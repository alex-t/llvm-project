//===-- AMDGPUSSARegisterAllocator.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// SSA-based Register Allocator for AMDGPU.
///
/// Implements width-descending multi-pass PEO coloring based on:
/// "Register Allocation for Programs in SSA-Form"
/// Sebastian Hack, Daniel Grund, Gerhard Goos (CC'06)
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSSAREGISTERALLOCATOR_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSSAREGISTERALLOCATOR_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "SSASpillEmitter.h"
#include "SSAForensicReporter.h"
#include "SSARegisterTree.h"
#include <memory>
#include <set>

namespace llvm {

class GCNSubtarget;
class SIInstrInfo;
class SIRegisterInfo;
struct GCNRegPressure;

class AMDGPUSSARegisterAllocator : public MachineFunctionPass {
  const SIRegisterInfo *TRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  MachineDominatorTree *MDT = nullptr;
  SlotIndexes *Indexes = nullptr;
  LiveIntervals *LIS = nullptr;
  MachineLoopInfo *MLI = nullptr;
  const GCNSubtarget *ST = nullptr;
  RegisterClassInfo RegClassInfo;

  std::set<unsigned, std::greater<unsigned>> ColoringOrder;
  DenseMap<Register, MCRegister> ColorMap;
  BitVector OccupiedRegUnits;

  // Width-tiered coloring (experiment, -amdgpu-ssa-virgin-order). Per tier the
  // explicit VIRGIN allocation order — aligned tuples of that (pool, width) that
  // NO already-allocated wider value of the SAME pool occupies anywhere in the
  // function. A value colored into one of these cannot interfere with any wider
  // value, so the tier is Hack-compliant BY CONSTRUCTION (auditable: this vector
  // IS the proof — no interference test needed within it). SGPR and VGPR pools
  // are disjoint, so a single pass could handle both; they are kept SEPARATE
  // tiers purely for transparency. Key = (isVector?1:0, widthBits) — unsigned,
  // not bool, since DenseMap's pair key needs a hashable integral. Built once
  // per tier in color()'s width loop.
  DenseMap<std::pair<unsigned, unsigned>, SmallVector<MCRegister, 64>>
      VirginTierOrder;
  void buildVirginTierOrder(bool IsVector, unsigned WidthBits);

  // Per-tier pick tallies (experiment). Keyed like VirginTierOrder, incremented
  // in pickFreePhysReg when a value of that (pool,width) is placed from the
  // virgin order vs the gap-scan. analyzeTierRank reads AND resets the key it
  // reports, so each entry counts exactly one (phase,pool,width) tier walk.
  DenseMap<std::pair<unsigned, unsigned>, unsigned> VirginPickByTier;
  DenseMap<std::pair<unsigned, unsigned>, unsigned> GapPickByTier;

  // Forensic (post-pass) feasibility analysis: over the vregs a tier actually
  // colored (\p TierVRegs) PLUS the ones it could not color (\p FailedVRegs —
  // they still competed for the same virgin pool, so they belong in the rank),
  // compute the tier's interference-graph rank via LIS and compare to the virgin
  // pool size. Emits one [TIERPROOF] line (via errs(), so it survives a
  // downstream crash) whose verdict separates colorer-fault from
  // spiller-under-spill:
  //   rank <= pool, no gap, no fail  -> HACK-OK (pure Hack held — the proof)
  //   rank <= pool, but gap or fail  -> COLORER fault (feasible yet Hack missed)
  //   rank >  pool, fail > 0         -> SPILLER under-spilled (infeasible tier)
  // NOT on the coloring decision path.
  void analyzeTierRank(unsigned Phase, bool IsVector, unsigned WidthBits,
                       ArrayRef<Register> TierVRegs,
                       ArrayRef<Register> FailedVRegs);

  // Gap-scan fallback (virgin exhausted): find a PR of \p RC whose colored
  // occupants' live intervals do NOT overlap \p VI along VI's whole range
  // (queried via LIS) — a gap opened by an occupant's death that we may reuse.
  // Returns 0 if none.
  MCRegister findNonInterferingGap(const TargetRegisterClass *RC,
                                   const LiveInterval &VI);
  void dumpSpanWidthDelta(const TargetRegisterClass *RC, const LiveInterval &VI);
  unsigned MaxVGPRIdx = 0;
  unsigned MaxSGPRIdx = 0;
  unsigned MaxAGPRIdx = 0;
  // VGPRs withheld from the vector (VGPR/AGPR) allocatable budget for the WWM
  // scratch that downstream SGPR-spill lowering needs. Set by the driver after
  // the SGPR allocation stage from the emitter's spilled-SGPR-lane count; 0
  // during the SGPR stage. Consumed by allocatablePool().
  unsigned VGPRReserve = 0;
  // (call def-slot, call instruction) for every call; a vreg live across a call
  // must avoid every register the call clobbers (regmask + explicit defs).
  SmallVector<std::pair<SlotIndex, const MachineInstr *>, 8> CallSites;
  unsigned DynVGPRBlockSize = 0;

  // Exec-safe spill/reload emitter, shared with the spiller pass. Used by the
  // approach-A spill-on-coloring-failure path: when color() cannot place a
  // value, the driver spills it here (store-at-def + dominance reloads) and
  // recolors. Created per function in runOnMachineFunction.
  std::unique_ptr<SSASpillEmitter> Emitter;

  // Forensic reporter (observer; -amdgpu-ssa-forensic*). Records observable
  // allocation FACTS for post-hoc analysis and NEVER mutates allocator state.
  // Every hook early-returns when the reporter is disabled (the default), so the
  // allocator is byte-identical ON vs OFF. Created per function in
  // runOnMachineFunction and shared with the emitter via setReporter().
  std::unique_ptr<SSAForensicReporter> Reporter;

  // === Shadow register-tree oracle (-amdgpu-ssa-shadow-tree, default off) ===
  //
  // A SHADOW SSARegisterTree that mirrors, for the VGPR_32 file ONLY, the exact
  // occupancy the allocator maintains in OccupiedRegUnits, and — at each real
  // VGPR_32 pick — logs what the tree WOULD have picked vs. what the allocator
  // actually chose. It NEVER influences allocation: its answer is discarded and
  // every mutation/compare is guarded behind the flag AND Reporter->active(), so
  // an off run (and a build without the flag set) is byte-identical.
  //
  // Mapping: leaf index == VGPR_32 allocation-order ordinal (getOrder index).
  // The tree requires a power-of-two leaf count, so it is sized to the padded
  // power of two >= the real allocatable VGPR_32 count; the padding leaves
  // [RealVGPR32Count, padded) are pre-allocated at construction so the tree's
  // pickFreeAligned can never return a nonexistent register. See the .cpp for
  // the leafOf() physreg->leaf map and the width-1 scope of this increment.
  std::unique_ptr<SSARegisterTree> ShadowTree;
  // getOrder(VGPR_32) ordinal for each MCRegister, or -1 if not a VGPR_32 in the
  // order. Built once per function in setupShadowTree(); the identity of the
  // physreg<->leaf bijection.
  DenseMap<unsigned, int> VGPR32Leaf;
  // MCRegUnit -> owning VGPR_32's leaf ordinal. Keyed by reg UNIT (not physreg)
  // because on targets with lo16/hi16 sub-registers the reg-unit roots of a
  // VGPR_32 are VGPRn_LO16/HI16, never VGPRn itself — so a physreg-id lookup off
  // a reg unit never matches. This unit->leaf map is the reliable bridge from an
  // OccupiedRegUnits bit (or a physreg's reg units) to its VGPR_32 leaf.
  DenseMap<unsigned, int> VGPR32UnitLeaf;
  unsigned RealVGPR32Count = 0;   // allocatable VGPR_32 regs (real, pre-padding)
  unsigned ShadowLeaves = 0;      // padded power-of-two leaf count of ShadowTree
  bool shadowActive() const;      // flag && Reporter && Reporter->active()
  void setupShadowTree();         // build the map + tree (per function)
  // Return the leaf index of \p PhysReg if it is a VGPR_32 in the mapped order,
  // else -1. Wider VGPR tuples map to their FIRST (lowest-index) sub-VGPR_32
  // leaf, which is the aligned block start; a non-VGPR physreg returns -1.
  int shadowLeafOf(MCRegister PhysReg) const;
  // Collect the leaf index of every VGPR_32 that \p PhysReg covers (its own leaf
  // if it is a VGPR_32; each sub-VGPR_32's leaf if it is a wider tuple), resolved
  // through the getOrder-ordinal map so no contiguity of a tuple's sub-registers
  // in leaf space is assumed. Empty for a non-VGPR physreg.
  void shadowLeavesOf(MCRegister PhysReg,
                      SmallVectorImpl<unsigned> &Leaves) const;
  // Mirror OccupiedRegUnits mutations into ShadowTree for the VGPR_32 file only.
  // These are no-ops unless shadowActive(). \p PhysReg is a full physreg (any
  // width); the width in leaves is derived from its VGPR-unit span.
  void shadowAllocate(MCRegister PhysReg);
  void shadowFree(MCRegister PhysReg);
  // Mirror a raw OccupiedRegUnits.reset(Unit): free the VGPR_32 leaf that owns
  // \p Unit (no-op if Unit is not a VGPR_32 unit). For the two sites that clear
  // single reg units directly rather than through markFree.
  void shadowFreeUnit(MCRegUnit Unit);
  void shadowResetToOccupied(); // rebuild tree occupancy from OccupiedRegUnits

  // Values that color() could not place (no physreg free across their whole
  // range — the %560/%1072 long-liver class). color() collects ALL of them and
  // finishes the walk (coloring everything else normally), rather than bailing
  // on the first, so that after the driver spills these the ONLY uncolored
  // vregs left are the short reload remainders — which provably settle.
  SmallVector<Register, 8> UncolorableVRegs;

  // === Coloring ===
  void classifyVRegs();
  // On unified-file targets (gfx90a/gfx942: VALU reads/writes AGPRs directly),
  // widen each VGPR-class vreg to the equivalent vector super-class (av_*) when
  // EVERY operand constraint already admits AGPRs. This lets a narrow value draw
  // the virgin AGPR tuples that wider VGPR tuples left free (the Greedy
  // spill-to-AGPR rescue, done as a sound up-front regclass widen rather than a
  // pick-time fallback — keeps each Hack tier drawing from one unified order).
  // Conservative: a sub-register operand blocks the widen (the whole-reg operand
  // constraint test does not apply to a subreg slice). Behind -amdgpu-ssa-agpr-
  // rescue. Must run before classifyVRegs so widened widths feed ColoringOrder.
  void widenToAVOnUnified();
  /// Run a full coloring walk. Colors every placeable value into ColorMap;
  /// appends any value it cannot place to UncolorableVRegs and skips it (does
  /// not occupy a register for it) so the rest of the walk proceeds as if that
  /// value were absent. Does not assert on failure.
  void color();
  /// Color a single value \p R in place against the CURRENT ColorMap /
  /// OccupiedRegUnits, without disturbing any existing assignment. Seeds
  /// occupancy from the colored values whose live range overlaps R's, then picks
  /// a free physreg across R's (short) range. Used to place reload remainders
  /// after a coloring-failure spill. Returns false if no register is free
  /// (should not happen for a width-1 reload — point pressure ≤ limit < file).
  bool colorOneInPlace(Register R);

  /// CSR(CS): the registers this allocation may use for \p RC that \p CallMI
  /// preserves. A call's regmask IS its preserved set, and ISel builds that mask
  /// from the CALLEE's calling convention, so this is a property of the call site
  /// -- two calls in one function can preserve different sets. A regmask can only
  /// be tested, never enumerated, so the candidates come from \p RC's allocation
  /// order (which for a vector class already spans VGPRs then AGPRs, and already
  /// holds the tuple registers of each width) and the mask decides which survive.
  SmallVector<MCRegister, 32> getCSRSet(const MachineInstr &CallMI,
                                        const TargetRegisterClass *RC) const;

  /// True if \p PR survives every clobber site the value described by \p VI is
  /// live at. A site is a call (its regmask clobbers the caller-saved partition,
  /// and an explicit def such as the return-address $sgpr30_sgpr31 clobbers that
  /// too) or an implicit def of an allocatable physreg -- an inline-asm register
  /// clobber, an implicit-def $vcc. A value colored onto a register that any
  /// site in its range writes is destroyed there, so this is the legality rule
  /// for every register handed to a value, wherever the decision is made.
  bool survivesClobberSites(const LiveInterval &VI, MCRegister PR) const;

  /// Assign registers to the values live across calls, BEFORE any coloring, so
  /// they get first pick of the registers calls preserve -- a value crossing a
  /// call can occupy nothing else, while the values the main walk places are free
  /// to sit anywhere. Call sites are walked in dominance order; at each, a value
  /// live across it keeps the register it already holds when this call preserves
  /// it, else takes one from CSR(CS), else is spilled across the call.
  void preassignValuesLiveAcrossCalls();

  /// What a recovery handler achieved (driver -> FSM stream). nextRecoveryState
  /// maps it to the next state; Resolved -> OK terminal, Infeasible -> terminal.
  enum class RecoveryResult {
    Resolved,  // value fully placed — done (FSM -> OK)
    Reduced,   // progress: shorter remnant (SelfSplit) or a blocker spilled
               // (SpillBlockers) — re-dispatch
    NoOp,      // precondition did not hold; nothing changed — next strategy
    Infeasible // genuine point-over-pressure — honest terminal
  };

  /// Recovery FSM states (FSM -> driver stream: which handler to run). Start
  /// classifies to a handler state; each handler's RecoveryResult drives
  /// nextRecoveryState; OK/Infeasible are terminals. The driver loops to a
  /// terminal with a monotone-progress detector (candidate length decreasing OR a
  /// blocker spilled) that breaks the SelfSplit<->CrossLiver cycle.
  enum class RecoveryState {
    Start, Web, CrossLiver, SelfSplit, AGPRRelief, Floor, OK, Infeasible
  };

  /// Spill a colored blocker B (occupying a physreg P legal for \p Failed) to
  /// free P over \p Failed's range. Two candidate classes, both requiring B live
  /// at F's end with NO use strictly inside (FS,FE) (so B's reload lands past FE
  /// -> no round-trip):
  ///  - LIVE-THROUGH (B.def <= FS): frees P over ALL of F -> Failed colors whole
  ///    -> Resolved.
  ///  - BORN-IN-F (B.def in (FS,FE)): frees P over F's TAIL [B.def,FE); F is
  ///    split at B.def, the tail colors into P, the HEAD [FS,B.def) is handed
  ///    back in \p Remnant -> Reduced.
  /// Multi-candidate pick = COVERAGE: live-through (frees all of F) beats
  /// born-in-F; among born-in-F the earliest def frees the longest tail. Returns
  /// NoOp if no clean candidate exists.
  RecoveryResult spillBlocker(Register Failed, unsigned RPLimit,
                              Register &Remnant);

  /// Close the PHI web seeded by \p Seed (a PHI result, or a PHI operand feeding
  /// one). Bidirectional closure over PHI operand/result edges, then the
  /// shared-slot soundness gate (declines if two ground operands interfere).
  /// Detection POLICY — RA-owned (moved out of the emitter's spillPhiWeb, now
  /// pure mechanics). Returns an INVALID PhiWeb (see PhiWeb::valid) if \p Seed is
  /// not part of a spillable web, so the caller falls back to a plain spill.
  PhiWeb closePhiWeb(Register Seed) const;

  /// RA-side feasibility gate for spilling \p Web: a web spill is a monotone
  /// wall-dissolution, so the ONLY failure is a reload landing where post-spill RP
  /// still exceeds \p Limit. Probe exactly that at each web member's EXTERNAL
  /// (non-PHI-edge) use via the shared reloadRPBeforeUse helper. \p IsVGPR selects
  /// the file. Enforces web-spill ATOMICITY: the RA proves feasibility BEFORE
  /// dispatch; spillPhiWeb is then pure spill/reload mechanics that cannot fail.
  bool webReloadFeasible(const PhiWeb &Web, bool IsVGPR, unsigned Limit) const;

  /// [Design: region-rp-reduction, Stage 1] Register file for region
  /// enumeration. AGPR is a DISTINCT file on non-unified targets (gfx908):
  /// separate budget, measured separately. On unified targets (gfx90a+) arch-VGPR
  /// and AGPR share one budget — enumerate only SGPR + (unified) VGPR there.
  enum class RegFile { SGPR, VGPR, AGPR };

  // The register file the current allocation stage owns. Allocation runs in two
  // independent stages — SGPR first, then VGPR (fileOf maps AGPR to VGPR, so the
  // vector stage handles VGPR+AGPR). color()/preSpill/region-rp process only
  // values of StageFile; the driver sets it before each stage.
  RegFile StageFile = RegFile::SGPR;

  /// [Stage 1] A tight region: a contiguous slot span within ONE block whose
  /// all-live RP in \p File exceeds the allocatable-pool limit. Half-open slot
  /// pair label; a value "crosses" it if its interval overlaps [Start,End).
  /// Block-local for v1 (cross-MBB coalescing deferred). Target is RP > pool:
  /// all-live RP already counts colored + uncolored crossers, so RP <= pool is
  /// exactly "placed + uncolored-crossers all fit".
  struct TightRegion {
    MachineBasicBlock *MBB;
    SlotIndex Start, End; // half-open, within MBB
    SlotIndex PeakSlot;   // slot where Peak RP is reached (a value must be live
                          // here to relieve the region by being spilled)
    RegFile File;
    unsigned Peak;  // max RP observed in the span
    unsigned Limit; // allocatable-pool count for the file
  };

  /// [Stage 1] Allocatable-pool size for \p File (SGPR_32 / VGPR_32 / AGPR_32
  /// count) — the number the colorer draws from; the region target is RP <= this.
  /// NOT raw getMaxNum* (102/64), NOT the spiller's margined value.
  unsigned allocatablePool(MachineFunction &MF, RegFile File) const;

  /// THE single source of truth for which physregs of \p RC this allocation may
  /// use: RegClassInfo::getOrder(RC) minus the WWM reserve (VGPRReserve, dropped
  /// from the tail for vector classes). The colorer scans this, and
  /// allocatablePool() is its size — so coloring capacity and the pressure
  /// budget never diverge.
  ArrayRef<MCPhysReg> availableOrder(const TargetRegisterClass *RC) const;

  /// [Stage 1] Per-file pressure at a tracker point. VGPR uses the UNIFIED count
  /// on gfx90a+ (arch+agpr+avgpr share one budget) and arch-VGPR alone otherwise;
  /// AGPR is the separate AGPR count (only meaningful on non-unified targets).
  unsigned pressureOf(const GCNRegPressure &P, RegFile File) const;

  /// Post-spill RP just BEFORE \p UseMI (per-use reload site): reset(*MI) seeds
  /// just after the use, recede steps to just-before; -W+W cancel => post-spill RP
  /// there. Feasibility POLICY — RA-owned (moved from the Emitter, which is pure
  /// spill/reload/SSA-repair mechanics). \p IsVGPR selects the file. NOTE: uses
  /// getVGPRNum(hasGFX90A) — NOT pressureOf's getArchVGPRNum — to stay bit-identical
  /// to the pre-move behavior (they differ by the AGPR term on non-unified targets).
  unsigned reloadRPBeforeUse(const MachineInstr *UseMI, bool IsVGPR) const;

  /// Post-spill RP at the END of \p NCD (shared hoisted-reload site). Valid for an
  /// empty NCD. Same file/accounting notes as reloadRPBeforeUse.
  unsigned reloadRPAtBlockEnd(const MachineBasicBlock *NCD, bool IsVGPR) const;

  /// [Stage 1] Enumerate tight regions for \p File: per block, maximal contiguous
  /// slot spans where all-live RP (GCNUpwardRPTracker) > allocatablePool(File).
  void findTightRegions(MachineFunction &MF, RegFile File,
                        SmallVectorImpl<TightRegion> &Out) const;

  /// Within tight region \p R, find the peak-RP slot at which \p V is LIVE (not
  /// R's global peak, which may fall outside V's range). Returns {slot, RP}; RP is
  /// 0 if V is live at no in-region slot. Same GCNUpwardRPTracker + pressureOf
  /// machinery as findTightRegions, so the RP is bit-identical.
  std::pair<SlotIndex, unsigned>
  peakSlotForValueInRegion(const TightRegion &R, Register V) const;

  /// Spill victims from ONE tight region until its total-dword excess (Peak-Limit)
  /// is gone. Candidates are frozen-\p Universe values of \p R's file, live at
  /// R.PeakSlot, not already in \p Spilled, and admitted by \p Eligible; chosen
  /// over-subscribed-then-widest-first, each decrementing the excess by its real
  /// width. Spilled victims are added to \p Spilled. Returns true if it spilled.
  /// Shared by the width-aware pre-spiller (Eligible = always) and AGPR-relief
  /// (Eligible = avReloadLegal, so reloads re-home to a free AGPR).
  /// If \p NumRecolored is non-null, it is incremented by the number of victims
  /// relieved by AGPR RECOLOR (not memory spill) — a MONOTONE action (AGPR budget
  /// and the frozen universe both strictly shrink), so a round that recolored is
  /// always real progress and the caller must NOT apply its rolling-wave guard.
  bool relieveTightRegion(const TightRegion &R,
                          const SmallDenseSet<Register, 128> &Universe,
                          SmallDenseSet<Register, 64> &Spilled,
                          llvm::function_ref<bool(Register)> Eligible,
                          unsigned *NumRecolored = nullptr);

  /// Naive up-front pre-spiller. At each tight region's peak slot, spill
  /// widest-first live values (kill-at-def memory spill) until point-RP <= the
  /// allocatable pool, iterating (re-measure) until no tight region remains. Runs
  /// BEFORE color() so the Hack fast-path colors by construction. Only guarantees
  /// the PRESSURE precondition — width>1 aligned-tuple colorability (chi>omega) is
  /// NOT ensured; such residuals still flow to color()'s recovery. Returns true if
  /// anything was spilled. Flag-gated (-amdgpu-ssa-pre-spill).
  bool preSpillToLimit(MachineFunction &MF);

  /// Width-aware up-front pre-spiller (-amdgpu-ssa-pre-spill-wa). Same tight-region
  /// / kill-at-def+reload-at-use machinery as preSpillToLimit, but the frozen victim
  /// universe spans ALL widths and victims are chosen WIDEST-FIRST, decrementing the
  /// region peak by each victim's real dword width. This relieves regions dominated
  /// by wide tuples (SGPR/VGPR vreg_64/128/...) that the width-1-only naive version
  /// cannot touch. Models per-width availability honestly (pool/W aligned tuples per
  /// class) but does NOT claim to resolve aligned-tuple fragmentation (chi>omega):
  /// pool-fit is necessary, not always sufficient; placement residuals still flow to
  /// color()'s recovery. Returns true if anything was spilled.
  bool preSpillToLimitWidthAware(MachineFunction &MF);

  /// [Recovery classifier, Stage 1] Register file of a class for the recovery
  /// window. AGPR folds into VGPR so the file matches pressureOf(VGPR)'s unified
  /// count (SGPR classes -> SGPR, everything else -> VGPR).
  RegFile fileOf(const TargetRegisterClass *RC) const;

  /// Control-flow ordering of two slots. THE SINGLE SOURCE OF TRUTH for "does
  /// slot A come before slot B" — based on DOMINANCE, never on block layout /
  /// SlotIndex numeric distance (layout order is NOT program order; comparing
  /// slot ordinals is a bug and is forbidden). Same block -> instruction order;
  /// different blocks -> dominator-tree relation; divergent paths -> Incomparable
  /// (neither precedes the other — e.g. sibling diamond arms).
  enum class SlotOrder { Before, After, Same, Incomparable };
  SlotOrder compareSlots(SlotIndex A, SlotIndex B) const;

  /// [Recovery classifier, Stage 1] A "recovery window" for one uncolored value:
  /// the forward slot span from the value's def down to the first non-PHI point
  /// where real register pressure drops below the file limit, plus the universe
  /// of already-colored, same-file crossers with NO use inside the window (the
  /// spill-candidate universe a later stage will draw from). SIDE-EFFECT-FREE:
  /// collected and logged only; it drives nothing in Stage 1.
  /// Why a recovery window stopped growing. Exactly one reason per window.
  /// BackEdge is the loop-carried-web signal (the window's single hop to a loop
  /// header PHI) and is distinct from ForkDivergence — do NOT lump them.
  enum class WindowStop {
    RPRecovered,    // closed cleanly: first non-PHI slot with RP < Limit
    ForkDivergence, // stopped at a >1-successor (or 0-successor) block
    BackEdge,       // followed a unique successor back into a visited block (loop)
    Cap             // hit MaxWindowSlots without RP recovering
  };

  struct RecoveryWindow {
    Register Uncolored;
    SlotIndex Start;                    // def slot of Uncolored
    SlotIndex End;                      // first non-PHI slot fwd with real RP < Limit
    SmallVector<Register, 16> Crossers; // spill-candidate universe (see collect)
    WindowStop Stop = WindowStop::RPRecovered;
    // PHI-web membership — the analyst signal. A PHI-web is one CFG primitive:
    // Uncolored feeds a PHI (a value-merge node). Loop-carried vs. divergent-
    // diamond is a LATER cost-model distinction, not a detection concern (YAGNI
    // now). WebPhi = the PHI result reg this value merges into, or invalid if the
    // value feeds no PHI. See Recursive_Recovery_Fix.md.
    Register WebPhi;
    // Stage-2 dispatch inputs (spill-around ranking). Per-crosser widths are NOT
    // stored — derived from the vreg id (consumer-side for the analyst, live for
    // the in-allocator dispatcher).
    unsigned UncoloredWidth = 0; // width of Uncolored in dwords (aligned-tuple feasibility)
    unsigned RPOvershoot = 0;    // peak (RP - Limit) across the window; 0 if never over
                                 // (spill-1 vs spill-N signal)
    // Branch-3 pick: the register-resident prefix the classifier proved peelable,
    // handed to trySelfSplitColor as its FIRST piece so the handler never
    // re-derives what routed it here. Valid iff classifyRecovery returned
    // SelfSplit; invalid on the CrossLiver/Web fall-through, where branch 3 was
    // never evaluated and the handler picks for itself.
    MCRegister PeelPR;
    SlotIndex PeelBound;
  };

  /// Pick the register free at \p V's start that stays free LONGEST, and decide
  /// whether that free run is worth peeling. Returns false when nothing is free at
  /// the start, or when the run does not reach past \p V's first use (genuine
  /// over-pressure rather than fragmentation). On true, \p PR is the register and
  /// \p Bound the slot where it becomes occupied (>= V's end means free across all
  /// of V). THE single split-across policy: classifyRecovery (branch 3) asks "will
  /// SelfSplit make progress?" and trySelfSplitColor asks "what do I peel now?",
  /// so the gate cannot disagree with the handler. Const — reads LIS/MRI/ColorMap.
  bool pickPeelableRun(Register V, MCRegister &PR, SlotIndex &Bound) const;

  /// [Recovery classifier, Stage 1] Collect the recovery window for \p Uncolored
  /// (see RecoveryWindow). Uses the trusted GCNUpwardRPTracker only. Const —
  /// reads LIS / MRI / ColorMap; mutates no allocator state.
  RecoveryWindow collectRecoveryWindow(Register Uncolored) const;


  /// Cross-liver PRECONDITION: true iff \p Failed has a cleanly-spillable
  /// live-through blocker — colored, same-file (getCommonSubClass), live across
  /// Failed's whole range with no use strictly inside (so its reload lands after
  /// FE), AND that reload's post-spill RP stays within \p RPLimit
  /// (reloadRPBeforeUse). The classifier calls this so CrossLiver is chosen ONLY
  /// when spillCrossLiver will find a feasible candidate (no spill-then-fail).
  bool hasCleanCrossLiver(Register Failed, unsigned RPLimit) const;

  /// A memory spill of \p R relieves it only if some non-PHI use has post-spill RP
  /// <= \p RPLimit (the reload lands below saturation). Else the reload re-enters
  /// the same pressure (a spill-reload thrash). No non-PHI use -> trivially viable.
  /// The Floor-vs-Infeasible discriminator.
  bool floorViable(Register R, bool IsVGPR, unsigned RPLimit) const;

  /// Unified-file (gfx90a+) VGPR-saturation relief: spill an av-LEGAL colored
  /// crosser of \p Failed so its reload re-homes to a free AGPR (availableOrder
  /// lists VGPRs then AGPRs; when VGPRs are saturated the reload falls through to
  /// an AGPR), freeing a VGPR across Failed's range. Resolved if Failed then
  /// colors; NoOp otherwise (freed nothing usable -> Floor).
  RecoveryResult agprRelief(Register Failed, unsigned RPLimit);

  /// True iff \p B's reload may legally be colored to an AGPR: every operand
  /// admits an AGPR (av_ is a subclass of each operand's required class) and no
  /// operand is a sub-register slice. Same legality test as widenToAVOnUnified.
  bool avReloadLegal(Register B) const;

  /// LAST-DITCH rescue fired right before reportPointOverPressure would abort. On
  /// a unified target, when \p R cannot be placed in the VGPR file (its whole
  /// range is VGPR-clobbered / over-pressure) but the AGPR file has a free tuple
  /// of R's width, HOME R in an AGPR and insert a short AGPR->VGPR copy before each
  /// VGPR-only-constrained use (the copy lives only [copy,use] -> trivially
  /// colorable). This is Greedy's v_accvgpr_read pattern for an asm-pinned block.
  /// Returns true if R was rescued (colored); false if the conditions do not hold
  /// (caller then screams). Flag-gated on EnableAGPRFirst.
  bool tryAGPRHomeRescue(Register R);

  /// FSM transition: given the current handler \p S and its \p R result, return
  /// the next state per the recovery transition table.
  RecoveryState nextRecoveryState(RecoveryState S, RecoveryResult R) const;

  /// [Recovery FSM] Classify \p RW into the FIRST handler STATE (web > cross-liver
  /// > self-split), or a terminal (Floor if a memory reload fits, else Infeasible)
  /// when no structural pattern applies. The driver runs the state's handler and
  /// advances via nextRecoveryState. Reads \p RW + feasibility helpers, and on a
  /// SelfSplit verdict writes back the peel pick (RW.PeelPR / RW.PeelBound).
  RecoveryState classifyRecovery(RecoveryWindow &RW) const;

  /// Emit the forensic recoveryWindow event for \p Failed / \p RW (analyst
  /// signal). Called by the recovery driver when the reporter is enabled; no
  /// effect on dispatch.
  void emitRecoveryWindow(Register Failed, const RecoveryWindow &RW) const;

  /// [Stage 2] Cost of spilling candidate \p B to relieve region \p R.
  ///   Feasible : no use is strictly inside R (case 2) nor in R's loop (case 3),
  ///              and every reload's post-spill RP <= R.Limit (Test 2). Else B
  ///              cannot relieve R — drop it.
  ///   Cost     : NReloads * Width. NReloads = 1 when B's uses are commonly
  ///              dominated and the shared reload hoists to their NCD; else one
  ///              reload per use.
  ///   Width    : dwords B frees in R per spill (RP relief).
  /// Ranking (Stage 3): Feasible, then Cost asc, then Width desc (equal traffic
  /// -> widest wins: frees more RP in R).
  struct SpillCost {
    bool Feasible;
    unsigned Cost;
    unsigned Width;
    explicit operator bool() const { return Feasible; }
    static SpillCost Infeasible() { return {false, 0, 0}; }
  };
  SpillCost costOfSpilling(Register B, const TightRegion &R);

  /// [Stage 3] Region RP-reduction driver. While tight regions remain, service
  /// the worst (highest Peak) by spilling the cheapest feasible crosser ACROSS
  /// that region (kill at R.Start so its register frees over R and the reload
  /// lands after R), then recompute regions globally. Returns true if any spill
  /// was performed (caller then re-colors from clean). Bounded by a round cap.
  bool reduceRegionPressure(MachineFunction &MF);

  /// Self-split (self-split branch of the recovery classifier): \p Failed is
  /// a long liver with no through-lane AND no live-through blocker to spill around
  /// (spillCrossLiver found nothing). Chop Failed into segments, each
  /// short enough that one physreg is free across it, coloring each into that reg.
  /// Only valid when Failed is POINT-FEASIBLE (some PR free at every slot); aborts
  /// (returns false -> caller memory-spills) if any slot has zero free PRs.
  /// \p FirstPR / \p FirstBound is the pick for the FIRST piece, supplied by the
  /// FSM (RecoveryWindow::PeelPR). Pass an invalid \p FirstPR to have the handler
  /// pick it, which is what the CrossLiver/Web fall-through entries do.
  RecoveryResult trySelfSplitColor(Register Failed, MCRegister FirstPR,
                                   SlotIndex FirstBound, Register &Remnant);

  /// Coloring-time recovery for one value \p Failed that color() could not place.
  /// Classifier-driven (see Recursive_Recovery_Fix.md): collectRecoveryWindow
  /// classifies the failure, then dispatch goes to the ONE branch whose
  /// precondition holds (web / cross-liver / self-split), falling through to the
  /// spill-self floor when no pattern matches. A fresh redef that cannot color is
  /// NOT recursed on — it is re-queued to UncolorableVRegs and retried by the
  /// caller's worklist fixpoint (no-progress terminal is the honest bottom).
  /// Returns true if Failed was resolved (colored, or spilled with its reload
  /// redefs placed).
  bool recoverUncolorable(Register Failed);

  /// Honest terminal for the classifier's no-pattern floor. Counts the values of
  /// \p R's register
  /// file live at \p R's def point and compares the total dword count to \p
  /// RPLimit, then report_fatal_error()s with the REAL NUMBERS: either genuine
  /// point-over-pressure (more live dwords than registers -> no coloring-time
  /// recovery exists) or, if feasible-yet-unrecovered, an honest allocator-bug
  /// diagnostic -- never the misleading "needs more up-front spilling". \p Ctx
  /// labels the call site. Does not return.
  [[noreturn]] void reportPointOverPressure(Register R, bool IsVGPR,
                                            unsigned RPLimit, const char *Ctx);

  /// Assign \p Piece -> \p PR in ColorMap and bump the file's high-water mark.
  /// Shared by the coloring-time split paths.
  void commitColor(Register Piece, MCRegister PR);

  /// Earliest slot > \p S in [\p S, \p End) where \p PR becomes occupied by an
  /// overlapping colored value in \p Overlappers or a call-clobber; returns \p S
  /// if PR is already occupied at S (not free here), else the bound (clamped to
  /// \p End). Helper for trySelfSplitColor's per-piece free-run search.
  SlotIndex
  firstBlockAfter(MCRegister PR, SlotIndex S, SlotIndex End,
                  ArrayRef<std::pair<Register, MCRegister>> Overlappers) const;

  /// Single linear scan over ColorMap for \p VI: the shared "collect" step of the
  /// gap-scan / split pipeline. ORs the register units of every colored occupant
  /// whose interval overlaps VI into \p OccupiedUnits. \p Overlappers is optional:
  /// when non-null it also collects (occupant vreg, its physreg) for each. The gap
  /// pick (findNonInterferingGap) passes nullptr (needs only occupancy); the
  /// splitter (spillCrossLiver) passes a vector. NOT cacheable across the
  /// two — they run in different phases with ColorMap mutated between.
  void scanOverlappersForVI(
      const LiveInterval &VI, BitVector &OccupiedUnits,
      SmallVectorImpl<std::pair<Register, MCRegister>> *Overlappers = nullptr) const;
  void seedOccupiedAtBBEntry(MachineBasicBlock *MBB);
  // True if the parallel PHI edge-copies for Pred->MBB cannot be safely placed
  // at Pred's terminator (they would clobber a value live into a sibling
  // successor, or need a scratch register for a cycle), i.e. the critical edge
  // must be split. A non-critical edge never needs splitting.
  bool
  edgeCopiesNeedSplit(MachineBasicBlock *Pred, MachineBasicBlock *MBB,
                      ArrayRef<std::pair<MCRegister, MCRegister>> Copies) const;
  void markOccupied(MCRegister PhysReg);
  void markFree(MCRegister PhysReg);

  /// Debug: print \p RC's allocation order at slot \p SI as an occupancy map,
  /// one char per register in order:
  ///   '.' free and usable    '#' occupied (ColorMap vreg live at SI)
  ///   'x' free but CLOBBERED by a call \p VI is live across (unusable by a
  ///       value pinned to callee-saved) — only marked when \p VI is given.
  /// So "####xxxx####" shows callee-saved full ('#') with only caller-saved
  /// ('x') free — the classic pinned-value exhaustion. Occupancy is derived by
  /// walking ColorMap for vregs live at SI. \p Tag labels the line.
  void dumpOccupancyMap(const TargetRegisterClass *RC, SlotIndex SI,
                        const char *Tag, const LiveInterval *VI = nullptr) const;

  /// Pure fact extraction shared by dumpOccupancyMap (debug print) and the
  /// forensic reporter (E16 snapshots): compute the occupancy view of \p RC at
  /// \p SI into \p Out. Const — reads OccupiedRegUnits / ColorMap / CallSites /
  /// LIS only, mutates nothing. Identical logic to the counting loop that used
  /// to live inline in dumpOccupancyMap; that function now calls this and prints.
  void collectOccupancy(const TargetRegisterClass *RC, SlotIndex SI,
                        const LiveInterval *VI, OccupancyFacts &Out) const;

  /// One colored value whose register could be freed across a coloring-failure
  /// region by spilling it: live-through the failed value's range with NO use
  /// strictly inside. \p OVI aliases the value's live interval (LIS-owned).
  struct SpillAcrossCandidate {
    Register V;
    MCRegister P;
    unsigned WidthDwords;
    const LiveInterval *OVI;
  };
  /// Pure fact extraction shared by the COLORFAIL debug block and the forensic
  /// reporter: over ColorMap, find colored values in \p Failed's register file
  /// that are live-through [FS,FE) (\p FS/\p FE = Failed's interval bounds). Sets
  /// \p NLiveThru to all live-through occupants and appends the no-interior-use
  /// subset (the spillable ones) to \p Out; \p LiveThruIdx gets the sorted vreg
  /// indices of the full live-through set. Const; mutates nothing.
  void collectSpillAcrossCandidates(
      Register Failed, SlotIndex FS, SlotIndex FE, bool FIsVGPR,
      unsigned &NLiveThru, SmallVectorImpl<SpillAcrossCandidate> &Out,
      SmallVectorImpl<unsigned> &LiveThruIdx) const;

  /// Forensic (facts-only, const): enumerate EVERY value live at \p SI into
  /// \p Out — the full liveness cross-section the analyst joins to the timeline.
  /// Reuses the same const LIS/ColorMap walk the allocator already relies on
  /// (liveAt over every vreg interval, ColorMap lookup for the physreg), so it
  /// adds no new pressure/LIS pass and mutates nothing. Only called when the
  /// forensic reporter is enabled (decision boundaries: E4/E10/E16).
  void collectLiveSet(SlotIndex SI, SmallVectorImpl<LiveSetEntry> &Out) const;

  // === Value-flow correctness verifier (-amdgpu-ssa-verify-value-flow) ===
  // Certifies SSA-destruction + physreg assignment preserved VALUE IDENTITY:
  // every physreg use holds the SSA value its vreg operand named. Catches the
  // clobber-while-live class (a live value overwritten in its register) that
  // liveness / reaching-def cannot see (both only tell "is there A value", never
  // "is it THE value"). Ground truth = a snapshot taken PRE-destruction, while
  // values are still vregs. v1 certifies single-basic-block functions (~92% of
  // the corpus green set); multi-block functions are reported SKIP (uncertified)
  // pending the meet-at-joins + dominance-rescue layer.
  struct VFOp {
    unsigned VReg;
    unsigned SubReg;
    bool IsDef;
  };
  DenseMap<const MachineInstr *, SmallVector<VFOp, 4>> VFIntent;
  DenseMap<Register, MCRegister> VFColor; // ColorMap frozen pre-destruction
  DenseMap<uint64_t, uint64_t> VFUF;      // union-find over (vreg,lane) keys
  // (vreg,lane) keys that receive a REAL (non-undef) definition. A use only
  // checks lanes in this set: a partial def `undef %V.sub0 = ...` leaves other
  // lanes intentionally undefined (don't-care), and a later read of %V must not
  // demand a value token for those lanes (else false clobber).
  DenseSet<uint64_t> VFDefinedLane;
  uint64_t vfFind(uint64_t X);
  void vfUnion(uint64_t A, uint64_t B);
  void snapshotValueFlow(MachineFunction &MF); // call BEFORE lowerPHIs
  bool verifyValueFlow(MachineFunction &MF);   // call AFTER finalizeProperties

  MCRegister pickFreePhysReg(
      const TargetRegisterClass *RC, const LiveInterval &VI,
      ArrayRef<std::pair<MCRegister, const LiveInterval *>> WiderDefs,
      ArrayRef<MCRegister> Hints = {}, uint64_t AttemptID = 0);

  // Option B affinity: collect already-colored phi-partner physregs for VReg
  // (phi results it feeds, and -- if VReg is a phi result -- its operands),
  // ordered by 2^loopdepth of the incoming edge (hottest first). Sub-register
  // relationships compose both ways: a phi-result VReg reading a slice of a
  // wider operand takes that slice of the operand's color (getSubReg); a wide
  // VReg feeding a narrow result via VReg.subN takes the super-register whose
  // slice is the result's color (getMatchingSuperReg -- the loop-carried tuple
  // case). Class-compatible partners only. Returns [] when VReg touches no
  // colored phi partner.
  SmallVector<MCRegister, 4> collectPhiHints(Register VReg,
                                             const TargetRegisterClass *RC);

  // === SSA Destruction + Operand Rewrite ===
  bool hasCFPseudos(MachineFunction &MF) const;
  // Rewrite one file's vregs to physregs (lowerPHIs + rewriteOperands +
  // eliminateRegSequences), scoped to \p Only. Called per allocation stage.
  void rewriteStage(MachineFunction &MF, RegFile Only);
  // Post-both-stages finalize: physreg live-ins, MF properties, value-flow check.
  void finalizeAfterRewrite(MachineFunction &MF);
  void lowerPHIs(MachineFunction &MF, RegFile Only);
  void resolvePermutation(
      MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
      SmallVectorImpl<std::pair<MCRegister, MCRegister>> &Copies);
  // Break a permutation cycle through a memory scratchpad when no free scratch
  // register exists in the cycle's file (store a member, walk with copies,
  // reload). Used by resolvePermutation as the fallback for full-file cycles.
  void breakCycleViaMemory(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator InsertPt,
                           MCRegister CycleStart,
                           DenseMap<MCRegister, MCRegister> &DstToSrc);
  // Find a physreg of RC's file free AT InsertPt (not live across it, not
  // reserved, not one of the cycle's own regs) to use as a transient permutation
  // scratch when the function-wide high-water reg does not fit. Returns null if
  // the point is genuinely saturated. (Approach A: local, zero-cost scratch.)
  MCRegister findLocalScratch(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator InsertPt,
                              const TargetRegisterClass *RC,
                              const DenseMap<MCRegister, MCRegister> &CycleRegs);
  void emitSwap(MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
                MCRegister RegA, MCRegister RegB);
  void rewriteOperands(MachineFunction &MF, RegFile Only);
  /// Before rewriteOperands: for each REG_SEQUENCE with an `undef` source, mark
  /// the result's uses that read an undef lane `undef`, so the flag survives onto
  /// the physical read (else the dead tuple lane is read-but-never-defined -> the
  /// post-RA LIS verifier fatals "missing from live-in list").
  void markRegSequenceUndefLaneUses(MachineFunction &MF, RegFile Only);
  void eliminateRegSequences(MachineFunction &MF);
  void addPhysRegLiveIns(MachineFunction &MF);
  void finalizeProperties(MachineFunction &MF);

public:
  static char ID;

  AMDGPUSSARegisterAllocator() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU SSA Register Allocator";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<SlotIndexesWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUSSAREGISTERALLOCATOR_H
