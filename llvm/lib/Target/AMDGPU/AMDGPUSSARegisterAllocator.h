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
  mutable DenseMap<const TargetRegisterClass *, unsigned> StrideCache;

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

  struct PendingTie {
    MachineInstr *MI;
    unsigned DefOpIdx;
    unsigned UseOpIdx;
  };
  // Exact tied operand pairs encountered by the current coloring walk.
  // Recovery may repoint the use after the def inherited its old color, so
  // validate these only after every coloring failure has been recovered.
  SmallVector<PendingTie, 8> PendingTies;

  // The cross-pool copies tryCrossFileHome itself minted. A copy that fails to
  // color joins UncolorableVRegs, and both the drain loop and the terminal sweep
  // walk values queued while they run, so without this the rescue would be
  // applied to its own copies, each minting another (unbounded). Rescuing a copy
  // can never help anyway: it exists precisely to hold one pool at one instruction.
  SmallDenseSet<Register, 8> RescueCopies;

  // Recovery mutations that must not repeat. A value moved to the sibling
  // vector pool may reclaim that home later, but is never moved back; a value
  // spilled as a blocker or by the memory floor is never selected for another
  // recovery spill. Cleared at the start of each allocation stage.
  SmallDenseSet<Register, 8> RehomedVRegs;
  SmallDenseSet<Register, 16> RecoverySpilledVRegs;

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
  bool tiedAssignmentsValid() const;
  void drainUncolorableWorklist(MachineFunction &MF);

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

  /// Result of one recovery strategy. NoChange guarantees that the strategy
  /// left MIR, LIS, register classes, and ColorMap unchanged.
  enum class RecoveryResult { Resolved, Changed, NoChange };

  /// Spill a colored blocker B (occupying a physreg P legal for \p Failed) to
  /// free P over \p Failed's range. Two candidate classes, both requiring B live
  /// at F's end with NO use strictly inside (FS,FE) (so B's reload lands past FE
  /// -> no round-trip):
  ///  - LIVE-THROUGH (B.def <= FS): frees P over ALL of F -> Failed colors whole
  ///    -> Resolved.
  ///  - BORN-IN-F (B.def in (FS,FE)): frees P over F's TAIL [B.def,FE); F is
  ///    split at B.def, the tail colors into P, the HEAD [FS,B.def) is handed
  ///    back in \p Remnant -> Changed.
  /// Multi-candidate pick = COVERAGE: live-through (frees all of F) beats
  /// born-in-F; among born-in-F the earliest def frees the longest tail. Returns
  /// NoChange if no clean candidate exists.
  RecoveryResult spillBlocker(Register Failed, Register &Remnant);

  /// Close the PHI web seeded by \p Seed (a PHI result, or a PHI operand feeding
  /// one). Bidirectional closure over PHI operand/result edges, then the
  /// shared-slot soundness gate (declines if two ground operands interfere).
  /// Detection POLICY — RA-owned (moved out of the emitter's spillPhiWeb, now
  /// pure mechanics). Returns an INVALID PhiWeb (see PhiWeb::valid) if \p Seed is
  /// not part of a spillable web, so the caller falls back to a plain spill.
  PhiWeb closePhiWeb(Register Seed) const;

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
    SlotIndex PeakSlot;   // slot carrying the largest deficiency (a value must be
                          // live here to relieve the region by being spilled)
    RegFile File;
    unsigned Deficiency; // max runs SHORT over the span, from demandDeficiency
    unsigned Limit;      // allocatable-pool count. A budget handed to the emitter
                         // and a number to report — never again a feasibility
                         // threshold: whether a slot fits is a packing question
                         // a pool size cannot answer.
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

  /// Spacing of legal first registers for \p RC, read off availableOrder: 1 where
  /// any register may start the value, 2 for 64-bit SGPRs and Align2 VGPR tuples,
  /// 4 for wider SGPRs. Cached per class.
  unsigned allocStride(const TargetRegisterClass *RC) const;

  /// The register POOL a value must be placed in — the tier axis of the demand
  /// oracle. Deliberately NOT fileOf(), which selects the allocation STAGE and
  /// folds AGPR into the vector stage: charging an AGPR value against the
  /// arch-VGPR pool measures a shortage that does not exist, because the two
  /// register sets are physically disjoint with separate allocation orders.
  RegFile poolOf(const TargetRegisterClass *RC) const;

  /// One (pool, width) tier at one slot: how many live values need it, and how
  /// many the oracle could actually place. The shortfall is Demand - Placed.
  struct TierDemand {
    const TargetRegisterClass *RC;
    unsigned Width, Stride, Demand, Placed;
  };

  /// THE demand oracle. At one slot, for \p Pool, per (pool, width) tier in
  /// DESCENDING width order, place each live value first-fit into the pool's
  /// remaining hardware registers, taking legal starts from availableOrder() so
  /// that stride, alignment, reserved registers and the withheld WWM tail all
  /// come from the one list the colorer itself scans. Returns the number of
  /// values that could not be placed: the deficiency, in RUNS.
  ///
  /// A dword total cannot answer this. Forty-four aligned pairs into exactly
  /// forty-four pair starts is at the limit with zero slack, and one 32-bit
  /// value landing on an even register destroys a start without changing any
  /// dword count. Descending width order is required because a wide tier
  /// constrains the narrow ones and never the reverse.
  ///
  /// \p Live is GCNRPTracker::LiveRegSet, spelled out here.
  unsigned demandDeficiency(const DenseMap<unsigned, LaneBitmask> &Live,
                            RegFile Pool,
                            SmallVectorImpl<TierDemand> &Out) const;

  /// The oracle bound to \p Pool, for the spill emitter's reload-placement
  /// queries. The emitter must ask the same question spill planning does: its
  /// own scalar count read getVGPRNum(), so for an AGPR value it compared a
  /// pool of AGPRs against a total of arch VGPRs.
  SSASpillEmitter::DemandFn demandFor(RegFile Pool) const;

  /// One in-region slot: the live set there and its deficiency. Carrying the SET
  /// rather than a number is what lets a candidate be priced by MEASUREMENT —
  /// re-query the oracle without the victim — instead of subtracting its width
  /// from a scalar that had already rounded.
  struct RegionSlot {
    SlotIndex SI;
    MachineInstr *MI;
    DenseMap<unsigned, LaneBitmask> Live;
    unsigned Short;
  };

  /// [Stage 1] Per-file pressure at a tracker point. VGPR is the arch-VGPR count
  /// on every target (rationale at the definition); AGPR is the separate AGPR
  /// count.
  unsigned pressureOf(const GCNRegPressure &P, RegFile File) const;

  /// 32-bit slots \p Lanes of \p RC occupy, in the SAME unit pressureOf reports:
  /// GCNRegPressure::inc charges a 32-bit class exactly 1 whatever its mask and a
  /// tuple the number of slots its live lanes cover. Use this, never the class
  /// width, whenever a value's demand is compared against a region peak.
  unsigned coveredSlots(const TargetRegisterClass *RC, LaneBitmask Lanes) const;

  /// 32-bit slots a spill of \p Lanes of \p V actually MOVES. Normally
  /// coveredSlots, because the store narrows to the subregister the mask names and
  /// the stack slot is sized to it; but a mask naming NO subregister falls back to
  /// storing the whole register, and then the traffic is the full class.
  unsigned spilledSlots(Register V, LaneBitmask Lanes) const;

  /// Post-spill RP at the END of \p NCD (shared hoisted-reload site). Valid for an
  /// empty NCD.
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

  /// What a region holds of one virtual register: the UNION of the lane masks the
  /// tracker reported live at in-region slots, and how many such slots. The mask is
  /// not decoration — pressure is charged per covered 32-bit slot, so it is what
  /// puts a victim's demand (and the spill that relieves it) in the peak's unit.
  struct RegionOccupancy {
    LaneBitmask Lanes = LaneBitmask::getNone();
    unsigned Slots = 0;
  };

  /// Largest demand DEFICIENCY of \p R's pool over ONLY [R.Start,R.End), from the
  /// oracle, so it is directly comparable to R.Deficiency. With \p Occupants the
  /// same walk also reports every virtual register live inside R — which IS the
  /// overlap test, hole-accurate by construction. With \p Slots it reports each
  /// in-region slot's instruction, live set, and deficiency, which the cumulative
  /// set planner copies and re-evaluates. With \p PeakTiers it reports which
  /// (pool, width) tiers were short at the worst slot.
  /// Returns 0 if the region packs, or if R's first instruction no longer exists.
  unsigned measureRegionPeak(
      const TightRegion &R,
      DenseMap<Register, RegionOccupancy> *Occupants = nullptr,
      SmallVectorImpl<RegionSlot> *Slots = nullptr,
      SmallVectorImpl<TierDemand> *PeakTiers = nullptr) const;

  /// Print the values live at the worst slot in \p Slots with their widths, so a
  /// relief prediction that did not come true can be compared against what
  /// actually stayed resident. Debug output only.
  void dumpWorstSlotLiveSet(ArrayRef<RegionSlot> Slots, const char *Tag) const;

  /// Diagnostic (-amdgpu-ssa-lane-waste-dump): per function and file, report the
  /// peak whole-tuple occupancy this allocator charges against the subrange
  /// occupancy LiveRegMatrix would charge. Mutates no allocator state.
  void reportLaneWaste(MachineFunction &MF) const;

  /// TEMPORARY diagnostic (-amdgpu-ssa-block-demand-dump): per slot, how many
  /// live values carry each (class width, allocation stride) pair, beside the
  /// lane-accurate pressure and the whole-block dword sum. Measurement input for
  /// the width-aware region gate; delete together with it. Mutates no state.
  void reportBlockDemand(MachineFunction &MF) const;

  /// Spill a complete area-planned set for one tight region. Candidates are
  /// frozen-\p Universe occupants admitted by \p Eligible. No source mutation is
  /// performed unless virtual oracle evaluation reaches zero deficiency.
  /// Spilled victims are added to \p Spilled.
  ///
  /// TEMPORARY: this borrows reduceRegionPressure's planner and measured-relief
  /// rule but remains a SECOND victim-selection loop over the same regions. The
  /// intended end state is one shared per-region loop parameterized by candidate
  /// admission; see the note at the top of the definition.
  /// The width-aware pre-spiller passes Eligible = always; other callers may
  /// restrict which frozen candidates participate in area planning.
  /// If \p NumRecolored is non-null, it is incremented by the number of victims
  /// relieved by AGPR RECOLOR (not memory spill) — a MONOTONE action (AGPR budget
  /// and the frozen universe both strictly shrink), so a round that recolored is
  /// always real progress and the caller must NOT apply its rolling-wave guard.
  bool relieveTightRegion(const TightRegion &R,
                          const SmallDenseSet<Register, 128> &Universe,
                          SmallDenseSet<Register, 64> &Spilled,
                          llvm::function_ref<bool(Register)> Eligible,
                          unsigned *NumRecolored = nullptr);

  /// Width-aware up-front pre-spiller. At each tight region's peak, spills frozen
  /// victims (kill-at-def store, reload at use) until the peak fits the allocatable
  /// pool; runs BEFORE color() so the coloring walk succeeds by construction. The
  /// victim universe spans ALL widths and victims are chosen WIDEST-FIRST,
  /// decrementing the region peak by each victim's real dword width, which is what
  /// lets it relieve regions dominated by wide tuples (SGPR/VGPR vreg_64/128/...).
  /// Models per-width availability honestly (pool/W aligned tuples per class) but
  /// does NOT claim to resolve aligned-tuple fragmentation (chi>omega): pool-fit is
  /// necessary, not always sufficient; placement residuals still flow to color()'s
  /// recovery. Returns true if anything was spilled.
  bool preSpillToLimitWidthAware(MachineFunction &MF);

  /// [Recovery classifier, Stage 1] Register file of a class for the recovery
  /// window. AGPR folds into VGPR so the file matches pressureOf(VGPR)'s unified
  /// count (SGPR classes -> SGPR, everything else -> VGPR).
  RegFile fileOf(const TargetRegisterClass *RC) const;

  /// Pick the register free at \p V's start that stays free LONGEST, and decide
  /// whether that free run is worth peeling. Returns false when nothing is free at
  /// the start, or when the run does not reach past \p V's first use (genuine
  /// over-pressure rather than fragmentation). On true, \p PR is the register and
  /// \p Bound the slot where it becomes occupied (>= V's end means free across all
  /// of V). This is the single split-across policy used by SelfSplit. Const —
  /// reads LIS/MRI/ColorMap.
  bool pickPeelableRun(Register V, MCRegister &PR, SlotIndex &Bound) const;

  /// True iff splitLiveRangeAt(V, SplitMI) will redirect at least one real use.
  /// The emitter creates its result vreg before discovering an empty split, so
  /// every recovery NoChange path must run this side-effect-free preflight.
  bool splitWouldRedirect(Register V, MachineInstr *SplitMI) const;

  /// Cross-file recovery strategy. First try to place \p R in its current
  /// class or in that class's sibling vector pool. If that fails, temporarily
  /// uncolor each colored crosser and try to move it from its ACTUAL physical
  /// pool to the sibling pool. A successful crosser move returns Changed so the
  /// recovery pipeline restarts from Web. Failed probes restore the old class and
  /// color exactly. Its internal forced-target probe bypasses current-home
  /// reclamation for a temporarily uncolored crosser.
  RecoveryResult tryCrossFileHome(Register R);

  /// [Stage 2] Cost of spilling candidate \p B to relieve region \p R.
  ///   Cost     : NReloads * Width. NReloads = 1 when B's uses are commonly
  ///              dominated and the shared reload hoists to their NCD; else one
  ///              reload per use, plus one per predecessor supplying a PHI use.
  ///   Width    : dwords the spill of \p Lanes MOVES (spilledSlots) — traffic, not
  ///              relief. The cumulative planner uses it only to price traffic;
  ///              area uses coveredSlots and actual freed RegionSlot sites.
  /// Every candidate is priceable: a reload only ever restores a register the
  /// reload point already needed. Traffic cost is a deterministic tie-breaker;
  /// cumulative oracle evaluation decides admission.
  struct SpillCost {
    unsigned Cost;
    unsigned Width;
  };
  SpillCost costOfSpilling(Register B, const TightRegion &R, LaneBitmask Lanes);

  struct SpillCandidateInput {
    Register V;
    LaneBitmask Lanes;
    bool CanRecolor = false;
  };

  struct AreaSpillAction {
    enum KindTy { Memory, Recolor };
    Register V;
    LaneBitmask Lanes;
    KindTy Kind = Memory;
    uint64_t Area = 0;
    unsigned Cost = 0;
    bool FootprintExact = false;
  };

  /// Build a cumulative non-emitting spill set over copies of \p Slots. Area
  /// orders the search; demandDeficiency remeasurement is authoritative. PHI
  /// webs are excluded until their multi-value footprint can be modeled as one
  /// atomic unit. Returns false without actions unless the virtual set reaches
  /// zero total deficiency.
  bool planAreaSpillSet(const TightRegion &R, ArrayRef<RegionSlot> Slots,
                        ArrayRef<SpillCandidateInput> Inputs,
                        unsigned RecolorBudget,
                        SmallVectorImpl<AreaSpillAction> &Actions,
                        unsigned *RemainingShort = nullptr);

  /// [Stage 3] Region RP-reduction driver. While tight regions remain, service
  /// the worst (highest Peak) by spilling the cheapest feasible crosser ACROSS
  /// that region (kill at R.Start so its register frees over R and the reload
  /// lands after R), then recompute regions globally. Returns true if any spill
  /// was performed (caller then re-colors from clean). Bounded by a round cap.
  bool reduceRegionPressure(MachineFunction &MF);

  /// SelfSplit recovery strategy: \p Failed is
  /// a long liver with no through-lane AND no live-through blocker to spill around
  /// (spillCrossLiver found nothing). Chop Failed into segments, each
  /// short enough that one physreg is free across it, coloring each into that reg.
  /// Only valid when Failed is POINT-FEASIBLE (some PR free at every slot); aborts
  /// (returns false -> caller memory-spills) if any slot has zero free PRs.
  /// \p FirstPR / \p FirstBound may provide the pick for the first piece. Pass
  /// an invalid \p FirstPR to have the strategy pick it from current state.
  RecoveryResult trySelfSplitColor(Register Failed, MCRegister FirstPR,
                                   SlotIndex FirstBound, Register &Remnant);

  /// Coloring-time recovery for one value \p Failed that color() could not place.
  /// Runs a mutation-aware fixpoint in priority order: Web, CrossFileHome,
  /// CrossLiver, SelfSplit. Any irreversible interference change restarts at Web
  /// with all predicates recomputed. A complete no-change iteration invokes the
  /// memory floor exactly once.
  bool recoverUncolorable(Register Failed);

  /// Honest terminal for a memory floor that cannot place its reload. Counts
  /// the values of
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

  /// Single linear scan over ColorMap for \p VI: the shared "collect" step of
  /// the split pipeline. ORs the register units of every colored occupant whose
  /// interval overlaps VI into \p OccupiedUnits. \p Overlappers is optional: when
  /// non-null it also collects (occupant vreg, its physreg) for each, as every
  /// current caller does. NOT cacheable across callers — they run in different
  /// phases with ColorMap mutated between.
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
  /// After rewriteOperands: erase a COPY whose source and destination were
  /// colored to the same physical register. lowerPHIs and eliminateRegSequences
  /// never emit such a copy; one that predates this pass — a live-in copy of a
  /// kernel argument, or a live-range split/narrow copy from the spill emitter —
  /// can only be recognized as a no-op once both its operands are physical.
  void eliminateIdentityCopies(MachineFunction &MF);
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
