//===-- SSASpillEmitter.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Exec-safe SSA spill/reload EMISSION mechanism, used by the SSA
/// register allocator.
///
/// This is NOT a pass. It is the pure "how to spill a value safely" mechanism:
/// a store-at-def plus dominance-ordered reloads plus inline SSA repair, with
/// no EXEC drift. Both the up-front spill planner and coloring (which discovers
/// a value with no free register during assignment) emit through it.
///
/// Policy — *which* value to spill and *when* — stays with the caller. This
/// class owns only the emission machinery and the per-value state it needs
/// (stack slots, store-at-def memo, reload cache, the SSA updater).
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SSASPILLEMITTER_H
#define LLVM_LIB_TARGET_AMDGPU_SSASPILLEMITTER_H

#include "GCNRegPressure.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "VRegMaskPair.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLaneSSAUpdater.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include <functional>
#include <memory>

namespace llvm {

class SSAForensicReporter;

// A spill store emitted by storeRegToStackSlot. Classify via the Spill TSFlag
// plus mayStore rather than an opcode list: this covers every register file
// (SGPR, VGPR, AGPR, and the AGPR-or-VGPR "AV" classes on gfx90a+) and every
// width. Shared by the spiller (policy skips these on its walk) and the emitter.
inline bool isSpillInstr(const MachineInstr *MI) {
  return SIInstrInfo::isSpill(MI->getDesc()) && MI->mayStore();
}

// A reload emitted by loadRegFromStackSlot. See isSpillInstr: classify via the
// Spill TSFlag plus mayLoad so AGPR and AGPR-or-VGPR ("AV") reloads are
// recognized too.
inline bool isReloadInstr(const MachineInstr *MI) {
  return SIInstrInfo::isSpill(MI->getDesc()) && MI->mayLoad();
}

/// True if \p MI still reads \p SpilledVMP with an overlapping lane mask (false
/// once SSA repair has rewritten the use). Pure predicate — shared by the
/// spiller's marker path and the emitter's reload placement.
bool usesSpilledVMP(const MachineInstr *MI, VRegMaskPair SpilledVMP,
                    const SIRegisterInfo *TRI, const MachineRegisterInfo *MRI);

/// Dom-group: head instruction dominates a list of other uses. (Moved here with
/// the reload machinery that builds and consumes it.)
class DomGroup {
  MachineInstr *Head;
  SmallVector<MachineInstr *, 4> DominatedUses;

public:
  DomGroup(MachineInstr *MI) : Head(MI) {}
  MachineInstr *getHead() const { return Head; }
  const SmallVector<MachineInstr *, 4> &getDominatedUses() const {
    return DominatedUses;
  }
  void addDominatedUse(MachineInstr *MI) { DominatedUses.push_back(MI); }
  void promoteHead(MachineInstr *NewHead) {
    DominatedUses.push_back(Head);
    Head = NewHead;
  }
  size_t size() const { return 1 + DominatedUses.size(); }
};

/// One ground (non-PHI) operand of a PHI web, on a specific PHI incoming edge.
/// The store + color-pinning COPY belongs at THAT predecessor's end (the classic
/// PHI-elimination edge), not at the ground def. A PHI operand may name a
/// SUB-REGISTER of a wider vreg (a 32-bit PHI reading %wide.subN); SubReg records
/// which lane to store (the slot is sized by the narrow PHI result).
struct GroundEdge {
  Register Reg;
  unsigned SubReg;
  MachineBasicBlock *Pred;
  MachineInstr *Phi; // the PHI whose operand reads Reg on this edge
  unsigned OpIdx;    // operand index of Reg in Phi (block operand is OpIdx+1)
};

/// A closed PHI web: the transitive PHI operand/result equivalence class, its
/// ground operands, and per-edge store sites. Detection POLICY is the RA's
/// (closePhiWeb builds this, runs the shared-slot soundness gate); the emitter's
/// spillPhiWeb consumes it as pure mechanics. An INVALID web (no root, no ground
/// operand, or interfering ground operands) means "not a spillable web" — the
/// caller falls back to a plain per-value spill.
struct PhiWeb {
  Register Root;                           // PHI result the web is keyed on
  SmallSetVector<Register, 32> PhiMembers; // PHIs in the web (candidates to erase)
  SmallSetVector<Register, 32> GroundOps;  // non-PHI operands (unique)
  SmallVector<GroundEdge, 32> GroundEdges; // every edge (no dedup)
  bool valid() const { return Root.isValid() && !GroundOps.empty(); }
};

/// SpillInfo: one value's spill decision with pre-built dom-groups.
struct SpillInfo {
  VRegMaskPair SpilledVMP;
  SlotIndex KillIdx;
  int FrameIndex;
  SmallVector<DomGroup, 4> DomGroups;
  // PHI uses of the spilled value, kept out of the dominance-merged DomGroups
  // (a PHI reads on the predecessor edge, not at its own slot). Reloaded via
  // insertReloadForUse's per-predecessor path.
  SmallVector<MachineInstr *, 4> PhiUses;
};

/// Exec-safe SSA spill/reload emitter. Construct once per function (it caches
/// per-function state: stack slots, store-at-def memo). The allocator holds one
/// and calls spillOneVMP() to spill a value. Call
/// beginPass() before each register-file pass to (re)create the SSA updater,
/// select the file mechanics, and bind the demand oracle for the pool.
class SSASpillEmitter {
public:
  /// The allocator's demand oracle for one register pool: given a live set,
  /// how many of its values have no placement in that pool. Injected rather
  /// than reimplemented so spill emission and spill planning share one
  /// pressure model.
  using DemandFn = std::function<unsigned(const GCNRPTracker::LiveRegSet &)>;

  /// Pre-emission register residency at caller-supplied instruction sites.
  /// Exact is false when dominance-sharing between reloads can only be decided
  /// after earlier reload redefs have been inserted and LiveIntervals recomputed.
  struct SpillFootprint {
    SmallBitVector ResidentSlots;
    bool Exact = true;
  };

private:
  // Analyses / target info (borrowed, not owned).
  MachineFunction &MF;
  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;
  const MachineLoopInfo *MLI;
  MachineRegisterInfo *MRI;
  MachineFrameInfo *FrameInfo;
  LiveIntervals *LIS;
  SlotIndexes *Indexes;
  MachineDominatorTree *DT;

  // SSA repair engine (reaching-VNI reconstruction + CFG-reachability queries).
  std::unique_ptr<MachineLaneSSAUpdater> SSAUpdater;

  // Per-function emission state (persists across the file passes so a value is
  // stored at its def only once).
  DenseMap<VRegMaskPair, int> Virt2StackSlotMap;      // value -> stack slot
  DenseMap<VRegMaskPair, MachineInstr *> StoredAtDefinition; // store-at-def memo

  struct FootprintUseAnalysis {
    SmallVector<DomGroup, 4> Groups;
    SmallVector<MachineInstr *, 4> PhiUses;
  };
  DenseMap<std::pair<MachineInstr *, VRegMaskPair>, FootprintUseAnalysis>
      FootprintUseCache;

  // Per-spill caches (cleared at the start of each emitReloadsAndRepairSSA).
  DenseMap<std::pair<MachineBasicBlock *, VRegMaskPair>, Register>
      BlockReloadCache;                                // per-block reload dedup
  // reload-hoist deficiency cache
  DenseMap<MachineBasicBlock *, unsigned> DeficiencyCache;

  void invalidateFootprintUseCache() { FootprintUseCache.clear(); }
  const FootprintUseAnalysis &
  getFootprintUseAnalysis(VRegMaskPair VMP, SlotIndex KillIdx);

  // Reload redefs create fresh vregs; callers exclude these from their own spill
  // candidate sets (a reload must not be immediately re-spilled). Written by the
  // emitter, read by policy via reloadedRegs().
  VRegMaskPairSet ReloadedRegs;

  // Current file being spilled: selects vector vs scalar spill MECHANICS (AGPR
  // is a vector file, so it too sets this). NOT a pool selector — the pool
  // lives in PoolDeficiency below. Set by beginPass().
  bool IsVGPRPass = false;

  // The allocator's width-aware demand oracle, bound to the pool being spilled.
  // Returns how many live values have no placement in that pool, so non-zero
  // means a reload put there has nowhere to land. Reload-hoist decisions ask
  // this instead of comparing a scalar count to a budget: the emitter must ask
  // the same question the planner does, and its own count charged an AGPR value
  // against the arch-VGPR total. Set by beginPass().
  DemandFn PoolDeficiency;
  bool ReloadEveryUse = false;

  // Set transiently if a reload redef leaves SSA broken; inline repair clears it.
  bool SSAInvalidated = false;

  // Forensic reporter (observer, borrowed, may be null). When non-null and
  // enabled, spillAtDefinition / phi-web stores (E14) and reload emission (E15)
  // record observable facts. NEVER mutated by this class beyond recording.
  SSAForensicReporter *Reporter = nullptr;

  // Total SGPR lanes spilled this function (each spilled SGPR value's dword
  // width), counted only when SGPR spills lower to VGPR lanes
  // (TRI->spillSGPRToVGPR()). The RA reads this after the SGPR allocation stage
  // to reserve ceil(lanes / wavesize) VGPRs for the downstream WWM SGPR-spill
  // lowering, so the VGPR stage does not consume the whole file. Reset per
  // function by the RA.
  unsigned NumSGPRSpillLanes = 0;

  // PHI web members erased by the last spillPhiWeb() call (caller prunes ColorMap).
  SmallVector<Register, 32> LastWebErased;
  // Ground operands the last spillPhiWeb() stored (the driver marks them Spilled
  // so they are not re-selected and double-spilled as plain victims).
  SmallVector<Register, 32> LastWebGround;

  // --- internal mechanism helpers (moved verbatim from the spiller) ---
  // Store \p VMP right after its def (EXEC full ⇒ captures all lanes). The store
  // half of spillOneVMP; its only caller. Returns the store instruction.
  MachineInstr *spillAtDefinition(VRegMaskPair VMP);
  int assignVirt2StackSlot(VRegMaskPair VMP);
  int createSpillSlot(const TargetRegisterClass *RC);
  void buildDomGroups(ArrayRef<MachineInstr *> Uses,
                      SmallVectorImpl<DomGroup> &Groups);
  void buildDomGroupsForSpill(SpillInfo &Info);
  void emitReloadsAndRepairSSA(SpillInfo &Info);
  std::pair<Register, MachineInstr *>
  getOrCreateReloadInBlock(MachineBasicBlock *BB, VRegMaskPair SpilledVMP,
                           MachineInstr *InsertBefore = nullptr,
                           LaneBitmask ReloadMask = LaneBitmask::getAll());
  bool insertReloadForUse(MachineInstr *UseMI, VRegMaskPair SpilledVMP,
                          MachineBasicBlock *KillBB);
  MachineBasicBlock *getEffectiveKillBB(MachineBasicBlock *SpillBB) const;
  std::pair<MachineBasicBlock *, MachineInstr *>
  adjustReloadForLoop(MachineBasicBlock *ReloadBB, MachineInstr *InsertBeforeMI,
                      MachineBasicBlock *KillBB, Register SpilledReg);
  std::pair<MachineBasicBlock *, MachineInstr *>
  reloadPlacementForUse(MachineInstr *UseMI, MachineBasicBlock *KillBB,
                        Register SpilledReg);
  unsigned maxDeficiencyForBlock(MachineBasicBlock *MBB);
  unsigned maxDeficiencyInBlockDownTo(MachineBasicBlock *MBB,
                                      MachineInstr *StopMI);
  // Max deficiency over the same-block span [DefMI, UseMI]; 0 if not same
  // block. Non-zero means a same-block reaching reload spans a point where the
  // pool cannot place everything live.
  unsigned maxDeficiencyBetween(MachineInstr *DefMI, MachineInstr *UseMI);
  // \p SpanLoop, when non-null, is a loop the hoisted reload stays live across
  // on every iteration; blocks inside it are measured in full rather than only
  // down to the first use.
  bool canHoistReloadTo(MachineBasicBlock *NCD, MachineInstr *InsertPoint,
                        Register SpilledReg,
                        const MachineLoop *SpanLoop = nullptr);
  bool ordinaryVALUUseIsExecSafe(MachineInstr *UseMI) const;
  bool walkPathsToUses(
      MachineBasicBlock *StartBB, Register SpilledReg,
      llvm::function_ref<bool(MachineBasicBlock *, MachineInstr *)> IsBad,
      bool StopOnBad = true) const;

public:
  SSASpillEmitter(MachineFunction &MF, LiveIntervals *LIS, SlotIndexes *Indexes,
                  MachineDominatorTree *DT, const MachineLoopInfo *MLI);

  /// (Re)create the SSA updater, select the file mechanics (\p IsVGPR: vector
  /// vs scalar), and bind \p Demand as the oracle for the pool being spilled.
  /// Call before a fresh register-file pass. Does NOT clear the store-at-def
  /// memo or stack-slot map (those persist per function) nor ReloadedRegs (the
  /// caller controls that via clearReloadedRegs()).
  void beginPass(bool IsVGPR, DemandFn Demand);
  void setReloadEveryUse(bool Enable) { ReloadEveryUse = Enable; }

  /// Attach the forensic reporter (observer; may be null). Records spill/reload
  /// facts (E14/E15) when set and enabled. Does not take ownership.
  void setReporter(SSAForensicReporter *R) { Reporter = R; }

  /// Provide the effective kill block for \p SpillBB after hoisting out of all
  /// enclosing loops (outermost preheader), or \p SpillBB unchanged. The spiller
  /// needs this for candidate filtering before it picks a KillIdx; exposed so
  /// policy and emission agree on the hoist target.
  MachineBasicBlock *effectiveKillBB(MachineBasicBlock *SpillBB) const {
    return getEffectiveKillBB(SpillBB);
  }

  /// Return true when the store-at-def and every ordinary reload-before-use
  /// position are legal. PHI-edge reloads are exempt from the EXEC-stability
  /// rule because each predecessor executes under that edge's lane mask.
  bool canSpill(VRegMaskPair VMP) const;

  /// THE primitive both callers use. Spill \p VMP: store at its definition
  /// (EXEC-safe — all lanes captured while EXEC is full), free the register from
  /// \p KillIdx onward, place dominance-ordered reloads at the reachable uses,
  /// and repair SSA inline. Reload-hoist decisions consult the pool oracle
  /// bound by beginPass().
  void spillOneVMP(VRegMaskPair VMP, SlotIndex KillIdx);

  /// Model which of \p Sites in the block-local \p RegionBB still require
  /// register residency after spilling \p VMP at \p KillIdx. Store-at-def,
  /// loop-adjusted reload placement, and PHI predecessor-edge reloads use the
  /// same placement helpers as emission.
  /// Dominance-sharing is conservatively over-approximated because its exact
  /// frontier is created incrementally by emitted reload redefs.
  SpillFootprint modelSpillFootprint(VRegMaskPair VMP, SlotIndex KillIdx,
                                     MachineBasicBlock *RegionBB,
                                     ArrayRef<MachineInstr *> Sites);

  /// In-memory PHI-web coalescing of an ALREADY-CLOSED, feasible web \p Web
  /// (detection, the shared-slot soundness gate, AND the reload-feasibility gate
  /// are the RA's job — see closePhiWeb / webReloadFeasible). Pure MECHANICS:
  /// assign ONE shared stack slot, store every non-PHI operand at its def (in its
  /// predecessor — SSA-legal), reload every EXTERNAL use of any web member from
  /// that slot (rolling window: one reg, dies after its use), and erase the
  /// now-dead PHIs. A MONOTONE wall-dissolution: it only REMOVES register pressure
  /// at the join. \p Web must be valid() and proven feasible; this call cannot
  /// fail. Ground operands are stored into the shared slot on their PHI edges.
  /// Differently-colored SGPRs MAY share the slot: the SILowerSGPRSpills lane
  /// assert is a per-store WIDTH check (reg-width <= slot lanes), NOT a color/count
  /// check, and the web's non-interference gate already proves the operands are
  /// never simultaneously live — so sequential writelanes into the shared lane are
  /// correct. No shared color is forced. A sub-register PHI operand is
  /// COPY-extracted to slot width first; that fresh short-lived vreg is colored via
  /// \p ColorFreshVReg (it lives only [copy, store], so a free reg always exists).
  void spillPhiWeb(const PhiWeb &Web,
                   llvm::function_ref<void(Register)> ColorFreshVReg);

  /// Members erased by the last spillPhiWeb() (for the caller to prune ColorMap).
  ArrayRef<Register> lastWebErased() const { return LastWebErased; }

  /// Ground operands stored by the last spillPhiWeb() (driver marks them Spilled).
  ArrayRef<Register> lastWebGround() const { return LastWebGround; }

  // reloadRPBeforeUse / reloadRPAtBlockEnd moved to the RA (feasibility policy).

  /// [Stage 2] Public forwarder to canHoistReloadTo: can \p B's shared reload
  /// hoist to \p NCD (reload at NCD end) with a placement on every NCD->use path?
  bool canHoistReload(MachineBasicBlock *NCD, Register B) {
    return canHoistReloadTo(NCD, /*InsertPoint=*/nullptr, B);
  }

  /// After a partial spill leaves \p WideVReg with only its \p RemnantMask lanes
  /// live (a contiguous sub-register named by \p SubIdx), extract that remnant
  /// into a fresh narrow vreg so WideVReg vacates its aligned tuple. Inserts
  /// `%new:SubRC = COPY WideVReg.SubIdx` right after WideVReg's def and uses the
  /// SSA updater's rewriteDominatedUses to redirect the remnant-lane uses to
  /// %new (dominance- and subreg-policy-correct, composing REG_SEQUENCEs where a
  /// use spans the split). Returns true if the remnant was narrowed.
  bool narrowRemnantToNewReg(Register WideVReg, unsigned SubIdx,
                             LaneBitmask RemnantMask);

  /// Live-range split (Hack-compatible, pre-coloring): insert
  /// `%new = COPY \p V` just before \p SplitPt and redirect every use of \p V
  /// at-or-after the copy (whose reaching value is the one live at the split) to
  /// %new. \p V then ends at the copy; the two halves no longer interfere across
  /// SplitPt, so coloring may place them in different physregs (reopening an
  /// aligned through-lane mid-life). Stays in SSA, so the interference graph
  /// stays chordal. Returns %new, or a null Register if nothing was redirected
  /// (dead copy removed). A prologue or terminator-sequence anchor is clamped to
  /// the nearest legal block-body boundary.
  Register splitLiveRangeAt(Register V, MachineBasicBlock::iterator SplitPt);

  /// Reload vregs created so far (fresh names from SSA repair). Policy layers
  /// subtract these from their spill-candidate sets so a reload is never
  /// immediately re-spilled.
  const VRegMaskPairSet &reloadedRegs() const { return ReloadedRegs; }
  void clearReloadedRegs() { ReloadedRegs.clear(); }

  /// Total SGPR lanes spilled so far (see NumSGPRSpillLanes). The RA reads this
  /// after the SGPR allocation stage to reserve VGPRs for WWM SGPR-spill
  /// lowering, and clears it per function.
  unsigned numSGPRSpillLanes() const { return NumSGPRSpillLanes; }
  void clearSGPRSpillLanes() { NumSGPRSpillLanes = 0; }

  /// True if any reload path left SSA transiently broken and it was not repaired
  /// (defensive; normally false — inline repair clears it).
  bool ssaInvalidated() const { return SSAInvalidated; }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SSASPILLEMITTER_H
