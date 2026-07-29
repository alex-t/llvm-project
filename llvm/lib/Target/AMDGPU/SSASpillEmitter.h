//===-- SSASpillEmitter.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Exec-safe SSA spill/reload EMISSION mechanism, shared by the SSA
/// spiller pass and the SSA register allocator (coloring).
///
/// This is NOT a pass. It is the pure "how to spill a value safely" mechanism,
/// factored out of AMDGPUSSARegisterSpiller so that BOTH the spiller (which
/// decides what to spill up front, by pressure) AND coloring (which discovers a
/// value with no free register during assignment) can emit a store-at-def +
/// dominance-ordered reloads + inline SSA repair, with no EXEC drift.
///
/// Policy — *which* value to spill and *when* — stays with each caller (the
/// spiller uses NextUseAnalysis + RP tracking; coloring uses a no-free-register
/// signal). This class owns only the emission machinery and the per-value state
/// it needs (stack slots, store-at-def memo, reload cache, the SSA updater).
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SSASPILLEMITTER_H
#define LLVM_LIB_TARGET_AMDGPU_SSASPILLEMITTER_H

#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "VRegMaskPair.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLaneSSAUpdater.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include <memory>

namespace llvm {

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
/// per-function state: stack slots, store-at-def memo). Both the spiller pass
/// and coloring hold one and call spillOneVMP() to spill a value. Call
/// beginPass() before each register-file pass to (re)create the SSA updater and
/// select the file for reload-RP checks.
class SSASpillEmitter {
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

  // Per-spill caches (cleared at the start of each emitReloadsAndRepairSSA).
  DenseMap<std::pair<MachineBasicBlock *, VRegMaskPair>, Register>
      BlockReloadCache;                                // per-block reload dedup
  DenseMap<MachineBasicBlock *, unsigned> MaxRPCache;  // reload-hoist RP cache

  // Reload redefs create fresh vregs; callers exclude these from their own spill
  // candidate sets (a reload must not be immediately re-spilled). Written by the
  // emitter, read by policy via reloadedRegs().
  VRegMaskPairSet ReloadedRegs;

  // Current file being spilled (VGPR vs SGPR) — affects reload-RP checks. Set by
  // beginPass().
  bool IsVGPRPass = false;

  // RP ceiling for reload-hoist decisions in the current spill. Policy's budget
  // for the current file, threaded in per spillOneVMP() call (not a member of
  // this class's concern otherwise). Set at the top of spillOneVMP().
  unsigned CurRPLimit = 0;

  // Set transiently if a reload redef leaves SSA broken; inline repair clears it.
  bool SSAInvalidated = false;

  // PHI web members erased by the last spillPhiWeb() call (caller prunes ColorMap).
  SmallVector<Register, 32> LastWebErased;
  // Ground operands the last spillPhiWeb() stored (the driver marks them Spilled
  // so they are not re-selected and double-spilled as plain victims).
  SmallVector<Register, 32> LastWebGround;
  // RP relief at the region peak delivered by the last spillPhiWeb().
  unsigned LastWebPeakRelief = 0;

  // --- internal mechanism helpers (moved verbatim from the spiller) ---
  // Store \p VMP right after its def (EXEC full ⇒ captures all lanes). The store
  // half of spillOneVMP; its only caller. Returns the store instruction.
  MachineInstr *spillAtDefinition(VRegMaskPair VMP);
  int assignVirt2StackSlot(VRegMaskPair VMP);
  int createSpillSlot(const TargetRegisterClass *RC);
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
  unsigned getMaxRPForBlock(MachineBasicBlock *MBB);
  unsigned getMaxRPInBlockDownTo(MachineBasicBlock *MBB, MachineInstr *StopMI);
  // Max RP (current file) over the same-block span [DefMI, UseMI]; 0 if not same
  // block. Decides whether a same-block reaching reload spans an RP-tight region.
  unsigned maxRPBetween(MachineInstr *DefMI, MachineInstr *UseMI);
  bool canHoistReloadTo(MachineBasicBlock *NCD, MachineInstr *InsertPoint,
                        unsigned RPLimit, Register SpilledReg);
  bool walkPathsToUses(
      MachineBasicBlock *StartBB, Register SpilledReg,
      llvm::function_ref<bool(MachineBasicBlock *, MachineInstr *)> IsBad,
      bool StopOnBad = true) const;

public:
  SSASpillEmitter(MachineFunction &MF, LiveIntervals *LIS, SlotIndexes *Indexes,
                  MachineDominatorTree *DT, const MachineLoopInfo *MLI);

  /// (Re)create the SSA updater and select the file (\p IsVGPR) for reload-RP
  /// checks. Call before a fresh register-file pass. Does NOT clear the
  /// store-at-def memo or stack-slot map (those persist per function) nor
  /// ReloadedRegs (the caller controls that via clearReloadedRegs()).
  void beginPass(bool IsVGPR);

  /// Provide the effective kill block for \p SpillBB after hoisting out of all
  /// enclosing loops (outermost preheader), or \p SpillBB unchanged. The spiller
  /// needs this for candidate filtering before it picks a KillIdx; exposed so
  /// policy and emission agree on the hoist target.
  MachineBasicBlock *effectiveKillBB(MachineBasicBlock *SpillBB) const {
    return getEffectiveKillBB(SpillBB);
  }

  /// THE primitive both callers use. Spill \p VMP: store at its definition
  /// (EXEC-safe — all lanes captured while EXEC is full), free the register from
  /// \p KillIdx onward, place dominance-ordered reloads at the reachable uses,
  /// and repair SSA inline. \p RPLimit bounds reload-hoist decisions.
  void spillOneVMP(VRegMaskPair VMP, SlotIndex KillIdx, unsigned RPLimit);

  /// In-memory PHI-web coalescing. \p PhiResult must be defined by a PHI. Closes
  /// the transitive operand/result equivalence class (union-find over PHI edges),
  /// assigns ONE shared stack slot, stores every non-PHI operand at its def (in
  /// its predecessor — SSA-legal), reloads every EXTERNAL use of any web member
  /// from that slot (rolling window: one reg, dies after its use), and erases the
  /// now-dead PHIs. This is a MONOTONE wall-dissolution: it only REMOVES register
  /// pressure at the join, so reloads are NOT RP-gated (gating on pre-spill RP is
  /// a false-positive — it measures the very wall we are removing). Returns true
  /// if the web was coalesced; false (no-op) if PhiResult is not a PHI or the web
  /// has no ground operand. \p PeakSlot is the region's peak-RP slot: the number
  /// of erased web members live there (the true RP relief the caller should credit
  /// so it does not over-spill) is recorded in lastWebPeakRelief().
  bool spillPhiWeb(Register PhiResult, unsigned RPLimit, SlotIndex RegS,
                   SlotIndex RegE);

  /// Members erased by the last spillPhiWeb() (for the caller to prune ColorMap).
  ArrayRef<Register> lastWebErased() const { return LastWebErased; }

  /// Ground operands stored by the last spillPhiWeb() (driver marks them Spilled).
  ArrayRef<Register> lastWebGround() const { return LastWebGround; }

  /// RP relief at the peak slot delivered by the last spillPhiWeb(): count of
  /// erased PHI members whose live range covered the peak (ground-operand stores
  /// are isKill=false and stay in registers, so they do NOT relieve the peak).
  unsigned lastWebPeakRelief() const { return LastWebPeakRelief; }

  /// [region-rp-reduction Stage 2] Post-spill RP just BEFORE \p UseMI (per-use
  /// reload site). File via beginPass() (2-way, POC). See .cpp for accounting.
  unsigned reloadRPBeforeUse(const MachineInstr *UseMI) const;

  /// [Stage 2] Post-spill RP at the END of \p NCD (shared hoisted-reload site).
  /// Valid for an empty NCD. Covers the NCD-block RP that canHoistReloadTo skips.
  unsigned reloadRPAtBlockEnd(const MachineBasicBlock *NCD) const;

  /// [Stage 2] Public forwarder to canHoistReloadTo: can \p B's shared reload
  /// hoist to \p NCD (reload at NCD end) within \p RPLimit on every NCD->use path?
  bool canHoistReload(MachineBasicBlock *NCD, unsigned RPLimit, Register B) {
    return canHoistReloadTo(NCD, /*InsertPoint=*/nullptr, RPLimit, B);
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
  /// (dead copy removed). \p SplitPt must be a non-PHI, mid-block position.
  Register splitLiveRangeAt(Register V, MachineBasicBlock::iterator SplitPt);

  /// Reload vregs created so far (fresh names from SSA repair). Policy layers
  /// subtract these from their spill-candidate sets so a reload is never
  /// immediately re-spilled.
  const VRegMaskPairSet &reloadedRegs() const { return ReloadedRegs; }
  void clearReloadedRegs() { ReloadedRegs.clear(); }

  /// True if any reload path left SSA transiently broken and it was not repaired
  /// (defensive; normally false — inline repair clears it).
  bool ssaInvalidated() const { return SSAInvalidated; }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SSASPILLEMITTER_H
