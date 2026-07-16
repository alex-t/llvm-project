//===--------------- AMDGPUSSARegisterSpiller.h  -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief SSA-aware Register Spiller for AMDGPU
///
/// This pass implements register spilling using the MachineLaneSSAUpdater
/// to maintain SSA form. Based on the approach from:
/// "Register Spilling and Live-Range Splitting for SSA-Form Programs"
/// Matthias Braun and Sebastian Hack, CC'09
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSSAREGISTERSPILLER_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSSAREGISTERSPILLER_H

#include "AMDGPUNextUseAnalysis.h"
#include "GCNRegPressure.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "VRegMaskPair.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLaneSSAUpdater.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/SlotIndexes.h"

namespace llvm {

/// Dom-group: head instruction dominates a list of other uses.
/// Built once at spill decision time, reused during processing.
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

/// SpillInfo: captures spill decision with pre-built dom-groups.
struct SpillInfo {
  VRegMaskPair SpilledVMP;
  SlotIndex KillIdx;
  int FrameIndex;
  SmallVector<DomGroup, 4> DomGroups;
};

class AMDGPUSSARegisterSpiller : public MachineFunctionPass {
  const SIRegisterInfo *TRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  const MachineLoopInfo *MLI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  const SIMachineFunctionInfo *MFI = nullptr;
  MachineFrameInfo *FrameInfo = nullptr;
  LiveIntervals *LIS = nullptr;
  SlotIndexes *Indexes = nullptr;
  MachineDominatorTree *DT = nullptr;

  // SSA updater: reaching-VNI SSA repair and CFG-reachability queries.
  std::unique_ptr<MachineLaneSSAUpdater> SSAUpdater;

  // Register pressure tracker (reused throughout the pass)
  std::unique_ptr<GCNUpwardRPTracker> RPTracker;

  // Next use analysis for spill candidate selection
  AMDGPUNextUseAnalysis::Result *NU = nullptr;

  // Stack slot management
  DenseMap<VRegMaskPair, int> Virt2StackSlotMap;

  // Track registers that have been stored at definition (to avoid EXEC drift)
  // When a register is selected for spilling, we store it right after
  // definition (when EXEC is full), then mark it dead at the "real spill" point
  // using pruneValue()
  DenseMap<VRegMaskPair, MachineInstr *> StoredAtDefinition;

  VRegMaskPairSet ReloadedRegs;

  // Register pressure limits (set during processFunction)
  unsigned VGPRLimit = 0;
  unsigned SGPRLimit = 0;

  // Second RP dimension: values that cross ANY call are pinned to callee-saved
  // registers for their whole range, so a per-point "preserved-RP" over the
  // pinned set must fit the callee-saved capacity k_cs (per file). See the
  // preserved-RP gate in processFunction and ACL_Pass_and_CallSite_Capacity.
  DenseSet<Register> PinnedVRegs;           // vregs crossing any call
  unsigned VGPRPreservedCap = 0;            // k_cs for VGPR file
  unsigned SGPRPreservedCap = 0;            // k_cs for SGPR file
  unsigned PreservedLimit = 0;              // k_cs for the current pass's file

  /// Classify PinnedVRegs (crosses any call) and compute VGPR/SGPRPreservedCap
  /// (min preserved allocatable count over the function's calls). Called before
  /// the first spill pass and re-run by the fixpoint loop after each pass, since
  /// spilling creates reload vregs that themselves cross calls (and are pinned).
  void computePinnedAndCap(MachineFunction &MF);

  /// True max width-weighted clique over the *current* PinnedVRegs of this
  /// pass's file, via an endpoint sweep over their live intervals. This is the
  /// authoritative preserved-RP the allocator will face after spilling; the
  /// fixpoint loop re-runs the pass while it exceeds PreservedLimit.
  unsigned maxPreservedClique() const;

  /// preserved-RP at \p MI: 32-bit-slot count of pinned vregs of the current
  /// pass's file live across \p MI.
  unsigned computePreservedRP(const MachineInstr &MI);

  // Current pass type for RP calculation.
  bool IsVGPRPass = false;

  // Set transiently when a reload redef breaks SSA; inline reconstruction
  // restores it before the pass returns (see emitReloadsAndRepairSSA).
  bool SSAInvalidated = false;

  // Reload optimizer: cached max RP per block (cleared per spill analysis)
  DenseMap<MachineBasicBlock *, unsigned> MaxRPCache;

  // Track reloads per block to avoid duplicates. Keyed by (block, reloaded
  // VReg+mask) so a narrow sub-slice reload and a wider reload of the same vreg
  // in one block are distinct cache entries.
  DenseMap<std::pair<MachineBasicBlock *, VRegMaskPair>, Register>
      BlockReloadCache;

  // TODO: Add tracking for spilled/reloaded registers if needed for
  // verification

  /// Inserts a virtual spill marker at the given position.
  void insertVirtualSpillMarker(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator I,
                                VRegMaskPair VMP);

  /// Returns a stack slot for the given VRegMaskPair, creating one if
  /// needed.
  int assignVirt2StackSlot(VRegMaskPair VMP);

  /// Creates a spill slot for the given register class.
  int createSpillSlot(const TargetRegisterClass *RC);

  /// Converts RPTracker's LiveRegSet to VRegMaskPairSet.
  VRegMaskPairSet
  convertLiveRegs(const GCNRPTracker::LiveRegSet &LiveRegs) const;

  /// Processes the entire function for one register class (SGPR or VGPR).
  /// This is called twice: first for SGPRs, then for VGPRs.
  /// Uses IsVGPRPass class member (set before calling).
  bool processFunction(MachineFunction &MF, unsigned RPLimit);

  /// Validates that final register pressure is within limits after all
  /// spilling. This is a temporary validation check until we properly handle
  /// clean path reloads.
  void validateFinalRegisterPressure(MachineFunction &MF, unsigned RPLimit,
                                     bool IsVGPR);

  /// Sorts the register set by next-use distance (descending).
  /// Registers with longer next-use distances are moved to the back.
  void sortRegSetByNextUse(MachineBasicBlock &MBB,
                           MachineBasicBlock::reverse_iterator I,
                           VRegMaskPairSet &Active);

  /// Spill selection algorithm: Selects which VRegMaskPairs to spill based on
  /// Belady's algorithm to reduce register pressure to the limit.
  ///
  /// This implements the core spill selection algorithm:
  /// 1. Calculate SizeToSpill = CurRP - RPLimit
  /// 2. Sort Active set by next-use distance (longest last)
  /// 3. Greedily select registers from the back until we've spilled enough
  /// 4. For registers larger than needed, split by subregister
  ///
  /// NOTE: This method only selects which registers to spill. The actual
  /// spill instruction emission is done by spillAndReload().
  ///
  /// Returns the set of VRegMaskPairs that were selected for spilling.
  VRegMaskPairSet getVMPsToSpill(MachineBasicBlock &MBB,
                                 MachineBasicBlock::reverse_iterator I,
                                 VRegMaskPairSet &Active, unsigned CurRP,
                                 unsigned RPLimit);

  /// High-level orchestration: Performs atomic spill+reload+SSA repair per
  /// register to keep MIR valid.
  ///
  /// IMPORTANT: Each register is completely spilled+reloaded+repaired before
  /// moving to the next to avoid invalid MIR state.
  ///
  /// Workflow:
  /// 1. Call getVMPsToSpill() to select and emit spill instructions
  /// 2. For each spilled VRegMaskPair, call emitReloadsAndRepairSSA()
  /// 3. MachineLaneSSAUpdater handles LiveInterval updates during SSA repair
  /// 4. Reset RPTracker after modifications
  ///
  /// Returns true if any spilling was performed.
  bool spillAndReload(MachineBasicBlock &MBB,
                      MachineBasicBlock::reverse_iterator I,
                      VRegMaskPairSet &Active, unsigned CurRP,
                      unsigned RPLimit);

  /// Stores register to stack slot right after its definition (when EXEC is
  /// full). This avoids EXEC drift issues by ensuring all lanes are stored
  /// before any divergent control flow can modify EXEC. Returns the store
  /// instruction.
  MachineInstr *spillAtDefinition(VRegMaskPair VMP);

  /// If the register was already stored at definition, uses pruneValue() to
  /// mark it dead at this point instead of emitting a new store.
  void spillBefore(MachineBasicBlock &MBB,
                   MachineBasicBlock::iterator InsertBefore, VRegMaskPair VMP);

  /// Emits a spill instruction before the given position (reverse iterator).
  /// This is used during backward traversal in processFunction().
  void spillBefore(MachineBasicBlock &MBB,
                   MachineBasicBlock::reverse_iterator I, VRegMaskPair VMP);

  /// Build dom-groups for a register at spill decision time.
  void buildDomGroupsForSpill(SpillInfo &Info);

  /// Emits reloads and repairs SSA using IDF-first PHI insertion.
  /// Uses pre-built dom-groups from SpillInfo.
  void emitReloadsAndRepairSSA(SpillInfo &Info);

  /// Finalize live intervals after all reloads and use rewriting.
  void finalizeLiveIntervals(Register SpilledReg);

  /// Get or create reload in a block.
  /// If InsertBefore is provided, inserts before that instruction.
  /// Otherwise inserts at block end (before terminator) and caches the result.
  /// Returns {ReloadReg, ReloadMI}. ReloadMI is nullptr if using cached reload.
  /// \p ReloadMask selects which of SpilledVMP's lanes to actually reload now
  /// (default: all of them). The stack slot is always the full SpilledVMP slot;
  /// only the reloaded sub-slice (dest subreg, class, and in-slot offset) is
  /// narrowed, so a use that reads a few lanes does not pull the whole tuple
  /// back into registers.
  std::pair<Register, MachineInstr *>
  getOrCreateReloadInBlock(MachineBasicBlock *BB, VRegMaskPair SpilledVMP,
                           MachineInstr *InsertBefore = nullptr,
                           LaneBitmask ReloadMask = LaneBitmask::getAll());

  /// Insert reload for a use instruction. For PHI uses, inserts in predecessor
  /// blocks. For non-PHI uses, handles loop adjustment.
  bool insertReloadForUse(MachineInstr *UseMI, VRegMaskPair SpilledVMP,
                          MachineBasicBlock *KillBB);

  /// Debug helper: dumps a register set to dbgs().
  void dumpRegSet(const VRegMaskPairSet &Regs) const;

  // ============================================================================
  // Divergent Path Optimization Helpers
  // ============================================================================

  /// Walk BFS from \p StartBB through blocks where \p SpilledReg is live.
  /// For each block, finds first use of SpilledReg (if any) and calls IsBad.
  /// \param StopOnBad If true (default), return immediately when IsBad returns
  /// true.
  ///                  If false, continue walking all paths.
  /// \returns true if all paths OK, false if any bad found.
  bool walkPathsToUses(
      MachineBasicBlock *StartBB, Register SpilledReg,
      llvm::function_ref<bool(MachineBasicBlock *, MachineInstr *)> IsBad,
      bool StopOnBad = true) const;

  /// Check if the given instruction still uses the spilled register with
  /// overlapping lane mask. Returns false if the use was rewritten by SSA
  /// repair.
  bool usesSpilledVMP(const MachineInstr *MI, VRegMaskPair SpilledVMP) const;

  // ============================================================================
  // Reload Optimizer
  // ============================================================================

  /// Computes and caches maximum register pressure within a basic block.
  unsigned getMaxRPForBlock(MachineBasicBlock *MBB);

  /// Computes max RP in block walking down from StopMI to block start.
  unsigned getMaxRPInBlockDownTo(MachineBasicBlock *MBB, MachineInstr *StopMI);

  /// Checks if reload can be hoisted to NCD by walking paths and checking RP.
  /// InsertPoint is the reload insertion point in NCD (nullptr if no use in
  /// NCD).
  bool canHoistReloadTo(MachineBasicBlock *NCD, MachineInstr *InsertPoint,
                        unsigned RPLimit, Register SpilledReg);

  // ============================================================================
  // Loop-Aware Spilling Helpers
  // ============================================================================

  /// Get the effective kill block after hoisting out of all enclosing loops.
  /// If SpillBB is inside loop(s), returns the outermost loop's preheader.
  /// Otherwise returns SpillBB unchanged.
  MachineBasicBlock *getEffectiveKillBB(MachineBasicBlock *SpillBB) const;

  /// Adjust reload placement for loop-aware spilling.
  /// If ReloadBB is in a loop but KillBB is outside, returns the preheader
  /// (only if it doesn't cause RP to exceed limit on the path).
  /// Returns the adjusted (ReloadBB, InsertBeforeMI) pair.
  std::pair<MachineBasicBlock *, MachineInstr *>
  adjustReloadForLoop(MachineBasicBlock *ReloadBB, MachineInstr *InsertBeforeMI,
                      MachineBasicBlock *KillBB, Register SpilledReg);

  /// Accounting only: compute how many VGPRs SGPR-spill-to-lane will consume,
  /// so Pass 2 (VGPR) sees the correct budget. Does NOT lower the pseudos and
  /// does NOT reserve physregs — physical lane reservation and the actual
  /// writelane/readlane materialization happen later, at SGPR coloring time
  /// (SuperReg must be physical for SGPRSpillBuilder).
  /// Called between Pass 1 (SGPR) and Pass 2 (VGPR).
  unsigned countSGPRSpillVGPRs(MachineFunction &MF);

public:
  static char ID;

  AMDGPUSSARegisterSpiller() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU SSA Register Spiller";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<SlotIndexesWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<AMDGPUNextUseAnalysisWrapper>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUSSAREGISTERSPILLER_H
