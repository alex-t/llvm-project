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
#include "SSASpillEmitter.h"
#include "VRegMaskPair.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/SlotIndexes.h"

namespace llvm {

// DomGroup and SpillInfo now live in SSASpillEmitter.h (the reload machinery
// that builds and consumes them moved there).

class AMDGPUSSARegisterSpiller : public MachineFunctionPass {
  const SIRegisterInfo *TRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  const MachineLoopInfo *MLI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  const SIMachineFunctionInfo *MFI = nullptr;
  LiveIntervals *LIS = nullptr;
  SlotIndexes *Indexes = nullptr;
  MachineDominatorTree *DT = nullptr;

  // Exec-safe spill/reload emission mechanism (store-at-def + dominance reloads
  // + inline SSA repair). Owns the SSA updater, stack slots, store-at-def memo,
  // reload caches, and the reloaded-vreg set. Created per function in
  // runOnMachineFunction; beginPass() selects the file before each spill pass.
  std::unique_ptr<SSASpillEmitter> Emitter;

  // Register pressure tracker (reused throughout the pass)
  std::unique_ptr<GCNUpwardRPTracker> RPTracker;

  // Next use analysis for spill candidate selection
  AMDGPUNextUseAnalysis::Result *NU = nullptr;

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

  /// Inserts a virtual spill marker at the given position. Test-only
  /// (-amdgpu-ssa-spill-markers, default off); only the Belady walk emits these,
  /// so this stays with the spiller policy rather than the emitter.
  void insertVirtualSpillMarker(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator I,
                                VRegMaskPair VMP);

  /// Converts RPTracker's LiveRegSet to VRegMaskPairSet.
  VRegMaskPairSet
  convertLiveRegs(const GCNRPTracker::LiveRegSet &LiveRegs) const;

  /// True if \p RC belongs to the file the current pass (IsVGPRPass) spills, in
  /// the SAME grouping the pressure metric (getVGPRNum/getSGPRNum) uses. For the
  /// VGPR pass this is VGPR *and* AGPR *and* the AGPR-or-VGPR vector-super
  /// ("av_") classes — getVGPRNum folds all of them into the VGPR count, so the
  /// spiller must be able to SEE and spill all of them. isVGPRClass/isAGPRClass
  /// are BOTH false for an av_ class (it is isVectorSuperClass), which is why a
  /// plain isVGPRClass test silently drops av_ tuples from candidate collection
  /// while they still count toward pressure.
  bool inCurrentFile(const TargetRegisterClass *RC) const;

  /// Live vregs at \p Slot belonging to the current pass's file (inCurrentFile).
  /// Replaces a single-RegKind getLiveRegs() call, which returns exactly one of
  /// {VGPR, AVGPR, AGPR} and so misses the av_/AGPR pressure the VGPR pass must
  /// spill.
  VRegMaskPairSet getLiveRegsForCurrentFile(SlotIndex Slot) const;

  /// Processes the entire function for one register class (SGPR or VGPR).
  /// This is called twice: first for SGPRs, then for VGPRs.
  /// Uses IsVGPRPass class member (set before calling).
  bool processFunction(MachineFunction &MF, unsigned RPLimit);

  /// ACL (around-call-liver) preserved-RP pass for the current file
  /// (IsVGPRPass). For each call C in program order, if the width-weighted set
  /// of pinned vregs live across C exceeds the callee-saved capacity k_cs, spill
  /// the excess (clean candidates first, then farthest next use) by
  /// store-at-def + free-across-C. The free point (KillIdx) is chosen per value:
  ///   - if V has a use that dominates C (a pre-call use on the path to C), kill
  ///     at the DEEPEST such use — V keeps its register up to there, is dead
  ///     across C, and post-call uses reload after C;
  ///   - otherwise (no C-dominating use) kill at C itself, which makes V dead
  ///     exactly at C;
  ///   - a value read AT C (call operand/target) is unspillable for C and only
  ///     contributes to the infeasibility floor.
  /// Runs before the ordinary total-RP processFunction pass for the same file.
  /// Returns true if any spill was performed. Relies on PinnedVRegs / the k_cs
  /// caps computed by computePinnedAndCap(MF).
  bool processACLCalls(MachineFunction &MF);

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

  /// Debug helper: dumps a register set to dbgs().
  void dumpRegSet(const VRegMaskPairSet &Regs) const;

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
