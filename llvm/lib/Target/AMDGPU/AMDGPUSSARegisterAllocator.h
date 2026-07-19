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
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "SSASpillEmitter.h"
#include <memory>
#include <set>

namespace llvm {

class GCNSubtarget;
class SIInstrInfo;
class SIRegisterInfo;

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
  unsigned MaxVGPRIdx = 0;
  unsigned MaxSGPRIdx = 0;
  unsigned MaxAGPRIdx = 0;
  // (call def-slot, call instruction) for every call; a vreg live across a call
  // must avoid every register the call clobbers (regmask + explicit defs).
  SmallVector<std::pair<SlotIndex, const MachineInstr *>, 8> CallSites;
  unsigned DynVGPRBlockSize = 0;

  // Exec-safe spill/reload emitter, shared with the spiller pass. Used by the
  // approach-A spill-on-coloring-failure path: when color() cannot place a
  // value, the driver spills it here (store-at-def + dominance reloads) and
  // recolors. Created per function in runOnMachineFunction.
  std::unique_ptr<SSASpillEmitter> Emitter;

  // Values that color() could not place (no physreg free across their whole
  // range — the %560/%1072 long-liver class). color() collects ALL of them and
  // finishes the walk (coloring everything else normally), rather than bailing
  // on the first, so that after the driver spills these the ONLY uncolored
  // vregs left are the short reload remainders — which provably settle.
  SmallVector<Register, 8> UncolorableVRegs;

  // === Coloring ===
  void classifyVRegs();
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

  MCRegister pickFreePhysReg(
      const TargetRegisterClass *RC, const LiveInterval &VI,
      ArrayRef<std::pair<MCRegister, const LiveInterval *>> WiderDefs,
      ArrayRef<MCRegister> Hints = {});

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
  void destroySSAAndRewrite(MachineFunction &MF);
  void lowerPHIs(MachineFunction &MF);
  void resolvePermutation(
      MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
      SmallVectorImpl<std::pair<MCRegister, MCRegister>> &Copies);
  void emitSwap(MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
                MCRegister RegA, MCRegister RegB);
  void rewriteOperands(MachineFunction &MF);
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
