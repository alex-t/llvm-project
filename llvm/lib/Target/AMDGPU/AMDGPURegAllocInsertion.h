//===-- AMDGPURegAllocInsertion.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Shared insertion-point rules for the AMDGPU SSA register allocator.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUREGALLOCINSERTION_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUREGALLOCINSERTION_H

#include "AMDGPULaneMaskUtils.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include <iterator>
#include <optional>

namespace llvm {
namespace AMDGPURegAllocInsertion {

/// Return the first legal position for a non-PHI instruction.
inline MachineBasicBlock::iterator bodyBegin(MachineBasicBlock &MBB) {
  return MBB.getFirstNonPHI();
}

/// Return the exclusive end of the legal non-PHI block body.
inline MachineBasicBlock::iterator bodyEnd(MachineBasicBlock &MBB) {
  return MBB.getFirstTerminator();
}

/// Clamp a requested before-position into [bodyBegin, bodyEnd].
inline MachineBasicBlock::iterator
legalBefore(MachineBasicBlock &MBB,
            MachineBasicBlock::iterator Requested) {
  auto Begin = bodyBegin(MBB);
  auto End = bodyEnd(MBB);

  for (auto I = MBB.begin(); I != Begin; ++I)
    if (I == Requested)
      return Begin;

  for (auto I = Begin;; ++I) {
    if (I == Requested)
      return I;
    if (I == End)
      break;
  }

  return End;
}

/// Return a legal position after Def, or nullopt when Def is in the terminator
/// sequence and no after-def position exists in the block body.
inline std::optional<MachineBasicBlock::iterator>
legalAfter(MachineInstr &Def) {
  MachineBasicBlock &MBB = *Def.getParent();
  auto Begin = bodyBegin(MBB);
  auto End = bodyEnd(MBB);

  if (Def.isPHI())
    return Begin;

  for (auto I = MBB.begin(); I != Begin; ++I)
    if (&*I == &Def)
      return Begin;

  for (auto I = Begin; I != End; ++I)
    if (&*I == &Def)
      return std::next(I);

  return std::nullopt;
}

/// Choose a position for a split or cross-file copy of CopySource. Requested must
/// already be a structurally legal insertion point. This does not place spill
/// stores, reloads, or PHI copies in newly split edges.
///
/// Scalar copies and ordinary body copies keep their requested position. At a
/// divergent join entry, place the copy after restoration of the mask saved by
/// the immediate dominator. An ordinary COPY can then define the value consumed
/// by the redirected uses. Return nullopt if that entry placement cannot be
/// established, or moving the copy would cross a reading use of CopySource.
/// CopySource is the value to be split or rehomed, before creating the new
/// definition. Callers must not redirect an earlier use to that new definition.
///
/// This query makes no MIR, liveness or RF changes. On nullopt, the caller must
/// decline this recovery action without mutation and let recovery try its next
/// action. Failure does not authorize inserting at Requested or selecting WWM.
inline std::optional<MachineBasicBlock::iterator>
copyPosition(MachineBasicBlock &MBB, MachineBasicBlock::iterator Requested,
             Register CopySource, MachineDominatorTree &MDT,
             const SIInstrInfo &TII) {
  const auto &ST = MBB.getParent()->getSubtarget<GCNSubtarget>();
  const auto &TRI = *ST.getRegisterInfo();
  const auto &MRI = MBB.getParent()->getRegInfo();
  if (TRI.isSGPRClass(TRI.getRegClassForReg(MRI, CopySource)))
    return Requested;

  // Debug instructions and labels do not turn an entry request into a body
  // request. Unlike SkipPHIsLabelsAndDebug, this does not classify a prologue.
  auto Entry = MBB.getFirstNonPHI();
  while (Entry != MBB.end() && (Entry->isDebugInstr() || Entry->isLabel()))
    ++Entry;
  auto Position = Requested;
  while (Position != MBB.end() &&
         (Position->isDebugInstr() || Position->isLabel()))
    ++Position;
  if (Position != Entry || MBB.pred_size() < 2)
    return Requested;

  auto *Node = MDT.getNode(&MBB);
  // Unreachable blocks may have no dominator node; this is not malformed MIR.
  if (!Node)
    return std::nullopt;
  if (Node == MDT.getRootNode())
    return Requested;
  assert(Node->getIDom() && "non-root dominator node must have an idom");
  MachineBasicBlock &DomBB = *Node->getIDom()->getBlock();
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  SmallVector<MachineOperand, 2> Cond;
  // hasDivergentBranch recognizes SI_IF/SI_ELSE/SI_LOOP before lowering.
  // Here the branch has already been lowered; use its analyzed condition.
  if (TII.analyzeBranch(DomBB, TBB, FBB, Cond))
    return std::nullopt;
  if (Cond.empty() || Cond[1].getReg() == AMDGPU::SCC)
    return Requested;

  const auto &LMC = AMDGPU::LaneMaskConstants::get(ST);
  const MachineInstr *MaskChange = nullptr;
  for (const MachineInstr &MI : llvm::reverse(DomBB)) {
    if (MI.isDebugInstr() || TII.isCopyFromExec(MI))
      continue;
    if (MI.modifiesRegister(LMC.ExecReg, &TRI)) {
      MaskChange = &MI;
      break;
    }
  }
  if (!MaskChange)
    return std::nullopt;
  const MachineInstr *Save = TII.findSavedExecBefore(*MaskChange);
  if (!Save)
    return std::nullopt;
  if (const MachineInstr *Complement =
          TII.findExecComplementBefore(*MaskChange, *Save))
    Save = Complement;
  SmallVector<Register, 4> SavedMaskCopies{Save->getOperand(0).getReg()};

  // Match a restoration of this saved mask, not merely an EXEC writer. OR
  // restores the missing lanes; OR_SAVEEXEC additionally retains the old EXEC
  // for the following arm. LaneMaskConstants handles both wave sizes.
  for (auto I = Position, E = MBB.getFirstTerminator(); I != E; ++I) {
    if (I->isDebugInstr())
      continue;
    if (I->readsRegister(CopySource, &TRI))
      return std::nullopt;
    // Read the copy source before invalidating its destination. Thus
    // "backup = COPY saved; saved = ..." retains backup as an available mask.
    Register CopiedMask;
    if (auto Copy = TII.isCopyInstr(*I)) {
      const MachineOperand &Src = *Copy->Source;
      const MachineOperand &Dst = *Copy->Destination;
      if (Src.readsReg() && !Src.getSubReg() && !Dst.getSubReg() &&
          llvm::is_contained(SavedMaskCopies, Src.getReg()))
        CopiedMask = Dst.getReg();
    }

    bool Restores = CopiedMask == LMC.ExecReg;
    if (I->getOpcode() == LMC.OrSaveExecOpc) {
      const MachineOperand &Mask = I->getOperand(1);
      Restores = Mask.isReg() &&
                 llvm::is_contained(SavedMaskCopies, Mask.getReg());
    } else if (I->getOpcode() == LMC.OrOpc &&
               I->getOperand(0).getReg() == LMC.ExecReg) {
      const MachineOperand &A = I->getOperand(1);
      const MachineOperand &B = I->getOperand(2);
      Restores = A.isReg() && B.isReg() &&
                 ((A.getReg() == LMC.ExecReg &&
                   llvm::is_contained(SavedMaskCopies, B.getReg())) ||
                  (B.getReg() == LMC.ExecReg &&
                   llvm::is_contained(SavedMaskCopies, A.getReg())));
    }
    if (Restores)
      return std::next(I);

    llvm::erase_if(SavedMaskCopies, [&](Register R) {
      return I->modifiesRegister(R, &TRI);
    });
    if (CopiedMask)
      SavedMaskCopies.push_back(CopiedMask);
    if (SavedMaskCopies.empty())
      return std::nullopt;
  }
  return std::nullopt;
}

/// True when Reg is not modified in [Begin, End).
inline bool registerStable(MachineBasicBlock::iterator Begin,
                           MachineBasicBlock::iterator End, MCRegister Reg,
                           const TargetRegisterInfo *TRI) {
  for (auto I = Begin; I != End; ++I)
    if (I->modifiesRegister(Reg, TRI))
      return false;
  return true;
}

} // namespace AMDGPURegAllocInsertion
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUREGALLOCINSERTION_H
