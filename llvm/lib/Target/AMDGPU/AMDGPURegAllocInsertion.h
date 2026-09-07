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
