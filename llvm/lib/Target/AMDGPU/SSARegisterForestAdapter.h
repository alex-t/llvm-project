//===- SSARegisterForestAdapter.h - Target homes for RF ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SSAREGISTERFORESTADAPTER_H
#define LLVM_LIB_TARGET_AMDGPU_SSAREGISTERFORESTADAPTER_H

#include "SSARegisterForest.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/MC/MCRegister.h"
#include <utility>

namespace llvm {

class MachineRegisterInfo;
class SIRegisterInfo;
class TargetRegisterClass;

/// Translate target register identities to temporal forest operations.
///
/// One instance serves one physical register file. PR arguments must be concrete
/// storage registers in that file, covering whole 32-bit leaves. The caller
/// chooses legal homes; these operations do not select or revalidate candidates.
/// The referenced forest, MRI and TRI must outlive this adapter.
class RegisterForestAdapter {
public:
  /// Supply the allocator's existing availableOrder(RC), including reservations
  /// and the withheld vector-register tail. The returned storage must remain
  /// valid while the caller traverses the order. The callable is owned; objects
  /// it references must outlive this adapter.
  using OrderProvider =
      unique_function<ArrayRef<MCPhysReg>(const TargetRegisterClass *) const>;

  RegisterForestAdapter(SSARegisterForest &Forest, const SIRegisterInfo &TRI,
                        const MachineRegisterInfo &MRI, OrderProvider GetOrder)
      : Forest(Forest), TRI(TRI), MRI(MRI), GetOrder(std::move(GetOrder)) {}

  /// Return legal physical homes for VR in the allocator's target order.
  /// Omitted registers remain holes in physical coordinates.
  ArrayRef<MCPhysReg> allocationOrder(Register VR) const;

  bool isFree(MCRegister PR, SlotIndex Start, SlotIndex End) const;
  bool contains(Register VR, MCRegister PR, SlotIndex Start, SlotIndex End) const;
  bool assign(Register VR, MCRegister PR, SlotIndex Start, SlotIndex End);
  bool release(Register VR, MCRegister PR, SlotIndex Start, SlotIndex End);

private:
  SSARegisterForest &Forest;
  const SIRegisterInfo &TRI;
  const MachineRegisterInfo &MRI;
  OrderProvider GetOrder;

  /// RF alone decomposes this span into canonical nodes. An unsupported size
  /// or an out-of-bounds home yields an invalid span, rejected by RF.
  SSARegisterForest::PhysicalSpan physicalSpan(MCRegister PR) const;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SSAREGISTERFORESTADAPTER_H
