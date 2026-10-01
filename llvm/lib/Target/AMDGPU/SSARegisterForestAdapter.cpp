//===- SSARegisterForestAdapter.cpp - Target homes for RF ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SSARegisterForestAdapter.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include <cassert>

using namespace llvm;

ArrayRef<MCPhysReg>
RegisterForestAdapter::allocationOrder(Register VR) const {
  assert(VR.isVirtual() && "allocation order requires a virtual register");
  return GetOrder(MRI.getRegClass(VR));
}

SSARegisterForest::PhysicalSpan
RegisterForestAdapter::physicalSpan(MCRegister PR) const {
  if (!PR || PR.id() >= TRI.getNumRegs())
    return {};
  const TargetRegisterClass *RC = TRI.getPhysRegBaseClass(PR);
  if (!RC)
    return {};
  const unsigned Bits = TRI.getRegSizeInBits(*RC);
  if (!Bits || Bits % 32)
    return {};

  const unsigned FirstPhysicalLeaf = TRI.getHWRegIndex(PR);
  const unsigned Width = Bits / 32;
  if (FirstPhysicalLeaf >= Forest.numLeaves() ||
      Width > Forest.numLeaves() - FirstPhysicalLeaf)
    return {};
  return {FirstPhysicalLeaf, FirstPhysicalLeaf + Width};
}

bool RegisterForestAdapter::isFree(MCRegister PR, SlotIndex Start,
                                   SlotIndex End) const {
  return Forest.isFree(physicalSpan(PR), Start, End);
}

bool RegisterForestAdapter::contains(Register VR, MCRegister PR,
                                     SlotIndex Start, SlotIndex End) const {
  return Forest.contains(physicalSpan(PR), Start, End, VR);
}

bool RegisterForestAdapter::assign(Register VR, MCRegister PR, SlotIndex Start,
                                   SlotIndex End) {
  return Forest.assign(physicalSpan(PR), Start, End, VR);
}

bool RegisterForestAdapter::release(Register VR, MCRegister PR, SlotIndex Start,
                                    SlotIndex End) {
  return Forest.release(physicalSpan(PR), Start, End, VR);
}
