//===- SSARegisterForestAdapter.cpp - Target homes for RF ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SSARegisterForestAdapter.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include <cassert>

using namespace llvm;

ArrayRef<MCPhysReg>
RegisterForestAdapter::allocationOrder(Register VR) const {
  assert(VR.isVirtual() && "allocation order requires a virtual register");
  return GetOrder(MRI.getRegClass(VR));
}

unsigned RegisterForestAdapter::firstPhysicalLeaf(MCRegister PR, unsigned Bits,
                                                   const SIRegisterInfo &TRI) {
  // Hardware indices count dwords; RF coordinates count 16-bit leaves.
  return 2 * TRI.getHWRegIndex(PR) +
         (Bits == 16 && AMDGPU::isHi16Reg(PR, TRI));
}

SSARegisterForest::PhysicalSpan
RegisterForestAdapter::physicalSpan(MCRegister PR) const {
  if (!PR || PR.id() >= TRI.getNumRegs())
    return {};
  const TargetRegisterClass *RC = TRI.getPhysRegBaseClass(PR);
  if (!RC)
    return {};
  const unsigned Bits = TRI.getRegSizeInBits(*RC);
  if (!Bits || Bits % 16)
    return {};

  const unsigned FirstPhysicalLeaf = firstPhysicalLeaf(PR, Bits, TRI);
  const unsigned Width = Bits / 16;
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

std::optional<SmallVector<SSARegisterForest::PhysicalSpan, 4>>
RegisterForestAdapter::physicalSpans(VRegMaskPair Value, MCRegister PR) const {
  Register VR = Value.getVReg();
  if (!VR.isVirtual() || VR.virtRegIndex() >= MRI.getNumVirtRegs())
    return std::nullopt;
  const TargetRegisterClass *RC = MRI.getRegClassOrNull(VR);
  SSARegisterForest::PhysicalSpan Whole = physicalSpan(PR);
  if (!RC || Whole.FirstPhysicalLeaf >= Whole.EndPhysicalLeaf ||
      TRI.getRegSizeInBits(*RC) != Whole.width() * 16)
    return std::nullopt;

  LaneBitmask Mask = Value.getLaneMask();
  LaneBitmask FullMask = MRI.getMaxLaneMaskForVReg(VR);
  if (Mask.none() || (Mask & ~FullMask).any())
    return std::nullopt;
  if (Whole.width() == 1 && Mask != FullMask)
    return std::nullopt;

  SmallVector<SSARegisterForest::PhysicalSpan, 4> Spans;
  // A true16 virtual register's full mask is relative to its own value. Its
  // physical home may be either half; physicalSpan has already selected it.
  if (Mask == FullMask) {
    Spans.push_back(Whole);
    return Spans;
  }
  LaneBitmask MappedMask = LaneBitmask::getNone();
  for (unsigned Half = 0; Half != Whole.width(); ++Half) {
    unsigned Channel = SIRegisterInfo::getSubRegFromChannel(Half / 2);
    unsigned HalfIndex = Half % 2 ? AMDGPU::hi16 : AMDGPU::lo16;
    LaneBitmask HalfMask = TRI.composeSubRegIndexLaneMask(
        Channel, TRI.getSubRegIndexLaneMask(HalfIndex));
    if ((Mask & HalfMask).none())
      continue;
    MappedMask |= Mask & HalfMask;
    unsigned Leaf = Whole.FirstPhysicalLeaf + Half;
    if (!Spans.empty() && Spans.back().EndPhysicalLeaf == Leaf)
      Spans.back().EndPhysicalLeaf = Leaf + 1;
    else
      Spans.push_back({Leaf, Leaf + 1});
  }
  if (MappedMask != Mask)
    return std::nullopt;
  return Spans;
}

bool RegisterForestAdapter::replace(Register VR, MCRegister PR, SlotIndex Start,
                                     SlotIndex End,
                                     ArrayRef<RetainedRegion> Retained) {
  SmallVector<SSARegisterForest::OwnershipRegion, 4> SurvivingOwnership;
  for (const RetainedRegion &R : Retained) {
    if (R.Value.getVReg() != VR)
      return false;
    auto Spans = physicalSpans(R.Value, PR);
    if (!Spans)
      return false;
    for (SSARegisterForest::PhysicalSpan Span : *Spans)
      SurvivingOwnership.push_back({Span, R.Start, R.End});
  }
  return Forest.replace({physicalSpan(PR), Start, End}, VR, SurvivingOwnership);
}

bool RegisterForestAdapter::assign(Register VR, MCRegister PR,
                                   ArrayRef<RetainedRegion> Regions) {
  SmallVector<SSARegisterForest::OwnershipRegion, 8> LiveOwnership;
  for (const RetainedRegion &R : Regions) {
    if (R.Value.getVReg() != VR)
      return false;
    auto Spans = physicalSpans(R.Value, PR);
    if (!Spans)
      return false;
    for (SSARegisterForest::PhysicalSpan Span : *Spans) {
      if (!Forest.isFree(Span, R.Start, R.End))
        return false;
      SSARegisterForest::OwnershipRegion Region{Span, R.Start, R.End};
      for (const auto &Previous : LiveOwnership)
        if (Region.overlaps(Previous))
          return false;
      LiveOwnership.push_back(Region);
    }
  }
  for (const auto &R : LiveOwnership) {
    bool Assigned = Forest.assign(R.Span, R.Start, R.End, VR);
    assert(Assigned && "preflighted forest insertion failed");
    (void)Assigned;
  }
  return true;
}
