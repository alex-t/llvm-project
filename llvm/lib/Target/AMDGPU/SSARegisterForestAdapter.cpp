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
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include <algorithm>
#include <cassert>
#include <tuple>

using namespace llvm;

std::optional<SSARegisterForest>
RegisterForestAdapter::createForest(const TargetRegisterClass &File,
                                    const SIRegisterInfo &TRI) {
  if (!TRI.isSGPRClass(&File) && !TRI.isVGPRClass(&File) &&
      !TRI.isAGPRClass(&File))
    return std::nullopt;
  unsigned Leaves = 8;
  for (unsigned R = 1; R != TRI.getNumRegs(); ++R) {
    MCRegister PR(R);
    const TargetRegisterClass *RC = getPhysicalRegClassOrNull(PR, TRI);
    if (!RC)
      continue;
    bool SameFile = (TRI.isSGPRClass(&File) && TRI.isSGPRClass(RC)) ||
                    (TRI.isVGPRClass(&File) && TRI.isVGPRClass(RC)) ||
                    (TRI.isAGPRClass(&File) && TRI.isAGPRClass(RC));
    if (!SameFile)
      continue;
    unsigned Bits = TRI.getRegSizeInBits(*RC);
    if (Bits && Bits % 16 == 0)
      Leaves = std::max(Leaves, firstPhysicalLeaf(PR, Bits, TRI) + Bits / 16);
  }
  return SSARegisterForest::create((Leaves + 7) / 8, 8);
}

bool RegisterForestAdapter::assign(Register VR, MCRegister PR,
                                   LiveIntervals &LIS) {
  Regions Live;
  if (!getVirtualRegClassOrNull(VR, MRI) || !LIS.hasInterval(VR) ||
      !projectLiveRegions(VR, PR, LIS, Live) ||
      hasRegMaskInterference(PR, LIS.getInterval(VR), LIS))
    return false;
  return Forest.assign(VR, PR, Live);
}

const TargetRegisterClass *RegisterForestAdapter::getVirtualRegClassOrNull(
    Register VR, const MachineRegisterInfo &MRI) {
  if (!VR.isVirtual() || VR.virtRegIndex() >= MRI.getNumVirtRegs())
    return nullptr;
  return MRI.getRegClassOrNull(VR);
}

const TargetRegisterClass *RegisterForestAdapter::getPhysicalRegClassOrNull(
    MCRegister PR, const SIRegisterInfo &TRI) {
  if (!PR || PR.id() >= TRI.getNumRegs())
    return nullptr;
  return TRI.getPhysRegBaseClass(PR);
}

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
  const TargetRegisterClass *RC = getPhysicalRegClassOrNull(PR, TRI);
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

bool RegisterForestAdapter::visitInterferences(
    MCRegister PR, SlotIndex Start, SlotIndex End,
    function_ref<void(const SSARegisterForest::Ownership &)> Visit) const {
  return Forest.visitInterferences(
      physicalSpan(PR), Start, End,
      [&](SSARegisterForest::PhysicalSpan,
          const SSARegisterForest::Ownership &Owned) { Visit(Owned); });
}

bool RegisterForestAdapter::visitInterferences(
    MCRegister PR, const LiveRange &Probe,
    function_ref<void(Register, SlotIndex, SlotIndex)> Visit) const {
  const SSARegisterForest::PhysicalSpan Span = physicalSpan(PR);
  if (Span.width() == 0)
    return false;

  for (const LiveRange::Segment &Segment : Probe.segments) {
    bool Valid = Forest.visitInterferences(
        Span, Segment.start, Segment.end,
        [&](SSARegisterForest::PhysicalSpan,
            const SSARegisterForest::Ownership &Owned) {
          Visit(Owned.Owner, std::max(Owned.Start, Segment.start),
                std::min(Owned.End, Segment.end));
        });
    if (!Valid)
      return false;
  }
  return true;
}

bool RegisterForestAdapter::visitLiveRanges(
    const LiveInterval &LI,
    function_ref<bool(const LiveRange &, LaneBitmask)> Visit) const {
  if (!getVirtualRegClassOrNull(LI.reg(), MRI))
    return false;
  if (!LI.hasSubRanges())
    return Visit(LI, MRI.getMaxLaneMaskForVReg(LI.reg()));
  for (const auto &Range : LI.subranges())
    if (!Visit(Range, Range.LaneMask))
      return false;
  return true;
}

std::optional<RegisterForestAdapter::Interference>
RegisterForestAdapter::interferences(MCRegister PR,
                                     const LiveInterval &Probe) const {
  Register VR = Probe.reg();
  if (!getVirtualRegClassOrNull(VR, MRI))
    return std::nullopt;
  LaneBitmask FullMask = MRI.getMaxLaneMaskForVReg(VR);
  // Validate the home even when the probe has no live segments.
  if (!physicalSpans(VRegMaskPair(VR, FullMask), PR))
    return std::nullopt;

  Interference Result;
  auto Collect = [&](const LiveRange &Range, LaneBitmask Mask) {
    auto Spans = physicalSpans(VRegMaskPair(VR, Mask), PR);
    if (!Spans)
      return false;
    for (const auto &Segment : Range.segments)
      for (auto Span : *Spans)
        if (!Forest.visitInterferences(
                Span, Segment.start, Segment.end,
                [&](SSARegisterForest::PhysicalSpan,
                    const SSARegisterForest::Ownership &Owned) {
                  if (Owned.Owner == SSARegisterForest::SELF_OWNED)
                    Result.HasFixedInterference = true;
                  else
                    Result.VirtualOwners.push_back(Owned.Owner);
                }))
          return false;
    return true;
  };
  if (!visitLiveRanges(Probe, Collect))
    return std::nullopt;

  auto &Owners = Result.VirtualOwners;
  llvm::sort(Owners, [](Register A, Register B) { return A.id() < B.id(); });
  Owners.erase(std::unique(Owners.begin(), Owners.end()), Owners.end());
  return Result;
}

bool RegisterForestAdapter::hasRegMaskInterference(
    MCRegister PR, const LiveInterval &Probe, LiveIntervals &LIS) const {
  if (LIS.getRegMaskSlots().empty())
    return false;
  // A set bit means every overlapping regmask preserves this register.
  BitVector PreservedRegisters;
  return LIS.checkRegMaskInterference(Probe, PreservedRegisters) &&
         !PreservedRegisters.test(PR);
}

std::optional<RegisterForestAdapter::Interference>
RegisterForestAdapter::interferences(MCRegister PR, const LiveInterval &Probe,
                                     LiveIntervals &LIS) const {
  auto Result = interferences(PR, Probe);
  if (Result && !Result->HasFixedInterference)
    Result->HasFixedInterference = hasRegMaskInterference(PR, Probe, LIS);
  return Result;
}

std::optional<SmallVector<Register, 4>>
RegisterForestAdapter::interferingOwners(MCRegister PR,
                                       const LiveInterval &Probe) const {
  auto Result = interferences(PR, Probe);
  if (!Result)
    return std::nullopt;
  return std::move(Result->VirtualOwners);
}

bool RegisterForestAdapter::assignFixed(ArrayRef<MCPhysReg> Registers,
                                        LiveIntervals &LIS) {
  using Region = SSARegisterForest::OwnershipRegion;
  SmallVector<Region, 32> Regions;
  BitVector Seen(Forest.numLeaves());
  for (MCRegister PR : Registers) {
    auto Whole = physicalSpan(PR);
    if (!Whole.width())
      return false;
    SmallVector<MCRegister, 8> Halves;
    if (Whole.width() == 1) {
      Halves.push_back(PR);
    } else {
      // Subregister indices describe storage even when a half has no base
      // register class, as with artificial SGPR/AGPR high halves.
      for (MCSubRegIndexIterator I(PR, &TRI); I.isValid(); ++I)
        if (TRI.getSubRegIdxSize(I.getSubRegIndex()) == 16)
          Halves.push_back(I.getSubReg());
    }
    if (Halves.size() != Whole.width())
      return false;
    for (MCRegister HalfReg : Halves) {
      unsigned Leaf = firstPhysicalLeaf(HalfReg, 16, TRI);
      SSARegisterForest::PhysicalSpan Half{Leaf, Leaf + 1};
      if (Half.FirstPhysicalLeaf < Whole.FirstPhysicalLeaf ||
          Half.EndPhysicalLeaf > Whole.EndPhysicalLeaf)
        return false;
      if (Seen.test(Leaf))
        continue;
      Seen.set(Leaf);
      // getRegUnit sees defs/uses of all aliases, not just the operand that
      // first led RA to this leaf. Thus a tuple and its halves import once.
      for (MCRegUnit Unit : TRI.regunits(HalfReg))
        for (const auto &S : LIS.getRegUnit(Unit).segments)
          Regions.push_back({Half, S.start, S.end});
    }
  }

  llvm::sort(Regions, [](const Region &A, const Region &B) {
    return std::make_tuple(A.Span.FirstPhysicalLeaf, A.Start, A.End) <
           std::make_tuple(B.Span.FirstPhysicalLeaf, B.Start, B.End);
  });
  SmallVector<Region, 32> Merged;
  for (const Region &R : Regions) {
    if (!Merged.empty() && Merged.back().Span == R.Span &&
        R.Start <= Merged.back().End) {
      Merged.back().End = std::max(Merged.back().End, R.End);
      continue;
    }
    Merged.push_back(R);
  }
  for (const Region &R : Merged)
    if (!Forest.isFree(R.Span, R.Start, R.End))
      return false;
  for (const Region &R : Merged) {
    bool Assigned = Forest.assign(R.Span, R.Start, R.End,
                                 SSARegisterForest::SELF_OWNED);
    assert(Assigned && "preflighted fixed ownership insertion failed");
    (void)Assigned;
  }
  return true;
}

bool RegisterForestAdapter::contains(Register VR, MCRegister PR,
                                     SlotIndex Start, SlotIndex End) const {
  return Forest.contains(physicalSpan(PR), Start, End, VR);
}

bool RegisterForestAdapter::assign(Register VR, MCRegister PR, SlotIndex Start,
                                   SlotIndex End) {
  const TargetRegisterClass *RC = getVirtualRegClassOrNull(VR, MRI);
  if (!RC || !RC->contains(PR))
    return false;
  SSARegisterForest::OwnershipRegion Region{physicalSpan(PR), Start, End};
  return Forest.assign(VR, PR, {Region});
}

bool RegisterForestAdapter::release(Register VR, MCRegister PR, SlotIndex Start,
                                    SlotIndex End) {
  return Forest.release(physicalSpan(PR), Start, End, VR);
}

std::optional<SmallVector<SSARegisterForest::PhysicalSpan, 4>>
RegisterForestAdapter::physicalSpans(VRegMaskPair Value, MCRegister PR) const {
  Register VR = Value.getVReg();
  const TargetRegisterClass *RC = getVirtualRegClassOrNull(VR, MRI);
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
  const TargetRegisterClass *RC = getVirtualRegClassOrNull(VR, MRI);
  if (!RC || !RC->contains(PR))
    return false;
  // Validate even an empty interval's home, including forest bounds.
  auto Whole = physicalSpan(PR);
  if (Whole.FirstPhysicalLeaf >= Whole.EndPhysicalLeaf ||
      Whole.EndPhysicalLeaf > Forest.numLeaves())
    return false;
  SmallVector<SSARegisterForest::OwnershipRegion, 8> LiveOwnership;
  for (const RetainedRegion &R : Regions) {
    if (R.Value.getVReg() != VR)
      return false;
    auto Spans = physicalSpans(R.Value, PR);
    if (!Spans)
      return false;
    for (auto Span : *Spans)
      LiveOwnership.push_back({Span, R.Start, R.End});
  }
  return Forest.assign(VR, PR, LiveOwnership);
}

void RegisterForestAdapter::releaseRegions(
    Register VR, ArrayRef<SSARegisterForest::OwnershipRegion> Regions) {
  for (const auto &R : Regions)
    if (!Forest.release(R.Span, R.Start, R.End, VR))
      report_fatal_error("stored RF ownership release failed");
}

bool RegisterForestAdapter::projectLiveRegions(Register VR, MCRegister PR,
                                               LiveIntervals &LIS,
                                               Regions &Out) const {
  const TargetRegisterClass *RC = getVirtualRegClassOrNull(VR, MRI);
  if (!RC || !RC->contains(PR))
    return false;
  const LiveInterval &LI = LIS.getInterval(VR);
  const std::size_t OriginalSize = Out.size();
  bool Valid = visitLiveRanges(LI, [&](const LiveRange &Range, LaneBitmask Mask) {
    auto Spans = physicalSpans(VRegMaskPair(VR, Mask), PR);
    if (!Spans)
      return false;
    for (const auto &S : Range.segments)
      for (auto Span : *Spans)
        Out.push_back({Span, S.start, S.end});
    return true;
  });
  if (!Valid)
    Out.resize(OriginalSize);
  return Valid;
}

static bool regionsOverlap(
    ArrayRef<SSARegisterForest::OwnershipRegion> A,
    ArrayRef<SSARegisterForest::OwnershipRegion> B) {
  for (const auto &Left : A)
    for (const auto &Right : B)
      if (Left.overlaps(Right))
        return true;
  return false;
}

SmallVector<Register, 4>
RegisterForestAdapter::onChange(ArrayRef<Register> Affected, LiveIntervals &LIS) {
  struct Change {
    Register VR;
    MCRegister Home;
    Regions Removed;
    Regions Additions;
    bool RetiredOrEmpty = false;
    bool CannotKeepHome = false;
    bool ConflictsWithPeer = false;
  };
  SmallVector<Change, 8> Changes;
  SmallSet<Register, 8> Seen;
  for (Register VR : Affected) {
    if (!VR.isVirtual() || VR.virtRegIndex() >= MRI.getNumVirtRegs())
      report_fatal_error("RF liveness notification requires a known virtual register");
    if (!Seen.insert(VR).second)
      continue;
    MCRegister Home = assignedHome(VR);
    if (!Home)
      continue; // New uncolored values remain unassigned.
    if (!LIS.hasInterval(VR) && !MRI.reg_nodbg_empty(VR))
      report_fatal_error("RF assigned owner lost its interval but still has operands");

    Change C{VR, Home, {}, {}};
    Regions After;
    C.RetiredOrEmpty = !LIS.hasInterval(VR) || LIS.getInterval(VR).empty();
    if (!C.RetiredOrEmpty) {
      // Invalid input is an integrity error. Validate the entire batch before
      // touching RF; failed projection must never discard existing ownership.
      if (!projectLiveRegions(VR, Home, LIS, After))
        report_fatal_error("invalid repaired live-region projection");
      C.CannotKeepHome = hasRegMaskInterference(Home, LIS.getInterval(VR), LIS);
    }
    // The old records come from RF, after LIS has already changed. Keep exact
    // matches in place and collect only the delta; no persistent before-image.
    Regions Before;
    Forest.visitOwnerAssignments(VR, [&](const auto &Owned) {
      Before.push_back({Owned.Span, Owned.Start, Owned.End});
    }); // False is valid for a recorded home with zero live regions.
    for (const auto &R : Before)
      if (!llvm::is_contained(After, R))
        C.Removed.push_back(R);
    for (const auto &R : After)
      if (!llvm::is_contained(Before, R))
        C.Additions.push_back(R);
    Changes.push_back(std::move(C));
  }

  // Remove obsolete regions from every affected owner before checking growth.
  // Exact surviving records stay installed and remain interference obstacles.
  for (const Change &C : Changes)
    releaseRegions(C.VR, C.Removed);

  for (Change &C : Changes)
    for (const auto &R : C.Additions)
      if (!Forest.isFree(R.Span, R.Start, R.End))
        C.CannotKeepHome = true;

  // Additions are not installed yet. Conflicting proposals invalidate both
  // homes; marking peers separately avoids making the result depend on order.
  for (auto I = Changes.begin(); I != Changes.end(); ++I) {
    if (I->RetiredOrEmpty || I->CannotKeepHome)
      continue;
    for (auto J = Changes.begin(); J != I; ++J) {
      if (!J->RetiredOrEmpty && !J->CannotKeepHome &&
          regionsOverlap(I->Additions, J->Additions))
        I->ConflictsWithPeer = J->ConflictsWithPeer = true;
    }
  }

  SmallVector<Register, 4> Invalidated;
  for (const Change &C : Changes) {
    if (C.RetiredOrEmpty || C.CannotKeepHome || C.ConflictsWithPeer) {
      // The delta may already have removed the final record and its home.
      // Otherwise also remove the unchanged records of this invalidated owner.
      if (assignedHome(C.VR) && !unassign(C.VR))
        report_fatal_error("RF invalidation lost an assigned owner");
      Invalidated.push_back(C.VR);
      continue;
    }
    // Removing a final old region can clear Home transiently. Publish it with
    // the checked additions, even if only unchanged regions remain.
    if (!Forest.assign(C.VR, C.Home, C.Additions))
      report_fatal_error("preflighted RF liveness insertion failed");
  }
  return Invalidated;
}
