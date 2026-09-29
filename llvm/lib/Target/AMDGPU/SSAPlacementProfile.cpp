//===- SSAPlacementProfile.cpp - Recovery placement facts ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SSAPlacementProfile.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>

using namespace llvm;

namespace {

bool affectsHome(const SmallBitVector &Homes,
                 PlacementProfile::HomeID Home) {
  return Home < Homes.size() && Homes.test(Home);
}

bool overlaps(PlacementProfile::SlotRange A,
              PlacementProfile::SlotRange B) {
  return A.Start < B.End && B.Start < A.End;
}

} // end anonymous namespace

NormalizedPlacementProfile
llvm::normalizePlacementProfile(const PlacementProfile &Profile) {
  NormalizedPlacementProfile Result;
  Result.Subject = Profile.Subject;
  Result.RC = Profile.RC;
  Result.Region = Profile.Region;
  Result.Homes.append(Profile.Homes.begin(), Profile.Homes.end());

  if (!Profile.Region.Start.isValid() || !Profile.Region.End.isValid() ||
      !(Profile.Region.Start < Profile.Region.End))
    return Result;

  for (PlacementProfile::HomeID Home = 0; Home != Profile.Homes.size(); ++Home) {
    SmallVector<SlotIndex, 16> Points;
    Points.push_back(Profile.Region.Start);
    Points.push_back(Profile.Region.End);

    for (const PlacementProfile::BlockerSpan &Span : Profile.Blockers) {
      if (!affectsHome(Span.BlockedHomes, Home) ||
          !overlaps(Span.Range, Profile.Region))
        continue;
      Points.push_back(Span.Range.Start < Profile.Region.Start
                           ? Profile.Region.Start
                           : Span.Range.Start);
      Points.push_back(Profile.Region.End < Span.Range.End ? Profile.Region.End
                                                           : Span.Range.End);
    }
    for (const PlacementProfile::HomeCut &Cut : Profile.Cuts)
      if (affectsHome(Cut.CutHomes, Home) &&
          Profile.Region.Start < Cut.At && Cut.At < Profile.Region.End)
        Points.push_back(Cut.At);

    llvm::sort(Points);
    Points.erase(std::unique(Points.begin(), Points.end()), Points.end());

    for (unsigned I = 0, E = Points.size() - 1; I != E; ++I) {
      PlacementProfileSlice Slice;
      Slice.Home = Home;
      Slice.Range = {Points[I], Points[I + 1]};

      for (const PlacementProfile::BlockerSpan &Span : Profile.Blockers)
        if (affectsHome(Span.BlockedHomes, Home) &&
            overlaps(Span.Range, Slice.Range))
          Slice.Blockers.push_back(Span.Blocker);

      llvm::sort(Slice.Blockers, [](Register A, Register B) {
        return A.id() < B.id();
      });
      Slice.Blockers.erase(
          std::unique(Slice.Blockers.begin(), Slice.Blockers.end()),
          Slice.Blockers.end());

      for (const PlacementProfile::HomeCut &Cut : Profile.Cuts)
        if (Cut.At == Slice.Range.End && affectsHome(Cut.CutHomes, Home)) {
          Slice.CutAfter = true;
          break;
        }

      if (!Result.Slices.empty()) {
        PlacementProfileSlice &Prev = Result.Slices.back();
        if (Prev.Home == Slice.Home && Prev.Range.End == Slice.Range.Start &&
            !Prev.CutAfter && Prev.Blockers == Slice.Blockers) {
          Prev.Range.End = Slice.Range.End;
          Prev.CutAfter = Slice.CutAfter;
          continue;
        }
      }
      Result.Slices.push_back(std::move(Slice));
    }
  }
  return Result;
}

void llvm::getFreeRuns(const PlacementProfile &Profile,
                       SmallVectorImpl<PlacementFreeRun> &Out) {
  Out.clear();
  NormalizedPlacementProfile Normalized = normalizePlacementProfile(Profile);
  for (const PlacementProfileSlice &Slice : Normalized.Slices)
    if (Slice.Blockers.empty())
      Out.push_back({Slice.Home, Normalized.Homes[Slice.Home], Slice.Range});
}

void llvm::getBlockers(const PlacementProfile &Profile,
                       PlacementProfile::SlotRange Range,
                       SmallVectorImpl<Register> &Out) {
  Out.clear();
  for (const PlacementProfile::BlockerSpan &Span : Profile.Blockers)
    if (overlaps(Span.Range, Range))
      Out.push_back(Span.Blocker);

  llvm::sort(Out, [](Register A, Register B) { return A.id() < B.id(); });
  Out.erase(std::unique(Out.begin(), Out.end()), Out.end());
}

void llvm::visitInterferenceSlices(
    const PlacementProfile &Profile,
    function_ref<void(PlacementProfile::HomeID,
                      PlacementProfile::SlotRange,
                      ArrayRef<Register>)> Visit) {
  NormalizedPlacementProfile Normalized = normalizePlacementProfile(Profile);
  for (const PlacementProfileSlice &Slice : Normalized.Slices)
    Visit(Slice.Home, Slice.Range, Slice.Blockers);
}
