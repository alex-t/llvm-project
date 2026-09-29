//===- SSAPlacementProfileTest.cpp - Placement profile tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SSAPlacementProfile.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(SSAPlacementProfileTest, NormalizesOverlappingBlockersAndFreeRuns) {
  IndexListEntry E1(nullptr, 1 * SlotIndex::InstrDist);
  IndexListEntry E2(nullptr, 2 * SlotIndex::InstrDist);
  IndexListEntry E4(nullptr, 4 * SlotIndex::InstrDist);
  IndexListEntry E5(nullptr, 5 * SlotIndex::InstrDist);
  IndexListEntry E7(nullptr, 7 * SlotIndex::InstrDist);
  IndexListEntry E9(nullptr, 9 * SlotIndex::InstrDist);
  SlotIndex S1(&E1, 0);
  SlotIndex S2(&E2, 0);
  SlotIndex S4(&E4, 0);
  SlotIndex S5(&E5, 0);
  SlotIndex S7(&E7, 0);
  SlotIndex S9(&E9, 0);

  Register A = Register::index2VirtReg(0);
  Register B = Register::index2VirtReg(1);

  PlacementProfile Profile;
  Profile.Region = {S1, S9};
  Profile.Homes.push_back(MCRegister::from(1));

  SmallBitVector Home0(1);
  Home0.set(0);
  Profile.Blockers.push_back({A, {S2, S5}, Home0});
  Profile.Blockers.push_back({B, {S4, S7}, Home0});

  NormalizedPlacementProfile Normalized =
      normalizePlacementProfile(Profile);
  ASSERT_EQ(Normalized.Slices.size(), 5u);

  EXPECT_TRUE(Normalized.Slices[0].Range ==
              (PlacementProfile::SlotRange{S1, S2}));
  EXPECT_TRUE(Normalized.Slices[0].Blockers.empty());

  EXPECT_TRUE(Normalized.Slices[1].Range ==
              (PlacementProfile::SlotRange{S2, S4}));
  ASSERT_EQ(Normalized.Slices[1].Blockers.size(), 1u);
  EXPECT_EQ(Normalized.Slices[1].Blockers[0], A);

  EXPECT_TRUE(Normalized.Slices[2].Range ==
              (PlacementProfile::SlotRange{S4, S5}));
  ASSERT_EQ(Normalized.Slices[2].Blockers.size(), 2u);
  EXPECT_EQ(Normalized.Slices[2].Blockers[0], A);
  EXPECT_EQ(Normalized.Slices[2].Blockers[1], B);

  EXPECT_TRUE(Normalized.Slices[3].Range ==
              (PlacementProfile::SlotRange{S5, S7}));
  ASSERT_EQ(Normalized.Slices[3].Blockers.size(), 1u);
  EXPECT_EQ(Normalized.Slices[3].Blockers[0], B);

  EXPECT_TRUE(Normalized.Slices[4].Range ==
              (PlacementProfile::SlotRange{S7, S9}));
  EXPECT_TRUE(Normalized.Slices[4].Blockers.empty());

  SmallVector<PlacementFreeRun, 2> FreeRuns;
  getFreeRuns(Profile, FreeRuns);
  ASSERT_EQ(FreeRuns.size(), 2u);
  EXPECT_EQ(FreeRuns[0].Home, 0u);
  EXPECT_EQ(FreeRuns[0].PhysReg, MCRegister::from(1));
  EXPECT_TRUE(FreeRuns[0].Range ==
              (PlacementProfile::SlotRange{S1, S2}));
  EXPECT_EQ(FreeRuns[1].Home, 0u);
  EXPECT_EQ(FreeRuns[1].PhysReg, MCRegister::from(1));
  EXPECT_TRUE(FreeRuns[1].Range ==
              (PlacementProfile::SlotRange{S7, S9}));
}

} // end anonymous namespace
