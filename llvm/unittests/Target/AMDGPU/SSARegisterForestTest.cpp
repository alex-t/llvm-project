//===- SSARegisterForestTest.cpp - Register forest geometry tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SSARegisterForest.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(SSARegisterForestTest, PreorderLayoutAndWidthTwoLevel) {
  std::optional<SSARegisterForest> MaybeForest =
      SSARegisterForest::create(3, 4);
  ASSERT_TRUE(MaybeForest);
  const SSARegisterForest &Forest = *MaybeForest;

  EXPECT_EQ(Forest.numTrees(), 3u);
  EXPECT_EQ(Forest.treeWidth(), 4u);
  EXPECT_EQ(Forest.treeStride(), 7u);
  EXPECT_EQ(Forest.numLeaves(), 12u);
  EXPECT_EQ(Forest.numNodes(), 21u);

  const SSARegisterForest::PhysicalSpan ExpectedLayout[] = {
      {0, 4},  {0, 2},  {0, 1}, {1, 2},  {2, 4},   {2, 3},   {3, 4},
      {4, 8},  {4, 6},  {4, 5}, {5, 6},  {6, 8},   {6, 7},   {7, 8},
      {8, 12}, {8, 10}, {8, 9}, {9, 10}, {10, 12}, {10, 11}, {11, 12}};

  for (unsigned I = 0; I != Forest.numNodes(); ++I) {
    std::optional<SSARegisterForest::NodeRef> Node = Forest.nodeAt(I);
    ASSERT_TRUE(Node);
    EXPECT_EQ(Node->MemoryIndex, I);
    EXPECT_EQ(Node->Span, ExpectedLayout[I]);
  }
  EXPECT_FALSE(Forest.nodeAt(Forest.numNodes()));

  const unsigned ExpectedIndices[] = {1, 4, 8, 11, 15, 18};
  const SSARegisterForest::PhysicalSpan ExpectedLevel[] = {
      {0, 2}, {2, 4}, {4, 6}, {6, 8}, {8, 10}, {10, 12}};

  SSARegisterForest::NodeLevelRange WidthTwo = Forest.nodeLevel(2);
  ASSERT_EQ(WidthTwo.size(), 6u);

  unsigned Position = 0;
  for (SSARegisterForest::NodeRef Node : WidthTwo) {
    ASSERT_LT(Position, WidthTwo.size());
    EXPECT_EQ(Node.MemoryIndex, ExpectedIndices[Position]);
    EXPECT_EQ(Node.Span, ExpectedLevel[Position]);
    ++Position;
  }
  EXPECT_EQ(Position, WidthTwo.size());
}

TEST(SSARegisterForestTest, AvailabilityMatchesSpatialAndTemporalOverlap) {
  IndexListEntry E1(nullptr, 1 * SlotIndex::InstrDist);
  IndexListEntry E2(nullptr, 2 * SlotIndex::InstrDist);
  IndexListEntry E3(nullptr, 3 * SlotIndex::InstrDist);
  IndexListEntry E4(nullptr, 4 * SlotIndex::InstrDist);
  IndexListEntry E5(nullptr, 5 * SlotIndex::InstrDist);
  IndexListEntry E6(nullptr, 6 * SlotIndex::InstrDist);
  IndexListEntry E7(nullptr, 7 * SlotIndex::InstrDist);
  SlotIndex S1(&E1, 0);
  SlotIndex S2(&E2, 0);
  SlotIndex S3(&E3, 0);
  SlotIndex S4(&E4, 0);
  SlotIndex S5(&E5, 0);
  SlotIndex S6(&E6, 0);
  SlotIndex S7(&E7, 0);

  struct SlotRange {
    SlotIndex Start;
    SlotIndex End;
    const char *Name;
  };
  const SlotRange ProbeRanges[] = {
      {S1, S2, "touches start"}, {S1, S3, "overlaps start"},
      {S3, S5, "contained"},     {S5, S7, "overlaps end"},
      {S6, S7, "touches end"},
  };

  const Register AssignedOwner = Register::index2VirtReg(0);
  const Register ProbeOwner = Register::index2VirtReg(1);

  // Enumerate every nonempty span, including odd-width, unaligned, and
  // cross-tree spans.
  std::optional<SSARegisterForest> MaybeForest =
      SSARegisterForest::create(2, 4);
  ASSERT_TRUE(MaybeForest);
  SSARegisterForest &Forest = *MaybeForest;

  SmallVector<SSARegisterForest::PhysicalSpan, 36> Spans;
  for (unsigned First = 0; First != Forest.numLeaves(); ++First)
    for (unsigned End = First + 1; End <= Forest.numLeaves(); ++End)
      Spans.push_back({First, End});

  for (SSARegisterForest::PhysicalSpan AssignedSpan : Spans) {
    ASSERT_TRUE(Forest.assign(AssignedSpan, S2, S6, AssignedOwner));

    // Only the complete original span contains this exact assignment. Its
    // canonical components are an internal representation, not assignments
    // that may be released independently.
    for (SSARegisterForest::PhysicalSpan CandidateSpan : Spans) {
      const bool ExactSpan = CandidateSpan == AssignedSpan;
      EXPECT_EQ(Forest.contains(CandidateSpan, S2, S6, AssignedOwner),
                ExactSpan)
          << "assigned [" << AssignedSpan.FirstPhysicalLeaf << ","
          << AssignedSpan.EndPhysicalLeaf << "), candidate ["
          << CandidateSpan.FirstPhysicalLeaf << ","
          << CandidateSpan.EndPhysicalLeaf << ")";
      if (!ExactSpan) {
        EXPECT_FALSE(Forest.release(CandidateSpan, S2, S6, AssignedOwner));
        EXPECT_TRUE(Forest.contains(AssignedSpan, S2, S6, AssignedOwner));
      }
    }

    for (SSARegisterForest::PhysicalSpan ProbeSpan : Spans) {
      const bool SpatialOverlap =
          AssignedSpan.FirstPhysicalLeaf < ProbeSpan.EndPhysicalLeaf &&
          ProbeSpan.FirstPhysicalLeaf < AssignedSpan.EndPhysicalLeaf;

      for (const SlotRange &Probe : ProbeRanges) {
        const bool TemporalOverlap = S2 < Probe.End && Probe.Start < S6;
        const bool ExpectedFree = !(SpatialOverlap && TemporalOverlap);

        EXPECT_EQ(Forest.isFree(ProbeSpan, Probe.Start, Probe.End),
                  ExpectedFree)
            << "assigned [" << AssignedSpan.FirstPhysicalLeaf << ","
            << AssignedSpan.EndPhysicalLeaf << "), probe ["
            << ProbeSpan.FirstPhysicalLeaf << "," << ProbeSpan.EndPhysicalLeaf
            << "), temporal case " << Probe.Name;

        const bool WasAssigned =
            Forest.assign(ProbeSpan, Probe.Start, Probe.End, ProbeOwner);
        EXPECT_EQ(WasAssigned, ExpectedFree)
            << "assigned [" << AssignedSpan.FirstPhysicalLeaf << ","
            << AssignedSpan.EndPhysicalLeaf << "), probe ["
            << ProbeSpan.FirstPhysicalLeaf << "," << ProbeSpan.EndPhysicalLeaf
            << "), temporal case " << Probe.Name;
        if (WasAssigned)
          ASSERT_TRUE(
              Forest.release(ProbeSpan, Probe.Start, Probe.End, ProbeOwner))
              << "probe [" << ProbeSpan.FirstPhysicalLeaf << ","
              << ProbeSpan.EndPhysicalLeaf << "), temporal case " << Probe.Name;
      }
    }

    ASSERT_TRUE(Forest.contains(AssignedSpan, S2, S6, AssignedOwner));
    ASSERT_TRUE(Forest.release(AssignedSpan, S2, S6, AssignedOwner));
    for (SSARegisterForest::PhysicalSpan ProbeSpan : Spans)
      EXPECT_TRUE(Forest.isFree(ProbeSpan, S2, S6));
  }

  // Adjacent records with identical owner and time do not become one
  // composite assignment merely because their union has the same canonical
  // cover that an unaligned assignment would use.
  constexpr SSARegisterForest::PhysicalSpan LeftSpan{1, 2};
  constexpr SSARegisterForest::PhysicalSpan RightSpan{2, 4};
  constexpr SSARegisterForest::PhysicalSpan CombinedSpan{1, 4};
  ASSERT_TRUE(Forest.assign(LeftSpan, S2, S6, AssignedOwner));
  ASSERT_TRUE(Forest.assign(RightSpan, S2, S6, AssignedOwner));
  EXPECT_FALSE(Forest.contains(CombinedSpan, S2, S6, AssignedOwner));
  EXPECT_FALSE(Forest.release(CombinedSpan, S2, S6, AssignedOwner));
  EXPECT_TRUE(Forest.contains(LeftSpan, S2, S6, AssignedOwner));
  EXPECT_TRUE(Forest.contains(RightSpan, S2, S6, AssignedOwner));
  ASSERT_TRUE(Forest.release(LeftSpan, S2, S6, AssignedOwner));
  ASSERT_TRUE(Forest.release(RightSpan, S2, S6, AssignedOwner));
}

TEST(SSARegisterForestTest,
     TemporalOwnershipOrderAndExactReleaseArePermutationInvariant) {
  IndexListEntry E1(nullptr, 1 * SlotIndex::InstrDist);
  IndexListEntry E2(nullptr, 2 * SlotIndex::InstrDist);
  IndexListEntry E3(nullptr, 3 * SlotIndex::InstrDist);
  IndexListEntry E4(nullptr, 4 * SlotIndex::InstrDist);
  IndexListEntry E5(nullptr, 5 * SlotIndex::InstrDist);
  IndexListEntry E6(nullptr, 6 * SlotIndex::InstrDist);
  SlotIndex S1(&E1, 0);
  SlotIndex S2(&E2, 0);
  SlotIndex S3(&E3, 0);
  SlotIndex S4(&E4, 0);
  SlotIndex S5(&E5, 0);
  SlotIndex S6(&E6, 0);

  struct Interval {
    SlotIndex Start;
    SlotIndex End;
    Register Owner;
  };
  const Interval Intervals[] = {
      {S1, S2, Register::index2VirtReg(0)},
      {S3, S4, Register::index2VirtReg(1)},
      {S5, S6, Register::index2VirtReg(2)},
  };
  const unsigned Orders[][3] = {
      {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0},
  };
  constexpr SSARegisterForest::PhysicalSpan OwnerSpan{0, 1};

  for (const auto &InsertOrder : Orders) {
    for (const auto &ReleaseOrder : Orders) {
      std::optional<SSARegisterForest> MaybeForest =
          SSARegisterForest::create(1, 4);
      ASSERT_TRUE(MaybeForest);
      SSARegisterForest &Forest = *MaybeForest;

      for (unsigned I : InsertOrder) {
        const Interval &Value = Intervals[I];
        ASSERT_TRUE(
            Forest.assign(OwnerSpan, Value.Start, Value.End, Value.Owner));
      }

      bool Present[] = {true, true, true};
      auto CheckOwners = [&] {
        for (unsigned I = 0; I != 3; ++I) {
          const Interval &Value = Intervals[I];
          EXPECT_EQ(
              Forest.contains(OwnerSpan, Value.Start, Value.End, Value.Owner),
              Present[I]);
        }
      };

      CheckOwners();
      EXPECT_TRUE(Forest.isFree(OwnerSpan, S2, S3));
      EXPECT_TRUE(Forest.isFree(OwnerSpan, S4, S5));

      // Neither the wrong owner nor a different interval may erase a record.
      EXPECT_FALSE(Forest.release(OwnerSpan, S3, S4, Intervals[0].Owner));
      EXPECT_FALSE(Forest.release(OwnerSpan, S3, S5, Intervals[1].Owner));
      CheckOwners();

      for (unsigned I : ReleaseOrder) {
        const Interval &Value = Intervals[I];
        ASSERT_TRUE(
            Forest.release(OwnerSpan, Value.Start, Value.End, Value.Owner));
        Present[I] = false;
        CheckOwners();
      }

      EXPECT_TRUE(Forest.isFree(OwnerSpan, S1, S6));
    }
  }
}

TEST(SSARegisterForestTest, InvalidOwnershipOperationsDoNotMutateState) {
  IndexListEntry E2(nullptr, 2 * SlotIndex::InstrDist);
  IndexListEntry E3(nullptr, 3 * SlotIndex::InstrDist);
  IndexListEntry E4(nullptr, 4 * SlotIndex::InstrDist);
  IndexListEntry E5(nullptr, 5 * SlotIndex::InstrDist);
  SlotIndex S2(&E2, 0);
  SlotIndex S3(&E3, 0);
  SlotIndex S4(&E4, 0);
  SlotIndex S5(&E5, 0);
  SlotIndex Invalid;

  std::optional<SSARegisterForest> MaybeForest =
      SSARegisterForest::create(1, 4);
  ASSERT_TRUE(MaybeForest);
  SSARegisterForest &Forest = *MaybeForest;

  constexpr SSARegisterForest::PhysicalSpan OwnerSpan{0, 4};
  const SSARegisterForest::PhysicalSpan InvalidSpan{Forest.numLeaves(),
                                                    Forest.numLeaves() + 1};
  const Register Owner = Register::index2VirtReg(0);
  const Register OtherOwner = Register::index2VirtReg(1);
  const Register NoOwner;
  const Register PhysicalOwner(1);

  ASSERT_TRUE(Forest.assign(OwnerSpan, S2, S4, Owner));

  auto ExpectOriginalState = [&] {
    EXPECT_TRUE(Forest.contains(OwnerSpan, S2, S4, Owner));
    EXPECT_TRUE(Forest.isFree(OwnerSpan, S4, S5));
  };
  auto ExpectRejectedAssignment =
      [&](SSARegisterForest::PhysicalSpan CandidateSpan, SlotIndex Start,
          SlotIndex End, Register CandidateOwner, const char *Case) {
        SCOPED_TRACE(Case);
        EXPECT_FALSE(Forest.assign(CandidateSpan, Start, End, CandidateOwner));
        ExpectOriginalState();
      };
  auto ExpectRejectedRelease =
      [&](SSARegisterForest::PhysicalSpan CandidateSpan, SlotIndex Start,
          SlotIndex End, Register CandidateOwner, const char *Case) {
        SCOPED_TRACE(Case);
        EXPECT_FALSE(Forest.release(CandidateSpan, Start, End, CandidateOwner));
        ExpectOriginalState();
      };

  ExpectRejectedAssignment(InvalidSpan, S4, S5, OtherOwner, "invalid span");
  ExpectRejectedAssignment(OwnerSpan, Invalid, S5, OtherOwner, "invalid start");
  ExpectRejectedAssignment(OwnerSpan, S4, Invalid, OtherOwner, "invalid end");
  ExpectRejectedAssignment(OwnerSpan, S4, S4, OtherOwner, "empty interval");
  ExpectRejectedAssignment(OwnerSpan, S5, S4, OtherOwner, "reversed interval");
  ExpectRejectedAssignment(OwnerSpan, S4, S5, NoOwner, "absent owner");
  ExpectRejectedAssignment(OwnerSpan, S4, S5, PhysicalOwner, "physical owner");

  ExpectRejectedRelease(InvalidSpan, S2, S4, Owner, "invalid span");
  ExpectRejectedRelease(OwnerSpan, Invalid, S4, Owner, "invalid start");
  ExpectRejectedRelease(OwnerSpan, S2, Invalid, Owner, "invalid end");
  ExpectRejectedRelease(OwnerSpan, S2, S2, Owner, "empty interval");
  ExpectRejectedRelease(OwnerSpan, S4, S2, Owner, "reversed interval");
  ExpectRejectedRelease(OwnerSpan, S2, S4, NoOwner, "absent owner");
  ExpectRejectedRelease(OwnerSpan, S2, S4, PhysicalOwner, "physical owner");

  EXPECT_FALSE(Forest.contains(InvalidSpan, S2, S4, Owner));
  EXPECT_FALSE(Forest.contains(OwnerSpan, Invalid, S4, Owner));
  EXPECT_FALSE(Forest.isFree(InvalidSpan, S4, S5));
  EXPECT_FALSE(Forest.isFree(OwnerSpan, Invalid, S5));
  EXPECT_FALSE(Forest.isFree(OwnerSpan, S4, Invalid));
  EXPECT_FALSE(Forest.isFree(OwnerSpan, S4, S4));
  EXPECT_FALSE(Forest.isFree(OwnerSpan, S5, S4));
  ExpectOriginalState();

  ASSERT_TRUE(Forest.release(OwnerSpan, S2, S4, Owner));
  EXPECT_TRUE(Forest.isFree(OwnerSpan, S2, S5));
}

} // end anonymous namespace
