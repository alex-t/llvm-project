//===- SSARegisterForestTest.cpp - Register forest geometry tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SSARegisterForest.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "gtest/gtest.h"

#include <tuple>

using namespace llvm;

namespace {

TEST(SSARegisterForestTest, SharedCompositeOwnershipAndOwnerIndex) {
  auto MaybeForest = SSARegisterForest::create(2, 4);
  ASSERT_TRUE(MaybeForest);
  auto &Forest = *MaybeForest;
  Register A = Register::index2VirtReg(3);
  Register B = Register::index2VirtReg(9);
  IndexListEntry E1(nullptr, SlotIndex::InstrDist);
  IndexListEntry E2(nullptr, 2 * SlotIndex::InstrDist);
  IndexListEntry E3(nullptr, 3 * SlotIndex::InstrDist);
  IndexListEntry E4(nullptr, 4 * SlotIndex::InstrDist);
  SlotIndex S1(&E1, 0), S2(&E2, 0), S3(&E3, 0), S4(&E4, 0);

  // Compare the indexed logical view with an independent full node walk.
  // Pointer deduplication is required only in this reference view.
  auto CheckOwner = [&](Register Owner, unsigned Count) {
    SmallPtrSet<const SSARegisterForest::Ownership *, 8> Expected, Actual;
    Forest.visitOwnershipComponents([&](auto, const auto &O) {
      if (O.Owner == Owner)
        Expected.insert(&O);
    });
    EXPECT_EQ(Count != 0, Forest.visitOwnerAssignments(Owner, [&](const auto &O) {
      EXPECT_EQ(O.Owner, Owner);
      EXPECT_TRUE(Actual.insert(&O).second);
      EXPECT_TRUE(Expected.contains(&O));
    }));
    EXPECT_EQ(Expected.size(), Count);
    EXPECT_EQ(Actual.size(), Expected.size());
  };

  // An unaligned assignment crossing two trees has four canonical references
  // to one record. Another span with the same owner/time is a distinct record.
  ASSERT_TRUE(Forest.assign({1, 7}, S1, S2, A));
  ASSERT_TRUE(Forest.assign({7, 8}, S1, S2, A));
  ASSERT_TRUE(Forest.assign({1, 7}, S2, S3, A));
  ASSERT_TRUE(Forest.assign({1, 7}, S3, S4, B));
  ASSERT_TRUE(Forest.assign({0, 1}, S1, S4, SSARegisterForest::SELF_OWNED));
  CheckOwner(A, 3);
  CheckOwner(B, 1);
  const SSARegisterForest::Ownership *Composite = nullptr;
  unsigned Components = 0;
  Forest.visitOwnershipComponents([&](auto Span, const auto &O) {
    if (O.Owner != A || O.Start != S1 || O.Span.FirstPhysicalLeaf != 1)
      return;
    EXPECT_EQ(O.Span, (SSARegisterForest::PhysicalSpan{1, 7}));
    EXPECT_GE(Span.FirstPhysicalLeaf, 1u);
    EXPECT_LE(Span.EndPhysicalLeaf, 7u);
    if (!Composite)
      Composite = &O;
    EXPECT_EQ(&O, Composite);
    ++Components;
  });
  EXPECT_EQ(Components, 4u);

  // Failed mutations preserve both views. Releasing one canonical component
  // must not accidentally release part of a logical composite assignment.
  EXPECT_FALSE(Forest.release({1, 2}, S1, S2, A));
  EXPECT_FALSE(Forest.assign({1, 7}, S3, S4, A));
  SSARegisterForest::OwnershipRegion Outside[] = {{{0, 3}, S2, S3}};
  EXPECT_FALSE(Forest.replace({{1, 7}, S2, S3}, A, Outside));
  CheckOwner(A, 3);
  EXPECT_TRUE(Forest.contains({1, 7}, S1, S2, A));

  // A still owns the same nodes at another time: their bits must survive.
  ASSERT_TRUE(Forest.release({1, 7}, S1, S2, A));
  CheckOwner(A, 2);
  SSARegisterForest::OwnershipRegion Retained[] = {
      {{1, 3}, S2, S3}, {{5, 7}, S2, S3}};
  ASSERT_TRUE(Forest.replace({{1, 7}, S2, S3}, A, Retained));
  CheckOwner(A, 3);
  EXPECT_TRUE(Forest.isFree({3, 5}, S2, S3));
  EXPECT_TRUE(Forest.contains({1, 3}, S2, S3, A));
  EXPECT_TRUE(Forest.contains({5, 7}, S2, S3, A));

  ASSERT_TRUE(Forest.releaseOwner(A));
  CheckOwner(A, 0);
  CheckOwner(B, 1);
  EXPECT_TRUE(Forest.contains({1, 7}, S3, S4, B));
  EXPECT_TRUE(Forest.contains({0, 1}, S1, S4, SSARegisterForest::SELF_OWNED));
  EXPECT_FALSE(Forest.releaseOwner(A));
  CheckOwner(Register::index2VirtReg(1000), 0);
  EXPECT_FALSE(Forest.releaseOwner(Register::index2VirtReg(1000)));
  EXPECT_FALSE(Forest.releaseOwner(SSARegisterForest::SELF_OWNED));
  EXPECT_FALSE(Forest.visitOwnerAssignments(SSARegisterForest::SELF_OWNED,
                                     [&](const auto &) { ADD_FAILURE(); }));

  // An emptied bitmap is reusable, including for a different canonical cover.
  ASSERT_TRUE(Forest.assign({0, 8}, S4, S4.getDeadSlot(), A));
  CheckOwner(A, 1);
  ASSERT_TRUE(Forest.replace({{0, 8}, S4, S4.getDeadSlot()}, A, {}));
  CheckOwner(A, 0);
}

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

TEST(SSARegisterForestTest, PartialSpillReplacementPreservesRemainingOwnership) {
  IndexListEntry E1(nullptr, SlotIndex::InstrDist);
  IndexListEntry E2(nullptr, 2 * SlotIndex::InstrDist);
  IndexListEntry E3(nullptr, 3 * SlotIndex::InstrDist);
  IndexListEntry E4(nullptr, 4 * SlotIndex::InstrDist);
  SlotIndex Start(&E1, 0), Spill(&E2, 0), End(&E3, 0), After(&E4, 0);
  const Register Owner = Register::index2VirtReg(0);
  const Register Neighbor = Register::index2VirtReg(1);
  const Register NewOwner = Register::index2VirtReg(2);
  using Span = SSARegisterForest::PhysicalSpan;
  using Region = SSARegisterForest::OwnershipRegion;

  // With 16-bit leaves, [4,8) is VGPR2_3. Also exercise a composite original
  // crossing the hardware-tree boundary: [6,10) is VGPR3_4.
  for (Span Whole : {Span{4, 8}, Span{6, 10}}) {
    SCOPED_TRACE(Whole.FirstPhysicalLeaf);
    auto MaybeForest = SSARegisterForest::create(2, 8);
    ASSERT_TRUE(MaybeForest);
    SSARegisterForest &Forest = *MaybeForest;
    const Span Freed{Whole.FirstPhysicalLeaf, Whole.FirstPhysicalLeaf + 2};
    const Span Retained{Whole.FirstPhysicalLeaf + 2, Whole.EndPhysicalLeaf};
    const Span Unrelated{0, 2};
    const Region Original{Whole, Start, End};
    const Region Head{Whole, Start, Spill};
    const Region Tail{Retained, Spill, End};
    ASSERT_TRUE(Forest.assign(Whole, Start, End, Owner));
    ASSERT_TRUE(Forest.assign(Unrelated, Start, End, Neighbor));
    ASSERT_TRUE(Forest.assign(Whole, End, After, Owner));
    SmallVector<const SSARegisterForest::Ownership *, 4> Unchanged;
    Forest.visitOwnershipComponents([&](Span S, const SSARegisterForest::Ownership &O) {
      if (S == Unrelated || O.Start == End)
        Unchanged.push_back(&O);
    });
    ASSERT_FALSE(Unchanged.empty());
    auto ExpectUnchanged = [&] {
      SmallVector<const SSARegisterForest::Ownership *, 4> Current;
      Forest.visitOwnershipComponents([&](Span S, const SSARegisterForest::Ownership &O) {
        if (S == Unrelated || O.Start == End)
          Current.push_back(&O);
      });
      EXPECT_EQ(Current, Unchanged);
      EXPECT_TRUE(Forest.contains(Whole, End, After, Owner));
      EXPECT_TRUE(Forest.contains(Unrelated, Start, End, Neighbor));
    };

    auto ExpectOriginal = [&] {
      EXPECT_TRUE(Forest.contains(Whole, Start, End, Owner));
      EXPECT_FALSE(Forest.isFree(Freed, Spill, End));
      EXPECT_TRUE(Forest.contains(Unrelated, Start, End, Neighbor));
      ExpectUnchanged();
    };
    const Region Overlapping[] = {Head, {Retained, Start, End}};
    EXPECT_FALSE(Forest.replace(Original, Owner, Overlapping));
    ExpectOriginal();
    const Region OutsideSpace[] = {
        {{Whole.FirstPhysicalLeaf - 1, Whole.EndPhysicalLeaf}, Spill, End}};
    EXPECT_FALSE(Forest.replace(Original, Owner, OutsideSpace));
    ExpectOriginal();
    const Region OutsideTime[] = {{Whole, Start, After}};
    EXPECT_FALSE(Forest.replace(Original, Owner, OutsideTime));
    ExpectOriginal();
    const Region ReversedTime[] = {{Whole, End, Start}};
    EXPECT_FALSE(Forest.replace(Original, Owner, ReversedTime));
    ExpectOriginal();
    EXPECT_FALSE(Forest.replace(Original, NewOwner, {}));
    ExpectOriginal();
    EXPECT_FALSE(Forest.replace({Retained, Start, End}, Owner, {}));
    ExpectOriginal();

    // Caller order does not determine temporal insertion order. The two pieces
    // touch at Spill, with only the high dword remaining live afterwards.
    const Region Replacements[] = {Tail, Head};
    ASSERT_TRUE(Forest.replace(Original, Owner, Replacements));
    EXPECT_FALSE(Forest.contains(Whole, Start, End, Owner));
    EXPECT_TRUE(Forest.contains(Head.Span, Head.Start, Head.End, Owner));
    EXPECT_TRUE(Forest.contains(Tail.Span, Tail.Start, Tail.End, Owner));
    EXPECT_FALSE(Forest.isFree(Freed, Start, Spill));
    EXPECT_TRUE(Forest.isFree(Freed, Spill, End));
    EXPECT_FALSE(Forest.isFree(Retained, Start, End));
    ExpectUnchanged();
    EXPECT_TRUE(Forest.contains(Unrelated, Start, End, Neighbor));

    ASSERT_TRUE(Forest.assign(Freed, Spill, End, NewOwner));
    EXPECT_TRUE(Forest.contains(Freed, Spill, End, NewOwner));
    EXPECT_FALSE(Forest.release(Whole, Start, End, Owner));
    EXPECT_TRUE(Forest.contains(Tail.Span, Tail.Start, Tail.End, Owner));
    ASSERT_TRUE(Forest.replace(Head, Owner, {}));
    EXPECT_TRUE(Forest.isFree(Whole, Start, Spill));
    EXPECT_TRUE(Forest.contains(Tail.Span, Tail.Start, Tail.End, Owner));
    EXPECT_TRUE(Forest.contains(Freed, Spill, End, NewOwner));
    ASSERT_TRUE(Forest.release(Tail.Span, Tail.Start, Tail.End, Owner));
    ASSERT_TRUE(Forest.release(Freed, Spill, End, NewOwner));
    ExpectUnchanged();
    ASSERT_TRUE(Forest.release(Whole, End, After, Owner));
    EXPECT_TRUE(Forest.isFree(Whole, Start, End));
    EXPECT_TRUE(Forest.contains(Unrelated, Start, End, Neighbor));
  }
}

TEST(SSARegisterForestTest, VisitsOnlyIntersectingOwnershipComponents) {
  IndexListEntry E1(nullptr, SlotIndex::InstrDist);
  IndexListEntry E2(nullptr, 2 * SlotIndex::InstrDist);
  IndexListEntry E3(nullptr, 3 * SlotIndex::InstrDist);
  IndexListEntry E4(nullptr, 4 * SlotIndex::InstrDist);
  SlotIndex S1(&E1, 0), S2(&E2, 0), S3(&E3, 0), S4(&E4, 0);
  using Span = SSARegisterForest::PhysicalSpan;
  using Hit = std::tuple<Span, SlotIndex, SlotIndex, Register>;
  const Register RootOwner = Register::index2VirtReg(0);
  const Register LeafOwner = Register::index2VirtReg(1);
  const Register CompositeOwner = Register::index2VirtReg(2);
  const Register Neighbor = Register::index2VirtReg(3);
  auto MaybeForest = SSARegisterForest::create(3, 8);
  ASSERT_TRUE(MaybeForest);
  SSARegisterForest &Forest = *MaybeForest;

  // The root owns all of the first tree before S2. Afterwards, its leaf and
  // a cross-tree composite own disjoint parts. A later root record and spatial
  // neighbors must not leak into the [5,11) x [S1,S3) query.
  ASSERT_TRUE(Forest.assign({0, 8}, S1, S2, RootOwner));
  ASSERT_TRUE(Forest.assign({5, 6}, S2, S3, LeafOwner));
  ASSERT_TRUE(Forest.assign({6, 10}, S2, S3, CompositeOwner));
  ASSERT_TRUE(Forest.assign({0, 8}, S3, S4, RootOwner));
  ASSERT_TRUE(Forest.assign({14, 16}, S1, S4, Neighbor));
  ASSERT_TRUE(Forest.assign({16, 24}, S1, S4, Neighbor));

  auto Expect = [&](Span Query, SlotIndex Start, SlotIndex End,
                    ArrayRef<Hit> Expected) {
    SmallVector<Hit, 8> Actual;
    EXPECT_TRUE(Forest.visitInterferences(
        Query, Start, End, [&](Span S, const SSARegisterForest::Ownership &O) {
          Actual.emplace_back(S, O.Start, O.End, O.Owner);
        }));
    EXPECT_EQ(ArrayRef<Hit>(Actual), Expected);
  };

  // The root spans several branches of this unaligned query but appears once.
  // Both components of the composite appear, with their original bounds.
  Expect({5, 11}, S1, S3,
         {{Span{0, 8}, S1, S2, RootOwner},
          {Span{5, 6}, S2, S3, LeafOwner},
          {Span{6, 8}, S2, S3, CompositeOwner},
          {Span{8, 10}, S2, S3, CompositeOwner}});
  // Leaf queries see ancestors; whole-tree queries see descendants. Touching
  // temporal and physical boundaries do not interfere.
  Expect({7, 8}, S1, S2, {{Span{0, 8}, S1, S2, RootOwner}});
  Expect({0, 8}, S1, S4,
         {{Span{0, 8}, S1, S2, RootOwner},
          {Span{0, 8}, S3, S4, RootOwner},
          {Span{5, 6}, S2, S3, LeafOwner},
          {Span{6, 8}, S2, S3, CompositeOwner}});
  Expect({0, 8}, S2, S3,
         {{Span{5, 6}, S2, S3, LeafOwner},
          {Span{6, 8}, S2, S3, CompositeOwner}});
  Expect({8, 9}, S2, S3, {{Span{8, 10}, S2, S3, CompositeOwner}});
  Expect({5, 11}, S3, S4, {{Span{0, 8}, S3, S4, RootOwner}});
  Expect({10, 14}, S1, S4, {});

  auto ExpectInvalid = [&](Span Query, SlotIndex Start, SlotIndex End) {
    EXPECT_FALSE(Forest.visitInterferences(
        Query, Start, End, [&](Span, const SSARegisterForest::Ownership &) {
          ADD_FAILURE() << "invalid query must not invoke its visitor";
        }));
  };
  ExpectInvalid({5, 5}, S1, S4);
  ExpectInvalid({6, 5}, S1, S4);
  ExpectInvalid({23, 25}, S1, S4);
  ExpectInvalid({5, 11}, SlotIndex(), S4);
  ExpectInvalid({5, 11}, S1, SlotIndex());
  ExpectInvalid({5, 11}, S2, S2);
  ExpectInvalid({5, 11}, S3, S2);

  // Released components are no longer enumerated; adjacent owners survive.
  ASSERT_TRUE(Forest.release({6, 10}, S2, S3, CompositeOwner));
  Expect({5, 11}, S2, S3, {{Span{5, 6}, S2, S3, LeafOwner}});
}

} // end anonymous namespace
