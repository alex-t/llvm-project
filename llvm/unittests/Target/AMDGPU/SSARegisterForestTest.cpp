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

  // Two trees exercise both cross-tree isolation and every preorder position
  // within a tree. Width eight supplies three non-leaf ancestor levels.
  std::optional<SSARegisterForest> MaybeForest =
      SSARegisterForest::create(2, 8);
  ASSERT_TRUE(MaybeForest);
  SSARegisterForest &Forest = *MaybeForest;

  for (SSARegisterForest::NodeIndex AssignedNode = 0;
       AssignedNode != Forest.numNodes(); ++AssignedNode) {
    ASSERT_TRUE(Forest.assign(AssignedNode, S2, S6, AssignedOwner));
    std::optional<SSARegisterForest::NodeRef> AssignedRef =
        Forest.nodeAt(AssignedNode);
    ASSERT_TRUE(AssignedRef);
    const SSARegisterForest::PhysicalSpan AssignedSpan = AssignedRef->Span;

    // Ownership is authoritative only at the exact assigned node.
    for (SSARegisterForest::NodeIndex I = 0; I != Forest.numNodes(); ++I) {
      std::optional<Register> Owner = Forest.ownerAt(I, S4);
      if (I == AssignedNode) {
        ASSERT_TRUE(Owner);
        EXPECT_EQ(*Owner, AssignedOwner);
      } else {
        EXPECT_FALSE(Owner);
      }
    }

    for (SSARegisterForest::NodeIndex ProbeNode = 0;
         ProbeNode != Forest.numNodes(); ++ProbeNode) {
      std::optional<SSARegisterForest::NodeRef> ProbeRef =
          Forest.nodeAt(ProbeNode);
      ASSERT_TRUE(ProbeRef);
      const SSARegisterForest::PhysicalSpan ProbeSpan = ProbeRef->Span;
      const bool SpatialOverlap =
          AssignedSpan.FirstPhysicalLeaf < ProbeSpan.EndPhysicalLeaf &&
          ProbeSpan.FirstPhysicalLeaf < AssignedSpan.EndPhysicalLeaf;

      for (const SlotRange &Probe : ProbeRanges) {
        const bool TemporalOverlap = S2 < Probe.End && Probe.Start < S6;
        const bool ExpectedFree = !(SpatialOverlap && TemporalOverlap);

        EXPECT_EQ(Forest.isFree(ProbeNode, Probe.Start, Probe.End),
                  ExpectedFree)
            << "assigned node " << AssignedNode << ", probe node " << ProbeNode
            << ", temporal case " << Probe.Name;

        const bool WasAssigned =
            Forest.assign(ProbeNode, Probe.Start, Probe.End, ProbeOwner);
        EXPECT_EQ(WasAssigned, ExpectedFree)
            << "assigned node " << AssignedNode << ", probe node " << ProbeNode
            << ", temporal case " << Probe.Name;
        if (WasAssigned)
          ASSERT_TRUE(
              Forest.release(ProbeNode, Probe.Start, Probe.End, ProbeOwner))
              << "assigned node " << AssignedNode << ", probe node "
              << ProbeNode << ", temporal case " << Probe.Name;
      }
    }

    std::optional<Register> Owner = Forest.ownerAt(AssignedNode, S4);
    ASSERT_TRUE(Owner);
    EXPECT_EQ(*Owner, AssignedOwner);
    ASSERT_TRUE(Forest.release(AssignedNode, S2, S6, AssignedOwner));
  }
}

} // end anonymous namespace
