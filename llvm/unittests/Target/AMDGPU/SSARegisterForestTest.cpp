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

} // end anonymous namespace
