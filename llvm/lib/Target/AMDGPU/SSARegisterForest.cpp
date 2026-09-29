//===- SSARegisterForest.cpp - Physical register forest geometry ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SSARegisterForest.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <cstdint>
#include <limits>

using namespace llvm;

std::optional<SSARegisterForest> SSARegisterForest::create(unsigned NumTrees,
                                                           unsigned TreeWidth) {
  if (NumTrees == 0 || TreeWidth == 0 || !isPowerOf2_32(TreeWidth))
    return std::nullopt;

  const uint64_t Stride = 2 * uint64_t(TreeWidth) - 1;
  const uint64_t LeafCount = uint64_t(NumTrees) * TreeWidth;
  const uint64_t NodeCount = uint64_t(NumTrees) * Stride;
  const uint64_t MaxIndex = std::numeric_limits<NodeIndex>::max();
  if (Stride > MaxIndex || LeafCount > MaxIndex || NodeCount > MaxIndex)
    return std::nullopt;

  return SSARegisterForest(NumTrees, TreeWidth, unsigned(Stride),
                           unsigned(LeafCount), unsigned(NodeCount));
}

bool SSARegisterForest::validNodeWidth(unsigned Width) const {
  return Width != 0 && Width <= TreeWidth && isPowerOf2_32(Width);
}

std::optional<SSARegisterForest::NodeRef>
SSARegisterForest::nodeAt(NodeIndex MemoryIndex) const {
  if (MemoryIndex >= NumNodes)
    return std::nullopt;

  const unsigned TreeOrdinal = MemoryIndex / TreeStride;
  unsigned TreeLocalIndex = MemoryIndex % TreeStride;
  unsigned SpanWidth = TreeWidth;
  unsigned FirstPhysicalLeaf = TreeOrdinal * TreeWidth;

  // In a preorder tree of Width leaves, the root is local offset zero, the
  // left subtree occupies [1, Width), and the right subtree starts at Width.
  while (TreeLocalIndex != 0) {
    assert(SpanWidth > 1 && "leaf cannot contain another preorder node");
    const unsigned Half = SpanWidth / 2;
    if (TreeLocalIndex < SpanWidth) {
      --TreeLocalIndex;
    } else {
      TreeLocalIndex -= SpanWidth;
      FirstPhysicalLeaf += Half;
    }
    SpanWidth = Half;
  }

  return NodeRef{MemoryIndex,
                 {FirstPhysicalLeaf, FirstPhysicalLeaf + SpanWidth}};
}

SSARegisterForest::NodeLevelRange
SSARegisterForest::nodeLevel(unsigned NodeWidth) const {
  if (!validNodeWidth(NodeWidth))
    return {};
  return NodeLevelRange(this, NodeWidth, NumLeaves / NodeWidth);
}

SSARegisterForest::NodeIndex
SSARegisterForest::nodeIndexForSpan(PhysicalSpan Span) const {
  assert(validNodeWidth(Span.width()));
  assert(Span.FirstPhysicalLeaf % Span.width() == 0);
  assert(Span.EndPhysicalLeaf <= NumLeaves);
  assert(Span.FirstPhysicalLeaf / TreeWidth ==
         (Span.EndPhysicalLeaf - 1) / TreeWidth);

  const unsigned TreeOrdinal = Span.FirstPhysicalLeaf / TreeWidth;
  const unsigned TargetFirstPhysicalLeaf = Span.FirstPhysicalLeaf % TreeWidth;
  unsigned CurrentFirstPhysicalLeaf = 0;
  unsigned CurrentWidth = TreeWidth;
  unsigned TreeLocalIndex = 0;

  while (CurrentWidth != Span.width()) {
    const unsigned Half = CurrentWidth / 2;
    if (TargetFirstPhysicalLeaf < CurrentFirstPhysicalLeaf + Half) {
      // The left child immediately follows its parent in preorder.
      ++TreeLocalIndex;
    } else {
      // The right child follows the complete left subtree. A tree whose root
      // covers CurrentWidth leaves contains 2*CurrentWidth-1 nodes, while its
      // left subtree contains CurrentWidth-1 nodes.
      TreeLocalIndex += CurrentWidth;
      CurrentFirstPhysicalLeaf += Half;
    }
    CurrentWidth = Half;
  }

  assert(CurrentFirstPhysicalLeaf == TargetFirstPhysicalLeaf);
  return TreeOrdinal * TreeStride + TreeLocalIndex;
}

SSARegisterForest::NodeRef
SSARegisterForest::nodeLevelRef(unsigned NodeWidth,
                                unsigned FirstPhysicalLeaf) const {
  assert(validNodeWidth(NodeWidth));
  assert(FirstPhysicalLeaf < NumLeaves);
  PhysicalSpan Span{FirstPhysicalLeaf, FirstPhysicalLeaf + NodeWidth};
  return {nodeIndexForSpan(Span), Span};
}

SSARegisterForest::NodeRef
SSARegisterForest::NodeLevelIterator::operator*() const {
  assert(Forest && "cannot dereference a default level iterator");
  return Forest->nodeLevelRef(NodeWidth, FirstPhysicalLeaf);
}

SSARegisterForest::NodeLevelIterator &
SSARegisterForest::NodeLevelIterator::operator++() {
  assert(Forest && "cannot increment a default level iterator");
  assert(FirstPhysicalLeaf < Forest->numLeaves() &&
         "cannot increment the end level iterator");
  FirstPhysicalLeaf += NodeWidth;
  return *this;
}

SSARegisterForest::NodeLevelIterator
SSARegisterForest::NodeLevelIterator::operator++(int) {
  NodeLevelIterator Previous = *this;
  ++*this;
  return Previous;
}

bool SSARegisterForest::NodeLevelIterator::operator==(
    const NodeLevelIterator &Other) const {
  return Forest == Other.Forest && NodeWidth == Other.NodeWidth &&
         FirstPhysicalLeaf == Other.FirstPhysicalLeaf;
}
