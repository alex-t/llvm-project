//===- SSARegisterForest.cpp - Physical register forest geometry ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SSARegisterForest.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
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

std::size_t
SSARegisterForest::NodeState::firstEndingAfter(SlotIndex Point) const {
  auto It = std::upper_bound(OwnedHere.begin(), OwnedHere.end(), Point,
                             [](SlotIndex Point, const Ownership &Owned) {
                               return Point < Owned.End;
                             });
  return It - OwnedHere.begin();
}

std::optional<Register>
SSARegisterForest::NodeState::ownerAt(SlotIndex At) const {
  const std::size_t Position = firstEndingAfter(At);
  if (Position == OwnedHere.size() || At < OwnedHere[Position].Start)
    return std::nullopt;
  return OwnedHere[Position].Owner;
}

bool SSARegisterForest::NodeState::interferes(SlotIndex Start,
                                              SlotIndex End) const {
  const std::size_t Position = firstEndingAfter(Start);
  return Position != OwnedHere.size() && OwnedHere[Position].Start < End;
}

void SSARegisterForest::NodeState::insert(Ownership Owned) {
  const std::size_t Position = firstEndingAfter(Owned.Start);
  assert((Position == OwnedHere.size() ||
          Owned.End <= OwnedHere[Position].Start) &&
         "overlapping temporal ownership");
  OwnedHere.insert(OwnedHere.begin() + Position, Owned);
}

bool SSARegisterForest::NodeState::erase(Ownership Owned) {
  const std::size_t Position = firstEndingAfter(Owned.Start);
  if (Position == OwnedHere.size() || !(OwnedHere[Position] == Owned))
    return false;
  OwnedHere.erase(OwnedHere.begin() + Position);
  return true;
}

bool SSARegisterForest::validOwnership(NodeIndex MemoryIndex, SlotIndex Start,
                                       SlotIndex End, Register Owner) const {
  return MemoryIndex < NumNodes && Start.isValid() && End.isValid() &&
         Start < End && Owner.isVirtual();
}

void SSARegisterForest::initializeTree(NodeIndex Root, unsigned LeafWidth) {
  assert(Root < NumNodes && LeafWidth != 0 && "invalid forest node geometry");
  Nodes[Root] = NodeState(LeafWidth);
  if (LeafWidth == 1)
    return;

  const unsigned ChildWidth = LeafWidth / 2;
  initializeTree(Root + 1, ChildWidth);
  initializeTree(Root + LeafWidth, ChildWidth);
}

SSARegisterForest::NodeIndex
SSARegisterForest::treeRootIndex(NodeIndex MemoryIndex) const {
  assert(MemoryIndex < NumNodes && "invalid forest node");
  return MemoryIndex - MemoryIndex % TreeStride;
}

SSARegisterForest::NodeIndex
SSARegisterForest::parentIndex(NodeIndex Child) const {
  assert(Child < NumNodes && "invalid forest node");
  assert(Child != treeRootIndex(Child) && "tree root has no parent");

  const unsigned ParentWidth = 2 * Nodes[Child].leafWidth();
  const NodeIndex LeftParentCandidate = Child - 1;
  if (Nodes[LeftParentCandidate].leafWidth() == ParentWidth)
    return LeftParentCandidate;

  assert(Child >= ParentWidth && "invalid right-child coordinate");
  const NodeIndex Parent = Child - ParentWidth;
  assert(Parent >= treeRootIndex(Child) &&
         Nodes[Parent].leafWidth() == ParentWidth && "invalid parent geometry");
  return Parent;
}

iota_range<SSARegisterForest::NodeIndex>
SSARegisterForest::subtree(NodeIndex Root) const {
  assert(Root < NumNodes && "invalid forest node");
  const unsigned Width = Nodes[Root].leafWidth();
  const NodeIndex SubtreeEnd = Root + Width + (Width - 1);
  const NodeIndex TreeEnd = treeRootIndex(Root) + TreeStride;
  assert(SubtreeEnd <= TreeEnd &&
         "subtree crosses a hardware-tree boundary");
  return seq(Root, SubtreeEnd);
}

SSARegisterForest::AncestorRange
SSARegisterForest::ancestors(NodeIndex Descendant) const {
  assert(Descendant < NumNodes && "invalid forest node");
  return AncestorRange(this, Descendant);
}

SSARegisterForest::NodeIndex
SSARegisterForest::AncestorIterator::operator*() const {
  assert(!AtEnd && "cannot dereference the end ancestor iterator");
  return Current;
}

SSARegisterForest::AncestorIterator &
SSARegisterForest::AncestorIterator::operator++() {
  assert(!AtEnd && "cannot increment the end ancestor iterator");
  if (Current == Forest->treeRootIndex(Current))
    AtEnd = true;
  else
    Current = Forest->parentIndex(Current);
  return *this;
}

SSARegisterForest::AncestorIterator
SSARegisterForest::AncestorIterator::operator++(int) {
  AncestorIterator Previous = *this;
  ++*this;
  return Previous;
}

bool SSARegisterForest::AncestorIterator::operator==(
    const AncestorIterator &Other) const {
  if (Forest != Other.Forest || AtEnd != Other.AtEnd)
    return false;
  return AtEnd || Current == Other.Current;
}

SSARegisterForest::AncestorIterator
SSARegisterForest::AncestorRange::begin() const {
  if (Descendant == Forest->treeRootIndex(Descendant))
    return end();
  return {Forest, Forest->parentIndex(Descendant), false};
}

bool SSARegisterForest::subtreeOrAncestorInterference(NodeIndex MemoryIndex,
                                                      SlotIndex Start,
                                                      SlotIndex End) const {
  for (NodeIndex I : subtree(MemoryIndex))
    if (Nodes[I].interferes(Start, End))
      return true;

  for (NodeIndex I : ancestors(MemoryIndex))
    if (Nodes[I].interferes(Start, End))
      return true;
  return false;
}

std::optional<Register> SSARegisterForest::ownerAt(NodeIndex MemoryIndex,
                                                   SlotIndex At) const {
  if (MemoryIndex >= NumNodes || !At.isValid())
    return std::nullopt;
  return Nodes[MemoryIndex].ownerAt(At);
}

bool SSARegisterForest::isFree(NodeIndex MemoryIndex, SlotIndex Start,
                               SlotIndex End) const {
  if (MemoryIndex >= NumNodes || !Start.isValid() || !End.isValid() ||
      !(Start < End))
    return false;
  return !subtreeOrAncestorInterference(MemoryIndex, Start, End);
}

bool SSARegisterForest::assign(NodeIndex MemoryIndex, SlotIndex Start,
                               SlotIndex End, Register Owner) {
  if (!validOwnership(MemoryIndex, Start, End, Owner) ||
      !isFree(MemoryIndex, Start, End))
    return false;

  Nodes[MemoryIndex].insert({Start, End, Owner});
  return true;
}

bool SSARegisterForest::release(NodeIndex MemoryIndex, SlotIndex Start,
                                SlotIndex End, Register Owner) {
  if (!validOwnership(MemoryIndex, Start, End, Owner))
    return false;
  return Nodes[MemoryIndex].erase({Start, End, Owner});
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
