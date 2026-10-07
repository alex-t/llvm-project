//===- SSARegisterForest.cpp - Physical register forest geometry ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SSARegisterForest.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <utility>

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
                             [](SlotIndex Point, const Ownership *Owned) {
                               return Point < Owned->End;
                             });
  return It - OwnedHere.begin();
}

const SSARegisterForest::Ownership *
SSARegisterForest::NodeState::find(SlotIndex Start, SlotIndex End,
                                   Register Owner) const {
  const std::size_t Position = firstEndingAfter(Start);
  if (Position == OwnedHere.size())
    return nullptr;

  const Ownership *Owned = OwnedHere[Position];
  return Owned->Start == Start && Owned->End == End && Owned->Owner == Owner
             ? Owned
             : nullptr;
}

bool SSARegisterForest::NodeState::interferes(SlotIndex Start,
                                              SlotIndex End) const {
  const std::size_t Position = firstEndingAfter(Start);
  return Position != OwnedHere.size() && OwnedHere[Position]->Start < End;
}

void SSARegisterForest::NodeState::visitInterferences(
    SlotIndex Start, SlotIndex End,
    function_ref<void(const Ownership &)> Visit) const {
  for (std::size_t I = firstEndingAfter(Start);
       I != OwnedHere.size() && OwnedHere[I]->Start < End; ++I)
    Visit(*OwnedHere[I]);
}

void SSARegisterForest::NodeState::insert(const Ownership *Owned) {
  const std::size_t Position = firstEndingAfter(Owned->Start);
  assert((Position == OwnedHere.size() ||
          Owned->End <= OwnedHere[Position]->Start) &&
         "overlapping temporal ownership");
  OwnedHere.insert(OwnedHere.begin() + Position, Owned);
}

bool SSARegisterForest::NodeState::erase(const Ownership *Owned) {
  const std::size_t Position = firstEndingAfter(Owned->Start);
  if (Position == OwnedHere.size() || OwnedHere[Position] != Owned)
    return false;
  OwnedHere.erase(OwnedHere.begin() + Position);
  return true;
}

bool SSARegisterForest::NodeState::hasOwner(Register Owner) const {
  for (const Ownership *Owned : OwnedHere)
    if (Owned->Owner == Owner)
      return true;
  return false;
}

void SSARegisterForest::NodeState::eraseOwner(Register Owner) {
  OwnedHere.erase(std::remove_if(OwnedHere.begin(), OwnedHere.end(),
                                 [Owner](const Ownership *Owned) {
                                   return Owned->Owner == Owner;
                                 }),
                  OwnedHere.end());
}

void SSARegisterForest::insertOwnership(NodeIndex Node,
                                        const Ownership *Owned) {
  Nodes[Node].insert(Owned);
  if (!Owned->Owner.isVirtual())
    return;
  unsigned OwnerIndex = Owned->Owner.virtRegIndex();
  if (Owners.size() <= OwnerIndex)
    Owners.resize(OwnerIndex + 1);
  BitVector &Membership = Owners[OwnerIndex].Nodes;
  if (Membership.empty())
    Membership.resize(NumNodes);
  Membership.set(Node);
}

void SSARegisterForest::eraseOwnership(NodeIndex Node,
                                       const Ownership *Owned) {
  const bool Erased = Nodes[Node].erase(Owned);
  assert(Erased && "preflighted ownership disappeared");
  (void)Erased;
  if (Owned->Owner.isVirtual() && !Nodes[Node].hasOwner(Owned->Owner))
    Owners[Owned->Owner.virtRegIndex()].Nodes.reset(Node);
}

bool SSARegisterForest::releaseOwner(Register Owner) {
  if (!Owner.isVirtual())
    return false;
  unsigned OwnerIndex = Owner.virtRegIndex();
  if (OwnerIndex >= Owners.size())
    return false;
  OwnerState &State = Owners[OwnerIndex];
  if (State.Nodes.none() && !State.Home)
    return false;
  for (unsigned Node : State.Nodes.set_bits())
    Nodes[Node].eraseOwner(Owner);
  State.Nodes.reset();
  State.Home = MCRegister();
  return true;
}

MCRegister SSARegisterForest::assignedHome(Register Owner) const {
  if (!Owner.isVirtual() || Owner.virtRegIndex() >= Owners.size())
    return MCRegister();
  return Owners[Owner.virtRegIndex()].Home;
}

void SSARegisterForest::visitAssignedOwners(
    function_ref<void(Register, MCRegister)> Visit) const {
  for (unsigned I = 0; I != Owners.size(); ++I)
    if (Owners[I].Home)
      Visit(Register::index2VirtReg(I), Owners[I].Home);
}

void SSARegisterForest::clearHomeIfEmpty(Register Owner) {
  if (!Owner.isVirtual())
    return;
  OwnerState &State = Owners[Owner.virtRegIndex()];
  if (State.Nodes.none())
    State.Home = MCRegister();
}

bool SSARegisterForest::assign(Register Owner, MCRegister Home,
                               ArrayRef<OwnershipRegion> Regions) {
  if (!Owner.isVirtual() || !Home)
    return false;
  unsigned Index = Owner.virtRegIndex();
  if (Index < Owners.size()) {
    const OwnerState &State = Owners[Index];
    if ((State.Home && State.Home != Home) ||
        (!State.Home && State.Nodes.any()))
      return false;
  }
  for (std::size_t I = 0; I != Regions.size(); ++I) {
    const OwnershipRegion &R = Regions[I];
    if (!isFree(R.Span, R.Start, R.End))
      return false;
    for (const OwnershipRegion &Previous : Regions.take_front(I))
      if (R.overlaps(Previous))
        return false;
  }
  // All recoverable failures precede mutation. Component insertion maintains
  // the node bitmap; the batch publishes the home in that same owner entry.
  for (const OwnershipRegion &R : Regions) {
    bool Inserted = assignRegion(R.Span, R.Start, R.End, Owner);
    assert(Inserted && "preflighted ownership insertion failed");
    (void)Inserted;
  }
  if (Owners.size() <= Index)
    Owners.resize(Index + 1);
  Owners[Index].Home = Home;
  return true;
}

bool SSARegisterForest::validPhysicalSpan(PhysicalSpan Span) const {
  return Span.FirstPhysicalLeaf < Span.EndPhysicalLeaf &&
         Span.EndPhysicalLeaf <= NumLeaves;
}

bool SSARegisterForest::validOwnership(SlotIndex Start, SlotIndex End,
                                       Register Owner) const {
  return Start.isValid() && End.isValid() && Start < End &&
         (Owner == SELF_OWNED || Owner.isVirtual());
}

std::optional<SSARegisterForest::NodeIndex>
SSARegisterForest::canonicalNode(PhysicalSpan Span) const {
  if (!validPhysicalSpan(Span) || !validNodeWidth(Span.width()) ||
      Span.FirstPhysicalLeaf % Span.width() != 0 ||
      Span.FirstPhysicalLeaf / TreeWidth !=
          (Span.EndPhysicalLeaf - 1) / TreeWidth)
    return std::nullopt;
  return nodeIndexForSpan(Span);
}

std::optional<SSARegisterForest::NodeCover>
SSARegisterForest::cover(PhysicalSpan Span) const {
  if (!validPhysicalSpan(Span))
    return std::nullopt;

  NodeCover Result;
  if (std::optional<NodeIndex> Node = canonicalNode(Span)) {
    Result.push_back(*Node);
    return Result;
  }

  unsigned Cursor = Span.FirstPhysicalLeaf;
  while (Cursor != Span.EndPhysicalLeaf) {
    const unsigned TreeEnd = (Cursor / TreeWidth + 1) * TreeWidth;
    const unsigned ComponentEnd = std::min(Span.EndPhysicalLeaf, TreeEnd);
    unsigned Width = llvm::bit_floor(ComponentEnd - Cursor);
    while (Cursor % Width != 0)
      Width /= 2;

    PhysicalSpan Component{Cursor, Cursor + Width};
    Result.push_back(nodeIndexForSpan(Component));
    Cursor += Width;
  }
  return Result;
}

const SSARegisterForest::Ownership *SSARegisterForest::findAssignment(
    PhysicalSpan Span, const NodeCover &Cover, SlotIndex Start, SlotIndex End,
    Register Owner) const {
  assert(!Cover.empty());
  const Ownership *Assignment = Nodes[Cover.front()].find(Start, End, Owner);
  if (!Assignment || !(Assignment->Span == Span))
    return nullptr;
  for (NodeIndex Node : Cover)
    if (Nodes[Node].find(Start, End, Owner) != Assignment)
      return nullptr;
  return Assignment;
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
  assert(SubtreeEnd <= TreeEnd && "subtree crosses a hardware-tree boundary");
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

bool SSARegisterForest::isFree(PhysicalSpan Span, SlotIndex Start,
                               SlotIndex End) const {
  std::optional<NodeCover> Cover = cover(Span);
  if (!Cover || !Start.isValid() || !End.isValid() || !(Start < End))
    return false;

  for (NodeIndex Node : *Cover)
    if (subtreeOrAncestorInterference(Node, Start, End))
      return false;
  return true;
}

void SSARegisterForest::visitInterferencesInTree(
    NodeRef Node, PhysicalSpan Span, SlotIndex Start, SlotIndex End,
    function_ref<void(PhysicalSpan, const Ownership &)> VisitComponent) const {
  Nodes[Node.MemoryIndex].visitInterferences(
      Start, End, [&](const Ownership &Owned) { VisitComponent(Node.Span, Owned); });

  const unsigned Width = Node.Span.width();
  if (Width == 1)
    return; // The leaf was visited above; it has no children.

  const unsigned Middle = Node.Span.FirstPhysicalLeaf + Width / 2;
  // Descend only into intersecting children. Visiting each node once also
  // visits shared ancestors once for unaligned queries.
  if (Span.FirstPhysicalLeaf < Middle)
    visitInterferencesInTree(
        {Node.MemoryIndex + 1, {Node.Span.FirstPhysicalLeaf, Middle}},
        Span, Start, End, VisitComponent);
  if (Middle < Span.EndPhysicalLeaf)
    visitInterferencesInTree(
        {Node.MemoryIndex + Width, {Middle, Node.Span.EndPhysicalLeaf}},
        Span, Start, End, VisitComponent);
}

bool SSARegisterForest::visitInterferences(
    PhysicalSpan Span, SlotIndex Start, SlotIndex End,
    function_ref<void(PhysicalSpan, const Ownership &)> VisitComponent) const {
  if (!validPhysicalSpan(Span) || !Start.isValid() || !End.isValid() ||
      !(Start < End))
    return false;

  const unsigned FirstTree = Span.FirstPhysicalLeaf / TreeWidth;
  const unsigned LastTree = (Span.EndPhysicalLeaf - 1) / TreeWidth;
  for (unsigned Tree = FirstTree; Tree <= LastTree; ++Tree)
    visitInterferencesInTree(
        {Tree * TreeStride, {Tree * TreeWidth, (Tree + 1) * TreeWidth}},
        Span, Start, End, VisitComponent);
  return true;
}

bool SSARegisterForest::contains(PhysicalSpan Span, SlotIndex Start,
                                 SlotIndex End, Register Owner) const {
  std::optional<NodeCover> Cover = cover(Span);
  if (!Cover || !validOwnership(Start, End, Owner))
    return false;

  return findAssignment(Span, *Cover, Start, End, Owner) != nullptr;
}

SSARegisterForest::Ownership *SSARegisterForest::createAssignment(
    PhysicalSpan Span, SlotIndex Start, SlotIndex End, Register Owner) {
  return new (OwnershipAllocator) Ownership{Start, End, Owner, Span};
}

bool SSARegisterForest::assign(PhysicalSpan Span, SlotIndex Start,
                               SlotIndex End, Register Owner) {
  if (assignedHome(Owner))
    return false;
  return assignRegion(Span, Start, End, Owner);
}

bool SSARegisterForest::assignRegion(PhysicalSpan Span, SlotIndex Start,
                                     SlotIndex End, Register Owner) {
  std::optional<NodeCover> Cover = cover(Span);
  if (!Cover || !validOwnership(Start, End, Owner))
    return false;

  for (NodeIndex Node : *Cover)
    if (subtreeOrAncestorInterference(Node, Start, End))
      return false;

  Ownership *Assignment = createAssignment(Span, Start, End, Owner);
  for (NodeIndex Node : *Cover)
    insertOwnership(Node, Assignment);
  return true;
}

bool SSARegisterForest::release(PhysicalSpan Span, SlotIndex Start,
                                SlotIndex End, Register Owner) {
  std::optional<NodeCover> Cover = cover(Span);
  if (!Cover || !validOwnership(Start, End, Owner))
    return false;

  const Ownership *Assignment = findAssignment(Span, *Cover, Start, End, Owner);
  if (!Assignment)
    return false;
  for (NodeIndex Node : *Cover)
    eraseOwnership(Node, Assignment);
  clearHomeIfEmpty(Owner);
  return true;
}

bool SSARegisterForest::replace(OwnershipRegion Original, Register Owner,
                                 ArrayRef<OwnershipRegion> Replacements) {
  std::optional<NodeCover> OriginalCover = cover(Original.Span);
  if (!OriginalCover || !validOwnership(Original.Start, Original.End, Owner))
    return false;

  const Ownership *OriginalAssignment = findAssignment(
      Original.Span, *OriginalCover, Original.Start, Original.End, Owner);
  if (!OriginalAssignment)
    return false;

  SmallVector<NodeCover, 4> Covers;
  for (std::size_t I = 0; I != Replacements.size(); ++I) {
    const OwnershipRegion &R = Replacements[I];
    std::optional<NodeCover> Cover = cover(R.Span);
    if (!Cover || !validOwnership(R.Start, R.End, Owner) ||
        R.Span.FirstPhysicalLeaf < Original.Span.FirstPhysicalLeaf ||
        Original.Span.EndPhysicalLeaf < R.Span.EndPhysicalLeaf ||
        R.Start < Original.Start || Original.End < R.End)
      return false;

    for (const OwnershipRegion &Previous : Replacements.take_front(I))
      if (R.overlaps(Previous))
        return false;
    Covers.push_back(std::move(*Cover));
  }

  // Containment in the exact original assignment proves that no other owner
  // can interfere. Prepare every replacement record before removing that owner;
  // all recoverable validation failures precede the first mutation.
  SmallVector<Ownership *, 4> Assignments;
  for (const OwnershipRegion &R : Replacements)
    Assignments.push_back(createAssignment(R.Span, R.Start, R.End, Owner));

  for (NodeIndex Node : *OriginalCover)
    eraseOwnership(Node, OriginalAssignment);
  for (std::size_t I = 0; I != Covers.size(); ++I)
    for (NodeIndex Node : Covers[I])
      insertOwnership(Node, Assignments[I]);
  clearHomeIfEmpty(Owner);
  return true;
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

void SSARegisterForest::visitOwnershipComponents(
    function_ref<void(PhysicalSpan, const Ownership &)> Visit) const {
  for (NodeIndex I = 0; I != numNodes(); ++I)
    for (const Ownership *Owned : Nodes[I].ownership())
      Visit(nodeAt(I)->Span, *Owned);
}

bool SSARegisterForest::visitOwnerAssignments(
    Register Owner, function_ref<void(const Ownership &)> VisitAssignment) const {
  if (!Owner.isVirtual())
    return false;
  unsigned OwnerIndex = Owner.virtRegIndex();
  if (OwnerIndex >= Owners.size() || Owners[OwnerIndex].Nodes.none())
    return false;
  for (unsigned Node : Owners[OwnerIndex].Nodes.set_bits()) {
    const unsigned FirstLeaf = nodeAt(Node)->Span.FirstPhysicalLeaf;
    for (const Ownership *Owned : Nodes[Node].ownership())
      // Only the first canonical component starts at the logical span's first
      // leaf. Emit there, avoiding a temporary set to deduplicate pointers.
      if (Owned->Owner == Owner && Owned->Span.FirstPhysicalLeaf == FirstLeaf)
        VisitAssignment(*Owned);
  }
  return true;
}
