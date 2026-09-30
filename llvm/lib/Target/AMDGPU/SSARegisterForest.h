//===- SSARegisterForest.h - Physical register forest geometry -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Geometry for the temporal physical-register forest used by AMDGPU SSA
/// register allocation.
///
/// Each equal-width hardware tree is encoded in preorder. Complete tree
/// encodings are concatenated, so NodeIndex is a transparent coordinate over
/// the whole forest without introducing a virtual root or inactive nodes.
/// Temporal ownership is recorded once, at the exact physical node assigned to
/// a value. Queries account for ownership on covering ancestors and contained
/// descendants without copying those records between nodes.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SSAREGISTERFOREST_H
#define LLVM_LIB_TARGET_AMDGPU_SSAREGISTERFOREST_H

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include <cstddef>
#include <iterator>
#include <optional>

namespace llvm {

class SSARegisterForest {
public:
  using NodeIndex = unsigned;

  /// A half-open interval on the global physical-leaf axis.
  struct PhysicalSpan {
    unsigned FirstPhysicalLeaf = 0;
    unsigned EndPhysicalLeaf = 0;

    unsigned width() const { return EndPhysicalLeaf - FirstPhysicalLeaf; }

    bool operator==(const PhysicalSpan &Other) const {
      return FirstPhysicalLeaf == Other.FirstPhysicalLeaf &&
             EndPhysicalLeaf == Other.EndPhysicalLeaf;
    }
  };

  /// A geometry view of one preorder memory coordinate.
  struct NodeRef {
    NodeIndex MemoryIndex = 0;
    PhysicalSpan Span;

    bool operator==(const NodeRef &Other) const {
      return MemoryIndex == Other.MemoryIndex && Span == Other.Span;
    }
  };

  /// One half-open temporal ownership interval stored at its exact physical
  /// node. Owner is a virtual register.
  struct Ownership {
    SlotIndex Start;
    SlotIndex End;
    Register Owner;

    bool operator==(const Ownership &Other) const {
      return Start == Other.Start && End == Other.End && Owner == Other.Owner;
    }
  };

  class NodeLevelRange;

  class NodeLevelIterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = NodeRef;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = NodeRef;

    NodeRef operator*() const;
    NodeLevelIterator &operator++();
    NodeLevelIterator operator++(int);

    bool operator==(const NodeLevelIterator &Other) const;
    bool operator!=(const NodeLevelIterator &Other) const {
      return !(*this == Other);
    }

  private:
    friend class NodeLevelRange;

    NodeLevelIterator(const SSARegisterForest *Forest, unsigned NodeWidth,
                      unsigned FirstPhysicalLeaf)
        : Forest(Forest), NodeWidth(NodeWidth),
          FirstPhysicalLeaf(FirstPhysicalLeaf) {}

    const SSARegisterForest *Forest = nullptr;
    unsigned NodeWidth = 0;
    unsigned FirstPhysicalLeaf = 0;
  };

  /// A sized forward range over every canonical node of one width, in
  /// increasing physical-leaf order. Nodes on one level need not be adjacent
  /// in preorder memory.
  class NodeLevelRange {
  public:
    NodeLevelIterator begin() const { return {Forest, NodeWidth, 0}; }
    NodeLevelIterator end() const {
      return {Forest, NodeWidth, Count * NodeWidth};
    }

    unsigned size() const { return Count; }
    bool empty() const { return Count == 0; }

  private:
    friend class SSARegisterForest;

    NodeLevelRange() = default;
    NodeLevelRange(const SSARegisterForest *Forest, unsigned NodeWidth,
                   unsigned Count)
        : Forest(Forest), NodeWidth(NodeWidth), Count(Count) {}

    const SSARegisterForest *Forest = nullptr;
    unsigned NodeWidth = 0;
    unsigned Count = 0;
  };

  /// Create NumTrees independent complete hardware trees, each covering
  /// TreeWidth leaves. TreeWidth must be a power of two and both arguments
  /// must be nonzero. Reject a topology whose leaf or node count cannot be
  /// represented by NodeIndex.
  static std::optional<SSARegisterForest> create(unsigned NumTrees,
                                                 unsigned TreeWidth);

  unsigned numTrees() const { return NumTrees; }
  unsigned treeWidth() const { return TreeWidth; }
  unsigned treeStride() const { return TreeStride; }
  unsigned numLeaves() const { return NumLeaves; }
  unsigned numNodes() const { return NumNodes; }

  /// Decode one coordinate in the concatenated preorder memory layout.
  std::optional<NodeRef> nodeAt(NodeIndex MemoryIndex) const;

  /// Return all canonical single-node spans of NodeWidth in physical-leaf
  /// order. This is not an arbitrary legal-register-tuple query. Invalid node
  /// widths produce an empty range.
  NodeLevelRange nodeLevel(unsigned NodeWidth) const;

  /// Return the owner recorded exactly at MemoryIndex at At, or no owner.
  /// This does not report ownership recorded on ancestors or descendants.
  std::optional<Register> ownerAt(NodeIndex MemoryIndex, SlotIndex At) const;

  /// Return whether the physical span represented by MemoryIndex is free for
  /// [Start, End). Ownership recorded at the node, on a covering ancestor, or
  /// in any contained descendant interferes. Invalid arguments return false.
  bool isFree(NodeIndex MemoryIndex, SlotIndex Start, SlotIndex End) const;

  /// Assign Owner to the physical span represented by MemoryIndex for
  /// [Start, End). Returns false without mutation if the arguments are invalid
  /// or the span is not free for the complete interval.
  bool assign(NodeIndex MemoryIndex, SlotIndex Start, SlotIndex End,
              Register Owner);

  /// Remove the exact ownership record. Returns false without mutation when no
  /// identical record exists at MemoryIndex.
  bool release(NodeIndex MemoryIndex, SlotIndex Start, SlotIndex End,
               Register Owner);

private:
  /// Geometry and temporal state owned by one exact physical node. LeafWidth
  /// is initialized with the forest topology and never changes. This class
  /// alone maintains the ordering and non-overlap invariant of the owner list.
  struct NodeState {
    NodeState() = default;
    explicit NodeState(unsigned LeafWidth) : LeafWidth(LeafWidth) {}

    unsigned leafWidth() const { return LeafWidth; }
    std::optional<Register> ownerAt(SlotIndex At) const;
    bool interferes(SlotIndex Start, SlotIndex End) const;
    void insert(Ownership Owned);
    bool erase(Ownership Owned);

  private:
    /// Return the index of the first interval whose end is after Point.
    std::size_t firstEndingAfter(SlotIndex Point) const;

    unsigned LeafWidth = 0;
    /// Sorted by Start and pairwise temporally disjoint.
    SmallVector<Ownership, 2> OwnedHere;
  };

  class AncestorRange;

  /// A lazy iterator over strict ancestors from the immediate parent towards
  /// the hardware-tree root.
  class AncestorIterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = NodeIndex;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = NodeIndex;

    NodeIndex operator*() const;
    AncestorIterator &operator++();
    AncestorIterator operator++(int);

    bool operator==(const AncestorIterator &Other) const;
    bool operator!=(const AncestorIterator &Other) const {
      return !(*this == Other);
    }

  private:
    friend class AncestorRange;

    AncestorIterator(const SSARegisterForest *Forest, NodeIndex Current,
                     bool AtEnd)
        : Forest(Forest), Current(Current), AtEnd(AtEnd) {}

    const SSARegisterForest *Forest = nullptr;
    NodeIndex Current = 0;
    bool AtEnd = true;
  };

  class AncestorRange {
  public:
    AncestorIterator begin() const;
    AncestorIterator end() const { return {Forest, 0, true}; }

  private:
    friend class SSARegisterForest;

    AncestorRange(const SSARegisterForest *Forest, NodeIndex Descendant)
        : Forest(Forest), Descendant(Descendant) {}

    const SSARegisterForest *Forest = nullptr;
    NodeIndex Descendant = 0;
  };

  SSARegisterForest(unsigned NumTrees, unsigned TreeWidth, unsigned TreeStride,
                    unsigned NumLeaves, unsigned NumNodes)
      : NumTrees(NumTrees), TreeWidth(TreeWidth), TreeStride(TreeStride),
        NumLeaves(NumLeaves), NumNodes(NumNodes), Nodes(NumNodes) {
    for (unsigned Tree = 0; Tree != NumTrees; ++Tree)
      initializeTree(Tree * TreeStride, TreeWidth);
  }

  unsigned NumTrees = 0;
  unsigned TreeWidth = 0;
  unsigned TreeStride = 0;
  unsigned NumLeaves = 0;
  unsigned NumNodes = 0;
  SmallVector<NodeState, 0> Nodes;

  bool validNodeWidth(unsigned Width) const;
  bool validOwnership(NodeIndex MemoryIndex, SlotIndex Start, SlotIndex End,
                      Register Owner) const;
  bool subtreeOrAncestorInterference(NodeIndex MemoryIndex, SlotIndex Start,
                                     SlotIndex End) const;
  void initializeTree(NodeIndex Root, unsigned LeafWidth);
  NodeIndex treeRootIndex(NodeIndex MemoryIndex) const;
  NodeIndex parentIndex(NodeIndex Child) const;
  iota_range<NodeIndex> subtree(NodeIndex Root) const;
  AncestorRange ancestors(NodeIndex Descendant) const;
  NodeIndex nodeIndexForSpan(PhysicalSpan Span) const;
  NodeRef nodeLevelRef(unsigned NodeWidth, unsigned FirstPhysicalLeaf) const;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SSAREGISTERFOREST_H
