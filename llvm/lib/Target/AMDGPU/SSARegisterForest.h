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
/// This first iteration contains no temporal owners or occupancy caches.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SSAREGISTERFOREST_H
#define LLVM_LIB_TARGET_AMDGPU_SSAREGISTERFOREST_H

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

private:
  SSARegisterForest(unsigned NumTrees, unsigned TreeWidth, unsigned TreeStride,
                    unsigned NumLeaves, unsigned NumNodes)
      : NumTrees(NumTrees), TreeWidth(TreeWidth), TreeStride(TreeStride),
        NumLeaves(NumLeaves), NumNodes(NumNodes) {}

  unsigned NumTrees = 0;
  unsigned TreeWidth = 0;
  unsigned TreeStride = 0;
  unsigned NumLeaves = 0;
  unsigned NumNodes = 0;

  bool validNodeWidth(unsigned Width) const;
  NodeIndex nodeIndexForSpan(PhysicalSpan Span) const;
  NodeRef nodeLevelRef(unsigned NodeWidth, unsigned FirstPhysicalLeaf) const;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SSAREGISTERFOREST_H
