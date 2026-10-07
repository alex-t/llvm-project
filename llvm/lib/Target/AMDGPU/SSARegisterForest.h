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
/// Temporal ownership is recorded on a canonical node or on the disjoint
/// canonical cover of an arbitrary physical span. Queries account for ownership
/// on covering ancestors and contained descendants without copying records into
/// every covered leaf.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SSAREGISTERFOREST_H
#define LLVM_LIB_TARGET_AMDGPU_SSAREGISTERFOREST_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/Allocator.h"
#include <cstddef>
#include <iterator>
#include <optional>

namespace llvm {

class SSARegisterForest {
public:
  using NodeIndex = unsigned;

  /// Fixed physical occupancy has no virtual owner to relocate. Reserve a
  /// nonvirtual Register encoding locally; this is not a stack-slot identity
  /// and must never be passed to MachineRegisterInfo. Zero remains invalid.
  static constexpr Register SELF_OWNED{Register::FirstStackSlot};

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

  /// One spatial and temporal region, independent of its canonical node cover.
  struct OwnershipRegion {
    PhysicalSpan Span;
    SlotIndex Start;
    SlotIndex End;

    bool operator==(const OwnershipRegion &Other) const {
      return Span == Other.Span && Start == Other.Start && End == Other.End;
    }

    /// Both regions must be nonempty. Touching boundaries do not overlap.
    bool overlaps(const OwnershipRegion &Other) const {
      return Span.FirstPhysicalLeaf < Other.Span.EndPhysicalLeaf &&
             Other.Span.FirstPhysicalLeaf < Span.EndPhysicalLeaf &&
             Start < Other.End && Other.Start < End;
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

  /// One logical assignment, shared by all canonical nodes covering Span.
  /// The forest owns this record; node lists contain non-owning references.
  /// Owner is a virtual register or SELF_OWNED. Pointer identity distinguishes
  /// separate assignments with equal owner/time but different physical spans.
  struct Ownership {
    SlotIndex Start;
    SlotIndex End;
    Register Owner;
    PhysicalSpan Span;
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

  /// Visit every live canonical ownership component. The first argument is
  /// the node's span; Ownership::Span is the complete logical assignment.
  /// Composite components reference the same Ownership record. The callback
  /// must not mutate this forest.
  void visitOwnershipComponents(
      function_ref<void(PhysicalSpan, const Ownership &)> VisitComponent) const;

  /// Visit each logical assignment of this virtual owner exactly once.
  /// Uses the owner-to-node bitmap, then filters those nodes' records. Does not
  /// scan unrelated nodes or consult LIS. The callback must not mutate RF.
  /// Return true only when assignments were visited. A nonvirtual or unassigned
  /// owner returns false without callbacks or index growth. Callers expecting
  /// existing ownership must treat false as an integrity error.
  bool visitOwnerAssignments(
      Register Owner, function_ref<void(const Ownership &)> VisitAssignment) const;

  /// Full target home recorded with this owner's node index. Constant-time
  /// lookup; zero means no target assignment. RF stores the identity opaquely:
  /// the adapter validates target classes and projects lanes to physical spans.
  MCRegister assignedHome(Register Owner) const;

  /// Visit each owner with a recorded home once, in virtual-register order.
  /// Includes homes with no live regions; excludes fixed physical ownership.
  /// Reads the existing owner index. The callback must not mutate this forest.
  void
  visitAssignedOwners(function_ref<void(Register, MCRegister)> Visit) const;

  /// Add a batch of regions at one target home, atomically. Reject a different
  /// existing home, unbound geometry-only ownership, or any overlapping region.
  /// The adapter must validate Home and every region's lane projection first.
  /// An empty batch records the home even when the interval has no live lanes.
  /// Metadata lives in the existing owner entry, not a second assignment map.
  bool assign(Register Owner, MCRegister Home,
              ArrayRef<OwnershipRegion> Regions);

  /// Remove all of a virtual owner's records, including complete composite
  /// assignments. Does not require an interval or a physical home from RA.
  /// Also clear its recorded target home. Return true when records or a home
  /// were removed, including a home assigned with no live regions. A nonvirtual
  /// or unassigned owner returns false without mutation. Callers expecting existing ownership
  /// must treat false as an integrity error.
  bool releaseOwner(Register Owner);

  /// Visit canonical ownership components overlapping Span and [Start, End).
  /// Each component is visited once, in node preorder then increasing time.
  /// The callback receives the canonical node span and the shared logical
  /// record, not their intersection with the query. Ownership::Span may extend
  /// beyond the queried node. A composite record may be reported more than once.
  /// The callback must not mutate this forest.
  /// Return false for invalid input, without callbacks; true for a valid query,
  /// including one with no interference.
  bool visitInterferences(
      PhysicalSpan Span, SlotIndex Start, SlotIndex End,
      function_ref<void(PhysicalSpan, const Ownership &)> VisitComponent) const;

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

  /// Return whether Span is free for [Start, End). Span may be aligned or
  /// unaligned and may cross canonical-node or hardware-tree boundaries.
  /// Invalid arguments return false.
  bool isFree(PhysicalSpan Span, SlotIndex Start, SlotIndex End) const;

  /// Return whether every canonical component of Span contains the exact
  /// assignment {Span, Start, End, Owner}. Invalid arguments return false.
  bool contains(PhysicalSpan Span, SlotIndex Start, SlotIndex End,
                Register Owner) const;

  /// Assign Owner to Span for [Start, End). The operation is all-or-nothing
  /// across Span's complete canonical cover. This geometry-only entry point
  /// rejects an owner with a recorded target home; use the home-aware batch
  /// overload for that owner so assignment metadata cannot be bypassed.
  bool assign(PhysicalSpan Span, SlotIndex Start, SlotIndex End,
              Register Owner);

  /// Remove the exact assignment {Span, Start, End, Owner}. The operation is
  /// all-or-nothing across Span's complete canonical cover.
  bool release(PhysicalSpan Span, SlotIndex Start, SlotIndex End,
               Register Owner);

  /// Replace one exact assignment with regions retained by the same owner.
  /// Every replacement must be nonempty and contained in Original. Replacements
  /// must not overlap in both space and time; touching boundaries are permitted.
  /// Empty Replacements removes Original. Each new region becomes an independent
  /// exact assignment. Invalid input leaves all ownership unchanged.
  bool replace(OwnershipRegion Original, Register Owner,
               ArrayRef<OwnershipRegion> Replacements);

private:
  /// Geometry and temporal state owned by one exact physical node. LeafWidth
  /// is initialized with the forest topology and never changes. This class
  /// alone maintains the ordering and non-overlap invariant of the owner list.
  struct NodeState {
    NodeState() = default;
    explicit NodeState(unsigned LeafWidth) : LeafWidth(LeafWidth) {}

    unsigned leafWidth() const { return LeafWidth; }
    ArrayRef<const Ownership *> ownership() const { return OwnedHere; }
    const Ownership *find(SlotIndex Start, SlotIndex End, Register Owner) const;
    bool interferes(SlotIndex Start, SlotIndex End) const;
    void visitInterferences(
        SlotIndex Start, SlotIndex End,
        function_ref<void(const Ownership &)> Visit) const;
    void insert(const Ownership *Owned);
    bool erase(const Ownership *Owned);
    bool hasOwner(Register Owner) const;
    void eraseOwner(Register Owner);

  private:
    /// Return the index of the first interval whose end is after Point.
    std::size_t firstEndingAfter(SlotIndex Point) const;

    unsigned LeafWidth = 0;
    /// Sorted by Start and pairwise temporally disjoint.
    SmallVector<const Ownership *, 2> OwnedHere;
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
  // Indexed by virtRegIndex(), grown on first assignment of a higher owner.
  // A bitmap is allocated only when that owner first receives a record.
  // Bit N means Nodes[N] contains at least one record belonging to that owner.
  // SELF_OWNED is not indexed. Empty entries/bitmaps may remain until destruction.
  struct OwnerState {
    BitVector Nodes;
    MCRegister Home;
  };
  SmallVector<OwnerState, 0> Owners;

  // Called after an entire release/replace, never between component edits.
  void clearHomeIfEmpty(Register Owner);

  // All component insertion/removal goes through these helpers so membership
  // follows the actual node lists, including during exact replacement.
  void insertOwnership(NodeIndex Node, const Ownership *Owned);
  void eraseOwnership(NodeIndex Node, const Ownership *Owned);
  /// Node lists share stable ownership addresses. Released records remain
  /// allocated until the function-local forest is destroyed.
  BumpPtrAllocator OwnershipAllocator;

  using NodeCover = SmallVector<NodeIndex, 4>;

  bool assignRegion(PhysicalSpan Span, SlotIndex Start, SlotIndex End,
                    Register Owner);
  bool validNodeWidth(unsigned Width) const;
  bool validPhysicalSpan(PhysicalSpan Span) const;
  bool validOwnership(SlotIndex Start, SlotIndex End, Register Owner) const;
  std::optional<NodeIndex> canonicalNode(PhysicalSpan Span) const;
  std::optional<NodeCover> cover(PhysicalSpan Span) const;
  const Ownership *findAssignment(PhysicalSpan Span, const NodeCover &Cover,
                                  SlotIndex Start, SlotIndex End,
                                  Register Owner) const;
  bool subtreeOrAncestorInterference(NodeIndex MemoryIndex, SlotIndex Start,
                                     SlotIndex End) const;
  void visitInterferencesInTree(
      NodeRef Node, PhysicalSpan Span, SlotIndex Start, SlotIndex End,
      function_ref<void(PhysicalSpan, const Ownership &)> VisitComponent) const;
  Ownership *createAssignment(PhysicalSpan Span, SlotIndex Start,
                              SlotIndex End, Register Owner);
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
