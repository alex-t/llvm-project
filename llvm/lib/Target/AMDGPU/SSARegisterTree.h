//===- SSARegisterTree.h - Aligned power-of-two register tree ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SSARegisterTree models the physical register file as a lazy binary segment
// tree over N leaves (N a power of two). It is the first, target-independent
// increment of the SSARA "register tree": a pure aligned tree with no LLVM
// register-class knowledge yet.
//
//   * leaves         = 32-bit register units,
//   * internal nodes = aligned power-of-two tuples (64/128/256-bit ...),
//   * root           = the widest aligned tuple (all N leaves).
//
// An "aligned block" of width w = 2^k starts at a leaf index that is a multiple
// of w and spans exactly w leaves. Every such block corresponds to exactly one
// node of the tree. All mutating operations touch a single node and then walk
// that node's root-path updating aggregates, giving O(log N) alloc/free and
// O(log N) (or better) queries.
//
// Later increments will layer odd widths and VCC/M0 on top; this class stays
// deliberately small and dependency-light so it is trivially unit-testable.
//
// ---------------------------------------------------------------------------
// INCREMENT 2a (this file): the core OCCUPANCY tree is now real --
// allocateAligned / freeAligned / isFree / freeCount / dump are implemented via
// a heap-laid-out binary segment tree with per-node free-leaf counts, mutated by
// a single root-path walk (one node per level). fullCountAtLevel and
// pickFreeAligned remain STUBBED (increments 2b/2c) and still return defaults.
// ---------------------------------------------------------------------------
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SSAREGISTERTREE_H
#define LLVM_LIB_TARGET_AMDGPU_SSAREGISTERTREE_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

class raw_ostream;

/// A lazy binary segment tree over N power-of-two-aligned leaves. See the file
/// header for the model. Throughout: every width is a power of two and every
/// block start index is a multiple of its width ("aligned").
class SSARegisterTree {
public:
  /// Build a tree over \p NumLeaves leaves. \p NumLeaves must be a power of two
  /// and >= 1. The tree starts completely free.
  explicit SSARegisterTree(unsigned NumLeaves);

  /// Total number of leaves (32-bit register units) the tree covers.
  unsigned numLeaves() const { return N; }

  /// Number of tree levels above (and including) the leaves. Leaves are level
  /// 0; a width-2^k aligned block is a level-k node; the root is level
  /// numLevels() (== Log2(numLeaves())).
  unsigned numLevels() const { return NumLevels; }

  /// Mark the aligned block [FirstLeaf, FirstLeaf+Width) occupied.
  ///
  /// \p Width must be a power of two, \p FirstLeaf must be Width-aligned, and
  /// the block must lie fully within [0, numLeaves()). Returns true on success.
  /// Returns false and makes NO change to the tree if the arguments are invalid
  /// or if ANY leaf in the block is already occupied (all-or-nothing).
  bool allocateAligned(unsigned FirstLeaf, unsigned Width);

  /// Free the aligned block [FirstLeaf, FirstLeaf+Width) previously marked
  /// occupied by allocateAligned. Arguments follow the same alignment/range
  /// rules as allocateAligned; invalid arguments are ignored (no-op).
  void freeAligned(unsigned FirstLeaf, unsigned Width);

  /// Return the FirstLeaf index of the lowest-indexed completely-free aligned
  /// block of the given power-of-two \p Width, or -1 if no such block exists.
  /// Implemented via the tree (O(log N)), not a linear scan.
  int pickFreeAligned(unsigned Width) const;

  /// Total number of free leaves across the whole tree (root aggregate). O(1).
  unsigned freeCount() const;

  /// True iff the aligned block [FirstLeaf, FirstLeaf+Width) is entirely free.
  /// Invalid arguments return false.
  bool isFree(unsigned FirstLeaf, unsigned Width) const;

  /// Packing-pressure invariant (I2): the number of FULLY-occupied aligned
  /// blocks of width 2^K currently formed by the placement. Allocating one leaf
  /// increments the level-0 count; allocating its sibling too increments the
  /// level-1 count; filling all four leaves under a width-4 block increments the
  /// level-2 count; and so on. Maintained incrementally on alloc/free, so this
  /// query is O(1).
  unsigned fullCountAtLevel(unsigned K) const;

  /// Print occupancy as one char per leaf, in leaf-index order: '#' for an
  /// occupied leaf, '.' for a free leaf. No trailing newline.
  void dump(raw_ostream &OS) const;

private:
  unsigned N = 0;         // number of leaves (power of two)
  unsigned NumLevels = 0; // Log2(N)

  // Heap-laid-out complete binary tree, 1-based. Node 1 is the root (covering
  // all N leaves); node i has children 2i and 2i+1; leaf j is stored at node
  // N + j. Nodes 1 .. 2N-1 are used, so the arrays have 2N entries (index 0
  // unused). A width-2^k aligned block [f, f+2^k) is exactly one node; walking
  // parent = node/2 from that node to the root visits one node per level.
  //
  // FreeLeaves[i] = number of currently-free leaves within node i's subtree.
  //   The root's value is freeCount(); a node is fully free iff this equals its
  //   span, fully occupied iff this is 0.
  SmallVector<unsigned, 32> FreeLeaves;

  // Marks the single node that an allocateAligned() call claimed, so freeAligned
  // and isFree can distinguish "this exact block was allocated" and so a block
  // is never double-counted. Indexed by node number (size 2N).
  BitVector NodeAllocated;

  // FullAtLevel[L] = number of currently fully-occupied aligned blocks of width
  // 2^L (packing-pressure invariant I2). Maintained incrementally on the
  // alloc/free root-path walk; size NumLevels + 1.
  SmallVector<unsigned, 8> FullAtLevel;

  // MaxFreeAligned[i] = width (in leaves) of the LARGEST completely-free aligned
  // block within node i's subtree (0 if none). Definition:
  //   * if node i is itself completely free (FreeLeaves[i] == spanOf(i)):
  //       MaxFreeAligned[i] = spanOf(i);
  //   * else, an internal node: max(MaxFreeAligned[2i], MaxFreeAligned[2i+1]);
  //   * a leaf: 1 if free, else 0 (subsumed by the "completely free" rule).
  // This is the sufficient admission test pickFreeAligned descends on -- unlike
  // FreeLeaves, which counts free leaves (necessary but NOT sufficient for an
  // aligned block to exist). Maintained incrementally on the same root-path walk.
  SmallVector<unsigned, 32> MaxFreeAligned;

  // Recompute MaxFreeAligned[Node] from its own fullness and its children.
  void refreshMaxFreeAligned(unsigned Node);

  // --- geometry helpers (pure functions of a node index) -------------------
  unsigned spanOf(unsigned Node) const;      // leaves covered by Node
  unsigned firstLeafOf(unsigned Node) const; // lowest leaf index under Node
  unsigned levelOf(unsigned Node) const;     // Log2(spanOf(Node)); root=NumLevels

  // Node index of the aligned block [FirstLeaf, FirstLeaf+Width); callers must
  // have validated the arguments with validBlock() first.
  unsigned nodeOf(unsigned FirstLeaf, unsigned Width) const;

  // True iff (FirstLeaf, Width) names a legal aligned power-of-two block that
  // lies fully within [0, N).
  bool validBlock(unsigned FirstLeaf, unsigned Width) const;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SSAREGISTERTREE_H
