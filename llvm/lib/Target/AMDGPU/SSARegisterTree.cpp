//===- SSARegisterTree.cpp - Aligned power-of-two register tree ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// INCREMENTS 2a + 2b.
//
// Storage is a heap-laid-out complete binary tree, 1-based:
//   * node 1        = root, spans all N leaves,
//   * node i        = children 2i (left) and 2i+1 (right),
//   * node N + j    = leaf j.
// So nodes 1 .. 2N-1 are live and the arrays hold 2N entries (index 0 unused).
//
// A width-2^k aligned block [f, f+2^k) is EXACTLY one node. Mutating it therefore
// touches that one node and then walks parent = node/2 up to the root -- one node
// per level, O(log N). We keep a lazy per-node free-leaf count: allocating a big
// block zeroes that node's count and decrements every ancestor, but leaves the
// (now irrelevant) descendant counts stale. To compensate, a query walks the
// node's ancestors checking the NodeAllocated flag so a block hidden underneath a
// larger allocation is correctly reported occupied.
//
// 2b: fullCountAtLevel(L) counts fully-occupied aligned blocks of width 2^L. It
// is maintained incrementally in the FullAtLevel[] array during the same
// root-path walk (no rescan). Allocating a width-2^k block:
//   * makes the block itself and ALL its sub-blocks full -- accounted directly by
//     FullAtLevel[j] += 2^(k-j) for j in 0..k (a full width-2^k block contains
//     2^(k-j) full sub-blocks at level j; j==k is the block itself);
//   * may cascade UPWARD -- an ancestor becomes full exactly when its FreeLeaves
//     hits 0 during the walk, so we bump FullAtLevel[levelOf(ancestor)] then.
// freeAligned reverses both halves symmetrically. Only O(log N) levels change.
//
// pickFreeAligned() is deliberately still stubbed here (increment 2c).
//
//===----------------------------------------------------------------------===//

#include "SSARegisterTree.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

SSARegisterTree::SSARegisterTree(unsigned NumLeaves) {
  N = NumLeaves;
  NumLevels = 0;
  for (unsigned W = 1; W < NumLeaves; W <<= 1)
    ++NumLevels;

  // 2N entries (1-based; slot 0 unused). Each node starts fully free: its free
  // count equals its span, and its largest free aligned block is its whole span.
  FreeLeaves.assign(2 * N, 0);
  MaxFreeAligned.assign(2 * N, 0);
  NodeAllocated.resize(2 * N, false);
  for (unsigned Node = 1; Node < 2 * N; ++Node) {
    FreeLeaves[Node] = spanOf(Node);
    MaxFreeAligned[Node] = spanOf(Node);
  }

  // Nothing is occupied yet, so no aligned block of any width is full.
  FullAtLevel.assign(NumLevels + 1, 0);
}

// MaxFreeAligned for a node: its whole span if the node is itself completely
// free, else the larger of its two children's aggregates. For a leaf (span 1)
// the "completely free" branch already yields 1-if-free / 0-if-taken.
void SSARegisterTree::refreshMaxFreeAligned(unsigned Node) {
  unsigned Span = spanOf(Node);
  if (FreeLeaves[Node] == Span) {
    MaxFreeAligned[Node] = Span;
    return;
  }
  if (Span == 1) {
    MaxFreeAligned[Node] = 0; // occupied leaf
    return;
  }
  unsigned Left = 2 * Node, Right = Left + 1;
  MaxFreeAligned[Node] = std::max(MaxFreeAligned[Left], MaxFreeAligned[Right]);
}

// depth(Node) = floor(log2(Node)); span = N >> depth.
unsigned SSARegisterTree::spanOf(unsigned Node) const {
  unsigned Depth = Log2_32(Node);
  return N >> Depth;
}

unsigned SSARegisterTree::firstLeafOf(unsigned Node) const {
  unsigned Depth = Log2_32(Node);
  unsigned IndexInLevel = Node - (1u << Depth);
  return IndexInLevel * (N >> Depth);
}

// A node at depth d spans N>>d leaves, i.e. width 2^(NumLevels-d), so its level
// is NumLevels - depth. The root (depth 0) is level NumLevels; a leaf is level 0.
unsigned SSARegisterTree::levelOf(unsigned Node) const {
  return NumLevels - Log2_32(Node);
}

unsigned SSARegisterTree::nodeOf(unsigned FirstLeaf, unsigned Width) const {
  unsigned K = Log2_32(Width);    // level of the block
  unsigned Depth = NumLevels - K; // depth from the root
  return (1u << Depth) + (FirstLeaf / Width);
}

bool SSARegisterTree::validBlock(unsigned FirstLeaf, unsigned Width) const {
  if (Width == 0 || !isPowerOf2_32(Width))
    return false;
  if (Width > N)
    return false;
  if ((FirstLeaf & (Width - 1)) != 0) // FirstLeaf must be Width-aligned
    return false;
  if (FirstLeaf + Width > N) // block must lie fully within [0, N)
    return false;
  return true;
}

bool SSARegisterTree::allocateAligned(unsigned FirstLeaf, unsigned Width) {
  if (!validBlock(FirstLeaf, Width))
    return false;

  unsigned Node = nodeOf(FirstLeaf, Width);

  // Reject unless the whole block is free: (a) no descendant leaf is occupied
  // -- the node's own count still equals its span; and (b) no ancestor block is
  // allocated over it.
  if (FreeLeaves[Node] != Width)
    return false;
  for (unsigned A = Node / 2; A >= 1; A /= 2)
    if (NodeAllocated[A])
      return false;

  // Claim the node. The block and all of its sub-blocks are now full: a full
  // width-2^k block contains 2^(k-j) full sub-blocks at each level j <= k.
  NodeAllocated[Node] = true;
  FreeLeaves[Node] = 0;
  // The node is now fully occupied, so it contains no free aligned block.
  // (Its stale descendants are never observed: the descent prunes here on 0.)
  MaxFreeAligned[Node] = 0;
  unsigned K = Log2_32(Width); // block's level
  for (unsigned J = 0; J <= K; ++J)
    FullAtLevel[J] += (1u << (K - J));

  // Walk the root-path: propagate the lost free leaves, refresh each ancestor's
  // largest-free-aligned aggregate from its (already-updated) children, and where
  // an ancestor transitions to fully occupied bump its level's full count.
  for (unsigned A = Node / 2; A >= 1; A /= 2) {
    FreeLeaves[A] -= Width;
    if (FreeLeaves[A] == 0)
      ++FullAtLevel[levelOf(A)];
    refreshMaxFreeAligned(A);
  }
  return true;
}

void SSARegisterTree::freeAligned(unsigned FirstLeaf, unsigned Width) {
  if (!validBlock(FirstLeaf, Width))
    return;

  unsigned Node = nodeOf(FirstLeaf, Width);
  // Only an exact, currently-allocated block may be freed.
  if (!NodeAllocated[Node])
    return;

  // Restore the freed node first: it is completely free again, so its whole span
  // is once more a free aligned block. This must happen before we refresh its
  // parent (refreshMaxFreeAligned reads the children's aggregates).
  NodeAllocated[Node] = false;
  FreeLeaves[Node] = Width;
  MaxFreeAligned[Node] = Width;

  // Undo the block-and-descendants full-count accounting.
  unsigned K = Log2_32(Width);
  for (unsigned J = 0; J <= K; ++J)
    FullAtLevel[J] -= (1u << (K - J));

  // Reverse the ancestor cascade: an ancestor that WAS full stops being full as
  // we add leaves back, so decrement its level's count before restoring the
  // count, then refresh its largest-free-aligned aggregate from its children.
  for (unsigned A = Node / 2; A >= 1; A /= 2) {
    if (FreeLeaves[A] == 0)
      --FullAtLevel[levelOf(A)];
    FreeLeaves[A] += Width;
    refreshMaxFreeAligned(A);
  }
}

bool SSARegisterTree::isFree(unsigned FirstLeaf, unsigned Width) const {
  if (!validBlock(FirstLeaf, Width))
    return false;

  unsigned Node = nodeOf(FirstLeaf, Width);
  if (FreeLeaves[Node] != Width)
    return false; // a descendant leaf is occupied
  for (unsigned A = Node / 2; A >= 1; A /= 2)
    if (NodeAllocated[A])
      return false; // hidden under a larger allocation
  return true;
}

unsigned SSARegisterTree::freeCount() const {
  return N == 0 ? 0 : FreeLeaves[1];
}

unsigned SSARegisterTree::fullCountAtLevel(unsigned K) const {
  return K < FullAtLevel.size() ? FullAtLevel[K] : 0;
}

int SSARegisterTree::pickFreeAligned(unsigned Width) const {
  // Mirror validBlock's argument rejects, but return -1 (not false).
  if (Width == 0 || !isPowerOf2_32(Width) || Width > N)
    return -1;

  // Root prune: MaxFreeAligned[1] is the widest free aligned block anywhere.
  if (MaxFreeAligned[1] < Width)
    return -1;

  // Descend to the target level (depth NumLevels-K, where each node spans W),
  // always taking the LEFT child first when its subtree still contains a free
  // aligned block of at least Width -- this yields the LOWEST firstLeaf.
  //
  // MaxFreeAligned (not FreeLeaves) is the correct admission test: FreeLeaves
  // only counts free leaves, which can be >= Width while no *aligned* block of
  // that width exists (e.g. occupancy ".#.#" has 2 free leaves but no free
  // aligned pair). MaxFreeAligned answers the aligned-block question directly.
  unsigned TargetDepth = NumLevels - Log2_32(Width);
  unsigned Node = 1;
  for (unsigned Depth = 0; Depth < TargetDepth; ++Depth) {
    unsigned Left = 2 * Node;
    unsigned Right = Left + 1;
    if (MaxFreeAligned[Left] >= Width)
      Node = Left; // prefer the lower-indexed subtree
    else
      Node = Right; // guaranteed by the invariant: parent had a free width-W
                    // block, so if the left child cannot host it the right can.
  }

  // Node is the lowest-indexed target-level node that is completely free.
  return static_cast<int>(firstLeafOf(Node));
}

void SSARegisterTree::dump(raw_ostream &OS) const {
  for (unsigned J = 0; J < N; ++J)
    OS << (isFree(J, 1) ? '.' : '#');
}
