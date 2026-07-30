//===- SSARegisterTreeTest.cpp - Unit tests for SSARegisterTree ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Golden-file (snapshot) unit tests for the pure aligned register tree
// (SSARegisterTree, Increment 1). The tree models the physical register file as
// a lazy binary segment tree over N leaves (N a power of two). Leaves are 32-bit
// register units; internal nodes are aligned power-of-two tuples.
//
// TEST DESIGN (per the tech lead):
//   * WRITE test  -- drives a fixed, deterministic SEQUENCE of allocateAligned /
//                    freeAligned calls and, after every step, snapshots the
//                    aggregates that a root-path update maintains (occupancy,
//                    freeCount, per-level fullCount). The concatenated trace is
//                    compared against a hand-authored GOLDEN string.
//   * READ  test  -- replays the sequence, then uses ONLY the read API (dump +
//                    query methods) to serialize the final tree state and
//                    compares it against a hand-authored GOLDEN string.
//
// GOLDEN STORAGE: the golden strings are embedded as raw string literals below
// (kWriteTraceGolden, kFinalStateGolden). This matches the existing AMDGPU
// unittest convention -- those tests are self-contained and do no file I/O, and
// there is no Inputs/ directory wired up for this unittest target. The literals
// are laid out one-line-per-fact for human review.
//
// STAGE 1 STATUS: SSARegisterTree is a STUB returning default values, so these
// tests are EXPECTED TO BE RED. The golden strings are the SPECIFICATION of the
// correct behavior the real (Stage 2) implementation must reproduce.
//
//===----------------------------------------------------------------------===//

#include "SSARegisterTree.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <string>

using namespace llvm;

namespace {

//===----------------------------------------------------------------------===//
// The deterministic write sequence (all on a 16-leaf tree).
//
//   #0 (initial)            empty tree
//   #1 allocateAligned(0,1) occupy leaf 0
//   #2 allocateAligned(1,1) occupy leaf 1        -> [0,2) becomes full
//   #3 allocateAligned(4,4) occupy leaves 4..7
//   #4 allocateAligned(2,2) occupy leaves 2,3    -> [0,4) becomes full
//   #5 allocateAligned(8,8) occupy leaves 8..15  -> whole tree full
//   #6 freeAligned(4,4)     free leaves 4..7
//
// Final occupancy: leaves {0,1,2,3, 8..15} occupied, {4,5,6,7} free.
//===----------------------------------------------------------------------===//

// Serialize one step of the write trace: label, occupancy map, freeCount, and
// the per-level fullCount vector (levels 0 .. numLevels()).
static std::string serializeStep(const char *Label, const SSARegisterTree &T) {
  std::string S;
  raw_string_ostream OS(S);
  OS << Label << '\n';
  OS << "  occupancy: ";
  T.dump(OS);
  OS << '\n';
  OS << "  freeCount: " << T.freeCount() << '\n';
  OS << "  full:     ";
  for (unsigned K = 0; K <= T.numLevels(); ++K)
    OS << ' ' << T.fullCountAtLevel(K);
  OS << '\n';
  return OS.str();
}

// Serialize the full final state via the READ api only.
static std::string serializeState(const SSARegisterTree &T) {
  std::string S;
  raw_string_ostream OS(S);
  OS << "SSARegisterTree state\n";
  OS << "leaves: " << T.numLeaves() << '\n';
  OS << "levels: " << T.numLevels() << '\n';
  OS << "occupancy: ";
  T.dump(OS);
  OS << '\n';
  OS << "freeCount: " << T.freeCount() << '\n';
  for (unsigned K = 0; K <= T.numLevels(); ++K)
    OS << "fullCountAtLevel[" << K << "]: " << T.fullCountAtLevel(K) << '\n';
  for (unsigned W = 1; W <= T.numLeaves(); W <<= 1)
    OS << "pickFreeAligned[" << W << "]: " << T.pickFreeAligned(W) << '\n';
  return OS.str();
}

// Drive the fixed sequence, appending a snapshot after every step.
static std::string runWriteTrace() {
  SSARegisterTree T(16);
  std::string Trace;
  Trace += serializeStep("#0 initial", T);
  T.allocateAligned(0, 1);
  Trace += serializeStep("#1 allocateAligned(0,1)", T);
  T.allocateAligned(1, 1);
  Trace += serializeStep("#2 allocateAligned(1,1)", T);
  T.allocateAligned(4, 4);
  Trace += serializeStep("#3 allocateAligned(4,4)", T);
  T.allocateAligned(2, 2);
  Trace += serializeStep("#4 allocateAligned(2,2)", T);
  T.allocateAligned(8, 8);
  Trace += serializeStep("#5 allocateAligned(8,8)", T);
  T.freeAligned(4, 4);
  Trace += serializeStep("#6 freeAligned(4,4)", T);
  return Trace;
}

// Replay the sequence and return the final tree (for the READ test).
static SSARegisterTree buildFinalTree() {
  SSARegisterTree T(16);
  T.allocateAligned(0, 1);
  T.allocateAligned(1, 1);
  T.allocateAligned(4, 4);
  T.allocateAligned(2, 2);
  T.allocateAligned(8, 8);
  T.freeAligned(4, 4);
  return T;
}

//===----------------------------------------------------------------------===//
// GOLDEN: step-by-step write trace.
//
// occupancy: one char per leaf ('#' occupied, '.' free), leaf 0 leftmost.
// full:      fullCountAtLevel for levels 0 1 2 3 4 (width 1 2 4 8 16).
//
// The packing-pressure invariant (I2) is visible here: e.g. after #2 the two
// filled siblings 0,1 form one full width-2 block (full[1] == 1); after #4 the
// four filled leaves 0..3 form one full width-4 block (full[2] == 1); after #5
// the whole tree is one full width-16 block (full[4] == 1).
//===----------------------------------------------------------------------===//
static const char *const kWriteTraceGolden =
    R"golden(#0 initial
  occupancy: ................
  freeCount: 16
  full:      0 0 0 0 0
#1 allocateAligned(0,1)
  occupancy: #...............
  freeCount: 15
  full:      1 0 0 0 0
#2 allocateAligned(1,1)
  occupancy: ##..............
  freeCount: 14
  full:      2 1 0 0 0
#3 allocateAligned(4,4)
  occupancy: ##..####........
  freeCount: 10
  full:      6 3 1 0 0
#4 allocateAligned(2,2)
  occupancy: ########........
  freeCount: 8
  full:      8 4 2 1 0
#5 allocateAligned(8,8)
  occupancy: ################
  freeCount: 0
  full:      16 8 4 2 1
#6 freeAligned(4,4)
  occupancy: ####....########
  freeCount: 4
  full:      12 6 3 1 0
)golden";

//===----------------------------------------------------------------------===//
// GOLDEN: final state after the full sequence, via the READ api.
//
// Final occupancy: {0,1,2,3, 8..15} occupied, {4,5,6,7} free.
//   freeCount        = 4
//   fullCountAtLevel = [12, 6, 3, 1, 0]
//   pickFreeAligned  = 4 for widths 1/2/4 (block [4,8) is the lowest free);
//                      -1 for widths 8/16 (no aligned free block that wide).
//===----------------------------------------------------------------------===//
static const char *const kFinalStateGolden =
    R"golden(SSARegisterTree state
leaves: 16
levels: 4
occupancy: ####....########
freeCount: 4
fullCountAtLevel[0]: 12
fullCountAtLevel[1]: 6
fullCountAtLevel[2]: 3
fullCountAtLevel[3]: 1
fullCountAtLevel[4]: 0
pickFreeAligned[1]: 4
pickFreeAligned[2]: 4
pickFreeAligned[4]: 4
pickFreeAligned[8]: -1
pickFreeAligned[16]: -1
)golden";

//===----------------------------------------------------------------------===//
// Tests.
//===----------------------------------------------------------------------===//

// WRITE test: the aggregate trace maintained on the way up the root-path must
// match the golden step-by-step snapshot.
TEST(SSARegisterTreeTest, WriteTraceMatchesGolden) {
  EXPECT_EQ(runWriteTrace(), std::string(kWriteTraceGolden));
}

// READ test: after the sequence, the read-api serialization must match the
// golden final-state snapshot.
TEST(SSARegisterTreeTest, FinalStateMatchesGolden) {
  SSARegisterTree T = buildFinalTree();
  EXPECT_EQ(serializeState(T), std::string(kFinalStateGolden));
}

//===----------------------------------------------------------------------===//
// Reject tests: a rejected allocateAligned must return false AND leave the tree
// byte-identical. We prove "byte-identical" by comparing the full read-api
// serialization before and after the rejected call.
//===----------------------------------------------------------------------===//

// Unaligned firstLeaf: 3 is not a multiple of 2, so allocateAligned(3,2) is
// illegal and must be a no-op.
TEST(SSARegisterTreeTest, RejectUnalignedFirstLeafNoOp) {
  SSARegisterTree T(16);
  // Put the tree into a non-trivial state first.
  ASSERT_TRUE(T.allocateAligned(0, 2));
  ASSERT_TRUE(T.allocateAligned(8, 4));

  std::string Before = serializeState(T);
  EXPECT_FALSE(T.allocateAligned(3, 2)); // firstLeaf 3 not width-2 aligned
  std::string After = serializeState(T);
  EXPECT_EQ(Before, After); // byte-identical: no state change
}

// Overlapping allocations: after claiming [4,8), any block that overlaps it must
// be rejected with no state change -- a sub-block (4,2), the containing
// super-block (0,8), a straddling/unaligned block (6,4), and the exact block
// again (4,4).
TEST(SSARegisterTreeTest, RejectOverlappingBlocksNoOp) {
  SSARegisterTree T(16);
  ASSERT_TRUE(T.allocateAligned(4, 4)); // occupy [4,8)

  std::string Before = serializeState(T);

  // Sub-block of the occupied [4,8): overlaps -> reject, no-op.
  EXPECT_FALSE(T.allocateAligned(4, 2));
  EXPECT_EQ(serializeState(T), Before);

  // Containing super-block [0,8) overlaps the occupied block -> reject, no-op.
  EXPECT_FALSE(T.allocateAligned(0, 8));
  EXPECT_EQ(serializeState(T), Before);

  // (6,4): straddles [4,8) and is not width-4 aligned -> reject, no-op.
  EXPECT_FALSE(T.allocateAligned(6, 4));
  EXPECT_EQ(serializeState(T), Before);

  // Re-allocating the exact same occupied block must also fail, no-op.
  EXPECT_FALSE(T.allocateAligned(4, 4));
  EXPECT_EQ(serializeState(T), Before);
}

} // namespace
