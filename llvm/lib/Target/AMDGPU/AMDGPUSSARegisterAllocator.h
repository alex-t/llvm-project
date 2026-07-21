//===-- AMDGPUSSARegisterAllocator.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// SSA-based Register Allocator for AMDGPU.
///
/// Implements width-descending multi-pass PEO coloring based on:
/// "Register Allocation for Programs in SSA-Form"
/// Sebastian Hack, Daniel Grund, Gerhard Goos (CC'06)
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSSAREGISTERALLOCATOR_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSSAREGISTERALLOCATOR_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "SSASpillEmitter.h"
#include <memory>
#include <set>

namespace llvm {

class GCNSubtarget;
class SIInstrInfo;
class SIRegisterInfo;

class AMDGPUSSARegisterAllocator : public MachineFunctionPass {
  const SIRegisterInfo *TRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  MachineDominatorTree *MDT = nullptr;
  SlotIndexes *Indexes = nullptr;
  LiveIntervals *LIS = nullptr;
  MachineLoopInfo *MLI = nullptr;
  const GCNSubtarget *ST = nullptr;
  RegisterClassInfo RegClassInfo;

  std::set<unsigned, std::greater<unsigned>> ColoringOrder;
  DenseMap<Register, MCRegister> ColorMap;
  BitVector OccupiedRegUnits;

  // Width-tiered coloring (experiment, -amdgpu-ssa-virgin-order). Per tier the
  // explicit VIRGIN allocation order — aligned tuples of that (pool, width) that
  // NO already-allocated wider value of the SAME pool occupies anywhere in the
  // function. A value colored into one of these cannot interfere with any wider
  // value, so the tier is Hack-compliant BY CONSTRUCTION (auditable: this vector
  // IS the proof — no interference test needed within it). SGPR and VGPR pools
  // are disjoint, so a single pass could handle both; they are kept SEPARATE
  // tiers purely for transparency. Key = (isVector?1:0, widthBits) — unsigned,
  // not bool, since DenseMap's pair key needs a hashable integral. Built once
  // per tier in color()'s width loop.
  DenseMap<std::pair<unsigned, unsigned>, SmallVector<MCRegister, 64>>
      VirginTierOrder;
  void buildVirginTierOrder(bool IsVector, unsigned WidthBits);

  // Per-tier pick tallies (experiment). Keyed like VirginTierOrder, incremented
  // in pickFreePhysReg when a value of that (pool,width) is placed from the
  // virgin order vs the gap-scan. analyzeTierRank reads AND resets the key it
  // reports, so each entry counts exactly one (phase,pool,width) tier walk.
  DenseMap<std::pair<unsigned, unsigned>, unsigned> VirginPickByTier;
  DenseMap<std::pair<unsigned, unsigned>, unsigned> GapPickByTier;

  // Forensic (post-pass) feasibility analysis: over the vregs a tier actually
  // colored (\p TierVRegs) PLUS the ones it could not color (\p FailedVRegs —
  // they still competed for the same virgin pool, so they belong in the rank),
  // compute the tier's interference-graph rank via LIS and compare to the virgin
  // pool size. Emits one [TIERPROOF] line (via errs(), so it survives a
  // downstream crash) whose verdict separates colorer-fault from
  // spiller-under-spill:
  //   rank <= pool, no gap, no fail  -> HACK-OK (pure Hack held — the proof)
  //   rank <= pool, but gap or fail  -> COLORER fault (feasible yet Hack missed)
  //   rank >  pool, fail > 0         -> SPILLER under-spilled (infeasible tier)
  // NOT on the coloring decision path.
  void analyzeTierRank(unsigned Phase, bool IsVector, unsigned WidthBits,
                       ArrayRef<Register> TierVRegs,
                       ArrayRef<Register> FailedVRegs);

  // Gap-scan fallback (virgin exhausted): find a PR of \p RC whose colored
  // occupants' live intervals do NOT overlap \p VI along VI's whole range
  // (queried via LIS) — a gap opened by an occupant's death that we may reuse.
  // Returns 0 if none.
  MCRegister findNonInterferingGap(const TargetRegisterClass *RC,
                                   const LiveInterval &VI);
  void dumpSpanWidthDelta(const TargetRegisterClass *RC, const LiveInterval &VI);
  unsigned MaxVGPRIdx = 0;
  unsigned MaxSGPRIdx = 0;
  unsigned MaxAGPRIdx = 0;
  // (call def-slot, call instruction) for every call; a vreg live across a call
  // must avoid every register the call clobbers (regmask + explicit defs).
  SmallVector<std::pair<SlotIndex, const MachineInstr *>, 8> CallSites;
  unsigned DynVGPRBlockSize = 0;

  // Exec-safe spill/reload emitter, shared with the spiller pass. Used by the
  // approach-A spill-on-coloring-failure path: when color() cannot place a
  // value, the driver spills it here (store-at-def + dominance reloads) and
  // recolors. Created per function in runOnMachineFunction.
  std::unique_ptr<SSASpillEmitter> Emitter;

  // Values that color() could not place (no physreg free across their whole
  // range — the %560/%1072 long-liver class). color() collects ALL of them and
  // finishes the walk (coloring everything else normally), rather than bailing
  // on the first, so that after the driver spills these the ONLY uncolored
  // vregs left are the short reload remainders — which provably settle.
  SmallVector<Register, 8> UncolorableVRegs;

  // === Coloring ===
  void classifyVRegs();
  // On unified-file targets (gfx90a/gfx942: VALU reads/writes AGPRs directly),
  // widen each VGPR-class vreg to the equivalent vector super-class (av_*) when
  // EVERY operand constraint already admits AGPRs. This lets a narrow value draw
  // the virgin AGPR tuples that wider VGPR tuples left free (the Greedy
  // spill-to-AGPR rescue, done as a sound up-front regclass widen rather than a
  // pick-time fallback — keeps each Hack tier drawing from one unified order).
  // Conservative: a sub-register operand blocks the widen (the whole-reg operand
  // constraint test does not apply to a subreg slice). Behind -amdgpu-ssa-agpr-
  // rescue. Must run before classifyVRegs so widened widths feed ColoringOrder.
  void widenToAVOnUnified();
  /// Run a full coloring walk. Colors every placeable value into ColorMap;
  /// appends any value it cannot place to UncolorableVRegs and skips it (does
  /// not occupy a register for it) so the rest of the walk proceeds as if that
  /// value were absent. Does not assert on failure.
  void color();
  /// Color a single value \p R in place against the CURRENT ColorMap /
  /// OccupiedRegUnits, without disturbing any existing assignment. Seeds
  /// occupancy from the colored values whose live range overlaps R's, then picks
  /// a free physreg across R's (short) range. Used to place reload remainders
  /// after a coloring-failure spill. Returns false if no register is free
  /// (should not happen for a width-1 reload — point pressure ≤ limit < file).
  bool colorOneInPlace(Register R);

  /// Coloring-time live-range split (experiment, gated by
  /// -amdgpu-ssa-split-live-ranges). \p Failed is a width-1 value with no
  /// through-lane. Find a colored liver B whose physreg P is occupied across
  /// Failed's whole (short) range ONLY by B and B is LIVE-THROUGH there (no use
  /// inside), spill B across so P frees over Failed's range with no interior
  /// reload, color Failed into P, and keep B's reload in P. Returns true if
  /// Failed was colored this way; false to fall back to spilling Failed itself.
  bool trySplitColorViaBlocker(Register Failed, unsigned RPLimit);

  /// Self-split (experiment, gated by -amdgpu-ssa-split-live-ranges): \p Failed is
  /// a long liver with no through-lane AND no live-through blocker to spill around
  /// (trySplitColorViaBlocker found nothing). Chop Failed into segments, each
  /// short enough that one physreg is free across it, coloring each into that reg.
  /// Only valid when Failed is POINT-FEASIBLE (some PR free at every slot); aborts
  /// (returns false -> caller memory-spills) if any slot has zero free PRs.
  bool trySelfSplitColor(Register Failed);
  // (declared once; defined in the .cpp)

  /// Single linear scan over ColorMap for \p VI: the shared "collect" step of the
  /// gap-scan / split pipeline. ORs the register units of every colored occupant
  /// whose interval overlaps VI into \p OccupiedUnits. \p Overlappers is optional:
  /// when non-null it also collects (occupant vreg, its physreg) for each. The gap
  /// pick (findNonInterferingGap) passes nullptr (needs only occupancy); the
  /// splitter (trySplitColorViaBlocker) passes a vector. NOT cacheable across the
  /// two — they run in different phases with ColorMap mutated between.
  void scanOverlappersForVI(
      const LiveInterval &VI, BitVector &OccupiedUnits,
      SmallVectorImpl<std::pair<Register, MCRegister>> *Overlappers = nullptr);
  void seedOccupiedAtBBEntry(MachineBasicBlock *MBB);
  // True if the parallel PHI edge-copies for Pred->MBB cannot be safely placed
  // at Pred's terminator (they would clobber a value live into a sibling
  // successor, or need a scratch register for a cycle), i.e. the critical edge
  // must be split. A non-critical edge never needs splitting.
  bool
  edgeCopiesNeedSplit(MachineBasicBlock *Pred, MachineBasicBlock *MBB,
                      ArrayRef<std::pair<MCRegister, MCRegister>> Copies) const;
  void markOccupied(MCRegister PhysReg);
  void markFree(MCRegister PhysReg);

  /// Debug: print \p RC's allocation order at slot \p SI as an occupancy map,
  /// one char per register in order:
  ///   '.' free and usable    '#' occupied (ColorMap vreg live at SI)
  ///   'x' free but CLOBBERED by a call \p VI is live across (unusable by a
  ///       value pinned to callee-saved) — only marked when \p VI is given.
  /// So "####xxxx####" shows callee-saved full ('#') with only caller-saved
  /// ('x') free — the classic pinned-value exhaustion. Occupancy is derived by
  /// walking ColorMap for vregs live at SI. \p Tag labels the line.
  void dumpOccupancyMap(const TargetRegisterClass *RC, SlotIndex SI,
                        const char *Tag, const LiveInterval *VI = nullptr) const;

  // === Value-flow correctness verifier (-amdgpu-ssa-verify-value-flow) ===
  // Certifies SSA-destruction + physreg assignment preserved VALUE IDENTITY:
  // every physreg use holds the SSA value its vreg operand named. Catches the
  // clobber-while-live class (a live value overwritten in its register) that
  // liveness / reaching-def cannot see (both only tell "is there A value", never
  // "is it THE value"). Ground truth = a snapshot taken PRE-destruction, while
  // values are still vregs. v1 certifies single-basic-block functions (~92% of
  // the corpus green set); multi-block functions are reported SKIP (uncertified)
  // pending the meet-at-joins + dominance-rescue layer.
  struct VFOp {
    unsigned VReg;
    unsigned SubReg;
    bool IsDef;
  };
  DenseMap<const MachineInstr *, SmallVector<VFOp, 4>> VFIntent;
  DenseMap<Register, MCRegister> VFColor; // ColorMap frozen pre-destruction
  DenseMap<uint64_t, uint64_t> VFUF;      // union-find over (vreg,lane) keys
  // (vreg,lane) keys that receive a REAL (non-undef) definition. A use only
  // checks lanes in this set: a partial def `undef %V.sub0 = ...` leaves other
  // lanes intentionally undefined (don't-care), and a later read of %V must not
  // demand a value token for those lanes (else false clobber).
  DenseSet<uint64_t> VFDefinedLane;
  uint64_t vfFind(uint64_t X);
  void vfUnion(uint64_t A, uint64_t B);
  void snapshotValueFlow(MachineFunction &MF); // call BEFORE lowerPHIs
  bool verifyValueFlow(MachineFunction &MF);   // call AFTER finalizeProperties

  MCRegister pickFreePhysReg(
      const TargetRegisterClass *RC, const LiveInterval &VI,
      ArrayRef<std::pair<MCRegister, const LiveInterval *>> WiderDefs,
      ArrayRef<MCRegister> Hints = {});

  // Option B affinity: collect already-colored phi-partner physregs for VReg
  // (phi results it feeds, and -- if VReg is a phi result -- its operands),
  // ordered by 2^loopdepth of the incoming edge (hottest first). Sub-register
  // relationships compose both ways: a phi-result VReg reading a slice of a
  // wider operand takes that slice of the operand's color (getSubReg); a wide
  // VReg feeding a narrow result via VReg.subN takes the super-register whose
  // slice is the result's color (getMatchingSuperReg -- the loop-carried tuple
  // case). Class-compatible partners only. Returns [] when VReg touches no
  // colored phi partner.
  SmallVector<MCRegister, 4> collectPhiHints(Register VReg,
                                             const TargetRegisterClass *RC);

  // === SSA Destruction + Operand Rewrite ===
  bool hasCFPseudos(MachineFunction &MF) const;
  void destroySSAAndRewrite(MachineFunction &MF);
  void lowerPHIs(MachineFunction &MF);
  void resolvePermutation(
      MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
      SmallVectorImpl<std::pair<MCRegister, MCRegister>> &Copies);
  void emitSwap(MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
                MCRegister RegA, MCRegister RegB);
  void rewriteOperands(MachineFunction &MF);
  void eliminateRegSequences(MachineFunction &MF);
  void addPhysRegLiveIns(MachineFunction &MF);
  void finalizeProperties(MachineFunction &MF);

public:
  static char ID;

  AMDGPUSSARegisterAllocator() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU SSA Register Allocator";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<SlotIndexesWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUSSAREGISTERALLOCATOR_H
