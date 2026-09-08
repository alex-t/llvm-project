//===-- AMDGPUSSARegisterAllocator.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUSSARegisterAllocator.h"
#include "AMDGPU.h"
#include "AMDGPURegAllocInsertion.h"
#include "GCNRegPressure.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/MathExtras.h"
#include "SSARATrace.h"
#include <limits>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-ssa-register-allocator"

static cl::opt<bool> EnableExperimentBail(
    "amdgpu-ssa-experiment-bail", cl::Hidden, cl::init(false),
    cl::desc("Forensic mode: when width-tiered coloring cannot place a value, "
             "bail early (leave it uncolored, skip spill+SSA-destruction) so "
             "[TIERPROOF]/-stats data is collected for every function instead "
             "of aborting on the first hard one. Produces INVALID output (a "
             "later NoVRegs pass will abort) — for measurement only. Off = the "
             "real width-1 spill path runs and the function completes."));

static cl::opt<bool> EnableVerifyValueFlow(
    "amdgpu-ssa-verify-value-flow", cl::Hidden, cl::init(false),
    cl::desc("Certify every physreg use holds the SSA value its vreg named "
             "(catches clobber-while-live; single-block functions in v1)"));

static cl::opt<bool> VerifyValueFlowFatal(
    "amdgpu-ssa-verify-value-flow-fatal", cl::Hidden, cl::init(false),
    cl::desc("Abort on a value-flow violation (default: warn to stderr)"));

// Shadow register-tree oracle (SSARegisterTree). When on AND the forensic sink
// is active, a shadow SSARegisterTree mirrors the VGPR_32 occupancy the
// allocator maintains and, at each real VGPR_32 pick, LOGS what the tree would
// have picked vs. the allocator's choice (behavior-neutral: the tree's answer is
// discarded; it never influences allocation). Default off; requires a forensic
// sink (-amdgpu-ssa-forensic-json/-trace) for the divergences to land anywhere.
static cl::opt<bool> EnableSSAShadowTree(
    "amdgpu-ssa-shadow-tree", cl::Hidden, cl::init(false),
    cl::desc("Run a SHADOW SSARegisterTree for the VGPR_32 file that mirrors the "
             "allocator's occupancy and logs its pick vs. the real pick to the "
             "forensic NDJSON (observer only; requires a forensic sink; default "
             "off; byte-identical output on vs off)"));

static cl::opt<bool> EnableLaneWasteDump(
    "amdgpu-ssa-lane-waste-dump", cl::Hidden, cl::init(false),
    cl::desc("Report per-function capacity held by dead lanes of partially "
             "live tuples: whole-tuple occupancy vs subrange occupancy."));

static cl::opt<bool> EnableBlockDemandDump(
    "amdgpu-ssa-block-demand-dump", cl::Hidden, cl::init(false),
    cl::desc("TEMPORARY measurement: per slot, the live-value histogram by "
             "class width and allocation stride, beside the lane-accurate "
             "pressure and the whole-block dword sum. Sizes the width-aware "
             "region gate; delete together with it."));

static cl::opt<bool> RecoveryOnly(
    "amdgpu-ssa-recovery-only", cl::Hidden, cl::init(false),
    cl::desc("Run recovery first, then one cheap spill-planning pass for "
             "remaining coloring failures"));

// Step-0 PHI-copy metric (see PHI_Coalescer design section 9). Counted at the
// copy-vs-fixed-point decision in lowerPHIs(); pure instrumentation, no MIR
// change. Baseline for the coalescer and regression guard for every later step.
#define PHI_METRIC_DEBUG_TYPE "amdgpu-phi-metric"
STATISTIC(NumPhiOperands, "PHI operands examined at SSA destruction");
STATISTIC(NumPhiCopies, "PHI operands lowered to a copy (not a fixed point)");
STATISTIC(NumPhiFixedPoints, "PHI operands already fixed points (Src==Dst)");
STATISTIC(NumPhiUndefEdges, "PHI operands with an undef source (no copy needed)");
STATISTIC(NumPhiCopyWeight, "Sum of 2^loopdepth over PHI-copy operands");
// Feasibility-ceiling split of the remaining copies (whole-register sources
// only): a copy can EVER become a fixed point only if the operand does not
// interfere with the PHI result. Infeasible copies are the ceiling residue no
// coalescer can remove; feasible copies are what a fixed-point coalescer
// (Option A) could still convert beyond greedy affinity (Option B).
STATISTIC(NumPhiCopyFeasible,
          "PHI-copy operands with no read-lane/result interference (coalescable)");
STATISTIC(NumPhiCopyInfeasible,
          "PHI-copy operands whose read lane interferes with the result (ceiling)");
STATISTIC(NumPhiCopySubreg,
          "PHI-copy operands with a sub-register source (context tally; overlaps "
          "the feasible/infeasible split, now lane-classified)");

STATISTIC(NumTierSpills,
          "Values that coloring could not place and that entered recovery");

STATISTIC(NumIdentityCopiesErased,
          "Copies whose source and destination were colored the same physreg");

char AMDGPUSSARegisterAllocator::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPUSSARegisterAllocator, DEBUG_TYPE,
                      "AMDGPU SSA Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUSSARegisterAllocator, DEBUG_TYPE,
                    "AMDGPU SSA Register Allocator", false, false)

// === Coloring ===

void AMDGPUSSARegisterAllocator::classifyVRegs() {
  SSARA_TRACE();
  ColoringOrder.clear();
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I < E; ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (MRI->reg_nodbg_empty(VReg))
      continue;
    ColoringOrder.insert(TRI->getRegSizeInBits(*MRI->getRegClass(VReg)));
  }

  LLVM_DEBUG({
    dbgs() << "Coloring order (width descending):";
    for (unsigned W : ColoringOrder)
      dbgs() << " " << W;
    dbgs() << "\n";
  });
}

void AMDGPUSSARegisterAllocator::widenToAVOnUnified() {
  SSARA_TRACE();
  if (!ST->hasGFX90AInsts()) // unified vector file only
    return;
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I < E; ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (MRI->reg_nodbg_empty(VReg))
      continue;
    const TargetRegisterClass *RC = MRI->getRegClass(VReg);
    // Only widen plain VGPR classes (av_ already unified; AGPR/SGPR untouched).
    if (!TRI->isVGPRClass(RC) || TRI->isAGPRClass(RC))
      continue;
    unsigned Bits = TRI->getRegSizeInBits(*RC);
    const TargetRegisterClass *AV = TRI->getVectorSuperClassForBitWidth(Bits);
    if (!AV || AV == RC)
      continue;
    // LEGALITY: AV must be a subclass of every operand's required class, i.e.
    // every instruction touching VReg must already accept an AGPR in that slot.
    // A sub-register operand blocks it (the whole-reg constraint test below does
    // not apply to a sub-register slice — conservative, revisit with
    // getMatchingSuperRegClass if wide subreg cases justify it).
    bool Legal = true;
    for (MachineOperand &MO : MRI->reg_nodbg_operands(VReg)) {
      if (MO.getSubReg()) {
        Legal = false;
        break;
      }
      MachineInstr *MI = MO.getParent();
      const TargetRegisterClass *OpRC =
          TII->getRegClass(MI->getDesc(), MO.getOperandNo(), TRI);
      if (!OpRC)
        continue; // COPY/PHI/REG_SEQUENCE: no encoding constraint
      if (TRI->getCommonSubClass(AV, OpRC) != AV) {
        Legal = false;
        break;
      }
    }
    if (Legal) {
      MRI->setRegClass(VReg, AV);
      LLVM_DEBUG(dbgs() << "  [AV-WIDEN] " << printReg(VReg, TRI) << " -> "
                        << TRI->getRegClassName(AV) << "\n");
    }
  }
}

void AMDGPUSSARegisterAllocator::markOccupied(MCRegister PhysReg) {
  SSARA_TRACE();
  for (MCRegUnit Unit : TRI->regunits(PhysReg))
    OccupiedRegUnits.set(Unit);
  shadowAllocate(PhysReg); // behavior-neutral mirror (no-op unless shadowActive)
}

void AMDGPUSSARegisterAllocator::markFree(MCRegister PhysReg) {
  SSARA_TRACE();
  for (MCRegUnit Unit : TRI->regunits(PhysReg))
    OccupiedRegUnits.reset(Unit);
  shadowFree(PhysReg); // behavior-neutral mirror (no-op unless shadowActive)
}

//===----------------------------------------------------------------------===//
// Shadow register-tree oracle (SSARegisterTree). VGPR_32 file ONLY.
//
// PURE OBSERVER. Every method here is a no-op unless shadowActive() (the flag is
// on AND the forensic reporter is live). The tree mirrors the VGPR_32 occupancy
// the allocator keeps in OccupiedRegUnits and, at the pick, logs what it would
// have chosen — its answer is DISCARDED. Nothing here reads or writes ColorMap /
// OccupiedRegUnits in a way the allocation can observe.
//===----------------------------------------------------------------------===//

bool AMDGPUSSARegisterAllocator::shadowActive() const {
  SSARA_TRACE();
  return EnableSSAShadowTree && Reporter && Reporter->active() && ShadowTree;
}

// Build the physreg<->leaf bijection and size the shadow tree. Leaf index i is
// the i-th register of getOrder(VGPR_32) (the same order pickFreePhysReg scans
// first-fit), so tree.pickFreeAligned(1) — the lowest free leaf — is directly
// comparable to the allocator's first-fit pick. The tree needs a power-of-two
// leaf count; we round the real allocatable VGPR_32 count UP to a power of two
// and pre-allocate the padding leaves [RealVGPR32Count, ShadowLeaves) so
// pickFreeAligned can never hand back a register that does not exist.
void AMDGPUSSARegisterAllocator::setupShadowTree() {
  SSARA_TRACE();
  ShadowTree.reset();
  VGPR32Leaf.clear();
  VGPR32UnitLeaf.clear();
  RealVGPR32Count = 0;
  ShadowLeaves = 0;
  // Only stand the tree up when it will actually be used; keeps the off path and
  // non-forensic runs at zero cost.
  if (!EnableSSAShadowTree || !Reporter || !Reporter->active())
    return;

  ArrayRef<MCPhysReg> Order = RegClassInfo.getOrder(&AMDGPU::VGPR_32RegClass);
  RealVGPR32Count = Order.size();
  if (RealVGPR32Count == 0)
    return;
  for (unsigned I = 0; I < RealVGPR32Count; ++I) {
    MCRegister PR(Order[I]);
    VGPR32Leaf[PR.id()] = (int)I;
    // Map every reg unit of this VGPR_32 to its leaf. A VGPR_32 owns 1 unit on
    // targets without 16-bit sub-regs and 2 (lo16/hi16) with them; all map to
    // the same leaf, so an occupied bit on any of them frees/occupies the leaf.
    for (MCRegUnit U : TRI->regunits(PR))
      VGPR32UnitLeaf[U] = (int)I;
  }

  // Round up to a power of two (tree requirement). PowerOf2Ceil(1)==1.
  ShadowLeaves = llvm::PowerOf2Ceil(RealVGPR32Count);
  ShadowTree = std::make_unique<SSARegisterTree>(ShadowLeaves);
  // Pre-mark the padding leaves occupied so they are never picked or double-freed.
  for (unsigned L = RealVGPR32Count; L < ShadowLeaves; ++L)
    ShadowTree->allocateAligned(L, 1);
}

// Map a physreg to a VGPR_32 leaf. A VGPR_32 maps directly. A wider VGPR tuple
// (vreg_64/96/128/...) maps to its LOWEST-index sub-VGPR_32's leaf, which — with
// leaf==getOrder-ordinal — is the aligned block start we mirror. A non-VGPR
// physreg (SGPR/AGPR/VCC/...) returns -1: out of scope for this increment.
int AMDGPUSSARegisterAllocator::shadowLeafOf(MCRegister PhysReg) const {
  SSARA_TRACE();
  auto Direct = VGPR32Leaf.find(PhysReg.id());
  if (Direct != VGPR32Leaf.end())
    return Direct->second;
  // Wider VGPR tuple: find the lowest leaf over its reg units (mapped via
  // VGPR32UnitLeaf, which handles lo16/hi16 unit roots correctly). A non-VGPR
  // physreg contributes no mapped unit and returns -1.
  int Best = -1;
  for (MCRegUnit Unit : TRI->regunits(PhysReg)) {
    auto It = VGPR32UnitLeaf.find(Unit);
    if (It != VGPR32UnitLeaf.end() && (Best < 0 || It->second < Best))
      Best = It->second;
  }
  return Best;
}

// Enumerate the ACTUAL leaf index of every VGPR_32 that \p PhysReg covers, into
// \p Leaves. A scalar VGPR_32 yields its own leaf; a wider tuple yields the leaf
// of each of its VGPR_32 sub-registers. We resolve each leaf through the
// getOrder-ordinal map (VGPR32Leaf) rather than assuming a tuple's sub-VGPRs are
// contiguous in leaf space — on targets that reserve VGPRs the allocation order
// is NOT the HW-index order, so a contiguous [Leaf, Leaf+W) block would mark the
// wrong leaves (a drift that would corrupt even the width-1 comparison). Each
// leaf is tracked as an independent width-1 cell, which is all pickFreeAligned(1)
// needs; aligned-block modeling of wide tuples is a later increment.
void AMDGPUSSARegisterAllocator::shadowLeavesOf(
    MCRegister PhysReg, SmallVectorImpl<unsigned> &Leaves) const {
  SSARA_TRACE();
  auto Direct = VGPR32Leaf.find(PhysReg.id());
  if (Direct != VGPR32Leaf.end()) {
    Leaves.push_back((unsigned)Direct->second);
    return;
  }
  // Wider tuple (or any physreg): collect the leaf of each covered reg unit via
  // VGPR32UnitLeaf (dedup — a VGPR_32 owns 2 units on lo16/hi16 targets, both
  // pointing at the same leaf). A non-VGPR physreg maps no units and yields [].
  for (MCRegUnit Unit : TRI->regunits(PhysReg)) {
    auto It = VGPR32UnitLeaf.find(Unit);
    if (It != VGPR32UnitLeaf.end() &&
        !llvm::is_contained(Leaves, (unsigned)It->second))
      Leaves.push_back((unsigned)It->second);
  }
}

void AMDGPUSSARegisterAllocator::shadowAllocate(MCRegister PhysReg) {
  SSARA_TRACE();
  if (!shadowActive())
    return;
  // Mirror as width-1 per VGPR_32 leaf so wider/unaligned tuples (out of the
  // width-1 pick scope) never wedge the aligned tree; each leaf tracks one dword.
  SmallVector<unsigned, 8> Leaves;
  shadowLeavesOf(PhysReg, Leaves);
  for (unsigned L : Leaves)
    if (L < RealVGPR32Count && ShadowTree->isFree(L, 1))
      ShadowTree->allocateAligned(L, 1);
}

void AMDGPUSSARegisterAllocator::shadowFree(MCRegister PhysReg) {
  SSARA_TRACE();
  if (!shadowActive())
    return;
  SmallVector<unsigned, 8> Leaves;
  shadowLeavesOf(PhysReg, Leaves);
  for (unsigned L : Leaves)
    if (L < RealVGPR32Count && !ShadowTree->isFree(L, 1))
      ShadowTree->freeAligned(L, 1);
}

void AMDGPUSSARegisterAllocator::shadowFreeUnit(MCRegUnit Unit) {
  SSARA_TRACE();
  if (!shadowActive())
    return;
  // Recover the VGPR_32 leaf that owns Unit directly from the unit map (the root
  // iterator would yield VGPRn_LO16/HI16, which are NOT in the physreg-keyed
  // map). A VGPR_32 owning 2 units means each is cleared separately; freeing an
  // already-free leaf is a guarded no-op.
  auto It = VGPR32UnitLeaf.find(Unit);
  if (It != VGPR32UnitLeaf.end()) {
    unsigned L = (unsigned)It->second;
    if (L < RealVGPR32Count && !ShadowTree->isFree(L, 1))
      ShadowTree->freeAligned(L, 1);
  }
}

// Resync the shadow tree to the AUTHORITATIVE OccupiedRegUnits. Called wherever
// the allocator resets/rebuilds OccupiedRegUnits wholesale (per-block seed,
// colorOneInPlace, deferred per-unit frees) so the mirror can never drift from
// the real occupancy even across the paths that touch the bitvector directly.
void AMDGPUSSARegisterAllocator::shadowResetToOccupied() {
  SSARA_TRACE();
  if (!shadowActive())
    return;
  // Free every real leaf, then re-occupy from the live OccupiedRegUnits. One
  // width-1 leaf per VGPR_32 whose reg unit is currently set.
  for (unsigned L = 0; L < RealVGPR32Count; ++L)
    if (!ShadowTree->isFree(L, 1))
      ShadowTree->freeAligned(L, 1);
  ArrayRef<MCPhysReg> Order = RegClassInfo.getOrder(&AMDGPU::VGPR_32RegClass);
  for (unsigned L = 0; L < RealVGPR32Count; ++L) {
    MCRegister PR(Order[L]);
    bool Occ = false;
    for (MCRegUnit U : TRI->regunits(PR))
      if (OccupiedRegUnits.test(U)) {
        Occ = true;
        break;
      }
    if (Occ)
      ShadowTree->allocateAligned(L, 1);
  }
}

void AMDGPUSSARegisterAllocator::collectOccupancy(const TargetRegisterClass *RC,
                                                  SlotIndex SI,
                                                  const LiveInterval *VI,
                                                  OccupancyFacts &Out) const {
  SSARA_TRACE();
  // Pure fact extraction: the counting loop lifted verbatim out of
  // dumpOccupancyMap. Reads OccupiedRegUnits / ColorMap / CallSites / LIS only;
  // mutates nothing. Fills \p Out with the same map string, tallies, and
  // phantom/usable register-name lists dumpOccupancyMap used to compute inline.
  //
  // Two occupancy views, to expose disagreements:
  //  Occ    = the LIVE OccupiedRegUnits bitvector pickFreePhysReg actually
  //           consults (running seed + mark/kill state at this program point).
  //  OccCM  = freshly rebuilt from ColorMap vregs live at SI.
  // If a reg is set in Occ but not OccCM, it is occupied by something NOT a
  // ColorMap-vreg-live-at-SI: a physreg live-in, a dead def still marked, or a
  // stale running-state bit — the exact thing to diagnose.
  const BitVector &Occ = OccupiedRegUnits;
  BitVector OccCM(TRI->getNumRegUnits());
  for (const auto &[VReg, PhysReg] : ColorMap)
    if (LIS->hasInterval(VReg) && LIS->getInterval(VReg).liveAt(SI))
      for (MCRegUnit U : TRI->regunits(PhysReg))
        OccCM.set(U);

  // A reg is clobbered for VI if some call VI is live across clobbers it.
  auto Clobbered = [&](MCRegister PR) -> bool {
    if (!VI)
      return false;
    for (const auto &[CS, CMI] : CallSites) {
      if (!VI->liveAt(CS))
        continue;
      if (CMI->modifiesRegister(PR, TRI))
        return true;
      for (const MachineOperand &MO : CMI->operands())
        if (MO.isRegMask() && MO.clobbersPhysReg(PR))
          return true;
    }
    return false;
  };

  Out = OccupancyFacts();
  Out.ClassName = TRI->getRegClassName(RC);
  std::string &Map = Out.Map;
  MCRegister First, Last;
  unsigned Count = 0;
  for (MCRegister PR : RegClassInfo.getOrder(RC)) {
    bool O = false, OCM = false;
    for (MCRegUnit U : TRI->regunits(PR)) {
      if (Occ.test(U)) O = true;
      if (OccCM.test(U)) OCM = true;
    }
    if (O) {
      Map.push_back('#');
      ++Out.Occupied;
      if (!OCM)
        Out.Phantom.push_back(TRI->getName(PR)); // running-state, no live vreg
    } else if (Clobbered(PR)) {
      Map.push_back('x');
      ++Out.FreeClobbered;
    } else {
      Map.push_back('.');
      ++Out.FreeUsable;
      Out.Usable.push_back(TRI->getName(PR));
    }
    if (Count == 0)
      First = PR;
    Last = PR;
    ++Count;
  }
  Out.Total = Count;
  if (Count) {
    Out.FirstReg = TRI->getName(First);
    Out.LastReg = TRI->getName(Last);
  }
}

void AMDGPUSSARegisterAllocator::dumpOccupancyMap(const TargetRegisterClass *RC,
                                                  SlotIndex SI, const char *Tag,
                                                  const LiveInterval *VI) const {
  SSARA_TRACE();
  OccupancyFacts F;
  collectOccupancy(RC, SI, VI, F);

  dbgs() << "  [OCCMAP " << Tag << "] " << F.ClassName << " @" << SI
         << "  usable=" << F.FreeUsable << " clobbered=" << F.FreeClobbered
         << " occupied=" << F.Occupied << " total=" << F.Total << "\n"
         << "    " << F.Map << "\n";
  if (F.Total)
    dbgs() << "    (" << F.FirstReg << " .. " << F.LastReg
           << ")  legend: # occ, x clobbered, . usable\n";
  // The key question: registers occupied by running-state but with NO live
  // ColorMap vreg (physreg live-ins, dead defs, or stale bits).
  if (!F.Phantom.empty()) {
    dbgs() << "    phantom-occupied (Occ set, no live ColorMap vreg):";
    for (const std::string &N : F.Phantom)
      dbgs() << " " << N;
    dbgs() << "\n";
  }
  if (!F.Usable.empty()) {
    dbgs() << "    usable regs:";
    for (const std::string &N : F.Usable)
      dbgs() << " " << N;
    dbgs() << "\n";
    // For each usable reg, find WIDER ColorMap values whose whole interval
    // OVERLAPS VI (pickFreePhysReg's OccupiedAtDef augmentation, lines ~195).
    // This is the occupancy the liveAt(SI) map view misses. The usable regs are
    // exactly the '.' entries of the map in getOrder(RC) order, so we zip the
    // allocation order with F.Map rather than re-collecting MCRegisters.
    if (VI) {
      unsigned VIWidth = TRI->getRegSizeInBits(*RC);
      unsigned Idx = 0;
      for (MCRegister PR : RegClassInfo.getOrder(RC)) {
        bool Usable = Idx < F.Map.size() && F.Map[Idx] == '.';
        ++Idx;
        if (!Usable)
          continue;
        for (const auto &[WReg, WPhys] : ColorMap) {
          if (TRI->getRegSizeInBits(*MRI->getRegClass(WReg)) <= VIWidth)
            continue;
          bool hitsPR = false;
          for (MCRegUnit U : TRI->regunits(WPhys))
            for (MCRegUnit PU : TRI->regunits(PR))
              if (U == PU) { hitsPR = true; break; }
          if (hitsPR && LIS->getInterval(WReg).overlaps(*VI)) {
            dbgs() << "      " << TRI->getName(PR) << " blocked by wider "
                   << printReg(WReg, TRI) << "->" << TRI->getName(WPhys)
                   << " (interval overlaps VI but not live@SI)\n";
            dbgs() << "        VI  " << printReg(VI->reg(), TRI) << ": " << *VI
                   << "\n        blk " << printReg(WReg, TRI) << ": "
                   << LIS->getInterval(WReg) << "\n";
          }
        }
      }
    }
  }
}

void AMDGPUSSARegisterAllocator::collectSpillAcrossCandidates(
    Register Failed, SlotIndex FS, SlotIndex FE, bool FIsVGPR,
    unsigned &NLiveThru, SmallVectorImpl<SpillAcrossCandidate> &Out,
    SmallVectorImpl<unsigned> &LiveThruIdx) const {
  SSARA_TRACE();
  // Pure fact extraction: the "which colored values could be spilled across the
  // failed value's region" scan lifted verbatim out of the COLORFAIL debug
  // block. ANSWER "is there a valid reg to spill across R?": count colored
  // values in R's FILE that are LIVE-THROUGH [FS,FE) with NO use strictly inside
  //  — each such value's register can be freed across the whole region by
  // spilling it (reload past FE). Reads ColorMap / LIS / MRI only.
  NLiveThru = 0;
  for (const auto &[V, P] : ColorMap) {
    if (V == Failed || !LIS->hasInterval(V))
      continue;
    const TargetRegisterClass *VRC = MRI->getRegClass(V);
    bool VIsVGPR = TRI->isVGPRClass(VRC) || TRI->isAGPRClass(VRC);
    if (VIsVGPR != FIsVGPR)
      continue; // wrong file
    const LiveInterval &OVI = LIS->getInterval(V);
    if (!OVI.liveAt(FS) || !OVI.liveAt(FE.getPrevSlot()))
      continue; // not live-through R
    ++NLiveThru;
    bool UsedInside = false;
    for (const MachineOperand &MO : MRI->use_operands(V)) {
      SlotIndex U = LIS->getInstructionIndex(*MO.getParent()).getRegSlot();
      if (FS < U && U < FE) {
        UsedInside = true;
        break;
      }
    }
    if (!UsedInside)
      Out.push_back(
          {V, P, (unsigned)(TRI->getRegSizeInBits(*VRC) / 32), &OVI});
  }
  // The FULL live-across set (reg indices sorted) so a round-to-round diff shows
  // exactly which vregs newly appear.
  for (const auto &[V, P] : ColorMap) {
    if (V == Failed || !LIS->hasInterval(V))
      continue;
    const TargetRegisterClass *VRC = MRI->getRegClass(V);
    bool VIsVGPR = TRI->isVGPRClass(VRC) || TRI->isAGPRClass(VRC);
    if (VIsVGPR != FIsVGPR)
      continue;
    const LiveInterval &OVI = LIS->getInterval(V);
    if (OVI.liveAt(FS) && OVI.liveAt(FE.getPrevSlot()))
      LiveThruIdx.push_back(V.virtRegIndex());
  }
  llvm::sort(LiveThruIdx);
}

void AMDGPUSSARegisterAllocator::collectLiveSet(
    SlotIndex SI, SmallVectorImpl<LiveSetEntry> &Out) const {
  SSARA_TRACE();
  // Facts-only const walk: every virtual register whose interval is live at SI,
  // joined to its physreg via ColorMap (uncolored => phys=-1). This is the same
  // liveAt(SI) test collectOccupancy already uses over ColorMap, generalized to
  // ALL vregs so the cross-section is complete (not just the colored ones). No
  // new LIS/pressure pass; nothing mutated.
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I < E; ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (MRI->reg_nodbg_empty(VReg) || !LIS->hasInterval(VReg))
      continue;
    const LiveInterval &LI = LIS->getInterval(VReg);
    if (LI.empty() || !LI.liveAt(SI))
      continue;
    const TargetRegisterClass *RC = MRI->getRegClass(VReg);
    LiveSetEntry Ent;
    Ent.VReg = VReg.virtRegIndex();
    {
      std::string S;
      raw_string_ostream OS(S);
      OS << LI.beginIndex();
      Ent.LR = OS.str();
    }
    auto It = ColorMap.find(VReg);
    if (It != ColorMap.end()) {
      Ent.Phys = (int64_t)It->second.id();
      Ent.PhysName = TRI->getName(It->second);
    } else {
      Ent.Phys = -1;
    }
    Ent.WidthBits = TRI->getRegSizeInBits(*RC);
    Ent.LaneMask = MRI->getMaxLaneMaskForVReg(VReg).getAsInteger();
    Out.push_back(std::move(Ent));
  }
}

void AMDGPUSSARegisterAllocator::scanOverlappersForVI(
    const LiveInterval &VI, BitVector &OccupiedUnits,
    SmallVectorImpl<std::pair<Register, MCRegister>> *Overlappers) const {
  SSARA_TRACE();
  // ONE walk over ColorMap: record which colored values' intervals overlap VI
  // (the full-live-interval test, catching same-width gaps a point check misses)
  // and OR their physreg units into OccupiedUnits. \p Overlappers is optional —
  // the gap pick needs only OccupiedUnits (pass nullptr to skip building the
  // list); the splitter also needs the occupant vregs, so passes a vector.
  //
  // NOTE the two callers run in DIFFERENT phases (gap scan during color(); the
  // splitter post-color() over UncolorableVRegs), with ColorMap mutated between,
  // so the result CANNOT be cached across them — each caller scans fresh.
  if (Overlappers)
    Overlappers->clear();
  OccupiedUnits.reset();
  OccupiedUnits.resize(TRI->getNumRegUnits());
  for (const auto &[WReg, WPhysReg] : ColorMap) {
    if (!LIS->hasInterval(WReg) || LIS->getInterval(WReg).empty() ||
        !LIS->getInterval(WReg).overlaps(VI))
      continue;
    if (Overlappers)
      Overlappers->emplace_back(WReg, WPhysReg);
    for (MCRegUnit WU : TRI->regunits(WPhysReg))
      OccupiedUnits.set(WU);
  }
}

MCRegister AMDGPUSSARegisterAllocator::pickFreePhysReg(
    const TargetRegisterClass *RC, const LiveInterval &VI,
    ArrayRef<std::pair<MCRegister, const LiveInterval *>> WiderDefs,
    ArrayRef<MCRegister> Hints, uint64_t AttemptID) {
  SSARA_TRACE();
  // Cache the forensic gate once (loop-invariant) — the per-candidate loops
  // below check it on every iteration.
  const bool Report = Reporter && Reporter->active();
  LLVM_DEBUG({
    dbgs() << "    Allocation order for " << TRI->getRegClassName(RC) << ":";
    for (MCRegister PR : RegClassInfo.getOrder(RC))
      dbgs() << " " << TRI->getName(PR);
    dbgs() << "\n";
  });

  // Interference against the colored values, one register unit at a time. A
  // colored value claims a unit only over the sub-range covering that unit's
  // lanes, so a tuple whose high lanes are dead no longer blocks the registers
  // those lanes map to. This replaces two approximations that were applied
  // together: a point-in-time scanline that marked every unit of an assigned
  // tuple regardless of live lanes, and a LiveInterval::overlaps() augmentation
  // that compared main ranges only and was restricted to wider values.
  //
  // VI is compared by its main range, which over-claims for VI itself. That is
  // conservative and self-correcting: VI's own dead lanes are accounted exactly
  // when some later value asks whether it may use the units VI holds.
  auto claimsUnit = [&](const LiveInterval &WLI, LaneBitmask UnitMask) {
    if (!WLI.hasSubRanges())
      return WLI.overlaps(VI);
    // Every sub-range touching the unit, not just the first: a unit covered by
    // two sub-ranges must be claimed if either is live across VI. An empty
    // sub-range is skipped because overlaps() asserts on an empty receiver.
    for (const LiveInterval::SubRange &S : WLI.subranges())
      if (!S.empty() && (S.LaneMask & UnitMask).any() && S.overlaps(VI))
        return true;
    return false;
  };
  BitVector OccupiedAtDef(TRI->getNumRegUnits());
  for (const auto &[WReg, WPhysReg] : ColorMap) {
    if (!LIS->hasInterval(WReg))
      continue;
    const LiveInterval &WLI = LIS->getInterval(WReg);
    if (WLI.empty() || !WLI.overlaps(VI))
      continue; // cheap main-range reject before the per-unit work
    for (MCRegUnitMaskIterator UI(WPhysReg, TRI); UI.isValid(); ++UI) {
      auto [Unit, UnitMask] = *UI;
      if (claimsUnit(WLI, UnitMask))
        OccupiedAtDef.set(Unit);
    }
  }

  // Shared legality test: a candidate PR is usable iff none of its reg units are
  // taken by a colored value or by physical-register liveness over VI's range,
  // AND no clobber site VI is live at writes it (a call's regmask or explicit
  // def, an inline-asm clobber, an implicit-def $vcc) - the value would be
  // undefined past that site.
  auto IsFree = [&](MCRegister PR) -> bool {
    for (MCRegUnit Unit : TRI->regunits(PR)) {
      if (OccupiedAtDef.test(Unit))
        return false;
      // Physical registers and block live-ins come from the reg-unit ranges
      // LiveIntervals already maintains, which are range-accurate, rather than
      // from a per-block live-in seed that held the whole block.
      // A unit range can exist but be empty; overlaps() asserts on an empty
      // receiver.
      if (const LiveRange *RU = LIS->getCachedRegUnit(Unit); RU && !RU->empty())
        if (RU->overlaps(VI))
          return false;
    }
    return survivesClobberSites(VI, PR);
  };

  // PRESSURE-TARGETED AGPR PREFERENCE (unified targets). An av_ value can live in
  // either arch-VGPR or AGPR. When the PEAK arch-VGPR pressure OVER THIS VALUE'S
  // RANGE exceeds the pool, this av value should drain an AGPR so arch-VGPRs stay
  // free for VGPR-only values (Greedy-style). Peak-over-range (not def-point
  // occupancy) is required: width-descending colors wide av values FIRST when the
  // file is still empty, so def-point occupancy is 0 and misses the pressure the
  // value's own long range creates across a hot region (e.g. a block pinned live
  // across an atomic loop). Computed BEFORE the phi-affinity hint so a physreg-copy
  // hint to an arch-VGPR (e.g. %v = COPY $vgpr0..31) does NOT pin this value into
  // the VGPR file under pressure — we accept a cheap v<->a copy at the fixed-reg
  // boundary instead (exactly Greedy's v_accvgpr scratch). Only under pressure ->
  // low-pressure functions are untouched.
  bool PreferAGPR = false;
  if (ST->hasGFX90AInsts() && TRI->isVectorSuperClass(RC)) {
    unsigned VGPRPool = allocatablePool(
        const_cast<MachineFunction &>(MRI->getMF()), RegFile::VGPR);
    unsigned Peak = 0;
    MachineFunction &MF = const_cast<MachineFunction &>(MRI->getMF());
    for (MachineBasicBlock &MBB : MF) {
      if (MBB.empty())
        continue;
      GCNUpwardRPTracker Tracker(*LIS);
      Tracker.reset(MBB);
      for (MachineInstr &MI : llvm::reverse(MBB)) {
        if (MI.isDebugInstr())
          continue;
        Tracker.recede(MI);
        if (MI.isPHI())
          continue;
        SlotIndex SI = LIS->getInstructionIndex(MI).getRegSlot();
        if (SI < VI.beginIndex() || VI.endIndex() <= SI)
          continue;
        Peak = std::max(Peak, pressureOf(Tracker.getPressure(), RegFile::VGPR));
      }
    }
    PreferAGPR = Peak > VGPRPool;
  }
  // Under pressure, take a free AGPR now (before the VGPR-affinity hint).
  if (PreferAGPR) {
    for (MCRegister PR : availableOrder(RC))
      if (TRI->isAGPRClass(TRI->getPhysRegBaseClass(PR)) && IsFree(PR)) {
        LLVM_DEBUG(dbgs() << "    AGPR-preferred pick: " << TRI->getName(PR)
                          << "\n");
        return PR;
      }
  }

  // Option B: prefer a phi-partner's color if it is a legal member of RC and
  // free. Hints are pre-ordered hottest-first by collectPhiHints; take the first
  // that fits. RC->contains guards against a partner whose class differs from RC.
  // Skipped under PreferAGPR: a VGPR-affinity hint would re-pin this value into
  // the saturated VGPR file (the AGPR scan above already tried the good target).
  uint64_t HintOrdinal = 0;
  for (MCRegister Hint : Hints) {
    if (PreferAGPR)
      break;
    if (!Hint || !RC->contains(Hint))
      continue;
    if (Report)
      Reporter->candidateConsidered(AttemptID, Hint.id(), TRI->getName(Hint),
                                    HintOrdinal, "phi-affinity-hint");
    if (IsFree(Hint)) {
      LLVM_DEBUG(dbgs() << "    phi-affinity hint taken: " << TRI->getName(Hint)
                        << "\n");
      if (Report)
        Reporter->candidateAccepted(AttemptID, Hint.id(), TRI->getName(Hint),
                                    HintOrdinal, "phi-affinity-hint");
      return Hint;
    }
    if (Report)
      Reporter->candidateRejected(AttemptID, Hint.id(), TRI->getName(Hint),
                                  HintOrdinal, "not-free");
    ++HintOrdinal;
  }

  // Fact-only reject-reason classifier (Q-B): observes WHY IsFree returned
  // false without changing any control flow. Used solely for candidate facts.
  auto RejectReason = [&](MCRegister PR) -> const char * {
    for (MCRegUnit Unit : TRI->regunits(PR))
      if (OccupiedAtDef.test(Unit))
        return "occupied-unit";
    for (const auto &[CallIdx, CallMI] : CallSites) {
      if (!VI.liveAt(CallIdx))
        continue;
      if (CallMI->modifiesRegister(PR, TRI))
        return "call-modifies";
      for (const MachineOperand &MO : CallMI->operands())
        if (MO.isRegMask() && MO.clobbersPhysReg(PR))
          return "regmask";
    }
    return "unknown";
  };

  uint64_t Ordinal = 0;
  for (MCRegister PR : availableOrder(RC)) {
    if (Report)
      Reporter->candidateConsidered(AttemptID, PR.id(), TRI->getName(PR),
                                    Ordinal, "first-fit-order");
    if (IsFree(PR)) {
      if (Report)
        Reporter->candidateAccepted(AttemptID, PR.id(), TRI->getName(PR),
                                    Ordinal, "first-fit-order");
      return PR;
    }
    if (Report)
      Reporter->candidateRejected(AttemptID, PR.id(), TRI->getName(PR), Ordinal,
                                  RejectReason(PR));
    ++Ordinal;
  }
  return MCRegister();
}

bool AMDGPUSSARegisterAllocator::colorOneInPlace(Register R) {
  SSARA_TRACE();
  // Color R against the CURRENT ColorMap without disturbing any assignment.
  // R is a reload remainder: a short interval [reload, use]. Seed occupancy
  // from exactly the colored values whose live range OVERLAPS R's range — that
  // is "what is live during R's span", i.e. the point-pressure at R expressed
  // as interval overlap (which also handles the endpoints: a value dying at R's
  // start or born at R's end does not block R). Any register left free is free
  // across all of R. For a width-1 reload one always exists: point pressure at
  // the use ≤ RPLimit < file size (the spiller's margin guarantees it).
  const TargetRegisterClass *RC = MRI->getRegClass(R);
  const LiveInterval &RI = LIS->getInterval(R);

  // pickFreePhysReg reads OccupiedRegUnits (same-or-narrower blockers) and scans
  // ColorMap itself for WIDER overlapping values. So seed OccupiedRegUnits with
  // the same-or-narrower colored values overlapping RI; let pickFreePhysReg
  // handle wider ones. WiderDefs is empty — the ColorMap scan inside
  // pickFreePhysReg already covers cross-block wider defs.
  unsigned RWidth = TRI->getRegSizeInBits(*RC);
  OccupiedRegUnits.reset();
  for (const auto &[VReg, PhysReg] : ColorMap) {
    if (VReg == R || !LIS->hasInterval(VReg) || LIS->getInterval(VReg).empty())
      continue; // empty interval (e.g. spilled to nothing) -> overlaps() asserts
    if (TRI->getRegSizeInBits(*MRI->getRegClass(VReg)) > RWidth)
      continue; // wider: handled by pickFreePhysReg's own overlap scan
    if (LIS->getInterval(VReg).overlaps(RI))
      markOccupied(PhysReg);
  }

  // The reset() above cleared OccupiedRegUnits directly (not through markFree),
  // so the mirror must be re-anchored to the just-rebuilt occupancy. No-op unless
  // shadowActive.
  shadowResetToOccupied();

  MCRegister Chosen = pickFreePhysReg(RC, RI, /*WiderDefs=*/{});
  if (!Chosen)
    return false;

  ColorMap[R] = Chosen;
  unsigned Idx = TRI->getHWRegIndex(Chosen);
  unsigned W = RWidth / 32;
  const TargetRegisterClass *PhysRC = TRI->getPhysRegBaseClass(Chosen);
  if (TRI->isVGPRClass(PhysRC))
    MaxVGPRIdx = std::max(MaxVGPRIdx, Idx + W);
  else if (TRI->isAGPRClass(PhysRC))
    MaxAGPRIdx = std::max(MaxAGPRIdx, Idx + W);
  else if (TRI->isSGPRClass(PhysRC))
    MaxSGPRIdx = std::max(MaxSGPRIdx, Idx + W);

  LLVM_DEBUG(dbgs() << "  in-place color: " << printReg(R, TRI) << " -> "
                    << TRI->getName(Chosen) << "\n");
  return true;
}

// Option B affinity hint collection. See header comment.
SmallVector<MCRegister, 4>
AMDGPUSSARegisterAllocator::collectPhiHints(Register VReg,
                                            const TargetRegisterClass *RC) {
  SSARA_TRACE();
  // (physreg, weight) candidates; dedup + weight-sort before returning.
  SmallVector<std::pair<MCRegister, uint64_t>, 4> Cand;

  // Record a candidate color for VReg, composing SubIdx onto the physical
  // register PR. PRIsSub says which side SubIdx slices:
  //   - PRIsSub == false: VReg is the sub-register, reading PR.SubIdx (a lane φ
  //     reading %593.sub3 of a wide colored operand, or a COPY of a slice of a
  //     physreg). VReg's color is that SLICE of PR -> getSubReg().
  //   - PRIsSub == true: PR is the sub-register; VReg's color is the SUPER
  //     register whose SubIdx slice is PR -> getMatchingSuperReg().
  // Shared by every hint direction below, so the legality rules live in exactly
  // one place.
  auto AddCandidate = [&](MCRegister PR, unsigned SubIdx, bool PRIsSub,
                          uint64_t W) {
    if (!PR)
      return;
    if (SubIdx) {
      PR = PRIsSub ? TRI->getMatchingSuperReg(PR, SubIdx, RC)
                   : TRI->getSubReg(PR, SubIdx);
      if (!PR)
        return; // no such slice/super in the physreg or class
    }
    if (!RC->contains(PR))
      return; // class/width mismatch after composition
    // Containment in RC does not imply the allocator may hand the register out:
    // SReg_64 contains EXEC, so a value defined by `COPY $exec` composes to a
    // hint onto the exec mask itself. Every other pick scans availableOrder(),
    // which excludes reserved registers, so this is the one path that can
    // introduce one. Coloring a value to EXEC and then spilling it emits a spill
    // of the exec mask, which SGPR spill lowering rejects outright.
    if (MRI->isReserved(PR))
      return;
    Cand.push_back({PR, W});
  };

  // Turn a colored φ partner into a candidate color for VReg. Direction A
  // reads Partner.SubIdx (PartnerIsSub == false); Direction B has Partner as
  // the narrow φ result of a loop-carried tuple, colored before this wide latch
  // operand (PartnerIsSub == true). Weight is the edge's loop depth, so a hot
  // back-edge outranks a cold one.
  auto AddPartner = [&](Register Partner, unsigned SubIdx, bool PartnerIsSub,
                        MachineBasicBlock *EdgeBlock) {
    if (!Partner.isVirtual())
      return;
    auto It = ColorMap.find(Partner);
    if (It == ColorMap.end())
      return; // partner not colored yet -- nothing to align to
    unsigned Depth = EdgeBlock ? MLI->getLoopDepth(EdgeBlock) : 0;
    uint64_t W = Depth < 63 ? (uint64_t(1) << Depth) : ~uint64_t(0);
    AddCandidate(It->second, SubIdx, PartnerIsSub, W);
  };

  MachineInstr *Def = MRI->getUniqueVRegDef(VReg);

  // Direction A -- VReg is a phi result: align to its (colored) operands. If an
  // operand reads a slice (%wide.subN), VReg's color is that slice of the
  // operand's color (PartnerIsSub = false).
  if (Def && Def->isPHI()) {
    for (unsigned I = 1, E = Def->getNumOperands(); I < E; I += 2) {
      MachineOperand &Src = Def->getOperand(I);
      // A web-spilled PHI operand is a FRAME INDEX (pred-tail relief), not a reg;
      // check isReg() FIRST (isUndef asserts on a non-reg operand).
      if (!Src.isReg() || Src.isUndef())
        continue;
      AddPartner(Src.getReg(), Src.getSubReg(), /*PartnerIsSub=*/false,
                 Def->getOperand(I + 1).getMBB());
    }
  }

  // Direction B -- VReg feeds one or more phi results: align to the (colored)
  // result. The incoming edge for weighting is VReg's own def block. When the φ
  // reads VReg via a sub-register (result is narrower than VReg -- the
  // loop-carried tuple case, where the header result is colored before this
  // wide latch operand), VReg's color is the super-register whose SubN slice is
  // the result's color (PartnerIsSub = true).
  MachineBasicBlock *DefBlock = Def ? Def->getParent() : nullptr;
  for (MachineInstr &UseMI : MRI->use_nodbg_instructions(VReg)) {
    if (!UseMI.isPHI())
      continue;
    for (unsigned I = 1, E = UseMI.getNumOperands(); I < E; I += 2) {
      MachineOperand &Src = UseMI.getOperand(I);
      if (Src.isReg() && Src.getReg() == VReg) {
        AddPartner(UseMI.getOperand(0).getReg(), Src.getSubReg(),
                   /*PartnerIsSub=*/true, DefBlock);
        break;
      }
    }
  }

  // Direction C -- physreg-copy affinity (the ABI live-in / live-out coalescing
  // hint the stock RegisterCoalescer applies and SSARA had dropped). If VReg is
  // defined by `COPY $phys` (an incoming argument / live-in) hint VReg->$phys; if
  // VReg is used by `$phys = COPY VReg` (an outgoing arg / return value) hint the
  // same. Keeping the value in its ABI register elides the copy. Sub-register
  // copies compose like the φ cases. The ABI edge is unconditional, so it
  // outweighs any loop-depth φ hint. A hint is only a preference: pickFreePhysReg
  // still gates it through IsFree for interference and call-clobber survival, and
  // AddCandidate drops registers the allocator may never hand out.
  constexpr uint64_t PhysCopyWeight = uint64_t(1) << 20;
  if (Def && Def->isCopy()) {
    const MachineOperand &Src = Def->getOperand(1);
    if (Src.isReg() && Src.getReg().isPhysical())
      // VReg = COPY $phys.SubIdx  ->  VReg's color is that slice of $phys.
      AddCandidate(Src.getReg(), Src.getSubReg(), /*PRIsSub=*/false,
                   PhysCopyWeight);
  }
  for (MachineInstr &UseMI : MRI->use_nodbg_instructions(VReg)) {
    if (!UseMI.isCopy())
      continue;
    const MachineOperand &Dst = UseMI.getOperand(0);
    const MachineOperand &Src = UseMI.getOperand(1);
    if (Dst.getReg().isPhysical() && Src.isReg() && Src.getReg() == VReg)
      // $phys = COPY VReg.SubIdx  ->  VReg's color's SubIdx slice is $phys, so
      // VReg's color is the super-register (PRIsSub = true).
      AddCandidate(Dst.getReg(), Src.getSubReg(), /*PRIsSub=*/true,
                   PhysCopyWeight);
  }

  // Hottest-first, deduped (keep max weight per physreg).
  llvm::stable_sort(Cand, [](auto &A, auto &B) { return A.second > B.second; });
  SmallVector<MCRegister, 4> Hints;
  for (auto &[PR, W] : Cand)
    if (!llvm::is_contained(Hints, PR))
      Hints.push_back(PR);
  return Hints;
}

void AMDGPUSSARegisterAllocator::seedOccupiedAtBBEntry(MachineBasicBlock *MBB) {
  SSARA_TRACE();
  OccupiedRegUnits.reset();
  SlotIndex BBStart = LIS->getMBBStartIdx(MBB);

  LLVM_DEBUG(dbgs() << "  Seed " << printMBBReference(*MBB) << ":\n");

  for (const auto &[VReg, PhysReg] : ColorMap) {
    if (LIS->getInterval(VReg).liveAt(BBStart)) {
      markOccupied(PhysReg);
      LLVM_DEBUG(dbgs() << "    live-in: " << printReg(VReg, TRI) << " -> "
                        << TRI->getName(PhysReg) << "\n");
    }
  }

  for (const auto &LI : MBB->liveins()) {
    markOccupied(LI.PhysReg);
    LLVM_DEBUG(dbgs() << "    phys live-in: " << TRI->getName(LI.PhysReg)
                      << "\n");
  }

  // Anchor the shadow tree to the freshly-seeded OccupiedRegUnits (the reset()
  // above dropped the previous block's mirror). No-op unless shadowActive.
  shadowResetToOccupied();
}

bool AMDGPUSSARegisterAllocator::edgeCopiesNeedSplit(
    MachineBasicBlock *Pred, MachineBasicBlock *MBB,
    ArrayRef<std::pair<MCRegister, MCRegister>> Copies) const {
  SSARA_TRACE();
  // Not a critical edge -> placing the copies at Pred's terminator is safe.
  if (Pred->succ_size() <= 1 || MBB->pred_size() <= 1)
    return false;

  // Reg units written by the edge copies (the PHI-result destinations).
  BitVector DstUnits(TRI->getNumRegUnits());
  for (auto &[SrcPhys, DstPhys] : Copies)
    for (MCRegUnit U : TRI->regunits(DstPhys))
      DstUnits.set(U);
  auto Overlaps = [&](MCRegister PhysReg) {
    for (MCRegUnit U : TRI->regunits(PhysReg))
      if (DstUnits.test(U))
        return true;
    return false;
  };

  // A permutation cycle among the copies does NOT force a split.
  // resolvePermutation breaks a cycle either with a scratch register or with
  // V_SWAP_B32/XOR:
  //   - the scratch is allocated above the high-water mark (VGPR0 + MaxVGPRIdx
  //   /
  //     SGPR0 + MaxSGPRIdx), so it is free on every out-edge by construction
  //     and cannot clobber a sibling successor;
  //   - V_SWAP_B32/XOR only touch the cycle's own registers, i.e. the copy
  //     destinations, which the destination-clobber check below already covers.
  // (This relies on resolvePermutation picking the scratch above the high-water
  // mark; revisit this guard if that ever changes to reuse a lower free reg.)

  // Sibling successors (usually one) and their entry slots.
  SmallVector<SlotIndex, 2> SibStarts;
  for (MachineBasicBlock *Succ : Pred->successors())
    if (Succ != MBB)
      SibStarts.push_back(LIS->getMBBStartIdx(Succ));
  if (SibStarts.empty())
    return false;

  // Single ColorMap pass: the cheap reg-unit bit-test filters out the vast
  // majority; only a color overlapping a destination pays for the liveAt query.
  for (const auto &[VReg, PhysReg] : ColorMap) {
    if (!Overlaps(PhysReg))
      continue;
    const LiveInterval &LI = LIS->getInterval(VReg);
    for (SlotIndex S : SibStarts)
      if (LI.liveAt(S))
        return true; // a copy destination would clobber a sibling-live value
  }

  // Pre-existing physical-register live-ins of the siblings.
  for (MachineBasicBlock *Succ : Pred->successors()) {
    if (Succ == MBB)
      continue;
    for (const auto &LI : Succ->liveins())
      if (Overlaps(LI.PhysReg))
        return true;
  }
  return false;
}

// [Design: region-rp-reduction, Stage 1] ---------------------------------------

// THE SINGLE SOURCE OF TRUTH for "which physregs of RC may this allocation use".
// Everything else — the colorer's candidate scan, the pressure budget
// (allocatablePool), and the recovery/floor limits — derives from this ONE
// function, so a register is available in exactly one, consistent sense.
//
// It is the target's allocation order (RegClassInfo::getOrder, the same list the
// colorer scans) MINUS the WWM reserve: the SGPR stage runs first and may spill
// SGPRs that lower to VGPR lanes, and the downstream WWM pass needs VGPRReserve
// VGPRs of scratch. We drop them from the TAIL of the order (lowest-priority =
// numeric-highest VGPRs, which is exactly where WWM's high-register reservation
// takes its scratch). VGPRReserve is 0 during the SGPR stage and for SGPR
// classes, so those are unaffected.
ArrayRef<MCPhysReg>
AMDGPUSSARegisterAllocator::availableOrder(const TargetRegisterClass *RC) const {
  SSARA_TRACE();
  ArrayRef<MCPhysReg> Order = RegClassInfo.getOrder(RC);
  // Reserve only from the vector file (VGPR/AGPR share the vector budget).
  if (VGPRReserve && !TRI->isSGPRClass(RC)) {
    unsigned Drop = std::min<unsigned>(VGPRReserve, Order.size());
    Order = Order.drop_back(Drop);
  }
  return Order;
}

unsigned AMDGPUSSARegisterAllocator::allocatablePool(MachineFunction &MF,
                                                     RegFile File) const {
  SSARA_TRACE();
  // The colorer's real capacity is exactly the number of registers it may use =
  // availableOrder().size(). (SReg_32 gives 96 = 94 SGPRs + VCC_LO/VCC_HI, which
  // VCC-liveness handles per value; the vector pool has VGPRReserve withheld for
  // WWM.) Deriving the budget from the SAME list the colorer scans keeps the
  // pressure gate and the coloring capacity in lockstep.
  const TargetRegisterClass *RC =
      File == RegFile::SGPR   ? &AMDGPU::SReg_32RegClass
      : File == RegFile::AGPR ? &AMDGPU::AGPR_32RegClass
                              : &AMDGPU::VGPR_32RegClass;
  return availableOrder(RC).size();
}

unsigned AMDGPUSSARegisterAllocator::pressureOf(const GCNRegPressure &P,
                                                RegFile File) const {
  SSARA_TRACE();
  switch (File) {
  case RegFile::SGPR:
    return P.getSGPRNum();
  case RegFile::AGPR:
    return P.getAGPRNum();
  case RegFile::VGPR:
    // TWO-FILE MODEL: the VGPR file's demand is arch-VGPR (arch + avgpr); AGPR is
    // its OWN file (case above), so a value colored to AGPR relieves the VGPR file,
    // which a unified arch+acc sum cannot express.
    //
    // This was gated behind a flag for a while, on the grounds that arch-VGPR
    // shifted spill decisions on AGPR-using code (agpr-rescue puts values in AGPRs,
    // so Value[AGPR] != 0 and the two counts genuinely differ there). The unified
    // count is now the configuration that fails on exactly that code:
    // buffer-fat-pointers-memcpy.ll aborts in recovery on gfx90a and gfx942 with
    // the unified count and completes with arch-VGPR.
    return P.getArchVGPRNum();
  }
  llvm_unreachable("bad RegFile");
}

unsigned
AMDGPUSSARegisterAllocator::coveredSlots(const TargetRegisterClass *RC,
                                         LaneBitmask Lanes) const {
  SSARA_TRACE();
  // Mirrors GCNRegPressure::inc: a 32-bit class charges exactly 1 whatever its
  // mask, a tuple charges the 32-bit slots its live lanes cover. Keeping this in
  // one place is what lets a victim's demand, its spill traffic and a region peak
  // be compared at all.
  if (TRI->getRegSizeInBits(*RC) == 32)
    return 1;
  return std::max(1u, SIRegisterInfo::getNumCoveredRegs(Lanes));
}

unsigned AMDGPUSSARegisterAllocator::spilledSlots(Register V,
                                                  LaneBitmask Lanes) const {
  SSARA_TRACE();
  const TargetRegisterClass *RC = MRI->getRegClass(V);
  // The store narrows to the subregister the mask NAMES. A mask naming none — an
  // unnamed lane span corresponds to no subregister at all — falls back to storing
  // the whole register (VRegMaskPair::getSubReg -> spillAtDefinition), so the
  // traffic really is the full class. divideCeil, not /32: a 16-bit class occupies
  // one 32-bit slot and must not truncate to zero.
  if (VRegMaskPair(V, Lanes).getSubReg(MRI, TRI) == AMDGPU::NoRegister)
    return divideCeil(TRI->getRegSizeInBits(*RC), 32);
  return coveredSlots(RC, Lanes);
}

unsigned AMDGPUSSARegisterAllocator::reloadRPAtBlockEnd(const MachineBasicBlock *NCD,
                                                        bool IsVGPR) const {
  SSARA_TRACE();
  GCNUpwardRPTracker Tracker(*LIS);
  Tracker.reset(*NCD);
  GCNRegPressure P = Tracker.getPressure();
  if (!IsVGPR)
    return P.getSGPRNum();
  return P.getArchVGPRNum();
}

unsigned
AMDGPUSSARegisterAllocator::allocStride(const TargetRegisterClass *RC) const {
  SSARA_TRACE();
  auto It = StrideCache.find(RC);
  if (It != StrideCache.end())
    return It->second;
  // The spacing of LEGAL STARTS, taken from the order the colorer actually scans
  // rather than from a table, so the two can never disagree. TSFlags cannot serve:
  // its RegTupleAlignUnits field carries only the Align2 VGPR requirement and
  // reads 1 for every SGPR class, including the even-aligned 64-bit ones.
  SmallVector<unsigned, 128> Idx;
  for (MCPhysReg PR : availableOrder(RC))
    Idx.push_back(TRI->getHWRegIndex(PR));
  llvm::sort(Idx);
  unsigned S = 0;
  for (unsigned I = 1, E = Idx.size(); I < E; ++I)
    if (unsigned D = Idx[I] - Idx[I - 1])
      S = S ? std::min(S, D) : D;
  return StrideCache[RC] = S ? S : 1;
}

AMDGPUSSARegisterAllocator::RegFile
AMDGPUSSARegisterAllocator::poolOf(const TargetRegisterClass *RC) const {
  SSARA_TRACE();
  if (TRI->isSGPRClass(RC))
    return RegFile::SGPR;
  if (TRI->isAGPRClass(RC))
    return RegFile::AGPR;
  return RegFile::VGPR;
}

SSASpillEmitter::DemandFn
AMDGPUSSARegisterAllocator::demandFor(RegFile Pool) const {
  return [this, Pool](const GCNRPTracker::LiveRegSet &Live) {
    SmallVector<TierDemand, 8> Tiers;
    return demandDeficiency(Live, Pool, Tiers);
  };
}

unsigned AMDGPUSSARegisterAllocator::demandDeficiency(
    const GCNRPTracker::LiveRegSet &Live, RegFile Pool,
    SmallVectorImpl<TierDemand> &Out) const {
  SSARA_TRACE();
  Out.clear();
  for (auto [RegNum, Mask] : Live) {
    Register Reg(RegNum);
    const TargetRegisterClass *RC =
        Reg.isVirtual() ? MRI->getRegClassOrNull(Reg) : nullptr;
    if (!RC || poolOf(RC) != Pool)
      continue;
    unsigned W = divideCeil(TRI->getRegSizeInBits(*RC), 32);
    unsigned S = allocStride(RC);
    auto *T = llvm::find_if(
        Out, [&](const TierDemand &X) { return X.Width == W && X.Stride == S; });
    if (T == Out.end())
      Out.push_back({RC, W, S, 1, 0});
    else
      ++T->Demand;
  }
  if (Out.empty())
    return 0;
  llvm::sort(Out, [](const TierDemand &A, const TierDemand &B) {
    return std::tie(A.Width, A.Stride) > std::tie(B.Width, B.Stride);
  });

  const TargetRegisterClass *Base =
      Pool == RegFile::SGPR   ? &AMDGPU::SReg_32RegClass
      : Pool == RegFile::AGPR ? &AMDGPU::AGPR_32RegClass
                              : &AMDGPU::VGPR_32RegClass;
  // Occupancy indexed by HARDWARE register number, because a run needs
  // consecutive hardware registers and allocation order stops being hardware
  // order as soon as registers are reserved. Every index the pool's 32-bit order
  // does not offer starts out taken, so reserved registers and the withheld WWM
  // tail are excluded by the same availableOrder() the colorer uses.
  unsigned NumIdx = 0;
  for (MCPhysReg PR : availableOrder(Base))
    NumIdx = std::max(NumIdx, TRI->getHWRegIndex(PR) + 1);
  for (const TierDemand &T : Out)
    for (MCPhysReg PR : availableOrder(T.RC))
      NumIdx = std::max(NumIdx, TRI->getHWRegIndex(PR) + T.Width);
  BitVector Taken(NumIdx, true);
  for (MCPhysReg PR : availableOrder(Base))
    Taken.reset(TRI->getHWRegIndex(PR));

  unsigned Short = 0;
  for (TierDemand &T : Out) {
    for (unsigned I = 0; I != T.Demand; ++I) {
      bool Placed = false;
      for (MCPhysReg PR : availableOrder(T.RC)) {
        unsigned First = TRI->getHWRegIndex(PR);
        bool Free = true;
        for (unsigned K = 0; K != T.Width && Free; ++K)
          Free = !Taken.test(First + K);
        if (!Free)
          continue;
        Taken.set(First, First + T.Width);
        ++T.Placed;
        Placed = true;
        break;
      }
      if (!Placed)
        ++Short;
    }
  }
  return Short;
}

void AMDGPUSSARegisterAllocator::findTightRegions(
    MachineFunction &MF, RegFile File,
    SmallVectorImpl<TightRegion> &Out) const {
  SSARA_TRACE();
  const unsigned Limit = allocatablePool(MF, File);
  for (MachineBasicBlock &MBB : MF) {
    if (MBB.empty())
      continue;
    // Upward RP tracker: seed live-out at block end, recede toward the top (same
    // machinery as SSASpillEmitter::maxRPBetween). Collect per-slot RP, then scan
    // for over-limit runs in program order.
    GCNUpwardRPTracker Tracker(*LIS);
    Tracker.reset(MBB);
    SmallVector<std::pair<SlotIndex, unsigned>, 32> SlotShort; // bottom-to-top
    for (MachineInstr &MI : llvm::reverse(MBB)) {
      if (MI.isDebugInstr())
        continue;
      // Registers needed AT this instruction: what it defines, plus the operands
      // that stay live. Operands that die here release theirs, which is why the
      // colorer frees kills before assigning defs. Walking bottom-up, that set is
      // what the tracker holds BEFORE recede. After recede it holds the set above
      // the instruction, which counts a def at the NEXT instruction and so starts
      // every region one instruction late.
      // A slot is tight when some (pool, width) tier cannot be placed. A dword
      // total errs in both directions: it calls a fragmented under-limit slot
      // fine, and an over-limit slot that packs perfectly tight.
      SmallVector<TierDemand, 8> Tiers;
      unsigned Short = demandDeficiency(Tracker.getLiveRegs(), File, Tiers);
      Tracker.recede(MI);
      // PHIs carry NO real register pressure: their operands are parallel-copy
      // semantics resolved at PREDECESSOR EDGES, not simultaneously live at the
      // join. Receding across a PHI wall over-counts (every result + all incoming
      // operands appear coexisting), producing PHANTOM tight regions. Skip PHI
      // slots — the true live set is at the first non-PHI point (PHI results only,
      // operands already collapsed), which the non-PHI slots below capture.
      if (MI.isPHI())
        continue;
      SlotIndex SI = LIS->getInstructionIndex(MI).getRegSlot();
      SlotShort.push_back({SI, Short});
      LLVM_DEBUG(if (Short) dbgs()
                 << "    slotShort " << printMBBReference(MBB) << " @" << SI
                 << " short=" << Short << " OVER\n");
    }
    std::reverse(SlotShort.begin(), SlotShort.end()); // program order

    // Consume each maximal deficient run as one region (while(Over){...} form).
    for (unsigned I = 0, N = SlotShort.size(); I < N;) {
      if (!SlotShort[I].second) {
        ++I;
        continue;
      }
      SlotIndex RS = SlotShort[I].first;
      unsigned Peak = 0;
      SlotIndex PeakSlot = RS;
      while (I < N && SlotShort[I].second) {
        if (SlotShort[I].second > Peak) {
          Peak = SlotShort[I].second;
          PeakSlot = SlotShort[I].first;
        }
        ++I;
      }
      // Half-open end: first packing slot, or block end if the run reaches it.
      SlotIndex RE = (I < N) ? SlotShort[I].first : LIS->getMBBEndIdx(&MBB);
      Out.push_back({&MBB, RS, RE, PeakSlot, File, Peak, Limit});
      LLVM_DEBUG(dbgs() << "  findTightRegions[" << (File == RegFile::SGPR ? "SGPR"
                        : File == RegFile::AGPR ? "AGPR" : "VGPR")
                        << "] " << printMBBReference(MBB) << " [" << RS << ","
                        << RE << ") short=" << Peak << "@" << PeakSlot
                        << " pool=" << Limit
                        << " loopdepth=" << MLI->getLoopDepth(&MBB) << "\n");
    }
  }
}

void AMDGPUSSARegisterAllocator::reportLaneWaste(MachineFunction &MF) const {
  SSARA_TRACE();
  // Two occupancy models over one tracker walk: what this allocator charges
  // (the whole tuple, from markOccupied) and what LiveRegMatrix would charge
  // (only the register units whose subrange is live). The difference is
  // capacity Greedy keeps and this allocator does not.
  for (RegFile File : {RegFile::SGPR, RegFile::VGPR}) {
    unsigned PeakWhole = 0, LaneAtPeak = 0, MaxWaste = 0;
    for (MachineBasicBlock &MBB : MF) {
      if (MBB.empty())
        continue;
      GCNUpwardRPTracker Tracker(*LIS);
      Tracker.reset(MBB);
      for (MachineInstr &MI : llvm::reverse(MBB)) {
        if (MI.isDebugInstr())
          continue;
        Tracker.recede(MI);
        if (MI.isPHI())
          continue;
        unsigned Whole = 0, Lanes = 0;
        for (auto [RegNum, Mask] : Tracker.getLiveRegs()) {
          Register Reg(RegNum);
          const TargetRegisterClass *RC =
              Reg.isVirtual() ? MRI->getRegClassOrNull(Reg) : nullptr;
          if (!RC || fileOf(RC) != File)
            continue;
          // divideCeil, not /32: a 16-bit class covers one 32-bit unit, and
          // truncating it to 0 makes Lanes exceed Whole.
          Whole += divideCeil(TRI->getRegSizeInBits(*RC), 32);
          Lanes += SIRegisterInfo::getNumCoveredRegs(Mask);
        }
        if (Whole > PeakWhole) {
          PeakWhole = Whole;
          LaneAtPeak = Lanes;
        }
        MaxWaste = std::max(MaxWaste, Whole - Lanes);
      }
    }
    errs() << "ssara-lane-waste: " << MF.getName()
           << " file=" << (File == RegFile::SGPR ? "SGPR" : "VGPR")
           << " pool=" << allocatablePool(MF, File)
           << " peakWhole=" << PeakWhole << " laneAtPeak=" << LaneAtPeak
           << " maxWaste=" << MaxWaste << "\n";
  }
}

void AMDGPUSSARegisterAllocator::reportBlockDemand(MachineFunction &MF) const {
  SSARA_TRACE();
  // TEMPORARY MEASUREMENT — delete with the width-aware region gate it sizes.
  //
  // Prints FACTS, not a verdict. At each slot: the lane-accurate pressure the
  // region gate uses today, the whole-block dword sum, and how many live values
  // carry each (class width, allocation stride) pair. The block arithmetic —
  // widest size first, leftovers of one size becoming the next size down — is
  // done OFFLINE from these numbers, so no capacity model is baked in here and
  // the numbers stay usable if the model turns out wrong.
  //
  // Stride comes from the real allocation order, not from a table: it is the
  // smallest gap between the hardware indices of two entries, which is exactly
  // the spacing of legal starts for the class (2 for 64-bit SGPRs, 4 for wider
  // SGPRs, 1 or 2 for VGPR tuples depending on the subtarget).
  for (RegFile File : {RegFile::SGPR, RegFile::VGPR, RegFile::AGPR}) {
    const unsigned Limit = allocatablePool(MF, File);
    const char *FN = File == RegFile::SGPR   ? "SGPR"
                     : File == RegFile::AGPR ? "AGPR"
                                             : "VGPR";
    // Widest-slot fallback, so a function that trips no interest test still
    // reports its worst point.
    unsigned MaxWhole = 0;
    std::string MaxLine;
    unsigned Printed = 0;

    for (MachineBasicBlock &MBB : MF) {
      if (MBB.empty())
        continue;
      GCNUpwardRPTracker Tracker(*LIS);
      Tracker.reset(MBB);
      for (MachineInstr &MI : llvm::reverse(MBB)) {
        if (MI.isDebugInstr())
          continue;
        Tracker.recede(MI);
        if (MI.isPHI())
          continue; // PHIs carry no real pressure (see findTightRegions)

        struct Bucket {
          unsigned W, S, N;
          const TargetRegisterClass *RC;
        };
        SmallVector<Bucket, 8> Hist;
        unsigned Whole = 0;
        for (auto [RegNum, Mask] : Tracker.getLiveRegs()) {
          Register Reg(RegNum);
          const TargetRegisterClass *RC =
              Reg.isVirtual() ? MRI->getRegClassOrNull(Reg) : nullptr;
          if (!RC || poolOf(RC) != File)
            continue;
          unsigned W = divideCeil(TRI->getRegSizeInBits(*RC), 32);
          unsigned S = allocStride(RC);
          Whole += W;
          auto *B = llvm::find_if(Hist, [&](const Bucket &X) {
            return X.W == W && X.S == S;
          });
          if (B == Hist.end())
            Hist.push_back({W, S, 1, RC});
          else
            B->N++;
        }
        if (Hist.empty())
          continue;
        llvm::sort(Hist, [](const Bucket &A, const Bucket &B) {
          return std::tie(A.W, A.S) > std::tie(B.W, B.S);
        });

        // Same-file register operands of the instruction AT this slot. A peak
        // sitting on an instruction that reads most of the live set cannot be
        // relieved by spilling: every victim needs a reload right before it.
        unsigned NUse = 0, NDef = 0;
        for (const MachineOperand &MO : MI.operands()) {
          if (!MO.isReg() || !MO.getReg().isVirtual())
            continue;
          const TargetRegisterClass *RC = MRI->getRegClassOrNull(MO.getReg());
          if (!RC || poolOf(RC) != File)
            continue;
          MO.isDef() ? ++NDef : ++NUse;
        }
        unsigned Lane = pressureOf(Tracker.getPressure(), File);
        std::string Line;
        raw_string_ostream OS(Line);
        OS << "ssara-block-demand: " << MF.getName() << " file=" << FN
           << " pool=" << Limit << " @"
           << LIS->getInstructionIndex(MI).getRegSlot() << " lane=" << Lane
           << " whole=" << Whole << " mi=" << TII->getName(MI.getOpcode())
           << " uses=" << NUse << " defs=" << NDef;
        bool Interesting = Whole > Limit;
        for (const Bucket &B : Hist) {
          // disjoint = how many values of this width the pool could hold if it
          // held nothing else; starts = legal first registers the order offers.
          unsigned Disjoint = Limit / B.W;
          OS << " | w" << B.W << "/s" << B.S << " n=" << B.N
             << " disjoint=" << Disjoint
             << " starts=" << availableOrder(B.RC).size();
          if (B.W > 1 && B.N > Disjoint)
            Interesting = true;
        }
        // Shadow the scalar figures with the oracle, so a slot where the two
        // disagree is visible in the same line. Unsound disagreement is the
        // oracle reporting no deficiency where coloring then failed.
        SmallVector<TierDemand, 8> Tiers;
        unsigned Shortfall =
            demandDeficiency(Tracker.getLiveRegs(), File, Tiers);
        OS << " || oracle short=" << Shortfall;
        for (const TierDemand &T : Tiers)
          OS << " w" << T.Width << " d=" << T.Demand << " p=" << T.Placed;
        if (Shortfall)
          Interesting = true;
        if (Whole > MaxWhole) {
          MaxWhole = Whole;
          MaxLine = Line;
        }
        if (Interesting && Printed < 16) {
          ++Printed;
          errs() << Line << " OVER\n";
        }
      }
    }
    if (!MaxLine.empty())
      errs() << MaxLine << " MAXWHOLE\n";
  }
}

// Recovery helpers ------------------------------------------------------------

AMDGPUSSARegisterAllocator::RegFile
AMDGPUSSARegisterAllocator::fileOf(const TargetRegisterClass *RC) const {
  SSARA_TRACE();
  // AGPR folds into VGPR so this matches pressureOf(VGPR)'s unified count; only
  // SGPR classes are the SGPR file.
  return TRI->isSGPRClass(RC) ? RegFile::SGPR : RegFile::VGPR;
}

AMDGPUSSARegisterAllocator::SpillCost
AMDGPUSSARegisterAllocator::costOfSpilling(Register B, const TightRegion &R,
                                           LaneBitmask Lanes) {
  SSARA_TRACE();
  // Traffic = what the spill MOVES. Usually \p Lanes, because the store narrows to
  // their subregister and the slot is sized to it; when the lanes name no
  // subregister it is the whole register. spilledSlots decides which, so the cost
  // is never billed for lanes that are not stored, nor excused for lanes that are.
  const unsigned Width = spilledSlots(B, Lanes);
  // 2-way file (POC): AGPR folds into the VGPR pass like the rest of the emitter.
  Emitter->beginPass(R.File != RegFile::SGPR, demandFor(R.File));

  // Walk B's uses: fold the NCD of the non-PHI uses and collect what the reloads
  // will cost. There is no placement gate: a reload only ever restores a register
  // that its own reload point already required, so it cannot inflate any slot and
  // cannot make a candidate infeasible. Whether the spill helps is decided by
  // cumulative oracle remeasurement.
  SmallVector<MachineInstr *, 8> Uses;
  SmallPtrSet<MachineBasicBlock *, 8> PhiEdgeBlocks;
  MachineBasicBlock *NCD = nullptr;
  for (MachineInstr &UseMI : MRI->use_nodbg_instructions(B)) {
    MachineBasicBlock *UBB = UseMI.getParent();

    // A PHI reads its operand on the PREDECESSOR edge, so the emitter places the
    // reload at the end of each incoming block that supplies B, not at the PHI's
    // own slot. B is live out of that block either way, so the reload frees
    // nothing there and costs nothing extra — but it IS traffic, one reload per
    // supplying block (getOrCreateReloadInBlock caches per block), and the
    // footprint model charges residency from the terminator to block end.
    if (UseMI.isPHI()) {
      for (unsigned I = 1, E = UseMI.getNumOperands(); I + 1 < E; I += 2) {
        if (UseMI.getOperand(I).getReg() != B)
          continue;
        PhiEdgeBlocks.insert(UseMI.getOperand(I + 1).getMBB());
      }
      // Do not fold the PHI block into the NCD dominance merge: its use point is
      // the predecessor edge, not the PHI's own block.
      continue;
    }

    // A use inside R is NOT by itself a reason to reject. The reload lands at that
    // use, so the victim keeps its register there — but across the slots where it
    // is live and untouched its register really is freed, and those may be the
    // over-limit ones. The footprint measures that per slot and frees nothing
    // when the victim is held everywhere R is tight. The same holds for a use in R's
    // loop: the reload restores the register at a point that reads B, so it adds
    // nothing that was not already there.
    NCD = NCD ? MDT->findNearestCommonDominator(NCD, UBB) : UBB;
    Uses.push_back(&UseMI);
  }

  // Reload set + Test 2. Commonly dominated + hoistable (hoist itself must not
  // cross R) -> ONE reload at NCD end; else one reload per use. Each reload's
  // post-spill RP must stay <= R.Limit. canHoistReloadTo (InsertPoint==null)
  // skips the NCD-block RP check, so reloadRPAtBlockEnd covers it here.
  const bool HoistOK = NCD && NCD != R.MBB && !MDT->dominates(NCD, R.MBB) &&
                       Emitter->canHoistReload(NCD, B);

  // reloadRPBeforeUse/reloadRPAtBlockEnd already include the reloaded value
  // present at the reload point (the -W+W cancel), so RP > Limit is the correct
  // "no room for the reload" test — do NOT add Width (that double-counts).
  // A SINGLE hoisted reload at the NCD block end is cheapest, but only legal if
  // that point has room. When it does NOT (the hoisted reload would pile all uses'
  // live-in at one over-pressure point — e.g. the 128-dword result block of a wide
  // bitcast, where a hoisted reload sees RP=128>64), DO NOT give up: fall through
  // to PER-USE reloads. A per-use reload lands right before each individual use and
  // dies immediately after it, so its point pressure is the ROLLING-WINDOW demand
  // (only a few reloads live at once), not the block's total throughput. This is
  // exactly how Greedy spills the wide-bitcast result block: 81 reloads all in
  // %end, but distributed through the sequential pack-4-bytes-and-store so no point
  // exceeds the limit. Rejecting at the hoist test (the old behavior) was the
  // deadlock: every crosser's shared reload piled in the 128-RP block -> all
  // rejected -> nothing spilled.
  // Reload traffic is LOOP-DEPTH WEIGHTED: a reload placed in a loop executes once
  // per iteration, so it costs 2^loopdepth (same weight the PHI-coalescer uses for
  // edges). A reload at depth 0 costs 1; at depth d costs 2^d. The cost is the SUM
  // over the reloads B forces, each weighted by its placement block's depth, times
  // Width (dwords moved per reload). The area planner uses this as a deterministic
  // tie-breaker, so equal-area choices still prefer lower dynamic traffic.
  auto depthWeight = [&](MachineBasicBlock *MBB) -> uint64_t {
    unsigned D = MBB ? MLI->getLoopDepth(MBB) : 0;
    // Saturate rather than shift past the width of the weight type.
    constexpr unsigned MaxShift = std::numeric_limits<uint64_t>::digits - 1;
    return D < MaxShift ? (uint64_t(1) << D) : ~uint64_t(0);
  };
  // One reload per predecessor supplying a PHI use, whichever way the non-PHI
  // uses are served. Omitting these priced a victim whose uses are ALL PHIs at
  // zero, which sorts it ahead of every real candidate.
  uint64_t PhiEdgeWeight = 0;
  for (MachineBasicBlock *PredBB : PhiEdgeBlocks)
    PhiEdgeWeight += depthWeight(PredBB);
  // An uncolored value is still a live virtual register, so the pressure trackers
  // below already count it. Adding it again here inflated every measurement taken
  // after coloring by the width of the uncolored values at that slot.
  const bool WantVG = R.File != RegFile::SGPR;
  uint64_t WeightedReloads;
  bool UseHoist = HoistOK && reloadRPAtBlockEnd(NCD, WantVG) <= R.Limit;
  if (UseHoist) {
    // One shared reload at the NCD block end.
    WeightedReloads = depthWeight(NCD) + PhiEdgeWeight;
  } else {
    LLVM_DEBUG(if (HoistOK) dbgs()
               << "      cost " << printReg(B, TRI) << ": hoist infeasible (NCD-end RP "
               << reloadRPAtBlockEnd(NCD, WantVG) << " > " << R.Limit
               << ") -> trying per-use\n");
    // Per-use reloads land at DISTINCT slots and die immediately, so they do not
    // accumulate against one another. Nor is there anything to reject here: B is
    // already live at its own use, so the reload restores a register that slot
    // needed anyway and the pressure there is the same spilled or not. Comparing
    // it against the limit says nothing about the spill, and inside a tight region
    // it is over the limit by definition, so it rejected every candidate and
    // nothing was ever spilled. The cumulative oracle decides whether the spill
    // helps; this loop only prices the traffic.
    uint64_t W = 0;
    for (MachineInstr *UseMI : Uses)
      W += depthWeight(UseMI->getParent());
    WeightedReloads = W + PhiEdgeWeight;
  }

  // Cost = weighted reload traffic * dwords moved. Clamp to unsigned for the
  // struct; deep loops saturate but stay ordered (huge => spilled last).
  uint64_t C = WeightedReloads * Width;
  unsigned Cost = C > ~0u ? ~0u : unsigned(C);
  LLVM_DEBUG(dbgs() << "      cost " << printReg(B, TRI) << ": FEASIBLE cost="
                    << Cost << " width=" << Width
                    << " weightedReloads=" << WeightedReloads << "\n");
  return {Cost, Width};
}

std::pair<SlotIndex, unsigned>
AMDGPUSSARegisterAllocator::peakSlotForValueInRegion(const TightRegion &R,
                                                     Register V) const {
  SSARA_TRACE();
  // Walk R's block bottom-to-top (same tracker as findTightRegions) and record
  // the max-RP slot at which V is live. This is the slot recovery must relieve:
  // spilling victims at R's GLOBAL peak is useless if V is dead there (a plateau
  // region where V occupies only a sub-span).
  const LiveInterval &VI = LIS->getInterval(V);
  SlotIndex BestSlot;
  unsigned BestRP = 0;
  GCNUpwardRPTracker Tracker(*LIS);
  Tracker.reset(*R.MBB);
  for (MachineInstr &MI : llvm::reverse(*R.MBB)) {
    if (MI.isDebugInstr())
      continue;
    Tracker.recede(MI);
    if (MI.isPHI())
      continue; // PHIs carry no real pressure (see findTightRegions)
    SlotIndex SI = LIS->getInstructionIndex(MI).getRegSlot();
    if (SI < R.Start || R.End <= SI)
      continue; // outside the region span
    if (!VI.liveAt(SI))
      continue; // V not live here -> spilling here cannot relieve V
    unsigned RP = pressureOf(Tracker.getPressure(), R.File);
    if (RP > BestRP) {
      BestRP = RP;
      BestSlot = SI;
    }
  }
  return {BestSlot, BestRP};
}

unsigned AMDGPUSSARegisterAllocator::measureRegionPeak(
    const TightRegion &R, DenseMap<Register, RegionOccupancy> *Occupants,
    SmallVectorImpl<RegionSlot> *Slots,
    SmallVectorImpl<TierDemand> *PeakTiers) const {
  SSARA_TRACE();
  // Same machinery findTightRegions used, so the number is comparable bit for bit,
  // but walking ONLY [R.Start,R.End): the tracker is seeded from LIS at the
  // bottom-most in-region instruction (reset(MI) = the live set just after MI, as
  // reloadRPBeforeUse does), so no block prefix or suffix is traversed.
  //
  // With \p Occupants the same walk also reports WHO is in the region: each virtual
  // register live at an in-region non-PHI slot, the union of its live lanes there,
  // and at how many slots. That IS the overlap test, hole-accurate for free, since
  // the tracker's live set never holds a value inside its own liveness hole.
  MachineInstr *StartMI = LIS->getInstructionFromIndex(R.Start);
  if (!StartMI || StartMI->getParent() != R.MBB) {
    // R's first instruction is gone (erased by an earlier spill this pass). Report
    // "fits": re-deriving the span here would measure a different region.
    LLVM_DEBUG(dbgs() << "    measureRegionPeak: start slot " << R.Start
                      << " no longer maps into " << printMBBReference(*R.MBB)
                      << "\n");
    return 0;
  }
  // R.End is half-open: the first slot NOT over the limit, or the block end index,
  // where no instruction exists.
  MachineInstr *EndMI = LIS->getInstructionFromIndex(R.End);
  MachineBasicBlock::iterator B = StartMI->getIterator();
  MachineBasicBlock::iterator E = (EndMI && EndMI->getParent() == R.MBB)
                                      ? EndMI->getIterator()
                                      : R.MBB->end();
  GCNUpwardRPTracker Tracker(*LIS);
  bool Seeded = false;
  unsigned Peak = 0;
  for (MachineBasicBlock::iterator I = E; I != B;) {
    MachineInstr &MI = *--I;
    if (MI.isDebugInstr())
      continue;
    if (!Seeded) {
      Tracker.reset(MI);
      Seeded = true;
    }
    // PHIs carry no real pressure (see findTightRegions).
    if (!MI.isPHI()) {
      // The live-out side, read before recede: what this instruction defines plus
      // what stays live past it. Occupancy is read from the SAME set, or a value
      // defined at a slot would not be an occupant of the region that needs it
      // spilled.
      auto &Live = Tracker.getLiveRegs();
      SmallVector<TierDemand, 8> Tiers;
      unsigned Short = demandDeficiency(Live, R.File, Tiers);
      if (Short > Peak) {
        Peak = Short;
        if (PeakTiers)
          *PeakTiers = Tiers;
      }
      if (Slots)
        Slots->push_back(
            {LIS->getInstructionIndex(MI).getRegSlot(), &MI, Live, Short});
      if (Occupants)
        for (auto [RegNum, Mask] : Live) {
          Register Reg(RegNum);
          if (!Reg.isVirtual())
            continue;
          RegionOccupancy &O = (*Occupants)[Reg];
          O.Lanes |= Mask; // union over the region: what R actually holds of Reg
          O.Slots += 1;
        }
    }
    Tracker.recede(MI);
  }
  return Peak;
}

void AMDGPUSSARegisterAllocator::dumpWorstSlotLiveSet(
    ArrayRef<RegionSlot> Slots, const char *Tag) const {
  LLVM_DEBUG({
    const RegionSlot *Worst = nullptr;
    for (const RegionSlot &S : Slots)
      if (!Worst || S.Short > Worst->Short)
        Worst = &S;
    if (Worst) {
      dbgs() << "    [WA] " << Tag << " worst@" << Worst->SI
             << " short=" << Worst->Short << " live:";
      SmallVector<std::pair<unsigned, unsigned>, 32> L; // (width, vreg index)
      for (const auto &KV : Worst->Live) {
        Register V(KV.first);
        unsigned W = TRI->getRegSizeInBits(*MRI->getRegClass(V)) / 32;
        L.push_back({W ? W : 1, V.virtRegIndex()});
      }
      llvm::sort(L, std::greater<std::pair<unsigned, unsigned>>());
      for (const auto &P : L)
        dbgs() << " %" << P.second << "(w" << P.first << ")";
      dbgs() << "\n";
    }
  });
}

bool AMDGPUSSARegisterAllocator::relieveTightRegion(
    const TightRegion &R, const SmallDenseSet<Register, 128> &Universe,
    SmallDenseSet<Register, 64> &Spilled,
    llvm::function_ref<bool(Register)> Eligible, unsigned *NumRecolored) {
  SSARA_TRACE();
  DenseMap<Register, RegionOccupancy> Occ;
  SmallVector<RegionSlot, 32> Slots;
  unsigned Peak = measureRegionPeak(R, &Occ, &Slots);
  if (!Peak)
    return false;
  dumpWorstSlotLiveSet(Slots, "pre ");

  SmallVector<SpillCandidateInput, 32> Inputs;
  for (Register V : Universe) {
    if (Spilled.count(V) || MRI->reg_nodbg_empty(V) || !LIS->hasInterval(V))
      continue;
    const TargetRegisterClass *RC = MRI->getRegClass(V);
    auto It = Occ.find(V);
    if (poolOf(RC) != R.File || It == Occ.end() || !Eligible(V))
      continue;
    bool CanRecolor =
        ST->hasGFX90AInsts() && R.File == RegFile::VGPR &&
        TRI->isVectorSuperClass(RC) && TRI->getEquivalentAGPRClass(RC);
    Inputs.push_back({V, It->second.Lanes, CanRecolor});
  }
  llvm::sort(Inputs, [](const SpillCandidateInput &A,
                        const SpillCandidateInput &B) {
    return A.V.id() < B.V.id();
  });

  unsigned AGPRBudget = 0;
  if (ST->hasGFX90AInsts() && R.File == RegFile::VGPR)
    AGPRBudget = allocatablePool(
        const_cast<MachineFunction &>(MRI->getMF()), RegFile::AGPR);

  Emitter->beginPass(R.File != RegFile::SGPR, demandFor(R.File));
  SmallVector<AreaSpillAction, 16> Actions;
  unsigned VirtualShort = 0;
  if (!planAreaSpillSet(R, Slots, Inputs, AGPRBudget, Actions,
                        &VirtualShort))
    return false;

  for (const AreaSpillAction &A : Actions)
    if (A.Kind == AreaSpillAction::Memory &&
        !Emitter->canSpill(VRegMaskPair(A.V, A.Lanes)))
      return false;

  for (const AreaSpillAction &A : Actions) {
    if (A.Kind == AreaSpillAction::Recolor) {
      const TargetRegisterClass *AGPR =
          TRI->getEquivalentAGPRClass(MRI->getRegClass(A.V));
      assert(AGPR && "planned AGPR recolor lost its target class");
      MRI->setRegClass(A.V, AGPR);
      if (NumRecolored)
        ++*NumRecolored;
    } else {
      Emitter->spillOneVMP(VRegMaskPair(A.V, A.Lanes),
                           LIS->getInterval(A.V).beginIndex());
    }
    Spilled.insert(A.V);
  }

  Slots.clear();
  unsigned Actual = measureRegionPeak(R, nullptr, &Slots);
  LLVM_DEBUG(dbgs() << "    [AREA] committed set=" << Actions.size()
                    << " virtual-short=" << VirtualShort
                    << " actual-short=" << Actual << "\n");
  dumpWorstSlotLiveSet(Slots, "post");
  return true;
}

bool AMDGPUSSARegisterAllocator::preSpillToLimitWidthAware(MachineFunction &MF) {
  SSARA_TRACE();
  // WIDTH-AWARE up-front spiller. Runs BEFORE color(): at each tight region's
  // peak it spills frozen victims (kill-at-def store, reload at use) until the
  // peak fits the allocatable pool, so the coloring walk succeeds by
  // construction. Two properties decide which regions it can relieve:
  //   (1) the frozen victim UNIVERSE spans ALL widths, and
  //   (2) victims are chosen WIDEST-FIRST and the region peak is decremented by
  //       the victim's REAL dword width.
  // Both are what reach a region dominated by wide tuples (vreg_64/128/... in
  // either file), where nothing of width 1 is live at the peak at all. The
  // SGPR-wide bookkeeping bug and the 128xfloat emergency-slot cases are exactly
  // these: the pressure is carried by wide SGPR/VGPR tuples.
  //
  // THE GATE IS WHOLE-BLOCK DEMAND, not lane-accurate pressure: this allocator
  // hands out whole tuples (markOccupied sets every unit of the assigned register),
  // so a partially-live tuple costs its full width and the lane reading understates
  // what the coloring walk needs. See wholeBlockDemand. The job ends there — the
  // pre-spiller only guarantees that at every slot SOME assignment exists. It does
  // not model fragmentation, which cannot exist before anything is coloured, and a
  // value left with no legal run is a coloring failure for recovery to handle.
  bool Any = false;
  SmallDenseSet<Register, 64> Spilled; // never re-pick a spilled value
  // FROZEN UNIVERSE (termination): every vreg that exists BEFORE any spilling, of
  // ANY width. Reload redefs spillOneVMP creates are fresh vregs NOT in the set,
  // so they can never become victims -> no rolling-wave regeneration. The
  // spillable set strictly shrinks; the loop is bounded by |Universe|.
  SmallDenseSet<Register, 128> Universe;
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register V = Register::index2VirtReg(I);
    if (MRI->reg_nodbg_empty(V) || !LIS->hasInterval(V))
      continue;
    Universe.insert(V);
  }
  // TERMINATION + DO-NO-HARM, one progress metric, NO cap. The measure is the
  // TOTAL deficiency = sum over tight regions of the runs they are short. A kept
  // round must STRICTLY reduce it:
  //  - relief (spill or AGPR-recolor) of any region lowers that region's excess,
  //    reducing the sum — even when a DIFFERENT region still holds the max (the
  //    reason a per-region-MAX metric wrongly stalled: it ignored progress made on
  //    a non-max region);
  //  - a memory spill whose reloads re-materialize pressure elsewhere (fresh vregs
  //    outside the frozen universe, unspillable) fails to reduce the sum — the
  //    rolling wave — and bails, leaving the residual to the colorer (do no harm).
  // The sum is a non-negative integer strictly decreasing on every kept round, so
  // the loop terminates in at most its initial value of rounds without a backstop.
  // The vector stage owns two DISJOINT register sets. fileOf folds them so one
  // coloring stage handles both, which is right for coloring and wrong for
  // demand: enumerated as one pool an AGPR shortage is invisible, and an AGPR
  // value inflates the arch-VGPR reading of a pool it will never occupy.
  SmallVector<RegFile, 2> Pools;
  if (StageFile == RegFile::SGPR)
    Pools.push_back(RegFile::SGPR);
  else {
    Pools.push_back(RegFile::VGPR);
    Pools.push_back(RegFile::AGPR);
  }
  long PrevTotal = -1; // total deficiency at the START of the last round
  while (true) {
    bool Changed = false;
    long TotalThisRound = 0;
    for (RegFile PF : Pools) {
      SmallVector<TightRegion, 8> PR;
      findTightRegions(MF, PF, PR);
      for (const TightRegion &R : PR)
        TotalThisRound += long(R.Deficiency);
    }
    if (TotalThisRound == 0)
      break; // every point fits — done
    if (PrevTotal >= 0 && TotalThisRound >= PrevTotal) {
      LLVM_DEBUG(dbgs() << "    [WA] no progress (total excess " << PrevTotal
                        << " -> " << TotalThisRound
                        << "): rolling wave, hand residual to colorer\n");
      break;
    }
    PrevTotal = TotalThisRound;
    for (RegFile File : Pools) {
      SmallVector<TightRegion, 8> Regions;
      findTightRegions(MF, File, Regions);
      if (Regions.empty())
        continue;
      Emitter->beginPass(File != RegFile::SGPR, demandFor(File));
      for (const TightRegion &R : Regions) {
        if (relieveTightRegion(R, Universe, Spilled,
                               [](Register) { return true; }))
          Any = Changed = true;
      }
    }
    if (!Changed)
      break; // fixpoint: every point <= Limit (or nothing left to spill)
  }
  return Any;
}

bool AMDGPUSSARegisterAllocator::planAreaSpillSet(
    const TightRegion &R, ArrayRef<RegionSlot> Slots,
    ArrayRef<SpillCandidateInput> Inputs, unsigned RecolorBudget,
    SmallVectorImpl<AreaSpillAction> &Actions, unsigned *RemainingShort) {
  SSARA_TRACE();
  Actions.clear();
  if (Slots.empty())
    return false;

  struct Candidate {
    AreaSpillAction Action;
    SmallVector<unsigned, 16> FreedSlots;
  };

  DenseMap<Register, unsigned> DirectIndex;
  SmallVector<SmallBitVector, 32> DirectLiveSlots;
  SmallVector<SmallBitVector, 32> DirectResidentSlots;
  if (RecoveryOnly) {
    for (unsigned I = 0; I != Inputs.size(); ++I) {
      DirectIndex.try_emplace(Inputs[I].V, I);
      DirectLiveSlots.emplace_back(Slots.size());
      DirectResidentSlots.emplace_back(Slots.size());
    }

    for (unsigned S = 0; S != Slots.size(); ++S) {
      for (unsigned I = 0; I != Inputs.size(); ++I)
        if (Slots[S].Live.count(Inputs[I].V.id()))
          DirectLiveSlots[I].set(S);

      for (const MachineOperand &MO : Slots[S].MI->operands()) {
        if (!MO.isReg() || !MO.getReg().isVirtual())
          continue;
        auto It = DirectIndex.find(MO.getReg());
        if (It == DirectIndex.end())
          continue;
        if ((VRegMaskPair(MO, TRI, MRI).getLaneMask() &
             Inputs[It->second].Lanes)
                .any())
          DirectResidentSlots[It->second].set(S);
      }
    }

    auto FirstTerm = R.MBB->getFirstTerminator();
    if (FirstTerm != R.MBB->end()) {
      SmallBitVector PhiEdgeSlots(Slots.size());
      SlotIndex FirstTermIdx =
          LIS->getInstructionIndex(*FirstTerm).getRegSlot();
      for (unsigned S = 0; S != Slots.size(); ++S)
        if (LIS->getInstructionIndex(*Slots[S].MI).getRegSlot() >=
            FirstTermIdx)
          PhiEdgeSlots.set(S);

      for (MachineBasicBlock *Succ : R.MBB->successors())
        for (const MachineInstr &Phi : *Succ) {
          if (!Phi.isPHI())
            break;
          for (unsigned I = 1, E = Phi.getNumOperands(); I + 1 < E; I += 2) {
            const MachineOperand &Val = Phi.getOperand(I);
            if (!Val.isReg() || !Val.getReg().isVirtual() ||
                Phi.getOperand(I + 1).getMBB() != R.MBB)
              continue;
            auto It = DirectIndex.find(Val.getReg());
            if (It == DirectIndex.end())
              continue;
            if ((VRegMaskPair(Val, TRI, MRI).getLaneMask() &
                 Inputs[It->second].Lanes)
                    .any())
              DirectResidentSlots[It->second] |= PhiEdgeSlots;
          }
        }
    }
  }

  SmallVector<MachineInstr *, 32> Sites;
  if (!RecoveryOnly)
    for (const RegionSlot &S : Slots)
      Sites.push_back(S.MI);

  SmallVector<Candidate, 32> Candidates;
  for (const SpillCandidateInput &I : Inputs) {
    if (!LIS->hasInterval(I.V) || LIS->getInterval(I.V).empty())
      continue;
    if (PhiWeb Web = closePhiWeb(I.V); Web.valid()) {
      LLVM_DEBUG(dbgs() << "    [AREA] exclude " << printReg(I.V, TRI)
                        << ": PHI web requires atomic multi-value footprint\n");
      continue;
    }

    const TargetRegisterClass *RC = MRI->getRegClass(I.V);
    unsigned Width = coveredSlots(RC, I.Lanes);
    Candidate C;
    C.Action.V = I.V;
    C.Action.Lanes = I.Lanes;
    if (!RecoveryOnly)
      C.Action.Cost = costOfSpilling(I.V, R, I.Lanes).Cost;
    if (RecoveryOnly) {
      unsigned INo = DirectIndex.lookup(I.V);
      SmallBitVector Freed = DirectLiveSlots[INo];
      Freed.reset(DirectResidentSlots[INo]);
      for (int S = Freed.find_first(); S >= 0; S = Freed.find_next(S))
        C.FreedSlots.push_back(S);
      C.Action.FootprintExact = true;
    } else {
      SSASpillEmitter::SpillFootprint Footprint =
          Emitter->modelSpillFootprint(VRegMaskPair(I.V, I.Lanes),
                                       LIS->getInterval(I.V).beginIndex(),
                                       R.MBB, Sites);
      C.Action.FootprintExact = Footprint.Exact;
      for (unsigned S = 0; S != Slots.size(); ++S)
        if (Slots[S].Live.count(I.V.id()) &&
            !Footprint.ResidentSlots.test(S))
          C.FreedSlots.push_back(S);
    }
    C.Action.Area = uint64_t(Width) * C.FreedSlots.size();
    if (C.Action.Area)
      Candidates.push_back(std::move(C));

    if (I.CanRecolor && Width <= RecolorBudget) {
      Candidate Recolor;
      Recolor.Action.V = I.V;
      Recolor.Action.Lanes = I.Lanes;
      Recolor.Action.Kind = AreaSpillAction::Recolor;
      Recolor.Action.FootprintExact = true;
      for (unsigned S = 0; S != Slots.size(); ++S)
        if (Slots[S].Live.count(I.V.id()))
          Recolor.FreedSlots.push_back(S);
      Recolor.Action.Area = uint64_t(Width) * Recolor.FreedSlots.size();
      if (Recolor.Action.Area)
        Candidates.push_back(std::move(Recolor));
    }
  }

  llvm::sort(Candidates, [](const Candidate &A, const Candidate &B) {
    if (A.Action.Area != B.Action.Area)
      return A.Action.Area > B.Action.Area;
    if (A.Action.Cost != B.Action.Cost)
      return A.Action.Cost < B.Action.Cost;
    return A.Action.V.id() < B.Action.V.id();
  });

  SmallVector<RegionSlot, 32> Virtual(Slots.begin(), Slots.end());
  SmallBitVector Selected(Candidates.size());
  SmallDenseSet<Register, 16> SelectedRegs;
  unsigned BudgetLeft = RecolorBudget;
  auto totalShort = [&]() {
    unsigned Total = 0;
    for (const RegionSlot &S : Virtual)
      Total += S.Short;
    return Total;
  };

  const unsigned InitialShort = totalShort();
  unsigned BestShort = InitialShort;
  unsigned BestSize = 0;

  while (totalShort()) {
    int Pick = -1;
    for (unsigned C = 0; C != Candidates.size(); ++C) {
      if (Selected.test(C))
        continue;
      const Candidate &Cand = Candidates[C];
      if (SelectedRegs.count(Cand.Action.V))
        continue;
      if (Cand.Action.Kind == AreaSpillAction::Recolor) {
        unsigned W = coveredSlots(MRI->getRegClass(Cand.Action.V),
                                  Cand.Action.Lanes);
        if (W > BudgetLeft)
          continue;
      }
      if (llvm::any_of(Cand.FreedSlots,
                       [&](unsigned S) { return Virtual[S].Short != 0; })) {
        Pick = C;
        break;
      }
    }
    if (Pick < 0)
      break;

    Candidate &Cand = Candidates[Pick];
    Selected.set(Pick);
    SelectedRegs.insert(Cand.Action.V);
    if (Cand.Action.Kind == AreaSpillAction::Recolor)
      BudgetLeft -= coveredSlots(MRI->getRegClass(Cand.Action.V),
                                 Cand.Action.Lanes);
    for (unsigned S : Cand.FreedSlots)
      Virtual[S].Live.erase(Cand.Action.V.id());
    for (RegionSlot &S : Virtual) {
      SmallVector<TierDemand, 8> Tiers;
      S.Short = demandDeficiency(S.Live, R.File, Tiers);
    }
    Actions.push_back(Cand.Action);
    unsigned CurrentShort = totalShort();
    if (CurrentShort < BestShort) {
      BestShort = CurrentShort;
      BestSize = Actions.size();
    }
    LLVM_DEBUG(dbgs() << "    [AREA] choose "
                      << printReg(Cand.Action.V, TRI)
                      << " area=" << Cand.Action.Area
                      << " remaining-short=" << CurrentShort
                      << " footprint="
                      << (Cand.Action.FootprintExact ? "exact" : "conservative")
                      << "\n");
    if (RecoveryOnly)
      break;
  }

  Actions.resize(BestSize);
  if (RemainingShort)
    *RemainingShort = BestShort;
  if (Actions.empty()) {
    LLVM_DEBUG(dbgs() << "    [AREA] no measured improvement; emit nothing\n");
    Actions.clear();
    return false;
  }
  LLVM_DEBUG(dbgs() << "    [AREA] retain improving prefix=" << BestSize
                    << " short=" << InitialShort << "->" << BestShort
                    << "; defer residual to coloring and recovery\n");
  return !Actions.empty();
}

bool AMDGPUSSARegisterAllocator::reduceRegionPressure(MachineFunction &MF) {
  SSARA_TRACE();
  // RP-PROFILE spill-across (post-color recovery). Tight regions come from
  // findTightRegions: per BLOCK, measured with GCNUpwardRPTracker, so liveness
  // HOLES, subranges and PHI semantics are the tracker's and a region can never
  // span blocks. The event sweep this replaces collapsed every value to its hull
  // [beginIndex,endIndex) — holes stopped existing — and ran on ONE global slot
  // axis, so a single "region" could cover several blocks and SUM mutually
  // exclusive divergent paths.
  //
  // Per region: construct one complete cumulative set over copied live snapshots,
  // emit it only if virtual deficiency reaches zero, then remeasure actual state.
  // The caller does not recolor-iterate; it recolors once and hands any model
  // discrepancy to the split path. Returns true if any spill was performed.
  bool AnySpill = false;
  SmallDenseSet<Register, 32> Spilled; // never re-pick within this pass

  // Only the current allocation stage's pools. The vector stage owns two of them:
  // arch-VGPR and AGPR are disjoint register sets with separate orders, so
  // enumerating them as one hides an AGPR shortage and inflates the arch-VGPR
  // reading with values that will never occupy that pool. Everything below is
  // per REGION (R.File, R.Limit), never per stage.
  SmallVector<RegFile, 2> Pools;
  if (StageFile == RegFile::SGPR)
    Pools.push_back(RegFile::SGPR);
  else {
    Pools.push_back(RegFile::VGPR);
    Pools.push_back(RegFile::AGPR);
  }

  SmallVector<TightRegion, 8> Regions;
  for (RegFile P : Pools)
    findTightRegions(MF, P, Regions);
  if (Regions.empty())
    return false;

  LLVM_DEBUG(dbgs() << "region-rp[" << (StageFile == RegFile::SGPR ? "SGPR"
                                                                   : "VGPR+AGPR")
                    << "]: tight-regions=" << Regions.size() << "\n");

  struct Cand {
    Register VReg;
    LaneBitmask Lanes; // lanes R holds; exactly what gets spilled
  };

  for (const TightRegion &R : Regions) {
    // ONE walk of R yields its current deficiency, its occupants, and the
    // per-slot live sets that the cumulative set planner evaluates.
    // Re-measuring matters: regions were enumerated up front, so spills made for
    // an earlier region may already have relieved this one (R.Deficiency is then
    // stale).
    // Not a pool selector: it picks the emitter's vector-vs-scalar spill
    // mechanics, and AGPR is a vector file.
    const bool IsVectorFile = R.File != RegFile::SGPR;
    DenseMap<Register, RegionOccupancy> Occupants;
    SmallVector<RegionSlot, 32> Slots;
    unsigned Peak = measureRegionPeak(R, &Occupants, &Slots);
    LLVM_DEBUG(dbgs() << "  region " << printMBBReference(*R.MBB) << " ["
                      << R.Start << "," << R.End << ") short=" << Peak
                      << " (enumerated " << R.Deficiency << ") occupants="
                      << Occupants.size() << "\n");
    if (!Peak)
      continue;

    // Candidates = occupant AND colored. Occupancy comes from the tracker's live
    // set, so it is hole-accurate by construction (a value is never in the set
    // inside its own liveness hole — the hull test this replaces admitted exactly
    // that, which is how a value sitting in its own hole was picked with cover=8),
    // and it is not liveAt(R.PeakSlot) either, because a peak can be a PLATEAU and
    // that strict test dropped victims covering most of R (cf512 28->6).
    // Uncolored occupants are counted in the peak but can be nobody's victim.
    SmallVector<Cand, 32> Cands;
    for (const auto &[V, Occ] : Occupants) {
      if (!ColorMap.count(V))
        continue;
      const TargetRegisterClass *RC = MRI->getRegClass(V);
      // poolOf, NOT isVectorRegister: the latter is isVGPRClass||isAGPRClass, both
      // FALSE for the AV_* vector-super classes this allocator deliberately widens
      // plain VGPR values into.
      if (poolOf(RC) != R.File)
        continue;
      Cands.push_back({V, Occ.Lanes});
    }
    if (Cands.empty())
      continue;
    // DenseMap order is not stable across runs and selection ties would leak it
    // into the output. (Iterating ColorMap had the same latent non-determinism.)
    llvm::sort(Cands, [](const Cand &A, const Cand &B) {
      return A.VReg.id() < B.VReg.id();
    });

    SmallVector<SpillCandidateInput, 32> Inputs;
    for (const Cand &C : Cands)
      if (!Spilled.count(C.VReg))
        Inputs.push_back({C.VReg, C.Lanes, false});

    Emitter->beginPass(IsVectorFile, demandFor(R.File));
    SmallVector<AreaSpillAction, 16> Actions;
    unsigned VirtualShort = 0;
    if (!planAreaSpillSet(R, Slots, Inputs, /*RecolorBudget=*/0, Actions,
                          &VirtualShort))
      continue;

    bool Legal = llvm::all_of(Actions, [&](const AreaSpillAction &A) {
      return A.Kind != AreaSpillAction::Memory ||
             Emitter->canSpill(VRegMaskPair(A.V, A.Lanes));
    });
    if (!Legal)
      continue;

    for (const AreaSpillAction &A : Actions) {
      assert(A.Kind == AreaSpillAction::Memory &&
             "post-color planner cannot recolor");
      assert(LIS->hasInterval(A.V) && !LIS->getInterval(A.V).empty() &&
             "planned victim lost its interval before set emission");
      LLVM_DEBUG(dbgs() << "    [AREA] spill " << printReg(A.V, TRI)
                        << " area=" << A.Area << " cost=" << A.Cost
                        << " lanes=" << PrintLaneMask(A.Lanes) << "\n");
      Emitter->spillOneVMP(VRegMaskPair(A.V, A.Lanes),
                           LIS->getInterval(A.V).beginIndex());
      ColorMap.erase(A.V);
      Spilled.insert(A.V);
      AnySpill = true;
    }

    Slots.clear();
    unsigned Actual = measureRegionPeak(R, nullptr, &Slots);
    LLVM_DEBUG(dbgs() << "    [AREA] committed set=" << Actions.size()
                      << " virtual-short=" << VirtualShort
                      << " actual-short=" << Actual << "\n");
  }

  LLVM_DEBUG(dbgs() << "region-rp: pass done, AnySpill=" << AnySpill << "\n");
  return AnySpill;
}

PhiWeb AMDGPUSSARegisterAllocator::closePhiWeb(Register Seed) const {
  SSARA_TRACE();
  PhiWeb Web;
  // Resolve the web root: Seed is a PHI result, or a PHI operand feeding one.
  // getUniqueVRegDef throughout (NOT getVRegDef, which asserts on multi-def): a
  // reload-created value has several redefs; treat it as a non-PHI ground/leaf.
  if (MachineInstr *D = MRI->getUniqueVRegDef(Seed); D && D->isPHI())
    Web.Root = Seed;
  else
    for (MachineInstr &U : MRI->use_nodbg_instructions(Seed))
      if (U.isPHI()) {
        Web.Root = U.getOperand(0).getReg();
        break;
      }
  if (!Web.Root)
    return Web; // invalid: Seed feeds no PHI

  // --- Close the equivalence class (analysis only; slot stays virtual). ---
  // The web equivalence result<->operand is BIDIRECTIONAL. Close it BOTH ways:
  //  - DOWN: a member PHI's operands (a PHI operand joins the web; a ground def is
  //    a store site).
  //  - UP: any PHI that USES the member as an operand (that consuming PHI is in the
  //    same class). Without the up-edge, a bb.N-Flow PHI reached as an operand of a
  //    bb.M-end PHI would be its OWN single-PHI web, and the edge to the end PHI
  //    would look EXTERNAL -> reloaded per-predecessor back into the join wall (the
  //    exact defect: web=1 everywhere, 128 reloads into RP=129 bb.1). Closing up
  //    makes that edge internal so it vanishes with the erased PHIs.
  SmallVector<Register, 32> Work;
  Web.PhiMembers.insert(Web.Root);
  Work.push_back(Web.Root);
  while (!Work.empty()) {
    Register R = Work.pop_back_val();
    MachineInstr *D = MRI->getUniqueVRegDef(R);
    if (D && D->isPHI()) {
      // DOWN: operands of this member PHI.
      for (unsigned I = 1, E = D->getNumOperands(); I + 1 < E; I += 2) {
        MachineOperand &Op = D->getOperand(I);
        if (!Op.isReg() || !Op.getReg().isVirtual())
          continue;
        // An UNDEF PHI operand (`undef %r.subN, %bb`) carries no live value on that
        // edge — %r is not live-out there. Storing it would emit a COPY reading an
        // undefined register (verifier: "reading vreg without a def"). Skip it: the
        // reload on that path reads whatever the slot holds, which is sound because
        // the incoming value was undef anyway.
        if (Op.isUndef())
          continue;
        Register OpReg = Op.getReg();
        MachineInstr *OpDef = MRI->getUniqueVRegDef(OpReg);
        if (OpDef && OpDef->isPHI()) {
          if (Web.PhiMembers.insert(OpReg))
            Work.push_back(OpReg);
        } else {
          // Ground operand on this PHI edge. The block operand is I+1. Record
          // EVERY edge (no dedup): only one edge executes at runtime, so each edge
          // carrying a web value must write the slot, even if the same vreg flows
          // on two edges. GroundOps stays unique for the interference gate/relief.
          MachineBasicBlock *PredBB = D->getOperand(I + 1).getMBB();
          Web.GroundOps.insert(OpReg);
          Web.GroundEdges.push_back({OpReg, Op.getSubReg(), PredBB, D, I});
        }
      }
    }
    // UP: any PHI consuming R as an operand joins the class.
    for (MachineInstr &U : MRI->use_nodbg_instructions(R)) {
      if (!U.isPHI())
        continue;
      Register UResult = U.getOperand(0).getReg();
      if (Web.PhiMembers.insert(UResult))
        Work.push_back(UResult);
    }
  }
  if (Web.GroundOps.empty()) {
    Web.Root = Register(); // invalid: all-undef web -> caller falls back
    return Web;
  }

  // Every non-undef ground value must be storable after its definition without
  // entering the block's terminator sequence. Decline before spillPhiWeb makes
  // any mutation when a legal store position does not exist.
  for (Register G : Web.GroundOps) {
    MachineInstr *DefMI = MRI->getUniqueVRegDef(G);
    if (DefMI && !DefMI->isImplicitDef() &&
        !AMDGPURegAllocInsertion::legalAfter(*DefMI)) {
      LLVM_DEBUG(dbgs() << "closePhiWeb(): DECLINE root "
                        << printReg(Web.Root, TRI) << " — ground "
                        << printReg(G, TRI)
                        << " is defined in a terminator sequence\n");
      Web.Root = Register();
      return Web;
    }
  }

  // SOUNDNESS GATE (option 2): all ground ops of the web share ONE stack slot. That
  // is only correct if no two of them are ever SIMULTANEOUSLY LIVE — i.e. the PHIs
  // are copy-less control-flow merges where exactly one incoming value reaches the
  // join on any path. If two operands interfere, the shared slot would clobber one
  // (the second store overwrites the first while the first is still needed) — a
  // MISCOMPILE. In-memory coalescing is register-coalescing with a slot as the
  // color, so it needs the SAME non-interference precondition. Prove it here; if any
  // pair interferes, DECLINE the web (invalidate) and let the caller fall back to a
  // plain per-value spill. (This is why 1024's divergent-select bitcast is safe: its
  // two operands come from mutually-exclusive cmp.true/cmp.false predecessors.)
  {
    SmallVector<Register, 16> GV(Web.GroundOps.begin(), Web.GroundOps.end());
    auto HasLive = [&](Register R) {
      return LIS->hasInterval(R) && !LIS->getInterval(R).empty();
    };
    for (unsigned A = 0; A < GV.size(); ++A)
      for (unsigned B = A + 1; B < GV.size(); ++B)
        if (HasLive(GV[A]) && HasLive(GV[B]) &&
            LIS->getInterval(GV[A]).overlaps(LIS->getInterval(GV[B]))) {
          LLVM_DEBUG(dbgs()
                     << "closePhiWeb(): DECLINE root " << printReg(Web.Root, TRI)
                     << " — operands " << printReg(GV[A], TRI) << " and "
                     << printReg(GV[B], TRI)
                     << " interfere (shared slot would clobber)\n");
          Web.Root = Register(); // invalid: shared slot would clobber
          return Web;
        }
  }
  return Web;
}

AMDGPUSSARegisterAllocator::RecoveryResult
AMDGPUSSARegisterAllocator::spillBlocker(Register Failed,
                                         Register &Remnant,
                                         RecoveryTransition &Transition) {
  SSARA_TRACE();
  const TargetRegisterClass *RC = MRI->getRegClass(Failed);
  const LiveInterval &FI = LIS->getInterval(Failed);
  SlotIndex FS = FI.beginIndex(), FE = FI.endIndex();
  bool IsVGPR = TRI->isVGPRClass(RC) || TRI->isAGPRClass(RC);

  // Candidate blockers from the SHARED overlapper scan. A clean candidate is a
  // colored value B whose physreg P is legal for Failed (same file), that is live
  // at F's END and has NO use strictly inside (FS,FE) — so B's reload lands past
  // FE and cannot round-trip into F's window. Two classes by where B STARTS:
  //  - LIVE-THROUGH (B.def <= FS, i.e. liveAt(FS)): freeing P clears ALL of F.
  //  - BORN-IN-F   (B.def in (FS,FE)):              freeing P clears F's TAIL.
  SmallVector<std::pair<Register, MCRegister>, 16> Overlappers;
  BitVector OccupiedUnits;
  scanOverlappersForVI(FI, OccupiedUnits, &Overlappers);

  PhiWeb WebBlocker;
  SmallVector<std::tuple<Register, MCRegister, bool>, 4> Cands; // (B, P, liveThru)
  unsigned NOverlap = 0, NClean = 0;
  for (const auto &[B, P] : Overlappers) {
    if (B == Failed || RecoverySpilledVRegs.count(B))
      continue;

    const LiveInterval &BI = LIS->getInterval(B);

    // A wider PHI-web tuple can block a narrower Failed value even though their
    // register classes have no common subclass. Redirect a live-through web to
    // the Web state when its tuple overlaps a register legal for Failed.
    bool BlocksFailed = false;
    for (MCRegister FP : RegClassInfo.getOrder(RC))
      if (TRI->regsOverlap(P, FP)) {
        BlocksFailed = true;
        break;
      }
    if (BlocksFailed && BI.liveAt(FS) && BI.liveAt(FE.getPrevSlot())) {
      PhiWeb Web = closePhiWeb(B);
      if (Web.valid() &&
          (!WebBlocker.valid() ||
           Web.Root.id() < WebBlocker.Root.id()))
        WebBlocker = std::move(Web);
    }

    // Same file: freeing P must open a Failed-legal register.
    const TargetRegisterClass *PRC = TRI->getPhysRegBaseClass(P);
    if (!TRI->getCommonSubClass(RC, PRC))
      continue;
    ++NOverlap;
    // Live at F's end covers BOTH classes (live-through: liveAt(FS)&&liveAt(FE);
    // born-in-F: B.end>FE => liveAt(FE.prev)).
    if (!BI.liveAt(FE.getPrevSlot()))
      continue;
    // B's reload must land PAST Failed's range, else it re-occupies the freed reg
    // inside [FS,FE) and Failed still cannot be placed. A use at slot U reloads
    // at R just BEFORE U (R < U), so a use at U == FE reloads at R with
    // FS < R < FE — INSIDE Failed's range. Therefore a use anywhere in (FS,FE]
    // (note the closed upper bound) disqualifies B, not just strictly inside.
    // (This is exactly the %206-vs-%208 case: %206 and Failed share their only
    // use at FE, so spilling %206 puts its reload where Failed is still live —
    // useless; %208's use is strictly past FE, so freeing it genuinely helps.)
    bool UsedInside = false;
    for (const MachineOperand &MO : MRI->use_operands(B)) {
      SlotIndex U = LIS->getInstructionIndex(*MO.getParent()).getRegSlot();
      if (FS < U && U <= FE) {
        UsedInside = true;
        break;
      }
    }
    if (UsedInside)
      continue;
    if (!Emitter->canSpill(
            VRegMaskPair(B, MRI->getMaxLaneMaskForVReg(B))))
      continue;
    ++NClean;
    Cands.emplace_back(B, P, BI.liveAt(FS));
  }

  LLVM_DEBUG(dbgs() << "  spill-blocker: " << printReg(Failed, TRI) << " ["
                    << FS << "," << FE << ") candidates: overlap=" << NOverlap
                    << " clean=" << NClean << "\n");

  if (Cands.empty() && WebBlocker.valid()) {
    LLVM_DEBUG(dbgs() << "  spill-blocker: transition to Web for "
                      << printReg(WebBlocker.Root, TRI) << ", then resume "
                      << printReg(Failed, TRI) << "\n");
    Transition.Next = {RecoveryState::Web, WebBlocker.Root};
    Transition.Resume = {RecoveryState::Blocker, Failed};
    return RecoveryResult::NoChange;
  }
  if (Cands.empty())
    return RecoveryResult::NoChange;

  // COVERAGE pick: a live-through blocker frees ALL of F; otherwise the born-in-F
  // blocker with the EARLIEST def frees the longest tail [B.def,FE). Blocker LI
  // LENGTH is irrelevant (the reload lands at the use).
  //
  // DETERMINISM: Cands is built in ColorMap (DenseMap) iteration order, which is
  // NOT stable — it depends on insertion history/rehashing. Picking the "first"
  // live-through candidate therefore made the choice depend on hash-map layout,
  // so an unrelated change to what/when ColorMap is populated (e.g. the SGPR
  // stage of the two-stage split) silently reordered candidates and picked a
  // different blocker — a spurious, input-order-dependent verdict. Choose by a
  // MEANINGFUL, stable key instead: live-through beats born-in-F (whole-F relief);
  // within born-in-F, earliest def; ties broken by vreg index. Result is
  // independent of ColorMap iteration order.
  Register B;
  MCRegister P;
  bool LiveThrough = false;
  SlotIndex BestDef;
  for (const auto &[CB, CP, CLive] : Cands) {
    if (LiveThrough && !CLive)
      continue; // a live-through pick already dominates any born-in-F
    if (CLive && !LiveThrough) {
      // First live-through seen — take it, then only a lower-index live-through
      // can replace it (deterministic tiebreak).
      B = CB;
      P = CP;
      LiveThrough = true;
      continue;
    }
    if (CLive) { // both live-through: lowest vreg index wins
      if (CB < B) {
        B = CB;
        P = CP;
      }
      continue;
    }
    // born-in-F (only reachable while no live-through found): earliest def, then
    // lowest vreg index.
    SlotIndex D = LIS->getInterval(CB).beginIndex();
    if (!B || D < BestDef || (D == BestDef && CB < B)) {
      B = CB;
      P = CP;
      BestDef = D;
    }
  }

  auto ColorInPlace = [&](Register R) -> bool {
    if (!R.isVirtual() || !LIS->hasInterval(R) || ColorMap.count(R) ||
        MRI->reg_nodbg_empty(R))
      return true;
    return colorOneInPlace(R);
  };

  if (LiveThrough) {
    LLVM_DEBUG(dbgs() << "  spill-blocker: live-through " << printReg(B, TRI)
                      << " (phys " << TRI->getName(P) << ") across "
                      << printReg(Failed, TRI) << "\n");
    // Store at B's def, reload at B's post-FE uses -> P free over all of FI.
    // spillOneVMP replaces B's long range with a head stub + narrow reload
    // redefs; recolor each surviving piece (forcing P onto a narrow reload is
    // unsound for a wide B). Freeing B's units opens the lane for Failed.
    RecoverySpilledVRegs.insert(B);
    Emitter->beginPass(IsVGPR, demandFor(poolOf(MRI->getRegClass(B))));
    ColorMap.erase(B);
    Emitter->spillOneVMP(VRegMaskPair(B, MRI->getMaxLaneMaskForVReg(B)), FS);
    // The erase and the spill are COMMITTED to LIS and MIR — there is no
    // rollback. Every value this uncoloured or created must therefore end up
    // either coloured or ON THE WORKLIST; a piece that is merely dropped reaches
    // final rewrite as a live virtual register, which is the emit-time abort
    // "non-undef virtual register not colored". The born-in-F branch below has
    // always worked this way; this one returned NoOp and dropped the pieces.
    // Observed on tahiti bitcast_v64bf16_to_v128i8_scalar: %2760 was evicted
    // here, declined to recolour, and asserted in rewriteOperands.
    auto ColorOrQueue = [&](Register R) {
      if (!ColorInPlace(R))
        UncolorableVRegs.push_back(R);
    };
    ColorOrQueue(B); // surviving head stub
    for (const VRegMaskPair &VMP : Emitter->reloadedRegs())
      ColorOrQueue(VMP.getVReg()); // fresh reload redefs
    // The verdict is about Failed ALONE. Reporting NoOp because a piece of the
    // collateral needs another round sent the caller on to its next recovery for
    // a value that is already placed, which then re-split and re-coloured it —
    // %2965 went to VGPR43 here and was overwritten with VGPR46 by the self-split
    // that followed. The collateral is on the worklist and does not need the
    // caller's help.
    if (!ColorInPlace(Failed)) {
      LLVM_DEBUG(dbgs() << "  spill-blocker: " << printReg(Failed, TRI)
                        << " stayed uncolorable after evicting "
                        << printReg(B, TRI) << "\n");
      Remnant = Failed;
      return RecoveryResult::Changed;
    }
    return RecoveryResult::Resolved;
  }

  // BORN-IN-F: freeing P clears the TAIL [B.def,FE). colorOneInPlace is
  // whole-range, so split F at B.def: color the tail into B's freed reg and hand
  // the HEAD [FS,B.def) back as the (strictly shorter) remnant -> Reduced.
  // A PHI-defined B has no mid-block def to split F at (splitLiveRangeAt needs a
  // real instruction); decline cleanly rather than mis-place the cut.
  MachineInstr *BDef = MRI->getVRegDef(B);
  assert(BDef && "colored blocker must have a def in SSA");
  if (BDef->isPHI())
    return RecoveryResult::NoChange;

  // CLEAN-CUT GATE. splitLiveRangeAt redirects uses by REACHING-VNI, not by slot:
  // it moves only the uses that read the VNI live just before the cut. F may be
  // MULTI-VNI (product of earlier splits); if a use at/after B.def reads a
  // DIFFERENT VNI, that use stays on F, F's endpoint does NOT move, and F is left
  // half-transformed and uncolored (an emit-time abort). Take born-in-F ONLY when
  // the cut is clean: every non-debug use of F at/after B.def reads the single VNI
  // live at B.def (and none is a PHI use — a value merge is inherently multi-VNI).
  // Then all tail uses redirect and F truly shrinks to [FS,B.def). Otherwise
  // decline (NoOp) and let SelfSplit/Floor handle F.
  SlotIndex BDefSlot = LIS->getInstructionIndex(*BDef).getRegSlot();
  VNInfo *CutVNI = FI.getVNInfoBefore(BDefSlot);
  if (!CutVNI)
    return RecoveryResult::NoChange;
  for (const MachineOperand &MO : MRI->use_operands(Failed)) {
    const MachineInstr *UMI = MO.getParent();
    if (UMI->isDebugInstr())
      continue;
    SlotIndex U = LIS->getInstructionIndex(*UMI).getRegSlot();
    if (U < BDefSlot)
      continue; // head use: keeps reading F after the cut — fine
    if (UMI->isPHI() || FI.getVNInfoBefore(U) != CutVNI)
      return RecoveryResult::NoChange; // tail use won't redirect -> cut not clean
  }

  LLVM_DEBUG(dbgs() << "  spill-blocker: born-in-F " << printReg(B, TRI)
                    << " (phys " << TRI->getName(P) << ") frees tail of "
                    << printReg(Failed, TRI) << " at " << BestDef << "\n");
  if (!splitWouldRedirect(Failed, BDef))
    return RecoveryResult::NoChange;
  Register Tail = Emitter->splitLiveRangeAt(Failed, BDef->getIterator());
  assert(Tail && "split preflight must guarantee a redirected use");
  RecoverySpilledVRegs.insert(B);
  Emitter->beginPass(IsVGPR, demandFor(poolOf(MRI->getRegClass(B))));
  ColorMap.erase(B);
  Emitter->spillOneVMP(VRegMaskPair(B, MRI->getMaxLaneMaskForVReg(B)),
                       LIS->getInterval(B).beginIndex());
  // Split + spill are now COMMITTED to LIS/MIR — there is no rollback. Any piece
  // that cannot color in place must be RE-QUEUED (dropping it leaks an uncolored
  // vreg to final rewrite = the emit-time abort); NoOp here would strand exactly
  // those pieces. Color what we can, queue the rest for the worklist fixpoint.
  // The pieces to place: B's SURVIVING range, B's reload redefs, and F's tail.
  // B is NOT always fully consumed — kill-at-def only removes B's range past its
  // def, but when B is itself a short split-Tail whose def is a COPY, a tiny stub
  // survives (def + store) that we erased from ColorMap and must re-place. So run
  // B through the same guarded path: it colors if a stub survives, skips if truly
  // consumed. (The live-through path kills at FS > B.def, always leaving a stub.)
  // Skip-guard the members: reloadedRegs() is CUMULATIVE across every spill in
  // the pass (beginPass does not clear it, only clearReloadedRegs does, which no
  // caller invokes per-spill), so it also contains STALE reloads from earlier
  // spills that are already colored / dead / interval-less. Skip those; only a
  // fresh, live, uncolored member that colorOneInPlace declines is a real queue.
  // TODO(honest fix): call Emitter->clearReloadedRegs() before spillOneVMP so
  // reloadedRegs() returns ONLY this spill's redefs; then the reload loop's skip
  // guard collapses to a strict assert (every member fresh+live). Left for its own
  // change — clearing is shared emitter state read by the driver.
  auto ColorOrQueue = [&](Register R) {
    if (!R.isVirtual() || !LIS->hasInterval(R) || ColorMap.count(R) ||
        MRI->reg_nodbg_empty(R))
      return;
    if (!colorOneInPlace(R))
      UncolorableVRegs.push_back(R);
  };
  ColorOrQueue(B); // B's surviving stub, if kill-at-def did not consume it
  for (const VRegMaskPair &VMP : Emitter->reloadedRegs())
    ColorOrQueue(VMP.getVReg()); // this + prior spills' reload redefs (cumulative)
  ColorOrQueue(Tail);            // F's tail into B's freed register
  // Clean-cut gate guarantees F's endpoint moved to B.def, so the head remnant is
  // strictly shorter -> hand it back for re-dispatch.
  Remnant = Failed;
  return RecoveryResult::Changed;
}

void AMDGPUSSARegisterAllocator::commitColor(Register Piece, MCRegister PR) {
  SSARA_TRACE();
  ColorMap[Piece] = PR;
  unsigned Idx = TRI->getHWRegIndex(PR);
  unsigned W = TRI->getRegSizeInBits(*MRI->getRegClass(Piece)) / 32;
  const TargetRegisterClass *PhysRC = TRI->getPhysRegBaseClass(PR);
  if (TRI->isVGPRClass(PhysRC))
    MaxVGPRIdx = std::max(MaxVGPRIdx, Idx + W);
  else if (TRI->isAGPRClass(PhysRC))
    MaxAGPRIdx = std::max(MaxAGPRIdx, Idx + W);
  else if (TRI->isSGPRClass(PhysRC))
    MaxSGPRIdx = std::max(MaxSGPRIdx, Idx + W);
}

SlotIndex AMDGPUSSARegisterAllocator::firstBlockAfter(
    MCRegister PR, SlotIndex S, SlotIndex End,
    ArrayRef<std::pair<Register, MCRegister>> Overlappers) const {
  SSARA_TRACE();
  // Call-clobber: a call in (S,End) clobbering PR bounds the free run there.
  SlotIndex Best = End;
  for (const auto &[CallIdx, CallMI] : CallSites) {
    if (CallIdx <= S || End <= CallIdx)
      continue;
    bool Clob = CallMI->modifiesRegister(PR, TRI);
    if (!Clob)
      for (const MachineOperand &MO : CallMI->operands())
        if (MO.isRegMask() && MO.clobbersPhysReg(PR)) {
          Clob = true;
          break;
        }
    if (Clob && CallIdx < Best)
      Best = CallIdx;
  }
  for (const auto &[WReg, WPhys] : Overlappers) {
    bool Touches = false;
    for (MCRegUnit WU : TRI->regunits(WPhys)) {
      for (MCRegUnit PU : TRI->regunits(PR))
        if (WU == PU) {
          Touches = true;
          break;
        }
      if (Touches)
        break;
    }
    if (!Touches)
      continue;
    const LiveInterval &WI = LIS->getInterval(WReg);
    for (const LiveRange::Segment &Seg : WI.segments) {
      if (Seg.end <= S)
        continue; // entirely before the piece start
      if (Seg.start <= S)
        return S; // occupied AT S -> PR not free here
      if (Seg.start < Best)
        Best = Seg.start; // first block after S
      break;              // segments are sorted; earliest found
    }
  }
  return Best;
}

bool AMDGPUSSARegisterAllocator::splitWouldRedirect(
    Register V, MachineInstr *SplitMI) const {
  SSARA_TRACE();
  if (!LIS->hasInterval(V))
    return false;
  MachineBasicBlock &MBB = *SplitMI->getParent();
  auto LegalPos =
      AMDGPURegAllocInsertion::legalBefore(MBB, SplitMI->getIterator());
  if (LegalPos == MBB.end())
    return false;
  MachineInstr *LegalAnchor = &*LegalPos;
  const LiveInterval &LI = LIS->getInterval(V);
  SlotIndex SplitSlot = LIS->getInstructionIndex(*LegalAnchor).getRegSlot();
  VNInfo *SrcVNI = LI.getVNInfoBefore(SplitSlot);
  if (!SrcVNI)
    return false;

  // Match SSASpillEmitter::splitLiveRangeAt exactly. A requested prologue or
  // terminator-sequence position is first clamped into the legal block body.
  for (const MachineOperand &MO : MRI->use_operands(V)) {
    const MachineInstr *UseMI = MO.getParent();
    if (UseMI->isDebugInstr() || isSpillInstr(UseMI) ||
        !MDT->dominates(LegalAnchor, UseMI))
      continue;
    SlotIndex UseSlot = LIS->getInstructionIndex(*UseMI).getRegSlot();
    if (LI.getVNInfoBefore(UseSlot) == SrcVNI)
      return true;
  }
  return false;
}

bool AMDGPUSSARegisterAllocator::pickPeelableRun(Register V, MCRegister &PR,
                                                 SlotIndex &Bound) const {
  SSARA_TRACE();
  // THE single split-across policy; see the header for why both callers share it.
  PR = MCRegister();
  Bound = SlotIndex();
  if (!LIS->hasInterval(V))
    return false;
  const LiveInterval &CI = LIS->getInterval(V);
  const TargetRegisterClass *RC = MRI->getRegClass(V);
  const SlotIndex S = CI.beginIndex(), E = CI.endIndex();

  SmallVector<std::pair<Register, MCRegister>, 16> Overlappers;
  BitVector Occ;
  scanOverlappersForVI(CI, Occ, &Overlappers);

  // Pick the available PR free at S that stays free the LONGEST (fewest future
  // splits). availableOrder(RC) already yields RC-width PRs and excludes the
  // tail reserved for downstream SGPR spill lowering and WWM allocation. PR
  // default-constructs to NoRegister (!PR tests it via MCRegister's unsigned
  // conversion).
  SlotIndex Best = S;
  for (MCRegister P : availableOrder(RC)) {
    SlotIndex B = firstBlockAfter(P, S, E, Overlappers);
    if (B <= S)
      continue; // not free at S
    if (!PR || Best < B) {
      PR = P;
      Best = B;
    }
  }
  if (!PR) {
    LLVM_DEBUG(dbgs() << "  peelable-run: " << printReg(V, TRI)
                      << " NONE (no free reg at " << S << ")\n");
    return false;
  }

  // A free run that does not even reach the next use is genuine over-pressure,
  // not fragmentation — peeling there would grind the region into use-less
  // confetti. A run covering all of V needs no such check.
  if (Best < E) {
    SlotIndex FirstUse;
    for (MachineInstr &U : MRI->use_nodbg_instructions(V)) {
      SlotIndex US = LIS->getInstructionIndex(U).getRegSlot();
      if (US > S && (!FirstUse.isValid() || US < FirstUse))
        FirstUse = US;
    }
    if (FirstUse.isValid() && Best <= FirstUse) {
      LLVM_DEBUG(dbgs() << "  peelable-run: " << printReg(V, TRI)
                        << " NONE (run [" << S << "," << Best
                        << ") does not reach first use " << FirstUse << ")\n");
      PR = MCRegister();
      return false;
    }
  }
  Bound = Best;
  LLVM_DEBUG(dbgs() << "  peelable-run: " << printReg(V, TRI) << " -> "
                    << TRI->getName(PR) << " [" << S << "," << Bound << ")\n");
  return true;
}

AMDGPUSSARegisterAllocator::RecoveryResult
AMDGPUSSARegisterAllocator::trySelfSplitColor(Register Failed, MCRegister FirstPR,
                                              SlotIndex FirstBound,
                                              Register &Remnant) {
  SSARA_TRACE();
  // SELF-SPLIT — Failed IS ITSELF the long liver: no single PR is free across its
  // whole range, and no separate live-through blocker exists to spill around.
  // Keep as much of Failed register-resident as possible: repeatedly peel off the
  // maximal PREFIX that some PR is free across, color that piece into that PR, and
  // recurse on the tail. Each peel = one splitLiveRangeAt (a COPY + reaching-VNI
  // use redirect, staying in SSA -> graph stays chordal -> Hack-compatible).
  //
  // Outcomes (NO memory-spill here — the Floor owns that):
  //  - Resolved : every piece colored.
  //  - Changed  : peeled >=1 prefix but a piece could not settle in a register;
  //               the SHORTER remnant is returned in \p Remnant for the driver to
  //               restart recovery (a shorter range may expose a clean cross-liver;
  //               deferring via the worklist could close that opportunity).
  //  - NoChange : could not peel anything (the first piece == whole Failed is
  //               already tight) -> driver floors the original Failed.
  // A piece that cannot settle in a register is NOT memory-spilled here (the Floor
  // owns memory-spilling). If nothing has been peeled yet it is NoChange (driver
  // floors the original Failed); if >=1 prefix was peeled the shorter remnant is
  // handed back as Changed so the pipeline restarts from Web.
  auto handBack = [&](Register Cur, unsigned Pieces) -> RecoveryResult {
    if (Pieces == 0)
      return RecoveryResult::NoChange;
    Remnant = Cur;
    return RecoveryResult::Changed;
  };

  Register Cur = Failed;
  unsigned Pieces = 0;
  const unsigned MaxPieces = 64; // runaway guard
  while (Pieces < MaxPieces) {
    // Cur is either Failed (which color() must hand over WITH an interval) or a
    // splitLiveRangeAt Tail (which computes its interval). A missing interval here
    // is malformed input / a broken split — a real bug, not a normal exit.
    assert(LIS->hasInterval(Cur) &&
           "self-split: piece lost its live interval (malformed split/input)");
    const LiveInterval &CI = LIS->getInterval(Cur);
    SlotIndex S = CI.beginIndex(), E = CI.endIndex();

    // FIRST piece: consume the pick the FSM already made when it routed here.
    // Later pieces are DIFFERENT intervals (fresh tails, previous piece now
    // colored), so they need their own pick — not a recomputation of the same
    // question. Either way the policy lives only in pickPeelableRun.
    MCRegister BestPR = (Pieces == 0) ? FirstPR : MCRegister();
    SlotIndex BestBound = (Pieces == 0) ? FirstBound : SlotIndex();
    if (!BestPR && !pickPeelableRun(Cur, BestPR, BestBound)) {
      LLVM_DEBUG(dbgs() << "  self-split: piece " << printReg(Cur, TRI)
                        << " cannot be placed -> hand back\n");
      return handBack(Cur, Pieces);
    }

    if (BestBound >= E) {
      // BestPR is free across the whole remaining piece: color it, done.
      commitColor(Cur, BestPR);
      LLVM_DEBUG(dbgs() << "  self-split: colored final piece "
                        << printReg(Cur, TRI) << " -> " << TRI->getName(BestPR)
                        << " (" << (Pieces + 1) << " pieces total)\n");
      return RecoveryResult::Resolved;
    }

    // Split at the boundary (where BestPR becomes occupied). splitLiveRangeAt
    // needs a clean non-PHI, mid-block instruction; if the boundary lands on a
    // PHI/gap, back up to the nearest earlier real instruction in (S, BestBound)
    // (the head is still a prefix of the free run). If none exists, hand back.
    MachineInstr *SplitMI = LIS->getInstructionFromIndex(BestBound);
    SlotIndex Probe = BestBound;
    while ((!SplitMI || SplitMI->isPHI() || SplitMI->isDebugInstr()) &&
           Probe > S) {
      Probe = Probe.getPrevIndex();
      SplitMI = LIS->getInstructionFromIndex(Probe);
    }
    if (!SplitMI || SplitMI->isPHI() || SplitMI->isDebugInstr() ||
        LIS->getInstructionIndex(*SplitMI).getRegSlot() <= S) {
      LLVM_DEBUG(dbgs() << "  self-split: no clean split point for piece "
                        << printReg(Cur, TRI) << " -> hand back\n");
      return handBack(Cur, Pieces);
    }
    if (!splitWouldRedirect(Cur, SplitMI))
      return handBack(Cur, Pieces);
    Register Tail = Emitter->splitLiveRangeAt(Cur, SplitMI->getIterator());
    assert(Tail && "split preflight must guarantee a redirected use");
    // Head piece (Cur, now [S,BestBound)) is free on BestPR: color it.
    commitColor(Cur, BestPR);
    LLVM_DEBUG(dbgs() << "  self-split: peeled piece " << printReg(Cur, TRI)
                      << " -> " << TRI->getName(BestPR) << " [" << S << ","
                      << BestBound << "), recurse on tail "
                      << printReg(Tail, TRI) << "\n");
    Cur = Tail;
    ++Pieces;
  }
  // Hit the piece cap mid-split: hand back the remainder (Pieces>0 here).
  LLVM_DEBUG(dbgs() << "  self-split: piece cap -> hand back remainder "
                    << printReg(Cur, TRI) << "\n");
  return handBack(Cur, Pieces);
}

AMDGPUSSARegisterAllocator::RecoveryResult
AMDGPUSSARegisterAllocator::tryCrossFileHome(Register R) {
  SSARA_TRACE();
  if (!ST->hasMAIInsts())
    return RecoveryResult::NoChange;

  auto TryOne = [&](Register V,
                    const TargetRegisterClass *ForcedTarget) -> bool {
    const TargetRegisterClass *RC = MRI->getRegClass(V);
    if ((!TRI->hasVGPRs(RC) && !TRI->isAGPRClass(RC)) ||
        !LIS->hasInterval(V) || ColorMap.count(V))
      return false;

    // A forced target is used for a temporarily uncolored crosser. Do not let it
    // reclaim the old physical home before probing the sibling pool.
    if (!ForcedTarget && colorOneInPlace(V))
      return true;
    if (RescueCopies.count(V) || RehomedVRegs.count(V))
      return false;

    const TargetRegisterClass *Home = ForcedTarget;
    if (!Home) {
      const bool ToAGPR = !TRI->isAGPRClass(RC);
      Home = ToAGPR ? TRI->getEquivalentAGPRClass(RC)
                    : TRI->getEquivalentVGPRClass(RC);
    }
    if (!Home)
      return false;
    const bool ToAGPR = TRI->isAGPRClass(Home);

    auto AcceptsHome = [&](const MachineOperand &MO) {
      const MachineInstr *MI = MO.getParent();
      if (MI->isPHI())
        return true;
      const TargetRegisterClass *OpRC =
          TII->getRegClass(MI->getDesc(), MO.getOperandNo(), TRI);
      return !OpRC || TRI->getCommonSubClass(Home, OpRC) != nullptr;
    };

    MachineOperand *SplitDef = nullptr;
    for (MachineOperand &MO : MRI->def_operands(V)) {
      if (AcceptsHome(MO))
        continue;
      if (SplitDef || MO.getSubReg() || MO.isTied())
        return false;
      SplitDef = &MO;
    }

    SmallVector<MachineOperand *, 8> NeedsCopy;
    for (MachineOperand &MO : MRI->use_operands(V)) {
      if (MO.getSubReg())
        return false;
      if (MO.getParent()->isPHI())
        continue;
      if (!AcceptsHome(MO))
        NeedsCopy.push_back(&MO);
    }

    std::optional<MachineBasicBlock::iterator> SplitDefInsert;
    if (SplitDef) {
      SplitDefInsert =
          AMDGPURegAllocInsertion::legalAfter(*SplitDef->getParent());
      if (!SplitDefInsert)
        return false;
    }

    const TargetRegisterClass *SavedRC = RC;
    MRI->setRegClass(V, Home);
    if (!colorOneInPlace(V)) {
      MRI->setRegClass(V, SavedRC);
      return false;
    }

    if (SplitDef) {
      MachineInstr *DefMI = SplitDef->getParent();
      Register Tmp = MRI->createVirtualRegister(SavedRC);
      SplitDef->setReg(Tmp);
      MachineInstr *Copy =
          BuildMI(*DefMI->getParent(), *SplitDefInsert,
                  DefMI->getDebugLoc(), TII->get(TargetOpcode::COPY), V)
              .addReg(Tmp);
      LIS->InsertMachineInstrInMaps(*Copy);
      LIS->createAndComputeVirtRegInterval(Tmp);
      RescueCopies.insert(Tmp);
      if (!colorOneInPlace(Tmp))
        UncolorableVRegs.push_back(Tmp);
    }

    for (MachineOperand *MO : NeedsCopy) {
      MachineInstr *MI = MO->getParent();
      Register Tmp = MRI->createVirtualRegister(SavedRC);
      auto InsertPt = AMDGPURegAllocInsertion::legalBefore(
          *MI->getParent(), MI->getIterator());
      MachineInstr *Copy =
          BuildMI(*MI->getParent(), InsertPt, MI->getDebugLoc(),
                  TII->get(TargetOpcode::COPY), Tmp)
              .addReg(V);
      LIS->InsertMachineInstrInMaps(*Copy);
      MO->setReg(Tmp);
      LIS->createAndComputeVirtRegInterval(Tmp);
      RescueCopies.insert(Tmp);
      if (!colorOneInPlace(Tmp))
        UncolorableVRegs.push_back(Tmp);
    }

    LIS->removeInterval(V);
    LIS->createAndComputeVirtRegInterval(V);
    RehomedVRegs.insert(V);
    LLVM_DEBUG(dbgs() << "  [cross-file-home] " << printReg(V, TRI) << " -> "
                      << TRI->getName(ColorMap.lookup(V)) << " ("
                      << (ToAGPR ? "VGPR->AGPR" : "AGPR->VGPR") << "), "
                      << NeedsCopy.size() << " copies back\n");
    if (SSAForensicReporter::enabled())
      Reporter->transformation(ToAGPR ? "cross-file-home-to-agpr"
                                      : "cross-file-home-to-vgpr",
                               V.virtRegIndex());
    return true;
  };

  // recoverUncolorable only dispatches live values. A failed TryOne preserves
  // this interval, so there is no meaningful post-probe missing-interval case.
  assert(LIS->hasInterval(R) &&
         "cross-file recovery requires an existing live interval");
  // First try the failed/uncolored current value itself.
  if (TryOne(R, nullptr))
    return RecoveryResult::Resolved;

  // Then move one colored crosser out of an actual physical pool Failed can use.
  // availableOrder is the exact set of legal candidate registers for Failed's
  // class (including target restrictions and reserved-register filtering), so
  // pool admission depends only on whether that set reaches the crosser's source
  // pool. Crosser width/class is irrelevant: moving it frees its actual lanes.
  const TargetRegisterClass *FailedRC = MRI->getRegClass(R);
  bool FailedCanUseVGPR = false;
  bool FailedCanUseAGPR = false;
  for (MCPhysReg Candidate : availableOrder(FailedRC)) {
    const TargetRegisterClass *CandidateRC =
        TRI->getPhysRegBaseClass(MCRegister(Candidate));
    FailedCanUseVGPR |= CandidateRC && TRI->isVGPRClass(CandidateRC);
    FailedCanUseAGPR |= CandidateRC && TRI->isAGPRClass(CandidateRC);
  }

  SmallVector<std::pair<Register, MCRegister>, 16> Crossers;
  BitVector OccupiedUnits;
  scanOverlappersForVI(LIS->getInterval(R), OccupiedUnits, &Crossers);
  llvm::sort(Crossers, [](const auto &A, const auto &B) {
    return A.first.id() < B.first.id();
  });
  for (const auto &[Crosser, OldPR] : Crossers) {
    if (Crosser == R || RescueCopies.count(Crosser) ||
        RehomedVRegs.count(Crosser))
      continue;
    const TargetRegisterClass *OldPhysRC = TRI->getPhysRegBaseClass(OldPR);
    if (!OldPhysRC)
      continue;
    const bool FromAGPR = TRI->isAGPRClass(OldPhysRC);
    const bool FromVGPR = TRI->isVGPRClass(OldPhysRC);
    if ((!FromAGPR && !FromVGPR) ||
        (FromAGPR && !FailedCanUseAGPR) ||
        (FromVGPR && !FailedCanUseVGPR))
      continue;

    const TargetRegisterClass *SavedRC = MRI->getRegClass(Crosser);
    const TargetRegisterClass *Sibling =
        FromAGPR ? TRI->getEquivalentVGPRClass(SavedRC)
                 : TRI->getEquivalentAGPRClass(SavedRC);
    if (!Sibling)
      continue;

    ColorMap.erase(Crosser);
    if (TryOne(Crosser, Sibling))
      return RecoveryResult::Changed;

    // No MIR is inserted before the forced color succeeds, so this restores the
    // exact pre-strategy state after a failed probe.
    MRI->setRegClass(Crosser, SavedRC);
    ColorMap[Crosser] = OldPR;
  }
  return RecoveryResult::NoChange;
}

void AMDGPUSSARegisterAllocator::reportPointOverPressure(Register R,
                                                         bool IsVGPR,
                                                         unsigned RPLimit,
                                                         const char *Ctx) {
  SSARA_TRACE();
  // Honest terminal for an unrecoverable memory floor. Find the point in R's range
  // with the most simultaneously-live dwords of R's register file and report the
  // REAL numbers. If that peak exceeds RPLimit no coloring-time recovery exists
  // (more values demand registers at one instant than the file has); otherwise
  // the point is feasible yet unrecovered, which is a genuine allocator bug —
  // say so, rather than the misleading "needs more up-front spilling".
  const LiveInterval &RI = LIS->getInterval(R);
  const RegFile Pool = poolOf(MRI->getRegClass(R));
  SlotIndex PeakSlot = RI.beginIndex();
  unsigned Short = 0, PeakDwords = 0;
  // Walk every real instruction slot in R's range and ask the oracle whether the
  // values live there can all be placed. R's range is short (a reload remainder),
  // so this is cheap.
  for (SlotIndex SI = RI.beginIndex(); SI < RI.endIndex();
       SI = Indexes->getNextNonNullIndex(SI)) {
    if (!Indexes->getInstructionFromIndex(SI))
      continue;
    GCNRPTracker::LiveRegSet Live;
    unsigned Dwords = 0;
    for (unsigned I = 0, E = MRI->getNumVirtRegs(); I < E; ++I) {
      Register V = Register::index2VirtReg(I);
      if (MRI->reg_nodbg_empty(V) || !LIS->hasInterval(V))
        continue;
      const LiveInterval &VI = LIS->getInterval(V);
      if (VI.empty() || !VI.liveAt(SI))
        continue;
      const TargetRegisterClass *VRC = MRI->getRegClass(V);
      Live[V.id()] = MRI->getMaxLaneMaskForVReg(V); // the oracle reads the class
      if (poolOf(VRC) == Pool)
        Dwords += TRI->getRegSizeInBits(*VRC) / 32;
    }
    SmallVector<TierDemand, 8> Tiers;
    unsigned S = demandDeficiency(Live, Pool, Tiers);
    if (S > Short) {
      Short = S;
      PeakDwords = Dwords;
      PeakSlot = SI;
    }
  }

  std::string Msg;
  raw_string_ostream OS(Msg);
  OS << "SSARA recursive-recovery [" << Ctx << "]: cannot place "
     << printReg(R, TRI) << " (" << (IsVGPR ? "VGPR" : "SGPR") << " file). ";
  if (Short)
    OS << "GENUINE POINT-OVER-PRESSURE: " << Short << " value(s) at " << PeakSlot
       << " have no legal placement (" << PeakDwords << " dwords live, " << RPLimit
       << " registers) — no coloring-time recovery can fit them.";
  else
    OS << "FEASIBLE YET UNRECOVERED (allocator bug): every value at " << PeakSlot
       << " has a legal placement (" << PeakDwords << " dwords, " << RPLimit
       << " registers), so recovery did not find one that exists.";
  if (SSAForensicReporter::enabled())
    Reporter->flushNow();
  report_fatal_error(StringRef(Msg));
}

bool AMDGPUSSARegisterAllocator::recoverUncolorable(Register Failed) {
  SSARA_TRACE();
  // Explicit recovery FSM. Each irreversible interference change transitions
  // back to Web so every later predicate observes current MIR/LIS/ColorMap.
  // Termination comes from monotone strategy-local measures: a web is erased
  // once, a value is re-homed at most once, a blocker is spilled at most once,
  // and SelfSplit hands back a strictly shorter remnant.
  const MachineFunction &MF = MRI->getMF();

  LLVM_DEBUG(dbgs() << "FALLBACK for " << printReg(Failed, TRI) << " ["
                    << LIS->getInterval(Failed).beginIndex() << ","
                    << LIS->getInterval(Failed).endIndex() << ")\n");

  auto ColorInPlace = [&](Register R) {
    if (!R.isVirtual() || !LIS->hasInterval(R) || ColorMap.count(R) ||
        MRI->reg_nodbg_empty(R))
      return;
    if (SSAForensicReporter::enabled())
      Reporter->flushNow();
    if (!colorOneInPlace(R))
      UncolorableVRegs.push_back(R);
  };
  auto CurLen = [&](Register R) {
    return LIS->hasInterval(R)
               ? LIS->getInterval(R).beginIndex().distance(
                     LIS->getInterval(R).endIndex())
               : 0u;
  };

  RecoveryFrame Current{RecoveryState::Entry, Failed};
  SmallVector<RecoveryFrame, 2> Continuations;
  auto CompleteCurrent = [&]() {
    if (Continuations.empty())
      return true;
    Current = Continuations.pop_back_val();
    return false;
  };
  auto Dispatch = [&](const RecoveryTransition &Transition) {
    assert(Transition.valid() && "recovery transition requires a target");
    if (Transition.Resume.valid())
      Continuations.push_back(Transition.Resume);
    Current = Transition.Next;
  };

  while (true) {
    Register Cur = Current.Value;
    switch (Current.State) {
    case RecoveryState::Entry:
      Current.State = RecoveryState::Web;
      [[fallthrough]];

    case RecoveryState::Web: {
      PhiWeb Web = closePhiWeb(Cur);
      if (!Web.valid()) {
        Current.State = RecoveryState::CrossFileHome;
        continue;
      }

      const TargetRegisterClass *RC = MRI->getRegClass(Cur);
      bool IsVGPR = TRI->isVGPRClass(RC) || TRI->isAGPRClass(RC);
      Emitter->beginPass(IsVGPR, demandFor(poolOf(RC)));
      auto CFV = [&](Register C) {
        if (C.isVirtual() && LIS->hasInterval(C) && !ColorMap.count(C))
          colorOneInPlace(C);
      };
      Emitter->spillPhiWeb(Web, CFV);
      if (SSAForensicReporter::enabled())
        Reporter->transformation("phi-web-spill", Web.Root.virtRegIndex());
      for (Register G : Emitter->lastWebGround())
        ColorInPlace(G);
      for (const VRegMaskPair &VMP : Emitter->reloadedRegs())
        ColorInPlace(VMP.getVReg());
      if (CompleteCurrent())
        return true;
      continue;
    }

    case RecoveryState::CrossFileHome: {
      RecoveryResult Home = tryCrossFileHome(Cur);
      if (Home == RecoveryResult::Resolved) {
        if (CompleteCurrent())
          return true;
        continue;
      }
      if (Home == RecoveryResult::Changed) {
        Current.State = RecoveryState::Entry;
        continue;
      }
      assert(Home == RecoveryResult::NoChange);
      Current.State = RecoveryState::Blocker;
      continue;
    }

    case RecoveryState::Blocker: {
      Register Remnant;
      RecoveryTransition Transition;
      RecoveryResult Blocker = spillBlocker(Cur, Remnant, Transition);
      if (Transition.valid()) {
        Dispatch(Transition);
        continue;
      }
      if (Blocker == RecoveryResult::Resolved) {
        if (SSAForensicReporter::enabled())
          Reporter->transformation("spill-blocker", Cur.virtRegIndex());
        if (CompleteCurrent())
          return true;
        continue;
      }
      if (Blocker == RecoveryResult::Changed) {
        assert(Remnant && "changed blocker spill must return a remnant");
        if (SSAForensicReporter::enabled())
          Reporter->transformation("spill-blocker", Cur.virtRegIndex());
        Current = {RecoveryState::Entry, Remnant};
        continue;
      }
      assert(Blocker == RecoveryResult::NoChange);
      Current.State = RecoveryState::SelfSplit;
      continue;
    }

    case RecoveryState::SelfSplit: {
      const unsigned LenBefore = CurLen(Cur);
      Register Remnant;
      RecoveryResult Split =
          trySelfSplitColor(Cur, MCRegister(), SlotIndex(), Remnant);
      if (Split == RecoveryResult::Resolved) {
        if (SSAForensicReporter::enabled())
          Reporter->transformation("self-split", Cur.virtRegIndex());
        if (CompleteCurrent())
          return true;
        continue;
      }
      if (Split == RecoveryResult::Changed) {
        assert(Remnant && CurLen(Remnant) < LenBefore &&
               "changed SelfSplit must return a strictly shorter remnant");
        Current = {RecoveryState::Entry, Remnant};
        continue;
      }
      assert(Split == RecoveryResult::NoChange);
      Current.State = RecoveryState::Floor;
      continue;
    }

    case RecoveryState::Floor: {
      const TargetRegisterClass *RC = MRI->getRegClass(Cur);
      bool IsVGPR = TRI->isVGPRClass(RC) || TRI->isAGPRClass(RC);
      unsigned RPLimit = allocatablePool(
          const_cast<MachineFunction &>(MF),
          IsVGPR ? RegFile::VGPR : RegFile::SGPR);
      VRegMaskPair SpillVMP(Cur, MRI->getMaxLaneMaskForVReg(Cur));
      if (!Emitter->canSpill(SpillVMP)) {
        LLVM_DEBUG(dbgs() << "  spill-self floor: no legal insertion plan for "
                          << printReg(Cur, TRI) << "\n");
        return false;
      }
      if (!RecoverySpilledVRegs.insert(Cur).second)
        reportPointOverPressure(Cur, IsVGPR, RPLimit, "re-spill-blocked");

      LLVM_DEBUG(dbgs() << "  spill-self floor\n");
      Emitter->beginPass(IsVGPR, demandFor(poolOf(RC)));
      MachineInstr *DefMI = MRI->getVRegDef(Cur);
      assert(DefMI && "uncolorable value must have a def in SSA");
      SlotIndex KillIdx = LIS->getInstructionIndex(*DefMI).getRegSlot();
      if (SSAForensicReporter::enabled())
        Reporter->transformation("memory-spill", Cur.virtRegIndex());
      Emitter->spillOneVMP(SpillVMP, KillIdx);
      SmallVector<Register, 8> Redefs;
      for (const VRegMaskPair &VMP : Emitter->reloadedRegs())
        Redefs.push_back(VMP.getVReg());
      ColorInPlace(Cur);
      for (Register RD : Redefs)
        ColorInPlace(RD);
      if (CompleteCurrent())
        return true;
      continue;
    }
    }
  }
}

bool AMDGPUSSARegisterAllocator::drainUncolorableWorklist(
    MachineFunction &MF, bool ReportFailure) {
  SSARA_TRACE();
  auto Colored = [&](Register R) { return ColorMap.count(R) != 0; };
  auto Skip = [&](Register R) {
    return MRI->reg_nodbg_empty(R) || !LIS->hasInterval(R) ||
           LIS->getInterval(R).empty();
  };

  size_t Cursor = 0, PassEnd = UncolorableVRegs.size();
  bool Progress = false;
  while (Cursor < UncolorableVRegs.size()) {
    Register Failed = UncolorableVRegs[Cursor++];
    if (Colored(Failed)) {
      Progress = true;
    } else if (!Skip(Failed)) {
      if (!ReportFailure && RecoverySpilledVRegs.count(Failed))
        return false;
      recoverUncolorable(Failed);
      if (Colored(Failed))
        Progress = true;
    }
    if (Cursor == PassEnd) {
      if (!Progress)
        break;
      Progress = false;
      PassEnd = UncolorableVRegs.size();
    }
  }

  for (size_t I = Cursor; I < UncolorableVRegs.size(); ++I) {
    Register R = UncolorableVRegs[I];
    if (Colored(R) || Skip(R))
      continue;
    if (!ReportFailure)
      return false;
    const TargetRegisterClass *RC = MRI->getRegClass(R);
    bool IsVGPR = TRI->isVGPRClass(RC) || TRI->isAGPRClass(RC);
    unsigned RPLimit =
        allocatablePool(MF, IsVGPR ? RegFile::VGPR : RegFile::SGPR);
    reportPointOverPressure(R, IsVGPR, RPLimit, "worklist-drained");
  }
  return true;
}

// A call destroys a register unless its regmask preserves it and the call does
// not define it (the return-address pair rides on the call as an explicit def,
// outside the mask). MachineInstr has no regmask accessor -- it is an operand,
// and a call carries exactly one.
static bool preservedByCall(const MachineInstr *CallMI, MCRegister PR,
                            const TargetRegisterInfo *TRI) {
  SSARA_TRACE();
  if (CallMI->modifiesRegister(PR, TRI))
    return false;
  for (const MachineOperand &MO : CallMI->operands())
    if (MO.isRegMask())
      return !MO.clobbersPhysReg(PR);
  return true;
}

bool AMDGPUSSARegisterAllocator::survivesClobberSites(const LiveInterval &VI,
                                                      MCRegister PR) const {
  SSARA_TRACE();
  for (const auto &[Idx, MI] : CallSites)
    if (VI.liveAt(Idx) && !preservedByCall(MI, PR, TRI))
      return false;
  return true;
}

SmallVector<MCRegister, 32>
AMDGPUSSARegisterAllocator::getCSRSet(const MachineInstr &CallMI,
                                      const TargetRegisterClass *RC) const {
  SSARA_TRACE();
  SmallVector<MCRegister, 32> CSRs;
  for (MCPhysReg Reg : availableOrder(RC))
    if (preservedByCall(&CallMI, MCRegister(Reg), TRI))
      CSRs.push_back(MCRegister(Reg));
  return CSRs;
}

void AMDGPUSSARegisterAllocator::preassignValuesLiveAcrossCalls() {
  SSARA_TRACE();
  // Real calls only: a regmask call is what confines a crossing value to the
  // preserved set, while a site that merely carries an implicit def (V_ADD_CO
  // defining VCC) constrains that one register and is left to the walk's own
  // legality test.
  //
  // DOMINANCE order, not slot order. Two things rest on it: a spill below
  // recomputes liveness, so a dominated call must not be handled before its
  // dominator; and a register handed out at a dominating call is still that
  // value's at every call it dominates, which is what lets this pass judge
  // occupancy from the call in hand alone. Slot indexes only order within a
  // block, so they do not give this.
  struct Site {
    SlotIndex CS;
    MachineInstr *CallMI;
    SmallVector<Register, 16> Live;
  };
  SmallVector<Site, 8> Sites;
  for (auto *N : depth_first(MDT->getRootNode()))
    for (MachineInstr &MI : *N->getBlock())
      if (MI.isCall())
        Sites.push_back({LIS->getInstructionIndex(MI).getRegSlot(), &MI, {}});
  if (Sites.empty())
    return;

  const MachineFunction &MF = MRI->getMF();
  const unsigned RPLimit =
      allocatablePool(const_cast<MachineFunction &>(MF), StageFile);
  const bool IsVGPR = StageFile != RegFile::SGPR;

  // Spilling a value across a call retires ITS crossing, but the reload left
  // behind serves every later use of the value, so when one of those uses sits
  // beyond a further call the crossing migrates to the reload rather than
  // disappearing. A reload that crosses a call is the same problem this pass
  // exists to solve, so the sweep below repeats until it spills nothing.
  //
  // This terminates. A reload is placed after the call it was spilled across, so
  // a value's crossing can only ever move FORWARD in program order, and there
  // are finitely many calls. The (call, value) pairs already spilled are
  // recorded as well, so a spill that fails to retire a crossing is never
  // retried -- such a value stays uncolored here and is left to the walk, which
  // is an honest terminal rather than a loop.
  DenseSet<uint64_t> Spilled;

  bool Changed = true;
  while (Changed) {
  Changed = false;

  // One live set per call, this stage's file only, rebuilt per sweep because the
  // previous sweep's spills introduced new values. Within a sweep a liveAt
  // re-check covers what the sweep's own spills retire. Sorted because the live
  // set is a hash map and its iteration order must not reach the result.
  //
  // The widths come from the same walk. Ordinary values never compete for the
  // registers a call preserves: the crossing values are placed here, before the
  // walk starts, and the walk only ever sees those registers as occupied. So the
  // width-descending order that keeps a narrow value from fragmenting the slot a
  // wide tuple needs is required only among the values placed here, and is kept
  // local rather than shared with the walk's tiers.
  std::set<unsigned, std::greater<unsigned>> Tiers;
  for (Site &S : Sites) {
    S.Live.clear();
    for (const auto &[Reg, LaneMask] : getLiveRegs(S.CS, *LIS, *MRI)) {
      Register V(Reg);
      const TargetRegisterClass *RC = MRI->getRegClassOrNull(V);
      if (!RC || fileOf(RC) != StageFile)
        continue;
      S.Live.push_back(V);
      Tiers.insert(TRI->getRegSizeInBits(*RC));
    }
    llvm::sort(S.Live, [](Register A, Register B) {
      return A.virtRegIndex() < B.virtRegIndex();
    });
  }

  for (unsigned Width : Tiers)
    for (unsigned SiteIdx = 0; SiteIdx != Sites.size(); ++SiteIdx) {
      Site &S = Sites[SiteIdx];
      // A value spilled at a call this one is dominated by may have stopped
      // crossing here as well -- its reload was placed at that earlier call.
      auto stillCrossing = [&](Register V) {
        return LIS->hasInterval(V) && LIS->getInterval(V).liveAt(S.CS);
      };

      // CSR(CS) is per class as well as per call, so it is built on first use
      // for each class that turns up in this call's live set.
      DenseMap<const TargetRegisterClass *, SmallVector<MCRegister, 32>> CSRSets;
      auto csrSet = [&](const TargetRegisterClass *RC)
          -> const SmallVector<MCRegister, 32> & {
        auto It = CSRSets.find(RC);
        if (It == CSRSets.end())
          It = CSRSets.try_emplace(RC, getCSRSet(*S.CallMI, RC)).first;
        return It->second;
      };

      LLVM_DEBUG(dbgs() << "\nacross-call assign at " << S.CS << ", " << Width
                        << "-bit\n");

      // One walk over what crosses this call: reserve the register held by every
      // value this call preserves (any width -- a wide value placed in an
      // earlier tier still holds its register here), and collect this tier's
      // values that still need one. The reservation must be complete before the
      // first pick, or a pick could take a register that a value further down
      // the list already carries in from a dominating call.
      SmallVector<MCRegister, 32> Taken;
      SmallVector<Register, 16> Pending;
      for (Register V : S.Live) {
        if (!stillCrossing(V))
          continue;
        MCRegister Held = ColorMap.lookup(V);
        bool Keeps = Held && survivesClobberSites(LIS->getInterval(V), Held);
        if (Keeps)
          Taken.push_back(Held);
        if (TRI->getRegSizeInBits(*MRI->getRegClass(V)) != Width)
          continue; // other tier: it counted for occupancy, nothing more here
        if (Keeps) {
          LLVM_DEBUG(dbgs() << "  " << printReg(V, TRI) << " keeps "
                            << TRI->getName(Held) << "\n");
          continue;
        }
        Pending.push_back(V);
      }
      auto isFree = [&](MCRegister PR) {
        return llvm::none_of(
            Taken, [&](MCRegister T) { return TRI->regsOverlap(PR, T); });
      };

      // Holds nothing, or holds one this call does not preserve: take one from
      // CSR(CS), and spill across the call when it has nothing free. CSR(CS) is
      // only a prefilter -- it answers for this call alone, while the register
      // has to survive every clobber site the value is live at, so each
      // candidate is checked against all of them before it is handed out.
      for (Register V : Pending) {
        const LiveInterval &VI = LIS->getInterval(V);
        MCRegister Pick;
        for (MCRegister C : csrSet(MRI->getRegClass(V)))
          if (isFree(C) && survivesClobberSites(VI, C)) {
            Pick = C;
            break;
          }
        if (Pick) {
          LLVM_DEBUG(dbgs() << "  " << printReg(V, TRI) << " -> "
                            << TRI->getName(Pick) << "\n");
          commitColor(V, Pick);
          Taken.push_back(Pick);
          continue;
        }
        // Nothing this call preserves is free -- spill V across it, unless that
        // was already tried here and left V crossing, in which case there is
        // nothing further this pass can do for it.
        if (!Spilled.insert((uint64_t(SiteIdx) << 32) | V.virtRegIndex()).second) {
          LLVM_DEBUG(dbgs() << "  " << printReg(V, TRI)
                            << " -> still crossing after spill, left to walk\n");
          continue;
        }
        LLVM_DEBUG(dbgs() << "  " << printReg(V, TRI) << " -> spill across\n");
        Emitter->beginPass(IsVGPR, demandFor(poolOf(MRI->getRegClass(V))));
        VRegMaskPair SpillVMP(V, MRI->getMaxLaneMaskForVReg(V));
        if (!Emitter->canSpill(SpillVMP)) {
          LLVM_DEBUG(dbgs() << "  " << printReg(V, TRI)
                            << " -> no legal spill insertion plan\n");
          continue;
        }
        if (SSAForensicReporter::enabled())
          Reporter->transformation("across-call-spill", V.virtRegIndex());
        Emitter->spillOneVMP(SpillVMP, S.CS);
        Changed = true;
      }
    }
  }
}

void AMDGPUSSARegisterAllocator::color() {
  SSARA_TRACE();
  // The vreg set has moved since the up-front classification: the earlier
  // stage's spills and the pre-spill work added values. Rebuild the width tiers
  // for this stage before anything consults them.
  classifyVRegs();

  PendingTies.clear();
  for (auto *Node : depth_first(MDT->getRootNode()))
    for (MachineInstr &MI : *Node->getBlock())
      for (const MachineOperand &DefMO : MI.operands()) {
        if (!DefMO.isReg() || !DefMO.isDef() || DefMO.isImplicit() ||
            !DefMO.getReg().isVirtual() ||
            fileOf(MRI->getRegClass(DefMO.getReg())) != StageFile)
          continue;
        unsigned UseOpIdx;
        if (MI.isRegTiedToUseOperand(DefMO.getOperandNo(), &UseOpIdx))
          PendingTies.push_back(
              {&MI, DefMO.getOperandNo(), UseOpIdx});
      }

  LLVM_DEBUG({
    dbgs() << "Coloring order (width descending):";
    for (unsigned W : ColoringOrder)
      dbgs() << " " << W;
    dbgs() << "\n";
  });

  // Function-wide width-descending: color ALL defs of the widest width across
  // all blocks before any narrower width. This prevents narrow defs from
  // fragmenting alignment slots needed by wider tuples (e.g., a VGPR_32 at an
  // odd index blocking an even-aligned VReg_64 pair on gfx90a).
  //
  // Wider assignments are committed to ColorMap before narrower passes start,
  // so seedOccupiedAtBBEntry naturally catches cross-block wider live-ins.
  // For wider defs born mid-block (not live at BBStart), a per-block WiderDefs
  // pre-scan collects them from ColorMap — O(|block|), same cost as the walk.

  // Collect clobber sites: a vreg live across one must not be assigned a
  // register the instruction clobbers (pickFreePhysReg consults these). Two
  // kinds: (1) a call regmask, and (2) an instruction with an IMPLICIT physical
  // register def - e.g. an inline-asm clobber list lowered to implicit-def dead
  // early-clobber $vgprN, or an instruction-description implicit clobber. These
  // carry no regmask and define no value, so nothing else models them. EXPLICIT
  // physreg defs are deliberately excluded: they are real values that the
  // forward walk already marks occupied within a block, and a call's explicit
  // result def is already covered because the call is a clobber site via its
  // regmask (modifiesRegister() catches the explicit def at pick time). Adding
  // explicit defs here would over-constrain coloring (large, correctness-neutral
  // allocation churn) without fixing any crash. Reserved registers are skipped:
  // pickFreePhysReg only ever picks allocatable registers (getOrder excludes
  // reserved), which share no reg unit with a reserved-only def.
  CallSites.clear();
  for (auto *Node : depth_first(MDT->getRootNode()))
    for (MachineInstr &MI : *Node->getBlock()) {
      bool IsClobberSite = false;
      for (const MachineOperand &MO : MI.operands()) {
        if (MO.isRegMask()) {
          // Fold the call's clobbers into MRI's used-physreg mask. The RA
          // framework (which we bypass) normally does this; without it
          // MRI::UsedPhysRegMask stays empty and MRI.isPhysRegUsed() reports
          // call-clobbered registers as unused. PrologEpilogInserter's
          // findUnusedRegister() would then pick a call-clobbered SGPR as the
          // whole-function frame-pointer save register, giving a value the call
          // destroys (read as undefined at restore). setBitsNotInMask marks the
          // registers the mask does NOT preserve, i.e. exactly the clobbers.
          MRI->addPhysRegsUsedFromRegMask(MO.getRegMask());
          IsClobberSite = true;
        } else if (MO.isReg() && MO.isDef() && MO.isImplicit() &&
                   MO.getReg().isPhysical() &&
                   MRI->isAllocatable(MO.getReg().asMCReg())) {
          // An implicit def of an ALLOCATABLE physreg is a clobber site: a value
          // live across it colored onto that reg would be destroyed (e.g. an
          // inline-asm register clobber, or an implicit-def $vcc on V_ADD_CO /
          // V_CMP -- VCC *is* allocatable on AMDGPU). This holds even for a DEAD
          // def: "dead" means the defined value is unused, but the register write
          // still happens, so a crossing value in that reg is still clobbered.
          // Non-allocatable defs (implicit-def $scc) can never hold an allocated
          // vreg and are excluded by isAllocatable. NB: these sites drive only
          // the exact per-register IsFree legality check, NOT the ACL priority
          // set (which is narrowed to real regmask calls below) -- so the flood
          // of VCC defs no longer perturbs coloring priority on call-free code.
          IsClobberSite = true;
        }
      }
      if (IsClobberSite)
        CallSites.push_back({LIS->getInstructionIndex(MI).getRegSlot(), &MI});
    }
  LLVM_DEBUG(dbgs() << "CallSites (regmask + allocatable implicit-def): "
                    << CallSites.size() << "\n");

  // Around-call-liver (ACL) set: vregs whose live interval spans a real CALL
  // (regmask site). These must go in registers the crossed call preserves
  // (enforced per-call by the IsFree regmask check). They are colored in a
  // SEPARATE, EARLIER width-descending walk (phase 0) over the whole function,
  // before ordinary vregs (phase 1). Priority — not just legality — is the
  // point: in a single combined walk an ordinary vreg defined before/between
  // calls grabs a preserved register first, leaving a later-crossing ACL with
  // nothing free even though IsFree would have allowed it. Coloring all ACLs
  // first reserves the preserved registers they need across the whole function.
  //
  // Only REGMASK (call) sites drive this priority set, NOT every clobber site.
  // A regmask clobbers a large caller-saved partition, so a value crossing it is
  // genuinely squeezed into the preserved subset and benefits from priority. A
  // lone implicit physreg def (e.g. a live implicit-def $vcc on V_ADD_CO) only
  // clobbers that ONE register; the exact per-register IsFree check already
  // rejects that single reg for a crossing value, and no phase-0 priority is
  // warranted. Including such sites floods the ACL set on call-free code
  // (V_ADD_CO/V_CMP emit VCC defs everywhere), needlessly reorders coloring, and
  // has triggered downstream SSA-destruction crashes. IsFree still consults ALL
  // of CallSites for legality — only the ACL priority membership is narrowed.
  // Values live across a call can only occupy a register the call preserves, so
  // they are assigned before anything else has a chance to take those registers.
  preassignValuesLiveAcrossCalls();

  // Rebuild the tiers again: the spills above introduce the reload remnants,
  // whose width can be one no vreg had before (a 128-bit reload in a function
  // whose tiers were 1024/64/32). The walk below visits defs one tier at a time,
  // so a width missing from the list is never visited and its vregs reach
  // operand rewrite uncolored.
  classifyVRegs();

  DenseSet<Register> ACLSet;
  SmallVector<SlotIndex, 8> CallOnlySites;
  for (const auto &[CallIdx, CallMI] : CallSites)
    if (CallMI->isCall())
      CallOnlySites.push_back(CallIdx);
  if (!CallOnlySites.empty())
    for (unsigned I = 0, E = MRI->getNumVirtRegs(); I < E; ++I) {
      Register VReg = Register::index2VirtReg(I);
      if (MRI->reg_nodbg_empty(VReg) || !LIS->hasInterval(VReg))
        continue;
      const LiveInterval &LI = LIS->getInterval(VReg);
      for (SlotIndex CS : CallOnlySites)
        if (LI.liveAt(CS)) {
          ACLSet.insert(VReg);
          break;
        }
    }
  LLVM_DEBUG(dbgs() << "ACL set: " << ACLSet.size()
                    << " vregs live across calls\n");

  // Phase 0 = ACL vregs, phase 1 = ordinary. Skip phase 0 when no ACLs exist.
  for (unsigned Phase = (ACLSet.empty() ? 1 : 0); Phase < 2; ++Phase) {
    LLVM_DEBUG(dbgs() << "\n=== Coloring phase " << Phase << " ("
                      << (Phase == 0 ? "ACL" : "ordinary") << ") ===\n");

  for (unsigned Width : ColoringOrder) {
    for (auto *Node : depth_first(MDT->getRootNode())) {
      MachineBasicBlock *MBB = Node->getBlock();

      LLVM_DEBUG(dbgs() << "\n=== Width pass: " << Width << "-bit, "
                        << printMBBReference(*MBB) << " ===\n");

      // Pre-scan: collect wider defs in THIS block from prior width passes.
      // These are defs not live at BBStart (born mid-block) whose physregs
      // must be avoided by the current narrower pass via LI.overlaps().
      SmallVector<std::pair<MCRegister, const LiveInterval *>, 8> WiderDefs;
      for (MachineInstr &MI : *MBB)
        for (MachineOperand &MO : MI.defs())
          if (MO.isReg() && MO.getReg().isVirtual()) {
            Register Reg = MO.getReg();
            if (TRI->getRegSizeInBits(*MRI->getRegClass(Reg)) > Width)
              if (auto It = ColorMap.find(Reg); It != ColorMap.end())
                WiderDefs.push_back({It->second, &LIS->getInterval(Reg)});
          }

      seedOccupiedAtBBEntry(MBB);

      for (MachineInstr &MI : *MBB) {
        // Physreg units / colored-vreg physregs whose freeing is deferred past
        // an early-clobber def (see below), freed after this instruction's defs
        // are colored.
        SmallVector<MCRegUnit, 8> DeferredUnits;
        SmallVector<MCRegister, 4> DeferredFree;

        // Kill uses before coloring defs: a def can reuse the physreg of
        // a source that dies at this instruction (no interference without
        // early-clobber). PHIs skipped: their sources are live only to
        // predecessor boundaries, and markFree would clear physregs that
        // preceding PHI defs already claimed.
        if (!MI.isPHI()) {
          // An early-clobber def is live while this instruction's uses are read,
          // so it must NOT reuse a dying use's physreg. Defer freeing dying uses
          // until after defs are colored (they are still freed for later
          // instructions, so no leak); non-early-clobber defs on other
          // instructions keep the reuse optimization.
          bool HasEC = false;
          for (const MachineOperand &MO : MI.operands())
            if (MO.isReg() && MO.isDef() && MO.isEarlyClobber()) {
              HasEC = true;
              break;
            }
          SlotIndex NextSI =
              LIS->getInstructionIndex(MI).getRegSlot().getNextSlot();
          // Iterate all operands filtered by the isUse flag rather than
          // MI.uses(): the range helpers key off operand POSITION
          // (getNumExplicitDefs), which is wrong for variadic instructions with
          // flag-interspersed operands (e.g. INLINEASM), whose def operands are
          // not leading. MI.uses() would then wrongly include those defs.
          for (const MachineOperand &MO : MI.operands()) {
            if (!MO.isReg() || !MO.isUse())
              continue;
            Register Reg = MO.getReg();
            if (Reg.isPhysical()) {
              for (MCRegUnit Unit : TRI->regunits(Reg))
                if (!LIS->getRegUnit(Unit).liveAt(NextSI)) {
                  if (HasEC)
                    DeferredUnits.push_back(Unit);
                  else {
                    OccupiedRegUnits.reset(Unit);
                    shadowFreeUnit(Unit); // mirror (no-op unless shadowActive)
                  }
                }
              continue;
            }
            auto It = ColorMap.find(Reg);
            if (It == ColorMap.end())
              continue;
            const LiveInterval &LI = LIS->getInterval(Reg);
            if (!LI.liveAt(NextSI)) {
              if (HasEC) {
                DeferredFree.push_back(It->second);
              } else {
                markFree(It->second);
                LLVM_DEBUG(dbgs()
                           << "    kill: " << printReg(Reg, TRI) << " free "
                           << TRI->getName(It->second) << "\n");
              }
            } else if (LI.hasSubRanges()) {
              // PARTIAL KILL: the whole value is still live, but some sub-lanes
              // are dead here — e.g. after the spiller stored sub0..sub2 of a
              // vreg_128, only sub3 remains live, yet the value is colored to the
              // whole tuple. Holding the dead lanes occupied is a soundness bug:
              // spilling N lanes must drop RP by N*32. Free the physreg units of
              // each subrange NOT live at NextSI so a narrower/aligned value can
              // use them.
              for (const LiveInterval::SubRange &S : LI.subranges()) {
                if (S.liveAt(NextSI))
                  continue;
                for (unsigned Ch = 0; Ch < 8; ++Ch) {
                  unsigned SubIdx = SIRegisterInfo::getSubRegFromChannel(Ch);
                  if ((TRI->getSubRegIndexLaneMask(SubIdx) & S.LaneMask).none())
                    continue;
                  if (MCRegister Sub = TRI->getSubReg(It->second, SubIdx)) {
                    if (HasEC)
                      for (MCRegUnit U : TRI->regunits(Sub))
                        DeferredUnits.push_back(U);
                    else
                      markFree(Sub);
                    LLVM_DEBUG(dbgs() << "    partial-kill: " << printReg(Reg, TRI)
                                      << " free dead " << TRI->getName(Sub)
                                      << "\n");
                  }
                }
              }
            }
          }
        }

        // Iterate all operands filtered by the isDef flag rather than
        // MI.defs(): the range helper returns only the leading explicit defs
        // ([0, getNumExplicitDefs())), which is empty for variadic instructions
        // like INLINEASM (getNumExplicitDefs()==0). Their def operands sit after
        // the asm string and flag immediates, so MI.defs() misses them and the
        // vreg they define never gets colored. Flag-based filtering visits every
        // real def regardless of operand position; flag immediates are !isReg().
        for (MachineOperand &MO : MI.operands()) {
          // Explicit defs only: implicit defs are call/instr clobbers (e.g.
          // implicit-def $scc, $sgpr32, and call clobber lists). MI.defs()
          // excluded them and coloring relied on that; marking them occupied
          // here (never freed) exhausts the file. INLINEASM's constraint reg
          // defs are explicit (only its clobbers are implicit), so they remain
          // covered.
          if (!MO.isReg() || !MO.isDef() || MO.isImplicit())
            continue;
          Register Reg = MO.getReg();
          if (!Reg.isVirtual()) {
            markOccupied(Reg);
            continue;
          }

          if (TRI->getRegSizeInBits(*MRI->getRegClass(Reg)) != Width) {
            if (auto It = ColorMap.find(Reg); It != ColorMap.end()) {
              markOccupied(It->second);
              LLVM_DEBUG(dbgs() << "    mark wider def: " << printReg(Reg, TRI)
                                << " -> " << TRI->getName(It->second) << "\n");
            }
            continue;
          }

          // Stage filter: allocation runs in two independent stages, SGPR then
          // VGPR/AGPR (fileOf maps AGPR to VGPR), so the VGPR budget can reserve
          // scratch for the SGPR spills the first stage made. A def for the other
          // stage is skipped; if it was already colored in the earlier stage,
          // mark its physreg occupied so this stage does not reuse it (same
          // treatment as a wider already-colored def above). Disjoint files, so
          // this only reorders coloring within each file, never across.
          if (fileOf(MRI->getRegClass(Reg)) != StageFile) {
            if (auto It = ColorMap.find(Reg); It != ColorMap.end())
              markOccupied(It->second);
            continue;
          }

          // Assigned by the across-call pass before coloring began. Keep that
          // register and mark it occupied at the def so this walk's values do not
          // reuse it (the kill path frees it at its last use, exactly as for a
          // wider already-colored def).
          if (auto It = ColorMap.find(Reg); It != ColorMap.end()) {
            markOccupied(It->second);
            continue;
          }

          // Phase filter: phase 0 colors only ACL vregs, phase 1 only the rest.
          // A def for the other phase is skipped; if already colored in phase 0
          // (an ACL def revisited in phase 1), mark its physreg occupied at its
          // def so phase-1 values do not reuse it (kill path frees it at its last
          // use, exactly as for a wider already-colored def).
          if (ACLSet.contains(Reg) != (Phase == 0)) {
            if (auto It = ColorMap.find(Reg); It != ColorMap.end())
              markOccupied(It->second);
            continue;
          }

          MCRegister Chosen;
          unsigned UseOpIdx;
          bool IsTied = MI.isRegTiedToUseOperand(MO.getOperandNo(), &UseOpIdx);
          MCRegister TiedUseColor;
          if (IsTied &&
              (TiedUseColor = ColorMap.lookup(MI.getOperand(UseOpIdx).getReg()))) {
            // Ordinary two-address def: inherit the tied use's color. When the
            // tied use reads a sub-register (e.g. a 32-bit V_MOV_B32_dpp or
            // V_WRITELANE_B32 tied to one lane of a wider value), the def's
            // class matches that lane, so inherit the sub-register of the color,
            // not the whole super-register.
            Chosen = TiedUseColor;
            if (unsigned UseSubIdx = MI.getOperand(UseOpIdx).getSubReg()) {
              Chosen = TRI->getSubReg(TiedUseColor, UseSubIdx);
              assert(Chosen && "Invalid tied-use subreg index");
            }
            LLVM_DEBUG(dbgs() << "    tied: " << printReg(Reg, TRI)
                              << " inherits " << TRI->getName(Chosen) << "\n");
          } else if (IsTied && MI.getOperand(UseOpIdx).isUndef()) {
            // The tied use is an `undef` passthrough (the DPP "old" source
            // `%N = V_..._dpp undef %N, ...`, a D16 load's untouched half, or a
            // MIX partial def). Its value is a don't-care, so there is no
            // earlier color to inherit -- color the def like a normal def.
            // rewriteOperands() then assigns the same physreg to the self-tied
            // use (same vreg), preserving two-address form.
            Chosen = pickFreePhysReg(MRI->getRegClass(Reg),
                                     LIS->getInterval(Reg), WiderDefs);
            if (!Chosen) {
              // Collect-and-skip, as at the main pick site below.
              UncolorableVRegs.push_back(Reg);
              continue;
            }
            LLVM_DEBUG(dbgs() << "    color (undef self-tie): "
                              << printReg(Reg, TRI) << " -> "
                              << TRI->getName(Chosen) << "\n");
          } else if (IsTied) {
            // The tied use failed earlier in this coloring walk. Defer it to
            // recovery. PendingTies is validated only after all coloring
            // failures have been recovered.
            Register TiedUse = MI.getOperand(UseOpIdx).getReg();
            if (!llvm::is_contained(UncolorableVRegs, TiedUse))
              UncolorableVRegs.push_back(TiedUse);
            LLVM_DEBUG(dbgs() << "    defer tied def " << printReg(Reg, TRI)
                              << ": use " << printReg(TiedUse, TRI)
                              << " is awaiting recovery\n");
            continue;
          } else {
            SmallVector<MCRegister, 4> Hints =
                collectPhiHints(Reg, MRI->getRegClass(Reg));
            // E4 AllocationAttemptStarted: record the attempt BEFORE the pick so
            // the candidate facts (E5/E6/E7) link back to it.
            uint64_t AttemptID = 0;
            if (SSAForensicReporter::enabled()) {
              const TargetRegisterClass *ARC = MRI->getRegClass(Reg);
              // Full liveness cross-section at the value's def slot (a decision
              // boundary). collectLiveSet is const and reuses the LIS liveAt
              // walk; only runs when the reporter is enabled.
              SmallVector<LiveSetEntry, 32> LiveSet;
              collectLiveSet(LIS->getInterval(Reg).beginIndex(), LiveSet);
              AttemptID = Reporter->attemptStarted(
                  Reg.virtRegIndex(), TRI->getRegSizeInBits(*ARC),
                  TRI->getRegClassName(ARC), "first-fit-order", LiveSet);
            }
            Chosen = pickFreePhysReg(MRI->getRegClass(Reg),
                                     LIS->getInterval(Reg), WiderDefs, Hints,
                                     AttemptID);
            if (!Chosen) {
              // No physreg is free across this value's whole range (the
              // %1072/%560 long-liver-through-tuple-churn case). Do NOT assert
              // and do NOT bail: record it and SKIP it (occupy nothing for it),
              // so the rest of the walk colors normally as if this value were
              // absent. The driver spills all collected values afterward, then
              // colors the short reload remainders in place. Skipping is correct
              // because the value is about to be spilled — it holds no register.
              // The COLORFAIL spill-across facts are needed only for the debug
              // dump or the forensic snapshot; skip the whole ColorMap scan on
              // the default path (preserves the original zero-cost behavior).
              bool WantColorFailFacts = SSAForensicReporter::enabled();
              LLVM_DEBUG(WantColorFailFacts = true);
              if (WantColorFailFacts) {
                const LiveInterval &FVI = LIS->getInterval(Reg);
                SlotIndex FS = FVI.beginIndex(), FE = FVI.endIndex();
                const TargetRegisterClass *FRC = MRI->getRegClass(Reg);
                bool FIsVGPR = TRI->isVGPRClass(FRC) || TRI->isAGPRClass(FRC);
                // Extract the spill-across facts once; used by both the debug
                // dump and the forensic record below.
                unsigned NLiveThru = 0;
                SmallVector<SpillAcrossCandidate, 8> Cands;
                SmallVector<unsigned, 128> LT;
                collectSpillAcrossCandidates(Reg, FS, FE, FIsVGPR, NLiveThru,
                                             Cands, LT);
                LLVM_DEBUG({
                  dbgs() << "!!! COLORFAIL " << printReg(Reg, TRI) << " " << FVI
                         << " class=" << TRI->getRegClassName(FRC) << "\n";
                  // ANSWER "is there a valid reg to spill across R?": count
                  // colored values in R's FILE that are LIVE-THROUGH [FS,FE)
                  // with NO use strictly inside — each such value's register can
                  // be freed across the whole region by spilling it (reload past
                  // FE).
                  for (const SpillAcrossCandidate &C : Cands)
                    dbgs() << "    SPILL-CANDIDATE " << printReg(C.V, TRI)
                           << " -> " << TRI->getName(C.P) << " w="
                           << C.WidthDwords << "  " << *C.OVI << "\n";
                  dbgs() << "  >>> VALID SPILL-ACROSS candidates for "
                         << printReg(Reg, TRI) << " [" << FS << "," << FE
                         << "): live-through=" << NLiveThru
                         << " no-interior-use=" << Cands.size() << "\n";
                  // DEBUG-LT: dump the FULL live-across set (reg indices sorted)
                  // so a round-to-round diff shows exactly which vregs newly
                  // appear.
                  dbgs() << "  DEBUGLT " << printReg(Reg, TRI) << " across("
                         << LT.size() << "): ";
                  for (unsigned I : LT)
                    dbgs() << I << " ";
                  dbgs() << "\n";
                });
                // E16 snapshot: record the spill-across facts and the register
                // occupancy at the failure point (facts only — the live-through
                // and no-interior counts, plus the occupancy map produced by the
                // refactored collectOccupancy), each carrying the full liveness
                // cross-section at the failure slot.
                if (SSAForensicReporter::enabled()) {
                  SmallVector<LiveSetEntry, 32> LiveSet;
                  collectLiveSet(FS, LiveSet);
                  Reporter->colorFailAnalysis(Reg.virtRegIndex(), NLiveThru,
                                              Cands.size(), AttemptID);
                  OccupancyFacts OF;
                  collectOccupancy(FRC, FS, &FVI, OF);
                  Reporter->snapshot("colorfail-occupancy", FS, OF, AttemptID,
                                     LiveSet);
                }
              }
              // E10 AllocationAttemptFailed: no physreg free across the range.
              // Carry the full liveness cross-section at the failure boundary.
              if (SSAForensicReporter::enabled()) {
                SmallVector<LiveSetEntry, 32> FailLiveSet;
                collectLiveSet(LIS->getInterval(Reg).beginIndex(), FailLiveSet);
                Reporter->attemptFailed(AttemptID, Reg.virtRegIndex(),
                                        "no-free-physreg-across-range",
                                        FailLiveSet);
              }
              UncolorableVRegs.push_back(Reg);
              continue;
            }
            // E9 AllocationAttemptCompleted: the pick succeeded.
            if (SSAForensicReporter::enabled())
              Reporter->attemptCompleted(
                  AttemptID, Reg.virtRegIndex(), Chosen.id(),
                  TRI->getName(Chosen), "first-fit-order");
            LLVM_DEBUG(dbgs() << "    color: " << printReg(Reg, TRI) << " -> "
                              << TRI->getName(Chosen) << "\n");
          }

          // SHADOW register-tree oracle: ask the tree what it WOULD pick and log
          // it against the allocator's actual Chosen. Behavior-neutral — the
          // tree's answer is discarded here, never fed back. Done BEFORE the
          // markOccupied below so the tree still shows Chosen free (its leaf must
          // be a candidate). Only the width-1 VGPR_32 pick is in scope; wider
          // tuples / non-VGPR files log a skip. No-op unless shadowActive.
          // Cause link: AttemptID exists only on the normal-def pick path (tied
          // and undef-self-tie defs create no E4 attempt), so use 0 (no-cause)
          // uniformly — the shadow event stands on its own vreg/leaf facts.
          if (shadowActive()) {
            const TargetRegisterClass *CRC = MRI->getRegClass(Reg);
            unsigned WDwords = TRI->getRegSizeInBits(*CRC) / 32;
            bool IsVGPR = TRI->isVGPRClass(CRC) && !TRI->isAGPRClass(CRC);
            int RealLeaf = shadowLeafOf(Chosen);
            if (!IsVGPR || WDwords != 1)
              Reporter->shadowTreeSkip(/*Cause=*/0, Reg.virtRegIndex(), WDwords,
                                       !IsVGPR ? "class" : "wide-tuple");
            else if (RealLeaf < 0)
              Reporter->shadowTreeSkip(/*Cause=*/0, Reg.virtRegIndex(), WDwords,
                                       "leaf-oob");
            else {
              LLVM_DEBUG({
                // Drift probe (DEBUG-only, neutral): the tree mirrors exactly
                // OccupiedRegUnits, so RealLeaf (which the allocator picked, thus
                // free in its OccupiedAtDef ⊇ OccupiedRegUnits view) MUST be free
                // in the tree. If not, the mirror drifted.
                if (!ShadowTree->isFree((unsigned)RealLeaf, 1))
                  dbgs() << "!!! SHADOW-DRIFT vreg" << Reg.virtRegIndex()
                         << " realLeaf=" << RealLeaf
                         << " but shadow tree says it is occupied (mirror bug; "
                            "freeCount=" << ShadowTree->freeCount() << ")\n";
              });
              int TreeLeaf = ShadowTree->pickFreeAligned(1);
              bool Match = (TreeLeaf == RealLeaf);
              Reporter->shadowTreePick(/*Cause=*/0, Reg.virtRegIndex(), WDwords,
                                       RealLeaf, TreeLeaf, Match,
                                       ShadowTree->freeCount(),
                                       ShadowTree->fullCountAtLevel(0));
            }
          }

          ColorMap[Reg] = Chosen;
          // A dead def (e.g. the unused carry-out of V_ADD_CO_U32_e64) is not
          // live past this instruction, so it must not reserve a register going
          // forward. Marking it occupied would leak: the kill path only frees
          // dying uses, never dead defs, so they accumulate until the class is
          // exhausted ("Failed to find free physreg"). It still needs a valid,
          // non-conflicting physreg (pickFreePhysReg above picked one free at
          // this point) and still counts toward the high-water mark below, but
          // is never added to OccupiedRegUnits.
          if (!MO.isDead())
            markOccupied(Chosen);

          // DEF-TIME PARTIAL KILL: a tuple def whose HIGH lanes are dead at the
          // def (e.g. %548 = V_LSHR_B64 where sub1 is never read: sub1 subrange is
          // [def,def+dead)) still gets its whole aligned physreg markOccupied'd
          // above, but the dead lanes must not stay reserved — otherwise the
          // colorer holds registers no value can use, contradicting lane-accurate
          // region pressure. Symmetric to the use-side partial-kill: free the
          // units of each subrange NOT live just after the def. (Skip if the
          // whole def is dead — already unoccupied — or has no subranges.)
          //
          // EARLY-CLOBBER GUARD: an early-clobber def (e.g. V_MAD_U64_U32 whose
          // early-clobber tuple result overlaps an input operand it also reads)
          // is defined at the EC slot, before the regular def slot, and may still
          // constrain a lane that liveAt(DefNext) reports dead. Freeing such a
          // lane lets a later value reuse it and the rewriter then sees a use with
          // no live segment ("No live segment at use" on gfx1100 true16/gisel).
          // The use-side partial-kill defers frees under HasEC for the same
          // reason; here we simply skip def-partial-kill for EC defs — the
          // dead-lane reclaim is a code-quality optimization we can forgo on the
          // rare EC-tuple instruction.
          if (!MO.isDead() && !MO.isEarlyClobber() && LIS->hasInterval(Reg)) {
            const LiveInterval &DLI = LIS->getInterval(Reg);
            if (DLI.hasSubRanges()) {
              SlotIndex DefNext =
                  LIS->getInstructionIndex(MI).getRegSlot().getNextSlot();
              for (const LiveInterval::SubRange &S : DLI.subranges()) {
                if (S.liveAt(DefNext))
                  continue;
                for (unsigned Ch = 0; Ch < 8; ++Ch) {
                  unsigned SubIdx = SIRegisterInfo::getSubRegFromChannel(Ch);
                  if ((TRI->getSubRegIndexLaneMask(SubIdx) & S.LaneMask).none())
                    continue;
                  if (MCRegister Sub = TRI->getSubReg(Chosen, SubIdx)) {
                    markFree(Sub);
                    LLVM_DEBUG(dbgs()
                               << "    def-partial-kill: " << printReg(Reg, TRI)
                               << " free dead " << TRI->getName(Sub) << "\n");
                  }
                }
              }
            }
          }

          unsigned Idx = TRI->getHWRegIndex(Chosen);
          unsigned W = TRI->getRegSizeInBits(*MRI->getRegClass(Reg)) / 32;
          // Classify by the CHOSEN physical register's file, not the vreg's
          // class: an AV (AGPR-or-VGPR) vreg is not isVGPRClass, so tracking by
          // vreg class would leave its high-water untracked.
          const TargetRegisterClass *PhysRC = TRI->getPhysRegBaseClass(Chosen);
          if (TRI->isVGPRClass(PhysRC))
            MaxVGPRIdx = std::max(MaxVGPRIdx, Idx + W);
          else if (TRI->isAGPRClass(PhysRC))
            MaxAGPRIdx = std::max(MaxAGPRIdx, Idx + W);
          else if (TRI->isSGPRClass(PhysRC))
            MaxSGPRIdx = std::max(MaxSGPRIdx, Idx + W);
        }

        // Free dying uses deferred past an early-clobber def now that its defs
        // are colored (they could not reuse these physregs).
        for (MCRegUnit Unit : DeferredUnits) {
          OccupiedRegUnits.reset(Unit);
          shadowFreeUnit(Unit); // mirror (no-op unless shadowActive)
        }
        for (MCRegister PR : DeferredFree)
          markFree(PR);
      }
    } // block walk

  } // width loop
  } // phase loop

  LLVM_DEBUG({
    dbgs() << "\nColoring result:\n";
    for (const auto &[VReg, PhysReg] : ColorMap)
      dbgs() << "  " << printReg(VReg, TRI) << " -> " << TRI->getName(PhysReg)
             << "\n";
  });
}

bool AMDGPUSSARegisterAllocator::tiedAssignmentsValid() const {
  SSARA_TRACE();
  auto OperandColor = [&](const MachineOperand &MO) -> MCRegister {
    Register R = MO.getReg();
    MCRegister PR = R.isPhysical() ? R.asMCReg() : ColorMap.lookup(R);
    if (PR && MO.getSubReg())
      PR = TRI->getSubReg(PR, MO.getSubReg());
    return PR;
  };

  for (const PendingTie &Tie : PendingTies) {
    const MachineOperand &DefMO = Tie.MI->getOperand(Tie.DefOpIdx);
    const MachineOperand &UseMO = Tie.MI->getOperand(Tie.UseOpIdx);
    assert(DefMO.isReg() && DefMO.isDef() && UseMO.isReg() && UseMO.isUse() &&
           "recorded tied operands changed shape");

    MCRegister DefPR = OperandColor(DefMO);
    MCRegister UsePR = OperandColor(UseMO);
    // rewriteOperands() copies the def's register into an uncolored undef
    // passthrough, so that case already has a valid final assignment.
    if (DefPR && !UsePR && UseMO.isUndef())
      continue;
    if (DefPR && DefPR == UsePR)
      continue;

    LLVM_DEBUG(dbgs() << "  postponed tie requires recolor: "
                      << printReg(DefMO.getReg(), TRI) << " / "
                      << printReg(UseMO.getReg(), TRI) << "\n");
    return false;
  }
  return true;
}

// === SSA Destruction + Operand Rewrite ===

bool AMDGPUSSARegisterAllocator::hasCFPseudos(MachineFunction &MF) const {
  SSARA_TRACE();
  for (const MachineBasicBlock &MBB : MF)
    for (const MachineInstr &MI : MBB.terminators())
      switch (MI.getOpcode()) {
      case AMDGPU::SI_IF:
      case AMDGPU::SI_ELSE:
      case AMDGPU::SI_IF_BREAK:
      case AMDGPU::SI_LOOP:
      case AMDGPU::SI_END_CF:
        return true;
      default:
        break;
      }
  return false;
}

void AMDGPUSSARegisterAllocator::emitSwap(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator InsertPt,
                                          MCRegister RegA, MCRegister RegB) {
  SSARA_TRACE();
  const TargetRegisterClass *RC = TRI->getPhysRegBaseClass(RegA);
  unsigned RegWidth = TRI->getRegSizeInBits(*RC);

  // In-place XOR swap: A ^= B; B ^= A; A ^= B.
  auto EmitXorTriplet = [&](unsigned Opc) {
    LIS->InsertMachineInstrInMaps(
        *BuildMI(MBB, InsertPt, DebugLoc(), TII->get(Opc), RegA)
             .addReg(RegA)
             .addReg(RegB));
    LIS->InsertMachineInstrInMaps(
        *BuildMI(MBB, InsertPt, DebugLoc(), TII->get(Opc), RegB)
             .addReg(RegA)
             .addReg(RegB));
    LIS->InsertMachineInstrInMaps(
        *BuildMI(MBB, InsertPt, DebugLoc(), TII->get(Opc), RegA)
             .addReg(RegA)
             .addReg(RegB));
  };

  // In-place XOR swap for the VOP3-encoded 16-bit XOR. Unlike the opcodes above
  // it carries source modifiers and op_sel, so the operand list is
  // dst, src0_mods, src0, src1_mods, src1, op_sel.
  auto EmitXorTripletT16 = [&] {
    auto Build = [&](MCRegister Dst) {
      LIS->InsertMachineInstrInMaps(
          *BuildMI(MBB, InsertPt, DebugLoc(),
                   TII->get(AMDGPU::V_XOR_B16_t16_e64), Dst)
               .addImm(0)
               .addReg(RegA)
               .addImm(0)
               .addReg(RegB)
               .addImm(0));
    };
    Build(RegA);
    Build(RegB);
    Build(RegA);
  };

  auto SwapInChunks = [&](unsigned ElemBytes) {
    for (int16_t SubIdx : TRI->getRegSplitParts(RC, ElemBytes))
      emitSwap(MBB, InsertPt, TRI->getSubReg(RegA, SubIdx),
               TRI->getSubReg(RegB, SubIdx));
  };

  if (!TRI->isVGPRClass(RC)) {
    // SGPR: no scalar swap instruction; use an S_XOR triplet with the widest
    // available scalar XOR (B64 for 64-bit chunks, B32 otherwise). S_XOR writes
    // SCC, so resolvePermutation only routes an SGPR cycle here when SCC is
    // dead.
    if (RegWidth == 32) {
      EmitXorTriplet(AMDGPU::S_XOR_B32);
    } else if (RegWidth == 64) {
      EmitXorTriplet(AMDGPU::S_XOR_B64);
    } else {
      // Wider: cover in aligned 64-bit chunks (S_XOR_B64), with a trailing
      // 32-bit chunk (S_XOR_B32) for an odd dword count -- e.g. 96-bit -> one
      // B64 (sub0_sub1) + one B32 (sub2).
      unsigned NumDWords = RegWidth / 32;
      unsigned Ch = 0;
      for (; Ch + 2 <= NumDWords; Ch += 2) {
        unsigned Sub = SIRegisterInfo::getSubRegFromChannel(Ch, 2);
        emitSwap(MBB, InsertPt, TRI->getSubReg(RegA, Sub),
                 TRI->getSubReg(RegB, Sub));
      }
      if (Ch < NumDWords) {
        unsigned Sub = SIRegisterInfo::getSubRegFromChannel(Ch, 1);
        emitSwap(MBB, InsertPt, TRI->getSubReg(RegA, Sub),
                 TRI->getSubReg(RegB, Sub));
      }
    }
    return;
  }

  // VGPR: only 32-bit swap primitives exist; decompose wider tuples.
  // 16-bit true16 lanes (e.g. two f16 PHI values packed into one VGPR's
  // lo16/hi16) cannot use V_SWAP_B32 -- its operands are VGPR_32. Use the
  // 16-bit swap (V_SWAP_B16, present on every true16 target, which is the only
  // place 16-bit VGPR subregs are allocated), or a 16-bit XOR triplet fallback.
  if (RegWidth == 16) {
    // V_SWAP_B16 is VOP1-encoded, so both operands must lie in VGPR_16_Lo128
    // (the lo16/hi16 halves of v0-v127). Coloring is free to place a 16-bit
    // value above v127, and there the swap must go through the VOP3-encoded
    // 16-bit XOR, whose operands are unrestricted VGPR_16.
    const TargetRegisterClass &Lo128 = AMDGPU::VGPR_16_Lo128RegClass;
    if (ST->hasTrue16BitInsts() && Lo128.contains(RegA) && Lo128.contains(RegB))
      LIS->InsertMachineInstrInMaps(
          *BuildMI(MBB, InsertPt, DebugLoc(), TII->get(AMDGPU::V_SWAP_B16), RegA)
               .addDef(RegB)
               .addReg(RegB)
               .addReg(RegA));
    else if (ST->hasTrue16BitInsts())
      EmitXorTripletT16();
    else
      EmitXorTriplet(AMDGPU::V_XOR_B16_fake16_e64);
    return;
  }
  if (RegWidth <= 32) {
    if (ST->hasSwap())
      LIS->InsertMachineInstrInMaps(
          *BuildMI(MBB, InsertPt, DebugLoc(), TII->get(AMDGPU::V_SWAP_B32), RegA)
               .addDef(RegB)
               .addReg(RegB)
               .addReg(RegA));
    else
      EmitXorTriplet(AMDGPU::V_XOR_B32_e64);
    return;
  }
  SwapInChunks(4);
}

MCRegister AMDGPUSSARegisterAllocator::findLocalScratch(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
    const TargetRegisterClass *RC,
    const DenseMap<MCRegister, MCRegister> &CycleRegs) {
  SSARA_TRACE();
  // Find a physreg of RC's file that is FREE AT THIS POINT: not live across
  // InsertPt, not reserved, and not one of the cycle's own registers (which are
  // all live here by definition). Register pressure is local, so a function full
  // at its peak can still have a free reg at this cycle's point. Uses the physreg
  // liveness query (LIS is not maintained past SSA destruction).
  const TargetRegisterClass *BaseRC =
      TRI->isSGPRClass(RC)   ? &AMDGPU::SGPR_32RegClass
      : TRI->isAGPRClass(RC) ? &AMDGPU::AGPR_32RegClass
                             : &AMDGPU::VGPR_32RegClass;
  // Must use availableOrder, not the raw allocation order: the tail of the
  // vector order is withheld for the lane holders and WWM scratch that the
  // downstream spill lowering and frame lowering take off the top of the file.
  // Those registers are not yet in MRI's reserved set here, so the isReserved
  // check below does not cover them.
  for (MCRegister PR : availableOrder(BaseRC)) {
    if (MRI->isReserved(PR))
      continue;
    // Skip the cycle's own registers (live here) and any that alias them.
    bool InCycle = false;
    for (const auto &[Dst, Src] : CycleRegs)
      if (TRI->regsOverlap(PR, Dst) || TRI->regsOverlap(PR, Src)) {
        InCycle = true;
        break;
      }
    if (InCycle)
      continue;
    if (MBB.computeRegisterLiveness(TRI, PR, InsertPt) ==
        MachineBasicBlock::LQR_Dead)
      return PR;
  }
  return MCRegister();
}

void AMDGPUSSARegisterAllocator::breakCycleViaMemory(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
    MCRegister CycleStart, DenseMap<MCRegister, MCRegister> &DstToSrc) {
  SSARA_TRACE();
  // Break a permutation cycle with a MEMORY scratchpad when no free scratch
  // register exists in the cycle's file (the file is full). Mirrors the
  // register-scratch path but the "saved value" lives on the stack:
  //   store CycleStart -> stack ; walk cycle with reg copies ; reload -> last reg.
  // storeRegToStackSlot/loadRegFromStackSlot emit SI_SPILL_* pseudos; the later
  // frame lowering (SILowerSGPRSpills / eliminateFrameIndex, both after this pass)
  // supplies any intermediate VGPR (AGPR spills) and keeps EXEC/SCC safe (SGPR
  // spills), so we do not manage those here.
  const TargetRegisterClass *RC = TRI->getPhysRegBaseClass(CycleStart);
  unsigned Bits = TRI->getRegSizeInBits(*RC);
  int FI = MBB.getParent()->getFrameInfo().CreateSpillStackObject(
      Bits / 8, Align(Bits / 8));

  LLVM_DEBUG(dbgs() << "    cycle via MEMORY scratch (fi=" << FI << "), start "
                    << TRI->getName(CycleStart) << ":\n");

  // Save CycleStart to the slot (its register is about to be overwritten).
  // Keep SlotIndexes consistent: a later allocation stage queries
  // getInstructionIndex over the whole function.
  TII->storeRegToStackSlot(MBB, InsertPt, CycleStart, /*isKill=*/false, FI, RC,
                           TRI, /*VReg=*/Register());
  LIS->InsertMachineInstrInMaps(*std::prev(InsertPt));

  // Walk the cycle: each Cur gets its Src via a register copy, except the final
  // member (whose Src is CycleStart, now on the stack) which is reloaded.
  MCRegister Cur = CycleStart;
  while (true) {
    MCRegister Src = DstToSrc[Cur];
    DstToSrc.erase(Cur);
    if (!DstToSrc.count(Src)) {
      assert(Src == CycleStart && "Cycle walk did not return to start");
      TII->loadRegFromStackSlot(MBB, InsertPt, Cur, FI, RC, TRI,
                                /*VReg=*/Register());
      LIS->InsertMachineInstrInMaps(*std::prev(InsertPt));
      LLVM_DEBUG(dbgs() << "      reload: fi=" << FI << " -> "
                        << TRI->getName(Cur) << "\n");
      break;
    }
    LIS->InsertMachineInstrInMaps(
        *BuildMI(MBB, InsertPt, DebugLoc(), TII->get(TargetOpcode::COPY), Cur)
             .addReg(Src));
    LLVM_DEBUG(dbgs() << "      " << TRI->getName(Src) << " -> "
                      << TRI->getName(Cur) << "\n");
    Cur = Src;
  }
}

void AMDGPUSSARegisterAllocator::resolvePermutation(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
    SmallVectorImpl<std::pair<MCRegister, MCRegister>> &Copies) {
  SSARA_TRACE();
  if (Copies.empty())
    return;
  InsertPt = AMDGPURegAllocInsertion::legalBefore(MBB, InsertPt);

  // Decompose every copy wider than one dword into per-dword (src.subK ->
  // dst.subK) copies BEFORE building the dependence map. DstToSrc/SrcRefCount
  // key on whole-MCRegister identity, which is blind to sub-register aliasing:
  // a parallel assignment mixing a wide slice with narrow slices over the SAME
  // physical dwords (e.g. v[0:3]<-v[28:31] together with v28<-v0, v29<-v1, ...)
  // looks like unrelated map entries, so the write-after-read hazard between the
  // wide write and a narrow read of the same dword goes undetected and the naive
  // Phase-1 chain drain emits copies in an order that clobbers live values. At
  // dword granularity aliasing becomes identity, so the hazard/cycle logic below
  // is exact. Src is already narrowed to the slice width by every caller, so
  // Src and Dst share the same width here.
  SmallVector<std::pair<MCRegister, MCRegister>, 8> DwordCopies;
  for (auto &[Src, Dst] : Copies) {
    unsigned Bits = TRI->getRegSizeInBits(*TRI->getPhysRegBaseClass(Dst));
    // A copy that is at most one dword wide (Bits <= 32, i.e. a 32-bit dword or a
    // sub-dword 16-bit true16 lo16/hi16 slice) has no wider sibling to alias, so
    // it is already atomic — pass it through unchanged. Only genuinely multi-dword
    // copies (Bits > 32) need splitting so a wide write cannot hide a
    // write-after-read hazard against a narrower copy of the same dword.
    if (Bits <= 32) {
      DwordCopies.push_back({Src, Dst});
      continue;
    }
    unsigned W = Bits / 32;
    for (unsigned K = 0; K < W; ++K) {
      unsigned SubIdx = SIRegisterInfo::getSubRegFromChannel(K);
      MCRegister S = TRI->getSubReg(Src, SubIdx);
      MCRegister D = TRI->getSubReg(Dst, SubIdx);
      assert(S && D && "per-dword subregister must exist");
      if (S != D) // a slice already in place needs no copy
        DwordCopies.push_back({S, D});
    }
  }

  DenseMap<MCRegister, MCRegister> DstToSrc;
  DenseMap<MCRegister, unsigned> SrcRefCount;
  for (auto &[Src, Dst] : DwordCopies) {
    DstToSrc[Dst] = Src;
    SrcRefCount[Src]++;
  }

  // Phase 1: emit chain copies via worklist.
  // Seed with all destinations that are not sources of any remaining copy.
  SmallVector<MCRegister> Ready;
  for (auto &[Dst, Src] : DstToSrc)
    if (SrcRefCount[Dst] == 0)
      Ready.push_back(Dst);

  while (!Ready.empty()) {
    MCRegister Dst = Ready.pop_back_val();
    MCRegister Src = DstToSrc[Dst];
    DstToSrc.erase(Dst);
    LIS->InsertMachineInstrInMaps(
        *BuildMI(MBB, InsertPt, DebugLoc(), TII->get(TargetOpcode::COPY), Dst)
             .addReg(Src));
    LLVM_DEBUG(dbgs() << "    copy: " << TRI->getName(Src) << " -> "
                      << TRI->getName(Dst) << "\n");
    if (--SrcRefCount[Src] == 0 && DstToSrc.count(Src))
      Ready.push_back(Src);
  }

  // Phase 2: all remaining entries form cycles (chains were drained above).
  // A permutation cycle is always confined to one register file — a VGPR
  // destination can never equal an SGPR source — so the file (and thus the
  // scratch counter, HW limit, occupancy model and swap lowering) is derived
  // per cycle from its own registers, never from a block-wide assumption.
  MachineFunction &MF = *MBB.getParent();

  // A cycle's scratch register is TRANSIENT: it is saved at the cycle start and
  // restored (its value moved out, leaving it dead) at the cycle end, so the
  // NEXT cycle can reuse the same index. Track each file's peak scratch usage
  // separately and fold it back into the reported high-water AFTER the loop, so
  // the base index we hand each cycle is the reused (pre-scratch) high-water
  // rather than one that permanently grows per cycle (which over-counted usage
  // and tripped the no-scratch asserts early).
  unsigned PeakVGPR = MaxVGPRIdx, PeakAGPR = MaxAGPRIdx, PeakSGPR = MaxSGPRIdx;

  while (!DstToSrc.empty()) {
    // Pick any entry as cycle start — all remaining entries form disjoint
    // cycles, and the walk traces the full cycle regardless of entry point.
    MCRegister CycleStart = DstToSrc.begin()->first;

    const TargetRegisterClass *CycleRC = TRI->getPhysRegBaseClass(CycleStart);
    bool IsVGPR = TRI->isVGPRClass(CycleRC);
    bool IsAGPR = TRI->isAGPRClass(CycleRC);
    unsigned &MaxIdx =
        IsVGPR ? MaxVGPRIdx : (IsAGPR ? MaxAGPRIdx : MaxSGPRIdx);
    // AGPRs draw from the vector register budget alongside VGPRs.
    unsigned MaxHWLimit =
        (IsVGPR || IsAGPR) ? ST->getMaxNumVGPRs(MF) : ST->getMaxNumSGPRs(MF);
    // The scratch below is VGPR0 + MaxIdx, one past the colored high-water, so
    // for the VGPR file the bound has to be the allocator's own availability
    // rather than the hardware limit. The tail of the VGPR order is withheld
    // (VGPRReserve) for the lane holder and the WWM scratch that the downstream
    // spill lowering and frame lowering take off the top of the file. Once the
    // colorer fills its pool the high-water lands exactly on the first withheld
    // register, so bounding by the hardware limit hands a cycle a register the
    // WWM pass is counting on and that pass then finds nothing free. Refusing
    // the scratch is safe: the cycle breaks in place via emitSwap's V_XOR
    // triplet, which needs no scratch. AGPRs keep the hardware bound — they are
    // a separate file, and the lane holders are always VGPRs.
    if (IsVGPR)
      MaxHWLimit = std::min(MaxHWLimit, allocatablePool(MF, RegFile::VGPR));
    unsigned CurrentOcc =
        IsVGPR ? ST->getOccupancyWithNumVGPRs(MaxIdx, DynVGPRBlockSize)
               : ST->getOccupancyWithNumSGPRs(MaxIdx);

    // The scratch is ALWAYS exactly one 32-bit register. Every cycle reaching
    // Phase 2 is at most one dword wide: the DwordCopies pre-pass decomposes any
    // Bits>32 copy into per-dword copies, so a cycle is either a 32-bit value or a
    // sub-dword (16-bit true16) value — both fit a single 32-bit scratch. (16-bit
    // cycles are VGPR-only and are broken in place by emitSwap/V_SWAP_B16, never
    // via this scratch path, so the scratch here is always a clean 32-bit
    // SGPR/AGPR.) Hence the reserve is one register, not CycleWidth.
    unsigned ScratchOcc =
        IsVGPR ? ST->getOccupancyWithNumVGPRs(MaxIdx + 1, DynVGPRBlockSize)
               : ST->getOccupancyWithNumSGPRs(MaxIdx + 1);
    bool ScratchFits = MaxIdx + 1 <= MaxHWLimit;

    // Decide between resolving the cycle with a scratch register (plain COPYs)
    // and in place via emitSwap.
    //   VGPR: emitSwap (V_SWAP_B32 or a V_XOR triplet) is scratch- and
    //   SCC-free,
    //         so prefer it; use a scratch only when swap is unavailable and it
    //         costs no occupancy.
    //   SGPR: there is no scalar swap. emitSwap uses an S_XOR triplet, which
    //         writes SCC and is therefore only safe when SCC is dead here.
    //         Otherwise a scratch COPY is the only SCC-preserving option.
    // NeedMemFallback: the cycle requires a scratch register but none is free
    // (the file is full). Rather than assert, break the cycle through a MEMORY
    // scratchpad (store a member, walk with copies, reload) — this is what Greedy
    // does when both files are full. storeRegToStackSlot/loadRegFromStackSlot emit
    // SI_SPILL_* pseudos whose intermediate-register/EXEC/SCC details are handled
    // by the later frame lowering (SILowerSGPRSpills runs after this pass).
    bool UseScratch;
    bool NeedMemFallback = false;
    if (IsVGPR) {
      UseScratch = !ST->hasSwap() && ScratchOcc == CurrentOcc && ScratchFits;
    } else if (IsAGPR) {
      // AGPRs have no swap or XOR primitive, so an in-place emitSwap is
      // impossible; a scratch AGPR (plain COPYs, legalized to AGPR moves
      // downstream) is the only way to break the cycle — or, if none fits, memory.
      UseScratch = ScratchFits;
      NeedMemFallback = !ScratchFits;
    } else {
      bool SccDead = MBB.computeRegisterLiveness(TRI, AMDGPU::SCC, InsertPt) ==
                     MachineBasicBlock::LQR_Dead;
      // SCC dead -> in-place S_XOR triplet. SCC live -> need a scratch COPY
      // (S_XOR would clobber SCC); if none fits, memory fallback.
      UseScratch = !SccDead && ScratchFits;
      NeedMemFallback = !SccDead && !ScratchFits;
    }

    // Approach A: before spilling to memory, try to find an SGPR/AGPR that is
    // actually FREE AT THIS CYCLE POINT (not just below the function-wide
    // high-water). Register pressure is local: a function can be full at its peak
    // yet have a free reg at the cycle's point. This costs nothing and avoids the
    // heavy memory fallback whenever the point has any slack. Only if the point is
    // GENUINELY saturated do we fall to memory.
    MCRegister LocalScratch;
    if (NeedMemFallback) {
      LocalScratch = findLocalScratch(MBB, InsertPt, CycleRC, DstToSrc);
      if (LocalScratch) {
        NeedMemFallback = false;
        LLVM_DEBUG(dbgs() << "    local scratch found: "
                          << TRI->getName(LocalScratch) << "\n");
      }
    }

    if (NeedMemFallback) {
      breakCycleViaMemory(MBB, InsertPt, CycleStart, DstToSrc);
      continue;
    }

    if ((UseScratch && ScratchFits) || LocalScratch) {
      // Prefer a locally-free scratch (Approach A) when the high-water reg does
      // not fit; otherwise one 32-bit scratch at the current high-water.
      MCRegister Scratch =
          LocalScratch ? LocalScratch
          : IsVGPR     ? MCRegister(AMDGPU::VGPR0 + MaxIdx)
          : IsAGPR     ? MCRegister(AMDGPU::AGPR0 + MaxIdx)
                       : MCRegister(AMDGPU::SGPR0 + MaxIdx);
      // A high-water scratch transiently occupies [MaxIdx, MaxIdx + 1): record
      // that as this file's peak, but do NOT advance MaxIdx — the scratch is dead
      // after this cycle's restore, so the next cycle reuses the same base index.
      // A LocalScratch is an already-counted in-use-elsewhere reg, so it adds no
      // peak.
      if (!LocalScratch) {
        unsigned &Peak = IsVGPR ? PeakVGPR : (IsAGPR ? PeakAGPR : PeakSGPR);
        Peak = std::max(Peak, MaxIdx + 1);
      }

      LLVM_DEBUG(dbgs() << "    cycle via scratch " << TRI->getName(Scratch)
                        << ":\n");

      // Save CycleStart — it will be overwritten by the first copy.
      // The last register in the walk receives this saved value.
      LIS->InsertMachineInstrInMaps(
          *BuildMI(MBB, InsertPt, DebugLoc(), TII->get(TargetOpcode::COPY),
                   Scratch)
               .addReg(CycleStart));
      LLVM_DEBUG(dbgs() << "      save: " << TRI->getName(CycleStart) << " -> "
                        << TRI->getName(Scratch) << "\n");

      MCRegister Cur = CycleStart;
      while (true) {
        MCRegister Src = DstToSrc[Cur];
        DstToSrc.erase(Cur);
        if (!DstToSrc.count(Src)) {
          assert(Src == CycleStart && "Cycle walk did not return to start");
          LIS->InsertMachineInstrInMaps(
              *BuildMI(MBB, InsertPt, DebugLoc(), TII->get(TargetOpcode::COPY),
                       Cur)
                   .addReg(Scratch));
          LLVM_DEBUG(dbgs() << "      restore: " << TRI->getName(Scratch)
                            << " -> " << TRI->getName(Cur) << "\n");
          break;
        }
        LIS->InsertMachineInstrInMaps(
            *BuildMI(MBB, InsertPt, DebugLoc(), TII->get(TargetOpcode::COPY),
                     Cur)
                 .addReg(Src));
        LLVM_DEBUG(dbgs() << "      " << TRI->getName(Src) << " -> "
                          << TRI->getName(Cur) << "\n");
        Cur = Src;
      }
      continue;
    }

    // Tier 2/3: break cycle pairwise, in place. emitSwap picks the right op per
    // register file: VGPR -> V_SWAP_B32 (GFX9+) or a V_XOR triplet; SGPR -> an
    // S_XOR triplet (only reached when SCC is dead, per the UseScratch decision
    // above, since S_XOR writes SCC). Collect the full cycle, then emit n-1
    // swaps from tail to head.
    LLVM_DEBUG(
        dbgs() << "    cycle via "
               << (!IsVGPR ? "S_XOR" : (ST->hasSwap() ? "V_SWAP_B32" : "V_XOR"))
               << ":\n");
    SmallVector<MCRegister> Cycle;
    MCRegister Cur = CycleStart;
    while (DstToSrc.count(Cur)) {
      Cycle.push_back(Cur);
      MCRegister Next = DstToSrc[Cur];
      DstToSrc.erase(Cur);
      Cur = Next;
    }
    for (int I = Cycle.size() - 1; I > 0; --I) {
      emitSwap(MBB, InsertPt, Cycle[I - 1], Cycle[I]);
      LLVM_DEBUG(dbgs() << "      swap " << TRI->getName(Cycle[I - 1])
                        << " <-> " << TRI->getName(Cycle[I]) << "\n");
    }
  }

  // Fold each file's transient-scratch peak back into the reported high-water.
  // (No-op unless a scratch cycle raised it above the entering value.)
  MaxVGPRIdx = std::max(MaxVGPRIdx, PeakVGPR);
  MaxAGPRIdx = std::max(MaxAGPRIdx, PeakAGPR);
  MaxSGPRIdx = std::max(MaxSGPRIdx, PeakSGPR);
}

void AMDGPUSSARegisterAllocator::lowerPHIs(MachineFunction &MF, RegFile Only) {
  SSARA_TRACE();
  LLVM_DEBUG(dbgs() << "\n=== SSA Destruction ===\n");

  SmallVector<MachineInstr *, 16> PHIsToErase;

  // Step-0 metric accumulators (see PHI_Coalescer section 9). Function-local;
  // folded into the STATISTIC counters as we go so -debug-only can print a
  // per-function line without disturbing the global totals.
  unsigned FnCopies = 0, FnFixed = 0, FnUndef = 0;
  uint64_t FnWeight = 0;

  for (MachineBasicBlock &MBB : MF) {
    if (MBB.empty() || !MBB.front().isPHI())
      continue;

    DenseMap<MachineBasicBlock *,
             SmallVector<std::pair<MCRegister, MCRegister>>>
        PredCopies;

    for (MachineInstr &MI : MBB) {
      if (!MI.isPHI())
        break;

      Register DstVReg = MI.getOperand(0).getReg();
      // Two-stage lowering: handle only this stage's file; the other file's PHIs
      // are lowered (and erased) in its own stage.
      if (fileOf(MRI->getRegClass(DstVReg)) != Only)
        continue;
      MCRegister DstPhys = ColorMap.lookup(DstVReg);
      assert(DstPhys && "PHI result not colored");

      // The PHI result physreg flows into this block from each predecessor.
      // After the PHI is erased, the block has no definition of DstPhys, so
      // we must declare it as a live-in so the verifier recognises it.
      if (!MBB.isLiveIn(DstPhys))
        MBB.addLiveIn(DstPhys);

      for (unsigned I = 1, E = MI.getNumOperands(); I < E; I += 2) {
        MachineOperand &SrcMO = MI.getOperand(I);
        MachineBasicBlock *Pred = MI.getOperand(I + 1).getMBB();
        ++NumPhiOperands;

        // An undef incoming value needs no copy, but DstPhys must still be
        // defined so it is live-out of Pred (DstPhys is a live-in of MBB).
        // Encode it as a copy with a null source; it is emitted as an
        // IMPLICIT_DEF of DstPhys during copy insertion below (as generic
        // PHIElimination does for undef PHI operands).
        if (SrcMO.isUndef()) {
          PredCopies[Pred].push_back({MCRegister(), DstPhys});
          ++NumPhiUndefEdges;
          ++FnUndef;
          continue;
        }

        MCRegister SrcPhys = ColorMap.lookup(SrcMO.getReg());
        assert(SrcPhys && "PHI source not colored");

        // A PHI source may name a subregister (e.g. %x.sub0). The copy must
        // move the corresponding sub-physreg, not the full tuple, otherwise we
        // emit an illegal width-mismatched copy.
        if (unsigned SubIdx = SrcMO.getSubReg()) {
          SrcPhys = TRI->getSubReg(SrcPhys, SubIdx);
          assert(SrcPhys && "Invalid subreg index on PHI source");
        }

        if (SrcPhys != DstPhys) {
          PredCopies[Pred].push_back({SrcPhys, DstPhys});
          // Not a fixed point: a copy will be emitted on this edge. Weight it
          // by 2^loopdepth(Pred) so loop-carried copies dominate the cost, per
          // the paper's cost_f (eq.1).
          ++NumPhiCopies;
          ++FnCopies;
          unsigned Depth = MLI->getLoopDepth(Pred);
          uint64_t W = Depth < 63 ? (uint64_t(1) << Depth) : ~uint64_t(0);
          NumPhiCopyWeight += W;
          FnWeight += W;
          // Feasibility ceiling: a copy can only ever become a fixed point if
          // the operand does not interfere with the PHI result. The operand may
          // read only a slice of a wider value (e.g. %x.sub0), so interference
          // must be tested at LANE granularity, not whole-vreg: a sibling lane
          // of the source can be live across the result's range while the READ
          // lane is not. Restrict the source interval to the operand's lane mask
          // (subranges are always present -- GCN enables subreg liveness
          // unconditionally) and overlap only those lanes with the result.
          const LiveInterval &SrcLI = LIS->getInterval(SrcMO.getReg());
          const LiveInterval &DstLI = LIS->getInterval(DstVReg);
          LaneBitmask ReadMask =
              SrcMO.getSubReg()
                  ? TRI->getSubRegIndexLaneMask(SrcMO.getSubReg())
                  : MRI->getMaxLaneMaskForVReg(SrcMO.getReg());
          bool Interferes;
          if (SrcLI.hasSubRanges()) {
            Interferes = false;
            for (const LiveInterval::SubRange &S : SrcLI.subranges())
              if ((S.LaneMask & ReadMask).any() && S.overlaps(DstLI)) {
                Interferes = true;
                break;
              }
          } else {
            // Whole-register value (no subranges): the read covers all lanes.
            Interferes = SrcLI.overlaps(DstLI);
          }
          if (Interferes)
            ++NumPhiCopyInfeasible;
          else
            ++NumPhiCopyFeasible;
          if (SrcMO.getSubReg())
            ++NumPhiCopySubreg; // keep the tuple-source tally for context
        } else {
          // SrcPhys == DstPhys: already a fixed point, no copy. This is exactly
          // what Option B / the coalescer manufactures.
          ++NumPhiFixedPoints;
          ++FnFixed;
        }
      }

      PHIsToErase.push_back(&MI);
    }
    MBB.sortUniqueLiveIns();

    for (auto &[Pred, Copies] : PredCopies) {
      MachineBasicBlock *InsertMBB = Pred;
      // The split decision covers null-source (IMPLICIT_DEF) entries too:
      // edgeCopiesNeedSplit only inspects the destination of each pair.
      if (edgeCopiesNeedSplit(Pred, &MBB, Copies)) {
        LLVM_DEBUG(dbgs() << "  Splitting critical edge "
                          << printMBBReference(*Pred) << " -> "
                          << printMBBReference(MBB) << "\n");
        InsertMBB = Pred->SplitCriticalEdge(&MBB, *this);
        assert(InsertMBB && "Failed to split critical edge");
      }

      LLVM_DEBUG(dbgs() << "  Edge " << printMBBReference(*InsertMBB) << " -> "
                        << printMBBReference(MBB) << ":\n");
      auto InsertPt = AMDGPURegAllocInsertion::bodyEnd(*InsertMBB);
      // Materialize undef edges (null source) as IMPLICIT_DEF of DstPhys and
      // drop them; the remainder are real copies handed to resolvePermutation.
      for (auto *It = Copies.begin(); It != Copies.end();) {
        if (!It->first) {
          MachineInstr *IDef =
              BuildMI(*InsertMBB, InsertPt, DebugLoc(),
                      TII->get(TargetOpcode::IMPLICIT_DEF), It->second);
          LIS->InsertMachineInstrInMaps(*IDef);
          It = Copies.erase(It);
        } else {
          ++It;
        }
      }
      resolvePermutation(*InsertMBB, InsertPt, Copies);
    }
  }

  for (MachineInstr *PHI : PHIsToErase) {
    // Keep SlotIndexes consistent: a later allocation stage queries
    // getInstructionIndex over the whole function, so an erased instr must leave
    // the maps.
    if (Indexes->hasIndex(*PHI))
      LIS->RemoveMachineInstrFromMaps(*PHI);
    PHI->eraseFromParent();
  }

  LLVM_DEBUG(dbgs() << "  Erased " << PHIsToErase.size() << " PHIs\n");

  // Per-function metric line (opt-in): a diff of two llc runs is a diff of these
  // lines. Gated on its own debug type so it is independent of the pass's
  // verbose -debug-only=amdgpu-ssa-register-allocator output.
  DEBUG_WITH_TYPE(PHI_METRIC_DEBUG_TYPE,
                  dbgs() << "phi-metric " << MF.getName() << ": copies="
                         << FnCopies << " fixed=" << FnFixed
                         << " undef=" << FnUndef << " weighted=" << FnWeight
                         << "\n");
}

void AMDGPUSSARegisterAllocator::rewriteOperands(MachineFunction &MF,
                                                 RegFile Only) {
  SSARA_TRACE();
  LLVM_DEBUG(dbgs() << "\n=== Operand Rewrite ===\n");

  for (MachineBasicBlock &MBB : MF) {
    // Use instrs() so operands of instructions *inside* BUNDLEs are rewritten
    // too (e.g. GWS ops: `BUNDLE implicit %r { DS_GWS_INIT %r, ... }`). Plain
    // MBB iteration visits only bundle headers, leaving the bundled
    // instruction's virtual operands un-rewritten ("Remaining virtual register").
    for (MachineInstr &MI : MBB.instrs()) {
      for (MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg().isVirtual())
          continue;

        Register VReg = MO.getReg();
        // Two-stage rewrite: the SGPR stage rewrites only SGPR vregs (leaving
        // VGPR vregs virtual for the VGPR stage), and vice versa.
        if (fileOf(MRI->getRegClass(VReg)) != Only)
          continue;
        MCRegister PhysReg = ColorMap.lookup(VReg);
        if (!PhysReg) {
          // A debug instruction does not compute a program value, and every
          // coloring path deliberately ignores a vreg with no non-debug use
          // (MRI->reg_nodbg_empty), so such a vreg never receives a color. The
          // location is simply unavailable: drop it, the same treatment
          // RegAllocFast gives a register that did not survive.
          if (MI.isDebugInstr()) {
            if (MI.isDebugValue()) {
              MI.setDebugValueUndef();
            } else {
              MO.setReg(Register());
              MO.setSubReg(0);
            }
            continue;
          }
          // A vreg that only ever appears as an `undef` operand has no value to
          // color (no def drives it). Its content is a don't-care; assign any
          // allocatable physreg of its class so the operand is well-formed. The
          // `undef` flag is preserved by setReg, so the verifier permits the
          // read of an otherwise-undefined physreg.
          assert(MO.isUndef() && "non-undef virtual register not colored");
          unsigned DefOpIdx;
          if (MO.isUse() &&
              MI.isRegTiedToDefOperand(MO.getOperandNo(), &DefOpIdx)) {
            // An undef use tied to a def (e.g. the DPP/PERMLANE "old" source
            // read as `undef %N.subX` where %N is never otherwise defined) has
            // a don't-care value, but two-address form still requires it to
            // equal the def. The def operand precedes this use and is already
            // rewritten to its physreg, which is the correct width for the tied
            // slot, so copy it verbatim and drop any sub-register.
            MCRegister DefPhys = MI.getOperand(DefOpIdx).getReg();
            assert(DefPhys.isPhysical() && "tied def not yet rewritten");
            MO.setSubReg(0);
            MO.setReg(DefPhys);
            continue;
          }
          const TargetRegisterClass *RC = MRI->getRegClass(VReg);
          ArrayRef<MCPhysReg> Order = RegClassInfo.getOrder(RC);
          assert(!Order.empty() && "empty allocation order for undef operand");
          PhysReg = Order.front();
        }

        unsigned OrigSubIdx = MO.getSubReg();
        if (OrigSubIdx) {
          PhysReg = TRI->getSubReg(PhysReg, OrigSubIdx);
          assert(PhysReg && "Invalid subreg index");
          MO.setSubReg(0);
        }

        // DEAD-LANE UNDEF PROPAGATION (reaching-VNI). Check the lanes THIS operand
        // actually reads — the full reg mask, or the subreg's lane mask for a
        // sub-tuple read (e.g. %x.sub0_sub1 of a 128b value). If ANY read lane has
        // no reaching value at the use, the read is partial-undef; in virtual MIR
        // the vreg's per-subrange liveness makes that legal, but once rewritten to
        // the physical tuple the dead lane's physreg looks read-but-never-defined
        // and LIS/verifier reject it ("needs to be live in ... missing from
        // live-in list"). LLVM's VirtRegRewriter marks such a read `undef`; match
        // it. Query each subrange for the VNInfo reaching the use (getVNInfoBefore
        // — the same reaching-VNI idiom splitLiveRangeAt / the emitter use).
        if (MO.isUse() && !MO.isUndef() && LIS->hasInterval(VReg)) {
          const LiveInterval &LI = LIS->getInterval(VReg);
          if (LI.hasSubRanges()) {
            SlotIndex UseIdx = LIS->getInstructionIndex(MI).getRegSlot();
            LaneBitmask ReadMask =
                OrigSubIdx ? TRI->getSubRegIndexLaneMask(OrigSubIdx)
                           : MRI->getMaxLaneMaskForVReg(VReg);
            LaneBitmask Reached;
            for (const LiveInterval::SubRange &S : LI.subranges())
              if (S.getVNInfoBefore(UseIdx))
                Reached |= S.LaneMask;
            if ((ReadMask & ~Reached).any()) // a READ lane has no reaching def
              MO.setIsUndef(true);
          }
        }
        MO.setReg(PhysReg);
      }
    }
  }
}

// Update MBB live-in sets with the physical registers assigned to virtual
// registers that are live at each block entry. VirtRegRewriter does this in
// the greedy RA path; without it the machine verifier reports "Using an
// undefined physical register" for cross-block physreg uses.
void AMDGPUSSARegisterAllocator::addPhysRegLiveIns(MachineFunction &MF) {
  SSARA_TRACE();
  for (MachineBasicBlock &MBB : MF) {
    SlotIndex BBStart = LIS->getMBBStartIdx(&MBB);
    for (const auto &[VReg, PhysReg] : ColorMap) {
      if (LIS->getInterval(VReg).liveAt(BBStart)) {
        if (!MBB.isLiveIn(PhysReg))
          MBB.addLiveIn(PhysReg);
      }
    }
    MBB.sortUniqueLiveIns();
  }
}

// Set all MachineFunction properties that downstream passes require after
// SSA destruction and physical register assignment are complete.
// Mirrors the state produced by VirtRegRewriter in the greedy RA path:
//   NoPHIs     — all PHI instructions removed by lowerPHIs()
//   NoVRegs    — all virtual registers replaced with physregs by
//   rewriteOperands() IsSSA      — cleared by leaveSSA() (not SSA anymore)
// TracksLiveness is deliberately preserved: MBB live-in sets contain only
// physregs and remain valid after the rewrite; clearing it would break
// post-RA passes such as MachineLICM that call livein_begin().
void AMDGPUSSARegisterAllocator::finalizeProperties(MachineFunction &MF) {
  SSARA_TRACE();
  MRI->leaveSSA();
  // Remove all virtual register declarations from MRI so that the verifier's
  // NoVRegs check (MRI->getNumVirtRegs() == 0) passes. VirtRegRewriter does
  // the same in the greedy RA path. Instruction operands are already physical
  // after rewriteOperands(); this only removes the stale vreg table entries.
  MRI->clearVirtRegs();
  MF.getProperties().set(MachineFunctionProperties::Property::NoPHIs);
  MF.getProperties().set(MachineFunctionProperties::Property::NoVRegs);
  // SSA RA gives each tied def the same physreg as its tied use, restoring
  // two-address form (as VirtRegRewriter does on the greedy path).
  MF.getProperties().set(MachineFunctionProperties::Property::TiedOpsRewritten);
}

// Eliminate REG_SEQUENCE instructions after physreg assignment.
// In the greedy RA path, VirtRegRewriter handles this. We skip VirtRegRewriter,
// so REG_SEQUENCEs that survived into post-RA MIR must be lowered here.
//
// A REG_SEQUENCE:  dst = REG_SEQUENCE src0, sub0, src1, sub1, ...
// is "trivial" if for every (src_i, sub_i): src_i == TRI->getSubReg(dst,
// sub_i). Trivial ones are deleted. Non-trivial ones are lowered to COPY
// instructions placed immediately before the REG_SEQUENCE, then the
// REG_SEQUENCE is deleted.
void AMDGPUSSARegisterAllocator::markRegSequenceUndefLaneUses(
    MachineFunction &MF, RegFile Only) {
  SSARA_TRACE();
  // A REG_SEQUENCE with an `undef` source leaves the destination lanes it feeds
  // undefined. That is legal on the vreg (per-subrange liveness), but once the
  // result is rewritten to a physical tuple the dead lane looks read-but-never-
  // defined -> the post-RA LiveIntervals verifier fatals "register $vgprN_vgprN+1
  // needs to be live in ... missing from the live-in list" (the "Invalid global
  // physical register" cluster). While the REG_SEQUENCE still exists — its result
  // is a virtual register, so its uses are findable via MRI — mark each use of the
  // result that READS an undef lane `undef`, so rewriteOperands carries the flag
  // onto the physical read. Only the lanes fed by undef sources are considered, so
  // a subreg use of a live lane is left untouched.
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB) {
      if (!MI.isRegSequence())
        continue;
      Register Dst = MI.getOperand(0).getReg();
      if (!Dst.isVirtual())
        continue;
      if (fileOf(MRI->getRegClass(Dst)) != Only)
        continue; // handled in the other file's stage
      LaneBitmask UndefLanes;
      for (unsigned I = 1, E = MI.getNumOperands(); I + 1 < E; I += 2)
        if (MI.getOperand(I).isUndef())
          UndefLanes |= TRI->getSubRegIndexLaneMask(MI.getOperand(I + 1).getImm());
      if (UndefLanes.none())
        continue;
      for (MachineOperand &UseMO : MRI->use_operands(Dst)) {
        if (UseMO.isUndef())
          continue;
        LaneBitmask ReadMask =
            UseMO.getSubReg() ? TRI->getSubRegIndexLaneMask(UseMO.getSubReg())
                              : MRI->getMaxLaneMaskForVReg(Dst);
        if ((ReadMask & UndefLanes).any())
          UseMO.setIsUndef(true);
      }
    }
}

void AMDGPUSSARegisterAllocator::eliminateRegSequences(MachineFunction &MF) {
  SSARA_TRACE();
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      if (!MI.isRegSequence())
        continue;

      // Two-stage rewrite: a REG_SEQUENCE whose result is still VIRTUAL belongs
      // to a file not yet rewritten (the other stage lowers it once its operands
      // are physical). Only lower RS whose result was rewritten to a physreg by
      // this stage's rewriteOperands.
      if (MI.getOperand(0).getReg().isVirtual())
        continue;
      MCRegister Dst = MI.getOperand(0).getReg().asMCReg();
      LLVM_DEBUG(dbgs() << "  [RegSeq] lowering " << MI);

      // A REG_SEQUENCE is a *parallel* assignment: all sources are read, then
      // each is written to its destination slice. Collect the non-trivial
      // (Src -> dst-slice) pairs and hand them to resolvePermutation, which
      // sequences them to respect write-after-read hazards (a slice that
      // overwrites a register another pair still needs) and cycles. Emitting
      // the copies naively in operand order corrupts such overlaps.
      SmallVector<std::pair<MCRegister, MCRegister>, 4> Copies;
      for (unsigned I = 1, E = MI.getNumOperands(); I < E; I += 2) {
        // An undef source slice (e.g. `undef %175.sub0`) is a don't-care: the
        // destination lanes it would fill are never read, so emit no copy.
        // Lowering it would COPY from the undef value's physreg, which is never
        // defined -> "Using an undefined physical register".
        if (MI.getOperand(I).isUndef())
          continue;
        MCRegister Src = MI.getOperand(I).getReg().asMCReg();
        unsigned SubIdx = MI.getOperand(I + 1).getImm();
        // The source class may be wider than the slice it fills (e.g. a 64-bit
        // value held in an sgpr_128 vreg). The slice index then also names the
        // matching sub-register of the source — narrow Src to it so the COPY is
        // width-correct. When Src already matches the slice width, SubIdx names
        // no sub-register of Src and getSubReg() returns 0, leaving Src as-is.
        if (MCRegister SubSrc = TRI->getSubReg(Src, SubIdx))
          Src = SubSrc;
        MCRegister Expected = TRI->getSubReg(Dst, SubIdx);
        if (Expected) {
          if (Src != Expected)
            Copies.push_back({Src, Expected});
          continue;
        }
        // SubIdx names no physical subregister of Dst: alignment-constrained
        // files have no tuple at this offset (SGPR tuples >=64-bit are generated
        // at aligned bases only, e.g. sub1_sub2 of an SGPR_96 == s1_2 does not
        // exist). Lower it as per-dword 32-bit copies, whose subregisters always
        // exist. Src is exactly the slice width here, so its dwords map 1:1 onto
        // the destination dwords of the slice.
        unsigned First = TRI->getChannelFromSubReg(SubIdx);
        unsigned NumDW = TRI->getSubRegIdxSize(SubIdx) / 32;
        for (unsigned K = 0; K < NumDW; ++K) {
          MCRegister D =
              TRI->getSubReg(Dst, SIRegisterInfo::getSubRegFromChannel(First + K));
          MCRegister S =
              (NumDW == 1)
                  ? Src
                  : TRI->getSubReg(Src, SIRegisterInfo::getSubRegFromChannel(K));
          assert(D && S && "per-dword subregister must exist");
          if (S != D)
            Copies.push_back({S, D});
        }
      }
      resolvePermutation(MBB, MI, Copies);
      // Keep SlotIndexes consistent for a later allocation stage's queries.
      if (Indexes->hasIndex(MI))
        LIS->RemoveMachineInstrFromMaps(MI);
      MI.eraseFromParent();
    }
  }
}

// A COPY that predates this pass becomes a no-op when both of its operands are
// colored to the same physical register: a kernel argument's live-in copy
// (`%v = COPY $sgpr4_sgpr5`, colored straight back onto $sgpr4_sgpr5), or a
// live-range split/narrow copy from the spill emitter whose tail legitimately
// lands back on the head's register. Leaving it in the output emits a
// register-to-itself move that nothing downstream is obliged to remove.
void AMDGPUSSARegisterAllocator::eliminateIdentityCopies(MachineFunction &MF) {
  SSARA_TRACE();
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      if (!MI.isCopy())
        continue;
      // A COPY carrying extra implicit operands is not a pure register move;
      // erasing it would drop them.
      if (MI.getNumOperands() != 2)
        continue;
      const MachineOperand &Dst = MI.getOperand(0);
      const MachineOperand &Src = MI.getOperand(1);
      // A virtual operand belongs to the file this stage has not rewritten yet;
      // that file's own stage decides. A cross-file copy is never identity.
      if (!Dst.getReg().isPhysical() || !Src.getReg().isPhysical())
        continue;
      // rewriteOperands folds a sub-register index into the physreg it names, so
      // a physical operand here is a whole register and comparison is exact.
      assert(!Dst.getSubReg() && !Src.getSubReg() &&
             "physical operand kept a sub-register index");
      if (Dst.getReg() != Src.getReg())
        continue;
      // An undef source defines nothing, so this copy is the only thing that
      // makes the register appear defined. Whether that read is dead is not this
      // function's decision.
      if (Src.isUndef())
        continue;
      LLVM_DEBUG(dbgs() << "  [IdentityCopy] erasing " << MI);
      // Keep SlotIndexes consistent for a later allocation stage's queries.
      if (Indexes->hasIndex(MI))
        LIS->RemoveMachineInstrFromMaps(MI);
      MI.eraseFromParent();
      ++NumIdentityCopiesErased;
    }
  }
}

// (vreg, dword-lane) packed into one key. 20 bits of lane is ample (max tuple is
// 32 dwords). vreg index fits the upper bits.
static uint64_t vfKey(unsigned VReg, unsigned Lane) {
  SSARA_TRACE();
  return (uint64_t(VReg) << 20) | (Lane & 0xFFFFF);
}

uint64_t AMDGPUSSARegisterAllocator::vfFind(uint64_t X) {
  SSARA_TRACE();
  auto It = VFUF.find(X);
  if (It == VFUF.end()) {
    VFUF[X] = X;
    return X;
  }
  // Iterative find + path compression (no deep recursion on long value chains).
  uint64_t R = X;
  while (VFUF[R] != R)
    R = VFUF[R];
  while (VFUF[X] != R) {
    uint64_t Next = VFUF[X];
    VFUF[X] = R;
    X = Next;
  }
  return R;
}

void AMDGPUSSARegisterAllocator::vfUnion(uint64_t A, uint64_t B) {
  SSARA_TRACE();
  VFUF[vfFind(A)] = vfFind(B);
}

void AMDGPUSSARegisterAllocator::snapshotValueFlow(MachineFunction &MF) {
  SSARA_TRACE();
  VFIntent.clear();
  VFUF.clear();
  VFColor.clear();
  VFDefinedLane.clear();
  for (auto &[V, P] : ColorMap)
    VFColor[V] = P;

  auto Lanes = [&](Register R) {
    return TRI->getRegSizeInBits(*MRI->getRegClass(R)) / 32;
  };
  // Record the lanes of a def operand that receive a REAL value. A whole-reg def
  // covers all lanes; a sub-register def covers [channel, +subLanes). An `undef`
  // def contributes NOTHING (its lanes stay don't-care).
  auto markDefined = [&](const MachineOperand &MO) {
    if (MO.isUndef())
      return;
    Register R = MO.getReg();
    unsigned Base = MO.getSubReg() ? TRI->getChannelFromSubReg(MO.getSubReg()) : 0;
    unsigned N = MO.getSubReg()
                     ? TRI->getSubRegIdxSize(MO.getSubReg()) / 32
                     : Lanes(R);
    for (unsigned K = 0; K < std::max(1u, N); ++K)
      VFDefinedLane.insert(vfKey(R.id(), Base + K));
  };

  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB) {
      // PHI and REG_SEQUENCE are DISSOLVED by SSA destruction; their result is
      // the SAME value as their operands (per lane). Merge the value classes so
      // the final walk treats them as one token.
      if (MI.isPHI()) {
        Register R = MI.getOperand(0).getReg();
        if (!R.isVirtual())
          continue;
        for (unsigned I = 1; I + 1 < MI.getNumOperands(); I += 2) {
          Register Op = MI.getOperand(I).getReg();
          if (!Op.isVirtual())
            continue;
          for (unsigned K = 0, E = Lanes(R); K < E; ++K)
            vfUnion(vfKey(R.id(), K), vfKey(Op.id(), K));
        }
        continue;
      }
      if (MI.isRegSequence()) {
        Register D = MI.getOperand(0).getReg();
        if (!D.isVirtual())
          continue;
        for (unsigned I = 1; I + 1 < MI.getNumOperands(); I += 2) {
          const MachineOperand &SrcMO = MI.getOperand(I);
          Register S = SrcMO.getReg();
          if (!S.isVirtual())
            continue;
          unsigned Base =
              TRI->getChannelFromSubReg(MI.getOperand(I + 1).getImm());
          // The source may itself read a sub-register (e.g. `%84.sub1`): its
          // lanes start at that channel, not lane 0. The slice width is the
          // dest sub-register's size.
          unsigned SrcBase =
              SrcMO.getSubReg() ? TRI->getChannelFromSubReg(SrcMO.getSubReg())
                                : 0;
          unsigned N = TRI->getSubRegIdxSize(MI.getOperand(I + 1).getImm()) / 32;
          for (unsigned K = 0, E = std::max(1u, N); K < E; ++K)
            vfUnion(vfKey(D.id(), Base + K), vfKey(S.id(), SrcBase + K));
        }
        continue;
      }
      // Ordinary instruction: record which value each vreg operand should carry,
      // by the STABLE MachineInstr* (survives rewriteOperands, which edits in
      // place). Skip fully-undef operands (don't-care value); record which lanes
      // each def really writes (markDefined skips undef defs).
      auto &Ops = VFIntent[&MI];
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg().isVirtual() || MO.isUndef())
          continue;
        Ops.push_back({MO.getReg().id(), MO.getSubReg(), MO.isDef()});
        if (MO.isDef())
          markDefined(MO);
      }
    }

  // Canonicalize the defined-lane set: a use lane is checkable iff SOME lane in
  // its union class (PHI/REG_SEQUENCE-merged) received a real def.
  DenseSet<uint64_t> Canon;
  for (uint64_t K : VFDefinedLane)
    Canon.insert(vfFind(K));
  VFDefinedLane = std::move(Canon);
}

bool AMDGPUSSARegisterAllocator::verifyValueFlow(MachineFunction &MF) {
  SSARA_TRACE();
  // v1: single-basic-block functions only. Multi-block needs the token-meet at
  // joins (+ dominance rescue on the ⊤ residue) — reported SKIP for now.
  if (MF.size() != 1) {
    LLVM_DEBUG(dbgs() << "[value-flow] SKIP multi-block " << MF.getName()
                      << "\n");
    return false;
  }
  MachineBasicBlock &MBB = MF.front();

  // Value tokens are tracked per 32-bit DWORD, not per reg-unit: on true16
  // targets a VGPR_32 has TWO reg-units (lo16/hi16) but exactly one value; and
  // the vfKey lane space is dwords. dwordKeys(P) returns one stable key per
  // dword of P — the first reg-unit of that dword's sub-register — so Actual and
  // the vfKey lane index agree at dword granularity.
  auto dwordKeys = [&](MCRegister P) {
    SmallVector<MCRegUnit, 8> Keys;
    unsigned Bits = TRI->getRegSizeInBits(*TRI->getPhysRegBaseClass(P));
    if (Bits <= 32) { // one dword or a sub-dword true16 slice: single key
      Keys.push_back(*TRI->regunits(P).begin());
      return Keys;
    }
    for (unsigned K = 0, NW = Bits / 32; K < NW; ++K) {
      MCRegister D = TRI->getSubReg(P, SIRegisterInfo::getSubRegFromChannel(K));
      Keys.push_back(*TRI->regunits(D).begin());
    }
    return Keys;
  };

  // Actual[dwordKey] = canonical value token currently held. High-bit spaces are
  // used for "unknown but consistent" seeds so they never collide with vfKey
  // value tokens.
  DenseMap<MCRegUnit, uint64_t> Actual;
  for (const auto &LI : MBB.liveins())
    for (MCRegUnit U : dwordKeys(LI.PhysReg))
      Actual[U] = (uint64_t(3) << 40) | U; // live-in: its own stable token

  // Map an original operand's (vreg, subreg) to its physical (sub)register and
  // the per-lane canonical token, using the frozen ColorMap.
  auto physOf = [&](const VFOp &O) -> MCRegister {
    MCRegister P = VFColor.lookup(O.VReg);
    if (O.SubReg && P)
      P = TRI->getSubReg(P, O.SubReg);
    return P;
  };
  auto laneBase = [&](unsigned SubReg) -> unsigned {
    return SubReg ? TRI->getChannelFromSubReg(SubReg) : 0;
  };

  unsigned Violations = 0;
  for (MachineInstr &MI : MBB) {
    auto It = VFIntent.find(&MI);
    if (It != VFIntent.end()) {
      // ORIGINAL instruction: check every use holds its intended value, THEN
      // apply defs (a two-address def reads its use first).
      for (const VFOp &O : It->second) {
        if (O.IsDef)
          continue;
        MCRegister P = physOf(O);
        if (!P)
          continue;
        unsigned Base = laneBase(O.SubReg);
        unsigned K = 0;
        for (MCRegUnit U : dwordKeys(P)) {
          uint64_t Want = vfFind(vfKey(O.VReg, Base + K));
          // Skip lanes that never received a real def (partial/undef value): a
          // read of such a lane is a don't-care, not a clobber.
          if (!VFDefinedLane.count(Want)) {
            ++K;
            continue;
          }
          auto A = Actual.find(U);
          if (A == Actual.end() || A->second != Want) {
            errs() << "[value-flow] CLOBBER: " << printRegUnit(U, TRI)
                   << " holds wrong value for " << printReg(O.VReg, TRI)
                   << " at:  " << MI;
            ++Violations;
            break; // one report per operand
          }
          ++K;
        }
      }
      for (const VFOp &O : It->second) {
        if (!O.IsDef)
          continue;
        MCRegister P = physOf(O);
        if (!P)
          continue;
        unsigned Base = laneBase(O.SubReg);
        unsigned K = 0;
        for (MCRegUnit U : dwordKeys(P))
          Actual[U] = vfFind(vfKey(O.VReg, Base + K++));
      }
      continue;
    }

    // INSERTED instruction (COPY / V_SWAP_B32 / spill save-restore / reg move):
    // propagate tokens. Read ALL sources from the pre-instruction state, then
    // apply ALL writes atomically (so a swap/permutation re-homes tokens without
    // a spurious mid-step "no holder" state).
    SmallVector<std::pair<MCRegUnit, uint64_t>, 8> Writes;
    auto tokOf = [&](MCRegUnit U) -> uint64_t {
      auto A = Actual.find(U);
      return A == Actual.end() ? 0 : A->second; // 0 = unknown/⊤
    };
    if (MI.isCopy()) {
      MCRegister D = MI.getOperand(0).getReg();
      MCRegister S = MI.getOperand(1).getReg();
      if (unsigned Sub = MI.getOperand(1).getSubReg())
        S = TRI->getSubReg(S, Sub);
      if (unsigned Sub = MI.getOperand(0).getSubReg())
        D = TRI->getSubReg(D, Sub);
      if (D && S) {
        SmallVector<MCRegUnit, 8> DK = dwordKeys(D), SK = dwordKeys(S);
        for (unsigned K = 0; K < DK.size(); ++K)
          Writes.push_back({DK[K], K < SK.size() ? tokOf(SK[K]) : 0});
      }
    } else {
      // General reg-to-reg case (V_SWAP_B32, V_MOV_B32/B64 reg, etc.): match def
      // operands to explicit reg-use operands positionally, per dword. Reads are
      // snapshotted before writes are applied (atomic).
      SmallVector<MCRegister, 4> Defs, Uses;
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg() || !MO.getReg().isPhysical())
          continue;
        if (MO.isDef())
          Defs.push_back(MO.getReg());
        else if (MO.readsReg() && !MO.isImplicit())
          Uses.push_back(MO.getReg());
      }
      // Only handle the shapes SSA-destruction emits: N defs paired with N uses
      // (swap: 2/2; move: 1/1). Anything else → treat defs as fresh unknowns
      // (a later original use against them carries real intent and will report if
      // genuinely wrong; fresh is only reached for opaque inserts).
      if (!Defs.empty() && Defs.size() == Uses.size()) {
        SmallVector<SmallVector<uint64_t, 4>, 4> SrcToks;
        for (MCRegister S : Uses) {
          SrcToks.emplace_back();
          for (MCRegUnit U : dwordKeys(S))
            SrcToks.back().push_back(tokOf(U));
        }
        for (unsigned I = 0; I < Defs.size(); ++I) {
          unsigned J = 0;
          for (MCRegUnit DK : dwordKeys(Defs[I]))
            Writes.push_back({DK, J < SrcToks[I].size() ? SrcToks[I][J++] : 0});
        }
      } else {
        for (MCRegister D : Defs)
          for (MCRegUnit DK : dwordKeys(D))
            Writes.push_back({DK, (uint64_t(7) << 40) | DK}); // fresh unknown
      }
    }
    for (auto &[U, T] : Writes)
      Actual[U] = T;
  }

  if (Violations) {
    errs() << "[value-flow] " << Violations << " violation(s) in "
           << MF.getName() << "\n";
    if (VerifyValueFlowFatal)
      report_fatal_error("value-flow violations detected");
  }
  return false;
}

void AMDGPUSSARegisterAllocator::rewriteStage(MachineFunction &MF,
                                              RegFile Only) {
  SSARA_TRACE();
  // Rewrite ONE file's vregs to physregs and lower its PHIs / REG_SEQUENCEs.
  // The other file stays virtual for its own stage; eliminateRegSequences skips
  // a still-virtual (other-file) RS result. The driver calls this per stage,
  // between that stage's coloring and the next stage's.
  lowerPHIs(MF, Only);
  markRegSequenceUndefLaneUses(MF, Only);
  rewriteOperands(MF, Only);
  eliminateRegSequences(MF);
  eliminateIdentityCopies(MF);
  // Add THIS stage's cross-block physreg live-ins now, while its ColorMap is
  // still intact (the next stage clears ColorMap). addPhysRegLiveIns reads
  // ColorMap + LIS, so a single end-of-run call would miss the earlier stage's
  // entries. It only ADDS (sortUniqueLiveIns), so running per stage is safe.
  addPhysRegLiveIns(MF);
}

void AMDGPUSSARegisterAllocator::finalizeAfterRewrite(MachineFunction &MF) {
  SSARA_TRACE();
  // Run ONCE after both files are physical.
  finalizeProperties(MF);
  if (EnableVerifyValueFlow)
    verifyValueFlow(MF); // AFTER: everything physical
}

// === Main entry point ===

// Callee-saved SGPRs are spilled by SILowerSGPRSpills into PHYSICAL VGPR lanes,
// a lane space separate from the virtual lane holders, and those holders are
// taken from the free registers (findUnusedRegister, highest first) BEFORE the
// WWM reservation is computed. The allocation has to leave room for them too,
// or that reservation comes up short. Counted exactly as spillCalleeSavedRegs
// does: one lane per saved register of the target's callee-saved list. Entry
// functions have an empty list and so cost nothing.
static unsigned countCalleeSavedSGPRLanes(MachineFunction &MF) {
  SSARA_TRACE();
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  BitVector SavedRegs;
  ST.getFrameLowering()->determineCalleeSavesSGPR(MF, SavedRegs);
  unsigned Lanes = 0;
  for (const MCPhysReg *CSRegs = MRI.getCalleeSavedRegs(); *CSRegs; ++CSRegs)
    if (SavedRegs.test(*CSRegs))
      ++Lanes;
  return Lanes;
}

bool AMDGPUSSARegisterAllocator::runOnMachineFunction(MachineFunction &MF) {
  SSARA_TRACE();
  TRI =
      static_cast<const SIRegisterInfo *>(MF.getSubtarget().getRegisterInfo());
  TII = static_cast<const SIInstrInfo *>(MF.getSubtarget().getInstrInfo());
  MRI = &MF.getRegInfo();
  ST = &MF.getSubtarget<GCNSubtarget>();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  RegClassInfo.runOnMachineFunction(MF);
  DynVGPRBlockSize =
      ST->isDynamicVGPREnabled() ? ST->getDynamicVGPRBlockSize() : 0;

  // SI control-flow pseudos (SI_IF/ELSE/LOOP/END_CF/IF_BREAK) must be lowered
  // before register allocation. If any survive, the pass pipeline is broken —
  // there is nothing sound to do (SSA destruction cannot run with unlowered CF).
  // Fail loudly rather than silently coloring and skipping the rewrite.
  if (hasCFPseudos(MF))
    report_fatal_error("AMDGPUSSARegisterAllocator: SI control-flow pseudos not "
                       "lowered before register allocation (broken pipeline)");

  LLVM_DEBUG(dbgs() << "AMDGPUSSARegisterAllocator: Processing " << MF.getName()
                    << "\n");

  // Forensic reporter (observer, default off). Created ONCE per pass instance
  // (not per function): the sink files are opened once and every function in the
  // module appends one NDJSON line with an incrementing reportID. Recreating it
  // per function would reopen+truncate the sink and reset the counter, dropping
  // all but the last function. E1 RunStarted records the run's identity BEFORE
  // any allocation, while the MIR is still the untouched input.
  if (!Reporter)
    Reporter = std::make_unique<SSAForensicReporter>();
  Reporter->beginRun(MF, TRI, MRI, LIS);

  // Approach-A emitter: spill values that coloring cannot place.
  Indexes = &getAnalysis<SlotIndexesWrapperPass>().getSI();
  Emitter = std::make_unique<SSASpillEmitter>(MF, LIS, Indexes, MDT, MLI);
  Emitter->setReporter(Reporter.get());
  Emitter->setReloadEveryUse(RecoveryOnly);

  // Erase fully-DEAD IMPLICIT_DEFs (def-only vreg, zero uses) before coloring.
  // Such an instruction produces no value, but if left in it is still colored to a
  // physreg and lowered to `dead $vgprN.. = IMPLICIT_DEF`, whose physical write
  // CLOBBERS any value live across that point (the colorer does not mark a dead
  // def occupied, so a later-colored overlapping value picks the same register ->
  // "Using an undefined physical register" at the clobbered value's next use).
  // Removing it is semantics-preserving (no uses) and eliminates the phantom
  // clobber at the source.
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      if (!MI.isImplicitDef())
        continue;
      Register D = MI.getOperand(0).getReg();
      if (!D.isVirtual() || !MRI->use_nodbg_empty(D))
        continue; // has a use (real undef source) -> keep
      LIS->RemoveMachineInstrFromMaps(MI);
      if (LIS->hasInterval(D))
        LIS->removeInterval(D);
      MI.eraseFromParent();
    }

  widenToAVOnUnified(); // before classifyVRegs: widened widths feed the order
  classifyVRegs();

  if (EnableLaneWasteDump)
    reportLaneWaste(MF);
  if (EnableBlockDemandDump)
    reportBlockDemand(MF);

  // TWO INDEPENDENT ALLOCATION STAGES: SGPR first, then VGPR/AGPR. The SGPR
  // stage may spill SGPRs; those spills lower (downstream) to VGPR lanes needing
  // WWM scratch. Between stages we reserve ceil(spilledSGPRlanes / wavesize)
  // VGPRs (VGPRReserve, withheld by allocatablePool) so the VGPR stage does not
  // consume the whole file. Each stage colors, recovers, and rewrites ONLY its
  // file's vregs (StageFile filters color()/preSpill/region-rp/rewriteStage);
  // the files are disjoint register sets, so this only reorders within a file.
  Emitter->clearSGPRSpillLanes();
  for (RegFile Stage : {RegFile::SGPR, RegFile::VGPR}) {
    StageFile = Stage;
    const unsigned WaveSize = ST->isWave32() ? 32u : 64u;
    VGPRReserve =
        (Stage == RegFile::VGPR)
            ? divideCeil(Emitter->numSGPRSpillLanes(), WaveSize) +
                  divideCeil(countCalleeSavedSGPRLanes(MF), WaveSize)
            : 0;
  // allocStride reads availableOrder, which depends on RegClassInfo and on the
  // reserve just set above. Keyed only by register class, so it must not outlive
  // either of them.
  StrideCache.clear();
  OccupiedRegUnits.clear();
  OccupiedRegUnits.resize(TRI->getNumRegUnits());
  ColorMap.clear();
  MaxVGPRIdx = 0;
  MaxSGPRIdx = 0;
  MaxAGPRIdx = 0;
  UncolorableVRegs.clear();
  // Only per function: a rescue copy stays a rescue copy across the full recolor
  // that clears UncolorableVRegs again below, since the copy instruction remains.
  RescueCopies.clear();
  RehomedVRegs.clear();
  RecoverySpilledVRegs.clear();
  // Shadow register-tree oracle (observer, default off). Build the VGPR_32
  // physreg<->leaf map + empty tree for this function; seedOccupiedAtBBEntry
  // re-anchors it per block. No-op / not built unless the flag AND a forensic
  // sink are on.
  setupShadowTree();
  if (!RecoveryOnly)
    preSpillToLimitWidthAware(MF);
  color();

  // [Stage 3] Region RP-reduction: do our BEST-EFFORT spill-across to drop the
  // point-pressure below Limit in the tight regions, then RECOLOR ONCE from clean.
  // We do NOT iterate to convergence — spill-across only relieves genuine RP
  // over-pressure; whatever remains uncolorable after one relieved recolor is the
  // range-interference/fragmentation residual, which the SPLIT path below handles.
  // (Iterating recolor here fought the split path and churned; one pass + hand off
  // to split is the design.)
  // Iterate spill-across + recolor to CONVERGENCE. A single pass is not enough:
  // recoloring from clean after the first round's spills can repack colored
  // values so that a value which WAS colorable now sits at a fresh over-pressure
  // point (measured Case B: colored point-RP == Limit over the value's own tiny
  // range, over by exactly its width). That new failure was not visible to the
  // first sweep (the value was still colored then). On the NEXT pass the now-
  // uncolorable value enters the sweep as an uncolored crosser, so the profile
  // sees RP > Limit at its range and spills a live-through across it. Loop until
  // no uncolorables remain, or a pass performs no spill (genuine residual for the
  // split path), guarded by a hard cap.
  if (!RecoveryOnly && !UncolorableVRegs.empty()) {
    // TEMPORARY / KNOWN-FLAWED termination (stopgap — see task #47 for the real
    // fix, an atomic region-relief transaction). The measure is the post-recolor
    // uncolorable COUNT: keep a round only if it strictly beats the previous
    // round's count. PrevCount starts at ~0u so round 0 is always kept.
    //
    // WHY IT'S FLAWED: the count is NOT a sound monotone measure. A round can
    // spill the failing value (count -1) while its reload remnant [reload,use] is
    // itself uncolorable (count +1) -> net equal -> this bails on a round that
    // actually made structural progress. And reload redefs are re-admitted as
    // candidates next round (unlike the pre-spiller's frozen universe), so without
    // this coarse break the loop can rolling-wave. The count break only bounds it;
    // it does not cleanly separate progress from churn. The real solution collects
    // the whole feasible spill+split+self-spill set for the region and commits it
    // atomically (task #47), making per-round convergence bookkeeping unnecessary.
    // Kept as-is only to unblock corpus measurement of the AGPR/dead-def work.
    unsigned PrevCount = ~0u;
    for (unsigned Round = 0; !UncolorableVRegs.empty(); ++Round) {
      LLVM_DEBUG(dbgs() << "=== region-rp round " << Round << ": "
                        << UncolorableVRegs.size()
                        << " uncolorable -> spill-across pass ===\n");
      // E2 RoundStarted.
      uint64_t RoundID = Reporter->roundStarted(Round, UncolorableVRegs.size());
      if (!reduceRegionPressure(MF)) {
        // E3 RoundCompleted (no spill -> residual for the split path).
        Reporter->roundCompleted(Round, UncolorableVRegs.size(),
                                 /*Spilled=*/false, RoundID);
        break; // nothing spilled this round -> residual is split-path work
      }
      OccupiedRegUnits.clear();
      OccupiedRegUnits.resize(TRI->getNumRegUnits());
      ColorMap.clear();
      MaxVGPRIdx = 0;
      MaxSGPRIdx = 0;
      MaxAGPRIdx = 0;
      UncolorableVRegs.clear();
      setupShadowTree(); // rebuild the shadow tree for the fresh recolor
      color();
      LLVM_DEBUG(dbgs() << "=== region-rp round " << Round << ": after recolor, "
                        << UncolorableVRegs.size() << " uncolorable remain ===\n");
      // E3 RoundCompleted (a spill happened this round).
      Reporter->roundCompleted(Round, UncolorableVRegs.size(),
                               /*Spilled=*/true, RoundID);
      // PROGRESS = strict decrease of the uncolorable count. This round spilled
      // (reduceRegionPressure returned true) and recolored from clean; if that did
      // not reduce how many values remain uncolorable, spill-across cannot relieve
      // the residual (e.g. diamond values used inside every crossing region), so
      // further rounds would only churn. Bail to the per-value split path. This is
      // the sole termination condition — the measure strictly decreases every kept
      // round and is bounded below by 0.
      unsigned CurCount = UncolorableVRegs.size();
      if (CurCount >= PrevCount) { // PrevCount==~0u on the first round -> kept
        LLVM_DEBUG(dbgs() << "=== region-rp: no progress (" << PrevCount << " -> "
                          << CurCount << " uncolorable) -> stop ===\n");
        break;
      }
      PrevCount = CurCount;
    }
    LLVM_DEBUG(dbgs() << "=== region-rp: converged with "
                      << UncolorableVRegs.size()
                      << " uncolorable remain -> split path ===\n");
  }

  // Spill-on-coloring-failure (approach A). A pure Hack coloring can fail on
  // AMDGPU even at RP ≤ limit (the %1072/%560 long-liver-through-tuple-churn
  // class): no single physreg is free across the value's whole range. color()
  // collected every such value in UncolorableVRegs and skipped it, so ColorMap
  // now holds a valid assignment for EVERYTHING ELSE — untouched from here on.
  //
  // For each collected value: spill it (store-at-def + reload-at-use), which
  // replaces its one long range with short reload ranges, then color those
  // reload remainders IN PLACE against the frozen ColorMap. We never re-color
  // an already-placed value, so no successfully-colored value can be perturbed
  // into a new failure (the reason we do NOT recolor from clean). Each width-1
  // reload provably settles: point pressure at the use ≤ RPLimit < file size.
  if (!UncolorableVRegs.empty() && EnableExperimentBail) {
    // EXPERIMENT MODE: do NOT attempt the (possibly-crashing) spill+recolor.
    // The forensic data we care about (the colorfail facts and the count of
    // values that reached spill) is already collected during color(). Classify
    // each failed value by width so the run terminates cleanly and -stats
    // flushes, giving us the failure shape on ALL tests instead of aborting on
    // the first.
    NumTierSpills += UncolorableVRegs.size();
    for (Register Failed : UncolorableVRegs) {
      const TargetRegisterClass *RC = MRI->getRegClass(Failed);
      LLVM_DEBUG(dbgs() << "  [EXPERIMENT] uncolorable " << printReg(Failed, TRI)
                        << " width=" << TRI->getRegSizeInBits(*RC)
                        << (TRI->getRegSizeInBits(*RC) == 32
                                ? "  (width-1: spiller under-spilled its pool)"
                                : "  (WIDE: spiller under-spilled a tuple)")
                        << "\n");
    }
    // E17 RunCompleted: flush the forensic record on this early-exit path too
    // (the colorfail facts were collected during color()).
    Reporter->endRun(UncolorableVRegs.size());
    // Bail cleanly — no destroySSA on an incompletely-colored function.
    return true;
  }

  // Recover every coloring failure in place. Recovery can repoint a tied use
  // after its def inherited the old use's color; if that happens, recolor the
  // mutated MIR from clean and run recovery again for any new failures. The
  // irreversible recovery sets are deliberately preserved across this loop.
  while (true) {
    bool RecoveryComplete = true;
    if (!UncolorableVRegs.empty()) {
      NumTierSpills += UncolorableVRegs.size();
      RecoveryComplete =
          drainUncolorableWorklist(MF, /*ReportFailure=*/!RecoveryOnly);
    }

    if (RecoveryOnly && !RecoveryComplete) {
      LLVM_DEBUG(dbgs() << "=== recovery-only: late cheap planner ===\n");
      if (reduceRegionPressure(MF)) {
        OccupiedRegUnits.clear();
        OccupiedRegUnits.resize(TRI->getNumRegUnits());
        ColorMap.clear();
        MaxVGPRIdx = 0;
        MaxSGPRIdx = 0;
        MaxAGPRIdx = 0;
        UncolorableVRegs.clear();
        setupShadowTree();
        color();
      }
      drainUncolorableWorklist(MF);
    }

    if (tiedAssignmentsValid())
      break;

    OccupiedRegUnits.clear();
    OccupiedRegUnits.resize(TRI->getNumRegUnits());
    ColorMap.clear();
    MaxVGPRIdx = 0;
    MaxSGPRIdx = 0;
    MaxAGPRIdx = 0;
    UncolorableVRegs.clear();
    setupShadowTree();
    color();

    if (UncolorableVRegs.empty() && !tiedAssignmentsValid())
      report_fatal_error(
          "SSARA produced inconsistent tied assignments after clean recolor");
  }

    // Rewrite THIS stage's vregs to physregs now, so the next stage starts with
    // only the other file still virtual (and the SGPR spill count is final for
    // the VGPR stage's VGPRReserve). The whole-function finalize runs once after
    // the loop.
    rewriteStage(MF, StageFile);
  } // end of the two allocation stages

  // E17 RunCompleted: flush this function's record to the configured sinks.
  Reporter->endRun(UncolorableVRegs.size());

  finalizeAfterRewrite(MF);

  return true;
}

MachineFunctionPass *llvm::createAMDGPUSSARegisterAllocatorPass() {
  SSARA_TRACE();
  return new AMDGPUSSARegisterAllocator();
}
