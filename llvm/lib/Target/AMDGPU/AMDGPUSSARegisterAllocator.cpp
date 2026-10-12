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
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "SSARegisterForestAdapter.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "SSARATrace.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <iterator>
#include <tuple>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-ssa-register-allocator"
static cl::opt<int> CoverageTraceVReg(
    "amdgpu-ssa-coverage-vreg", cl::Hidden, cl::init(-1),
    cl::desc("Limit recovery debug output to this virtual register index"));
static cl::opt<bool> EnableRecoveryTiming(
    "amdgpu-ssa-phase-timing", cl::Hidden, cl::init(false),
    cl::desc("Report SSA allocator phase timings without enabling debug dumps"));
static cl::opt<bool> EliminatePostColorReloads(
    "amdgpu-ssa-eliminate-reloads", cl::Hidden, cl::init(true),
    cl::desc("Eliminate redundant reloads after SSA register coloring"));
static constexpr unsigned MaxRecoverySplits = 64;

namespace {
struct RecoveryTimingEntry {
  StringRef Name;
  double Milliseconds = 0;
  unsigned Calls = 0;
};
static SmallVector<RecoveryTimingEntry, 16> RecoveryTimings;
class RecoveryScopeTimer {
  StringRef Name;
  std::chrono::steady_clock::time_point Start;
  bool Enabled = false;
public:
  explicit RecoveryScopeTimer(StringRef Name) : Name(Name) {
    Enabled = EnableRecoveryTiming;
    DEBUG_WITH_TYPE("amdgpu-ssa-phase-time", Enabled = true);
    if (Enabled)
      Start = std::chrono::steady_clock::now();
  }
  void finish() {
    if (!Enabled)
      return;
    double Ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - Start).count();
    auto I = llvm::find_if(RecoveryTimings,
                          [&](const RecoveryTimingEntry &E) { return E.Name == Name; });
    if (I == RecoveryTimings.end()) {
      RecoveryTimings.push_back({Name, Ms, 1});
    } else {
      I->Milliseconds += Ms;
      ++I->Calls;
    }
    Enabled = false;
  }
  ~RecoveryScopeTimer() { finish(); }
};
static void dumpRecoveryTimings() {
  bool Enabled = EnableRecoveryTiming;
  DEBUG_WITH_TYPE("amdgpu-ssa-phase-time", Enabled = true);
  if (Enabled)
    for (const RecoveryTimingEntry &E : RecoveryTimings)
      errs() << "phase-total " << E.Name << " ms=" << E.Milliseconds
             << " calls=" << E.Calls << "\n";
}
} // namespace


/// An `undef` flag applies to the entire selected operand. Missing lanes make
/// a read partial-undef, but the operand remains a real use while any selected
/// lane has a value. This matches RegisterCoalescer::addUndefFlag and
/// VirtRegRewriter::readsUndefSubreg.
static bool readsOnlyUndefLanes(LaneBitmask ReadMask,
                                LaneBitmask DefinedMask) {
  return (ReadMask & DefinedMask).none();
}

AMDGPUSSARegisterAllocator::AMDGPUSSARegisterAllocator()
    : MachineFunctionPass(ID) {}
AMDGPUSSARegisterAllocator::~AMDGPUSSARegisterAllocator() = default;

MCRegister AMDGPUSSARegisterAllocator::assignedHome(Register VR) const {
  MCRegister Home;
  for (const auto &Forest : Forests) {
    if (!Forest)
      continue;
    MCRegister Candidate = Forest->assignedHome(VR);
    if (!Candidate)
      continue;
    if (Home)
      report_fatal_error("owner assigned in more than one register file");
    Home = Candidate;
  }
  return Home;
}

bool AMDGPUSSARegisterAllocator::hasAssignedTiedPartner(Register VR) const {
  for (const MachineOperand &MO : MRI->reg_nodbg_operands(VR)) {
    if (!MO.isTied())
      continue;
    const MachineInstr &MI = *MO.getParent();
    const MachineOperand &Other =
        MI.getOperand(MI.findTiedOperandIdx(MO.getOperandNo()));
    const MachineOperand &Use = MO.isUse() ? MO : Other;
    // An undef passthrough imposes no input-home constraint. A future,
    // unassigned partner likewise has no assignment to preserve yet.
    if (Use.isUndef())
      continue;
    Register Partner = Other.getReg();
    if (Partner == VR)
      continue;
    if (Partner.isPhysical() || assignedHome(Partner))
      return true;
  }
  return false;
}

void AMDGPUSSARegisterAllocator::initializeForests() {
  assert(!Forests[0] && "coloring requires a fresh register forest");
  const TargetRegisterClass *Classes[] = {&AMDGPU::SGPR_32RegClass,
                                          &AMDGPU::VGPR_32RegClass,
                                          &AMDGPU::AGPR_32RegClass};
  for (unsigned F = 0; F != Forests.size(); ++F) {
    Forests[F] = RegisterForestAdapter::createForest(*Classes[F], *TRI);
    if (!Forests[F])
      report_fatal_error("invalid target register-forest topology");
    Adapters[F] = std::make_unique<RegisterForestAdapter>(
        *Forests[F], *TRI, *MRI,
        [this](const TargetRegisterClass *RC) { return availableOrder(RC); });
  }
}

RegisterForestAdapter *
AMDGPUSSARegisterAllocator::adapterFor(MCRegister PR) const {
  if (!PR || PR.id() >= TRI->getNumRegs())
    return nullptr;
  const TargetRegisterClass *RC = TRI->getPhysRegBaseClass(PR);
  if (!RC ||
      !(TRI->isSGPRClass(RC) || TRI->isVGPRClass(RC) || TRI->isAGPRClass(RC)))
    return nullptr;
  return Adapters[static_cast<unsigned>(poolOf(RC))].get();
}

void AMDGPUSSARegisterAllocator::visitAssignments(
    function_ref<void(Register, MCRegister)> Visit) const {
  for (const auto &Forest : Forests)
    if (Forest)
      Forest->visitAssignedOwners(Visit);
}

std::optional<RegisterForestAdapter::Interference>
AMDGPUSSARegisterAllocator::forestInterferences(Register VR,
                                                MCRegister PR) const {
  RegisterForestAdapter *Adapter = adapterFor(PR);
  if (!Adapter || !VR.isVirtual() || VR.virtRegIndex() >= MRI->getNumVirtRegs())
    return std::nullopt;
  const TargetRegisterClass *RC = MRI->getRegClassOrNull(VR);
  if (!RC || !RC->contains(PR) || !LIS->hasInterval(VR))
    return std::nullopt;
  return Adapter->interferences(PR, LIS->getInterval(VR), *LIS);
}

bool AMDGPUSSARegisterAllocator::placementIsFree(Register VR,
                                                 MCRegister PR) const {
  auto Query = forestInterferences(VR, PR);
  if (!Query)
    report_fatal_error("invalid register-forest placement query");
  return !Query->HasFixedInterference && Query->VirtualOwners.empty();
}

bool AMDGPUSSARegisterAllocator::isFreeAt(MCRegister PR, SlotIndex At) const {
  RegisterForestAdapter *Adapter = adapterFor(PR);
  if (!Adapter || !At.isValid())
    report_fatal_error("invalid register-forest point query");
  return Adapter->isFree(PR, At, At.getNextSlot());
}

void AMDGPUSSARegisterAllocator::assignColor(Register VR, MCRegister PR) {
  RegisterForestAdapter *Adapter = adapterFor(PR);
  if (!Adapter || !VR.isVirtual() || assignedHome(VR))
    report_fatal_error(
        "register assignment requires an uncolored virtual owner");
  if (!Adapter->assign(VR, PR, *LIS))
    report_fatal_error("register-forest ownership insertion failed");
}

void AMDGPUSSARegisterAllocator::unassignColor(Register VR) {
  MCRegister Home = assignedHome(VR);
  RegisterForestAdapter *Adapter = adapterFor(Home);
  if (!Adapter || !Adapter->unassign(VR))
    report_fatal_error("register removal requires an existing assignment");
}

void AMDGPUSSARegisterAllocator::clearColors() {
  for (auto &Adapter : Adapters)
    Adapter.reset();
  for (auto &Forest : Forests)
    Forest.reset();
}

SmallVector<RegisterForestAdapter *, 3>
AMDGPUSSARegisterAllocator::repairAdapters(ArrayRef<Register> Affected) {
  SmallVector<RegisterForestAdapter *, 3> Result;
  if (!Forests[0])
    return Result;
  bool SGPR = false, VGPR = false, AGPR = false;
  for (Register VR : Affected) {
    const TargetRegisterClass *RC = MRI->getRegClass(VR);
    SGPR |= TRI->hasSGPRs(RC);
    VGPR |= TRI->hasVGPRs(RC);
    AGPR |= TRI->hasAGPRs(RC);
  }
  if (SGPR)
    Result.push_back(Adapters[static_cast<unsigned>(RegFile::SGPR)].get());
  if (VGPR)
    Result.push_back(Adapters[static_cast<unsigned>(RegFile::VGPR)].get());
  if (AGPR)
    Result.push_back(Adapters[static_cast<unsigned>(RegFile::AGPR)].get());
  return Result;
}

void AMDGPUSSARegisterAllocator::notifyLivenessChanged(
    ArrayRef<Register> Affected) {
  // Producers retain register classes. A class-changing repair must notify
  // both its previous file and its new file before allocation resumes.
  for (RegisterForestAdapter *Adapter : repairAdapters(Affected))
    Adapter->onChange(Affected, *LIS);
}

bool AMDGPUSSARegisterAllocator::needsRecovery(Register VR) {
  return LIS->hasInterval(VR) && !LIS->getInterval(VR).empty() &&
         !MRI->reg_nodbg_empty(VR) && !assignedHome(VR);
}

void AMDGPUSSARegisterAllocator::queueUnassignedValues(
    ArrayRef<Register> Affected) {
  // Consumed entries remain in the vector; their presence is not a pending bit.
  for (Register VR : Affected)
    if (needsRecovery(VR))
      UncolorableVRegs.push_back(VR);
}

void AMDGPUSSARegisterAllocator::placeRepairedValues(
    ArrayRef<Register> Affected) {
  RecoveryScopeTimer PhaseTimer("place-repaired");
  for (Register VR : Affected) {
    if (!needsRecovery(VR))
      continue;
    if (SSAForensicReporter::enabled())
      Reporter->flushNow();
    // The vector retains already-consumed entries. A failed value must be
    // appended again so the drain revisits it after this repair.
    if (!colorOneInPlace(VR))
      UncolorableVRegs.push_back(VR);
  }
}

SSASpillEmitter::RepairResult
AMDGPUSSARegisterAllocator::spillWithForest(VRegMaskPair Value,
                                                SlotIndex Kill) {
  RecoveryScopeTimer PhaseTimer("ordinary-spill");
  Register V = Value.getVReg();
  auto Dump = [&](StringRef Phase) {
    DEBUG_WITH_TYPE("amdgpu-ssa-spill-history", {
      if (CoverageTraceVReg < 0 ||
          V.virtRegIndex() == unsigned(CoverageTraceVReg)) {
        dbgs() << "spill-history " << Phase << " function="
               << MRI->getMF().getName() << " value=" << printReg(V, TRI)
               << " home=" << printReg(assignedHome(V), TRI) << "\n";
        if (LIS->hasInterval(V))
          LIS->getInterval(V).print(dbgs());
        dbgs() << "\n";
        for (MachineInstr &MI : MRI->reg_nodbg_instructions(V))
          dbgs() << "  bb." << MI.getParent()->getNumber() << " "
                 << LIS->getInstructionIndex(MI) << " " << MI;
      }
    });
  };
  using Segments = SmallVector<std::pair<SlotIndex, SlotIndex>, 8>;
  SmallVector<std::pair<LaneBitmask, Segments>, 4> Before;
  bool Repeated = RecoverySpilledValues.overlaps(Value);
  if (Repeated) {
    const LiveInterval &LI = LIS->getInterval(V);
    auto Save = [&](LaneBitmask Mask, const LiveRange &LR) {
      Before.emplace_back(Mask, Segments());
      for (const auto &S : LR.segments)
        Before.back().second.emplace_back(S.start, S.end);
    };
    if (LI.hasSubRanges()) {
      LaneBitmask Missing = Value.getLaneMask();
      for (const auto &SR : LI.subranges()) {
        LaneBitmask Mask = SR.LaneMask & Value.getLaneMask();
        if (Mask.any())
          Save(Mask, SR);
        Missing &= ~SR.LaneMask;
      }
      if (Missing.any())
        Before.emplace_back(Missing, Segments());
    } else {
      Save(Value.getLaneMask(), LI);
    }
  }
  Dump("before");
  DEBUG_WITH_TYPE("amdgpu-ssa-spill-history", {
    if (CoverageTraceVReg < 0 ||
        V.virtRegIndex() == unsigned(CoverageTraceVReg)) {
      dbgs() << "  kill=" << Kill << " ";
      if (MachineInstr *MI = Indexes->getInstructionFromIndex(Kill))
        dbgs() << *MI;
    }
  });
  auto Result = Emitter->spillOneVMP(
      Value, Kill, [this](ArrayRef<Register> Affected) {
        notifyLivenessChanged(Affected);
      });
  if (Repeated) {
    // SlotIndex handles survive insertion/renumbering. Compare lane coverage,
    // ignoring value-number identities introduced by SSA repair.
    auto Covered = [](const Segments &A, const Segments &B) {
      size_t J = 0;
      for (auto [Start, End] : A) {
        while (J < B.size() && B[J].second <= Start)
          ++J;
        SlotIndex At = Start;
        for (size_t K = J; K < B.size() && B[K].first <= At; ++K) {
          At = std::max(At, B[K].second);
          if (End <= At)
            break;
        }
        if (At < End)
          return false;
      }
      return true;
    };
    bool Shrunk = false;
    const LiveInterval *After = LIS->hasInterval(V) ? &LIS->getInterval(V) : nullptr;
    for (const auto &[Mask, Old] : Before) {
      for (unsigned Bit = 0; Bit < LaneBitmask::BitWidth; ++Bit) {
        LaneBitmask Lane = LaneBitmask::getLane(Bit);
        if ((Mask & Lane).none())
          continue;
        Segments Now;
        auto Add = [&](const LiveRange &LR) {
          for (const auto &S : LR.segments)
            Now.emplace_back(S.start, S.end);
        };
        if (After) {
          if (After->hasSubRanges()) {
            for (const auto &SR : After->subranges())
              if ((SR.LaneMask & Lane).any())
                Add(SR);
          } else {
            Add(*After);
          }
        }
        if (!Covered(Now, Old))
          report_fatal_error("repeated spill expanded original lane liveness");
        Shrunk |= !Covered(Old, Now);
      }
    }
    DEBUG_WITH_TYPE("amdgpu-ssa-spill-history", dbgs()
        << "repeated-spill " << printReg(V, TRI) << " strict-shrink="
        << Shrunk << "\n");
    if (!Shrunk)
      report_fatal_error("repeated spill did not shrink original lane liveness");
  }
  RecoverySpilledValues.insert(Value);
  Dump("after");
  return Result;
}

std::optional<SSASpillEmitter::RepairResult>
AMDGPUSSARegisterAllocator::splitWithForest(
    Register VR, MachineBasicBlock::iterator Point) {
  RecoveryScopeTimer PhaseTimer("split-repair");
  const TargetRegisterClass *RC = MRI->getRegClass(VR);
  Emitter->beginPass(TRI->isVGPRClass(RC) || TRI->isAGPRClass(RC));
  auto Result = Emitter->splitLiveRangeAt(
      VR, Point, [this](ArrayRef<Register> Affected) {
        notifyLivenessChanged(Affected);
      });
  if (Result)
    ++RecoverySplitCount;
  return Result;
}

static cl::opt<bool> EnableLaneWasteDump(
    "amdgpu-ssa-lane-waste-dump", cl::Hidden, cl::init(false),
    cl::desc("Report per-function capacity held by dead lanes of partially "
             "live tuples: whole-tuple occupancy vs subrange occupancy."));

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

void AMDGPUSSARegisterAllocator::collectOccupancy(const TargetRegisterClass *RC,
                                                  SlotIndex SI,
                                                  const LiveInterval *VI,
                                                  OccupancyFacts &Out) const {
  SSARA_TRACE();
  Out = OccupancyFacts();
  Out.ClassName = TRI->getRegClassName(RC);
  for (MCRegister PR : availableOrder(RC)) {
    if (!Out.Total)
      Out.FirstReg = TRI->getName(PR);
    Out.LastReg = TRI->getName(PR);
    ++Out.Total;
    if (!isFreeAt(PR, SI)) {
      Out.Map.push_back('#');
      ++Out.Occupied;
    } else if (VI && !placementIsFree(VI->reg(), PR)) {
      Out.Map.push_back('x');
      ++Out.FreeClobbered;
    } else {
      Out.Map.push_back('.');
      ++Out.FreeUsable;
      Out.Usable.push_back(TRI->getName(PR));
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
  // values in R's FILE that are LIVE-THROUGH [FS,FE) with NO use strictly
  // inside
  //  — each such value's register can be freed across the whole region by
  // spilling it (reload past FE). Reads RF ownership / LIS / MRI only.
  NLiveThru = 0;
  visitAssignments([&](Register V, MCRegister P) {
    if (V == Failed || !LIS->hasInterval(V))
      return;
    const TargetRegisterClass *VRC = MRI->getRegClass(V);
    bool VIsVGPR = TRI->isVGPRClass(VRC) || TRI->isAGPRClass(VRC);
    if (VIsVGPR != FIsVGPR)
      return; // wrong file
    const LiveInterval &OVI = LIS->getInterval(V);
    if (!OVI.liveAt(FS) || !OVI.liveAt(FE.getPrevSlot()))
      return; // not live-through R
    ++NLiveThru;
    LiveThruIdx.push_back(V.virtRegIndex());
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
  });
  llvm::sort(LiveThruIdx);
}

void AMDGPUSSARegisterAllocator::collectLiveSet(
    SlotIndex SI, SmallVectorImpl<LiveSetEntry> &Out) const {
  SSARA_TRACE();
  // Facts-only const walk: every virtual register whose interval is live at SI,
  // joined to its physreg via RF ownership (uncolored => phys=-1). This is the
  // same liveAt(SI) test collectOccupancy already uses over RF ownership,
  // generalized to ALL vregs so the cross-section is complete (not just the
  // colored ones). No new LIS/pressure pass; nothing mutated.
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
    if (MCRegister Home = assignedHome(VReg)) {
      Ent.Phys = (int64_t)Home.id();
      Ent.PhysName = TRI->getName(Home);
    } else {
      Ent.Phys = -1;
    }
    Ent.WidthBits = TRI->getRegSizeInBits(*RC);
    Ent.LaneMask = MRI->getMaxLaneMaskForVReg(VReg).getAsInteger();
    Out.push_back(std::move(Ent));
  }
}

void AMDGPUSSARegisterAllocator::collectBlockers(
    Register Subject,
    SmallVectorImpl<std::pair<Register, MCRegister>> &Blockers) const {
  RecoveryScopeTimer PhaseTimer("collect-blockers");
  SmallDenseSet<Register, 16> Seen;
  for (MCRegister Home : availableOrder(MRI->getRegClass(Subject))) {
    auto Query = forestInterferences(Subject, Home);
    if (!Query)
      report_fatal_error("invalid register-forest blocker query");
    for (Register Owner : Query->VirtualOwners) {
      if (!Seen.insert(Owner).second)
        continue;
      MCRegister Assigned = assignedHome(Owner);
      if (!Assigned)
        report_fatal_error("register-forest blocker has no recorded home");
      Blockers.emplace_back(Owner, Assigned);
    }
  }
}

MCRegister AMDGPUSSARegisterAllocator::pickFreePhysReg(
    const TargetRegisterClass *RC, const LiveInterval &VI,
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

  // A tied result can outlive its input. Check every inherited home over
  // the result's complete live interval, including when placing a repair COPY.
  // Ordinary virtual blockers can be evicted at the result; a blocker with an
  // assigned tied partner must retain its home. Undef uses impose no constraint.
  auto TiedHomesAllowPlacement = [&](MCRegister PR) {
    SmallVector<std::pair<Register, MCRegister>, 4> Worklist{{VI.reg(), PR}};
    for (size_t I = 0; I != Worklist.size(); ++I) {
      auto [Input, InputHome] = Worklist[I];
      for (const MachineOperand &Use : MRI->use_nodbg_operands(Input)) {
        if (!Use.isTied() || Use.isUndef())
          continue;
        const MachineInstr &MI = *Use.getParent();
        Register Result =
            MI.getOperand(MI.findTiedOperandIdx(Use.getOperandNo())).getReg();
        MCRegister Home = InputHome;
        if (unsigned SubIdx = Use.getSubReg()) {
          Home = TRI->getSubReg(Home, SubIdx);
          assert(Home && "Invalid tied-use subreg index");
        }
        std::pair<Register, MCRegister> Assignment{Result, Home};
        if (llvm::is_contained(Worklist, Assignment))
          continue;
        // A candidate in another register file may not satisfy this tied
        // result's class. Reject that candidate before querying ownership.
        if (!MRI->getRegClass(Result)->contains(Home))
          return false;
        auto Query = forestInterferences(Result, Home);
        if (!Query.has_value())
          report_fatal_error("SSARA tied placement has an invalid RF query");
        if (Query->HasFixedInterference)
          return false;
        for (Register Owner : Query->VirtualOwners)
          if (Owner != Result && hasAssignedTiedPartner(Owner))
            return false;
        Worklist.push_back(Assignment);
      }
    }
    return true;
  };

  auto IsFree = [&](MCRegister PR) {
    return placementIsFree(VI.reg(), PR) && TiedHomesAllowPlacement(PR);
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
                                  "rf-placement-conflict");
    ++Ordinal;
  }
  return MCRegister();
}

MCRegister AMDGPUSSARegisterAllocator::colorOneInPlace(Register R) {
  SSARA_TRACE();
  if (assignedHome(R))
    report_fatal_error("in-place coloring requires an uncolored owner");
  // Query the complete lane live ranges against the current RF assignments.
  const TargetRegisterClass *RC = MRI->getRegClass(R);
  const LiveInterval &RI = LIS->getInterval(R);

  MCRegister Chosen = pickFreePhysReg(RC, RI);
  if (!Chosen)
    return MCRegister();

  assignColor(R, Chosen);
  unsigned Idx = TRI->getHWRegIndex(Chosen);
  unsigned W = TRI->getRegSizeInBits(*RC) / 32;
  const TargetRegisterClass *PhysRC = TRI->getPhysRegBaseClass(Chosen);
  if (TRI->isVGPRClass(PhysRC))
    MaxVGPRIdx = std::max(MaxVGPRIdx, Idx + W);
  else if (TRI->isAGPRClass(PhysRC))
    MaxAGPRIdx = std::max(MaxAGPRIdx, Idx + W);
  else if (TRI->isSGPRClass(PhysRC))
    MaxSGPRIdx = std::max(MaxSGPRIdx, Idx + W);

  LLVM_DEBUG(dbgs() << "  in-place color: " << printReg(R, TRI) << " -> "
                    << TRI->getName(Chosen) << "\n");
  return Chosen;
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
    MCRegister PartnerHome = assignedHome(Partner);
    if (!PartnerHome)
      return; // partner not colored yet -- nothing to align to
    unsigned Depth = EdgeBlock ? MLI->getLoopDepth(EdgeBlock) : 0;
    uint64_t W = Depth < 63 ? (uint64_t(1) << Depth) : ~uint64_t(0);
    AddCandidate(PartnerHome, SubIdx, PartnerIsSub, W);
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

bool AMDGPUSSARegisterAllocator::edgeCopiesNeedSplit(
    MachineBasicBlock *Pred, MachineBasicBlock *MBB,
    ArrayRef<std::pair<MCRegister, MCRegister>> Copies) const {
  if (Pred->succ_size() <= 1 || MBB->pred_size() <= 1)
    return false;
  for (MachineBasicBlock *Succ : Pred->successors()) {
    if (Succ == MBB)
      continue;
    SlotIndex Start = LIS->getMBBStartIdx(Succ);
    for (const auto &[Source, Destination] : Copies)
      if (!isFreeAt(Destination, Start))
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

void AMDGPUSSARegisterAllocator::reportLaneWaste(MachineFunction &MF) const {
  SSARA_TRACE();
  // Two occupancy models over one tracker walk: what this allocator charges
  // (the whole tuple) and what lane-aware allocation actually occupies
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

// Recovery helpers ------------------------------------------------------------

AMDGPUSSARegisterAllocator::RegFile
AMDGPUSSARegisterAllocator::fileOf(const TargetRegisterClass *RC) const {
  SSARA_TRACE();
  // AGPR folds into VGPR so this matches pressureOf(VGPR)'s unified count; only
  // SGPR classes are the SGPR file.
  return TRI->isSGPRClass(RC) ? RegFile::SGPR : RegFile::VGPR;
}

PhiWeb AMDGPUSSARegisterAllocator::closePhiWeb(Register Seed) const {
  RecoveryScopeTimer PhaseTimer("close-phi-web");
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

void AMDGPUSSARegisterAllocator::commitColor(Register Piece, MCRegister PR) {
  SSARA_TRACE();
  assignColor(Piece, PR);
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

void AMDGPUSSARegisterAllocator::initializePlacementProfile(
    Register Subject, PlacementProfile &Out) const {
  SSARA_TRACE();
  Out = PlacementProfile();
  Out.Subject = Subject;
  Out.RC = MRI->getRegClass(Subject);
  Out.Homes.append(availableOrder(Out.RC).begin(),
                   availableOrder(Out.RC).end());

  if (!LIS->hasInterval(Subject) || LIS->getInterval(Subject).empty())
    return;

  const LiveInterval &SubjectLI = LIS->getInterval(Subject);
  Out.Region = {SubjectLI.beginIndex(), SubjectLI.endIndex()};

  // Mask cuts prevent a resident piece from crossing a clobber. Fixed
  // physical defs and lifetimes are already represented by RF ownership.
  ArrayRef<SlotIndex> Slots = LIS->getRegMaskSlots();
  ArrayRef<const uint32_t *> Masks = LIS->getRegMaskBits();
  for (unsigned I = 0; I != Slots.size(); ++I) {
    SlotIndex At = Slots[I];
    if (At <= Out.Region.Start || Out.Region.End <= At)
      continue;
    SmallBitVector CutHomes(Out.Homes.size());
    for (PlacementProfile::HomeID Home = 0; Home != Out.Homes.size(); ++Home)
      if (MachineOperand::clobbersPhysReg(Masks[I], Out.Homes[Home]))
        CutHomes.set(Home);
    if (CutHomes.any())
      Out.Cuts.push_back({At, std::move(CutHomes)});
  }
}

void AMDGPUSSARegisterAllocator::collectPlacementBlockers(
    PlacementProfile &Profile) const {
  assert(Profile.Blockers.empty() && "placement blockers already populated");
  if (!Profile.Region.Start.isValid())
    return;
  const LiveInterval &SubjectLI = LIS->getInterval(Profile.Subject);
  for (PlacementProfile::HomeID Home = 0; Home != Profile.Homes.size();
       ++Home) {
    RegisterForestAdapter *Adapter = adapterFor(Profile.Homes[Home]);
    if (!Adapter)
      report_fatal_error("unsupported register-forest placement home");
    SmallBitVector BlockedHomes(Profile.Homes.size());
    BlockedHomes.set(Home);
    if (!Adapter->visitInterferences(
            Profile.Homes[Home], SubjectLI,
            [&](Register Owner, SlotIndex Start, SlotIndex End) {
              if (Owner == Profile.Subject)
                return;
              // Fixed owners block placement but are never movable victims.
              Profile.Blockers.push_back({Owner, {Start, End}, BlockedHomes});
            }))
      report_fatal_error("invalid register-forest placement query");
  }
}

bool AMDGPUSSARegisterAllocator::pickPeelableRun(Register V, MCRegister &PR,
                                                 SlotIndex &Bound) const {
  RecoveryScopeTimer PhaseTimer("prefix-free-runs");
  PlacementProfile Profile;
  initializePlacementProfile(V, Profile);
  collectPlacementBlockers(Profile);
  return selectPeelableRun(Profile, PR, Bound);
}

SlotIndex AMDGPUSSARegisterAllocator::firstUseAfter(Register V,
                                                    SlotIndex Start) const {
  SlotIndex FirstUse;
  for (MachineInstr &U : MRI->use_nodbg_instructions(V)) {
    SlotIndex Use = LIS->getInstructionIndex(U).getRegSlot();
    if (Start < Use && (!FirstUse.isValid() || Use < FirstUse))
      FirstUse = Use;
  }
  return FirstUse;
}

bool AMDGPUSSARegisterAllocator::selectPeelableRun(
    const PlacementProfile &Profile, MCRegister &PR, SlotIndex &Bound) const {
  PR = MCRegister();
  Bound = SlotIndex();

  if (!Profile.Subject.isVirtual() || !Profile.Region.Start.isValid() ||
      !Profile.Region.End.isValid() ||
      !(Profile.Region.Start < Profile.Region.End))
    return false;

  const SlotIndex S = Profile.Region.Start;
  const SlotIndex E = Profile.Region.End;

  SmallVector<PlacementFreeRun, 32> FreeRuns;
  getFreeRuns(Profile, FreeRuns);

  // FreeRuns is grouped in target home order. Updating only for a strictly
  // longer run therefore preserves the target-order first-home tie-break.
  SlotIndex Best = S;
  for (const PlacementFreeRun &Run : FreeRuns) {
    if (Run.Range.Start != S)
      continue;
    if (!PR || Best < Run.Range.End) {
      PR = Run.PhysReg;
      Best = Run.Range.End;
    }
  }
  if (!PR)
    return false;

  // A partial prefix must extend past the first use.
  if (Best < E) {
    SlotIndex FirstUse = firstUseAfter(Profile.Subject, S);
    if (FirstUse.isValid() && Best <= FirstUse) {
      PR = MCRegister();
      return false;
    }
  }

  Bound = Best;
  return true;
}

AMDGPUSSARegisterAllocator::RecoveryResult
AMDGPUSSARegisterAllocator::trySelfSplitColor(Register Failed) {
  RecoveryScopeTimer PhaseTimer("self-split");
  SSARA_TRACE();
  const LiveInterval &LI = LIS->getInterval(Failed);
  SlotIndex Start = LI.beginIndex(), End = LI.endIndex();
  MCRegister Home;
  SlotIndex Bound;
  if (!pickPeelableRun(Failed, Home, Bound))
    return RecoveryResult::NoChange;
  if (Bound >= End) {
    commitColor(Failed, Home);
    return RecoveryResult::Resolved;
  }

  // Worklist retries cannot restart the function-wide split budget.
  // Exhaustion uses the floor.
  if (RecoverySplitCount >= MaxRecoverySplits)
    return RecoveryResult::NoChange;
  MachineInstr *SplitMI = LIS->getInstructionFromIndex(Bound);
  SlotIndex Probe = Bound;
  while ((!SplitMI || SplitMI->isPHI() || SplitMI->isDebugInstr()) &&
         Probe > Start) {
    Probe = Probe.getPrevIndex();
    SplitMI = LIS->getInstructionFromIndex(Probe);
  }
  if (!SplitMI || SplitMI->isPHI() || SplitMI->isDebugInstr() ||
      LIS->getInstructionIndex(*SplitMI).getRegSlot() <= Start)
    return RecoveryResult::NoChange;

  auto Split = splitWithForest(Failed, SplitMI->getIterator());
  if (!Split)
    return RecoveryResult::NoChange;
  // The candidate prefix was only a proposal. Do not assign it after mutation;
  // subsequent recovery must place the actual repaired intervals.
  queueUnassignedValues(Split->Affected);
  return RecoveryResult::Deferred;
}

bool AMDGPUSSARegisterAllocator::tryEvict(Register B, SlotIndex Start,
                                         SlotIndex End) {
  RecoveryScopeTimer PhaseTimer("cheap-rehome");
  MCRegister OldHome = assignedHome(B);
  if (!OldHome || !LIS->hasInterval(B) || LIS->getInterval(B).empty() ||
      !LIS->getInterval(B).overlaps(Start, End) || RescueCopies.count(B))
    return false;
  if (hasAssignedTiedPartner(B))
    return false;
  const TargetRegisterClass *RC = MRI->getRegClass(B);
  auto TryClass = [&](const TargetRegisterClass *HomeRC) {
    if (!HomeRC)
      return false;
    for (MCPhysReg PR : availableOrder(HomeRC)) {
      if (TRI->regsOverlap(PR, OldHome))
        continue;
      bool Accepts = llvm::all_of(
          MRI->reg_nodbg_operands(B), [&](const MachineOperand &MO) {
            if (MO.isUse() && !MO.readsReg())
              return true;
            if ((MO.getSubReg() || MO.isTied()) && HomeRC != RC)
              return false;
            const MachineInstr *MI = MO.getParent();
            if (MI->isPHI())
              return true;
            const TargetRegisterClass *OperandRC =
                TII->getRegClass(MI->getDesc(), MO.getOperandNo(), TRI);
            MCRegister OperandHome = MO.getSubReg()
                ? TRI->getSubReg(PR, MO.getSubReg()) : MCRegister(PR);
            return OperandHome &&
                   (!OperandRC || OperandRC->contains(OperandHome));
          });
      if (!Accepts)
        continue;
      // Probe the whole interval while retaining the existing assignment.
      // No copies, splits, spills, or recursive blocker eviction are allowed.
      MRI->setRegClass(B, HomeRC);
      if (!placementIsFree(B, PR)) {
        MRI->setRegClass(B, RC);
        continue;
      }
      unassignColor(B);
      commitColor(B, PR);
      if (HomeRC != RC)
        RehomedVRegs.insert(B);
      DEBUG_WITH_TYPE("amdgpu-ssa-coverage", dbgs()
          << "whole-range rehome " << printReg(B, TRI) << " "
          << printReg(OldHome, TRI) << " -> " << printReg(PR, TRI) << "\n");
      return true;
    }
    return false;
  };
  if (TryClass(RC))
    return true;
  if (!ST->hasMAIInsts() || RehomedVRegs.count(B) ||
      (!TRI->hasVGPRs(RC) && !TRI->isAGPRClass(RC)))
    return false;
  return TryClass(TRI->isAGPRClass(RC) ? TRI->getEquivalentVGPRClass(RC)
                                      : TRI->getEquivalentAGPRClass(RC));
}

AMDGPUSSARegisterAllocator::RecoveryResult
AMDGPUSSARegisterAllocator::tryCrossFileHome(Register R) {
  RecoveryScopeTimer PhaseTimer("cross-file");
  SSARA_TRACE();
  if (colorOneInPlace(R))
    return RecoveryResult::Resolved;
  if (!ST->hasMAIInsts())
    return RecoveryResult::NoChange;

  auto TryOne = [&](Register V) -> bool {
    const TargetRegisterClass *RC = MRI->getRegClass(V);
    if ((!TRI->hasVGPRs(RC) && !TRI->isAGPRClass(RC)) ||
        RescueCopies.count(V) || RehomedVRegs.count(V))
      return false;
    const bool ToAGPR = !TRI->isAGPRClass(RC);
    const TargetRegisterClass *Home = ToAGPR
        ? TRI->getEquivalentAGPRClass(RC) : TRI->getEquivalentVGPRClass(RC);
    if (!Home)
      return false;

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

    {
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
    }
    notifyLivenessChanged({V});
    // Copy insertion only narrows V's lifetime. Losing its checked home here
    // indicates broken liveness maintenance, not an ordinary placement failure.
    if (!assignedHome(V))
      report_fatal_error("cross-file copies invalidated their assigned owner");
    RehomedVRegs.insert(V);
    LLVM_DEBUG(dbgs() << "  [cross-file-home] " << printReg(V, TRI) << " -> "
                      << TRI->getName(assignedHome(V)) << " ("
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
  // The current-file attempt above failed. Try only this value's sibling.
  if (TryOne(R))
    return RecoveryResult::Resolved;

  return RecoveryResult::NoChange;
}

void AMDGPUSSARegisterAllocator::reportPointOverPressure(Register R,
                                                         bool IsVGPR,
                                                         unsigned RPLimit,
                                                         const char *Ctx) {
  RecoveryScopeTimer DiagnosticTimer("terminal-diagnostic");
  SSARA_TRACE();
  // Honest terminal for an unrecoverable memory floor. Find the point in R's range
  // with the most simultaneously-live dwords of R's register file and report the
  // REAL numbers. If that peak exceeds RPLimit no coloring-time recovery exists
  // (more values demand registers at one instant than the file has); otherwise
  // the point is feasible yet unrecovered, which is a genuine allocator bug —
  // say so, rather than the misleading "needs more up-front spilling".
  const LiveInterval &RI = LIS->getInterval(R);
  DEBUG_WITH_TYPE("amdgpu-ssa-spill-history", {
    dbgs() << "spill-history failure=" << Ctx << " function="
           << MRI->getMF().getName() << " value=" << printReg(R, TRI) << "\n";
    RI.print(dbgs());
    dbgs() << "\n";
    for (MachineInstr &MI : MRI->reg_nodbg_instructions(R))
      dbgs() << "  bb." << MI.getParent()->getNumber() << " "
             << LIS->getInstructionIndex(MI) << " " << MI;
  });
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
  DiagnosticTimer.finish();
  dumpRecoveryTimings();
  report_fatal_error(StringRef(Msg));
}

bool AMDGPUSSARegisterAllocator::recoverUncolorable(Register Failed) {
  SSARA_TRACE();
  // Each attempt advances from self rehome through one blocker snapshot to
  // prefix self-splitting. A blocker mutation never restarts this attempt.
  // The snapshot bounds this attempt, and explicit split copies consume the
  // function-wide budget. Same-file homes may change in later attempts, and
  // repairs can append new values to the worklist; these local restrictions
  // alone do not establish termination of the entire recovery worklist.
  const MachineFunction &MF = MRI->getMF();

  LLVM_DEBUG(dbgs() << "FALLBACK for " << printReg(Failed, TRI) << " ["
                    << LIS->getInterval(Failed).beginIndex() << ","
                    << LIS->getInterval(Failed).endIndex() << ")\n");

  RecoveryState State = RecoveryState::Web;
  Register Cur = Failed;
  while (true) {
    DEBUG_WITH_TYPE("amdgpu-ssa-fsm-trace", {
      if (CoverageTraceVReg < 0 || Cur.virtRegIndex() == unsigned(CoverageTraceVReg)) {
        const char *Names[] = {"Web", "CrossFileHome", "EvictBlockers", "SelfSplit", "Floor"};
        dbgs() << "fsm " << printReg(Cur, TRI) << " state=" << Names[unsigned(State)]
               << " home=" << printReg(assignedHome(Cur), TRI) << "\n";
      }
    });
    switch (State) {
    case RecoveryState::Web: {
      RecoveryScopeTimer PhaseTimer("web-state");
      PhiWeb Web = closePhiWeb(Cur);
      if (!Web.valid()) {
        State = RecoveryState::CrossFileHome;
        continue;
      }

      SmallVector<VRegMaskPair, 8> GroundStores;
      for (const GroundEdge &GE : Web.GroundEdges) {
        MachineInstr *Def = MRI->getUniqueVRegDef(GE.Reg);
        if (!Def || Def->isImplicitDef())
          continue;
        GroundStores.emplace_back(
            GE.Reg, GE.SubReg ? TRI->getSubRegIndexLaneMask(GE.SubReg)
                             : MRI->getMaxLaneMaskForVReg(GE.Reg));
      }
      if (llvm::any_of(GroundStores, [&](VRegMaskPair VMP) {
            return RecoverySpilledValues.overlaps(VMP);
          })) {
        State = RecoveryState::CrossFileHome;
        continue;
      }
      const TargetRegisterClass *RC = MRI->getRegClass(Cur);
      bool IsVGPR = TRI->isVGPRClass(RC) || TRI->isAGPRClass(RC);
      Emitter->beginPass(IsVGPR);
      auto Result = Emitter->spillPhiWeb(
          Web, [this](ArrayRef<Register> Affected) {
            notifyLivenessChanged(Affected);
          });
      RecoverySpilledValues.insert(GroundStores.begin(), GroundStores.end());
      placeRepairedValues(Result.Affected);
      if (SSAForensicReporter::enabled())
        Reporter->transformation("phi-web-spill", Web.Root.virtRegIndex());
      return true;
    }

    case RecoveryState::CrossFileHome: {
      RecoveryResult Home = tryCrossFileHome(Cur);
      if (Home == RecoveryResult::Resolved) {
        return true;
      }
      assert(Home == RecoveryResult::NoChange);
      State = RecoveryState::EvictBlockers;
      continue;
    }

    case RecoveryState::EvictBlockers: {
      RecoveryScopeTimer PhaseTimer("eviction-state");
      SmallVector<std::pair<Register, MCRegister>, 16> Blockers;
      collectBlockers(Cur, Blockers);
      unsigned Tried = 0, Moved = 0;
      for (const auto &[B, OldHome] : Blockers) {
        if (!needsRecovery(Cur))
          break;
        if (assignedHome(B) != OldHome || !LIS->hasInterval(B) ||
            !LIS->getInterval(B).overlaps(LIS->getInterval(Cur)))
          continue;
        // Mutations can renumber SlotIndexes. Read the current window for every
        // surviving snapshot member and intentionally ignore its result.
        SlotIndex Start = LIS->getInterval(Cur).beginIndex();
        SlotIndex End = LIS->getInterval(Cur).endIndex();
        DEBUG_WITH_TYPE("amdgpu-ssa-coverage", dbgs()
            << "evict principal=" << printReg(Cur, TRI)
            << " blocker=" << printReg(B, TRI) << " begin\n");
        ++Tried;
        Moved += tryEvict(B, Start, End);
        DEBUG_WITH_TYPE("amdgpu-ssa-coverage", dbgs()
            << "evict principal=" << printReg(Cur, TRI)
            << " blocker=" << printReg(B, TRI) << " end\n");
      }
      DEBUG_WITH_TYPE("amdgpu-ssa-fsm-trace", {
        if (CoverageTraceVReg < 0 || Cur.virtRegIndex() == unsigned(CoverageTraceVReg))
          dbgs() << "fsm blockers snapshot=" << Blockers.size()
                 << " tried=" << Tried << " moved=" << Moved << "\n";
      });
      DEBUG_WITH_TYPE("amdgpu-ssa-blocker-detail", {
        if (CoverageTraceVReg >= 0 &&
            Cur.virtRegIndex() == unsigned(CoverageTraceVReg)) {
          dbgs() << "blocker-detail principal=" << printReg(Cur, TRI)
                 << " moved=" << Moved << " needs=" << needsRecovery(Cur)
                 << "\n";
          for (const auto &[B, OldHome] : Blockers) {
            dbgs() << "  blocker=" << printReg(B, TRI)
                   << " home=" << printReg(assignedHome(B), TRI)
                   << " old=" << printReg(OldHome, TRI)
                   << " rescue=" << RescueCopies.count(B)
                   << " tied=" << hasAssignedTiedPartner(B)
                   << " spilled=" << RecoverySpilledValues.overlaps(
                          VRegMaskPair(B, MRI->getMaxLaneMaskForVReg(B)))
                   << "\n";
            if (LIS->hasInterval(B))
              LIS->getInterval(B).print(dbgs());
            dbgs() << "\n";
            for (MachineOperand &MO : MRI->reg_nodbg_operands(B)) {
              MachineInstr *MI = MO.getParent();
              dbgs() << "    bb." << MI->getParent()->getNumber() << " ";
              if (Indexes->hasIndex(*MI))
                dbgs() << LIS->getInstructionIndex(*MI) << " ";
              MI->print(dbgs());
            }
          }
        }
      });
      if (needsRecovery(Cur)) {
        // Rehoming did not place the principal. Spill eligible live-through
        // blockers from this snapshot until placement succeeds or it is exhausted.
        // Count once per blocker; preserve collection order for equal counts.
        SmallDenseMap<Register, unsigned, 16> UseCounts;
        for (const auto &[B, OldHome] : Blockers) {
          unsigned Count = llvm::count_if(
              MRI->use_nodbg_operands(B),
              [](const MachineOperand &MO) { return MO.readsReg(); });
          UseCounts[B] = Count;
        }
        llvm::stable_sort(Blockers, [&](const auto &A, const auto &B) {
          return UseCounts.lookup(A.first) < UseCounts.lookup(B.first);
        });
        for (const auto &[B, OldHome] : Blockers) {
          if (!needsRecovery(Cur))
            break;
          // Each repair may renumber indexes or change the principal interval.
          MachineInstr *KillMI = MRI->getVRegDef(Cur);
          SlotIndex Start = LIS->getInstructionIndex(*KillMI).getRegSlot();
          SlotIndex End = LIS->getInterval(Cur).endIndex();
          if (assignedHome(B) != OldHome || !LIS->hasInterval(B) ||
              RescueCopies.count(B))
            continue;
          VRegMaskPair VMP(B, MRI->getMaxLaneMaskForVReg(B));
          const LiveInterval &BI = LIS->getInterval(B);
          MachineInstr *Def = MRI->getVRegDef(B);
          if (!BI.liveAt(Start) || !BI.liveAt(End.getPrevSlot()) ||
              !MDT->dominates(Def, KillMI))
            continue;
          bool UsedInWindow = llvm::any_of(
              MRI->use_nodbg_operands(B), [&](const MachineOperand &MO) {
                if (!MO.readsReg())
                  return false;
                const MachineInstr *MI = MO.getParent();
                SlotIndex UseAt = MI->isPHI()
                    ? LIS->getMBBEndIdx(
                          MI->getOperand(MO.getOperandNo() + 1).getMBB())
                          .getPrevSlot()
                    : LIS->getInstructionIndex(*MI).getRegSlot();
                return Start.getBaseIndex() <= UseAt && UseAt <= End;
              });
          if (UsedInWindow || !Emitter->canSpill(VMP))
            continue;
          // Keep already assigned tied results in their existing homes. Only
          // admit direct, whole-register reloads immediately before a dominated
          // tied use; no alternative-home search or tied-chain rehoming here.
          struct TiedReload {
            MachineInstr *Use;
            unsigned Operand;
            Register Result;
            MCRegister Home;
          };
          SmallVector<TiedReload, 4> TiedReloads;
          bool TiesFit = true;
          bool PreserveTiedDefHome = false;
          for (MachineOperand &MO : MRI->reg_nodbg_operands(B)) {
            if (!MO.isTied())
              continue;
            MachineInstr *MI = MO.getParent();
            MachineOperand &Other =
                MI->getOperand(MI->findTiedOperandIdx(MO.getOperandNo()));
            const MachineOperand &Use = MO.isUse() ? MO : Other;
            if (Use.isUndef())
              continue;
            // A store after this tied definition does not alter the tie at
            // the definition. Preserve its home and shorten only its tail.
            if (MO.isDef()) {
              PreserveTiedDefHome = true;
              continue;
            }
            Register Partner = Other.getReg();
            MCRegister Home = Partner.isPhysical() ? MCRegister(Partner)
                                                  : assignedHome(Partner);
            if (!Home)
              continue;
            if (MO.getSubReg() || Other.getSubReg() ||
                Other.isEarlyClobber() || !Partner.isVirtual() ||
                !MDT->dominates(KillMI, MI) || MI->isBundled() ||
                AMDGPURegAllocInsertion::legalBefore(
                    *MI->getParent(), MI->getIterator()) != MI->getIterator() ||
                !MRI->getRegClass(B)->contains(Home)) {
              TiesFit = false;
              break;
            }
            SlotIndex UseAt = LIS->getInstructionIndex(*MI).getRegSlot();
            // Include the gap before the instruction, where its new reload
            // will be inserted. The victim's old ownership will be shortened.
            bool Free = true;
            auto *Adapter = adapterFor(Home);
            if (!Adapter || !Adapter->visitInterferences(
                    Home, UseAt.getBaseIndex().getPrevSlot(), UseAt,
                    [&, Victim = B](const SSARegisterForest::Ownership &O) {
                      if (O.Owner != Victim)
                        Free = false;
                    }))
              Free = false;
            for (const TiedReload &T : TiedReloads)
              if (T.Use == MI && TRI->regsOverlap(T.Home, Home))
                Free = false;
            if (!Free) {
              TiesFit = false;
              break;
            }
            TiedReloads.push_back({MI, MO.getOperandNo(), Partner, Home});
          }
          if (!TiesFit)
            continue;
          const TargetRegisterClass *RC = MRI->getRegClass(B);
          Emitter->beginPass(TRI->hasVGPRs(RC) || TRI->isAGPRClass(RC));
          DEBUG_WITH_TYPE("amdgpu-ssa-fsm-trace", dbgs()
              << "fsm spill-blocker principal=" << printReg(Cur, TRI)
              << " victim=" << printReg(B, TRI) << " kill=" << Start
              << " end=" << End << "\n");
          if (RecoverySpilledValues.overlaps(VMP) &&
              !Emitter->prepareRepeatedSpill(VMP, Start))
            continue;
          auto Repair = spillWithForest(VMP, Start);
          if (PreserveTiedDefHome && assignedHome(B) != OldHome)
            report_fatal_error("blocker spill changed the tied definition home");
          // Commit the checked reload homes before general repair placement
          // can choose a different home for a reload feeding an assigned tie.
          for (const TiedReload &T : TiedReloads) {
            Register Reload = T.Use->getOperand(T.Operand).getReg();
            if (Reload == B || assignedHome(T.Result) != T.Home ||
                !MRI->getRegClass(Reload)->contains(T.Home) ||
                (assignedHome(Reload) && assignedHome(Reload) != T.Home))
              report_fatal_error("tied blocker spill did not preserve reload home");
            if (!assignedHome(Reload)) {
              if (!placementIsFree(Reload, T.Home))
                report_fatal_error("tied blocker reload home is not free after repair");
              assignColor(Reload, T.Home);
            }
            DEBUG_WITH_TYPE("amdgpu-ssa-fsm-trace", dbgs()
                << "fsm tied-blocker-reload victim=" << printReg(B, TRI)
                << " reload=" << printReg(Reload, TRI)
                << " result=" << printReg(T.Result, TRI)
                << " home=" << printReg(T.Home, TRI) << "\n");
          }
          placeRepairedValues(Repair.Affected);
          if (needsRecovery(Cur))
            colorOneInPlace(Cur);
          DEBUG_WITH_TYPE("amdgpu-ssa-fsm-trace", dbgs()
              << "fsm after-blocker-spill principal=" << printReg(Cur, TRI)
              << " home=" << printReg(assignedHome(Cur), TRI) << "\n");
        }
      }
      State = RecoveryState::SelfSplit;
      continue;
    }

    case RecoveryState::SelfSplit: {
      if (!needsRecovery(Cur)) {
        return true;
      }
      RecoveryResult Split = trySelfSplitColor(Cur);
      DEBUG_WITH_TYPE("amdgpu-ssa-fsm-trace", {
        if (CoverageTraceVReg < 0 || Cur.virtRegIndex() == unsigned(CoverageTraceVReg))
          dbgs() << "fsm self-split result=" << unsigned(Split)
                 << " split-count=" << RecoverySplitCount << "\n";
      });
      if (Split == RecoveryResult::Resolved ||
          Split == RecoveryResult::Deferred) {
        if (SSAForensicReporter::enabled())
          Reporter->transformation("self-split", Cur.virtRegIndex());
        // Deferred results are already on UncolorableVRegs. Finish this attempt
        // instead of inventing a single shorter remnant to recover recursively.
        return true;
      }
      assert(Split == RecoveryResult::NoChange);
      State = RecoveryState::Floor;
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
      Emitter->beginPass(IsVGPR);
      MachineInstr *DefMI = MRI->getVRegDef(Cur);
      assert(DefMI && "uncolorable value must have a def in SSA");
      SlotIndex KillIdx = LIS->getInstructionIndex(*DefMI).getRegSlot();
      if (RecoverySpilledValues.overlaps(SpillVMP) &&
          !Emitter->prepareRepeatedSpill(SpillVMP, KillIdx))
        reportPointOverPressure(Cur, IsVGPR, RPLimit,
                                "re-spill-no-progress-or-backing");

      if (SSAForensicReporter::enabled())
        Reporter->transformation("memory-spill", Cur.virtRegIndex());
      auto Repair = spillWithForest(SpillVMP, KillIdx);
      placeRepairedValues(Repair.Affected);
      return true;
    }
    }
  }
}

void AMDGPUSSARegisterAllocator::drainUncolorableWorklist(
    MachineFunction &MF) {
  SSARA_TRACE();
  auto Colored = [&](Register R) { return assignedHome(R) != 0; };
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
      // Successful recovery either places the value or commits a repair.
      // Spill and PHI-web repairs can leave an uncolored def-to-store remnant;
      // process that queued remnant through recovery on the next pass.
      if (recoverUncolorable(Failed))
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
    const TargetRegisterClass *RC = MRI->getRegClass(R);
    bool IsVGPR = TRI->isVGPRClass(RC) || TRI->isAGPRClass(RC);
    unsigned RPLimit =
        allocatablePool(MF, IsVGPR ? RegFile::VGPR : RegFile::SGPR);
    reportPointOverPressure(R, IsVGPR, RPLimit, "worklist-drained");
  }
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

      SmallVector<Register, 16> Pending;
      for (Register V : S.Live) {
        if (!stillCrossing(V) ||
            TRI->getRegSizeInBits(*MRI->getRegClass(V)) != Width)
          continue;
        // Assignment and onChange already check every regmask over the full
        // interval. An assigned value therefore needs no second legality scan.
        if (!assignedHome(V))
          Pending.push_back(V);
      }

      // Holds nothing, or holds one this call does not preserve: take one from
      // CSR(CS), and spill across the call when it has nothing free. CSR(CS) is
      // only a prefilter -- it answers for this call alone, while the register
      // has to survive every clobber site the value is live at, so each
      // candidate is checked against all of them before it is handed out.
      for (Register V : Pending) {
        MCRegister Pick;
        for (MCRegister C : csrSet(MRI->getRegClass(V)))
          if (placementIsFree(V, C)) {
            Pick = C;
            break;
          }
        if (Pick) {
          LLVM_DEBUG(dbgs() << "  " << printReg(V, TRI) << " -> "
                            << TRI->getName(Pick) << "\n");
          if (assignedHome(V))
            unassignColor(V);
          commitColor(V, Pick);
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
        Emitter->beginPass(IsVGPR);
        VRegMaskPair SpillVMP(V, MRI->getMaxLaneMaskForVReg(V));
        if (!Emitter->canSpill(SpillVMP)) {
          LLVM_DEBUG(dbgs() << "  " << printReg(V, TRI)
                            << " -> no legal spill insertion plan\n");
          continue;
        }
        if (SSAForensicReporter::enabled())
          Reporter->transformation("across-call-spill", V.virtRegIndex());
        spillWithForest(SpillVMP, S.CS);
        Changed = true;
      }
    }
  }
}

MCRegister AMDGPUSSARegisterAllocator::prepareTiedDefHome(
    const PendingTie &Tie, MCRegister InputHome) {
  assert(InputHome && "tied input must have a home before inheritance");
  MachineInstr &MI = *Tie.MI;
  MachineBasicBlock *MBB = MI.getParent();
  Register Reg = MI.getOperand(Tie.DefOpIdx).getReg();
  unsigned UseOpIdx = Tie.UseOpIdx;
  MachineOperand &UseMO = MI.getOperand(UseOpIdx);
  if (UseMO.isUndef()) {
    // No input value needs preserving. Make the passthrough follow the result
    // so its former home does not constrain placement or final tie validation.
    UseMO.setReg(Reg);
    UseMO.setSubReg(MI.getOperand(Tie.DefOpIdx).getSubReg());
    MCRegister Home =
        pickFreePhysReg(MRI->getRegClass(Reg), LIS->getInterval(Reg));
    if (!Home && !llvm::is_contained(UncolorableVRegs, Reg))
      UncolorableVRegs.push_back(Reg);
    return Home;
  }

  // A tied result can extend ownership past its input's last use, into a
  // previously assigned blocker. Keep displaced owners uncolored until every
  // ordinary def (including the other tied results) has been visited.
  auto ReleaseTiedBlockers = [&](
      Register Result, MCRegister Home,
      const RegisterForestAdapter::Interference &Query) {
    const auto &Blockers = Query.VirtualOwners;

    // Repair must have selected a home with only evictable virtual blockers.
    assert(MRI->getRegClass(Result)->contains(Home) &&
           "tied repair selected an incompatible home");
    assert(!Query.HasFixedInterference &&
           "tied repair left fixed physical interference");

    // Validate every owner before mutation. The adapter already returns
    // distinct owners in register-ID order.
    assert(llvm::none_of(Blockers, [&](Register Owner) {
      return hasAssignedTiedPartner(Owner);
    }) && "tied repair left an owner with an assigned tied partner");

    for (Register Owner : Blockers) {
      LLVM_DEBUG(dbgs() << "    tied: evict " << printReg(Owner, TRI)
                        << " for " << printReg(Result, TRI) << " -> "
                        << TRI->getName(Home) << "\n");
      unassignColor(Owner);
      if (!llvm::is_contained(UncolorableVRegs, Owner))
        UncolorableVRegs.push_back(Owner);
    }
  };

  // Ordinary two-address def: inherit the tied use's color. When the
  // tied use reads a sub-register (e.g. a 32-bit V_MOV_B32_dpp or
  // V_WRITELANE_B32 tied to one lane of a wider value), the def's
  // class matches that lane, so inherit the sub-register of the color,
  // not the whole super-register.
  MCRegister Chosen = InputHome;
  if (unsigned UseSubIdx = MI.getOperand(UseOpIdx).getSubReg()) {
    Chosen = TRI->getSubReg(InputHome, UseSubIdx);
    assert(Chosen && "Invalid tied-use subreg index");
  }
  auto Query = forestInterferences(Reg, Chosen);
  if (!Query.has_value())
    report_fatal_error(
        "SSARA tied inheritance has an invalid RF query");
  Register Input = UseMO.getReg();
  bool PreserveInput =
      llvm::is_contained(Query->VirtualOwners, Input);
  bool PreserveBlockers = llvm::any_of(
      Query->VirtualOwners, [&](Register Owner) {
        return hasAssignedTiedPartner(Owner);
      });
  if (Query->HasFixedInterference || PreserveInput || PreserveBlockers) {
    // Copy only this tied use when the input remains live or its inherited
    // home conflicts with fixed ownership or an existing tied assignment.
    // Keep that assignment intact; later unassigned ties do not
    // protect an otherwise evictable owner.
    auto InsertPt =
        AMDGPURegAllocInsertion::legalBefore(*MBB, MI.getIterator());
    if (InsertPt != MI.getIterator())
      report_fatal_error(
          "SSARA tied-input copy requires insertion immediately "
          "before the tied instruction");
    Register Tmp = MRI->createVirtualRegister(MRI->getRegClass(Reg));
    unsigned InputSubReg = UseMO.getSubReg();
    {
      MachineInstr *Copy = BuildMI(*MBB, InsertPt, MI.getDebugLoc(),
                                   TII->get(TargetOpcode::COPY), Tmp)
                               .addReg(Input, 0, InputSubReg);
      LIS->InsertMachineInstrInMaps(*Copy);
      UseMO.setReg(Tmp);
      UseMO.setSubReg(0);
      UseMO.setIsKill(true);
      // The copy reads earlier within the existing input range.
      LIS->shrinkToUses(&LIS->getInterval(Input));
      LIS->createAndComputeVirtRegInterval(Tmp);
    }
    notifyLivenessChanged({Input});
    RescueCopies.insert(Tmp);
    LLVM_DEBUG(dbgs() << "    tied: preserve " << printReg(Input, TRI)
                      << " via " << printReg(Tmp, TRI) << " for "
                      << printReg(Reg, TRI) << "\n");
    // Common placement checks both the copy's own interval and the
    // tied result's inherited home. On failure, the existing drain
    // and pending-tie continuation retry the repaired MIR.
    Chosen = colorOneInPlace(Tmp);
    if (!Chosen) {
      UncolorableVRegs.push_back(Tmp);
      return MCRegister();
    }
    Query = forestInterferences(Reg, Chosen);
    if (!Query.has_value())
      report_fatal_error(
          "SSARA tied inheritance has an invalid RF query");
  }
  ReleaseTiedBlockers(Reg, Chosen, *Query);

  LLVM_DEBUG(dbgs() << "    tied: " << printReg(Reg, TRI)
                    << " inherits " << TRI->getName(Chosen) << "\n");
  return Chosen;
}

bool AMDGPUSSARegisterAllocator::resumePendingTies() {
  bool Changed = false;
  for (const PendingTie &Tie : PendingTies) {
    Register Result = Tie.MI->getOperand(Tie.DefOpIdx).getReg();
    if (assignedHome(Result))
      continue;
    MCRegister InputHome =
        assignedHome(Tie.MI->getOperand(Tie.UseOpIdx).getReg());
    if (!InputHome)
      continue;

    // The input recovered after the ordinary def walk passed this instruction.
    // Finish inheritance against the retained assignments. A failed local COPY
    // is queued by the same repair path used during the ordinary walk.
    if (MCRegister Home = prepareTiedDefHome(Tie, InputHome))
      commitColor(Result, Home);
    Changed = true; // Assigned the result or queued the repaired tie.
  }
  return Changed;
}

void AMDGPUSSARegisterAllocator::color() {
  RecoveryScopeTimer PhaseTimer("color");
  SSARA_TRACE();
  initializeForests();
  // The vreg set may have moved since the earlier classification: recovery in
  // the preceding allocation stage can add reload values. Rebuild the width
  // tiers for this stage before anything consults them.
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
  // Discover concrete fixed operands once per epoch. The adapter imports
  // their lane lifetimes; register-mask interference is queried through LIS.
  BitVector PhysicalRegistersToImport(TRI->getNumRegs());
  for (const MachineBasicBlock &MBB : MRI->getMF()) {
    for (const auto &LiveIn : MBB.liveins())
      if (MRI->isAllocatable(LiveIn.PhysReg))
        PhysicalRegistersToImport.set(LiveIn.PhysReg);
    for (const MachineInstr &MI : MBB) {
      if (MI.isDebugInstr())
        continue;
      for (const MachineOperand &MO : MI.operands()) {
        if (MO.isReg() && MO.getReg().isPhysical() && MO.getReg() &&
            MRI->isAllocatable(MO.getReg().asMCReg()))
          PhysicalRegistersToImport.set(MO.getReg());
        if (MO.isRegMask())
          MRI->addPhysRegsUsedFromRegMask(MO.getRegMask());
      }
    }
  }
  std::array<SmallVector<MCPhysReg, 16>, 3> FixedByFile;
  for (int R : PhysicalRegistersToImport.set_bits()) {
    if (!adapterFor(MCRegister(R)))
      report_fatal_error("unsupported fixed physical register");
    RegFile File = poolOf(TRI->getPhysRegBaseClass(MCRegister(R)));
    FixedByFile[static_cast<unsigned>(File)].push_back(R);
  }
  for (unsigned F = 0; F != Adapters.size(); ++F)
    if (!Adapters[F]->assignFixed(FixedByFile[F], *LIS))
      report_fatal_error("fixed physical ownership insertion failed");

  // Give values crossing calls first choice of preserved homes.
  preassignValuesLiveAcrossCalls();

  // Rebuild the tiers again: the spills above introduce the reload remnants,
  // whose width can be one no vreg had before (a 128-bit reload in a function
  // whose tiers were 1024/64/32). The walk below visits defs one tier at a time,
  // so a width missing from the list is never visited and its vregs reach
  // operand rewrite uncolored.
  classifyVRegs();

  DenseSet<Register> ACLSet;
  SmallVector<SlotIndex, 8> CallOnlySites;
  for (const MachineBasicBlock &MBB : MRI->getMF())
    for (const MachineInstr &MI : MBB)
      if (MI.isCall())
        CallOnlySites.push_back(LIS->getInstructionIndex(MI).getRegSlot());
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

      for (MachineInstr &MI : *MBB) {
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
          if (!Reg.isVirtual() ||
              TRI->getRegSizeInBits(*MRI->getRegClass(Reg)) != Width ||
              fileOf(MRI->getRegClass(Reg)) != StageFile || assignedHome(Reg) ||
              ACLSet.contains(Reg) != (Phase == 0))
            continue;
          // Evicted owners are handled after the ordinary definition walk.
          if (llvm::is_contained(UncolorableVRegs, Reg))
            continue;

          MCRegister Chosen;
          unsigned UseOpIdx;
          bool IsTied = MI.isRegTiedToUseOperand(MO.getOperandNo(), &UseOpIdx);
          MCRegister TiedUseColor;
          if (IsTied &&
              (TiedUseColor = assignedHome(MI.getOperand(UseOpIdx).getReg()))) {
            Chosen = prepareTiedDefHome(
                {&MI, MO.getOperandNo(), UseOpIdx}, TiedUseColor);
            if (!Chosen)
              continue;
          } else if (IsTied && MI.getOperand(UseOpIdx).isUndef()) {
            // The tied use is an `undef` passthrough (the DPP "old" source
            // `%N = V_..._dpp undef %N, ...`, a D16 load's untouched half, or a
            // MIX partial def). Its value is a don't-care, so there is no
            // earlier color to inherit -- color the def like a normal def.
            // rewriteOperands() then assigns the same physreg to the self-tied
            // use (same vreg), preserving two-address form.
            Chosen =
                pickFreePhysReg(MRI->getRegClass(Reg), LIS->getInterval(Reg));
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
                                     LIS->getInterval(Reg), Hints, AttemptID);
            if (!Chosen) {
              // No physreg is free across this value's whole range (the
              // %1072/%560 long-liver-through-tuple-churn case). Do NOT assert
              // and do NOT bail: record it and SKIP it (occupy nothing for it),
              // so the rest of the walk colors normally as if this value were
              // absent. The driver spills all collected values afterward, then
              // colors the short reload remainders in place. Skipping is
              // correct because the value is about to be spilled — it holds no
              // register. The COLORFAIL spill-across facts are needed only for
              // the debug dump or the forensic snapshot; skip the whole RF
              // ownership scan on the default path (preserves the original
              // zero-cost behavior).
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

          assignColor(Reg, Chosen);

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

      }
    } // block walk

  } // width loop
  } // phase loop

  LLVM_DEBUG({
    dbgs() << "\nColoring result:\n";
    visitAssignments([&](Register VReg, MCRegister PhysReg) {
      dbgs() << "  " << printReg(VReg, TRI) << " -> " << TRI->getName(PhysReg)
             << "\n";
    });
  });
}

bool AMDGPUSSARegisterAllocator::tiedAssignmentsValid() const {
  SSARA_TRACE();
  auto OperandColor = [&](const MachineOperand &MO) -> MCRegister {
    Register R = MO.getReg();
    MCRegister PR = R.isPhysical() ? R.asMCReg() : assignedHome(R);
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
    // swaps from head to tail.
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
    for (unsigned I = 1; I < Cycle.size(); ++I) {
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
  RecoveryScopeTimer PhaseTimer("lower-phis");
  SSARA_TRACE();
  LLVM_DEBUG(dbgs() << "\n=== SSA Destruction ===\n");

  // Decide all edge placements while RF still describes the original CFG.
  // Splitting edges and emitting physical copies happens only after this plan
  // is complete; no interference query observes partially rewritten liveness.
  struct EdgeCopyPlan {
    MachineBasicBlock *Pred;
    MachineBasicBlock *Successor;
    SmallVector<std::pair<MCRegister, MCRegister>> Copies;
    bool Split;
  };
  SmallVector<EdgeCopyPlan, 8> EdgeCopies;

  SmallVector<MachineInstr *, 16> PHIsToErase;

  // Step-0 metric accumulators (see PHI_Coalescer section 9). Function-local;
  // folded into the STATISTIC counters as we go so -debug-only can print a
  // per-function line without disturbing the global totals.
  unsigned FnCopies = 0, FnFixed = 0, FnUndef = 0;
  uint64_t FnWeight = 0;

  for (MachineBasicBlock &MBB : MF) {
    if (MBB.empty() || !MBB.front().isPHI())
      continue;

    DenseMap<unsigned, SmallVector<std::pair<MCRegister, MCRegister>>>
        PredCopies;

    for (MachineInstr &MI : MBB) {
      if (!MI.isPHI())
        break;

      Register DstVReg = MI.getOperand(0).getReg();
      // Two-stage lowering: handle only this stage's file; the other file's PHIs
      // are lowered (and erased) in its own stage.
      if (fileOf(MRI->getRegClass(DstVReg)) != Only)
        continue;
      MCRegister DstPhys = assignedHome(DstVReg);
      assert(DstPhys && "PHI result not colored");

      // The PHI result physreg flows into this block from each predecessor.
      // After the PHI is erased, the block has no definition of DstPhys, so
      // we must declare it as a live-in so the verifier recognises it.
      if (!MBB.isLiveIn(DstPhys))
        MBB.addLiveIn(DstPhys);

      for (unsigned I = 1, E = MI.getNumOperands(); I < E; I += 2) {
        MachineOperand &SrcMO = MI.getOperand(I);
        MachineBasicBlock *Pred = MI.getOperand(I + 1).getMBB();
        int PredNumber = Pred->getNumber();
        assert(PredNumber >= 0 && "PHI predecessor must have a block number");
        auto &Copies = PredCopies[static_cast<unsigned>(PredNumber)];
        ++NumPhiOperands;

        // An undef incoming value needs no copy, but DstPhys must still be
        // defined so it is live-out of Pred (DstPhys is a live-in of MBB).
        // Encode it as a copy with a null source; it is emitted as an
        // IMPLICIT_DEF of DstPhys during copy insertion below (as generic
        // PHIElimination does for undef PHI operands).
        if (SrcMO.isUndef()) {
          Copies.push_back({MCRegister(), DstPhys});
          ++NumPhiUndefEdges;
          ++FnUndef;
          continue;
        }

        MCRegister SrcPhys = assignedHome(SrcMO.getReg());
        assert(SrcPhys && "PHI source not colored");

        // A PHI source may name a subregister (e.g. %x.sub0). The copy must
        // move the corresponding sub-physreg, not the full tuple, otherwise we
        // emit an illegal width-mismatched copy.
        if (unsigned SubIdx = SrcMO.getSubReg()) {
          SrcPhys = TRI->getSubReg(SrcPhys, SubIdx);
          assert(SrcPhys && "Invalid subreg index on PHI source");
        }

        if (SrcPhys != DstPhys) {
          Copies.push_back({SrcPhys, DstPhys});
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

    SmallVector<unsigned> PredNumbers;
    PredNumbers.reserve(PredCopies.size());
    for (const auto &Entry : PredCopies)
      PredNumbers.push_back(Entry.first);
    llvm::sort(PredNumbers);

    for (unsigned PredNumber : PredNumbers) {
      MachineBasicBlock *Pred = MF.getBlockNumbered(PredNumber);
      assert(Pred && "PHI predecessor block number must resolve");
      auto &Copies = PredCopies.find(PredNumber)->second;
      bool Split = edgeCopiesNeedSplit(Pred, &MBB, Copies);
      EdgeCopies.push_back({Pred, &MBB, std::move(Copies), Split});
    }
  }

  for (auto &Plan : EdgeCopies) {
    MachineBasicBlock *Pred = Plan.Pred;
    MachineBasicBlock &MBB = *Plan.Successor;
    auto &Copies = Plan.Copies;
    MachineBasicBlock *InsertMBB = Pred;
    // The split decision covers null-source (IMPLICIT_DEF) entries too:
    // edgeCopiesNeedSplit only inspects the destination of each pair.
    if (Plan.Split) {
      LLVM_DEBUG(dbgs() << "  Splitting critical edge "
                        << printMBBReference(*Pred) << " -> "
                        << printMBBReference(MBB) << "\n");
      InsertMBB = Pred->SplitCriticalEdge(&MBB, *this);
      assert(InsertMBB && "Failed to split critical edge");
    }

    LLVM_DEBUG(dbgs() << "  Edge " << printMBBReference(*InsertMBB) << " -> "
                      << printMBBReference(MBB) << ":\n");
    auto InsertPt = AMDGPURegAllocInsertion::bodyEnd(*InsertMBB);
    // Remove undef edges from the permutation, but defer their definitions:
    // an undef destination may still hold another PHI's incoming value.
    SmallVector<MCRegister, 4> UndefDests;
    for (auto *It = Copies.begin(); It != Copies.end();) {
      if (!It->first) {
        UndefDests.push_back(It->second);
        It = Copies.erase(It);
      } else {
        ++It;
      }
    }
    resolvePermutation(*InsertMBB, InsertPt, Copies);
    // All incoming copy sources have been consumed before redefining storage.
    for (MCRegister Dst : UndefDests) {
      MachineInstr *IDef =
          BuildMI(*InsertMBB, InsertPt, DebugLoc(),
                  TII->get(TargetOpcode::IMPLICIT_DEF), Dst);
      LIS->InsertMachineInstrInMaps(*IDef);
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
        MCRegister PhysReg = assignedHome(VReg);
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

        // Mark a read `undef` only when NONE of its selected lanes has a
        // reaching value. A full-register use of a partially defined tuple is a
        // real use: LLVM models its defined lanes with subranges and may assign
        // unrelated values to the other lanes. The operand-level flag cannot
        // represent that partial state.
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
            if (readsOnlyUndefLanes(ReadMask, Reached))
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
    visitAssignments([&](Register VReg, MCRegister PhysReg) {
      if (LIS->getInterval(VReg).liveAt(BBStart) && !MBB.isLiveIn(PhysReg))
        MBB.addLiveIn(PhysReg);
    });
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
  // A REG_SEQUENCE with an `undef` source leaves those destination lanes
  // undefined. Mark a use of its result `undef` only when every selected lane
  // comes from an undef source; a partial-undef use remains a real use.
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
        if (readsOnlyUndefLanes(ReadMask, ~UndefLanes))
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
      SmallVector<MCRegister, 4> UndefDests;
      for (unsigned I = 1, E = MI.getNumOperands(); I < E; I += 2) {
        unsigned SubIdx = MI.getOperand(I + 1).getImm();
        MCRegister Expected = TRI->getSubReg(Dst, SubIdx);

        // An undef source requires no copy, but its destination slice must still
        // have a physical definition when another instruction reads the whole
        // REG_SEQUENCE result. Defer that definition until after the parallel
        // copies: the destination may still be one of their sources.
        if (MI.getOperand(I).isUndef()) {
          if (Expected) {
            UndefDests.push_back(Expected);
          } else {
            unsigned First = TRI->getChannelFromSubReg(SubIdx);
            unsigned NumDW = TRI->getSubRegIdxSize(SubIdx) / 32;
            for (unsigned K = 0; K < NumDW; ++K) {
              MCRegister D = TRI->getSubReg(
                  Dst, SIRegisterInfo::getSubRegFromChannel(First + K));
              assert(D && "per-dword destination subregister must exist");
              UndefDests.push_back(D);
            }
          }
          continue;
        }

        MCRegister Src = MI.getOperand(I).getReg().asMCReg();
        // The source class may be wider than the slice it fills (e.g. a 64-bit
        // value held in an sgpr_128 vreg). The slice index then also names the
        // matching sub-register of the source — narrow Src to it so the COPY is
        // width-correct. When Src already matches the slice width, SubIdx names
        // no sub-register of Src and getSubReg() returns 0, leaving Src as-is.
        if (MCRegister SubSrc = TRI->getSubReg(Src, SubIdx))
          Src = SubSrc;
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
      for (MCRegister UndefDst : UndefDests) {
        MachineInstr *IDef =
            BuildMI(MBB, MI, DebugLoc(),
                    TII->get(TargetOpcode::IMPLICIT_DEF), UndefDst);
        LIS->InsertMachineInstrInMaps(*IDef);
      }
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

void AMDGPUSSARegisterAllocator::rewriteStage(MachineFunction &MF,
                                              RegFile Only) {
  RecoveryScopeTimer PhaseTimer("rewrite-stage");
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
  // Add THIS stage's cross-block physreg live-ins now, while its RF ownership
  // is still intact (the next stage clears RF ownership). addPhysRegLiveIns
  // reads RF ownership + LIS, so a single end-of-run call would miss the
  // earlier stage's entries. It only ADDS (sortUniqueLiveIns), so running per
  // stage is safe.
  addPhysRegLiveIns(MF);
}

// Physical assignments are final; spill slots still identify reload sources.
void AMDGPUSSARegisterAllocator::eliminateRedundantReloads(MachineFunction &MF) {
  RecoveryScopeTimer Timer("post-color-reloads");
  enum class ReloadKind { Scalar, Vector, WholeWaveVector };
  struct SliceKey {
    int FrameIndex;
    unsigned ByteOffset;
    unsigned Bytes;
    ReloadKind Kind;

    bool operator==(const SliceKey &Other) const {
      return FrameIndex == Other.FrameIndex && ByteOffset == Other.ByteOffset &&
             Bytes == Other.Bytes && Kind == Other.Kind;
    }
  };
  struct AvailableSlice {
    SliceKey Key;
    MachineInstr *Def; // Retained reload, used for memory-alias queries.
  };
  unsigned RemovedBytes = 0;

  for (MachineBasicBlock &MBB : MF) {
    DenseMap<Register, AvailableSlice> Available;
    bool Changed = false;
    auto EraseIf = [&](auto Predicate) {
      for (auto I = Available.begin(), E = Available.end(); I != E;) {
        auto Current = I++;
        if (Predicate(*Current))
          Available.erase(Current);
      }
    };
    auto Invalidate = [&](const MachineInstr &MI) {
      // Only definitions and regmasks can clobber cached physical values.
      // Aliases include dwords overlapping a true16 or tuple definition.
      for (const MachineOperand &MO : MI.operands()) {
        if (MO.isReg() && MO.isDef() && MO.getReg()) {
          Register Reg = MO.getReg();
          if (MO.getSubReg())
            Reg = TRI->getSubReg(Reg, MO.getSubReg());
          for (MCRegAliasIterator A(Reg, TRI, true); A.isValid(); ++A)
            Available.erase(*A);
        } else if (MO.isRegMask()) {
          EraseIf([&](const auto &Entry) {
            return MO.clobbersPhysReg(Entry.first);
          });
        }
      }
      // Scalar and whole-wave restores are independent of the current EXEC.
      if (MI.modifiesRegister(AMDGPU::EXEC, TRI))
        EraseIf([](const auto &Entry) {
          return Entry.second.Key.Kind == ReloadKind::Vector;
        });
      if (!MI.mayStore())
        return;

      int StoreFI = 0;
      Register StoredReg;
      if (TII->isSpill(MI))
        StoredReg = TII->isStoreToStackSlot(MI, StoreFI);
      unsigned StoreOffset = 0;
      unsigned StoreBytes = 0;
      if (StoredReg) {
        StoreBytes = TRI->getSpillSize(*TRI->getPhysRegBaseClass(StoredReg));
        if (const MachineOperand *Offset =
                TII->getNamedOperand(MI, AMDGPU::OpName::offset))
          StoreOffset = Offset->getImm();
      }
      EraseIf([&](const auto &Entry) {
        const AvailableSlice &A = Entry.second;
        // Overwriting slot bytes breaks their equality with the cached register,
        // even when the instruction does not write that register.
        if (StoredReg)
          return A.Key.FrameIndex == StoreFI &&
                 A.Key.ByteOffset < StoreOffset + StoreBytes &&
                 StoreOffset < A.Key.ByteOffset + A.Key.Bytes;
        // Other stores invalidate only possibly aliasing spill sources. This
        // retains entries across stores proven disjoint by LLVM's memory info.
        return MI.mayAlias(static_cast<AAResults *>(nullptr), *A.Def, false);
      });
    };

    // Visit bundle interiors for their clobbers, but do not rewrite bundles.
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB.instrs())) {
      if (MI.isDebugInstr())
        continue;
      int FI = 0;
      Register Dst;
      if (TII->isSpill(MI)) {
        if (MI.getOpcode() == AMDGPU::SI_SPILL_S32_RESTORE_OFFSET) {
          FI = TII->getNamedOperand(MI, AMDGPU::OpName::addr)->getIndex();
          Dst = MI.getOperand(0).getReg();
        } else {
          Dst = TII->isLoadFromStackSlot(MI, FI);
        }
      }
      if (!Dst || MI.isBundled() || MI.hasOrderedMemoryRef()) {
        Invalidate(MI);
        continue;
      }
      assert(Dst.isPhysical() && "reload optimizer runs after physical rewrite");
      assert(MI.getNumExplicitDefs() == 1 &&
             "spill reload must have one explicit destination");
      const ReloadKind Kind =
          TII->isSGPRSpill(MI) ? ReloadKind::Scalar
          : TII->isWWMRegSpillOpcode(MI.getOpcode()) ? ReloadKind::WholeWaveVector
                                                  : ReloadKind::Vector;
      unsigned SourceOffset = 0;
      if (const MachineOperand *Offset =
              TII->getNamedOperand(MI, AMDGPU::OpName::offset))
        SourceOffset = Offset->getImm() * (Kind == ReloadKind::Scalar ? 4 : 1);
      const auto *RC = TRI->getPhysRegBaseClass(Dst);
      const unsigned Bytes = TRI->getSpillSize(*RC);
      // A true16 restore stays a single two-byte slice. Wider registers are
      // tracked as individual dwords, so writing one dword keeps the others.
      const unsigned PartBytes = std::min(Bytes, 4u);
      SmallVector<Register, 32> Parts;
      if (Bytes > 4)
        for (int16_t Sub : TRI->getRegSplitParts(RC, 4))
          Parts.push_back(TRI->getSubReg(Dst, Sub));
      else
        Parts.push_back(Dst);
      assert(Parts.size() * PartBytes == Bytes && "incomplete reload split");
      SmallVector<unsigned, 32> Missing;
      for (unsigned I = 0; I != Parts.size(); ++I) {
        SliceKey Key{FI, SourceOffset + I * PartBytes, PartBytes, Kind};
        auto Found = Available.find(Parts[I]);
        if (Found == Available.end() || !(Found->second.Key == Key))
          Missing.push_back(I);
      }
      if (Missing.size() == Parts.size()) {
        // Keep the original wide operation if nothing can be eliminated.
        Invalidate(MI);
        for (unsigned I = 0; I != Parts.size(); ++I)
          Available[Parts[I]] = {
              {FI, SourceOffset + I * PartBytes, PartBytes, Kind}, &MI};
        continue;
      }

      // Whole-wave restore pseudos are already single-dword operations.
      assert(Kind != ReloadKind::WholeWaveVector || Missing.empty());
      for (unsigned I : Missing) {
        const unsigned Offset = SourceOffset + I * PartBytes;
        unsigned Opcode;
        if (Kind == ReloadKind::Scalar)
          Opcode = AMDGPU::SI_SPILL_S32_RESTORE_OFFSET;
        else
          Opcode = TII->getVectorRegSpillRestoreOpcode(
              Parts[I], TRI->getPhysRegBaseClass(Parts[I]), PartBytes, *MFI);
        auto Reload = BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(Opcode), Parts[I])
                          .addFrameIndex(FI);
        if (Kind == ReloadKind::Scalar) {
          Reload.addImm(Offset / 4);
          for (const MachineOperand &MO : MI.implicit_operands())
            if (MO.isReg() && MO.isUse() && MO.getReg() != AMDGPU::EXEC)
              Reload.addReg(MO.getReg(), RegState::Implicit);
        } else {
          // Vector restore offsets are already expressed in bytes.
          Reload.addReg(TII->getNamedOperand(MI, AMDGPU::OpName::soffset)->getReg())
              .addImm(Offset);
        }
        for (MachineMemOperand *MMO : MI.memoperands())
          Reload.addMemOperand(MF.getMachineMemOperand(MMO, I * PartBytes,
                                                       PartBytes));
        LIS->InsertMachineInstrInMaps(*Reload);
        Invalidate(*Reload);
        Available[Parts[I]] = {{FI, Offset, PartBytes, Kind},
                               Reload.getInstr()};
      }
      RemovedBytes += (Parts.size() - Missing.size()) * PartBytes;
      if (LIS->getSlotIndexes()->hasIndex(MI))
        LIS->RemoveMachineInstrFromMaps(MI);
      MI.eraseFromParent();
      Changed = true;
    }
    // Sharing is confined to this block, so successor live-ins are unchanged.
    // Repair flags once, not once for every eliminated reload and every slice.
    if (Changed)
      recomputeLivenessFlags(MBB);
  }
  LLVM_DEBUG(dbgs() << "  post-color reload bytes removed: " << RemovedBytes
                    << "\n");
}

void AMDGPUSSARegisterAllocator::finalizeAfterRewrite(MachineFunction &MF) {
  SSARA_TRACE();
  // Run ONCE after both files are physical.
  if (EliminatePostColorReloads)
    eliminateRedundantReloads(MF);
  finalizeProperties(MF);
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
  RecoveryTimings.clear();
  RecoveryScopeTimer AllocatorTimer("allocator-total");
  SSARA_TRACE();
  TRI =
      static_cast<const SIRegisterInfo *>(MF.getSubtarget().getRegisterInfo());
  TII = static_cast<const SIInstrInfo *>(MF.getSubtarget().getInstrInfo());
  MRI = &MF.getRegInfo();
  MFI = MF.getInfo<SIMachineFunctionInfo>();
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
  Emitter = std::make_unique<SSASpillEmitter>(MF, LIS, Indexes, MDT);
  Emitter->setReporter(Reporter.get());

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

  // TWO INDEPENDENT ALLOCATION STAGES: SGPR first, then VGPR/AGPR. The SGPR
  // stage may spill SGPRs; those spills lower (downstream) to VGPR lanes needing
  // WWM scratch. Between stages we reserve ceil(spilledSGPRlanes / wavesize)
  // VGPRs (VGPRReserve, withheld by allocatablePool) so the VGPR stage does not
  // consume the whole file. Each stage colors, recovers, and rewrites ONLY its
  // file's vregs (StageFile filters color()/recovery/rewriteStage);
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

  clearColors();
  MaxVGPRIdx = 0;
  MaxSGPRIdx = 0;
  MaxAGPRIdx = 0;
  UncolorableVRegs.clear();
  // Reset recovery bookkeeping at the start of this allocation stage.
  RescueCopies.clear();
  RehomedVRegs.clear();
  RecoverySpilledValues.clear();
  RecoverySplitCount = 0;
  color();

  // Spill-on-coloring-failure (approach A). A pure Hack coloring can fail on
  // AMDGPU even at RP ≤ limit (the %1072/%560 long-liver-through-tuple-churn
  // class): no single physreg is free across the value's whole range. color()
  // collected every such value in UncolorableVRegs and skipped it, so RF
  // ownership now holds a valid assignment for EVERYTHING ELSE — untouched from
  // here on.
  //
  // For each collected value: spill it (store-at-def + reload-at-use), which
  // replaces its one long range with short reload ranges, then color those
  // reload remainders IN PLACE against the frozen RF ownership. We never
  // re-color an already-placed value, so no successfully-colored value can be
  // perturbed into a new failure (the reason we do NOT recolor from clean).
  // Recover failures against retained ownership. Do not restart coloring after
  // a spill or tied-operand mismatch.
  while (true) {
    if (!UncolorableVRegs.empty()) {
      NumTierSpills += UncolorableVRegs.size();
      drainUncolorableWorklist(MF);
    }

    // Recovery can home an input whose tied def was deferred by color().
    // Resume those defs against the retained assignments. New evictions or
    // repair copies go through the existing drain on the next iteration.
    if (resumePendingTies())
      continue;

    if (tiedAssignmentsValid())
      break;

    report_fatal_error(
        "SSARA retained assignments require local tied-operand repair");
  }

    // Rewrite THIS stage's vregs to physregs now, so the next stage starts with
    // only the other file still virtual (and the SGPR spill count is final for
    // the VGPR stage's VGPRReserve). The whole-function finalize runs once after
    // the loop.

    rewriteStage(MF, StageFile);
    clearColors();
  } // end of the two allocation stages

  // E17 RunCompleted: flush this function's record to the configured sinks.
  Reporter->endRun(UncolorableVRegs.size());

  finalizeAfterRewrite(MF);

  AllocatorTimer.finish();
  if (EnableRecoveryTiming)
    errs() << "phase-function " << MF.getName() << "\n";
  dumpRecoveryTimings();
  return true;
}

MachineFunctionPass *llvm::createAMDGPUSSARegisterAllocatorPass() {
  SSARA_TRACE();
  return new AMDGPUSSARegisterAllocator();
}
