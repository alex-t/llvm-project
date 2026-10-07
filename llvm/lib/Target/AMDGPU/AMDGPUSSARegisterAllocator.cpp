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
#include "SSARegisterForestAdapter.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "SSARATrace.h"
#include <algorithm>
#include <array>
#include <iterator>
#include <tuple>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-ssa-register-allocator"

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
  return Emitter->spillOneVMP(
      Value, Kill, [this](ArrayRef<Register> Affected) {
        notifyLivenessChanged(Affected);
      });
}

std::optional<SSASpillEmitter::RepairResult>
AMDGPUSSARegisterAllocator::splitWithForest(
    Register VR, MachineBasicBlock::iterator Point) {
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

void AMDGPUSSARegisterAllocator::dumpOccupancyMap(
    const TargetRegisterClass *RC, SlotIndex SI, const char *Tag,
    const LiveInterval *VI) const {
  OccupancyFacts Facts;
  collectOccupancy(RC, SI, VI, Facts);
  dbgs()
      << "  [OCCMAP " << Tag << "] " << Facts.ClassName << " @" << SI
      << " usable=" << Facts.FreeUsable << " blocked=" << Facts.FreeClobbered
      << " occupied=" << Facts.Occupied << " total=" << Facts.Total << '\n'
      << "    " << Facts.Map << '\n'
      << "    # occupied here; x blocked elsewhere in the interval; . free\n";
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

bool AMDGPUSSARegisterAllocator::planAreaSpillSet(
    const TightRegion &R, ArrayRef<RegionSlot> Slots,
    ArrayRef<SpillCandidateInput> Inputs,
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
      if (LIS->getInstructionIndex(*Slots[S].MI).getRegSlot() >= FirstTermIdx)
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
    unsigned INo = DirectIndex.lookup(I.V);
    SmallBitVector Freed = DirectLiveSlots[INo];
    Freed.reset(DirectResidentSlots[INo]);
    for (int S = Freed.find_first(); S >= 0; S = Freed.find_next(S))
      C.FreedSlots.push_back(S);
    C.Action.Area = uint64_t(Width) * C.FreedSlots.size();
    if (C.Action.Area)
      Candidates.push_back(std::move(C));
  }

  llvm::sort(Candidates, [](const Candidate &A, const Candidate &B) {
    if (A.Action.Area != B.Action.Area)
      return A.Action.Area > B.Action.Area;
    return A.Action.V.id() < B.Action.V.id();
  });

  SmallVector<RegionSlot, 32> Virtual(Slots.begin(), Slots.end());
  auto totalShort = [&]() {
    unsigned Total = 0;
    for (const RegionSlot &S : Virtual)
      Total += S.Short;
    return Total;
  };

  const unsigned InitialShort = totalShort();
  unsigned BestShort = InitialShort;

  for (Candidate &Cand : Candidates) {
    if (!llvm::any_of(Cand.FreedSlots,
                      [&](unsigned S) { return Virtual[S].Short != 0; }))
      continue;
    for (unsigned S : Cand.FreedSlots)
      Virtual[S].Live.erase(Cand.Action.V.id());
    for (RegionSlot &S : Virtual) {
      SmallVector<TierDemand, 8> Tiers;
      S.Short = demandDeficiency(S.Live, R.File, Tiers);
    }
    unsigned CurrentShort = totalShort();
    if (CurrentShort < InitialShort) {
      BestShort = CurrentShort;
      Actions.push_back(Cand.Action);
    }
    LLVM_DEBUG(dbgs() << "    [AREA] choose "
                      << printReg(Cand.Action.V, TRI)
                      << " area=" << Cand.Action.Area
                      << " remaining-short=" << CurrentShort
                      << " footprint=direct"
                      << "\n");
    break;
  }

  if (RemainingShort)
    *RemainingShort = BestShort;
  if (Actions.empty()) {
    LLVM_DEBUG(dbgs() << "    [AREA] no measured improvement; emit nothing\n");
    Actions.clear();
    return false;
  }
  LLVM_DEBUG(dbgs() << "    [AREA] retain one direct candidate short="
                    << InitialShort << "->" << BestShort
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
  // Per region: select at most one candidate whose direct reload footprint
  // improves copied deficiency. The caller recolors once and runs recovery again.
  // Returns true if any spill was performed.
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
    // per-slot live sets that the direct planner evaluates.
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
      if (!assignedHome(V))
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
    // into the output.
    llvm::sort(Cands, [](const Cand &A, const Cand &B) {
      return A.VReg.id() < B.VReg.id();
    });

    SmallVector<SpillCandidateInput, 32> Inputs;
    for (const Cand &C : Cands)
      if (!Spilled.count(C.VReg))
        Inputs.push_back({C.VReg, C.Lanes});

    Emitter->beginPass(IsVectorFile);
    SmallVector<AreaSpillAction, 16> Actions;
    unsigned VirtualShort = 0;
    if (!planAreaSpillSet(R, Slots, Inputs, Actions, &VirtualShort))
      continue;

    bool Legal = llvm::all_of(Actions, [&](const AreaSpillAction &A) {
      return Emitter->canSpill(VRegMaskPair(A.V, A.Lanes));
    });
    if (!Legal)
      continue;

    for (const AreaSpillAction &A : Actions) {
      assert(LIS->hasInterval(A.V) && !LIS->getInterval(A.V).empty() &&
             "planned victim lost its interval before set emission");
      LLVM_DEBUG(dbgs() << "    [AREA] spill " << printReg(A.V, TRI)
                        << " area=" << A.Area
                        << " lanes=" << PrintLaneMask(A.Lanes) << "\n");
      unassignColor(A.V);
      spillWithForest(VRegMaskPair(A.V, A.Lanes),
                      LIS->getInterval(A.V).beginIndex());
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
  collectBlockers(Failed, Overlappers);

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

  // COVERAGE pick: a live-through blocker frees ALL of F; otherwise the
  // born-in-F blocker with the EARLIEST def frees the longest tail [B.def,FE).
  // Blocker LI LENGTH is irrelevant (the reload lands at the use).
  //
  // Prefer live-through victims, then earlier definitions, then register ID.
  // Query traversal order must not decide which victim recovery chooses.
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
    if (!R.isVirtual() || !LIS->hasInterval(R) || assignedHome(R) ||
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
    Emitter->beginPass(IsVGPR);
    unassignColor(B);
    auto Repair =
        spillWithForest(VRegMaskPair(B, MRI->getMaxLaneMaskForVReg(B)), FS);
    // The erase and the spill are COMMITTED to LIS and MIR — there is no
    // rollback. Every value this uncoloured or created must therefore end up
    // either coloured or ON THE WORKLIST; a piece that is merely dropped reaches
    // final rewrite as a live virtual register, which is the emit-time abort
    // "non-undef virtual register not colored". The born-in-F branch below has
    // always worked this way; this one returned NoOp and dropped the pieces.
    // Observed on tahiti bitcast_v64bf16_to_v128i8_scalar: %2760 was evicted
    // here, declined to recolour, and asserted in rewriteOperands.
    placeRepairedValues(Repair.Affected);
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

  // The blocker starts inside Failed. SSA repair may create multiple split
  // results at joins/backedges; retain all of them for the common worklist.
  MachineInstr *BDef = MRI->getVRegDef(B);
  assert(BDef && "colored blocker must have a def in SSA");
  if (BDef->isPHI())
    return RecoveryResult::NoChange;
  auto Split = splitWithForest(Failed, BDef->getIterator());
  if (!Split)
    return RecoveryResult::NoChange;

  RecoverySpilledVRegs.insert(B);
  Emitter->beginPass(IsVGPR);
  unassignColor(B);
  auto Repair =
      spillWithForest(VRegMaskPair(B, MRI->getMaxLaneMaskForVReg(B)),
                      LIS->getInterval(B).beginIndex());
  placeRepairedValues(Repair.Affected);
  queueUnassignedValues(Split->Affected);
  return RecoveryResult::Deferred;
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
              // Fixed owners block placement but are never movable victims.
              Profile.Blockers.push_back({Owner, {Start, End}, BlockedHomes});
            }))
      report_fatal_error("invalid register-forest placement query");
  }
}

bool AMDGPUSSARegisterAllocator::pickPeelableRun(Register V, MCRegister &PR,
                                                 SlotIndex &Bound) const {
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

  // Compatibility only: the final solver will model mandatory in-register
  // occurrences directly and delete this heuristic gate.
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

  // Use the old numerical limit across this function, including blocker splits,
  // so worklist retries cannot restart the budget. Exhaustion uses the floor.
  if (RecoverySplitCount >= 64)
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

AMDGPUSSARegisterAllocator::RecoveryResult
AMDGPUSSARegisterAllocator::tryCrossFileHome(Register R) {
  SSARA_TRACE();
  if (!ST->hasMAIInsts())
    return RecoveryResult::NoChange;

  auto TryOne = [&](Register V,
                    const TargetRegisterClass *ForcedTarget) -> bool {
    const TargetRegisterClass *RC = MRI->getRegClass(V);
    if ((!TRI->hasVGPRs(RC) && !TRI->isAGPRClass(RC)) || !LIS->hasInterval(V) ||
        assignedHome(V))
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
  collectBlockers(R, Crossers);
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

    unassignColor(Crosser);
    if (TryOne(Crosser, Sibling))
      return RecoveryResult::Changed;

    // No MIR is inserted before the forced color succeeds, so this restores the
    // exact pre-strategy state after a failed probe.
    MRI->setRegClass(Crosser, SavedRC);
    assignColor(Crosser, OldPR);
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
  // back to Web so every later predicate observes current MIR/LIS/RF ownership.
  // Termination comes from monotone strategy-local measures: a web is erased
  // once, a value is re-homed at most once, a blocker is spilled at most once,
  // and self-splitting consumes a function-wide budget across worklist retries.
  const MachineFunction &MF = MRI->getMF();

  LLVM_DEBUG(dbgs() << "FALLBACK for " << printReg(Failed, TRI) << " ["
                    << LIS->getInterval(Failed).beginIndex() << ","
                    << LIS->getInterval(Failed).endIndex() << ")\n");

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
      Emitter->beginPass(IsVGPR);
      auto Result = Emitter->spillPhiWeb(
          Web, [this](ArrayRef<Register> Affected) {
            notifyLivenessChanged(Affected);
          });
      placeRepairedValues(Result.Affected);
      if (SSAForensicReporter::enabled())
        Reporter->transformation("phi-web-spill", Web.Root.virtRegIndex());
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
      if (Blocker == RecoveryResult::Resolved ||
          Blocker == RecoveryResult::Deferred) {
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
      RecoveryResult Split = trySelfSplitColor(Cur);
      if (Split == RecoveryResult::Resolved ||
          Split == RecoveryResult::Deferred) {
        if (SSAForensicReporter::enabled())
          Reporter->transformation("self-split", Cur.virtRegIndex());
        // Deferred results are already on UncolorableVRegs. Finish this attempt
        // instead of inventing a single shorter remnant to recover recursively.
        if (CompleteCurrent())
          return true;
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
      Emitter->beginPass(IsVGPR);
      MachineInstr *DefMI = MRI->getVRegDef(Cur);
      assert(DefMI && "uncolorable value must have a def in SSA");
      SlotIndex KillIdx = LIS->getInstructionIndex(*DefMI).getRegSlot();
      if (SSAForensicReporter::enabled())
        Reporter->transformation("memory-spill", Cur.virtRegIndex());
      auto Repair = spillWithForest(SpillVMP, KillIdx);
      placeRepairedValues(Repair.Affected);
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
      if (!ReportFailure && RecoverySpilledVRegs.count(Failed))
        return false;
      unsigned SplitsBefore = RecoverySplitCount;
      recoverUncolorable(Failed);

      // A split can queue only uncolored results. Process them on the next
      // pass even when this original was not colored during the current pass.
      if (Colored(Failed) || RecoverySplitCount != SplitsBefore)
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
  // Add THIS stage's cross-block physreg live-ins now, while its RF ownership
  // is still intact (the next stage clears RF ownership). addPhysRegLiveIns
  // reads RF ownership + LIS, so a single end-of-run call would miss the
  // earlier stage's entries. It only ADDS (sortUniqueLiveIns), so running per
  // stage is safe.
  addPhysRegLiveIns(MF);
}

void AMDGPUSSARegisterAllocator::finalizeAfterRewrite(MachineFunction &MF) {
  SSARA_TRACE();
  // Run ONCE after both files are physical.
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
  // Only per function: a rescue copy stays a rescue copy across the full recolor
  // that clears UncolorableVRegs again below, since the copy instruction remains.
  RescueCopies.clear();
  RehomedVRegs.clear();
  RecoverySpilledVRegs.clear();
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
  // Each width-1 reload provably settles: point pressure at the use ≤ RPLimit <
  // file size. Recover every coloring failure in place. Recovery can repoint a
  // tied use after its def inherited the old use's color; if that happens,
  // recolor the mutated MIR from clean and run recovery again for any new
  // failures. The irreversible recovery sets are deliberately preserved across
  // this loop.
  while (true) {
    bool RecoveryComplete = true;
    if (!UncolorableVRegs.empty()) {
      NumTierSpills += UncolorableVRegs.size();
      RecoveryComplete =
          drainUncolorableWorklist(MF, /*ReportFailure=*/false);
    }

    if (!RecoveryComplete) {
      LLVM_DEBUG(dbgs() << "=== recovery: late direct planner ===\n");
      if (reduceRegionPressure(MF)) {

        clearColors();
        MaxVGPRIdx = 0;
        MaxSGPRIdx = 0;
        MaxAGPRIdx = 0;
        UncolorableVRegs.clear();

        color();
      }
      drainUncolorableWorklist(MF);
    }

    // Recovery can home an input whose tied def was deferred by color().
    // Resume those defs before considering a clean recolor. New evictions or
    // repair copies go through the existing drain on the next iteration.
    if (resumePendingTies())
      continue;

    if (tiedAssignmentsValid())
      break;

    clearColors();
    MaxVGPRIdx = 0;
    MaxSGPRIdx = 0;
    MaxAGPRIdx = 0;
    UncolorableVRegs.clear();

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
    clearColors();
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
