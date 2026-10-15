//===-- SSASpillEmitter.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Exec-safe SSA spill/reload emission mechanism (see SSASpillEmitter.h).
/// Reloads are placed directly before reachable uses.
///
//===----------------------------------------------------------------------===//

#include "SSASpillEmitter.h"
#include "SSAForensicReporter.h"
#include "AMDGPU.h"
#include "AMDGPURegAllocInsertion.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIRegisterInfo.h"
#include "VRegMaskPair.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/Register.h"
#include <algorithm>
#include "SSARATrace.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-ssa-register-spiller"

STATISTIC(NumSpills, "Number of register spills");
STATISTIC(NumReloads, "Number of register reloads");

bool llvm::usesSpilledVMP(const MachineInstr *MI, VRegMaskPair SpilledVMP,
                          const SIRegisterInfo *TRI,
                          const MachineRegisterInfo *MRI) {
  SSARA_TRACE();
  Register SpilledReg = SpilledVMP.getVReg();
  LaneBitmask SpilledMask = SpilledVMP.getLaneMask();

  // Quick check: does the instruction read this virtual register at all?
  // This handles partial defines correctly (read-modify-write)
  if (!MI->readsVirtualRegister(SpilledReg))
    return false;

  // Found a use, now check if it overlaps with spilled lanes
  for (const MachineOperand &MO : MI->uses()) {
    if (MO.isReg() && MO.getReg() == SpilledReg) {
      LaneBitmask UseMask = VRegMaskPair(MO, TRI, MRI).getLaneMask();
      // Check if this use overlaps with the spilled lanes
      if ((UseMask & SpilledMask).any()) {
        return true;
      }
    }
  }

  return false;
}

SSASpillEmitter::SSASpillEmitter(MachineFunction &MF, LiveIntervals *LIS,
                                 SlotIndexes *Indexes, MachineDominatorTree *DT)
    : MF(MF), LIS(LIS), Indexes(Indexes), DT(DT) {
  SSARA_TRACE();
  TRI = static_cast<const SIRegisterInfo *>(MF.getSubtarget().getRegisterInfo());
  TII = static_cast<const SIInstrInfo *>(MF.getSubtarget().getInstrInfo());
  MRI = &MF.getRegInfo();
  FrameInfo = &MF.getFrameInfo();
}

void SSASpillEmitter::beginPass(bool IsVGPR) {
  SSARA_TRACE();
  IsVGPRPass = IsVGPR;
  // Fresh SSA updater per pass (caches IDF computations per run).
  SSAUpdater = std::make_unique<MachineLaneSSAUpdater>(MF, *LIS, *DT, *TRI);
}

bool SSASpillEmitter::ordinaryVALUUseIsExecSafe(MachineInstr *UseMI) const {
  assert(UseMI && !UseMI->isPHI() && "expected an ordinary use");
  MachineBasicBlock &MBB = *UseMI->getParent();
  auto UsePos = UseMI->getIterator();
  auto ReloadPos = AMDGPURegAllocInsertion::legalBefore(MBB, UsePos);
  return AMDGPURegAllocInsertion::registerStable(
      ReloadPos, UsePos, AMDGPU::EXEC, TRI);
}

bool SSASpillEmitter::canSpill(VRegMaskPair VMP) const {
  Register VReg = VMP.getVReg();
  MachineInstr *DefMI = MRI->getVRegDef(VReg);
  if (DefMI && !DefMI->isImplicitDef() && !MRI->use_nodbg_empty(VReg) &&
      !DefMI->registerDefIsDead(VReg, /*TRI=*/nullptr) &&
      !AMDGPURegAllocInsertion::legalAfter(*DefMI))
    return false;

  const TargetRegisterClass *RC = VMP.getRegClass(MRI, TRI);
  if (!TRI->isVGPRClass(RC) && !TRI->isAGPRClass(RC))
    return true;

  for (MachineInstr &UseMI : MRI->use_nodbg_instructions(VReg)) {
    if (UseMI.isPHI() || isSpillInstr(&UseMI) ||
        !usesSpilledVMP(&UseMI, VMP, TRI, MRI))
      continue;
    if (!ordinaryVALUUseIsExecSafe(&UseMI))
      return false;
  }
  return true;
}

bool SSASpillEmitter::prepareRepeatedSpill(VRegMaskPair VMP,
                                             SlotIndex KillIdx) {
  SmallVector<MachineInstr *, 8> Uses, PhiUses;
  collectReloadUses(VMP, KillIdx, Uses, PhiUses);
  if (Uses.empty() && PhiUses.empty())
    return false;
  MachineInstr *Store = WebGroundStores.lookup(VMP);
  if (MachineInstr *Existing = StoredAtDefinition.lookup(VMP))
    if (Existing != Store)
      return true;
  if (!Store)
    return false;
  int FI = 0;
  bool FoundFI = false;
  for (const MachineOperand &MO : Store->operands())
    if (MO.isFI()) {
      FI = MO.getIndex();
      FoundFI = true;
      break;
    }
  assert(FoundFI && "spill store must name a frame index");

  SmallPtrSet<MachineInstr *, 16> Targets;
  for (MachineInstr *Use : Uses)
    Targets.insert(Use);
  SmallPtrSet<MachineBasicBlock *, 8> EdgeTargets;
  for (MachineInstr *Phi : PhiUses)
    for (unsigned I = 1; I < Phi->getNumOperands(); I += 2) {
      const MachineOperand &MO = Phi->getOperand(I);
      if (MO.isReg() && MO.getReg() == VMP.getVReg() && MO.readsReg() &&
          VRegMaskPair(MO, TRI, MRI).overlaps(VMP))
        EdgeTargets.insert(Phi->getOperand(I + 1).getMBB());
    }

  // A different ground writer must not reach any prospective reload without
  // passing the store of this value again. Scan each reachable block once per
  // writer; this is a conservative CFG check, not reload commoning.
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB) {
      if (&MI == Store || !MI.mayStore() ||
          !llvm::any_of(MI.operands(), [FI](const MachineOperand &MO) {
            return MO.isFI() && MO.getIndex() == FI;
          }))
        continue;
      SmallPtrSet<MachineBasicBlock *, 16> Seen;
      SmallVector<MachineBasicBlock *, 16> Work;
      auto Visit = [&](MachineBasicBlock &BB,
                       MachineBasicBlock::iterator Begin) {
        for (auto I = Begin; I != BB.end(); ++I) {
          if (&*I == Store)
            return false;
          if (Targets.count(&*I))
            return true;
        }
        if (EdgeTargets.count(&BB))
          return true;
        for (MachineBasicBlock *Succ : BB.successors())
          if (Seen.insert(Succ).second)
            Work.push_back(Succ);
        return false;
      };
      if (Visit(MBB, std::next(MI.getIterator())))
        return false;
      while (!Work.empty()) {
        MachineBasicBlock *BB = Work.pop_back_val();
        if (Visit(*BB, BB->begin()))
          return false;
      }
    }
  Virt2StackSlotMap[VMP] = FI;
  StoredAtDefinition[VMP] = Store;
  return true;
}

int SSASpillEmitter::assignVirt2StackSlot(VRegMaskPair VMP) {
  SSARA_TRACE();
  assert(VMP.getVReg().isVirtual() && "Expected virtual register");

  // Check if we already have a stack slot for this VRegMaskPair
  auto It = Virt2StackSlotMap.find(VMP);
  if (It != Virt2StackSlotMap.end())
    return It->second;

  // Create a new stack slot
  const TargetRegisterClass *RC = VMP.getRegClass(MRI, TRI);
  int FI = createSpillSlot(RC);
  Virt2StackSlotMap[VMP] = FI;
  // Count SGPR spill lanes once per spilled value (this is the miss path — a
  // dedup hit above returns early). Only when SGPR spills consume VGPR lanes
  // downstream; the RA reserves VGPRs against this count for WWM lowering.
  if (TRI->spillSGPRToVGPR() && TRI->isSGPRClass(RC))
    NumSGPRSpillLanes += TRI->getRegSizeInBits(*RC) / 32;
  return FI;
}

int SSASpillEmitter::createSpillSlot(const TargetRegisterClass *RC) {
  SSARA_TRACE();
  unsigned SpillSize = TRI->getSpillSize(*RC);
  Align SpillAlign = TRI->getSpillAlign(*RC);
  return FrameInfo->CreateSpillStackObject(SpillSize, SpillAlign);
}

SSASpillEmitter::RepairResult
SSASpillEmitter::spillOneVMP(VRegMaskPair VMP, SlotIndex KillIdx,
                            LivenessCallback OnChange) {
  SSARA_TRACE();
  assert(canSpill(VMP) && "spill has no legal, EXEC-safe insertion plan");

  LLVM_DEBUG({
    Register VReg = VMP.getVReg();
    StringRef Name = MRI->getVRegName(VReg);
    dbgs() << "\nspillOneVMP(): Processing VMP ";
    if (!Name.empty())
      dbgs() << "%" << Name;
    else
      dbgs() << printReg(VReg, TRI);
    dbgs() << " with mask " << PrintLaneMask(VMP.getLaneMask())
           << ", KillIdx=" << KillIdx << "\n";
  });

  // Step 2a: Store register at definition point (when EXEC is full).
  // This avoids EXEC drift issues by ensuring all lanes are stored before any
  // divergent control flow can modify EXEC. Store placement is fixed at the def
  // and is independent of KillIdx (which only decides where the reg is freed).
  MachineInstr *DefStoreMI = spillAtDefinition(VMP);
  // DefStoreMI may be null when the value's def is IMPLICIT_DEF (undef): no store
  // is emitted (storing undef is illegal MIR). Reloads then load an uninitialized
  // slot, which is a semantic don't-care for an undef value.
  (void)DefStoreMI;

  // Step 2c: Get stack slot for reload phase
  assignVirt2StackSlot(VMP);

  // Step 2d: Place reloads at uses reachable from KillIdx, so uses above KillIdx
  // keep the original register and are not reloaded.
  return emitReloadsAndRepairSSA(VMP, KillIdx, OnChange);
}

SSASpillEmitter::RepairResult
SSASpillEmitter::spillPhiWeb(const PhiWeb &Web, LivenessCallback OnChange) {
  SSARA_TRACE();
  LastWebErased.clear();
  // Detection, the shared-slot soundness gate, AND the reload-feasibility gate all
  // ran in the RA (closePhiWeb + webReloadFeasible); this is pure mechanics over an
  // already-closed, sound, feasible web — it cannot fail.
  assert(Web.valid() && "spillPhiWeb requires a closed, sound web");
  const unsigned FirstNewVReg = MRI->getNumVirtRegs();
  Register PhiResult = Web.Root;
  const auto &PhiMembers = Web.PhiMembers;
  const auto &GroundOps = Web.GroundOps;
  const auto &GroundEdges = Web.GroundEdges;
  RepairResult Result;
  Result.Affected.append(PhiMembers.begin(), PhiMembers.end());
  for (Register G : GroundOps)
    if (G.isVirtual())
      Result.Affected.push_back(G);
  auto CollectChanges = [&](ArrayRef<Register> Affected) {
    Result.Affected.append(Affected.begin(), Affected.end());
  };

  LLVM_DEBUG(dbgs() << "\nspillPhiWeb(): root " << printReg(PhiResult, TRI)
                    << " web=" << PhiMembers.size() << " PHIs, "
                    << GroundOps.size() << " ground ops\n");

  // Freeze every member's external-use list before stores or SSA repair can
  // change operands. Membership is that of the original closed web.
  struct MemberReloads {
    Register Member;
    SmallVector<MachineInstr *, 8> Uses;
    SmallVector<MachineInstr *, 4> PhiUses;
  };
  SmallVector<MemberReloads, 8> ReloadPlans;
  for (Register M : PhiMembers) {
    MachineInstr *Def = MRI->getUniqueVRegDef(M);
    assert(Def && Def->isPHI() && "closed web member must have a PHI def");
    MemberReloads Plan{M, {}, {}};
    collectReloadUses(VRegMaskPair(M, MRI->getMaxLaneMaskForVReg(M)),
                      LIS->getInstructionIndex(*Def).getRegSlot(), Plan.Uses,
                      Plan.PhiUses);
    llvm::erase_if(Plan.PhiUses, [&](MachineInstr *Use) {
      return PhiMembers.count(Use->getOperand(0).getReg());
    });
    if (!Plan.Uses.empty() || !Plan.PhiUses.empty())
      ReloadPlans.push_back(std::move(Plan));
  }

  VRegMaskPair RootVMP(PhiResult, MRI->getMaxLaneMaskForVReg(PhiResult));
  int FI = assignVirt2StackSlot(RootVMP);

  // --- 3. Store every web ground operand into the shared slot on its PHI edge,
  // all reaching the slot in ONE physreg R (required: the SGPR-lane lowering sees
  // one lane for a width-1 slot and asserts on N distinct source SGPRs). We do NOT
  // recolor the long-lived ground ops (a color free across their full ranges
  // rarely exists). Instead ONE operand is the "main": it keeps its own color R
  // and is stored directly (NO copy — so it never interferes with itself, so R
  // stays reusable). Every OTHER operand gets a short-lived `c = COPY op` at its
  // predecessor end, colored R and killed into the store; R need only be free at
  // THAT pred end (where the main op is not live). The color is chosen FIRST, from
  // the operands as they are — inserting copies before choosing would make each
  // copy compete for R at the saturated pred ends (the 49-decline bug).
  // (A valid() web has >=1 ground op, hence >=1 ground edge — no empty guard.)

  // Store each web ground operand AT ITS DEF into the shared slot. Storing at the
  // def (not the predecessor end) is what DRAINS the predecessor tail: these
  // operands are live [def, block-end] only because they are live-out to the join
  // PHI; once that PHI use is gone (the PHI is erased below) and the value is in
  // memory from its def onward, each collapses to [def, store], vacating the tail
  // where N of them piled up (the COLORFAIL cluster). A sub-register operand stores
  // only its lane (the slot is the result-narrow class). Dedup identical
  // (value,pred) edges — a value on two edges from one pred needs one store.
  SmallDenseSet<std::pair<unsigned, MachineBasicBlock *>, 16> StoredEdge;
  for (const GroundEdge &GE : GroundEdges) {
    Register G = GE.Reg;
    if (!G.isVirtual())
      continue;
    MachineInstr *GDef = MRI->getUniqueVRegDef(G);
    if (!GDef || GDef->isImplicitDef())
      continue; // live-in / undef: nothing to store
    if (!StoredEdge.insert({G.id(), GE.Pred}).second)
      continue;
    MachineBasicBlock &GBB = *GDef->getParent();
    auto LegalAfter = AMDGPURegAllocInsertion::legalAfter(*GDef);
    assert(LegalAfter && "closed PHI web has no legal ground-store position");
    MachineBasicBlock::iterator DIt = *LegalAfter;
    const TargetRegisterClass *StoreRC =
        GE.SubReg ? TRI->getSubRegisterClass(MRI->getRegClass(G), GE.SubReg)
                  : MRI->getRegClass(G);
    TII->storeRegToStackSlot(GBB, DIt, G, /*isKill=*/false, FI, StoreRC, TRI, G,
                             MachineInstr::NoFlags, GE.SubReg);
    MachineInstr *GroundStore = &*std::prev(DIt);
    LIS->InsertMachineInstrInMaps(*GroundStore);
    WebGroundStores[VRegMaskPair(
        G, GE.SubReg ? TRI->getSubRegIndexLaneMask(GE.SubReg)
                     : MRI->getMaxLaneMaskForVReg(G))] = GroundStore;
    ++NumSpills;
    if (Reporter && SSAForensicReporter::enabled())
      Reporter->spillEmitted(G.virtRegIndex(), "phi-web-store-at-def");
    LLVM_DEBUG(dbgs() << "  phi-web: store " << printReg(G, TRI)
                      << (GE.SubReg ? ".subN" : "") << " AT DEF -> FI" << FI
                      << " in " << printMBBReference(GBB) << "\n");
  }

  // Reload each PHI member's EXTERNAL uses from the shared slot and erase the PHIs.
  // The reload redefs are fresh vregs returned to the driver for placement;
  // the value delivery is memory on whichever
  // edge ran. This is the join-dissolution: the spilled web is FINAL (no register
  // coalescing remains — the coalescer's job on it is done by the spill).
  for (const MemberReloads &Plan : ReloadPlans) {
    VRegMaskPair VMP(Plan.Member, MRI->getMaxLaneMaskForVReg(Plan.Member));
    Virt2StackSlotMap[VMP] = FI;
    emitReloadsAndRepairSSA(VMP, Plan.Uses, Plan.PhiUses, CollectChanges);
  }

  for (Register M : PhiMembers) {
    // A member already spilled as a plain value owns a store-at-def to its OWN
    // slot. The web supersedes that spill: the ground ops write the
    // shared web slot FI, and any external uses of M were reloaded from FI by the
    // loop above (it repointed M via Virt2StackSlotMap[VMP] = FI). So M's old
    // store-at-def is now dead — nothing reads its slot — and erasing M's PHI def
    // below would leave it reading a defless, uncolored vreg (asserts in
    // rewriteOperands). Remove the stale store and its memo.
    VRegMaskPair MVMP(M, MRI->getMaxLaneMaskForVReg(M));
    if (MachineInstr *MStore = StoredAtDefinition.lookup(MVMP)) {
      LIS->RemoveMachineInstrFromMaps(*MStore);
      MStore->eraseFromParent();
      StoredAtDefinition.erase(MVMP);
    }
    if (MachineInstr *D = MRI->getUniqueVRegDef(M)) {
      // Repair may have changed these operands. Their final liveness must
      // reflect removal of this PHI use.
      for (const MachineOperand &MO : D->uses())
        if (MO.isReg() && MO.getReg().isVirtual())
          Result.Affected.push_back(MO.getReg());
      LIS->RemoveMachineInstrFromMaps(*D);
      D->eraseFromParent();
    }
    if (LIS->hasInterval(M))
      LIS->removeInterval(M);
    LastWebErased.push_back(M);
  }

  // All original web PHIs are gone. Classify uses against the complete set
  // of PHIs created by this repair, so internal cycles do not keep themselves
  // alive. A PHI feeding an external reader keeps its PHI dependencies alive.
  SmallSetVector<MachineInstr *, 16> RepairPHIs;
  for (Register R : Result.Affected) {
    if (R.virtRegIndex() < FirstNewVReg)
      continue;
    if (MachineInstr *Def = MRI->getUniqueVRegDef(R))
      if (Def->isPHI())
        RepairPHIs.insert(Def);
  }
  SmallSetVector<MachineInstr *, 16> LivePHIs;
  for (MachineInstr *Phi : RepairPHIs) {
    Register R = Phi->getOperand(0).getReg();
    if (llvm::any_of(MRI->use_nodbg_operands(R), [&](MachineOperand &MO) {
          return MO.readsReg() && !RepairPHIs.count(MO.getParent());
        }))
      LivePHIs.insert(Phi);
  }
  for (unsigned I = 0; I != LivePHIs.size(); ++I) {
    MachineInstr *Phi = LivePHIs[I];
    for (const MachineOperand &MO : Phi->uses()) {
      if (!MO.isReg() || !MO.getReg().isVirtual() || !MO.readsReg())
        continue;
      MachineInstr *Def = MRI->getUniqueVRegDef(MO.getReg());
      if (RepairPHIs.count(Def))
        LivePHIs.insert(Def);
    }
  }
  // No surviving reader depends on this group; erase it as a whole.
  for (MachineInstr *Phi : RepairPHIs) {
    if (LivePHIs.count(Phi))
      continue;
    for (const MachineOperand &MO : Phi->uses())
      if (MO.isReg() && MO.getReg().isVirtual())
        Result.Affected.push_back(MO.getReg());
    MRI->markUsesInDebugValueAsUndef(Phi->getOperand(0).getReg());
    LIS->RemoveMachineInstrFromMaps(*Phi);
    Phi->eraseFromParent();
  }

  // Original web PHIs and unused repair-PHI groups have been removed.
  // Only now can a reload's remaining readers determine whether it is dead.
  for (unsigned I = 0; I != Result.Affected.size(); ++I) {
    Register R = Result.Affected[I];
    if (R.virtRegIndex() < FirstNewVReg)
      continue;
    MachineInstr *Def = MRI->getUniqueVRegDef(R);
    if (!Def || !isReloadInstr(Def) ||
        llvm::any_of(MRI->use_nodbg_operands(R),
                     [](const MachineOperand &MO) { return MO.readsReg(); }))
      continue;
    for (const MachineOperand &MO : Def->uses())
      if (MO.isReg() && MO.getReg().isVirtual())
        Result.Affected.push_back(MO.getReg());
    MRI->markUsesInDebugValueAsUndef(R);
    LIS->RemoveMachineInstrFromMaps(*Def);
    Def->eraseFromParent();
  }

  llvm::sort(Result.Affected,
             [](Register A, Register B) { return A.id() < B.id(); });
  Result.Affected.erase(
      std::unique(Result.Affected.begin(), Result.Affected.end()),
      Result.Affected.end());
  // Recompute from the completed transformation before publishing to the forest.
  // Retired IDs remain in Affected so their old ownership is removed.
  for (Register R : Result.Affected) {
    if (LIS->hasInterval(R))
      LIS->removeInterval(R);
    if (!MRI->def_empty(R)) {
      LIS->createAndComputeVirtRegInterval(R);
      continue;
    }
    assert(!llvm::any_of(MRI->use_nodbg_operands(R),
                        [](const MachineOperand &MO) { return MO.readsReg(); }) &&
           "PHI-web spill left a live use without a definition");
    // Only undef operands remain. The verifier requires an interval for every
    // referenced vreg; this empty interval carries no liveness to allocate.
    if (!MRI->reg_nodbg_empty(R))
      LIS->createEmptyInterval(R);
  }
  OnChange(Result.Affected);

  LLVM_DEBUG(dbgs() << "  phi-web: coalesced, erased " << LastWebErased.size()
                    << "/" << PhiMembers.size() << " PHIs, " << GroundOps.size()
                    << " ground stores -> FI" << FI << "\n");
  return Result;
}

std::optional<SSASpillEmitter::RepairResult>
SSASpillEmitter::splitLiveRangeAt(Register V,
                                  MachineBasicBlock::iterator SplitPt,
                                  LivenessCallback OnChange) {
  SSARA_TRACE();
  if (!LIS->hasInterval(V))
    return std::nullopt;
  MachineBasicBlock &MBB = *SplitPt->getParent();
  SplitPt = AMDGPURegAllocInsertion::legalBefore(MBB, SplitPt);
  auto CopyPt = AMDGPURegAllocInsertion::copyPosition(MBB, SplitPt, V, *DT, *TII);
  // Decline this split without changing MIR or liveness. The caller retains
  // the original interval and can continue recovery with another action.
  if (!CopyPt)
    return std::nullopt;
  SplitPt = *CopyPt;
  if (SplitPt == MBB.end())
    return std::nullopt;
  SlotIndex Before = LIS->getInstructionIndex(*SplitPt).getBaseIndex();
  if (!LIS->getInterval(V).liveAt(Before))
    return std::nullopt;
  assert(MRI->hasOneDef(V) && "split requires an SSA input");

  // Placement is chosen before liveness and SSA repair. The new definition
  // serves the redirected uses; other paths retain the original definition.
  MachineInstr *CopyMI =
      BuildMI(MBB, SplitPt, SplitPt->getDebugLoc(),
              TII->get(TargetOpcode::COPY), V)
          .addReg(V);
  LIS->InsertMachineInstrInMaps(*CopyMI);
  LIS->removeInterval(V);
  LIS->createAndComputeVirtRegInterval(V);
  RepairResult Result =
      SSAUpdater->repairSSAForNewDefs(V, {CopyMI}, OnChange);

  Register CopyResult = CopyMI->getOperand(0).getReg();
  if (!llvm::any_of(MRI->use_nodbg_operands(CopyResult),
                    [](const MachineOperand &MO) { return MO.readsReg(); }))
    report_fatal_error("live split point produced no reachable uses after SSA "
                       "repair; inconsistent liveness or broken repair");
  LLVM_DEBUG(dbgs() << "splitLiveRangeAt(): " << printReg(V, TRI) << " -> "
                    << printReg(CopyResult, TRI) << "\n");
  return Result;
}

MachineInstr *SSASpillEmitter::spillAtDefinition(VRegMaskPair VMP) {
  SSARA_TRACE();
  if (MachineInstr *Existing = StoredAtDefinition.lookup(VMP)) {
    LLVM_DEBUG({
      StringRef Name = MRI->getVRegName(VMP.getVReg());
      dbgs() << "spillAtDefinition(): Already stored ";
      if (!Name.empty())
        dbgs() << "%" << Name;
      else
        dbgs() << printReg(VMP.getVReg(), TRI);
      dbgs() << " at definition\n";
    });
    return Existing;
  }

  Register VReg = VMP.getVReg();
  LaneBitmask Mask = VMP.getLaneMask();

  LLVM_DEBUG({
    StringRef Name = MRI->getVRegName(VReg);
    dbgs() << "spillAtDefinition(): Storing ";
    if (!Name.empty())
      dbgs() << "%" << Name;
    else
      dbgs() << printReg(VReg, TRI);
    dbgs() << " with mask " << PrintLaneMask(Mask)
           << " right after definition\n";
  });

  // Find the definition point
  MachineInstr *DefMI = MRI->getVRegDef(VReg);
  if (!DefMI) {
    LLVM_DEBUG(
        dbgs() << "spillAtDefinition(): No definition found (live-in?)\n");
    return nullptr;
  }

  // The value is an undef (IMPLICIT_DEF). There is nothing to save — storing it
  // would emit `SI_SPILL_*_SAVE <undef reg>` (the source is dead right after the
  // IMPLICIT_DEF), which the machine verifier rejects as "using an undefined
  // physical register". Skip the store; the reload side re-materializes the undef
  // (an uninitialized slot load of an undef value is semantically a don't-care).
  if (DefMI->isImplicitDef()) {
    LLVM_DEBUG(dbgs() << "spillAtDefinition(): def is IMPLICIT_DEF (undef) -> "
                         "no store emitted\n");
    return nullptr;
  }

  // The def is dead: the value has no reader, so there is nothing to save and
  // nothing will be reloaded. Emitting the store anyway leaves `dead %v = ...`
  // immediately followed by a read of %v, which the verifier rejects once colored
  // with "Using an undefined physical register" -- the same failure the
  // IMPLICIT_DEF case above avoids (si-sgpr-spill: `dead %202:sreg_64` is the
  // unused sdst of a V_DIV_SCALE, stored into %stack.12 as $sgpr52_sgpr53).
  if (MRI->use_nodbg_empty(VReg) ||
      DefMI->registerDefIsDead(VReg, /*TRI=*/nullptr)) {
    LLVM_DEBUG(dbgs() << "spillAtDefinition(): def is dead (no readers) -> "
                         "no store emitted\n");
    return nullptr;
  }

  MachineBasicBlock *DefMBB = DefMI->getParent();
  // Store right after the def. When the def is a PHI, all PHIs must stay
  // contiguous at the block top, so std::next(PHI) could land the store between
  // PHIs ("PHI after non-PHI"). Insert after the last PHI instead.
  auto LegalAfter = AMDGPURegAllocInsertion::legalAfter(*DefMI);
  assert(LegalAfter && "spill definition lies in the terminator sequence");
  MachineBasicBlock::iterator InsertAfter = *LegalAfter;

  // Get or create stack slot
  int FI = assignVirt2StackSlot(VMP);

  // Determine SubRegIdx from lane mask
  unsigned SubRegIdx = VMP.getSubReg(MRI, TRI);

  // Get the appropriate register class
  const TargetRegisterClass *FullRC = TRI->getRegClassForReg(*MRI, VReg);
  const TargetRegisterClass *RC =
      SubRegIdx == AMDGPU::NoRegister
          ? nullptr
          : TRI->getSubRegisterClass(FullRC, SubRegIdx);
  if (!RC) {
    // A named lane span need not have a register class — sub6..sub15 of a
    // 512-bit tuple is 10 channels wide and the legal widths stop at 8 before
    // jumping to 16. Store the whole register, which is what VRegMaskPair's
    // getRegClass sized the stack slot for.
    RC = FullRC;
    SubRegIdx = AMDGPU::NoRegister;
  }

  LLVM_DEBUG({
    if (SubRegIdx != AMDGPU::NoRegister) {
      StringRef Name = MRI->getVRegName(VReg);
      dbgs() << "spillAtDefinition(): Storing subregister "
             << TRI->getSubRegIndexName(SubRegIdx) << " of ";
      if (!Name.empty())
        dbgs() << "%" << Name;
      else
        dbgs() << printReg(VReg, TRI);
      dbgs() << "\n";
    }
  });

  // Emit the store instruction right after definition with isKill=false
  // This ensures all lanes are stored when EXEC is full
  TII->storeRegToStackSlot(*DefMBB, InsertAfter, VReg, /*isKill=*/false, FI, RC,
                           TRI, VReg, MachineInstr::NoFlags, SubRegIdx);

  // Get the inserted store instruction
  MachineInstr &StoreMI = *std::prev(InsertAfter);

  // Update LiveIntervals
  LIS->InsertMachineInstrInMaps(StoreMI);

  // Mark this register as stored at definition
  StoredAtDefinition[VMP] = &StoreMI;

  LLVM_DEBUG(dbgs() << "spillAtDefinition(): Stored: " << StoreMI);
  ++NumSpills;
  if (Reporter && SSAForensicReporter::enabled())
    Reporter->spillEmitted(VReg.virtRegIndex(), "store-at-def");

  return &StoreMI;
}

std::pair<Register, MachineInstr *>
SSASpillEmitter::getOrCreateReloadInBlock(MachineBasicBlock *BB,
                                          VRegMaskPair SpilledVMP,
                                          MachineInstr *InsertBefore,
                                          LaneBitmask ReloadMask) {
  SSARA_TRACE();
  Register OrigVReg = SpilledVMP.getVReg();

  // Vector reloads may narrow to the lanes requested by the use. SGPR restore
  // pseudos have no source offset, so they must reload the entire stored slice.
  LaneBitmask Slice = ReloadMask & SpilledVMP.getLaneMask();
  if (Slice.none())
    Slice = SpilledVMP.getLaneMask();

  // The reload REDEFINES OrigVReg[.sub] (a transient SSA violation) that the
  // spiller repairs inline via reaching-VNI reconstruction (see
  // emitReloadsAndRepairSSA). RC/SubRegIdx describe the reloaded SLICE (which
  // may be narrower than the spilled VMP when a use only reads some lanes).
  //
  // Derive the slice's subreg from its 32-bit channel span. VRegMaskPair's
  // getSubReg only matches an EXACT subreg lane mask, which fails for a
  // contiguous-but-not-named range (e.g. sub17..sub31 of a vreg_1024) and would
  // silently fall back to a full-width reload. Instead round the slice up to
  // whole channels [FirstChan, LastChan] (reloading an extra covered lane is
  // always safe -- the full slot holds it) and name that span directly via
  // getSubRegFromChannel. The byte offset into the slot is FirstChan * 4.
  const TargetRegisterClass *FullRC = TRI->getRegClassForReg(*MRI, OrigVReg);
  LaneBitmask FullMask = MRI->getMaxLaneMaskForVReg(OrigVReg);
  const TargetRegisterClass *RC = FullRC;
  unsigned SubRegIdx = 0;
  unsigned FirstChan = 0;
  unsigned TotalChans = TRI->getNumCoveredRegs(FullMask);
  if (TRI->isSGPRClass(FullRC)) {
    // Match spillAtDefinition: a named, supported slice starts at slot lane 0;
    // otherwise the store and slot cover the whole register. The destination
    // subreg identifies the lanes redefined for SSA repair, not a slot offset.
    unsigned SpillSubIdx = SpilledVMP.getSubReg(MRI, TRI);
    if (const TargetRegisterClass *SpillRC =
            SpillSubIdx ? TRI->getSubRegisterClass(FullRC, SpillSubIdx) : nullptr) {
      RC = SpillRC;
      SubRegIdx = SpillSubIdx;
    }
  } else if (Slice != FullMask) {
    // Contiguous channel span covering the slice.
    unsigned First = ~0u, Last = 0;
    for (unsigned C = 0; C < TotalChans; ++C) {
      LaneBitmask ChMask =
          TRI->getSubRegIndexLaneMask(TRI->getSubRegFromChannel(C));
      if ((ChMask & Slice).any()) {
        First = std::min(First, C);
        Last = std::max(Last, C);
      }
    }
    unsigned NumChans = Last - First + 1;

    // getSubRegFromChannel only names legal AMDGPU tuple widths. Round the span
    // up to the next legal width and slide the window down if the tail would
    // overrun the register. Reloading a few extra covered channels is always
    // safe -- the full slot holds them. If no legal window narrower than the
    // whole register is found, fall back to a whole-register reload.
    static constexpr unsigned LegalWidths[] = {1, 2, 3, 4, 5, 6, 7, 8, 16};
    const TargetRegisterClass *SubRC = nullptr;
    unsigned SubIdx = 0, Start = First;
    for (unsigned W : LegalWidths) {
      if (W < NumChans || W >= TotalChans)
        continue;
      unsigned S = std::min(First, TotalChans - W);
      unsigned Idx = TRI->getSubRegFromChannel(S, W);
      const TargetRegisterClass *Cand =
          Idx ? TRI->getSubRegisterClass(FullRC, Idx) : nullptr;
      if (Cand) {
        // getSubRegisterClass(av_*, subN) returns a VGPR-only slice class, so a
        // slice of an AGPR-eligible (vector-super) spill loses AGPR-eligibility ->
        // its reload can only land in an arch-VGPR. Promote the slice to the av
        // super-class of the same width so the reload may re-home to a free AGPR
        // (Greedy's v_accvgpr scratch).
        if (TRI->isVectorSuperClass(FullRC))
          if (const TargetRegisterClass *AV =
                  TRI->getVectorSuperClassForBitWidth(W * 32))
            Cand = AV;
        SubRC = Cand;
        SubIdx = Idx;
        Start = S;
        break;
      }
    }
    if (SubRC) {
      RC = SubRC;
      SubRegIdx = SubIdx;
      FirstChan = Start;
    }
    // else: leave RC=FullRC, SubRegIdx=0 -> whole-register reload (safe).
  }

  // Determine insertion point within the legal non-PHI block body.
  auto Requested =
      InsertBefore ? InsertBefore->getIterator() : BB->getFirstTerminator();
  auto InsertIt = AMDGPURegAllocInsertion::legalBefore(*BB, Requested);
  assert((!InsertBefore || !IsVGPRPass ||
          AMDGPURegAllocInsertion::registerStable(
              InsertIt, Requested, AMDGPU::EXEC, TRI)) &&
         "ordinary VALU reload crosses an EXEC modification");
  int FI = assignVirt2StackSlot(SpilledVMP);

  TII->loadRegFromStackSlot(*BB, InsertIt, OrigVReg, FI, RC, TRI, Register(),
                            MachineInstr::NoFlags, SubRegIdx);

  // Get the reload instruction and add to slot indexes
  MachineInstr *ReloadMI = &*std::prev(InsertIt);
  LIS->InsertMachineInstrInMaps(*ReloadMI);

  // Vector reloads encode the source position as a byte offset. SGPR reloads
  // start at slot lane 0 and reload the same width as the store above.
  if (SubRegIdx != 0 && FirstChan != 0) {
    if (MachineOperand *Off =
            TII->getNamedOperand(*ReloadMI, AMDGPU::OpName::offset)) {
      Off->setImm(Off->getImm() + FirstChan * 4);
      LLVM_DEBUG(dbgs() << "    reload sub-slice offset: channel " << FirstChan
                        << " -> byte " << FirstChan * 4 << "\n");
    }
  }

  // loadRegFromStackSlot no longer marks a partial (subreg) reload def undef.
  // Under reload-as-redef of OrigVReg the un-reloaded (complement) lanes are
  // usually still live -- they were never spilled -- so the partial redef must
  // PRESERVE them (an implicit RMW read), keeping them live in the recomputed
  // interval so the reaching-VNI reconstruction can source them. Mark the def
  // undef only in the rare case where the complement is dead across the reload;
  // otherwise a plain partial redef would read lanes with no reaching def.
  if (SubRegIdx != 0) {
    // Complement = all lanes of OrigVReg NOT redefined by this reload. The
    // reload redefines the rounded channel span (SubRegIdx's lane mask), which
    // may be slightly wider than the requested Slice -- use the actual redefined
    // mask so preserved (complement) lanes are computed correctly.
    LaneBitmask RedefMask = TRI->getSubRegIndexLaneMask(SubRegIdx);
    LaneBitmask Complement = MRI->getMaxLaneMaskForVReg(OrigVReg) & ~RedefMask;
    SlotIndex RSlot = LIS->getInstructionIndex(*ReloadMI).getRegSlot();
    const LiveInterval &LI = LIS->getInterval(OrigVReg);
    bool ComplementLive = false;
    if (LI.hasSubRanges()) {
      for (const LiveInterval::SubRange &S : LI.subranges())
        if ((S.LaneMask & Complement).any() && S.liveAt(RSlot))
          ComplementLive = true;
    } else if (Complement.any() && LI.liveAt(RSlot))
      ComplementLive = true;
    ReloadMI->getOperand(0).setIsUndef(!ComplementLive);
  }

  LLVM_DEBUG(dbgs() << "    Created reload (redef) in "
                    << printMBBReference(*BB)
                    << (InsertBefore ? " before use" : " at block end") << ": "
                    << printReg(OrigVReg, TRI) << "\n");
  ++NumReloads;
  if (Reporter && SSAForensicReporter::enabled())
    Reporter->reloadEmitted(OrigVReg.virtRegIndex(),
                            InsertBefore ? "reload-before-use"
                                         : "reload-at-block-end");
  return {OrigVReg, ReloadMI};
}

bool SSASpillEmitter::insertReloadForUse(MachineInstr *UseMI,
                                         VRegMaskPair SpilledVMP) {
  SSARA_TRACE();
  Register SpilledReg = SpilledVMP.getVReg();
  LaneBitmask SpilledMask = SpilledVMP.getLaneMask();

  if (UseMI->isPHI()) {
    // PHI use: reload must be in predecessor block(s) that provide the spilled
    // reg
    bool InsertedAny = false;
    for (unsigned I = 1; I < UseMI->getNumOperands(); I += 2) {
      MachineOperand &ValOp = UseMI->getOperand(I);
      MachineOperand &BBOp = UseMI->getOperand(I + 1);
      if (!ValOp.isReg() || ValOp.getReg() != SpilledReg)
        continue;

      // Check if this PHI operand's lanes overlap with spilled lanes
      LaneBitmask UseMask = VRegMaskPair(ValOp, TRI, MRI).getLaneMask();
      if ((UseMask & SpilledMask).none())
        continue;

      MachineBasicBlock *PredBB = BBOp.getMBB();

      // Place the reload redef; SSA is repaired inline after all reloads.
      // Reload only the lanes this PHI operand reads.
      getOrCreateReloadInBlock(PredBB, SpilledVMP, nullptr,
                               UseMask & SpilledMask);
      InsertedAny = true;
      LLVM_DEBUG(dbgs() << "    PHI use: reload in "
                        << printMBBReference(*PredBB) << "\n");
    }
    return InsertedAny;
  }

  // Non-PHI use: reload only the lanes this instruction actually reads (union
  // over its operands that read SpilledReg, intersected with the spilled lanes).
  // A use reading a sub-slice of a wide tuple (e.g. a REG_SEQUENCE operand
  // %r.sub6_sub7...) then pulls back only those lanes, not the whole tuple.
  LaneBitmask UseMask = LaneBitmask::getNone();
  for (const MachineOperand &MO : UseMI->uses())
    if (MO.isReg() && MO.getReg() == SpilledReg)
      UseMask |= VRegMaskPair(MO, TRI, MRI).getLaneMask();
  UseMask &= SpilledMask;

  getOrCreateReloadInBlock(UseMI->getParent(), SpilledVMP, UseMI, UseMask);
  return true;
}

void SSASpillEmitter::collectReloadUses(
    VRegMaskPair SpilledVMP, SlotIndex KillIdx,
    SmallVectorImpl<MachineInstr *> &Uses,
    SmallVectorImpl<MachineInstr *> &PhiUses) const {
  Register SpilledReg = SpilledVMP.getVReg();
  MachineInstr *KillMI = Indexes->getInstructionFromIndex(KillIdx);
  assert(KillMI && "kill must name an instruction");
  // Collect ordinary uses reachable from the kill and PHI edge uses. Direct
  // placement intentionally emits one reload at every such use; a post-color
  // coalescer may merge redundant reloads after physical assignments are known.
  for (MachineInstr &UseMI : MRI->use_nodbg_instructions(SpilledReg)) {
    if (isSpillInstr(&UseMI))
      continue;
    if (!usesSpilledVMP(&UseMI, SpilledVMP, TRI, MRI))
      continue;
    if (UseMI.isPHI()) {
      PhiUses.push_back(&UseMI);
      continue;
    }
    if (!DT->dominates(KillMI, &UseMI) &&
        !SSAUpdater->isUseReachableFromDef(KillMI, &UseMI, SpilledReg))
      continue;
    Uses.push_back(&UseMI);
  }
  llvm::sort(Uses, [this](MachineInstr *A, MachineInstr *B) {
    if (A == B)
      return false;
    if (DT->dominates(A, B))
      return true;
    if (DT->dominates(B, A))
      return false;
    return LIS->getInstructionIndex(*A) < LIS->getInstructionIndex(*B);
  });
}

SSASpillEmitter::RepairResult SSASpillEmitter::emitReloadsAndRepairSSA(
    VRegMaskPair SpilledVMP, SlotIndex KillIdx, LivenessCallback OnChange) {
  SmallVector<MachineInstr *, 8> Uses;
  SmallVector<MachineInstr *, 4> PhiUses;
  collectReloadUses(SpilledVMP, KillIdx, Uses, PhiUses);
  return emitReloadsAndRepairSSA(SpilledVMP, Uses, PhiUses, OnChange);
}

SSASpillEmitter::RepairResult SSASpillEmitter::emitReloadsAndRepairSSA(
    VRegMaskPair SpilledVMP, ArrayRef<MachineInstr *> Uses,
    ArrayRef<MachineInstr *> PhiUses, LivenessCallback OnChange) {
  SSARA_TRACE();
  Register SpilledReg = SpilledVMP.getVReg();

  LLVM_DEBUG({
    dbgs() << "\n=== emitReloadsAndRepairSSA() [Option 3: redef-only] ===\n";
    dbgs() << "Spilled: " << printReg(SpilledReg, TRI) << " mask "
           << PrintLaneMask(SpilledVMP.getLaneMask()) << "\n";
  });

  for (MachineInstr *U : Uses) {
    if (!usesSpilledVMP(U, SpilledVMP, TRI, MRI))
      continue;
    insertReloadForUse(U, SpilledVMP);
  }

  for (MachineInstr *U : PhiUses) {
    if (!usesSpilledVMP(U, SpilledVMP, TRI, MRI))
      continue;
    insertReloadForUse(U, SpilledVMP);
  }

  // Final recompute (reflects all reloads) for the reconstruction. Correct
  // placement above put a reload on every freed edge that needs one, so the
  // reload redefs kill the original throughout the freed region -- the
  // recompute's merges are then genuine (original only on kill-free paths).
  if (LIS->hasInterval(SpilledReg))
    LIS->removeInterval(SpilledReg);
  LIS->createAndComputeVirtRegInterval(SpilledReg);

  // TRIAL: inline reaching-VNI repair, one call per reload redef. Each call
  // renames the reload def to a fresh vreg, places PHIs at the merges recorded
  // in OrigVReg's recomputed interval, and rewrites dominated uses -- restoring
  // SSA and keeping LiveIntervals correct inline (so the spiller's RP stays
  // accurate on the next iteration).
  SmallVector<MachineInstr *, 4> ReloadDefs;
  for (MachineInstr &D : MRI->def_instructions(SpilledReg))
    if (isReloadInstr(&D))
      ReloadDefs.push_back(&D);
  RepairResult Result =
      SSAUpdater->repairSSAForNewDefs(SpilledReg, ReloadDefs, OnChange);
  LLVM_DEBUG(dbgs() << "\nemitReloadsAndRepairSSA() complete\n");
  return Result;
}

