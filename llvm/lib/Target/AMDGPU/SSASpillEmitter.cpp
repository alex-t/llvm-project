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
                                 SlotIndexes *Indexes, MachineDominatorTree *DT,
                                 const MachineLoopInfo *MLI)
    : MF(MF), MLI(MLI), LIS(LIS), Indexes(Indexes), DT(DT) {
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

void SSASpillEmitter::spillOneVMP(VRegMaskPair VMP, SlotIndex KillIdx) {
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
  emitReloadsAndRepairSSA(VMP, KillIdx);
}

void SSASpillEmitter::spillPhiWeb(
    const PhiWeb &Web, llvm::function_ref<void(Register)> ColorFreshVReg) {
  SSARA_TRACE();
  LastWebErased.clear();
  LastWebGround.clear();
  // Detection, the shared-slot soundness gate, AND the reload-feasibility gate all
  // ran in the RA (closePhiWeb + webReloadFeasible); this is pure mechanics over an
  // already-closed, sound, feasible web — it cannot fail.
  assert(Web.valid() && "spillPhiWeb requires a closed, sound web");
  Register PhiResult = Web.Root;
  const auto &PhiMembers = Web.PhiMembers;
  const auto &GroundOps = Web.GroundOps;
  const auto &GroundEdges = Web.GroundEdges;

  // Report the ground ops so the driver marks them Spilled (a web stores each once;
  // without this the driver re-selects a stored ground op as a plain victim and
  // double-spills it — proven on cf512: %999/%944/%1003/%948 spilled WEB then PLAIN).
  for (Register G : GroundOps)
    LastWebGround.push_back(G);

  LLVM_DEBUG(dbgs() << "\nspillPhiWeb(): root " << printReg(PhiResult, TRI)
                    << " web=" << PhiMembers.size() << " PHIs, "
                    << GroundOps.size() << " ground ops\n");

  // --- 2. Collect the members that have an EXTERNAL use (non-web-edge). Internal
  // edges (a member feeding another web PHI) vanish when the PHIs are erased. ---
  auto isInternalEdge = [&](MachineInstr &U) {
    return U.isPHI() && PhiMembers.count(U.getOperand(0).getReg());
  };
  SmallSetVector<Register, 32> MembersWithExtUse;
  for (Register M : PhiMembers)
    for (MachineInstr &U : MRI->reg_nodbg_instructions(M))
      if (U.readsVirtualRegister(M) && !isInternalEdge(U))
        MembersWithExtUse.insert(M);

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
    LIS->InsertMachineInstrInMaps(*std::prev(DIt));
    ++NumSpills;
    if (Reporter && SSAForensicReporter::enabled())
      Reporter->spillEmitted(G.virtRegIndex(), "phi-web-store-at-def");
    LLVM_DEBUG(dbgs() << "  phi-web: store " << printReg(G, TRI)
                      << (GE.SubReg ? ".subN" : "") << " AT DEF -> FI" << FI
                      << " in " << printMBBReference(GBB) << "\n");
  }

  // Reload each PHI member's EXTERNAL uses from the shared slot and erase the PHIs.
  // The reload redefs are fresh vregs colored in place by the driver
  // (ColorFreshVReg via reloadedRegs()); the value delivery is memory on whichever
  // edge ran. This is the join-dissolution: the spilled web is FINAL (no register
  // coalescing remains — the coalescer's job on it is done by the spill).
  for (Register M : MembersWithExtUse) {
    MachineInstr *MDef = MRI->getUniqueVRegDef(M);
    if (!MDef)
      continue;
    VRegMaskPair VMP(M, MRI->getMaxLaneMaskForVReg(M));
    Virt2StackSlotMap[VMP] = FI;
    emitReloadsAndRepairSSA(VMP,
                            LIS->getInstructionIndex(*MDef).getRegSlot());
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
      LIS->RemoveMachineInstrFromMaps(*D);
      D->eraseFromParent();
    }
    if (LIS->hasInterval(M))
      LIS->removeInterval(M);
    LastWebErased.push_back(M);
  }

  for (Register G : GroundOps) {
    if (!G.isVirtual() || MRI->reg_nodbg_empty(G))
      continue;
    if (LIS->hasInterval(G))
      LIS->removeInterval(G);
    LIS->createAndComputeVirtRegInterval(G);
  }

  LLVM_DEBUG(dbgs() << "  phi-web: coalesced, erased " << LastWebErased.size()
                    << "/" << PhiMembers.size() << " PHIs, " << GroundOps.size()
                    << " ground stores -> FI" << FI << "\n");
}

bool SSASpillEmitter::narrowRemnantToNewReg(Register WideVReg, unsigned SubIdx,
                                            LaneBitmask RemnantMask) {
  SSARA_TRACE();
  MachineInstr *DefMI = MRI->getVRegDef(WideVReg);
  if (!DefMI)
    return false;
  const TargetRegisterClass *RC = MRI->getRegClass(WideVReg);
  const TargetRegisterClass *SubRC = TRI->getSubRegisterClass(RC, SubIdx);
  if (!SubRC)
    return false;

  // Capture the remnant into an INDEPENDENT narrow vreg right after the wide def:
  //   %new:SubRC = COPY WideVReg.SubIdx
  // %new is a distinct value that happens to equal WideVReg's remnant lanes at
  // this point. We then redirect the remnant-lane USES of WideVReg to %new. Once
  // its last remnant use is redirected, WideVReg is live only [def, this copy],
  // so its wide aligned tuple frees for the rest of the range.
  MachineBasicBlock &MBB = *DefMI->getParent();
  // Insert after the def. If the def is a PHI, all PHIs must stay contiguous at
  // the block top, so insert after the last PHI (getFirstNonPHI), not
  // immediately after this PHI (which would put a non-PHI COPY between PHIs ->
  // "PHI after non-PHI" verifier error).
  auto LegalAfter = AMDGPURegAllocInsertion::legalAfter(*DefMI);
  if (!LegalAfter)
    return false;
  MachineBasicBlock::iterator InsertPt = *LegalAfter;
  Register NewReg = MRI->createVirtualRegister(SubRC);
  MachineInstr *CopyMI = BuildMI(MBB, InsertPt, DefMI->getDebugLoc(),
                                 TII->get(TargetOpcode::COPY), NewReg)
                             .addReg(WideVReg, 0, SubIdx);
  LIS->InsertMachineInstrInMaps(*CopyMI);
  SlotIndex CopySlot = LIS->getInstructionIndex(*CopyMI).getRegSlot();

  // Redirect WideVReg's remnant-lane uses to %new — but ONLY those whose
  // REACHING value is WideVReg's ORIGINAL def (the value the COPY captured).
  // Dominance alone is unsound here (a use may be dominated yet its reaching VNI
  // be a PHI merge or another def); the codebase deliberately uses reaching-VNI
  // ownership, so mirror it: query WideVReg's LiveInterval for the VNInfo
  // reaching each use and only rewrite when it is the COPY's source VNI. Only
  // the remnant-mask lanes of the operand are eligible.
  const LiveInterval &LI = LIS->getInterval(WideVReg);
  VNInfo *SrcVNI = LI.getVNInfoBefore(CopySlot); // value the COPY read
  bool Changed = false;
  for (MachineOperand &MO :
       llvm::make_early_inc_range(MRI->use_operands(WideVReg))) {
    MachineInstr *UseMI = MO.getParent();
    if (UseMI == CopyMI || UseMI->isDebugInstr() || isSpillInstr(UseMI))
      continue;
    // Operand must read only lanes within the remnant (a use of a spilled lane
    // is served by a reload; a use spanning spilled+remnant is left alone —
    // conservative, its reload path handles it).
    LaneBitmask OpMask = MO.getSubReg()
                             ? TRI->getSubRegIndexLaneMask(MO.getSubReg())
                             : MRI->getMaxLaneMaskForVReg(WideVReg);
    if ((OpMask & ~RemnantMask).any())
      continue;
    // Reaching-VNI gate: this use must read the same value the COPY captured.
    SlotIndex UseSlot = LIS->getInstructionIndex(*UseMI).getRegSlot();
    VNInfo *AtUse = LI.getVNInfoBefore(UseSlot);
    if (!AtUse || AtUse != SrcVNI)
      continue;
    // The operand read exactly the remnant (SubIdx) lanes -> it maps to all of
    // %new (SubReg 0). (Guaranteed: OpMask ⊆ RemnantMask and we only narrow a
    // single contiguous remnant; a strict sub-slice would need a re-based
    // sub-index, left for the updater-API version.)
    if (OpMask != RemnantMask)
      continue;
    MO.setReg(NewReg);
    MO.setSubReg(AMDGPU::NoRegister);
    Changed = true;
  }

  if (!Changed) {
    // No use was redirected; the copy is dead. Remove it and report no-op.
    LIS->RemoveMachineInstrFromMaps(*CopyMI);
    CopyMI->eraseFromParent();
    if (LIS->hasInterval(NewReg))
      LIS->removeInterval(NewReg);
    return false;
  }

  LIS->createAndComputeVirtRegInterval(NewReg);
  if (LIS->hasInterval(WideVReg))
    LIS->removeInterval(WideVReg);
  LIS->createAndComputeVirtRegInterval(WideVReg);

  LLVM_DEBUG(dbgs() << "narrowRemnantToNewReg(): " << printReg(WideVReg, TRI)
                    << " remnant " << PrintLaneMask(RemnantMask) << " -> "
                    << printReg(NewReg, TRI) << " ("
                    << TRI->getRegClassName(SubRC) << ")\n");
  return true;
}

Register SSASpillEmitter::splitLiveRangeAt(Register V,
                                           MachineBasicBlock::iterator SplitPt) {
  SSARA_TRACE();
  if (!LIS->hasInterval(V))
    return Register();
  MachineBasicBlock &MBB = *SplitPt->getParent();
  const TargetRegisterClass *RC = MRI->getRegClass(V);
  SplitPt = AMDGPURegAllocInsertion::legalBefore(MBB, SplitPt);
  if (SplitPt == MBB.end())
    return Register();

  // Insert %new = COPY %v at the legal block-body position nearest SplitPt.
  Register NewReg = MRI->createVirtualRegister(RC);
  MachineInstr *CopyMI = BuildMI(MBB, SplitPt, SplitPt->getDebugLoc(),
                                 TII->get(TargetOpcode::COPY), NewReg)
                             .addReg(V);
  LIS->InsertMachineInstrInMaps(*CopyMI);
  SlotIndex CopySlot = LIS->getInstructionIndex(*CopyMI).getRegSlot();

  // Redirect uses of %v that (a) sit at-or-after the copy and (b) read the same
  // value the copy captured. Reaching-VNI ownership (not dominance) mirrors
  // narrowRemnantToNewReg: a use may be dominated yet reach a different VNI
  // (PHI merge / another def), so query %v's interval for the VNInfo reaching
  // each use and only rewrite when it is the copy's source VNI.
  const LiveInterval &LI = LIS->getInterval(V);
  VNInfo *SrcVNI = LI.getVNInfoBefore(CopySlot);
  bool Changed = false;
  for (MachineOperand &MO : llvm::make_early_inc_range(MRI->use_operands(V))) {
    MachineInstr *UseMI = MO.getParent();
    if (UseMI == CopyMI || UseMI->isDebugInstr() || isSpillInstr(UseMI))
      continue;
    SlotIndex UseSlot = LIS->getInstructionIndex(*UseMI).getRegSlot();
    // Redirect a use to %new only if the split COPY actually DOMINATES it. A
    // slot-index comparison (UseSlot < CopySlot) is NOT a dominance test: for a
    // multi-block value a use in a block the split point does not dominate can
    // still have a later slot index, and redirecting it to %new leaves %new used
    // where its def does not dominate ("defs don't dominate all uses").
    if (!DT->dominates(CopyMI, UseMI))
      continue; // not dominated by the split: keeps reading %v
    VNInfo *AtUse = LI.getVNInfoBefore(UseSlot);
    if (!AtUse || AtUse != SrcVNI)
      continue; // different reaching value: leave alone
    MO.setReg(NewReg); // sub-register index (if any) preserved unchanged
    Changed = true;
  }

  if (!Changed) {
    LIS->RemoveMachineInstrFromMaps(*CopyMI);
    CopyMI->eraseFromParent();
    if (LIS->hasInterval(NewReg))
      LIS->removeInterval(NewReg);
    return Register();
  }

  LIS->createAndComputeVirtRegInterval(NewReg);
  if (LIS->hasInterval(V))
    LIS->removeInterval(V);
  LIS->createAndComputeVirtRegInterval(V);

  LLVM_DEBUG(dbgs() << "splitLiveRangeAt(): " << printReg(V, TRI) << " @ "
                    << CopySlot << " -> " << printReg(NewReg, TRI) << "\n");
  return NewReg;
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

  // Narrow the reload to the lanes actually requested. The stack slot stays the
  // full SpilledVMP slot (the store side is untouched); we only reload the
  // sub-slice a use needs, from within that slot. A full-width request
  // (getAll()) reproduces the original whole-VMP reload.
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
  if (Slice != FullMask) {
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

  // When the reloaded slice starts above channel 0 of the full slot, point the
  // load at the right sub-slice. For VGPR/AV reloads the slot is memory and the
  // in-slot position is a byte offset (channel N is stored at byte N*4); set the
  // pseudo's immediate `offset` operand. For SGPR reloads (spill-to-VGPR-lane)
  // the narrowed dest subreg already selects the correct lanes in restoreSGPR,
  // so no offset is needed -- and there is no offset operand to set.
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

  SSAInvalidated =
      true; // redef of OrigVReg breaks SSA; inline repair restores it

  // NOTE: do NOT mark OrigVReg reloaded here -- that would subtract these lanes
  // from OrigVReg's active set globally and corrupt spill-candidate selection.
  // The reloaded value is tracked after inline repair renames it to a fresh
  // vreg (see emitReloadsAndRepairSSA).

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

void SSASpillEmitter::emitReloadsAndRepairSSA(VRegMaskPair SpilledVMP,
                                              SlotIndex KillIdx) {
  SSARA_TRACE();
  Register SpilledReg = SpilledVMP.getVReg();

  MachineInstr *KillMI = Indexes->getInstructionFromIndex(KillIdx);
  assert(KillMI && "KillIdx must correspond to an instruction");
  MachineBasicBlock *KillBB = KillMI->getParent();

  LLVM_DEBUG({
    dbgs() << "\n=== emitReloadsAndRepairSSA() [Option 3: redef-only] ===\n";
    dbgs() << "Spilled: " << printReg(SpilledReg, TRI) << " mask "
           << PrintLaneMask(SpilledVMP.getLaneMask()) << "\n";
  });

  // Collect ordinary uses reachable from the kill and PHI edge uses. Direct
  // placement intentionally emits one reload at every such use; a post-color
  // coalescer may merge redundant reloads after physical assignments are known.
  SmallVector<MachineInstr *, 8> Uses;
  SmallVector<MachineInstr *, 4> PhiUses;
  for (MachineInstr &UseMI : MRI->use_nodbg_instructions(SpilledReg)) {
    if (isSpillInstr(&UseMI))
      continue;
    MachineOperand *UseOp =
        UseMI.findRegisterUseOperand(SpilledReg, TRI, /*isKill=*/false);
    if (!UseOp || !VRegMaskPair(*UseOp, TRI, MRI).overlaps(SpilledVMP))
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
  // Freeze the reaching oracle ONCE, here: the interval was just recomputed (line
  // above) and already contains every reload redef, and nothing renames it until
  // the repair loop below. All repairSSAForNewDef calls share this one snapshot,
  // so no per-call re-freeze (resetSession) is needed -- the emitter no longer
  // disturbs the interval between repair calls.
  FrozenInterval Oracle = SSAUpdater->freezeInterval(SpilledReg);
  bool InsertedPHI = false;
  for (MachineInstr *RMI : ReloadDefs) {
    SmallVector<MachineOperand *> PHIDefs;
    SSAUpdater->repairSSAForNewDef(*RMI, SpilledReg, PHIDefs, &Oracle);
    if (!PHIDefs.empty())
      InsertedPHI = true;
    // Track the reloaded value -- now a renamed fresh vreg -- so the forward
    // walk does not immediately re-spill it. (Tracking OrigVReg would corrupt
    // its active-lane accounting; see getOrCreateReloadInBlock.)
    Register ReloadReg = RMI->getOperand(0).getReg();
    if (ReloadReg.isVirtual() && ReloadReg != SpilledReg)
      ReloadedRegs.insert(
          VRegMaskPair(ReloadReg, MRI->getMaxLaneMaskForVReg(ReloadReg)));
  }
  // Every reload redef has been renamed, so SpilledReg's subranges are final.
  // Rebuild its main range as their union: repair left it spanning the defs that
  // moved to the reload vregs, which would make SpilledReg claim registers it no
  // longer occupies once coloring consults it.
  SSAUpdater->rebuildMainRangeFromSubranges(SpilledReg);
  // SSA is restored inline; do not clear the IsSSA property at pass end.
  SSAInvalidated = false;

  // Only clear NoPHIs if reconstruction actually inserted a merge PHI. Clearing
  // it otherwise wrongly enables verifier checks (e.g. the physreg-live-in
  // check) that assume the function may contain PHIs. (Cf. X86CmovConversion.)
  if (InsertedPHI)
    KillBB->getParent()->getProperties().reset(
        MachineFunctionProperties::Property::NoPHIs);

  LLVM_DEBUG(dbgs() << "\nemitReloadsAndRepairSSA() complete\n");
}

MachineBasicBlock *
SSASpillEmitter::getEffectiveKillBB(MachineBasicBlock *SpillBB) const {
  SSARA_TRACE();
  // Find outermost loop containing spill point
  MachineLoop *Loop = MLI->getLoopFor(SpillBB);
  if (!Loop)
    return SpillBB; // Not in any loop

  // Walk up to outermost loop
  while (MachineLoop *Parent = Loop->getParentLoop())
    Loop = Parent;

  // Get outermost loop's preheader
  MachineBasicBlock *Preheader = Loop->getLoopPreheader();
  if (Preheader) {
    LLVM_DEBUG(dbgs() << "  Hoisting spill point from "
                      << printMBBReference(*SpillBB) << " to preheader "
                      << printMBBReference(*Preheader) << "\n");
    return Preheader;
  }

  // Irreducible loop - can't hoist
  LLVM_DEBUG(
      dbgs() << "  Warning: No preheader for loop containing spill point\n");
  return SpillBB;
}

