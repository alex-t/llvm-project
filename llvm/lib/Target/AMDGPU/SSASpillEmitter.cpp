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
/// The bodies here were factored verbatim out of AMDGPUSSARegisterSpiller; the
/// only rewrites are (1) VGPRLimit/SGPRLimit reads become the per-spill
/// CurRPLimit threaded through spillOneVMP, and (2) usesSpilledVMP is a shared
/// free function taking TRI/MRI rather than a member.
///
//===----------------------------------------------------------------------===//

#include "SSASpillEmitter.h"
#include "AMDGPU.h"
#include "GCNRegPressure.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIRegisterInfo.h"
#include "VRegMaskPair.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/Register.h"
#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-ssa-register-spiller"

STATISTIC(NumSpills, "Number of register spills");
STATISTIC(NumReloads, "Number of register reloads");

bool llvm::usesSpilledVMP(const MachineInstr *MI, VRegMaskPair SpilledVMP,
                          const SIRegisterInfo *TRI,
                          const MachineRegisterInfo *MRI) {
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
  TRI = static_cast<const SIRegisterInfo *>(MF.getSubtarget().getRegisterInfo());
  TII = static_cast<const SIInstrInfo *>(MF.getSubtarget().getInstrInfo());
  MRI = &MF.getRegInfo();
  FrameInfo = &MF.getFrameInfo();
}

void SSASpillEmitter::beginPass(bool IsVGPR) {
  IsVGPRPass = IsVGPR;
  // Fresh SSA updater per pass (caches IDF computations per run).
  SSAUpdater = std::make_unique<MachineLaneSSAUpdater>(MF, *LIS, *DT, *TRI);
}

int SSASpillEmitter::assignVirt2StackSlot(VRegMaskPair VMP) {
  assert(VMP.getVReg().isVirtual() && "Expected virtual register");

  // Check if we already have a stack slot for this VRegMaskPair
  auto It = Virt2StackSlotMap.find(VMP);
  if (It != Virt2StackSlotMap.end())
    return It->second;

  // Create a new stack slot
  const TargetRegisterClass *RC = VMP.getRegClass(MRI, TRI);
  int FI = createSpillSlot(RC);
  Virt2StackSlotMap[VMP] = FI;
  return FI;
}

int SSASpillEmitter::createSpillSlot(const TargetRegisterClass *RC) {
  unsigned SpillSize = TRI->getSpillSize(*RC);
  Align SpillAlign = TRI->getSpillAlign(*RC);
  return FrameInfo->CreateSpillStackObject(SpillSize, SpillAlign);
}

void SSASpillEmitter::spillOneVMP(VRegMaskPair VMP, SlotIndex KillIdx,
                                  unsigned RPLimit) {
  // Reload-hoist decisions in the reload path use this file's RP budget, chosen
  // by policy and threaded in per spill.
  CurRPLimit = RPLimit;

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
  assert(DefStoreMI && "Virtual register must have a definition in SSA form");
  (void)DefStoreMI;

  // Step 2c: Get stack slot for reload phase
  int FI = assignVirt2StackSlot(VMP);

  // Step 2d: Build SpillInfo with dom-groups and emit reloads. Reloads are
  // placed at uses reachable from KillIdx (dominance-ordered), so uses above
  // KillIdx keep the original register and are not reloaded.
  SpillInfo Info;
  Info.SpilledVMP = VMP;
  Info.KillIdx = KillIdx;
  Info.FrameIndex = FI;
  buildDomGroupsForSpill(Info);
  emitReloadsAndRepairSSA(Info);
}

bool SSASpillEmitter::narrowRemnantToNewReg(Register WideVReg, unsigned SubIdx,
                                            LaneBitmask RemnantMask) {
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
  MachineBasicBlock::iterator InsertPt =
      DefMI->isPHI() ? MBB.getFirstNonPHI() : std::next(DefMI->getIterator());
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

MachineInstr *SSASpillEmitter::spillAtDefinition(VRegMaskPair VMP) {
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

  MachineBasicBlock *DefMBB = DefMI->getParent();
  // Store right after the def. When the def is a PHI, all PHIs must stay
  // contiguous at the block top, so std::next(PHI) could land the store between
  // PHIs ("PHI after non-PHI"). Insert after the last PHI instead.
  MachineBasicBlock::iterator InsertAfter =
      DefMI->isPHI() ? DefMBB->getFirstNonPHI()
                     : std::next(DefMI->getIterator());

  // Get or create stack slot
  int FI = assignVirt2StackSlot(VMP);

  // Determine SubRegIdx from lane mask
  unsigned SubRegIdx = VMP.getSubReg(MRI, TRI);

  // Get the appropriate register class
  const TargetRegisterClass *RC =
      (SubRegIdx == AMDGPU::NoRegister)
          ? TRI->getRegClassForReg(*MRI, VReg)
          : TRI->getSubRegisterClass(TRI->getRegClassForReg(*MRI, VReg),
                                     SubRegIdx);

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

  return &StoreMI;
}

// ===========================================================================
// IDF-First PHI Insertion Strategy
// ===========================================================================

void SSASpillEmitter::buildDomGroupsForSpill(SpillInfo &Info) {
  Register SpilledReg = Info.SpilledVMP.getVReg();
  MachineInstr *KillMI = Indexes->getInstructionFromIndex(Info.KillIdx);

  LLVM_DEBUG(dbgs() << "buildDomGroupsForSpill for "
                    << printReg(SpilledReg, TRI) << "\n");

  // Collect all uses into a vector for sorting. PHI uses are collected
  // SEPARATELY (Info.PhiUses): a PHI reads its operand on the PREDECESSOR edge,
  // not at the PHI's own slot, so it cannot take part in the dominance-merge
  // grouping below (which reasons about the use instruction's position) and its
  // reload belongs in the predecessor. emitReloadsAndRepairSSA routes PHI uses
  // straight to insertReloadForUse, which already places a per-predecessor
  // reload. Skipping them entirely (the old behavior) left a PHI-only-used value
  // with a store but NO reload, so its live range never shrank and it was
  // re-spilled forever.
  SmallVector<MachineInstr *, 8> AllUses;
  for (MachineInstr &UseMI : MRI->use_nodbg_instructions(SpilledReg)) {
    if (isSpillInstr(&UseMI))
      continue;

    MachineOperand *UseOp =
        UseMI.findRegisterUseOperand(SpilledReg, TRI, /*isKill=*/false);
    if (!UseOp)
      continue;

    VRegMaskPair UseVMP(*UseOp, TRI, MRI);
    if (!UseVMP.overlaps(Info.SpilledVMP))
      continue;

    if (UseMI.isPHI()) {
      // Reachability for a PHI is per-edge (each operand's incoming block);
      // insertReloadForUse re-checks each operand, so collect unconditionally.
      Info.PhiUses.push_back(&UseMI);
      continue;
    }

    // Only consider uses reachable from KillMI
    if (!DT->dominates(KillMI, &UseMI) &&
        !SSAUpdater->isUseReachableFromDef(KillMI, &UseMI, SpilledReg))
      continue;

    AllUses.push_back(&UseMI);
  }

  // Sort uses by dominance order (dominating first)
  llvm::sort(AllUses, [this](MachineInstr *A, MachineInstr *B) {
    if (DT->dominates(A, B))
      return true;
    if (DT->dominates(B, A))
      return false;
    // For unrelated uses, use slot index as tiebreaker
    return Indexes->getInstructionIndex(*A) < Indexes->getInstructionIndex(*B);
  });

  // Build groups: for each use, either merge into existing group or create new
  for (MachineInstr *UseMI : AllUses) {
    bool Merged = false;
    for (DomGroup &G : Info.DomGroups) {
      if (DT->dominates(G.getHead(), UseMI)) {
        G.addDominatedUse(UseMI);
        Merged = true;
        break;
      }
      if (DT->dominates(UseMI, G.getHead())) {
        G.promoteHead(UseMI);
        Merged = true;
        break;
      }
    }
    if (!Merged) {
      Info.DomGroups.emplace_back(UseMI);
    }
  }

  LLVM_DEBUG(dbgs() << "  Built " << Info.DomGroups.size() << " dom-groups\n");
}

std::pair<Register, MachineInstr *>
SSASpillEmitter::getOrCreateReloadInBlock(MachineBasicBlock *BB,
                                          VRegMaskPair SpilledVMP,
                                          MachineInstr *InsertBefore,
                                          LaneBitmask ReloadMask) {
  Register OrigVReg = SpilledVMP.getVReg();

  // Narrow the reload to the lanes actually requested. The stack slot stays the
  // full SpilledVMP slot (the store side is untouched); we only reload the
  // sub-slice a use needs, from within that slot. A full-width request
  // (getAll()) reproduces the original whole-VMP reload.
  LaneBitmask Slice = ReloadMask & SpilledVMP.getLaneMask();
  if (Slice.none())
    Slice = SpilledVMP.getLaneMask();
  VRegMaskPair ReloadVMP(OrigVReg, Slice);

  auto Key = std::make_pair(BB, ReloadVMP);

  // Only use cache for block-end reloads (InsertBefore == nullptr)
  if (!InsertBefore) {
    auto It = BlockReloadCache.find(Key);
    if (It != BlockReloadCache.end()) {
      LLVM_DEBUG(dbgs() << "    Reusing cached reload in "
                        << printMBBReference(*BB) << ": "
                        << printReg(It->second, TRI) << "\n");
      return {It->second, nullptr}; // Cached - no new instruction
    }
  }

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

  // Determine insertion point: before specified instruction or at block end
  auto InsertIt =
      InsertBefore ? InsertBefore->getIterator() : BB->getFirstTerminator();
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

  // Cache only block-end reloads
  if (!InsertBefore)
    BlockReloadCache[Key] = OrigVReg;

  LLVM_DEBUG(dbgs() << "    Created reload (redef) in "
                    << printMBBReference(*BB)
                    << (InsertBefore ? " before use" : " at block end") << ": "
                    << printReg(OrigVReg, TRI) << "\n");
  ++NumReloads;
  return {OrigVReg, ReloadMI};
}

bool SSASpillEmitter::insertReloadForUse(MachineInstr *UseMI,
                                         VRegMaskPair SpilledVMP,
                                         MachineBasicBlock *KillBB) {
  Register SpilledReg = SpilledVMP.getVReg();
  LaneBitmask SpilledMask = SpilledVMP.getLaneMask();
  unsigned RPLimit = CurRPLimit;

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

      // Check RP in predecessor
      unsigned PredRP = getMaxRPForBlock(PredBB);
      if (PredRP > RPLimit) {
        LLVM_DEBUG(dbgs() << "    WARNING: Predecessor "
                          << printMBBReference(*PredBB) << " has RP=" << PredRP
                          << " > limit=" << RPLimit
                          << ", but must insert reload for PHI use\n");
      }

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

  // Non-PHI use: insert before use with loop adjustment
  auto Adjusted =
      adjustReloadForLoop(UseMI->getParent(), UseMI, KillBB, SpilledReg);
  MachineInstr *InsertBeforeUse =
      (Adjusted.first == UseMI->getParent()) ? UseMI : nullptr;
  // Place the reload redef; SSA is repaired inline after all reloads.
  getOrCreateReloadInBlock(Adjusted.first, SpilledVMP, InsertBeforeUse, UseMask);
  return true;
}

void SSASpillEmitter::emitReloadsAndRepairSSA(SpillInfo &Info) {
  VRegMaskPair SpilledVMP = Info.SpilledVMP;
  Register SpilledReg = SpilledVMP.getVReg();

  MaxRPCache.clear();
  BlockReloadCache.clear();

  MachineInstr *KillMI = Indexes->getInstructionFromIndex(Info.KillIdx);
  assert(KillMI && "KillIdx must correspond to an instruction");
  MachineBasicBlock *KillBB = KillMI->getParent();

  LLVM_DEBUG({
    dbgs() << "\n=== emitReloadsAndRepairSSA() [Option 3: redef-only] ===\n";
    dbgs() << "Spilled: " << printReg(SpilledReg, TRI) << " mask "
           << PrintLaneMask(SpilledVMP.getLaneMask()) << "\n";
    dbgs() << "DomGroups: " << Info.DomGroups.size() << "\n";
  });

  // Dominance-ordered reload-on-demand (see Reload_join_phi_coalescing.md).
  // Conceptually we cut OrigVReg's live range at the kill: a use in the freed
  // region then reaches no original value and needs a reload, while a use
  // outside it still reaches the original. We realize the cut without surgery
  // -- reloads are redefs, so once the frontier reloads are placed the
  // recomputed interval already merges them as isPHIDef VNInfos and keeps the
  // original on non-kill paths; the existing reconstruction turns those into
  // PHIs/reuses. Here we only pick the frontier: a freed-region use whose
  // spilled lanes still reach the ORIGINAL def (not a reload and not an
  // isPHIDef merge) gets a reload; everything else is left to reconstruction.
  // No reload optimizer: processing dominators first makes a dominating reload
  // visible to dominated uses (query sees it), so intra-chain sharing is
  // automatic.
  SmallVector<MachineInstr *, 8> Uses;
  for (DomGroup &G : Info.DomGroups) {
    Uses.push_back(G.getHead());
    for (MachineInstr *U : G.getDominatedUses())
      Uses.push_back(U);
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

  const LaneBitmask SpillMask = SpilledVMP.getLaneMask();
  const SlotIndex KillSlot = Info.KillIdx.getRegSlot();

  // Decide whether use U needs a reload. Atomic-process invariant: we must
  // NEVER prune the live LIS interval -- the RPTracker (canHoistReloadTo /
  // adjustReloadForLoop) reads it, and removing OrigVReg's liveness there would
  // corrupt pressure and the hoist decision. Instead we recompute the live
  // interval (RP-safe: a reload is a redef of OrigVReg with the same one-reg
  // footprint, and per the reload-analysis invariant OrigVReg stays counted as
  // live), DEEP-COPY it, and CUT the COPY at the kill. On the copy the original
  // is pruned from the kill onward (surviving only on kill-free paths) while
  // reload values are untouched; the live LIS the RPTracker reads is intact.
  auto NeedsReload = [&](MachineInstr *U) -> bool {
    if (LIS->hasInterval(SpilledReg))
      LIS->removeInterval(SpilledReg);
    LiveInterval &Live = LIS->createAndComputeVirtRegInterval(SpilledReg);

    // Deep copy (allocator declared first so the copy destructs before it).
    VNInfo::Allocator CutAlloc;
    LiveInterval Cut(SpilledReg, 0.0f);
    Cut.assign(Live, CutAlloc);
    for (const LiveInterval::SubRange &S : Live.subranges())
      Cut.createSubRangeFrom(CutAlloc, S.LaneMask, S);

    // Cut the COPY at the kill (never the live interval).
    SmallVector<SlotIndex, 8> Ends;
    if (Cut.hasSubRanges()) {
      for (LiveInterval::SubRange &S : Cut.subranges())
        if ((S.LaneMask & SpillMask).any() && S.getVNInfoAt(KillSlot))
          LIS->pruneValue(S, KillSlot, &Ends);
    } else if (Cut.getVNInfoAt(KillSlot)) {
      LIS->pruneValue(static_cast<LiveRange &>(Cut), KillSlot, &Ends);
    }

    // Per-edge availability on the cut copy: reload iff some spilled lane is
    // not available on every incoming path. No value reaches the use, or the
    // reaching value is live-in but a predecessor edge carries no value (a
    // freed edge) -> reload. A value defined in U's own block (a local reload)
    // dominates U and covers it. A live-in value on ALL predecessors is a
    // genuine merge -> reconstruction inserts a PHI, no reload here.
    MachineBasicBlock *B = U->getParent();
    SlotIndex UIdx = LIS->getInstructionIndex(*U).getRegSlot();
    auto LaneNeedsReload = [&](LiveRange &LR) -> bool {
      VNInfo *AtUse = LR.getVNInfoBefore(UIdx);
      if (!AtUse)
        return true;
      if (MachineInstr *DMI = LIS->getInstructionFromIndex(AtUse->def))
        if (DMI->getParent() == B) {
          // A same-block reaching def normally covers U with no new reload. But
          // if the reaching value's live range [DMI, U] SPANS an RP-tight region
          // (max RP between them exceeds the limit), reusing it keeps a register
          // pinned across that region — the exact C1 pathology (a reload live
          // across a high-pressure INLINEASM etc. that never lowers RP there and
          // makes coloring fail). Force a fresh reload right before U instead, so
          // the reaching value's range ends before the tight point.
          if (CurRPLimit && maxRPBetween(DMI, U) > CurRPLimit)
            return true;
          return false;
        }
      for (MachineBasicBlock *P : B->predecessors())
        if (!LR.getVNInfoBefore(LIS->getMBBEndIdx(P)))
          return true;
      return false;
    };
    if (Cut.hasSubRanges()) {
      for (LiveInterval::SubRange &S : Cut.subranges())
        if ((S.LaneMask & SpillMask).any() && LaneNeedsReload(S))
          return true;
      return false;
    }
    return LaneNeedsReload(Cut);
  };

  // Dominators-first: a reload placed for a dominator/sibling is visible to
  // later uses, so intra-chain sharing and join PHIs fall out with no reload
  // optimizer.
  for (MachineInstr *U : Uses) {
    if (!usesSpilledVMP(U, SpilledVMP, TRI, MRI))
      continue;
    if (NeedsReload(U))
      insertReloadForUse(U, SpilledVMP, KillBB);
  }

  // PHI uses: reload on the predecessor edge(s). These bypass the NeedsReload
  // gate above — that gate reasons about the use instruction's own block/slot,
  // which is meaningless for a PHI (its operand lives on the incoming edge, not
  // at the PHI). insertReloadForUse's PHI branch places a reload in each
  // predecessor that supplies the spilled reg, which is exactly the cut that
  // frees the value across the def→PHI span. Without this a PHI-only-used value
  // keeps the original live to the PHI and never sheds pressure.
  for (MachineInstr *U : Info.PhiUses) {
    if (!usesSpilledVMP(U, SpilledVMP, TRI, MRI))
      continue;
    insertReloadForUse(U, SpilledVMP, KillBB);
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
  // This spillAndReload added new reload redefs of SpilledReg. Force a fresh
  // reaching-oracle freeze so repair sees them; the updater otherwise caches
  // the frozen interval per OrigVReg across our incremental spills, which would
  // make reaching resolution miss these redefs (leaving dead reloads).
  SSAUpdater->resetSession();
  bool InsertedPHI = false;
  for (MachineInstr *RMI : ReloadDefs) {
    SmallVector<MachineOperand *> PHIDefs;
    SSAUpdater->repairSSAForNewDef(*RMI, SpilledReg, PHIDefs);
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

// ============================================================================
// Reload Optimizer
// ============================================================================

// TODO: Investigate profitability/possibility of early return when RP > Limit.
// Callers only care whether RP exceeds the limit, not by how much.
// Optimization: if we find RP > Limit at any point, return early and cache
// that value - no need to compute the actual maximum.
unsigned SSASpillEmitter::getMaxRPForBlock(MachineBasicBlock *MBB) {
  auto It = MaxRPCache.find(MBB);
  if (It != MaxRPCache.end())
    return It->second;

  // Compute max RP by tracking backwards through the block
  GCNUpwardRPTracker Tracker(*LIS);
  Tracker.reset(*MBB);

  const GCNSubtarget &ST = MBB->getParent()->getSubtarget<GCNSubtarget>();

  // Include initial pressure (live-out at block end)
  GCNRegPressure InitPressure = Tracker.getPressure();
  unsigned MaxRP = IsVGPRPass ? InitPressure.getVGPRNum(ST.hasGFX90AInsts())
                              : InitPressure.getSGPRNum();

  for (MachineInstr &MI : reverse(*MBB)) {
    if (MI.isDebugInstr())
      continue;
    Tracker.recede(MI);
    GCNRegPressure Pressure = Tracker.getPressure();
    unsigned CurRP = IsVGPRPass ? Pressure.getVGPRNum(ST.hasGFX90AInsts())
                                : Pressure.getSGPRNum();
    MaxRP = std::max(MaxRP, CurRP);
  }

  MaxRPCache[MBB] = MaxRP;
  return MaxRP;
}

unsigned SSASpillEmitter::maxRPBetween(MachineInstr *DefMI,
                                       MachineInstr *UseMI) {
  // Max RP (current pass's file) at the program points strictly between DefMI
  // and UseMI in the same block, inclusive of the span the reaching value would
  // occupy. Used to decide whether a same-block reaching reload SPANS an
  // RP-tight region: if so, the shared reload must not be reused across it — a
  // fresh reload is forced right before UseMI so the span ends before the tight
  // point (C1: a reload live across a pressure region does not lower RP there).
  MachineBasicBlock *MBB = UseMI->getParent();
  if (!DefMI || DefMI->getParent() != MBB)
    return 0;
  const GCNSubtarget &ST = MBB->getParent()->getSubtarget<GCNSubtarget>();
  GCNUpwardRPTracker Tracker(*LIS);
  Tracker.reset(*UseMI);
  unsigned MaxRP = 0;
  for (auto It = UseMI->getReverseIterator(); It != MBB->rend(); ++It) {
    MachineInstr &MI = *It;
    if (MI.isDebugInstr())
      continue;
    Tracker.recede(MI);
    GCNRegPressure P = Tracker.getPressure();
    unsigned RP = IsVGPRPass ? P.getVGPRNum(ST.hasGFX90AInsts()) : P.getSGPRNum();
    MaxRP = std::max(MaxRP, RP);
    if (&MI == DefMI)
      break; // reached the reaching def; span is [DefMI, UseMI]
  }
  return MaxRP;
}

unsigned SSASpillEmitter::getMaxRPInBlockDownTo(MachineBasicBlock *MBB,
                                                MachineInstr *StopMI) {
  if (!StopMI || StopMI->getParent() != MBB)
    return getMaxRPForBlock(MBB);

  // Compute max RP from block start up to (not including) StopMI
  GCNUpwardRPTracker Tracker(*LIS);

  // Start from StopMI and track backwards to block start
  Tracker.reset(*StopMI);

  unsigned MaxRP = 0;
  const GCNSubtarget &ST = MBB->getParent()->getSubtarget<GCNSubtarget>();

  for (auto It = StopMI->getReverseIterator(); It != MBB->rend(); ++It) {
    MachineInstr &MI = *It;
    if (MI.isDebugInstr())
      continue;
    Tracker.recede(MI);
    GCNRegPressure Pressure = Tracker.getPressure();
    unsigned CurRP = IsVGPRPass ? Pressure.getVGPRNum(ST.hasGFX90AInsts())
                                : Pressure.getSGPRNum();
    MaxRP = std::max(MaxRP, CurRP);
  }

  return MaxRP;
}

bool SSASpillEmitter::canHoistReloadTo(MachineBasicBlock *NCD,
                                       MachineInstr *InsertPoint,
                                       unsigned RPLimit, Register SpilledReg) {
  // Spilled register is already counted as live, so MaxRP > RPLimit means
  // no room for reload (no +1 needed).

  // Check RP in NCD block only if reload is placed inside NCD (InsertPoint set)
  // If InsertPoint is nullptr, reload goes at NCD end - skip NCD RP check,
  // walkPathsToUses will check paths from NCD to uses.
  if (InsertPoint) {
    unsigned NCDRP = getMaxRPInBlockDownTo(NCD, InsertPoint);
    if (NCDRP > RPLimit)
      return false;
  }

  auto IsHighRP = [&](MachineBasicBlock *BB, MachineInstr *UseMI) -> bool {
    unsigned CurRP =
        UseMI ? getMaxRPInBlockDownTo(BB, UseMI) : getMaxRPForBlock(BB);
    return CurRP > RPLimit;
  };

  return walkPathsToUses(NCD, SpilledReg, IsHighRP);
}

// ============================================================================
// Loop-Aware Spilling Helpers
// ============================================================================

MachineBasicBlock *
SSASpillEmitter::getEffectiveKillBB(MachineBasicBlock *SpillBB) const {
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

std::pair<MachineBasicBlock *, MachineInstr *>
SSASpillEmitter::adjustReloadForLoop(MachineBasicBlock *ReloadBB,
                                     MachineInstr *InsertBeforeMI,
                                     MachineBasicBlock *KillBB,
                                     Register SpilledReg) {
  MachineLoop *ReloadLoop = MLI->getLoopFor(ReloadBB);
  if (ReloadLoop && !ReloadLoop->contains(KillBB)) {
    // Do NOT hoist the reload to the preheader if the loop body contains a call
    // that clobbers this value's file: hoisting makes the value live across the
    // whole loop, hence across that in-loop call — but a value live across a
    // call may occupy only registers the call preserves. A cross-call value
    // hoisted here would be un-colorable (Failed to find free physreg). Keep the
    // reload inside the loop (at the use, after the call), accepting a per-
    // iteration reload — correctness outranks the reload-count optimization.
    bool LoopHasClobberingCall = false;
    for (MachineBasicBlock *LB : ReloadLoop->blocks()) {
      for (MachineInstr &LMI : *LB) {
        if (!LMI.isCall())
          continue;
        for (const MachineOperand &MO : LMI.operands())
          if (MO.isRegMask() &&
              MO.clobbersPhysReg(
                  (IsVGPRPass ? AMDGPU::VGPR0 : AMDGPU::SGPR0))) {
            LoopHasClobberingCall = true;
            break;
          }
        if (LoopHasClobberingCall)
          break;
      }
      if (LoopHasClobberingCall)
        break;
    }
    if (LoopHasClobberingCall) {
      LLVM_DEBUG(dbgs() << "  Not hoisting reload: loop contains a call that "
                           "clobbers this file; keeping reload inside loop\n");
      return {ReloadBB, InsertBeforeMI};
    }

    // Use in loop, spill outside - consider hoisting reload to preheader
    MachineBasicBlock *Preheader = ReloadLoop->getLoopPreheader();
    if (Preheader) {
      unsigned RPLimit = CurRPLimit;
      MachineInstr *InsertPoint = nullptr;
      auto TermIt = Preheader->getFirstTerminator();
      if (TermIt != Preheader->end())
        InsertPoint = &*TermIt;

      bool CanHoist =
          canHoistReloadTo(Preheader, InsertPoint, RPLimit, SpilledReg);

      if (!CanHoist) {
        LLVM_DEBUG(
            dbgs() << "  Cannot hoist reload to preheader: "
                   << "RP exceeds limit on path, keeping reload inside loop\n");
        return {ReloadBB,
                InsertBeforeMI}; // Don't hoist - accept reload in loop
      }

      LLVM_DEBUG(dbgs() << "  Hoisting reload from "
                        << printMBBReference(*ReloadBB) << " to preheader "
                        << printMBBReference(*Preheader) << "\n");
      return {Preheader, nullptr}; // Insert at end of preheader
    }
  }
  return {ReloadBB, InsertBeforeMI};
}

// ============================================================================
// Divergent Path Optimization Helpers
// ============================================================================

bool SSASpillEmitter::walkPathsToUses(
    MachineBasicBlock *StartBB, Register SpilledReg,
    llvm::function_ref<bool(MachineBasicBlock *, MachineInstr *)> IsBad,
    bool StopOnBad) const {

  const LiveInterval &LI = LIS->getInterval(SpilledReg);

  SmallPtrSet<MachineBasicBlock *, 8> Visited;
  SmallVector<MachineBasicBlock *, 8> Worklist(StartBB->successors());
  bool FoundBad = false;

  while (!Worklist.empty()) {
    MachineBasicBlock *BB = Worklist.pop_back_val();
    if (!Visited.insert(BB).second)
      continue;

    // Skip blocks where spilled register is not live
    SlotIndex BBStart = Indexes->getMBBStartIdx(BB);
    if (!LI.liveAt(BBStart))
      continue;

    // Find first use of SpilledReg in this block (if any)
    MachineInstr *FirstUseMI = nullptr;
    for (MachineInstr &MI : *BB) {
      if (MI.readsRegister(SpilledReg, TRI)) {
        FirstUseMI = &MI;
        break;
      }
    }

    // Check predicate
    if (IsBad(BB, FirstUseMI)) {
      if (StopOnBad)
        return false;
      FoundBad = true;
    }

    // Continue to successors
    for (MachineBasicBlock *Succ : BB->successors())
      if (!Visited.count(Succ))
        Worklist.push_back(Succ);
  }

  return !FoundBad;
}
