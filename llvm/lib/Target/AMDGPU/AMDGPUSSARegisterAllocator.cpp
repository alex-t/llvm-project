//===-- AMDGPUSSARegisterAllocator.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUSSARegisterAllocator.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-ssa-register-allocator"

// Master switch for the around-call-liver (ACL) work: the allocator's two-phase
// (ACL-first) coloring walk AND the spiller's preserved-RP (callee-saved
// capacity) gate. This work is uncommitted/unproven and currently regresses the
// corpus (malformed MIR out of the two-phase walk on call functions). Default
// OFF so the tree matches the committed `ssara` baseline; flip on to develop it.
// The spiller reads this via a declaration in AMDGPUSSARegisterSpiller.cpp.
namespace llvm {
cl::opt<bool> EnableAMDGPUSSAACLColoring(
    "amdgpu-ssa-acl-coloring",
    cl::desc("Enable around-call-liver two-phase coloring and the spiller's "
             "preserved-register-pressure gate (default off; work in progress)"),
    cl::init(false), cl::Hidden);
} // namespace llvm

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

void AMDGPUSSARegisterAllocator::markOccupied(MCRegister PhysReg) {
  for (MCRegUnit Unit : TRI->regunits(PhysReg))
    OccupiedRegUnits.set(Unit);
}

void AMDGPUSSARegisterAllocator::markFree(MCRegister PhysReg) {
  for (MCRegUnit Unit : TRI->regunits(PhysReg))
    OccupiedRegUnits.reset(Unit);
}

void AMDGPUSSARegisterAllocator::dumpOccupancyMap(const TargetRegisterClass *RC,
                                                  SlotIndex SI, const char *Tag,
                                                  const LiveInterval *VI) const {
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

  std::string Map;
  SmallVector<MCRegister, 128> Order;
  unsigned FreeUsable = 0, FreeClobbered = 0, Occupied = 0;
  // Registers set in Occ (running state) but NOT in OccCM (ColorMap-live): the
  // "phantom" occupancy the map used to miss. Also collect the truly-usable regs.
  SmallVector<MCRegister, 8> Phantom, Usable;
  for (MCRegister PR : RegClassInfo.getOrder(RC)) {
    bool O = false, OCM = false;
    for (MCRegUnit U : TRI->regunits(PR)) {
      if (Occ.test(U)) O = true;
      if (OccCM.test(U)) OCM = true;
    }
    if (O) {
      Map.push_back('#');
      ++Occupied;
      if (!OCM)
        Phantom.push_back(PR); // occupied by running-state but no live ColorMap vreg
    } else if (Clobbered(PR)) {
      Map.push_back('x');
      ++FreeClobbered;
    } else {
      Map.push_back('.');
      ++FreeUsable;
      Usable.push_back(PR);
    }
    Order.push_back(PR);
  }
  dbgs() << "  [OCCMAP " << Tag << "] " << TRI->getRegClassName(RC) << " @" << SI
         << "  usable=" << FreeUsable << " clobbered=" << FreeClobbered
         << " occupied=" << Occupied << " total=" << Order.size() << "\n"
         << "    " << Map << "\n";
  if (!Order.empty())
    dbgs() << "    (" << TRI->getName(Order.front()) << " .. "
           << TRI->getName(Order.back()) << ")  legend: # occ, x clobbered, . usable\n";
  // The key question: registers occupied by running-state but with NO live
  // ColorMap vreg (physreg live-ins, dead defs, or stale bits).
  if (!Phantom.empty()) {
    dbgs() << "    phantom-occupied (Occ set, no live ColorMap vreg):";
    for (MCRegister PR : Phantom)
      dbgs() << " " << TRI->getName(PR);
    dbgs() << "\n";
  }
  if (!Usable.empty()) {
    dbgs() << "    usable regs:";
    for (MCRegister PR : Usable)
      dbgs() << " " << TRI->getName(PR);
    dbgs() << "\n";
    // For each usable reg, find WIDER ColorMap values whose whole interval
    // OVERLAPS VI (pickFreePhysReg's OccupiedAtDef augmentation, lines ~195).
    // This is the occupancy the liveAt(SI) map view misses.
    if (VI) {
      unsigned VIWidth = TRI->getRegSizeInBits(*RC);
      for (MCRegister PR : Usable) {
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

MCRegister AMDGPUSSARegisterAllocator::pickFreePhysReg(
    const TargetRegisterClass *RC, const LiveInterval &VI,
    ArrayRef<std::pair<MCRegister, const LiveInterval *>> WiderDefs,
    ArrayRef<MCRegister> Hints) {
  LLVM_DEBUG({
    dbgs() << "    Allocation order for " << TRI->getRegClassName(RC) << ":";
    for (MCRegister PR : RegClassInfo.getOrder(RC))
      dbgs() << " " << TRI->getName(PR);
    dbgs() << "\n";
  });

  // Augment OccupiedRegUnits with wider-width assignments that overlap VI.
  // Two sources: (1) WiderDefs — wider defs in THIS block not yet live at
  // BBStart (O(k), k = wider defs in block); (2) ColorMap scan for wider
  // cross-block entries (O(|ColorMap|), but only in narrower width passes
  // after wider passes committed their assignments).
  unsigned VIWidth = TRI->getRegSizeInBits(*RC);
  BitVector OccupiedAtDef = OccupiedRegUnits;
  for (const auto &[WPhysReg, WLI] : WiderDefs) {
    if (WLI->overlaps(VI)) {
      for (MCRegUnit Unit : TRI->regunits(WPhysReg))
        OccupiedAtDef.set(Unit);
    }
  }
  for (const auto &[WReg, WPhysReg] : ColorMap) {
    if (TRI->getRegSizeInBits(*MRI->getRegClass(WReg)) <= VIWidth)
      continue;
    if (LIS->getInterval(WReg).overlaps(VI)) {
      for (MCRegUnit Unit : TRI->regunits(WPhysReg))
        OccupiedAtDef.set(Unit);
    }
  }

  // Shared legality test: a candidate PR is usable iff none of its reg units are
  // occupied at this def AND it is not clobbered by any call VI is live across.
  // A value live across a call cannot occupy a register the call clobbers
  // (regmask-clobbered caller-saved regs, or an explicit def such as the
  // return-address $sgpr30_sgpr31) - it would be undefined after the call.
  auto IsFree = [&](MCRegister PR) -> bool {
    for (MCRegUnit Unit : TRI->regunits(PR))
      if (OccupiedAtDef.test(Unit))
        return false;
    for (const auto &[CallIdx, CallMI] : CallSites) {
      if (!VI.liveAt(CallIdx))
        continue;
      if (CallMI->modifiesRegister(PR, TRI))
        return false;
      for (const MachineOperand &MO : CallMI->operands())
        if (MO.isRegMask() && MO.clobbersPhysReg(PR))
          return false;
    }
    return true;
  };

  // Option B: prefer a phi-partner's color if it is a legal member of RC and
  // free. Hints are pre-ordered hottest-first by collectPhiHints; take the first
  // that fits. RC->contains guards against a partner whose class differs from RC.
  for (MCRegister Hint : Hints) {
    if (!Hint || !RC->contains(Hint))
      continue;
    if (IsFree(Hint)) {
      LLVM_DEBUG(dbgs() << "    phi-affinity hint taken: " << TRI->getName(Hint)
                        << "\n");
      return Hint;
    }
  }

  for (MCRegister PR : RegClassInfo.getOrder(RC))
    if (IsFree(PR))
      return PR;
  return MCRegister();
}

// Option B affinity hint collection. See header comment.
SmallVector<MCRegister, 4>
AMDGPUSSARegisterAllocator::collectPhiHints(Register VReg,
                                            const TargetRegisterClass *RC) {
  // (physreg, weight) candidates; dedup + weight-sort before returning.
  SmallVector<std::pair<MCRegister, uint64_t>, 4> Cand;

  // Turn a colored φ partner into a candidate color for VReg. SubIdx is the
  // sub-register index relating the two values; PartnerIsSub says which side it
  // slices:
  //   - PartnerIsSub == false (Direction A): VReg is the sub-register, reading
  //     Partner.SubIdx (a lane φ reading %593.sub3 of a wide colored operand).
  //     VReg's color is that SLICE of Partner's color -> getSubReg().
  //   - PartnerIsSub == true  (Direction B): Partner is the sub-register; the φ
  //     reads VReg.SubIdx into the narrow result Partner (a loop-carried tuple
  //     whose header result is colored before the wide latch operand VReg).
  //     VReg's color is the SUPER-register whose SubIdx slice is Partner's
  //     color -> getMatchingSuperReg().
  // Either composition must land in RC (VReg's class) to be a legal hint.
  auto AddPartner = [&](Register Partner, unsigned SubIdx, bool PartnerIsSub,
                        MachineBasicBlock *EdgeBlock) {
    if (!Partner.isVirtual())
      return;
    auto It = ColorMap.find(Partner);
    if (It == ColorMap.end())
      return; // partner not colored yet -- nothing to align to
    MCRegister PR = It->second;
    if (SubIdx) {
      PR = PartnerIsSub ? TRI->getMatchingSuperReg(PR, SubIdx, RC)
                        : TRI->getSubReg(PR, SubIdx);
      if (!PR)
        return; // no such slice/super in the physreg or class
    }
    if (!RC->contains(PR))
      return; // class/width mismatch after composition
    unsigned Depth = EdgeBlock ? MLI->getLoopDepth(EdgeBlock) : 0;
    uint64_t W = Depth < 63 ? (uint64_t(1) << Depth) : ~uint64_t(0);
    Cand.push_back({PR, W});
  };

  MachineInstr *Def = MRI->getUniqueVRegDef(VReg);

  // Direction A -- VReg is a phi result: align to its (colored) operands. If an
  // operand reads a slice (%wide.subN), VReg's color is that slice of the
  // operand's color (PartnerIsSub = false).
  if (Def && Def->isPHI()) {
    for (unsigned I = 1, E = Def->getNumOperands(); I < E; I += 2) {
      MachineOperand &Src = Def->getOperand(I);
      if (Src.isUndef() || !Src.isReg())
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

  // Hottest-first, deduped (keep max weight per physreg).
  llvm::stable_sort(Cand, [](auto &A, auto &B) { return A.second > B.second; });
  SmallVector<MCRegister, 4> Hints;
  for (auto &[PR, W] : Cand)
    if (!llvm::is_contained(Hints, PR))
      Hints.push_back(PR);
  return Hints;
}

void AMDGPUSSARegisterAllocator::seedOccupiedAtBBEntry(MachineBasicBlock *MBB) {
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
}

bool AMDGPUSSARegisterAllocator::edgeCopiesNeedSplit(
    MachineBasicBlock *Pred, MachineBasicBlock *MBB,
    ArrayRef<std::pair<MCRegister, MCRegister>> Copies) const {
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

void AMDGPUSSARegisterAllocator::color() {
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
  DenseSet<Register> ACLSet;
  if (EnableAMDGPUSSAACLColoring) {
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
                  else
                    OccupiedRegUnits.reset(Unit);
                }
              continue;
            }
            auto It = ColorMap.find(Reg);
            if (It == ColorMap.end())
              continue;
            if (!LIS->getInterval(Reg).liveAt(NextSI)) {
              if (HasEC) {
                DeferredFree.push_back(It->second);
              } else {
                markFree(It->second);
                LLVM_DEBUG(dbgs()
                           << "    kill: " << printReg(Reg, TRI) << " free "
                           << TRI->getName(It->second) << "\n");
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
            assert(Chosen && "Failed to find free physreg");
            LLVM_DEBUG(dbgs() << "    color (undef self-tie): "
                              << printReg(Reg, TRI) << " -> "
                              << TRI->getName(Chosen) << "\n");
          } else if (IsTied) {
            llvm_unreachable("Tied use must be colored already or undef");
          } else {
            SmallVector<MCRegister, 4> Hints =
                collectPhiHints(Reg, MRI->getRegClass(Reg));
            Chosen = pickFreePhysReg(MRI->getRegClass(Reg),
                                     LIS->getInterval(Reg), WiderDefs, Hints);
            if (!Chosen) {
              const LiveInterval &FVI = LIS->getInterval(Reg);
              dbgs() << "!!! COLORFAIL " << printReg(Reg, TRI) << " " << FVI
                     << " class=" << TRI->getRegClassName(MRI->getRegClass(Reg))
                     << "\n";
              // Every already-colored value whose interval OVERLAPS %308, with
              // its physreg + interval. These are the true blockers.
              dbgs() << "  overlapping colored values (blockers):\n";
              for (const auto &[V, P] : ColorMap) {
                if (!LIS->hasInterval(V))
                  continue;
                const LiveInterval &OVI = LIS->getInterval(V);
                if (OVI.overlaps(FVI))
                  dbgs() << "    " << printReg(V, TRI) << " -> "
                         << TRI->getName(P) << "  " << OVI << "\n";
              }
            }
            assert(Chosen && "Failed to find free physreg");
            LLVM_DEBUG(dbgs() << "    color: " << printReg(Reg, TRI) << " -> "
                              << TRI->getName(Chosen) << "\n");
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
        for (MCRegUnit Unit : DeferredUnits)
          OccupiedRegUnits.reset(Unit);
        for (MCRegister PR : DeferredFree)
          markFree(PR);
      }
    }
  } // width loop
  } // phase loop

  LLVM_DEBUG({
    dbgs() << "\nColoring result:\n";
    for (const auto &[VReg, PhysReg] : ColorMap)
      dbgs() << "  " << printReg(VReg, TRI) << " -> " << TRI->getName(PhysReg)
             << "\n";
  });
}

// === SSA Destruction + Operand Rewrite ===

bool AMDGPUSSARegisterAllocator::hasCFPseudos(MachineFunction &MF) const {
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
  const TargetRegisterClass *RC = TRI->getPhysRegBaseClass(RegA);
  unsigned RegWidth = TRI->getRegSizeInBits(*RC);

  // In-place XOR swap: A ^= B; B ^= A; A ^= B.
  auto EmitXorTriplet = [&](unsigned Opc) {
    BuildMI(MBB, InsertPt, DebugLoc(), TII->get(Opc), RegA)
        .addReg(RegA)
        .addReg(RegB);
    BuildMI(MBB, InsertPt, DebugLoc(), TII->get(Opc), RegB)
        .addReg(RegA)
        .addReg(RegB);
    BuildMI(MBB, InsertPt, DebugLoc(), TII->get(Opc), RegA)
        .addReg(RegA)
        .addReg(RegB);
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
    if (ST->hasTrue16BitInsts())
      BuildMI(MBB, InsertPt, DebugLoc(), TII->get(AMDGPU::V_SWAP_B16), RegA)
          .addDef(RegB)
          .addReg(RegB)
          .addReg(RegA);
    else
      EmitXorTriplet(AMDGPU::V_XOR_B16_fake16_e64);
    return;
  }
  if (RegWidth <= 32) {
    if (ST->hasSwap())
      BuildMI(MBB, InsertPt, DebugLoc(), TII->get(AMDGPU::V_SWAP_B32), RegA)
          .addDef(RegB)
          .addReg(RegB)
          .addReg(RegA);
    else
      EmitXorTriplet(AMDGPU::V_XOR_B32_e64);
    return;
  }
  SwapInChunks(4);
}

void AMDGPUSSARegisterAllocator::resolvePermutation(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
    SmallVectorImpl<std::pair<MCRegister, MCRegister>> &Copies) {
  if (Copies.empty())
    return;

  DenseMap<MCRegister, MCRegister> DstToSrc;
  DenseMap<MCRegister, unsigned> SrcRefCount;
  for (auto &[Src, Dst] : Copies) {
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
    BuildMI(MBB, InsertPt, DebugLoc(), TII->get(TargetOpcode::COPY), Dst)
        .addReg(Src);
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
  const MachineFunction &MF = *MBB.getParent();

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
    unsigned CurrentOcc =
        IsVGPR ? ST->getOccupancyWithNumVGPRs(MaxIdx, DynVGPRBlockSize)
               : ST->getOccupancyWithNumSGPRs(MaxIdx);

    // Scratch must match the cycle's register width.
    unsigned CycleWidth =
        TRI->getRegSizeInBits(*TRI->getPhysRegBaseClass(CycleStart)) / 32;
    unsigned ScratchOcc =
        IsVGPR ? ST->getOccupancyWithNumVGPRs(MaxIdx + CycleWidth,
                                              DynVGPRBlockSize)
               : ST->getOccupancyWithNumSGPRs(MaxIdx + CycleWidth);
    bool ScratchFits = MaxIdx + CycleWidth <= MaxHWLimit;

    // Decide between resolving the cycle with a scratch register (plain COPYs)
    // and in place via emitSwap.
    //   VGPR: emitSwap (V_SWAP_B32 or a V_XOR triplet) is scratch- and
    //   SCC-free,
    //         so prefer it; use a scratch only when swap is unavailable and it
    //         costs no occupancy.
    //   SGPR: there is no scalar swap. emitSwap uses an S_XOR triplet, which
    //         writes SCC and is therefore only safe when SCC is dead here.
    //         Otherwise a scratch COPY is the only SCC-preserving option.
    bool UseScratch;
    if (IsVGPR) {
      UseScratch = !ST->hasSwap() && ScratchOcc == CurrentOcc && ScratchFits;
    } else if (IsAGPR) {
      // AGPRs have no swap or XOR primitive, so an in-place emitSwap is
      // impossible; a scratch AGPR (plain COPYs, legalized to AGPR moves
      // downstream) is the only way to break the cycle.
      UseScratch = true;
      assert(ScratchFits &&
             "AGPR permutation cycle with no free scratch register");
    } else {
      bool SccDead = MBB.computeRegisterLiveness(TRI, AMDGPU::SCC, InsertPt) ==
                     MachineBasicBlock::LQR_Dead;
      UseScratch = !SccDead;
      assert(
          (!UseScratch || ScratchFits) &&
          "SGPR permutation cycle with live SCC and no free scratch register");
    }

    if (UseScratch && ScratchFits) {
      MCRegister ScratchBase =
          IsVGPR ? MCRegister(AMDGPU::VGPR0 + MaxIdx)
                 : IsAGPR ? MCRegister(AMDGPU::AGPR0 + MaxIdx)
                          : MCRegister(AMDGPU::SGPR0 + MaxIdx);
      MCRegister Scratch =
          (CycleWidth == 1)
              ? ScratchBase
              : TRI->getMatchingSuperReg(ScratchBase, AMDGPU::sub0,
                                         TRI->getPhysRegBaseClass(CycleStart));
      MaxIdx += CycleWidth;

      LLVM_DEBUG(dbgs() << "    cycle via scratch " << TRI->getName(Scratch)
                        << ":\n");

      // Save CycleStart — it will be overwritten by the first copy.
      // The last register in the walk receives this saved value.
      BuildMI(MBB, InsertPt, DebugLoc(), TII->get(TargetOpcode::COPY), Scratch)
          .addReg(CycleStart);
      LLVM_DEBUG(dbgs() << "      save: " << TRI->getName(CycleStart) << " -> "
                        << TRI->getName(Scratch) << "\n");

      MCRegister Cur = CycleStart;
      while (true) {
        MCRegister Src = DstToSrc[Cur];
        DstToSrc.erase(Cur);
        if (!DstToSrc.count(Src)) {
          assert(Src == CycleStart && "Cycle walk did not return to start");
          BuildMI(MBB, InsertPt, DebugLoc(), TII->get(TargetOpcode::COPY), Cur)
              .addReg(Scratch);
          LLVM_DEBUG(dbgs() << "      restore: " << TRI->getName(Scratch)
                            << " -> " << TRI->getName(Cur) << "\n");
          break;
        }
        BuildMI(MBB, InsertPt, DebugLoc(), TII->get(TargetOpcode::COPY), Cur)
            .addReg(Src);
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
}

void AMDGPUSSARegisterAllocator::lowerPHIs(MachineFunction &MF) {
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
      auto InsertPt = InsertMBB->getFirstTerminator();
      // Materialize undef edges (null source) as IMPLICIT_DEF of DstPhys and
      // drop them; the remainder are real copies handed to resolvePermutation.
      for (auto *It = Copies.begin(); It != Copies.end();) {
        if (!It->first) {
          BuildMI(*InsertMBB, InsertPt, DebugLoc(),
                  TII->get(TargetOpcode::IMPLICIT_DEF), It->second);
          It = Copies.erase(It);
        } else {
          ++It;
        }
      }
      resolvePermutation(*InsertMBB, InsertPt, Copies);
    }
  }

  for (MachineInstr *PHI : PHIsToErase)
    PHI->eraseFromParent();

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

void AMDGPUSSARegisterAllocator::rewriteOperands(MachineFunction &MF) {
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
        MCRegister PhysReg = ColorMap.lookup(VReg);
        if (!PhysReg) {
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

        unsigned SubIdx = MO.getSubReg();
        if (SubIdx) {
          PhysReg = TRI->getSubReg(PhysReg, SubIdx);
          assert(PhysReg && "Invalid subreg index");
          MO.setSubReg(0);
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
void AMDGPUSSARegisterAllocator::eliminateRegSequences(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      if (!MI.isRegSequence())
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
      MI.eraseFromParent();
    }
  }
}

void AMDGPUSSARegisterAllocator::destroySSAAndRewrite(MachineFunction &MF) {
  if (hasCFPseudos(MF)) {
    LLVM_DEBUG(dbgs() << "SSA Destruction: skipped — "
                         "SI control-flow pseudos present\n");
    return;
  }

  lowerPHIs(MF);
  rewriteOperands(MF);
  eliminateRegSequences(MF);
  addPhysRegLiveIns(MF);
  finalizeProperties(MF);
}

// === Main entry point ===

bool AMDGPUSSARegisterAllocator::runOnMachineFunction(MachineFunction &MF) {
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

  LLVM_DEBUG(dbgs() << "AMDGPUSSARegisterAllocator: Processing " << MF.getName()
                    << "\n");

  classifyVRegs();
  OccupiedRegUnits.clear();
  OccupiedRegUnits.resize(TRI->getNumRegUnits());

  ColorMap.clear();
  MaxVGPRIdx = 0;
  MaxSGPRIdx = 0;
  MaxAGPRIdx = 0;

  color();
  destroySSAAndRewrite(MF);

  return true;
}

MachineFunctionPass *llvm::createAMDGPUSSARegisterAllocatorPass() {
  return new AMDGPUSSARegisterAllocator();
}
