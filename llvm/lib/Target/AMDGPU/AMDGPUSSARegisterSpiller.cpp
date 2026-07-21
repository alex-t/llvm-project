//===--------------- AMDGPUSSARegisterSpiller.cpp  -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUSSARegisterSpiller.h"
#include "AMDGPU.h"
#include "GCNRegPressure.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIRegisterInfo.h"
#include "VRegMaskPair.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"
#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-ssa-register-spiller"

// NumSpills/NumReloads statistics moved to SSASpillEmitter.cpp with the emission
// machinery that increments them.

static cl::opt<bool> EnableVirtualSpillMarkers(
    "amdgpu-ssa-spill-markers",
    cl::desc("Emit SI_VIRTUAL_SPILL_MARKER instructions for SSA spiller tests"),
    cl::Hidden, cl::init(false));

// ============================================================================
static cl::opt<bool>
    DisableReloadOptimizer("amdgpu-ssa-spill-no-reload-opt",
                           cl::desc("Disable reload optimizer in SSA spiller"),
                           cl::init(false), cl::Hidden);

// ============================================================================
static cl::opt<bool> EnableNarrowRemnant(
    "amdgpu-ssa-narrow-spill-remnant", cl::init(true), cl::Hidden,
    cl::desc("After a partial (strict-subset) spill of a wide value, extract the "
             "un-spilled remnant lanes into a fresh narrow vreg so the wide vreg "
             "vacates its aligned tuple (unblocks aligned placement)"));

namespace llvm {
cl::opt<bool> EnableSplitLiveRanges(
    "amdgpu-ssa-split-live-ranges", cl::init(false), cl::Hidden,
    cl::desc("After RP spilling, split values that are point-feasible but have no "
             "physreg free across their whole span (span_free=0) at the region "
             "boundary, so coloring can place the halves separately (SSARA "
             "live-range-splitting experiment)"));
} // namespace llvm

// ============================================================================
static cl::opt<cl::boolOrDefault> VerifyFinalRP(
    "amdgpu-ssa-spiller-verify-rp",
    cl::desc("Verify final register pressure stays within the limit after SSA "
             "spilling (default: on in expensive-checks builds)"),
    cl::Hidden);

// ============================================================================
static cl::opt<bool> EnablePreservedRPFixpoint(
    "amdgpu-ssa-spiller-presrp-fixpoint",
    cl::desc("Re-classify pinned vregs and re-run the spill pass until the "
             "preserved-RP clique fits callee-saved capacity (Option A)"),
    cl::init(true), cl::Hidden);

// Master switch for all around-call-liver (ACL) work, defined in
// AMDGPUSSARegisterAllocator.cpp. When off, the spiller skips the preserved-RP
// (callee-saved capacity) gate entirely, matching the committed baseline.
namespace llvm {
extern cl::opt<bool> EnableAMDGPUSSAACLColoring;
}

// isSpillInstr / isReloadInstr are now shared inline helpers in
// SSASpillEmitter.h (used by both the spiller policy and the emitter).

char AMDGPUSSARegisterSpiller::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPUSSARegisterSpiller, DEBUG_TYPE,
                      "AMDGPU SSA Register Spiller", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AMDGPUNextUseAnalysisWrapper)
INITIALIZE_PASS_END(AMDGPUSSARegisterSpiller, DEBUG_TYPE,
                    "AMDGPU SSA Register Spiller", false, false)

VRegMaskPairSet AMDGPUSSARegisterSpiller::convertLiveRegs(
    const GCNRPTracker::LiveRegSet &LiveRegs) const {
  VRegMaskPairSet Result;
  for (const auto &[Reg, Mask] : LiveRegs) {
    if (Register::isVirtualRegister(Reg)) {
      Result.insert(VRegMaskPair(Register(Reg), Mask));
    }
  }
  return Result;
}

Printable printVRegMaskPairSet(const VRegMaskPairSet &VMPSet) {
  return Printable([&](raw_ostream &OS) { VMPSet.dump(); });
}

bool AMDGPUSSARegisterSpiller::inCurrentFile(
    const TargetRegisterClass *RC) const {
  if (!IsVGPRPass)
    return TRI->isSGPRClass(RC);
  // VGPR pass owns everything getVGPRNum folds into the VGPR count: plain VGPR,
  // AGPR, and the AGPR-or-VGPR vector-super ("av_") classes. isVGPRClass and
  // isAGPRClass are both false for an av_ class, so it must be caught explicitly.
  return TRI->isVGPRClass(RC) || TRI->isAGPRClass(RC) ||
         TRI->isVectorSuperClass(RC);
}

VRegMaskPairSet
AMDGPUSSARegisterSpiller::getLiveRegsForCurrentFile(SlotIndex Slot) const {
  // TOTAL_KINDS = no kind filter; we filter by inCurrentFile so the av_/AGPR
  // pressure the VGPR pass counts is also visible as spill candidates.
  GCNRPTracker::LiveRegSet Live =
      llvm::getLiveRegs(Slot, *LIS, *MRI, GCNRegPressure::TOTAL_KINDS);
  VRegMaskPairSet Result;
  for (const auto &[Reg, Mask] : Live) {
    if (!Register::isVirtualRegister(Reg))
      continue;
    if (!inCurrentFile(MRI->getRegClass(Reg)))
      continue;
    Result.insert(VRegMaskPair(Register(Reg), Mask));
  }
  return Result;
}

void AMDGPUSSARegisterSpiller::validateFinalRegisterPressure(
    MachineFunction &MF, unsigned RPLimit, bool IsVGPR) {

  const char *RegClassName = IsVGPR ? "VGPR" : "SGPR";

  LLVM_DEBUG(dbgs() << "\n=== Validating Final Register Pressure ("
                    << RegClassName << ") ===\n");

  // Traverse basic blocks same as in processFunction
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);

  for (MachineBasicBlock *MBB : RPOT) {
    if (MBB->empty())
      continue;

    // Walk forward through the block
    for (auto I = MBB->begin(), E = MBB->end(); I != E; ++I) {
      MachineInstr &MI = *I;

      // Skip spill/reload instructions (same as in processFunction)
      if (isSpillInstr(&MI) || isReloadInstr(&MI))
        continue;

      // Validate against the same peak (read/write phase) metric the spiller
      // targets; the after-instruction live set alone underestimates pressure
      // when operands die in place. A PHI is excluded: its operands are not
      // read at the PHI (the sources are live out of the predecessors and moved
      // by edge copies during SSA destruction), so its pressure is the
      // block-entry live set, not a read peak.
      RPTracker->reset(MI);
      GCNRegPressure CurPressure;
      if (MI.isPHI()) {
        CurPressure = RPTracker->getPressure();
      } else {
        RPTracker->recede(MI);
        CurPressure = RPTracker->getMaxPressure();
      }
      const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
      unsigned CurRP = IsVGPR ? CurPressure.getVGPRNum(ST.hasGFX90AInsts())
                              : CurPressure.getSGPRNum();

      if (CurRP > RPLimit) {
        std::string Msg;
        raw_string_ostream OS(Msg);
        OS << "SSA Spiller FINAL RP VALIDATION FAILED!\n";
        OS << "  Register class: " << RegClassName << "\n";
        OS << "  Current RP: " << CurRP << "\n";
        OS << "  RP Limit: " << RPLimit << "\n";
        OS << "  At instruction: " << MI << "\n";
        OS << "  In block: " << printMBBReference(*MBB) << "\n";
        OS << "\nThis indicates the spiller failed to keep RP within limits.\n";
        report_fatal_error(Twine(OS.str()));
      }
    }
  }

  LLVM_DEBUG(dbgs() << "✅ Final RP validation passed for " << RegClassName
                    << "\n");
}

void AMDGPUSSARegisterSpiller::computePinnedAndCap(MachineFunction &MF) {
  // Classify PINNED vregs: those whose live interval crosses ANY call. Such a
  // value must sit in a callee-saved register for its whole range (this RA does
  // not split live ranges), so it contributes to preserved-RP everywhere it is
  // live. Computed independently in the spiller (not shared with the allocator).
  PinnedVRegs.clear();
  SmallVector<SlotIndex, 8> CallSlots;
  SmallVector<const uint32_t *, 4> CallMasks;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (MI.isCall())
        for (const MachineOperand &MO : MI.operands())
          if (MO.isRegMask()) {
            CallSlots.push_back(LIS->getInstructionIndex(MI).getRegSlot());
            CallMasks.push_back(MO.getRegMask());
            break;
          }

  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I < E; ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (MRI->reg_nodbg_empty(VReg) || !LIS->hasInterval(VReg))
      continue;
    const LiveInterval &LI = LIS->getInterval(VReg);
    for (SlotIndex S : CallSlots)
      if (LI.liveAt(S)) {
        PinnedVRegs.insert(VReg);
        break;
      }
  }

  // k_cs per file = min over all calls of |allocatable(file) preserved by the
  // call's regmask|. Function-wide min is the conservative bound: a pinned value
  // may cross any call, so it must fit the most restrictive preserved set.
  auto MinPreserved = [&](const TargetRegisterClass *RC) -> unsigned {
    BitVector Alloc = TRI->getAllocatableSet(MF, RC);
    unsigned MinN = Alloc.count(); // no calls -> full file (no ceiling effect)
    for (const uint32_t *Mask : CallMasks) {
      unsigned N = 0;
      for (unsigned Reg : Alloc.set_bits())
        if (!MachineOperand::clobbersPhysReg(Mask, MCRegister(Reg)))
          ++N;
      MinN = std::min(MinN, N);
    }
    return MinN;
  };
  VGPRPreservedCap = MinPreserved(&AMDGPU::VGPR_32RegClass);
  SGPRPreservedCap = MinPreserved(&AMDGPU::SGPR_32RegClass);
  LLVM_DEBUG(dbgs() << "Pinned vregs: " << PinnedVRegs.size()
                    << "; k_cs VGPR=" << VGPRPreservedCap
                    << " SGPR=" << SGPRPreservedCap << "\n");

  // DIAGNOSTIC (answers "does splitting suffice?"): at EACH call, the
  // width-weighted count of vregs live ACROSS it (its own live-across set, not
  // the whole pinned set) per file, vs that call's preserved capacity. If every
  // call's live-across ≤ its capacity, there is NO per-call capacity violation →
  // the exhaustion is placement/interference (splitting or a point-accurate
  // metric fixes it). If some call exceeds → real spilling is unavoidable there.
  LLVM_DEBUG({
    for (unsigned CI = 0; CI < CallSlots.size(); ++CI) {
      SlotIndex CS = CallSlots[CI];
      const uint32_t *Mask = CallMasks[CI];
      auto CapFor = [&](const TargetRegisterClass *RC) {
        BitVector A = TRI->getAllocatableSet(MF, RC);
        unsigned N = 0;
        for (unsigned R : A.set_bits())
          if (!MachineOperand::clobbersPhysReg(Mask, MCRegister(R)))
            ++N;
        return N;
      };
      unsigned AcrossV = 0, AcrossS = 0;
      for (unsigned I = 0, E = MRI->getNumVirtRegs(); I < E; ++I) {
        Register VReg = Register::index2VirtReg(I);
        if (MRI->reg_nodbg_empty(VReg) || !LIS->hasInterval(VReg))
          continue;
        if (!LIS->getInterval(VReg).liveAt(CS))
          continue;
        const TargetRegisterClass *RC = MRI->getRegClass(VReg);
        unsigned W = TRI->getRegSizeInBits(*RC) / 32;
        if (TRI->isSGPRClass(RC))
          AcrossS += W;
        else if (TRI->isVGPRClass(RC) || TRI->isAGPRClass(RC))
          AcrossV += W;
      }
      dbgs() << "  [XCALL] call@" << CS << " acrossVGPR=" << AcrossV << "/"
             << CapFor(&AMDGPU::VGPR_32RegClass) << " acrossSGPR=" << AcrossS
             << "/" << CapFor(&AMDGPU::SGPR_32RegClass)
             << (AcrossS > CapFor(&AMDGPU::SGPR_32RegClass) ||
                         AcrossV > CapFor(&AMDGPU::VGPR_32RegClass)
                     ? "  *** EXCEEDS ***"
                     : "")
             << "\n";
      // [SPLITCAND] Classify SGPR values live across an EXCEEDING call by whether
      // they have a use STRICTLY BEFORE the call slot (both-sides -> spilling
      // births a crossing reload -> iterates) vs post-call-only / passthrough
      // (clean: store-at-def + reload-at-post-use removes the crossing in one
      // shot). Counts the clean pool vs the shed target (across - cap).
      if (AcrossS > CapFor(&AMDGPU::SGPR_32RegClass)) {
        unsigned CleanSlots = 0, BothSlots = 0;
        for (unsigned I = 0, E = MRI->getNumVirtRegs(); I < E; ++I) {
          Register VReg = Register::index2VirtReg(I);
          if (MRI->reg_nodbg_empty(VReg) || !LIS->hasInterval(VReg))
            continue;
          const TargetRegisterClass *RC = MRI->getRegClass(VReg);
          if (!TRI->isSGPRClass(RC))
            continue;
          if (!LIS->getInterval(VReg).liveAt(CS))
            continue;
          unsigned W = TRI->getRegSizeInBits(*RC) / 32;
          bool HasPreUse = false;
          for (MachineInstr &U : MRI->use_nodbg_instructions(VReg)) {
            SlotIndex UI = LIS->getInstructionIndex(U).getRegSlot();
            if (UI < CS) { HasPreUse = true; break; }
          }
          if (HasPreUse) BothSlots += W;
          else CleanSlots += W;
          dbgs() << "    [SPLITCAND] " << printReg(VReg, TRI) << " w=" << W
                 << (HasPreUse ? " BOTH-SIDES" : " CLEAN(post/passthru)")
                 << " " << LIS->getInterval(VReg) << "\n";
        }
        dbgs() << "  [SPLITSUM] call@" << CS << " shedTarget="
               << (AcrossS - CapFor(&AMDGPU::SGPR_32RegClass))
               << " cleanPool=" << CleanSlots << " bothSides=" << BothSlots
               << (CleanSlots >= AcrossS - CapFor(&AMDGPU::SGPR_32RegClass)
                       ? "  -> CLEAN POOL SUFFICES"
                       : "  -> NEEDS BOTH-SIDES (iteration)")
               << "\n";
      }
    }
  });
}

unsigned AMDGPUSSARegisterSpiller::computePreservedRP(const MachineInstr &MI) {
  // preserved-RP at MI = 32-bit-slot count of PINNED vregs of this pass's file
  // live across MI (dead slot: values that survive MI). This is the second RP
  // dimension bounded by k_cs.
  if (PinnedVRegs.empty())
    return 0;
  SlotIndex Slot = LIS->getInstructionIndex(MI).getRegSlot();
  // Current pass's whole file (VGPR ∪ AGPR ∪ av_), matching the pressure metric
  // — a single-RegKind query would miss av_ tuples that count toward pressure.
  VRegMaskPairSet Live = getLiveRegsForCurrentFile(Slot);
  unsigned N = 0, NAll = 0;
  for (const VRegMaskPair &VMP : Live) {
    unsigned W = VMP.getSizeInRegs(TRI);
    NAll += W;
    if (PinnedVRegs.contains(VMP.getVReg()))
      N += W;
  }
  LLVM_DEBUG(if (N > PreservedLimit) dbgs()
                 << "  [PRESRP] @"
                 << LIS->getInstructionIndex(MI).getRegSlot() << " pinnedN=" << N
                 << " allLiveN=" << NAll << " k_cs=" << PreservedLimit
                 << (MI.isCall() ? " (call)" : "") << "\n");
  return N;
}

bool AMDGPUSSARegisterSpiller::processFunction(MachineFunction &MF,
                                               unsigned RPLimit) {
  LLVM_DEBUG(dbgs() << "processFunction: " << (IsVGPRPass ? "VGPR" : "SGPR")
                    << " pass, limit=" << RPLimit << "\n");

  // (Re)create the emitter's SSA updater and select the file. The updater is
  // reused throughout the pass and caches IDF computations.
  // FIXME: Clear cache if CFG changes during spilling
  Emitter->beginPass(IsVGPRPass);

  // Initialize register pressure tracker (reused throughout the pass)
  RPTracker = std::make_unique<GCNUpwardRPTracker>(*LIS);

  // Store RP limits for reload budget checking
  if (IsVGPRPass)
    VGPRLimit = RPLimit;
  else
    SGPRLimit = RPLimit;

  // Preserved-register (callee-saved) capacity for this pass's file — the second
  // RP dimension's ceiling (see computePreservedRP and the gate in the walk).
  PreservedLimit = IsVGPRPass ? VGPRPreservedCap : SGPRPreservedCap;

  // Track if we made any modifications
  bool Changed = false;

  // Traverse basic blocks in reverse post-order (RPO)
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);

  Emitter->clearReloadedRegs();

  for (MachineBasicBlock *MBB : RPOT) {
    LLVM_DEBUG(dbgs() << "\nProcessing " << printMBBReference(*MBB) << "\n");

    if (MBB->empty())
      continue;

    // Seed physreg pressure from block live-ins.
    // GCNRegPressure only tracks virtual registers; physical registers
    // reduce the available budget and must be counted separately.
    unsigned LivePhysRP = 0;
    for (const auto &LI : MBB->liveins()) {
      const TargetRegisterClass *RC = TRI->getPhysRegBaseClass(LI.PhysReg);
      if (RC && !MRI->isReserved(LI.PhysReg) && inCurrentFile(RC))
        LivePhysRP += TRI->getRegSizeInBits(*RC) / 32;
    }

    // Traverse instructions forward (from beginning to end)
    // When we spill at point P, pressure drops from P forward
    //
    // Design Note: We use reset() + forward walk (not recede() + backward walk)
    // because:
    // - Spill insertion at I reduces pressure from I *forward* (down in control
    // flow)
    // - Walking forward with reset(I) naturally sees reduced pressure at I+1
    // after spilling at I
    // - Walking backward with recede() would detect high RP at I *after*
    // already processing
    //   instructions I+1, I+2, ... that would benefit from the spill (timing
    //   mismatch)
    // - reset() cost is O(n) per instruction, acceptable for typical block
    // sizes
    for (auto I = MBB->begin(), E = MBB->end(); I != E; ++I) {
      MachineInstr &MI = *I;

      // Skip spill and reload instructions we create
      if (isSpillInstr(&MI) || isReloadInstr(&MI)) {
        LLVM_DEBUG(dbgs() << "  Skipping spill/reload: " << MI);
        continue;
      }

      LLVM_DEBUG(dbgs() << "  Processing: " << MI);

      // Update physreg pressure on the fly: kills before defs.
      // Pressure is counted in 32-bit register slots. For wide physregs,
      // each 32-bit sub-register is checked independently (partial kills).
      // Reserved registers (EXEC, M0, etc.) are skipped — they have no
      // LiveRange in LiveIntervals and are excluded from allocation.
      SlotIndex NextSI =
          LIS->getInstructionIndex(MI).getRegSlot().getNextSlot();
      unsigned PhysDefs = 0;
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg().isPhysical())
          continue;
        Register Reg = MO.getReg();
        const TargetRegisterClass *RC = TRI->getPhysRegBaseClass(Reg);
        if (!RC || MRI->isReserved(Reg) || !RC->isAllocatable() ||
            !(IsVGPRPass ? TRI->isVGPRClass(RC) : TRI->isSGPRClass(RC)))
          continue;
        unsigned Width = TRI->getRegSizeInBits(*RC) / 32;
        // A physreg use frees pressure only for the 32-bit slots it kills. An
        // `undef` use reads no live value and was never added to LivePhysRP, so
        // it must not decrement it -- otherwise the unsigned counter underflows
        // (e.g. SI_RETURN_TO_EPILOG's `implicit undef $vgprN` operands), which
        // fabricates enormous pressure and forces a spurious spill.
        if (MO.isUse() && !MO.isUndef()) {
          // Pressure is counted in 32-bit slots. A single 32-bit register spans
          // several reg units (its 16-bit sub-lanes); a wider tuple is split
          // into its dword sub-registers. Either way, a slot is dead only if all
          // of its units are dead, and each dead slot frees one unit of pressure.
          SmallVector<MCRegister, 8> Slots;
          if (Width == 1)
            Slots.push_back(Reg);
          else
            for (int16_t SubIdx : TRI->getRegSplitParts(RC, /*DWordBytes=*/4))
              Slots.push_back(TRI->getSubReg(Reg, SubIdx));
          for (MCRegister Slot : Slots) {
            bool Dead = true;
            for (MCRegUnit Unit : TRI->regunits(Slot))
              if (LIS->getRegUnit(Unit).liveAt(NextSI)) {
                Dead = false;
                break;
              }
            if (Dead)
              --LivePhysRP;
          }
        }
        if (MO.isDef())
          PhysDefs += Width;
      }
      LivePhysRP += PhysDefs;

      // An instruction's register pressure is its peak simultaneous demand:
      // the maximum of the read phase (all uses + values live across it) and
      // the write phase (all defs + values live across it). The live set after
      // the instruction is not enough — when operands die in place, a
      // multi-input or early-clobber instruction can require more registers
      // while executing than remain live once it has finished.
      //
      // reset(MI) seeds the tracker with the live set just after MI; recede(MI)
      // moves back across MI, folding in both phases (early-clobber aware) so
      // the peak lands in getMaxPressure(). The per-instruction reset re-seeds
      // the otherwise-running MaxPressure, scoping it to this instruction.
      //
      // A PHI has no read phase: its operands are not read here — the source
      // values are live out of the predecessors and moved by copies on the
      // edges during SSA destruction. Its pressure is the block-entry live set,
      // so use the post-instruction set directly (no recede).
      RPTracker->reset(MI);
      GCNRegPressure CurPressure;
      if (MI.isPHI()) {
        CurPressure = RPTracker->getPressure();
      } else {
        RPTracker->recede(MI);
        CurPressure = RPTracker->getMaxPressure();
      }

      // Get pressure for the current pass using the appropriate API
      const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
      unsigned CurRP = IsVGPRPass ? CurPressure.getVGPRNum(ST.hasGFX90AInsts())
                                  : CurPressure.getSGPRNum();
      CurRP += LivePhysRP;

      LLVM_DEBUG(dbgs() << "    " << (IsVGPRPass ? "VGPR" : "SGPR")
                        << " pressure: " << CurRP << "\n");

      // Check if we need to spill
      if (CurRP > RPLimit) {
        LLVM_DEBUG(dbgs() << "  " << (IsVGPRPass ? "VGPR" : "SGPR")
                          << " pressure " << CurRP << " > limit " << RPLimit
                          << ", need to spill\n");

        // Get the slot index for the current instruction
        SlotIndex Slot = LIS->getInstructionIndex(MI).getRegSlot();

        // Live vregs of the current pass's whole file (VGPR ∪ AGPR ∪ av_ for the
        // VGPR pass) as spill candidates. A single-RegKind getLiveRegs(VGPR)
        // would drop av_ tuples that DO count toward getVGPRNum pressure, so the
        // pass would see pressure over the limit but find nothing to spill.
        VRegMaskPairSet ActiveRegs = getLiveRegsForCurrentFile(Slot);

        LLVM_DEBUG(dbgs() << "ActiveRegs: " << printVRegMaskPairSet(ActiveRegs)
                          << "\n");
        LLVM_DEBUG(dbgs() << "ReloadedRegs: "
                          << printVRegMaskPairSet(Emitter->reloadedRegs())
                          << "\n");
        ActiveRegs.set_subtract(Emitter->reloadedRegs());
        LLVM_DEBUG(dbgs() << "ActiveRegs after subtracting ReloadedRegs: "
                          << printVRegMaskPairSet(ActiveRegs) << "\n");

        // CRITICAL: Exclude registers DEFINED by the current instruction!
        // RPTracker.reset(MI) gives us RP AFTER MI executes, which includes
        // registers defined by MI. But we insert spills BEFORE MI, so we
        // cannot spill a register that doesn't exist yet. We must exclude
        // MI.defs().
        VRegMaskPairSet ToRemove;
        for (const MachineOperand &MO : MI.defs()) {
          if (MO.getReg().isVirtual()) {
            // Create VRegMaskPair from the def operand to match both reg and
            // mask
            VRegMaskPair Def(MO, TRI, MRI);
            // Look for matching VMP in active set
            for (const auto &VMP : ActiveRegs) {
              if (Def == VMP) {
                ToRemove.insert(VMP);
                LLVM_DEBUG(dbgs()
                           << "  Excluding " << printReg(Def.getVReg(), TRI)
                           << " with mask " << PrintLaneMask(Def.getLaneMask())
                           << " (defined by current instruction)\n");
              }
            }
          }
        }
        // Remove them from active candidates
        ActiveRegs.set_subtract(ToRemove);

        MachineBasicBlock::reverse_iterator ReverseI(std::next(I));

        // Call spillAndReload to handle atomic spill+reload+SSA repair
        bool Spilled =
            spillAndReload(*MBB, ReverseI, ActiveRegs, CurRP, RPLimit);
        if (Spilled) {
          Changed = true;
        }

        // Note: After spilling at point P, the spilled register's pressure
        // contribution is removed from P forward. We continue walking forward
        // and will see lower pressure at subsequent instructions.
      }
    }
  }

  // Verify the spiller kept register pressure within the limit at every
  // instruction. This re-walks the function (O(n)); it runs when explicitly
  // requested (-amdgpu-ssa-spiller-verify-rp) and by default in
  // expensive-checks builds.
  bool DoVerifyRP = VerifyFinalRP == cl::BOU_TRUE;
#ifdef EXPENSIVE_CHECKS
  if (VerifyFinalRP == cl::BOU_UNSET)
    DoVerifyRP = true;
#endif
  if (DoVerifyRP)
    validateFinalRegisterPressure(MF, RPLimit, IsVGPRPass);

  return Changed;
}

unsigned AMDGPUSSARegisterSpiller::maxPreservedClique() const {
  // Authoritative preserved-RP: the max width-weighted clique over the current
  // PinnedVRegs of this pass's file, computed by an endpoint sweep over their
  // live intervals. Unlike a per-instruction getLiveRegs scan, this counts a
  // simultaneously-live pinned set even at a point where no instruction sits,
  // and reflects reload vregs that computePinnedAndCap picked up on its most
  // recent (post-spill) run. This is exactly the CSR clique the allocator's
  // ACL phase will try to color; while it exceeds k_cs, coloring must fail.
  struct Ev {
    SlotIndex S;
    int D;
    unsigned W;
  };
  SmallVector<Ev, 64> Evs;
  for (Register V : PinnedVRegs) {
    if (!LIS->hasInterval(V))
      continue;
    const TargetRegisterClass *RC = MRI->getRegClass(V);
    if (!inCurrentFile(RC))
      continue;
    unsigned W = TRI->getRegSizeInBits(*RC) / 32;
    for (const LiveRange::Segment &Seg : LIS->getInterval(V)) {
      Evs.push_back({Seg.start, +1, W});
      Evs.push_back({Seg.end, -1, W});
    }
  }
  llvm::sort(Evs, [](const Ev &A, const Ev &B) {
    if (A.S != B.S)
      return A.S < B.S;
    return A.D < B.D; // ends (-1) before starts (+1) at the same point
  });
  unsigned Cur = 0, Clique = 0;
  for (const Ev &E : Evs) {
    if (E.D > 0) {
      Cur += E.W;
      Clique = std::max(Clique, Cur);
    } else {
      Cur -= E.W;
    }
  }
  return Clique;
}

bool AMDGPUSSARegisterSpiller::processACLCalls(MachineFunction &MF) {
  // Per-call preserved-RP (ACL) pass for the current file (IsVGPRPass). For each
  // call C, the set of pinned vregs (crosses any call) live across C in this
  // file must fit the callee-saved capacity k_cs. Where it does not, spill the
  // excess by store-at-def + free-across-C, choosing the free point (KillIdx)
  // relative to C so the spilled value stops occupying a register at C. See
  // ACL_Pass_and_CallSite_Capacity.md "Part 1b".
  if (PinnedVRegs.empty())
    return false;

  // SSA repair + RP tracking machinery, same as processFunction sets up. This
  // pass runs before processFunction, so it must initialize them itself; both
  // are recreated per pass by design (they cache per-run state). beginPass()
  // (re)creates the emitter's SSA updater and selects the file.
  Emitter->beginPass(IsVGPRPass);
  RPTracker = std::make_unique<GCNUpwardRPTracker>(*LIS);
  PreservedLimit = IsVGPRPass ? VGPRPreservedCap : SGPRPreservedCap;

  const unsigned KCS = IsVGPRPass ? VGPRPreservedCap : SGPRPreservedCap;

  // Collect calls (regmask calls only — the preserved-RP constraint is a
  // real-call property; implicit clobbers are handled by the coloring IsFree
  // legality check, not here).
  //
  // Order matters and must be DOMINANCE order, not layout/slot order. The calls
  // are NOT independent: spillOneVMP() below recomputes liveness, so when the
  // loop reaches call C its liveAt/clean-across tests already reflect the spills
  // done for earlier calls. If a call dominated by C were processed before C,
  // we could shed excess at the dominated call that C's own spill would have
  // covered — over-spilling and mis-choosing the "clean across C" kill point.
  // A dominator-tree preorder never visits a dominated call before its
  // dominator (dominance is only partial — sibling-branch calls are
  // incomparable — but a dom-tree preorder is a valid linearization of it).
  // Raw SlotIndex order does NOT give this: indexes only order within a block.
  SmallVector<std::pair<SlotIndex, MachineInstr *>, 8> Calls;
  for (auto *Node : depth_first(DT->getRootNode()))
    for (MachineInstr &MI : *Node->getBlock())
      if (MI.isCall())
        Calls.push_back({LIS->getInstructionIndex(MI).getRegSlot(), &MI});

  bool Changed = false;

  auto Width = [&](Register V) {
    return TRI->getRegSizeInBits(*MRI->getRegClass(V)) / 32u;
  };
  auto RightFile = [&](Register V) {
    return inCurrentFile(MRI->getRegClass(V));
  };

  for (auto &[CS, CallMI] : Calls) {
    // Candidate = pinned vreg of this file, live across C.
    struct Cand {
      Register V;
      unsigned W;
      bool Clean;           // every use is dominated by C (no pre-call/sibling use)
      unsigned NextUseDist; // farthest-first ordering key
    };
    SmallVector<Cand, 16> Cands;
    unsigned CrossRP = 0, FloorRP = 0;

    // Values read AT the call (operands): unspillable for C.
    DenseSet<Register> UsedAtC;
    for (const MachineOperand &MO : CallMI->uses())
      if (MO.isReg() && MO.getReg().isVirtual())
        UsedAtC.insert(MO.getReg());

    for (Register V : PinnedVRegs) {
      if (!LIS->hasInterval(V) || !RightFile(V))
        continue;
      const LiveInterval &LI = LIS->getInterval(V);
      if (!LI.liveAt(CS))
        continue;
      unsigned W = Width(V);
      CrossRP += W;

      if (UsedAtC.contains(V)) {
        FloorRP += W; // read by the call itself -> cannot relieve C
        continue;
      }

      // "clean for C" = every use is dominated by C, i.e. strictly after C on
      // every path. Such a value has no pre-call or sibling-path register need,
      // so freeing it across C reloads only post-call uses. A use NOT dominated
      // by C (before C, or on a sibling branch) makes it "both-sides".
      // Dominance -- NOT SlotIndex order, which is literal layout order and only
      // meaningful within a single block.
      bool Clean = true;
      for (MachineInstr &U : MRI->use_nodbg_instructions(V))
        if (!DT->dominates(CallMI, &U)) {
          Clean = false;
          break;
        }
      Cands.push_back({V, W, Clean, 0});
    }

    if (CrossRP <= KCS)
      continue;

    unsigned Excess = CrossRP - KCS;
    if (FloorRP > KCS) {
      LLVM_DEBUG(dbgs() << "  [ACL] call@" << CS << " INFEASIBLE: floorRP="
                        << FloorRP << " > k_cs=" << KCS << " (unspillable "
                        << "values used at the call exceed capacity)\n");
      // No amount of spilling can relieve C; leave it and let later stages
      // report. Do not loop.
      continue;
    }

    // Order: clean first, then farthest-next-use (measured from C).
    for (Cand &Cd : Cands)
      Cd.NextUseDist = NU->getNextUseDistance(
          CallMI->getIterator(),
          VRegMaskPair(Cd.V, MRI->getMaxLaneMaskForVReg(Cd.V)));
    llvm::sort(Cands, [](const Cand &A, const Cand &B) {
      if (A.Clean != B.Clean)
        return A.Clean; // clean candidates first
      return A.NextUseDist > B.NextUseDist; // farthest next use first
    });

    LLVM_DEBUG(dbgs() << "  [ACL] call@" << CS << " crossRP=" << CrossRP
                      << " k_cs=" << KCS << " excess=" << Excess
                      << " floorRP=" << FloorRP << " cands=" << Cands.size()
                      << "\n");

    unsigned Shed = 0;
    for (const Cand &Cd : Cands) {
      if (Shed >= Excess)
        break;

      // Kill point relative to C. Killing at C's slot makes V dead exactly at C
      // (measured: liveAt(CS) 1->0), which frees it across C. But if V has a use
      // that DOMINATES C (a real pre-call use on the path to C), killing at C
      // would also free the register before that use. So place the kill at the
      // DEEPEST C-dominating use instead: V keeps its register up to that use,
      // is dead across C, and post-call uses (dominated by C) reload after C.
      // Uses that dominate C are totally ordered (they lie on one path to C), so
      // "deepest" is well defined. If no use dominates C (clean, or the only
      // non-post-call uses are on sibling paths), kill at C. Dominance -- never
      // SlotIndex order.
      MachineInstr *DeepestPre = nullptr;
      for (MachineInstr &U : MRI->use_nodbg_instructions(Cd.V)) {
        if (!DT->dominates(&U, CallMI))
          continue; // not a pre-call use on the path to C
        if (!DeepestPre || DT->dominates(DeepestPre, &U))
          DeepestPre = &U; // U is at least as deep as the current best
      }
      SlotIndex KillIdx =
          DeepestPre ? LIS->getInstructionIndex(*DeepestPre).getRegSlot() : CS;

      VRegMaskPair VMP(Cd.V, MRI->getMaxLaneMaskForVReg(Cd.V));
      // RPLimit for reload-hoist decisions: the current file's limit. At ACL
      // time (before processFunction runs for this file) it is still 0, matching
      // the pre-extraction behavior where the moved reload code read the unset
      // VGPRLimit/SGPRLimit members.
      Emitter->spillOneVMP(VMP, KillIdx, IsVGPRPass ? VGPRLimit : SGPRLimit);
      Shed += Cd.W;
      Changed = true;
    }
  }

  return Changed;
}

void AMDGPUSSARegisterSpiller::sortRegSetByNextUse(
    MachineBasicBlock &MBB, MachineBasicBlock::reverse_iterator I,
    VRegMaskPairSet &Active) {
  // Pre-compute next-use distances for all registers to avoid redundant calls
  // during sorting (sort makes O(n log n) comparisons, but we only need O(n)
  // distance calculations)
  DenseMap<VRegMaskPair, unsigned> DistanceMap;

  // Get the current instruction
  MachineInstr *MI = &(*I);

  // Rank spill candidates by next-use distance measured at MI's slot. The lanes
  // an instruction reads cannot be freed at that instruction — if spilled they
  // would be reloaded right before it — so they are subtracted from each
  // candidate, and only the remaining lanes (those live across MI) are ranked.
  // Working at lane granularity keeps a sub-register spillable when a sibling
  // lane of the same tuple is read here (e.g. %x.sub1 stays a candidate while
  // %x.sub0 is read by MI). This also subsumes the former early-clobber special
  // case: an early-clobber use is just a read here, and recede() already
  // accounts for early-clobber in the pressure metric. Defs never appear in
  // Active (SSA defs are whole-register and excluded earlier), so only reads
  // are subtracted.
  MachineBasicBlock::iterator MIIter = MI->getIterator();

  // Lanes read by MI, per virtual register.
  DenseMap<Register, LaneBitmask> UsedLanes;
  for (const MachineOperand &MO : MI->operands())
    if (MO.isReg() && MO.isUse() && MO.getReg().isVirtual())
      UsedLanes[MO.getReg()] |= VRegMaskPair(MO, TRI, MRI).getLaneMask();

  // Restrict each candidate to the lanes NOT read by MI; drop any fully
  // consumed here. Rebuild Active so downstream spilling operates on exactly
  // the spillable lanes.
  SmallVector<VRegMaskPair, 8> Candidates;
  for (const VRegMaskPair &VMP : Active) {
    LaneBitmask CandMask = VMP.getLaneMask();
    auto It = UsedLanes.find(VMP.getVReg());
    if (It != UsedLanes.end())
      CandMask &= ~It->second;
    if (CandMask.none())
      continue;
    Candidates.emplace_back(VMP.getVReg(), CandMask);
  }

  Active.clear();
  for (const VRegMaskPair &VMP : Candidates) {
    Active.insert(VMP);
    DistanceMap[VMP] = NU->getNextUseDistance(MIIter, VMP);
  }

  // Sort using pre-computed distances
  Active.sort([&](const VRegMaskPair &A, const VRegMaskPair &B) {
    unsigned DistA = DistanceMap[A];
    unsigned DistB = DistanceMap[B];

    // Primary sort: Shorter distance first (longer distance at back for
    // spilling)
    if (DistA != DistB)
      return DistA < DistB;

    // Tie-breaker: If distances are equal, prefer SMALLER register to spill
    // We pop from the back, so put LARGER registers first (smaller at back)
    // This ensures we spill exactly the amount needed, not more
    // Example: Need to free 2 VGPRs, both v64 and v128 have same distance
    //   → Put v128 first, v64 at back → pop v64 (2 VGPRs) instead of v128 (4
    //   VGPRs)
    unsigned SizeA = A.getLaneMask().getNumLanes();
    unsigned SizeB = B.getLaneMask().getNumLanes();
    return SizeA > SizeB; // Larger first, so smaller is at back for popping
  });

  LLVM_DEBUG({
    dbgs() << "sortRegSetByNextUse: Active set sorted at " << *MI;
    dbgs() << " (read lanes excluded, ranked by next use at MI)\n";

    for (const auto &VMP : Active) {
      Register VReg = VMP.getVReg();
      StringRef Name = MRI->getVRegName(VReg);
      if (!Name.empty())
        dbgs() << "  %" << Name;
      else
        dbgs() << "  " << printReg(VReg, TRI);
      dbgs() << " (mask " << PrintLaneMask(VMP.getLaneMask())
             << ") : " << DistanceMap[VMP] << "\n";
    }
  });
}

VRegMaskPairSet AMDGPUSSARegisterSpiller::getVMPsToSpill(
    MachineBasicBlock &MBB, MachineBasicBlock::reverse_iterator I,
    VRegMaskPairSet &Active, unsigned CurRP, unsigned RPLimit) {

  VRegMaskPairSet ToSpill;

  LLVM_DEBUG(dbgs() << "getVMPsToSpill(): CurRP=" << CurRP
                    << ", RPLimit=" << RPLimit << "\n");

  // Step 1: Calculate how much we need to spill
  if (CurRP <= RPLimit) {
    LLVM_DEBUG(
        dbgs() << "getVMPsToSpill(): No spilling needed (RP <= limit)\n");
    return ToSpill;
  }

  unsigned SizeToSpill = CurRP - RPLimit;
  LLVM_DEBUG(dbgs() << "getVMPsToSpill(): Need to spill " << SizeToSpill
                    << " 32-bit register units\n");

  // Step 2: Sort Active set by next-use distance (longest last)
  sortRegSetByNextUse(MBB, I, Active);

  // Step 2.5: Loop-aware candidate filtering
  // If spill point is inside a loop, we hoist it to the outermost preheader.
  // Only candidates whose def dominates the effective kill point are valid.
  MachineBasicBlock *EffectiveKillBB = Emitter->effectiveKillBB(&MBB);

  if (EffectiveKillBB != &MBB) {
    // Spill point was hoisted - filter candidates
    LLVM_DEBUG(dbgs() << "getVMPsToSpill(): Spill in loop, effective kill at "
                      << printMBBReference(*EffectiveKillBB) << "\n");

    // Rebuild Active with only valid candidates (preserving NUD order)
    SmallVector<VRegMaskPair> ValidCandidates;
    for (const auto &VMP : Active) {
      MachineInstr *DefMI = MRI->getVRegDef(VMP.getVReg());
      if (!DefMI)
        continue;

      MachineBasicBlock *DefBB = DefMI->getParent();
      // Def must dominate the effective (hoisted) kill point
      if (DT->dominates(DefBB, EffectiveKillBB)) {
        ValidCandidates.push_back(VMP);
        LLVM_DEBUG({
          StringRef Name = MRI->getVRegName(VMP.getVReg());
          dbgs() << "  Valid candidate: ";
          if (!Name.empty())
            dbgs() << "%" << Name;
          else
            dbgs() << printReg(VMP.getVReg(), TRI);
          dbgs() << " (def dominates effective kill)\n";
        });
      } else {
        LLVM_DEBUG({
          StringRef Name = MRI->getVRegName(VMP.getVReg());
          dbgs() << "  Filtered out: ";
          if (!Name.empty())
            dbgs() << "%" << Name;
          else
            dbgs() << printReg(VMP.getVReg(), TRI);
          dbgs() << " (def in loop, doesn't dominate effective kill)\n";
        });
      }
    }

    if (ValidCandidates.empty()) {
      LLVM_DEBUG(
          dbgs()
          << "getVMPsToSpill(): No valid candidates after loop filter!\n");
      // TODO: Fallback - pick best invalid candidate and use loop exit sinking
      return ToSpill;
    }

    // Rebuild Active from valid candidates (preserves NUD order)
    Active.clear();
    for (const auto &VMP : ValidCandidates)
      Active.insert(VMP);
  }

  // Step 3: Greedily select registers to spill from the back
  unsigned RemainingToSpill = SizeToSpill;

  LLVM_DEBUG(dbgs() << "getVMPsToSpill(): Need to reduce RP by "
                    << RemainingToSpill << " units\n");

  while (RemainingToSpill > 0 && !Active.empty()) {
    VRegMaskPair Candidate = Active.pop_back_val();
    unsigned CandidateSize = Candidate.getSizeInRegs(TRI);

    LLVM_DEBUG({
      Register VReg = Candidate.getVReg();
      StringRef Name = MRI->getVRegName(VReg);
      dbgs() << "getVMPsToSpill(): Considering candidate ";
      if (!Name.empty())
        dbgs() << "%" << Name;
      else
        dbgs() << printReg(VReg, TRI);
      dbgs() << " with mask " << PrintLaneMask(Candidate.getLaneMask())
             << " (size " << CandidateSize << ")\n";
    });

    // If this register is larger than what we need to spill, split it by
    // subregisters and only spill what's needed
    if (CandidateSize > RemainingToSpill) {
      LLVM_DEBUG(dbgs() << "getVMPsToSpill(): Candidate is too large ("
                        << CandidateSize << " > " << RemainingToSpill
                        << "), splitting by subregisters\n");

      // Get subregisters sorted by next-use distance (longest first)
      // Use same query position as sortRegSetByNextUse: after instruction
      // Handle end-of-block case to avoid dereferencing end() iterator
      MachineBasicBlock::iterator AfterCurrent = std::next(I.getReverse());

      SmallVector<VRegMaskPair> SortedSubregs;
      if (AfterCurrent == MBB.end()) {
        // At block end, use block-level query
        SortedSubregs = NU->getSortedSubregUses(MBB, Candidate);
      } else {
        SortedSubregs = NU->getSortedSubregUses(AfterCurrent, Candidate);
      }

      if (!SortedSubregs.empty()) {
        // Split by subregisters and spill only what's needed
        for (const auto &SubReg : SortedSubregs) {
          unsigned SubRegSize = SubReg.getSizeInRegs(TRI);
          if (SubRegSize <= RemainingToSpill) {
            ToSpill.insert(SubReg);
            RemainingToSpill -= SubRegSize;

            if (RemainingToSpill == 0)
              break;
          } else {
            // SubReg is still too wide. Decompose into 32-bit parts using
            // getRegSplitParts and take only as many parts as needed.
            // RemainingToSpill counts 32-bit pressure units; one unit occupies
            // multiple lane-mask bits (e.g. sreg_32 uses mask 0x03).
            const TargetRegisterClass *SubRC = SubReg.getRegClass(MRI, TRI);
            ArrayRef<int16_t> Parts = TRI->getRegSplitParts(SubRC, 4);
            for (int16_t PartSubRegIdx : Parts) {
              LaneBitmask PartMask = TRI->getSubRegIndexLaneMask(PartSubRegIdx);
              unsigned PartSize = TRI->getNumCoveredRegs(PartMask);
              ToSpill.insert(VRegMaskPair(SubReg.getVReg(), PartMask));
              RemainingToSpill -= std::min(PartSize, RemainingToSpill);
              if (RemainingToSpill == 0)
                break;
            }
            break;
          }
        }

        // Note: We don't need to insert remaining lanes back into Active.
        // RPTracker will provide the correct live set on the next instruction.
      } else {
        // Fallback: no subregister info available, spill the whole register
        LLVM_DEBUG(dbgs() << "getVMPsToSpill(): No subregister info, "
                             "spilling whole register\n");
        ToSpill.insert(Candidate);
        RemainingToSpill = 0;
      }
    } else {
      // This register fits within our spill budget
      ToSpill.insert(Candidate);
      RemainingToSpill -= CandidateSize;
    }
  }

  LLVM_DEBUG(dbgs() << "getVMPsToSpill(): Selected " << ToSpill.size()
                    << " VMP(s) for spilling:\n";
             dumpRegSet(ToSpill));

  return ToSpill;
}

void AMDGPUSSARegisterSpiller::insertVirtualSpillMarker(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator I, VRegMaskPair VMP) {
  // Avoid dropping a marker immediately after the actual spill store
  // for the same VReg/Lane mask.
  MachineInstr *PrevMI = nullptr;
  if (I == MBB.end()) {
    if (!MBB.empty()) {
      auto PrevIt = MBB.end();
      --PrevIt;
      PrevMI = &*PrevIt;
    }
  } else if (I != MBB.begin()) {
    auto PrevIt = I;
    --PrevIt;
    PrevMI = &*PrevIt;
  }

  if (!PrevMI || !isSpillInstr(PrevMI) ||
      !usesSpilledVMP(PrevMI, VMP, TRI, MRI)) {
    // Insert MIR-visible marker so tests can assert the virtual spill point.
    DebugLoc SpillDL = I == MBB.end() ? DebugLoc() : I->getDebugLoc();
    MachineInstr *MarkerMI =
        BuildMI(MBB, I, SpillDL, TII->get(AMDGPU::SI_VIRTUAL_SPILL_MARKER))
            .addImm(VMP.getVReg().virtRegIndex())
            .addImm(VMP.getLaneMask().getAsInteger());
    LIS->InsertMachineInstrInMaps(*MarkerMI);
  } else {
    LLVM_DEBUG(
        dbgs()
        << "Skipping virtual spill marker (adjacent real spill of same VMP)\n");
  }
}

bool AMDGPUSSARegisterSpiller::spillAndReload(
    MachineBasicBlock &MBB, MachineBasicBlock::reverse_iterator I,
    VRegMaskPairSet &Active, unsigned CurRP, unsigned RPLimit) {

  LLVM_DEBUG(dbgs() << "\n=== spillAndReload() at " << *I << "\n");
  LLVM_DEBUG(dbgs() << "CurRP=" << CurRP << ", RPLimit=" << RPLimit << "\n");

  // Step 1: Select which VRegMaskPairs to spill using Belady's algorithm
  VRegMaskPairSet ToSpill = getVMPsToSpill(MBB, I, Active, CurRP, RPLimit);

  if (ToSpill.empty()) {
    LLVM_DEBUG(dbgs() << "spillAndReload(): Nothing to spill\n");
    return false;
  }

  // TODO: this message is not correct. spillAndReload will spill as much as
  // CurRP - RPLimit, but here we print a total number of VMPs available for
  // spill.
  LLVM_DEBUG(dbgs() << "spillAndReload(): Will spill " << ToSpill.size()
                    << " VMP(s)\n");

  // Step 2: For each selected VMP, perform atomic spill+reload+SSA repair
  // IMPORTANT: Process one VMP at a time to keep MIR valid
  for (const auto &VMP : ToSpill) {
    // Step 2b: Set virtual "spill point" at the high-pressure point
    // This is where RP exceeded, but we don't prune the LiveInterval here.
    // The LiveInterval will be shrunk later by shrinkToUses() after all reloads
    // are placed.
    MachineBasicBlock::iterator SpillPos = I.getReverse();

    // Get the effective kill block (hoisted out of loops if needed)
    MachineBasicBlock *EffectiveKillBB = Emitter->effectiveKillBB(&MBB);

    SlotIndex KillIdx;
    MachineBasicBlock::iterator MarkerPos;
    MachineBasicBlock *MarkerBB = EffectiveKillBB;

    if (EffectiveKillBB != &MBB) {
      // Spill hoisted to preheader - kill at preheader end
      KillIdx = Indexes->getMBBEndIdx(EffectiveKillBB).getPrevSlot();
      MarkerPos = EffectiveKillBB->getFirstTerminator();
    } else if (SpillPos == MBB.end()) {
      KillIdx = Indexes->getMBBEndIdx(&MBB).getPrevSlot();
      MarkerPos = MBB.end();
    } else {
      KillIdx = Indexes->getInstructionIndex(*SpillPos).getRegSlot();
      MarkerPos = SpillPos;
    }

    // Virtual spill marker at effective kill point
    if (EnableVirtualSpillMarkers) {
      if (I->isPHI() && EffectiveKillBB == &MBB) {
        // PHI case only applies when not hoisted
        LLVM_DEBUG(
            dbgs() << "Virtual spill marker for PHI goes to predecessors\n");
        for (auto *Pred : MBB.predecessors()) {
          insertVirtualSpillMarker(*Pred, Pred->getFirstTerminator(), VMP);
        }
      } else {
        insertVirtualSpillMarker(*MarkerBB, MarkerPos, VMP);
      }
    }

    Emitter->spillOneVMP(VMP, KillIdx, RPLimit);
  }

  // If a wide value had only a STRICT SUBSET of its lanes spilled, its remnant
  // still occupies the wide (aligned-tuple) vreg. Narrow it so the tuple frees.
  if (EnableNarrowRemnant)
    narrowSpilledRemnants(ToSpill);

  LLVM_DEBUG(dbgs() << "spillAndReload(): Completed, spilled " << ToSpill.size()
                    << " VMP(s)\n");
  LLVM_DEBUG(dbgs() << "===================================\n\n");

  return true;
}

bool AMDGPUSSARegisterSpiller::narrowSpilledRemnants(
    const VRegMaskPairSet &Spilled) {
  // Aggregate the spilled lane mask per wide vreg.
  DenseMap<Register, LaneBitmask> SpilledMask;
  for (const auto &VMP : Spilled)
    SpilledMask[VMP.getVReg()] |= VMP.getLaneMask();

  bool Changed = false;
  for (auto &[VReg, Mask] : SpilledMask) {
    if (!VReg.isVirtual() || !LIS->hasInterval(VReg))
      continue;
    const TargetRegisterClass *RC = MRI->getRegClass(VReg);
    if (!inCurrentFile(RC))
      continue;
    LaneBitmask Full = MRI->getMaxLaneMaskForVReg(VReg);
    LaneBitmask Remnant = Full & ~Mask;
    // Only act on a PARTIAL spill (some lanes spilled, a remnant survives) of a
    // genuinely wide value.
    if (Remnant.none() || Remnant == Full)
      continue;
    if (TRI->getRegSizeInBits(*RC) <= 32)
      continue;

    // Only narrow when the remnant is a SINGLE 32-bit lane. That is the case that
    // actually relieves aligned-tuple pressure: a lone surviving lane still pins
    // the wide value's aligned slot, blocking a wider value that needs it (the
    // %123.sub3 case). A still-wide remnant (e.g. 6 of 8 dwords) gains nothing
    // from being copied to another equally-wide vreg — worse, it just relocates
    // the aligned-tuple burden and can itself become uncolorable (regressed
    // schedule-amdgpu-tracker-physreg-crash: a VReg_192 remnant copied to a fresh
    // VReg_192 that then failed coloring). So restrict to width-1 remnants.
    if (TRI->getNumCoveredRegs(Remnant) != 1)
      continue;

    // The remnant must be a single contiguous sub-register we can name and copy.
    unsigned SubIdx = TRI->getSubRegIndexForLaneMask(Remnant);
    if (!SubIdx)
      continue;

    // Delegate the actual COPY + SSA-correct use rewrite to the emitter, which
    // owns the MachineLaneSSAUpdater. It inserts `%new = COPY VReg.SubIdx` at the
    // def and calls rewriteDominatedUses(VReg, %new, Remnant) — dominance- and
    // subreg-policy-aware, composing REG_SEQUENCEs where a use spans the split.
    if (Emitter->narrowRemnantToNewReg(VReg, SubIdx, Remnant))
      Changed = true;
  }
  return Changed;
}

void AMDGPUSSARegisterSpiller::dumpRegSet(const VRegMaskPairSet &Regs) const {
  for (const auto &VMP : Regs) {
    Register VReg = VMP.getVReg();
    dbgs() << "  ";

    // Print original name if available (e.g., %large), otherwise print number
    StringRef Name = MRI->getVRegName(VReg);
    if (!Name.empty())
      dbgs() << "%" << Name;
    else
      dbgs() << printReg(VReg, TRI);

    dbgs() << " (mask " << PrintLaneMask(VMP.getLaneMask()) << ", size "
           << VMP.getSizeInRegs(TRI) << ")\n";
  }
}
unsigned AMDGPUSSARegisterSpiller::countSGPRSpillVGPRs(MachineFunction &MF) {
  SIMachineFunctionInfo *FuncInfo = MF.getInfo<SIMachineFunctionInfo>();
  if (!FuncInfo->hasSpilledSGPRs())
    return 0;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  MachineFrameInfo &LocalMFI = MF.getFrameInfo();
  const unsigned WaveSize = ST.getWavefrontSize();

  // Lane VGPRs are packed WaveSize 32-bit slots per VGPR, accumulated across
  // all distinct SGPR-spill frame indices (same packing as
  // allocateSGPRSpillToVGPRLane). Count slots from frame-object sizes only —
  // no physreg, no SuperReg, no side effects.
  DenseSet<int> SeenFIs;
  unsigned TotalLanes = 0;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!TII->isSGPRSpill(MI))
        continue;
      const MachineOperand *Addr =
          TII->getNamedOperand(MI, AMDGPU::OpName::addr);
      if (!Addr || !Addr->isFI())
        continue;
      int FI = Addr->getIndex();
      if (LocalMFI.getStackID(FI) != TargetStackID::SGPRSpill)
        continue;
      if (!SeenFIs.insert(FI).second)
        continue;
      TotalLanes += LocalMFI.getObjectSize(FI) / 4;
    }
  }

  unsigned NumSpillVGPRs = (TotalLanes + WaveSize - 1) / WaveSize;
  LLVM_DEBUG(dbgs() << "countSGPRSpillVGPRs(): " << TotalLanes
                    << " lane slot(s) -> " << NumSpillVGPRs << " VGPR(s)\n");
  return NumSpillVGPRs;
}

bool AMDGPUSSARegisterSpiller::runOnMachineFunction(MachineFunction &MF) {
  // Initialize pass dependencies
  TRI =
      static_cast<const SIRegisterInfo *>(MF.getSubtarget().getRegisterInfo());
  TII = static_cast<const SIInstrInfo *>(MF.getSubtarget().getInstrInfo());
  MRI = &MF.getRegInfo();
  MFI = MF.getInfo<SIMachineFunctionInfo>();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  Indexes = &getAnalysis<SlotIndexesWrapperPass>().getSI();
  DT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();

  // Get Next Use Analysis result
  NU = &getAnalysis<AMDGPUNextUseAnalysisWrapper>().getNU();

  // Fresh spill/reload emitter per function. It owns the per-function state
  // (stack slots, store-at-def memo) that was previously cleared here: a fresh
  // instance starts empty, so no stale entry from a prior function can alias a
  // different value (virtual register numbers restart per function).
  Emitter = std::make_unique<SSASpillEmitter>(MF, LIS, Indexes, DT, MLI);

  LLVM_DEBUG(dbgs() << "AMDGPUSSARegisterSpiller: Processing function "
                    << MF.getName() << "\n");

  // Calculate register pressure limits based on subtarget and function
  // requirements These limits are determined by:
  // - Target architecture capabilities
  // - Desired occupancy (waves per execution unit)
  // - Function-specific requirements (e.g., flat scratch, dynamic stack)
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();

  // Get the maximum number of registers for this function
  // These methods account for:
  // - Hardware limits
  // - Addressable register limits
  // - Occupancy requirements
  // - Dynamic VGPR block size (for unified register file architectures)
  unsigned VGPRLimit = ST.getMaxNumVGPRs(MF);
  unsigned SGPRLimit = ST.getMaxNumSGPRs(MF);

  // Cap each budget by the number of registers the *allocator* can actually
  // hand out for that class -- the size of its allocatable set (getOrder in
  // color()). getMaxNumVGPRs returns the occupancy/addressable target over the
  // whole vector register budget, which on split-file targets (gfx90a: separate
  // VGPR and AGPR files) exceeds the VGPR_32 file the allocator draws from: a
  // VGPR value cannot be colored to an AGPR. If the spiller trusts the larger
  // number it under-spills, leaving true pressure above what color() can place,
  // and color() then aborts ("Failed to find free physreg"). Taking the min
  // keeps the spiller's target consistent with the allocator's real capacity.
  VGPRLimit = std::min(
      VGPRLimit, TRI->getAllocatableSet(MF, &AMDGPU::VGPR_32RegClass).count());
  SGPRLimit = std::min(
      SGPRLimit, TRI->getAllocatableSet(MF, &AMDGPU::SGPR_32RegClass).count());

  // Reserve a ~10% safety margin (RA temporaries, compiler temporaries, ABI
  // reserved registers). Subtract floor(10%) rather than computing (N*9)/10:
  // the latter truncates the *limit* down, which rounds the *margin up* and
  // cuts 25-50% from small budgets (e.g. N=3 -> 2). Subtracting N/10 makes the
  // margin a true floor(10%) — 0 for budgets < 10, ~10% for large files — so a
  // small but satisfiable budget is targeted exactly instead of infeasibly.
  VGPRLimit -= VGPRLimit / 10;
  SGPRLimit -= SGPRLimit / 10;

  LLVM_DEBUG(dbgs() << "Register pressure limits (90% of max): VGPR="
                    << VGPRLimit << ", SGPR=" << SGPRLimit << "\n");
  LLVM_DEBUG(dbgs() << "  (Architecture max: VGPR=" << ST.getMaxNumVGPRs(MF)
                    << ", SGPR=" << ST.getMaxNumSGPRs(MF) << ")\n");

  // Classify pinned (crosses-a-call) vregs and compute per-file callee-saved
  // capacity k_cs. Only the ACL passes consume this, so skip it entirely when
  // ACL is off — the spiller then behaves exactly as the original two total-RP
  // passes with no ACL-related work.
  if (EnableAMDGPUSSAACLColoring)
    computePinnedAndCap(MF);

  // Two-pass approach:
  // Pass 1: Process SGPRs (spilled to VGPR lanes if needed)
  // Pass 2: Process VGPRs (spilled to memory)

  // Four-pass structure when ACL is enabled (mirrors the coloring side, which
  // colors ACL then ordinary):
  //   1. ACL_SGPR   preserved-RP, SGPR   (processACLCalls)  -> spills to lanes
  //   2. main_SGPR  total-RP, SGPR       (processFunction)  -> more lanes
  //      -- count SGPR-spill lanes; debit VGPR total (and preserved) budget --
  //   3. ACL_VGPR   preserved-RP, VGPR   (processACLCalls)
  //   4. main_VGPR  total-RP, VGPR       (processFunction)
  // When ACL is off, this reduces to the original two total-RP passes.

  // Pass 1: ACL_SGPR — spill SGPR around-call-livers to fit callee-saved
  // capacity, before ordinary SGPR spilling sees them.
  bool ChangedSGPR = false;
  if (EnableAMDGPUSSAACLColoring) {
    IsVGPRPass = false;
    ChangedSGPR |= processACLCalls(MF);
  }

  // Pass 2: main_SGPR — ordinary total-RP SGPR spilling.
  LLVM_DEBUG(dbgs() << "\n=== Pass 2: Processing SGPRs (total-RP) ===\n");
  IsVGPRPass = false;
  ChangedSGPR |= processFunction(MF, SGPRLimit);

  // Account for VGPRs consumed by SGPR-spill-to-lane and shrink the VGPR
  // budget for the VGPR passes. Actual lowering happens later (at SGPR
  // coloring). SGPR spills materialize as WWM VGPR lanes (added later by
  // SILowerSGPRSpills), which need physical VGPRs on top of the per-thread
  // allocation. Reserve them only when we actually spilled SGPRs, and credit
  // the proportional margin that is already held back: if the margin
  // (getMaxNumVGPRs/10) already covers the lanes, no extra reservation;
  // otherwise reserve only the shortfall.
  if (ChangedSGPR) {
    unsigned SpillVGPRsUsed = countSGPRSpillVGPRs(MF);
    unsigned MarginReserved = ST.getMaxNumVGPRs(MF) / 10;
    unsigned ExtraReserve =
        SpillVGPRsUsed > MarginReserved ? SpillVGPRsUsed - MarginReserved : 0;
    assert(ExtraReserve <= VGPRLimit && "SGPR spill lanes exceed VGPR budget");
    VGPRLimit -= ExtraReserve;
  }

  // Recompute the pinned set / caps for the VGPR file: the SGPR passes created
  // reload vregs and (via SGPR->lane spills) new VGPR around-call-livers, none
  // of which the initial computePinnedAndCap saw. ACL_VGPR needs the fresh set.
  // NOTE: debiting VGPRPreservedCap by the call-crossing spill lanes is a
  // separate follow-up (see ACL_Pass_and_CallSite_Capacity.md Part 1b) — this
  // recompute picks up reload-vreg pins but not the not-yet-materialized lanes.
  if (EnableAMDGPUSSAACLColoring)
    computePinnedAndCap(MF);

  // Pass 3: ACL_VGPR — spill VGPR around-call-livers to fit callee-saved
  // capacity, before ordinary VGPR spilling.
  bool ChangedVGPR = false;
  if (EnableAMDGPUSSAACLColoring) {
    IsVGPRPass = true;
    ChangedVGPR |= processACLCalls(MF);
  }

  // Pass 4: main_VGPR — ordinary total-RP VGPR spilling.
  LLVM_DEBUG(dbgs() << "\n=== Pass 4: Processing VGPRs (total-RP) ===\n");
  IsVGPRPass = true;
  ChangedVGPR |= processFunction(MF, VGPRLimit);

  LLVM_DEBUG(dbgs() << "\nAMDGPUSSARegisterSpiller: Completed processing "
                    << MF.getName() << "\n");

  // Dump final LiveIntervals state for testing/verification
  LLVM_DEBUG({
    dbgs() << "\n********** FINAL LIVE INTERVALS **********\n";
    LIS->print(dbgs());
  });

  // Reloads redefine OrigVReg, breaking SSA transiently; inline reconstruction
  // (emitReloadsAndRepairSSA) restores it and clears the flag, so this normally
  // does not fire. Kept as a defensive net: if any reload path ever left SSA
  // unrepaired, clearing IsSSA makes the (SSA-requiring) allocator fail loudly
  // rather than silently consuming non-SSA MIR.
  if (Emitter->ssaInvalidated())
    MF.getProperties().reset(MachineFunctionProperties::Property::IsSSA);

  // Return true if either pass made modifications
  return ChangedSGPR || ChangedVGPR;
}

// Create function for pass manager
MachineFunctionPass *llvm::createAMDGPUSSARegisterSpillerPass() {
  return new AMDGPUSSARegisterSpiller();
}
