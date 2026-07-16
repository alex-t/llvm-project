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

STATISTIC(NumSpills, "Number of register spills");
STATISTIC(NumReloads, "Number of register reloads");

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

// Helper function to identify spill instructions
// ============================================================================

// A spill store emitted by storeRegToStackSlot. Classify via the Spill TSFlag
// plus mayStore rather than enumerating opcodes: this covers every register
// file (SGPR, VGPR, AGPR, and the AGPR-or-VGPR "AV" classes used on gfx90a+)
// and every register width without an opcode list that silently omits some.
static bool isSpillInstr(const MachineInstr *MI) {
  return SIInstrInfo::isSpill(MI->getDesc()) && MI->mayStore();
}

// A reload emitted by loadRegFromStackSlot. See isSpillInstr: classify via the
// Spill TSFlag plus mayLoad so AGPR and AGPR-or-VGPR ("AV") reloads are
// recognized too. The prior opcode list omitted the A/AV variants, so on
// gfx90a+ an AV reload redef was never renamed during SSA repair, leaving the
// spilled vreg multi-defined and tripping getVRegDef on the next spill.
static bool isReloadInstr(const MachineInstr *MI) {
  return SIInstrInfo::isSpill(MI->getDesc()) && MI->mayLoad();
}

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

int AMDGPUSSARegisterSpiller::assignVirt2StackSlot(VRegMaskPair VMP) {
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

int AMDGPUSSARegisterSpiller::createSpillSlot(const TargetRegisterClass *RC) {
  unsigned SpillSize = TRI->getSpillSize(*RC);
  Align SpillAlign = TRI->getSpillAlign(*RC);
  return FrameInfo->CreateSpillStackObject(SpillSize, SpillAlign);
}

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
  GCNRegPressure::RegKind Kind =
      IsVGPRPass ? GCNRegPressure::VGPR : GCNRegPressure::SGPR;
  SlotIndex Slot = LIS->getInstructionIndex(MI).getRegSlot();
  GCNRPTracker::LiveRegSet Live = llvm::getLiveRegs(Slot, *LIS, *MRI, Kind);
  unsigned N = 0, NAll = 0;
  for (const auto &[Reg, Mask] : Live) {
    Register VReg(Reg);
    unsigned W = VRegMaskPair(VReg, Mask).getSizeInRegs(TRI);
    NAll += W;
    if (PinnedVRegs.contains(VReg))
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

  // Initialize SSA updater (reused throughout the pass, caches IDF
  // computations)
  // FIXME: Clear cache if CFG changes during spilling
  SSAUpdater = std::make_unique<MachineLaneSSAUpdater>(MF, *LIS, *DT, *TRI);

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

  ReloadedRegs.clear();

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
      if (RC && !MRI->isReserved(LI.PhysReg) &&
          (IsVGPRPass ? TRI->isVGPRClass(RC) : TRI->isSGPRClass(RC)))
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

        // Determine the register kind for filtering
        GCNRegPressure::RegKind Kind =
            IsVGPRPass ? GCNRegPressure::VGPR : GCNRegPressure::SGPR;

        // Get the slot index for the current instruction
        SlotIndex Slot = LIS->getInstructionIndex(MI).getRegSlot();

        // Get live registers filtered by register kind using getLiveRegs helper
        GCNRPTracker::LiveRegSet LiveRegsMap =
            llvm::getLiveRegs(Slot, *LIS, *MRI, Kind);

        // Convert to VRegMaskPairSet for spill candidate selection
        VRegMaskPairSet ActiveRegs = convertLiveRegs(LiveRegsMap);

        LLVM_DEBUG(dbgs() << "ActiveRegs: " << printVRegMaskPairSet(ActiveRegs)
                          << "\n");
        LLVM_DEBUG(dbgs() << "ReloadedRegs: "
                          << printVRegMaskPairSet(ReloadedRegs) << "\n");
        ActiveRegs.set_subtract(ReloadedRegs);
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
    bool RightFile = IsVGPRPass ? (TRI->isVGPRClass(RC) || TRI->isAGPRClass(RC))
                                : TRI->isSGPRClass(RC);
    if (!RightFile)
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
  // are recreated per pass by design (they cache per-run state).
  SSAUpdater = std::make_unique<MachineLaneSSAUpdater>(MF, *LIS, *DT, *TRI);
  RPTracker = std::make_unique<GCNUpwardRPTracker>(*LIS);
  PreservedLimit = IsVGPRPass ? VGPRPreservedCap : SGPRPreservedCap;

  const unsigned KCS = IsVGPRPass ? VGPRPreservedCap : SGPRPreservedCap;

  // Collect call slots in program order (regmask calls only — the preserved-RP
  // constraint is a real-call property; implicit clobbers are handled by the
  // coloring IsFree legality check, not here).
  SmallVector<std::pair<SlotIndex, MachineInstr *>, 8> Calls;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (MI.isCall())
        Calls.push_back({LIS->getInstructionIndex(MI).getRegSlot(), &MI});
  llvm::sort(Calls, [](const auto &A, const auto &B) { return A.first < B.first; });

  bool Changed = false;

  auto Width = [&](Register V) {
    return TRI->getRegSizeInBits(*MRI->getRegClass(V)) / 32u;
  };
  auto RightFile = [&](Register V) {
    const TargetRegisterClass *RC = MRI->getRegClass(V);
    return IsVGPRPass ? (TRI->isVGPRClass(RC) || TRI->isAGPRClass(RC))
                      : TRI->isSGPRClass(RC);
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
      spillOneVMP(VMP, KillIdx);
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
  MachineBasicBlock *EffectiveKillBB = getEffectiveKillBB(&MBB);

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

  if (!PrevMI || !isSpillInstr(PrevMI) || !usesSpilledVMP(PrevMI, VMP)) {
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
    MachineBasicBlock *EffectiveKillBB = getEffectiveKillBB(&MBB);

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

    spillOneVMP(VMP, KillIdx);
  }

  LLVM_DEBUG(dbgs() << "spillAndReload(): Completed, spilled " << ToSpill.size()
                    << " VMP(s)\n");
  LLVM_DEBUG(dbgs() << "===================================\n\n");

  return true;
}

void AMDGPUSSARegisterSpiller::spillOneVMP(VRegMaskPair VMP, SlotIndex KillIdx) {
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

// ============================================================================
// Reload Optimizer
// ============================================================================

// TODO: Investigate profitability/possibility of early return when RP > Limit.
// Callers only care whether RP exceeds the limit, not by how much.
// Optimization: if we find RP > Limit at any point, return early and cache
// that value - no need to compute the actual maximum.
unsigned AMDGPUSSARegisterSpiller::getMaxRPForBlock(MachineBasicBlock *MBB) {
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

unsigned AMDGPUSSARegisterSpiller::getMaxRPInBlockDownTo(MachineBasicBlock *MBB,
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

bool AMDGPUSSARegisterSpiller::canHoistReloadTo(MachineBasicBlock *NCD,
                                                MachineInstr *InsertPoint,
                                                unsigned RPLimit,
                                                Register SpilledReg) {
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
AMDGPUSSARegisterSpiller::getEffectiveKillBB(MachineBasicBlock *SpillBB) const {
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
AMDGPUSSARegisterSpiller::adjustReloadForLoop(MachineBasicBlock *ReloadBB,
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
      unsigned RPLimit = IsVGPRPass ? VGPRLimit : SGPRLimit;
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

// ===========================================================================
// IDF-First PHI Insertion Strategy
// ===========================================================================

void AMDGPUSSARegisterSpiller::buildDomGroupsForSpill(SpillInfo &Info) {
  Register SpilledReg = Info.SpilledVMP.getVReg();
  MachineInstr *KillMI = Indexes->getInstructionFromIndex(Info.KillIdx);

  LLVM_DEBUG(dbgs() << "buildDomGroupsForSpill for "
                    << printReg(SpilledReg, TRI) << "\n");

  // Collect all uses into a vector for sorting
  SmallVector<MachineInstr *, 8> AllUses;
  for (MachineInstr &UseMI : MRI->use_nodbg_instructions(SpilledReg)) {
    if (isSpillInstr(&UseMI) || UseMI.isPHI())
      continue;

    MachineOperand *UseOp =
        UseMI.findRegisterUseOperand(SpilledReg, TRI, /*isKill=*/false);
    if (!UseOp)
      continue;

    VRegMaskPair UseVMP(*UseOp, TRI, MRI);
    if (!UseVMP.overlaps(Info.SpilledVMP))
      continue;

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
AMDGPUSSARegisterSpiller::getOrCreateReloadInBlock(MachineBasicBlock *BB,
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

bool AMDGPUSSARegisterSpiller::insertReloadForUse(MachineInstr *UseMI,
                                                  VRegMaskPair SpilledVMP,
                                                  MachineBasicBlock *KillBB) {
  Register SpilledReg = SpilledVMP.getVReg();
  LaneBitmask SpilledMask = SpilledVMP.getLaneMask();
  unsigned RPLimit = IsVGPRPass ? VGPRLimit : SGPRLimit;

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

void AMDGPUSSARegisterSpiller::finalizeLiveIntervals(Register SpilledReg) {
  // Reload intervals may already exist if inline SSA repair created them via
  // LIS.getInterval() auto-creation. Only create intervals for reloads that
  // weren't processed through that path.
  for (auto &Entry : BlockReloadCache) {
    Register ReloadReg = Entry.second;
    if (!LIS->hasInterval(ReloadReg))
      LIS->createAndComputeVirtRegInterval(ReloadReg);
  }

  // SpilledReg always has an interval - recompute it to reflect rewritten uses
  LIS->removeInterval(SpilledReg);
  LIS->createAndComputeVirtRegInterval(SpilledReg);
}

void AMDGPUSSARegisterSpiller::emitReloadsAndRepairSSA(SpillInfo &Info) {
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
        if (DMI->getParent() == B)
          return false;
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
    if (!usesSpilledVMP(U, SpilledVMP))
      continue;
    if (NeedsReload(U))
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
// Primary spill/reload methods (forward iterators) - used for SSA repair
// ============================================================================

MachineInstr *AMDGPUSSARegisterSpiller::spillAtDefinition(VRegMaskPair VMP) {
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

void AMDGPUSSARegisterSpiller::spillBefore(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertBefore,
    VRegMaskPair VMP) {
  Register VReg = VMP.getVReg();
  LaneBitmask Mask = VMP.getLaneMask();

  // Check if this register was already stored at definition
  if (StoredAtDefinition.count(VMP)) {
    LLVM_DEBUG({
      StringRef Name = MRI->getVRegName(VReg);
      dbgs() << "spillBefore(): Register ";
      if (!Name.empty())
        dbgs() << "%" << Name;
      else
        dbgs() << printReg(VReg, TRI);
      dbgs() << " already stored at definition, marking dead at real spill "
                "point\n";
    });

    // Get the SlotIndex for the "real spill" point (right before InsertBefore)
    SlotIndex KillIdx;
    if (InsertBefore == MBB.begin()) {
      // InsertBefore is at block start, use block start index
      KillIdx = Indexes->getMBBStartIdx(&MBB);
    } else {
      // InsertBefore points to an instruction, mark dead at its base index
      // (right before it executes)
      KillIdx = Indexes->getInstructionIndex(*InsertBefore).getBaseIndex();
    }

    // Prune the LiveInterval to mark register dead at this point
    if (LIS->hasInterval(VReg)) {
      LiveInterval &LI = LIS->getInterval(VReg);

      // Prune the main live range (cast LiveInterval to LiveRange*)
      LIS->pruneValue(*static_cast<LiveRange *>(&LI), KillIdx, nullptr);

      // Prune all subranges as well
      for (LiveInterval::SubRange &SR : LI.subranges()) {
        LIS->pruneValue(SR, KillIdx, nullptr);
      }

      // TEMPORARY DEBUG: Dump %0's LiveInterval AFTER pruning and compare with
      // %2
      LLVM_DEBUG({
        if (InsertBefore != MBB.begin()) {
          dbgs() << "spillBefore(): %0 LiveInterval AFTER pruning: " << LI
                 << "\n";

          // Find %2 again and check if %0 dies before %2 becomes live
          MachineInstr *DefMI = &*InsertBefore;
          Register VReg2;
          for (const MachineOperand &MO : DefMI->operands()) {
            if (MO.isReg() && MO.isDef() && MO.getReg().isVirtual()) {
              VReg2 = MO.getReg();
              break;
            }
          }

          if (VReg2.isValid() && LIS->hasInterval(VReg2)) {
            LiveInterval &LI2 = LIS->getInterval(VReg2);
            SlotIndex Def2Idx =
                Indexes->getInstructionIndex(*DefMI).getRegSlot();

            // Check if %2 is live at KillIdx (where %0 dies)
            LiveQueryResult LRQ2 = LI2.Query(KillIdx);
            bool VReg2LiveAtKill = (LRQ2.valueIn() != nullptr);

            dbgs() << "spillBefore(): %2 defined at: " << Def2Idx << "\n";
            dbgs() << "spillBefore(): %0 dies at: " << KillIdx << "\n";
            dbgs() << "spillBefore(): %2 live at KillIdx (" << KillIdx
                   << "): " << (VReg2LiveAtKill ? "YES" : "NO") << "\n";

            if (VReg2LiveAtKill && Def2Idx >= KillIdx) {
              dbgs() << "spillBefore(): WARNING: %0 dies BEFORE %2 becomes "
                        "live!\n";
            } else {
              dbgs() << "spillBefore(): OK: %0 dies after %2 becomes live or "
                        "%2 not live yet\n";
            }
          }
        }
      });
    }

    LLVM_DEBUG(dbgs() << "spillBefore(): Pruned LiveInterval at " << KillIdx
                      << "\n");
    return;
  }

  // Original behavior: emit spill instruction
  LLVM_DEBUG({
    StringRef Name = MRI->getVRegName(VReg);
    dbgs() << "spillBefore(): Emitting spill for ";
    if (!Name.empty())
      dbgs() << "%" << Name;
    else
      dbgs() << printReg(VReg, TRI);
    dbgs() << " with mask " << PrintLaneMask(Mask) << "\n";
  });

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
      dbgs() << "spillBefore(): Spilling subregister "
             << TRI->getSubRegIndexName(SubRegIdx) << " of ";
      if (!Name.empty())
        dbgs() << "%" << Name;
      else
        dbgs() << printReg(VReg, TRI);
      dbgs() << "\n";
    }
  });

  // Emit the spill instruction
  bool IsKill = (SubRegIdx == AMDGPU::NoRegister);
  TII->storeRegToStackSlot(MBB, InsertBefore, VReg, IsKill, FI, RC, TRI, VReg,
                           MachineInstr::NoFlags, SubRegIdx);

  // Get the inserted spill instruction (it's right before InsertBefore)
  MachineInstr &SpillMI = *std::prev(InsertBefore);

  // Update LiveIntervals
  LIS->InsertMachineInstrInMaps(SpillMI);

  // NOTE: We do NOT call shrinkToUses here because the original uses still
  // exist. After emitReloadsAndRepairSSA() rewrites uses to point to reloaded
  // registers, we'll shrink the LiveInterval there to reflect the reduced
  // liveness.

  LLVM_DEBUG(dbgs() << "spillBefore(): Emitted: " << SpillMI);
  ++NumSpills;
}

// ============================================================================
// Backward traversal helper (reverse iterator) - used during spilling phase
// ============================================================================

void AMDGPUSSARegisterSpiller::spillBefore(
    MachineBasicBlock &MBB, MachineBasicBlock::reverse_iterator I,
    VRegMaskPair VMP) {
  // Convert reverse iterator to forward iterator
  // The reverse iterator I is set up so that *I is the instruction MI where
  // we detected high pressure. We want to insert the spill BEFORE MI.
  // Simply get the forward iterator directly from the instruction.
  MachineInstr *MI = &(*I);
  MachineBasicBlock::iterator InsertPt = MI->getIterator();

  // If we're at a terminator, insert before the first terminator
  // to keep the block valid (terminators must be at the end)
  if (InsertPt != MBB.end() && InsertPt->isTerminator()) {
    InsertPt = MBB.getFirstTerminator();
  }

  spillBefore(MBB, InsertPt, VMP);
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

// ============================================================================
// Divergent Path Optimization Helpers
// ============================================================================

bool AMDGPUSSARegisterSpiller::usesSpilledVMP(const MachineInstr *MI,
                                              VRegMaskPair SpilledVMP) const {
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

bool AMDGPUSSARegisterSpiller::walkPathsToUses(
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
  FrameInfo = &MF.getFrameInfo();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  Indexes = &getAnalysis<SlotIndexesWrapperPass>().getSI();
  DT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();

  // Get Next Use Analysis result
  NU = &getAnalysis<AMDGPUNextUseAnalysisWrapper>().getNU();

  SSAInvalidated = false;

  // Stack slots and the store-at-definition memo are per-function state keyed by
  // VRegMaskPair. Virtual register numbers restart in every function, so a stale
  // entry from a previously-processed function would alias a different value:
  // Virt2StackSlotMap would return a frame index that is out of range for this
  // function's MachineFrameInfo (getObjectAlign "Invalid Object Idx"), and
  // StoredAtDefinition would hand back a dangling store from the prior function.
  Virt2StackSlotMap.clear();
  StoredAtDefinition.clear();

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
  LLVM_DEBUG(dbgs() << "Total spills: " << NumSpills
                    << ", Total reloads: " << NumReloads << "\n");

  // Dump final LiveIntervals state for testing/verification
  LLVM_DEBUG({
    dbgs() << "\n********** FINAL LIVE INTERVALS **********\n";
    LIS->print(dbgs());
  });

  // Reloads redefine OrigVReg, breaking SSA transiently; inline reconstruction
  // (emitReloadsAndRepairSSA) restores it and clears SSAInvalidated, so this
  // normally does not fire. Kept as a defensive net: if any reload path ever
  // left SSA unrepaired, clearing IsSSA makes the (SSA-requiring) allocator
  // fail loudly rather than silently consuming non-SSA MIR.
  if (SSAInvalidated)
    MF.getProperties().reset(MachineFunctionProperties::Property::IsSSA);

  // Return true if either pass made modifications
  return ChangedSGPR || ChangedVGPR;
}

// Create function for pass manager
MachineFunctionPass *llvm::createAMDGPUSSARegisterSpillerPass() {
  return new AMDGPUSSARegisterSpiller();
}
