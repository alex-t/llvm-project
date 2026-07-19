//===-- AMDGPUVerifyPhysRegLiveness.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Independent post-allocation physical-register liveness validator.
///
/// *** WIP / NOT YET CALIBRATED (2026-07-19). ***
/// The dataflow is implemented and the pass runs (behind the default-off flag
/// -amdgpu-verify-physreg-liveness), BUT it currently FALSE-POSITIVES on
/// known-good code (e.g. add.ll add64_in_branch reports 6 bogus "clobbers" on
/// block-live-in SGPRs defined in a dominating block). Do NOT trust its output
/// as a correctness verdict until the cross-block defined-set propagation /
/// block-liveins handling is fixed and it is calibrated: (1) silent on a set of
/// known-good tests, (2) FIRES on a known clobber (e.g. the reverted stall-guard
/// miscompile). Committed WIP for a rollback point and so the correctness-gate
/// infrastructure is not lost. See [[project note: SSARA correctness gate]].
///
/// Catches the CLOBBER class of miscompile that the MachineVerifier cannot: it
/// recomputes per-physreg-UNIT reaching-definition state FROM THE INSTRUCTION
/// STREAM ALONE (ignoring LiveIntervals / the allocator's own bookkeeping) and
/// reports a use of an allocatable unit that is not defined on every path
/// reaching it — i.e. a value read out of a register that was overwritten (or
/// never defined). The MachineVerifier trusts the allocation's liveness, so a
/// structurally-valid-but-wrong assignment (register reused while live) passes
/// it; this pass re-derives truth from the emitted instructions and flags it.
///
/// Run after the SSA register allocator, e.g.:
///   llc ... -amdgpu-ssa-regalloc -amdgpu-verify-physreg-liveness ...
///
/// Model: forward per-block scan tracking a BitVector of "defined" reg units;
/// cross-block via fixpoint — a block's live-in defined set is the INTERSECTION
/// of predecessors' defined-out sets (a unit is safely available only if defined
/// on ALL incoming paths). Entry live-ins and block liveins seed the defined
/// set. Reserved regs, EXEC, and undef uses are ignored (only allocatable
/// VGPR/SGPR/AGPR units are tracked).
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-verify-physreg-liveness"

static cl::opt<bool> EnableVerifyPhysRegLiveness(
    "amdgpu-verify-physreg-liveness", cl::Hidden, cl::init(false),
    cl::desc("Independent post-alloc physreg clobber check (SSARA correctness "
             "gate)"));

static cl::opt<bool> VerifyPhysRegFatal(
    "amdgpu-verify-physreg-liveness-fatal", cl::Hidden, cl::init(false),
    cl::desc("Abort on a physreg-liveness violation (default: warn to stderr)"));

namespace {

class AMDGPUVerifyPhysRegLiveness : public MachineFunctionPass {
  const SIRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  unsigned NumUnits = 0;

  // Per-block "defined-out" reg-unit sets, indexed by MBB number.
  DenseMap<unsigned, BitVector> DefinedOut;

  // Mark all allocatable units of PhysReg into BV.
  void markUnits(BitVector &BV, MCRegister PhysReg) const {
    for (MCRegUnit U : TRI->regunits(PhysReg))
      BV.set(U);
  }

  // Is this a unit we track (belongs to an allocatable reg; skip exec/scc/etc.)?
  // We test allocatability at the operand's register, not per unit, before
  // marking — units of reserved regs are simply never set, so a use of them is
  // not flagged.
  bool trackReg(MCRegister PhysReg) const {
    if (!PhysReg || MRI->isReserved(PhysReg))
      return false;
    const TargetRegisterClass *RC = TRI->getPhysRegBaseClass(PhysReg);
    return RC && RC->isAllocatable();
  }

public:
  static char ID;
  AMDGPUVerifyPhysRegLiveness() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AMDGPU Verify PhysReg Liveness";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (!EnableVerifyPhysRegLiveness)
      return false;

    TRI = static_cast<const SIRegisterInfo *>(
        MF.getSubtarget().getRegisterInfo());
    MRI = &MF.getRegInfo();
    NumUnits = TRI->getNumRegUnits();
    DefinedOut.clear();

    // Seed every block's defined-out to "all" so the intersection fixpoint can
    // shrink toward the true set; entry starts from its live-ins.
    for (MachineBasicBlock &MBB : MF) {
      BitVector All(NumUnits, true);
      DefinedOut[MBB.getNumber()] = All;
    }

    unsigned Violations = 0;
    bool Changed = true;
    unsigned Iter = 0;
    // Fixpoint: recompute each block's defined-out from predecessors' outs
    // (intersection) + its own defs, until stable. RPO for fast convergence.
    ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
    while (Changed && Iter++ < NumUnits + 8) {
      Changed = false;
      for (MachineBasicBlock *MBB : RPOT) {
        BitVector In(NumUnits, false);
        if (MBB->pred_empty()) {
          // Entry (or unreachable): seed from block live-ins.
          for (const auto &LI : MBB->liveins())
            if (trackReg(LI.PhysReg))
              markUnits(In, LI.PhysReg);
        } else {
          In.resize(NumUnits, true); // start full, intersect preds
          for (MachineBasicBlock *P : MBB->predecessors())
            In &= DefinedOut[P->getNumber()];
        }
        // Also honor explicit block liveins (physregs declared live-in are
        // defined on entry regardless of the intersection — they were produced
        // upstream / are ABI live-ins).
        for (const auto &LI : MBB->liveins())
          if (trackReg(LI.PhysReg))
            markUnits(In, LI.PhysReg);

        BitVector Cur = In;
        for (MachineInstr &MI : *MBB) {
          // Defs first? No — uses are read before defs execute. Process uses
          // against Cur, then apply defs. (Two-address/EC defs read as uses via
          // the use operand, which we see.)
          for (const MachineOperand &MO : MI.operands()) {
            if (!MO.isReg() || !MO.readsReg() || MO.isUndef())
              continue;
            MCRegister R = MO.getReg().asMCReg();
            if (!R || !R.isPhysical() || !trackReg(R))
              continue;
            for (MCRegUnit U : TRI->regunits(R)) {
              if (!Cur.test(U)) {
                // Only the fixpoint's FINAL iteration should report, else we
                // report transient states. Defer reporting to a post-fixpoint
                // pass; here we just note instability is impossible to use.
              }
            }
          }
          // Apply defs.
          for (const MachineOperand &MO : MI.operands()) {
            if (!MO.isReg() || !MO.isDef())
              continue;
            MCRegister R = MO.getReg().asMCReg();
            if (!R || !R.isPhysical())
              continue;
            markUnits(Cur, R);
          }
        }
        if (Cur != DefinedOut[MBB->getNumber()]) {
          DefinedOut[MBB->getNumber()] = Cur;
          Changed = true;
        }
      }
    }

    // Reporting pass: with defined-out stable, re-scan and flag undefined uses.
    for (MachineBasicBlock *MBB : RPOT) {
      BitVector Cur(NumUnits, false);
      if (MBB->pred_empty()) {
        for (const auto &LI : MBB->liveins())
          if (trackReg(LI.PhysReg))
            markUnits(Cur, LI.PhysReg);
      } else {
        Cur.resize(NumUnits, true);
        for (MachineBasicBlock *P : MBB->predecessors())
          Cur &= DefinedOut[P->getNumber()];
      }
      for (const auto &LI : MBB->liveins())
        if (trackReg(LI.PhysReg))
          markUnits(Cur, LI.PhysReg);

      for (MachineInstr &MI : *MBB) {
        for (const MachineOperand &MO : MI.operands()) {
          if (!MO.isReg() || !MO.readsReg() || MO.isUndef())
            continue;
          MCRegister R = MO.getReg().asMCReg();
          if (!R || !R.isPhysical() || !trackReg(R))
            continue;
          for (MCRegUnit U : TRI->regunits(R)) {
            if (!Cur.test(U)) {
              ++Violations;
              std::string Msg;
              raw_string_ostream OS(Msg);
              OS << "[physreg-liveness] read of undefined/clobbered unit "
                 << printRegUnit(U, TRI) << " (" << printReg(R, TRI)
                 << ") in " << printMBBReference(*MBB) << ":\n  " << MI;
              errs() << OS.str();
              break; // one report per operand
            }
          }
        }
        for (const MachineOperand &MO : MI.operands()) {
          if (!MO.isReg() || !MO.isDef())
            continue;
          MCRegister R = MO.getReg().asMCReg();
          if (!R || !R.isPhysical())
            continue;
          markUnits(Cur, R);
        }
      }
    }

    if (Violations) {
      errs() << "[physreg-liveness] " << Violations
             << " violation(s) in " << MF.getName() << "\n";
      if (VerifyPhysRegFatal)
        report_fatal_error("physreg-liveness violations detected");
    }
    return false;
  }
};

} // end anonymous namespace

char AMDGPUVerifyPhysRegLiveness::ID = 0;

INITIALIZE_PASS(AMDGPUVerifyPhysRegLiveness, DEBUG_TYPE,
                "AMDGPU Verify PhysReg Liveness", false, false)

MachineFunctionPass *llvm::createAMDGPUVerifyPhysRegLivenessPass() {
  return new AMDGPUVerifyPhysRegLiveness();
}
