//===-- AMDGPUPHISimplifier.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Undef-aware PHI SIMPLIFICATION for the SSA register-allocation stack.
//
// NOTE: this is NOT a PHI coalescer. It performs no interference-graph
// coalescing and no Hack-style permutation fixed point; it is a few local,
// SSA-preserving peephole rewrites run PRE-RA that remove the artificial
// register-pressure inflation the structurizer's undef-PHIs create. The real
// PHI coalescer (Hack sec. 4.3) is a separate, later component.
//
// Runs after SSA reconstruction and before the SSA spiller. It targets the
// register-pressure inflation caused by "one real operand, rest undef" PHIs,
// which the structurizer produces at every diamond merge: each merged lane is a
//
//     %res = PHI %real, <real-edge>, undef, <other-edge>
//
// where <other-edge> supplies an IMPLICIT_DEF placeholder for a lane the other
// arm never wrote. Two problems follow if this reaches allocation unchanged:
//
//   1. The all-undef placeholder (often a wide IMPLICIT_DEF tuple, e.g. a
//      vreg_512 read as %593.subN by 16 lane PHIs) is colored to real
//      registers and held live across the region -- pure waste.
//   2. %res is colored independently of %real, double-counting one value at the
//      merge. An N-lane diamond then presents an N-wide simultaneous peak even
//      though no control-flow edge actually carries N distinct real values.
//      Worse, the spiller cannot relieve it: %res's "use" is the PHI and a PHI
//      operand is live-out of its predecessor by definition, so there is no
//      program point at which storing it shortens the range that crosses the
//      edge.
//
// The transform is three local, SSA-preserving rewrites:
//
//   (a) Flag every PHI operand that reads a fully-undef value (a vreg whose sole
//       def is IMPLICIT_DEF) with the `undef` flag. This does not change
//       semantics -- the read was already undef -- but it lets liveness see the
//       placeholder as dead, reclaiming its registers.
//
//   (b) When, after (a), every non-undef operand of a PHI reads one and the same
//       value `%real` and %real's def dominates the PHI's block, replace the PHI
//       result with %real and delete the PHI. This is the classic undef-PHI
//       simplification: on a real edge %res IS %real; on every other edge %res
//       is a don't-care, so %real (available by dominance) is a legal value
//       there too. It also covers the PHI whose edges all carry the identical
//       value, which is a copy rather than a merge. The value is no longer
//       double-counted, and %real -- an ordinary def/use range -- is spillable
//       where the φ-operand was not.
//
//   (c) Erase an IMPLICIT_DEF once (a) has made every read of it undef. Nothing
//       reads a defined value from it, so holding a register for it is waste.
//       The operands reading it keep referring to a def-less vreg, which is
//       valid for an undef read, and PHI lowering materialises an IMPLICIT_DEF
//       on the PHYSICAL register for the undef edge where one is needed.
//
// This is the tractable, high-value slice of a full PHI-aware coalescer (Hack):
// it needs only "one real operand" plus a dominance test, not an interference
// graph. It does not lower a genuine simultaneous-liveness peak (that is the
// spiller's job); it removes the *artificial* inflation so the spiller's
// ordinary def/use model can do its work.
//
// LiveIntervals/SlotIndexes are intentionally not preserved: the pass edits MIR
// directly and the following spiller re-requires (hence recomputes) them.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-phi-simplifier"

STATISTIC(NumUndefFlagged, "Number of fully-undef PHI operands flagged undef");
STATISTIC(NumPHIsSimplified, "Number of single-real PHIs folded to their operand");
STATISTIC(NumImplicitDefsErased,
          "Number of IMPLICIT_DEFs erased once all their reads were undef");

// Escape hatch for A/B measurement and bisection. On by default.
static cl::opt<bool> EnablePHISimplifier(
    "amdgpu-phi-simplify", cl::Hidden, cl::init(true),
    cl::desc("Enable undef-aware PHI simplification before the SSA spiller"));

// Sub-flag for differential diagnostics: attribute crash/pressure effects to
// the single-real fold independently. On by default; the master flag above
// still gates the whole pass.
static cl::opt<bool> EnableSingleRealFold(
    "amdgpu-phi-simplify-fold", cl::Hidden, cl::init(true),
    cl::desc("(b) fold single-real PHIs onto their real operand"));

namespace {

class AMDGPUPHISimplifier : public MachineFunctionPass {
  MachineDominatorTree *MDT = nullptr;
  MachineRegisterInfo *MRI = nullptr;

  // True if VReg's sole definition is an IMPLICIT_DEF, i.e. every read of it is
  // an undef read regardless of the operand's flag.
  bool isFullyUndef(Register VReg) const {
    if (!VReg.isVirtual())
      return false;
    MachineInstr *Def = MRI->getUniqueVRegDef(VReg);
    return Def && Def->isImplicitDef();
  }

public:
  static char ID;

  AMDGPUPHISimplifier() : MachineFunctionPass(ID) {
    initializeAMDGPUPHISimplifierPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char AMDGPUPHISimplifier::ID = 0;

char &llvm::AMDGPUPHISimplifierID = AMDGPUPHISimplifier::ID;

INITIALIZE_PASS_BEGIN(AMDGPUPHISimplifier, DEBUG_TYPE, "AMDGPU PHI Simplifier",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(AMDGPUPHISimplifier, DEBUG_TYPE, "AMDGPU PHI Simplifier",
                    false, false)

bool AMDGPUPHISimplifier::runOnMachineFunction(MachineFunction &MF) {
  if (!EnablePHISimplifier)
    return false;

  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  MRI = &MF.getRegInfo();

  if (!MRI->isSSA())
    return false;

  LLVM_DEBUG(dbgs() << "\n=== AMDGPUPHISimplifier on " << MF.getName()
                    << " ===\n");

  bool Changed = false;

  // Collect PHIs up front so a folded PHI can be erased immediately. The
  // erase must be immediate, not deferred: replaceRegWith(Res, Real) rewrites
  // *every* operand mentioning Res, including the PHI's own def operand, so
  // until the PHI is gone Real transiently has two defs. A deferred erase would
  // let a later getVRegDef(Real) assert on the multiple definition.
  SmallVector<MachineInstr *, 32> PHIs;
  SmallVector<MachineInstr *, 32> ImplicitDefs;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB) {
      if (MI.isPHI())
        PHIs.push_back(&MI);
      else if (MI.isImplicitDef())
        ImplicitDefs.push_back(&MI);
    }

  for (MachineInstr *PHIPtr : PHIs) {
    MachineInstr &PHI = *PHIPtr;
    // (a) Flag every operand that reads a fully-undef value as undef, so the
    // placeholder it reads is seen as dead by liveness. Simultaneously find
    // the sole non-undef (real) operand, if there is exactly one.
    MachineOperand *SoleReal = nullptr;
    bool MultipleReal = false;
    for (unsigned I = 1, E = PHI.getNumOperands(); I < E; I += 2) {
      MachineOperand &Src = PHI.getOperand(I);
      if (Src.isUndef())
        continue;
      if (isFullyUndef(Src.getReg())) {
        Src.setIsUndef(true);
        ++NumUndefFlagged;
        Changed = true;
        continue;
      }
      if (SoleReal) {
        // The same (register, sub-register) read on several edges is ONE real
        // value: every edge carries the identical value, so the PHI is a copy of
        // it. Only a genuinely different source makes the PHI a real merge.
        if (SoleReal->getReg() != Src.getReg() ||
            SoleReal->getSubReg() != Src.getSubReg())
          MultipleReal = true;
      } else
        SoleReal = &Src;
    }

    // (b) Fold a single-real PHI onto its real operand `%real` and delete
    // the PHI, replacing every use of the result `%res` with `%real`. Only
    // whole-register sources qualify (a sub-register source would need index
    // composition; left as an ordinary PHI).
    //
    // The fold is legal iff `%real`'s def dominates the PHI *instruction*.
    // That single instruction-level test is exactly the right condition:
    //   - `%res` is defined only by the PHI, so every use of `%res` is
    //     dominated by the PHI (ordinary uses directly; a downstream
    //     PHI-operand edge use in predecessor P requires, by SSA validity,
    //     the PHI's block M to dominate P). If `RealDef` dominates the PHI,
    //     its block strictly dominates M and therefore dominates every such
    //     use point -- so replacing all uses of `%res` with `%real` cannot
    //     break SSA.
    //   - It rejects the loop-carried induction case that a block-level or
    //     per-use test lets slip: for a header PHI
    //       %res = PHI %init/undef, <preheader>, %real, <latch>
    //     the back-edge value `%real` is computed *from* `%res` in the loop
    //     body (e.g. %real = ADD 1, %res), so its def sits below the header
    //     PHI and does NOT dominate it. `%res` and `%real` are distinct
    //     per-iteration values, not a copy; folding would emit the self-
    //     reference `%real = ADD 1, %real`. Requiring dominance of the PHI
    //     instruction correctly declines.
    if (!EnableSingleRealFold)
      continue;
    if (!SoleReal || MultipleReal || SoleReal->getSubReg())
      continue;
    Register Real = SoleReal->getReg();
    Register Res = PHI.getOperand(0).getReg();
    if (!Real.isVirtual() || Real == Res)
      continue;
    // Result and operand must share a register class for a plain replacement.
    if (MRI->getRegClass(Real) != MRI->getRegClass(Res))
      continue;
    MachineInstr *RealDef = MRI->getVRegDef(Real);
    if (!RealDef || !MDT->dominates(RealDef, &PHI))
      continue;

    LLVM_DEBUG(dbgs() << "  simplify " << printReg(Res) << " := "
                      << printReg(Real) << " (" << PHI);
    MRI->replaceRegWith(Res, Real);
    PHI.eraseFromParent();
    ++NumPHIsSimplified;
    Changed = true;
  }

  // (c) An IMPLICIT_DEF all of whose reads are undef has no reader liveness can
  // see, yet it is still assigned a physical register. Erase it; the operands
  // reading it keep referring to a now def-less vreg, which is valid for an
  // undef read, and PHI lowering materialises an IMPLICIT_DEF on the physical
  // register for the undef edge where the value must appear defined.
  for (MachineInstr *ID : ImplicitDefs) {
    Register Def = ID->getOperand(0).getReg();
    if (!Def.isVirtual())
      continue;
    if (!all_of(MRI->use_nodbg_operands(Def),
                [](const MachineOperand &MO) { return MO.isUndef(); }))
      continue;
    ID->eraseFromParent();
    ++NumImplicitDefsErased;
    Changed = true;
  }

  return Changed;
}

MachineFunctionPass *llvm::createAMDGPUPHISimplifierPass() {
  return new AMDGPUPHISimplifier();
}
