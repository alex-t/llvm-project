//===-- AMDGPURebuildSSA.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reconstruct SSA form from non-SSA MIR by splitting multi-def vregs into
// single-def vregs and inserting PHI nodes.
//
// This is a temporary bridge: converts post-PHIElimination MIR back to SSA
// so the SSA Register Allocator can run.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLaneSSAUpdater.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-rebuild-ssa"

namespace {

class AMDGPURebuildSSALegacy : public MachineFunctionPass {
  LiveIntervals *LIS = nullptr;
  MachineDominatorTree *MDT = nullptr;
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;

public:
  static char ID;
  AMDGPURebuildSSALegacy() : MachineFunctionPass(ID) {
    initializeAMDGPURebuildSSALegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  // Collapse nested REG_SEQUENCE towers produced by lane-by-lane
  // reconstruction into flat REG_SEQUENCEs (see definition).
  void flattenRegSequences(MachineFunction &MF);

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredTransitiveID(MachineDominatorsID);
    AU.addPreservedID(MachineDominatorsID);
    AU.addRequired<LiveIntervalsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

// === runOnMachineFunction ===

bool AMDGPURebuildSSALegacy::runOnMachineFunction(MachineFunction &MF) {
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  MRI = &MF.getRegInfo();
  TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();

  if (MRI->isSSA())
    return false;

  LLVM_DEBUG(dbgs() << "\n=== AMDGPURebuildSSA on " << MF.getName()
                    << " ===\n");

  DenseMap<MachineBasicBlock *, unsigned> DomPreorder;
  {
    unsigned N = 0;
    for (auto *Node : depth_first(MDT->getRootNode()))
      DomPreorder[Node->getBlock()] = N++;
  }

  MachineLaneSSAUpdater Updater(MF, *LIS, *MDT, *TRI);
  DenseSet<Register> Processed;
  SmallVector<MachineOperand *, 4> PHIDefs;

  for (auto &B : MF) {
    for (auto &I : B) {
      // Iterate ALL operands filtered by the isDef flag rather than I.defs():
      // the defs() range helper returns only the leading explicit defs
      // ([0, getNumExplicitDefs())), which is EMPTY for INLINEASM (its def
      // operands sit after the asm string and flag immediates, not leading). So
      // I.defs() misses a vreg defined only by INLINEASM (e.g. two INLINEASM
      // lane-defs building a tuple: `def undef %V.sub1` + `def %V.sub0`), leaving
      // it with two subreg defs and never converted to SSA -> MachineVerifier
      // "Multiple virtual register defs in SSA form". Flag-based filtering visits
      // every real def regardless of operand position. (Same fix as
      // AMDGPUSSARegisterAllocator.cpp's def walk.)
      for (MachineOperand &Def : I.operands()) {
        if (!Def.isReg() || !Def.isDef() || !Def.getReg().isVirtual())
          continue;

        Register VReg = Def.getReg();
        if (!LIS->hasInterval(VReg) || !Processed.insert(VReg).second)
          continue;

        // Collect VNInfos before any repair calls: repairSSAForNewDef
        // replaces OrigVReg's interval object, invalidating any reference
        // held across calls.
        LiveInterval &LI = LIS->getInterval(VReg);
        if (LI.getNumValNums() == 1)
          continue;

        LLVM_DEBUG(dbgs() << "\n[VREG] " << printReg(VReg) << " has "
                          << LI.getNumValNums() << " VNs\n");

        // Order defs by dominator-tree preorder, breaking ties within a block
        // by definition slot, so the establishing (earliest) def sorts first.
        auto IsEarlier = [&](VNInfo *A, VNInfo *B) {
          unsigned PA = DomPreorder[LIS->getMBBFromIndex(A->def)];
          unsigned PB = DomPreorder[LIS->getMBBFromIndex(B->def)];
          if (PA != PB)
            return PA < PB;
          return A->def < B->def;
        };

        // A wide vreg can be assembled by a chain of partial subregister defs
        // where every def after the first read-modify-writes the super-register
        // (a subreg def without `undef` implicitly reads the lanes it does not
        // write). Such a re-def must be renamed before the establishing def it
        // reads, otherwise recomputing OrigVReg's interval mid-chain reads lanes
        // whose def was just renamed away. This ordering constraint is
        // INTRA-block (a dominance chain by slot, no back-edges), so it is safe
        // even when the vreg also has defs in other blocks or loops -- it is
        // applied per block in the WorkList sort below (not as a global
        // reverse), and the establishing def is kept as Root (processed last).
        bool HasRMWRedef = false;
        for (const MachineOperand &MO : MRI->def_operands(VReg))
          if (MO.getSubReg() && !MO.isUndef()) {
            HasRMWRedef = true;
            break;
          }

        // Find the establishing (earliest) non-PHI definition. This is the
        // unique original SSA def that anchors the vreg. For an RMW chain the
        // earliest (slot-ordered) def must win so it becomes Root and is kept.
        VNInfo *Root = nullptr;
        for (VNInfo *V : LI.vnis()) {
          if (!V || V->isUnused() || V->isPHIDef())
            continue;
          if (HasRMWRedef) {
            if (!Root || IsEarlier(V, Root))
              Root = V;
          } else if (!Root ||
                     DomPreorder[LIS->getMBBFromIndex(V->def)] <
                         DomPreorder[LIS->getMBBFromIndex(Root->def)]) {
            Root = V;
          }
        }

        // Collect re-defs (non-Root, non-PHI) in dom-preorder, Root last.
        // PHI VNInfos carry no instruction; they are handled implicitly
        // by the IDF computation inside repairSSAForNewDef.
        // Root is placed last so the updater sees OrigVReg's interval
        // reduced to Root's lanes only when computing Root's IDF.
        SmallVector<VNInfo *, 8> WorkList;
        for (VNInfo *V : LI.vnis())
          if (V && !V->isUnused() && !V->isPHIDef() && V != Root)
            WorkList.push_back(V);
        llvm::sort(WorkList, [&](VNInfo *A, VNInfo *B) {
          // Same-block RMW chain: rename later (RMW-reader) defs before the
          // earlier establishing def they read -- IsEarlier(B, A) reverses the
          // slot order (block equal here). Renaming the establisher first would
          // strip a lane the reader still needs. Intra-block only, so
          // dominance-safe (no back-edges) even for multi-block/loop vregs.
          // Otherwise: dominator-preorder, then slot.
          if (HasRMWRedef &&
              LIS->getMBBFromIndex(A->def) == LIS->getMBBFromIndex(B->def))
            return IsEarlier(B, A);
          return IsEarlier(A, B);
        });
        WorkList.push_back(Root);
        // LI not used below this point.

        assert(Root && "live interval with multiple VNs must have a non-PHI def");

        // Rename each re-def to a fresh vreg. repairSSAForNewDef inserts
        // lane-aware PHIs at pruned IDF blocks and rewrites dominated uses
        // with exact/subset/super policy, then recomputes LiveIntervals for
        // all affected vregs.
        // A full-register Root def is the unique SSA def for its lanes and
        // needs no renaming. A partial (subreg) Root def leaves OrigVReg as
        // a wide register class with only a subset of lanes live, causing
        // downstream passes to allocate the full register tuple unnecessarily.
        for (VNInfo *VNI : WorkList) {
          MachineInstr *DefMI = LIS->getInstructionFromIndex(VNI->def);
          assert(DefMI && "non-PHI VNInfo must have an instruction");
          if (VNI == Root) {
            // Root is processed last. For a single-block RMW chain the re-defs
            // have already been renamed by now, so narrowing a subreg Root here
            // breaks nothing and avoids leaving OrigVReg as a wide register with
            // only a subset of lanes live (which would inflate pressure).
            int Idx = DefMI->findRegisterDefOperandIdx(VReg, TRI,
                                                       /*IsDead=*/false,
                                                       /*Overlaps=*/true);
            if (Idx < 0 || !DefMI->getOperand(Idx).getSubReg())
              continue;
          }
          PHIDefs.clear();
          Updater.repairSSAForNewDef(*DefMI, VReg, PHIDefs);
        }

        // Restore intra-instruction register coupling that lane-by-lane repair
        // may have severed (e.g. V_MOVRELS src0 must stay a subreg of the
        // implicit vector use).
        Updater.finalizeCoupledUses(VReg);

        // All defs this pass intended to rename are renamed; VReg's main range
        // is the union of its now-final subranges.
        Updater.rebuildMainRangeFromSubranges(VReg);
      }
    }
  }

  Processed.clear();

  // Flatten nested REG_SEQUENCE towers the reconstruction produced, so wide
  // values rebuilt lane-by-lane don't carry growing-width live intermediates
  // into register allocation.
  flattenRegSequences(MF);

  MF.getProperties().set(MachineFunctionProperties::Property::IsSSA);
  MF.getProperties().reset(MachineFunctionProperties::Property::NoPHIs);
  // Re-SSA-ifying turns rewritten two-address tied operands back into distinct
  // SSA values (def != use), so the function is no longer in two-address form.
  MF.getProperties().reset(MachineFunctionProperties::Property::TiedOpsRewritten);

  LLVM_DEBUG({
    dbgs() << "=== verify ===\n";
    MF.verify();
  });
  return MRI->isSSA();
}

// Collapse nested REG_SEQUENCE towers into flat REG_SEQUENCEs.
//
// Lane-by-lane SSA reconstruction rewrites each partial re-def by wrapping the
// prior value in a 2-source REG_SEQUENCE, so a wide value rebuilt from many
// single-lane defs becomes a chain of growing-width intermediates
// (areg_64 -> areg_96 -> ... -> areg_384). Every intermediate is a live wide
// tuple; with no coalescer they inflate register pressure (observed: an MFMA
// accumulator using 43 AGPRs instead of 32, dropping occupancy 8 -> 5).
// Inlining a single-use, whole-register REG_SEQUENCE source leaves one flat
// REG_SEQUENCE whose only live values are the narrow sources and the result.
void AMDGPURebuildSSALegacy::flattenRegSequences(MachineFunction &MF) {
  DenseSet<Register> Touched;

  for (MachineBasicBlock &MBB : MF) {
    // MI (the parent) is never erased here; only its child REG_SEQUENCEs are,
    // and in SSA a child (def of a source) precedes MI, so erasing it never
    // invalidates this forward iterator.
    for (MachineInstr &MI : MBB) {
      if (!MI.isRegSequence())
        continue;

      Register PReg = MI.getOperand(0).getReg();
      LaneBitmask PFull = MRI->getMaxLaneMaskForVReg(PReg);

      // Drain this parent's tower: repeatedly inline a single-use, whole-read
      // child REG_SEQUENCE source until none remain (inlining one may expose a
      // grandchild as a new direct source).
      bool Inlined = true;
      while (Inlined) {
        Inlined = false;
        for (unsigned I = 1; I < MI.getNumOperands(); I += 2) {
          MachineOperand &SrcMO = MI.getOperand(I);
          if (!SrcMO.isReg() || SrcMO.getSubReg() ||
              !SrcMO.getReg().isVirtual())
            continue;
          MachineInstr *Child = MRI->getVRegDef(SrcMO.getReg());
          if (!Child || !Child->isRegSequence() ||
              !MRI->hasOneNonDBGUse(SrcMO.getReg()))
            continue;

          unsigned DestSub = MI.getOperand(I + 1).getImm();
          // Append each child source, mapping its child-local dest lanes into
          // the parent's namespace via lane masks. (composeSubRegIndices is
          // unsafe: it silently mis-handles compositions that do not land on a
          // defined subreg index, which AMDGPU's irregular areg lattice can
          // produce.)
          for (unsigned J = 1; J < Child->getNumOperands(); J += 2) {
            MachineOperand CS = Child->getOperand(J); // copy: preserves undef
            CS.setIsKill(false); // liveness recomputed below; avoid stale kills
            unsigned CSub = Child->getOperand(J + 1).getImm();
            LaneBitmask ChildLanes = TRI->getSubRegIndexLaneMask(CSub);
            LaneBitmask PLanes =
                TRI->composeSubRegIndexLaneMask(DestSub, ChildLanes);
            unsigned Composed =
                (PLanes == PFull) ? 0 : TRI->getSubRegIndexForLaneMask(PLanes);
            assert((PLanes == PFull || Composed) &&
                   "flattened REG_SEQUENCE slice has no subreg index");
            MI.addOperand(MF, CS);
            MI.addOperand(MF, MachineOperand::CreateImm(Composed));
            if (CS.getReg().isVirtual())
              Touched.insert(CS.getReg());
          }

          // Drop the inlined (Src, DestSub) pair; added sources are at the end,
          // so these indices are still valid. Then erase the now-dead child.
          MI.removeOperand(I + 1);
          MI.removeOperand(I);

          Register Dead = Child->getOperand(0).getReg();
          LIS->RemoveMachineInstrFromMaps(*Child);
          Child->eraseFromParent();
          LIS->removeInterval(Dead);

          Inlined = true;
          break; // MI's operand list changed; re-scan it
        }
      }
    }
  }

  // Inlined sources' single use moved from the erased child to the parent MI;
  // recompute their intervals. Parent results are unchanged (same def/uses).
  for (Register R : Touched)
    if (R.isVirtual() && LIS->hasInterval(R)) {
      LIS->removeInterval(R);
      LIS->createAndComputeVirtRegInterval(R);
    }
}

char AMDGPURebuildSSALegacy::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPURebuildSSALegacy, DEBUG_TYPE, "AMDGPU Rebuild SSA",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(AMDGPURebuildSSALegacy, DEBUG_TYPE, "AMDGPU Rebuild SSA",
                    false, false)

FunctionPass *llvm::createAMDGPURebuildSSALegacyPass() {
  return new AMDGPURebuildSSALegacy();
}
