//===- SIFixSGPRCopies.cpp - Remove potential VGPR => SGPR copies ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Copies from VGPR to SGPR registers are illegal and the register coalescer
/// will sometimes generate these illegal copies in situations like this:
///
///  Register Class <vsrc> is the union of <vgpr> and <sgpr>
///
/// BB0:
///   %0 <sgpr> = SCALAR_INST
///   %1 <vsrc> = COPY %0 <sgpr>
///    ...
///    BRANCH %cond BB1, BB2
///  BB1:
///    %2 <vgpr> = VECTOR_INST
///    %3 <vsrc> = COPY %2 <vgpr>
///  BB2:
///    %4 <vsrc> = PHI %1 <vsrc>, <%bb.0>, %3 <vrsc>, <%bb.1>
///    %5 <vgpr> = VECTOR_INST %4 <vsrc>
///
///
/// The coalescer will begin at BB0 and eliminate its copy, then the resulting
/// code will look like this:
///
/// BB0:
///   %0 <sgpr> = SCALAR_INST
///    ...
///    BRANCH %cond BB1, BB2
/// BB1:
///   %2 <vgpr> = VECTOR_INST
///   %3 <vsrc> = COPY %2 <vgpr>
/// BB2:
///   %4 <sgpr> = PHI %0 <sgpr>, <%bb.0>, %3 <vsrc>, <%bb.1>
///   %5 <vgpr> = VECTOR_INST %4 <sgpr>
///
/// Now that the result of the PHI instruction is an SGPR, the register
/// allocator is now forced to constrain the register class of %3 to
/// <sgpr> so we end up with final code like this:
///
/// BB0:
///   %0 <sgpr> = SCALAR_INST
///    ...
///    BRANCH %cond BB1, BB2
/// BB1:
///   %2 <vgpr> = VECTOR_INST
///   %3 <sgpr> = COPY %2 <vgpr>
/// BB2:
///   %4 <sgpr> = PHI %0 <sgpr>, <%bb.0>, %3 <sgpr>, <%bb.1>
///   %5 <vgpr> = VECTOR_INST %4 <sgpr>
///
/// Now this code contains an illegal copy from a VGPR to an SGPR.
///
/// In order to avoid this problem, this pass searches for PHI instructions
/// which define a <vsrc> register and constrains its definition class to
/// <vgpr> if the user of the PHI's definition register is a vector instruction.
/// If the PHI's definition class is constrained to <vgpr> then the coalescer
/// will be unable to perform the COPY removal from the above example  which
/// ultimately led to the creation of an illegal COPY.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "si-fix-sgpr-copies"

static cl::opt<bool> EnableM0Merge(
  "amdgpu-enable-merge-m0",
  cl::desc("Merge and hoist M0 initializations"),
  cl::init(true));

namespace {

class SIFixSGPRCopies : public MachineFunctionPass {
  MachineDominatorTree *MDT;

  class UserInfo {
    MachineInstr *Copy;
    unsigned SScore, VScore, SVCopies;
    SetVector<MachineInstr *> Descendants;

  public:
    UserInfo() : Copy(nullptr), SScore(0), VScore(0), SVCopies(0){};
    // UserInfo(MachineInstr *C) : Copy(C) {}
    UserInfo(MachineInstr *C, unsigned S, unsigned V, unsigned SV,
             SetVector<MachineInstr *> D)
        : Copy(C), SScore(S), VScore(V), SVCopies(SV), Descendants(D){};
    unsigned getSScore() { return SScore; }
    unsigned getVScore() { return VScore; }
    unsigned getSVcopies() { return SVCopies; }
    bool Reaching(MachineInstr *X) { return Descendants.contains(X); }
    /* void setScores(unsigned S, unsigned V, unsigned SV) {
      SScore = S;
      VScore = V;
      SVCopies = SV;
    }*/
    MachineInstr *getCopy() { return Copy; }
    SetVector<MachineInstr *> &getDescendants() { return Descendants; }
    // void addDescendants(SetVector<MachineInstr *> Desc) {
    //   Descendants.insert(Desc.begin(), Desc.end());
    // }
    explicit operator bool() { return Copy != nullptr; }
    void dump() const;
    void print(raw_ostream &OS) const;
  };

  class V2SCopyInfo {
    MachineInstr *Copy;
    struct Sibling {
      MachineInstr *SiblingCopy;
      MachineInstr *CommonUser;
      unsigned CommonUserSALUScore;
      unsigned CommonUserVALUScore;
      unsigned CommonUserS2VCopies;
    };
    SmallVector<Sibling, 4> Siblings;
    unsigned SALUChainLength;
    unsigned VALUChainLength;
    unsigned NumSVCopies;
    static DenseMap<MachineInstr *, V2SCopyInfo> V2SCopies;
    static V2SCopyInfo NullInfo;

  public:
    MachineInstr *getCopy() { return Copy; }
    unsigned getSALUScore() { return SALUChainLength; }
    unsigned getVALUScore() { return VALUChainLength; }
    unsigned getSVCopies() { return NumSVCopies; }
    void setSALUScore(unsigned Score) { SALUChainLength = Score; }
    void setVALUScore(unsigned Score) { VALUChainLength = Score; }
    void incSALU() { SALUChainLength++; }
    void incVALU() { VALUChainLength++; }
    void incSVCopies() { NumSVCopies++; }

    void addSibling(MachineInstr *SiblingCopy, MachineInstr *CommonUser,
                    UserInfo& Info, bool UpdateScores = false) {
      Siblings.push_back({SiblingCopy, CommonUser, Info.getSScore(),
                          Info.getVScore(), Info.getSVcopies()});
      if (UpdateScores) {
        SALUChainLength += Info.getSScore();
        VALUChainLength += Info.getVScore();
        NumSVCopies += Info.getSVcopies();
      }
    }
    //void addSibling(MachineInstr *SiblingCopy, MachineInstr *CommonUser) {
    //  Siblings.push_back({SiblingCopy, CommonUser, SALUChainLength,
    //                      VALUChainLength, NumSVCopies});
    //}
    SmallVector<Sibling, 4> &getSiblings() { return Siblings; }
    int getScore() {
      // VALUChainLength is not very useful. It reflects the summary length
      // of all the VALU chains originated from the copy. Each VALU chain
      // originated from the V2S copy requires S2V copy OR VALU instruction
      // accepting SGPR operandand and producing result in VGPR
      // TODO: deside if/how we should use this metric.
      //
      // NumSVCopies  - the real number of the VALU users of the root V2S copy.
      // If we have many VALU users of the root V2S copy we'll likely need many
      // S2V copies if we turn root copy to v_readfirstlane_b32. This number of
      // S2V copis required considered a penaulty.
      //
      // NumReadfirstlanes - number of v_readfirstlane_b32 instructions that
      // need to be added to keep the corresponding subtree in SALU.
      //
      // SALUChainLength - summarized length of all the continuous SALU tracks
      // over all V2S copy users.
      unsigned NumReadfirstlanes = 1 + Siblings.size();
      return SALUChainLength - (NumReadfirstlanes /* + NumSVCopies*/);
    }
    V2SCopyInfo()
        : Copy(nullptr), SALUChainLength(0), VALUChainLength(0),
          NumSVCopies(0){};
    V2SCopyInfo(MachineInstr *MI)
        : Copy(MI), SALUChainLength(0), VALUChainLength(0), NumSVCopies(0){};
    V2SCopyInfo(MachineInstr *MI, uint16_t nS, uint16_t nV)
        : Copy(MI), SALUChainLength(nS), VALUChainLength(nV), NumSVCopies(0){};
    ~V2SCopyInfo(){};

    explicit operator bool() {
      return Copy != nullptr;
    }

    static V2SCopyInfo& getV2SCopy(MachineInstr *MI) {
      if (V2SCopies.count(MI))
        return V2SCopies[MI];
      return NullInfo;
    }

    static void addV2SCopy(MachineInstr *MI, V2SCopyInfo Copy) {
      V2SCopies[MI] = Copy;
    }

    static void eraseV2SCopy(MachineInstr *MI) {
      if (V2SCopyInfo &Info = getV2SCopy(MI)) {
        // We turn to VALU V2S copy and all its SSA users subtree
        // including the instruction common between the copy and its
        // sibling Sibling scores reflect the numbers for the common
        // instruction subtree but not instruction itself. Since it is
        // going to turn to VALU we add one point to sibling VALU score
        // and subtract one from its SALU score.
        for (auto &S : Info.getSiblings()) {
          if (V2SCopyInfo &SInfo = getV2SCopy(S.SiblingCopy)) {
            unsigned SScore = Info.getSALUScore();
            unsigned SiblingSScore = SInfo.getSALUScore();
            SInfo.setSALUScore(
                SiblingSScore > SScore + 1 ? SiblingSScore - SScore - 1 : 0);
            SInfo.setVALUScore(SInfo.getVALUScore() + Info.getVALUScore() + 1);
            for (auto &SS : SInfo.getSiblings())
              if (SS.SiblingCopy == MI) {
                SInfo.getSiblings().erase(&SS);
                break;
              }
          }
        }
        // Remove all the related bookkeeping
        LLVM_DEBUG(dbgs() << "Erasing V2S copy:" << *MI);
        V2SCopies.erase(MI);
      }
    }

    static SmallVector<V2SCopyInfo, 8> getSorted() {
      SmallVector<V2SCopyInfo, 8> Ret;
      struct {
        bool operator()(V2SCopyInfo I, V2SCopyInfo J) {
          return I.getScore() < J.getScore();
        }
      } Pred;
      for (auto P : V2SCopies)
        Ret.push_back(P.getSecond());
      std::sort(Ret.begin(), Ret.end(), Pred);
      return Ret;
    }
    static void clear() { V2SCopies.clear(); }
    void dump() const;
    void print(raw_ostream &OS) const;
    static void print();
  };

public:
  static char ID;

  MachineRegisterInfo *MRI;
  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;

  SIFixSGPRCopies() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;
  void collectV2SCopyInfo(MachineFunction &MF);
  void lowerV2SCopies(MachineFunction &MF);
  bool LowerSpecialCase(MachineInstr &MI);
  void testIterative(MachineFunction &MF);
  void testIterative1(MachineFunction &MF);

  MachineBasicBlock *processPHINode(MachineInstr &MI);

  StringRef getPassName() const override { return "SI Fix SGPR copies"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineDominatorTree>();
    AU.addPreserved<MachineDominatorTree>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

DenseMap<MachineInstr *, SIFixSGPRCopies::V2SCopyInfo>
    SIFixSGPRCopies::V2SCopyInfo::V2SCopies =
        DenseMap<MachineInstr *, SIFixSGPRCopies::V2SCopyInfo>();
SIFixSGPRCopies::V2SCopyInfo SIFixSGPRCopies::V2SCopyInfo::NullInfo =
    V2SCopyInfo();

INITIALIZE_PASS_BEGIN(SIFixSGPRCopies, DEBUG_TYPE,
                     "SI Fix SGPR copies", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_END(SIFixSGPRCopies, DEBUG_TYPE,
                     "SI Fix SGPR copies", false, false)

char SIFixSGPRCopies::ID = 0;

char &llvm::SIFixSGPRCopiesID = SIFixSGPRCopies::ID;

FunctionPass *llvm::createSIFixSGPRCopiesPass() {
  return new SIFixSGPRCopies();
}

static bool hasVectorOperands(const MachineInstr &MI,
                              const SIRegisterInfo *TRI) {
  const MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.getReg().isVirtual())
      continue;

    if (TRI->hasVectorRegisters(MRI.getRegClass(MO.getReg())))
      return true;
  }
  return false;
}

static std::pair<const TargetRegisterClass *, const TargetRegisterClass *>
getCopyRegClasses(const MachineInstr &Copy,
                  const SIRegisterInfo &TRI,
                  const MachineRegisterInfo &MRI) {
  Register DstReg = Copy.getOperand(0).getReg();
  Register SrcReg = Copy.getOperand(1).getReg();

  const TargetRegisterClass *SrcRC = SrcReg.isVirtual()
                                         ? MRI.getRegClass(SrcReg)
                                         : TRI.getPhysRegClass(SrcReg);

  // We don't really care about the subregister here.
  // SrcRC = TRI.getSubRegClass(SrcRC, Copy.getOperand(1).getSubReg());

  const TargetRegisterClass *DstRC = DstReg.isVirtual()
                                         ? MRI.getRegClass(DstReg)
                                         : TRI.getPhysRegClass(DstReg);

  return std::make_pair(SrcRC, DstRC);
}

static bool isVGPRToSGPRCopy(const TargetRegisterClass *SrcRC,
                             const TargetRegisterClass *DstRC,
                             const SIRegisterInfo &TRI) {
  return SrcRC != &AMDGPU::VReg_1RegClass && TRI.isSGPRClass(DstRC) &&
         TRI.hasVectorRegisters(SrcRC);
}

static bool isSGPRToVGPRCopy(const TargetRegisterClass *SrcRC,
                             const TargetRegisterClass *DstRC,
                             const SIRegisterInfo &TRI) {
  return DstRC != &AMDGPU::VReg_1RegClass && TRI.isSGPRClass(SrcRC) &&
         TRI.hasVectorRegisters(DstRC);
}

static bool tryChangeVGPRtoSGPRinCopy(MachineInstr &MI,
                                      const SIRegisterInfo *TRI,
                                      const SIInstrInfo *TII) {
  MachineRegisterInfo &MRI = MI.getParent()->getParent()->getRegInfo();
  auto &Src = MI.getOperand(1);
  Register DstReg = MI.getOperand(0).getReg();
  Register SrcReg = Src.getReg();
  if (!SrcReg.isVirtual() || !DstReg.isVirtual())
    return false;

  for (const auto &MO : MRI.reg_nodbg_operands(DstReg)) {
    const auto *UseMI = MO.getParent();
    if (UseMI == &MI)
      continue;
    if (MO.isDef() || UseMI->getParent() != MI.getParent() ||
        UseMI->getOpcode() <= TargetOpcode::GENERIC_OP_END)
      return false;

    unsigned OpIdx = UseMI->getOperandNo(&MO);
    if (OpIdx >= UseMI->getDesc().getNumOperands() ||
        !TII->isOperandLegal(*UseMI, OpIdx, &Src))
      return false;
  }
  // Change VGPR to SGPR destination.
  MRI.setRegClass(DstReg, TRI->getEquivalentSGPRClass(MRI.getRegClass(DstReg)));
  return true;
}

// Distribute an SGPR->VGPR copy of a REG_SEQUENCE into a VGPR REG_SEQUENCE.
//
// SGPRx = ...
// SGPRy = REG_SEQUENCE SGPRx, sub0 ...
// VGPRz = COPY SGPRy
//
// ==>
//
// VGPRx = COPY SGPRx
// VGPRz = REG_SEQUENCE VGPRx, sub0
//
// This exposes immediate folding opportunities when materializing 64-bit
// immediates.
static bool foldVGPRCopyIntoRegSequence(MachineInstr &MI,
                                        const SIRegisterInfo *TRI,
                                        const SIInstrInfo *TII,
                                        MachineRegisterInfo &MRI) {
  assert(MI.isRegSequence());

  Register DstReg = MI.getOperand(0).getReg();
  if (!TRI->isSGPRClass(MRI.getRegClass(DstReg)))
    return false;

  if (!MRI.hasOneUse(DstReg))
    return false;

  MachineInstr &CopyUse = *MRI.use_instr_begin(DstReg);
  if (!CopyUse.isCopy())
    return false;

  // It is illegal to have vreg inputs to a physreg defining reg_sequence.
  if (CopyUse.getOperand(0).getReg().isPhysical())
    return false;

  const TargetRegisterClass *SrcRC, *DstRC;
  std::tie(SrcRC, DstRC) = getCopyRegClasses(CopyUse, *TRI, MRI);

  if (!isSGPRToVGPRCopy(SrcRC, DstRC, *TRI))
    return false;

  if (tryChangeVGPRtoSGPRinCopy(CopyUse, TRI, TII))
    return true;

  // TODO: Could have multiple extracts?
  unsigned SubReg = CopyUse.getOperand(1).getSubReg();
  if (SubReg != AMDGPU::NoSubRegister)
    return false;

  MRI.setRegClass(DstReg, DstRC);

  // SGPRx = ...
  // SGPRy = REG_SEQUENCE SGPRx, sub0 ...
  // VGPRz = COPY SGPRy

  // =>
  // VGPRx = COPY SGPRx
  // VGPRz = REG_SEQUENCE VGPRx, sub0

  MI.getOperand(0).setReg(CopyUse.getOperand(0).getReg());
  bool IsAGPR = TRI->isAGPRClass(DstRC);

  for (unsigned I = 1, N = MI.getNumOperands(); I != N; I += 2) {
    Register SrcReg = MI.getOperand(I).getReg();
    unsigned SrcSubReg = MI.getOperand(I).getSubReg();

    const TargetRegisterClass *SrcRC = MRI.getRegClass(SrcReg);
    assert(TRI->isSGPRClass(SrcRC) &&
           "Expected SGPR REG_SEQUENCE to only have SGPR inputs");

    SrcRC = TRI->getSubRegClass(SrcRC, SrcSubReg);
    const TargetRegisterClass *NewSrcRC = TRI->getEquivalentVGPRClass(SrcRC);

    Register TmpReg = MRI.createVirtualRegister(NewSrcRC);

    BuildMI(*MI.getParent(), &MI, MI.getDebugLoc(), TII->get(AMDGPU::COPY),
            TmpReg)
        .add(MI.getOperand(I));

    if (IsAGPR) {
      const TargetRegisterClass *NewSrcRC = TRI->getEquivalentAGPRClass(SrcRC);
      Register TmpAReg = MRI.createVirtualRegister(NewSrcRC);
      unsigned Opc = NewSrcRC == &AMDGPU::AGPR_32RegClass ?
        AMDGPU::V_ACCVGPR_WRITE_B32_e64 : AMDGPU::COPY;
      BuildMI(*MI.getParent(), &MI, MI.getDebugLoc(), TII->get(Opc),
            TmpAReg)
        .addReg(TmpReg, RegState::Kill);
      TmpReg = TmpAReg;
    }

    MI.getOperand(I).setReg(TmpReg);
  }

  CopyUse.eraseFromParent();
  return true;
}

static bool isSafeToFoldImmIntoCopy(const MachineInstr *Copy,
                                    const MachineInstr *MoveImm,
                                    const SIInstrInfo *TII,
                                    unsigned &SMovOp,
                                    int64_t &Imm) {
  if (Copy->getOpcode() != AMDGPU::COPY)
    return false;

  if (!MoveImm->isMoveImmediate())
    return false;

  const MachineOperand *ImmOp =
      TII->getNamedOperand(*MoveImm, AMDGPU::OpName::src0);
  if (!ImmOp->isImm())
    return false;

  // FIXME: Handle copies with sub-regs.
  if (Copy->getOperand(0).getSubReg())
    return false;

  switch (MoveImm->getOpcode()) {
  default:
    return false;
  case AMDGPU::V_MOV_B32_e32:
    SMovOp = AMDGPU::S_MOV_B32;
    break;
  case AMDGPU::V_MOV_B64_PSEUDO:
    SMovOp = AMDGPU::S_MOV_B64;
    break;
  }
  Imm = ImmOp->getImm();
  return true;
}

template <class UnaryPredicate>
bool searchPredecessors(const MachineBasicBlock *MBB,
                        const MachineBasicBlock *CutOff,
                        UnaryPredicate Predicate) {
  if (MBB == CutOff)
    return false;

  DenseSet<const MachineBasicBlock *> Visited;
  SmallVector<MachineBasicBlock *, 4> Worklist(MBB->predecessors());

  while (!Worklist.empty()) {
    MachineBasicBlock *MBB = Worklist.pop_back_val();

    if (!Visited.insert(MBB).second)
      continue;
    if (MBB == CutOff)
      continue;
    if (Predicate(MBB))
      return true;

    Worklist.append(MBB->pred_begin(), MBB->pred_end());
  }

  return false;
}

// Checks if there is potential path From instruction To instruction.
// If CutOff is specified and it sits in between of that path we ignore
// a higher portion of the path and report it is not reachable.
static bool isReachable(const MachineInstr *From,
                        const MachineInstr *To,
                        const MachineBasicBlock *CutOff,
                        MachineDominatorTree &MDT) {
  if (MDT.dominates(From, To))
    return true;

  const MachineBasicBlock *MBBFrom = From->getParent();
  const MachineBasicBlock *MBBTo = To->getParent();

  // Do predecessor search.
  // We should almost never get here since we do not usually produce M0 stores
  // other than -1.
  return searchPredecessors(MBBTo, CutOff, [MBBFrom]
           (const MachineBasicBlock *MBB) { return MBB == MBBFrom; });
}

// Return the first non-prologue instruction in the block.
static MachineBasicBlock::iterator
getFirstNonPrologue(MachineBasicBlock *MBB, const TargetInstrInfo *TII) {
  MachineBasicBlock::iterator I = MBB->getFirstNonPHI();
  while (I != MBB->end() && TII->isBasicBlockPrologue(*I))
    ++I;

  return I;
}

// Hoist and merge identical SGPR initializations into a common predecessor.
// This is intended to combine M0 initializations, but can work with any
// SGPR. A VGPR cannot be processed since we cannot guarantee vector
// executioon.
static bool hoistAndMergeSGPRInits(unsigned Reg,
                                   const MachineRegisterInfo &MRI,
                                   const TargetRegisterInfo *TRI,
                                   MachineDominatorTree &MDT,
                                   const TargetInstrInfo *TII) {
  // List of inits by immediate value.
  using InitListMap = std::map<unsigned, std::list<MachineInstr *>>;
  InitListMap Inits;
  // List of clobbering instructions.
  SmallVector<MachineInstr*, 8> Clobbers;
  // List of instructions marked for deletion.
  SmallSet<MachineInstr*, 8> MergedInstrs;

  bool Changed = false;

  for (auto &MI : MRI.def_instructions(Reg)) {
    MachineOperand *Imm = nullptr;
    for (auto &MO : MI.operands()) {
      if ((MO.isReg() && ((MO.isDef() && MO.getReg() != Reg) || !MO.isDef())) ||
          (!MO.isImm() && !MO.isReg()) || (MO.isImm() && Imm)) {
        Imm = nullptr;
        break;
      } else if (MO.isImm())
        Imm = &MO;
    }
    if (Imm)
      Inits[Imm->getImm()].push_front(&MI);
    else
      Clobbers.push_back(&MI);
  }

  for (auto &Init : Inits) {
    auto &Defs = Init.second;

    for (auto I1 = Defs.begin(), E = Defs.end(); I1 != E; ) {
      MachineInstr *MI1 = *I1;

      for (auto I2 = std::next(I1); I2 != E; ) {
        MachineInstr *MI2 = *I2;

        // Check any possible interference
        auto interferes = [&](MachineBasicBlock::iterator From,
                              MachineBasicBlock::iterator To) -> bool {

          assert(MDT.dominates(&*To, &*From));

          auto interferes = [&MDT, From, To](MachineInstr* &Clobber) -> bool {
            const MachineBasicBlock *MBBFrom = From->getParent();
            const MachineBasicBlock *MBBTo = To->getParent();
            bool MayClobberFrom = isReachable(Clobber, &*From, MBBTo, MDT);
            bool MayClobberTo = isReachable(Clobber, &*To, MBBTo, MDT);
            if (!MayClobberFrom && !MayClobberTo)
              return false;
            if ((MayClobberFrom && !MayClobberTo) ||
                (!MayClobberFrom && MayClobberTo))
              return true;
            // Both can clobber, this is not an interference only if both are
            // dominated by Clobber and belong to the same block or if Clobber
            // properly dominates To, given that To >> From, so it dominates
            // both and located in a common dominator.
            return !((MBBFrom == MBBTo &&
                      MDT.dominates(Clobber, &*From) &&
                      MDT.dominates(Clobber, &*To)) ||
                     MDT.properlyDominates(Clobber->getParent(), MBBTo));
          };

          return (llvm::any_of(Clobbers, interferes)) ||
                 (llvm::any_of(Inits, [&](InitListMap::value_type &C) {
                    return C.first != Init.first &&
                           llvm::any_of(C.second, interferes);
                  }));
        };

        if (MDT.dominates(MI1, MI2)) {
          if (!interferes(MI2, MI1)) {
            LLVM_DEBUG(dbgs()
                       << "Erasing from "
                       << printMBBReference(*MI2->getParent()) << " " << *MI2);
            MergedInstrs.insert(MI2);
            Changed = true;
            ++I2;
            continue;
          }
        } else if (MDT.dominates(MI2, MI1)) {
          if (!interferes(MI1, MI2)) {
            LLVM_DEBUG(dbgs()
                       << "Erasing from "
                       << printMBBReference(*MI1->getParent()) << " " << *MI1);
            MergedInstrs.insert(MI1);
            Changed = true;
            ++I1;
            break;
          }
        } else {
          auto *MBB = MDT.findNearestCommonDominator(MI1->getParent(),
                                                     MI2->getParent());
          if (!MBB) {
            ++I2;
            continue;
          }

          MachineBasicBlock::iterator I = getFirstNonPrologue(MBB, TII);
          if (!interferes(MI1, I) && !interferes(MI2, I)) {
            LLVM_DEBUG(dbgs()
                       << "Erasing from "
                       << printMBBReference(*MI1->getParent()) << " " << *MI1
                       << "and moving from "
                       << printMBBReference(*MI2->getParent()) << " to "
                       << printMBBReference(*I->getParent()) << " " << *MI2);
            I->getParent()->splice(I, MI2->getParent(), MI2);
            MergedInstrs.insert(MI1);
            Changed = true;
            ++I1;
            break;
          }
        }
        ++I2;
      }
      ++I1;
    }
  }

  // Remove initializations that were merged into another.
  for (auto &Init : Inits) {
    auto &Defs = Init.second;
    auto I = Defs.begin();
    while (I != Defs.end()) {
      if (MergedInstrs.count(*I)) {
        (*I)->eraseFromParent();
        I = Defs.erase(I);
      } else
        ++I;
    }
  }

  // Try to schedule SGPR initializations as early as possible in the MBB.
  for (auto &Init : Inits) {
    auto &Defs = Init.second;
    for (auto MI : Defs) {
      auto MBB = MI->getParent();
      MachineInstr &BoundaryMI = *getFirstNonPrologue(MBB, TII);
      MachineBasicBlock::reverse_iterator B(BoundaryMI);
      // Check if B should actually be a boundary. If not set the previous
      // instruction as the boundary instead.
      if (!TII->isBasicBlockPrologue(*B))
        B++;

      auto R = std::next(MI->getReverseIterator());
      const unsigned Threshold = 50;
      // Search until B or Threshold for a place to insert the initialization.
      for (unsigned I = 0; R != B && I < Threshold; ++R, ++I)
        if (R->readsRegister(Reg, TRI) || R->definesRegister(Reg, TRI) ||
            TII->isSchedulingBoundary(*R, MBB, *MBB->getParent()))
          break;

      // Move to directly after R.
      if (&*--R != MI)
        MBB->splice(*R, MBB, MI);
    }
  }

  if (Changed)
    MRI.clearKillFlags(Reg);

  return Changed;
}

bool SIFixSGPRCopies::runOnMachineFunction(MachineFunction &MF) {
  // Only need to run this in SelectionDAG path.
  if (MF.getProperties().hasProperty(
        MachineFunctionProperties::Property::Selected))
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  MRI = &MF.getRegInfo();
  TRI = ST.getRegisterInfo();
  TII = ST.getInstrInfo();
  MDT = &getAnalysis<MachineDominatorTree>();

  testIterative1(MF);

  collectV2SCopyInfo(MF);
  lowerV2SCopies(MF);

  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
                                                  BI != BE; ++BI) {
    MachineBasicBlock *MBB = &*BI;
    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E;
         ++I) {
      MachineInstr &MI = *I;

      switch (MI.getOpcode()) {
      default:
        continue;
      case AMDGPU::COPY:
      case AMDGPU::WQM:
      case AMDGPU::STRICT_WQM:
      case AMDGPU::SOFT_WQM:
      case AMDGPU::STRICT_WWM: {
        Register DstReg = MI.getOperand(0).getReg();
        const TargetRegisterClass *SrcRC, *DstRC;
        std::tie(SrcRC, DstRC) = getCopyRegClasses(MI, *TRI, *MRI);

        if (MI.isCopy()) {
          Register SrcReg = MI.getOperand(1).getReg();
          if (SrcReg == AMDGPU::SCC) {
            Register SCCCopy = MRI->createVirtualRegister(
                TRI->getRegClass(AMDGPU::SReg_1_XEXECRegClassID));
            I = BuildMI(*MI.getParent(),
                        std::next(MachineBasicBlock::iterator(MI)),
                        MI.getDebugLoc(),
                        TII->get(ST.isWave32() ? AMDGPU::S_CSELECT_B32
                                               : AMDGPU::S_CSELECT_B64),
                        SCCCopy)
                    .addImm(-1)
                    .addImm(0);
            I = BuildMI(*MI.getParent(), std::next(I), I->getDebugLoc(),
                        TII->get(AMDGPU::COPY), DstReg)
                    .addReg(SCCCopy);
            MI.eraseFromParent();
            continue;
          } else if (DstReg == AMDGPU::SCC) {
            unsigned Opcode =
                ST.isWave64() ? AMDGPU::S_AND_B64 : AMDGPU::S_AND_B32;
            Register Exec = ST.isWave64() ? AMDGPU::EXEC : AMDGPU::EXEC_LO;
            Register Tmp = MRI->createVirtualRegister(TRI->getBoolRC());
            I = BuildMI(*MI.getParent(),
                        std::next(MachineBasicBlock::iterator(MI)),
                        MI.getDebugLoc(), TII->get(Opcode))
                    .addReg(Tmp, getDefRegState(true))
                    .addReg(SrcReg)
                    .addReg(Exec);
            MI.eraseFromParent();
            continue;
          }
        }

        if (!DstReg.isVirtual()) {
          // If the destination register is a physical register there isn't
          // really much we can do to fix this.
          // Some special instructions use M0 as an input. Some even only use
          // the first lane. Insert a readfirstlane and hope for the best.
          if (DstReg == AMDGPU::M0 && TRI->hasVectorRegisters(SrcRC)) {
            Register TmpReg
              = MRI->createVirtualRegister(&AMDGPU::SReg_32_XM0RegClass);

            BuildMI(*MBB, MI, MI.getDebugLoc(),
                    TII->get(AMDGPU::V_READFIRSTLANE_B32), TmpReg)
                .add(MI.getOperand(1));
            MI.getOperand(1).setReg(TmpReg);
          }

          continue;
        }

        if (isSGPRToVGPRCopy(SrcRC, DstRC, *TRI)) {
          tryChangeVGPRtoSGPRinCopy(MI, TRI, TII);
        }

        break;
      }
      case AMDGPU::PHI: {
        MachineBasicBlock *NewBB = processPHINode(MI);
        if (NewBB && NewBB != MBB) {
          MBB = NewBB;
          E = MBB->end();
          BI = MachineFunction::iterator(MBB);
          BE = MF.end();
        }
        assert((!NewBB || NewBB == I->getParent()) &&
               "moveToVALU did not return the right basic block");
        break;
      }
      case AMDGPU::REG_SEQUENCE: {
        if (TRI->hasVectorRegisters(TII->getOpRegClass(MI, 0)) ||
            !hasVectorOperands(MI, TRI)) {
          foldVGPRCopyIntoRegSequence(MI, TRI, TII, *MRI);
          continue;
        }

        LLVM_DEBUG(dbgs() << "Fixing REG_SEQUENCE: " << MI);

        MachineBasicBlock *NewBB = TII->moveToVALU(MI, MDT);
        if (NewBB && NewBB != MBB) {
          MBB = NewBB;
          E = MBB->end();
          BI = MachineFunction::iterator(MBB);
          BE = MF.end();
        }
        assert((!NewBB || NewBB == I->getParent()) &&
               "moveToVALU did not return the right basic block");
        break;
      }
      case AMDGPU::INSERT_SUBREG: {
        const TargetRegisterClass *DstRC, *Src0RC, *Src1RC;
        DstRC = MRI->getRegClass(MI.getOperand(0).getReg());
        Src0RC = MRI->getRegClass(MI.getOperand(1).getReg());
        Src1RC = MRI->getRegClass(MI.getOperand(2).getReg());
        if (TRI->isSGPRClass(DstRC) &&
            (TRI->hasVectorRegisters(Src0RC) ||
             TRI->hasVectorRegisters(Src1RC))) {
          LLVM_DEBUG(dbgs() << " Fixing INSERT_SUBREG: " << MI);
          MachineBasicBlock *NewBB = TII->moveToVALU(MI, MDT);
          if (NewBB && NewBB != MBB) {
            MBB = NewBB;
            E = MBB->end();
            BI = MachineFunction::iterator(MBB);
            BE = MF.end();
          }
          assert((!NewBB || NewBB == I->getParent()) &&
                 "moveToVALU did not return the right basic block");
        }
        break;
      }
      case AMDGPU::V_WRITELANE_B32: {
        // Some architectures allow more than one constant bus access without
        // SGPR restriction
        if (ST.getConstantBusLimit(MI.getOpcode()) != 1)
          break;

        // Writelane is special in that it can use SGPR and M0 (which would
        // normally count as using the constant bus twice - but in this case it
        // is allowed since the lane selector doesn't count as a use of the
        // constant bus). However, it is still required to abide by the 1 SGPR
        // rule. Apply a fix here as we might have multiple SGPRs after
        // legalizing VGPRs to SGPRs
        int Src0Idx =
            AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::src0);
        int Src1Idx =
            AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::src1);
        MachineOperand &Src0 = MI.getOperand(Src0Idx);
        MachineOperand &Src1 = MI.getOperand(Src1Idx);

        // Check to see if the instruction violates the 1 SGPR rule
        if ((Src0.isReg() && TRI->isSGPRReg(*MRI, Src0.getReg()) &&
             Src0.getReg() != AMDGPU::M0) &&
            (Src1.isReg() && TRI->isSGPRReg(*MRI, Src1.getReg()) &&
             Src1.getReg() != AMDGPU::M0)) {

          // Check for trivially easy constant prop into one of the operands
          // If this is the case then perform the operation now to resolve SGPR
          // issue. If we don't do that here we will always insert a mov to m0
          // that can't be resolved in later operand folding pass
          bool Resolved = false;
          for (MachineOperand *MO : {&Src0, &Src1}) {
            if (MO->getReg().isVirtual()) {
              MachineInstr *DefMI = MRI->getVRegDef(MO->getReg());
              if (DefMI && TII->isFoldableCopy(*DefMI)) {
                const MachineOperand &Def = DefMI->getOperand(0);
                if (Def.isReg() &&
                    MO->getReg() == Def.getReg() &&
                    MO->getSubReg() == Def.getSubReg()) {
                  const MachineOperand &Copied = DefMI->getOperand(1);
                  if (Copied.isImm() &&
                      TII->isInlineConstant(APInt(64, Copied.getImm(), true))) {
                    MO->ChangeToImmediate(Copied.getImm());
                    Resolved = true;
                    break;
                  }
                }
              }
            }
          }

          if (!Resolved) {
            // Haven't managed to resolve by replacing an SGPR with an immediate
            // Move src1 to be in M0
            BuildMI(*MI.getParent(), MI, MI.getDebugLoc(),
                    TII->get(AMDGPU::COPY), AMDGPU::M0)
                .add(Src1);
            Src1.ChangeToRegister(AMDGPU::M0, false);
          }
        }
        break;
      }
      }
    }
  }

  if (MF.getTarget().getOptLevel() > CodeGenOpt::None && EnableM0Merge)
    hoistAndMergeSGPRInits(AMDGPU::M0, *MRI, TRI, *MDT, TII);

  return true;
}

MachineBasicBlock *SIFixSGPRCopies::processPHINode(MachineInstr &MI) {
  unsigned numVGPRUses = 0;
  bool AllAGPRUses = true;
  SetVector<const MachineInstr *> worklist;
  SmallSet<const MachineInstr *, 4> Visited;
  SetVector<MachineInstr *> PHIOperands;
  MachineBasicBlock *CreatedBB = nullptr;
  worklist.insert(&MI);
  Visited.insert(&MI);
  while (!worklist.empty()) {
    const MachineInstr *Instr = worklist.pop_back_val();
    Register Reg = Instr->getOperand(0).getReg();
    for (const auto &Use : MRI->use_operands(Reg)) {
      const MachineInstr *UseMI = Use.getParent();
      AllAGPRUses &= (UseMI->isCopy() &&
                      TRI->isAGPR(*MRI, UseMI->getOperand(0).getReg())) ||
                     TRI->isAGPR(*MRI, Use.getReg());
      if (UseMI->isCopy() || UseMI->isRegSequence()) {
        if (UseMI->isCopy() &&
          UseMI->getOperand(0).getReg().isPhysical() &&
          !TRI->isSGPRReg(*MRI, UseMI->getOperand(0).getReg())) {
          numVGPRUses++;
        }
        if (Visited.insert(UseMI).second)
          worklist.insert(UseMI);

        continue;
      }

      if (UseMI->isPHI()) {
        const TargetRegisterClass *UseRC = MRI->getRegClass(Use.getReg());
        if (!TRI->isSGPRReg(*MRI, Use.getReg()) &&
          UseRC != &AMDGPU::VReg_1RegClass)
          numVGPRUses++;
        continue;
      }

      const TargetRegisterClass *OpRC =
        TII->getOpRegClass(*UseMI, UseMI->getOperandNo(&Use));
      if (!TRI->isSGPRClass(OpRC) && OpRC != &AMDGPU::VS_32RegClass &&
        OpRC != &AMDGPU::VS_64RegClass) {
        numVGPRUses++;
      }
    }
  }

  Register PHIRes = MI.getOperand(0).getReg();
  const TargetRegisterClass *RC0 = MRI->getRegClass(PHIRes);
  if (AllAGPRUses && numVGPRUses && !TRI->isAGPRClass(RC0)) {
    LLVM_DEBUG(dbgs() << "Moving PHI to AGPR: " << MI);
    MRI->setRegClass(PHIRes, TRI->getEquivalentAGPRClass(RC0));
    for (unsigned I = 1, N = MI.getNumOperands(); I != N; I += 2) {
      MachineInstr *DefMI = MRI->getVRegDef(MI.getOperand(I).getReg());
      if (DefMI && DefMI->isPHI())
        PHIOperands.insert(DefMI);
    }
  }

  bool hasVGPRInput = false;
  for (unsigned i = 1; i < MI.getNumOperands(); i += 2) {
    Register InputReg = MI.getOperand(i).getReg();
    MachineInstr *Def = MRI->getVRegDef(InputReg);
    if (TRI->isVectorRegister(*MRI, InputReg)) {
      if (Def->isCopy()) {
        Register SrcReg = Def->getOperand(1).getReg();
        const TargetRegisterClass *RC =
          TRI->getRegClassForReg(*MRI, SrcReg);
        if (TRI->isSGPRClass(RC))
          continue;
      }
      hasVGPRInput = true;
      break;
    }
    else if (Def->isCopy() &&
      TRI->isVectorRegister(*MRI, Def->getOperand(1).getReg())) {
      Register SrcReg = Def->getOperand(1).getReg();
      MachineInstr *SrcDef = MRI->getVRegDef(SrcReg);
      unsigned SMovOp;
      int64_t Imm;
      if (!isSafeToFoldImmIntoCopy(Def, SrcDef, TII, SMovOp, Imm)) {
        hasVGPRInput = true;
        break;
      } else {
        // Formally, if we did not do this right away
        // it would be done on the next iteration of the
        // runOnMachineFunction main loop. But why not if we can?
        MachineFunction *MF = MI.getParent()->getParent();
        Def->getOperand(1).ChangeToImmediate(Imm);
        Def->addImplicitDefUseOperands(*MF);
        Def->setDesc(TII->get(SMovOp));
      }
    }
  }

  if ((!TRI->isVectorRegister(*MRI, PHIRes) &&
       RC0 != &AMDGPU::VReg_1RegClass) &&
    (hasVGPRInput || numVGPRUses > 1)) {
    LLVM_DEBUG(dbgs() << "Fixing PHI: " << MI);
    CreatedBB = TII->moveToVALU(MI);
  }
  else {
    LLVM_DEBUG(dbgs() << "Legalizing PHI: " << MI);
    TII->legalizeOperands(MI, MDT);
  }

  // Propagate register class back to PHI operands which are PHI themselves.
  while (!PHIOperands.empty()) {
    processPHINode(*PHIOperands.pop_back_val());
  }
  return CreatedBB;
}

void SIFixSGPRCopies::collectV2SCopyInfo(MachineFunction &MF) {
  V2SCopyInfo::clear();
  DenseMap<MachineInstr *, UserInfo> Instructions;
  auto needProcessing = [](MachineInstr &MI) -> bool {
    switch (MI.getOpcode()) {
    case AMDGPU::COPY:
    case AMDGPU::WQM:
    case AMDGPU::STRICT_WQM:
    case AMDGPU::SOFT_WQM:
    case AMDGPU::STRICT_WWM:
      return true;
    default:
      return false;
    }
  };
  auto trackDown = [&](const auto &self, V2SCopyInfo *Copy, MachineInstr *Root,
                       unsigned &SScore, unsigned &VScore, unsigned &SVScore,
                       SetVector<MachineInstr *>& Reaching,
                       SetVector<MachineInstr *>& Visited) {
    if (!Visited.insert(Root))
      return;
    //if (Instructions.find(Root) != Instructions.end()) {
    if (UserInfo UInfo = Instructions.lookup(Root)) {

      //UserInfo UInfo = Instructions.find(Root)->getSecond();
      MachineInstr *SiblingCopy = UInfo.getCopy();
      if (SiblingCopy != Copy->getCopy()) {

        V2SCopyInfo& SInfo = V2SCopyInfo::getV2SCopy(SiblingCopy);

        for (auto &S : SInfo.getSiblings()) {
          if (UserInfo SCUInfo = Instructions.lookup(S.CommonUser)) {
            V2SCopyInfo &SSInfo = V2SCopyInfo::getV2SCopy(S.SiblingCopy);
            if (UInfo.Reaching(S.CommonUser)) {
              Copy->addSibling(S.SiblingCopy, S.CommonUser, SCUInfo, true);
              SSInfo.addSibling(Copy->getCopy(), S.CommonUser, SCUInfo);
            } else if (SCUInfo.Reaching(Root)) {
              V2SCopyInfo &SSInfo = V2SCopyInfo::getV2SCopy(S.SiblingCopy);
              Copy->addSibling(SSInfo.getCopy(), Root, UInfo);
              SSInfo.addSibling(Copy->getCopy(), Root, UInfo);
            }
          }
        }

        SInfo.addSibling(Copy->getCopy(), Root, UInfo);
        Copy->addSibling(SiblingCopy, Root, UInfo, true);

      }
      SScore += UInfo.getSScore();
      VScore += UInfo.getVScore();
      SVScore += UInfo.getSVcopies();
    } else {
      unsigned STmp = SScore, VTmp = VScore, SVTmp = SVScore;
      SetVector<MachineInstr *> List;
      //UserInfo UInfo(Copy->getCopy());
      if (Root->getNumExplicitDefs() != 0) {
        Register Reg = Root->getOperand(0).getReg();
        for (auto &U : MRI->use_instructions(Reg)) {
          //if (UserInfo Cached = Instructions[&U]) {
           // dbgs() << "TEST\n";
          //  SScore += Cached.getSScore();
          //  VScore += Cached.getVScore();
          //  SVScore += Cached.getSVcopies();
          //  Reaching.insert(Cached.getDescendants().begin(),
          //                  Cached.getDescendants().end());
          //}
          if (!U.isCopy() && !U.isRegSequence()) {
            if (TRI->isSGPRReg(*MRI, Reg)) {
              Copy->incSALU();
              SScore++;
            } else {
              Copy->incVALU();
              VScore++;
            }
          } else if (TRI->isVGPR(*MRI, U.getOperand(0).getReg())) {
            Copy->incSVCopies();
            SVScore++;
          }
          List.clear();
          self(self, Copy, &U, SScore, VScore, SVScore, List, Visited);
          Reaching.insert(List.begin(), List.end());
          //UInfo.addDescendants(List);
        }
      }
      //UInfo.addDescendants(Reaching);
      //UInfo.setScores(SScore - STmp, VScore - VTmp, SVScore - SVTmp);
      Instructions.insert(std::make_pair(
          Root, UserInfo(Copy->getCopy(), SScore - STmp, VScore - VTmp,
                         SVScore - SVTmp, Reaching)));
      Reaching.insert(Root);
    }
  };
  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end(); BI != BE;
       ++BI) {
    MachineBasicBlock *MBB = &*BI;
    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E;
         ++I) {
      MachineInstr &MI = *I;
      if (!needProcessing(MI))
        continue;
      if (!LowerSpecialCase(MI)) {
        V2SCopyInfo CopyInfo(&MI);

        SetVector<MachineInstr *> Visited, Reaching;
        Register Reg = MI.getOperand(0).getReg();
        unsigned S = 0, V = 0, VS = 0;
        for (auto &U : MRI->use_instructions(Reg)) {
          // Reg here is always scalar. Hence ++S on each user.
          CopyInfo.incSALU();
          trackDown(trackDown, &CopyInfo, &U, ++S, V, VS, Reaching, Visited);
        }
        LLVM_DEBUG(for (auto &UI
                        : Instructions) {
          dbgs() << "\nInst: " << *UI.getFirst();
          UI.getSecond().dump();
        });
        V2SCopyInfo::addV2SCopy(&MI, CopyInfo);
      }
    }
  }
  LLVM_DEBUG(SIFixSGPRCopies::V2SCopyInfo::print());
}

// REMOVE ME: temporary stuff to check the ISel correctness!

static void checkIfDefMILegal(MachineRegisterInfo *MRI, Register CopySource,
                              Register CopyDest) {
  MachineInstr *DefMI = MRI->getVRegDef(CopySource);
  if (DefMI->isDivergent()) {
    for (auto &U : MRI->use_instructions(CopyDest)) {
      unsigned Opc = U.getOpcode();
      if (Opc != AMDGPU::V_WRITELANE_B32 &&
          Opc != AMDGPU::S_BUFFER_LOAD_DWORD_IMM &&
          Opc != AMDGPU::BUFFER_LOAD_FORMAT_X_OFFSET &&
          Opc != AMDGPU::BUFFER_LOAD_FORMAT_X_IDXEN &&
          Opc != AMDGPU::BUFFER_LOAD_FORMAT_X_OFFEN &&
          Opc != AMDGPU::BUFFER_LOAD_FORMAT_X_BOTHEN &&
          Opc != AMDGPU::IMAGE_SAMPLE_V1_V2) {
        dbgs() << *DefMI->getParent() << "\n" << DefMI;
        assert(false && "Error porcessing VGPR2SGPR copy\n");
      }
    }
  }
}

bool SIFixSGPRCopies::LowerSpecialCase(MachineInstr &MI) {
  MachineBasicBlock *MBB = MI.getParent();
  const TargetRegisterClass *SrcRC, *DstRC;
  std::tie(SrcRC, DstRC) = getCopyRegClasses(MI, *TRI, *MRI);

  // We return true to indicate that no further processing needed
  if (!isVGPRToSGPRCopy(SrcRC, DstRC, *TRI))
    return true;

  Register SrcReg = MI.getOperand(1).getReg();
  Register DstReg = MI.getOperand(0).getReg();
  if (!SrcReg.isVirtual() || TRI->isAGPR(*MRI, SrcReg)) {
    TII->moveToVALU(MI, MDT);
    return true;
  }

  checkIfDefMILegal(MRI, SrcReg, DstReg);

  unsigned SMovOp;
  int64_t Imm;
  // If we are just copying an immediate, we can replace the copy with
  // s_mov_b32.
  if (isSafeToFoldImmIntoCopy(&MI, MRI->getVRegDef(SrcReg), TII, SMovOp, Imm)) {
    MI.getOperand(1).ChangeToImmediate(Imm);
    MI.addImplicitDefUseOperands(*MBB->getParent());
    MI.setDesc(TII->get(SMovOp));
    return true;
  }
  return false;
}

void SIFixSGPRCopies::lowerV2SCopies(MachineFunction &MF) {
  for (auto &V : V2SCopyInfo::getSorted()) {
    MachineInstr *MI = V.getCopy();
    MachineBasicBlock *MBB = MI->getParent();
    if (V2SCopyInfo::getV2SCopy(MI).getScore() > 2) {
      // We decide to turn V2S copy to v_readfirstlanre_b32
      // remove it from the V2SCopies and remove it from all its siblings
      LLVM_DEBUG(dbgs() << "V2S copy " << *MI
                        << " is being turned to v_readfirstlane_b32");
      uint16_t SubRegs[4] = {AMDGPU::sub0, AMDGPU::sub1, AMDGPU::sub2,
                             AMDGPU::sub3};
      Register DstReg = MI->getOperand(0).getReg();
      Register SrcReg = MI->getOperand(1).getReg();
      unsigned SubReg = MI->getOperand(1).getSubReg();
      bool IsSubReg = SubReg != AMDGPU::NoSubRegister;
      const TargetRegisterClass *SrcRC = TRI->getRegClassForReg(*MRI, SrcReg);
      if (IsSubReg)
        SrcRC = TRI->getSubRegClass(SrcRC, SubReg);
      if (TRI->getRegSizeInBits(*SrcRC) == 32) {
        auto MIB = BuildMI(*MBB, MI, MI->getDebugLoc(),
                           TII->get(AMDGPU::V_READFIRSTLANE_B32), DstReg);
        if (IsSubReg)
          MIB.addReg(SrcReg, 0, SubReg);
        else
          MIB.addReg(SrcReg);
      } else {
        auto Result = BuildMI(*MBB, MI, MI->getDebugLoc(),
                              TII->get(AMDGPU::REG_SEQUENCE), DstReg);
        int N = TRI->getRegSizeInBits(*SrcRC) / 32;
        for (int i = 0; i < N; i++) {
          Register PartialSrc =
              TII->buildExtractSubReg(Result, *MRI, MI->getOperand(1), SrcRC,
                                      SubRegs[i], &AMDGPU::VGPR_32RegClass);
          Register PartialDst =
              MRI->createVirtualRegister(&AMDGPU::SReg_32RegClass);
          BuildMI(*MBB, *Result, Result->getDebugLoc(),
                  TII->get(AMDGPU::V_READFIRSTLANE_B32), PartialDst)
              .addReg(PartialSrc);
          Result.addReg(PartialDst).addImm(SubRegs[i]);
        }
      }
      MI->eraseFromParent();
    } else {
      // We decide to convert the V2S copy
      // and all its SSA subtree to VALU
      // We need to update its siblings scores
      // to reflect the change we've made
      // Also, remove all the related bookkeeping
      LLVM_DEBUG(dbgs() << "V2S copy " << MI << " is being turned to VALU\n");
      V2SCopyInfo::eraseV2SCopy(MI);

      TII->moveToVALU(*MI, MDT);
    }
  }
}

// V2SCopyInfo print methods
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void SIFixSGPRCopies::V2SCopyInfo::dump() const {
  print(dbgs());
}

LLVM_DUMP_METHOD void SIFixSGPRCopies::UserInfo::dump() const {
  print(dbgs());
}
#endif

void SIFixSGPRCopies::V2SCopyInfo::print(raw_ostream &OS) const {
  OS << "\nRoot V2S Copy: " << *Copy << "\n\t";
  OS << "SALU chain length: " << SALUChainLength << "\n\t";
  OS << "VALU chain length: " << VALUChainLength << "\n\t";
  OS << "S2V copies number: " << NumSVCopies << "\n";
  unsigned i = 0;
  for (auto &S : Siblings) {
    if (++i == 1)
      OS << "\n\tSiblings: \n";
    OS << "\t\t" << i << ": " << *S.SiblingCopy;
    OS << "\t\t\tCommonUser: " << *S.CommonUser;
    OS << "\t\t\tCommon user SSA subtree has: \n\t\t\t" << S.CommonUserSALUScore
       << " SALU users and\n\t\t\t";
    OS << S.CommonUserVALUScore << " VALU users and\n\t\t\t";
    OS << S.CommonUserS2VCopies << " S2V copies\n-------------------------\n";
  }
}

void SIFixSGPRCopies::UserInfo::print(raw_ostream &OS) const {
  dbgs() << "Info: \n\t"
         << " S:" << SScore
         << " V:" << VScore
         << " SV:" << SVCopies << "\n";
  dbgs() << "Descendants:\n\t";
  for (auto &D : Descendants)
    dbgs() << *D << "\t";
}

void SIFixSGPRCopies::V2SCopyInfo::print() {
  for (auto &I : V2SCopies) {
    I.getSecond().dump();
  }
}


void SIFixSGPRCopies::testIterative(MachineFunction& MF) {

  auto needProcessing = [](MachineInstr &MI) -> bool {
    switch (MI.getOpcode()) {
    case AMDGPU::COPY:
    case AMDGPU::WQM:
    case AMDGPU::STRICT_WQM:
    case AMDGPU::SOFT_WQM:
    case AMDGPU::STRICT_WWM:
      return true;
    default:
      return false;
    }
  };
  DenseMap<MachineInstr *, unsigned> SiblingPenaulty;
  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end(); BI != BE;
       ++BI) {
    MachineBasicBlock *MBB = &*BI;
    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E;
         ++I) {
      MachineInstr &MI = *I;
      if (!needProcessing(MI))
        continue;
      if (!LowerSpecialCase(MI)) {
        unsigned S = 0, V = 0, SV = 0;
        SmallVector<MachineInstr *, 8> worklist;
        SmallSet<MachineInstr *, 8> visited;
        worklist.push_back(&MI);
        unsigned i = 0;
        SmallVector<MachineInstr *, 8>::iterator I = worklist.begin();
        while (true) {
          MachineInstr *Inst = *I;
          if (visited.insert(Inst).second) {
            (*I)->dump();
            if (Inst->getNumExplicitDefs() != 0) {
              Register Reg = Inst->getOperand(0).getReg();
              for (auto &U : MRI->use_instructions(Reg)) {
                if (!U.isCopy() && !U.isRegSequence()) {
                  if (TRI->isSGPRReg(*MRI, Reg)) {
                    S++;
                  } else {
                    V++;
                  }
                } else if (TRI->isVGPR(*MRI, U.getOperand(0).getReg())) {
                  SV++;
                }
                worklist.push_back(&U);
              }
            }
          }

          if (*I == worklist.back())
            break;
          I = &worklist[++i];
        }
        dbgs() << "\nS:" << S << " V:" << V << " SV:" << SV << "\n";
      }
    }
  }

}


void SIFixSGPRCopies::testIterative1(MachineFunction &MF) {
  struct Info {
    MachineInstr *Copy;
    SmallVector<MachineInstr *, 8> SChain;
    unsigned S = 0, V = 0, SV = 0;
  };
  DenseMap<MachineInstr *, struct Info> Copies;
  DenseMap<MachineInstr *, unsigned> SiblingPenaulty;

  auto needProcessing = [](MachineInstr &MI) -> bool {
    switch (MI.getOpcode()) {
    case AMDGPU::COPY:
    case AMDGPU::WQM:
    case AMDGPU::STRICT_WQM:
    case AMDGPU::SOFT_WQM:
    case AMDGPU::STRICT_WWM:
      return true;
    default:
      return false;
    }
  };
  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end(); BI != BE;
       ++BI) {
    MachineBasicBlock *MBB = &*BI;
    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E;
         ++I) {
      MachineInstr &MI = *I;
      if (!needProcessing(MI))
        continue;
      if (!LowerSpecialCase(MI)) {
        Copies[&MI].Copy = &MI;
        SmallVector<MachineInstr *,8> worklist;
        DenseSet<MachineInstr *> visited;
        worklist.push_back(&MI);
        while (!worklist.empty()) {
          MachineInstr* Inst = worklist.pop_back_val();

          if (visited.insert(Inst).second) {
            if (SiblingPenaulty.count(Inst))
              SiblingPenaulty[Inst]++;
            else
              SiblingPenaulty[Inst] = 1;
            if (Inst->getNumExplicitDefs() != 0) {
              Register Reg = Inst->getOperand(0).getReg();
              for (auto &U : MRI->use_instructions(Reg)) {
                if (!U.isCopy() && !U.isRegSequence()) {
                  if (TRI->isSGPRReg(*MRI, Reg)) {
                    Copies[&MI].S++;
                    Copies[&MI].SChain.push_back(&U);
                  } else {
                    Copies[&MI].V++;
                  }
                } else if (TRI->isVGPR(*MRI, U.getOperand(0).getReg())) {
                  Copies[&MI].SV++;
                }
                worklist.push_back(&U);
              }
            }
          }
        }
        //dbgs() << "\nS:" << S << " V:" << V << " SV:" << SV << "\n";
      }
    }
  }
  for (auto &P : Copies) {
    auto Pred = [&](MachineInstr *A, MachineInstr *B) -> bool {
      return SiblingPenaulty[A] < SiblingPenaulty[B];
    };
    dbgs() << *P.first << "\n\tS:" << P.second.S << "\n\tV:" << P.second.V
           << "\n\tSV:" << P.second.SV;
    dbgs() << "\nSChain:\n";
    for (auto &U : P.second.SChain)
      dbgs() << *U;
    dbgs() << "Max SP: "
           << SiblingPenaulty[*std::max_element(P.second.SChain.begin(),
                                              P.second.SChain.end(), Pred)];
  }
}
