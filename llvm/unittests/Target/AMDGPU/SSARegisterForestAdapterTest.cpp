//===- SSARegisterForestAdapterTest.cpp - Target home mapping tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SSARegisterForestAdapter.h"
#include "AMDGPUTargetMachine.h"
#include "AMDGPUUnitTests.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "gtest/gtest.h"
#include <tuple>

using namespace llvm;

namespace {

TEST(SSARegisterForestAdapterTest, ReconcileCompletedRepairFromStoredOwnership) {
  auto TM = createAMDGPUTargetMachine("amdgcn-amd-", "gfx906", "");
  ASSERT_TRUE(TM);
  GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()), *TM);
  LLVMContext Context;
  Module M("forest-on-change", Context);
  M.setDataLayout(TM->createDataLayout());
  auto *FT = FunctionType::get(Type::getVoidTy(Context), false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "test", &M);
  MachineModuleInfo MMI(TM.get());
  MachineFunction MF(*F, *TM, ST, MMI.getContext(), 0);
  MF.initTargetMachineFunctionInfo(ST);
  auto &MRI = MF.getRegInfo();
  const auto &TRI = *ST.getRegisterInfo();
  const auto &TII = *ST.getInstrInfo();
  Register Pair = MRI.createVirtualRegister(&AMDGPU::VReg_64RegClass);
  Register Retired = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  Register Unchanged = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  Register New = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  auto *BB = MF.CreateMachineBasicBlock();
  MF.push_back(BB);
  MachineInstr *First = BuildMI(*BB, BB->end(), DebugLoc(),
                                TII.get(AMDGPU::S_NOP)).addImm(0);
  MachineInstr *Middle = BuildMI(*BB, BB->end(), DebugLoc(),
                                 TII.get(AMDGPU::S_NOP)).addImm(0);
  MachineInstr *Last = BuildMI(*BB, BB->end(), DebugLoc(),
                               TII.get(AMDGPU::S_ENDPGM)).addImm(0);
  MRI.freezeReservedRegs();
  MachineFunctionAnalysisManager MFAM;
  MFAM.registerPass([] { return PassInstrumentationAnalysis(); });
  MFAM.registerPass([] { return SlotIndexesAnalysis(); });
  MFAM.registerPass([] { return MachineDominatorTreeAnalysis(); });
  MFAM.registerPass([] { return LiveIntervalsAnalysis(); });
  auto &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  SlotIndex A = LIS.getInstructionIndex(*First).getRegSlot();
  SlotIndex B = LIS.getInstructionIndex(*Middle).getRegSlot();
  SlotIndex D = LIS.getInstructionIndex(*Last).getRegSlot();
  auto Forest = SSARegisterForest::create(1, 8);
  ASSERT_TRUE(Forest);
  RegisterForestAdapter Adapter(
      *Forest, TRI, MRI,
      [](const TargetRegisterClass *) -> ArrayRef<MCPhysReg> { return {}; });

  // Store a tuple home, then retain only its low lane. Occupancy alone no
  // longer identifies that full home. replace() must update the owner index.
  LaneBitmask Low = TRI.getSubRegIndexLaneMask(AMDGPU::sub0);
  ASSERT_TRUE(Adapter.assign(Pair, AMDGPU::VGPR0_VGPR1, A, B));
  RegisterForestAdapter::RetainedRegion LowOnly[] = {
      {VRegMaskPair(Pair, Low), A, B}};
  ASSERT_TRUE(Adapter.replace(Pair, AMDGPU::VGPR0_VGPR1, A, B, LowOnly));
  EXPECT_EQ(Adapter.assignedHome(Pair), AMDGPU::VGPR0_VGPR1);
  EXPECT_FALSE(Adapter.assign(Pair, AMDGPU::VGPR2_VGPR3, A, B));
  EXPECT_FALSE(Forest->assign({4, 6}, A, B, Pair));
  EXPECT_EQ(Adapter.assignedHome(Pair), AMDGPU::VGPR0_VGPR1);
  ASSERT_TRUE(Adapter.assign(Retired, AMDGPU::VGPR0, B, D));
  ASSERT_TRUE(Adapter.assign(Unchanged, AMDGPU::VGPR2, A, D));
  ASSERT_TRUE(Forest->assign({2, 4}, B, D, SSARegisterForest::SELF_OWNED));
  const SSARegisterForest::Ownership *OriginalRecord = nullptr;
  Forest->visitOwnershipComponents([&](auto, const auto &O) {
    if (O.Owner == Unchanged)
      OriginalRecord = &O;
  });
  ASSERT_NE(OriginalRecord, nullptr);

  // Supply the completed repair's after-state. The retired owner has no LI;
  // the new value has an LI but must not acquire a physical assignment.
  auto SetAfter = [&](Register VR) -> LiveInterval & {
    if (LIS.hasInterval(VR))
      LIS.removeInterval(VR);
    auto &LI = LIS.createEmptyInterval(VR);
    LI.addSegment({A, D, LI.getNextValue(A, LIS.getVNInfoAllocator())});
    return LI;
  };
  auto &PairLI = SetAfter(Pair);
  auto *Sub = PairLI.createSubRange(LIS.getVNInfoAllocator(), Low);
  Sub->addSegment({A, D, Sub->getNextValue(A, LIS.getVNInfoAllocator())});
  SetAfter(Unchanged);
  SetAfter(New);
  if (LIS.hasInterval(Retired))
    LIS.removeInterval(Retired);
  // A malformed projection is an error, not a request to discard the home.
  // Put retirement first to exercise validation of the whole batch before edits.
  Sub->LaneMask = LaneBitmask::getNone();
  EXPECT_DEATH(Adapter.onChange({Retired, Pair}, LIS),
               "invalid repaired live-region projection");
  Sub->LaneMask = Low;
  auto Invalidated = Adapter.onChange({Pair, New, Unchanged, Retired, Pair}, LIS);
  ASSERT_EQ(Invalidated.size(), 1u);
  EXPECT_EQ(Invalidated[0], Retired);
  EXPECT_EQ(Adapter.assignedHome(Pair), AMDGPU::VGPR0_VGPR1);
  EXPECT_FALSE(Adapter.assignedHome(New));
  EXPECT_FALSE(Adapter.assignedHome(Retired));
  EXPECT_FALSE(Forest->visitOwnerAssignments(Retired, [](const auto &) {}));
  EXPECT_TRUE(Adapter.contains(Pair, AMDGPU::VGPR0, A, D));
  Forest->visitOwnershipComponents([&](auto, const auto &O) {
    if (O.Owner == Unchanged)
      EXPECT_EQ(&O, OriginalRecord);
  });

  // Repair now makes both tuple lanes live. Its high lane hits fixed occupancy;
  // invalidate the entire home, including the previously retained low lane.
  SetAfter(Pair);
  Invalidated = Adapter.onChange({Pair}, LIS);
  ASSERT_EQ(Invalidated.size(), 1u);
  EXPECT_EQ(Invalidated[0], Pair);
  EXPECT_FALSE(Adapter.assignedHome(Pair));
  EXPECT_FALSE(Forest->visitOwnerAssignments(Pair, [](const auto &) {}));
  EXPECT_TRUE(Adapter.isFree(AMDGPU::VGPR0, A, D));
  EXPECT_TRUE(Forest->contains({2, 4}, B, D, SSARegisterForest::SELF_OWNED));
  EXPECT_TRUE(Adapter.contains(Unchanged, AMDGPU::VGPR2, A, D));

  // Reuse the emptied owner entry. Empty occupancy still has a recorded home
  // and can be explicitly unassigned; a second removal must fail.
  ASSERT_TRUE(Adapter.assign(Pair, AMDGPU::VGPR0_VGPR1,
                             ArrayRef<RegisterForestAdapter::RetainedRegion>{}));
  EXPECT_EQ(Adapter.assignedHome(Pair), AMDGPU::VGPR0_VGPR1);
  EXPECT_TRUE(Adapter.unassign(Pair));
  EXPECT_FALSE(Adapter.assignedHome(Pair));
  EXPECT_FALSE(Adapter.unassign(Pair));

  // Two affected values grow into the same previously unoccupied time/lanes.
  // Neither proposal is installed first; both homes must be invalidated in
  // either notification order. The unrelated VGPR2 owner stays untouched.
  for (bool Reverse : {false, true}) {
    ASSERT_TRUE(Adapter.assign(Pair, AMDGPU::VGPR0_VGPR1, LowOnly));
    ASSERT_TRUE(Adapter.assign(New, AMDGPU::VGPR0, B, D));
    auto &LI = SetAfter(Pair);
    auto *LowRange = LI.createSubRange(LIS.getVNInfoAllocator(), Low);
    LowRange->addSegment({A, D, LowRange->getNextValue(A, LIS.getVNInfoAllocator())});
    SetAfter(New);
    SmallVector<Register, 2> Affected = Reverse
                                          ? SmallVector<Register, 2>{New, Pair}
                                          : SmallVector<Register, 2>{Pair, New};
    Invalidated = Adapter.onChange(Affected, LIS);
    ASSERT_EQ(Invalidated.size(), 2u);
    EXPECT_FALSE(Adapter.assignedHome(Pair));
    EXPECT_FALSE(Adapter.assignedHome(New));
    EXPECT_TRUE(Adapter.isFree(AMDGPU::VGPR0, A, D));
    EXPECT_TRUE(Adapter.contains(Unchanged, AMDGPU::VGPR2, A, D));
  }
}

TEST(SSARegisterForestAdapterTest, RegMaskInterferenceInSharedQuery) {
  auto TM = createAMDGPUTargetMachine("amdgcn-amd-", "gfx950", "");
  ASSERT_TRUE(TM);
  GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()), *TM);
  LLVMContext Context;
  Module M("forest-call-mask", Context);
  M.setDataLayout(TM->createDataLayout());
  auto *FT = FunctionType::get(Type::getVoidTy(Context), false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "test", &M);
  MachineModuleInfo MMI(TM.get());
  MachineFunction MF(*F, *TM, ST, MMI.getContext(), 0);
  MF.initTargetMachineFunctionInfo(ST);
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIRegisterInfo &TRI = *ST.getRegisterInfo();
  const SIInstrInfo &TII = *ST.getInstrInfo();
  Register ProbeVR = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  Register Owner = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  auto *BB = MF.CreateMachineBasicBlock();
  MF.push_back(BB);

  // One call clobbers VGPR0 and its aliases, while preserving VGPR1.
  SmallVector<uint32_t, 32> Mask((TRI.getNumRegs() + 31) / 32, ~uint32_t(0));
  for (MCRegAliasIterator I(AMDGPU::VGPR0, &TRI, /*IncludeSelf=*/true);
       I.isValid(); ++I)
    Mask[*I / 32] &= ~(uint32_t(1) << (*I % 32));
  MachineInstr *Before =
      BuildMI(*BB, BB->end(), DebugLoc(), TII.get(AMDGPU::S_NOP)).addImm(0);
  MachineInstr *Call =
      BuildMI(*BB, BB->end(), DebugLoc(), TII.get(AMDGPU::SI_CALL),
              AMDGPU::SGPR30_SGPR31)
          .addReg(AMDGPU::SGPR4_SGPR5, RegState::Undef)
          .addImm(0)
          .addRegMask(Mask.data());
  Call->getOperand(0).setIsDead();
  MachineInstr *After =
      BuildMI(*BB, BB->end(), DebugLoc(), TII.get(AMDGPU::S_ENDPGM)).addImm(0);
  MRI.freezeReservedRegs();

  MachineFunctionAnalysisManager MFAM;
  MFAM.registerPass([] { return PassInstrumentationAnalysis(); });
  MFAM.registerPass([] { return SlotIndexesAnalysis(); });
  MFAM.registerPass([] { return MachineDominatorTreeAnalysis(); });
  MFAM.registerPass([] { return LiveIntervalsAnalysis(); });
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  SlotIndex A = LIS.getInstructionIndex(*Before).getRegSlot();
  SlotIndex C = LIS.getInstructionIndex(*Call).getRegSlot();
  SlotIndex E = LIS.getInstructionIndex(*After).getRegSlot();
  ASSERT_TRUE(Call->isCall());
  ASSERT_EQ(LIS.getRegMaskSlots().size(), 1u);

  auto Forest = SSARegisterForest::create(1, 8);
  ASSERT_TRUE(Forest);
  RegisterForestAdapter Adapter(
      *Forest, TRI, MRI,
      [](const TargetRegisterClass *) -> ArrayRef<MCPhysReg> { return {}; });
  BumpPtrAllocator Allocator;
  LiveInterval Probe(ProbeVR, 0.0f);
  Probe.addSegment({A, E, Probe.getNextValue(A, Allocator)});
  auto Result = Adapter.interferences(AMDGPU::VGPR0, Probe, LIS);
  ASSERT_TRUE(Result);
  EXPECT_TRUE(Result->HasFixedInterference);
  EXPECT_TRUE(Result->VirtualOwners.empty());
  Result = Adapter.interferences(AMDGPU::VGPR1, Probe, LIS);
  ASSERT_TRUE(Result);
  EXPECT_FALSE(Result->HasFixedInterference);

  // Ordinary dying inputs exclude the mask slot; starts at that slot include it.
  Probe.clear();
  Probe.addSegment({A, C, Probe.getNextValue(A, Allocator)});
  Result = Adapter.interferences(AMDGPU::VGPR0, Probe, LIS);
  ASSERT_TRUE(Result);
  EXPECT_FALSE(Result->HasFixedInterference);
  Probe.clear();
  Probe.addSegment({C, E, Probe.getNextValue(C, Allocator)});
  Result = Adapter.interferences(AMDGPU::VGPR0, Probe, LIS);
  ASSERT_TRUE(Result);
  EXPECT_TRUE(Result->HasFixedInterference);

  // A mask in a liveness gap does not interfere.
  Probe.clear();
  Probe.addSegment({A, C, Probe.getNextValue(A, Allocator)});
  Probe.addSegment({C.getDeadSlot(), E,
                    Probe.getNextValue(C.getDeadSlot(), Allocator)});
  Result = Adapter.interferences(AMDGPU::VGPR0, Probe, LIS);
  ASSERT_TRUE(Result);
  EXPECT_FALSE(Result->HasFixedInterference);

  Probe.clear();
  Probe.addSegment({A, E, Probe.getNextValue(A, Allocator)});
  ASSERT_TRUE(Adapter.assign(Owner, AMDGPU::VGPR0, A, E));
  Result = Adapter.interferences(AMDGPU::VGPR0, Probe, LIS);
  ASSERT_TRUE(Result);
  EXPECT_TRUE(Result->HasFixedInterference);
  const Register ExpectedOwners[] = {Owner};
  EXPECT_EQ(ArrayRef<Register>(Result->VirtualOwners),
            ArrayRef<Register>(ExpectedOwners));
  ASSERT_TRUE(Adapter.release(Owner, AMDGPU::VGPR0, A, E));
  Result = Adapter.interferences(AMDGPU::VGPR0, Probe, LIS);
  ASSERT_TRUE(Result);
  EXPECT_TRUE(Result->HasFixedInterference);
  EXPECT_TRUE(Result->VirtualOwners.empty());
}

TEST(SSARegisterForestAdapterTest, ImportsFixedSGPRLiveIn) {
  // Match a flat-scratch target: the callable ABI leaves SGPR0 available
  // for arguments instead of reserving SGPR0-3 as a scratch descriptor.
  auto TM = createAMDGPUTargetMachine("amdgcn-amd-", "gfx950", "");
  ASSERT_TRUE(TM);
  GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()), *TM);
  LLVMContext Context;
  Module M("forest-fixed-sgpr", Context);
  M.setDataLayout(TM->createDataLayout());
  auto *FT = FunctionType::get(Type::getVoidTy(Context), false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "test", &M);
  MachineModuleInfo MMI(TM.get());
  MachineFunction MF(*F, *TM, ST, MMI.getContext(), 0);
  MF.initTargetMachineFunctionInfo(ST);
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIRegisterInfo &TRI = *ST.getRegisterInfo();
  const SIInstrInfo &TII = *ST.getInstrInfo();
  Register CopyResult = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
  Register ProbeVR = MRI.createVirtualRegister(&AMDGPU::SGPR_32RegClass);
  auto *BB = MF.CreateMachineBasicBlock();
  MF.push_back(BB);
  MRI.addLiveIn(AMDGPU::SGPR0, CopyResult);
  BB->addLiveIn(AMDGPU::SGPR0);
  MachineInstr *Use =
      BuildMI(*BB, BB->end(), DebugLoc(), TII.get(TargetOpcode::COPY), CopyResult)
          .addReg(AMDGPU::SGPR0);
  Use->getOperand(0).setIsDead();
  MachineInstr *EndMI =
      BuildMI(*BB, BB->end(), DebugLoc(), TII.get(AMDGPU::S_ENDPGM)).addImm(0);
  MRI.freezeReservedRegs();
  ASSERT_TRUE(MRI.isAllocatable(AMDGPU::SGPR0));

  MachineFunctionAnalysisManager MFAM;
  MFAM.registerPass([] { return PassInstrumentationAnalysis(); });
  MFAM.registerPass([] { return SlotIndexesAnalysis(); });
  MFAM.registerPass([] { return MachineDominatorTreeAnalysis(); });
  MFAM.registerPass([] { return LiveIntervalsAnalysis(); });
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  SlotIndex Start = LIS.getMBBStartIdx(BB);
  SlotIndex LastUse = LIS.getInstructionIndex(*Use).getRegSlot();
  SlotIndex End = LIS.getInstructionIndex(*EndMI).getRegSlot();

  auto Forest = SSARegisterForest::create(1, 8);
  ASSERT_TRUE(Forest);
  RegisterForestAdapter Adapter(
      *Forest, TRI, MRI,
      [](const TargetRegisterClass *) -> ArrayRef<MCPhysReg> { return {}; });
  // The high half exists as storage but has no standalone base class.
  ASSERT_EQ(TRI.getPhysRegBaseClass(AMDGPU::SGPR0_HI16), nullptr);
  const MCPhysReg Registers[] = {AMDGPU::SGPR0, AMDGPU::SGPR0_LO16};
  ASSERT_TRUE(Adapter.assignFixed(Registers, LIS));

  // The incoming 32-bit value occupies both leaves until its COPY use.
  SmallVector<unsigned, 2> Leaves;
  Forest->visitOwnershipComponents([&](SSARegisterForest::PhysicalSpan Span,
                            const SSARegisterForest::Ownership &O) {
    EXPECT_EQ(Span.width(), 1u);
    EXPECT_EQ(O.Owner, SSARegisterForest::SELF_OWNED);
    EXPECT_EQ(O.Start, Start);
    EXPECT_EQ(O.End, LastUse);
    Leaves.push_back(Span.FirstPhysicalLeaf);
  });
  const unsigned ExpectedLeaves[] = {0, 1};
  EXPECT_EQ(ArrayRef<unsigned>(Leaves), ArrayRef<unsigned>(ExpectedLeaves));
  EXPECT_FALSE(Adapter.isFree(AMDGPU::SGPR0, Start, LastUse));
  EXPECT_TRUE(Adapter.isFree(AMDGPU::SGPR0, LastUse, End));

  BumpPtrAllocator Allocator;
  LiveInterval Probe(ProbeVR, 0.0f);
  Probe.addSegment({Start, LastUse, Probe.getNextValue(Start, Allocator)});
  auto Result = Adapter.interferences(AMDGPU::SGPR0, Probe);
  ASSERT_TRUE(Result);
  EXPECT_TRUE(Result->HasFixedInterference);
  EXPECT_TRUE(Result->VirtualOwners.empty());
}

TEST(SSARegisterForestAdapterTest, ImportsFixedOccupancyAndClassifiesInterference) {
  auto TM = createAMDGPUTargetMachine("amdgcn-amd-", "gfx906", "");
  ASSERT_TRUE(TM);
  GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()), *TM);
  LLVMContext Context;
  Module M("forest-fixed-occupancy", Context);
  M.setDataLayout(TM->createDataLayout());
  auto *FT = FunctionType::get(Type::getVoidTy(Context), false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "test", &M);
  MachineModuleInfo MMI(TM.get());
  MachineFunction MF(*F, *TM, ST, MMI.getContext(), 0);
  MF.initTargetMachineFunctionInfo(ST);
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIRegisterInfo &TRI = *ST.getRegisterInfo();
  const SIInstrInfo &TII = *ST.getInstrInfo();
  Register CopyResult = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  Register Owner = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  Register ProbeVR = MRI.createVirtualRegister(&AMDGPU::VReg_64RegClass);
  auto *BB = MF.CreateMachineBasicBlock();
  MF.push_back(BB);

  // Straight-line CFG: VGPR1 contains a used value from B.reg to C.reg.
  // Its dead write at E still occupies [E.reg,E.dead). The gap is free.
  MachineInstr *A = BuildMI(*BB, BB->end(), DebugLoc(), TII.get(AMDGPU::S_NOP))
                        .addImm(0);
  MachineInstr *B =
      BuildMI(*BB, BB->end(), DebugLoc(), TII.get(AMDGPU::V_MOV_B32_e32),
              AMDGPU::VGPR1).addImm(0);
  MachineInstr *C =
      BuildMI(*BB, BB->end(), DebugLoc(), TII.get(TargetOpcode::COPY), CopyResult)
          .addReg(AMDGPU::VGPR1);
  C->getOperand(0).setIsDead();
  BuildMI(*BB, BB->end(), DebugLoc(), TII.get(AMDGPU::S_NOP)).addImm(0);
  MachineInstr *E =
      BuildMI(*BB, BB->end(), DebugLoc(), TII.get(AMDGPU::V_MOV_B32_e32),
              AMDGPU::VGPR1).addImm(1);
  E->getOperand(0).setIsDead();
  BuildMI(*BB, BB->end(), DebugLoc(), TII.get(AMDGPU::S_NOP)).addImm(0);
  MachineInstr *G =
      BuildMI(*BB, BB->end(), DebugLoc(), TII.get(AMDGPU::S_ENDPGM)).addImm(0);
  MRI.freezeReservedRegs();

  MachineFunctionAnalysisManager MFAM;
  MFAM.registerPass([] { return PassInstrumentationAnalysis(); });
  MFAM.registerPass([] { return SlotIndexesAnalysis(); });
  MFAM.registerPass([] { return MachineDominatorTreeAnalysis(); });
  MFAM.registerPass([] { return LiveIntervalsAnalysis(); });
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  SlotIndex Start = LIS.getInstructionIndex(*A).getRegSlot();
  SlotIndex Def = LIS.getInstructionIndex(*B).getRegSlot();
  SlotIndex LastUse = LIS.getInstructionIndex(*C).getRegSlot();
  SlotIndex DeadDef = LIS.getInstructionIndex(*E).getRegSlot();
  SlotIndex DeadEnd = LIS.getInstructionIndex(*E).getDeadSlot();
  SlotIndex End = LIS.getInstructionIndex(*G).getRegSlot();

  auto Forest = SSARegisterForest::create(1, 8);
  ASSERT_TRUE(Forest);
  RegisterForestAdapter Adapter(
      *Forest, TRI, MRI,
      [](const TargetRegisterClass *) -> ArrayRef<MCPhysReg> { return {}; });
  const MCPhysReg Registers[] = {AMDGPU::VGPR0_VGPR1, AMDGPU::VGPR1,
                                AMDGPU::VGPR1_LO16};
  ASSERT_TRUE(Adapter.assignFixed(Registers, LIS));

  // Check the physical meaning independently of the importer's projection:
  // VGPR1 occupies half-register leaves 2 and 3, with two intervals each.
  unsigned FixedRecords = 0;
  Forest->visitOwnershipComponents([&](SSARegisterForest::PhysicalSpan Span,
                            const SSARegisterForest::Ownership &O) {
    ++FixedRecords;
    EXPECT_EQ(O.Owner, SSARegisterForest::SELF_OWNED);
    EXPECT_EQ(Span.width(), 1u);
    EXPECT_TRUE(Span.FirstPhysicalLeaf == 2 || Span.FirstPhysicalLeaf == 3);
    EXPECT_TRUE((O.Start == Def && O.End == LastUse) ||
                (O.Start == DeadDef && O.End == DeadEnd));
  });
  EXPECT_EQ(FixedRecords, 4u);
  EXPECT_TRUE(Adapter.isFree(AMDGPU::VGPR1, LastUse, DeadDef));
  EXPECT_TRUE(Adapter.isFree(AMDGPU::VGPR1, DeadEnd, End));
  ASSERT_TRUE(Adapter.assign(Owner, AMDGPU::VGPR0, Start, End));

  BumpPtrAllocator Allocator;
  LiveInterval Probe(ProbeVR, 0.0f);
  Probe.addSegment({Start, End, Probe.getNextValue(Start, Allocator)});
  auto *Low = Probe.createSubRange(
      Allocator, TRI.getSubRegIndexLaneMask(AMDGPU::sub0));
  Low->addSegment({Start, End, Low->getNextValue(Start, Allocator)});
  auto *High = Probe.createSubRange(
      Allocator, TRI.getSubRegIndexLaneMask(AMDGPU::sub1));
  High->addSegment({Def, LastUse, High->getNextValue(Def, Allocator)});
  auto Result = Adapter.interferences(AMDGPU::VGPR0_VGPR1, Probe);
  ASSERT_TRUE(Result);
  EXPECT_TRUE(Result->HasFixedInterference);
  const Register ExpectedOwners[] = {Owner};
  EXPECT_EQ(ArrayRef<Register>(Result->VirtualOwners),
            ArrayRef<Register>(ExpectedOwners));

  High->clear();
  High->addSegment({LastUse, DeadDef, High->getNextValue(LastUse, Allocator)});
  Result = Adapter.interferences(AMDGPU::VGPR0_VGPR1, Probe);
  ASSERT_TRUE(Result);
  EXPECT_FALSE(Result->HasFixedInterference);
  EXPECT_EQ(ArrayRef<Register>(Result->VirtualOwners),
            ArrayRef<Register>(ExpectedOwners));

  High->clear();
  High->addSegment({DeadDef, DeadEnd, High->getNextValue(DeadDef, Allocator)});
  ASSERT_TRUE(Adapter.release(Owner, AMDGPU::VGPR0, Start, End));
  Result = Adapter.interferences(AMDGPU::VGPR0_VGPR1, Probe);
  ASSERT_TRUE(Result);
  EXPECT_TRUE(Result->HasFixedInterference);
  EXPECT_TRUE(Result->VirtualOwners.empty());
  auto VirtualOnly = Adapter.interferingOwners(AMDGPU::VGPR0_VGPR1, Probe);
  ASSERT_TRUE(VirtualOnly);
  EXPECT_TRUE(VirtualOnly->empty());

  EXPECT_FALSE(Adapter.assign(Owner, AMDGPU::VGPR1, Def, LastUse));
  EXPECT_FALSE(Adapter.isFree(AMDGPU::VGPR1, Def, LastUse));
  EXPECT_TRUE(Adapter.assign(Owner, AMDGPU::VGPR1, LastUse, DeadDef));
}

TEST(SSARegisterForestAdapterTest, LaneAwareInterferingOwners) {
  auto TM = createAMDGPUTargetMachine("amdgcn-amd-", "gfx906", "");
  ASSERT_TRUE(TM);
  GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()), *TM);
  LLVMContext Context;
  Module M("forest-lane-query", Context);
  M.setDataLayout(TM->createDataLayout());
  auto *FT = FunctionType::get(Type::getVoidTy(Context), false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "test", &M);
  MachineModuleInfo MMI(TM.get());
  MachineFunction MF(*F, *TM, ST, MMI.getContext(), 0);
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIRegisterInfo &TRI = *ST.getRegisterInfo();

  Register ProbeVR = MRI.createVirtualRegister(&AMDGPU::VReg_64RegClass);
  Register LowHole = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  Register HighEarly = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  Register GapOnly = MRI.createVirtualRegister(&AMDGPU::VReg_64RegClass);
  Register BothHit = MRI.createVirtualRegister(&AMDGPU::VReg_64RegClass);
  Register HighHit = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  Register Touching = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  auto Forest = SSARegisterForest::create(1, 8);
  ASSERT_TRUE(Forest);
  RegisterForestAdapter Adapter(
      *Forest, TRI, MRI,
      [](const TargetRegisterClass *) -> ArrayRef<MCPhysReg> { return {}; });

  IndexListEntry E1(nullptr, SlotIndex::InstrDist);
  IndexListEntry E2(nullptr, 2 * SlotIndex::InstrDist);
  IndexListEntry E3(nullptr, 3 * SlotIndex::InstrDist);
  IndexListEntry E4(nullptr, 4 * SlotIndex::InstrDist);
  IndexListEntry E5(nullptr, 5 * SlotIndex::InstrDist);
  IndexListEntry E6(nullptr, 6 * SlotIndex::InstrDist);
  IndexListEntry E7(nullptr, 7 * SlotIndex::InstrDist);
  SlotIndex A(&E1, 0), B(&E2, 0), C(&E3, 0), D(&E4, 0);
  SlotIndex E(&E5, 0), FEnd(&E6, 0), G(&E7, 0);
  BumpPtrAllocator Allocator;
  LiveInterval Probe(ProbeVR, 0.0f);
  Probe.addSegment({A, C, Probe.getNextValue(A, Allocator)});
  Probe.addSegment({D, FEnd, Probe.getNextValue(D, Allocator)});
  auto *Low = Probe.createSubRange(
      Allocator, TRI.getSubRegIndexLaneMask(AMDGPU::sub0));
  Low->addSegment({A, B, Low->getNextValue(A, Allocator)});
  Low->addSegment({D, FEnd, Low->getNextValue(D, Allocator)});
  auto *High = Probe.createSubRange(
      Allocator, TRI.getSubRegIndexLaneMask(AMDGPU::sub1));
  High->addSegment({B, C, High->getNextValue(B, Allocator)});
  High->addSegment({E, FEnd, High->getNextValue(E, Allocator)});

  // These owners occupy only dead probe lanes, a gap, or the touching end.
  ASSERT_TRUE(Adapter.assign(LowHole, AMDGPU::VGPR0, B, C));
  ASSERT_TRUE(Adapter.assign(HighEarly, AMDGPU::VGPR1, A, B));
  ASSERT_TRUE(Adapter.assign(GapOnly, AMDGPU::VGPR0_VGPR1, C, D));
  ASSERT_TRUE(Adapter.assign(Touching, AMDGPU::VGPR0, FEnd, G));
  ASSERT_TRUE(Adapter.assign(HighHit, AMDGPU::VGPR1, B, C));
  ASSERT_TRUE(Adapter.assign(BothHit, AMDGPU::VGPR0_VGPR1, E, FEnd));

  auto Owners = Adapter.interferingOwners(AMDGPU::VGPR0_VGPR1, Probe);
  ASSERT_TRUE(Owners.has_value());
  // BothHit intersects both subranges but must appear only once. ID order
  // must not depend on the subrange list or RF traversal order.
  const Register Expected[] = {BothHit, HighHit};
  EXPECT_EQ(ArrayRef<Register>(*Owners), ArrayRef<Register>(Expected));
  EXPECT_FALSE(Adapter.interferingOwners(MCRegister(), Probe).has_value());

  ASSERT_TRUE(Adapter.release(HighHit, AMDGPU::VGPR1, B, C));
  ASSERT_TRUE(Adapter.release(BothHit, AMDGPU::VGPR0_VGPR1, E, FEnd));
  Owners = Adapter.interferingOwners(AMDGPU::VGPR0_VGPR1, Probe);
  ASSERT_TRUE(Owners.has_value());
  EXPECT_TRUE(Owners->empty());
}

TEST(SSARegisterForestAdapterTest, QueriesProbeSegmentsAndClipsOwnership) {
  auto TM = createAMDGPUTargetMachine("amdgcn-amd-", "gfx906", "");
  ASSERT_TRUE(TM);
  GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()), *TM);
  LLVMContext Context;
  Module M("forest-adapter", Context);
  M.setDataLayout(TM->createDataLayout());
  auto *FT = FunctionType::get(Type::getVoidTy(Context), false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "test", &M);
  MachineModuleInfo MMI(TM.get());
  MachineFunction MF(*F, *TM, ST, MMI.getContext(), 0);
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIRegisterInfo &TRI = *ST.getRegisterInfo();

  Register GapOwner = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  Register CrossingOwner = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  auto Forest = SSARegisterForest::create(1, 8);
  ASSERT_TRUE(Forest);
  RegisterForestAdapter Adapter(
      *Forest, TRI, MRI,
      [](const TargetRegisterClass *) -> ArrayRef<MCPhysReg> { return {}; });

  IndexListEntry E1(nullptr, SlotIndex::InstrDist);
  IndexListEntry E2(nullptr, 2 * SlotIndex::InstrDist);
  IndexListEntry E3(nullptr, 3 * SlotIndex::InstrDist);
  IndexListEntry E4(nullptr, 4 * SlotIndex::InstrDist);
  SlotIndex A(&E1, 0), B(&E2, 0), C(&E3, 0), D(&E4, 0);
  BumpPtrAllocator Allocator;
  LiveRange Probe;
  VNInfo *Value = Probe.getNextValue(A, Allocator);
  Probe.addSegment({A, B, Value});
  Probe.addSegment({C, D, Value});

  // Distinct dwords: one owner lives only in the probe gap, while the other
  // remains live across both probe segments and their gap.
  ASSERT_TRUE(Adapter.assign(GapOwner, AMDGPU::VGPR0, B, C));
  ASSERT_TRUE(Adapter.assign(CrossingOwner, AMDGPU::VGPR1, A, D));
  unsigned WholeWindowHits = 0;
  ASSERT_TRUE(Adapter.visitInterferences(
      AMDGPU::VGPR0_VGPR1, A, D,
      [&](const SSARegisterForest::Ownership &) { ++WholeWindowHits; }));
  EXPECT_EQ(WholeWindowHits, 2u);

  using Hit = std::tuple<Register, SlotIndex, SlotIndex>;
  SmallVector<Hit, 2> Hits;
  ASSERT_TRUE(Adapter.visitInterferences(
      AMDGPU::VGPR0_VGPR1, Probe,
      [&](Register Owner, SlotIndex Start, SlotIndex End) {
        Hits.emplace_back(Owner, Start, End);
      }));
  const Hit Expected[] = {{CrossingOwner, A, B}, {CrossingOwner, C, D}};
  EXPECT_EQ(ArrayRef<Hit>(Hits), ArrayRef<Hit>(Expected));

  auto ExpectNoHit = [&](Register, SlotIndex, SlotIndex) {
    ADD_FAILURE() << "query must not report an owner";
  };
  EXPECT_TRUE(Adapter.visitInterferences(AMDGPU::VGPR0, Probe, ExpectNoHit));
  LiveRange Empty;
  EXPECT_TRUE(Adapter.visitInterferences(AMDGPU::VGPR1, Empty, ExpectNoHit));
  EXPECT_FALSE(Adapter.visitInterferences(MCRegister(), Probe, ExpectNoHit));
  EXPECT_TRUE(Adapter.contains(GapOwner, AMDGPU::VGPR0, B, C));
  EXPECT_TRUE(Adapter.contains(CrossingOwner, AMDGPU::VGPR1, A, D));
}

TEST(SSARegisterForestAdapterTest, SparseInsertionIsAtomic) {
  auto TM = createAMDGPUTargetMachine("amdgcn-amd-", "gfx906", "");
  ASSERT_TRUE(TM);
  GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()), *TM);
  LLVMContext Context;
  Module M("forest-adapter", Context);
  M.setDataLayout(TM->createDataLayout());
  auto *FT = FunctionType::get(Type::getVoidTy(Context), false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "test", &M);
  MachineModuleInfo MMI(TM.get());
  MachineFunction MF(*F, *TM, ST, MMI.getContext(), 0);
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIRegisterInfo &TRI = *ST.getRegisterInfo();

  Register Pair = MRI.createVirtualRegister(&AMDGPU::VReg_64RegClass);
  Register Other = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
  auto Forest = SSARegisterForest::create(1, 8);
  ASSERT_TRUE(Forest);
  RegisterForestAdapter Adapter(
      *Forest, TRI, MRI,
      [](const TargetRegisterClass *) -> ArrayRef<MCPhysReg> { return {}; });
  IndexListEntry E1(nullptr, SlotIndex::InstrDist);
  IndexListEntry E2(nullptr, 2 * SlotIndex::InstrDist);
  IndexListEntry E3(nullptr, 3 * SlotIndex::InstrDist);
  SlotIndex Start(&E1, 0), Split(&E2, 0), End(&E3, 0);
  LaneBitmask Low = TRI.getSubRegIndexLaneMask(AMDGPU::sub0);
  LaneBitmask High = TRI.getSubRegIndexLaneMask(AMDGPU::sub1);

  // The high dword belongs to another value until Split. Inserting a whole
  // pair and subsequently shrinking it would incorrectly reject this case.
  ASSERT_TRUE(Adapter.assign(Other, AMDGPU::VGPR1, Start, Split));
  RegisterForestAdapter::RetainedRegion Invalid[] = {
      {VRegMaskPair(Pair, Low), Start, End},
      {VRegMaskPair(Pair, High), Start, End}};
  EXPECT_FALSE(Adapter.assign(Pair, AMDGPU::VGPR0_VGPR1, Invalid));
  EXPECT_TRUE(Adapter.isFree(AMDGPU::VGPR0, Start, End));
  EXPECT_TRUE(Adapter.contains(Other, AMDGPU::VGPR1, Start, Split));

  RegisterForestAdapter::RetainedRegion Live[] = {
      {VRegMaskPair(Pair, Low), Start, End},
      {VRegMaskPair(Pair, High), Split, End}};
  ASSERT_TRUE(Adapter.assign(Pair, AMDGPU::VGPR0_VGPR1, Live));
  EXPECT_TRUE(Adapter.contains(Pair, AMDGPU::VGPR0, Start, End));
  EXPECT_TRUE(Adapter.contains(Pair, AMDGPU::VGPR1, Split, End));
  EXPECT_TRUE(Adapter.contains(Other, AMDGPU::VGPR1, Start, Split));

  unsigned Components = 0;
  Forest->visitOwnershipComponents([&](SSARegisterForest::PhysicalSpan Span,
                             const SSARegisterForest::Ownership &O) {
    ++Components;
    if (Span.FirstPhysicalLeaf == 0) {
      EXPECT_EQ(Span.EndPhysicalLeaf, 2u);
      EXPECT_EQ(O.Owner, Pair);
      EXPECT_EQ(O.Start, Start);
      EXPECT_EQ(O.End, End);
    } else {
      EXPECT_EQ(Span.FirstPhysicalLeaf, 2u);
      EXPECT_EQ(Span.EndPhysicalLeaf, 4u);
      EXPECT_EQ(O.Owner, O.Start == Start ? Other : Pair);
      EXPECT_EQ(O.End, O.Start == Start ? Split : End);
    }
  });
  EXPECT_EQ(Components, 3u);
}

TEST(SSARegisterForestAdapterTest, TargetOrderAndPhysicalOverlap) {
  auto TM = createAMDGPUTargetMachine("amdgcn-amd-", "gfx906", "");
  ASSERT_TRUE(TM);
  GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()), *TM);
  LLVMContext Context;
  Module M("forest-adapter", Context);
  M.setDataLayout(TM->createDataLayout());
  auto *FT = FunctionType::get(Type::getVoidTy(Context), false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "test", &M);
  MachineModuleInfo MMI(TM.get());
  MachineFunction MF(*F, *TM, ST, MMI.getContext(), 0);
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIRegisterInfo &TRI = *ST.getRegisterInfo();
  Register PairVR = MRI.createVirtualRegister(&AMDGPU::VReg_64RegClass);

  auto Forest = SSARegisterForest::create(2, 8);
  ASSERT_TRUE(Forest);
  // Deliberately nonnumeric target preference, with VGPR2 and VGPR5 omitted.
  // This models a filtered allocator order, not a new reservation policy.
  const MCPhysReg Order[] = {AMDGPU::VGPR6_VGPR7, AMDGPU::VGPR0_VGPR1,
                            AMDGPU::VGPR3_VGPR4};
  unsigned OrderCalls = 0;
  RegisterForestAdapter Adapter(
      *Forest, TRI, MRI,
      [&](const TargetRegisterClass *RC) -> ArrayRef<MCPhysReg> {
        EXPECT_EQ(RC, &AMDGPU::VReg_64RegClass);
        ++OrderCalls;
        return Order;
      });
  ArrayRef<MCPhysReg> Homes = Adapter.allocationOrder(PairVR);
  ASSERT_EQ(Homes.size(), 3u);
  EXPECT_EQ(OrderCalls, 1u);
  for (unsigned I = 0; I != Homes.size(); ++I)
    EXPECT_EQ(Homes[I], Order[I]);

  IndexListEntry E1(nullptr, SlotIndex::InstrDist);
  IndexListEntry E2(nullptr, 2 * SlotIndex::InstrDist);
  IndexListEntry E3(nullptr, 3 * SlotIndex::InstrDist);
  SlotIndex Start(&E1, 0), End(&E2, 0), After(&E3, 0);
  const MCPhysReg DWords[] = {
      AMDGPU::VGPR0, AMDGPU::VGPR1, AMDGPU::VGPR2, AMDGPU::VGPR3,
      AMDGPU::VGPR4, AMDGPU::VGPR5, AMDGPU::VGPR6, AMDGPU::VGPR7};

  // Check the forwarded order against physical occupancy. Reindexing the
  // filtered list would wrongly occupy a gap or a lower-priority leaf.
  ASSERT_TRUE(Adapter.assign(PairVR, Homes.front(), Start, End));
  for (unsigned I = 0; I != 8; ++I)
    EXPECT_EQ(Adapter.isFree(DWords[I], Start, End), I < 6);
  ASSERT_TRUE(Adapter.release(PairVR, Homes.front(), Start, End));

  struct MappingCase {
    MCPhysReg PR;
    unsigned First;
    unsigned End;
  };
  // Independent physical oracle: the seven 32-bit-and-wider nodes of tree zero,
  // then an unaligned pair, a cross-tree pair, an odd tuple and a wider tuple.
  const MappingCase Cases[] = {
      {AMDGPU::VGPR0_VGPR1_VGPR2_VGPR3, 0, 4},
      {AMDGPU::VGPR0_VGPR1, 0, 2},
      {AMDGPU::VGPR0, 0, 1},
      {AMDGPU::VGPR1, 1, 2},
      {AMDGPU::VGPR2_VGPR3, 2, 4},
      {AMDGPU::VGPR2, 2, 3},
      {AMDGPU::VGPR3, 3, 4},
      {AMDGPU::VGPR1_VGPR2, 1, 3},
      {AMDGPU::VGPR3_VGPR4, 3, 5},
      {AMDGPU::VGPR1_VGPR2_VGPR3, 1, 4},
      {AMDGPU::VGPR0_VGPR1_VGPR2_VGPR3_VGPR4_VGPR5_VGPR6_VGPR7, 0, 8}};
  for (const MappingCase &C : Cases) {
    SCOPED_TRACE(TRI.getName(C.PR));
    Register VR = MRI.createVirtualRegister(TRI.getPhysRegBaseClass(C.PR));
    ASSERT_TRUE(Adapter.assign(VR, C.PR, Start, End));
    ASSERT_TRUE(Adapter.contains(VR, C.PR, Start, End));
    for (unsigned I = 0; I != 8; ++I) {
      const bool ExpectedFree = I < C.First || I >= C.End;
      EXPECT_EQ(Adapter.isFree(DWords[I], Start, End), ExpectedFree);
      EXPECT_TRUE(Adapter.isFree(DWords[I], End, After));
    }
    if (C.End - C.First > 1) {
      EXPECT_FALSE(Adapter.contains(VR, DWords[C.First], Start, End));
      EXPECT_FALSE(Adapter.release(VR, DWords[C.First], Start, End));
      EXPECT_TRUE(Adapter.contains(VR, C.PR, Start, End));
    }
    ASSERT_TRUE(Adapter.release(VR, C.PR, Start, End));
    for (MCPhysReg PR : DWords)
      EXPECT_TRUE(Adapter.isFree(PR, Start, End));
  }

  // A separate invocation uses SGPR identities on the same leaf geometry.
  auto ScalarForest = SSARegisterForest::create(2, 8);
  ASSERT_TRUE(ScalarForest);
  Register ScalarVR = MRI.createVirtualRegister(&AMDGPU::SReg_64RegClass);
  const MCPhysReg ScalarOrder[] = {AMDGPU::SGPR6_SGPR7, AMDGPU::SGPR0_SGPR1};
  RegisterForestAdapter ScalarAdapter(
      *ScalarForest, TRI, MRI,
      [&](const TargetRegisterClass *RC) -> ArrayRef<MCPhysReg> {
        EXPECT_EQ(RC, &AMDGPU::SReg_64RegClass);
        return ScalarOrder;
      });
  auto ScalarHomes = ScalarAdapter.allocationOrder(ScalarVR);
  ASSERT_EQ(ScalarHomes.size(), 2u);
  EXPECT_EQ(ScalarHomes[0], ScalarOrder[0]);
  EXPECT_EQ(ScalarHomes[1], ScalarOrder[1]);
  ASSERT_TRUE(ScalarAdapter.assign(ScalarVR, ScalarHomes[0], Start, End));
  EXPECT_TRUE(ScalarAdapter.isFree(AMDGPU::SGPR5, Start, End));
  EXPECT_FALSE(ScalarAdapter.isFree(AMDGPU::SGPR6, Start, End));
  EXPECT_FALSE(ScalarAdapter.isFree(AMDGPU::SGPR7, Start, End));
  ASSERT_TRUE(ScalarAdapter.release(ScalarVR, ScalarHomes[0], Start, End));
  EXPECT_TRUE(ScalarAdapter.isFree(AMDGPU::SGPR6, Start, End));
  EXPECT_TRUE(ScalarAdapter.isFree(AMDGPU::SGPR7, Start, End));
}

TEST(SSARegisterForestAdapterTest, True16OwnershipAndTupleOverlap) {
  auto TM = createAMDGPUTargetMachine("amdgcn-amd-", "gfx1100", "");
  ASSERT_TRUE(TM);
  GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()), *TM);
  LLVMContext Context;
  Module M("forest-true16", Context);
  M.setDataLayout(TM->createDataLayout());
  auto *FT = FunctionType::get(Type::getVoidTy(Context), false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "test", &M);
  MachineModuleInfo MMI(TM.get());
  MachineFunction MF(*F, *TM, ST, MMI.getContext(), 0);
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIRegisterInfo &TRI = *ST.getRegisterInfo();

  // Two trees still cover eight dwords, now with 16 independently owned halves.
  auto Forest = SSARegisterForest::create(2, 8);
  ASSERT_TRUE(Forest);
  EXPECT_EQ(Forest->numLeaves(), 16u);
  EXPECT_EQ(Forest->numNodes(), 30u);
  Register LowVR = MRI.createVirtualRegister(&AMDGPU::VGPR_16RegClass);
  Register HighVR = MRI.createVirtualRegister(&AMDGPU::VGPR_16RegClass);
  const MCPhysReg Order[] = {AMDGPU::VGPR2_HI16, AMDGPU::VGPR0_LO16,
                            AMDGPU::VGPR2_LO16};
  RegisterForestAdapter Adapter(
      *Forest, TRI, MRI,
      [&](const TargetRegisterClass *RC) -> ArrayRef<MCPhysReg> {
        EXPECT_EQ(RC, &AMDGPU::VGPR_16RegClass);
        return Order;
      });
  auto Homes = Adapter.allocationOrder(LowVR);
  ASSERT_EQ(Homes.size(), 3u);
  for (unsigned I = 0; I != Homes.size(); ++I)
    EXPECT_EQ(Homes[I], Order[I]);

  IndexListEntry E1(nullptr, SlotIndex::InstrDist);
  IndexListEntry E2(nullptr, 2 * SlotIndex::InstrDist);
  IndexListEntry E3(nullptr, 3 * SlotIndex::InstrDist);
  SlotIndex Start(&E1, 0), End(&E2, 0), After(&E3, 0);

  // The physical half coordinate 4 is stored at preorder node 10; coordinate
  // 5 is at node 11. The parent VGPR2 is node 9, not physical coordinate 2.
  ASSERT_TRUE(Adapter.assign(LowVR, AMDGPU::VGPR2_LO16, Start, End));
  ASSERT_TRUE(Forest->contains({4, 5}, Start, End, LowVR));
  EXPECT_TRUE(Adapter.isFree(AMDGPU::VGPR2_HI16, Start, End));
  EXPECT_FALSE(Adapter.isFree(AMDGPU::VGPR2, Start, End));
  ASSERT_TRUE(Adapter.assign(HighVR, AMDGPU::VGPR2_HI16, Start, End));
  ASSERT_TRUE(Forest->contains({5, 6}, Start, End, HighVR));
  EXPECT_FALSE(Adapter.release(LowVR, AMDGPU::VGPR2, Start, End));
  EXPECT_TRUE(Adapter.contains(LowVR, AMDGPU::VGPR2_LO16, Start, End));
  EXPECT_TRUE(Adapter.contains(HighVR, AMDGPU::VGPR2_HI16, Start, End));
  ASSERT_TRUE(Adapter.release(LowVR, AMDGPU::VGPR2_LO16, Start, End));
  EXPECT_TRUE(Adapter.isFree(AMDGPU::VGPR2_LO16, Start, End));
  EXPECT_FALSE(Adapter.isFree(AMDGPU::VGPR2, Start, End));
  EXPECT_TRUE(Adapter.contains(HighVR, AMDGPU::VGPR2_HI16, Start, End));
  ASSERT_TRUE(Adapter.release(HighVR, AMDGPU::VGPR2_HI16, Start, End));
  EXPECT_TRUE(Adapter.isFree(AMDGPU::VGPR2, Start, End));

  // Use target register-unit aliasing as an independent overlap oracle. Include
  // boundary halves, aligned parents, and unaligned/cross-tree composite homes.
  const MCPhysReg Registers[] = {
      AMDGPU::VGPR0_LO16, AMDGPU::VGPR0_HI16,
      AMDGPU::VGPR2_LO16, AMDGPU::VGPR2_HI16,
      AMDGPU::VGPR3_HI16, AMDGPU::VGPR4_LO16, AMDGPU::VGPR7_HI16,
      AMDGPU::VGPR0, AMDGPU::VGPR2, AMDGPU::VGPR7,
      AMDGPU::VGPR2_VGPR3, AMDGPU::VGPR3_VGPR4,
      AMDGPU::VGPR1_VGPR2_VGPR3, AMDGPU::VGPR0_VGPR1_VGPR2_VGPR3,
      AMDGPU::VGPR0_VGPR1_VGPR2_VGPR3_VGPR4_VGPR5_VGPR6_VGPR7};
  for (MCPhysReg Assigned : Registers) {
    SCOPED_TRACE(TRI.getName(Assigned));
    const TargetRegisterClass *RC = TRI.getPhysRegBaseClass(Assigned);
    // Low-numbered true16 homes have a non-allocatable physical base class.
    if (TRI.getRegSizeInBits(*RC) == 16)
      RC = &AMDGPU::VGPR_16RegClass;
    Register VR = MRI.createVirtualRegister(RC);
    ASSERT_TRUE(Adapter.assign(VR, Assigned, Start, End));
    for (MCPhysReg Probe : Registers) {
      SCOPED_TRACE(TRI.getName(Probe));
      EXPECT_EQ(Adapter.isFree(Probe, Start, End),
                !TRI.regsOverlap(Assigned, Probe));
      EXPECT_TRUE(Adapter.isFree(Probe, End, After));
    }
    ASSERT_TRUE(Adapter.release(VR, Assigned, Start, End));
    for (MCPhysReg Probe : Registers)
      EXPECT_TRUE(Adapter.isFree(Probe, Start, End));
  }
  EXPECT_FALSE(Adapter.assign(LowVR, AMDGPU::VGPR8_LO16, Start, End));
  EXPECT_TRUE(Adapter.isFree(AMDGPU::VGPR7_HI16, Start, End));
}

TEST(SSARegisterForestAdapterTest, MaskedReplacementPreservesPhysicalLanes) {
  auto TM = createAMDGPUTargetMachine("amdgcn-amd-", "gfx1100", "");
  ASSERT_TRUE(TM);
  GCNSubtarget ST(TM->getTargetTriple(), std::string(TM->getTargetCPU()),
                  std::string(TM->getTargetFeatureString()), *TM);
  LLVMContext Context;
  Module M("forest-masked-replacement", Context);
  M.setDataLayout(TM->createDataLayout());
  auto *FT = FunctionType::get(Type::getVoidTy(Context), false);
  auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, "test", &M);
  MachineModuleInfo MMI(TM.get());
  MachineFunction MF(*F, *TM, ST, MMI.getContext(), 0);
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIRegisterInfo &TRI = *ST.getRegisterInfo();
  using Region = RegisterForestAdapter::RetainedRegion;
  IndexListEntry E1(nullptr, SlotIndex::InstrDist);
  IndexListEntry E2(nullptr, 2 * SlotIndex::InstrDist);
  IndexListEntry E3(nullptr, 3 * SlotIndex::InstrDist);
  SlotIndex Start(&E1, 0), Spill(&E2, 0), End(&E3, 0);

  struct Case {
    const TargetRegisterClass *RC;
    MCPhysReg PR;
    LaneBitmask Retain;
    unsigned FirstHalf;
    unsigned EndHalf;
    // Independent expected physical-leaf bitmap, not an LLVM lane mask.
    unsigned ExpectedTail;
  };
  const Case Cases[] = {
      {&AMDGPU::VReg_64RegClass, AMDGPU::VGPR2_VGPR3,
       TRI.getSubRegIndexLaneMask(AMDGPU::sub1), 4, 8, 0xc0},
      {&AMDGPU::VReg_128RegClass, AMDGPU::VGPR0_VGPR1_VGPR2_VGPR3,
       TRI.getSubRegIndexLaneMask(AMDGPU::hi16) |
           TRI.getSubRegIndexLaneMask(AMDGPU::sub2) |
           TRI.getSubRegIndexLaneMask(AMDGPU::sub3_lo16),
       0, 8, 0x72}};
  const MCPhysReg Halves[] = {
      AMDGPU::VGPR0_LO16, AMDGPU::VGPR0_HI16,
      AMDGPU::VGPR1_LO16, AMDGPU::VGPR1_HI16,
      AMDGPU::VGPR2_LO16, AMDGPU::VGPR2_HI16,
      AMDGPU::VGPR3_LO16, AMDGPU::VGPR3_HI16};

  for (const Case &C : Cases) {
    SCOPED_TRACE(TRI.getName(C.PR));
    auto Forest = SSARegisterForest::create(2, 8);
    ASSERT_TRUE(Forest);
    RegisterForestAdapter Adapter(
        *Forest, TRI, MRI,
        [](const TargetRegisterClass *) -> ArrayRef<MCPhysReg> { return {}; });
    Register VR = MRI.createVirtualRegister(C.RC);
    Register Other = MRI.createVirtualRegister(&AMDGPU::VGPR_32RegClass);
    LaneBitmask Full = MRI.getMaxLaneMaskForVReg(VR);
    ASSERT_TRUE(Adapter.assign(VR, C.PR, Start, End));
    ASSERT_TRUE(Adapter.assign(Other, AMDGPU::VGPR4, Start, End));

    const Region WrongOwner[] = {{VRegMaskPair(Other, Full), Start, End}};
    EXPECT_FALSE(Adapter.replace(VR, C.PR, Start, End, WrongOwner));
    const Region EmptyMask[] = {
        {VRegMaskPair(VR, LaneBitmask::getNone()), Start, End}};
    EXPECT_FALSE(Adapter.replace(VR, C.PR, Start, End, EmptyMask));
    const Region OutOfClass[] = {
        {VRegMaskPair(VR, TRI.getSubRegIndexLaneMask(AMDGPU::sub4)), Start, End}};
    EXPECT_FALSE(Adapter.replace(VR, C.PR, Start, End, OutOfClass));
    const Region Overlapping[] = {
        {VRegMaskPair(VR, Full), Start, End},
        {VRegMaskPair(VR, C.Retain), Spill, End}};
    EXPECT_FALSE(Adapter.replace(VR, C.PR, Start, End, Overlapping));
    ASSERT_TRUE(Adapter.contains(VR, C.PR, Start, End));
    ASSERT_TRUE(Adapter.contains(Other, AMDGPU::VGPR4, Start, End));

    const Region Retained[] = {{VRegMaskPair(VR, Full), Start, Spill},
                               {VRegMaskPair(VR, C.Retain), Spill, End}};
    ASSERT_TRUE(Adapter.replace(VR, C.PR, Start, End, Retained));
    EXPECT_TRUE(Adapter.contains(VR, C.PR, Start, Spill));
    for (unsigned I = 0; I != 8; ++I) {
      SCOPED_TRACE(I);
      EXPECT_EQ(Adapter.isFree(Halves[I], Start, Spill),
                I < C.FirstHalf || I >= C.EndHalf);
      EXPECT_EQ(Adapter.isFree(Halves[I], Spill, End),
                (C.ExpectedTail & (1u << I)) == 0);
    }
    EXPECT_TRUE(Adapter.contains(Other, AMDGPU::VGPR4, Start, End));
    // Empty replacement removes the exact surviving head only.
    ASSERT_TRUE(Adapter.replace(VR, C.PR, Start, Spill, {}));
    EXPECT_TRUE(Adapter.isFree(C.PR, Start, Spill));
    EXPECT_FALSE(Adapter.isFree(C.PR, Spill, End));
  }
}

} // end anonymous namespace
