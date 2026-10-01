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
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

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
  Forest->visitOwnership([&](SSARegisterForest::PhysicalSpan Span,
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
