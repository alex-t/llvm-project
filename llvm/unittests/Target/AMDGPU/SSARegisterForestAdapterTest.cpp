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

  auto Forest = SSARegisterForest::create(2, 4);
  ASSERT_TRUE(Forest);
  // Deliberately nonnumeric target preference, with leaves 2 and 5 omitted.
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
  const MCPhysReg Leaves[] = {
      AMDGPU::VGPR0, AMDGPU::VGPR1, AMDGPU::VGPR2, AMDGPU::VGPR3,
      AMDGPU::VGPR4, AMDGPU::VGPR5, AMDGPU::VGPR6, AMDGPU::VGPR7};

  // Check the forwarded order against physical occupancy. Reindexing the
  // filtered list would wrongly occupy a gap or a lower-priority leaf.
  ASSERT_TRUE(Adapter.assign(PairVR, Homes.front(), Start, End));
  for (unsigned I = 0; I != 8; ++I)
    EXPECT_EQ(Adapter.isFree(Leaves[I], Start, End), I < 6);
  ASSERT_TRUE(Adapter.release(PairVR, Homes.front(), Start, End));

  struct MappingCase {
    MCPhysReg PR;
    unsigned First;
    unsigned End;
  };
  // Independent physical oracle: all seven canonical nodes of the first tree,
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
      EXPECT_EQ(Adapter.isFree(Leaves[I], Start, End), ExpectedFree);
      EXPECT_TRUE(Adapter.isFree(Leaves[I], End, After));
    }
    if (C.End - C.First > 1) {
      EXPECT_FALSE(Adapter.contains(VR, Leaves[C.First], Start, End));
      EXPECT_FALSE(Adapter.release(VR, Leaves[C.First], Start, End));
      EXPECT_TRUE(Adapter.contains(VR, C.PR, Start, End));
    }
    ASSERT_TRUE(Adapter.release(VR, C.PR, Start, End));
    for (MCPhysReg PR : Leaves)
      EXPECT_TRUE(Adapter.isFree(PR, Start, End));
  }

  // A separate invocation uses SGPR identities on the same leaf geometry.
  auto ScalarForest = SSARegisterForest::create(2, 4);
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

} // end anonymous namespace
