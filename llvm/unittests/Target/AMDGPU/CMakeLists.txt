include_directories(
  ${PROJECT_SOURCE_DIR}/lib/Target/AMDGPU
  ${PROJECT_BINARY_DIR}/lib/Target/AMDGPU
  )

set(LLVM_LINK_COMPONENTS
  AMDGPUCodeGen
  AMDGPUDesc
  AMDGPUInfo
  AMDGPUUtils
  CodeGen
  CodeGenTypes
  Core
  GlobalISel
  MC
  MIRParser
  Support
  TargetParser
  )

add_llvm_target_unittest(AMDGPUTests
  PARTIAL_SOURCES_INTENDED
  AMDGPUUnitTests.cpp
  CSETest.cpp
  DwarfRegMappings.cpp
  ExecMayBeModifiedBeforeAnyUse.cpp
  LiveRegUnits.cpp
  NextUseAnalysisTest.cpp
  PALMetadata.cpp
  SSARegisterForestTest.cpp
  SSARegisterForestAdapterTest.cpp
  SSARegisterTreeTest.cpp
  VRegMaskPairTest.cpp
  )

# Keep the placement-profile unit isolated from unrelated AMDGPU test sources:
# it exercises a pure fact representation and needs no target fixture.
add_llvm_target_unittest(SSAPlacementProfileTests
  PARTIAL_SOURCES_INTENDED
  SSAPlacementProfileTest.cpp
  )
