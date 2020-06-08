; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tahiti -verify-machineinstrs | FileCheck %s -check-prefixes=FUNC,GCN,SICIVI,SI
; RUN: llc < %s -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s -check-prefixes=FUNC,GCN,SICIVI,VI
; RUN: llc < %s -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=gfx900 -verify-machineinstrs | FileCheck %s -check-prefixes=FUNC,GCN,GFX9


declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32) nounwind readnone
declare { i64, i1 } @llvm.sadd.with.overflow.i64(i64, i64) nounwind readnone


declare { <2 x i32>, <2 x i1> } @llvm.sadd.with.overflow.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

define amdgpu_kernel void @saddo_i64_zext(i64 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
; SI-LABEL: saddo_i64_zext:
; SI:       ; %bb.0:
; SI-NEXT:    s_load_dwordx4 s[4:7], s[0:1], 0x9
; SI-NEXT:    s_load_dwordx2 s[8:9], s[0:1], 0xd
; SI-NEXT:    s_mov_b32 s3, 0xf000
; SI-NEXT:    s_mov_b32 s2, -1
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    v_mov_b32_e32 v0, s6
; SI-NEXT:    s_add_u32 s10, s6, s8
; SI-NEXT:    s_addc_u32 s11, s7, s9
; SI-NEXT:    v_mov_b32_e32 v1, s7
; SI-NEXT:    v_cmp_lt_i64_e32 vcc, s[10:11], v[0:1]
; SI-NEXT:    v_cmp_lt_i64_e64 s[6:7], s[8:9], 0
; SI-NEXT:    s_mov_b32 s0, s4
; SI-NEXT:    s_mov_b32 s1, s5
; SI-NEXT:    s_xor_b64 s[4:5], s[6:7], vcc
; SI-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; SI-NEXT:    v_mov_b32_e32 v1, s11
; SI-NEXT:    v_add_i32_e32 v0, vcc, s10, v0
; SI-NEXT:    v_addc_u32_e32 v1, vcc, 0, v1, vcc
; SI-NEXT:    buffer_store_dwordx2 v[0:1], off, s[0:3], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: saddo_i64_zext:
; VI:       ; %bb.0:
; VI-NEXT:    s_load_dwordx4 s[4:7], s[0:1], 0x24
; VI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x34
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    v_mov_b32_e32 v1, s6
; VI-NEXT:    s_add_u32 s2, s6, s0
; VI-NEXT:    s_addc_u32 s3, s7, s1
; VI-NEXT:    v_mov_b32_e32 v2, s7
; VI-NEXT:    v_cmp_lt_i64_e32 vcc, s[2:3], v[1:2]
; VI-NEXT:    v_cmp_lt_i64_e64 s[8:9], s[0:1], 0
; VI-NEXT:    v_mov_b32_e32 v3, s3
; VI-NEXT:    s_xor_b64 s[0:1], s[8:9], vcc
; VI-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[0:1]
; VI-NEXT:    v_add_u32_e32 v2, vcc, s2, v2
; VI-NEXT:    v_mov_b32_e32 v0, s4
; VI-NEXT:    v_mov_b32_e32 v1, s5
; VI-NEXT:    v_addc_u32_e32 v3, vcc, 0, v3, vcc
; VI-NEXT:    flat_store_dwordx2 v[0:1], v[2:3]
; VI-NEXT:    s_endpgm
;
; GFX9-LABEL: saddo_i64_zext:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_load_dwordx4 s[4:7], s[0:1], 0x24
; GFX9-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x34
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v1, s6
; GFX9-NEXT:    s_add_u32 s2, s6, s0
; GFX9-NEXT:    s_addc_u32 s3, s7, s1
; GFX9-NEXT:    v_mov_b32_e32 v2, s7
; GFX9-NEXT:    v_cmp_lt_i64_e32 vcc, s[2:3], v[1:2]
; GFX9-NEXT:    v_cmp_lt_i64_e64 s[8:9], s[0:1], 0
; GFX9-NEXT:    v_mov_b32_e32 v3, s3
; GFX9-NEXT:    s_xor_b64 s[0:1], s[8:9], vcc
; GFX9-NEXT:    v_cndmask_b32_e64 v2, 0, 1, s[0:1]
; GFX9-NEXT:    v_add_co_u32_e32 v2, vcc, s2, v2
; GFX9-NEXT:    v_mov_b32_e32 v0, s4
; GFX9-NEXT:    v_mov_b32_e32 v1, s5
; GFX9-NEXT:    v_addc_co_u32_e32 v3, vcc, 0, v3, vcc
; GFX9-NEXT:    global_store_dwordx2 v[0:1], v[2:3], off
; GFX9-NEXT:    s_endpgm
  %sadd = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %a, i64 %b) nounwind
  %val = extractvalue { i64, i1 } %sadd, 0
  %carry = extractvalue { i64, i1 } %sadd, 1
  %ext = zext i1 %carry to i64
  %add2 = add i64 %val, %ext
  store i64 %add2, i64 addrspace(1)* %out, align 8
  ret void
}

define amdgpu_kernel void @s_saddo_i32(i32 addrspace(1)* %out, i1 addrspace(1)* %carryout, i32 %a, i32 %b) nounwind {
; SI-LABEL: s_saddo_i32:
; SI:       ; %bb.0:
; SI-NEXT:    s_load_dwordx4 s[4:7], s[0:1], 0x9
; SI-NEXT:    s_load_dwordx2 s[8:9], s[0:1], 0xd
; SI-NEXT:    s_mov_b32 s3, 0xf000
; SI-NEXT:    s_mov_b32 s2, -1
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_mov_b32 s0, s4
; SI-NEXT:    s_add_i32 s12, s8, s9
; SI-NEXT:    s_cmp_lt_i32 s9, 0
; SI-NEXT:    s_cselect_b64 s[10:11], 1, 0
; SI-NEXT:    s_cmp_lt_i32 s12, s8
; SI-NEXT:    s_mov_b32 s1, s5
; SI-NEXT:    v_mov_b32_e32 v0, s12
; SI-NEXT:    s_cselect_b64 s[8:9], 1, 0
; SI-NEXT:    buffer_store_dword v0, off, s[0:3], 0
; SI-NEXT:    s_xor_b64 s[0:1], s[10:11], s[8:9]
; SI-NEXT:    s_mov_b32 s4, s6
; SI-NEXT:    s_mov_b32 s5, s7
; SI-NEXT:    s_mov_b32 s6, s2
; SI-NEXT:    s_mov_b32 s7, s3
; SI-NEXT:    s_waitcnt expcnt(0)
; SI-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[0:1]
; SI-NEXT:    buffer_store_byte v0, off, s[4:7], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: s_saddo_i32:
; VI:       ; %bb.0:
; VI-NEXT:    s_load_dwordx4 s[4:7], s[0:1], 0x24
; VI-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x34
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    v_mov_b32_e32 v0, s4
; VI-NEXT:    s_add_i32 s4, s0, s1
; VI-NEXT:    s_cmp_lt_i32 s1, 0
; VI-NEXT:    s_cselect_b64 s[2:3], 1, 0
; VI-NEXT:    s_cmp_lt_i32 s4, s0
; VI-NEXT:    s_cselect_b64 s[0:1], 1, 0
; VI-NEXT:    v_mov_b32_e32 v1, s5
; VI-NEXT:    v_mov_b32_e32 v4, s4
; VI-NEXT:    s_xor_b64 s[0:1], s[2:3], s[0:1]
; VI-NEXT:    flat_store_dword v[0:1], v4
; VI-NEXT:    v_mov_b32_e32 v2, s6
; VI-NEXT:    v_mov_b32_e32 v3, s7
; VI-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[0:1]
; VI-NEXT:    flat_store_byte v[2:3], v0
; VI-NEXT:    s_endpgm
;
; GFX9-LABEL: s_saddo_i32:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_load_dwordx4 s[4:7], s[0:1], 0x24
; GFX9-NEXT:    s_load_dwordx2 s[0:1], s[0:1], 0x34
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v0, s4
; GFX9-NEXT:    s_add_i32 s4, s0, s1
; GFX9-NEXT:    s_cmp_lt_i32 s1, 0
; GFX9-NEXT:    s_cselect_b64 s[2:3], 1, 0
; GFX9-NEXT:    s_cmp_lt_i32 s4, s0
; GFX9-NEXT:    s_cselect_b64 s[0:1], 1, 0
; GFX9-NEXT:    v_mov_b32_e32 v1, s5
; GFX9-NEXT:    v_mov_b32_e32 v4, s4
; GFX9-NEXT:    s_xor_b64 s[0:1], s[2:3], s[0:1]
; GFX9-NEXT:    global_store_dword v[0:1], v4, off
; GFX9-NEXT:    v_mov_b32_e32 v2, s6
; GFX9-NEXT:    v_mov_b32_e32 v3, s7
; GFX9-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[0:1]
; GFX9-NEXT:    global_store_byte v[2:3], v0, off
; GFX9-NEXT:    s_endpgm
  %sadd = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %a, i32 %b) nounwind
  %val = extractvalue { i32, i1 } %sadd, 0
  %carry = extractvalue { i32, i1 } %sadd, 1
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

define amdgpu_kernel void @v_saddo_i32(i32 addrspace(1)* %out, i1 addrspace(1)* %carryout, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
; SI-LABEL: v_saddo_i32:
; SI:       ; %bb.0:
; SI-NEXT:    s_load_dwordx8 s[0:7], s[0:1], 0x9
; SI-NEXT:    s_mov_b32 s11, 0xf000
; SI-NEXT:    s_mov_b32 s10, -1
; SI-NEXT:    s_mov_b32 s14, s10
; SI-NEXT:    s_mov_b32 s15, s11
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_mov_b32 s8, s0
; SI-NEXT:    s_mov_b32 s9, s1
; SI-NEXT:    s_mov_b32 s12, s2
; SI-NEXT:    s_mov_b32 s13, s3
; SI-NEXT:    s_mov_b32 s0, s4
; SI-NEXT:    s_mov_b32 s1, s5
; SI-NEXT:    s_mov_b32 s2, s10
; SI-NEXT:    s_mov_b32 s3, s11
; SI-NEXT:    s_mov_b32 s4, s6
; SI-NEXT:    s_mov_b32 s5, s7
; SI-NEXT:    s_mov_b32 s6, s10
; SI-NEXT:    s_mov_b32 s7, s11
; SI-NEXT:    buffer_load_dword v0, off, s[0:3], 0
; SI-NEXT:    buffer_load_dword v1, off, s[4:7], 0
; SI-NEXT:    s_waitcnt vmcnt(0)
; SI-NEXT:    v_add_i32_e32 v2, vcc, v1, v0
; SI-NEXT:    v_cmp_gt_i32_e32 vcc, 0, v1
; SI-NEXT:    v_cmp_lt_i32_e64 s[0:1], v2, v0
; SI-NEXT:    s_xor_b64 s[0:1], vcc, s[0:1]
; SI-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[0:1]
; SI-NEXT:    buffer_store_dword v2, off, s[8:11], 0
; SI-NEXT:    buffer_store_byte v0, off, s[12:15], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: v_saddo_i32:
; VI:       ; %bb.0:
; VI-NEXT:    s_load_dwordx8 s[0:7], s[0:1], 0x24
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    v_mov_b32_e32 v4, s4
; VI-NEXT:    v_mov_b32_e32 v5, s5
; VI-NEXT:    v_mov_b32_e32 v6, s6
; VI-NEXT:    v_mov_b32_e32 v7, s7
; VI-NEXT:    flat_load_dword v4, v[4:5]
; VI-NEXT:    flat_load_dword v5, v[6:7]
; VI-NEXT:    v_mov_b32_e32 v0, s0
; VI-NEXT:    v_mov_b32_e32 v1, s1
; VI-NEXT:    v_mov_b32_e32 v2, s2
; VI-NEXT:    v_mov_b32_e32 v3, s3
; VI-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; VI-NEXT:    v_add_u32_e32 v6, vcc, v5, v4
; VI-NEXT:    v_cmp_gt_i32_e32 vcc, 0, v5
; VI-NEXT:    v_cmp_lt_i32_e64 s[0:1], v6, v4
; VI-NEXT:    s_xor_b64 s[0:1], vcc, s[0:1]
; VI-NEXT:    flat_store_dword v[0:1], v6
; VI-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[0:1]
; VI-NEXT:    flat_store_byte v[2:3], v0
; VI-NEXT:    s_endpgm
;
; GFX9-LABEL: v_saddo_i32:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_load_dwordx8 s[0:7], s[0:1], 0x24
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v4, s4
; GFX9-NEXT:    v_mov_b32_e32 v5, s5
; GFX9-NEXT:    v_mov_b32_e32 v6, s6
; GFX9-NEXT:    v_mov_b32_e32 v7, s7
; GFX9-NEXT:    global_load_dword v4, v[4:5], off
; GFX9-NEXT:    global_load_dword v5, v[6:7], off
; GFX9-NEXT:    v_mov_b32_e32 v0, s0
; GFX9-NEXT:    v_mov_b32_e32 v1, s1
; GFX9-NEXT:    v_mov_b32_e32 v2, s2
; GFX9-NEXT:    v_mov_b32_e32 v3, s3
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_add_u32_e32 v6, v4, v5
; GFX9-NEXT:    v_cmp_gt_i32_e32 vcc, 0, v5
; GFX9-NEXT:    v_cmp_lt_i32_e64 s[0:1], v6, v4
; GFX9-NEXT:    s_xor_b64 s[0:1], vcc, s[0:1]
; GFX9-NEXT:    global_store_dword v[0:1], v6, off
; GFX9-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[0:1]
; GFX9-NEXT:    global_store_byte v[2:3], v0, off
; GFX9-NEXT:    s_endpgm
  %a = load i32, i32 addrspace(1)* %aptr, align 4
  %b = load i32, i32 addrspace(1)* %bptr, align 4
  %sadd = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %a, i32 %b) nounwind
  %val = extractvalue { i32, i1 } %sadd, 0
  %carry = extractvalue { i32, i1 } %sadd, 1
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

define amdgpu_kernel void @s_saddo_i64(i64 addrspace(1)* %out, i1 addrspace(1)* %carryout, i64 %a, i64 %b) nounwind {
; SI-LABEL: s_saddo_i64:
; SI:       ; %bb.0:
; SI-NEXT:    s_load_dwordx8 s[0:7], s[0:1], 0x9
; SI-NEXT:    s_mov_b32 s11, 0xf000
; SI-NEXT:    s_mov_b32 s10, -1
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_add_u32 s12, s4, s6
; SI-NEXT:    v_mov_b32_e32 v0, s4
; SI-NEXT:    s_addc_u32 s13, s5, s7
; SI-NEXT:    v_mov_b32_e32 v1, s5
; SI-NEXT:    v_cmp_lt_i64_e32 vcc, s[12:13], v[0:1]
; SI-NEXT:    v_cmp_lt_i64_e64 s[4:5], s[6:7], 0
; SI-NEXT:    v_mov_b32_e32 v0, s12
; SI-NEXT:    s_mov_b32 s8, s0
; SI-NEXT:    s_mov_b32 s9, s1
; SI-NEXT:    v_mov_b32_e32 v1, s13
; SI-NEXT:    s_xor_b64 s[4:5], s[4:5], vcc
; SI-NEXT:    s_mov_b32 s0, s2
; SI-NEXT:    s_mov_b32 s1, s3
; SI-NEXT:    buffer_store_dwordx2 v[0:1], off, s[8:11], 0
; SI-NEXT:    s_mov_b32 s2, s10
; SI-NEXT:    s_mov_b32 s3, s11
; SI-NEXT:    s_waitcnt expcnt(0)
; SI-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[4:5]
; SI-NEXT:    buffer_store_byte v0, off, s[0:3], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: s_saddo_i64:
; VI:       ; %bb.0:
; VI-NEXT:    s_load_dwordx8 s[0:7], s[0:1], 0x24
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    v_mov_b32_e32 v0, s0
; VI-NEXT:    v_mov_b32_e32 v4, s4
; VI-NEXT:    s_add_u32 s0, s4, s6
; VI-NEXT:    v_mov_b32_e32 v1, s1
; VI-NEXT:    s_addc_u32 s1, s5, s7
; VI-NEXT:    v_mov_b32_e32 v5, s5
; VI-NEXT:    v_cmp_lt_i64_e32 vcc, s[0:1], v[4:5]
; VI-NEXT:    v_mov_b32_e32 v2, s2
; VI-NEXT:    v_mov_b32_e32 v3, s3
; VI-NEXT:    v_cmp_lt_i64_e64 s[2:3], s[6:7], 0
; VI-NEXT:    v_mov_b32_e32 v5, s1
; VI-NEXT:    v_mov_b32_e32 v4, s0
; VI-NEXT:    s_xor_b64 s[0:1], s[2:3], vcc
; VI-NEXT:    flat_store_dwordx2 v[0:1], v[4:5]
; VI-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[0:1]
; VI-NEXT:    flat_store_byte v[2:3], v0
; VI-NEXT:    s_endpgm
;
; GFX9-LABEL: s_saddo_i64:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_load_dwordx8 s[0:7], s[0:1], 0x24
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v0, s0
; GFX9-NEXT:    v_mov_b32_e32 v4, s4
; GFX9-NEXT:    s_add_u32 s0, s4, s6
; GFX9-NEXT:    v_mov_b32_e32 v1, s1
; GFX9-NEXT:    s_addc_u32 s1, s5, s7
; GFX9-NEXT:    v_mov_b32_e32 v5, s5
; GFX9-NEXT:    v_cmp_lt_i64_e32 vcc, s[0:1], v[4:5]
; GFX9-NEXT:    v_mov_b32_e32 v2, s2
; GFX9-NEXT:    v_mov_b32_e32 v3, s3
; GFX9-NEXT:    v_cmp_lt_i64_e64 s[2:3], s[6:7], 0
; GFX9-NEXT:    v_mov_b32_e32 v5, s1
; GFX9-NEXT:    v_mov_b32_e32 v4, s0
; GFX9-NEXT:    s_xor_b64 s[0:1], s[2:3], vcc
; GFX9-NEXT:    global_store_dwordx2 v[0:1], v[4:5], off
; GFX9-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[0:1]
; GFX9-NEXT:    global_store_byte v[2:3], v0, off
; GFX9-NEXT:    s_endpgm
  %sadd = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %a, i64 %b) nounwind
  %val = extractvalue { i64, i1 } %sadd, 0
  %carry = extractvalue { i64, i1 } %sadd, 1
  store i64 %val, i64 addrspace(1)* %out, align 8
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

define amdgpu_kernel void @v_saddo_i64(i64 addrspace(1)* %out, i1 addrspace(1)* %carryout, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr) nounwind {
; SI-LABEL: v_saddo_i64:
; SI:       ; %bb.0:
; SI-NEXT:    s_load_dwordx8 s[0:7], s[0:1], 0x9
; SI-NEXT:    s_mov_b32 s11, 0xf000
; SI-NEXT:    s_mov_b32 s10, -1
; SI-NEXT:    s_mov_b32 s14, s10
; SI-NEXT:    s_mov_b32 s15, s11
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_mov_b32 s8, s0
; SI-NEXT:    s_mov_b32 s9, s1
; SI-NEXT:    s_mov_b32 s12, s2
; SI-NEXT:    s_mov_b32 s13, s3
; SI-NEXT:    s_mov_b32 s0, s4
; SI-NEXT:    s_mov_b32 s1, s5
; SI-NEXT:    s_mov_b32 s2, s10
; SI-NEXT:    s_mov_b32 s3, s11
; SI-NEXT:    s_mov_b32 s4, s6
; SI-NEXT:    s_mov_b32 s5, s7
; SI-NEXT:    s_mov_b32 s6, s10
; SI-NEXT:    s_mov_b32 s7, s11
; SI-NEXT:    buffer_load_dwordx2 v[0:1], off, s[0:3], 0
; SI-NEXT:    buffer_load_dwordx2 v[2:3], off, s[4:7], 0
; SI-NEXT:    s_waitcnt vmcnt(0)
; SI-NEXT:    v_add_i32_e32 v4, vcc, v0, v2
; SI-NEXT:    v_addc_u32_e32 v5, vcc, v1, v3, vcc
; SI-NEXT:    v_cmp_gt_i64_e32 vcc, 0, v[2:3]
; SI-NEXT:    v_cmp_lt_i64_e64 s[0:1], v[4:5], v[0:1]
; SI-NEXT:    buffer_store_dwordx2 v[4:5], off, s[8:11], 0
; SI-NEXT:    s_xor_b64 s[0:1], vcc, s[0:1]
; SI-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[0:1]
; SI-NEXT:    buffer_store_byte v0, off, s[12:15], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: v_saddo_i64:
; VI:       ; %bb.0:
; VI-NEXT:    s_load_dwordx8 s[0:7], s[0:1], 0x24
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    v_mov_b32_e32 v4, s4
; VI-NEXT:    v_mov_b32_e32 v5, s5
; VI-NEXT:    v_mov_b32_e32 v6, s6
; VI-NEXT:    v_mov_b32_e32 v7, s7
; VI-NEXT:    flat_load_dwordx2 v[4:5], v[4:5]
; VI-NEXT:    flat_load_dwordx2 v[6:7], v[6:7]
; VI-NEXT:    v_mov_b32_e32 v0, s0
; VI-NEXT:    v_mov_b32_e32 v1, s1
; VI-NEXT:    v_mov_b32_e32 v2, s2
; VI-NEXT:    v_mov_b32_e32 v3, s3
; VI-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; VI-NEXT:    v_add_u32_e32 v8, vcc, v4, v6
; VI-NEXT:    v_addc_u32_e32 v9, vcc, v5, v7, vcc
; VI-NEXT:    v_cmp_gt_i64_e32 vcc, 0, v[6:7]
; VI-NEXT:    v_cmp_lt_i64_e64 s[0:1], v[8:9], v[4:5]
; VI-NEXT:    flat_store_dwordx2 v[0:1], v[8:9]
; VI-NEXT:    s_xor_b64 s[0:1], vcc, s[0:1]
; VI-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[0:1]
; VI-NEXT:    flat_store_byte v[2:3], v0
; VI-NEXT:    s_endpgm
;
; GFX9-LABEL: v_saddo_i64:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_load_dwordx8 s[0:7], s[0:1], 0x24
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v4, s4
; GFX9-NEXT:    v_mov_b32_e32 v5, s5
; GFX9-NEXT:    v_mov_b32_e32 v6, s6
; GFX9-NEXT:    v_mov_b32_e32 v7, s7
; GFX9-NEXT:    global_load_dwordx2 v[4:5], v[4:5], off
; GFX9-NEXT:    global_load_dwordx2 v[6:7], v[6:7], off
; GFX9-NEXT:    v_mov_b32_e32 v0, s0
; GFX9-NEXT:    v_mov_b32_e32 v1, s1
; GFX9-NEXT:    v_mov_b32_e32 v2, s2
; GFX9-NEXT:    v_mov_b32_e32 v3, s3
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_add_co_u32_e32 v8, vcc, v4, v6
; GFX9-NEXT:    v_addc_co_u32_e32 v9, vcc, v5, v7, vcc
; GFX9-NEXT:    v_cmp_gt_i64_e32 vcc, 0, v[6:7]
; GFX9-NEXT:    v_cmp_lt_i64_e64 s[0:1], v[8:9], v[4:5]
; GFX9-NEXT:    global_store_dwordx2 v[0:1], v[8:9], off
; GFX9-NEXT:    s_xor_b64 s[0:1], vcc, s[0:1]
; GFX9-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[0:1]
; GFX9-NEXT:    global_store_byte v[2:3], v0, off
; GFX9-NEXT:    s_endpgm
  %a = load i64, i64 addrspace(1)* %aptr, align 4
  %b = load i64, i64 addrspace(1)* %bptr, align 4
  %sadd = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %a, i64 %b) nounwind
  %val = extractvalue { i64, i1 } %sadd, 0
  %carry = extractvalue { i64, i1 } %sadd, 1
  store i64 %val, i64 addrspace(1)* %out, align 8
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

define amdgpu_kernel void @v_saddo_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %carryout, <2 x i32> addrspace(1)* %aptr, <2 x i32> addrspace(1)* %bptr) nounwind {
; SI-LABEL: v_saddo_v2i32:
; SI:       ; %bb.0:
; SI-NEXT:    s_load_dwordx8 s[0:7], s[0:1], 0x9
; SI-NEXT:    s_mov_b32 s11, 0xf000
; SI-NEXT:    s_mov_b32 s10, -1
; SI-NEXT:    s_mov_b32 s14, s10
; SI-NEXT:    s_mov_b32 s15, s11
; SI-NEXT:    s_waitcnt lgkmcnt(0)
; SI-NEXT:    s_mov_b32 s8, s0
; SI-NEXT:    s_mov_b32 s9, s1
; SI-NEXT:    s_mov_b32 s12, s2
; SI-NEXT:    s_mov_b32 s13, s3
; SI-NEXT:    s_mov_b32 s0, s4
; SI-NEXT:    s_mov_b32 s1, s5
; SI-NEXT:    s_mov_b32 s2, s10
; SI-NEXT:    s_mov_b32 s3, s11
; SI-NEXT:    s_mov_b32 s4, s6
; SI-NEXT:    s_mov_b32 s5, s7
; SI-NEXT:    s_mov_b32 s6, s10
; SI-NEXT:    s_mov_b32 s7, s11
; SI-NEXT:    buffer_load_dwordx2 v[0:1], off, s[0:3], 0
; SI-NEXT:    buffer_load_dwordx2 v[2:3], off, s[4:7], 0
; SI-NEXT:    s_waitcnt vmcnt(0)
; SI-NEXT:    v_add_i32_e32 v5, vcc, v1, v3
; SI-NEXT:    v_add_i32_e32 v4, vcc, v0, v2
; SI-NEXT:    v_cmp_gt_i32_e64 s[0:1], 0, v3
; SI-NEXT:    v_cmp_lt_i32_e64 s[4:5], v5, v1
; SI-NEXT:    s_xor_b64 s[0:1], s[0:1], s[4:5]
; SI-NEXT:    v_cmp_gt_i32_e32 vcc, 0, v2
; SI-NEXT:    v_cmp_lt_i32_e64 s[2:3], v4, v0
; SI-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[0:1]
; SI-NEXT:    s_xor_b64 s[0:1], vcc, s[2:3]
; SI-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[0:1]
; SI-NEXT:    buffer_store_dwordx2 v[4:5], off, s[8:11], 0
; SI-NEXT:    buffer_store_dwordx2 v[0:1], off, s[12:15], 0
; SI-NEXT:    s_endpgm
;
; VI-LABEL: v_saddo_v2i32:
; VI:       ; %bb.0:
; VI-NEXT:    s_load_dwordx8 s[0:7], s[0:1], 0x24
; VI-NEXT:    s_waitcnt lgkmcnt(0)
; VI-NEXT:    v_mov_b32_e32 v4, s4
; VI-NEXT:    v_mov_b32_e32 v5, s5
; VI-NEXT:    v_mov_b32_e32 v6, s6
; VI-NEXT:    v_mov_b32_e32 v7, s7
; VI-NEXT:    flat_load_dwordx2 v[4:5], v[4:5]
; VI-NEXT:    flat_load_dwordx2 v[6:7], v[6:7]
; VI-NEXT:    v_mov_b32_e32 v0, s0
; VI-NEXT:    v_mov_b32_e32 v1, s1
; VI-NEXT:    v_mov_b32_e32 v2, s2
; VI-NEXT:    v_mov_b32_e32 v3, s3
; VI-NEXT:    s_waitcnt vmcnt(0) lgkmcnt(0)
; VI-NEXT:    v_add_u32_e32 v9, vcc, v5, v7
; VI-NEXT:    v_add_u32_e32 v8, vcc, v4, v6
; VI-NEXT:    v_cmp_gt_i32_e64 s[0:1], 0, v7
; VI-NEXT:    v_cmp_lt_i32_e64 s[4:5], v9, v5
; VI-NEXT:    v_cmp_gt_i32_e32 vcc, 0, v6
; VI-NEXT:    v_cmp_lt_i32_e64 s[2:3], v8, v4
; VI-NEXT:    s_xor_b64 s[0:1], s[0:1], s[4:5]
; VI-NEXT:    flat_store_dwordx2 v[0:1], v[8:9]
; VI-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[0:1]
; VI-NEXT:    s_xor_b64 s[0:1], vcc, s[2:3]
; VI-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[0:1]
; VI-NEXT:    flat_store_dwordx2 v[2:3], v[0:1]
; VI-NEXT:    s_endpgm
;
; GFX9-LABEL: v_saddo_v2i32:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_load_dwordx8 s[0:7], s[0:1], 0x24
; GFX9-NEXT:    s_waitcnt lgkmcnt(0)
; GFX9-NEXT:    v_mov_b32_e32 v4, s4
; GFX9-NEXT:    v_mov_b32_e32 v5, s5
; GFX9-NEXT:    v_mov_b32_e32 v6, s6
; GFX9-NEXT:    v_mov_b32_e32 v7, s7
; GFX9-NEXT:    global_load_dwordx2 v[4:5], v[4:5], off
; GFX9-NEXT:    global_load_dwordx2 v[6:7], v[6:7], off
; GFX9-NEXT:    v_mov_b32_e32 v0, s0
; GFX9-NEXT:    v_mov_b32_e32 v1, s1
; GFX9-NEXT:    v_mov_b32_e32 v2, s2
; GFX9-NEXT:    v_mov_b32_e32 v3, s3
; GFX9-NEXT:    s_waitcnt vmcnt(0)
; GFX9-NEXT:    v_add_u32_e32 v9, v5, v7
; GFX9-NEXT:    v_add_u32_e32 v8, v4, v6
; GFX9-NEXT:    v_cmp_gt_i32_e64 s[0:1], 0, v7
; GFX9-NEXT:    v_cmp_lt_i32_e64 s[4:5], v9, v5
; GFX9-NEXT:    v_cmp_gt_i32_e32 vcc, 0, v6
; GFX9-NEXT:    v_cmp_lt_i32_e64 s[2:3], v8, v4
; GFX9-NEXT:    s_xor_b64 s[0:1], s[0:1], s[4:5]
; GFX9-NEXT:    global_store_dwordx2 v[0:1], v[8:9], off
; GFX9-NEXT:    v_cndmask_b32_e64 v1, 0, 1, s[0:1]
; GFX9-NEXT:    s_xor_b64 s[0:1], vcc, s[2:3]
; GFX9-NEXT:    v_cndmask_b32_e64 v0, 0, 1, s[0:1]
; GFX9-NEXT:    global_store_dwordx2 v[2:3], v[0:1], off
; GFX9-NEXT:    s_endpgm
  %a = load <2 x i32>, <2 x i32> addrspace(1)* %aptr, align 4
  %b = load <2 x i32>, <2 x i32> addrspace(1)* %bptr, align 4
  %sadd = call { <2 x i32>, <2 x i1> } @llvm.sadd.with.overflow.v2i32(<2 x i32> %a, <2 x i32> %b) nounwind
  %val = extractvalue { <2 x i32>, <2 x i1> } %sadd, 0
  %carry = extractvalue { <2 x i32>, <2 x i1> } %sadd, 1
  store <2 x i32> %val, <2 x i32> addrspace(1)* %out, align 4
  %carry.ext = zext <2 x i1> %carry to <2 x i32>
  store <2 x i32> %carry.ext, <2 x i32> addrspace(1)* %carryout
  ret void
}
