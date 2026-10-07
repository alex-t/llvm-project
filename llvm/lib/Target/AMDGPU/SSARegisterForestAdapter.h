//===- SSARegisterForestAdapter.h - Target homes for RF ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SSAREGISTERFORESTADAPTER_H
#define LLVM_LIB_TARGET_AMDGPU_SSAREGISTERFORESTADAPTER_H

#include "SSARegisterForest.h"
#include "VRegMaskPair.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/MC/MCRegister.h"
#include <utility>

namespace llvm {

class LiveInterval;
class LiveIntervals;
class LiveRange;
class MachineRegisterInfo;
class SIRegisterInfo;
class TargetRegisterClass;

/// Translate target register identities to temporal forest operations.
///
/// One instance serves one physical register file. PR arguments must be concrete
/// storage registers in that file. Forest coordinates count 16-bit leaves:
/// a 32-bit register covers two leaves; its low and high halves cover one each.
/// To group four 32-bit registers per tree, construct Forest with TreeWidth=8
/// (15 nodes per tree). The caller chooses legal homes; these operations do not
/// select or revalidate candidates.
/// The referenced forest, MRI and TRI must outlive this adapter.
class RegisterForestAdapter {
public:
  /// Build storage for one concrete SGPR, VGPR or AGPR file. Topology includes
  /// target coordinate holes and uses trees of four dwords (eight half leaves).
  static std::optional<SSARegisterForest>
  createForest(const TargetRegisterClass &File, const SIRegisterInfo &TRI);

  /// Assign a full target home with its current lane live ranges from LIS.
  /// Reject an invalid home, mask clobber or overlapping ownership atomically.
  bool assign(Register VR, MCRegister PR, LiveIntervals &LIS);

  /// Supply the allocator's existing availableOrder(RC), including reservations
  /// and the withheld vector-register tail. The returned storage must remain
  /// valid while the caller traverses the order. The callable is owned; objects
  /// it references must outlive this adapter.
  using OrderProvider =
      unique_function<ArrayRef<MCPhysReg>(const TargetRegisterClass *) const>;

  RegisterForestAdapter(SSARegisterForest &Forest, const SIRegisterInfo &TRI,
                        const MachineRegisterInfo &MRI, OrderProvider GetOrder)
      : Forest(Forest), TRI(TRI), MRI(MRI), GetOrder(std::move(GetOrder)) {}

  /// Return legal physical homes for VR in the allocator's target order.
  /// Omitted registers remain holes in physical coordinates.
  ArrayRef<MCPhysReg> allocationOrder(Register VR) const;

  /// Lanes retained over one half-open time interval. Masks are relative to
  /// the virtual register, not to RF coordinates or physical subregister IDs.
  struct RetainedRegion {
    VRegMaskPair Value;
    SlotIndex Start;
    SlotIndex End;
  };

  bool isFree(MCRegister PR, SlotIndex Start, SlotIndex End) const;
  /// Visit stored ownership components interfering with PR over [Start, End).
  /// Mapping stays private; callbacks receive complete stored intervals and
  /// owners (including SELF_OWNED), without forest coordinates. Owners can appear more than
  /// once. The callback must not mutate the forest. Return false for invalid
  /// input without callbacks; true for a valid query, even with no interference.
  bool visitInterferences(
      MCRegister PR, SlotIndex Start, SlotIndex End,
      function_ref<void(const SSARegisterForest::Ownership &)> Visit) const;
  /// Query each disjoint segment of Probe separately. The callback receives
  /// the owner and its occupied time clipped to that probe segment. An owner
  /// spanning a probe gap is reported as separate pieces; gap-only owners are
  /// omitted. Composite ownership can also produce multiple callbacks.
  /// Probe must satisfy LiveRange invariants. A valid empty probe succeeds
  /// without callbacks. Invalid PR returns false. The callback must not mutate
  /// the forest or Probe. This overload does not inspect lane subranges.
  bool visitInterferences(
      MCRegister PR, const LiveRange &Probe,
      function_ref<void(Register Owner, SlotIndex Start, SlotIndex End)> Visit)
      const;

  /// Import fixed physical lifetimes before virtual assignments begin.
  /// Registers are concrete physical operands/live-ins found by RA, all in
  /// this adapter's file. LIS supplies their unit ranges for the same function,
  /// computing uncached ranges on demand. Aliases and overlapping unit ranges
  /// are merged per leaf; liveness gaps remain free. Dead defs retain their
  /// short unit intervals. Register-mask clobbers are not unit lifetimes.
  /// Call once to initialize an epoch. This imports the current LIS state;
  /// subsequent physical-lifetime edits must be mirrored separately in RF.
  /// Invalid mapping or conflicting ownership returns false without insertion.
  bool assignFixed(ArrayRef<MCPhysReg> Registers, LiveIntervals &LIS);

  struct Interference {
    SmallVector<Register, 4> VirtualOwners;
    bool HasFixedInterference = false;
  };

  /// PR is the proposed home; Probe supplies live lanes and disjoint segments.
  /// Query stored virtual and SELF_OWNED records in one traversal. Fixed
  /// conflicts cannot be removed by relocating VirtualOwners. nullopt means
  /// invalid input, not a free home. Register-mask clobbers remain separate.
  std::optional<Interference> interferences(MCRegister PR,
                                          const LiveInterval &Probe) const;

  /// Include call/inline-asm regmask constraints from LIS in the query above.
  /// PR is the proposed home; Probe supplies its disjoint live segments. LIS
  /// must describe the same function as this adapter. A rejected mask sets
  /// HasFixedInterference, never a movable VirtualOwner. Mask checks apply to
  /// the whole PR over the main interval, following LiveIntervals semantics.
  /// Ownership queries remain lane-aware. Does not mutate ownership or MIR.
  std::optional<Interference> interferences(MCRegister PR,
                                          const LiveInterval &Probe,
                                          LiveIntervals &LIS) const;

  /// Return distinct virtual owners intersecting Probe placed at PR. Query
  /// each lane subrange and each of its disjoint segments; without subranges,
  /// use the full virtual-register mask. The result is sorted by register ID.
  /// Stored ownership supplies blocker lanes. No owner is excluded, including
  /// Probe.reg() if already assigned. This query does not mutate ownership.
  /// An empty vector means no virtual interference; nullopt means an invalid
  /// owner, home or lane projection. This compatibility query omits fixed
  /// conflicts; use interferences() for placement legality. Probe must satisfy
  /// LiveInterval invariants.
  std::optional<SmallVector<Register, 4>>
  interferingOwners(MCRegister PR, const LiveInterval &Probe) const;

  /// Constant-time lookup of the full home stored in RF, including currently
  /// dead lanes. Zero means unassigned; no target home is inferred from spans.
  MCRegister assignedHome(Register VR) const { return Forest.assignedHome(VR); }

  /// Remove stored ownership and its home, even after VR's interval is retired.
  /// False means no assignment exists; a caller expecting one must report it.
  bool unassign(Register VR) { return Forest.releaseOwner(VR); }

  /// Consume the complete affected set after SSA/LIS repair in this register
  /// file. RF supplies old ownership; LIS supplies the final repaired intervals.
  /// The producer must not query allocation state during this synchronous call.
  /// Duplicate IDs are allowed. New unassigned VRs remain unassigned.
  /// Malformed input or failed projection aborts before any RF mutation;
  /// retirement and legal projection into an occupied home are distinct cases.
  ///
  /// Return owners whose assignments were cleared: retired/empty values and
  /// values unable to retain their home. RA must update its bookkeeping and
  /// recover surviving uncolored values before resuming allocation. Unchanged
  /// regions keep their records. Unrelated owners are never evicted. Conflicting
  /// additions invalidate both affected homes, without choosing a winner.
  /// A VGPR/AGPR transfer must notify both file adapters explicitly.
  SmallVector<Register, 4> onChange(ArrayRef<Register> Affected,
                                  LiveIntervals &LIS);

  bool contains(Register VR, MCRegister PR, SlotIndex Start, SlotIndex End) const;
  bool assign(Register VR, MCRegister PR, SlotIndex Start, SlotIndex End);
  bool release(Register VR, MCRegister PR, SlotIndex Start, SlotIndex End);

  /// Insert only the supplied live lanes and time intervals. Validate the
  /// complete batch before mutation; overlapping regions are rejected, even
  /// for the same owner. Empty Regions records the full home with no occupancy.
  bool assign(Register VR, MCRegister PR, ArrayRef<RetainedRegion> Regions);

  /// Refine the exact whole-home assignment {VR, PR, Start, End}. Each retained
  /// region must name VR and a nonempty mask within its class's lane mask.
  /// Sparse masks retain their holes. RF checks containment and non-overlap of
  /// the complete replacement before mutation. Empty Retained removes the
  /// assignment. Callers supply time bounds; this method does not read or alter
  /// live intervals. Each emitted contiguous span becomes an exact assignment.
  bool replace(Register VR, MCRegister PR, SlotIndex Start, SlotIndex End,
               ArrayRef<RetainedRegion> Retained);

private:
  /// Return null for a nonvirtual, unknown or classless register.
  static const TargetRegisterClass *
  getVirtualRegClassOrNull(Register VR, const MachineRegisterInfo &MRI);

  /// Return null for a missing, unknown or unsupported physical register.
  static const TargetRegisterClass *
  getPhysicalRegClassOrNull(MCRegister PR, const SIRegisterInfo &TRI);

  /// Visit each lane subrange, or the main range with the full class mask.
  /// Return false for an invalid virtual register or when Visit returns false.
  /// Visit must not mutate LI. Projection and segment processing belong to
  /// the caller, so physical spans can be computed once per range.
  bool visitLiveRanges(
      const LiveInterval &LI,
      function_ref<bool(const LiveRange &, LaneBitmask)> Visit) const;

  /// PR has already been validated. Probe and LIS belong to this function.
  /// Use LLVM's regmask boundary rules, including special live-through uses.
  bool hasRegMaskInterference(MCRegister PR, const LiveInterval &Probe,
                              LiveIntervals &LIS) const;

  static unsigned firstPhysicalLeaf(MCRegister PR, unsigned Bits,
                                    const SIRegisterInfo &TRI);

  SSARegisterForest &Forest;
  const SIRegisterInfo &TRI;
  const MachineRegisterInfo &MRI;
  OrderProvider GetOrder;

  using Regions = SmallVector<SSARegisterForest::OwnershipRegion, 8>;
  /// Append projected regions on success; leave Out unchanged on failure.
  /// LIS must contain VR. Does not mutate LIS or forest ownership.
  bool projectLiveRegions(Register VR, MCRegister PR, LiveIntervals &LIS,
                          Regions &Out) const;
  void releaseRegions(Register VR, ArrayRef<SSARegisterForest::OwnershipRegion>);

  /// RF alone decomposes this span into canonical nodes. An unsupported size
  /// or an out-of-bounds home yields an invalid span, rejected by RF.
  SSARegisterForest::PhysicalSpan physicalSpan(MCRegister PR) const;
  std::optional<SmallVector<SSARegisterForest::PhysicalSpan, 4>>
  physicalSpans(VRegMaskPair Value, MCRegister PR) const;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SSAREGISTERFORESTADAPTER_H
