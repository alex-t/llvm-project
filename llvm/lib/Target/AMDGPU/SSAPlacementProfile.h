//===- SSAPlacementProfile.h - Recovery placement facts -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Canonical placement and interference facts used by AMDGPU SSA register
/// allocation recovery.  The profile contains no recovery policy or mutation
/// plan; its queries are derived views of the recorded facts.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SSAPLACEMENTPROFILE_H
#define LLVM_LIB_TARGET_AMDGPU_SSAPLACEMENTPROFILE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/MC/MCRegisterInfo.h"

namespace llvm {

class TargetRegisterClass;

struct PlacementProfile {
  using HomeID = unsigned;

  struct SlotRange {
    SlotIndex Start;
    SlotIndex End;

    bool operator==(const SlotRange &Other) const {
      return Start == Other.Start && End == Other.End;
    }
  };

  struct BlockerSpan {
    Register Blocker;
    SlotRange Range;

    // HomeID indexes Homes. Owner identity is payload, never a search or
    // preference key.
    SmallBitVector BlockedHomes;
  };

  struct HomeCut {
    SlotIndex At;

    // Homes that cannot carry one resident piece across At.
    SmallBitVector CutHomes;
  };

  Register Subject;
  const TargetRegisterClass *RC = nullptr;
  SlotRange Region;

  // Exact allocator order, including its current target-order priority.
  SmallVector<MCRegister, 32> Homes;

  // Factorized interference: one blocker fact can affect several homes.
  SmallVector<BlockerSpan, 32> Blockers;

  // Ownerless temporal legality cuts.
  SmallVector<HomeCut, 8> Cuts;
};

struct PlacementProfileSlice {
  PlacementProfile::HomeID Home;
  PlacementProfile::SlotRange Range;
  SmallVector<Register, 4> Blockers;

  // A cut at Range.End prevents this slice from being joined to the next one,
  // even when both have the same blocker set.
  bool CutAfter = false;

  bool operator==(const PlacementProfileSlice &Other) const {
    return Home == Other.Home && Range == Other.Range &&
           Blockers == Other.Blockers && CutAfter == Other.CutAfter;
  }
};

struct NormalizedPlacementProfile {
  Register Subject;
  const TargetRegisterClass *RC = nullptr;
  PlacementProfile::SlotRange Region;
  SmallVector<MCRegister, 32> Homes;
  SmallVector<PlacementProfileSlice, 32> Slices;

  bool operator==(const NormalizedPlacementProfile &Other) const {
    return Subject == Other.Subject && RC == Other.RC &&
           Region == Other.Region && Homes == Other.Homes &&
           Slices == Other.Slices;
  }
};

struct PlacementFreeRun {
  PlacementProfile::HomeID Home;
  MCRegister PhysReg;
  PlacementProfile::SlotRange Range;
};

/// Return a decomposition-independent view: target-ordered homes and maximal
/// temporal slices. Blockers are ordered by register ID only to canonicalize
/// identity; that order carries no recovery preference.
NormalizedPlacementProfile
normalizePlacementProfile(const PlacementProfile &Profile);

/// Return every globally free temporal run, grouped in target home order.
/// Applicable cuts keep otherwise adjacent free runs separate.
void getFreeRuns(const PlacementProfile &Profile,
                 SmallVectorImpl<PlacementFreeRun> &Out);

/// Collect every distinct blocker intersecting Range on any legal home.
void getBlockers(const PlacementProfile &Profile,
                 PlacementProfile::SlotRange Range,
                 SmallVectorImpl<Register> &Out);

/// Visit the same maximal slices used by normalization.
void visitInterferenceSlices(
    const PlacementProfile &Profile,
    function_ref<void(PlacementProfile::HomeID,
                      PlacementProfile::SlotRange,
                      ArrayRef<Register>)> Visit);

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SSAPLACEMENTPROFILE_H
