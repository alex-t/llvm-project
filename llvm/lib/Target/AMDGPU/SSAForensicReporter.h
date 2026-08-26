//===-- SSAForensicReporter.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Forensic reporting infrastructure for the SSA register allocator.
///
/// This records OBSERVABLE FACTS about an allocation run (which reg units were
/// occupied at a program point, which pick strategy chose a physreg, which
/// values a tier colored or failed to color, which transformations the fallback
/// paths applied) for post-hoc analysis. It is a pure OBSERVER:
///
///   * It NEVER mutates allocator state. It borrows const pointers to the
///     analyses it needs for name/interval lookup (TRI/MRI/LIS) and calls only
///     const methods. It takes no non-const reference to ColorMap /
///     OccupiedRegUnits / UncolorableVRegs.
///   * It records FACTS ONLY. It computes no severity, feasibility judgment, or
///     synthetic score. Where the allocator picks first-fit with no score, the
///     reporter records `strategy=first-fit-order` plus the candidate
///     free/blocked facts rather than inventing a number.
///   * When disabled (the default) every hook early-returns after a single
///     bool test, so the allocator produces byte-identical output ON vs OFF.
///
/// Output sinks (both optional, both gated by the master flag):
///   * NDJSON  (-amdgpu-ssa-forensic-json=<file>): one JSON object per function
///     per line, with a versioned header and the full ordered event stream.
///   * Trace   (-amdgpu-ssa-forensic-trace=<file>): a human-readable event log.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SSAFORENSICREPORTER_H
#define LLVM_LIB_TARGET_AMDGPU_SSAFORENSICREPORTER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llvm {

class MachineFunction;
class MachineRegisterInfo;
class TargetRegisterInfo;
class LiveIntervals;
class TargetRegisterClass;

// Master switch and sink flags. Defined in SSAForensicReporter.cpp; declared
// here so the allocator can read the master gate cheaply on the hot path.
extern cl::opt<bool> EnableSSAForensic;
extern cl::opt<std::string> SSAForensicJSONFile;
extern cl::opt<std::string> SSAForensicTraceFile;
// Colorfail-function scope gate (default TRUE). When true, only functions with
// >= 1 attempt-failed event are flushed; clean functions are dropped entirely.
extern cl::opt<bool> SSAForensicColorfailOnly;
// Reserved (Phase 3 / out of scope for v1) — the flag exists so downstream
// tooling can be wired to it, but nothing reads it yet.
extern cl::opt<bool> EnableSSAForensicCounterfactual;

/// The event taxonomy (E1..E17 from the plan). Rollback is reserved (Q-A): the
/// enum value exists for schema stability but is not emitted in v1.
enum class ForensicEventKind : uint8_t {
  RunStarted = 1,             // E1
  RoundStarted,               // E2
  RoundCompleted,             // E3
  AllocationAttemptStarted,   // E4
  CandidateConsidered,        // E5
  CandidateRejected,          // E6
  CandidateAccepted,          // E7
  Snapshot,                   // E16 (numbered E8 slot; ordered by ID)
  AllocationAttemptCompleted, // E9
  AllocationAttemptFailed,    // E10
  Transformation,             // E11 (split) / E12 (self-split / web)
  SpillEmitted,               // E14
  ReloadEmitted,              // E15
  RunCompleted,               // E17
  Rollback,                   // reserved (Q-A) — unused in v1
  ShadowTreePick,             // E18 — SSARegisterTree shadow-oracle comparison
  RecoveryWindow,             // E19 — recovery-classifier Stage 1 observation
};

/// One value live at a decision-boundary slot — the full liveness cross-section
/// the analyst joins to the objects table by VReg / LR. Facts only: the value's
/// virtual-register id, its live-range identity (printed begin slot), its
/// assigned physreg (or -1 / empty name if still uncolored), width, and lanes.
struct LiveSetEntry {
  unsigned VReg;        // VirtualRegisterID = Register::virtRegIndex()
  std::string LR;       // LiveRangeID (printed interval begin SlotIndex)
  int64_t Phys;         // PhysicalRegisterNodeID (MCRegister::id()), -1=uncolored
  std::string PhysName; // TRI name, empty if uncolored
  unsigned WidthBits;
  uint64_t LaneMask;
};

/// One recorded fact-set. Facts are stored as string or integer key/value
/// pairs; score-like data is ALWAYS a set of named terms (StringMap<int64_t>),
/// never a scalar (§4 of the plan). Causality is expressed by ID references
/// (§5): Causes = earlier events that led here; Consequences filled by link().
struct ForensicEvent {
  uint64_t ID;
  ForensicEventKind Kind;
  // Ordered so serialization is deterministic without a post-sort.
  SmallVector<std::pair<std::string, std::string>, 4> StrFacts;
  SmallVector<std::pair<std::string, int64_t>, 4> IntFacts;
  // Named numeric terms (e.g. score components) — facts, not a synthesized score.
  StringMap<int64_t> Terms;
  // Full liveness cross-section at a decision boundary (E4/E10/E16). Empty for
  // events that are not decision boundaries. Sorted by VReg before serialize.
  SmallVector<LiveSetEntry, 0> LiveSet;
  SmallVector<uint64_t, 2> Causes;
  SmallVector<uint64_t, 2> Consequences;
};

/// Occupancy of one register class at one program point, as observed facts.
/// Mirrors exactly what dumpOccupancyMap prints; produced by the allocator's
/// const collectOccupancy() so the reporter never re-derives allocator state.
struct OccupancyFacts {
  std::string Map;   // one char per reg in order: '#','x','.'
  unsigned FreeUsable = 0;
  unsigned FreeClobbered = 0;
  unsigned Occupied = 0;
  unsigned Total = 0;
  std::string ClassName;
  std::string FirstReg;
  std::string LastReg;
  SmallVector<std::string, 8> Phantom;
  SmallVector<std::string, 8> Usable;
};

class SSAForensicReporter {
public:
  SSAForensicReporter();
  ~SSAForensicReporter();

  /// Master gate. All hooks early-return when this is false, so an off run is
  /// byte-identical to a build without the reporter. The reporter is on when the
  /// master switch is set OR either sink file was named (a sink flag implies the
  /// reporter, consistent with the "-json/-trace enable it" flag docs). cl::opt
  /// values are fixed after command-line parsing, so this is constant for the
  /// whole run; \ref active() caches it to avoid recomputing on the hot path.
  static bool enabled() {
    return EnableSSAForensic || !SSAForensicJSONFile.empty() ||
           !SSAForensicTraceFile.empty();
  }

  /// Cached form of enabled() for the coloring hot path (e.g. pickFreePhysReg's
  /// per-candidate loop): one bool load instead of a flag + two string empties.
  /// Set once in beginRun; identical value to enabled() (loop-invariant), so
  /// substituting it is behavior-neutral.
  bool active() const { return Active; }

  // === Run lifecycle ===

  /// E1. Begin a run for \p MF. Borrows const analysis pointers for name/id
  /// lookup used while serializing (const-only use — no mutation).
  void beginRun(const MachineFunction &MF, const TargetRegisterInfo *TRI,
                const MachineRegisterInfo *MRI, const LiveIntervals *LIS);

  /// E17. Record the run summary, then flush this function's record to the
  /// configured sinks and reset per-function state. \p Uncolorable is the final
  /// count left for the fallback paths.
  void endRun(uint64_t Uncolorable);

  /// Flush this function's buffered record to the sinks EARLY, before the
  /// normal endRun. This exists so a colorfail path that ends in a hard
  /// assert/abort() (which fires before endRun) still gets its forensic report
  /// on disk. It performs the SAME gated flush endRun does (colorfail-only gate
  /// respected) via the shared emitReport() helper, so the on-disk format and
  /// gate semantics are unchanged — it is purely "flush the same report
  /// earlier". IDEMPOTENT: a second flushNow(), or the normal endRun after a
  /// flushNow(), does not double-emit. No-op if called before beginRun or when
  /// disabled. Unlike endRun it does NOT reset per-function state (the caller is
  /// typically about to abort, or endRun runs later and does the reset).
  void flushNow();

  // === Region-RP rounds (E2/E3) ===

  /// E2. Returns the event ID (a cause handle for the round's spills).
  uint64_t roundStarted(unsigned Round, uint64_t UncolorableIn);
  /// E3.
  void roundCompleted(unsigned Round, uint64_t UncolorableOut, bool Spilled,
                      uint64_t StartedCause);

  // === Per-value coloring attempt (E4/E9/E10) ===

  /// E4. Begin an attempt to color \p VReg (width in bits, class name). Returns
  /// the attempt's event ID; pass it back as \p AttemptCause to the candidate
  /// and completion hooks so the stream is causally linked. \p LiveSet is the
  /// full liveness cross-section at the value's def slot (a decision boundary).
  uint64_t attemptStarted(unsigned VRegIdx, unsigned WidthBits,
                          StringRef ClassName, StringRef Strategy,
                          ArrayRef<LiveSetEntry> LiveSet = {});
  /// E9. \p PhysRegID = MCRegister::id() of the accepted reg.
  void attemptCompleted(uint64_t AttemptCause, unsigned VRegIdx,
                        unsigned PhysRegID, StringRef PhysRegName,
                        StringRef Strategy);
  /// E10. No physreg was free across the value's range. \p LiveSet is the full
  /// liveness cross-section at the failure boundary.
  void attemptFailed(uint64_t AttemptCause, unsigned VRegIdx, StringRef Reason,
                     ArrayRef<LiveSetEntry> LiveSet = {});

  // === Candidate facts inside pickFreePhysReg (E5/E6/E7) ===
  //
  // SCHEMA NOTE: candidate-* events are emitted on the first-fit pick path (and
  // the phi-affinity-hint prelude). A pick that returns before that loop runs
  // carries NO candidate trail, so consumers must not assume every
  // attempt-completed has preceding candidate events.

  /// E5. A candidate physreg was considered at a given first-fit ordinal.
  void candidateConsidered(uint64_t AttemptCause, unsigned PhysRegID,
                           StringRef PhysRegName, uint64_t Ordinal,
                           StringRef Strategy);
  /// E6. A candidate was rejected. \p Reason is one of the observable reject
  /// facts: "occupied-unit", "call-modifies", "regmask", "class-mismatch".
  void candidateRejected(uint64_t AttemptCause, unsigned PhysRegID,
                         StringRef PhysRegName, uint64_t Ordinal,
                         StringRef Reason);
  /// E7. A candidate was accepted (the pick).
  void candidateAccepted(uint64_t AttemptCause, unsigned PhysRegID,
                         StringRef PhysRegName, uint64_t Ordinal,
                         StringRef Strategy);

  // === Snapshots (E16) ===

  /// E16. Record an occupancy snapshot (facts produced by collectOccupancy),
  /// plus the full liveness cross-section at the slot.
  void snapshot(StringRef Tag, SlotIndex SI, const OccupancyFacts &F,
                uint64_t Cause = 0, ArrayRef<LiveSetEntry> LiveSet = {});

  // === Transformations (E11/E12) ===

  /// E11/E12. A fallback transformation was applied. \p Kind names the
  /// transform ("split-blocker", "self-split", "phi-web-spill",
  /// "memory-spill"); \p VRegIdx is the value it acted on. Returns the event ID.
  uint64_t transformation(StringRef TransformKind, unsigned VRegIdx,
                          uint64_t Cause = 0);

  /// Record the spill-across analysis facts observed at a coloring failure:
  /// the count of colored values live-through the failed value's range, and the
  /// no-interior-use subset (the ones whose register could be freed across it).
  /// Facts only — no judgment. \p Cause links back to the failed attempt.
  void colorFailAnalysis(unsigned VRegIdx, uint64_t LiveThrough,
                         uint64_t NoInteriorUse, uint64_t Cause = 0);

  // === Spill / reload emission (E14/E15) ===

  /// E14. A store-at-def / spill was emitted for \p VRegIdx.
  void spillEmitted(unsigned VRegIdx, StringRef Site);
  /// E15. A reload was emitted for \p VRegIdx.
  void reloadEmitted(unsigned VRegIdx, StringRef Site);

  // === Shadow register-tree oracle (E18) ===

  /// E18. Record a shadow SSARegisterTree comparison at a real physreg pick.
  /// PURE OBSERVER — the tree's answer is discarded; this only logs the
  /// divergence between what the real allocator chose and what the shadow tree
  /// would have picked. \p RealLeaf is the leaf index of the physreg the
  /// allocator actually chose; \p TreeLeaf is tree.pickFreeAligned(width) (-1 if
  /// the tree found nothing); \p Match is RealLeaf==TreeLeaf. \p FreeCount and
  /// \p FullAtWidthLevel give the tree's aggregate context. \p AttemptCause links
  /// back to the E4 attempt.
  void shadowTreePick(uint64_t AttemptCause, unsigned VRegIdx, unsigned WidthDwords,
                      int64_t RealLeaf, int64_t TreeLeaf, bool Match,
                      unsigned FreeCount, unsigned FullAtWidthLevel);

  /// E18 (skip variant). The shadow tree could not evaluate this pick and was
  /// skipped. \p Reason is one of "class" (non-VGPR_32 file), "leaf-oob"
  /// (physreg outside the mapped VGPR_32 order), or "unmapped".
  void shadowTreeSkip(uint64_t AttemptCause, unsigned VRegIdx, unsigned WidthDwords,
                      StringRef Reason);

  /// E19. Recovery-classifier Stage 1 observation: the recovery window computed
  /// for an uncolored value. Plain fields (not the allocator's RecoveryWindow
  /// struct) to avoid a layering dependency. \p Crossers is the spill-candidate
  /// universe as vreg indices (width derived consumer-side). Endpoints are BLOCK
  /// NUMBERS (not slot ordinals — layout order is not program order). \p WebPhiIdx
  /// is the PHI result reg the value merges into, or 0 if none. Purely
  /// descriptive — records what the classifier saw; drives nothing.
  void recoveryWindow(unsigned UncoloredVRegIdx, int StartBlock, int EndBlock,
                      ArrayRef<unsigned> Crossers, StringRef StopReason,
                      unsigned WebPhiIdx, unsigned UncoloredWidth,
                      unsigned RPOvershoot);

  // === Causality ===

  /// §5. Link cause->consequence bidirectionally. Both IDs must be events in
  /// the current function's stream; no-op if either is 0 (the "no cause"
  /// sentinel) or out of range.
  void link(uint64_t CauseID, uint64_t ConsequenceID);

private:
  // Append a fresh event of \p Kind and return a reference to fill in facts.
  ForensicEvent &newEvent(ForensicEventKind Kind);
  ForensicEvent *findEvent(uint64_t ID);

  void flushJSON();
  void flushTrace();

  /// The single shared flush path used by BOTH endRun() and flushNow(). Applies
  /// the colorfail-only gate, opens the sinks, writes the JSON/trace records,
  /// bumps ReportCtr — but only once per function (guarded by \ref Emitted).
  /// Does NOT reset per-function state; the caller owns that.
  void emitReport();

  // Borrowed const analysis pointers (name/id lookup only; never mutated).
  const MachineFunction *MF = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  const MachineRegisterInfo *MRI = nullptr;
  const LiveIntervals *LIS = nullptr;

  // Cached enabled() (set in beginRun). Loop-invariant for the whole run.
  bool Active = false;

  // Identity counters. EventCtr is the per-run monotonic event id (starts at 1;
  // 0 is the "no cause" sentinel). ReportCtr numbers the functions across the
  // whole module (persists across beginRun/endRun cycles, incrementing 1,2,3...).
  uint64_t EventCtr = 0;
  uint64_t ReportCtr = 0;

  // Per-function record, reset in beginRun.
  std::string FunctionName;
  std::string MFHash; // MD5 over dom-tree-ordered MIR
  std::vector<ForensicEvent> Events;

  // Colorfail-function scope gate. Set true when this function records >= 1
  // attempt-failed event; read at flush time (endRun). When false the whole
  // function record is dropped (no NDJSON / no trace line) — the corpus run
  // cares only about functions that colorfailed, so dumping every function
  // would produce GBs of noise. Reset per function in beginRun. Note this
  // gates ONLY whether the buffered record is flushed; it does NOT change what
  // events are captured, so an on-vs-off run stays byte-identical. For a
  // function that DID fail, the FULL event stream is emitted (including the
  // successful attempts) so the analyst has the passing attempts as contrast.
  bool FailingFunction = false;

  // Idempotence guard for the shared flush path. Set true by emitReport() when
  // it actually writes this function's record; reset per function in beginRun.
  // Ensures a flushNow() (early, pre-abort) followed by the normal endRun does
  // NOT double-emit, and that two flushNow() calls emit at most once.
  bool Emitted = false;

  // Lazily opened output streams (persist across functions within a run).
  std::unique_ptr<raw_fd_ostream> JSONOut;
  std::unique_ptr<raw_fd_ostream> TraceOut;
  bool SinksTried = false;
  void ensureSinks();

  static constexpr unsigned ReportVersion = 1;
  static constexpr unsigned SchemaVersion = 1;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_SSAFORENSICREPORTER_H
