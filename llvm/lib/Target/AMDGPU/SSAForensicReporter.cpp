//===-- SSAForensicReporter.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Implementation of the SSA register-allocator forensic reporter.
/// See SSAForensicReporter.h. This file owns the cl::opt flags, the per-run
/// event buffer, and the two output sinks (NDJSON + human-readable trace).
///
//===----------------------------------------------------------------------===//

#include "SSAForensicReporter.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// Flags (Q-C: NDJSON; Q-D: cl::opt-gated, default OFF; Q-E: counterfactual
// reserved-but-unused).
//===----------------------------------------------------------------------===//

namespace llvm {
cl::opt<bool> EnableSSAForensic(
    "amdgpu-ssa-forensic", cl::Hidden, cl::init(false),
    cl::desc("Master switch for the SSA register-allocator forensic reporter "
             "(records observable allocation facts; behavior-neutral; default "
             "off). Implied by -amdgpu-ssa-forensic-json / -trace."));

cl::opt<std::string> SSAForensicJSONFile(
    "amdgpu-ssa-forensic-json", cl::Hidden, cl::init(""),
    cl::desc("Write forensic allocation records as NDJSON (one JSON object per "
             "function per line) to this file. Enables the reporter."));

cl::opt<std::string> SSAForensicTraceFile(
    "amdgpu-ssa-forensic-trace", cl::Hidden, cl::init(""),
    cl::desc("Write a human-readable forensic allocation trace to this file. "
             "Enables the reporter."));

cl::opt<bool> SSAForensicColorfailOnly(
    "amdgpu-ssa-forensic-colorfail-only", cl::Hidden, cl::init(true),
    cl::desc("Colorfail-function scope gate (default ON): emit a function's "
             "forensic record ONLY IF that function had >= 1 attempt-failed "
             "event; drop clean functions entirely. For a failing function the "
             "FULL event stream is emitted (successful attempts kept as "
             "contrast). Set =false to dump every function (schema tests)."));

cl::opt<bool> EnableSSAForensicCounterfactual(
    "amdgpu-ssa-forensic-counterfactual", cl::Hidden, cl::init(false),
    cl::desc("Reserved (Phase 3): collect counterfactual/replay data. Unused "
             "in v1; the flag exists for forward compatibility."));
} // namespace llvm

//===----------------------------------------------------------------------===//
// Event-kind names (stable schema strings).
//===----------------------------------------------------------------------===//

static const char *kindName(ForensicEventKind K) {
  switch (K) {
  case ForensicEventKind::RunStarted:
    return "run-started";
  case ForensicEventKind::RoundStarted:
    return "round-started";
  case ForensicEventKind::RoundCompleted:
    return "round-completed";
  case ForensicEventKind::AllocationAttemptStarted:
    return "attempt-started";
  case ForensicEventKind::CandidateConsidered:
    return "candidate-considered";
  case ForensicEventKind::CandidateRejected:
    return "candidate-rejected";
  case ForensicEventKind::CandidateAccepted:
    return "candidate-accepted";
  case ForensicEventKind::Snapshot:
    return "snapshot";
  case ForensicEventKind::AllocationAttemptCompleted:
    return "attempt-completed";
  case ForensicEventKind::AllocationAttemptFailed:
    return "attempt-failed";
  case ForensicEventKind::Transformation:
    return "transformation";
  case ForensicEventKind::SpillEmitted:
    return "spill-emitted";
  case ForensicEventKind::ReloadEmitted:
    return "reload-emitted";
  case ForensicEventKind::RunCompleted:
    return "run-completed";
  case ForensicEventKind::Rollback:
    return "rollback"; // reserved (Q-A) — never emitted in v1
  case ForensicEventKind::ShadowTreePick:
    return "shadow-tree-pick";
  }
  return "unknown";
}

//===----------------------------------------------------------------------===//
// Construction / teardown.
//===----------------------------------------------------------------------===//

SSAForensicReporter::SSAForensicReporter() = default;
SSAForensicReporter::~SSAForensicReporter() = default;

void SSAForensicReporter::ensureSinks() {
  if (SinksTried)
    return;
  SinksTried = true;
  if (!SSAForensicJSONFile.empty()) {
    std::error_code EC;
    auto OS = std::make_unique<raw_fd_ostream>(SSAForensicJSONFile, EC,
                                               sys::fs::OF_Text);
    if (!EC)
      JSONOut = std::move(OS);
    else
      errs() << "SSAForensicReporter: cannot open JSON sink '"
             << SSAForensicJSONFile << "': " << EC.message() << "\n";
  }
  if (!SSAForensicTraceFile.empty()) {
    std::error_code EC;
    auto OS = std::make_unique<raw_fd_ostream>(SSAForensicTraceFile, EC,
                                               sys::fs::OF_Text);
    if (!EC)
      TraceOut = std::move(OS);
    else
      errs() << "SSAForensicReporter: cannot open trace sink '"
             << SSAForensicTraceFile << "': " << EC.message() << "\n";
  }
}

//===----------------------------------------------------------------------===//
// Identity: MD5 over dom-tree-ordered MIR.
//===----------------------------------------------------------------------===//

// Hash the function's instructions in dom-tree (RPO-ish) order so the value is
// stable across runs but sensitive to the actual MIR at run start. This is a
// FACT (an identity fingerprint), not a judgment.
static std::string hashMIR(const MachineFunction &MF) {
  MD5 Hash;
  SmallString<256> Buf;
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      Buf.clear();
      raw_svector_ostream OS(Buf);
      MI.print(OS, /*IsStandalone=*/true, /*SkipOpers=*/false,
               /*SkipDebugLoc=*/true, /*AddNewLine=*/false);
      Hash.update(Buf);
    }
  }
  MD5::MD5Result Result;
  Hash.final(Result);
  return Result.digest().str().str();
}

//===----------------------------------------------------------------------===//
// Event buffer helpers.
//===----------------------------------------------------------------------===//

ForensicEvent &SSAForensicReporter::newEvent(ForensicEventKind Kind) {
  Events.emplace_back();
  ForensicEvent &E = Events.back();
  E.ID = ++EventCtr;
  E.Kind = Kind;
  return E;
}

// Attach a live-set to an event, sorted by VirtualRegisterID for determinism
// (same rule as every other DenseMap-derived collection).
static void attachLiveSet(ForensicEvent &E, ArrayRef<LiveSetEntry> LiveSet) {
  if (LiveSet.empty())
    return;
  E.LiveSet.assign(LiveSet.begin(), LiveSet.end());
  llvm::sort(E.LiveSet, [](const LiveSetEntry &A, const LiveSetEntry &B) {
    return A.VReg < B.VReg;
  });
}

ForensicEvent *SSAForensicReporter::findEvent(uint64_t ID) {
  if (ID == 0)
    return nullptr;
  // Events are appended in ID order (ID == index+first), so the lookup is O(1)
  // when the id is in range; guard for out-of-run references.
  if (ID >= 1 && ID <= Events.size() && Events[ID - 1].ID == ID)
    return &Events[ID - 1];
  return nullptr;
}

void SSAForensicReporter::link(uint64_t CauseID, uint64_t ConsequenceID) {
  if (!enabled())
    return;
  ForensicEvent *C = findEvent(CauseID);
  ForensicEvent *Q = findEvent(ConsequenceID);
  if (!C || !Q)
    return;
  C->Consequences.push_back(ConsequenceID);
  Q->Causes.push_back(CauseID);
}

//===----------------------------------------------------------------------===//
// Run lifecycle.
//===----------------------------------------------------------------------===//

void SSAForensicReporter::beginRun(const MachineFunction &MFn,
                                   const TargetRegisterInfo *TRIn,
                                   const MachineRegisterInfo *MRIn,
                                   const LiveIntervals *LISn) {
  Active = enabled();
  if (!Active)
    return;
  MF = &MFn;
  TRI = TRIn;
  MRI = MRIn;
  LIS = LISn;
  FunctionName = MFn.getName().str();
  MFHash = hashMIR(MFn);
  Events.clear();
  EventCtr = 0;
  FailingFunction = false;
  Emitted = false;
  // ReportCtr is NOT bumped here: reportID numbers only EMITTED (failing)
  // functions, so the NDJSON is densely numbered 1,2,3.. over the functions
  // that colorfailed. It is incremented at flush time in endRun, gated on
  // FailingFunction. mfHash remains per-function regardless.

  ForensicEvent &E = newEvent(ForensicEventKind::RunStarted);
  E.StrFacts.push_back({"function", FunctionName});
  E.StrFacts.push_back({"mfHash", MFHash});
  E.IntFacts.push_back({"numVirtRegs", (int64_t)MRIn->getNumVirtRegs()});
}

// The single shared flush path used by BOTH endRun() and flushNow(). Idempotent
// per function via Emitted, so a flushNow() (early, pre-abort) followed by the
// normal endRun does not double-emit. Applies the colorfail-only gate; does NOT
// reset per-function state (the caller owns that).
void SSAForensicReporter::emitReport() {
  if (Emitted)
    return;
  // Nothing buffered means beginRun never ran (or already reset) — nothing to
  // emit. Guards flushNow() called before any beginRun.
  if (Events.empty())
    return;

  // Colorfail-function scope gate: when SSAForensicColorfailOnly (default ON),
  // emit this function's record ONLY IF it had >= 1 attempt-failed event. Clean
  // functions are dropped entirely (no NDJSON line, no trace line) to keep the
  // corpus-wide output to the colorfail set. The full event stream (including
  // the successful attempts of a failing function) is preserved as contrast.
  // reportID advances only for emitted functions so the NDJSON is densely
  // numbered 1,2,3.. over the emitted functions.
  if (!SSAForensicColorfailOnly || FailingFunction) {
    ++ReportCtr;
    ensureSinks();
    if (JSONOut)
      flushJSON();
    if (TraceOut)
      flushTrace();
  }
  // Mark emitted even when the gate dropped this (clean) function: a later
  // flushNow()/endRun must not re-run the gate and emit it then.
  Emitted = true;
}

void SSAForensicReporter::endRun(uint64_t Uncolorable) {
  if (!enabled())
    return;
  // If a flushNow() already emitted this function's record early (pre-abort
  // path that then recovered), don't append a second RunCompleted or re-emit.
  // Still reset per-function state so the next function starts clean.
  if (!Emitted) {
    ForensicEvent &E = newEvent(ForensicEventKind::RunCompleted);
    E.IntFacts.push_back({"uncolorableRemaining", (int64_t)Uncolorable});
    E.IntFacts.push_back({"totalEvents", (int64_t)Events.size()});
    emitReport();
  }

  Events.clear();
  MF = nullptr;
  TRI = nullptr;
  MRI = nullptr;
  LIS = nullptr;
}

// Flush the current function's buffered record EARLY (before endRun), so a
// colorfail path that ends in a hard assert/abort still leaves its report on
// disk. Idempotent and format-identical to endRun's flush (see emitReport).
void SSAForensicReporter::flushNow() {
  if (!enabled())
    return;
  emitReport();
}

//===----------------------------------------------------------------------===//
// Region-RP rounds.
//===----------------------------------------------------------------------===//

uint64_t SSAForensicReporter::roundStarted(unsigned Round,
                                           uint64_t UncolorableIn) {
  if (!enabled())
    return 0;
  ForensicEvent &E = newEvent(ForensicEventKind::RoundStarted);
  E.IntFacts.push_back({"round", (int64_t)Round});
  E.IntFacts.push_back({"uncolorableIn", (int64_t)UncolorableIn});
  return E.ID;
}

void SSAForensicReporter::roundCompleted(unsigned Round, uint64_t UncolorableOut,
                                         bool Spilled, uint64_t StartedCause) {
  if (!enabled())
    return;
  ForensicEvent &E = newEvent(ForensicEventKind::RoundCompleted);
  E.IntFacts.push_back({"round", (int64_t)Round});
  E.IntFacts.push_back({"uncolorableOut", (int64_t)UncolorableOut});
  E.StrFacts.push_back({"spilled", Spilled ? "true" : "false"});
  uint64_t ID = E.ID;
  link(StartedCause, ID);
}

//===----------------------------------------------------------------------===//
// Per-value coloring attempt.
//===----------------------------------------------------------------------===//

uint64_t SSAForensicReporter::attemptStarted(unsigned VRegIdx,
                                             unsigned WidthBits,
                                             StringRef ClassName,
                                             StringRef Strategy,
                                             ArrayRef<LiveSetEntry> LiveSet) {
  if (!enabled())
    return 0;
  ForensicEvent &E = newEvent(ForensicEventKind::AllocationAttemptStarted);
  E.IntFacts.push_back({"vreg", (int64_t)VRegIdx});
  E.IntFacts.push_back({"widthBits", (int64_t)WidthBits});
  E.StrFacts.push_back({"class", ClassName.str()});
  E.StrFacts.push_back({"strategy", Strategy.str()});
  attachLiveSet(E, LiveSet);
  return E.ID;
}

void SSAForensicReporter::attemptCompleted(uint64_t AttemptCause,
                                           unsigned VRegIdx, unsigned PhysRegID,
                                           StringRef PhysRegName,
                                           StringRef Strategy) {
  if (!enabled())
    return;
  ForensicEvent &E = newEvent(ForensicEventKind::AllocationAttemptCompleted);
  E.IntFacts.push_back({"vreg", (int64_t)VRegIdx});
  E.IntFacts.push_back({"physReg", (int64_t)PhysRegID});
  E.StrFacts.push_back({"physRegName", PhysRegName.str()});
  E.StrFacts.push_back({"strategy", Strategy.str()});
  uint64_t ID = E.ID;
  link(AttemptCause, ID);
}

void SSAForensicReporter::attemptFailed(uint64_t AttemptCause, unsigned VRegIdx,
                                        StringRef Reason,
                                        ArrayRef<LiveSetEntry> LiveSet) {
  if (!enabled())
    return;
  ForensicEvent &E = newEvent(ForensicEventKind::AllocationAttemptFailed);
  E.IntFacts.push_back({"vreg", (int64_t)VRegIdx});
  E.StrFacts.push_back({"reason", Reason.str()});
  attachLiveSet(E, LiveSet);
  // Colorfail-function scope gate: this function had a coloring failure, so its
  // whole record will be flushed at endRun (clean functions are dropped).
  FailingFunction = true;
  uint64_t ID = E.ID;
  link(AttemptCause, ID);
}

//===----------------------------------------------------------------------===//
// Candidate facts (E5/E6/E7).
//===----------------------------------------------------------------------===//

void SSAForensicReporter::candidateConsidered(uint64_t AttemptCause,
                                              unsigned PhysRegID,
                                              StringRef PhysRegName,
                                              uint64_t Ordinal,
                                              StringRef Strategy) {
  if (!enabled())
    return;
  ForensicEvent &E = newEvent(ForensicEventKind::CandidateConsidered);
  E.IntFacts.push_back({"physReg", (int64_t)PhysRegID});
  E.StrFacts.push_back({"physRegName", PhysRegName.str()});
  E.IntFacts.push_back({"ordinal", (int64_t)Ordinal});
  E.StrFacts.push_back({"strategy", Strategy.str()});
  uint64_t ID = E.ID;
  link(AttemptCause, ID);
}

void SSAForensicReporter::candidateRejected(uint64_t AttemptCause,
                                            unsigned PhysRegID,
                                            StringRef PhysRegName,
                                            uint64_t Ordinal,
                                            StringRef Reason) {
  if (!enabled())
    return;
  ForensicEvent &E = newEvent(ForensicEventKind::CandidateRejected);
  E.IntFacts.push_back({"physReg", (int64_t)PhysRegID});
  E.StrFacts.push_back({"physRegName", PhysRegName.str()});
  E.IntFacts.push_back({"ordinal", (int64_t)Ordinal});
  E.StrFacts.push_back({"reason", Reason.str()});
  uint64_t ID = E.ID;
  link(AttemptCause, ID);
}

void SSAForensicReporter::candidateAccepted(uint64_t AttemptCause,
                                            unsigned PhysRegID,
                                            StringRef PhysRegName,
                                            uint64_t Ordinal,
                                            StringRef Strategy) {
  if (!enabled())
    return;
  ForensicEvent &E = newEvent(ForensicEventKind::CandidateAccepted);
  E.IntFacts.push_back({"physReg", (int64_t)PhysRegID});
  E.StrFacts.push_back({"physRegName", PhysRegName.str()});
  E.IntFacts.push_back({"ordinal", (int64_t)Ordinal});
  E.StrFacts.push_back({"strategy", Strategy.str()});
  uint64_t ID = E.ID;
  link(AttemptCause, ID);
}

//===----------------------------------------------------------------------===//
// Snapshots (E16).
//===----------------------------------------------------------------------===//

void SSAForensicReporter::snapshot(StringRef Tag, SlotIndex SI,
                                   const OccupancyFacts &F, uint64_t Cause,
                                   ArrayRef<LiveSetEntry> LiveSet) {
  if (!enabled())
    return;
  ForensicEvent &E = newEvent(ForensicEventKind::Snapshot);
  E.StrFacts.push_back({"tag", Tag.str()});
  {
    std::string S;
    raw_string_ostream OS(S);
    OS << SI;
    E.StrFacts.push_back({"slot", OS.str()});
  }
  E.StrFacts.push_back({"class", F.ClassName});
  E.StrFacts.push_back({"map", F.Map});
  E.IntFacts.push_back({"freeUsable", (int64_t)F.FreeUsable});
  E.IntFacts.push_back({"freeClobbered", (int64_t)F.FreeClobbered});
  E.IntFacts.push_back({"occupied", (int64_t)F.Occupied});
  E.IntFacts.push_back({"total", (int64_t)F.Total});
  if (!F.FirstReg.empty())
    E.StrFacts.push_back({"firstReg", F.FirstReg});
  if (!F.LastReg.empty())
    E.StrFacts.push_back({"lastReg", F.LastReg});
  {
    std::string S;
    for (const std::string &R : F.Phantom) {
      if (!S.empty())
        S += " ";
      S += R;
    }
    if (!S.empty())
      E.StrFacts.push_back({"phantom", S});
  }
  {
    std::string S;
    for (const std::string &R : F.Usable) {
      if (!S.empty())
        S += " ";
      S += R;
    }
    if (!S.empty())
      E.StrFacts.push_back({"usable", S});
  }
  attachLiveSet(E, LiveSet);
  uint64_t ID = E.ID;
  link(Cause, ID);
}

//===----------------------------------------------------------------------===//
// Transformations (E11/E12).
//===----------------------------------------------------------------------===//

uint64_t SSAForensicReporter::transformation(StringRef TransformKind,
                                             unsigned VRegIdx, uint64_t Cause) {
  if (!enabled())
    return 0;
  ForensicEvent &E = newEvent(ForensicEventKind::Transformation);
  E.StrFacts.push_back({"transform", TransformKind.str()});
  E.IntFacts.push_back({"vreg", (int64_t)VRegIdx});
  uint64_t ID = E.ID;
  link(Cause, ID);
  return ID;
}

void SSAForensicReporter::colorFailAnalysis(unsigned VRegIdx,
                                            uint64_t LiveThrough,
                                            uint64_t NoInteriorUse,
                                            uint64_t Cause) {
  if (!enabled())
    return;
  ForensicEvent &E = newEvent(ForensicEventKind::Snapshot);
  E.StrFacts.push_back({"tag", "colorfail-spill-across"});
  E.IntFacts.push_back({"vreg", (int64_t)VRegIdx});
  E.IntFacts.push_back({"liveThrough", (int64_t)LiveThrough});
  E.IntFacts.push_back({"noInteriorUse", (int64_t)NoInteriorUse});
  uint64_t ID = E.ID;
  link(Cause, ID);
}

//===----------------------------------------------------------------------===//
// Spill / reload emission (E14/E15).
//===----------------------------------------------------------------------===//

void SSAForensicReporter::spillEmitted(unsigned VRegIdx, StringRef Site) {
  if (!enabled())
    return;
  ForensicEvent &E = newEvent(ForensicEventKind::SpillEmitted);
  E.IntFacts.push_back({"vreg", (int64_t)VRegIdx});
  E.StrFacts.push_back({"site", Site.str()});
}

void SSAForensicReporter::reloadEmitted(unsigned VRegIdx, StringRef Site) {
  if (!enabled())
    return;
  ForensicEvent &E = newEvent(ForensicEventKind::ReloadEmitted);
  E.IntFacts.push_back({"vreg", (int64_t)VRegIdx});
  E.StrFacts.push_back({"site", Site.str()});
}

//===----------------------------------------------------------------------===//
// Shadow register-tree oracle (E18).
//===----------------------------------------------------------------------===//

void SSAForensicReporter::shadowTreePick(uint64_t AttemptCause, unsigned VRegIdx,
                                         unsigned WidthDwords, int64_t RealLeaf,
                                         int64_t TreeLeaf, bool Match,
                                         unsigned FreeCount,
                                         unsigned FullAtWidthLevel) {
  if (!enabled())
    return;
  ForensicEvent &E = newEvent(ForensicEventKind::ShadowTreePick);
  E.IntFacts.push_back({"vreg", (int64_t)VRegIdx});
  E.IntFacts.push_back({"widthDwords", (int64_t)WidthDwords});
  E.IntFacts.push_back({"realLeaf", RealLeaf});
  E.IntFacts.push_back({"treeLeaf", TreeLeaf});
  E.StrFacts.push_back({"match", Match ? "true" : "false"});
  E.IntFacts.push_back({"freeCount", (int64_t)FreeCount});
  E.IntFacts.push_back({"fullAtWidthLevel", (int64_t)FullAtWidthLevel});
  uint64_t ID = E.ID;
  link(AttemptCause, ID);
}

void SSAForensicReporter::shadowTreeSkip(uint64_t AttemptCause, unsigned VRegIdx,
                                         unsigned WidthDwords, StringRef Reason) {
  if (!enabled())
    return;
  ForensicEvent &E = newEvent(ForensicEventKind::ShadowTreePick);
  E.IntFacts.push_back({"vreg", (int64_t)VRegIdx});
  E.IntFacts.push_back({"widthDwords", (int64_t)WidthDwords});
  E.StrFacts.push_back({"skipped", Reason.str()});
  uint64_t ID = E.ID;
  link(AttemptCause, ID);
}

//===----------------------------------------------------------------------===//
// Serialization.
//===----------------------------------------------------------------------===//

// One JSON object per function per line (NDJSON, Q-C). Header carries the
// versioning triple (reportVersion / schemaVersion / allocatorVersion) plus the
// function identity, followed by the ordered event stream.
void SSAForensicReporter::flushJSON() {
  json::OStream J(*JSONOut, /*IndentSize=*/0);
  J.object([&] {
    J.attribute("reportVersion", (int64_t)ReportVersion);
    J.attribute("schemaVersion", (int64_t)SchemaVersion);
    J.attribute("allocatorVersion", "amdgpu-ssa-ra-v1");
    J.attribute("reportID", (int64_t)ReportCtr);
    J.attribute("function", FunctionName);
    J.attribute("mfHash", MFHash);
    J.attributeArray("events", [&] {
      for (const ForensicEvent &E : Events) {
        J.object([&] {
          J.attribute("id", (int64_t)E.ID);
          J.attribute("kind", kindName(E.Kind));
          for (const auto &KV : E.StrFacts)
            J.attribute(KV.first, KV.second);
          for (const auto &KV : E.IntFacts)
            J.attribute(KV.first, KV.second);
          if (!E.Terms.empty()) {
            // Score components serialized as named terms (§4) — sorted by key
            // for determinism. Never a scalar.
            SmallVector<StringRef, 8> Keys;
            for (const auto &T : E.Terms)
              Keys.push_back(T.first());
            llvm::sort(Keys);
            J.attributeObject("terms", [&] {
              for (StringRef K : Keys)
                J.attribute(K, (int64_t)E.Terms.lookup(K));
            });
          }
          if (!E.LiveSet.empty()) {
            // Full liveness cross-section at this decision boundary (schema v1
            // first-class field). Sorted by vreg at capture time.
            J.attributeArray("liveSet", [&] {
              for (const LiveSetEntry &L : E.LiveSet)
                J.object([&] {
                  J.attribute("vreg", (int64_t)L.VReg);
                  J.attribute("lr", L.LR);
                  if (L.Phys < 0)
                    J.attribute("phys", nullptr);
                  else
                    J.attribute("phys", L.Phys);
                  if (!L.PhysName.empty())
                    J.attribute("physName", L.PhysName);
                  J.attribute("width", (int64_t)L.WidthBits);
                  {
                    std::string H;
                    raw_string_ostream OS(H);
                    OS << format_hex(L.LaneMask, 0);
                    J.attribute("laneMask", OS.str());
                  }
                });
            });
          }
          if (!E.Causes.empty())
            J.attributeArray("causes", [&] {
              for (uint64_t C : E.Causes)
                J.value((int64_t)C);
            });
          if (!E.Consequences.empty())
            J.attributeArray("consequences", [&] {
              for (uint64_t C : E.Consequences)
                J.value((int64_t)C);
            });
        });
      }
    });
  });
  *JSONOut << "\n"; // NDJSON: newline terminates the function record
  JSONOut->flush();
}

void SSAForensicReporter::flushTrace() {
  raw_fd_ostream &OS = *TraceOut;
  OS << "=== forensic report #" << ReportCtr << " function " << FunctionName
     << " (mfHash " << MFHash << ", schema v" << SchemaVersion << ") ===\n";
  for (const ForensicEvent &E : Events) {
    OS << "  [" << E.ID << "] " << kindName(E.Kind);
    for (const auto &KV : E.StrFacts)
      OS << " " << KV.first << "=" << KV.second;
    for (const auto &KV : E.IntFacts)
      OS << " " << KV.first << "=" << KV.second;
    if (!E.Terms.empty()) {
      SmallVector<StringRef, 8> Keys;
      for (const auto &T : E.Terms)
        Keys.push_back(T.first());
      llvm::sort(Keys);
      OS << " terms{";
      bool First = true;
      for (StringRef K : Keys) {
        if (!First)
          OS << ",";
        First = false;
        OS << K << "=" << E.Terms.lookup(K);
      }
      OS << "}";
    }
    if (!E.LiveSet.empty()) {
      OS << " liveSet[" << E.LiveSet.size() << "]{";
      bool First = true;
      for (const LiveSetEntry &L : E.LiveSet) {
        if (!First)
          OS << ",";
        First = false;
        OS << "%" << L.VReg << "->";
        if (L.Phys < 0)
          OS << "(uncolored)";
        else
          OS << L.PhysName;
      }
      OS << "}";
    }
    if (!E.Causes.empty()) {
      OS << " <-";
      for (uint64_t C : E.Causes)
        OS << " " << C;
    }
    OS << "\n";
  }
  OS.flush();
}
