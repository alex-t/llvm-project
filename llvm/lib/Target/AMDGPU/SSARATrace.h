//===-- SSARATrace.h - executed-call-graph trace ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// TEMPORARY instrumentation. Records the caller -> callee edges the allocator
/// actually walks, one line per distinct pair per process, so a corpus run
/// yields the real call graph of the code that runs. The set of callees is the
/// reachability answer; the edges are the workflow.
///
/// Set SSARA_TRACE to a path prefix. Each process appends to <prefix>.<pid>, so
/// parallel harness jobs never share a file and the harness needs no change (it
/// keeps stderr only for FAILING tests). The first line of each file is that
/// process's command line, so a trace identifies its own test and the graph can
/// later be split by outcome.
///
///   export SSARA_TRACE=/tmp/ssara-trace/hits
///   cat hits.* | grep -v '^#' | sort -u > edges.tsv   # the graph
///   cut -f2 edges.tsv | sort -u > executed.txt        # reachability
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_SSARATRACE_H
#define LLVM_LIB_TARGET_AMDGPU_SSARATRACE_H

#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace llvm {
namespace ssara {

inline std::FILE *traceSink() {
  static std::FILE *F = []() -> std::FILE * {
    const char *P = std::getenv("SSARA_TRACE");
    if (!P || !*P)
      return nullptr;
    std::string Path = std::string(P) + "." + std::to_string(::getpid());
    std::FILE *S = std::fopen(Path.c_str(), "a");
    if (!S)
      return nullptr;
    // Self-identify: cmdline is NUL-separated, so join it for one readable line.
    if (std::FILE *C = std::fopen("/proc/self/cmdline", "r")) {
      char Buf[4096];
      size_t N = std::fread(Buf, 1, sizeof(Buf) - 1, C);
      std::fclose(C);
      for (size_t I = 0; I + 1 < N; ++I)
        if (!Buf[I])
          Buf[I] = ' ';
      Buf[N] = '\0';
      std::fprintf(S, "# cmd %s\n", Buf);
      std::fflush(S);
    }
    return S;
  }();
  return F;
}

/// Functions currently entered; back() is the caller of the one being entered.
/// llc is single-threaded on this path, so no locking.
inline std::vector<const char *> &traceStack() {
  static std::vector<const char *> S;
  return S;
}

/// Edges already written. __PRETTY_FUNCTION__ is a per-site constant, so this
/// compares pointers and never hashes a string.
inline bool firstEdge(const char *From, const char *To) {
  static std::set<std::pair<const char *, const char *>> Seen;
  return Seen.emplace(From, To).second;
}

/// Flushes on every new edge because the runs worth studying end in abort(),
/// which would discard buffered output.
struct TraceScope {
  bool Active;
  explicit TraceScope(const char *Name) : Active(traceSink() != nullptr) {
    if (!Active)
      return;
    const char *From = traceStack().empty() ? "(root)" : traceStack().back();
    if (firstEdge(From, Name)) {
      std::fprintf(traceSink(), "%s\t%s\n", From, Name);
      std::fflush(traceSink());
    }
    traceStack().push_back(Name);
  }
  ~TraceScope() {
    if (Active)
      traceStack().pop_back();
  }
};

} // namespace ssara
} // namespace llvm

#define SSARA_TRACE()                                                          \
  ::llvm::ssara::TraceScope SSARATraceScope__(__PRETTY_FUNCTION__)

#endif // LLVM_LIB_TARGET_AMDGPU_SSARATRACE_H
