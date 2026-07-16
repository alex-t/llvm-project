# Cursor Safety Net & Command Allowlist — Agent Runbook

A reproducible summary of the human↔AI command-safety setup, for another AI agent
to understand and replicate. Verified on Cursor **3.2.16** (Windows client + Linux
remote worktree).

## Model: two layers

**Layer 1 — UI Command Allowlist (convenience: what auto-runs)**
- Settings → Cursor Settings → Agents → Approvals & Execution → Auto-Run Mode = **Use Allowlist**.
- Stored **client-side** (Windows `AppData\Roaming\Cursor`) — NOT editable from the Linux box; entries are added via the UI.
- Matching is **command-string PREFIX** based:
  - `git log` also matches `git log --oneline`.
  - `ninja -C build/user-debug` matches any target in that dir but NOT `ninja` elsewhere → builds scoped to the correct folder.
  - Never allowlist bare `git` (would allow `git push`/`commit`); use per-subcommand entries.
- **Governs background subagents too**: a non-allowlisted command inside a subagent surfaces a View/Allow prompt in the parent UI and BLOCKS that subagent until clicked (not a silent hang, but it stalls).
- Populated with a read-only set + scoped build (file/inspection tools, read-only `git` subcommands, `ninja -C build/user-debug`).

**Layer 2 — `beforeShellExecution` hook (safety: hard blocks)**
- Files: `.cursor/hooks.json` → `.cursor/hooks/shell-guard.sh`; audit log at `.cursor/hooks/state/shell-guard.log`. `failClosed:false`.
- **Verified precedence facts (measured, not assumed):**
  - Hook **`allow` is IGNORED for subagents** — they fall back to the UI allowlist. (So never rely on hook `allow`.)
  - Hook **`deny` is HONORED for BOTH parent and subagents** — a hard block that wins over the allowlist everywhere.
  - Empty stdout + `exit 0` = **abstain** (does NOT default to allow; the allowlist stays authoritative).
- Correct usage: **deny-only, fail-safe** — hard-deny known destructive patterns, abstain otherwise.

## Key constraints / gotchas

- **Prefix matching cannot exclude a mutating suffix.** Allowlisting any `find …`/`sed …`
  prefix also admits `find … -delete`, `sed -i`, etc. So destructive carve-outs must be
  handled by the **deny hook**, not the allowlist.
- **Do NOT build an allow-based "sandbox by path" hook.** Parsing shell in a regex to
  auto-`allow` "safe" mutations fails OPEN (absolute-path binaries `/bin/rm`, interpreters
  pointed at `/tmp`, `..` traversal, uncaptured paths, command substitution all bypass it).
  A deny-based hook fails SAFE (a missed pattern → abstain → prompt).
- **External-File Protection** is a separate Cursor prompt for editing files *outside* the
  workspace root (e.g. a sibling docs worktree). It is not governed by the allowlist or hook.
- Rules are re-scanned per message (reach open chats next turn); **skills** are discovered at
  startup (only in NEW chats).

## Reproduce elsewhere

1. Auto-Run Mode = **Use Allowlist**; add the read-only set + `ninja -C <build-dir>`
   (per-subcommand git; never bare `git`).
2. Add a project `beforeShellExecution` hook that **audit-logs** and **denies** destructive
   sequences of any tool you allowlist; abstain otherwise. Deny is the only hook decision that
   works everywhere.
3. To safely allowlist `find`/`sed`: allowlist them for convenience, and have the deny hook
   hard-block `find … -delete/-exec/-execdir/-ok/-okdir/-fprintf/-fls/-fprint` and
   `sed -i/--in-place` (match the tool as a word or via absolute path).

## Reference: deny-based `shell-guard.sh`

```bash
#!/bin/bash
# beforeShellExecution guard: audit-log; hard-DENY destructive carve-outs of
# allowlisted tools (find/sed); abstain otherwise. Deny is honored for parent AND
# subagents; fails safe. failClosed=false in hooks.json.
set -uo pipefail
input="$(cat)"
LOG=".cursor/hooks/state/shell-guard.log"
mkdir -p "$(dirname "$LOG")"
printf '%s\t%s\n' "$(date -Is)" "$input" >> "$LOG"

cmd="$(printf '%s' "$input" | jq -r '.command // empty' 2>/dev/null)"
[ -z "$cmd" ] && exit 0   # abstain

deny() {
  printf '{"permission":"deny","user_message":"Blocked by shell-guard: %s.","agent_message":"shell-guard hard-blocked a destructive pattern (%s). Do NOT retry or work around it; ask the user to run it manually if intended."}\n' "$1" "$1"
  exit 0
}

# find invoked as a word or via absolute path, WITH a destructive action.
if printf '%s' "$cmd" | grep -Eq '(^|[][[:space:];&|(`$])/?([^[:space:];&|]*/)?find([[:space:]]|$)' \
&& printf '%s' "$cmd" | grep -Eq '[[:space:]]-(delete|exec|execdir|ok|okdir|fprintf|fprint|fprint0|fls)([[:space:]]|$)'; then
  deny "find with -delete/-exec/-fprintf/..."
fi

# sed in-place edit (-i / --in-place, incl. bundled short flags like -ni and -i.bak).
if printf '%s' "$cmd" | grep -Eq '(^|[][[:space:];&|(`$])/?([^[:space:];&|]*/)?sed([[:space:]]|$)' \
&& printf '%s' "$cmd" | grep -Eq '([[:space:]]-[a-zA-Z]*i([[:space:]]|=|\.|$)|--in-place)'; then
  deny "sed -i/--in-place"
fi

exit 0   # abstain
```

## Status note

- Applied on this machine: the **audit-only** variant of `shell-guard.sh` (log + abstain).
- The deny-based variant above is the recommended target; apply it together with allowlisting
  `find`/`sed` if you want those auto-running without their destructive forms.
