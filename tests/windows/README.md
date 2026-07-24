# The Windows launcher harness (F4.5)

`start.cmd` is the file most SME clients will double-click, and until 2026-07-22
**it had never been executed by anybody**. It was written blind, three times
over — the preflight, the update flow and the tarball load were all added
without ever running the file.

This harness runs it. `./drive.sh --all` drives every branch under Wine's
`cmd.exe`, with scripted stand-ins for `docker`, `curl` and `timeout`.

## ⚠️ What a pass here means, and what it does not

**Wine's `cmd.exe` is a reimplementation.** A green run means *"no gross batch
error"* — the branch was reached, variables expanded, the message read
correctly, the exit code was right. It does **not** mean *"runs on Windows"*,
and no claim produced from this harness may be worded that way. Only **F5.9**,
on a real Windows machine, can say that.

Three things this harness structurally cannot see:

- **Docker Desktop.** Every `docker` call is a stub. The launcher barely depends
  on Docker's behaviour, which is why this is still worth running — but the
  named-pipe transport, the WSL2 backend and `host.docker.internal` on Windows
  are all invisible here.
- **Line endings.** Wine's `cmd` is indifferent to them; real `cmd.exe` is not,
  particularly around labels and `goto`. `.gitattributes` pins `*.cmd` to CRLF
  for that reason — a change made because it is free, not because this harness
  caught it.
- **Its own divergences.** It has already been caught differing from Windows
  once. See `findstr.c`: Wine's `findstr` treats a bare `^` as a literal caret,
  where real Windows defaults to regex and matches every line. `start.cmd`
  relies on the Windows behaviour twice ("is this folder empty?", "is the
  container running?"), and under Wine's version both answered "no lines" —
  which reads *exactly* like a defect in `start.cmd` and is not one. **That
  near-miss is the argument for reading every surprise here twice** before
  writing it down as a finding.

## Why the stubs are compiled C and not `.cmd` files

Because a batch file that invokes another batch file **without `call` never
returns to the caller**. A `docker.cmd` stub would silently terminate
`start.cmd` at its first `docker info`, and every scenario would "pass" by
ending early. The stubs have to be real PE executables, so the harness needs a
mingw-w64 cross compiler. Verified: Wine returns the stub's real exit code,
honours `>nul 2>&1`, and feeds `for /f`.

Wine's *Unix*-binary escape hatch was tried first and rejected — it returns a
constant exit code of 104 regardless of what the program exits with, and does
not feed `for /f` at all.

## Running it

```sh
./drive.sh --all            # every scenario, one line each
./drive.sh 14-container-crashed   # one scenario, full output + the stub log
```

Prerequisites, neither of which is in the deliverable:

- `wine`
- `mingw-w64`. If it is not installed system-wide, `apt-get download` the five
  packages listed at the top of `drive.sh` (this needs no root), `dpkg -x` them
  into a prefix, and pass `MINGW_ROOT=…`.

## How a scenario works

`scenarios/<name>.rules` is the stub's behaviour, one rule per line:

```
<pattern>|<exit code>|<stdout>
```

The pattern is matched against `"<tool> <args…>"`; a trailing `*` makes it a
prefix match. First match wins, so overrides go above `_base.rules`, which is
concatenated underneath and describes a fully healthy machine.

`scenarios/<name>.setup` shapes the delivery folder (`have_docs`, `docs_empty`,
`have_tarball`).

**An invocation matching no rule is logged `[NO RULE]`.** That is a hole in the
scenario, never something to read past — it means the launcher did something the
scenario did not anticipate.

The per-run `stub.log` is the real evidence: it shows which commands ran, in
what order, with what exit code, and what `BASE_IMAGE` was set to. The printed
messages say what the client sees; the log says what actually happened.

## What it found the first time it was run

Three defects, all live, plus one divergence between the launchers:

1. **The Compose check was unreachable.** It sat *after* both model probes — and
   both probes are `docker compose run`. A client with Docker but no Compose
   plugin was told their Ollama networking was broken. Present in `start.sh`
   too; fixed in both.
2. **`exit /b 1` inside `call :load_base` did not exit.** The corrupt-tarball
   path printed "ask me for it again", paused, and carried on to declare
   **"Ready. Opening http://localhost:8000"** and exit **0**. `start.sh` uses an
   inline `if/else` and never had it. The `call` is gone rather than worked
   around.
3. **The two launchers picked different tarballs** when more than one was
   present — `start.sh` the first, `start.cmd` the last, so the Windows one
   loaded a file the build then did not ask for.
4. **"Docker is not running"** was shown to machines with no Docker installed.

Also cleared, and worth recording as *checked* rather than assumed: a delivery
folder at `…\Ana Paula\Área de Trabalho\assistente da empresa\` — spaces and
accents — works, and all non-ASCII in `start.cmd` is confined to `REM`
comments, so nothing the client reads can arrive as mojibake under the OEM
codepage.
