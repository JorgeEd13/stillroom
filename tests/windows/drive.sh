#!/usr/bin/env bash
#
# F4.5 — drive start.cmd through one scenario under Wine's cmd.exe.
#
#   ./drive.sh <scenario>      one scenario, full output + the stub log
#   ./drive.sh --all           every scenario, one line each
#
# ⚠️ READ THIS BEFORE QUOTING A RESULT.
#
# Wine's cmd.exe is a REIMPLEMENTATION of Windows' cmd.exe. A pass here means
# "no gross batch error" — the branch was reached, the variables expanded, the
# message read correctly. It does NOT mean "runs on Windows", and nothing this
# harness produces may be described that way. It cannot see Docker Desktop, it
# is indifferent to line endings, and it has already been caught differing from
# Windows once (see findstr.c). Only F5.9, on a real Windows machine, can say
# the launcher works.
#
# What it IS good for: the defects a blind-written .cmd actually has are
# batch-language defects — `call :label` scoping, delayed expansion, `for /f`
# token parsing, variables set inside `if (…)` blocks. Wine's cmd exercises
# exactly those, and it found three real ones the first time it was run.
#
# Prerequisites (neither is in the deliverable; this is a development tool):
#   - wine
#   - a mingw-w64 cross compiler. The stubs MUST be real PE executables: a .cmd
#     stub would be a harness artifact, because batch invoking batch without
#     `call` never returns to the caller.
#       apt-get download gcc-mingw-w64-base gcc-mingw-w64-x86-64-win32 \
#                        binutils-mingw-w64-x86-64 mingw-w64-x86-64-dev \
#                        mingw-w64-common
#     then `dpkg -x` each into a prefix and point $MINGW_ROOT at it, or install
#     mingw-w64 system-wide and leave $MINGW_ROOT unset.
set -u

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)          # the stillroom repo root
work=${STILLROOM_WINTEST_DIR:-/tmp/stillroom-wintest}
export WINEPREFIX=$work/prefix
export WINEDEBUG=-all

CC=${MINGW_CC:-}
if [ -z "$CC" ]; then
  if [ -n "${MINGW_ROOT:-}" ]; then
    CC="$MINGW_ROOT/usr/bin/x86_64-w64-mingw32-gcc-win32 --sysroot=$MINGW_ROOT/usr"
    CC="$CC -B $MINGW_ROOT/usr/lib/gcc/x86_64-w64-mingw32/13-win32/"
    CC="$CC -I $MINGW_ROOT/usr/share/mingw-w64/include"
    CC="$CC -I $MINGW_ROOT/usr/x86_64-w64-mingw32/include"
    CC="$CC -L $MINGW_ROOT/usr/x86_64-w64-mingw32/lib"
    CC="$CC -L $MINGW_ROOT/usr/lib/gcc/x86_64-w64-mingw32/13-win32"
  else
    CC="x86_64-w64-mingw32-gcc"
  fi
fi

# --- one-time setup --------------------------------------------------------
mkdir -p "$work/bin"
[ -d "$WINEPREFIX" ] || wineboot -i >/dev/null 2>&1
if [ ! -f "$work/bin/docker.exe" ] || [ "$here/stub.c" -nt "$work/bin/docker.exe" ]; then
  $CC -O2 -o "$work/bin/docker.exe" "$here/stub.c" || exit 2
  cp "$work/bin/docker.exe" "$work/bin/curl.exe"
  cp "$work/bin/docker.exe" "$work/bin/timeout.exe"
  $CC -O2 -o "$work/bin/findstr.exe" "$here/findstr.c" || exit 2
  # `start "" http://…` on the happy path must not open a real browser.
  printf 'int main(void){return 0;}\n' > "$work/noop.c"
  $CC -O2 -o "$work/bin/noop.exe" "$work/noop.c" || exit 2
  cp "$work/bin/noop.exe" "$WINEPREFIX/drive_c/windows/system32/winebrowser.exe" 2>/dev/null
fi
wpath() { wine winepath -w "$1" 2>/dev/null | tr -d '\r'; }

run_one() {
  name=$1; nodocker=${2:-0}
  run=$work/run/$name
  rm -rf "$run"; mkdir -p "$run"

  cp "$root/start.cmd" "$root/Dockerfile" "$run/" || return 2
  cmp -s "$root/start.cmd" "$run/start.cmd" || { echo "start.cmd copy differs"; return 2; }

  have_docs=1; docs_empty=0; have_tarball=1
  [ -f "$here/scenarios/$name.setup" ] && . "$here/scenarios/$name.setup"
  if [ "$have_docs" = 1 ]; then
    mkdir -p "$run/documents"
    [ "$docs_empty" = 0 ] && printf 'Refunds are available within 14 days.\n' > "$run/documents/policy.md"
  fi
  [ "$have_tarball" = 1 ] && printf 'not-a-real-tarball' > "$run/stillroom-base-0.1.0.tar.gz"
  printf '[client]\nname = "harness"\n' > "$run/client.toml"

  bindir=$work/bin
  if [ "$nodocker" = 1 ]; then           # a machine with no Docker at all
    bindir=$work/bin-nodocker
    mkdir -p "$bindir"; cp "$work"/bin/curl.exe "$work"/bin/timeout.exe "$work"/bin/findstr.exe "$bindir/"
  fi

  : > "$run/stub.log"
  printf '@echo off\nset PATH=%s;%%PATH%%\nset STUB_SCRIPT=%s\nset STUB_LOG=%s\ncall start.cmd\necho [HARNESS] start.cmd exited with %%errorlevel%%\n' \
    "$(wpath "$bindir")" "$(wpath "$here/scenarios/$name.rules")" "$(wpath "$run/stub.log")" \
    > "$run/_harness.cmd"

  ( cd "$run" && yes '' | head -60 | timeout 900 wine cmd /c "_harness.cmd" 2>/dev/null ) \
    | tr -d '\r' | grep -v 'cwd was reset'
}

summarise() {
  printf '%-28s %-14s %s\n' "$1" \
    "$(printf '%s' "$2" | grep -o 'exited with [0-9]*' | tail -1)" \
    "$(printf '%s' "$2" | grep -vE '^$|^-----|Starting your document|Press any key' | head -1)"
}

if [ "${1:-}" = "--all" ]; then
  for f in "$here"/scenarios/[0-9]*.rules; do
    n=$(basename "$f" .rules)
    summarise "$n" "$(run_one "$n")"
  done
  # not a .rules scenario: Docker absent from PATH entirely
  summarise "00-docker-absent" "$(run_one 16-happy-path 1)"
else
  run_one "${1:?usage: drive.sh <scenario>|--all}"
  echo "--- stub log:"
  cat "$work/run/${1}/stub.log"
fi
