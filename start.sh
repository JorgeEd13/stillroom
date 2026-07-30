#!/usr/bin/env bash
#
# Start the assistant. Double-click it, or run ./start.sh from a terminal.
#
# ⚠️ This file is written for the CLIENT, not for me. Every message it prints is
# read by somebody who did not choose to be looking at a terminal, and who will
# forward whatever appears here to me if it goes wrong. So it explains rather
# than reports: no stack traces, no exit codes, no jargon — and it never tries
# to fix anything on their machine (ADR-003, verify never install).

set -u

cd "$(dirname "$0")" || exit 1

say() { printf '%s\n' "$1"; }
hr() { say "------------------------------------------------------------"; }

hr
say "  Starting your document assistant"
hr
say ""

# --- 1. Docker ------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  say "Docker is not installed on this computer."
  say ""
  say "The assistant runs inside Docker. Install Docker Desktop from Docker's"
  say "own website, start it, then run this file again."
  say ""
  read -r -p "Press Enter to close." _ || true
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  say "Docker is installed but not running."
  say ""
  say "Start Docker Desktop, wait until it says it is running, then run this"
  say "file again."
  say ""
  read -r -p "Press Enter to close." _ || true
  exit 1
fi

# --- 1b. Compose, BEFORE anything that uses Compose -----------------------
# This has to precede the probes below, because they ARE `docker compose run`.
# See the note at 2b for what it cost while it sat further down.
if ! docker compose version >/dev/null 2>&1; then
  say "This computer has Docker, but not Docker Compose."
  say ""
  say "Compose is included with Docker Desktop. If Docker was installed some"
  say "other way, install the Compose plugin, then run this file again."
  say ""
  read -r -p "Press Enter to close." _ || true
  exit 1
fi

# --- 2. Ollama ------------------------------------------------------------
# ⚠️ **Checked from inside a container, and the previous version's "from here"
# was the defect** (leg B #12).
#
# This script used to run `curl http://localhost:11434/api/tags` on the HOST,
# where Ollama is reachable — and then hand off to a container, where it is not.
# On Linux, `host.docker.internal` is the docker0 bridge and Ollama binds
# `127.0.0.1` by default, so the host check passes, the launcher says
# "Starting…", and the assistant can never answer a single live question. That
# is ADR-039's "reasoning about the wrong computer", in the launcher.
#
# The container this probe runs in is the same kind the assistant runs in, so a
# pass here means the assistant can genuinely reach the model. No Python, no
# image build — busybox is ~4 MB and the client already has Docker.
#
# ⚠️ **Ask Compose where the model actually is; do not assume.** An earlier
# version hardcoded `host.docker.internal`, which is right for the ordinary
# deployment and wrong for every other one — the on-premises engagement points
# at a GPU box elsewhere on the client's network (ADR-023), and a rehearsal
# points at a container. Hardcoding made the launcher refuse to start a build
# that was perfectly healthy.
# ⛔ **AND THE PROBE ITSELF WAS ON THE WRONG NETWORK** (ADR-060, and it is the
# same defect one level deeper).
#
# The previous version ran `docker run --rm busybox …` with no network, which
# lands on the **default** bridge. Compose builds its own network for the
# project, with a different subnet and a different bridge interface on the host
# — so a firewall, a route or a policy can allow one and refuse the other.
# Measured here, on exactly such a host:
#
#     from the default bridge (what this checked)      -> HTTP 200
#     from the compose network (what the assistant uses) -> HTTP 000
#
# The preflight passed, the launcher said "Ready", and every live question hung.
# Running the probe as a **compose service** is what makes "the same network"
# structural: Compose puts it on the project network with the same address and
# the same `extra_hosts` as the assistant, because they share one anchor.
#
# It also removes the old carve-out. The previous probe could not resolve a name
# that exists only on the assistant's network, so it skipped the on-premises
# GPU-box engagement entirely (ADR-023) and checked nothing. On the project
# network that name resolves, so **every deployment is now actually checked.**
say "Checking the AI model is reachable…"
OLLAMA_URL="$(docker compose config 2>/dev/null \
  | grep -oE 'OLLAMA_HOST[=:][[:space:]]*[^[:space:]]+' \
  | head -1 | sed -E 's/^OLLAMA_HOST[=:][[:space:]]*//')"
[ -n "${OLLAMA_URL:-}" ] || OLLAMA_URL="the address in docker-compose.yml"

if ! docker compose run --rm probe >/dev/null 2>&1; then

  # Distinguish the two worlds before saying anything: they need opposite
  # actions, and telling a client to start software that is already running is
  # how a support thread starts (leg B #13).
  if curl -fsS --max-time 5 http://localhost:11434/api/tags >/dev/null 2>&1; then
    say ""
    say "Ollama is running, but the assistant cannot reach it at ${OLLAMA_URL}."
    say ""
    say "Ollama is only listening to this computer itself, and the assistant"
    say "runs in its own container, which counts as somewhere else. Ollama"
    say "needs to accept connections from it."
    say ""
    say "On Linux there are two common causes, and it may be either or both."
    say ""
    say "1. Ollama listens only to this computer. Run:"
    say ""
    say "       sudo snap set ollama host=0.0.0.0     (if installed via snap)"
    say "       sudo snap restart ollama.listener"
    say ""
    say "   or, for a package install:"
    say ""
    say "       sudo systemctl edit --force ollama"
    say "         (add:  [Service]"
    say "                Environment=\"OLLAMA_HOST=0.0.0.0\" )"
    say "       sudo systemctl restart ollama"
    say ""
    say "2. The firewall blocks the assistant's container. Allow the whole"
    say "   Docker address range, not one interface — this program runs on its"
    say "   own private network, which is not the one you see first:"
    say ""
    say "       sudo ufw allow from 172.16.0.0/12 to any port 11434 proto tcp"
    say ""
    say "Then run this file again. On Windows or macOS neither is usually"
    say "needed — if you see this message there, send me this screen."
  else
    say ""
    say "Ollama is not responding (tried ${OLLAMA_URL})."
    say ""
    say "Ollama is what runs the AI model locally — it is the reason your"
    say "documents never leave this machine. Start Ollama, wait a few seconds,"
    say "then run this file again."
  fi
  say ""
  read -r -p "Press Enter to close." _ || true
  exit 1
fi

# --- 2a. The SECOND model, which the client has no reason to know about ---
# ⚠️ **New with ADR-059, and it is the likeliest first-run failure of any
# delivery after it.**
#
# Reading the client's documents used to need nothing but the image; it now
# needs `bge-m3` on their Ollama. The runbook told them to pull the model that
# writes the answers, so a client who did exactly what they were told still
# fails — inside the build, several minutes in, on a name they have never seen.
#
# Probed as a compose service for the same reason as the check above (ADR-060):
# same network, same address, same anchor. And it asks for an EMBEDDING rather
# than for the tag list, because a tag proves the weights are on disk and only
# an embedding proves the server can load them.
say "Checking the model that reads your documents…"
EMBEDDING_MODEL="$(docker compose config 2>/dev/null \
  | grep -oE 'STILLROOM_EMBEDDING_MODEL[=:][[:space:]]*[^[:space:]]+' \
  | head -1 | sed -E 's/^STILLROOM_EMBEDDING_MODEL[=:][[:space:]]*//')"
[ -n "${EMBEDDING_MODEL:-}" ] || EMBEDDING_MODEL="bge-m3:567m"

if ! docker compose run --rm probe-embedding >/dev/null 2>&1; then
  say ""
  say "The assistant needs a second AI model, and it is not installed yet."
  say ""
  say "There are two: one writes the answers, and one reads your documents so"
  say "the right passage can be found. Only the first one is installed."
  say ""
  say "Install the second one — this downloads about 1.2 GB, once:"
  say ""
  say "       ollama pull ${EMBEDDING_MODEL}"
  say ""
  say "Then run this file again."
  say ""
  read -r -p "Press Enter to close." _ || true
  exit 1
fi

# --- 2b. The things that make the first build fail ------------------------
# Each of these is cheap, and each one otherwise surfaces as a traceback or a
# daemon error in front of the client, during the one minute that decides
# whether they trust this.
#
# ⛔ The Compose check used to live HERE, and that made it dead code in the one
# situation it exists for (found by F4.5, driving the Windows twin's branches).
# Both model probes above are `docker compose run`, so on a machine with Docker
# but no Compose plugin the probe failed first and the client was told their
# Ollama networking was broken — a diagnosis that sends the support thread in
# entirely the wrong direction. It now runs immediately after `docker info`,
# before anything that uses Compose. Same move in start.cmd.

if [ ! -d documents ] || [ -z "$(ls -A documents 2>/dev/null)" ]; then
  say "The 'documents' folder is empty, or is not next to this file."
  say ""
  say "The assistant answers questions about the documents in that folder."
  say "Put them there, then run this file again."
  say ""
  read -r -p "Press Enter to close." _ || true
  exit 1
fi

# --- 2b2. The prepared image, from the file next to this one --------------
# ⚠️ **This has to come before anything that builds** — including the document
# check below, which runs `docker compose run` and will happily build the image
# first. Loading it here is what makes the client's first build touch nothing but
# their own Ollama (ADR-058).
#
# Without it the build downloads ~90 libraries from PyPI. That is a chance to
# fail on a corporate proxy, a TLS-inspecting firewall, or a bad afternoon at a
# CDN. It used to also download an embedding model from a bucket in another
# company's cloud — which was the step OBSERVED failing here, and which ADR-059
# removed entirely by moving the embedder onto the client's own Ollama.
# ⚠️ `BASE_TAG`, not `BASE_IMAGE`. `BASE_IMAGE` is the name Compose reads from
# the environment for the build argument of the same name, and setting it here
# would hand the build a tag it must resolve from a registry it cannot reach.
BASE_TAG="$(sed -n 's/^ARG BASE_IMAGE=//p' Dockerfile | head -1)"
[ -n "${BASE_TAG:-}" ] || BASE_TAG="stillroom-base:0.1.0"

if ! docker image inspect "$BASE_TAG" >/dev/null 2>&1; then
  TARBALL="$(ls -1 stillroom-base-*.tar.gz 2>/dev/null | head -1)"

  if [ -n "$TARBALL" ]; then
    say "Preparing the assistant — this happens once, and takes a minute…"
    if ! docker load -i "$TARBALL" >/dev/null 2>&1; then
      say ""
      say "The prepared file '${TARBALL}' could not be read. It is usually"
      say "incomplete, which happens when a download or a copy is interrupted."
      say ""
      say "Delete it, ask me for it again, and put the new one next to this file."
      say ""
      read -r -p "Press Enter to close." _ || true
      exit 1
    fi
  else
    # ⚠️ It says so and continues, rather than refusing. Same judgement as the
    # model probe above: refusing to start a build that would have worked is a
    # worse failure than starting one that explains itself. A client who has the
    # internet is fine here; a client who does not now knows what to ask for,
    # instead of reading a Python traceback.
    say "The prepared assistant file is not in this folder, so this first run"
    say "will download what it needs from the internet. It usually works. If it"
    say "stops with an error, send me this screen — I will send you the file"
    say "and it will not need the internet at all."
    say ""
    # Compose reads this variable for the build argument of the same name; it
    # names the Dockerfile's own `base` STAGE, so the build makes the base
    # itself instead of expecting a loaded image.
    export BASE_IMAGE=base
  fi
fi

# --- 2c. Documents changed? Re-read them BEFORE starting ------------------
# ⚠️ **Re-running this file used to do nothing at all** for a client who had
# added a document. `docker compose up -d` does not rebuild an image that
# already exists, so the new file was simply not in the assistant's knowledge
# — silently, with no error anywhere (leg B #21).
#
# This is the whole update story, and it is one action: put the file in the
# folder, run this again. The check is cheap because it only reads and chunks
# the documents; the expensive embedding runs only when something actually
# moved.
if docker compose ps --status running --quiet assistant 2>/dev/null | grep -q .; then
  # It is already up. Stopping first is required, not tidiness: the running
  # service pins its corpus when it starts, and re-indexing underneath it is
  # what used to make every question fail.
  say "Stopping the assistant so the documents can be re-read…"
  docker compose stop assistant >/dev/null 2>&1
fi

# ⚠️ BUILD EXPLICITLY, AND BEFORE THE REFRESH — the Linux twin of the same fix
# in `start.cmd` (F5.9, 2026-07-27). `docker compose run` builds the image when
# it is missing, so a BUILD failure arrived wearing the refresh step's error
# message, which blames the client's DOCUMENTS. Found on Windows, where the
# build could not reach Ollama; present identically here, and it survived on
# this side for the usual reason — the machine that tested it could always
# reach its own model.
#
# It also makes the refresh message below true: that message says the assistant
# "will keep answering from the documents it already knew", which was false on a
# first run. Refresh is now only reachable after a successful build.
say "Preparing your assistant — this can take several minutes the first time…"
docker compose build assistant 2>&1 | sed 's/^/    /'
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  say ""
  say "The assistant could not be prepared. Your documents were not touched"
  say "and nothing on this machine has been changed."
  say ""
  say "If the lines above mention a connection being refused, then Ollama is"
  say "running but this build cannot reach it. That is a setting on this"
  say "machine, not a fault in your documents and not something you did wrong."
  say ""
  say "Send me this screen and I will tell you the exact change to make."
  say ""
  read -r -p "Press Enter to close." _ || true
  exit 1
fi

# ⛔ `${PIPESTATUS[0]}`, not `if ! … | sed`, and this one was ALREADY BROKEN.
#
# Without `set -o pipefail`, a pipeline's status is the LAST command's — `sed`,
# which always exits 0. So `if ! docker compose run … | sed` tested sed. This
# branch has never been reachable on Linux: a failing `refresh` was reported as
# a success and the script carried on to `docker compose up -d`.
#
# Found 2026-07-27 while adding the build step above, by nearly writing it the
# same way. `start.cmd` checks `errorlevel 1` and has always been correct, so
# this is the launcher pair disagreeing again — and it survived because the one
# machine that ran `start.sh` end to end could always reach its own model, so
# `refresh` never failed on it.
say "Checking your documents…"
docker compose run --rm --no-deps assistant \
     stillroom refresh --config client.toml 2>&1 | sed 's/^/    /'
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  say ""
  say "Your documents could not be read. The assistant has NOT been changed —"
  say "it will keep answering from the documents it already knew."
  say ""
  say "The most likely cause is more documents than this assistant is set up"
  say "for. The lines above say which. Send me this screen if not."
  say ""
  read -r -p "Press Enter to close." _ || true
  exit 1
fi

# --- 3. Start -------------------------------------------------------------
say ""
say "Starting… the first run can take a minute."
say ""

if ! docker compose up -d; then
  say ""
  say "The assistant could not start. Please send me everything above this"
  say "line and I will tell you what it means."
  say ""
  read -r -p "Press Enter to close." _ || true
  exit 1
fi

# --- 4. Wait for it to be ready, then open the browser --------------------
# The wait is generous on purpose: loading a model on a CPU-only machine is
# genuinely slow, and a browser opened onto a page that is not up yet reads as
# "it is broken" rather than "it is still starting".
say "Waiting for the assistant to be ready…"
ready=""
stopped=""
for _ in $(seq 1 60); do
  if curl -fsS --max-time 3 http://localhost:8000/api/health >/dev/null 2>&1; then
    ready="yes"
    break
  fi
  # ⚠️ **A crashed container and a slow one are not the same thing, and this
  # loop used to report both as "slow"** (leg B #9). The assistant had already
  # exited — the engine raises during startup when it cannot reach the model —
  # and the client was told it was "taking longer than expected", pointed at a
  # URL that would never load, and the script exited 0.
  #
  # Asking Compose whether the thing is still running costs nothing and turns a
  # 120-second wait for a corpse into an immediate, true sentence.
  if ! docker compose ps --status running --quiet assistant 2>/dev/null | grep -q .; then
    stopped="yes"
    break
  fi
  sleep 2
done

if [ -n "$stopped" ]; then
  say ""
  say "The assistant started and then stopped."
  say ""
  say "This is usually the AI model: the assistant checks it can reach it as it"
  say "starts, and gives up if it cannot. What it said:"
  say ""
  docker compose logs --tail=15 assistant 2>&1 | sed 's/^/    /'
  say ""
  say "Send me everything above this line and I will tell you what it means."
  say ""
  read -r -p "Press Enter to close." _ || true
  exit 1
fi

if [ -z "$ready" ]; then
  say ""
  say "It is taking longer than expected. It may still be starting up."
  say "Open this address in your browser in a minute:  http://localhost:8000"
  say ""
  read -r -p "Press Enter to close." _ || true
  exit 0
fi

say ""
hr
say "  Ready.  Open:  http://localhost:8000"
hr
say ""
say "To stop it later, run stop.sh."
say ""

# Best effort — never an error if the machine has no desktop browser.
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open http://localhost:8000 >/dev/null 2>&1 &
elif command -v open >/dev/null 2>&1; then
  open http://localhost:8000 >/dev/null 2>&1 &
fi
