#!/usr/bin/env bash
#
# F5 — rehearse a client build before shipping it.
#
# This assembles a delivery folder EXACTLY as a client receives it, drives the
# REAL `start.sh` through it against a real Docker, a real Ollama and the real
# 170 MB base tarball, then asks the running assistant live questions over the
# port the client actually gets. It is the Linux twin of `tests/windows/` — that
# harness drives `start.cmd` with stubbed tools to prove the batch logic; this
# one drives `start.sh` with nothing stubbed to prove a build stands up and
# answers.
#
# ⚠️ READ THIS BEFORE QUOTING A RESULT. Two honesty boundaries, both load-bearing.
#
#  1. IT REHEARSES A MODEL CLASS, NOT THE CLIENT'S GPU. The build is exercised
#     against `$MODEL` on THIS machine's Ollama. A pass means "the software, the
#     config and this class of model produce a working, grounded assistant" — it
#     says nothing about whether the client's specific card can host it. That is
#     what the qualifier (`machine_scanner`) decides before purchase, and what
#     F5.9 confirms on real hardware. Never describe a green run here as "it runs
#     on the client's machine".
#
#  2. IT RUNS THE COMPOSE PATH, FROM THE TARBALL — because the one path that was
#     never in doubt is the one a `--network host` rehearsal re-proves (leg B
#     #10/#11). F3.5's live-answer check ran `--network host`, which cancels the
#     compose file's loopback-only publishing; the bridge-network failure
#     survived it, and a leftover container was still listening on 0.0.0.0:8000
#     fifteen hours later. So this harness FORCES the tarball-load branch (it
#     removes the pre-loaded base image first) and then asserts the running
#     service is published on 127.0.0.1 and NOT 0.0.0.0. A rehearsal that skips
#     either half rehearses a path no client takes.
#
# It is deliberately NOT in the pytest suite: rule 7 says `tests/` runs offline,
# and this needs Docker, Ollama and the network. Like the Windows harness, it
# lives under tests/ for proximity to what it exercises, not for collection.
#
#   ./rehearse.sh                 # full run against qwen2.5:7b
#   STILLROOM_REHEARSAL_MODEL=qwen2.5:1.5b ./rehearse.sh   # a leaner class
#
# Prerequisites (none is in the deliverable; this is a development tool):
#   - Docker with the Compose plugin, and the daemon running
#   - Ollama serving, with $MODEL and $EMBEDDING pulled
#   - the base tarball built in dist/ (tools/build_base.sh)
#   - jq, python3, curl

set -u

# --- configuration --------------------------------------------------------
MODEL=${STILLROOM_REHEARSAL_MODEL:-qwen2.5:7b}
EMBEDDING=${STILLROOM_REHEARSAL_EMBEDDING:-bge-m3:567m}
OLLAMA_URL=${STILLROOM_REHEARSAL_OLLAMA:-http://localhost:11434}
API_KEY=${STILLROOM_REHEARSAL_KEY:-rehearsal-key-not-a-secret}
PORT=8000

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)                 # the stillroom repo root
work=${STILLROOM_REHEARSAL_DIR:-/tmp/stillroom-rehearsal}
delivery="$work/delivery"

VERSION="$(sed -n 's/^version[[:space:]]*=[[:space:]]*"\(.*\)".*/\1/p' "$root/pyproject.toml" | head -1)"
[ -n "$VERSION" ] || VERSION="0.1.0"
BASE_TAG="stillroom-base:${VERSION}"
TARBALL="$root/dist/stillroom-base-${VERSION}.tar.gz"

# --- reporting ------------------------------------------------------------
fails=0
pass()  { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fails=$((fails + 1)); }
info()  { printf '  ....  %s\n' "$1"; }
section() { printf '\n=== %s\n' "$1"; }
die()   { printf '\n\033[31mABORT\033[0m  %s\n' "$1" >&2; exit 2; }

# `assert <label> <condition-cmd...>` — runs the command, PASS on exit 0.
assert() { local label=$1; shift; if "$@" >/dev/null 2>&1; then pass "$label"; else fail "$label"; fi; }

# --- teardown, always -----------------------------------------------------
# ⚠️ The leftover-container failure (leg B #11) is exactly what an un-torn-down
# rehearsal leaves behind, so cleanup runs on EVERY exit, success or abort.
teardown() {
  if [ -d "$delivery" ]; then
    ( cd "$delivery" && docker compose down -v >/dev/null 2>&1 )
  fi
}
trap teardown EXIT

printf '\n%s\n' "------------------------------------------------------------"
printf '  F5 rehearsal — model class: %s   base: %s\n' "$MODEL" "$BASE_TAG"
printf '%s\n' "------------------------------------------------------------"

# --- 1. Preconditions -----------------------------------------------------
section "1. Preconditions"
command -v docker  >/dev/null 2>&1 || die "docker is not installed."
command -v jq      >/dev/null 2>&1 || die "jq is not installed."
command -v python3 >/dev/null 2>&1 || die "python3 is not installed."
docker info >/dev/null 2>&1        || die "the Docker daemon is not running."
docker compose version >/dev/null 2>&1 || die "the Docker Compose plugin is missing."
curl -fsS --max-time 5 "$OLLAMA_URL/api/tags" >/dev/null 2>&1 || die "Ollama is not reachable at $OLLAMA_URL."
[ -f "$TARBALL" ] || die "the base tarball is missing: $TARBALL (run tools/build_base.sh)."

tags="$(curl -fsS "$OLLAMA_URL/api/tags")"
printf '%s' "$tags" | jq -e --arg m "$MODEL"     '.models[]?.name | select(. == $m)'     >/dev/null \
  || die "the chat model '$MODEL' is not pulled on this Ollama."
printf '%s' "$tags" | jq -e --arg m "$EMBEDDING" '.models[]?.name | select(. == $m)'     >/dev/null \
  || die "the embedding model '$EMBEDDING' is not pulled on this Ollama."
info "docker, compose, ollama, both models and the tarball are all present."

# --- 2. Assemble the delivery folder --------------------------------------
# Exactly what a client receives: the launcher, the compose file and Dockerfile,
# the engine source (it enters the image only in the `client` stage), a realistic
# corpus, the pinned config, and the tarball next to it.
section "2. Assemble the delivery folder — what the client actually receives"
rm -rf "$delivery"
mkdir -p "$delivery/documents"
cp "$root/start.sh" "$root/stop.sh" "$root/docker-compose.yml" "$root/Dockerfile" \
   "$root/.dockerignore" "$root/pyproject.toml" "$delivery/" || die "could not copy the deliverable."
cp -r "$root/src" "$delivery/src"
cp "$here/corpus/"*.md "$delivery/documents/"
cp "$TARBALL" "$delivery/"

# ⚠️ We drive the ARTIFACT, not a variant of it. If the copied launcher is not
# byte-identical to the one in the repo, the rehearsal is rehearsing something
# the client will never run.
assert "the delivery start.sh is byte-identical to the repo's" \
  cmp -s "$root/start.sh" "$delivery/start.sh"

# The config a client gets: model class PINNED (not "auto" — auto resolves
# against THIS machine, and the point is to rehearse the class their hardware
# supports), loopback publishing left to compose, container-internal paths.
cat > "$delivery/client.toml" <<TOML
client = "Rehearsal Ltd"
language = "en"
index_path = "/app/index"
collection = "documents"
api_key = "${API_KEY}"

[model]
kind = "ollama"
name = "${MODEL}"
keep_alive = "10m"

[corpus]
path = "/app/documents"
include = [".md", ".txt", ".pdf", ".docx"]
chunk_chars = 1200
chunk_overlap = 150

[retrieval]
k = 5
min_similarity = 0.50

[retrieval.spelling_retry]
enabled = true

[embedding]
model = "${EMBEDDING}"

[answer_cache]
enabled = true
threshold = 0.90
curated = [
    "What is the refund window?",
    "How much notice do I have to give?",
    "Who approves an expense over the limit?",
]

[conversation]
enabled = true
max_turns = 6
max_chars_per_turn = 600
TOML
info "delivery folder assembled at $delivery ($(ls "$delivery/documents" | wc -l) documents)."

# --- 3. Force the tarball-load path ---------------------------------------
# If the base image is already loaded, start.sh skips the tarball. Removing it
# is what makes "from the tarball" real rather than assumed. The client image
# and any stale volume go too, so this is a genuine first-run build.
section "3. Force the tarball-load path (remove any pre-loaded image)"
( cd "$delivery" && docker compose down -v >/dev/null 2>&1 )
docker rmi -f "$BASE_TAG" "stillroom-client:latest" >/dev/null 2>&1 || true
if docker image inspect "$BASE_TAG" >/dev/null 2>&1; then
  fail "the base image is still present; start.sh would skip the tarball."
else
  pass "no pre-loaded base image — start.sh must load the tarball."
fi

# --- 4. Drive the real start.sh -------------------------------------------
# `</dev/null` so the error-path `read` prompts return immediately; the happy
# path never blocks. This is the client's whole first run: load the tarball,
# probe Ollama, build the client image (ingest + bake), start on the bridge
# network, wait for health.
section "4. Drive the real start.sh (this builds and starts — can take a few minutes)"
( cd "$delivery" && ./start.sh </dev/null ) 2>&1 | tee "$work/start.log" | sed 's/^/      /'
start_rc=${PIPESTATUS[0]}

if grep -q "Preparing the assistant" "$work/start.log"; then
  pass "start.sh took the tarball-load branch (\"Preparing the assistant…\")."
else
  fail "start.sh did NOT report loading the tarball — the wrong branch ran."
fi
if [ "$start_rc" = 0 ] && grep -q "Ready.  Open" "$work/start.log"; then
  pass "start.sh reached \"Ready\" and exited 0."
else
  fail "start.sh did not reach a clean \"Ready\" (exit $start_rc)."
fi

# --- 5. The compose path: loopback, NOT --network host --------------------
# This is honesty boundary #2, made mechanical. A --network host run would have
# no compose port mapping at all and would be reachable on 0.0.0.0; the compose
# path publishes on 127.0.0.1 only.
section "5. Verify the compose path — published on loopback, not 0.0.0.0"
published="$(cd "$delivery" && docker compose port assistant "$PORT" 2>/dev/null)"
info "compose publishes the service at: ${published:-<nothing>}"
case "$published" in
  127.0.0.1:*) pass "the service is published on loopback (127.0.0.1) — the client's deployment." ;;
  0.0.0.0:*|:::*|*) fail "the service is NOT loopback-only (got '${published:-none}')." ;;
esac
# Belt and suspenders: nothing anywhere is exposing 8000 on 0.0.0.0.
if docker ps --format '{{.Ports}}' | grep -q "0.0.0.0:${PORT}"; then
  fail "a container is publishing ${PORT} on 0.0.0.0 — a --network host leftover class."
else
  pass "no container publishes ${PORT} on 0.0.0.0."
fi

# --- 6. Live-answer verification over the client's port -------------------
# Reached at http://127.0.0.1:8000 — the same address the client's browser uses,
# on the same network the assistant serves on.
section "6. Live answers over http://127.0.0.1:${PORT} (the port the client gets)"
base="http://127.0.0.1:${PORT}"
jqget() { python3 -c 'import sys,json;print(json.load(sys.stdin).get(sys.argv[1],""))' "$1"; }

health="$(curl -fsS --max-time 10 "$base/api/health" 2>/dev/null)"
if [ -n "$health" ]; then
  ready="$(printf '%s' "$health" | jqget ready)"
  onprem="$(printf '%s' "$health" | jqget documents_stay_on_premises)"
  [ "$ready" = "True" ]   && pass "/api/health reports ready."   || fail "/api/health not ready ($ready)."
  [ "$onprem" = "True" ]  && pass "/api/health confirms documents stay on premises." \
                          || fail "/api/health does not confirm on-premises ($onprem)."
else
  fail "/api/health did not respond."
fi

# ask <question> -> prints "served_by\tciting?\treply" (tab-separated)
ask() {
  local q body
  body="$(jq -nc --arg q "$1" '{question:$q}')"
  local resp
  resp="$(curl -fsS --max-time 120 -X POST "$base/api/ask" \
            -H "X-API-Key: ${API_KEY}" -H 'Content-Type: application/json' \
            -d "$body" 2>/dev/null)"
  printf '%s' "$resp" | python3 -c '
import sys, json
d = json.load(sys.stdin)
cites = d.get("citations") or []
print("\t".join([d.get("served_by",""), str(len(cites)), (d.get("reply","") or "").replace("\n"," ")[:80]]))
'
}

# 6a. A curated question — must come back as an instant answer (bake worked).
IFS=$'\t' read -r sv nc reply <<<"$(ask "What is the refund window?")"
info "curated:   served_by=$sv  citations=$nc  reply=\"$reply\""
[ "$sv" = "cache" ] && pass "curated question served instantly from the baked cache." \
                    || fail "curated question was not served from cache (served_by=$sv)."

# 6b. A NON-curated on-topic question — the live model path, grounded + cited.
IFS=$'\t' read -r sv nc reply <<<"$(ask "How long does international shipping take?")"
info "on-topic:  served_by=$sv  citations=$nc  reply=\"$reply\""
[ "$sv" = "model" ] && pass "on-topic question ran the live model." \
                    || fail "on-topic question did not run the model (served_by=$sv)."
[ "${nc:-0}" -gt 0 ] 2>/dev/null && pass "the on-topic answer carries a citation." \
                    || fail "the on-topic answer has no citation."

# 6c. An off-topic question — the 0.50 floor, live. Must refuse with NO model call.
IFS=$'\t' read -r sv nc reply <<<"$(ask "What is the boiling point of water?")"
info "off-topic: served_by=$sv  citations=$nc  reply=\"$reply\""
[ "$sv" = "no-match" ] && pass "off-topic question refused at the relevance floor (no model call)." \
                       || fail "off-topic question was NOT refused (served_by=$sv)."

# --- 7. The client's stop path, and no leftovers --------------------------
section "7. Stop with the real stop.sh, and prove nothing is left listening"
( cd "$delivery" && ./stop.sh </dev/null ) 2>&1 | sed 's/^/      /'
sleep 1
if docker ps --format '{{.Ports}}' | grep -q ":${PORT}"; then
  fail "something is still publishing ${PORT} after stop.sh."
else
  pass "stop.sh left nothing listening on ${PORT}."
fi

# --- Report ---------------------------------------------------------------
section "Result"
if [ "$fails" -eq 0 ]; then
  printf '  \033[32mAll checks passed.\033[0m\n'
else
  printf '  \033[31m%d check(s) failed.\033[0m\n' "$fails"
fi
cat <<'BOUNDARIES'

  What this run proved, and what it did NOT:
    • It rehearsed a MODEL CLASS on this machine's Ollama. It did NOT test the
      client's GPU — the qualifier decides that before purchase, F5.9 confirms
      it on real Windows hardware.
    • It ran the COMPOSE path, from the 170 MB tarball, published on loopback.
      That is the client's deployment. A --network host rehearsal would prove a
      path no client takes.
BOUNDARIES

exit $([ "$fails" -eq 0 ] && echo 0 || echo 1)
