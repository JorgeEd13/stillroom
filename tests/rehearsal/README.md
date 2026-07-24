# The rehearsal harness (F5)

Before a build is handed to a client, it should be run the way the client will
run it — not read, run. This harness assembles a delivery folder exactly as a
client receives it, drives the **real `start.sh`** through it against a real
Docker, a real Ollama and the real 170 MB base tarball, and then asks the running
assistant live questions over the port the client actually gets.

It is the Linux twin of [`../windows/`](../windows/README.md). That harness drives
`start.cmd` with **stubbed** tools to prove the batch logic on a platform this
machine cannot run; this one drives `start.sh` with **nothing stubbed** to prove
that a real build stands up, publishes where it should, and answers.

```sh
./rehearse.sh                                    # full run against qwen2.5:7b
STILLROOM_REHEARSAL_MODEL=qwen2.5:1.5b ./rehearse.sh   # a leaner model class
```

## ⚠️ What a pass here means, and what it does not

Two honesty boundaries, and every claim produced from this harness has to carry
both.

### 1. It rehearses a MODEL CLASS, not the client's GPU

The build is exercised against `$MODEL` on **this** machine's Ollama. A green run
means *"the software, the config and this class of model produce a working,
grounded, refusing assistant"*. It says **nothing** about whether the client's
specific card can host that model — that is what the qualifier
(`machine_scanner`) decides *before* purchase, and what **F5.9** confirms on real
Windows hardware. Never describe a pass here as "it runs on the client's machine".

The config pins the model **explicitly** rather than using `auto`. `auto`
resolves against whatever machine runs it, so on a rehearsal box it would silently
pick a different class than the client's hardware supports — which is the exact
substitution this boundary exists to refuse.

### 2. It runs the COMPOSE path, from the TARBALL — not `--network host`

The one path that was never in doubt is the one a `--network host` rehearsal
re-proves (leg B #10/#11). F3.5's live-answer check ran that way, which **cancels
the compose file's loopback-only publishing**: the bridge-network failure
survived it, and a leftover rehearsal container was still listening on
`0.0.0.0:8000` fifteen hours later. So this harness does two things a
`--network host` run cannot:

- **It forces the tarball-load branch.** It removes any pre-loaded base image
  first, so `start.sh` genuinely runs `docker load` on the 170 MB tarball — the
  client's actual first-build path, not a shortcut past it.
- **It asserts the running service is published on `127.0.0.1`, not `0.0.0.0`.**
  That is mechanical proof the compose path ran. A `--network host` container has
  no compose port mapping at all and is reachable on every interface.

A rehearsal that skips either half rehearses a path no client takes.

## Why it is not in the pytest suite

Rule 7 (`../../CLAUDE.md`): `tests/` runs offline — no model download, no Ollama,
no network. This harness needs all three. Like the Windows harness it lives under
`tests/` for proximity to the launcher it exercises, not because pytest collects
it (there is no `test_*.py` here, so pytest does not).

## The corpus is realistic on purpose

`corpus/` is seven short business documents — returns, shipping, a staff
handbook, expenses, pricing, security, support — the same set that re-derived the
`0.50` relevance floor in `benchmarks/retrieval_floor.py`. It is deliberately not
a one-paragraph fixture: the §fixture-hides-defect rule cost this product a floor
that was measured on four paragraphs and quoted as a gate for four phases while
it gated nothing. A rehearsal against a toy corpus would answer a question no
client asks.

It is synthetic and clean-room — no client material ever becomes a fixture here
(the clean-room rule).

## What the run checks

1. **Preconditions** — Docker, Compose, Ollama, both models pulled, the tarball built.
2. **Delivery folder** — assembled from the real deliverable; `start.sh` verified
   byte-identical to the repo's (we drive the artifact, not a variant).
3. **Tarball-load path forced** — the pre-loaded base image is removed.
4. **The real `start.sh`** — loads the tarball, probes Ollama, builds the client
   image (ingest + bake), starts on the bridge network, waits for health. The run
   asserts it announced *"Preparing the assistant…"* (the tarball branch) and
   reached *"Ready"* with exit 0.
5. **The compose path** — the service is published on `127.0.0.1:8000`, and no
   container anywhere is exposing `8000` on `0.0.0.0`.
6. **Live answers** over `http://127.0.0.1:8000`, the client's own port:
   - a **curated** question returns instantly from the baked cache (`served_by=cache`);
   - a **non-curated on-topic** question runs the live model and carries a
     citation (`served_by=model`);
   - an **off-topic** question is refused at the relevance floor with **no model
     call** (`served_by=no-match`) — the `0.50` floor, proven live rather than
     in a benchmark.
7. **Teardown** — the real `stop.sh` runs, and the harness proves nothing is left
   listening on `8000` (the leg B #11 leftover-container class). Cleanup also runs
   on every exit, including an abort.

## Prerequisites (none is in the deliverable)

- Docker with the Compose plugin, daemon running
- Ollama serving, with the chat model and `bge-m3:567m` pulled
- the base tarball built in `dist/` (`tools/build_base.sh`)
- `jq`, `python3`, `curl`
