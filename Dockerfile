# The client's deliverable.
#
# Four things here exist because they already went wrong once — three on the
# an earlier project, one on this product. Each is
# marked ⚠️ — do not "simplify" one without reading the note.

# ⚠️ LANDMINE 5 — the client's build must not need the internet.
#
# The build failed once, here, on the embedding-model download from Chroma's S3.
# On a client machine that is `docker compose up` dying with a Python traceback
# on their FIRST double-click — the single worst moment to fail, and not their
# fault or ours. `pip install` has the same exposure to PyPI, a corporate proxy
# or TLS interception.
#
# So the two halves of this file are delivered differently:
#
#   `base`   — client-independent. Built HERE, handed over as a `docker load`-able
#              tarball, and therefore NEVER built on the client's machine.
#   `client` — their config and their documents. Needs nothing but their own
#              Ollama, which is a stated prerequisite.
#
# `BASE_IMAGE` names the loaded image by default. Pass `--build-arg
# BASE_IMAGE=base` to build the base stage inline instead — the fallback for a
# machine that has the internet but not the tarball, and what CI would use.
#
#     docker build --target base -t stillroom-base:0.1.0 .   # ours, once
#     tools/build_base.sh                                    # …and the tarball
#
# ⚠️ The tag carries a VERSION and it must move when the DEPENDENCIES move, not
# when the engine changes: `client` reinstalls the engine from the delivery
# folder on every build, so a stale base is a stale dependency closure and
# nothing else. It is `pyproject.toml`'s version because that is the file both
# facts live in.
ARG BASE_IMAGE=stillroom-base:0.1.0


FROM python:3.12-slim AS base

# ⚠️ LANDMINE 1 — the embedding model must be downloaded AS THE RUNTIME USER.
#
# On an earlier deployment, the build downloaded the ONNX MiniLM as root while
# the container ran as a non-root user. The model was therefore missing from
# that user's cache at runtime, so it re-downloaded on first request, arrived
# TRUNCATED, and the app crash-looped with:
#     INVALID_PROTOBUF: Protobuf parsing failed  ->  Application startup failed
#
# Creating the user FIRST and doing every download as that user is what makes
# the cache land in the right home directory. For this product the stakes are
# higher than a crash: a runtime download means the container reaches the
# network to serve an answer, which contradicts the entire claim being sold.
RUN groupadd --gid 1000 app \
 && useradd --uid 1000 --gid 1000 --create-home app

WORKDIR /app
RUN chown app:app /app
USER app
ENV PATH="/home/app/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ⚠️ `pyproject.toml` and NOTHING ELSE, so this image is a pure function of the
# dependency declaration — which is what makes the tag honest (the tarball says the
# tag tracks the dependency set, not the engine).
#
# `README.md` used to be copied here too, out of habit. `pyproject.toml` declares
# no `readme` key, so the build never needed it — but its presence meant editing
# the README made this image drift from the tarball on disk, silently, with the
# version unchanged. Found by the Fluxo L deploy-target check, not by a failure.
COPY --chown=app:app pyproject.toml ./

# ⚠️ **No engine source in the base image, and the placeholder is why**.
#
# The obvious version of this stage copies `src/` and installs it, because that
# is how the dependency closure gets resolved. It also puts a frozen copy of our
# tree inside a layer of a tarball we hand over — and a later layer overwriting
# it does not remove it, exactly as a later commit does not remove what an
# earlier one published. That is the shape of an ignore-pattern leak,
# and it is cheap to avoid: resolve the dependencies against an EMPTY package of
# the same name.
#
# What the base therefore contains is third-party software and nothing else,
# which is also what makes its tag mean "this dependency closure" rather than
# "this engine".
RUN mkdir -p src/stillroom && touch src/stillroom/__init__.py

# The build backend, installed explicitly so the `client` stage can reinstall the
# engine with `--no-build-isolation` — i.e. without pip building an isolated
# environment, which is the step that would reach PyPI.
RUN pip install --no-cache-dir --user "setuptools>=68" wheel \
 && pip install --no-cache-dir --user ".[ollama,docs]"

# ⚠️ **LANDMINE 1 IS DEAD, and the step that used to be here is GONE**.
#
# This is where the ONNX MiniLM was downloaded from Chroma's S3 and warmed into
# `app`'s cache. `bge-m3` is served by the client's own Ollama, so there is no
# embedding model in this image at all any more. Three consequences, and none of
# them is cosmetic:
#
#   1. **The one step whose upstream we do not control is gone.** Chroma's S3 was
#      observed failing mid-session, and it is the reason the base ships as a
#      tarball. The tarball is still right — PyPI is still an exposure — but the
#      worst of the four failure modes has been removed rather than routed around.
#   2. **We are no longer redistributing model weights.**
#      `bom.py` and `notices.py` say so; if this line ever comes back, they have
#      to change back with it, in the same commit.
#   3. **The image is smaller by the size of the model.**
#
# ⚠️ Landmine 1's *lesson* is still live even though its instance is dead: any
# download added here must run as the RUNTIME user, or it lands in root's home
# and is silently re-fetched at runtime — which for this product means the
# container reaching the network to answer a question, contradicting the claim
# being sold.

# The third-party licence texts are generated FROM this image by
# `tools/build_base.sh`, and they land beside the tarball rather than inside it —
# a document the client is obliged to receive is a document they should be able
# to read without running anything.


# ---------------------------------------------------------------------------
# The per-client image. Built on the CLIENT's machine, with their config and
# their documents; this is the artifact that gets handed over.
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS client

# ⚠️ This is where the engine enters, and it enters only here.
#
# The base carries the dependency closure against an empty package of this name;
# this install replaces it with the real one, from the delivery folder. So the
# code that RUNS is provably the code the client can read — which is the whole
# point of a source-available licence: an auditor reading `src/`
# beside this file is reading what answers their questions.
# `--no-deps --no-build-isolation` keeps it offline; nothing is resolved and
# nothing is fetched.
COPY --chown=app:app pyproject.toml ./
COPY --chown=app:app src/ ./src/
RUN pip install --no-cache-dir --user --no-deps --no-build-isolation .

# The client's configuration and corpus. Both are mounted or copied at build
# time depending on the engagement — never baked into a public image.
COPY --chown=app:app client.toml ./client.toml
COPY --chown=app:app documents/ ./documents/

# The delivery manifest, if this build has one. `client.toml*` also matches
# `client.toml`, so the pattern always has at least one hit and the COPY cannot
# fail on a delivery folder that predates manifests — Docker errors on a COPY
# whose every source is missing, and a build that breaks because an OPTIONAL
# file is absent is worse than the check it was carrying.
#
# ⚠️ It is written by us at handover and NOT regenerated by the client's build.
# That asymmetry is the whole mechanism: a rebuild after an edited config keeps
# the delivered manifest, so `doctor` can see the two disagree. It records
# nothing about the documents and nothing confidential — only the agreed limits.
COPY --chown=app:app client.toml* DELIVERY_MANIFEST.jso[n] ./

# Provenance, not protection. Answers "which build is this?" when a support
# bundle arrives, and survives `docker inspect` on the client's own machine.
ARG BUILD_ID=""
LABEL org.opencontainers.image.title="stillroom" \
      org.opencontainers.image.licenses="LicenseRef-PolyForm-Shield-1.0.0" \
      io.stillroom.build-id="${BUILD_ID}"

# ⚠️ LANDMINE 2 — BAKE, do not warm at startup.
#
# An earlier project seeds its curated cache at BUILD time, which is why its headline
# questions answer in ~1.4 s from the very first request after every deploy. If
# this ran at startup instead, the client's first session would be the slow one
# — the worst possible first impression, on the exact feature the product
# rests on. `bake` also exits non-zero when a curated question cannot be answered
#, so a corpus that does not answer the client's own questions fails
# the BUILD rather than being discovered at handover.
#
# ⚠️ THEREFORE THIS BUILD NEEDS OLLAMA REACHABLE — and since the embedder swap it is the
# ONLY thing this build needs. Build it with:
#
#     docker build --network host --target client -t stillroom-<client> .
#
# This differs from that project and the difference is intrinsic, not an
# oversight: it seeds curated *plans* — hand-written SQL that needs no
# model — whereas this bakes whole *answers*, and an answer requires
# the model that will serve it. Verified: build reaches the host's Ollama over
# `--network host`, and a fresh container then answers a curated question in
# **218 ms on its very first request**.
#
# ⚠️ **BOTH lines below now need Ollama, and until the move to bge-m3 only the second did.**
# `ingest` used to embed with an ONNX model inside this image, so it could not
# fail for network reasons; it now calls `bge-m3` on the client's Ollama. The
# practical consequence is that the build fails EARLIER and for a different
# reason than it used to, and the failure names a model the client has probably
# not pulled — which is why both launcher preflights check the embedding model
# by name rather than checking that "Ollama is up".
RUN stillroom ingest --config client.toml \
 && stillroom bake   --config client.toml

EXPOSE 8000

# Bound to 0.0.0.0 inside the container only; the compose file decides what is
# published, and by default that is loopback on the host.
CMD ["stillroom", "serve", "--config", "client.toml", "--host", "0.0.0.0", "--port", "8000"]
