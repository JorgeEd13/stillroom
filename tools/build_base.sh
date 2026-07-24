#!/usr/bin/env bash
#
# Build the base image and the tarball that ships with a delivery.
#
# ⚠️ This one is OURS, not the client's. Everything at the repo root is written
# for somebody who did not choose to be looking at a terminal; this is written
# for me, and it is allowed to have exit codes in it.
#
# What it produces, in `dist/`:
#
#   stillroom-base-<version>.tar.gz   the image, `docker load`-able
#   THIRD-PARTY-NOTICES.md            the licence texts the tarball must carry
#   stillroom-base-<version>.sha256   what the client can check the file against
#
# The tarball is what removes PyPI, Chroma's S3, the client's proxy and their
# TLS interception from the client's first build. After it, the only external
# thing that build touches is their own Ollama.

set -euo pipefail

cd "$(dirname "$0")/.."

VERSION="$(grep -m1 '^version' pyproject.toml | sed -E 's/.*"(.*)".*/\1/')"
IMAGE="stillroom-base:${VERSION}"
OUT="dist"
TARBALL="${OUT}/stillroom-base-${VERSION}.tar.gz"

mkdir -p "$OUT"

echo "==> Building ${IMAGE}"
# ⚠️ No `--network host` here, and that is the point of the whole exercise: the
# BASE build is the one that needs the internet, and it happens on this machine
# where a failed download is a retry rather than a client's first impression.
docker build --target base --build-arg "BASE_IMAGE=base" -t "$IMAGE" .

echo "==> Generating the third-party notices"
# ⚠️ Generated FROM the image and written OUTSIDE it, and both halves matter.
#
# From the image, because licence texts are read from installed distributions and
# a source checkout would describe nothing — it refuses rather than writing a
# confident, empty document.
#
# Outside it, because this is a document the client is obliged to receive; asking
# them to run a container to read their own licence notices would be a strange
# way to discharge an obligation. The engine's own source is bind-mounted for the
# run because the base image deliberately does not contain it.
# ⚠️ Mount a COPY of src with no `*.egg-info`. A stale editable-install egg-info
# on a dev machine sits first on `PYTHONPATH` and shadows the image's own (and
# correct) installed metadata, so `importlib.metadata` reads IT — and an egg-info
# written before a dependency was added silently drops that dependency from the
# notices (this is how the `docs`/`ollama` extras went undeclared). A
# clean checkout has no egg-info (it is gitignored); this makes a dev build match.
NOTICES_SRC="$(mktemp -d)"
cp -r src/. "$NOTICES_SRC/"
find "$NOTICES_SRC" -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
docker run --rm \
  -v "$NOTICES_SRC:/src:ro" -v "$PWD/${OUT}:/out" \
  -e PYTHONPATH=/src \
  "$IMAGE" python -m stillroom.notices --output /out/THIRD-PARTY-NOTICES.md
rm -rf "$NOTICES_SRC"

echo "==> Saving ${TARBALL}"
docker save "$IMAGE" | gzip -9 > "$TARBALL"

( cd "$OUT" && sha256sum "$(basename "$TARBALL")" > "$(basename "${TARBALL%.tar.gz}").sha256" )

echo
echo "==> Done"
ls -lh "$TARBALL" "${OUT}/THIRD-PARTY-NOTICES.md" | sed 's/^/    /'
echo
echo "    Load it with:  docker load -i $(basename "$TARBALL")"
echo "    The launcher does this by itself when the image is missing."
