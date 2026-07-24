# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""The corpus fingerprint — the thing that makes the answer cache honest.

The immutable-corpus rule permits caching a whole answer, which `receivables-agent` refuses to do,
and the entire justification is one property: **a document corpus does not
change between ingests.** A cached answer is therefore exact right up until the
documents change, and lying begins the instant a re-ingest fails to bust the
cache.

So the fingerprint is not a nicety, it is the load-bearing part. It covers
everything that could change an answer without the question changing:

* the chunk IDs and their content — the obvious part;
* the **chunking parameters**, because re-chunking the same bytes changes which
  passages are retrievable and therefore what the model was shown;
* the **embedding function**, because swapping it changes what "similar" means,
  and the cached answer was produced under the old geometry.

Anything else that would change an answer (the model, the prompt, `k`) is
deliberately *not* in here — see `answers.cache`, which keys on those separately.
The split matters: a corpus change invalidates every cached answer, whereas
raising `k` invalidates them for a different reason, and conflating the two
makes it impossible to reason about which cache entries are still true.
"""

from __future__ import annotations

import hashlib

from stillroom.ingest.chunking import Chunk

# Bump when a change to chunking or ID construction should invalidate every
# existing fingerprint even though the client's documents did not change.
FINGERPRINT_VERSION = "1"


def corpus_fingerprint(
    chunks: list[Chunk],
    *,
    chunk_chars: int,
    chunk_overlap: int,
    embedding_name: str,
) -> str:
    """A stable hex digest of exactly what the retriever can see.

    Order-independent by construction: chunk digests are sorted before hashing,
    so two ingests of the same corpus agree even if the files were walked in a
    different order.
    """
    digest = hashlib.sha256()
    digest.update(
        f"v{FINGERPRINT_VERSION}|{chunk_chars}|{chunk_overlap}|{embedding_name}|".encode()
    )

    per_chunk = sorted(
        hashlib.sha256(f"{chunk.id}\x00{chunk.text}".encode("utf-8")).hexdigest()
        for chunk in chunks
    )
    for item in per_chunk:
        digest.update(item.encode("ascii"))

    return digest.hexdigest()
