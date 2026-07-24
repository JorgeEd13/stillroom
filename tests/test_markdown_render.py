"""Runs the JavaScript renderer test, so `pytest` stays the one command.

The renderer is the fix for landmine 3 and it is written in JavaScript, so it
has to be exercised in JavaScript (`tests/js/markdown_test.mjs`). Wiring it in
here means nobody has to remember a second command — the failure mode otherwise
is a suite that is green while the only new code in the phase is unverified.

⚠️ **A skip here is not a pass.** If Node is missing, the renderer was not
tested at all: no `<strong>`, no citation chips, no proof that a partial answer
does not hang the browser. Node is a *test-time* tool only — it is not in the
container, not in the deliverable, and no package is ever installed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SUITE = Path(__file__).parent / "js" / "markdown_test.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_markdown_renderer_passes_its_own_suite():
    result = subprocess.run(
        ["node", "--test", str(SUITE)],
        capture_output=True,
        text=True,
        timeout=120,
        # No network, no package manager, no node_modules: the suite uses only
        # `node:test` and a stub DOM it defines itself.
        cwd=SUITE.parent,
    )

    assert result.returncode == 0, result.stdout + result.stderr
