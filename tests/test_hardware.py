"""Model selection. Pure functions, no machine and no Ollama contacted.

The selection logic is what stops a client build recommending a model that will
not load — and the F1 finding it exists to avoid (counting integrated-GPU
"VRAM" as if it were dedicated) is pinned here so it cannot come back.
"""

from __future__ import annotations

import stillroom.hardware as hw
from stillroom.hardware import CATALOG, FLOOR_MODEL, Profile, recommend_model


def _spec(name: str):
    return next(s for s in CATALOG if s.name == name)


def test_a_gpu_box_budgets_on_vram():
    profile = Profile(ram_total_gb=64.0, vram_gb=8.0)

    assert profile.has_gpu
    assert profile.effective_memory_gb == 8.0


def test_a_cpu_box_keeps_headroom_below_total_ram():
    profile = Profile(ram_total_gb=16.0, vram_gb=0.0)

    assert not profile.has_gpu
    # The OS, this process and the index all need memory while the model loads.
    assert profile.effective_memory_gb == 12.8


def test_a_tiny_gpu_is_ignored_in_favour_of_system_ram():
    """A card too small to hold a model plus its KV cache is not a GPU here."""
    profile = Profile(ram_total_gb=32.0, vram_gb=1.0)

    assert not profile.has_gpu
    assert profile.effective_memory_gb == 25.6


def test_catalog_is_ordered_by_quality():
    qualities = [spec.quality for spec in CATALOG]

    assert qualities == sorted(qualities)


def test_a_gpu_model_needs_more_memory_than_the_cpu_figure():
    """VRAM requirements exceed RAM ones; picking the wrong one over-promises."""
    spec = _spec("qwen2.5:7b")

    assert spec.requirement_for(has_gpu=True) == spec.vram_gb
    assert spec.requirement_for(has_gpu=False) == spec.ram_gb


def test_selection_picks_the_best_model_that_fits(monkeypatch):
    monkeypatch.setattr(hw, "detect_profile", lambda: Profile(64.0, 8.0))
    monkeypatch.setattr(hw, "list_downloaded", lambda url, timeout=2.0: set())

    assert recommend_model() == "gemma2:9b"


def test_selection_prefers_a_model_already_downloaded(monkeypatch):
    """So the first question after handover does not stall on a large pull."""
    monkeypatch.setattr(hw, "detect_profile", lambda: Profile(64.0, 8.0))
    monkeypatch.setattr(hw, "list_downloaded", lambda url, timeout=2.0: {"qwen2.5:7b"})

    assert recommend_model() == "qwen2.5:7b"


def test_a_downloaded_model_that_does_not_fit_is_not_chosen(monkeypatch):
    """Already pulled is a tie-breaker among models that fit, not an override."""
    monkeypatch.setattr(hw, "detect_profile", lambda: Profile(8.0, 0.0))
    monkeypatch.setattr(hw, "list_downloaded", lambda url, timeout=2.0: {"qwen2.5:14b"})

    assert recommend_model() != "qwen2.5:14b"


def test_a_machine_below_the_floor_still_starts(monkeypatch):
    """The gate that screens this machine out runs before delivery, not here."""
    monkeypatch.setattr(hw, "detect_profile", lambda: Profile(0.5, 0.0))
    monkeypatch.setattr(hw, "list_downloaded", lambda url, timeout=2.0: set())

    assert recommend_model() == FLOOR_MODEL


def test_an_unreachable_ollama_does_not_break_selection(monkeypatch, allow_network):
    """Selection degrades to the catalog; the provider raises later if it must."""
    monkeypatch.setattr(hw, "detect_profile", lambda: Profile(64.0, 8.0))

    # A real call against a closed port, not a stub — this is the actual path
    # taken when Ollama is down at startup.
    assert hw.list_downloaded("http://127.0.0.1:9", timeout=0.2) == set()
    assert recommend_model("http://127.0.0.1:9") == "gemma2:9b"


# --------------------------------------------------- licences ---
#
# "Free to download" is not "free to use commercially", and the obligation
# lands on the CLIENT — which the engagement terms forbid us from creating for
# them. These tests exist because the defect was invisible: `qwen2.5:3b` was
# simply the best model that fits a 4 GB machine, and nothing recorded that it
# is the one size in its family under a research licence.


def test_every_catalog_entry_declares_a_licence():
    """No default, no blank. A model with unrecorded terms cannot be put in a
    bill of materials, and one that cannot be listed cannot be delivered."""
    for spec in CATALOG:
        assert spec.licence.name, f"{spec.name} has no licence recorded"
        assert isinstance(spec.licence.commercial, bool)


def test_the_research_licensed_model_is_never_recommended(monkeypatch):
    """The exact defect: it fits, it is downloaded, it is the highest-quality
    model that fits — and it still must not be chosen."""
    monkeypatch.setattr(hw, "detect_profile", lambda: Profile(32.0, 0.0))
    monkeypatch.setattr(hw, "list_downloaded", lambda *a, **k: {"qwen2.5:3b"})

    assert _spec("qwen2.5:3b").licence.commercial is False
    assert recommend_model() != "qwen2.5:3b"


def test_a_4gb_machine_still_gets_a_model_it_may_lawfully_run(monkeypatch):
    """Excluding it must not leave that hardware band empty."""
    monkeypatch.setattr(hw, "detect_profile", lambda: Profile(5.0, 0.0))
    monkeypatch.setattr(hw, "list_downloaded", lambda *a, **k: set())

    chosen = recommend_model()
    assert hw.licence_for(chosen).commercial is True


def test_nothing_selectable_is_non_commercial():
    for spec in hw.DELIVERABLE:
        assert spec.licence.commercial, f"{spec.name} is not deliverable"


def test_the_floor_model_is_commercially_licensed():
    """The floor is what a struggling machine falls back to — it is delivered
    more often than any other, and by definition without deliberation."""
    assert hw.licence_for(FLOOR_MODEL).commercial is True


def test_obligations_are_recorded_for_the_licences_that_have_them():
    """These are what the client actually has to DO, and they belong in the
    bill of materials rather than in a lawyer's later discovery."""
    assert hw.licence_for("llama3.1:8b").use_obligations
    assert hw.licence_for("gemma2:9b").use_obligations
    # Permissive licences bind nobody who merely runs the thing, and saying so
    # is the useful answer.
    assert hw.licence_for("qwen2.5:7b").use_obligations == ()


def test_running_a_model_is_not_distributing_it():
    """The delivery model ships no weights — the client's own Ollama pulls them.
    Both agreements hang attribution and notice duties on *Distribution*, so
    those must not be reported as things the client has to do.

    ⚠️ If the delivery model ever changes to handing weights over, this test is
    the one that should start looking wrong."""
    llama = hw.licence_for("llama3.1:8b")
    attribution = "Built with Llama"

    assert any(attribution in ob for ob in llama.distribution_obligations)
    assert not any(attribution in ob for ob in llama.use_obligations)


def test_the_two_llama_versions_are_not_one_licence():
    """Separate agreements, different notice strings, different use-policy URLs.
    §6.2 asks for *the applicable licence* per item, so one shared object would
    put the wrong document in front of a client's lawyer."""
    three_one = hw.licence_for("llama3.1:8b")
    three_two = hw.licence_for("llama3.2:3b")

    assert three_one is not three_two
    assert "3.1" in three_one.name
    assert "3.2" in three_two.name


def test_the_mau_threshold_is_a_gate_and_not_a_bar():
    """§2 sends a very large organisation to Meta to *request* a licence, which
    Meta may grant. Reporting it as "not available" would tell such a client
    they may not use what they may."""
    (mau,) = [
        ob for ob in hw.licence_for("llama3.1:8b").use_obligations
        if "monthly active users" in ob
    ]
    assert "request a licence" in mau


def test_every_verified_licence_cites_an_archived_document():
    """The archive's rule, enforced: a legal fact needs a quote from a document
    in `docs/legal/`, never a search result, a summary or a memory."""
    for spec in CATALOG:
        if spec.name in hw.UNVERIFIED:
            continue
        assert spec.licence.source, f"{spec.name} states terms it cannot cite"
        assert "legal/" in spec.licence.source


def test_the_unverified_set_matches_what_is_actually_uncited():
    """Two representations of the same fact, deliberately: `UNVERIFIED` is the
    expectation and `source` is the evidence. Drift between them means somebody
    filled in a citation without reading, or pinned a model without one."""
    uncited = {spec.name for spec in CATALOG if not spec.licence.source}
    assert uncited == set(hw.UNVERIFIED)


def test_every_deliverable_model_has_a_licence_somebody_actually_read():
    """The state to hold: nothing reaches a client whose terms were inferred.

    This briefly failed on `llama3.2:3b`, whose archived PDF was a *valid file
    with no readable body* — which is worse than a missing one, because it
    passes a presence check and fails only when somebody opens it."""
    for spec in hw.DELIVERABLE:
        assert hw.licence_is_verified(spec.name), f"{spec.name} was never read"

    assert hw.licence_is_verified("some-model:70b") is False


def test_an_unvetted_model_reports_unknown_terms_rather_than_guessing():
    """A config that pins a model we never checked must not silently look
    approved — somebody has to read its licence by hand."""
    assert hw.licence_for("some-model:70b") is None
