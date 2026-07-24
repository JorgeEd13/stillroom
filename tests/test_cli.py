"""The CLI, which is both my build loop and the client's re-ingest script.

The exit codes matter more than they look. `bake` returning non-zero on a
question the corpus cannot answer is what makes that finding impossible to
scroll past during a build — and the client's own re-ingest script inherits the
same behaviour.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from stillroom import cli
from stillroom.config import ClientConfig
from stillroom.index.embeddings import DeterministicEmbeddingFunction


@pytest.fixture
def config_file(tmp_path: Path, config: ClientConfig, monkeypatch) -> Path:
    """A config on disk, with the deterministic embedder forced everywhere.

    The CLI builds its own embedder, so the test substitutes the offline one
    rather than letting the suite reach the client's Ollama (`ingest`
    now needs it, and `tests/` runs offline by assertion, not by intention).
    """
    import stillroom.engine
    import stillroom.pipeline

    embedding = DeterministicEmbeddingFunction()
    monkeypatch.setattr(stillroom.pipeline, "embedding_function_for", lambda config: embedding)
    monkeypatch.setattr(stillroom.engine, "embedding_function_for", lambda config: embedding)

    path = tmp_path / "client.toml"
    path.write_text(
        f'client = "Test Ltd"\n'
        f'index_path = "{config.index_path}"\n'
        f"[corpus]\npath = \"{config.corpus.path}\"\n"
        f"[retrieval]\nk = 3\nmin_similarity = 0.20\n"
        f'[answer_cache]\ncurated = ["What is the refund window?"]\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def stub_model(monkeypatch, fake_model):
    import stillroom.engine

    monkeypatch.setattr(stillroom.engine, "build_chat_model", lambda config: fake_model)
    monkeypatch.setattr(stillroom.engine, "model_label", lambda config: "stub")
    monkeypatch.setattr(cli, "model_label", lambda config: "stub")
    return fake_model


def test_the_example_config_is_valid(tmp_path: Path):
    """It is copied into every engagement; a broken one wastes the first hour."""
    example = Path(__file__).resolve().parents[1] / "configs" / "example.toml"

    config = ClientConfig.model_validate(tomllib.loads(example.read_text(encoding="utf-8")))

    assert config.uses_local_model()
    assert config.answer_cache.curated


def test_ingest_reports_what_it_indexed(config_file: Path, stub_model, capsys):
    assert cli.main(["ingest", "--config", str(config_file)]) == 0

    out = capsys.readouterr().out
    assert "Indexed 2 documents" in out
    assert "Corpus fingerprint:" in out


def test_ingest_names_the_files_it_skipped(config_file: Path, config, stub_model, capsys):
    (Path(config.corpus.path) / "scan.pdf").write_bytes(b"not a pdf")

    cli.main(["ingest", "--config", str(config_file)])

    out = capsys.readouterr().out
    # The client is told which document is missing from their chatbot, and why.
    assert "Skipped 1 file(s)" in out
    assert "scan.pdf" in out


def test_ask_prints_the_answer_and_its_sources(config_file: Path, stub_model, capsys):
    cli.main(["ingest", "--config", str(config_file)])
    capsys.readouterr()

    assert cli.main(["ask", "--config", str(config_file), "What is the refund window?"]) == 0

    out = capsys.readouterr().out
    assert stub_model.reply in out
    assert "handbook.md" in out
    assert "(model)" in out


def test_ask_before_ingest_exits_with_a_clear_error(config_file: Path, stub_model, capsys):
    code = cli.main(["ask", "--config", str(config_file), "anything?"])

    assert code == 2
    assert "Run an ingest first" in capsys.readouterr().err


def test_bake_succeeds_when_every_curated_question_is_answerable(
    config_file: Path, stub_model, capsys
):
    cli.main(["ingest", "--config", str(config_file)])
    capsys.readouterr()

    assert cli.main(["bake", "--config", str(config_file)]) == 0
    assert "Baked 1 instant answer(s)" in capsys.readouterr().out


def test_bake_fails_loudly_when_the_corpus_cannot_answer_a_question(
    config_file: Path, stub_model, capsys
):
    """A non-zero exit is what stops this being scrolled past during a build."""
    cli.main(["ingest", "--config", str(config_file)])
    capsys.readouterr()

    code = cli.main(
        ["bake", "--config", str(config_file), "--question", "What is our space travel policy?"]
    )

    assert code == 1
    out = capsys.readouterr().out
    assert "found NOTHING" in out
    assert "space travel" in out


def test_a_missing_config_exits_without_a_traceback(tmp_path: Path, capsys):
    code = cli.main(["ingest", "--config", str(tmp_path / "nope.toml")])

    assert code == 1
    assert "Error:" in capsys.readouterr().err
