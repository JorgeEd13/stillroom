"""The third-party notices that travel with the base image.

This document exists because of a change in the DELIVERY MODEL, not in the code.
Until the base became a tarball, the client's own machine fetched every library
and the embedding weights, and we redistributed nothing. A tarball redistributes
all of it, and MIT, BSD and Apache-2.0 permit that only with their notices
attached.

The failure mode worth testing for is the same one `stillroom bom` guards: a
document that looks complete and carries nothing.
"""

from __future__ import annotations

from stillroom import notices


def _notice(name="thing", texts=(("LICENSE", "MIT License\nCopyright (c) X"),)):
    return notices.Notice(name=name, version="1.0", licence="MIT", texts=tuple(texts))


def test_licence_files_are_recognised_by_name_whatever_the_extension():
    for name in ("LICENSE", "LICENCE.txt", "COPYING.gpl", "NOTICE", "license.md"):
        assert notices._is_licence_file(name), name


def test_a_readme_is_not_a_licence():
    """Apache-2.0 §4(d) makes NOTICE its own obligation, so the net is wide —
    wide enough that it has to stop somewhere sensible."""
    for name in ("README.md", "RECORD", "METADATA", "top_level.txt"):
        assert not notices._is_licence_file(name)


def test_the_notices_say_the_embedding_MODEL_IS_NOT_IN_THE_IMAGE():
    """⚠️ **The inverse of what this test used to assert**.

    It used to require a section describing the embedding weights *inside* the
    image, because leg 3's tarball made us their distributor. `bge-m3`
    is pulled by the client's own Ollama, so it is not in the image and MIT's
    notice condition is not triggered by us.

    A notices file still carrying that section would describe a model that is
    not in the artifact it documents — handed over, at delivery, as a licence
    document. So the section is replaced rather than deleted: silence would read
    as an omission to anybody auditing the image against the bill of materials."""
    text = notices.render([_notice()])

    assert "not in this image" in text.lower()
    assert notices.DEFAULT_EMBEDDING_NAME in text
    assert "MIT" in text
    # The old claims must be gone, not merely reworded around.
    assert "included in the image" not in text
    assert "onnx-minilm" not in text.lower()


def test_a_library_with_no_licence_text_is_listed_rather_than_dropped():
    """Some wheels declare a licence and ship no copy of it. Silently omitting
    those turns "we could not find it" into "there was nothing to find"."""
    text = notices.render([_notice(), _notice(name="quiet", texts=())])

    assert "quiet" in text
    assert "carried no licence file" in text


def test_the_same_licence_recorded_twice_is_read_once(tmp_path):
    """PEP 639 moved licence files into `.dist-info/licenses/` and wheels often
    record both locations. Printing one licence twice makes a document look
    padded rather than thorough, so the dedup is by CONTENT."""

    class _Path:
        def __init__(self, path, parts):
            self._path, self.parts = path, parts

        @property
        def name(self):
            return self.parts[-1]

        def locate(self):
            return self._path

    body = "MIT License\nCopyright (c) X"
    (tmp_path / "a").write_text(body, encoding="utf-8")
    (tmp_path / "b").write_text(body, encoding="utf-8")
    (tmp_path / "c").write_text("BSD, different text", encoding="utf-8")

    class _Dist:
        files = [
            _Path(tmp_path / "a", ("thing-1.0.dist-info", "LICENSE")),
            _Path(tmp_path / "b", ("thing-1.0.dist-info", "licenses", "LICENSE")),
            _Path(tmp_path / "c", ("thing-1.0.dist-info", "NOTICE")),
            # Not in the .dist-info at all, so not ours to reproduce.
            _Path(tmp_path / "c", ("thing", "vendored", "LICENSE")),
        ]

    texts = notices._texts_of(_Dist())

    assert [body, "BSD, different text"] == sorted((text for _, text in texts), reverse=True)


def test_it_refuses_rather_than_writing_an_empty_document(tmp_path, monkeypatch, capsys):
    """⚠️ Run in a source checkout, nothing is installed and nothing can be read.

    Writing the file anyway would produce a confident, complete-looking document
    with no licence in it — and an obligation that *looks* discharged is worse
    than one that visibly is not. Same posture as `stillroom bom`.
    """
    monkeypatch.setattr(notices, "collect", lambda: [_notice(texts=())])
    output = tmp_path / "THIRD-PARTY-NOTICES.md"

    assert notices.main(["--output", str(output)]) == 1
    assert not output.exists()
    assert "not in a source checkout" in capsys.readouterr().err


def test_it_writes_the_document_when_there_is_something_to_write(tmp_path, monkeypatch):
    monkeypatch.setattr(notices, "collect", lambda: [_notice()])
    output = tmp_path / "THIRD-PARTY-NOTICES.md"

    assert notices.main(["--output", str(output)]) == 0
    assert "MIT License" in output.read_text(encoding="utf-8")
