"""The web page: what it serves, what it refuses, and what it must never contain.

Three of the tests here assert **absences**, which is unusual enough to say why.
The UI's risky properties are all things that are true until somebody adds one
convenient line: an external font, an `innerHTML`, a CORS header. None of those
break a feature when introduced, so nothing else in the suite would fail. These
tests are the only thing standing between a future edit and a product whose page
quietly contacts a third party or renders model output as HTML.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from stillroom import ui as ui_assets
from stillroom.api import create_app
from stillroom.config import ClientConfig
from stillroom.engine import Answer

API_KEY = "test-key-123"
CURATED = ("What is the refund window?", "How much notice must I give?")


def _code_only(source: str) -> str:
    """The file with its comments removed.

    Needed because these tests scan for forbidden strings, and the files
    explain — in comments, at the point it matters — exactly which strings are
    forbidden and why. Scanning the raw text finds the warning label rather than
    the hazard, which is worse than not scanning at all: it fails when the
    documentation is good.
    """
    without_blocks = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", "", without_blocks, flags=re.MULTILINE)


class StubEngine:
    fingerprint = "abc123"

    def ask(self, question: str) -> Answer:
        return Answer(text="30 days [1].", citations=[], served_by="model")

    async def astream(self, question: str):
        yield {"type": "answer", "reply": "30 days [1].", "citations": []}


def _app(**ui: Any):
    config = ClientConfig.model_validate(
        {
            "client": "Acme",
            "api_key": API_KEY,
            "corpus": {"path": "/tmp/docs"},
            "answer_cache": {"curated": CURATED},
            "ui": ui,
        }
    )
    return create_app(config, engine_builder=lambda _: StubEngine())


def _client(**ui: Any) -> TestClient:
    return TestClient(_app(**ui), base_url="http://localhost")


@pytest.fixture
def client() -> Any:
    with _client() as test_client:
        yield test_client


# --------------------------------------------------------------- the page ---


def test_the_page_is_served_at_the_root(client: TestClient):
    """The whole handover instruction is "open this address"; this is it."""
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "/ui/app.js" in response.text


def test_the_page_and_its_assets_are_the_only_files_served(client: TestClient):
    for name in ("app.js", "app.css", "favicon.svg"):
        assert client.get(f"/ui/{name}").status_code == 200

    # Not a directory mount: a file that appears in the folder later is not
    # automatically published, and neither is anything above it.
    assert client.get("/ui/__init__.py").status_code == 404
    assert client.get("/ui/../config.py").status_code in (307, 404)


def test_a_headless_build_has_no_page_at_all(client: TestClient):
    with _client(enabled=False) as headless:
        assert headless.get("/").status_code == 404
        assert headless.get("/ui/app.js").status_code == 404
        # The API it exists for still works.
        assert headless.get("/api/health").status_code == 200


# ------------------------------------------------------- what it must not do ---


def test_the_ui_never_reaches_a_third_party():
    """The claim the client paid for, asserted against the page they look at.

    A CDN script, a webfont or a tracking pixel would mean the browser telling
    somebody else, on every single view, which internal tool this company runs —
    on the one screen where the client could have seen it happening.
    """
    for path in sorted(ui_assets.STATIC_DIR.iterdir()):
        source = _code_only(path.read_text(encoding="utf-8"))
        # The SVG namespace is an identifier, not an address: nothing is fetched
        # from it. It is the one string here that looks like a URL and is not.
        source = source.replace('xmlns="http://www.w3.org/2000/svg"', "")

        for offender in ("http://", "https://", "//cdn", "@import", "@font-face"):
            assert offender not in source, f"{path.name} reaches outside: {offender}"


def test_the_answer_pane_is_built_from_dom_nodes_not_html():
    """Landmine 3's fix, and the guard on it.

    Answers are Markdown written by a model about documents somebody else
    supplied. Rendering them means building elements; the moment any of these
    appear, the page is assigning attacker-influenced text to a parser.
    """
    source = _code_only((ui_assets.STATIC_DIR / "app.js").read_text(encoding="utf-8"))

    for offender in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
    ):
        assert offender not in source, f"app.js uses {offender}"


def test_no_cors_headers_are_ever_sent(client: TestClient):
    """Without these, another site cannot read this service's answers.

    Adding CORS "so the front end can be developed separately" would undo the
    Host check as well — a rebound origin would then be allowed to read the
    response it provoked.
    """
    response = client.get("/api/health", headers={"Origin": "https://example.com"})

    assert "access-control-allow-origin" not in {k.lower() for k in response.headers}


def test_an_unknown_host_header_is_refused():
    """DNS rebinding: a public page can resolve its own name to 127.0.0.1 and
    then talk to this service from inside an employee's browser."""
    with TestClient(_app(), base_url="http://evil.example.com") as rebound:
        assert rebound.get("/api/health").status_code == 400


def test_a_client_hostname_can_be_allowed_explicitly():
    with TestClient(
        _app(allowed_hosts=["assistant.acme.internal"]),
        base_url="http://assistant.acme.internal",
    ) as proxied:
        assert proxied.get("/api/health").status_code == 200


# ----------------------------------------------------------- what it needs ---


def test_the_public_payload_carries_chrome_but_never_the_curated_questions(
    client: TestClient,
):
    """`/api/ui` is open by necessity — the key prompt has to be rendered before
    there is a key. So the client's own most-asked questions must not be in it:
    read together, that list describes what the business worries about."""
    body = client.get("/api/ui").json()

    assert body["title"] == "Acme — document assistant"
    assert body["languages"] == ["en"]
    assert body["strings"]["en"]["send"]

    serialised = json.dumps(body)
    for question in CURATED:
        assert question not in serialised


def test_key_mode_requires_the_key_on_data_routes():
    # `key` mode is the shared-office-network deployment: the team pastes the key
    # from the config once per browser. Without it, the data routes must refuse.
    with _client(access="key") as keyed:
        assert keyed.get("/api/suggestions").status_code == 401

        response = keyed.get("/api/suggestions", headers={"X-API-Key": API_KEY})
        assert response.status_code == 200
        assert response.json()["questions"] == list(CURATED)


def test_open_mode_serves_data_routes_without_a_key():
    # The loopback-boundary rule: in `open` mode — the single-machine default — the boundary
    # is the loopback binding, and the page sends no X-API-Key by design.
    # Requiring one here bounced an open-mode client to an access-key gate on
    # their very first question, so open mode could not answer at all. The data
    # routes must serve keyless in open mode. The default `client` fixture is
    # open, which is why the old outcome test (asserting 401 here) passed while
    # encoding the bug.
    with _client(access="open") as opened:
        response = opened.get("/api/suggestions")
        assert response.status_code == 200
        assert response.json()["questions"] == list(CURATED)


def test_suggestions_can_be_switched_off_for_a_client_who_wants_a_bare_page():
    with _client(suggestions=False) as bare:
        response = bare.get("/api/suggestions", headers={"X-API-Key": API_KEY})
        assert response.json()["questions"] == []


# ---------------------------------------------------------------- theming ---


def test_the_theme_is_generated_from_the_config_not_stored():
    with _client(theme={"preset": "forest", "accent": "#123456", "shape": "sharp"}) as c:
        css = c.get("/ui/theme.css").text

    assert "--accent: #123456;" in css
    assert "--radius: 0px;" in css
    # Both appearances always ship: half the team has a dark-mode laptop.
    assert ':root[data-theme="dark"]' in css
    assert ':root[data-theme="light"]' in css


def test_the_manual_toggle_is_written_after_the_ambient_rule():
    """A forced light theme on a dark-mode laptop is the case a naive
    `prefers-color-scheme` block gets wrong — and it is what somebody does right
    before sharing their screen in a meeting."""
    with _client() as c:
        css = c.get("/ui/theme.css").text

    assert css.index("prefers-color-scheme") < css.index(':root[data-theme="light"]')


def test_a_fixed_appearance_does_not_follow_the_operating_system():
    with _client(theme={"mode": "dark"}) as c:
        css = c.get("/ui/theme.css").text

    assert "prefers-color-scheme" not in css


def test_a_colour_that_is_actually_css_is_rejected():
    """Theme values are interpolated into a stylesheet, so they are matched
    rather than trusted."""
    with pytest.raises(ValueError, match="hex colour"):
        ClientConfig.model_validate(
            {
                "client": "Acme",
                "corpus": {"path": "/tmp/docs"},
                "ui": {"theme": {"accent": "red; } body { display: none"}},
            }
        )


def test_the_stylesheet_hard_codes_no_colour():
    """The rule that makes a re-skin a TOML edit: every colour in the UI's own
    stylesheet is a variable rendered from the client's config. One hard-coded
    `#fff` here is the pixel that stays wrong in every future client's build."""
    css = _code_only((ui_assets.STATIC_DIR / "app.css").read_text(encoding="utf-8"))

    assert "#" not in css, "a literal colour is in app.css"
    # The documented exception: overlays that must sit on top of any palette.
    assert css.count("rgba(") == 1


def test_a_client_stylesheet_is_served_and_loads_last(tmp_path: Path):
    """The escape hatch for anything the tokens do not cover. It is CSS and
    never JS: a brand file can restyle every element without being able to touch
    what the page does with the documents."""
    config_path = tmp_path / "client.toml"
    (tmp_path / "brand.css").write_text(".title { letter-spacing: 0.2em; }")
    config_path.write_text(
        'client = "Acme"\n'
        f'index_path = "{tmp_path / "index"}"\n'
        "[corpus]\n"
        f'path = "{tmp_path}"\n'
        "[ui.theme]\n"
        'custom_css = "brand.css"\n'
    )
    config = ClientConfig.load(config_path)

    with TestClient(
        create_app(config, engine_builder=lambda _: StubEngine()),
        base_url="http://localhost",
    ) as branded:
        assert "letter-spacing" in branded.get("/ui/custom.css").text
        page = branded.get("/").text
        assert page.index("/ui/theme.css") < page.index("/ui/custom.css")


def test_a_missing_brand_asset_fails_the_build_rather_than_the_handover(tmp_path: Path):
    config_path = tmp_path / "client.toml"
    config_path.write_text(
        'client = "Acme"\n'
        f'index_path = "{tmp_path / "index"}"\n'
        "[corpus]\n"
        f'path = "{tmp_path}"\n'
        "[ui.theme]\n"
        'logo = "logo.svg"\n'
    )
    config = ClientConfig.load(config_path)

    with pytest.raises(FileNotFoundError, match="logo.svg"):
        create_app(config, engine_builder=lambda _: StubEngine())


# --------------------------------------------------------------- languages ---


def test_every_locale_carries_every_string():
    """A missing key renders as its own name — `shortcutStop` on screen, in
    front of the client, at handover."""
    reference = ui_assets.load_locale("en")

    for name in ui_assets.AVAILABLE_LOCALES:
        strings = ui_assets.load_locale(name)
        assert set(strings) == set(reference), f"{name}.json has drifted from en.json"
        assert all(value.strip() for value in strings.values())


def test_a_single_language_build_gets_no_switcher(client: TestClient):
    body = client.get("/api/ui").json()

    assert body["languages"] == ["en"]
    assert set(body["strings"]) == {"en"}


def test_a_two_language_build_defaults_to_the_first():
    with _client(languages=["pt-BR", "en"]) as bilingual:
        body = bilingual.get("/api/ui").json()

    assert body["languages"] == ["pt-BR", "en"]
    assert body["strings"]["pt-BR"]["send"] == "Enviar"


def test_an_unknown_language_is_refused_with_the_list_of_real_ones():
    with pytest.raises(ValueError, match="Unknown UI locale"):
        ClientConfig.model_validate(
            {
                "client": "Acme",
                "corpus": {"path": "/tmp/docs"},
                "ui": {"languages": ["fr"]},
            }
        )


def test_the_privacy_sentence_is_translated_not_shipped_finished(client: TestClient):
    """The claim the purchase rests on, in the reader's language.

    Sent as a token plus a detail so the page composes it from its own string
    file — otherwise a Portuguese build shows the single most important sentence
    on the page in English, in the header, above every answer. Found by looking
    at a screenshot of the running bilingual build, not by reading the code.
    """
    body = client.get("/api/ui").json()

    assert body["posture_kind"] == "local"
    assert body["posture_detail"] is None
    # The finished English sentence is not what the page renders from.
    assert "privacy_posture" not in body

    for name in ui_assets.AVAILABLE_LOCALES:
        strings = ui_assets.load_locale(name)
        assert "{detail}" in strings["postureOnPremises"]
        assert "{detail}" in strings["postureThirdParty"]


def test_a_byok_build_says_so_in_the_header():
    """A client who bought the honest exception is told, on every page view,
    that this one sends documents out. It is the posture they agreed to, and
    hiding it in a config file would make the header a lie by omission."""
    config = ClientConfig.model_validate(
        {
            "client": "Acme",
            "corpus": {"path": "/tmp/docs"},
            "model": {
                "kind": "byok",
                "provider": "anthropic",
                "name": "claude-sonnet-5",
                "api_key_env": "ANTHROPIC_API_KEY",
            },
        }
    )
    payload = ui_assets.ui_payload(config)

    assert payload["posture_kind"] == "third_party"
    assert payload["posture_detail"] == "anthropic"
    assert payload["documents_stay_on_premises"] is False


def test_the_answers_language_boundary_is_stated_in_every_locale():
    """Translating the chrome must not imply translated answers. Answers come
    back in the language the client's documents are written in."""
    for name in ui_assets.AVAILABLE_LOCALES:
        note = ui_assets.load_locale(name)["answersLanguageNote"]
        assert len(note) > 40
