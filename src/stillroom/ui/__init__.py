# Copyright (c) Jorge Ribeiro.
# Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
# Source-available, not open source: any purpose is permitted except
# providing a product that competes with this software.
"""The front-end core: the page the client's team opens, and its theming.

**This package is the deliverable's face, and it is built to be re-skinned per
engagement without touching code.** Everything visual resolves from CSS custom
properties that this module renders from the client's config (`[ui.theme]`), so
a brand match, a different element style or a calmer motion budget is a TOML
edit — the same "a client is a config, not a fork" rule the engine follows, applied to the part the buyer actually looks at.

Three constraints shaped what is here, and none of them is stylistic:

* **No build step.** The page is HTML, CSS and JavaScript served straight from
  disk — no Node in the image, no bundler, no `dist/` the client cannot read.
  A source-available licence invites the client to audit what they
  bought; a minified bundle answers that invitation with noise. It also means
  changing an accent colour never requires rebuilding anything.
* **Nothing is fetched from anywhere.** No CDN, no webfont, no analytics, no
  external image. This service exists because the client's documents must not
  leave the building; a page that phones a third party on every view would
  contradict that on the one screen they see it on.
* **The answer pane never receives HTML.** Model output is Markdown (see
  `api.AskResponse`), and it is rendered by building DOM nodes in
  `static/app.js` — never by assigning to `innerHTML`. A test asserts the
  absence, because that is the sort of thing a future convenience edit removes.

Locales are string files, so an extra interface language is a translated JSON
file rather than a code change.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # `config` imports this module's AVAILABLE_LOCALES at runtime.
    from stillroom.config import ClientConfig, ThemeConfig

UI_DIR = Path(__file__).parent
STATIC_DIR = UI_DIR / "static"
LOCALES_DIR = UI_DIR / "locales"

# The chrome languages that ship with the engine. A deployment uses one or
# more of them; adding another language means adding a file here.
AVAILABLE_LOCALES: tuple[str, ...] = tuple(
    sorted(path.stem for path in LOCALES_DIR.glob("*.json"))
)

# ---------------------------------------------------------------------------
# Palettes.
#
# A preset is a light pair and a dark pair, never one of the two: a client build
# is opened by whichever employees are sitting there, and half of them have
# their laptop in dark mode. Shipping only one appearance means shipping a
# product that looks broken to half the team on day one.
# ---------------------------------------------------------------------------

_PRESETS: dict[str, dict[str, dict[str, str]]] = {
    "slate": {
        "light": {
            "bg": "#f6f7f9",
            "panel": "#ffffff",
            "panel-2": "#eef1f5",
            "text": "#18202e",
            "muted": "#586274",
            "faint": "#8b95a6",
            "line": "#dfe4ea",
            "accent": "#3b5bdb",
            "accent-text": "#ffffff",
            "error": "#c92a2a",
        },
        "dark": {
            "bg": "#0f1115",
            "panel": "#171a21",
            "panel-2": "#1f2530",
            "text": "#e7e9ee",
            "muted": "#9aa3b2",
            "faint": "#7c8697",
            "line": "#2b3340",
            "accent": "#5b8def",
            "accent-text": "#0b0d11",
            "error": "#f87171",
        },
    },
    "warm": {
        "light": {
            "bg": "#faf7f2",
            "panel": "#ffffff",
            "panel-2": "#f2ece2",
            "text": "#2a2118",
            "muted": "#6b5d4c",
            "faint": "#9a8b78",
            "line": "#e6ddd0",
            "accent": "#b45309",
            "accent-text": "#ffffff",
            "error": "#b91c1c",
        },
        "dark": {
            "bg": "#15110c",
            "panel": "#1e1811",
            "panel-2": "#2a2119",
            "text": "#f0e8dc",
            "muted": "#b6a894",
            "faint": "#93856f",
            "line": "#3a2f22",
            "accent": "#f0a24a",
            "accent-text": "#15110c",
            "error": "#fca5a5",
        },
    },
    "forest": {
        "light": {
            "bg": "#f4f8f5",
            "panel": "#ffffff",
            "panel-2": "#e7f0e9",
            "text": "#152119",
            "muted": "#4c6154",
            "faint": "#7d9185",
            "line": "#d6e3d9",
            "accent": "#16794f",
            "accent-text": "#ffffff",
            "error": "#b02525",
        },
        "dark": {
            "bg": "#0c1310",
            "panel": "#131c17",
            "panel-2": "#1b271f",
            "text": "#e3ece6",
            "muted": "#94a99c",
            "faint": "#77897f",
            "line": "#24332a",
            "accent": "#4ec08a",
            "accent-text": "#0c1310",
            "error": "#f28b8b",
        },
    },
    "mono": {
        "light": {
            "bg": "#ffffff",
            "panel": "#ffffff",
            "panel-2": "#f1f1f1",
            "text": "#111111",
            "muted": "#5a5a5a",
            "faint": "#8a8a8a",
            "line": "#e0e0e0",
            "accent": "#111111",
            "accent-text": "#ffffff",
            "error": "#a30000",
        },
        "dark": {
            "bg": "#0d0d0d",
            "panel": "#151515",
            "panel-2": "#1e1e1e",
            "text": "#ededed",
            "muted": "#a0a0a0",
            "faint": "#7a7a7a",
            "line": "#2a2a2a",
            "accent": "#ededed",
            "accent-text": "#0d0d0d",
            "error": "#ef7676",
        },
    },
}

# Font stacks, all of them faces that already exist on the machine. There is no
# `@font-face` and no webfont URL anywhere in this product, on purpose: a
# webfont is the client's browser announcing to a third party, on every page
# view, which internal tool they are using.
_FONTS = {
    "system": 'system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif',
    "serif": 'Georgia, "Times New Roman", "Nimbus Roman", serif',
    "mono": '"JetBrains Mono", "Cascadia Mono", Consolas, "DejaVu Sans Mono", monospace',
    "rounded": '"Nunito", "Segoe UI Variable", "SF Pro Rounded", system-ui, sans-serif',
}

_SHAPES = {
    "soft": {"radius": "10px", "radius-sm": "6px", "radius-lg": "16px"},
    "sharp": {"radius": "0px", "radius-sm": "0px", "radius-lg": "0px"},
    "pill": {"radius": "999px", "radius-sm": "999px", "radius-lg": "24px"},
}

_DENSITIES = {
    "comfortable": {"gap": "14px", "pad": "14px 16px", "fs": "16px", "lh": "1.6"},
    "compact": {"gap": "8px", "pad": "8px 12px", "fs": "14.5px", "lh": "1.5"},
}

_EFFECTS = {
    # `--dur` is consumed by every transition in the stylesheet, so "none"
    # switches the whole interface to instant. That is an accessibility setting
    # and a performance one: this product runs on machines where the CPU is
    # already busy generating the answer.
    "full": {"dur": "160ms", "shadow": "0 6px 20px rgba(0,0,0,0.10)", "lift": "-1px"},
    "subtle": {"dur": "90ms", "shadow": "0 2px 8px rgba(0,0,0,0.07)", "lift": "0px"},
    "none": {"dur": "0ms", "shadow": "none", "lift": "0px"},
}


def _backdrop(background: str, mode: str) -> dict[str, str]:
    """The page backdrop, drawn in CSS. No image is ever fetched."""
    line = "rgba(127,127,127,0.10)" if mode == "light" else "rgba(255,255,255,0.05)"
    if background == "gradient":
        tint = "rgba(127,127,127,0.10)" if mode == "light" else "rgba(255,255,255,0.04)"
        return {
            "backdrop-image": f"radial-gradient(120% 80% at 50% 0%, {tint} 0%, transparent 60%)",
            "backdrop-size": "auto",
        }
    if background == "grid":
        return {
            "backdrop-image": (
                f"linear-gradient({line} 1px, transparent 1px), "
                f"linear-gradient(90deg, {line} 1px, transparent 1px)"
            ),
            "backdrop-size": "32px 32px",
        }
    if background == "dots":
        return {
            "backdrop-image": f"radial-gradient({line} 1px, transparent 1px)",
            "backdrop-size": "20px 20px",
        }
    return {"backdrop-image": "none", "backdrop-size": "auto"}


def _tokens(theme: "ThemeConfig", mode: str) -> dict[str, str]:
    """Every CSS variable for one appearance, resolved from the config."""
    palette = dict(_PRESETS[theme.preset][mode])

    override = theme.accent if mode == "light" else theme.accent_dark
    if override:
        palette["accent"] = override

    tokens = dict(palette)
    tokens.update(_SHAPES[theme.shape])
    tokens.update(_DENSITIES[theme.density])
    tokens.update(_EFFECTS[theme.effects])
    tokens.update(_backdrop(theme.background, mode))
    tokens["font"] = _FONTS[theme.font]
    tokens["font-mono"] = _FONTS["mono"]
    return tokens


def _block(selector: str, tokens: dict[str, str], indent: str = "") -> str:
    lines = [f"{indent}{selector} {{"]
    lines += [f"{indent}  --{name}: {value};" for name, value in tokens.items()]
    lines.append(f"{indent}}}")
    return "\n".join(lines)


def render_theme_css(config: "ClientConfig") -> str:
    """The client's theme as a stylesheet, generated from their config.

    Served at `/ui/theme.css` and loaded *before* the UI's own stylesheet, which
    reads these variables and hard-codes none of them.

    The appearance rules are ordered so the in-page toggle wins in **both**
    directions — a forced light theme on a dark-mode laptop is the case that a
    naive `prefers-color-scheme` block gets wrong, and it is exactly what an
    employee does when they are about to share their screen in a meeting.
    """
    theme = config.ui.theme
    light = _tokens(theme, "light")
    dark = _tokens(theme, "dark")

    parts = [
        "/* Generated from [ui.theme] in this client's config. Do not edit by",
        "   hand: it is rendered fresh on every request from the TOML file. */",
    ]

    if theme.mode == "dark":
        parts.append(_block(":root", dark))
    else:
        parts.append(_block(":root", light))
        if theme.mode == "system":
            parts.append(
                "@media (prefers-color-scheme: dark) {\n"
                + _block(":root", dark, indent="  ")
                + "\n}"
            )

    # The manual toggle, last, so it beats the ambient signal either way.
    parts.append(_block(':root[data-theme="light"]', light))
    parts.append(_block(':root[data-theme="dark"]', dark))

    return "\n\n".join(parts) + "\n"


@lru_cache(maxsize=None)
def load_locale(name: str) -> dict[str, str]:
    """One chrome-language string file."""
    return json.loads((LOCALES_DIR / f"{name}.json").read_text(encoding="utf-8"))


def read_asset(name: str) -> bytes:
    """A static file from this package, by exact name — never a client path."""
    return (STATIC_DIR / name).read_bytes()


def ui_payload(config: "ClientConfig") -> dict[str, Any]:
    """What the page needs before anyone has authenticated.

    ⚠️ **Deliberately excludes the curated questions.** They are the client's
    own most-asked questions — "what is our severance formula", "which supplier
    is on the exclusivity clause" — and the list is a readable summary of what
    the business worries about. The chrome is not sensitive; that list is, so it
    lives behind the key on `/api/suggestions`.
    """
    ui = config.ui
    return {
        "title": ui.title or f"{config.client} — document assistant",
        "intro": ui.intro,
        "languages": list(ui.languages),
        "strings": {name: load_locale(name) for name in ui.languages},
        "access": ui.access,
        "mode": ui.theme.mode,
        "has_logo": ui.theme.logo is not None,
        "suggestions": ui.suggestions,
        # ⚠️ **The page must never carry its own copy of this number**.
        # The meter shows how full the conversation is; the server is what
        # actually trims it. Two constants would drift the first time one is
        # tuned for a client, and the visible one would then be a lie about the
        # invisible one.
        "max_turns": config.conversation.max_turns if config.conversation.enabled else 0,
        # Stated in the interface rather than only in the runbook: the sentence
        # the whole purchase rests on should be visible where the answers are.
        # Sent as a token plus its detail, not as the finished English sentence,
        # so a bilingual build does not show it in the wrong language.
        "posture_kind": config.privacy_posture_kind(),
        "posture_detail": config.privacy_posture_detail(),
        "documents_stay_on_premises": config.documents_stay_on_premises(),
    }
