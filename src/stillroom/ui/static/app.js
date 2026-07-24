/*
 * Copyright (c) Jorge Ribeiro.
 * Licensed under PolyForm Shield 1.0.0 - see LICENSE.md.
 * Source-available, not open source: any purpose is permitted except
 * providing a product that competes with this software.
 */
/*
  The client's assistant, in one file, with no dependencies and no build step.

  ⚠️ THE ONE RULE HERE: **model output never becomes HTML.** There is no
  `innerHTML`, no `insertAdjacentHTML`, no `document.write`, no `eval` and no
  `new Function` in this file, and a test asserts their absence. Answers arrive
  as Markdown (see `api.AskResponse`) and are turned into real elements by
  `renderMarkdown` below, one `createElement`/`textContent` at a time. That is
  what makes it safe to render text that a model wrote about documents somebody
  else supplied — and it is what stops a UI showing
  `**bold**` and `##` on screen as literal syntax.

  Everything the client can change (colours, shape, density, motion, background,
  logo) lives in their config and arrives as CSS variables in /ui/theme.css.
  Nothing in this file decides what the product looks like.
*/

(function () {
  "use strict";

  var KEY_STORE = "stillroom.key";
  var LANG_STORE = "stillroom.lang";
  var THEME_STORE = "stillroom.theme";

  var el = function (id) {
    return document.getElementById(id);
  };

  var state = {
    ui: null, // the /api/ui payload
    lang: "en",
    strings: {},
    key: null, // only ever set in access = "key" mode
    busy: false,
    controller: null, // AbortController for the running stream
    suggestions: [],
    /*
      The conversation, and it lives HERE rather than on the server.

      Two consequences worth stating: the service stays stateless, so a
      transcript of the client's most confidential questions is never written to
      a volume that outlives the tab — and "New chat" is honest, because
      dropping this array is genuinely all there is to clear.

      Only completed exchanges are pushed. A question that errored, was stopped,
      or came back empty is not part of the conversation, and feeding a failed
      turn back as context would teach the model to repeat it.
    */
    history: [],
  };

  /*
    Kept small on this side too: the server trims again, but sending fifty turns
    so the server can throw away forty-four wastes the client's own bandwidth and
    memory on the machine that is already running the model.

    ⚠️ The value comes from `/api/ui`, never from a constant here. The
    server is what actually trims; a second copy of the number would drift the
    first time it is tuned for a client, and the meter would then be showing a
    limit that is not the limit. This is only the fallback for the moment before
    that payload has arrived.
  */
  var maxTurns = 6;

  /*
    Heat starts at half-full. Below that the conversation is comfortable and a
    colouring bar would be noise; above it the user is approaching the point
    where their oldest exchange is silently dropped, which is the only moment
    the meter exists to signal.
  */
  function updateBudget() {
    var meter = el("budget");
    if (!meter) return;
    if (!maxTurns) {
      meter.hidden = true; /* memory switched off for this build */
      return;
    }

    var ratio = Math.min(1, state.history.length / maxTurns);
    meter.hidden = state.history.length === 0;
    el("budget-fill").style.setProperty("--budget-fill", Math.round(ratio * 100) + "%");
    el("budget-fill").style.setProperty(
      "--budget-heat",
      Math.round(Math.max(0, (ratio - 0.5) / 0.5) * 100) + "%"
    );
    el("budget-label").textContent = t("budgetLabel")
      .replace("{n}", String(state.history.length))
      .replace("{max}", String(maxTurns));
  }

  var t = function (name) {
    return (state.strings && state.strings[name]) || name;
  };

  // ------------------------------------------------------------- markdown ---

  /*
    A deliberately small Markdown subset: headings, paragraphs, bold, italic,
    strikethrough, inline code, fenced code, blockquotes, ordered and unordered
    lists, pipe tables, horizontal rules, and the `[1]` citation markers.

    Small is the point. Every construct here is one a local model actually emits
    when answering a question about a policy document; anything rarer is better
    shown as plain text than parsed by more code than the feature is worth.
  */

  var INLINE = /(`[^`]+`|\*\*[^*]+\*\*|__[^_]+__|~~[^~]+~~|\*[^*\n]+\*|_[^_\n]+_|\[[^\]\n]*\]\([^)\s]+\)|\[\d+\])/;

  function appendInline(parent, text) {
    var parts = text.split(INLINE);
    for (var i = 0; i < parts.length; i++) {
      var piece = parts[i];
      if (!piece) continue;

      if (piece.length > 1 && piece[0] === "`" && piece[piece.length - 1] === "`") {
        parent.appendChild(tag("code", piece.slice(1, -1)));
      } else if (piece.slice(0, 2) === "**" && piece.slice(-2) === "**") {
        parent.appendChild(tag("strong", piece.slice(2, -2)));
      } else if (piece.slice(0, 2) === "__" && piece.slice(-2) === "__") {
        parent.appendChild(tag("strong", piece.slice(2, -2)));
      } else if (piece.slice(0, 2) === "~~" && piece.slice(-2) === "~~") {
        parent.appendChild(tag("s", piece.slice(2, -2)));
      } else if (
        (piece[0] === "*" && piece[piece.length - 1] === "*") ||
        (piece[0] === "_" && piece[piece.length - 1] === "_")
      ) {
        parent.appendChild(tag("em", piece.slice(1, -1)));
      } else if (/^\[\d+\]$/.test(piece)) {
        parent.appendChild(citation(piece.slice(1, -1)));
      } else if (piece[0] === "[") {
        appendLink(parent, piece);
      } else {
        parent.appendChild(document.createTextNode(piece));
      }
    }
  }

  /*
    A link in an answer is rendered as TEXT, never as a clickable anchor.

    The model is writing about the client's own documents, and a URL it produces
    is either copied from one of them or invented. Neither case earns a
    click-through: an invented address turns a grounded answer into a navigation
    vector, and this page is an internal tool over confidential files. The
    address is shown in full so a person can decide for themselves.
  */
  function appendLink(parent, piece) {
    var split = piece.indexOf("](");
    var label = piece.slice(1, split);
    var href = piece.slice(split + 2, -1);
    parent.appendChild(tag("span", label, "link-text"));
    parent.appendChild(tag("span", " (" + href + ")", "link-href"));
  }

  function citation(number) {
    var chip = tag("span", "[" + number + "]", "cite");
    chip.dataset.n = number;
    chip.setAttribute("role", "button");
    chip.tabIndex = 0;
    return chip;
  }

  function tag(name, text, className) {
    var node = document.createElement(name);
    if (text !== undefined && text !== null) node.textContent = text;
    if (className) node.className = className;
    return node;
  }

  function renderMarkdown(source, target) {
    var lines = String(source == null ? "" : source).split("\n");
    var blocks = document.createDocumentFragment();
    var i = 0;

    while (i < lines.length) {
      var line = lines[i];

      if (/^```/.test(line)) {
        var code = [];
        i++;
        while (i < lines.length && !/^```/.test(lines[i])) code.push(lines[i++]);
        i++; // closing fence
        var pre = document.createElement("pre");
        pre.appendChild(tag("code", code.join("\n")));
        blocks.appendChild(pre);
        continue;
      }

      if (/^\s*$/.test(line)) {
        i++;
        continue;
      }

      if (/^(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
        blocks.appendChild(document.createElement("hr"));
        i++;
        continue;
      }

      var heading = /^(#{1,6})\s+(.*)$/.exec(line);
      if (heading) {
        // Levels are collapsed into h3/h4: an answer sits inside the page's own
        // heading hierarchy, and a model writing `#` does not know that.
        var node = document.createElement(heading[1].length <= 2 ? "h3" : "h4");
        appendInline(node, heading[2]);
        blocks.appendChild(node);
        i++;
        continue;
      }

      if (/^\s*>\s?/.test(line)) {
        var quoted = [];
        while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
          quoted.push(lines[i].replace(/^\s*>\s?/, ""));
          i++;
        }
        var quote = document.createElement("blockquote");
        renderMarkdown(quoted.join("\n"), quote);
        blocks.appendChild(quote);
        continue;
      }

      if (/^\s*\|.*\|\s*$/.test(line) && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1] || "")) {
        var rows = [];
        while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(lines[i++]);
        blocks.appendChild(buildTable(rows));
        continue;
      }

      var bullet = /^\s*([-*+]|\d+[.)])\s+/.exec(line);
      if (bullet) {
        var ordered = /\d/.test(bullet[1]);
        var list = document.createElement(ordered ? "ol" : "ul");
        while (i < lines.length) {
          var item = /^\s*([-*+]|\d+[.)])\s+(.*)$/.exec(lines[i]);
          if (!item) break;
          if (/\d/.test(item[1]) !== ordered) break;
          var li = document.createElement("li");
          appendInline(li, item[2]);
          list.appendChild(li);
          i++;
        }
        blocks.appendChild(list);
        continue;
      }

      // Everything else is a paragraph, gathering following lines until a break.
      //
      // ⚠️ The first line is taken unconditionally, and that is load-bearing:
      // this branch is reached by lines that *look* like a block start but were
      // not claimed by one — a `| a |` whose separator row has not streamed in
      // yet is the real case. Testing the condition before consuming anything
      // let the loop make no progress and spin forever, which in a browser is a
      // frozen tab in front of the client, mid-answer.
      var paragraph = [lines[i++]];
      while (i < lines.length && !/^\s*$/.test(lines[i]) && !isBlockStart(lines[i])) {
        paragraph.push(lines[i++]);
      }
      var p = document.createElement("p");
      appendInline(p, paragraph.join(" "));
      blocks.appendChild(p);
    }

    target.replaceChildren(blocks);
  }

  function isBlockStart(line) {
    return (
      /^```/.test(line) ||
      /^#{1,6}\s/.test(line) ||
      /^\s*>\s?/.test(line) ||
      /^\s*([-*+]|\d+[.)])\s+/.test(line) ||
      /^\s*\|.*\|\s*$/.test(line)
    );
  }

  function buildTable(rows) {
    var cells = function (row) {
      return row
        .trim()
        .replace(/^\||\|$/g, "")
        .split("|")
        .map(function (cell) {
          return cell.trim();
        });
    };

    var table = document.createElement("table");
    var head = document.createElement("thead");
    var headRow = document.createElement("tr");
    cells(rows[0]).forEach(function (text) {
      var th = document.createElement("th");
      appendInline(th, text);
      headRow.appendChild(th);
    });
    head.appendChild(headRow);
    table.appendChild(head);

    var body = document.createElement("tbody");
    rows.slice(2).forEach(function (row) {
      var tr = document.createElement("tr");
      cells(row).forEach(function (text) {
        var td = document.createElement("td");
        appendInline(td, text);
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });
    table.appendChild(body);
    return table;
  }

  // ------------------------------------------------------------ transport ---

  function headers() {
    var head = { "Content-Type": "application/json" };
    if (state.key) head["X-API-Key"] = state.key;
    return head;
  }

  function onUnauthorized() {
    // The stored key stopped working — it was rotated, or was never right.
    // Clearing it is the only way the gate can come back.
    try {
      localStorage.removeItem(KEY_STORE);
    } catch (err) {
      /* private browsing: nothing to clear */
    }
    state.key = null;
    showGate(t("keyError"));
  }

  /*
    Server-sent events over POST. The stream is what turns a slow local model
    into visible progress rather than a page that looks frozen — the honest
    trade-off named up front, made legible.
  */
  function streamAsk(question, onEvent, signal) {
    return fetch("/api/ask/stream", {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({ question: question, history: state.history }),
      signal: signal,
    }).then(function (response) {
      if (response.status === 401) {
        onUnauthorized();
        throw new Error("unauthorized");
      }
      if (!response.ok || !response.body) {
        throw new Error("HTTP " + response.status);
      }

      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";

      function pump() {
        return reader.read().then(function (result) {
          if (result.done) return;
          buffer += decoder.decode(result.value, { stream: true });

          var separator;
          while ((separator = buffer.indexOf("\n\n")) !== -1) {
            var frame = buffer.slice(0, separator);
            buffer = buffer.slice(separator + 2);
            var line = frame.split("\n").find(function (candidate) {
              return candidate.indexOf("data: ") === 0;
            });
            if (!line) continue;
            var payload = line.slice(6);
            if (payload === "[DONE]") return;
            try {
              onEvent(JSON.parse(payload));
            } catch (err) {
              /* a malformed frame is skipped rather than breaking the stream */
            }
          }
          return pump();
        });
      }

      return pump();
    });
  }

  // ----------------------------------------------------------------- chat ---

  function turn(role) {
    var wrapper = tag("div", null, "turn " + role);
    wrapper.appendChild(tag("div", role === "user" ? t("you") : t("assistant"), "who"));
    var bubble = tag("div", null, "bubble");
    wrapper.appendChild(bubble);
    el("chat").appendChild(wrapper);
    return bubble;
  }

  function scrollToEnd() {
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  }

  function renderSources(bubble, citations) {
    if (!citations || !citations.length) return;

    var box = tag("div", null, "sources");
    box.appendChild(tag("h3", t("sources")));
    citations.forEach(function (citation, index) {
      var row = tag("div", null, "source");
      row.dataset.n = String(index + 1);
      row.appendChild(tag("span", "[" + (index + 1) + "]", "source-n"));
      var label = citation.heading
        ? citation.source + " — " + citation.heading
        : citation.source;
      row.appendChild(tag("span", label));
      box.appendChild(row);
    });
    bubble.appendChild(box);

    // Clicking a [1] in the answer highlights the passage it came from. The
    // citation IS the product claim, so it is worth making it inspectable.
    bubble.querySelectorAll(".cite").forEach(function (chip) {
      chip.addEventListener("click", function () {
        highlight(bubble, chip.dataset.n);
      });
      chip.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          highlight(bubble, chip.dataset.n);
        }
      });
    });
  }

  function highlight(bubble, number) {
    bubble.querySelectorAll(".source, .cite").forEach(function (node) {
      node.setAttribute("aria-current", node.dataset.n === number ? "true" : "false");
    });
  }

  function ask(question) {
    if (state.busy || !question.trim()) return;
    state.busy = true;
    el("send").disabled = true;
    el("input").value = "";
    autoGrow();
    clearEmptyState();

    var userBubble = turn("user");
    userBubble.appendChild(tag("div", question));

    var bubble = turn("assistant");
    var status = tag("div", null, "status");
    status.appendChild(tag("span", null, "dot"));
    status.appendChild(tag("span", t("thinking")));
    bubble.appendChild(status);

    var slowTimer = setTimeout(function () {
      if (bubble.contains(status)) status.appendChild(tag("span", t("slowHint"), "hint"));
    }, 4000);

    scrollToEnd();

    var body = tag("div", null, "md");
    var answer = "";
    var citations = [];
    var pending = false;

    // Re-rendering the whole answer on every token is correct but janky at
    // speed, so the paint is throttled to the frame. The parse is cheap; the
    // layout is not.
    function paint() {
      if (pending) return;
      pending = true;
      requestAnimationFrame(function () {
        pending = false;
        renderMarkdown(answer, body);
      });
    }

    state.controller = new AbortController();

    streamAsk(
      question,
      function (event) {
        if (event.type === "cached") {
          clearTimeout(slowTimer);
          answer = event.reply;
          citations = event.citations || [];
          bubble.replaceChildren(
            badge(event.curated ? t("instantBadge") : t("cachedBadge")),
            body
          );
          renderMarkdown(answer, body);
          renderSources(bubble, citations);
        } else if (event.type === "sources") {
          citations = event.citations || [];
          status.replaceChildren(tag("span", null, "dot"), tag("span", t("generating")));
        } else if (event.type === "token") {
          clearTimeout(slowTimer);
          if (bubble.contains(status)) bubble.replaceChildren(body);
          answer += event.text;
          paint();
        } else if (event.type === "answer") {
          clearTimeout(slowTimer);
          answer = event.reply;
          citations = event.citations || citations;
          bubble.replaceChildren(body);
          renderMarkdown(answer, body);
          renderSources(bubble, citations);
        } else if (event.type === "error") {
          clearTimeout(slowTimer);
          bubble.replaceChildren(errorBox(event.message));
        }
      },
      state.controller.signal
    )
      .then(function () {
        if (answer) {
          renderMarkdown(answer, body);
          if (!bubble.querySelector(".sources")) renderSources(bubble, citations);
          state.history.push({ question: question, answer: answer });
          if (state.history.length > maxTurns) {
            state.history = state.history.slice(-maxTurns);
          }
          updateBudget();
        }
      })
      .catch(function (error) {
        clearTimeout(slowTimer);
        if (error.name === "AbortError") {
          bubble.appendChild(tag("div", t("stopped"), "hint"));
        } else if (error.message !== "unauthorized") {
          bubble.replaceChildren(errorBox(t("offline")));
        }
      })
      .finally(function () {
        clearTimeout(slowTimer);
        state.busy = false;
        state.controller = null;
        el("send").disabled = false;
        el("send").textContent = t("send");
        scrollToEnd();
      });

    el("send").textContent = t("stop");
    el("send").disabled = false;
  }

  function badge(text) {
    var row = tag("div", null, "status");
    row.appendChild(tag("span", text, "badge"));
    return row;
  }

  function errorBox(message) {
    var box = tag("div", null, "error");
    box.appendChild(tag("strong", t("errorTitle")));
    box.appendChild(tag("div", message));
    return box;
  }

  function clearEmptyState() {
    var empty = document.querySelector(".empty");
    if (empty) empty.remove();
  }

  function showEmptyState() {
    var empty = tag("div", null, "empty");
    empty.appendChild(tag("h2", t("emptyTitle")));
    empty.appendChild(tag("p", t("emptyBody")));
    el("chat").replaceChildren(empty);
  }

  // ---------------------------------------------------------- suggestions ---

  function loadSuggestions() {
    if (!state.ui.suggestions) return;
    fetch("/api/suggestions", { headers: headers() })
      .then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
      })
      .then(function (data) {
        state.suggestions = data.questions || [];
        renderSuggestions();
      })
      .catch(function () {
        /* suggestions are a convenience; the assistant works without them */
      });
  }

  function renderSuggestions() {
    if (!state.suggestions.length) return;
    el("suggestions").hidden = false;
    el("suggestions-title").textContent = t("instantTitle");
    el("suggestions-hint").textContent = t("instantHint");

    var chips = document.createDocumentFragment();
    state.suggestions.forEach(function (question, index) {
      var chip = tag("button", null, "chip");
      chip.type = "button";
      if (index < 9) chip.appendChild(tag("span", index + 1, "chip-n"));
      chip.appendChild(tag("span", question));
      chip.addEventListener("click", function () {
        ask(question);
      });
      chips.appendChild(chip);
    });
    el("chips").replaceChildren(chips);
  }

  // ------------------------------------------------------------- shortcuts ---

  /*
    One map, used for both the behaviour and the help sheet — so a shortcut
    added here documents itself and cannot drift out of the list a client reads.
  */
  var SHORTCUTS = [
    { keys: ["/"], label: "shortcutFocus" },
    { keys: ["Enter"], label: "shortcutSend" },
    { keys: ["Shift", "Enter"], label: "shortcutNewline" },
    { keys: ["Esc"], label: "shortcutStop" },
    { keys: ["Alt", "N"], label: "shortcutNew" },
    { keys: ["Alt", "T"], label: "shortcutTheme" },
    { keys: ["?"], label: "shortcutHelp" },
  ];

  function renderShortcuts() {
    el("shortcuts-title").textContent = t("shortcutsTitle");
    el("shortcuts-close").textContent = t("close");
    var list = document.createDocumentFragment();
    SHORTCUTS.forEach(function (shortcut) {
      var dt = document.createElement("dt");
      shortcut.keys.forEach(function (key, index) {
        if (index) dt.appendChild(document.createTextNode(" + "));
        dt.appendChild(tag("kbd", key));
      });
      list.appendChild(dt);
      list.appendChild(tag("dd", t(shortcut.label)));
    });
    el("shortcuts-list").replaceChildren(list);
  }

  function bindShortcuts() {
    document.addEventListener("keydown", function (event) {
      var typing =
        document.activeElement &&
        (document.activeElement.tagName === "TEXTAREA" ||
          document.activeElement.tagName === "INPUT");

      if (event.key === "Escape" && state.controller) {
        state.controller.abort();
        return;
      }
      if (event.altKey && (event.key === "n" || event.key === "N")) {
        event.preventDefault();
        newChat();
        return;
      }
      if (event.altKey && (event.key === "t" || event.key === "T")) {
        event.preventDefault();
        toggleTheme();
        return;
      }
      if (typing) return;
      if (event.key === "/") {
        event.preventDefault();
        el("input").focus();
      } else if (event.key === "?") {
        event.preventDefault();
        el("shortcuts").showModal();
      } else if (/^[1-9]$/.test(event.key) && state.suggestions.length) {
        var picked = state.suggestions[Number(event.key) - 1];
        if (picked) ask(picked);
      }
    });
  }

  // ---------------------------------------------------------------- chrome ---

  function applyStrings() {
    document.documentElement.lang = state.lang;
    el("input").placeholder = t("placeholder");
    el("send").textContent = t("send");
    el("new").textContent = t("newChat");
    el("signout").textContent = t("signOut");
    el("theme").textContent =
      currentTheme() === "dark" ? t("themeToLight") : t("themeToDark");
    renderShortcuts();
    // Re-composed on a language switch, like every other string on the page.
    if (state.ui.posture_kind) el("posture").textContent = posture();
    if (state.ui.languages.length > 1) {
      el("lang-note").hidden = false;
      el("lang-note").textContent = t("answersLanguageNote");
    }
    if (state.suggestions.length) renderSuggestions();
    if (!el("chat").querySelector(".turn")) showEmptyState();
  }

  function currentTheme() {
    var stored = document.documentElement.getAttribute("data-theme");
    if (stored) return stored;
    if (state.ui && state.ui.mode !== "system") return state.ui.mode;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function toggleTheme() {
    var next = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem(THEME_STORE, next);
    } catch (err) {
      /* private browsing: the choice simply does not persist */
    }
    el("theme").textContent = next === "dark" ? t("themeToLight") : t("themeToDark");
  }

  function newChat() {
    if (state.controller) state.controller.abort();
    state.history = [];
    updateBudget();
    showEmptyState();
    el("input").focus();
  }

  function setLanguage(name) {
    state.lang = name;
    state.strings = state.ui.strings[name] || {};
    try {
      localStorage.setItem(LANG_STORE, name);
    } catch (err) {
      /* private browsing: the choice simply does not persist */
    }
    applyStrings();
  }

  function buildLanguageMenu() {
    var languages = state.ui.languages;
    if (languages.length < 2) return; // one language: no switcher at all.

    var menu = el("lang");
    menu.hidden = false;
    languages.forEach(function (name) {
      var option = document.createElement("option");
      option.value = name;
      option.textContent = (state.ui.strings[name] || {}).langName || name;
      menu.appendChild(option);
    });
    menu.value = state.lang;
    menu.addEventListener("change", function () {
      setLanguage(menu.value);
    });
  }

  // ------------------------------------------------------------------ gate ---

  function showGate(message) {
    el("app").hidden = true;
    el("gate").hidden = false;
    el("gate-title").textContent = t("keyTitle");
    el("gate-body").textContent = t("keyBody");
    el("gate-label").textContent = t("keyLabel");
    el("gate-button").textContent = t("keyButton");
    if (message) {
      el("gate-error").hidden = false;
      el("gate-error").textContent = message;
    }
    el("gate-input").focus();
  }

  function unlock(value) {
    state.key = value;
    // Verified against a real route before it is stored, so a wrong key is
    // rejected here rather than on the client's first real question.
    fetch("/api/suggestions", { headers: headers() }).then(function (response) {
      if (response.status === 401) {
        state.key = null;
        el("gate-error").hidden = false;
        el("gate-error").textContent = t("keyError");
        return;
      }
      try {
        localStorage.setItem(KEY_STORE, value);
      } catch (err) {
        /* private browsing: they will be asked again next session */
      }
      el("gate").hidden = true;
      start();
    });
  }

  // ----------------------------------------------------------------- boot ---

  /*
    The privacy sentence, composed here rather than sent finished.

    It is the claim the whole purchase rests on, so on a bilingual build it must
    not be the one line stuck in English above every answer. The server sends a
    token and a detail; the wording comes from the string file for the language
    the reader picked.
  */
  function posture() {
    var byKind = {
      local: "postureLocal",
      on_premises: "postureOnPremises",
      third_party: "postureThirdParty",
    };
    var sentence = t(byKind[state.ui.posture_kind] || "postureLocal");
    return sentence.replace("{detail}", state.ui.posture_detail || "");
  }

  function start() {
    el("app").hidden = false;
    el("title").textContent = state.ui.title;
    document.title = state.ui.title;
    el("posture").textContent = state.ui.intro
      ? state.ui.intro + " · " + posture()
      : posture();
    if (state.ui.has_logo) {
      el("logo").src = "/ui/logo";
      el("logo").hidden = false;
    }
    if (state.ui.access === "key") el("signout").hidden = false;

    applyStrings();
    loadSuggestions();
    el("input").focus();
  }

  function autoGrow() {
    var input = el("input");
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 192) + "px";
  }

  function boot() {
    try {
      var storedTheme = localStorage.getItem(THEME_STORE);
      if (storedTheme) document.documentElement.setAttribute("data-theme", storedTheme);
    } catch (err) {
      /* private browsing: fall back to the configured default */
    }

    fetch("/api/ui")
      .then(function (response) {
        return response.json();
      })
      .then(function (payload) {
        state.ui = payload;
        /* The server's own limit, not a copy of it. */
        if (typeof payload.max_turns === "number") maxTurns = payload.max_turns;
        updateBudget();

        var stored = null;
        try {
          stored = localStorage.getItem(LANG_STORE);
        } catch (err) {
          /* private browsing */
        }
        state.lang =
          stored && payload.languages.indexOf(stored) !== -1
            ? stored
            : payload.languages[0];
        state.strings = payload.strings[state.lang] || {};
        buildLanguageMenu();

        if (payload.access === "key") {
          try {
            state.key = localStorage.getItem(KEY_STORE);
          } catch (err) {
            /* private browsing */
          }
          if (!state.key) {
            showGate(null);
            return;
          }
        }
        start();
      });

    el("composer").addEventListener("submit", function (event) {
      event.preventDefault();
      if (state.busy && state.controller) {
        state.controller.abort();
        return;
      }
      ask(el("input").value);
    });

    el("input").addEventListener("input", autoGrow);
    el("input").addEventListener("keydown", function (event) {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        ask(el("input").value);
      }
    });

    el("theme").addEventListener("click", toggleTheme);
    el("new").addEventListener("click", newChat);
    el("help").addEventListener("click", function () {
      el("shortcuts").showModal();
    });
    el("shortcuts-close").addEventListener("click", function () {
      el("shortcuts").close();
    });
    el("signout").addEventListener("click", function () {
      try {
        localStorage.removeItem(KEY_STORE);
      } catch (err) {
        /* private browsing: nothing was stored */
      }
      state.key = null;
      showGate(null);
    });
    el("gate-form").addEventListener("submit", function (event) {
      event.preventDefault();
      unlock(el("gate-input").value);
    });

    bindShortcuts();
  }

  document.addEventListener("DOMContentLoaded", boot);

  // Handed to the offline renderer test (tests/js/markdown_test.mjs), which
  // runs this file against a stub DOM. A browser has no `module`, so this line
  // does nothing in the product — and the alternative is shipping the one piece
  // of this phase that exists to fix a known defect with nothing proving it works.
  if (typeof module !== "undefined") {
    module.exports = { renderMarkdown: renderMarkdown };
  }
})();
