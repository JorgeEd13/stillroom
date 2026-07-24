/*
  The Markdown renderer, exercised against a stub DOM.

  This is the one test in this repo that is not Python, and it earns the
  exception: landmine 3 — a chat UI that shows `**bold**` and `##` on screen as
  literal syntax — is the defect this whole phase exists to fix, and the fix is
  100 lines of JavaScript. Asserting it in the language it is written in is the
  only way to assert it at all.

  It uses **no packages**: `node --test` is built in, and the DOM below is forty
  lines of stub. That matters — the product's claim is that its front end has no
  dependency tree, and a test suite that quietly installed one would make that
  claim false in the place a client would look first (`package.json`).

  `app.js` is evaluated with `document`, `window`, `localStorage` and `module`
  injected as parameters, so nothing here touches a real global and the file
  under test is the exact byte-for-byte file the server serves.
*/

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const APP_JS = join(HERE, "..", "..", "src", "stillroom", "ui", "static", "app.js");

// --------------------------------------------------------------- stub DOM ---

function makeNode(tagName) {
  return {
    tagName,
    children: [],
    text: null,
    className: "",
    dataset: {},
    attributes: {},
    tabIndex: 0,
    appendChild(child) {
      this.children.push(child);
      return child;
    },
    replaceChildren(...nodes) {
      this.children = [];
      for (const node of nodes) {
        if (node && node.__fragment) this.children.push(...node.children);
        else this.children.push(node);
      }
    },
    setAttribute(name, value) {
      this.attributes[name] = value;
    },
    querySelectorAll() {
      return [];
    },
    set textContent(value) {
      this.text = String(value);
      this.children = [];
    },
    get textContent() {
      return this.text;
    },
  };
}

const document = {
  createElement: (name) => makeNode(name),
  createTextNode: (text) => ({ tagName: "#text", text, children: [] }),
  createDocumentFragment: () => Object.assign(makeNode("#fragment"), { __fragment: true }),
  getElementById: () => null,
  querySelector: () => null,
  addEventListener: () => {},
  documentElement: { setAttribute: () => {}, getAttribute: () => null },
  activeElement: null,
};

const window = { matchMedia: () => ({ matches: false }), scrollTo: () => {} };
const localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };

/** Serialise the stub tree the way a browser would render it, for assertions. */
function html(node) {
  if (node.tagName === "#text") return node.text;
  const inner =
    node.text !== null && node.text !== undefined
      ? node.text
      : node.children.map(html).join("");
  if (node.tagName === "#fragment") return inner;
  const cls = node.className ? ` class="${node.className}"` : "";
  return `<${node.tagName}${cls}>${inner}</${node.tagName}>`;
}

const source = readFileSync(APP_JS, "utf8");
const module = { exports: {} };
new Function("document", "window", "localStorage", "module", source)(
  document,
  window,
  localStorage,
  module
);
const { renderMarkdown } = module.exports;

function render(markdown) {
  const root = makeNode("div");
  renderMarkdown(markdown, root);
  return root.children.map(html).join("");
}

// ----------------------------------------------------------------- tests ---

test("landmine 3: bold and headings become elements, not visible syntax", () => {
  assert.equal(render("**30 days**"), "<p><strong>30 days</strong></p>");
  assert.equal(render("## Refund window"), "<h3>Refund window</h3>");
  assert.equal(render("### Detail"), "<h4>Detail</h4>");
  // The exact strings that appeared on screen in an earlier project.
  const rendered = render("The window is **30 days**.\n\n## Notice");
  assert.ok(!rendered.includes("**"));
  assert.ok(!rendered.includes("##"));
});

test("emphasis, strikethrough and inline code", () => {
  assert.equal(render("_soon_"), "<p><em>soon</em></p>");
  assert.equal(render("*soon*"), "<p><em>soon</em></p>");
  assert.equal(render("~~never~~"), "<p><s>never</s></p>");
  assert.equal(render("run `stillroom ingest`"), "<p>run <code>stillroom ingest</code></p>");
});

test("lists keep their kind", () => {
  assert.equal(render("- one\n- two"), "<ul><li>one</li><li>two</li></ul>");
  assert.equal(render("1. one\n2. two"), "<ol><li>one</li><li>two</li></ol>");
});

test("a paragraph is joined across soft line breaks", () => {
  assert.equal(render("thirty days\nfrom purchase"), "<p>thirty days from purchase</p>");
});

test("fenced code is not parsed as Markdown inside", () => {
  assert.equal(
    render("```\n**not bold**\n```"),
    "<pre><code>**not bold**</code></pre>"
  );
});

test("blockquotes and tables", () => {
  assert.equal(render("> quoted"), "<blockquote><p>quoted</p></blockquote>");
  assert.equal(
    render("| a | b |\n| --- | --- |\n| 1 | 2 |"),
    "<table><thead><tr><th>a</th><th>b</th></tr></thead>" +
      "<tbody><tr><td>1</td><td>2</td></tr></tbody></table>"
  );
});

test("citation markers become inspectable chips, not text", () => {
  const root = makeNode("div");
  renderMarkdown("Refunds take 30 days [2].", root);
  const paragraph = root.children[0];
  const chip = paragraph.children.find((node) => node.className === "cite");

  assert.ok(chip, "no citation chip was produced");
  assert.equal(chip.dataset.n, "2");
  assert.equal(chip.textContent, "[2]");
});

test("a link is rendered as text, never as a clickable anchor", () => {
  const rendered = render("see [the policy](http://intranet/policy)");

  assert.ok(!rendered.includes("<a"), "an answer produced a clickable link");
  assert.ok(rendered.includes("the policy"));
  assert.ok(rendered.includes("http://intranet/policy"));
});

test("HTML in model output is shown, never interpreted", () => {
  // The property that makes it safe to render text a model wrote about
  // documents somebody else supplied: this arrives as characters in a text
  // node. The stub DOM cannot execute anything, so the assertion is that the
  // tag was set as TEXT rather than parsed into an element.
  const root = makeNode("div");
  renderMarkdown("<img src=x onerror=alert(1)>", root);
  const paragraph = root.children[0];

  assert.equal(paragraph.tagName, "p");
  assert.equal(paragraph.children.length, 1);
  assert.equal(paragraph.children[0].tagName, "#text");
  assert.equal(paragraph.children[0].text, "<img src=x onerror=alert(1)>");
});

test("an empty or partial answer renders without throwing", () => {
  // The streaming path re-renders on every token, so the parser sees every
  // prefix of the answer — including "**" on its own.
  assert.doesNotThrow(() => render(""));
  assert.doesNotThrow(() => render("**"));
  assert.doesNotThrow(() => render("| a |"));
  assert.doesNotThrow(() => render("```"));
  assert.doesNotThrow(() => render("- "));
});
