<!--
  The plain-language runbook the client receives. Copy it into the engagement
  folder, replace every <PLACEHOLDER>, delete what does not apply to their build,
  and DELETE THIS COMMENT.

  ⚠️ Written for somebody who did not choose to be reading documentation. Rules
  for editing it:
  * No jargon that is not explained in the same sentence. No "container", no
    "volume", no "endpoint", no "SSE".
  * Every instruction is something they can do without a terminal, unless the
    heading says otherwise.
  * It states what NOT to expect, out loud. Speed and refusals are the two
    things that generate "is it broken?" messages, and both are the product
    working correctly.
  * The definition of done for this file is the phase's: somebody following only
    this document, on a machine they have not been shown, reaches a working
    chatbot.
-->

# Your document assistant — how to run it

This assistant answers questions about **your own documents**, on **your own
computer**. Nothing is sent anywhere. You can unplug the network and it still
works.

---

## Before you start (once)

Two programs need to be installed on the computer that will run the assistant.
Both are free, and both come from their own makers — install them yourself from
their official websites, not from me:

1. **Docker Desktop** — runs the assistant.
2. **Ollama** — runs the AI model on your machine.

After installing, **open both once** so they are running. Ollama is quiet: it
sits in the system tray or menu bar.

### Then: two AI models

The assistant uses **two** models, and they do different jobs. Open a terminal
(Command Prompt on Windows) and run these once. Each downloads once and is then
yours — nothing is downloaded again afterwards, and nothing is sent anywhere.

```
ollama pull <CHAT MODEL>
ollama pull bge-m3:567m
```

- The first one **writes the answers**.
- The second one **reads your documents**, so the assistant can find the right
  passage to answer from. It is about 1.2 GB.

> **If you only install the first one**, the assistant will stop while it is
> setting up and tell you the second one is missing. Nothing is broken — run the
> command it shows you and start it again.

> I never install anything on your computer, and nothing I send you does either.
> If a step here needs software, you install it from the maker's own site.

---

## Every day: starting it

- **Windows:** double-click **`start.cmd`**
- **Mac or Linux:** double-click **`start.sh`** (or run `./start.sh`)

A window opens, tells you what it is doing, and then your browser opens at:

**http://localhost:8000**

**The very first time you run it, it takes several minutes** — it is assembling
the assistant and reading your documents, and it only ever does that once. Leave
the window open; it tells you when it is ready.

> **Keep the big `stillroom-base-….tar.gz` file in this folder.** It is the
> prepared assistant, and it is why the first start does not need to download
> anything. If it is missing, the first start tries the internet instead and can
> fail on an office network that blocks it — the window will say so, and I will
> send the file again.

After that, a start takes a minute at most: the model has to be loaded into
memory, and that is all.

> ⚠️ **Ollama must already be running before the first start**, because that
> first run prepares your instant answers and needs the model to do it.

**To stop it:** double-click **`stop.cmd`** (Windows) or **`stop.sh`**. Your
documents, your settings and everything the assistant has learned are kept.

---

## Using it

Type a question and press **Enter**. The answer arrives with the **sources** it
came from, listed underneath. Click a number like `[1]` in the answer to see
which document it came from.

**The buttons at the top:**

| Button | What it does |
|---|---|
| **New chat** | Forgets the conversation and starts fresh. Your documents are untouched. |
| **Light / Dark** | Switches the appearance. Your choice is remembered. |
| **?** | Shows the keyboard shortcuts. |
| **Language** | Changes the language of *this page*. <!-- bilingual builds only --> |

**It follows the conversation.** Ask *"and after that?"* or *"how long does it
take?"* and it knows what you were talking about. It remembers the last few
exchanges only, and **only until you press New chat** — nothing about your
conversation is saved anywhere.

Two things it will not do, on purpose. It will not treat something *it* said as
a fact: every answer is still built from your documents, and still shows which
one. And if a follow-up wanders somewhere your documents do not cover, it says
so rather than reaching for the nearest document that looks similar.

**Instant answers.** The questions listed under *Instant answers* were prepared
in advance when the assistant was built, so they come back in about a second.
Press `1`–`9` to ask one without typing. Any other question runs the model,
which takes longer.

---

## What to expect — and what is not a fault

**It is slower than a cloud chatbot.** That is the trade. A cloud service runs
on a rack of graphics cards in a data centre and your documents go with it. This
runs on your machine, so your documents stay here. While it is working you can
see it writing the answer — that is it thinking, not a fault.

**Sometimes it says it does not know.** If your documents do not contain the
answer, it says so instead of inventing one. **This is the most important thing
it does.** Every chatbot can answer a question it knows; the reason this one is
safe to put in front of your team is that it does not guess.

**It only knows your documents.** It has no general knowledge, no internet, and
no memory of anything outside the documents it was given.

---

## When something looks wrong

| What you see | What it means | What to do |
|---|---|---|
| The page does not open at all | The assistant is not running | Run `start` again; it explains what is missing |
| "Docker is not running" | Docker Desktop was closed | Open Docker Desktop, wait for it to say it is running, start again |
| "Ollama is not responding" | Ollama was closed | Open Ollama, wait a few seconds, start again |
| "needs a second AI model" | Only the chat model was installed | Run `ollama pull bge-m3:567m`, then start again |
| It asks for an **access key** | Normal on a shared setup | Enter the key from your configuration file — once per computer |
| Answers are about old documents | The documents changed but were not re-read | Run `start` again — see *Adding documents* |
| It does not know about a document you added | The documents were not re-read | Run `start` again — see *Adding documents* |
| It is very slow on the first question of the day | The model is loading into memory | Wait; the next ones are quick |

If none of these fit, send me: what you asked, what it answered, and what the
`start` window said. That is enough for me to tell you what happened.

---

## Adding or changing documents

Put the new files in the **`documents`** folder, then **run `start` again**.
That is the whole procedure.

It will stop the assistant, re-read your documents, and start it again. If
nothing changed it takes a few seconds; if you added a lot it can take a few
minutes. When it says *Ready*, reload the page in your browser.

Two things worth knowing:

- **If something is wrong with the new documents, nothing is lost.** The
  assistant keeps answering from the documents it already had, and the window
  tells you which file was the problem.
- **Prepared instant answers are rebuilt** when the documents change, so an
  instant answer can never quietly describe a document you have replaced.
- **Scanned documents do not work.** A PDF that is a *photograph* of a page has
  no text in it to read. A PDF exported from Word does. If a file is skipped,
  the window tells you which one and why.

<!-- If you would rather run it by hand, this is the same work:
     docker compose run --rm assistant stillroom refresh --config client.toml
     followed by stop and start. -->

<!-- scheduled-refresh builds only: -->
## Automatic refresh <!-- scheduled-refresh builds only -->

This build re-checks your documents every **<INTERVAL>** minutes on its own. If
nothing changed, it does nothing. You do not have to run anything.

---

## Checking it is really private

You do not have to take my word for it:

1. Open **http://localhost:8000/api/health** in your browser. It states, in a
   sentence, where the model runs — and whether the assistant is answering from
   your current documents.
2. **Disconnect the computer from the internet** and ask a question. It still
   answers. <!-- Delete this line for a bring-your-own-key build. -->

---

## What is included, and what is not

**Included:** up to <N> documents, <M> prepared instant answers,
sources on every answer, and <SUPPORT WINDOW> of support from handover.

**Not included:** reading scanned documents, connecting to databases or other
systems, and any messaging platform other than this web page. Each of those is
real work and I would rather quote it properly than half-do it here.

**Ownership:** your documents, your configuration, your index and this
deployment are yours outright. The engine underneath is my own pre-existing
tooling, and you have a permanent, paid-up licence to use and change it inside
your organisation. Nothing phones home. Nothing expires.

---

**Contact:** <HOW THEY REACH ME> · **Support until:** <DATE>
