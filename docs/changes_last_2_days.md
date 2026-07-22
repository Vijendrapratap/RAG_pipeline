# What We Changed in the Last 2 Days

*A simple story of the work, written so anyone can follow it.*

## First, what is this thing we're building?

Imagine a **giant library** with millions of pages — all of them are talks
(discourses) that were turned into text by a computer that listens to audio.

We're building a **super-smart librarian** for that library. You ask it a
question, and it runs off, finds the exact pages that answer you, and reads
them back. That librarian is what we work on.

Over the last two days we did two big things:

1. **Cleaned up the messy pages** so the librarian isn't reading junk.
2. **Made the librarian smarter** about *how* it searches.

---

## Day 1 (July 9) — Cleaning up the messy pages

Some of the pages in our library were broken or repeated, like a scratched
record that says the same word over and over. If the librarian reads those, it
gives bad answers. So we fixed the pages.

- **Found the broken pages.** We ran a test that showed exactly where the
  librarian was tripping up, and saved proof of what was wrong.
- **Taught it to cut sentences properly.** Our language (Hindi) ends sentences
  with a special mark called a *danda* ( । ). We taught the page-cutter to
  respect it, so sentences don't get chopped in the wrong place.
- **Threw out the "broken record" pages.** Pages where the computer got stuck
  repeating itself were removed, and pages that were too big were split into
  neat smaller ones.
- **Built a "fix one page" button.** A new tool that can cleanly pull a single
  bad page out of every place it's stored and re-do it — without messing up the
  rest of the library.
- **Re-did 50 poisoned pages** and double-checked that all our storage boxes
  now agree with each other.
- **Stopped losing our labels.** When the storage box restarted, the extra
  notes we'd added to pages (like "this talk mentions Namdev") were getting
  lost. We made those notes *stick permanently*.
- **Turned the label-maker back on.** A helper that reads a page and writes
  short labels for it (people, places, topics) was stuck — we unblocked it and
  made it faster, and stopped it from getting stuck in loops.
- **Made sure the catalog gets a fair shot.** We have a special list (a
  spreadsheet of talks). We fixed the search so pages from that list aren't
  thrown away too early before the librarian picks the best answer.

**In short:** the library pages went from *messy and unreliable* to *clean and
trustworthy*.

---

## Day 2 (July 11) — Making the librarian smarter

Now that the pages were clean, we made the librarian better at *thinking*.

### Better labels and a study guide

- **Recovered 171 "lost" pages.** The label-maker had given up on 171 pages.
  We forced it to fill in the blanks the right way and rescued all of them.
- **Built a summary index** — like the little blurb on the back of a book — so
  the librarian can skim summaries first, then dive into details.
- **Fixed doubled-up notes** on 391 files (some notes were written twice) and
  pushed the clean versions back into storage.

### A "report card" for the librarian

- **Made a fair test.** We wrote a set of real questions *with* the correct
  answers — including trick questions the librarian should politely refuse to
  answer. Now we can actually score how good it is.
- **Tried lots of settings** to find which knobs make it answer best, and
  wrote down the score *before* our big new changes — so we can prove the new
  changes help.

### The big new brain upgrade (called "Stage 4")

This is the headline. We taught the librarian to **notice what *kind* of
question you're asking** and search differently for each:

- **A question-sorter.** Before searching, the librarian quietly guesses: "Is
  this asking about a *person*? a *quote*? a *topic*? a *number/count*? or a
  *follow-up* to what we were just talking about?" It gets this right about
  **97 out of 100 times**, in a tiny fraction of a second.
- **Different search style for each kind.** A question about a person's name is
  searched differently from a question about an idea — because that gets better
  answers.
- **Smarter topic search ("HyDE").** For big "tell me about..." questions, the
  librarian first *imagines* what a good answer would look like, then uses that
  to find better pages. We also fixed a bug where the model was accidentally
  mumbling its private thoughts into the search — now it stays quiet and clean.
- **Understands follow-up questions.** If you ask "and where else is *that*
  mentioned?", the word "that" is confusing on its own. The librarian now looks
  back at the last thing you talked about and rewrites your question to spell
  it out — so it actually finds the right pages.

### An important, careful choice

Every one of these new brain upgrades has an **on/off switch, and we left them
all switched OFF for now.** Why? Because the rule on this project is: *don't
change how it works for real users until we've proven the new way is better and
still fast.* We built them, we tested them, we wrote down the results — and now
a grown-up gets to decide when to flip each switch on.

---

## The one-sentence version

**We scrubbed the library clean, gave it a fair report card, and built a
smarter librarian brain — then left the new brain's switches off until someone
decides it's ready to turn on.**
