# Session opener -- paste this as your first message in the new session

Also upload, in the same message:
  1. `HANDOFF_2026-07-23_session_close.md`  (the full handoff -- read it first)
  2. `FRESHNESS_2026-07-20.md`              (only when we reach item 2)

---

## Paste from here

You are resuming work on **genomic-variant-classifier**, a whole-genome,
multi-modal, multi-model variant pathogenicity classifier. Repository:
`github.com/monzia-moodie-repo-projects/genomic-variant-classifier`. Local:
`C:\Projects\genomic-variant-classifier`. Python 3.12.10, venv `.venv312`,
PowerShell 5.1. My downloads always land in `C:\Users\monzi\Downloads\`.

I have uploaded `HANDOFF_2026-07-23_session_close.md`. **Read it in full before
doing anything else.** It records the entire previous session: what was completed
with commit hashes, what remains with protocols, and the status of every larger
deliverable. Do not re-derive any of it and do not relitigate settled decisions.

### First action, before you propose anything

Clone the remote fresh and verify the current state yourself rather than trusting
the handoff or your memory:

- confirm `main` HEAD, expected `644a184`
- confirm the suite-size ratchet in `tests/EXPECTED_SUITE_SIZE`, expected **2874**
- confirm the README test badge agrees with the ratchet
- report anything that differs from the handoff, before proceeding

### The immediate next action

**Item 3 -- push S0 Commit 2.** It is already built, verified and waiting in my
Downloads folder. Section 3 of the handoff has the four files, their SHA-256
hashes, and the complete runnable command sequence. Net +19, ratchet 2874 -> 2893.

**Warning recorded in the handoff:** `install_ratchet_bump_2879_2026-07-22.py` is
STALE and its pre-check will fail. Use `install_ratchet_bump_2893_2026-07-23.py`
(SHA-256 `04d340f7e358b48f121adc4dffc542ebdcc3b2a655da1b66bdd02050dc3a9129`).

### The queue after that, in this order

1. **Repository-wide Python-handle-into-Arrow audit.** 328 `pandas.read_parquet`
   call sites were found by syntax-tree walk. Handoff section 4 explains why a
   blanket ban is the wrong answer and gives the five-step classification protocol.
   Do not skip the classification step.
2. **Data-source freshness.** AlphaFold HTTP 404 and LOVD HTTP 400. Handoff
   section 5. Treat each as its own root-cause investigation; do not assume
   endpoint drift.
3. **Environment reproducibility.** `requirements.lock` and `requirements-dev.lock`
   exist but are referenced zero times in `ci.yml`. Handoff section 6.
4. **`docs/METRICS.md` audit** against the current code, which has not been done
   since the R3a/R3b split and the protocol-inheritance work landed.

### How I want you to work

Read handoff section 8 -- the session-durable lessons -- and hold to them. The ones
that cost real time last session:

- **Every step I must perform gets a runnable command**, including `Copy-Item`,
  `git commit` and `git push`. Prose steps do not get run. If a step is genuinely
  manual, such as a button in the Actions interface, mark it as manual explicitly.
- **Hashes prove a copy landed, test counts do not.** Three times last session the
  suite was green while my working tree held stale files.
- **Compute every sum in a tool call**, never from memory, and never state a rate
  without its denominator.
- **Zero events is a bound, not proof.** Report the rule-of-three upper bound.
- **Verify against a fresh clone**, not against your recollection. A shallow clone
  reports the wrong dates.
- **Never paste angle-bracket placeholders into PowerShell.**
- **Check an installer's printed prose against its own arithmetic** before
  delivering it.
- Spell out every acronym on first use.

Assume nothing about my actions, reasoning or intent. When context is missing, ask.
Aggressively pursue every warning, skip, stub, drift, stale snapshot and
discrepancy, and document with dates. Avoid patchwork; build from the ground up. No
blockers is the lowest bar, not the goal.

Begin by verifying the repository state, then give me the command sequence for
item 3.

## Paste to here
