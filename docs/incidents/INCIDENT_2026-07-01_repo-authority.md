# INCIDENT / DECISION RECORD: Repository authority ambiguity

- **Date:** 2026-07-01
- **Status:** OPEN -- awaiting one factual confirmation from GitHub (see "Resolution gate")
- **Severity:** Process / provenance (no code or data loss)
- **Author:** recorded during the Unit 1 / Unit 2 session

## Summary

There is an unresolved ambiguity about which GitHub remote is the AUTHORITATIVE
repository for this project, and -- critically -- **no prior written record exists**
of a repository move that was supposed to have happened days ago. This document
exists to close that documentation gap and to force the ambiguity to be settled by
evidence rather than memory.

## What is factually known (from `git` output on 2026-07-01)

- The local working copy at `C:\Projects\genomic-variant-classifier` has:
  - `origin (fetch) = https://github.com/monzia-moodie-repo-projects/genomic-variant-classifier.git`
  - `origin (push)  = https://github.com/monzia-moodie-repo-projects/genomic-variant-classifier.git`
- All recent work has been pushed to `-repo-projects`:
  - `bbfee34` feat(orchestrator): lazy agent registry (Phase 1) -- CI GREEN
  - `ebbcbbc` fix(agents): remove vestigial phantom-drift path (Unit 1) -- CI GREEN
  - `ab56cde` test(agent-layer): reactivate message-bus suite (Unit 2) -- CI running at time of record
- `git ls-remote origin` shows `refs/heads/main = ab56cde...` and a second branch `run9a-prep`.
- Therefore **the most up-to-date state of the project currently lives in
  `-repo-projects`.** Whatever was intended earlier, the real work is here today.

## The ambiguity

The project owner stated (2026-07-01) that the authoritative repository was to be
moved/reclaimed to `https://github.com/monzia-moodie/genomic-variant-classifier`
"some days ago", and that `-repo-projects` was to become stale. However:

1. No documentation of that move exists anywhere in `docs/`, `docs/sessions/`,
   `docs/incidents/`, `README.md`, or the roadmap (searched 2026-07-01).
2. The local `origin` was never re-pointed -- it still targets `-repo-projects`.
3. Consequently, days of commits (through `ab56cde`) went to `-repo-projects`.

If the reclaim to `monzia-moodie/...` was never executed at the git level, then
`monzia-moodie/...` is BEHIND (missing all recent work) and `-repo-projects` is
authoritative-in-practice. This is the most likely state given (2) and (3).

## Resolution gate (settle by evidence, not memory)

Run and record the output of:

    git ls-remote https://github.com/monzia-moodie/genomic-variant-classifier.git main
    git ls-remote https://github.com/monzia-moodie-repo-projects/genomic-variant-classifier.git main
    git rev-parse HEAD

Decision rule:
- If `monzia-moodie/...` errors or is far behind `ab56cde` -> `-repo-projects` is
  authoritative; keep `origin` as-is; mark this record RESOLVED (authoritative =
  -repo-projects).
- If `monzia-moodie/...` has recent history -> decide migration explicitly; migrate
  `-repo-projects` commits FORWARD into it (never abandon the newer work); re-point
  `origin`; mark RESOLVED (authoritative = monzia-moodie, migrated at <hash>).

## Corrective action for the ROOT cause (the missing record)

The root cause was **not** the repo move itself -- it was the absence of a written
record of it. Going forward, ANY change to remotes, repo identity, or push targets
MUST be recorded in this incidents directory AND the session log AT THE TIME it is
done, per the project's standing documentation law. This file is the first such
record and the template for future ones.
