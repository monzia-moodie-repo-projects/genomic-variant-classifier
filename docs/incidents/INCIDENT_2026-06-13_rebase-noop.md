# INCIDENT 2026-06-13 -- silent no-op rebase force-pushed content-identical history

## Summary
While regrouping the session's work into clean commits, the af_1kg `_join_gnomad` wiring landed in
the WRONG commit (the L1 leakage commit instead of the af_1kg commit). The first attempt to correct
this with an interactive rebase **silently did nothing** and was force-pushed -- rewriting every
commit hash while leaving the file-to-commit distribution unchanged. A second, hardened rebase fixed
it correctly. No code was lost or corrupted at any point; the defect was purely in commit history.

## Timeline
1. **Origin of the misplacement.** During `git add -p` on real_data_prep.py (which carried two
   independent hunks -- af_1kg in _join_gnomad, L1 in _gene_aware_split), both hunks were staged
   (answered `y`/`y`) instead of staging only L1 (`n`/`y`). Commit f25bbb2 (L1) therefore included
   the af_1kg block (real_data_prep.py 40 lines = 24 L1 + 16 af_1kg); the af_1kg commit 6b0494c
   excluded it.
2. **First corrective rebase (NO-OP).** An interactive rebase marked the L1 and af_1kg commits as
   `edit` and was meant to run a remover script at one stop and an adder at the other. The two helper
   scripts were in Downloads but **never copied into the repo root**, so `python .\remove_af1kg_rdp.py`
   failed with "No such file or directory." Each subsequent `git add` staged nothing, each
   `git commit --amend --no-edit` re-committed identical content under a new hash. The rebase rewrote
   all hashes (b3894e0 -> 6b00bd8 era) but moved nothing.
3. **Insufficient gate.** The post-rebase check `git diff backup HEAD` returned empty -- which is true
   BOTH when the rebase correctly reorganises history AND when it does nothing -- so it passed on the
   no-op. The per-commit check `git show HEAD~4 ... 1KGP` DID print the block (signalling failure), but
   it was framed as "expect nothing" rather than a hard abort, and the force-push proceeded.
4. **Detection.** A later diagnostic on the new hashes confirmed the misplacement persisted:
   `git show 3d4b126 ... 1KGP`.Count == 1 (L1 still entangled), `git show 46877b9 ...`.Count == 0
   (af_1kg still missing the wiring).
5. **Second corrective rebase (SUCCESS).** Same plan, three hardenings: (a) scripts invoked by
   ABSOLUTE path from Downloads -- their TARGET is repo-relative, so CWD=repo is sufficient and there
   is no copy step to forget; (b) each stop gated on the script's success message AND a non-empty
   staged diff; (c) the decisive verification is a per-commit content count that must read 0 (L1) then
   1 (af_1kg) -- the exact inverse of the broken state, so it cannot pass on a no-op. Result: 0 then 1,
   `git diff backup-pre-rebase2 HEAD` empty, suite 989 green, force-push d9bd23b -> 8c19f9b.

## Root causes
- A required setup step (copy scripts into the repo) was written as prose preamble, not a numbered,
  verified step -- so it was skipped.
- The chosen verification gate (`git diff backup HEAD` empty) could not distinguish "fixed" from
  "unchanged"; both produce an empty diff.

## Fixes / preventions
- Run helper scripts by absolute path (`python "$HOME\Downloads\<script>.py"`) so a missing copy
  cannot cause a silent no-op; their relative TARGET resolves against the repo CWD.
- Gate every history-edit stop on (1) the script's explicit success line and (2) a non-empty
  `git diff --cached`. An empty staged diff after a script that should change a file => abort.
- Verify history rewrites with a per-commit CONTENT check whose pass condition is the inverse of the
  broken state (here: af_1kg-block count 0 in L1, 1 in af_1kg), not only a whole-tree diff.

## Learned
- "History reorganised correctly" and "history not touched" can look identical to a whole-tree diff.
  A correct rewrite must be proven at the per-commit level.
- On a solo repo, force-push is low-risk for collaborators but still rewrites the remote -- so the
  pre-push gate must be unambiguous, because the push is the irreversible step.
- A `--force-with-lease` and a backup tag (`git reset --hard <tag>`) made every attempt fully
  recoverable; nothing was ever at risk despite two rounds.
