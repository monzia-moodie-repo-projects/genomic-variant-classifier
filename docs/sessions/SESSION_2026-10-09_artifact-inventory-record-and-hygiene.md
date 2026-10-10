# Session record, 2026-10-09: the first artifact-inventory record, repository hygiene, and three procedure defects

Author: Monzia Moodie. Design authority: owner rulings 2026-10-08c, 2026-10-08e and 2026-10-08f (the record and its index) and the
hygiene plan the owner approved by digest. Base: branch `run-intent-2026-10-09`, commit `1228c8f2e37d90f98ebce0715cde1e67b7f4db49`,
tree `5b365c3f77d663f4ebb3907dacebd6ee4e40c499` (the validated run-intent candidate; this change follows it).

## 1. The record

The owner ran the collector on the artifact store on 2026-10-09 from 13:40:36 to 13:42:13 UTC (implementation tree `6ef5da1f`, main
after pull request #50). It wrote `REC-ece23653bf8e45dbad3da26303870dc6.json` (561,366 bytes, SHA-256
`bb716d48ad628ae593e104263efd7ca846db237d25dff2b3f35ba0018d192c22`), a readiness decision, a summary and a run record, and packed
them in `artifact_inventory_20261009T134034Z.evidence.zip` (SHA-256 `0d14514dd11c471dabc731a7b0d549bc0ca9eb7c75345f80c4f21ae111108df6`,
82,136 bytes). The four members were uploaded separately; their digests:

| Member | SHA-256 | Bound by |
| --- | --- | --- |
| `REC-ece23653bf8e45dbad3da26303870dc6.json` | `bb716d48ad628ae593e104263efd7ca846db237d25dff2b3f35ba0018d192c22` | the collector's printed line, the summary and the decision |
| `readiness_decision.json` | `5b554f50c475073da601c655b8831b60278efd908fdc05ec28a1fdade3af380f` | the summary |
| `summary.json` | `caffae999d32ce8935f056acf567368509c5c045bb293b136c24d224894c559a` | the evidence archive only |
| `run_record.json` | `829d371f3d9f9a73680d5e4f24064917d399cac4ef121a13ab33dc8c249b0565` | the evidence archive only |

Because the summary and the run record are bound only by the archive, the installer re-verifies all four members against the owner's
archive by its digest before anything is committed.

What the record says, read from the record itself: 313 requirements (177 Windows binaries, 112 sources, 13 local binaries, 11
run-evidence bundles), all "match" with the location hint confirmed; 214 content objects, all found; 416 locations, all read, on one
volume, each a distinct filesystem file; 202 are second copies (189 acquired archives in their acquisition run directory and in
`accepted/`; 13 local builds in their build run directory and in `accepted/local_binary`); the renv bootstrap archive and the eleven
run-evidence archives exist once. 214 searches, all complete. Three R packages are supplied by the runtime (Matrix 1.7-5, codetools
0.2-20, lattice 0.22-9). One known gap: the evidence bundle of `fixtures_20261007T022427Z` was never retained, so it has no digest to
verify. The readiness decision (99 rows, all present) is historical artifact-input availability only.

## 2. Verification before commit

`make_record_unit_v2.py` (SHA-256 `64f1387e7ff53320bd29af5d2aab4b8606a888cb38017f3670dfb0b2c39536e3`; kept outside the repository with
the collector it imports) refuses unless every check passes, and writes nothing before: member digests (full 64 hexadecimal
characters; a prefix is refused, never completed); exact member set; byte-identical round-trip through the typed owner; record id equals
file name; first record; verifier equals the collector's exact bytes; summary and decision name the record; the summary binds the
decision's bytes and equals the decision's implementation; the summary's counts equal `record.counts()`; every loaded checkout module
named by the decision (11) has that canonical digest in this tree; the measured tree is the tree of a commit in this history; the run
record is admitted, completed, stage `readiness`, and its interval contains the measurement and the evaluation; readiness re-derived
from the record and the pinned plans (`c578eb10...`, `cc1251f8...`) equals the decision's rows. Eleven seeded cases (ten refusals and
one positive control in archive mode) behaved as stated.

## 3. Repository hygiene (performed by the owner, measured here afterwards)

The hygiene plan (`1a3d9b75...686a`) was removed item by item; the removal log reports 54 deletions, 0 skipped, 0 failed: three
disposable validation clones (receipts kept), 20 GitHub branches and 31 local branches. Measured on GitHub after the removal: the
remaining branches are `main` (`702ebfe`), `run-intent-2026-10-09` (`1228c8f2`) and `run9a-prep` (`c1c01920`, 180 commits not in
main, deliberately kept). Every deleted branch's last commit, GitHub and local, is an ancestor of GitHub main, so no commit was lost;
each restore command is in the removal log.

## 4. Procedure defects found in the owner's output, and their remedies

1. **A false "PASSED".** The post-merge check was run before the merge. It raised "merged tree is 93b2f138..., not the validated
   5b365c3f", and the next pasted line still printed "PASSED": in an interactive PowerShell session each pasted line runs on its own,
   so a `throw` stops only its own line. Remedy, applied from now on: every command block is one script block, `& { ... }`, so a
   failure stops the block; a block that depends on a merge checks that precondition first and says so in its heading.
2. **A local branch removed that I had said would stay.** The plan listed the LOCAL `run9a-prep` (`c568971a...`, an ancestor of main)
   as merged, and it was deleted; my message had said "run9a-prep stays", meaning the GitHub branch, which does stay. Nothing was lost
   (the commit is in main). Restore if wanted: `git branch run9a-prep c568971adafdc58a0e80c281381300c79dc8b9ad`. Remedy: the probe now
   reports a same-named branch whose local and GitHub tips differ as a separate class that is never removed automatically.
3. **A guessed path.** The block that looked for the store copy of the evidence archive guessed a path inside the run directory and
   found nothing. The run block that published it names the copy `runs\artifact_inventory_20261009T134034Z.evidence.zip`, beside
   the run directory, as every earlier run's archive is in the record (eleven `runs/*.evidence.zip` locations). Remedy: locations are
   taken from the script that wrote them and checked by listing, never guessed.
