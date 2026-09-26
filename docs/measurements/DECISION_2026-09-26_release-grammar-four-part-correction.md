# DECISION 2026-09-26 -- release names with four components remain review items (correction)

Author: Monzia Moodie. Records a correction the owner made on 2026-09-26, before change B implements
the source monitor's release grammar.

## What is corrected

The owner's review package `GVC_approval_control_refinements_2026-09-25` has a README whose section B
(line 114) says to "accept the deliberately supported two/three/four-component forms", and also permits
unrestricted trailing-zero normalisation. That README is identified by SHA-256
`6b92da9ae8196940d25ced5c0de8926521a5993ce09e5f1a0e97b48dba09e114` (recomputed from the package, 2026-09-26).

The owner's ruling of 2026-09-26 (`decision.txt`, SHA-256
`8ee408fddf39b785118219e3018c20480f2eb82ff5bf56faaa2ca0d2c5fd1f7a`, 344 lines, preserved 2026-09-26T00:58:21Z)
calls that sentence a mistake and supersedes it on exactly two points: four-component acceptance and
unrestricted trailing-zero normalisation. The release-policy ruling of 2026-09-25 (decision 4) governs.

## The rule change B implements

- Only two- and three-component stable names are ordered automatically.
- `4.1.1.0`, `4.1.1.1`, `4.2.0.0` and `v4.1.1.0` are **unsupported release names**: explicit review findings
  (exit 1), never newer, never dropped.
- The component count is validated **before** any normalisation: trimming trailing zeros must never turn
  `4.1.1.0` into an accepted `4.1.1`. (Measured on main 38987f54: `4.1.1.0` currently raises a false
  "newer than the approved 4.1.1" alert, because the tuple (4, 1, 1, 0) sorts after (4, 1, 1).)
- `4.2` and `4.2.0` share one ordering key only; each raw name keeps its own finding.
- The grammar identity joins the monitor's interpretation fingerprint (change B).

## What this does not change

Nothing in the monitor's behaviour changes until change B. The recorded 4.1.1 approval
(docs/approvals/APPROVAL_2026-09-24_gnomad-4.1.1.json) is unaffected.
