# Package 1 -- the prediction contract

Author: Monzia Moodie

Package 1 makes the stored convex blend the single authoritative score and makes every score traceable to the fitted predictor
that produced it. It is delivered in three sequential sets; each leaves every affected, reachable operation either correct or
explicitly refused (owner ruling 2026-11-07, decision 38f15c7a).

| File | What it is |
|---|---|
| `PACKAGE1_ACCEPTANCE_MATRIX_2026-11-07.md` | The frozen acceptance matrix (revision 1), preserved unchanged: sha256 3669fe07275edadba407f8c160fd8ded1d41edb186e0099f851ef1c009a479f3 |
| `PACKAGE1_ACCEPTANCE_MATRIX_2026-11-07_rev2.md` | Revision 2, a superseding amendment that records decisions ccffb8cf (the exact kernel; census and entrypoint closure) and 38f15c7a (three sets with transition containment; record-to-artifact binding): sha256 fa0b3b8925305f3351a1fb5f77de8a551cc7d42da065474c56825806cf287192 |
| `PACKAGE1_SET1A_CENSUS_LEDGER.json` | One disposition after set 1a for each of the 282 score-surface candidates of the census (revision 5), with the set's changed paths; built by a program from the census and the final change set, never by hand |

Measurement: `docs/measurements/MEASUREMENT_2026-11-07_exact-blend-kernel-run15.md`. Session record:
`docs/sessions/SESSION_2026-11-07_prediction-contract-1a.md`.

| Set | Implements | State after merge |
|---|---|---|
| 1a | Contract core, exact kernel, model adapters, serving, readiness gate, record-to-artifact binding, transition refusals | Supported prediction paths use the authoritative contract; incompatible consumers cannot execute or publish |
| 1b | Original out-of-fold identities, completion manifest, explanation and calibration writers | Migrated writers consume bound predictions and name their target; the rest stay refused |
| 1c | Exporter metadata, threshold workflow, validators, forensic consumers | Every in-scope consumer delegates or keeps a documented refusal; package 1 closes |

A gate changes only by a recorded owner ruling. Rollback respects the order: 1a is not reverted alone after 1b or 1c has landed.
