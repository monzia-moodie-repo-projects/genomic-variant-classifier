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
| `PACKAGE1_SET1B_DELIVERY.md` | Set 1b against the gates (G2, G7.3, G8 and the 1b row of G12), the refinements implemented for the owner's ruling, and the findings recorded but not fixed |
| `PACKAGE1_SET1B_CENSUS_LEDGER.json` | The disposition after set 1b of each of the 282 candidates, re-checked for every name whose behaviour or presence 1b changes; built by a program from the set-1a ledger and the final change set |
| `RUN15_OOF_IDENTITY_DERIVED_RECORD.json` | The separately identified derived record of Run 15's out-of-fold row identity (gate G2.3) and the code provenance of its ENSEMBLE_STACKER rows (G7.3), derived from the artifact by `scripts/derive_run15_oof_identity.py`; the original OOF file is never rewritten |

Measurement: `docs/measurements/MEASUREMENT_2026-11-07_exact-blend-kernel-run15.md`. Session records:
`docs/sessions/SESSION_2026-11-07_prediction-contract-1a.md`, `docs/sessions/SESSION_2026-11-08_prediction-contract-1b.md`.

| Set | Implements | State after merge |
|---|---|---|
| 1a | Contract core, exact kernel, model adapters, serving, readiness gate, record-to-artifact binding, transition refusals | Supported prediction paths use the authoritative contract; incompatible consumers cannot execute or publish |
| 1b | Original out-of-fold identities, completion manifest, explanation and calibration writers | Migrated writers consume bound predictions and name their target; the rest stay refused |
| 1c | Exporter metadata, threshold workflow, validators, forensic consumers | Every in-scope consumer delegates or keeps a documented refusal; package 1 closes |

A gate changes only by a recorded owner ruling. Rollback respects the order: 1a is not reverted alone after 1b or 1c has landed.
