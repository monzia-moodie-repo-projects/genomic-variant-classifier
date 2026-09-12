# SESSION 2026-09-11 to 2026-09-12 -- measurement overturns the claim, five times

Predecessor at the start: `cba2dfe7457f2eb446044b1cb48085d93aaf6a88`
Head at the end:          `58886e34c2d551636902a4237498aba8c2042c9b`

Four units installed. Five confident claims overturned by reading the artifact
they described. Three fabricated comparison values recorded against myself.

## 0. What this session did

    N  cba2dfe7 -> 324592cf  one evidence field, read two ways, reconciled
    C  324592cf -> d181e739  collection bytes stop being discarded
    E  d181e739 -> 5d45487a  a verifier that replays the bytes it judges
    W  5d45487a -> 58886e34  documentation corrected to agree with its code

Suite: 6470 -> 6481 -> 6498 -> 6528 -> 6528.
Every count MEASURED at the live repository inside a rolled-back transaction.

## 1. Unit N -- one field, two incompatible readings

MEASURED 2026-09-11 against the owner installed at 07bc7a5: a
DELIBERATE_RETIREMENT with `added=(B, B)` and `removed=(A, A)` at counts 1 and
1 was EMITTED as an attestation record. Membership comparison collapsed the
duplicates to sets while the arithmetic read their lengths as 2 and 2. Both
errors cancelled.

The repair keeps a frozenset as the correct object for a DECLARATION --
membership is what a declaration claims -- and requires a list or tuple for an
OBSERVATION, because one arriving deduplicated has destroyed the evidence the
check exists to read.

A sequence-only constructor was CONSIDERED AND REJECTED. MEASURED:
`tuple(set([A, A, B]))` passes a list-or-tuple check. The duplicates vanished
before the boundary. Provenance needs a controlled producer, not a type.

All EIGHT previously installed installers' actual ADDED_NODEIDS values were
parsed from source and constructed against the new contract: 0 refused.

## 2. Unit C -- the bytes existed and were thrown away

MEASURED at d181e739 in the installer at 66772b3a:

    collect_output  line 388  capture_output=True -- THE BYTES EXIST
                    line 392  proc.stdout.decode("utf-8", "replace")
                              returns str; `proc` leaves scope; bytes gone

Three call sites; `run_gate` has the same shape for execution evidence. The
unit-O interpretation migration had to be RECONSTRUCTED from ten identities
recorded in prose because this output was never kept.

### The first apply was refused by the gate

4 failed, 6479 passed. All four were this unit's OWN tests:

    assert b't.py::test_a\r\n' == b't.py::test_a\n'

Python's stdout is a TEXT stream, and Windows text mode translates. THE
COMPONENT WAS CORRECT -- it retained exactly what the child produced. The
fixture demanded POSIX bytes from a Windows child and now writes through
`sys.stdout.buffer`.

## 3. Unit E -- the sink overwrote its own evidence

MEASURED 2026-09-12 against the sink installed at d181e739. Two captures, same
phase, IDENTICAL STDOUT, different stderr:

    same retained stderr path : True
    first attempt's stderr file now contains: b'second'
    first attempt's evidence still matches   : False

No SHA-256 collision. Identical stdout across repeated collection is ORDINARY.
The stem was `phase-stdout_sha256[:16]` and every write overwrote. CONTENT
DIGESTS IDENTIFY ARTIFACTS; ATTEMPT IDENTIFIERS DISTINGUISH EXECUTIONS.

THE SEVENTEEN INSTALLED CAPTURE TESTS PASSED AGAINST THE DEFECT AND AGAINST ITS
REPAIR. They never exercised repeated capture, so the suite could not tell the
implementations apart. Passing tests are not coverage.

### Four admission defects, each reproduced first

    a gate naming candidate C with an after-observation naming X  ACCEPTED
    a baseline expectation with no suite pin                      ACCEPTED
    load_bound accepting duplicate keys and NaN                   ACCEPTED
    CollectionEvidenceUnavailable defined and raised ZERO times

Acceptance now takes RAW BYTE BUNDLES and reconstructs the snapshots itself.

### The gate found a platform difference, and the obvious fix was wrong

    test_a_directory_key_is_refused
    PermissionError: [Errno 13] ... '\evidence\sub'

Opening a directory raises IsADirectoryError on POSIX and PermissionError on
Windows. CATCHING PermissionError TOO WOULD HAVE BEEN WRONG: it maps a real
denial onto "names a directory" and destroys the absence-versus-failure
distinction. The condition is now TESTED with `path.is_dir()`.

## 4. Unit W -- two sentences that denied their own code

`enrich_gene_counts` said "the count uses only labeled rows, not the test set".
TEST ROWS ARE LABELED ROWS. Under the gene-disjoint split a held-out gene's
count derives entirely from held-out labels. That is INCIDENT_2026-06-13, which
the project measured at lone-feature test AUROC 0.7181 corpus-wide against
0.5000 train-only, and repaired in BOTH split paths.

`scripts/train.py` defaulted `--clinvar` to `clinvar_grch38.parquet`, which
`_assert_clean_cohort` REFUSES -- 13,295 rows ending ':na:na' and 1,311
identifiers covering more than one accession, measured 2026-09-12. That is
INCIDENT_2026-05-31_null-key-leak.

### A reconstructed preimage, accepted by the repository

A PowerShell console pipeline decoded git's output as code page 437, turning
each of six em-dashes (U+2014, bytes E2 80 94) into three characters for 18
bytes of excess over the committed 73,528. Reversing the misdecode reproduced
the committed digest EXACTLY, and the installer's precondition check -- pinned
to the repository's own value -- confirmed it.

## 5. Five claims overturned by measurement

| I claimed | measured |
| --- | --- |
| nine ratchet entries carry unit T's prose | EIGHT; P and S predate T |
| unit W wrote ADDITION for a NEUTRAL transition | W wrote NO ENTRY |
| the trainer's cohort is not the file I measured | train.py:98 names exactly it |
| identifier merging has no fix anywhere | `_assert_clean_cohort` refuses it |
| target leakage, demonstrated | already repaired in both split paths |

The last two matter most. Every cohort defect measured this session was one the
project had already found, dated, fixed and guarded. The scientific
contribution was two false sentences corrected -- not a discovery.

## 6. Three fabricated comparison values

Recorded against myself. A 40-character string presented as a SHA-256 in the
register review; `29124 B / 6fd02e5d` checked against unit I's attestation; and
`4dc4d5a0c1e27bbb` reported as unit W's dry-run digest when the transcript says
`b23e59b907c3daa3`.

Every instance was caught by re-reading the artifact from disk. COMPARISON
VALUES MUST BE READ, NEVER RECALLED.

## 7. Findings

### Repaired

- duplicated difference evidence in the transition projection -- unit N.
- collection evidence discarded at the point of capture -- unit C.
- the sink overwriting earlier attempts; admission accepting a replacement as
  net-zero; the optional baseline pin -- unit E.
- two documentation sentences contradicting their code -- unit W.

### Open

- `integration_not_done`: the producer and the verifier are each qualified and
  NOTHING CONNECTS THEM. Until an installer captures through the sink and an
  operation retrieves through the reader into acceptance, the capability is
  verified, not operational.
- `installer_template_duplication`: nine edited copies of one engine. The
  template is corrected for future derivations; the nine are unchanged.
- `ratchet_log_misattribution`: eight historical entries carry unit T's
  description. Correcting an append-only record of what units did is a
  judgment about that log's purpose.
- `parser_strip_aliases_trailing_space`: unchanged; a separately measured
  interpretation change.

### Measured, not a defect

`tests/EXPECTED_SUITE_SIZE` is 7,984 lines and 488,239 bytes holding exactly
one non-comment value: 6528. `conftest.py` reads one number.

## 8. Ending state

    HEAD           58886e34c2d551636902a4237498aba8c2042c9b
    suite          6528 collected; 6513 passed, 15 skipped, 0 failed
    working tree   clean, untracked included

## 9. Next intended action

Decisions, not measurements: whether to correct the eight historical ratchet
entries; whether to consolidate nine installers into one engine; the
integration connecting capture to acceptance; and on the scientific side
`Q-tier-remedy` and the disposition of identifiers carrying multiple label
values.

## 10. Remaining uncertainty

- Which cohort a real training run consumed. `train.py` names
  `clinvar_grch38.parquet` by default and the pipeline REFUSES it, so any
  successful run passed `--clinvar` explicitly. What it passed is unrecorded
  here.
- Whether `n_pathogenic_in_gene` reaches a model at all. It is computed,
  corrected, and ablated; whether it is in TABULAR_FEATURES was not measured.
- `split_protocol_v2` supports five- and six-way schemas and two modes. Which
  protocol any run used is unmeasured, so the 334-identifier figure describes
  the legacy three-way partition only.
