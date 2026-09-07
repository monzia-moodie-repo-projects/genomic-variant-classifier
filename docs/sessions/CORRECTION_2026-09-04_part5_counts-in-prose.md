# CORRECTION 2026-09-04 part 5 -- a rule this repository already adopted, and five places that break it

**Author: Monzia Moodie**
**Measured at commit:** e6ad5d3

**Applies to three tracked artifacts, each pinned at all sixty-four characters
and re-verified on disk by the installer:**

```
docs/architecture/decisions/ADR-0001-authority-and-contract-governance.md
    75ac005fcb1ca7ef24a489ab1c61021f9a7db144d8f23e0b00aa2525d92d4413
tests/unit/test_adr_contract.py
    72c63926670dad37f7537a616cd7239d2a9773ad1c4b473eed992d651fd3b488
configs/data_manifest.yaml
    7e64b53432bf9a7a99a4103b31150f6c0877c16858f3a170797d2c2a1e940d18
```

Two further findings named below are NOT pinned, and cannot be.
`AUTHORITATIVE-PROBE-HAS-NO-DEFINED-POPULATION-1` is a finding about an ABSENT
definition -- there is no artifact to pin. `INSTALLER-WALL-CLOCK-COUNT-STALE-1`
concerns three installers that live in `C:\Users\monzi\Downloads`, OUTSIDE the
repository, so the installer's own guard -- which resolves each target against
the repository root -- would fail on a path that does not exist there.

Widening that guard to accept untracked paths would weaken the check that makes
it worth having. Stating the limit is the honest option.

Corrections belong BESIDE records, never inside them. No file named here is
edited by this document.

---

## 0. The rule is not mine, and it is not new

`ADR-0001-authority-and-contract-governance.md`, lines 167 to 176, section
**Counts are rendered, never primary**:

> No count is architecture. `5213` tests, `95` features, `13` models, `22`
> agents, `54` open items -- each is a measurement of a state at a time. Counts
> belong in executable contracts that enforce them and in generated summaries
> that display them. **They do not belong in identity prose.**
>
> This does not weaken any fail-loud contract. `EXPECTED_TABULAR_FEATURE_COUNT`
> and `tests/EXPECTED_SUITE_SIZE` remain enforcing invariants. The rule is that
> **prose must not become a second, unenforced authority for the same number.**

I registered five findings across this session as five separate documentary
defects. They are ONE defect with five instances, and the repository named the
class before I encountered any of them. `CLAUDE.md`'s stale feature count,
repaired at `2c94ae3` this morning, was a SIXTH instance of the same rule --
repaired without noticing the rule that forbade it.

The ADR even uses `5213` as its example. That is the figure the superseded
starting prompt still carried while the ratchet read 6237.

---

## 1. `ADR-CONTRACT-DOCSTRING-COUNTS-STALE-1` -- inside the enforcement itself

`tests/unit/test_adr_contract.py`, 15,870 bytes, 390 lines, digest
`72c63926670dad37f7537a616cd7239d2a9773ad1c4b473eed992d651fd3b488`, read in
full at `e6ad5d3`.

Line 51, in the MODULE DOCSTRING, undated:

> Five of the twelve tests are negative controls.

MEASURED from the parse tree and the file's own section structure:

```
test functions                                          15
negative controls under section 3, "Negative controls"   5
    test_the_parser_rejects_the_byline_as_a_metadata_field        292
    test_the_parser_rejects_a_field_with_an_empty_value           305
    test_the_parser_ignores_metadata_that_has_drifted_out_of_...  314
    test_the_domain_check_rejects_an_unknown_domain               322
    test_the_filename_check_rejects_near_misses                   332
negative control under section 4                                  1
    test_the_index_check_detects_a_missing_and_a_phantom_entry    378
                                                          --
TOTAL negative controls                                    6
```

So BOTH numbers are stale: twelve should be fifteen, five should be six.

**A HEURISTIC THAT FOUND ONE OF SIX.** A first scan looked for `pytest.raises`
in the body or the phrase in the docstring, and found ONE. The file explains
why at line 112: *Parsing. Factored out so the negative controls exercise the
SHIPPING code.* They assert on `parse_header`, `unknown_domains` and `FILENAME`
directly and raise nothing. Only reading the file settled it -- the same
three-way ambiguity as the `preflight_data_guard` count, where a scan could not
distinguish a real absence from a scan too narrow to see.

**WHAT IS NOT WRONG IN THAT FILE.** Line 13, *Measured on 2026-08-22 across all
three accepted records*, and line 166, *Three are accepted as of 2026-08-22*,
are DATED. A dated measurement is a record of a state at a time, which is
exactly what ADR-0001 permits. Only the undated line 51 breaks the rule.

---

## 2. `ADR-0001-EVIDENCE-INGEST-MANIFEST-IS-UNBUILT-1` -- WITHDRAWN AND RESTATED

I registered this as `ADR-0001-DECISION-MANIFEST-IS-UNBUILT-1`, describing a
*decision-sequence preservation manifest*, and attributed it to
`ADR-0001-repository-record-roles.md`. **That file does not exist.** The record
is `ADR-0001-authority-and-contract-governance.md`, 243 lines, and its
specification is about EVIDENCE INGEST, not about decisions as a record class.

Lines 160 to 165, verbatim:

> Preserved copies are renamed on ingest to `decision_<NN>_<YYYY-MM-DD>.txt` and
> recorded in a manifest carrying the original filename, the receipt date, the
> SHA-256, the byte count, the line-ending kind, whether the file ends with a
> newline, and explicit `supersedes` / `superseded_by` edges. The bytes are
> preserved exactly; only the filename disambiguates.

MEASURED at `e6ad5d3` over every tracked file:

```
files named decision_<NN>_<YYYY-MM-DD>.txt          NONE
files named output_<NN>...                          NONE
tracked files under records/                        20
    all install attestations, one manifest.json, one reconstruction
```

`records/attestations/installations/manifest.json` is the INSTALLATION
ATTESTATION archive manifest -- a different artifact for a different plane, and
not this.

The rolling-name sequence itself is REAL: twenty-four `decision.txt` versions
have been received, the most recent at 118,033 bytes, digest
`effad57756bc440fd8b06b6b6401f2ddbc057d1a4e19b8c04e8ed3ae79adbc21`. **Its
preservation half was never built.**

---

## 3. `AUTHORITATIVE-PROBE-HAS-NO-DEFINED-POPULATION-1` -- open, unchanged

Recorded 2026-09-05. Four candidate definitions of *authoritative probe*, all
measured and all failing: 67 tracked files named like a probe across FIFTEEN
locations, four of them not Python; `scripts/forensics` holding 70 files under
SIXTEEN leading verbs; `git grep` exiting 1 on every registry name -- a
MEASURED ABSENCE, categorically different from the pathspec that exited 0
SILENTLY; and exactly one probe invoked in continuous integration, which is
none of the four migrated ones.

It belongs in this correction because it is the same class: a term used as
though it named a population, with no enforced definition anywhere.

---

## 4. `INSTALLER-WALL-CLOCK-COUNT-STALE-1` -- mine, shipped THREE times

Every installer prints a wall-clock note naming how many acceptance-gate
observations exist. MEASURED across this session's runs:

```
Install_ArtifactKey_2026-09-04   said fifteen   sixteen had run
Install_Correction4_2026-09-04   said fifteen   eighteen had run  -- NOTICED, not fixed
Install_Estate_2026-09-04        said fifteen   nineteen had run  -- INHERITED
```

The true count at this correction is TWENTY, and the observed range is
unchanged at minimum 842.9 seconds, maximum 1355.5 seconds, with warnings at 33
in every one of the twenty.

**WHY IT PROPAGATED.** Each installer is DERIVED from a proven predecessor,
which is the right practice: it avoids a private notion of one shape. But
derivation re-pins DIGESTS and leaves DISPLAY TEXT untouched, so a count in
prose rides along unexamined. I named the defect in the second instance and
shipped it again in the third.

That is precisely ADR-0001's sentence: prose became a second, unenforced
authority for a number that the observation list already holds.

---

## 5. `REFERENCE-DECLARATION-NAMES-A-RETIRED-CONSUMER-1` -- open

`configs/data_manifest.yaml`, the `reference` declaration, line 144:

> Read form: the UNCOMPRESSED
> `data/external/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa` (+
> `.fai`), opened by `seq_windows.open_reference` (pyfaidx, `rebuild=False`) in
> `populate_fasta_seq`. **CODE-REFERENCED -- do NOT rename.**

MEASURED at `e6ad5d3` over every tracked file:

```
seq_windows.py           tracked at: NOWHERE
populate_fasta_seq.py    tracked at: NOWHERE
```

`tests/test_build_seq_windows.py:5` states it: *Phase 3 retires
`data/seq_windows.py` and `data/populate_fasta_seq.py`.*

A live declaration protects a path on behalf of a module that no longer exists
in the repository. The PATH is correct and now populated -- the grch38
consolidation at `70674c1` put the FASTA there -- so the instruction is not
harmful. Its stated REASON is false.

---

## 6. What this decides

Nothing is repaired here. Five findings are restated as one class with the
repository's own name for it, one is WITHDRAWN and correctly re-attributed, and
one is newly registered against my own tooling.

A repair for the class would be an executable check -- the shape ADR-0001
prescribes when it says counts belong *in executable contracts that enforce
them*. `tests/unit/test_claude_md_claims.py`, landed at `2c94ae3`, is that
shape for one document. Whether the same binding should extend to test
docstrings and installer prose is a design question this correction does not
answer.

---

## 7. What is NOT claimed

That the six negative controls are correctly SIZED or CORRECTLY CHOSEN. They
were counted and located; whether six is the right number for fifteen tests is
not a question a count can answer.

That `records/` should contain an evidence-ingest manifest. ADR-0001 specifies
one; whether it should be built, amended or withdrawn is a decision.

That the dated counts anywhere were audited. Only `test_adr_contract.py`'s were
examined, and its two dated statements were found CORRECT for their dates.

That the installer wall-clock note should be removed. It carries real
information -- the observed range and the refusal to claim an interval -- and
only its COUNT is stale.

That any other artifact in the repository carries a count in identity prose. No
repository-wide census for this class was run. Five instances are named because
five were encountered, not because five is the total.
