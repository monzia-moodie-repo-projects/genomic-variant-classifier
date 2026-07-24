#!/usr/bin/env python3
"""Does the cohort's label column agree with the term sets production matches against?

WHY THIS SCRIPT EXISTS
======================
`real_data_prep._load_and_label` assigns the binary training label with an EXACT,
CASE-SENSITIVE membership test against two hardcoded term sets. It applies
`.fillna("").str.strip()` first and nothing else -- no lowercasing, no
underscore folding. If the cohort stores its clinical significance in any other
spelling, every row silently fails both tests, receives no label, and is dropped
by the `notna()` filter a few lines later. Nothing raises. Nothing warns. The
cohort simply comes out smaller.

`docs/CONTAINMENT_2026-07-24_R2.md` section 8 named that possibility the single
question outranking every other open item, on the grounds that a repair built on
a cohort whose labelling is broken would have to be redone.

WHAT THIS SCRIPT SUPERSEDES
---------------------------
An earlier, undated-in-repository probe (`probe_label_column_terms_2026-07-24.py`,
SHA-256 B818E323...54AB60D0) was run twice on 2026-07-24 from a Downloads folder,
was never committed, and its output was never written to a file. It carried two
defects that this script exists to remove:

  1. It read ONE hardcoded cohort artifact,
     `data/processed/clinvar_grch38_clean.parquet`. That is not the artifact
     `real_data_prep` defaults to (`clinvar_grch38.parquet`, line 29) nor the one
     `preflight_run16_inputs` defaults to (`clinvar_grch38_clean_seq.parquet`,
     line 176). The measurement therefore covered one of at least three
     candidates and could not answer the question it was asked.
  2. It exited 1 on ANY divergence, including a divergence confined to a column
     production does not label from. Its own closing text said the blocking
     condition was the labelling column specifically. An operator following exit
     codes would have halted for a non-blocker. A fail-loud guard that fails for
     a false reason is worse than no guard, because it trains the operator to
     ignore it -- the Run-16 preflight lesson of 2026-07-20.

DO NOT RUN THE SUPERSEDED COPY AGAIN.

WHAT THIS SCRIPT DOES NOT HARDCODE
----------------------------------
Two things that a previous generation of tooling in this project got wrong by
copying values into a checker and letting them drift:

  * THE TERM SETS. `PATHOGENIC_TERMS` and `BENIGN_TERMS` are read from
    `src/genomic_variant_classifier/data/real_data_prep.py` by abstract syntax
    tree (AST) parse at run time. That module imports XGBoost, LightGBM, CatBoost
    and PyTorch at module scope, so it cannot simply be imported for inspection;
    the AST route is the same one `scripts/preflight_run16_inputs.py` uses. If
    the term sets change, this probe follows them. It cannot go stale.
  * WHICH COLUMN PRODUCTION LABELS FROM. Discovered, not assumed, by locating
    the `.isin(PATHOGENIC_TERMS)` / `.isin(BENIGN_TERMS)` call sites inside
    `_load_and_label` and reading the subscript they are applied to. If more than
    one distinct column is found, or none, the probe refuses to guess and exits
    with the environment code.

EXIT CODES -- THESE DISCRIMINATE, WHICH IS THE POINT
-----------------------------------------------------
    0  No divergence in any inspected column of any inspected cohort.
    1  DIVERGENCE IN THE LABELLING COLUMN of at least one cohort. This is the
       blocker described in CONTAINMENT_2026-07-24_R2 section 8. Production is
       silently dropping labelled variants.
    3  Divergence confined to columns production does NOT label from. A finding
       worth recording -- it usually means an annotation column was written by a
       different normalisation than the one now in force -- but it does not block
       the repair, and it must not be reported as though it did.
    2  Contract or environment failure: the term sets could not be extracted, the
       labelling column could not be identified unambiguously, no cohort artifact
       was found, or a named cohort lacked every inspected column. An absence of
       measurement is never reported as a clean measurement.

READ-ONLY. In normal operation this script opens no file for writing. The single
exception is `--self-test`, which round-trips one small synthetic table through
the system temporary directory in order to exercise the real Parquet read path;
it touches nothing under the repository.

Parquet is read with `pyarrow.parquet.read_table`, never `pandas.read_parquet`,
per commit 644a184 (2026-07-23), which removed a non-deterministic interpreter
abort in `arrow::py::PyReadableFile::~PyReadableFile()` by eliminating the
faulting object rather than suppressing its symptom.

VALIDATION ANCHOR
-----------------
`--self-test` reconstructs the exact value-count distribution measured on
`data/processed/clinvar_grch38_clean.parquet` on 2026-07-24 (4,399,089 rows, the
100 distinct `clinical_sig` values summarised by their measured top twenty plus a
642-row remainder, and the 5 distinct `pathogenicity` values) and asserts that
this script reproduces the three figures that run actually produced:

    clinical_sig   exact 1,686,333 (38.334%)  normalised 1,686,333  difference 0
    pathogenicity  exact         0 ( 0.000%)  normalised 1,848,225  difference +1,848,225

Reproducing an independently measured answer is the only evidence that a
rewritten tool is not worse than the one it replaces.

USAGE (PowerShell 5.1)
----------------------
    python "C:\\Users\\monzi\\Downloads\\probe_label_column_terms.py" --self-test

    python "C:\\Users\\monzi\\Downloads\\probe_label_column_terms.py" `
        --repo "C:\\Projects\\genomic-variant-classifier"

    python "C:\\Users\\monzi\\Downloads\\probe_label_column_terms.py" `
        --repo "C:\\Projects\\genomic-variant-classifier" `
        --cohort "data\\processed\\clinvar_grch38.parquet"

Acronyms on first use. AST = abstract syntax tree. ClinVar = the National Center
for Biotechnology Information's Clinical Variation archive. VCF = variant call
format.

Author: written for Monzia Moodie, 2026-07-24. Supersedes
probe_label_column_terms_2026-07-24.py.
"""
from __future__ import annotations

import argparse
import ast
import logging
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

EXIT_CLEAN = 0
EXIT_LABEL_COLUMN_DIVERGES = 1
EXIT_ENVIRONMENT = 2
EXIT_NON_LABEL_DIVERGENCE = 3

REAL_DATA_PREP_REL = "src/genomic_variant_classifier/data/real_data_prep.py"
LABEL_FN_NAME = "_load_and_label"
TERM_SET_NAMES = ("PATHOGENIC_TERMS", "BENIGN_TERMS")

#: Cohort artifacts inspected when --cohort is not given. Each is a real default
#: somewhere in the tree; the probe reports which of them exist rather than
#: assuming one. Ordered most-canonical-claim first.
DEFAULT_COHORTS = (
    "data/processed/clinvar_grch38.parquet",             # real_data_prep.py:29
    "data/processed/clinvar_grch38_clean.parquet",       # probe_tier_filter_impact.py:142
    "data/processed/clinvar_grch38_clean_seq.parquet",   # preflight_run16_inputs.py:176
)

#: Columns inspected when present. The labelling column is discovered, not taken
#: from this list; this list only decides what else gets reported alongside it.
CANDIDATE_LABEL_COLUMNS = ("clinical_sig", "pathogenicity", "clnsig", "significance")

_WHITESPACE = re.compile(r"\s+")

TOP_N = 20


class ContractError(RuntimeError):
    """The repository does not look the way this probe requires. Exit code 2.

    Distinct from any data finding: a probe that cannot establish what production
    does has not measured anything, and must not print a verdict as though it
    had.
    """


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalise(value: object) -> str:
    """Lowercase, underscores to spaces, collapse whitespace, strip.

    This is deliberately the SAME transformation as
    `genomic_variant_classifier.data.review_status.normalise`. It is duplicated
    here rather than imported because importing anything from the package pulls
    in the heavyweight training stack, and because this probe must be runnable
    from a Downloads folder before the package is even installed. The duplication
    is bounded to four operations and is asserted against the review_status
    implementation by --self-test when that module is importable.
    """
    if value is None:
        return ""
    return _WHITESPACE.sub(" ", str(value).lower().replace("_", " ")).strip()


def production_key(value: object) -> str:
    """Exactly what production compares: `.fillna("").str.strip()`, nothing else.

    real_data_prep.py:510 applies `fillna("")` then `str.strip()` to
    `clinical_sig`, then tests membership. No lowercasing. No underscore folding.
    Reproducing that faithfully is the whole measurement; a probe that normalised
    here would answer a different question and report agreement that production
    does not enjoy.
    """
    if value is None:
        return ""
    return str(value).strip()


# ---------------------------------------------------------------------------
# Reading production's contract out of the source, by AST
# ---------------------------------------------------------------------------
def _literal_string_collection(node: ast.AST) -> tuple[str, ...] | None:
    """Return the strings of a set/list/tuple/frozenset literal, or None."""
    if isinstance(node, ast.Call):
        fn = node.func
        is_frozenset = (isinstance(fn, ast.Name) and fn.id == "frozenset") or (
            isinstance(fn, ast.Attribute) and fn.attr == "frozenset")
        if is_frozenset and len(node.args) == 1:
            return _literal_string_collection(node.args[0])
        return None
    if not isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        return None
    out: list[str] = []
    for elt in node.elts:
        if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
            return None
        out.append(elt.value)
    return tuple(out)


def extract_term_sets(source: str, origin: str) -> dict[str, tuple[tuple[str, ...], int]]:
    """PATHOGENIC_TERMS and BENIGN_TERMS as written, with their line numbers.

    Raises ContractError rather than returning a partial result. A probe that
    silently proceeded with one of the two term sets would report a label count
    that is wrong by the whole benign side and look entirely plausible doing it.
    """
    tree = ast.parse(source, filename=origin)
    found: dict[str, tuple[tuple[str, ...], int]] = {}
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for tgt in targets:
            if not (isinstance(tgt, ast.Name) and tgt.id in TERM_SET_NAMES):
                continue
            value = node.value
            if value is None:
                continue
            strings = _literal_string_collection(value)
            if strings is None:
                raise ContractError(
                    f"{origin}:{node.lineno}: {tgt.id} is not a literal collection of "
                    f"strings, so it cannot be read statically. This probe refuses to "
                    f"guess what production matches against."
                )
            if tgt.id in found:
                raise ContractError(
                    f"{origin}: {tgt.id} is assigned more than once "
                    f"(lines {found[tgt.id][1]} and {node.lineno}). Which one production "
                    f"uses is not statically obvious; refusing to guess."
                )
            found[tgt.id] = (strings, node.lineno)
    missing = [n for n in TERM_SET_NAMES if n not in found]
    if missing:
        raise ContractError(
            f"{origin}: could not find {', '.join(missing)}. Either the term sets were "
            f"renamed or the labelling contract moved. This probe measures nothing until "
            f"that is resolved."
        )
    return found


def extract_label_column(source: str, origin: str) -> tuple[str, tuple[int, ...]]:
    """The column `_load_and_label` actually tests, discovered from the call sites.

    Looks for `<something>[ "<col>" ].isin(PATHOGENIC_TERMS | BENIGN_TERMS)` inside
    the function and reads the subscript. Assumption-free: if the function starts
    labelling from a different column, this probe follows it there.
    """
    tree = ast.parse(source, filename=origin)
    fn: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == LABEL_FN_NAME:
            if fn is not None:
                raise ContractError(
                    f"{origin}: {LABEL_FN_NAME} is defined more than once; refusing to guess."
                )
            fn = node  # type: ignore[assignment]
    if fn is None:
        raise ContractError(
            f"{origin}: no function named {LABEL_FN_NAME}. The labelling entry point "
            f"moved or was renamed; this probe cannot locate the column under test."
        )
    columns: dict[str, list[int]] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "isin"):
            continue
        if not any(isinstance(a, ast.Name) and a.id in TERM_SET_NAMES for a in node.args):
            continue
        sub = func.value
        if not isinstance(sub, ast.Subscript):
            raise ContractError(
                f"{origin}:{node.lineno}: an .isin() against a term set is applied to "
                f"something that is not a simple column subscript, so the column under "
                f"test cannot be read statically."
            )
        key = sub.slice
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            raise ContractError(
                f"{origin}:{node.lineno}: the column subscript is not a string literal."
            )
        columns.setdefault(key.value, []).append(node.lineno)
    if not columns:
        raise ContractError(
            f"{origin}: found {LABEL_FN_NAME} but no `.isin(PATHOGENIC_TERMS)` or "
            f"`.isin(BENIGN_TERMS)` call inside it. The labelling mechanism changed."
        )
    if len(columns) > 1:
        detail = "; ".join(f"{c!r} at line(s) {', '.join(map(str, ls))}"
                           for c, ls in sorted(columns.items()))
        raise ContractError(
            f"{origin}: {LABEL_FN_NAME} labels from MORE THAN ONE column -- {detail}. "
            f"That is itself a finding worth investigating, and this probe will not "
            f"pick one of them silently."
        )
    col, lines = next(iter(columns.items()))
    return col, tuple(sorted(lines))


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------
class ColumnReport:
    """One column of one cohort, measured two ways."""

    def __init__(self, column: str, is_label_column: bool, rows: int,
                 counts: dict[object, int], exact_terms: frozenset[str],
                 normalised_terms: frozenset[str]) -> None:
        self.column = column
        self.is_label_column = is_label_column
        self.rows = rows
        self.counts = counts
        self.distinct = len(counts)
        self.exact_matched = 0
        self.normalised_matched = 0
        self.responsible: list[tuple[object, int]] = []
        for value, n in counts.items():
            hit_exact = production_key(value) in exact_terms
            hit_norm = normalise(value) in normalised_terms
            if hit_exact:
                self.exact_matched += n
            if hit_norm:
                self.normalised_matched += n
            if hit_norm and not hit_exact:
                self.responsible.append((value, n))
        self.responsible.sort(key=lambda kv: -kv[1])

    @property
    def difference(self) -> int:
        return self.normalised_matched - self.exact_matched

    @property
    def diverges(self) -> bool:
        return self.difference != 0

    def _pct(self, n: int) -> float:
        return 0.0 if not self.rows else 100.0 * n / self.rows

    def render(self, out: list[str]) -> None:
        role = "LABELLING COLUMN -- production labels from this" if self.is_label_column \
            else "not the labelling column"
        out.append("=" * 78)
        out.append(f"COLUMN: {self.column}   ({self.rows:,} rows, {self.distinct:,} distinct values)")
        out.append(f"  role: {role}")
        out.append("=" * 78)
        out.append(f"  top {TOP_N} values, as Python reprs so whitespace cannot hide:")
        for value, n in sorted(self.counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[:TOP_N]:
            out.append(f"    {n:>12,}  {value!r}")
        shown = min(TOP_N, self.distinct)
        if self.distinct > shown:
            tail_rows = self.rows - sum(sorted(self.counts.values(), reverse=True)[:shown])
            out.append(f"    ... and {self.distinct - shown:,} further distinct value(s), "
                       f"{tail_rows:,} row(s)")
        out.append("")
        out.append(f"  rows labelled under production's EXACT match : {self.exact_matched:>12,} "
                   f"({self._pct(self.exact_matched):7.3f}%)")
        out.append(f"  rows labelled under a NORMALISED match       : {self.normalised_matched:>12,} "
                   f"({self._pct(self.normalised_matched):7.3f}%)")
        out.append(f"  difference                                   : {self.difference:>+12,}")
        out.append("")
        if not self.diverges:
            out.append("  NO DIVERGENCE: the term sets match this column's values exactly.")
            return
        severity = "BLOCKER" if self.is_label_column else "FINDING (not a blocker)"
        out.append(f"  *** DIVERGENCE -- {severity} ***")
        out.append(f"  {self.difference:,} row(s) would be labelled under a normalised match and")
        out.append("  are NOT labelled by production's exact match. Values responsible:")
        for value, n in self.responsible:
            out.append(f"    {n:>12,}  {value!r}")
        if self.is_label_column:
            out.append("")
            out.append("  Production drops these rows as unlabelled. This is the condition")
            out.append("  CONTAINMENT_2026-07-24_R2 section 8 names as outranking every other")
            out.append("  open item. No repair should begin until it is resolved.")
        else:
            out.append("")
            out.append("  Production does not label from this column, so no training row is")
            out.append("  lost by this divergence. Record it; do not treat it as a blocker.")


def value_counts(table, column: str) -> dict[object, int]:
    """Distinct values and their row counts, via Arrow compute.

    Nulls are counted under the key None so that a null-heavy column cannot be
    mistaken for a sparse one, and so that `production_key(None) == ""` is
    exercised on real data rather than only in the self-test.
    """
    import pyarrow.compute as pc

    arr = table.column(column).combine_chunks()
    counts: dict[object, int] = {}
    vc = pc.value_counts(arr)
    for entry in vc:
        value = entry["values"].as_py()
        counts[value] = counts.get(value, 0) + entry["counts"].as_py()
    return counts


def inspect_cohort(path: Path, label_column: str, exact_terms: frozenset[str],
                   normalised_terms: frozenset[str], out: list[str]) -> list[ColumnReport]:
    """Measure every candidate column present in one cohort artifact."""
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    present = [c for c in CANDIDATE_LABEL_COLUMNS if c in schema.names]
    out.append("")
    out.append("#" * 78)
    out.append(f"COHORT: {path}")
    out.append(f"  size on disk: {path.stat().st_size / (1024 * 1024):,.1f} MiB")
    out.append(f"  label-like columns present: {present if present else 'NONE'}")
    out.append("#" * 78)
    if not present:
        raise ContractError(
            f"{path}: none of {list(CANDIDATE_LABEL_COLUMNS)} is present. This artifact "
            f"cannot be assessed for the labelling question."
        )
    if label_column not in present:
        out.append("")
        out.append(f"  WARNING: production's labelling column {label_column!r} is ABSENT from")
        out.append("  this artifact. real_data_prep would raise a KeyError on it. The columns")
        out.append("  below are reported for completeness only.")
    table = pq.read_table(path, columns=present)   # native Arrow; never pd.read_parquet
    rows = table.num_rows
    reports: list[ColumnReport] = []
    for column in present:
        counts = value_counts(table, column)
        total = sum(counts.values())
        if total != rows:
            raise ContractError(
                f"{path}: value counts for {column!r} sum to {total:,} but the table has "
                f"{rows:,} rows. The count is not trustworthy and no verdict is issued."
            )
        report = ColumnReport(column, column == label_column, rows, counts,
                              exact_terms, normalised_terms)
        out.append("")
        report.render(out)
        reports.append(report)
    return reports


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
#: The distribution measured on data/processed/clinvar_grch38_clean.parquet,
#: 2026-07-24, by the superseded probe. Top twenty of 100 distinct values.
MEASURED_CLINICAL_SIG_TOP20 = {
    "Uncertain significance": 2_293_565, "Likely benign": 1_081_571, "-": 245_148,
    "Benign": 209_078, "Pathogenic": 185_012,
    "Conflicting classifications of pathogenicity": 161_319,
    "Likely pathogenic": 110_527, "Benign/Likely benign": 63_534,
    "Pathogenic/Likely pathogenic": 36_611, "not provided": 6_805,
    "drug response": 1_856, "other": 1_528,
    "no classification for the single variant": 627, "risk factor": 347,
    "association": 324, "Uncertain significance/Uncertain risk allele": 144,
    "no classifications from unflagged records": 133, "Affects": 132,
    "Likely risk allele": 93, "Pathogenic, low penetrance": 93,
}
MEASURED_PATHOGENICITY = {
    "uncertain": 2_550_864, "likely_benign": 1_081_595, "pathogenic": 383_277,
    "benign": 272_688, "likely_pathogenic": 110_665,
}
MEASURED_ROWS = 4_399_089
MEASURED_TAIL_ROWS = MEASURED_ROWS - sum(MEASURED_CLINICAL_SIG_TOP20.values())  # 642

EXPECT_CLINICAL_SIG_EXACT = 1_686_333
EXPECT_CLINICAL_SIG_NORMALISED = 1_686_333
EXPECT_PATHOGENICITY_EXACT = 0
EXPECT_PATHOGENICITY_NORMALISED = 1_848_225


def _synthetic_counts() -> tuple[dict[object, int], dict[object, int]]:
    """The measured distribution, with the 642-row remainder given a filler value.

    The filler must not match any term set under either comparison, which is
    asserted by the self-test rather than assumed -- if it did match, the
    reproduction of the measured figures would be luck.
    """
    cs = dict(MEASURED_CLINICAL_SIG_TOP20)
    cs["__unlisted_tail_filler__"] = MEASURED_TAIL_ROWS
    return cs, dict(MEASURED_PATHOGENICITY)


def _check(ok: bool, label: str, got: object, want: object, failures: list[str]) -> None:
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {label}: got {got!r}, want {want!r}")
    if not ok:
        failures.append(label)


def self_test(repo: Path | None) -> int:
    """Outcome-assert this probe against independently measured ground truth."""
    print("=" * 78)
    print("SELF-TEST -- reproducing the figures measured on 2026-07-24")
    print("=" * 78)
    failures: list[str] = []

    exact_terms = frozenset({"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic",
                             "Benign", "Likely benign", "Benign/Likely benign"})
    normalised_terms = frozenset(normalise(t) for t in exact_terms)

    print("\n-- arithmetic of the fixture itself --")
    cs_counts, pg_counts = _synthetic_counts()
    _check(sum(cs_counts.values()) == MEASURED_ROWS, "clinical_sig fixture sums to cohort rows",
           sum(cs_counts.values()), MEASURED_ROWS, failures)
    _check(sum(pg_counts.values()) == MEASURED_ROWS, "pathogenicity fixture sums to cohort rows",
           sum(pg_counts.values()), MEASURED_ROWS, failures)
    _check(MEASURED_TAIL_ROWS == 642, "unlisted tail", MEASURED_TAIL_ROWS, 642, failures)
    filler = "__unlisted_tail_filler__"
    _check(production_key(filler) not in exact_terms and normalise(filler) not in normalised_terms,
           "tail filler matches no term set under either comparison", "no match", "no match",
           failures)

    print("\n-- clinical_sig, measured two ways --")
    cs = ColumnReport("clinical_sig", True, MEASURED_ROWS, cs_counts, exact_terms, normalised_terms)
    _check(cs.exact_matched == EXPECT_CLINICAL_SIG_EXACT, "clinical_sig exact",
           cs.exact_matched, EXPECT_CLINICAL_SIG_EXACT, failures)
    _check(cs.normalised_matched == EXPECT_CLINICAL_SIG_NORMALISED, "clinical_sig normalised",
           cs.normalised_matched, EXPECT_CLINICAL_SIG_NORMALISED, failures)
    _check(cs.difference == 0, "clinical_sig difference", cs.difference, 0, failures)
    _check(f"{cs._pct(cs.exact_matched):.3f}" == "38.334", "clinical_sig exact percent",
           f"{cs._pct(cs.exact_matched):.3f}", "38.334", failures)
    _check(cs.diverges is False, "clinical_sig does not diverge", cs.diverges, False, failures)

    print("\n-- pathogenicity, measured two ways --")
    pg = ColumnReport("pathogenicity", False, MEASURED_ROWS, pg_counts, exact_terms, normalised_terms)
    _check(pg.exact_matched == EXPECT_PATHOGENICITY_EXACT, "pathogenicity exact",
           pg.exact_matched, EXPECT_PATHOGENICITY_EXACT, failures)
    _check(pg.normalised_matched == EXPECT_PATHOGENICITY_NORMALISED, "pathogenicity normalised",
           pg.normalised_matched, EXPECT_PATHOGENICITY_NORMALISED, failures)
    _check(pg.difference == EXPECT_PATHOGENICITY_NORMALISED, "pathogenicity difference",
           pg.difference, EXPECT_PATHOGENICITY_NORMALISED, failures)
    _check(f"{pg._pct(pg.normalised_matched):.3f}" == "42.014", "pathogenicity normalised percent",
           f"{pg._pct(pg.normalised_matched):.3f}", "42.014", failures)
    _check(len(pg.responsible) == 4, "pathogenicity responsible value count",
           len(pg.responsible), 4, failures)
    _check(sum(n for _, n in pg.responsible) == EXPECT_PATHOGENICITY_NORMALISED,
           "responsible values sum to the difference",
           sum(n for _, n in pg.responsible), EXPECT_PATHOGENICITY_NORMALISED, failures)

    print("\n-- exit-code discrimination (the defect this probe was written to remove) --")
    _check(decide_exit([cs, pg]) == EXIT_NON_LABEL_DIVERGENCE,
           "divergence only outside the labelling column -> 3",
           decide_exit([cs, pg]), EXIT_NON_LABEL_DIVERGENCE, failures)
    cs_bad = ColumnReport("clinical_sig", True, MEASURED_ROWS,
                          {"likely_pathogenic": 100, "Benign": 50, "Uncertain significance": 850},
                          exact_terms, normalised_terms)
    _check(decide_exit([cs_bad]) == EXIT_LABEL_COLUMN_DIVERGES,
           "divergence in the labelling column -> 1",
           decide_exit([cs_bad]), EXIT_LABEL_COLUMN_DIVERGES, failures)
    _check(decide_exit([cs]) == EXIT_CLEAN, "no divergence anywhere -> 0",
           decide_exit([cs]), EXIT_CLEAN, failures)
    _check(decide_exit([]) == EXIT_ENVIRONMENT, "nothing measured -> 2",
           decide_exit([]), EXIT_ENVIRONMENT, failures)

    print("\n-- normalisation boundary cases --")
    for raw, want in (("Likely_Pathogenic", "likely pathogenic"),
                      ("  LIKELY   BENIGN  ", "likely benign"),
                      ("Pathogenic/Likely_pathogenic", "pathogenic/likely pathogenic"),
                      (None, ""), ("", ""), ("_", "")):
        _check(normalise(raw) == want, f"normalise({raw!r})", normalise(raw), want, failures)
    for raw, want in ((None, ""), ("  Pathogenic  ", "Pathogenic"), ("Pathogenic", "Pathogenic")):
        _check(production_key(raw) == want, f"production_key({raw!r})",
               production_key(raw), want, failures)
    _check(production_key("pathogenic") not in exact_terms,
           "production's match is case-SENSITIVE", "no match", "no match", failures)

    print("\n-- the real Parquet read path, round-tripped through a temporary file --")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        values, counts = [], []
        for v, n in list(cs_counts.items())[:5]:
            values.append(v); counts.append(n)
        col = []
        for v, n in zip(values, [1, 2, 3, 4, 5]):
            col.extend([v] * n)
        col.append(None)
        table = pa.table({"clinical_sig": pa.array(col, type=pa.string())})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "roundtrip.parquet"
            pq.write_table(table, p)
            back = pq.read_table(p, columns=["clinical_sig"])
            vc = value_counts(back, "clinical_sig")
            _check(sum(vc.values()) == back.num_rows, "value_counts sums to row count",
                   sum(vc.values()), back.num_rows, failures)
            _check(None in vc and vc[None] == 1, "a null row is counted, not dropped",
                   vc.get(None), 1, failures)
    except Exception as exc:                                    # noqa: BLE001
        _check(False, "Parquet round-trip", f"{type(exc).__name__}: {exc}", "success", failures)

    print("\n-- normalise() agrees with the package's single source of truth --")
    agreed = None
    if repo is not None:
        rs = repo / "src" / "genomic_variant_classifier" / "data" / "review_status.py"
        if rs.is_file():
            try:
                ns: dict[str, object] = {}
                exec(compile(rs.read_text(encoding="utf-8"), str(rs), "exec"), ns)
                other = ns["normalise"]
                agreed = all(other(r) == normalise(r) for r in                 # type: ignore[operator]
                             ["Criteria_Provided,_Single_Submitter", "  A  B  ", "", "_", None,
                              "Likely_Pathogenic", "PATHOGENIC"])
            except Exception as exc:                            # noqa: BLE001
                print(f"  [SKIP] review_status.py present but not evaluable: "
                      f"{type(exc).__name__}: {exc}")
    if agreed is None:
        print("  [SKIP] review_status.py not present (it is untracked as of 2026-07-24);")
        print("         this check arms itself automatically once that module lands.")
    else:
        _check(agreed, "normalise() matches review_status.normalise()", agreed, True, failures)

    print()
    print("=" * 78)
    if failures:
        print(f"SELF-TEST FAILED -- {len(failures)} check(s): {', '.join(failures)}")
        print("=" * 78)
        return 1
    print("SELF-TEST PASSED -- every figure measured on 2026-07-24 is reproduced,")
    print("and each exit code is produced by the condition it is supposed to signal.")
    print("=" * 78)
    return 0


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def decide_exit(reports: Iterable[ColumnReport]) -> int:
    """Blocker, finding, clean, or nothing-measured -- as four distinct codes."""
    reports = list(reports)
    if not reports:
        return EXIT_ENVIRONMENT
    if any(r.diverges and r.is_label_column for r in reports):
        return EXIT_LABEL_COLUMN_DIVERGES
    if any(r.diverges for r in reports):
        return EXIT_NON_LABEL_DIVERGENCE
    return EXIT_CLEAN


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Does the cohort's label column agree with production's term sets?")
    ap.add_argument("--repo", type=Path, default=None,
                    help="repository root; defaults to this script's parent repository")
    ap.add_argument("--cohort", action="append", default=None, metavar="PATH",
                    help="cohort Parquet, repeatable; relative to --repo or absolute. "
                         "Default: every artifact in DEFAULT_COHORTS that exists.")
    ap.add_argument("--self-test", action="store_true",
                    help="reproduce the 2026-07-24 measurements and the exit-code matrix")
    args = ap.parse_args(argv)

    repo = args.repo
    if repo is None:
        guess = Path(__file__).resolve().parents[1]
        repo = guess if (guess / REAL_DATA_PREP_REL).is_file() else None
    if repo is not None:
        repo = repo.resolve()

    if args.self_test:
        return self_test(repo)

    if repo is None:
        print("ERROR: could not infer the repository root. Pass --repo explicitly, for example",
              file=sys.stderr)
        print('       --repo "C:\\Projects\\genomic-variant-classifier"', file=sys.stderr)
        return EXIT_ENVIRONMENT

    out: list[str] = []
    try:
        rdp = repo / REAL_DATA_PREP_REL
        if not rdp.is_file():
            raise ContractError(f"{rdp} does not exist; --repo does not point at the repository.")
        source = rdp.read_text(encoding="utf-8")
        terms = extract_term_sets(source, REAL_DATA_PREP_REL)
        label_column, label_lines = extract_label_column(source, REAL_DATA_PREP_REL)

        exact_terms = frozenset(t for pair in terms.values() for t in pair[0])
        normalised_terms = frozenset(normalise(t) for t in exact_terms)

        out.append("=" * 78)
        out.append("LABEL COLUMN AND TERM-SET CHECK")
        out.append("READ-ONLY. Writes nothing.")
        out.append("=" * 78)
        out.append(f"  repository : {repo}")
        out.append("")
        out.append("  Production's labelling contract, read from source by AST at run time")
        out.append("  (never hardcoded here, so it cannot go stale):")
        out.append(f"    column under test : {label_column!r}")
        out.append(f"    tested at         : {REAL_DATA_PREP_REL}:"
                   f"{', '.join(map(str, label_lines))}")
        out.append(f"    match rule        : exact, case-sensitive, after "
                   f'.fillna("").str.strip()')
        for name in TERM_SET_NAMES:
            strings, lineno = terms[name]
            out.append(f"    {name} ({REAL_DATA_PREP_REL}:{lineno}) = {sorted(strings)}")

        cohort_args = args.cohort if args.cohort else list(DEFAULT_COHORTS)
        resolved: list[Path] = []
        skipped: list[str] = []
        for c in cohort_args:
            p = Path(c)
            p = p if p.is_absolute() else (repo / c)
            (resolved if p.is_file() else skipped).append(p if p.is_file() else c)  # type: ignore[arg-type]
        out.append("")
        out.append(f"  cohorts to inspect : {len(resolved)}")
        for p in resolved:
            out.append(f"    PRESENT  {p}")
        for c in skipped:
            out.append(f"    ABSENT   {c}")
        if not resolved:
            raise ContractError(
                "no cohort artifact found. Nothing was measured, and an absence of "
                "measurement is not a clean result."
            )

        all_reports: list[ColumnReport] = []
        per_cohort: list[tuple[Path, int]] = []
        for p in resolved:
            reports = inspect_cohort(p, label_column, exact_terms, normalised_terms, out)
            all_reports.extend(reports)
            per_cohort.append((p, decide_exit(reports)))

        code = decide_exit(all_reports)
        out.append("")
        out.append("=" * 78)
        out.append("VERDICT")
        out.append("=" * 78)
        for p, c in per_cohort:
            meaning = {EXIT_CLEAN: "clean",
                       EXIT_LABEL_COLUMN_DIVERGES: "BLOCKER -- labelling column diverges",
                       EXIT_NON_LABEL_DIVERGENCE: "finding -- divergence outside the labelling column",
                       EXIT_ENVIRONMENT: "nothing measured"}[c]
            out.append(f"  exit {c}  {meaning}")
            out.append(f"          {p}")
        out.append("")
        out.append(f"  AGGREGATE EXIT {code} -- the worst verdict across all cohorts inspected.")
        if code == EXIT_LABEL_COLUMN_DIVERGES:
            out.append("  Production is silently dropping labelled variants. No repair should")
            out.append("  begin until this is resolved (CONTAINMENT_2026-07-24_R2 section 8).")
        elif code == EXIT_NON_LABEL_DIVERGENCE:
            out.append("  The labelling column is sound in every cohort inspected. The")
            out.append("  divergence lies in a column production does not label from, so it")
            out.append("  is a finding to record, NOT a blocker on the repair.")
        elif code == EXIT_CLEAN:
            out.append("  No divergence in any inspected column of any inspected cohort.")
        print("\n".join(out))
        return code
    except ContractError as exc:
        print("\n".join(out))
        print("")
        print("=" * 78)
        print("CONTRACT FAILURE -- nothing was measured, and no verdict is issued.")
        print("=" * 78)
        print(f"  {exc}")
        return EXIT_ENVIRONMENT


if __name__ == "__main__":
    raise SystemExit(main())
