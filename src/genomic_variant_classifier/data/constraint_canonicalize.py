"""Canonicalise the gnomAD MANE Select constraint table, and validate it.

WHY THIS MODULE EXISTS -- DUPLICATE-1A
======================================
`variant_ensemble.py` aliased `loeuf` into `gene_constraint_oe`:

    feats["gene_constraint_oe"] = df.get("gene_constraint_oe",
                                         df.get("loeuf", <constant>))

and `connector_gnomad_constraint.py` maps `lof.oe_ci.upper` -> `loeuf`. The two
columns were therefore bit-identical, measured 2026-08-09:

    identical = True   max abs diff = 0.0   correlation = 1.0

The justification on record, at `patch_constraint_oe_from_loeuf.py:10`, is that
"LOEUF is the LoF observed/expected upper-bound fraction, so loeuf is the
correct source." That inference does not hold. LOEUF is the upper bound of the
90 per cent confidence interval around the loss-of-function observed/expected
ratio; the ratio itself is the point estimate. The source column name says so:
`lof.oe_ci.upper` is not `lof.oe`.

The genuine point estimate is present in the same file, and its arithmetic was
verified against its own inputs on 2026-08-09:

    max | lof.obs / lof.exp  -  lof.oe |  =  3.25e-4     (rounding)

THE STRUCTURE OF THE SOURCE FILE, MEASURED NOT ASSUMED
=======================================================
`gnomad.v4.1.constraint_metrics.tsv`, MANE Select rows only:

    34,962 rows
    17,486 distinct gene symbols
         8 rows carrying a NULL gene symbol

Every row has a partner. gnomAD emits each MANE Select transcript twice, once
per annotation namespace:

    gene_id '26009'           transcript 'NM_015534.6'      RefSeq
    gene_id 'ENSG00000036549' transcript 'ENST00000370801'  Ensembl

The Ensembl row carries the richer record -- rank, decile, chromosome, coding
length, exon count -- while the RefSeq row has those as null. The BIOLOGICAL
metrics agree exactly:

    LOEUF identical within a gene symbol: 17,486 of 17,486
    LOEUF differing within a gene symbol:      0

So this is namespace duplication, not two competing measurements, and the
correct treatment is a collapse with an equality invariant rather than a
filter. Three pairing shapes exist:

    17,468 pairs sharing one gene symbol
         5 pairs split by SYMBOL DISAGREEMENT
           (SCHIP1/IQCJ-SCHIP1, ZNF177/ZNF559-ZNF177, EEF1AKNMT/METTL13,
            FAM207A/SLX9, METTL11B/NTMT2 -- identical metrics, different names)
         8 pairs split because the Ensembl side has NO SYMBOL and the RefSeq
           side uses a provisional LOC* placeholder

THREE DESIGN RULINGS, EACH FORCED BY A MEASUREMENT
===================================================
1.  NEVER `drop_duplicates`. The surviving row would depend on source order.
2.  NEVER key on `gene_id` alone. Measured: `gene_id` is unique PER ROW in this
    file, so keying on it preserves the duplication rather than resolving it.
3.  NEVER let `groupby` drop nulls implicitly. pandas defaults to
    `dropna=True`, which would silently discard the 8 null-symbol rows -- the
    exact silent-loss shape this project has repeatedly been bitten by. Those
    rows are excluded EXPLICITLY, with a recorded count, and nothing is lost
    because their RefSeq partners carry identical metrics under LOC* symbols.

WHAT IS AND IS NOT A MODEL FEATURE
===================================
`lof.obs`, `lof.exp` and `oe_exceeds_reported_upper_bound` are ingested for
SOURCE VALIDATION and live in the audit record. They are deliberately NOT added
to TABULAR_FEATURES: source-validation information does not automatically
become predictive information, and DUPLICATE-1A repairs an identity rather than
growing the roster.

WHAT THIS MODULE DELIBERATELY DOES NOT CLAIM
=============================================
Twelve well-powered genes have `lof.oe > lof.oe_ci.upper` (DNMT3A, TET2, LZTR1
and others, all with lof.exp above 100). The reported upper bounds cluster just
below 2.0. That is a PATTERN, not a documented mechanism. This module records
the observation as `oe_exceeds_reported_upper_bound` and asserts nothing about
censoring, clipping or an upstream defect. It does not clip `lof.oe`: taking
`min(oe, loeuf)` would manufacture a statistic that gnomAD never published.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Source column names, exactly as gnomAD publishes them.
COL_GENE = "gene"
COL_GENE_ID = "gene_id"
COL_TRANSCRIPT = "transcript"
COL_MANE = "mane_select"
COL_CANONICAL = "canonical"
COL_LOF_OBS = "lof.obs"
COL_LOF_EXP = "lof.exp"
COL_LOF_OE = "lof.oe"
COL_LOEUF = "lof.oe_ci.upper"

# Metrics that must agree between two namespace representations of one gene.
EQUIVALENCE_METRICS = (COL_LOF_OBS, COL_LOF_EXP, COL_LOF_OE, COL_LOEUF)

# TWO TOLERANCES, because they encode two different claims. Sharing one number
# because it happened to be nearby would import a rounding allowance into a
# place where no calculation occurs.
#
#   ARITHMETIC: published lof.oe ~= lof.obs / lof.exp. An independent
#     calculation against published four-decimal fields, so rounding is
#     expected. Measured 2026-08-09 across 191,811 rows: max |error| =
#     3.25426e-4.
#
#   NAMESPACE: the RefSeq and Ensembl rows are two encodings of ONE published
#     record. No calculation happens between them, so nothing may differ --
#     including which values are missing. Measured: 0 disagreements across
#     17,486 gene symbols.
OE_ARITHMETIC_ATOL = 5e-4
NAMESPACE_ATOL = 0.0

_MANE_TRUE = frozenset({"true", "1", "yes", "t", "y"})


_ENSEMBL_GENE_ID = re.compile(r"^ENSG\d+(\.\d+)?$")


class TranscriptSelectionTier(str, Enum):
    """WHICH transcript a gene's constraint came from, recorded per gene.

    MEASURED 2026-08-09. The source file holds 211,523 rows across 18,203 gene
    symbols; MANE Select covers 17,486 of them. So roughly 718 genes have NO
    MANE Select transcript at all -- non-coding, deprecated, or otherwise
    MANE-absent.

    A MANE-only filter would silently DROP those genes, moving their variants
    from "matched" to "unmatched" and, given the connector's fill, from a real
    value to a fabricated one. That trades one invisible defect for another.

    A tiered selection keeps them and RECORDS WHY each gene was chosen, so a
    later decision to exclude a tier is a filter over an explicit field rather
    than a re-engineering of the selection logic. The tier travels with the
    row; nothing downstream has to infer it.
    """
    MANE_SELECT = "mane_select"
    CANONICAL = "canonical"
    #
    # AUDIT-ONLY. Never a production tier. "Neither MANE nor canonical" is not
    # a selection policy -- it is the absence of one. A gene with three
    # unflagged transcripts has no defensible choice among them, and retaining
    # an arbitrary one would state a biological measurement the source does not
    # support. Such genes carry MISSING constraint, and the count is recorded.
    #
    #     an arbitrary transcript says  "here is a measurement"
    #     a missing value says          "no transcript satisfies the contract"
    #
    # Those are not interchangeable, and the whole of DUPLICATE-1 is what
    # happens when they are treated as if they were.
    UNSELECTED = "unselected"


class GeneIdNamespace(str, Enum):
    ENSEMBL = "ensembl"
    NCBI_GENE = "ncbi_gene"


class ConstraintSourceError(ValueError):
    """The gnomAD constraint source violated a contract this module asserts."""


@dataclass(frozen=True)
class OeValidation:
    n_checked: int
    n_failed: int
    max_abs_error: float


@dataclass(frozen=True)
class TranscriptSourceFacts:
    """What the gnomAD file CONTAINS. Invariant under project policy.

    AUDITCOUNT-1, measured 2026-08-10. `no_declared_transcript` reported 1 with
    the canonical fallback enabled and 2 with it disabled, FOR THE SAME SOURCE,
    because a gene that HAS a canonical transcript was folded into "no declared
    transcript" whenever policy declined to use it. A biological fact about
    gnomAD changed when a project switch moved -- and the switch exists
    precisely for MANE-only threshold calibration, which is the configuration
    where that count would be read.

    Sealing the record made it immutable; it did not make its contents correct.

    Renaming the field would have fixed the number. Splitting the record makes
    the category error IMPOSSIBLE:

        what gnomAD contains   !=   what the project chose to select
    """
    n_gene_symbols: int = 0
    n_with_mane_select: int = 0
    n_with_canonical: int = 0
    n_without_mane_select: int = 0
    n_without_declared_transcript: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class TranscriptSelectionAudit:
    """What the PROJECT did with those facts. Varies with policy, by design."""
    allow_canonical_fallback: bool = True
    n_selected_mane: int = 0
    n_selected_canonical: int = 0
    n_excluded_by_policy: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class _AuditBuilder:
    """MUTABLE while the index is being built. Sealed to a frozen record before
    it is returned -- see CanonicalizationAudit.

    FIELD NAMES SAY WHAT THEY COUNT. `n_rows_mane` previously held the count of
    ALL SELECTED rows, including the canonical-fallback tier, so it was false
    whenever the fallback contributed anything -- and it did, for 696 genes on
    the real source. A field named for one tier while holding the sum of two is
    the same semantic drift as a comment claiming "first = MANE transcript" over
    a drop_duplicates.
    """
    source_path: str = ""
    source_sha256: str = ""
    n_rows_input: int = 0
    n_rows_selected: int = 0
    n_rows_null_symbol_excluded: int = 0
    n_rows_grouped: int = 0
    n_genes_canonical: int = 0
    n_genes_ensembl_preferred: int = 0
    n_genes_refseq_fallback: int = 0
    n_genes_without_constraint: int = 0
    n_oe_exceeds_reported_upper_bound: int = 0
    oe_validation: OeValidation | None = None
    excluded_null_symbol_gene_ids: tuple = ()
    tier_counts: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    source_facts: TranscriptSourceFacts = field(default_factory=TranscriptSourceFacts)
    selection: TranscriptSelectionAudit = field(default_factory=TranscriptSelectionAudit)
    source_sha256_verified: bool = False

    def seal(self) -> "CanonicalizationAudit":
        """Emit the immutable record. After this the evidence cannot change."""
        return CanonicalizationAudit(
            source_path=self.source_path,
            source_sha256=self.source_sha256,
            n_rows_input=self.n_rows_input,
            n_rows_selected=self.n_rows_selected,
            n_rows_null_symbol_excluded=self.n_rows_null_symbol_excluded,
            n_rows_grouped=self.n_rows_grouped,
            n_genes_canonical=self.n_genes_canonical,
            n_genes_ensembl_preferred=self.n_genes_ensembl_preferred,
            n_genes_refseq_fallback=self.n_genes_refseq_fallback,
            n_genes_without_constraint=self.n_genes_without_constraint,
            n_oe_exceeds_reported_upper_bound=self.n_oe_exceeds_reported_upper_bound,
            oe_validation=self.oe_validation,
            excluded_null_symbol_gene_ids=tuple(self.excluded_null_symbol_gene_ids),
            tier_counts=tuple(sorted(self.tier_counts.items())),
            notes=tuple(self.notes),
            source_facts=self.source_facts,
            selection=self.selection,
            source_sha256_verified=self.source_sha256_verified)


@dataclass(frozen=True)
class CanonicalizationAudit:
    """IMMUTABLE evidence of how the index was built.

    Frozen, with tuple collections throughout. An earlier version froze only
    `notes` and left every other field writable while a comment called the
    result sealed; a probe then set n_genes_canonical = 999, a NEGATIVE tier
    count and a fabricated note, all accepted. A record that documents itself
    as evidence must be unable to change after emission.
    """
    source_path: str = ""
    source_sha256: str = ""
    n_rows_input: int = 0
    n_rows_selected: int = 0
    n_rows_null_symbol_excluded: int = 0
    n_rows_grouped: int = 0
    n_genes_canonical: int = 0
    n_genes_ensembl_preferred: int = 0
    n_genes_refseq_fallback: int = 0
    n_genes_without_constraint: int = 0
    n_oe_exceeds_reported_upper_bound: int = 0
    oe_validation: OeValidation | None = None
    excluded_null_symbol_gene_ids: tuple = ()
    tier_counts: tuple = ()
    notes: tuple = ()
    source_facts: TranscriptSourceFacts = TranscriptSourceFacts()
    selection: TranscriptSelectionAudit = TranscriptSelectionAudit()
    #: PROVENANCE-ASSERT-1: True only when the digest was RECOMPUTED from
    #: the file on disk. A caller-supplied digest is recorded but not
    #: believed -- a provenance field nobody verifies is decoration.
    source_sha256_verified: bool = False

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "oe_validation"}
        d["tier_counts"] = dict(self.tier_counts)
        d["source_facts"] = self.source_facts.as_dict()
        d["selection"] = self.selection.as_dict()
        d["oe_validation"] = (
            None if self.oe_validation is None else dict(self.oe_validation.__dict__))
        return d


def sha256_file(path: str) -> str:
    """Digest RAW BYTES. Never a parsed-and-reserialised object: normalisation
    would silently change the identity of the artefact being recorded."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def namespace_of(gene_id: object) -> GeneIdNamespace:
    """Classify a gnomAD gene identifier, or REFUSE.

    The previous form returned NCBI_GENE for anything not starting with ENSG,
    so None, NaN, an empty string and outright garbage all classified as a
    valid NCBI Gene identifier. That is a fail-open default in the function
    that decides which representation is canonical.
    """
    # ONE check, not two. An earlier form tested None/NaN explicitly AND then
    # tested the stringified value for null-like text. The first branch was
    # fully redundant -- str(None) is "None", str(nan) is "nan", str(pd.NA) is
    # "<NA>", all caught by the second -- so a sabotage run that deleted it
    # changed no behaviour and went undetected. A guard that cannot fail is
    # not a guard; the surviving check is the one under test.
    s = str(gene_id).strip()
    if not s or s.lower() in ("nan", "none", "<na>", "null"):
        raise ConstraintSourceError(
            "gnomAD gene_id is missing, blank or null-like: {!r}".format(gene_id))
    # ENSG followed by DIGITS, optionally a version suffix. A bare "ENSG_"
    # satisfied startswith() and classified as a valid Ensembl identifier.
    if _ENSEMBL_GENE_ID.match(s):
        return GeneIdNamespace.ENSEMBL
    if s.isdigit():
        return GeneIdNamespace.NCBI_GENE
    raise ConstraintSourceError(
        "unrecognised gnomAD gene_id namespace: {!r}. Expected an Ensembl "
        "'ENSG...' identifier or an all-digit NCBI Gene identifier.".format(gene_id))


def to_numeric_strict(series, *, column: str):
    """Coerce to numeric, distinguishing MISSING from MALFORMED.

    STRICTNUMERIC-1. `pd.to_numeric(errors="coerce")` turns "1.7", "" and
    "not_a_number" all into NaN, so a genuinely absent measurement and a
    corrupted field arrive at the model as the same value. That is the defect
    this whole line of work exists to end, reproduced inside the coercion.

    A value that was ALREADY missing stays missing. A value that is PRESENT and
    does not parse RAISES, because a source that suddenly emits unparseable
    numbers is a source-integrity failure, not a data gap.
    """
    s = pd.Series(series)
    was_missing = s.isna() | (s.astype(str).str.strip() == "")
    coerced = pd.to_numeric(s, errors="coerce")
    malformed = coerced.isna() & ~was_missing
    if bool(malformed.any()):
        offenders = s[malformed].astype(str).head(5).tolist()
        raise ConstraintSourceError(
            "column {!r}: {} value(s) are PRESENT but do not parse as numeric, "
            "e.g. {}. A malformed value is not a missing one; refusing to "
            "silently coerce it to NaN.".format(
                column, int(malformed.sum()), offenders))
    return coerced


def _flag_true(series) -> "pd.Series":
    return series.astype(str).str.strip().str.lower().isin(_MANE_TRUE)


def describe_transcript_source(raw: pd.DataFrame) -> TranscriptSourceFacts:
    """Count what the SOURCE contains. Takes no policy argument, deliberately.

    Every quantity here is a property of the gnomAD file. A function that
    cannot see a policy cannot let one contaminate a fact, which is a stronger
    guarantee than remembering not to.
    """
    missing = [c for c in (COL_GENE, COL_MANE) if c not in raw.columns]
    if missing:
        raise ConstraintSourceError(
            "gnomAD constraint source is missing required column(s): {}".format(
                ", ".join(missing)))
    named = raw[raw[COL_GENE].notna()]
    genes = set(named[COL_GENE])
    with_mane = set(named.loc[_flag_true(named[COL_MANE]), COL_GENE])
    with_canon = (set(named.loc[_flag_true(named[COL_CANONICAL]), COL_GENE])
                  if COL_CANONICAL in named.columns else set())
    return TranscriptSourceFacts(
        n_gene_symbols=len(genes),
        n_with_mane_select=len(with_mane),
        n_with_canonical=len(with_canon),
        n_without_mane_select=len(genes - with_mane),
        # NEITHER flag. Independent of whether the project USES the canonical
        # tier: this counts genes for which the source declares no transcript
        # at all.
        n_without_declared_transcript=len(genes - with_mane - with_canon))


def select_constraint_transcripts(raw: pd.DataFrame, *,
                                  allow_canonical_fallback: bool = True):
    """Choose rows per gene by an explicit tier ladder, and record the tier.

    MANE Select where available; otherwise the canonical transcript. A gene with
    NEITHER is NOT retained under any flag: it is counted as
    `no_declared_transcript` and carries missing constraint. An earlier version
    of this docstring described a retained UNSELECTED tier that the
    implementation never produces.

    Returns (frame_with_tier_column, tier_counts).

    The ladder is DECLARED, not inferred from row order. `drop_duplicates` is
    never used: on this source `gene_id` is unique per row and the file is
    ordered by neither gene nor transcript significance, so "first wins" is a
    coin toss. Measured 2026-08-09: first-row selection disagrees with MANE
    Select on 5,468 of 17,473 genes (31.3%), median absolute LOEUF difference
    0.039, maximum 1.689, with 132 genes crossing the 0.35 boundary.
    """
    missing = [c for c in (COL_GENE, COL_GENE_ID, COL_MANE) if c not in raw.columns]
    if missing:
        raise ConstraintSourceError(
            "gnomAD constraint source is missing required column(s): {}".format(
                ", ".join(missing)))
    # NULL SYMBOLS ARE NOT FILTERED HERE. canonicalize_mane_constraint excludes
    # them explicitly and records the count; doing it in two places broke the
    # conservation identity the moment this ladder was added. One exclusion,
    # one place, one recorded number.
    df = raw.copy()

    is_mane = _flag_true(df[COL_MANE])
    is_canon = (_flag_true(df[COL_CANONICAL]) if COL_CANONICAL in df.columns
                else pd.Series(False, index=df.index))

    named = df[COL_GENE].notna()
    mane_genes = set(df.loc[is_mane & named, COL_GENE])
    canon_only = set(df.loc[is_canon & named, COL_GENE]) - mane_genes
    all_named = set(df.loc[named, COL_GENE])
    rest = (all_named - mane_genes - canon_only if allow_canonical_fallback
            else all_named - mane_genes)

    # A MANE row is kept whatever its symbol, so the eight null-symbol MANE
    # rows still reach the explicit exclusion and the row count is unchanged.
    keep_mane = df[is_mane].assign(
        _tier=TranscriptSelectionTier.MANE_SELECT.value)
    if allow_canonical_fallback:
        keep_canon = df[is_canon & named & df[COL_GENE].isin(canon_only)].assign(
            _tier=TranscriptSelectionTier.CANONICAL.value)
    else:
        keep_canon = df.iloc[0:0].assign(_tier="")
        canon_only = set()

    out = pd.concat([keep_mane, keep_canon], axis=0)
    counts = {
        TranscriptSelectionTier.MANE_SELECT.value: len(mane_genes),
        TranscriptSelectionTier.CANONICAL.value: len(canon_only),
        # Genes with NO declared transcript. Recorded, never retained: they
        # carry missing constraint rather than an invented measurement.
        "no_declared_transcript": len(rest),
        "canonical_fallback_enabled": bool(allow_canonical_fallback),
    }
    return out, counts


def select_mane(raw: pd.DataFrame) -> pd.DataFrame:
    """MANE-only selection.

    REMOVED FROM THE PRODUCTION PATH on 2026-08-09. It was called by
    canonicalize_mane_constraint, which meant the tier ladder selected a
    canonical-tier gene and this function then discarded it again before the
    index was built. Thirty-six tests passed and not one asked whether a
    canonical-tier gene reached the OUTPUT -- the component was tested, the
    path was not.

    Retained only for MANE-only calibration work: gnomAD's v4.1.1 percentile
    table is derived from 17,063 MANE Select transcripts, so threshold
    calibration must use this population and not the fallback-extended one.
    """
    frame, _ = select_constraint_transcripts(raw, allow_canonical_fallback=False)
    return frame[frame["_tier"] == TranscriptSelectionTier.MANE_SELECT.value].copy()


def validate_published_oe(df: pd.DataFrame, *, atol: float = OE_ARITHMETIC_ATOL) -> OeValidation:
    """Assert lof.oe equals lof.obs / lof.exp within tolerance.

    CONDITIONAL, not global: only finite triples with a positive expected count
    are checkable, and a row that cannot be checked must not silently count as
    passing. Both the failure COUNT and the maximum error are returned so a
    later reader can see the envelope rather than only a boolean.
    """
    for c in (COL_LOF_OBS, COL_LOF_EXP, COL_LOF_OE):
        if c not in df.columns:
            raise ConstraintSourceError(
                "cannot validate lof.oe arithmetic: column {!r} absent".format(c))
    obs = pd.to_numeric(df[COL_LOF_OBS], errors="coerce")
    exp = pd.to_numeric(df[COL_LOF_EXP], errors="coerce")
    oe = pd.to_numeric(df[COL_LOF_OE], errors="coerce")

    usable = (np.isfinite(obs) & np.isfinite(exp) & np.isfinite(oe) & (exp > 0))
    if not usable.any():
        return OeValidation(n_checked=0, n_failed=0, max_abs_error=float("nan"))

    err = (obs[usable] / exp[usable] - oe[usable]).abs()
    failed = err > atol
    result = OeValidation(int(usable.sum()), int(failed.sum()),
                          float(err.max()))
    if result.n_failed:
        worst = err.nlargest(10)
        raise ConstraintSourceError(
            "gnomAD lof.oe failed its obs/exp identity: {}/{} row(s) exceed "
            "atol={}; max_abs_error={:.8g}; worst row labels={}".format(
                result.n_failed, result.n_checked, atol,
                result.max_abs_error, worst.index.tolist()))
    return result


def assert_row_conservation(n_grouped: int, n_retained: int, n_excluded: int,
                            n_selected: int) -> None:
    """Every MANE row is grouped or explicitly excluded -- checked TWICE.

    TWO IDENTITIES, because one can be satisfied by cancellation. A sabotage
    run on 2026-08-09 removed the explicit null-symbol filter AND reverted
    pandas' dropna default together: groupby then dropped the row silently
    (n_grouped fell by one) while the excluded COUNT still reported one,
    because it was computed from the intended exclusion rather than an
    observed one. The single identity balanced and the loss went undetected.

        n_grouped  == n_retained         the group loop saw every retained row
        n_retained + n_excluded == n_selected  nothing vanished before it

    A count of what was MEANT to be excluded is not a measurement of what was.

    Measured on the real file: 34,954 grouped == 34,954 retained;
    34,954 + 8 = 34,962 MANE rows.
    """
    if n_grouped != n_retained:
        raise ConstraintSourceError(
            "canonicalisation LOST rows inside grouping: {} grouped != {} "
            "retained. pandas groupby drops null keys by default; every "
            "retained row must reach the loop.".format(n_grouped, n_retained))
    if n_retained + n_excluded != n_selected:
        raise ConstraintSourceError(
            "canonicalisation LOST rows before grouping: {} retained + {} "
            "excluded != {} SELECTED rows.".format(n_retained, n_excluded, n_selected))


def _assert_namespace_equivalence(group: pd.DataFrame, symbol: object,
                                  atol: float = NAMESPACE_ATOL) -> None:
    """The upstream-drift tripwire.

    Two namespace representations of one gene must agree on every biological
    metric. Measured 2026-08-09: they agree on all 17,486 gene symbols. If a
    future release ever disagrees, this raises rather than silently preferring
    one -- which is the whole reason to collapse rather than filter.
    """
    if len(group) <= 1:
        return
    for col in EQUIVALENCE_METRICS:
        if col not in group.columns:
            raise ConstraintSourceError(
                "namespace-equivalence metric {!r} is absent from the source; "
                "equivalence cannot be established".format(col))
        x = pd.to_numeric(group[col], errors="coerce")

        # MISSINGNESS IS PART OF EQUIVALENCE. dropna() before comparing would
        # let a populated RefSeq value and a missing Ensembl value look
        # equivalent -- and since Ensembl is then preferred, a real measurement
        # would be silently replaced by missing data. That is the exact failure
        # class this canonicaliser exists to prevent.
        missing = x.isna()
        if bool(missing.any()) and not bool(missing.all()):
            raise ConstraintSourceError(
                "gnomAD MANE namespace representations DISAGREE for gene={!r} "
                "on {!r}: one representation is missing while another is "
                "populated. values={}".format(symbol, col, x.tolist()))
        if bool(missing.all()):
            continue

        vals = x.dropna()
        spread = float(vals.max() - vals.min())
        if spread > atol:
            raise ConstraintSourceError(
                "gnomAD MANE namespace representations DISAGREE for gene={!r} "
                "on {!r}: values={} spread={:.8g} > atol={}. These are meant to "
                "be two encodings of one transcript; a disagreement means the "
                "collapse is no longer safe.".format(
                    symbol, col, vals.tolist(), spread, atol))


def canonicalize_mane_constraint(raw: pd.DataFrame, *,
                                 oe_arithmetic_atol: float = OE_ARITHMETIC_ATOL,
                                 allow_canonical_fallback: bool = True,
                                 source_path: str = "",
                                 source_sha256: str = "") -> tuple:
    """Collapse paired namespace rows to one canonical row per gene symbol.

    Returns (canonical_frame, audit).
    """
    # PROVENANCE-ASSERT-1: a caller-supplied digest is a CLAIM. Recompute it
    # when the file is readable; otherwise record the claim as UNVERIFIED
    # rather than presenting it as established provenance.
    if source_path and source_sha256:
        try:
            actual = sha256_file(source_path)
        except OSError:
            actual = None
        if actual is not None and actual != source_sha256:
            raise ConstraintSourceError(
                "provenance mismatch for {!r}: caller supplied SHA-256 {}, the "
                "file on disk is {}".format(source_path, source_sha256, actual))
        _verified = actual is not None
    else:
        _verified = False

    audit = _AuditBuilder(source_path=source_path,
                                  source_sha256=source_sha256,
                                  n_rows_input=int(len(raw)))
    # SOURCE FACTS FIRST, before any policy is consulted. These are computed
    # from the file alone; no argument of this function can change them.
    audit.source_sha256_verified = _verified
    audit.source_facts = describe_transcript_source(raw)

    mane, tier_counts = select_constraint_transcripts(
        raw, allow_canonical_fallback=allow_canonical_fallback)
    audit.selection = TranscriptSelectionAudit(
        allow_canonical_fallback=bool(allow_canonical_fallback),
        n_selected_mane=int(tier_counts.get(
            TranscriptSelectionTier.MANE_SELECT.value, 0)),
        n_selected_canonical=int(tier_counts.get(
            TranscriptSelectionTier.CANONICAL.value, 0)),
        n_excluded_by_policy=int(
            audit.source_facts.n_gene_symbols
            - tier_counts.get(TranscriptSelectionTier.MANE_SELECT.value, 0)
            - tier_counts.get(TranscriptSelectionTier.CANONICAL.value, 0)))
    audit.n_rows_selected = int(len(mane))
    audit.tier_counts = dict(tier_counts)

    # EXPLICIT null-symbol exclusion. pandas groupby defaults to dropna=True and
    # would remove these silently. Measured 2026-08-09: 8 such rows, every one
    # an Ensembl record whose RefSeq partner carries IDENTICAL metrics under a
    # provisional LOC* symbol, so excluding them loses no measurement.
    null_symbol = mane[mane[COL_GENE].isna()]
    audit.n_rows_null_symbol_excluded = int(len(null_symbol))
    audit.excluded_null_symbol_gene_ids = tuple(
        str(x) for x in null_symbol[COL_GENE_ID].tolist())
    if len(null_symbol):
        # The runtime does NOT prove the partner claim; it merely excludes.
        # The eight-for-eight pairing was established by a source audit, and
        # historical evidence must be labelled historical.
        audit.notes.append(
            "{} MANE row(s) explicitly excluded because the canonical index "
            "requires a gene symbol. A source audit on 2026-08-09 identified "
            "paired RefSeq representations for all eight v4.1 cases; this run "
            "does not re-establish that pairing.".format(len(null_symbol)))
    mane = mane[mane[COL_GENE].notna()]
    # OBSERVED, not intended: the count of rows that actually survived the
    # filter, which is what conservation is checked against.
    n_retained = int(len(mane))

    audit.oe_validation = validate_published_oe(mane, atol=oe_arithmetic_atol)

    chosen = []
    n_grouped = 0
    # dropna=False is DEFENCE IN DEPTH and is currently unreachable: null
    # symbols were excluded explicitly above, so no null group can form. It is
    # retained so that removing that exclusion cannot silently reintroduce
    # pandas' default row-dropping -- and the LIVE guard against that is
    # assert_row_conservation below, which a sabotage run confirmed catches it.
    for symbol, group in mane.groupby(COL_GENE, sort=False, dropna=False):
        n_grouped += len(group)
        # NAMESPACE_ATOL explicitly, NOT the function's `atol` parameter --
        # that one carries OE_ARITHMETIC_ATOL and would import a 5e-4 rounding
        # allowance into a comparison where no rounding occurs. Caught
        # 2026-08-09 by a test asserting a 1e-9 difference must raise: the
        # constant existed and the call site never used it.
        _assert_namespace_equivalence(group, symbol, NAMESPACE_ATOL)
        ns = group[COL_GENE_ID].map(namespace_of)
        ens = group[ns == GeneIdNamespace.ENSEMBL]
        if len(ens) == 1:
            row = ens.iloc[0].copy()
            audit.n_genes_ensembl_preferred += 1
        elif len(ens) > 1:
            raise ConstraintSourceError(
                "gene={!r}: {} Ensembl MANE rows. Transcript ambiguity must be "
                "resolved by an explicit policy, never by source order.".format(
                    symbol, len(ens)))
        elif len(group) == 1:
            row = group.iloc[0].copy()
            audit.n_genes_refseq_fallback += 1
        else:
            raise ConstraintSourceError(
                "gene={!r}: {} non-Ensembl MANE rows and no Ensembl row; cannot "
                "resolve deterministically.".format(symbol, len(group)))
        row["_n_source_representations"] = int(len(group))
        # The tier travels with the row because `row` is taken FROM `group`,
        # which already carries `_tier` from selection. An explicit
        # re-assignment here was redundant, and a mixed-tier guard beside it was
        # unreachable: select_constraint_transcripts assigns tier per GENE, so a
        # gene belongs to exactly one tier by construction. Both were removed on
        # 2026-08-09 after sabotage showed neither could fail. The property they
        # were meant to protect is asserted directly, at its source, by
        # test_each_gene_belongs_to_exactly_one_tier.
        row["_namespace"] = str(namespace_of(row[COL_GENE_ID]).value)
        chosen.append(row)

    # CONSERVATION. Every MANE row must be either grouped or explicitly
    # excluded. Measured 2026-08-09: 34,954 grouped + 8 excluded = 34,962.
    assert_row_conservation(n_grouped, n_retained,
                            audit.n_rows_null_symbol_excluded, audit.n_rows_selected)
    audit.n_rows_grouped = n_grouped

    out = pd.DataFrame(chosen).reset_index(drop=True)
    if len(out) and not out[COL_GENE].is_unique:
        dupes = out.loc[out[COL_GENE].duplicated(keep=False), COL_GENE].tolist()
        raise ConstraintSourceError(
            "canonical constraint index is not unique by gene symbol: "
            "{}".format(sorted(set(dupes))[:20]))
    audit.n_genes_canonical = int(len(out))

    if len(out):
        oe = pd.to_numeric(out.get(COL_LOF_OE), errors="coerce")
        loeuf = pd.to_numeric(out.get(COL_LOEUF), errors="coerce")
        # DESCRIPTIVE ONLY. Says what was observed, nothing about why.
        out["oe_exceeds_reported_upper_bound"] = (
            oe.notna() & loeuf.notna() & (oe > loeuf))
        audit.n_oe_exceeds_reported_upper_bound = int(
            out["oe_exceeds_reported_upper_bound"].sum())
        audit.n_genes_without_constraint = int(loeuf.isna().sum())

    # SEAL the audit's mutable collections. An audit object is EVIDENCE once
    # emitted, and evidence that can still be appended to is not evidence.
    sealed = audit.seal()
    # "selected row(s)", not "MANE row(s)": the count includes the
    # canonical-fallback tier, which contributes 696 genes on the real source.
    logger.info("canonical gnomAD constraint index: %d gene(s) from %d selected "
                "row(s); tiers %s", sealed.n_genes_canonical,
                sealed.n_rows_selected, dict(sealed.tier_counts))
    return out, sealed


def derive_gene_is_constrained(loeuf, *, threshold: float):
    """Three states, not two: constrained, not constrained, UNKNOWN.

        C(l) = 1   if l <  threshold
               0   if l >= threshold
               NA  if l is unavailable

    `np.nan < 0.35` evaluates False, so a naive comparison records every gene
    without constraint data as NOT CONSTRAINED -- reproducing CONSTRAINTFILL-1
    one layer downstream. Absence of evidence is not evidence of tolerance.

    Returns a nullable Int8 for the SEMANTIC layer. The model-facing matrix
    should carry Float32 {0.0, 1.0, NaN}: pandas extension dtypes create
    needless friction with NumPy and scikit-learn paths, and the meaning
    matters while the storage dtype does not.
    """
    x = pd.to_numeric(loeuf, errors="coerce")
    out = pd.Series(pd.NA, index=x.index, dtype="Int8",
                    name="gene_is_constrained")
    known = x.notna()
    out.loc[known] = (x.loc[known] < threshold).astype("int8")
    return out


# ---------------------------------------------------------------------------
# Exact-duplicate detection -- the gate that would have caught DUPLICATE-1
# ---------------------------------------------------------------------------

def _fingerprint(s: pd.Series) -> str:
    h = pd.util.hash_pandas_object(s, index=False, categorize=True)
    return hashlib.sha256(np.asarray(h).tobytes()).hexdigest()


def exact_duplicate_groups(frame: pd.DataFrame) -> list:
    """Columns that are bit-identical to one another.

    RUN THIS ON THE PRE-TRANSFORM MATRIX. On a standardised matrix every
    constant column becomes 0.0, so two unrelated dead features would look
    identical and the gate would drown in meaningless pairs.

    Degenerate columns are skipped: a constant column is a VITALITY failure,
    which the vitality contract owns, not a duplicate-signal failure.

    Hashing identifies CANDIDATES in linear time; equality establishes the
    FINDING. A hash collision must never become a reported duplicate.
    """
    buckets = {}
    for name in frame.columns:
        s = frame[name]
        if s.nunique(dropna=False) <= 1:
            continue
        buckets.setdefault(_fingerprint(s), []).append(name)

    out = []
    for names in buckets.values():
        if len(names) < 2:
            continue
        remaining = list(names)
        while remaining:
            head = remaining.pop(0)
            group = [head]
            keep = []
            for other in remaining:
                if frame[head].equals(frame[other]):
                    group.append(other)
                else:
                    keep.append(other)
            remaining = keep
            if len(group) > 1:
                out.append(tuple(group))
    return out
