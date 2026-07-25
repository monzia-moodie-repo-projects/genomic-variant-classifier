#!/usr/bin/env python3
"""Step 1b: complete clinical_sig ontology census + multi-axis parser validation.

READ-ONLY. Writes one measurement file; touches no source, runs no git command.

WHY THIS EXISTS (decision 2026-07-25)
=====================================
The P6 evidence-adjudication audit surfaced a hard blocker: 71 distinct clinical_sig
values across 245,925 rows fell through the flat evidence-state classifier as
UNRECOGNISED. The decision is NOT to patch those 71 into a flat string->state map,
but to replace string-level classification with a lossless multi-axis parse:

  clinical_sig string
    -> grammar-based, fail-closed parse into a typed evidence object (source semantics)
    -> a SEPARATE, versioned binary-target policy derives label 1/0/None (task view)

This decouples "what ClinVar says" (preserved in the canonical cohort) from "what the
current binary model is authorised to learn" (a derived, versioned view), so the cohort
need not be rebuilt when the learning task later expands to risk / penetrance / PGx.

This probe is the COMPLETE ontology census + parser validation the decision requires
before any v2 builder is written. It:
  * enumerates EVERY distinct clinical_sig value (not just the 71 unknown), with row
    count, unique-variant count, delimiter structure, and parsed atomic components;
  * runs the fail-closed grammar parser on every value and reports any UNCONSUMED
    token or unparsed value (the gate: unrecognised = 0, unconsumed = 0 before adoption);
  * assigns each value a multi-axis evidence vector and a proposed binary consequence
    under BINARY_TARGET_POLICY v2.0;
  * reports three target-policy sensitivity views (strict-Mendelian / inclusive-disease
    / current-production-compatible) so ontology-repair gain is separable from P6 order
    correction.

It does NOT rewire clean_cohort, does NOT build the v2 cohort, and does NOT adopt any
policy. It is the evidence for designing the v2 ontology + parser.

VERSIONS (independent, per the decision):
  CLINICAL_SIGNIFICANCE_ONTOLOGY_VERSION = "1.0"
  BINARY_PATHOGENICITY_POLICY_VERSION    = "2.0"

INPUT   : data/processed/clinvar_grch38.parquet  (RAW cohort)
OUTPUT  : docs/measurements/CLINSIG_ONTOLOGY_CENSUS_2026-07-25.txt

USAGE (PowerShell 5.1):
    python "C:\\Users\\monzi\\Downloads\\probe_clinsig_ontology_census_2026-07-25.py"

Exit 0 on success, 2 if no cohort or no clinical_sig column. Never edits.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

CLINICAL_SIGNIFICANCE_ONTOLOGY_VERSION = "1.0"
BINARY_PATHOGENICITY_POLICY_VERSION = "2.0"

REPO = Path(r"C:\Projects\genomic-variant-classifier")
RAW = REPO / "data" / "processed" / "clinvar_grch38.parquet"
CLEAN = REPO / "data" / "processed" / "clinvar_grch38_clean.parquet"
OUT = REPO / "docs" / "measurements" / "CLINSIG_ONTOLOGY_CENSUS_2026-07-25.txt"

EXIT_OK = 0
EXIT_FAIL = 2

# ---------------------------------------------------------------------------
# Recognised atomic vocabulary. Longest-match: multi-word atomic phrases (whose
# internal comma is intrinsic, e.g. "pathogenic, low penetrance") are matched as a
# unit BEFORE any delimiter splitting. Verified against NCBI ClinVar clinsig docs
# and this cohort's observed values (2026-07-25).
# ---------------------------------------------------------------------------
# Each atomic term maps to a partial evidence assignment (a dict of axis -> value).
# The parser accumulates these; conflicting assignments on the same axis are recorded.

# disease_pathogenicity axis values
P, LP, VUS, LB, B, NONE = "P", "LP", "UNCERTAIN", "LB", "B", "NONE"

# Atomic terms carrying a disease-pathogenicity call (possibly with a modifier).
# Order within this dict does not matter; matching is longest-string-first.
ATOMIC_TERMS: dict[str, dict] = {
    # --- ACMG/AMP 5-tier germline pathogenicity ---
    "pathogenic": {"path": P},
    "likely pathogenic": {"path": LP},
    "uncertain significance": {"path": VUS},
    "likely benign": {"path": LB},
    "benign": {"path": B},
    # --- aggregate compatibility classes (slash-joined, but standard atomic labels) ---
    "pathogenic/likely pathogenic": {"path": P},          # P/LP aggregate -> treat as P-side
    "benign/likely benign": {"path": B},                  # B/LB aggregate -> treat as B-side
    # --- ClinGen low-penetrance pathogenic ---
    "pathogenic, low penetrance": {"path": P, "penetrance": "LOW"},
    "likely pathogenic, low penetrance": {"path": LP, "penetrance": "LOW"},
    # --- ClinGen risk alleles ---
    "established risk allele": {"risk": "ESTABLISHED"},
    "likely risk allele": {"risk": "LIKELY"},
    "uncertain risk allele": {"risk": "UNCERTAIN"},
    # --- VUS subtypes (ClinGen) ---
    "vus-high": {"path": VUS, "uncertain_subtype": "HIGH"},
    "vus-mid": {"path": VUS, "uncertain_subtype": "MID"},
    "vus-low": {"path": VUS, "uncertain_subtype": "LOW"},
    # --- pharmacogenomic / other non-binary assertions ---
    "drug response": {"pgx": True},
    "risk factor": {"other": "risk_factor"},
    "association": {"other": "association"},
    "protective": {"other": "protective"},
    "affects": {"other": "affects"},
    "confers sensitivity": {"other": "confers_sensitivity"},
    "other": {"other": "other"},
    "not provided": {"classification_present": False, "provenance": "not_provided"},
    "no classification provided": {"classification_present": False, "provenance": "no_classification"},
    "no classification for the single variant": {"classification_present": False, "provenance": "no_classification"},
    "no classifications from unflagged records": {"classification_present": False, "provenance": "no_classification"},
    # --- explicit conflict (clinical_sig form) ---
    "conflicting classifications of pathogenicity": {"explicit_conflict": True},
    "conflicting interpretations of pathogenicity": {"explicit_conflict": True},
    # older ClinVar aggregate-conflict term (submitter disagreement); same conflict
    # class as the two above. Verified vs NCBI ClinVar docs 2026-07-25.
    "conflicting data from submitters": {"explicit_conflict": True},
    # recognized negative counterpart to "association"; a typed non-binary assertion.
    "association not found": {"other": "association_not_found"},
}

# Absent/serialization markers -> classification_present=False, provenance=missing.
MISSING_MARKERS = {"", "-", ".", "na", "nan", "none", "null", "<na>"}

# clean_cohort's current production binary treatment (for the "current" column).
PROD_PATHOGENIC = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}
PROD_BENIGN = {"benign", "likely benign", "benign/likely benign"}


def normalise(v: object) -> str:
    if v is None:
        return ""
    try:
        import math
        if isinstance(v, float) and math.isnan(v):
            return ""
    except Exception:
        pass
    return " ".join(str(v).strip().lower().replace("_", " ").split())


@dataclass
class ParsedClinSig:
    raw: str
    normalized: str
    components: list = field(default_factory=list)     # list of atomic term strings
    unconsumed: str = ""                               # leftover text (fail-closed signal)
    # multi-axis evidence
    path: str = NONE                                   # strongest disease-pathogenicity call
    path_calls: set = field(default_factory=set)       # all disease-path calls seen
    penetrance: str = "UNSPECIFIED"
    pgx: bool = False
    risk: str = "NONE"
    uncertain_subtype: str | None = None
    other: set = field(default_factory=set)
    explicit_conflict: bool = False
    classification_present: bool = True
    provenance: str = "present"

    @property
    def recognised(self) -> bool:
        return self.unconsumed == "" and (len(self.components) > 0 or not self.classification_present)


def _split_top_level(norm: str) -> list[str]:
    """Split a normalized string on ';' and '/' (assertion separators), but ONLY after
    atomic multi-word phrases have been reserved. Comma is NOT a top-level separator
    because it appears intrinsically inside atomic terms (', low penetrance'). We first
    protect known comma-atomics, then split on ; and /."""
    protected = norm
    placeholders = {}
    # protect comma-bearing atomic phrases so their comma is not seen as a separator
    for i, term in enumerate(t for t in ATOMIC_TERMS if "," in t):
        if term in protected:
            ph = f"\x00{i}\x00"
            placeholders[ph] = term
            protected = protected.replace(term, ph)
    # now split on ; and /
    parts = []
    for chunk in protected.replace("/", ";").split(";"):
        c = chunk.strip()
        if c:
            parts.append(c)
    # restore placeholders
    restored = []
    for p in parts:
        for ph, term in placeholders.items():
            p = p.replace(ph, term)
        restored.append(p.strip())
    return restored


def parse_clinsig(raw: object) -> ParsedClinSig:
    """Grammar-based, fail-closed parse. Longest-match atomic phrases first; any token
    that does not resolve to a recognised atomic term is left in `unconsumed`."""
    norm = normalise(raw)
    res = ParsedClinSig(raw="" if raw is None else str(raw), normalized=norm)

    if norm in MISSING_MARKERS:
        res.classification_present = False
        res.provenance = "missing"
        return res

    # whole-string atomic match first (handles slash-aggregates and comma-atomics)
    if norm in ATOMIC_TERMS:
        _apply(res, norm, ATOMIC_TERMS[norm])
        res.components.append(norm)
        return res

    # otherwise split into assertion parts and match each atomically
    parts = _split_top_level(norm)
    leftover = []
    for part in parts:
        if part in ATOMIC_TERMS:
            _apply(res, part, ATOMIC_TERMS[part])
            res.components.append(part)
        else:
            leftover.append(part)
    res.unconsumed = "; ".join(leftover)
    return res


# strength order for choosing the "strongest" disease-path call
_PATH_STRENGTH = {P: 5, LP: 4, VUS: 3, LB: 2, B: 1, NONE: 0}


def _apply(res: ParsedClinSig, term: str, assign: dict) -> None:
    if "path" in assign:
        res.path_calls.add(assign["path"])
        if _PATH_STRENGTH[assign["path"]] > _PATH_STRENGTH[res.path]:
            res.path = assign["path"]
    if assign.get("penetrance"):
        res.penetrance = assign["penetrance"]
    if assign.get("pgx"):
        res.pgx = True
    if assign.get("risk"):
        res.risk = assign["risk"]
    if assign.get("uncertain_subtype"):
        res.uncertain_subtype = assign["uncertain_subtype"]
    if assign.get("other"):
        res.other.add(assign["other"])
    if assign.get("explicit_conflict"):
        res.explicit_conflict = True
    if assign.get("classification_present") is False:
        res.classification_present = False
        res.provenance = assign.get("provenance", res.provenance)


# ---------------------------------------------------------------------------
# BINARY TARGET POLICY v2.0 -- derived SEPARATELY from the evidence object.
# ---------------------------------------------------------------------------
def derive_binary_label(res: ParsedClinSig, view: str = "inclusive") -> object:
    """Return 1 / 0 / None under a named target-policy view. Never called during
    parsing; the evidence object is source-truth, this is the task view.

    views:
      inclusive   -- P/LP (incl low-penetrance) -> 1; B/LB -> 0; else None.
                     Pathogenic + PGx compound with a clear single disease call -> 1;
                     if path_calls is ambiguous (mixed) -> None (context ambiguous).
      strict      -- exclude low-penetrance, risk alleles, and any compound with PGx or
                     other assertions; only clean single P/LP or B/LB.
      production  -- only the exact simple terms in the current PATHOGENIC/BENIGN sets.
    """
    if not res.classification_present:
        return None
    if res.explicit_conflict:
        return None

    if view == "production":
        if res.normalized in PROD_PATHOGENIC:
            return 1
        if res.normalized in PROD_BENIGN:
            return 0
        return None

    # need an unambiguous disease-pathogenicity call
    pos = {P, LP}
    neg = {B, LB}
    has_pos = bool(res.path_calls & pos)
    has_neg = bool(res.path_calls & neg)
    if has_pos and has_neg:
        return None  # mixed disease call within one value -> ambiguous
    is_pos = has_pos and not has_neg
    is_neg = has_neg and not has_pos

    if view == "strict":
        if res.penetrance == "LOW":
            return None
        if res.risk != "NONE":
            return None
        if res.pgx or res.other:
            return None
        if res.uncertain_subtype:
            return None
        if is_pos:
            return 1
        if is_neg:
            return 0
        return None

    # inclusive. A disease call coexisting with ANY non-disease assertion (pharmaco-
    # genomic, risk-allele, or other typed assertion: protective/association/affects/
    # confers-sensitivity/other) is context-ambiguous -- risk association and Mendelian
    # pathogenicity are different estimands, and a flattened compound can mix contexts.
    # Such compounds are WITHHELD (None) from the binary view rather than voted into it;
    # the full evidence remains in the parsed object for a future risk/PGx-aware target.
    has_nondisease = res.pgx or bool(res.other) or (res.risk != "NONE")
    if is_pos:
        if has_nondisease:
            return None
        return 1
    if is_neg:
        if has_nondisease:
            return None
        return 0
    return None


def main() -> int:
    import pyarrow.parquet as pq

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s); print(s)

    emit("=" * 78)
    emit("STEP 1b CLINICAL_SIG ONTOLOGY CENSUS + PARSER VALIDATION -- 2026-07-25 (read-only)")
    emit(f"ontology v{CLINICAL_SIGNIFICANCE_ONTOLOGY_VERSION}  "
         f"binary-target-policy v{BINARY_PATHOGENICITY_POLICY_VERSION}")
    emit("=" * 78)

    cohort = RAW if RAW.is_file() else (CLEAN if CLEAN.is_file() else None)
    if cohort is None:
        print(f"  FAIL: no cohort at {RAW} or {CLEAN}"); return EXIT_FAIL
    emit(f"\n  cohort: {cohort.name}")

    tbl = pq.read_table(cohort)
    have = list(tbl.column_names)
    sig_col = None
    for c in ("clinical_sig", "clinical_significance", "clnsig"):
        if c in have:
            sig_col = c; break
    if sig_col is None:
        print("  FAIL: no clinical_sig column"); return EXIT_FAIL
    sigs = tbl.column(sig_col).to_pylist()
    vids = tbl.column("variant_id").to_pylist() if "variant_id" in have else [None] * len(sigs)
    n = len(sigs)
    emit(f"  rows: {n:,}   clinical_sig column: {sig_col}")

    # census: per distinct normalized value -> row count, variant set
    row_count: Counter = Counter()
    var_sets: dict = defaultdict(set)
    raw_forms: dict = defaultdict(Counter)
    for i in range(n):
        norm = normalise(sigs[i])
        row_count[norm] += 1
        var_sets[norm].add(vids[i])
        raw_forms[norm][("" if sigs[i] is None else str(sigs[i]))] += 1
    emit(f"  distinct normalized clinical_sig values: {len(row_count):,}")

    # parse every distinct value; bucket by recognised / unconsumed
    parsed_by_norm = {norm: parse_clinsig(norm) for norm in row_count}
    unrecognised = {norm: p for norm, p in parsed_by_norm.items() if not p.recognised}
    unconsumed_rows = sum(row_count[norm] for norm in unrecognised)

    emit("\n" + "=" * 78)
    emit("PARSER FAIL-CLOSED GATE")
    emit("-" * 78)
    emit(f"  distinct values that fully parse (recognised)   : "
         f"{len(row_count) - len(unrecognised):,}")
    emit(f"  distinct values with UNCONSUMED tokens          : {len(unrecognised):,}")
    emit(f"  rows affected by unconsumed tokens              : {unconsumed_rows:,}")
    if unrecognised:
        emit("  -- these must be resolved before v2 adoption (gate: unconsumed = 0): --")
        for norm, p in sorted(unrecognised.items(), key=lambda kv: -row_count[kv[0]])[:40]:
            emit(f"    {row_count[norm]:>10,}  norm={norm!r}  unconsumed={p.unconsumed!r}  "
                 f"components={p.components}")
    else:
        emit("  ALL values fully parse with zero unconsumed tokens. Gate SATISFIED.")

    # full census table (every distinct value)
    emit("\n" + "=" * 78)
    emit("COMPLETE ONTOLOGY CENSUS (every distinct clinical_sig value)")
    emit("-" * 78)
    emit("  rows | variants | delim | path | pen | pgx | risk | subtype | other | "
         "conflict | present | prod | v2-inclusive")
    def delim_of(norm):
        d = []
        if "," in norm and norm not in ATOMIC_TERMS: d.append(",")
        if ";" in norm: d.append(";")
        if "/" in norm and norm not in ATOMIC_TERMS: d.append("/")
        return "".join(d) or "-"
    for norm in sorted(row_count, key=lambda k: -row_count[k]):
        p = parsed_by_norm[norm]
        prod = ("1" if norm in PROD_PATHOGENIC else "0" if norm in PROD_BENIGN else "None")
        v2 = derive_binary_label(p, "inclusive")
        emit(f"  {row_count[norm]:>9,} | {len(var_sets[norm]):>8,} | {delim_of(norm):>5} | "
             f"{p.path:>4} | {p.penetrance[:3]:>3} | {str(p.pgx)[:1]} | {p.risk[:4]:>4} | "
             f"{str(p.uncertain_subtype):>7} | {','.join(sorted(p.other)) or '-':>18} | "
             f"{str(p.explicit_conflict)[:1]} | {str(p.classification_present)[:1]} | "
             f"{prod:>4} | {str(v2):>5}   {norm!r}")

    # target-policy sensitivity views: how many rows get label 1 / 0 / None under each
    emit("\n" + "=" * 78)
    emit("TARGET-POLICY SENSITIVITY VIEWS (row-level label assignment)")
    emit("-" * 78)
    for view in ("production", "strict", "inclusive"):
        pos = neg = none = 0
        for norm, cnt in row_count.items():
            lab = derive_binary_label(parsed_by_norm[norm], view)
            if lab == 1: pos += cnt
            elif lab == 0: neg += cnt
            else: none += cnt
        emit(f"  {view:<12}: positive={pos:>10,}  negative={neg:>10,}  withheld/None={none:>10,}")
    emit("\n  'production' = only current PATHOGENIC_TERMS/BENIGN_TERMS exact matches.")
    emit("  'strict'     = clean single P/LP or B/LB; excludes low-penetrance, risk, PGx, VUS.")
    emit("  'inclusive'  = P/LP (incl low-penetrance) -> 1, B/LB -> 0; compound w/ PGx or")
    emit("                 other assertion withheld as context-ambiguous; conflict withheld.")
    emit("  Difference (inclusive - production) positives = labels recoverable by ontology")
    emit("  repair; this is separable from the P6 order-correction gain measured earlier.")

    # component frequency: how many rows carry each axis
    emit("\n" + "=" * 78)
    emit("EVIDENCE-AXIS ROW COVERAGE")
    emit("-" * 78)
    axis = Counter()
    for norm, cnt in row_count.items():
        p = parsed_by_norm[norm]
        if p.path in (P, LP): axis["disease: P/LP"] += cnt
        if p.path in (B, LB): axis["disease: B/LB"] += cnt
        if p.path == VUS: axis["disease: VUS"] += cnt
        if p.penetrance == "LOW": axis["penetrance: LOW"] += cnt
        if p.pgx: axis["pharmacogenomic"] += cnt
        if p.risk != "NONE": axis["risk allele"] += cnt
        if p.uncertain_subtype: axis["VUS subtype"] += cnt
        if p.other: axis["other non-binary"] += cnt
        if p.explicit_conflict: axis["explicit conflict"] += cnt
        if not p.classification_present: axis["absent/no-classification"] += cnt
    for k in sorted(axis, key=lambda x: -axis[x]):
        emit(f"  {k:<28}: {axis[k]:>12,} rows")

    emit("\n" + "=" * 78)
    emit("ACCEPTANCE GATE READOUT")
    emit("-" * 78)
    emit(f"  distinct clinical_sig values                 : {len(row_count):,}")
    emit(f"  values with unconsumed tokens (must be 0)    : {len(unrecognised):,}")
    emit(f"  rows with unconsumed tokens (must be 0)      : {unconsumed_rows:,}")
    emit("  When both are 0, the parser is complete and the v2 ontology + binary-target")
    emit("  policy can be implemented against it. Until then, do NOT build the v2 cohort:")
    emit("  any unconsumed value would enter certified construction through an implicit")
    emit("  branch, which is exactly the silent gap this census exists to close.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"\nWROTE {OUT}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
