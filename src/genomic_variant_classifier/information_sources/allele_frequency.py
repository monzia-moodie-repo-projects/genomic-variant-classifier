"""Stage 1D-1c.1B: allele-frequency observations from the verified gnomAD extract, the strict selection policy, and claim
eligibility -- three separate planes. SHADOW MODE: nothing in production imports this module (stage 1D-1c.1C feeds
engineer_features from it).

Owner rulings 482cd291, 4bef3910 and b937681e (2026-11-14); audits AUDIT_OF_RULING_1D1c_2026-11-14 (sections 7-8) and
AUDIT_OF_RULING_1D1c1B_2026-11-15. The 1D-1b.2 pilot ``materialization.materialize_gnomad_af`` is NOT modified: its code
identity is sealed by docs/census/PILOTS_1D1b2.json and it remains the executable record of the legacy join.

The physical file ``data/processed/gnomad_v4_exomes.parquet`` holds two LOGICAL sources, read once through one verified
byte stream:

* ``dataset:gnomad_af`` -- the primary extract written by scripts/build_gnomad_parquet.py: four-field identifiers
  ``chrom:pos:ref:alt`` on autosomes and X only (measured; Y and MT were outside its query, build_gnomad_ymt_af.py line 4).
  Its statistic is ``gnomad_preprocessed_mixed_af_unfiltered_v1``: the INFO field AF, else AF_genome where AF is absent,
  fallback events not recorded, VCF FILTER not applied, no AC/AN, ClinVar-ascertained query universe (completeness
  unverified), six significant digits.
* ``dataset:gnomad_ymt_api`` -- records appended 2026-06-16 by scripts/build_gnomad_ymt_af.py from the live gnomAD API:
  five-field identifiers ``gnomad:Y|MT:pos:ref:alt``. Y values are exome AF else genome AF; MT values are ac_hom / an
  (homoplasmic fraction), where a missing ac_hom became 0 (line 136), so an MT zero is ambiguous. Never selectable into
  the strict scalar.

Refused, never repaired: an identifier outside the two grammars, a key occurring twice in a logical source (no
first-wins), a non-floating frequency column, a frequency outside [0, 1] or non-finite, a cohort row whose variant_id is
not its coordinate columns, and an allele that is neither ACGT nor ACGT with unknown bases (N).

Source states are source facts (observation.py vocabulary) with source-specific details. "No row in the extracted table"
is never "not in gnomAD". Selection (policy) and claim eligibility (task population) are separate projections
(``select_strict``); neither reads labels, and selection never reads detail codes.
"""
from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from genomic_variant_classifier.information_sources.census import HEX64, canonical_bytes, sha256_bytes, symbol_closure_digest
from genomic_variant_classifier.information_sources.columns import CODE, STATE_CODES, ObservationColumn, Population, readonly_array
from genomic_variant_classifier.information_sources.materialization import Parsed, check_registered
from genomic_variant_classifier.information_sources.observation import ObservationState, SourceIdentity

logger = logging.getLogger(__name__)

SCHEMA = "gvc.allele-frequency.1d1c1/1"
RECEIPT_SCHEMA = "gvc.materialization-receipt.1d1c1/2"
PRIMARY = "dataset:gnomad_af"
YMT_API = "dataset:gnomad_ymt_api"
_THIS_FILE = "src/genomic_variant_classifier/information_sources/allele_frequency.py"
S = ObservationState

#: Every statistic the file carries, declared as measured (the builders were read in full; field names are theirs).
STATISTICS: Mapping[str, Mapping[str, Any]] = {
    "gnomad_preprocessed_mixed_af_unfiltered_v1": {
        "logical_source": PRIMARY, "builder": "scripts/build_gnomad_parquet.py", "primary_field": "AF",
        "fallback_field": "AF_genome", "fallback_rule": "exome_else_genome", "fallback_events_recorded": False,
        "filter_policy": {"applied": False, "source_filter_values_available": False},
        "allele_counts_available": False, "allele_numbers_available": False,
        "query_universe": {"kind": "clinvar_ascertained_extract", "completeness": "unverified"},
        "release": "not recorded by the extract", "serialization_precision": "6 significant digits (VCF text)",
        "claims": {"exome_only": False, "filter_qualified": False, "complete_gnomad_catalogue": False}},
    "gnomad_api_y_exome_else_genome_af_v1": {
        "logical_source": YMT_API, "builder": "scripts/build_gnomad_ymt_af.py", "quantity": "exome.af else genome.af",
        "api_dataset": "gnomad_r4 (live API, not pinned)", "fallback_events_recorded": False,
        "query_universe": {"kind": "cohort_y_genes_with_clean_symbols", "completeness": "unverified"},
        "par_records": "X-PAR records re-keyed to Y (GNOMAD-YPAR-REKEY-DUPLICATES-1)"},
    "gnomad_api_mt_homoplasmic_fraction_v1": {
        "logical_source": YMT_API, "builder": "scripts/build_gnomad_ymt_af.py", "quantity": "ac_hom / an",
        "api_dataset": "gnomad_r4 (live API, not pinned)", "zero": "a missing ac_hom became 0: a zero is ambiguous",
        "query_universe": {"kind": "mitochondrial_region_1_16569", "completeness": "one region call"}},
}
#: The only statistic the strict policy may select.
SELECTABLE_STATISTICS: frozenset[str] = frozenset({"gnomad_preprocessed_mixed_af_unfiltered_v1"})

AUTOSOMES = tuple(str(i) for i in range(1, 23))
PRIMARY_DOMAIN = frozenset(AUTOSOMES) | {"X"}
CHROMOSOMES = frozenset(AUTOSOMES) | {"X", "Y", "MT"}
_POS = r"[1-9][0-9]*"
PRIMARY_GRAMMAR = rf"(?:{'|'.join(AUTOSOMES)}|X):{_POS}:[ACGT]+:[ACGT]+"
YMT_GRAMMAR = rf"gnomad:(?:Y|MT):{_POS}:[ACGT]+:[ACGT]+"

PRIMARY_DETAILS: tuple[str, ...] = ("", "reported_zero_filter_unknown", "record_without_value", "no_row_in_extracted_table",
                                    "allele_sequence_incomplete", "chromosome_outside_extract_scope")
API_DETAILS: tuple[str, ...] = ("", "zero_ac_hom_or_missing_ac_hom", "no_row_in_api_extract", "allele_sequence_incomplete",
                                "chromosome_outside_extract_scope")


class AlleleForm(str, Enum):
    FULLY_SPECIFIED = "fully_specified"          # REF and ALT in [ACGT]+
    UNKNOWN_BASE = "unknown_base"                # [ACGTN]+ with at least one N (no symbolic allele can match this)


# ------------------------------------------------------------------------------------------------- the source table
@dataclass(frozen=True)
class SourceTable:
    """One logical source's records: key (bare chrom:pos:ref:alt), value (NaN = record without value)."""
    logical_id: str
    keys: pd.Index
    values: np.ndarray


def read_source_tables(raw_parquet: bytes) -> tuple[SourceTable, SourceTable]:
    """The two logical sources of the verified bytes, by the strict grammars. Refuses unknown forms, duplicated keys, a
    non-floating frequency column and any value outside [0, 1]."""
    g = pd.read_parquet(io.BytesIO(raw_parquet), columns=["variant_id", "allele_freq"])
    if not pd.api.types.is_float_dtype(g["allele_freq"].dtype):
        raise ValueError(f"allele_freq must be a floating-point column, not {g['allele_freq'].dtype}")
    if g["variant_id"].isna().any():
        raise ValueError(f"{int(g['variant_id'].isna().sum())} source records have no identifier")
    vid = g["variant_id"].astype(str)
    is_primary = vid.str.fullmatch(PRIMARY_GRAMMAR).to_numpy(dtype=bool)
    is_api = vid.str.fullmatch(YMT_GRAMMAR).to_numpy(dtype=bool)
    unknown = ~(is_primary | is_api)
    if unknown.any():
        raise ValueError(f"{int(unknown.sum())} source identifiers match neither grammar, e.g. "
                         f"{vid[unknown].head(3).tolist()}")
    af = g["allele_freq"].to_numpy(dtype=np.float64)
    present = ~np.isnan(af)
    bad = present & ~(np.isfinite(af) & (af >= 0.0) & (af <= 1.0))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} source frequencies lie outside [0, 1] or are not finite, e.g. "
                         f"{af[bad][:3].tolist()}; refused, never clipped")
    key = np.where(is_api, vid.str.slice(len("gnomad:")).to_numpy(dtype=object), vid.to_numpy(dtype=object))
    ref_alt_differ = pd.Series(key).str.split(":").map(lambda p: p[2] != p[3]).to_numpy(dtype=bool)
    if not ref_alt_differ.all():
        raise ValueError(f"{int((~ref_alt_differ).sum())} source identifiers have REF equal to ALT")
    tables = []
    for logical, mask in ((PRIMARY, is_primary), (YMT_API, is_api)):
        k = pd.Index(key[mask])
        if k.has_duplicates:
            dup = k[k.duplicated()].unique()
            raise ValueError(f"{logical}: {len(dup)} keys occur more than once (e.g. {list(dup[:3])}); first-wins is refused")
        tables.append(SourceTable(logical, k, readonly_array(af[mask])))
    if len(tables[0].keys.intersection(tables[1].keys)):
        raise AssertionError("the two logical sources share a key; their chromosome scopes are disjoint by grammar")
    return tables[0], tables[1]


# --------------------------------------------------------------------------------------------------- the cohort side
def _cohort_alleles(frame: pd.DataFrame, population: Population) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bare key, chromosome, allele form) per population row; the frame is read for identity columns only."""
    n = len(population)
    need = ["variant_id", "chrom", "pos", "ref", "alt"]
    missing = [c for c in need if c not in frame.columns]
    if missing:
        raise ValueError(f"the cohort frame lacks {missing}")
    f = frame[need]
    if len(f) != n or not (f["variant_id"].astype(str).to_numpy() == np.asarray(population.row_ids, dtype=object)).all():
        raise ValueError("the frame's rows are not the population's rows in population order")
    chrom = f["chrom"].astype(str).to_numpy(dtype=object)
    if not np.isin(chrom, list(CHROMOSOMES)).all():
        raise ValueError(f"unknown chromosome names: {sorted(set(chrom) - CHROMOSOMES)[:5]}")
    pos = pd.to_numeric(f["pos"], errors="coerce")
    if pos.isna().any() or (pos <= 0).any() or not (pos == pos.round()).all():
        raise ValueError("every cohort position is a positive integer")
    ref, alt = f["ref"].astype(str), f["alt"].astype(str)
    # Built and compared as Python objects: one cohort allele is 19,119 characters, and a fixed-width NumPy string array
    # of 1.49 million such rows would need over 100 GB.
    key_s = f["chrom"].astype(str) + ":" + pos.astype(np.int64).astype(str) + ":" + ref + ":" + alt
    if not ("clinvar:" + key_s).equals(f["variant_id"].astype(str)):
        raise ValueError("a cohort variant_id is not 'clinvar:' + its chrom:pos:ref:alt columns")
    key = key_s.to_numpy(dtype=object)
    full = (ref.str.fullmatch(r"[ACGT]+") & alt.str.fullmatch(r"[ACGT]+")).to_numpy(dtype=bool)
    with_n = (ref.str.fullmatch(r"[ACGTN]+") & alt.str.fullmatch(r"[ACGTN]+")).to_numpy(dtype=bool) & ~full
    if not (full | with_n).all():
        raise ValueError(f"{int((~(full | with_n)).sum())} cohort alleles are neither ACGT nor ACGT with unknown bases "
                         "(symbolic or other forms need their own adjudicated treatment)")
    if ((ref == alt).to_numpy() & full).any():
        raise ValueError("a cohort row has REF equal to ALT")
    form = np.where(full, AlleleForm.FULLY_SPECIFIED.value, AlleleForm.UNKNOWN_BASE.value).astype(object)
    return key, chrom, form


# ------------------------------------------------------------------------------------------------ the materialisation
@dataclass(frozen=True)
class AlleleFrequencyMaterialization:
    primary: ObservationColumn          # af_raw from dataset:gnomad_af
    y_api: ObservationColumn            # af_y_api from dataset:gnomad_ymt_api (Y statistic)
    mt_api: ObservationColumn           # af_hom_mt_api from dataset:gnomad_ymt_api (MT homoplasmic fraction)
    chromosome: np.ndarray              # per row (object), read-only by convention of the caller
    allele_form: np.ndarray             # per row: AlleleForm values
    statistics: Mapping[str, str]       # column feature -> statistic id
    receipt: Mapping[str, Any]

    def columns(self) -> tuple[ObservationColumn, ObservationColumn, ObservationColumn]:
        return self.primary, self.y_api, self.mt_api


def _column(feature: str, logical: str, sha: str, population: Population, values: np.ndarray, states: np.ndarray,
            details: np.ndarray, vocabulary: tuple[str, ...]) -> ObservationColumn:
    return ObservationColumn(feature, SourceIdentity(logical, sha, None), population, values.astype(np.float64),
                             states.astype(np.uint8), details.astype(np.uint16), vocabulary)


def materialize_allele_frequency(frame: pd.DataFrame, population: Population, source: Parsed, *,
                                 registered_datasets: Iterable[str]) -> AlleleFrequencyMaterialization:
    """The three evidence columns for ``population`` from the verified bytes ``source`` (read_verified_bytes)."""
    registered = tuple(registered_datasets)
    check_registered(PRIMARY, registered)
    check_registered(YMT_API, registered)
    if not isinstance(source, Parsed) or not isinstance(source.value, bytes) or not HEX64.match(source.sha256):
        raise ValueError("source must be the Parsed result of read_verified_bytes")
    primary, api = read_source_tables(source.value)
    key, chrom, form = _cohort_alleles(frame, population)
    n = len(population)
    det_p = {d: i for i, d in enumerate(PRIMARY_DETAILS)}
    det_a = {d: i for i, d in enumerate(API_DETAILS)}
    in_domain = np.isin(chrom, list(PRIMARY_DOMAIN))
    is_y, is_mt = chrom == "Y", chrom == "MT"
    unknown_base = form == AlleleForm.UNKNOWN_BASE.value

    def lookup(table: SourceTable) -> tuple[np.ndarray, np.ndarray]:
        pos = table.keys.get_indexer(pd.Index(key))
        found = pos >= 0
        vals = np.full(n, np.nan)
        vals[found] = table.values[pos[found]]
        return found, vals

    # dataset:gnomad_af
    found, vals = lookup(primary)
    v = np.full(n, np.nan)
    st = np.full(n, CODE[S.NOT_IN_SOURCE], np.uint8)
    dt = np.full(n, det_p["no_row_in_extracted_table"], np.uint16)
    measured = found & ~np.isnan(vals)
    st[measured], dt[measured], v[measured] = CODE[S.OBSERVED], det_p[""], vals[measured]
    dt[measured & (vals == 0.0)] = det_p["reported_zero_filter_unknown"]
    dt[found & np.isnan(vals)] = det_p["record_without_value"]
    oos = in_domain & unknown_base
    st[oos], dt[oos], v[oos] = CODE[S.UNSUPPORTED_VARIANT], det_p["allele_sequence_incomplete"], np.nan
    st[~in_domain], dt[~in_domain], v[~in_domain] = CODE[S.UNSUPPORTED_VARIANT], det_p["chromosome_outside_extract_scope"], np.nan
    if (found & (oos | ~in_domain)).any():
        raise AssertionError("an out-of-scope row matched a primary record; the grammar forbids it")
    primary_col = _column("af_raw", PRIMARY, source.sha256, population, v, st, dt, PRIMARY_DETAILS)

    # dataset:gnomad_ymt_api: one column per statistic
    found_a, vals_a = lookup(api)
    if np.isnan(vals_a[found_a]).any():
        raise AssertionError("the API builder never writes a record without a value (build_gnomad_ymt_af.py parse_*_af)")
    api_cols = []
    for feature, chrom_mask in (("af_y_api", is_y), ("af_hom_mt_api", is_mt)):
        v = np.full(n, np.nan)
        st = np.full(n, CODE[S.UNSUPPORTED_VARIANT], np.uint8)
        dt = np.full(n, det_a["chromosome_outside_extract_scope"], np.uint16)
        scope = chrom_mask
        st[scope], dt[scope] = CODE[S.NOT_IN_SOURCE], det_a["no_row_in_api_extract"]
        hit = scope & found_a
        st[hit], dt[hit], v[hit] = CODE[S.OBSERVED], det_a[""], vals_a[hit]
        if feature == "af_hom_mt_api":
            dt[hit & (vals_a == 0.0)] = det_a["zero_ac_hom_or_missing_ac_hom"]
        n_scope_n = scope & unknown_base
        st[n_scope_n], dt[n_scope_n], v[n_scope_n] = CODE[S.UNSUPPORTED_VARIANT], det_a["allele_sequence_incomplete"], np.nan
        api_cols.append(_column(feature, YMT_API, source.sha256, population, v, st, dt, API_DETAILS))

    if (found_a & ~(is_y | is_mt)).any():
        raise AssertionError("an autosome or X row matched an API record; the grammar forbids it")
    requested = {PRIMARY: int(found.sum()), YMT_API: int(found_a.sum())}
    receipt = _receipt([primary_col, *api_cols], source, {PRIMARY: len(primary.keys), YMT_API: len(api.keys)}, requested)
    statistics = {"af_raw": "gnomad_preprocessed_mixed_af_unfiltered_v1",
                  "af_y_api": "gnomad_api_y_exome_else_genome_af_v1",
                  "af_hom_mt_api": "gnomad_api_mt_homoplasmic_fraction_v1"}
    return AlleleFrequencyMaterialization(primary_col, api_cols[0], api_cols[1], readonly_array(chrom.astype(str)),
                                          readonly_array(form.astype(str)), statistics, receipt)


def code_identity(qualname: str) -> dict[str, str]:
    symbol = f"{_THIS_FILE}::{qualname}"
    return {"symbol": symbol, "closure_sha256": symbol_closure_digest(Path(__file__).resolve().parents[3], symbol)}


def _receipt(cols: list[ObservationColumn], source: Parsed, records: Mapping[str, int],
             requested: Mapping[str, int]) -> dict[str, Any]:
    pop = cols[0].population
    return {"schema": RECEIPT_SCHEMA, "content_sha256": source.sha256, "bytes": source.n_bytes,
            "parser": code_identity("materialize_allele_frequency"),
            "population_id": pop.population_id, "population_sha256": pop.sha256,
            "population_identity_schema": "gvc.population-order/2", "n_population": len(pop),
            "source_records": dict(records), "source_records_requested_by_population": dict(requested),
            "columns": {c.feature: {"logical_source": c.source.logical_id, "column_sha256": c.sha256,
                                    "state_counts": c.counts(), "detail_counts": c.detail_counts()} for c in cols}}


# ------------------------------------------------------------------------------------------- selection and the claim
class UnknownBaseTreatment(str, Enum):
    EXCLUDE_FROM_SEQUENCE_RESOLVED_PRIMARY = "exclude_from_sequence_resolved_primary"   # ruling b937681e, section 3


class Arm(str, Enum):
    RETAIN_REPORTED_ZEROS = "retain_reported_zeros"          # arm A: the source-reconstruction baseline
    WITHHOLD_REPORTED_ZEROS = "withhold_reported_zeros"      # arm B: sensitivity
    WITHOUT_AF = "without_af"                                # arm C: the AF family withheld


_UNSET = object()


@dataclass(frozen=True)
class AFSelectionPolicy:
    """The strict selection policy. The unknown-base treatment must be declared: there is no default."""
    unknown_base_treatment: Any = _UNSET
    policy_id: str = "strict_primary_exome_autosomes_x_v2"
    statistic: str = "gnomad_preprocessed_mixed_af_unfiltered_v1"
    primary_domain: tuple[str, ...] = (*AUTOSOMES, "X")
    arm: Arm = Arm.RETAIN_REPORTED_ZEROS

    def __post_init__(self) -> None:
        if self.unknown_base_treatment is _UNSET or not isinstance(self.unknown_base_treatment, UnknownBaseTreatment):
            raise ValueError("the policy must declare its unknown-base treatment (an UnknownBaseTreatment); none is assumed")
        if self.statistic not in SELECTABLE_STATISTICS:
            raise ValueError(f"statistic {self.statistic!r} is not selectable; only {sorted(SELECTABLE_STATISTICS)}")
        if tuple(self.primary_domain) != (*AUTOSOMES, "X"):
            raise ValueError("the primary domain of this policy is autosomes 1-22 and X")
        if not isinstance(self.arm, Arm):
            raise ValueError("arm must be an Arm")

    def as_dict(self) -> dict[str, Any]:
        return {"policy_id": self.policy_id, "statistic": self.statistic, "primary_domain": list(self.primary_domain),
                "unknown_base_treatment": self.unknown_base_treatment.value, "arm": self.arm.value}


CLAIM_STATES: tuple[str, ...] = ("eligible", "incomplete_allele", "outside_domain")
SELECTION_STATES: tuple[str, ...] = ("selected", "no_qualified_observation", "allele_sequence_incomplete",
                                     "policy_out_of_scope", "withheld_zero_sensitivity", "withheld_af_ablation")


@dataclass(frozen=True)
class Selection:
    af_model: np.ndarray           # float64, NaN unless selected; read-only
    selection: np.ndarray          # uint8 codes into SELECTION_STATES; read-only
    claim: np.ndarray              # uint8 codes into CLAIM_STATES; read-only
    identity: str
    policy: Mapping[str, Any]
    selection_vocabulary: tuple[str, ...] = SELECTION_STATES
    claim_vocabulary: tuple[str, ...] = CLAIM_STATES

    def census(self) -> dict[str, int]:
        sel = self.selection == SELECTION_STATES.index("selected")
        elig = self.claim == CLAIM_STATES.index("eligible")
        out = {"full": int(len(self.claim)), "primary_domain": int((self.claim != CLAIM_STATES.index("outside_domain")).sum()),
               "sequence_resolved_primary": int(elig.sum()), "selected": int(sel.sum()),
               "eligible_unselected": int((elig & ~sel).sum()),
               "unknown_base": int((self.claim == CLAIM_STATES.index("incomplete_allele")).sum()),
               "outside_domain": int((self.claim == CLAIM_STATES.index("outside_domain")).sum()),
               "selected_reported_zero": int((sel & (self.af_model == 0.0)).sum())}
        if out["selected"] + out["eligible_unselected"] + out["unknown_base"] + out["outside_domain"] != out["full"]:
            raise AssertionError("the population partition does not reconcile")
        return out


def select_strict(m: AlleleFrequencyMaterialization, policy: AFSelectionPolicy, *, reference_verified: np.ndarray,
                  reference_evidence_sha256: str) -> Selection:
    """Claim eligibility (from chromosome, allele form and reference evidence -- never from AF or labels) and the strict
    model value (from source STATES and values only -- never from detail codes)."""
    if not isinstance(policy, AFSelectionPolicy):
        raise ValueError("policy must be an AFSelectionPolicy")
    if not isinstance(reference_evidence_sha256, str) or not HEX64.match(reference_evidence_sha256):
        raise ValueError("reference_evidence_sha256 must be the SHA-256 of the reference-identity evidence")
    p = m.primary
    n = len(p.population)
    ref_ok = np.asarray(reference_verified)
    if ref_ok.dtype != bool or ref_ok.shape != (n,):
        raise ValueError("reference_verified is one boolean per population row")
    in_domain = np.isin(m.chromosome, list(PRIMARY_DOMAIN))
    unknown = m.allele_form == AlleleForm.UNKNOWN_BASE.value
    if (unknown & ref_ok).any():
        raise ValueError("an allele with unknown bases cannot carry verified reference identity")
    if (in_domain & ~unknown & ~ref_ok).any():
        raise ValueError(f"{int((in_domain & ~unknown & ~ref_ok).sum())} fully specified primary-domain alleles lack "
                         "verified reference identity")
    claim = np.full(n, CLAIM_STATES.index("outside_domain"), np.uint8)
    claim[in_domain] = CLAIM_STATES.index("eligible")
    claim[in_domain & unknown] = CLAIM_STATES.index("incomplete_allele")
    observed = p.states == CODE[S.OBSERVED]
    eligible = claim == CLAIM_STATES.index("eligible")
    sel = np.full(n, SELECTION_STATES.index("no_qualified_observation"), np.uint8)
    sel[~in_domain] = SELECTION_STATES.index("policy_out_of_scope")
    sel[in_domain & unknown] = SELECTION_STATES.index("allele_sequence_incomplete")
    chosen = eligible & observed
    af = np.full(n, np.nan)
    if policy.arm is Arm.WITHOUT_AF:
        sel[chosen] = SELECTION_STATES.index("withheld_af_ablation")
    else:
        take = chosen.copy()
        if policy.arm is Arm.WITHHOLD_REPORTED_ZEROS:
            zero = chosen & (p.values == 0.0)
            sel[zero] = SELECTION_STATES.index("withheld_zero_sensitivity")
            take &= ~zero
        sel[take] = SELECTION_STATES.index("selected")
        af[take] = p.values[take]
    identity = sha256_bytes(canonical_bytes({
        "schema": SCHEMA, "policy": policy.as_dict(), "primary_column_sha256": p.sha256,
        "reference_evidence_sha256": reference_evidence_sha256, "code": code_identity("select_strict")}))
    return Selection(readonly_array(af), readonly_array(sel), readonly_array(claim), identity, policy.as_dict())


# ------------------------------------------------------------------------------------------- rare-flag description
def audit_binary_flag(*, n_positive: int, n_negative: int, flagged_positive: int, flagged_negative: int) -> dict[str, Any]:
    """A rare indicator described beyond AUROC: counts, within-group prevalence with its exact (Clopper-Pearson) 95%
    interval, positive likelihood ratio with its log-method 95% interval, the AUROC of the binary flag (ties count half)
    and the mutual information with the label in bits. Descriptive; not a test of a causal or transportable effect."""
    for x in (n_positive, n_negative, flagged_positive, flagged_negative):
        if type(x) is not int or x < 0:
            raise ValueError("counts are non-negative integers")
    if n_positive == 0 or n_negative == 0 or flagged_positive > n_positive or flagged_negative > n_negative:
        raise ValueError("invalid class denominators or flagged counts")
    from scipy.stats import beta

    a, b = flagged_positive, flagged_negative
    k = a + b
    tpr, fpr = a / n_positive, b / n_negative
    lo = 0.0 if a == 0 else float(beta.ppf(0.025, a, k - a + 1))
    hi = 1.0 if a == k else float(beta.ppf(0.975, a + 1, k - a))
    if a and b:
        lr = tpr / fpr
        se = math.sqrt(1 / a - 1 / n_positive + 1 / b - 1 / n_negative)
        lr_ci: list[float] | None = [lr * math.exp(-1.959963984540054 * se), lr * math.exp(1.959963984540054 * se)]
    else:
        lr, lr_ci = (math.inf if a else 0.0), None

    def h(q: float) -> float:
        return 0.0 if q in (0.0, 1.0) else -(q * math.log2(q) + (1 - q) * math.log2(1 - q))
    total = n_positive + n_negative
    pm = k / total
    mi = h(n_positive / total) - (pm * h(a / k if k else 0.0) + (1 - pm) * h((n_positive - a) / (total - k)))
    return {"n": k, "pathogenic": a, "benign": b, "prevalence": a / k if k else math.nan, "prevalence_95ci": [lo, hi],
            "lr_plus": lr, "lr_plus_95ci": lr_ci, "auroc": 0.5 * (1 + tpr - fpr), "mutual_information_bits": mi}


__all__ = ["API_DETAILS", "AFSelectionPolicy", "AlleleForm", "AlleleFrequencyMaterialization", "Arm", "CLAIM_STATES",
           "PRIMARY", "PRIMARY_DETAILS", "PRIMARY_GRAMMAR", "RECEIPT_SCHEMA", "SCHEMA", "SELECTABLE_STATISTICS",
           "SELECTION_STATES", "STATISTICS", "Selection", "SourceTable", "UnknownBaseTreatment", "YMT_API",
           "YMT_GRAMMAR", "audit_binary_flag", "code_identity", "materialize_allele_frequency", "read_source_tables",
           "select_strict"]
