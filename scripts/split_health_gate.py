#!/usr/bin/env python3
"""split_health_gate.py -- Author: Monzia Moodie

A precise GO/NO_GO gate on split feature health, to run AFTER a data-prep/re-prep
and BEFORE training (locally or on the VM). It improves on the DataReadiness
">=50% degenerate" heuristic, which conflates known stubs with real breakage and
could pass a split where a CORE predictor silently died (as long as <50% overall
are dead). This gate sorts hard-degenerate columns (ALL_ZERO / ALL_NULL / CONSTANT)
into three buckets:

  * CORE_FEATURES degenerate          -> NO_GO (a predictor we rely on is dead)
  * UNEXPECTED degenerate             -> NO_GO (a should-be-live feature is dead,
                                          e.g. a re-prep that did not actually revive gtex_*/af_1kg_*)
  * EXPECTED_ZERO (known stubs) deg.  -> allowed (eve/alphafold/clingen/omim/hgmd/phylop, etc.)

NEAR_CONSTANT (naturally-rare binaries like is_mitochondrial) is a WARNING, not a
gate failure. This is the automated form of "rerun on the new splits to confirm
the dead columns came alive": after re-prep the only degenerate columns should be
EXPECTED_ZERO stubs, and CORE + the previously-stale families must be healthy.

The classify() core is pure/import-free and unit-tested. main() globs the splits
and scores columns via genomic_variant_classifier.data.feature_health.col_health
(the same library DataReadiness + audit_split_feature_health use).

Usage:
  python scripts/split_health_gate.py --splits-dir outputs/<run>/full/splits
Exit: 0 GO | 1 NO_GO | 2 splits not found.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Columns permitted to be degenerate even after a full re-prep: data is genuinely
# absent/blocked (registry STUB/BLOCKED) or the column is a non-numeric id/seq that
# should not be a model feature (flagged for removal from X, not a training blocker).
EXPECTED_ZERO = {
    # genuine stub / blocked / absent-data sources
    "eve_score",                                                     # EVE (needs HGVSp + EVE data)
    "alphafold_plddt", "dist_to_active_site", "solvent_accessibility",
    "secondary_structure_context",                                  # AlphaFold structure (stub)
    "clingen_validity_score",                                       # ClinGen (stub / 404)
    "omim_n_diseases", "omim_is_autosomal_dominant",                # OMIM (stub)
    "hgmd_is_disease_mutation", "hgmd_n_reports",                   # HGMD (procurement-blocked)
    "phylop_score",                                                 # PhyloP (absent)
    "esm2_delta_norm",                                             # ESM-2 (gated on HGVSp parser, still outstanding)
    # non-numeric id/sequence columns that should not be model features (investigate: drop from X)
    "fasta_seq", "protein_change", "transcript_id", "source_db", "lovd_variant_class",
}

# Predictors that MUST be healthy; any degeneracy here is an immediate NO_GO.
CORE_FEATURES = {
    "alphamissense_score", "revel_score", "revel_pathogenic", "sift_score",
    "sift_deleterious", "cadd_phred", "cadd_high", "splice_ai_score", "is_splice",
    "gnn_score", "hetero_gnn_score", "n_pathogenic_in_gene", "loeuf", "gerp_score",
    "pli_score", "consequence_severity",
}

# Core features that prep.run() does NOT emit -- they are written during the GNN
# training stage (run_phase2_eval re-persists X_* with gnn_score/hetero_gnn_score).
# In a prep-only validation these are legitimately ABSENT, so their absence must
# not trip the presence check; in a full post-training split they must be present.
GNN_STAGE_FEATURES = {"gnn_score", "hetero_gnn_score"}

# Reasons that count as HARD degeneracy (gating). NEAR_CONSTANT is a soft warning.
_HARD = ("ALL_ZERO", "ALL_NULL", "CONSTANT")


def is_hard_degenerate(reason: str) -> bool:
    """True for ALL_ZERO/ALL_NULL/CONSTANT; False for NEAR_CONSTANT-only / healthy."""
    if not reason:
        return False
    # "CONSTANT" matches, but a bare "NEAR_CONSTANT(..)" must NOT count as CONSTANT.
    tokens = {t.split("(")[0] for t in reason.split(";")}
    return any(h in tokens for h in _HARD)


def reason_from_health(health) -> str:
    """Extract the degeneracy-reason STRING from a feature_health.col_health() return.

    The library contract is: ``col_health(s, near_constant) -> dict`` whose
    ``"degenerate"`` key is ``";".join(reasons)`` (or ``""`` when healthy). We require
    that exact shape and raise LOUDLY if it ever changes, so a future feature_health
    refactor can never silently neuter this gate (passing the raw dict downstream is
    precisely the bug this guards against)."""
    if not isinstance(health, dict) or "degenerate" not in health:
        raise TypeError(
            "col_health() must return a dict with a 'degenerate' key; got "
            f"{type(health).__name__} -- feature_health contract changed?")
    return health["degenerate"] or ""


def classify(degenerate: dict[str, str], *, present=None, prep_only: bool = False,
             expected_zero=EXPECTED_ZERO, core_features=CORE_FEATURES,
             max_unexpected: int = 0) -> dict:
    """degenerate: {column: reason}. ``present`` (optional) is the set of all column
    names seen across the splits; when supplied, a CORE feature that is silently
    ABSENT (not just degenerate) is a NO_GO -- absence is a failure mode too. In
    ``prep_only`` mode the GNN-stage features are exempt from the presence check
    (prep.run() does not emit them; they are added during GNN training). Pure."""
    hard = {c for c, r in degenerate.items() if is_hard_degenerate(r)}
    near = sorted(c for c, r in degenerate.items()
                  if c not in hard and "NEAR_CONSTANT" in (r or ""))
    core_deg = sorted(hard & set(core_features))
    expected_deg = sorted(hard & set(expected_zero))
    unexpected_deg = sorted(hard - set(expected_zero) - set(core_features))

    missing_core: list[str] = []
    if present is not None:
        exempt = GNN_STAGE_FEATURES if prep_only else set()
        missing_core = sorted((set(core_features) - exempt) - set(present))

    reasons, verdict = [], "GO"
    if core_deg:
        verdict = "NO_GO"
        reasons.append(f"{len(core_deg)} CORE feature(s) degenerate: {core_deg}")
    if missing_core:
        verdict = "NO_GO"
        reasons.append(f"{len(missing_core)} CORE feature(s) ABSENT from splits "
                       f"(silent dropout): {missing_core}")
    if len(unexpected_deg) > max_unexpected:
        verdict = "NO_GO"
        reasons.append(f"{len(unexpected_deg)} unexpected degenerate (not known stubs) -- "
                       f"re-prep did not revive these: {unexpected_deg}")
    if verdict == "GO":
        reasons.append(f"GO: only {len(expected_deg)} expected-stub column(s) degenerate; "
                       f"core features + previously-stale families healthy")
    return {"verdict": verdict, "core_degenerate": core_deg, "missing_core": missing_core,
            "unexpected_degenerate": unexpected_deg, "expected_degenerate": expected_deg,
            "near_constant_warnings": near, "reasons": reasons}


# ----------------------------------------------------------------------------- main
def _score_splits(splits_dir: Path, near_constant_frac: float,
                  glob: str = "X_*.parquet") -> tuple[dict[str, str], set]:
    """Score the FEATURE MATRIX (X_*) columns. Returns (degenerate, present) where a
    column degenerate in ANY split file is reported degenerate (matching
    audit_split_feature_health) and ``present`` is every column seen. Reads fail
    LOUDLY -- an unreadable split must halt the gate, never be silently skipped."""
    import pandas as pd
    from genomic_variant_classifier.data.feature_health import col_health
    splits_dir = Path(splits_dir)
    degenerate: dict[str, str] = {}
    present: set = set()
    files = sorted(splits_dir.rglob(glob))
    if not files:
        raise FileNotFoundError(f"no parquet matching {glob!r} under {splits_dir}")
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception as e:                       # loud, not silent-skip
            raise RuntimeError(f"failed reading split parquet {f}: {e}") from e
        present.update(map(str, df.columns))
        for c in df.columns:
            why = reason_from_health(col_health(df[c], near_constant_frac))
            if why:                                  # degenerate in THIS file
                degenerate[c] = degenerate.get(c) or why
    return degenerate, present


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--glob", default="X_*.parquet",
                    help="parquet glob for the feature matrix (default X_*.parquet)")
    ap.add_argument("--near-constant-frac", type=float, default=0.999)
    ap.add_argument("--max-unexpected", type=int, default=0)
    ap.add_argument("--prep-only", action="store_true",
                    help="splits came from prep.run() without the GNN stage; exempt "
                         "gnn_score/hetero_gnn_score from the core-presence check")
    args = ap.parse_args(argv)

    # sanity: the two curated sets must be disjoint
    overlap = EXPECTED_ZERO & CORE_FEATURES
    if overlap:
        print(f"CONFIG ERROR: EXPECTED_ZERO and CORE_FEATURES overlap: {sorted(overlap)}")
        return 2

    sd = Path(args.splits_dir)
    if not sd.exists():
        print(f"splits dir not found: {sd.resolve()} -- STOP.")
        return 2
    try:
        degenerate, present = _score_splits(sd, args.near_constant_frac, args.glob)
    except FileNotFoundError as e:
        print(f"{e} -- STOP.")
        return 2
    res = classify(degenerate, present=present, prep_only=args.prep_only,
                   max_unexpected=args.max_unexpected)

    print(f"split-health gate  --  {sd.resolve()}"
          + ("  [prep-only]" if args.prep_only else ""))
    print(f"  VERDICT: {res['verdict']}")
    for r in res["reasons"]:
        print(f"   - {r}")
    if res["expected_degenerate"]:
        print(f"  expected-stub degenerate ({len(res['expected_degenerate'])}): "
              f"{res['expected_degenerate']}")
    if res["near_constant_warnings"]:
        print(f"  near-constant warnings (not gating): {res['near_constant_warnings']}")
    return 0 if res["verdict"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
