#!/usr/bin/env python3
"""split_health_gate.py -- Author: Monzia Moodie

GO/NO_GO gate on split feature health, to run AFTER a prep/re-prep and BEFORE
training (locally via regen_splits_local.py, or in-run via gate_frames()). It sorts
hard-degenerate columns (ALL_ZERO / ALL_NULL / CONSTANT) into buckets and accounts
for the pipeline's real staging semantics so it does not false-alarm:

  * CORE_FEATURES degenerate                  -> NO_GO (a predictor we rely on is dead)
  * CORE absent (silent dropout)              -> NO_GO
  * UNEXPECTED degenerate                     -> NO_GO (a should-be-live feature is dead)
  * EXPECTED_ZERO (known stubs/unwired) deg.  -> allowed

Staging semantics (verified against real_data_prep.py @9f9ced7):
  * GNN_STAGE_FEATURES (gnn_score, hetero_gnn_score) are written as 0.5 placeholders by
    prep.run() and overwritten by the GNN training stage. Under --prep-only they are
    exempt from BOTH the presence and the degeneracy checks.
  * TRAIN_ONLY_FEATURES (n_pathogenic_in_gene, gene_has_known_disease) are recomputed
    train-only post-split (leakage fix INCIDENT_2026-06-13); with gene-disjoint splits
    they are legitimately zero in val/test. They are scored on the TRAIN frame ONLY.

NEAR_CONSTANT (naturally-rare binaries like is_mitochondrial) is a WARNING, not a gate
failure. The classify() core is pure and unit-tested.

EXPECTED_ZERO is wiring-dependent and MUST be updated as sources activate (e.g. when
--eve-path / --omim-path are wired into run_phase2_eval, move eve_score/omim_* OUT of
EXPECTED_ZERO into enforced columns). See the per-entry notes.

Usage:
  python scripts/split_health_gate.py --splits-dir outputs/<run>/full/splits [--prep-only]
Exit: 0 GO | 1 NO_GO | 2 splits not found.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Columns permitted to be degenerate: data genuinely absent/blocked, the source is not
# wired into run_phase2_eval yet, or the column is a non-numeric id/seq not used as a
# model feature. Each entry notes WHY + what would move it out of this set.
EXPECTED_ZERO = {
    # --- genuine stub / blocked / no-data sources (no near-term activation) ---
    "alphafold_plddt", "dist_to_active_site", "solvent_accessibility",
    "secondary_structure_context",                                 # AlphaFold structure (no bulk .cif)
    "clingen_validity_score",                                      # ClinGen (no data / unwired)
    "hgmd_is_disease_mutation", "hgmd_n_reports",                  # HGMD (procurement-blocked)
    "phylop_score",                                               # PhyloP (no bigWig yet)
    # --- IN SCOPE to wire this cycle: data available, arg just not wired -> move OUT once wired ---
    "eve_score",                                                  # EVE: connector fixed; needs --eve-path + HGVSp for coverage
    "omim_n_diseases", "omim_is_autosomal_dominant",              # OMIM: license held; needs --omim-path (mim2gene)
    "dbsnp_af",                                                   # dbSNP: GCF file in Downloads; needs move + --dbsnp-path
    "has_uniprot_annotation", "n_known_pathogenic_protein_variants",  # UniProt gene features: prep.run() not passed uniprot_path
    # --- VEP / sequence dependent (RNA pipeline defaults; need VEP exon + dist_to_donor/acceptor + fasta_seq) ---
    "codon_position", "exon_number", "dist_to_splice_site", "is_canonical_splice",
    "maxentscan_score", "maxentscan_delta",
    # --- GTEx eQTL trio: bulk --gtex-path mode provides expression only; eQTL defaults to 0 BY DESIGN ---
    "gtex_is_eqtl", "gtex_min_eqtl_pval", "gtex_max_abs_effect",
    # --- ESM-2 derived (stubbed pending HGVSp parser) ---
    "esm2_delta_norm", "esm2_llr",
    # --- non-numeric id/sequence columns (not model features; investigate dropping from X) ---
    "fasta_seq", "protein_change", "transcript_id", "source_db", "lovd_variant_class",
}

# Predictors that MUST be healthy (in the appropriate frame); degeneracy -> NO_GO.
CORE_FEATURES = {
    "alphamissense_score", "revel_score", "revel_pathogenic", "sift_score",
    "sift_deleterious", "cadd_phred", "cadd_high", "splice_ai_score", "is_splice",
    "gnn_score", "hetero_gnn_score", "n_pathogenic_in_gene", "loeuf", "gerp_score",
    "pli_score", "consequence_severity",
}

# Written as 0.5 placeholders by prep.run(); overwritten at the GNN stage. Exempt from
# presence AND degeneracy under --prep-only.
GNN_STAGE_FEATURES = {"gnn_score", "hetero_gnn_score"}

# Recomputed train-only post-split (leakage fix); zero in val/test by design with
# gene-disjoint splits. Scored on the TRAIN frame only.
TRAIN_ONLY_FEATURES = {"n_pathogenic_in_gene", "gene_has_known_disease"}

_HARD = ("ALL_ZERO", "ALL_NULL", "CONSTANT")


def is_hard_degenerate(reason: str) -> bool:
    """True for ALL_ZERO/ALL_NULL/CONSTANT; False for NEAR_CONSTANT-only / healthy."""
    if not reason:
        return False
    tokens = {t.split("(")[0] for t in reason.split(";")}
    return any(h in tokens for h in _HARD)


def reason_from_health(health) -> str:
    """Extract the degeneracy-reason STRING from feature_health.col_health()'s dict.
    Raises LOUDLY if the contract changes, so a refactor cannot silently neuter the gate."""
    if not isinstance(health, dict) or "degenerate" not in health:
        raise TypeError(
            "col_health() must return a dict with a 'degenerate' key; got "
            f"{type(health).__name__} -- feature_health contract changed?")
    return health["degenerate"] or ""


def classify(degenerate: dict[str, str], *, present=None, prep_only: bool = False,
             expected_zero=EXPECTED_ZERO, core_features=CORE_FEATURES,
             gnn_stage=GNN_STAGE_FEATURES, max_unexpected: int = 0) -> dict:
    """degenerate: {column: reason} (already train-only-aware; see _accumulate). Pure."""
    hard = {c for c, r in degenerate.items() if is_hard_degenerate(r)}
    near = sorted(c for c, r in degenerate.items()
                  if c not in hard and "NEAR_CONSTANT" in (r or ""))
    if prep_only:
        hard = hard - set(gnn_stage)               # placeholders, filled at GNN stage

    core_deg = sorted(hard & set(core_features))
    expected_deg = sorted(hard & set(expected_zero))
    unexpected_deg = sorted(hard - set(expected_zero) - set(core_features))

    missing_core: list[str] = []
    if present is not None:
        exempt = set(gnn_stage) if prep_only else set()
        missing_core = sorted((set(core_features) - exempt) - set(present))

    reasons, verdict = [], "GO"
    if core_deg:
        verdict = "NO_GO"
        reasons.append(f"{len(core_deg)} CORE feature(s) degenerate: {core_deg}")
    if missing_core:
        verdict = "NO_GO"
        reasons.append(f"{len(missing_core)} CORE feature(s) ABSENT (silent dropout): {missing_core}")
    if len(unexpected_deg) > max_unexpected:
        verdict = "NO_GO"
        reasons.append(f"{len(unexpected_deg)} unexpected degenerate (not known stubs) -- "
                       f"re-prep did not revive these: {unexpected_deg}")
    if verdict == "GO":
        reasons.append(f"GO: only {len(expected_deg)} expected-stub column(s) degenerate; "
                       f"core + previously-stale families healthy")
    return {"verdict": verdict, "core_degenerate": core_deg, "missing_core": missing_core,
            "unexpected_degenerate": unexpected_deg, "expected_degenerate": expected_deg,
            "near_constant_warnings": near, "reasons": reasons}


# ----------------------------------------------------------------------------- scoring
def _accumulate(degenerate: dict, present: set, frame_name: str, df,
                near_constant_frac: float, train_only=TRAIN_ONLY_FEATURES) -> None:
    """Fold one split frame into (degenerate, present). TRAIN_ONLY_FEATURES are scored
    only on the train frame (zero in val/test is by design with gene-disjoint splits)."""
    from genomic_variant_classifier.data.feature_health import col_health
    is_train = "train" in frame_name.lower()
    present.update(map(str, df.columns))
    for c in df.columns:
        if c in train_only and not is_train:
            continue
        why = reason_from_health(col_health(df[c], near_constant_frac))
        if why:
            degenerate[c] = degenerate.get(c) or why


def gate_frames(frames, *, near_constant_frac: float = 0.999,
                prep_only: bool = False, max_unexpected: int = 0) -> dict:
    """Score in-memory frames (e.g. {'X_train':df,'X_val':df,'X_test':df}) and classify.
    For the post-prep, pre-train gate inside run_phase2_eval. Use prep_only=True there."""
    degenerate: dict[str, str] = {}
    present: set = set()
    for name, df in frames.items():
        _accumulate(degenerate, present, name, df, near_constant_frac)
    return classify(degenerate, present=present, prep_only=prep_only,
                    max_unexpected=max_unexpected)


def _score_splits(splits_dir, near_constant_frac: float,
                  glob: str = "X_*.parquet") -> tuple[dict[str, str], set]:
    """Score the feature matrix (X_*) files. Reads fail LOUDLY (no silent skip)."""
    import pandas as pd
    splits_dir = Path(splits_dir)
    degenerate: dict[str, str] = {}
    present: set = set()
    files = sorted(splits_dir.rglob(glob))
    if not files:
        raise FileNotFoundError(f"no parquet matching {glob!r} under {splits_dir}")
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            raise RuntimeError(f"failed reading split parquet {f}: {e}") from e
        _accumulate(degenerate, present, f.stem, df, near_constant_frac)
    return degenerate, present


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--glob", default="X_*.parquet")
    ap.add_argument("--near-constant-frac", type=float, default=0.999)
    ap.add_argument("--max-unexpected", type=int, default=0)
    ap.add_argument("--prep-only", action="store_true",
                    help="splits came from prep.run() without the GNN stage; exempt "
                         "gnn_score/hetero_gnn_score from presence + degeneracy checks")
    args = ap.parse_args(argv)

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

    print(f"split-health gate  --  {sd.resolve()}" + ("  [prep-only]" if args.prep_only else ""))
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
