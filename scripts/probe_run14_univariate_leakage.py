#!/usr/bin/env python
"""
probe_run14_univariate_leakage.py  (2026-07-08)  -- READ-ONLY. Writes nothing.
==========================================================================
APPLIES THE METRIC STACK (genomic_variant_classifier.evaluation.metrics) to Run 14's
surviving splits, to answer questions that need no model predictions:

    WHICH SINGLE FEATURE, ON ITS OWN, PREDICTS THE LABEL?
    AND WAS `n_pathogenic_in_gene` A LIVE LEAK IN RUN 14, OR ALREADY A DEAD FEATURE?

WHY, AND WHY IT MATTERS MORE THAN THE ABLATIONS

    Run 14 (80ac62c, 2026-05-26) reported test AUROC 0.9975. Its feature matrix carries
    `n_pathogenic_in_gene`. real_data_prep.py:1687-1690 records:

        # --- Leakage fix (INCIDENT_2026-06-13): train-only n_pathogenic_in_gene
        # enrich_gene_counts() computes this count corpus-wide (train+val+test)
        # PRE-split. With the gene-disjoint GroupShuffleSplit above, a held-out
        # gene's count would derive entirely from its own held-out labels ...

    That fix is dated 2026-06-13 -- EIGHTEEN DAYS AFTER Run 14.

    `outputs/ablation_run15/ablation_results.parquet` shows `no_gene_prevalence` costs
    only ΔAUROC 0.0002. That does NOT exonerate Run 14: Run 15 POSTDATES the fix, so its
    `n_pathogenic_in_gene` is identically zero on the gene-disjoint test set
    (tests/unit/test_d1_d2.py::TestRecomputeNpig::test_test_npig_always_zero). Zeroing a
    feature that is already zero changes nothing by construction.

    Run 14's splits predate the fix. They are the only surviving evidence.

    The matrix also carries `hgmd_is_disease_mutation`, `hgmd_n_reports`,
    `lovd_variant_class`, `clingen_validity_score`, `n_known_pathogenic_protein_variants`.
    HGMD is a curated register of disease-causing mutations. **No ablation exists for any
    of them.** AUROC 0.998 on gene-disjoint ClinVar is far above published metapredictors
    (REVEL/CADD/AlphaMissense: 0.85-0.95) and demands an explanation.

    `oof_predictions.parquet` cannot help: 1,017,633 rows against X_train's 1,197,216,
    with no variant_id, index, or fold column to join on -- because run_phase2_eval.py
    never calls RunArtifactWriter, whose save_oof_predictions asserts that key exists.

WHAT IT REPORTS

    1. DEAD FEATURES (nunique <= 1). Run 14 trained on 78 features of which several
       (af_1kg_*, dbsnp_af, phylop_score) are identically zero for every row.
    2. UNIVARIATE POWER: standalone AUROC of each feature against y, direction-agnostic,
       plus AUPRC against the no-skill floor (pos_rate) and the finite-value coverage
       (a feature with NaNs is scored on a DIFFERENT subpopulation -- say so).
    3. LEAK CANDIDATES: any feature whose standalone power exceeds the threshold. A single
       biological annotation should not separate the classes on its own.
    4. STRATIFICATION of the strongest features by variant representation, because a
       headline over a cohort that is 86% SNVs and 8.5% padded deletions (the latter 3.5x
       pathogenic-enriched, with sixteen constant features) describes neither.

THREE-WAY VERDICT ON n_pathogenic_in_gene (fixed before the data is seen)

    A. nunique > 1 on TEST and standalone power >= 0.90
         -> LIVE PRE-FIX LEAK. With a gene-disjoint split the value derives from the
            held-out gene's own labels. Run 14's 0.9975 is not a measurement of variant
            pathogenicity. Every regime-v0 number (Runs 9-14) is void, independently of
            the coordinate bug.
    B. nunique == 1 on TEST (dead)
         -> POST-FIX SIGNATURE. Not a leak here, but a TRAIN/TEST DISTRIBUTION SHIFT: the
            model trains on values in [0, N] and predicts with the feature pinned at 0.
            The remedy is removal, not zeroing.
    C. nunique > 1 but power < 0.90
         -> present and varying, but not separating on its own. Report the power; the
            multivariate contribution still needs the ablation, read with sec 6.2.3 of
            the incident report.

    Independently: any of the HGMD / LOVD / ClinGen suspects with nunique > 1 and power
    >= 0.80 is a curated disease register inside the feature matrix. Justify in METHODS
    or remove.

USAGE (from project root, .venv312 active)
    python scripts/probe_run14_univariate_leakage.py
    python scripts/probe_run14_univariate_leakage.py --split train --top 30
    python scripts/probe_run14_univariate_leakage.py --x outputs/run14/full/X_test.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from genomic_variant_classifier.evaluation.metrics import (
    auroc, auprc, no_skill_auprc, stratified_evaluate,
)

# Features whose presence in a training matrix demands justification.
SUSPECT = [
    "n_pathogenic_in_gene",                 # INCIDENT_2026-06-13: corpus-wide, pre-split
    "n_known_pathogenic_protein_variants",
    "gene_has_known_disease",
    "hgmd_is_disease_mutation",             # curated disease-mutation register
    "hgmd_n_reports",
    "lovd_variant_class",
    "clingen_validity_score",
]
DECISIVE = "n_pathogenic_in_gene"

LEAK_THRESHOLD = 0.90
SUSPICIOUS_THRESHOLD = 0.80


def representation(ref: str, alt: str) -> str:
    if not ref or not alt:
        return "empty"
    if len(ref) == 1 and len(alt) == 1:
        return "SNV"
    if len(alt) < len(ref) and ref.startswith(alt):
        return "padded_deletion"
    if len(ref) < len(alt) and alt.startswith(ref):
        return "padded_insertion"
    if ref[0] == alt[0]:
        return "padded_other"
    return "delins"


def _numeric(col: pd.Series) -> np.ndarray:
    """Coerce safely. A nullable Int64 holding pd.NA raises under .to_numpy(float)."""
    return pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Univariate leakage panel via the metric stack.")
    ap.add_argument("--splits", default="outputs/run14/full/splits")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--x", default=None, help="override X path (two candidates exist for test)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=0)
    a = ap.parse_args(argv)

    sp = Path(a.splits)
    xp = Path(a.x) if a.x else sp / f"X_{a.split}.parquet"
    yp, mp = sp / f"y_{a.split}.parquet", sp / f"meta_{a.split}.parquet"
    for p in (xp, yp, mp):
        if not p.exists():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            return 2

    print("=" * 86)
    print(f"UNIVARIATE LEAKAGE PANEL  (metric stack)   split={a.split}")
    print(f"  X    {xp}  ({xp.stat().st_size:,} B)")
    print(f"  y    {yp}\n  meta {mp}")
    print("=" * 86)

    X = pd.read_parquet(xp)
    y = pd.read_parquet(yp).iloc[:, 0].to_numpy().astype(int)
    meta = pd.read_parquet(mp)
    if not (len(X) == len(y) == len(meta)):
        print(f"ABORT: length mismatch X={len(X):,} y={len(y):,} meta={len(meta):,}", file=sys.stderr)
        print("Row alignment is positional and cannot be assumed. Stop.", file=sys.stderr)
        return 3

    base = no_skill_auprc(y)
    print(f"rows {len(X):,}   features {X.shape[1]}   pos_rate {100*base:.4f}%  "
          f"(= AUPRC no-skill floor)")

    # --- 1. dead features ---------------------------------------------------
    nuniq = X.nunique(dropna=False)
    dead = sorted(nuniq[nuniq <= 1].index)
    print(f"\n--- DEAD FEATURES (nunique <= 1): {len(dead)} of {X.shape[1]} ---")
    print("  " + (", ".join(dead) if dead else "(none)"))
    if dead:
        print("  These contributed nothing. They are not zeros the model learned from;")
        print("  they are columns that never varied.")

    # --- 2. univariate power ------------------------------------------------
    live = [c for c in X.columns if c not in dead]
    rows = []
    for c in live:
        s = _numeric(X[c])
        finite = np.isfinite(s)
        if finite.sum() < 2 or len(np.unique(y[finite])) < 2:
            continue
        au = auroc(y, s)
        if not np.isfinite(au):
            continue
        rows.append({
            "feature": c,
            "auroc": au,
            "power": max(au, 1 - au),          # direction-agnostic separability
            "auprc": auprc(y, s if au >= 0.5 else -s),
            "nunique": int(nuniq[c]),
            "pct_finite": round(100 * finite.mean(), 3),
        })
    if not rows:
        print("ABORT: no feature could be scored.", file=sys.stderr)
        return 3
    tab = pd.DataFrame(rows).set_index("feature")
    tab["auprc_lift"] = tab["auprc"] / base
    tab = tab.sort_values("power", ascending=False)

    print(f"\n--- TOP {a.top} FEATURES BY STANDALONE DISCRIMINATIVE POWER ---")
    print("  power = max(AUROC, 1-AUROC): a feature can leak in either direction")
    print("  pct_finite < 100 means that feature's AUROC was computed on a SUBPOPULATION")
    print(tab.head(a.top).round(4).to_string())

    # --- 3. leak candidates -------------------------------------------------
    leaks = tab[tab["power"] >= LEAK_THRESHOLD]
    susp = tab[(tab["power"] >= SUSPICIOUS_THRESHOLD) & (tab["power"] < LEAK_THRESHOLD)]
    print("\n" + "-" * 86)
    print(f"LEAK CANDIDATES (standalone power >= {LEAK_THRESHOLD})")
    print("-" * 86)
    print(leaks.round(4).to_string() if len(leaks) else "  (none)")
    if len(susp):
        print(f"\nSUSPICIOUS ({SUSPICIOUS_THRESHOLD} <= power < {LEAK_THRESHOLD}):")
        print(susp.round(4).to_string())

    print("\n--- NAMED SUSPECTS ---")
    for c in SUSPECT:
        if c in tab.index:
            r = tab.loc[c]
            tag = ("  <-- LEAK" if r["power"] >= LEAK_THRESHOLD
                   else "  <-- suspicious" if r["power"] >= SUSPICIOUS_THRESHOLD else "")
            print(f"  {c:<38} power {r['power']:.4f}  auroc {r['auroc']:.4f}  "
                  f"nuniq {int(r['nunique']):>6}  finite {r['pct_finite']:.1f}%{tag}")
        elif c in dead:
            print(f"  {c:<38} DEAD (nunique={int(nuniq[c])}) -- present but never varied")
        else:
            print(f"  {c:<38} absent from X")

    # --- the decisive feature, reported whether live or dead ----------------
    print("\n" + "-" * 86)
    print(f"DECISIVE FEATURE: {DECISIVE}")
    print("-" * 86)
    if DECISIVE in X.columns:
        col = _numeric(X[DECISIVE])
        nu = int(nuniq[DECISIVE])
        print(f"  nunique  : {nu}")
        print(f"  min/max  : {np.nanmin(col):.6g} / {np.nanmax(col):.6g}")
        print(f"  n_finite : {int(np.isfinite(col).sum()):,} of {len(col):,}")
        power = float(tab.loc[DECISIVE, "power"]) if DECISIVE in tab.index else float("nan")
        if np.isfinite(power):
            print(f"  standalone power : {power:.4f}   (auroc {tab.loc[DECISIVE, 'auroc']:.4f})")
    else:
        nu, power = 0, float("nan")
        print("  ABSENT from X.")

    # --- 4. stratify the strongest features by variant representation -------
    if {"ref", "alt"} <= set(meta.columns):
        rep = pd.Series([representation(str(r), str(x)) for r, x in zip(meta["ref"], meta["alt"])])
    else:
        parts = meta["variant_id"].astype(str).str.split(":", expand=True)
        rep = pd.Series([representation(r, x) for r, x in zip(parts[3], parts[4])])

    to_stratify = list(dict.fromkeys(
        list(tab.index[:3]) + [c for c in SUSPECT if c in tab.index][:2]
    ))
    print("\n" + "-" * 86)
    print("TOP FEATURES, STRATIFIED BY VARIANT REPRESENTATION")
    print("-" * 86)
    for c in to_stratify:
        s = _numeric(X[c])
        if tab.loc[c, "auroc"] < 0.5:
            s = -s
        df = stratified_evaluate(y, np.nan_to_num(s, nan=np.nanmedian(s)), rep, n_boot=a.n_boot)
        print(f"\n  feature: {c}")
        print(df[["n", "pos_rate", "auroc", "auprc", "auprc_no_skill", "auprc_lift"]]
              .round(4).to_string())

    # --- verdict ------------------------------------------------------------
    print("\n" + "=" * 86)
    if DECISIVE not in X.columns:
        print(f"VERDICT: {DECISIVE} is absent from this split. Nothing to conclude about it.")
    elif nu <= 1:
        print("VERDICT (B): POST-FIX SIGNATURE -- the feature is DEAD on this split.")
        print(f"  `{DECISIVE}` has nunique={nu}: identically constant on every row.")
        print("  Not a leak here. But the split is gene-disjoint and the fix computes the")
        print("  count from TRAIN labels only, so the model trains on values in [0, N] and")
        print("  predicts with the feature pinned at 0 -- a TRAIN/TEST DISTRIBUTION SHIFT.")
        print("  If it learned 'high count => pathogenic', every unseen gene is pushed toward")
        print("  benign. The remedy is REMOVAL, or an out-of-fold train-only encoding.")
        print("  NOTE: Run 14 predates the 2026-06-13 fix; a dead feature here would be")
        print("  UNEXPECTED. Check --split train, and confirm which code produced these splits.")
    elif np.isfinite(power) and power >= LEAK_THRESHOLD:
        print("VERDICT (A): LIVE PRE-FIX LEAK.")
        print(f"  `{DECISIVE}` alone separates the classes at power {power:.4f} on the TEST split,")
        print(f"  and varies across {nu} distinct values. The split is gene-disjoint and")
        print("  (pre-INCIDENT_2026-06-13) this count was computed corpus-wide, so a held-out")
        print("  gene's value derives from its own held-out labels.")
        print()
        print("  Run 14's test AUROC of 0.9975 is not a measurement of variant pathogenicity.")
        print("  Every regime-v0 number (Runs 9-14) is void, INDEPENDENTLY of the coordinate")
        print("  bug. Re-baseline on a post-fix, coordinate-corrected cohort.")
    else:
        print("VERDICT (C): present and varying, but not separating on its own.")
        print(f"  `{DECISIVE}`: nunique={nu}, standalone power={power:.4f} (< {LEAK_THRESHOLD}).")
        print("  A pre-split leak can still contribute multivariately. Read alongside")
        print("  docs/incidents/INCIDENT_2026-07-08... sec 6.2.3 and scripts/run9_ablations.py.")

    if len(leaks):
        print(f"\n  OTHER LEAK CANDIDATES: {list(leaks.index)}")
        print("  Any curated disease register (HGMD / LOVD / ClinGen) in the matrix must be")
        print("  justified in METHODS or removed.")
    print()
    print("  Regardless of verdict: train and test share the coordinate corruption")
    print("  identically, so no metric computed on these splits can detect it. See")
    print("  docs/incidents/INCIDENT_2026-07-08_deletion-reviewstatus-loss.md sec 3.")
    print("=" * 86)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
