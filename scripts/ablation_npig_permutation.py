"""
scripts/ablation_npig_permutation.py
======================================
Permutation ablation for the ``n_pathogenic_in_gene`` (npig) feature.

C3 hypothesis
-------------
Run 14/15 AUROC may be inflated by gene-prevalence memorisation via npig.
This script tests whether npig encodes genuine variant-level signal or is
merely a proxy for gene identity (i.e. the model has memorised which genes
have many ClinVar pathogenic entries).

Protocol
--------
1. Load the clean-cohort training splits produced by DataPrepPipeline.
2. Run N permutation rounds:
   a. Shuffle ``y_train`` labels (breaks gene–label association).
   b. Recompute ``n_pathogenic_in_gene`` and ``gene_has_known_disease``
      from the *shuffled* labels.  ← critical: see FINDING F-10.
   c. Replace both columns in X_train / X_test.
   d. Retrain a fast LightGBM proxy (CPU, no NN/KAN/GNN).
   e. Evaluate on the original (un-shuffled) y_test.
   f. Record test AUROC.
3. Compare observed AUROC (original npig, original labels) to the
   95th percentile of the permuted null distribution.
   - observed > p95 → npig encodes genuine signal beyond gene identity.
   - observed ≤ p95 → dominated by gene memorisation (C3 CONFIRMED →
     run npig-free ablation and document in session notes).

Standing Rules compliance
--------------------------
- Runs LOCALLY (LightGBM CPU, --skip-nn flag equivalent, no ensemble).
- Wall-clock budget gate: abort if any single round exceeds 600 s.
- Estimated runtime: ~2 min/round on 1.2M rows.  Default N=50 → ~100 min.
  Use --n-permutations 10 --max-train 50000 for a 10-min smoke test.
- Pre-flight: splits must exist at --splits-dir before running.

Usage
-----
    # Smoke test (~10 min local)
    python scripts/ablation_npig_permutation.py \\
        --splits-dir outputs/run15/full/splits \\
        --n-permutations 10 --max-train 50000 \\
        --output outputs/run15/ablation_npig_smoke

    # Full ablation (~100 min local)
    python scripts/ablation_npig_permutation.py \\
        --splits-dir outputs/run15/full/splits \\
        --n-permutations 50 \\
        --output outputs/run15/ablation_npig_full

Exit codes
----------
0  C3 result computed and saved (SIGNAL or MEMORISATION).
1  C3 result computed: MEMORISATION — document and plan npig-free ablation.
2  Pipeline error (missing files, budget exceeded, import failure).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ablation_npig")

# Hard wall-clock budget per permutation round.
_PER_ROUND_BUDGET_S: int = 600


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="n_pathogenic_in_gene permutation ablation (C3 hypothesis)"
    )
    p.add_argument(
        "--splits-dir", required=True,
        help="Directory with X_train/y_train/X_test/y_test/meta_train/meta_test "
             "parquets (output of DataPrepPipeline).",
    )
    p.add_argument("--n-permutations", type=int, default=50)
    p.add_argument("--output",         default="outputs/ablation_npig")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument(
        "--max-train", type=int, default=None,
        help="Subsample training rows for speed (e.g. 50000 for a smoke test).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _recompute_npig(
    X_train:    pd.DataFrame,
    y_shuffled: pd.Series,
    meta_train: pd.DataFrame,
    X_test:     pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replace n_pathogenic_in_gene and gene_has_known_disease in X_train and
    X_test using counts derived from *shuffled* labels.

    CRITICAL (FINDING F-10): npig MUST be recomputed from the SHUFFLED y_train,
    not from the original labels.  Using original labels in the ablation would
    test "what happens to AUROC when npig is correct but labels are random"
    rather than "what happens when npig encodes no real signal" — which is the
    opposite of what C3 needs to know.

    For X_test: test genes were never in train; their shuffled npig count is 0
    by construction (no variants from test genes appear in the shuffled train
    positives).  This is correct: an unseen gene has no historical signal.
    """
    if "gene_symbol" not in meta_train.columns:
        raise ValueError(
            "_recompute_npig: 'gene_symbol' missing from meta_train.  "
            "Ensure DataPrepPipeline persists meta_train.parquet (Patch 6b)."
        )

    # Count label=1 per gene in the SHUFFLED train set.
    tmp = meta_train[["gene_symbol"]].copy().reset_index(drop=True)
    tmp["label_shuffled"] = y_shuffled.values
    gene_counts = (
        tmp[tmp["label_shuffled"] == 1]
        .groupby("gene_symbol")
        .size()
        .rename("n_pathogenic_in_gene")
        .reset_index()
    )

    # Rebuild X_train with new npig.
    X_tr = X_train.copy()
    for col in ("n_pathogenic_in_gene", "gene_has_known_disease"):
        if col in X_tr.columns:
            X_tr = X_tr.drop(columns=[col])
    join_base = meta_train[["gene_symbol"]].copy().reset_index(drop=True)
    join_base = join_base.merge(gene_counts, on="gene_symbol", how="left")
    join_base["n_pathogenic_in_gene"] = (
        join_base["n_pathogenic_in_gene"].fillna(0).astype(int)
    )
    X_tr["n_pathogenic_in_gene"]  = join_base["n_pathogenic_in_gene"].values
    X_tr["gene_has_known_disease"] = (X_tr["n_pathogenic_in_gene"] > 0).astype(int)

    # X_test: all npig = 0 (test genes unseen in shuffled train).
    X_te = X_test.copy()
    for col in ("n_pathogenic_in_gene", "gene_has_known_disease"):
        if col in X_te.columns:
            X_te[col] = 0

    return X_tr, X_te


def _fit_eval_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
) -> float:
    """Fit LightGBM (CPU, 200 trees) and return test AUROC."""
    try:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise ImportError(
            "LightGBM required for npig permutation ablation.  "
            "Install: pip install lightgbm --break-system-packages"
        ) from exc

    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=63,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, proba))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    rng     = np.random.default_rng(args.seed)
    outdir  = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    splits  = Path(args.splits_dir)

    # Pre-flight: verify all required files.
    required = (
        "X_train.parquet", "y_train.parquet",
        "X_test.parquet",  "y_test.parquet",
        "meta_train.parquet", "meta_test.parquet",
    )
    for fname in required:
        if not (splits / fname).exists():
            logger.error("Missing required file: %s", splits / fname)
            logger.error(
                "Run DataPrepPipeline on clinvar_grch38_clean.parquet first."
            )
            return 2

    logger.info("Loading splits from %s ...", splits)
    X_train    = pd.read_parquet(splits / "X_train.parquet")
    y_train    = pd.read_parquet(splits / "y_train.parquet")["label"]
    X_test     = pd.read_parquet(splits / "X_test.parquet")
    y_test     = pd.read_parquet(splits / "y_test.parquet")["label"]
    meta_train = pd.read_parquet(splits / "meta_train.parquet")
    logger.info(
        "Loaded: train=%d  test=%d  features=%d",
        len(X_train), len(X_test), X_train.shape[1],
    )

    if "n_pathogenic_in_gene" not in X_train.columns:
        logger.error(
            "n_pathogenic_in_gene not in X_train.  "
            "Ensure DataPrepPipeline.run() called enrich_gene_counts()."
        )
        return 2

    # Optional subsample.
    if args.max_train and len(y_train) > args.max_train:
        idx = rng.choice(len(y_train), args.max_train, replace=False)
        idx.sort()
        X_train    = X_train.iloc[idx].reset_index(drop=True)
        y_train    = y_train.iloc[idx].reset_index(drop=True)
        meta_train = meta_train.iloc[idx].reset_index(drop=True)
        logger.info("Subsampled train to %d rows", args.max_train)

    # Observed AUROC (original npig, original labels).
    logger.info("Computing observed AUROC (original npig) ...")
    t0 = time.perf_counter()
    observed_auroc = _fit_eval_lgbm(X_train, y_train, X_test, y_test)
    logger.info(
        "Observed AUROC: %.4f  (%.1fs)", observed_auroc, time.perf_counter() - t0
    )

    # Permutation rounds.
    permuted_aurocs: list[float] = []
    for i in range(args.n_permutations):
        t_round = time.perf_counter()

        y_shuffled = pd.Series(
            rng.permutation(y_train.values), index=y_train.index
        )
        X_tr_p, X_te_p = _recompute_npig(
            X_train, y_shuffled, meta_train, X_test
        )
        auroc_p = _fit_eval_lgbm(X_tr_p, y_shuffled, X_te_p, y_test)
        elapsed = time.perf_counter() - t_round

        permuted_aurocs.append(float(auroc_p))
        logger.info(
            "  round %3d/%d  perm_auroc=%.4f  (%.1fs)",
            i + 1, args.n_permutations, auroc_p, elapsed,
        )

        if elapsed > _PER_ROUND_BUDGET_S:
            logger.error(
                "Per-round wall clock %.1fs exceeded budget %ds.  "
                "Use --max-train to subsample and rerun.",
                elapsed, _PER_ROUND_BUDGET_S,
            )
            return 2

    perm_arr = np.array(permuted_aurocs)
    p95      = float(np.percentile(perm_arr, 95))
    verdict  = "SIGNAL" if observed_auroc > p95 else "MEMORISATION"

    interpretation = (
        "n_pathogenic_in_gene encodes genuine variant-level signal beyond gene "
        "identity (observed AUROC > 95th percentile of null distribution)."
        if verdict == "SIGNAL"
        else
        "n_pathogenic_in_gene is dominated by gene-identity memorisation "
        "(observed AUROC ≤ 95th percentile of null distribution).  "
        "C3 hypothesis CONFIRMED — run npig-free ablation and document in "
        "session notes before publishing."
    )

    result = {
        "observed_auroc":        float(observed_auroc),
        "permuted_auroc_mean":   float(perm_arr.mean()),
        "permuted_auroc_std":    float(perm_arr.std()),
        "permuted_auroc_p95":    p95,
        "n_permutations":        args.n_permutations,
        "c3_verdict":            verdict,
        "interpretation":        interpretation,
        "splits_dir":            str(splits),
        "max_train":             args.max_train,
        "seed":                  args.seed,
    }

    (outdir / "npig_ablation_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    pd.DataFrame({"permuted_auroc": permuted_aurocs}).to_csv(
        outdir / "npig_ablation_rounds.csv", index=False
    )

    sep = "=" * 58
    print(f"\n{sep}")
    print("  n_pathogenic_in_gene Permutation Ablation (C3)")
    print(sep)
    print(f"  Observed AUROC  : {observed_auroc:.4f}")
    print(f"  Permuted mean   : {perm_arr.mean():.4f} ± {perm_arr.std():.4f}")
    print(f"  Permuted p95    : {p95:.4f}")
    print(f"  C3 verdict      : {verdict}")
    print(f"\n  {interpretation}\n{sep}\n")

    return 0 if verdict == "SIGNAL" else 1


if __name__ == "__main__":
    sys.exit(main())
