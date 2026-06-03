r"""
ablate_gnn.py - GNN measurement harness + GPU timing probe.

Runs the GNN on the real verify_sources splits for a small epoch budget and records
the metrics every GNN optimization must be judged on: per-epoch wall-clock, peak VRAM,
best val AUROC, and gnn_score std. Writes one JSON row per run so baseline-vs-variant
ablations are directly comparable.

Use it FIRST on the GPU with --epochs 2 as the timing probe (the one number not to
estimate), then re-run per variant as Tier-1/Tier-2 changes land.

Run from repo root:
    python scripts/ablate_gnn.py --tag baseline --epochs 2 --subsample 8000
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from genomic_variant_classifier.models.gnn import (
    StringDBGraph, build_pyg_dataset, train_gnn_pipeline, GNNScorer,
)

SPLITS = Path("outputs/verify_sources/splits")
OUT = Path("outputs/gnn_ablation")


def _assemble(subsample: int, seed: int) -> tuple[pd.DataFrame, list[str]]:
    X = pd.read_parquet(SPLITS / "X_train.parquet").reset_index(drop=True)
    meta = pd.read_parquet(SPLITS / "meta_train.parquet").reset_index(drop=True)
    feat = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    df = X.copy()
    df["gene_symbol"] = meta["gene_symbol"].values
    df["variant_id"] = (meta["variant_id"].values if "variant_id" in meta.columns
                        else [f"v{i}" for i in range(len(df))])
    if "acmg_label" in meta.columns:
        df["acmg_label"] = meta["acmg_label"].values
    else:
        df["acmg_label"] = pd.read_parquet(SPLITS / "y_train.parquet").iloc[:, 0].values
    df = df.dropna(subset=["gene_symbol", "acmg_label"])
    n = min(subsample, len(df))
    return df.sample(n=n, random_state=seed).reset_index(drop=True), feat


def summarize(tag: str, hist: list[dict], gnn_std: float, peak_vram_mb: float,
              wall_s: float, n_rows: int, device: str) -> dict:
    """Pure metric reducer (testable without a GPU)."""
    epochs = len(hist)
    return {
        "tag": tag,
        "utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "epochs": epochs,
        "rows": n_rows,
        "wall_s": round(wall_s, 1),
        "s_per_epoch": round(wall_s / max(epochs, 1), 1),
        "best_val_auc": round(max((h["val_auc"] for h in hist), default=float("nan")), 4),
        "final_train_loss": round(hist[-1]["train_loss"], 4) if hist else None,
        "gnn_score_std": round(gnn_std, 4),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "all_finite": all(np.isfinite(h["train_loss"]) and np.isfinite(h["val_auc"]) for h in hist),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--subsample", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df, feat = _assemble(a.subsample, a.seed)
    print(f"[{a.tag}] {len(df)} rows | {len(feat)} feats | device={device}")

    graph = StringDBGraph(
        combined_score_threshold=700,
        local_links_path=Path("data/external/string/9606.protein.links.detailed.v12.0.txt.gz"),
        local_info_path=Path("data/external/string/9606.protein.info.v12.0.txt.gz"),
    ).build()
    print(f"graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model, trainer, hist = train_gnn_pipeline(df, feat, graph=graph, epochs=a.epochs, test_split=0.2)
    wall = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() / 1024**2) if device == "cuda" else 0.0

    full = build_pyg_dataset(df, graph, feat)
    sc = GNNScorer.from_trainer(trainer, full, df).score_dataframe(df)

    row = summarize(a.tag, hist, float(sc.std()), peak, wall, len(df), device)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"ablation_{a.tag}.json"
    path.write_text(json.dumps(row, indent=2))
    print(json.dumps(row, indent=2))
    print(f"\nwrote {path.resolve()}")
    if not row["all_finite"]:
        print("WARNING: non-finite values present")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
