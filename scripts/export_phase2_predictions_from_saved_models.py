from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


RUN_DIR = Path("outputs/rnaseq_pred_write_smoke")


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def positive_score(pred) -> np.ndarray:
    arr = np.asarray(pred)
    if arr.ndim == 2:
        if arr.shape[1] < 2:
            fail(f"predict_proba returned invalid 2D shape: {arr.shape}")
        arr = arr[:, 1]
    elif arr.ndim != 1:
        fail(f"predict_proba returned invalid shape: {arr.shape}")
    arr = arr.astype(float)
    if not np.isfinite(arr).all():
        fail("non-finite predictions")
    if not ((arr >= 0).all() and (arr <= 1).all()):
        fail("predictions outside [0,1]")
    return arr


def predict_one(model, x_scaled: np.ndarray) -> np.ndarray:
    return positive_score(model.predict_proba(x_scaled))


def load_split(split: str):
    split_dir = RUN_DIR / "splits"
    x = pd.read_parquet(split_dir / f"X_{split}.parquet")
    y = pd.read_parquet(split_dir / f"y_{split}.parquet")
    meta = pd.read_parquet(split_dir / f"meta_{split}.parquet")

    if y.shape[1] != 1:
        fail(f"{split}: expected one y column, got {y.columns.tolist()}")
    if len(x) != len(y) or len(x) != len(meta):
        fail(f"{split}: length mismatch X={len(x)} y={len(y)} meta={len(meta)}")
    if "variant_id" not in meta.columns or "gene_symbol" not in meta.columns:
        fail(f"{split}: meta missing variant_id or gene_symbol")

    return x, y.iloc[:, 0].astype(int), meta


def main() -> int:
    state_path = RUN_DIR / "models" / "ensemble.joblib"
    metrics_path = RUN_DIR / "metrics.json"

    for p in [state_path, metrics_path]:
        if not p.exists() or p.stat().st_size <= 0:
            fail(f"missing required artifact: {p}")

    state = joblib.load(state_path)
    metrics = json.loads(metrics_path.read_text())

    if not isinstance(state, dict):
        fail(f"expected ensemble state dict, got {type(state)}")

    names = list(state["oof_model_names_"])
    weights = np.asarray(state["blend_weights_"], dtype=float)
    saved_paths = dict(state["saved_model_paths"])

    if len(names) != len(weights):
        fail(f"model/weight mismatch: {len(names)} names vs {len(weights)} weights")
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        fail(f"invalid blend weights: {weights}")

    weights = weights / weights.sum()
    print("blend weights:", dict(zip(names, weights.round(6))))

    models = {}
    for name in names:
        path = Path(saved_paths.get(name, ""))
        if not path.is_absolute():
            path = RUN_DIR / "models" / path
        if not path.exists():
            path = RUN_DIR / "models" / "ensemble_models" / f"{name}.joblib"
        if not path.exists() or path.stat().st_size <= 0:
            fail(f"missing saved model for {name}: {path}")
        models[name] = joblib.load(path)

    expected_auc = {"test": metrics["auroc"], "val": metrics["val_auroc"]}

    for split in ["test", "val"]:
        x, y, meta = load_split(split)

        base_scores = []
        for name in names:
            score = predict_one(models[name], x)
            if len(score) != len(x):
                fail(f"{split}/{name}: prediction length mismatch")
            base_scores.append(score)

        base = np.vstack(base_scores).T
        y_score = base @ weights

        auc = roc_auc_score(y, y_score)
        print(f"{split}: reconstructed blended AUROC={auc:.4f}; metrics.json={expected_auc[split]:.4f}")

        if abs(auc - expected_auc[split]) > 0.005:
            fail(
                f"{split}: reconstructed AUROC differs from metrics.json too much: "
                f"{auc:.6f} vs {expected_auc[split]:.6f}"
            )

        out = pd.DataFrame(
            {
                "variant_id": meta["variant_id"].astype(str),
                "gene_symbol": meta["gene_symbol"].astype(str),
                "label": y.to_numpy(dtype=int),
                "y_score": y_score,
                "split": split,
            }
        )

        for i, name in enumerate(names):
            out[f"score_{name}"] = base[:, i]

        path = RUN_DIR / f"predictions_{split}.parquet"
        out.to_parquet(path, index=False)
        print("wrote", path, out.shape)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
