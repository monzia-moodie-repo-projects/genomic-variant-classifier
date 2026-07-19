from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RUN = Path("outputs/rnaseq_pred_write_smoke")

state = joblib.load(RUN / "models" / "ensemble.joblib")
scaler = joblib.load(RUN / "scaler.joblib")
per = pd.read_csv(RUN / "per_model_metrics.csv", index_col=0)

names = list(state["oof_model_names_"])
weights = np.asarray(state["blend_weights_"], dtype=float)
weights = weights / weights.sum()
meta_learner = state["meta_learner"]

X_raw = pd.read_parquet(RUN / "splits" / "X_test.parquet")
X_scaled = scaler.transform(X_raw)
y = pd.read_parquet(RUN / "splits" / "y_test.parquet")["label"].astype(int).to_numpy()

print("Expected test AUROC from per_model_metrics.csv:")
print(per["auroc"].to_string())

def pos(pred):
    a = np.asarray(pred)
    return a[:, 1] if a.ndim == 2 else a

def load_model(name):
    p = RUN / "models" / "ensemble_models" / f"{name}.joblib"
    return joblib.load(p)

for mode, X in [("raw", X_raw), ("scaled", X_scaled)]:
    print(f"\n===== MODE: {mode} =====")
    base = []
    for name in names:
        model = load_model(name)
        try:
            s = pos(model.predict_proba(X))
        except Exception as e:
            print(name, "FAILED", type(e).__name__, str(e)[:120])
            s = np.full(len(y), np.nan)
        if np.isfinite(s).all():
            auc = roc_auc_score(y, s)
            expected = float(per.loc[name, "auroc"]) if name in per.index else np.nan
            print(f"{name:22s} auc={auc:.4f} expected={expected:.4f} delta={auc-expected:+.4f}")
        base.append(s)

    B = np.vstack(base).T
    if np.isfinite(B).all():
        blend = B @ weights
        print(f"weighted_blend auc={roc_auc_score(y, blend):.4f}")

        try:
            stack = pos(meta_learner.predict_proba(B))
            print(f"meta_learner_stack auc={roc_auc_score(y, stack):.4f}")
        except Exception as e:
            print("meta_learner_stack FAILED", type(e).__name__, str(e))
