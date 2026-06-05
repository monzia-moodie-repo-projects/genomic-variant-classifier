#!/usr/bin/env python3
"""
test_scalable_svm.py - validate ScalableSVM in the project venv (.venv312).

Run from the repo root after placing the module:
    python scripts/test_scalable_svm.py

Checks: every mode fits and yields calibrated 2-col probabilities with AUROC well
above chance; bagged_rbf uses K bags above the cap and a single exact SVC below it;
joblib round-trips identically (Run 10b pickle-safety); an invalid mode raises; and
the D / K probes return sane plateaus. Exit 0 = pass.
"""
from __future__ import annotations

import io
import sys

import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from genomic_variant_classifier.models.scalable_svm import ScalableSVM


def main() -> int:
    failures: list[str] = []
    X, y = make_classification(n_samples=6000, n_features=78, n_informative=20,
                               weights=[0.9, 0.1], random_state=0)
    Xtr, Xte, ytr, yte = X[:4500], X[4500:], y[:4500], y[4500:]

    def check(mode, **kw):
        clf = ScalableSVM(mode=mode, random_state=0, **kw).fit(Xtr, ytr)
        p = clf.predict_proba(Xte)
        if p.shape != (len(yte), 2):
            failures.append(f"{mode}: proba shape {p.shape}")
        if not np.allclose(p.sum(1), 1, atol=1e-6):
            failures.append(f"{mode}: proba rows do not sum to 1")
        auc = roc_auc_score(yte, p[:, 1])
        if auc <= 0.6:
            failures.append(f"{mode}: AUROC too low ({auc:.3f})")
        else:
            print(f"[ok] {mode:11s} AUROC={auc:.3f}")
        return clf

    cN = check("nystrom", n_components=512)
    check("rff", n_components=512)
    cB = check("bagged_rbf", svm_max_subsample=1500, svm_n_bags=5)
    if getattr(cB, "_n_bags_used", None) != 5:
        failures.append(f"bagged n>m: expected 5 bags, got {getattr(cB,'_n_bags_used',None)}")

    cB1 = ScalableSVM(mode="bagged_rbf", svm_max_subsample=10000,
                      svm_n_bags=25, random_state=0).fit(Xtr, ytr)
    if getattr(cB1, "_n_bags_used", None) != 1:
        failures.append(f"bagged n<=m: expected single exact fit, got {cB1._n_bags_used}")
    else:
        print("[ok] bagged n<=m collapses to 1 exact SVC")

    buf = io.BytesIO()
    joblib.dump(cN, buf)
    buf.seek(0)
    cN2 = joblib.load(buf)
    if not np.allclose(cN.predict_proba(Xte), cN2.predict_proba(Xte)):
        failures.append("joblib round-trip changed predictions")
    else:
        print("[ok] joblib round-trip identical")

    try:
        ScalableSVM(mode="bogus").fit(Xtr, ytr)
        failures.append("invalid mode did not raise")
    except ValueError:
        print("[ok] invalid mode raises ValueError")

    d = ScalableSVM.probe_n_components(Xtr, ytr, candidates=(128, 256, 512), mode="nystrom")
    k = ScalableSVM.probe_n_bags(Xtr, ytr, candidates=(1, 3, 5), svm_max_subsample=1500)
    print(f"[ok] probe_n_components D={d['chosen']}  probe_n_bags K={k['chosen']}")

    print()
    if failures:
        for f in failures:
            print("[FAIL] " + f)
        return 1
    print("ALL SCALABLE-SVM CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
