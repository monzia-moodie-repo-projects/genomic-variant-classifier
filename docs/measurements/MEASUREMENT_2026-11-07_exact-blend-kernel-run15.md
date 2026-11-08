# MEASUREMENT 2026-11-07 -- the exact blend kernel on Run 15's real out-of-fold matrix

Author: Monzia Moodie. Package 1 (the prediction contract), set 1a. Owner ruling 2026-11-07 (decision ccffb8cf, revision 5):
the authoritative kernel is `exact_binary64_products_sum_rne_v1` -- the exact real sum of the stored binary64 products, rounded
once to binary64, nearest with ties to even; canonical +0; a result above one refused. This record measures it on real data.

## Input

Run 15's out-of-fold member matrix, `outputs/run15_rerun_report/full/oof_predictions.parquet` (80,807,001 bytes; 883,127 rows;
the 13 member columns in file order: random_forest, xgboost, lightgbm, svm, svm_bagged_rbf, logistic_regression,
gradient_boosting, catboost, tabular_nn, cnn_1d, kan, mc_dropout, deep_ensemble), and the 13 stored blend weights, recorded
bit-exactly as hexadecimal floats below (their exact sum minus one is 0.0). Development rows only; no labels are read and
nothing is refitted. The reference module is `numerical_score_reference.py` of the owner's review bundle, revision 5
(gvc_package1_boundary_review.zip, sha256 02dd91fe0b84befba50afabb5faba04c67ebf10def13a598f9cbca4b3026d0d6).

## Results

| comparison | rows bitwise different | ULP histogram | max absolute difference | call changes at 0.1/0.3/0.5/0.7/0.9 (>= and >) | order |
|---|---|---|---|---|---|
| set 1a's `combiner.exact_blend` vs the owner's reference | 0 | -- | 0 | -- | identical |
| independent exact method (fsum over error-free products) vs the reference | 0 | -- | 0 | -- | identical |
| historical `P @ w` vs exact | 356,050 | 0: 527,077; 1: 346,719; 2: 9,316; 3: 15 | 2.22e-16 | 0 and 0 | identical |
| Dot2 vs exact | 0 | 0: 883,127 | 0 | 0 and 0 | identical |

- Both the exact scores and the historical scores take 883,127 DISTINCT values, and their sorted orders agree at every position:
  one strict total order. Every rank-based metric -- the area under the receiver operating characteristic curve (AUROC), the
  area under the precision-recall curve (AUPRC) computed from the ranking, and the risk-coverage curve -- is therefore identical
  under the two kernels; a score-valued metric (the Brier score, for one) moves by at most 2.22e-16 per row.
- The 3-ULP maximum is an observation on this matrix, NOT a tolerance: a kernel change is a predictor-identity change.
- Dot2 agrees with the exact sum on this matrix and is still not exact in general (the owner's counterexamples, reproduced in
  tests/unit/test_prediction_contract.py): it stays a non-selectable candidate.
- The independent method is exact only when every nonzero product is at least 2^-968 (here the smallest is 7.86e-39, no row
  outside the domain), and CPython documents that `math.fsum` may double-round on builds with extended-precision addition; it
  is a cross-check, never the definition. The integer accumulator has no platform condition.
- Time on the development host (single runs, recorded below): the reference 8.47 s (measurement 1) and 9.70 s (measurement 2),
  set 1a's implementation 5.40 s (about 6.1 microseconds per row; a 1,000-variant request about 6 ms).
- Identity of the reconstructed matrix: the second measurement rebuilt the matrix from the parquet file and reproduced the
  first measurement's fingerprint exactly (356,050 rows; the same ULP histogram).

## Measurement 1 (2026-11-07 12:38 UTC): exact_kernel_run15.py, sha256 762bc5d0c1d54b02a512d33cbb38a48264754174c09c50b76d5e03e7ac0ef269

```python
"""Exact once-rounded stored-value dot product on Run 15's real 883,127 x 13 OOF member matrix. Author: Monzia Moodie.

Usage: python -I exact_kernel_run15.py REFINE_DIR OWNER_R5_DIR -> JSON.
Compares: the owner's integer reference (exact_binary64_products_sum_rne_v1), an INDEPENDENT second exact method
(math.fsum over error-free TwoProduct pairs, valid when every nonzero product >= 2**-968), the historical P @ w, and
Dot2. Development rows only; no labels are read; nothing is refitted.
"""
import json, math, pickle, sys, time
import numpy as np
R, ref = sys.argv[1], sys.argv[2]
sys.path.insert(0, ref)
from numerical_score_reference import exact_blend_reference, dot2_candidate, probability_ulp_distance  # noqa: E402

P = np.load(R + "/run15_oof_predictions_matrix.npy", allow_pickle=False)
with open(R + "/run15_combiners.pkl", "rb") as f:
    w = tuple(float(x) for x in pickle.load(f)["blend_weights_"])

SPLIT = 134217729.0
def two_product(a, b):
    p = a * b
    c = SPLIT * a; ah = c - (c - a); al = a - ah
    c = SPLIT * b; bh = c - (c - b); bl = b - bh
    return p, al * bl - (((p - ah * bh) - al * bh) - ah * bl)

def fsum_exact(P, w):
    W = np.asarray(w)
    hi, lo = two_product(P, W[None, :])
    prod = P * W[None, :]
    safe = ((prod == 0) & ((P == 0) | (W[None, :] == 0))) | (prod >= 2.0 ** -968)
    ok_rows = safe.all(axis=1)
    out = np.empty(len(P))
    pairs = np.concatenate([hi, lo], axis=1)
    for i in range(len(P)):
        out[i] = math.fsum(pairs[i].tolist()) if ok_rows[i] else np.nan
    return out, int((~ok_rows).sum())

res = {"rows": int(P.shape[0]), "members": int(P.shape[1]), "dtype": str(P.dtype),
       "weights_hex": [x.hex() for x in w], "weight_mass_fsum_minus_1": math.fsum(w) - 1.0}
t = time.perf_counter(); exact = exact_blend_reference(np.ascontiguousarray(P, dtype=np.float64), w)
res["integer_reference_seconds"] = time.perf_counter() - t
t = time.perf_counter(); fs, unsafe = fsum_exact(P.astype(np.float64), w)
res["fsum_independent_seconds"] = time.perf_counter() - t
res["fsum_rows_outside_its_exactness_domain"] = unsafe
safe = ~np.isnan(fs)
res["fsum_vs_integer_bitwise_different_rows"] = int((fs[safe].view(np.int64) != exact[safe].view(np.int64)).sum())
hist = P.astype(np.float64) @ np.asarray(w)
d2 = dot2_candidate(P.astype(np.float64), w)
for name, v in (("historical_matmul", hist), ("dot2", d2)):
    u = probability_ulp_distance(exact, np.where(v == 0, 0.0, v))
    edges = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    side = lambda x: np.searchsorted(edges, x, side="right")
    strict = lambda x: np.searchsorted(edges, x, side="left")
    res[name + "_vs_exact"] = {"rows_bitwise_different": int((u != 0).sum()), "max_ulp": int(u.max()),
        "ulp_histogram": {int(k): int(c) for k, c in zip(*np.unique(u, return_counts=True))},
        "max_abs_difference": float(np.max(np.abs(v - exact))),
        "changes_ge_rule_any_edge": int((side(v) != side(exact)).sum()),
        "changes_gt_rule_any_edge": int((strict(v) != strict(exact)).sum()),
        "ordering_pairs_changed_among_ties": None}
    # Rank agreement: positions where the sort order differs.
    oe, ov = np.argsort(exact, kind="stable"), np.argsort(v, kind="stable")
    res[name + "_vs_exact"]["sorted_positions_differing"] = int((oe != ov).sum())
res["exact_ties_count_rows_sharing_a_value"] = int(len(exact) - len(np.unique(exact)))
res["min_nonzero_product"] = float(np.min((P * np.asarray(w))[(P * np.asarray(w)) > 0]))
print(json.dumps(res, indent=1))
```

Result, sha256 19b44811f59138fe43c3e371c5833bc6334d8634cf423e27f092977db3f7ecc8:

```json
{
 "rows": 883127,
 "members": 13,
 "dtype": "float64",
 "weights_hex": [
  "0x1.37362f52544b0p-5",
  "0x1.d938a0872c50fp-2",
  "0x1.6d00046cbc25fp-5",
  "0x1.26809a0948619p-8",
  "0x1.53c18d8780659p-13",
  "0x1.533b8c0261533p-10",
  "0x1.4f5b51f9e666ep-3",
  "0x1.5eddbb1ecb7ccp-3",
  "0x1.90fd383f7c747p-5",
  "0x1.6600720ced48ep-8",
  "0x1.627a099f7607dp-21",
  "0x1.ca306a82bb816p-6",
  "0x1.058c04398064ap-5"
 ],
 "weight_mass_fsum_minus_1": 0.0,
 "integer_reference_seconds": 8.472986474000209,
 "fsum_independent_seconds": 1.0888945789997706,
 "fsum_rows_outside_its_exactness_domain": 0,
 "fsum_vs_integer_bitwise_different_rows": 0,
 "historical_matmul_vs_exact": {
  "rows_bitwise_different": 356050,
  "max_ulp": 3,
  "ulp_histogram": {
   "0": 527077,
   "1": 346719,
   "2": 9316,
   "3": 15
  },
  "max_abs_difference": 2.220446049250313e-16,
  "changes_ge_rule_any_edge": 0,
  "changes_gt_rule_any_edge": 0,
  "ordering_pairs_changed_among_ties": null,
  "sorted_positions_differing": 0
 },
 "dot2_vs_exact": {
  "rows_bitwise_different": 0,
  "max_ulp": 0,
  "ulp_histogram": {
   "0": 883127
  },
  "max_abs_difference": 0.0,
  "changes_ge_rule_any_edge": 0,
  "changes_gt_rule_any_edge": 0,
  "ordering_pairs_changed_among_ties": null,
  "sorted_positions_differing": 0
 },
 "exact_ties_count_rows_sharing_a_value": 0,
 "min_nonzero_product": 7.861777338818042e-39
}
```

## Measurement 2 (2026-11-07, set 1a's implementation): integrated_kernel_run15.py, sha256 742aeff16a88a6aed8526d0a1ea24071f5779831c261358052cc5677f68bff11

```python
"""The INTEGRATED kernel (combiner.exact_blend, set 1a) against the owner's reference on Run 15's real OOF matrix.
Author: Monzia Moodie. Usage: python integrated_kernel_run15.py OOF_PARQUET REFERENCE_DIR MEASUREMENT_JSON -> JSON.
The weights are the 13 stored blend weights recorded bit-exactly (hex) in the earlier measurement; the matrix is the
member columns of oof_predictions.parquet in file order. Identity of the reconstruction is checked by fingerprint: the
historical P @ w must differ from the exact sum on exactly the rows, and by exactly the ULP histogram, measured before.
No labels are read; nothing is refitted.
"""
import json, sys, time
import numpy as np, pyarrow.parquet as pq
oof, ref_dir, prior_path = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, ref_dir)
from numerical_score_reference import exact_blend_reference, probability_ulp_distance  # noqa: E402
from genomic_variant_classifier.models.combiner import exact_blend, BLEND_KERNEL  # noqa: E402
prior = json.load(open(prior_path))
t = pq.read_table(oof)
names = [c for c in t.column_names if c != "_train_row_idx"]
P = np.column_stack([t.column(c).to_numpy() for c in names]).astype(np.float64)
w = tuple(float.fromhex(h) for h in prior["weights_hex"])
res = {"oof_columns": names, "rows": int(P.shape[0]), "kernel": BLEND_KERNEL}
t0 = time.perf_counter(); ref = exact_blend_reference(np.ascontiguousarray(P), w); res["reference_seconds"] = time.perf_counter() - t0
t0 = time.perf_counter(); got = exact_blend(np.ascontiguousarray(P), w); res["integrated_seconds"] = time.perf_counter() - t0
res["integrated_vs_reference_bitwise_different_rows"] = int((np.asarray(got).view(np.int64) != ref.view(np.int64)).sum())
hist = P @ np.asarray(w)
u = probability_ulp_distance(ref, np.where(hist == 0, 0.0, hist))
fp = {"rows_bitwise_different": int((u != 0).sum()), "ulp_histogram": {str(int(k)): int(c) for k, c in zip(*np.unique(u, return_counts=True))}}
want = prior["historical_matmul_vs_exact"]
res["fingerprint"] = fp
res["fingerprint_matches_prior_measurement"] = (fp["rows_bitwise_different"] == want["rows_bitwise_different"]
                                                and fp["ulp_histogram"] == {str(k): v for k, v in want["ulp_histogram"].items()})
oe, oh = np.argsort(ref, kind="stable"), np.argsort(hist, kind="stable")
res["distinct_exact_values"] = int(len(np.unique(ref)))
res["distinct_historical_values"] = int(len(np.unique(hist)))
res["sorted_positions_differing_historical_vs_exact"] = int((oe != oh).sum())
print(json.dumps(res, indent=1))
```

Result, sha256 8fa678c8c5816f9339dec099d92ff23eb2352485a84deab109d39752a4a66bd5:

```json
{
 "oof_columns": [
  "random_forest",
  "xgboost",
  "lightgbm",
  "svm",
  "svm_bagged_rbf",
  "logistic_regression",
  "gradient_boosting",
  "catboost",
  "tabular_nn",
  "cnn_1d",
  "kan",
  "mc_dropout",
  "deep_ensemble"
 ],
 "rows": 883127,
 "kernel": "exact_binary64_products_sum_rne_v1",
 "reference_seconds": 9.703590206999706,
 "integrated_seconds": 5.40428325799985,
 "integrated_vs_reference_bitwise_different_rows": 0,
 "fingerprint": {
  "rows_bitwise_different": 356050,
  "ulp_histogram": {
   "0": 527077,
   "1": 346719,
   "2": 9316,
   "3": 15
  }
 },
 "fingerprint_matches_prior_measurement": true,
 "distinct_exact_values": 883127,
 "distinct_historical_values": 883127,
 "sorted_positions_differing_historical_vs_exact": 0
}
```
