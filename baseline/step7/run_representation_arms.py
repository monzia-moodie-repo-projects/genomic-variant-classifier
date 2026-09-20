"""Step 7: is the logistic-regression / LightGBM gap representational or capacity?

Four arms over the SAME admissible information (core tier only), differing only
in how that information is encoded for the linear model:

  A lr_current        current encoding: scaled continuous, one-hot variant type
  B lr_representation log1p lengths, log10 allele frequency, consequence as a
                      CATEGORICAL rather than a numeric severity rank
  C lr_splines        restricted cubic splines on the continuous terms
  D lightgbm          the same admissible information, gradient boosting

Constraint features are EXCLUDED. The constraint re-measurement showed they
degrade LightGBM probability quality on unseen genes (+0.006159 Brier, interval
excluding zero in 5/5 seeds), so including them would confound the representation
question with a term that harms the strongest arm.

Comparable tuning effort is achieved by giving every arm NONE: all library
defaults, no search. Equal by construction rather than by judgement.

Evaluation: the VALIDATION partition. The test partition is recorded in the
exposure ledger as test_feedback.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

CONSEQUENCE_SEVERITY = {
    "transcript_ablation": 10, "splice_acceptor_variant": 9, "splice_donor_variant": 9,
    "stop_gained": 9, "frameshift_variant": 8, "stop_lost": 8, "start_lost": 8,
    "transcript_amplification": 7, "inframe_insertion": 6, "inframe_deletion": 6,
    "missense_variant": 5, "protein_altering_variant": 5, "splice_region_variant": 4,
    "incomplete_terminal_codon_variant": 3, "start_retained_variant": 3,
    "stop_retained_variant": 3, "synonymous_variant": 2, "coding_sequence_variant": 2,
    "5_prime_UTR_variant": 2, "3_prime_UTR_variant": 2,
    "non_coding_transcript_exon_variant": 1, "intron_variant": 1,
    "NMD_transcript_variant": 1, "upstream_gene_variant": 0,
    "downstream_gene_variant": 0, "intergenic_variant": 0,
}
VARIANT_TYPES = ["SNV", "insertion", "deletion", "substitution"]
SEVERITY_LEVELS = [float(v) for v in range(0, 11)]
CONTINUOUS = ["af_raw", "ref_len", "alt_len", "consequence_severity"]
BINARY = ["af_is_absent"]
CATEGORICAL = ["variant_type_category"]
LOG_CONTINUOUS = ["af_log10", "ref_len_log1p", "alt_len_log1p"]
ARMS = ("lr_current", "lr_representation", "lr_splines", "lightgbm")
REFERENCE_ARM = "lr_current"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_features(df):
    out = pd.DataFrame(index=df.index)
    af = pd.to_numeric(df["allele_freq"], errors="coerce")
    out["af_is_absent"] = af.isna().astype(int)
    out["af_raw"] = af.fillna(0.0)
    out["af_log10"] = np.log10(out["af_raw"].clip(lower=0.0) + 1e-9)
    ref = df["ref"].astype(str)
    alt = df["alt"].astype(str)
    out["ref_len"] = ref.str.len()
    out["alt_len"] = alt.str.len()
    out["ref_len_log1p"] = np.log1p(out["ref_len"])
    out["alt_len_log1p"] = np.log1p(out["alt_len"])

    def vtype(r, a):
        if len(r) == 1 and len(a) == 1: return "SNV"
        if len(a) > len(r): return "insertion"
        if len(r) > len(a): return "deletion"
        return "substitution"

    out["variant_type_category"] = [vtype(r, a) for r, a in zip(ref, alt)]
    out["consequence_severity"] = df["consequence"].fillna("").map(
        lambda c: max((CONSEQUENCE_SEVERITY.get(t, 0) for t in str(c).split("&")),
                      default=0)).astype(float)
    return out


def make_preprocessor(arm):
    ohe = lambda cats: OneHotEncoder(categories=[cats], handle_unknown="ignore",
                                     sparse_output=False)
    if arm in ("lr_current", "lightgbm"):
        return ColumnTransformer([
            ("continuous", StandardScaler(), CONTINUOUS),
            ("binary", "passthrough", BINARY),
            ("variant_type", ohe(VARIANT_TYPES), CATEGORICAL),
        ], remainder="drop")
    if arm == "lr_representation":
        return ColumnTransformer([
            ("log_continuous", StandardScaler(), LOG_CONTINUOUS),
            ("binary", "passthrough", BINARY),
            ("variant_type", ohe(VARIANT_TYPES), CATEGORICAL),
            ("severity", ohe(SEVERITY_LEVELS), ["consequence_severity"]),
        ], remainder="drop")
    if arm == "lr_splines":
        return ColumnTransformer([
            ("splines", SplineTransformer(n_knots=5, degree=3, include_bias=False),
             CONTINUOUS),
            ("binary", "passthrough", BINARY),
            ("variant_type", ohe(VARIANT_TYPES), CATEGORICAL),
        ], remainder="drop")
    raise ValueError(f"Unknown arm: {arm}")


def make_model(arm, random_state):
    if arm == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(random_state=random_state, verbose=-1)
    return LogisticRegression(max_iter=2000, random_state=random_state)


def metrics(y, p):
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p))}


def run(membership_path, cohort_path, src_root, output_dir,
        seeds=(0, 1, 2, 3, 4), n_boot=2000, random_state=42):
    import lightgbm

    sys.path.insert(0, str(Path(src_root)))
    from genomic_variant_classifier.evaluation.metrics import cluster_bootstrap_ci

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists -- refusing to overwrite")

    mem_all = pd.read_parquet(membership_path)
    mem = mem_all[mem_all["legacy_member"] | mem_all["corrected_member"]].reset_index(drop=True)
    if mem["label"].isna().any():
        raise ValueError("arm-eligible rows carry a missing label")
    cohort = pd.read_parquet(cohort_path, columns=["variant_id", "ref", "alt",
                                                    "allele_freq", "consequence"])
    df = mem.merge(cohort, on="variant_id", how="left", validate="one_to_one")
    if df[["ref", "alt"]].isna().any().any():
        raise ValueError("membership rows missing allele data after merge")

    feats = build_features(df)
    feats["gene_symbol"] = df["gene_symbol"].values
    feats["label"] = df["label"].astype(int).values
    feats["partition"] = df["partition"].values
    feats["variant_id"] = df["variant_id"].values

    train = feats[feats["partition"].eq("train")]
    val = feats[feats["partition"].eq("validation")].reset_index(drop=True)
    print(f"train {len(train):,} rows / {train['gene_symbol'].nunique():,} genes "
          f"(prevalence {train['label'].mean():.6f})")
    print(f"VALIDATION {len(val):,} rows / {val['gene_symbol'].nunique():,} genes "
          f"(prevalence {val['label'].mean():.6f})")
    overlap = set(train["gene_symbol"]) & set(val["gene_symbol"])
    if overlap:
        raise ValueError(f"train/validation gene overlap: {len(overlap)}")
    print("  gene disjointness train/validation: overlap=0")
    print("  constraint features EXCLUDED (they degrade the strongest arm)")

    y_tr = train["label"].to_numpy()
    y_va = val["label"].to_numpy()
    preds, arm_reports = {}, {}
    print()
    for arm in ARMS:
        pre = make_preprocessor(arm)
        X_tr = pre.fit_transform(train)
        X_va = pre.transform(val)
        if X_tr.shape[1] != X_va.shape[1]:
            raise ValueError(f"{arm}: train/val width mismatch")
        model = make_model(arm, random_state)
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_va)[:, 1]
        preds[arm] = p
        m = metrics(y_va, p)
        arm_reports[arm] = {"n_features": int(X_tr.shape[1]), **m}
        print(f"{arm}: features={X_tr.shape[1]:>3}  AUROC={m['auroc']:.4f}  "
              f"AUPRC={m['auprc']:.4f}  Brier={m['brier']:.5f}")

    print()
    print(f"=== paired Brier contrast vs {REFERENCE_ARM} (negative = arm is BETTER) ===")
    mean_fn = lambda yy, ss: float(np.mean(ss))
    genes = val["gene_symbol"].to_numpy()
    yf = y_va.astype(float)
    contrasts = {}
    for arm in ARMS:
        if arm == REFERENCE_ARM:
            continue
        d = (preds[arm] - yf) ** 2 - (preds[REFERENCE_ARM] - yf) ** 2
        point = float(np.mean(d))
        per_seed = []
        for s in seeds:
            lo, hi, de = cluster_bootstrap_ci(mean_fn, yf, d, genes, n_boot=n_boot,
                                               seed=s, return_design_effect=True)
            per_seed.append({"seed": int(s), "ci_low": float(lo), "ci_high": float(hi),
                             "design_effect": float(de),
                             "excludes_zero": bool((lo > 0) or (hi < 0))})
        verdicts = {r["excludes_zero"] for r in per_seed}
        contrasts[arm] = {"delta_brier_vs_reference": point, "per_seed": per_seed,
                          "verdict_stable": len(verdicts) == 1}
        flag = "STABLE" if len(verdicts) == 1 else "*** UNSTABLE ***"
        print(f"  {arm}: delta={point:+.6f} -> {flag}")
        for r in per_seed:
            print(f"      seed {r['seed']}: [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] "
                  f"design_effect={r['design_effect']:.3f} excludes_zero={r['excludes_zero']}")

    gap = arm_reports[REFERENCE_ARM]["brier"] - arm_reports["lightgbm"]["brier"]
    best_lr = min((a for a in ARMS if a.startswith("lr_")),
                  key=lambda a: arm_reports[a]["brier"])
    closed = arm_reports[REFERENCE_ARM]["brier"] - arm_reports[best_lr]["brier"]
    fraction = (closed / gap) if gap != 0 else float("nan")
    print()
    print(f"reference Brier {arm_reports[REFERENCE_ARM]['brier']:.5f} | "
          f"lightgbm {arm_reports['lightgbm']['brier']:.5f} | gap {gap:.5f}")
    print(f"best linear arm: {best_lr} (Brier {arm_reports[best_lr]['brier']:.5f}), "
          f"closing {100*fraction:.1f}% of the gap")

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "question": "is the logistic-regression / LightGBM gap representational or capacity?",
        "constraint_features": "EXCLUDED -- degrade the strongest arm on unseen genes",
        "tuning": "none for any arm; comparable effort by construction",
        "evaluation_population": "validation partition (test excluded: test_feedback exposure)",
        "membership_sha256": sha256_file(membership_path),
        "cohort_sha256": sha256_file(cohort_path),
        "n_train": len(train), "n_validation": len(val),
        "n_validation_genes": int(val["gene_symbol"].nunique()),
        "validation_prevalence": float(y_va.mean()),
        "arms": arm_reports,
        "reference_arm": REFERENCE_ARM,
        "paired_contrasts": contrasts,
        "gap_analysis": {"reference_brier": arm_reports[REFERENCE_ARM]["brier"],
                         "lightgbm_brier": arm_reports["lightgbm"]["brier"],
                         "gap": gap, "best_linear_arm": best_lr,
                         "fraction_of_gap_closed": fraction},
        "caveat": "AUROC and AUPRC per arm carry NO uncertainty estimate. Only the "
                  "paired Brier contrasts have intervals.",
    }
    (output_dir / "representation_arms_report.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    out = val[["variant_id", "gene_symbol", "label"]].copy()
    for arm in ARMS:
        out[arm] = preds[arm]
    out.to_parquet(output_dir / "validation_predictions.parquet", index=False)
    print()
    print(f"Wrote {output_dir}")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--membership", required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--src-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()
    run(args.membership, args.cohort, args.src_root, args.output_dir,
        tuple(args.seeds), args.n_boot)


if __name__ == "__main__":
    main()
