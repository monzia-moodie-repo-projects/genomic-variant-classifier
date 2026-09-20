"""Re-measure the constraint extension on the REPAIRED cohort, with honest
gene-clustered uncertainty (ruling step 7 / section 9).

The earlier core-vs-constraint result (+0.0088 AUROC, +0.06-0.08 AUPRC) was
measured on the LEGACY cohort -- the one missing 97.3% of eligible deletions --
and carried NO uncertainty estimate. Design effects of 3.2x-19.2x have since
been measured on these data, so a delta of that size may be null.

Evaluation population: the VALIDATION partition. The test partition is recorded
in the exposure ledger as test_feedback; the ruling forbids repeating model
choices against an already-inspected test population.

The two tiers differ ONLY by loeuf and mis_z plus their explicit missingness
indicators. The tabular backend is REQUIRED, never silently substituted.
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
CONTINUOUS = ["af_raw", "ref_len", "alt_len", "consequence_severity"]
CORE_BINARY = ["af_is_absent"]
CONSTRAINT_BINARY = ["loeuf_is_missing", "mis_z_is_missing"]
CATEGORICAL = ["variant_type_category"]
CONSTRAINT = ["loeuf", "mis_z"]
VARIANT_TYPES = ["SNV", "insertion", "deletion", "substitution"]
TIERS = ("core", "core_plus_constraint")


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
    ref = df["ref"].astype(str)
    alt = df["alt"].astype(str)
    out["ref_len"] = ref.str.len()
    out["alt_len"] = alt.str.len()

    def vtype(r, a):
        if len(r) == 1 and len(a) == 1: return "SNV"
        if len(a) > len(r): return "insertion"
        if len(r) > len(a): return "deletion"
        return "substitution"

    out["variant_type_category"] = [vtype(r, a) for r, a in zip(ref, alt)]
    out["consequence_severity"] = df["consequence"].fillna("").map(
        lambda c: max((CONSEQUENCE_SEVERITY.get(t, 0) for t in str(c).split("&")), default=0))
    return out


def load_constraint(path, prefer="ENST"):
    cols = ["gene", "transcript", "mane_select", "lof.oe_ci.upper", "mis.z_score"]
    df = pd.read_csv(path, sep="\t", usecols=lambda c: c in cols, low_memory=False)
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Constraint source missing columns: {sorted(missing)}")
    c = df[df["mane_select"] == True].copy()
    t = c["transcript"].astype(str)
    c["namespace"] = "other"
    c.loc[t.str.startswith("ENST"), "namespace"] = "ENST"
    c.loc[t.str.startswith(("NM_", "NR_", "XM_", "XR_")), "namespace"] = "NCBI"
    c = c[c["namespace"].isin(["ENST", "NCBI"]) & c["gene"].notna()].copy()
    fallback = "NCBI" if prefer == "ENST" else "ENST"
    c["_rank"] = c["namespace"].map({prefer: 0, fallback: 1})
    c = c.sort_values(["gene", "_rank"]).drop_duplicates(subset=["gene"], keep="first")
    if c["gene"].duplicated().any():
        raise ValueError("Canonical constraint table not unique per gene")
    return c.rename(columns={"lof.oe_ci.upper": "loeuf", "mis.z_score": "mis_z"})[
        ["gene", "loeuf", "mis_z"]]


def make_preprocessor(tier):
    transformers = [
        ("continuous", StandardScaler(), CONTINUOUS),
        ("binary", "passthrough", CORE_BINARY),
        ("categorical", OneHotEncoder(categories=[VARIANT_TYPES], handle_unknown="ignore",
                                      sparse_output=False), CATEGORICAL),
    ]
    if tier == "core_plus_constraint":
        transformers.append(("constraint_missing", "passthrough", CONSTRAINT_BINARY))
        transformers.append(("constraint_value",
                             Pipeline([("impute", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())]), CONSTRAINT))
    elif tier != "core":
        raise ValueError(f"Unknown tier: {tier}")
    return ColumnTransformer(transformers, remainder="drop")


def metrics(y, p):
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "n": int(len(y)),
            "prevalence": float(np.mean(y))}


def run(membership_path, cohort_path, gnomad_path, src_root, output_dir,
        seeds=(0, 1, 2, 3, 4), n_boot=2000, random_state=42):
    import lightgbm as lgb

    sys.path.insert(0, str(Path(src_root)))
    from genomic_variant_classifier.evaluation.metrics import cluster_bootstrap_ci

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists -- refusing to overwrite")

    mem_all = pd.read_parquet(membership_path)
    mem = mem_all[mem_all["legacy_member"] | mem_all["corrected_member"]].reset_index(drop=True)
    if mem["label"].isna().any():
        raise ValueError("arm-eligible rows carry a missing label")
    print(f"universe {len(mem_all):,} | eligible {len(mem):,}")

    cohort = pd.read_parquet(cohort_path, columns=["variant_id", "ref", "alt",
                                                    "allele_freq", "consequence"])
    df = mem.merge(cohort, on="variant_id", how="left", validate="one_to_one")
    if df[["ref", "alt"]].isna().any().any():
        raise ValueError("membership rows missing allele data after merge")

    feats = build_features(df)
    constraint = load_constraint(gnomad_path)
    feats["gene_symbol"] = df["gene_symbol"].values
    feats = feats.merge(constraint, left_on="gene_symbol", right_on="gene",
                        how="left").drop(columns=["gene"])
    feats["loeuf_is_missing"] = feats["loeuf"].isna().astype(int)
    feats["mis_z_is_missing"] = feats["mis_z"].isna().astype(int)
    feats["label"] = df["label"].astype(int).values
    feats["partition"] = df["partition"].values
    feats["variant_id"] = df["variant_id"].values

    train = feats[feats["partition"].eq("train")]
    val = feats[feats["partition"].eq("validation")].reset_index(drop=True)
    print(f"train {len(train):,} rows / {train['gene_symbol'].nunique():,} genes "
          f"(prevalence {train['label'].mean():.6f})")
    print(f"VALIDATION {len(val):,} rows / {val['gene_symbol'].nunique():,} genes "
          f"(prevalence {val['label'].mean():.6f})")
    print("  test partition deliberately NOT used: recorded as test_feedback exposure")

    overlap = set(train["gene_symbol"]) & set(val["gene_symbol"])
    if overlap:
        raise ValueError(f"train/validation gene overlap: {len(overlap)}")
    print("  gene disjointness train/validation: overlap=0")

    preds = {}
    tier_reports = {}
    widths = []
    fitted = {}
    for tier in TIERS:
        pre = make_preprocessor(tier)
        X_tr = pre.fit_transform(train)
        X_va = pre.transform(val)
        if X_tr.shape[1] != X_va.shape[1]:
            raise ValueError(f"{tier}: train/val width mismatch {X_tr.shape[1]} vs {X_va.shape[1]}")
        widths.append((tier, int(X_tr.shape[1])))
        y_tr = train["label"].to_numpy()
        tier_reports[tier] = {"n_features": int(X_tr.shape[1]), "by_model": {}}
        print()
        print(f"=== tier {tier}: {X_tr.shape[1]} features ===")
        for name, model in (("logistic_regression",
                             LogisticRegression(max_iter=2000, random_state=random_state)),
                            ("lightgbm",
                             lgb.LGBMClassifier(random_state=random_state, verbose=-1))):
            model.fit(X_tr, y_tr)
            p = model.predict_proba(X_va)[:, 1]
            preds.setdefault(name, {})[tier] = p
            fitted[f"{tier}__{name}"] = {"preprocessor": pre, "model": model}
            m = metrics(val["label"].to_numpy(), p)
            tier_reports[tier]["by_model"][name] = m
            print(f"  {name}: AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} Brier={m['brier']:.5f}")

    print()
    print(f"=== paired tier contrast on VALIDATION, gene-cluster bootstrap (n_boot={n_boot}) ===")
    print("    negative delta = constraint tier has LOWER Brier loss")
    mean_fn = lambda yy, ss: float(np.mean(ss))
    y = val["label"].to_numpy().astype(float)
    genes = val["gene_symbol"].to_numpy()
    contrasts = {}
    for name in preds:
        d = (preds[name]["core_plus_constraint"] - y) ** 2 - (preds[name]["core"] - y) ** 2
        point = float(np.mean(d))
        per_seed = []
        for s in seeds:
            lo, hi, de = cluster_bootstrap_ci(mean_fn, y, d, genes, n_boot=n_boot,
                                               seed=s, return_design_effect=True)
            per_seed.append({"seed": int(s), "ci_low": float(lo), "ci_high": float(hi),
                             "design_effect": float(de),
                             "excludes_zero": bool((lo > 0) or (hi < 0))})
        verdicts = {r["excludes_zero"] for r in per_seed}
        contrasts[name] = {"delta_brier": point, "per_seed": per_seed,
                           "verdict_stable": len(verdicts) == 1}
        flag = "STABLE" if len(verdicts) == 1 else "*** UNSTABLE ***"
        print(f"  {name}: delta={point:+.6f} -> {flag}")
        for r in per_seed:
            print(f"      seed {r['seed']}: [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}] "
                  f"design_effect={r['design_effect']:.3f} excludes_zero={r['excludes_zero']}")

    if len({w for _, w in widths}) != 2:
        raise ValueError(f"tiers must differ in width: {widths}")

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "re-measure constraint extension on the REPAIRED cohort with "
                   "gene-clustered uncertainty; supersedes the legacy-cohort result",
        "evaluation_population": "validation partition (test excluded: test_feedback exposure)",
        "membership_path": str(Path(membership_path).resolve()),
        "membership_sha256": sha256_file(membership_path),
        "cohort_path": str(Path(cohort_path).resolve()),
        "cohort_sha256": sha256_file(cohort_path),
        "gnomad_path": str(Path(gnomad_path).resolve()),
        "gnomad_sha256": sha256_file(gnomad_path),
        "tabular_backend": "lightgbm",
        "feature_widths": dict(widths),
        "n_train": len(train), "n_train_genes": int(train["gene_symbol"].nunique()),
        "train_prevalence": float(train["label"].mean()),
        "n_validation": len(val), "n_validation_genes": int(val["gene_symbol"].nunique()),
        "validation_prevalence": float(val["label"].mean()),
        "tiers": tier_reports,
        "paired_contrasts": contrasts,
        "note": "AUPRC is comparable across tiers here because the population is the "
                "same; it is not comparable across populations of differing prevalence.",
    }
    (output_dir / "constraint_remeasure_report.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    # Persist EXECUTION-MATCHED feature and availability data alongside the
    # predictions. v1 wrote identity + predictions only, which forced any
    # downstream attribution to RECOMPUTE the constraint join and merely hope
    # it reproduced what the run used.
    persisted = ["variant_id", "gene_symbol", "label",
                 "loeuf", "mis_z", "loeuf_is_missing", "mis_z_is_missing",
                 *CONTINUOUS, *CORE_BINARY, *CATEGORICAL]
    missing_cols = [c for c in persisted if c not in val.columns]
    if missing_cols:
        raise ValueError(f"cannot persist execution-matched data; absent: {missing_cols}")
    out = val[persisted].copy()
    for name in preds:
        for tier in TIERS:
            out[f"{name}__{tier}"] = preds[name][tier]
    out.to_parquet(output_dir / "validation_predictions.parquet", index=False)
    import joblib
    joblib.dump(fitted, output_dir / "fitted_pipelines.joblib")
    print(f"  persisted {len(fitted)} fitted pipelines (preprocessor + model per tier/model)")
    print(f"  persisted {out.shape[1]} columns "
          f"(identity + execution-matched features/availability + predictions)")
    print()
    print(f"Wrote {output_dir}")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--membership", required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--gnomad-constraint", required=True)
    p.add_argument("--src-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()
    run(args.membership, args.cohort, args.gnomad_constraint, args.src_root,
        args.output_dir, tuple(args.seeds), args.n_boot)


if __name__ == "__main__":
    main()



