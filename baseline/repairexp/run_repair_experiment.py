"""The two-by-two repair experiment (governing ruling, section 5).

Train two models that differ ONLY in cohort eligibility policy:
  M_L  legacy    -- rows eligible under the VCF-join-derived ReviewStatus
  M_C  corrected -- rows eligible under the authenticated metadata review status
Both use the SAME frozen gene->partition registry, the SAME feature
definitions, the SAME model families, and the SAME metrics.

Score BOTH models on BOTH evaluation cells of the test partition:
  E_common  rows eligible under both policies
  E_added   rows eligible only under the corrected policy

The tabular backend is REQUIRED, never silently substituted.
"""
import argparse
import hashlib
import json
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

from migration import brier_factorial

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
BINARY = ["af_is_absent", "loeuf_is_missing", "mis_z_is_missing"]
CATEGORICAL = ["variant_type_category"]
CONSTRAINT = ["loeuf", "mis_z"]
# Frozen category vocabulary. Learning categories per-arm would give the two arms
# DIFFERENT feature dimensionality, because the legacy arm is nearly all SNVs --
# the excluded class IS the deletions.
VARIANT_TYPES = ["SNV", "insertion", "deletion", "substitution"]


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
    cons = df["consequence"].fillna("")
    out["consequence_severity"] = cons.map(
        lambda c: max((CONSEQUENCE_SEVERITY.get(t, 0) for t in str(c).split("&")), default=0))
    return out


def load_constraint(path, prefer="ENST"):
    cols = ["gene", "transcript", "mane_select", "lof.oe_ci.upper", "mis.z_score"]
    df = pd.read_csv(path, sep="\t", usecols=lambda c: c in cols, low_memory=False)
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Constraint source missing columns: {sorted(missing)}")
    c = df[df["mane_select"] == True].copy()
    c["namespace"] = "other"
    t = c["transcript"].astype(str)
    c.loc[t.str.startswith("ENST"), "namespace"] = "ENST"
    c.loc[t.str.startswith(("NM_", "NR_", "XM_", "XR_")), "namespace"] = "NCBI"
    c = c[c["namespace"].isin(["ENST", "NCBI"])]
    n_null = int(c["gene"].isna().sum())
    c = c[c["gene"].notna()].copy()
    fallback = "NCBI" if prefer == "ENST" else "ENST"
    c["_rank"] = c["namespace"].map({prefer: 0, fallback: 1})
    c = c.sort_values(["gene", "_rank"]).drop_duplicates(subset=["gene"], keep="first")
    if c["gene"].duplicated().any():
        raise ValueError("Canonical constraint table not unique per gene")
    return c.rename(columns={"lof.oe_ci.upper": "loeuf", "mis.z_score": "mis_z"})[
        ["gene", "loeuf", "mis_z"]], n_null


def make_preprocessor():
    return ColumnTransformer([
        ("continuous", StandardScaler(), CONTINUOUS),
        # Missingness indicators are explicit precomputed binary columns passed
        # through unscaled -- never add_indicator, whose output depends on which
        # columns happened to be missing in that arm.
        ("binary", "passthrough", BINARY),
        ("categorical", OneHotEncoder(categories=[VARIANT_TYPES], handle_unknown="ignore",
                                      sparse_output=False), CATEGORICAL),
        ("constraint_value", Pipeline([("impute", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())]), CONSTRAINT),
    ], remainder="drop")


def metrics(y, p):
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "n": int(len(y)),
            "prevalence": float(np.mean(y))}


def run(membership_path, cohort_path, gnomad_path, output_dir, random_state=42):
    import lightgbm as lgb   # REQUIRED backend; no silent substitution

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists -- refusing to overwrite")

    mem_all = pd.read_parquet(membership_path)
    n_universe = len(mem_all)
    # The membership table spans the WHOLE identity universe, including rows
    # eligible under neither policy (VUS, conflicting, sub-threshold review).
    # Those carry a missing label by construction and are never trained on or
    # evaluated. Restrict to arm-eligible rows, then assert the label is
    # complete -- rather than coercing NA into an integer.
    mem = mem_all[mem_all["legacy_member"] | mem_all["corrected_member"]].reset_index(drop=True)
    n_dropped = n_universe - len(mem)
    print(f"universe rows: {n_universe:,} | eligible under at least one arm: {len(mem):,} "
          f"| never-eligible dropped: {n_dropped:,}")
    if mem["label"].isna().any():
        raise ValueError(
            f"{int(mem['label'].isna().sum())} arm-eligible rows carry a missing label; "
            f"membership and label policy disagree")

    cohort = pd.read_parquet(cohort_path, columns=["variant_id", "ref", "alt",
                                                    "allele_freq", "consequence"])
    df = mem.merge(cohort, on="variant_id", how="left", validate="one_to_one")
    if df[["ref", "alt"]].isna().any().any():
        raise ValueError("Membership rows missing allele data after merge")

    feats = build_features(df)
    constraint, n_null_gene = load_constraint(gnomad_path)
    feats["gene_symbol"] = df["gene_symbol"].values
    feats = feats.merge(constraint, left_on="gene_symbol", right_on="gene", how="left").drop(columns=["gene"])
    feats["loeuf_is_missing"] = feats["loeuf"].isna().astype(int)
    feats["mis_z_is_missing"] = feats["mis_z"].isna().astype(int)
    feats["label"] = df["label"].astype(int).values
    feats["partition"] = df["partition"].values
    feats["legacy_member"] = df["legacy_member"].values
    feats["corrected_member"] = df["corrected_member"].values
    feats["variant_id"] = df["variant_id"].values

    test = feats["partition"].eq("test")
    cell = pd.Series("none", index=feats.index)
    cell[test & feats["legacy_member"] & feats["corrected_member"]] = "common"
    cell[test & ~feats["legacy_member"] & feats["corrected_member"]] = "added"
    feats["evaluation_cell"] = cell

    e_rows = feats[feats["evaluation_cell"].isin(["common", "added"])].reset_index(drop=True)
    print(f"evaluation rows: common={int((e_rows.evaluation_cell=='common').sum()):,} "
          f"added={int((e_rows.evaluation_cell=='added').sum()):,}")

    train_genes = set(feats.loc[feats["partition"].eq("train"), "gene_symbol"])
    eval_genes = set(e_rows["gene_symbol"])
    overlap = train_genes & eval_genes
    if overlap:
        raise ValueError(f"Evaluation genes overlap training genes: {len(overlap)}")
    print(f"gene disjointness: train={len(train_genes):,} eval={len(eval_genes):,} overlap=0")

    predictions = {}
    arm_reports = {}
    feature_widths = []
    for arm, member_col in (("legacy", "legacy_member"), ("corrected", "corrected_member")):
        tr = feats[feats["partition"].eq("train") & feats[member_col]]
        print()
        print(f"=== {arm}: fitting on {len(tr):,} rows, {tr['gene_symbol'].nunique():,} genes, "
              f"prevalence {tr['label'].mean():.6f} ===")
        pre = make_preprocessor()
        X_tr = pre.fit_transform(tr)
        y_tr = tr["label"].to_numpy()
        X_ev = pre.transform(e_rows)

        if X_tr.shape[1] != X_ev.shape[1]:
            raise ValueError(f"{arm}: train/eval feature width mismatch {X_tr.shape[1]} vs {X_ev.shape[1]}")
        feature_widths.append((arm, int(X_tr.shape[1])))
        arm_reports[arm] = {"n_train": len(tr), "n_train_genes": int(tr["gene_symbol"].nunique()),
                            "train_prevalence": float(tr["label"].mean()),
                            "n_features": int(X_tr.shape[1]), "by_model": {}}

        models = {
            "logistic_regression": LogisticRegression(max_iter=2000, random_state=random_state),
            "lightgbm": lgb.LGBMClassifier(random_state=random_state, verbose=-1),
        }
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            p = model.predict_proba(X_ev)[:, 1]
            predictions.setdefault(name, {})[arm] = p
            per_cell = {}
            for c in ("common", "added"):
                sel = (e_rows["evaluation_cell"] == c).to_numpy()
                per_cell[c] = metrics(e_rows.loc[sel, "label"].to_numpy(), p[sel])
            arm_reports[arm]["by_model"][name] = per_cell
            print(f"  {name}: common AUROC={per_cell['common']['auroc']:.4f} "
                  f"Brier={per_cell['common']['brier']:.5f} | "
                  f"added AUROC={per_cell['added']['auroc']:.4f} "
                  f"Brier={per_cell['added']['brier']:.5f}")

    widths = {w for _, w in feature_widths}
    if len(widths) != 1:
        raise ValueError(f"Arms have non-comparable feature spaces: {feature_widths}")
    print()
    print(f"feature space identical across arms: {feature_widths[0][1]} columns")

    print()
    print("=== paired factorial contrasts (Brier; negative = corrected better) ===")
    factorial = {}
    for name in predictions:
        rows = [{"variant_id": vid, "evaluation_cell": c, "label": int(y),
                 "legacy_probability": float(pl), "corrected_probability": float(pc)}
                for vid, c, y, pl, pc in zip(e_rows["variant_id"], e_rows["evaluation_cell"],
                                              e_rows["label"], predictions[name]["legacy"],
                                              predictions[name]["corrected"])]
        factorial[name] = brier_factorial(rows)
        f = factorial[name]
        print(f"  {name}: delta_common={f['delta_common']:+.6f} "
              f"delta_added={f['delta_added']:+.6f} interaction={f['interaction']:+.6f}")

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "two-by-two cohort-eligibility repair",
        "frozen": ["feature_definitions", "constraint_source_policy", "model_families",
                   "preprocessing_policy", "label_snapshot", "metrics"],
        "membership_path": str(Path(membership_path).resolve()),
        "membership_sha256": sha256_file(membership_path),
        "cohort_sha256": sha256_file(cohort_path),
        "gnomad_path": str(Path(gnomad_path).resolve()),
        "gnomad_sha256": sha256_file(gnomad_path),
        "constraint_null_gene_excluded": n_null_gene,
        "n_universe": n_universe,
        "n_arm_eligible": len(mem),
        "n_never_eligible_dropped": n_dropped,
        "tabular_backend": "lightgbm",
        "feature_width": feature_widths[0][1],
        "n_eval_common": int((e_rows.evaluation_cell == "common").sum()),
        "n_eval_added": int((e_rows.evaluation_cell == "added").sum()),
        "eval_prevalence_common": float(e_rows.loc[e_rows.evaluation_cell == "common", "label"].mean()),
        "eval_prevalence_added": float(e_rows.loc[e_rows.evaluation_cell == "added", "label"].mean()),
        "arms": arm_reports,
        "factorial": factorial,
    }
    (output_dir / "repair_experiment_report.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    pred_out = e_rows[["variant_id", "gene_symbol", "evaluation_cell", "label"]].copy()
    for name in predictions:
        for arm in ("legacy", "corrected"):
            pred_out[f"{name}__{arm}"] = predictions[name][arm]
    pred_out.to_parquet(output_dir / "evaluation_predictions.parquet", index=False)
    print()
    print(f"Wrote {output_dir}")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--membership", required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--gnomad-constraint", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    run(args.membership, args.cohort, args.gnomad_constraint, args.output_dir)


if __name__ == "__main__":
    main()

