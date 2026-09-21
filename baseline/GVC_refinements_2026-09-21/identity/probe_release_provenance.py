"""Read-only probe: are the VCF and variant_summary the same ClinVar release, and
what do the GENEINFO anomalies mean?

File modification times are not release dates. This reads release evidence from the
files' own contents: the VCF ##fileDate header, and the variant_summary LastEvaluated
maximum (a LOWER BOUND on its release date, not the date itself). Writes one JSON.
"""
import argparse, collections, gzip, hashlib, json, re, sys
from datetime import datetime, timezone
from pathlib import Path

GI_RE = re.compile(r"(?:^|;)GENEINFO=([^;]+)")
VC_RE = re.compile(r"(?:^|;)CLNVC=([^;]+)")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def probe_vcf(vcf):
    meta, ids, vc_at10, vc_all, raw_malformed = [], set(), collections.Counter(), collections.Counter(), []
    at10_ids = []
    with gzip.open(vcf, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("##"):
                if any(line.startswith(k) for k in ("##fileDate", "##source", "##reference")):
                    meta.append(line.strip())
                continue
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t", 8)
            ids.add(c[2])
            vc = VC_RE.search(c[7]); vc = vc.group(1) if vc else "NONE"
            vc_all[vc] += 1
            m = GI_RE.search(c[7])
            if not m:
                continue
            entries = m.group(1).split("|")
            if any(e.count(":") != 1 for e in entries):
                raw_malformed.append({"id": c[2], "geneinfo": m.group(1)[:200]})
            if len(entries) == 10:
                vc_at10[vc] += 1; at10_ids.append(c[2])
    return meta, ids, vc_all, vc_at10, raw_malformed, at10_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--variant-summary", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    if Path(a.output).exists():
        sys.exit(f"ABORT: {a.output} exists; refusing to overwrite")
    import pandas as pd

    meta, vcf_ids, vc_all, vc_at10, malformed, at10_ids = probe_vcf(a.vcf)
    print("=== VCF release metadata (from file contents) ===")
    for m in meta: print("  ", m)

    with gzip.open(a.variant_summary, "rt", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
    print(f"\n=== variant_summary: {len(header)} columns ===\n  {header}")
    want = ["VariationID", "GeneID", "GeneSymbol", "HGNC_ID", "Assembly", "LastEvaluated", "Type",
            "PositionVCF", "ReferenceAlleleVCF", "AlternateAlleleVCF", "Start"]
    present = [c for c in want if c in header]
    absent = [c for c in want if c not in header]
    print("  requested-and-present:", present)
    print("  requested-and-ABSENT:", absent or "none")
    vs = pd.read_csv(a.variant_summary, sep="\t", usecols=present, dtype=str, low_memory=False)
    vs38 = vs[vs["Assembly"] == "GRCh38"] if "Assembly" in vs else vs
    le = pd.to_datetime(vs38.get("LastEvaluated"), errors="coerce", format="mixed") if "LastEvaluated" in vs38 else None

    cohort = pd.read_parquet(a.cohort, columns=["source_id"])
    cid = set(cohort["source_id"].astype(str))
    vsid = set(vs38["VariationID"].astype(str))

    at10 = vs38[vs38["VariationID"].isin(at10_ids)] if at10_ids else vs38.iloc[0:0]
    out = {
        "vcf_meta": meta, "vcf_sha256": sha256_file(a.vcf),
        "variant_summary_sha256": sha256_file(a.variant_summary),
        "variant_summary_columns": header, "absent_requested_columns": absent,
        "vs_grch38_rows": int(len(vs38)),
        "vs_grch38_distinct_variation_ids": int(vs38["VariationID"].nunique()),
        "vs_last_evaluated_max": str(le.max()) if le is not None else None,
        "vs_last_evaluated_unparseable": int(le.isna().sum()) if le is not None else None,
        "vcf_ids": len(vcf_ids), "cohort_source_ids": len(cid),
        "cohort_in_vcf": len(cid & vcf_ids), "cohort_in_vs": len(cid & vsid),
        "cohort_in_neither": len(cid - vcf_ids - vsid),
        "cohort_in_vs_not_vcf": len((cid & vsid) - vcf_ids),
        "cohort_in_vcf_not_vs": len((cid & vcf_ids) - vsid),
        "clnvc_all": dict(vc_all.most_common()), "clnvc_at_10_genes": dict(vc_at10.most_common()),
        "malformed_geneinfo_raw": malformed,
    }
    if "GeneID" in vs38:
        out["vs_geneid_value_counts_top"] = vs38["GeneID"].value_counts().head(8).to_dict()
        out["vs_multigene_symbol_rows"] = int(vs38["GeneSymbol"].fillna("").str.contains(";").sum())
        if len(at10):
            out["vs_at10_genesymbol_semicolon_count_distribution"] = (
                at10["GeneSymbol"].fillna("").str.count(";").add(1).value_counts().sort_index().to_dict())
            out["vs_at10_geneid_top"] = at10["GeneID"].value_counts().head(5).to_dict()
    if {"PositionVCF", "Start"} <= set(vs38.columns):
        both = vs38[["PositionVCF", "Start"]].dropna()
        out["vs_rows_positionvcf_ne_start"] = int((both["PositionVCF"] != both["Start"]).sum())
    out["run_utc"] = datetime.now(timezone.utc).isoformat()
    Path(a.output).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    print(f"\n=== release overlap ===")
    for k in ("vcf_ids", "vs_grch38_distinct_variation_ids", "cohort_source_ids", "cohort_in_vcf",
              "cohort_in_vs", "cohort_in_neither", "cohort_in_vs_not_vcf", "cohort_in_vcf_not_vs",
              "vs_last_evaluated_max"):
        v = out.get(k); print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    print(f"\n=== variant type among the {sum(vc_at10.values()):,} ten-gene records vs all ===")
    print("  at 10 genes:", out["clnvc_at_10_genes"])
    print("  all records (top 6):", dict(list(out["clnvc_all"].items())[:6]))
    for k in ("vs_at10_genesymbol_semicolon_count_distribution", "vs_at10_geneid_top",
              "vs_geneid_value_counts_top", "vs_multigene_symbol_rows", "vs_rows_positionvcf_ne_start"):
        if k in out: print(f"  {k}: {out[k]}")
    print("\n=== raw malformed GENEINFO ===")
    for r in malformed: print("  ", r)
    print(f"\nWrote {a.output}")


if __name__ == "__main__":
    main()
