"""Close two residuals left by authenticate_same_release.py, over ALL rows.

1. Every clinical-significance disagreement, not only the top 25 patterns stored:
   is it purely the secondary-term delimiter ("; " in the cohort, "|" in the VCF)?
   Any remainder is printed in full.
2. Every cohort row sharing a VariationID: is each pair one variant on chrX and chrY
   at the same position and alleles? Reports positions against GRCh38
   pseudoautosomal-region bounds SUPPLIED AS ARGUMENTS (not hard-coded belief), plus
   gene agreement and binary-label eligibility of the pairs.
Read-only; writes one JSON.
"""
import argparse, collections, gzip, json, re, sys
from datetime import datetime, timezone
from pathlib import Path

SIG_RE = re.compile(r"(?:^|;)CLNSIG=([^;]+)")
BINARY = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic",
          "Benign", "Likely benign", "Benign/Likely benign"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vcf", required=True); ap.add_argument("--cohort", required=True)
    for n in ("par1-x", "par1-y", "par2-x", "par2-y"):
        ap.add_argument(f"--{n}", required=True, help="start-end, 1-based inclusive")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    if Path(a.output).exists(): sys.exit(f"ABORT: {a.output} exists; refusing to overwrite")
    import pandas as pd
    def iv(spec):
        m = re.fullmatch(r"(\d+)-(\d+)", spec)
        if not m or int(m.group(1)) > int(m.group(2)): sys.exit(f"ABORT: malformed interval {spec!r}")
        return int(m.group(1)), int(m.group(2))
    p1x, p1y, p2x, p2y = iv(a.par1_x), iv(a.par1_y), iv(a.par2_x), iv(a.par2_y)
    if p1x[1] - p1x[0] != p1y[1] - p1y[0] or p2x[1] - p2x[0] != p2y[1] - p2y[0]:
        sys.exit("ABORT: X and Y lengths of a PAR differ; intervals are inconsistent")
    off1, off2 = p1x[0] - p1y[0], p2x[0] - p2y[0]   # X position minus Y position, per region

    df = pd.read_parquet(a.cohort, columns=["variant_id", "source_id", "chrom", "pos", "ref", "alt",
                                             "gene_symbol", "clinical_sig"])
    df["source_id"] = df["source_id"].astype(str)
    ids = set(df["source_id"])
    vsig = {}
    with gzip.open(a.vcf, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"): continue
            c = line.rstrip("\n").rstrip("\r").split("\t", 8)
            if c[2] in ids:
                s = SIG_RE.search(c[7]); vsig[c[2]] = s.group(1).replace("_", " ") if s else None

    kind, remainder = collections.Counter(), collections.Counter()
    for sid, cs in zip(df["source_id"], df["clinical_sig"]):
        v = vsig.get(sid); c = (cs or "").strip()
        if v is None or v == c: continue
        if c.replace("; ", "|") == v: kind["delimiter_only"] += 1
        else: kind["other"] += 1; remainder[(c, v)] += 1

    dup = df[df["source_id"].duplicated(keep=False)].copy()
    dup["chrom"] = dup["chrom"].astype(str)
    inside = lambda iv_, p: iv_[0] <= p <= iv_[1]
    checks = collections.Counter(); non_xy = []
    for sid, g in dup.groupby("source_id"):
        if sorted(g["chrom"]) != ["X", "Y"] or g[["ref", "alt"]].astype(str).drop_duplicates().shape[0] != 1:
            checks["NOT_xy_pair_with_same_alleles"] += 1
            if len(non_xy) < 20: non_xy.append(g[["source_id", "variant_id", "chrom", "pos"]].astype(str).to_dict("records"))
            continue
        x = g[g["chrom"] == "X"].iloc[0]; y = g[g["chrom"] == "Y"].iloc[0]
        px, py = int(x["pos"]), int(y["pos"])
        if inside(p1x, px) and inside(p1y, py) and px - py == off1: region = "par1_offset_consistent"
        elif inside(p2x, px) and inside(p2y, py) and px - py == off2: region = "par2_offset_consistent"
        else:
            region = "xy_pair_NOT_a_consistent_par_pair"
            if len(non_xy) < 20: non_xy.append(g[["source_id", "variant_id", "chrom", "pos"]].astype(str).to_dict("records"))
        checks[region] += 1
        checks["same_gene" if x["gene_symbol"] == y["gene_symbol"] else "gene_differs"] += 1
        if str(x["clinical_sig"]).strip() in BINARY: checks["binary_label_eligible_pairs"] += 1
    out = {"clinical_sig_disagreement_kinds": dict(kind),
           "clinical_sig_non_delimiter_disagreements": [
               {"cohort": k[0], "vcf": k[1], "rows": n} for k, n in remainder.most_common()],
           "par_intervals_supplied": {"par1_x": p1x, "par1_y": p1y, "par2_x": p2x, "par2_y": p2y, "x_minus_y_offsets": [off1, off2]}, "shared_id_checks": dict(checks),
           "shared_id_ids": int(dup["source_id"].nunique()), "non_xy_examples": non_xy,
           "run_utc": datetime.now(timezone.utc).isoformat()}
    Path(a.output).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("clinical_sig disagreement kinds:", dict(kind), "| total", sum(kind.values()))
    for r in out["clinical_sig_non_delimiter_disagreements"]:
        print(f"    {r['rows']:>6}  cohort={r['cohort']!r}  vcf={r['vcf']!r}")
    print(f"\nshared-ID groups: {out['shared_id_ids']:,}")
    for k, v in sorted(checks.items()): print(f"  {k}: {v:,}")
    for e in non_xy: print("   needs inspection:", e)
    print(f"\nWrote {a.output}")


if __name__ == "__main__":
    main()
