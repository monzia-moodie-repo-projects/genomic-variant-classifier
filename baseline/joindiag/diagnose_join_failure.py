"""Read-only diagnostic: explain WHY augment_reviewstatus.py's VCF join failed.
Hypothesis: variant_summary and VCF use different position conventions for
indels. Modifies nothing.
"""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def norm_chrom(c):
    c = str(c)
    return c[3:] if c.lower().startswith("chr") else c


def representation(ref, alt):
    if not isinstance(ref, str) or not isinstance(alt, str) or not ref or not alt:
        return "unresolved"
    if set(ref + alt) - set("ACGT") or ref == alt:
        return "unresolved"
    if len(ref) == len(alt) == 1:
        return "SNV"
    if len(alt) > len(ref):
        return "net_length_gain"
    if len(ref) > len(alt):
        return "net_length_loss"
    return "equal_length_replacement"


def build_vcf_keysets(vcf_path):
    exact = set()
    n_records = 0
    with gzip.open(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 8:
                continue
            info = dict(kv.split("=", 1) for kv in p[7].split(";") if "=" in kv)
            if not info.get("CLNREVSTAT"):
                continue
            n_records += 1
            chrom = norm_chrom(p[0])
            pos = p[1]
            ref = p[3]
            for a in p[4].split(","):
                exact.add(f"{chrom}:{pos}:{ref}:{a}")
    return exact, n_records


def candidate_keys(chrom, pos, ref, alt):
    pos = int(pos)
    out = {"as_written": f"{chrom}:{pos}:{ref}:{alt}"}
    out["pos_minus_1"] = f"{chrom}:{pos - 1}:{ref}:{alt}"
    out["pos_plus_1"] = f"{chrom}:{pos + 1}:{ref}:{alt}"
    return out


def diagnose(cohort_path, vcf_path, sample_size, output_path):
    print(f"Reading cohort: {cohort_path}")
    df = pd.read_parquet(cohort_path, columns=["variant_id", "chrom", "pos", "ref", "alt",
                                                "ReviewStatus", "clinical_sig"])
    print(f"  {len(df):,} rows")

    print(f"Parsing VCF: {vcf_path}")
    exact_keys, n_records = build_vcf_keysets(vcf_path)
    print(f"  {n_records:,} VCF records with CLNREVSTAT, {len(exact_keys):,} distinct keys")
    print()

    df["representation"] = [representation(r, a) for r, a in zip(df["ref"], df["alt"])]
    df["join_failed"] = df["ReviewStatus"].fillna("") == ""

    report = {
        "cohort_rows": len(df),
        "vcf_records_with_clnrevstat": n_records,
        "vcf_distinct_keys": len(exact_keys),
        "join_failure_by_representation": {},
        "candidate_key_match_rates": {},
    }

    print("=== Join failure rate by representation class ===")
    for rep, group in df.groupby("representation"):
        failed = int(group["join_failed"].sum())
        total = len(group)
        report["join_failure_by_representation"][rep] = {
            "total": total, "failed": failed,
            "failure_rate": round(failed / total, 6) if total else None,
        }
        print(f"  {rep}: {failed:,} / {total:,} failed ({100*failed/total:.2f}%)")
    print()

    print("=== Candidate key forms tested on FAILED rows, by representation ===")
    for rep in sorted(df["representation"].unique()):
        failed_rows = df[(df["representation"] == rep) & df["join_failed"]]
        if failed_rows.empty:
            continue
        sample = failed_rows.head(sample_size)
        counter = Counter()
        for _, row in sample.iterrows():
            keys = candidate_keys(norm_chrom(row["chrom"]), row["pos"], str(row["ref"]), str(row["alt"]))
            matched_any = False
            for form, key in keys.items():
                if key in exact_keys:
                    counter[form] += 1
                    matched_any = True
            if not matched_any:
                counter["no_candidate_matched"] += 1
        report["candidate_key_match_rates"][rep] = {
            "sampled": len(sample), "matches": dict(counter),
        }
        print(f"  {rep} (sampled {len(sample):,} failed rows):")
        for form, n in counter.most_common():
            print(f"    {form}: {n:,} ({100*n/len(sample):.2f}%)")
    print()

    if output_path:
        Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {output_path}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", required=True)
    p.add_argument("--vcf", required=True)
    p.add_argument("--sample-size", type=int, default=20000)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    diagnose(args.cohort, args.vcf, args.sample_size, args.output)


if __name__ == "__main__":
    main()
