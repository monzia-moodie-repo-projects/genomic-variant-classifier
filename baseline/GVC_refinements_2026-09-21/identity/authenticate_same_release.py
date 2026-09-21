"""Authenticate cohort classification and review status against the SAME-RELEASE VCF,
keyed by VariationID -- never by coordinates.

The original defect was the coordinate join, not the VCF content. Every cohort
VariationID is present in the March 2026 VCF, which carries CLNSIG and CLNREVSTAT for
the same release. Review statuses are compared through the project's own canonical
normaliser; clinical significance is compared after the VCF's underscore encoding is
decoded, and every disagreement is reported with examples rather than forced to agree.

Also measures cohort rows that SHARE a VariationID: whether they are distinct alleles
under one record (a classified set) or exact duplicates. Read-only; writes one JSON.
"""
import argparse, collections, gzip, importlib.util, json, re, sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

SIG_RE = re.compile(r"(?:^|;)CLNSIG=([^;]+)")
REV_RE = re.compile(r"(?:^|;)CLNREVSTAT=([^;]+)")


def load_normaliser(repo):
    path = Path(repo) / "src/genomic_variant_classifier/data/review_status.py"
    spec = importlib.util.spec_from_file_location("_rs_auth", path)
    m = importlib.util.module_from_spec(spec); sys.modules["_rs_auth"] = m; spec.loader.exec_module(m)
    return m.normalise


def nested_review(v):
    if isinstance(v, Mapping): return v.get("review_status")
    if isinstance(v, str): return json.loads(v).get("review_status")
    return None


def read_vcf(vcf):
    out = {}
    with gzip.open(vcf, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"): continue
            c = line.rstrip("\n").split("\t", 8)
            s = SIG_RE.search(c[7]); r = REV_RE.search(c[7])
            out[c[2]] = (s.group(1).replace("_", " ") if s else None,
                         r.group(1) if r else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vcf", required=True); ap.add_argument("--cohort", required=True)
    ap.add_argument("--repo", required=True); ap.add_argument("--output", required=True)
    a = ap.parse_args()
    if Path(a.output).exists(): sys.exit(f"ABORT: {a.output} exists; refusing to overwrite")
    import pandas as pd
    norm = load_normaliser(a.repo)
    vcf = read_vcf(a.vcf)
    df = pd.read_parquet(a.cohort, columns=["variant_id", "source_id", "chrom", "pos", "ref", "alt",
                                             "clinical_sig", "metadata"])
    df["source_id"] = df["source_id"].astype(str)
    sig_state, rev_state = collections.Counter(), collections.Counter()
    sig_pairs, rev_pairs = collections.Counter(), collections.Counter()
    for sid, csig, meta in zip(df["source_id"], df["clinical_sig"], df["metadata"]):
        if sid not in vcf:
            sig_state["cohort_id_absent_from_vcf"] += 1; rev_state["cohort_id_absent_from_vcf"] += 1; continue
        vsig, vrev = vcf[sid]
        c = (csig or "").strip()
        if vsig is None: sig_state["vcf_has_no_clnsig"] += 1
        elif vsig == c: sig_state["agree"] += 1
        else: sig_state["disagree"] += 1; sig_pairs[(c, vsig)] += 1
        cr = nested_review(meta)
        if vrev is None: rev_state["vcf_has_no_clnrevstat"] += 1
        elif norm(cr) == norm(vrev): rev_state["agree"] += 1
        else: rev_state["disagree"] += 1; rev_pairs[(str(cr), vrev)] += 1

    dup = df[df["source_id"].duplicated(keep=False)]
    per = dup.groupby("source_id").agg(rows=("variant_id", "size"),
                                       distinct_alleles=("variant_id", "nunique"),
                                       distinct_sig=("clinical_sig", "nunique"))
    out = {
        "cohort_rows": len(df), "vcf_records": len(vcf),
        "clinical_sig": dict(sig_state), "clinical_sig_disagreement_top": [
            {"cohort": k[0], "vcf": k[1], "rows": v} for k, v in sig_pairs.most_common(25)],
        "review_status": dict(rev_state), "review_status_disagreement_top": [
            {"cohort": k[0], "vcf": k[1], "rows": v} for k, v in rev_pairs.most_common(25)],
        "shared_variation_id": {
            "ids": int(len(per)), "rows": int(len(dup)),
            "rows_per_id": per["rows"].value_counts().sort_index().to_dict(),
            "ids_whose_rows_are_distinct_alleles": int((per["distinct_alleles"] == per["rows"]).sum()),
            "ids_with_exact_duplicate_rows": int((per["distinct_alleles"] < per["rows"]).sum()),
            "ids_whose_rows_disagree_on_clinical_sig": int((per["distinct_sig"] > 1).sum()),
            "examples": dup.sort_values(["source_id", "variant_id"]).head(12)[
                ["source_id", "variant_id", "clinical_sig"]].astype(str).to_dict("records"),
        },
        "run_utc": datetime.now(timezone.utc).isoformat(),
    }
    Path(a.output).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    for k in ("clinical_sig", "review_status"):
        tot = sum(out[k].values())
        print(f"{k}: {out[k]}  (states sum {tot:,} == rows {len(df):,}: {tot == len(df)})")
        for d in out[f"{k}_disagreement_top"][:10]:
            print(f"    {d['rows']:>8,}  cohort={d['cohort']!r}  vcf={d['vcf']!r}")
    s = out["shared_variation_id"]
    print(f"\nshared VariationID: {s['ids']:,} IDs over {s['rows']:,} rows | rows per ID {s['rows_per_id']}")
    print(f"  distinct alleles under one ID: {s['ids_whose_rows_are_distinct_alleles']:,} | "
          f"exact duplicates: {s['ids_with_exact_duplicate_rows']:,} | "
          f"rows disagree on clinical_sig: {s['ids_whose_rows_disagree_on_clinical_sig']:,}")
    for e in s["examples"]: print("   ", e)
    print(f"\nWrote {a.output}")


if __name__ == "__main__":
    main()
