"""build_gnomad_ymt_af.py -- Monzia Moodie

Fetch gnomAD population allele frequencies for the cohort's chrY and chrMT variants -- the two
chromosomes that are absent from gnomad_v4_exomes.parquet (its build omitted the Y/MT genes) AND from
the 1000G high-coverage panel (no Y/MT in that release). Without this, ~6,300 cohort variants carry a
silent allele_freq of 0 from every source.

  Y  : gnomAD exome (genome fallback) AF via gene(gene_symbol){ variants } -- per Y gene_symbol.
  MT : gnomAD mtDNA homoplasmic AF computed as ac_hom/an via region(chrom:"M"){ mitochondrial_variants }
       in ONE call (~10,850 mtDNA variants total). af_hom is the ACMG-standard mtDNA population frequency.

Output: data/processed/gnomad_ymt_af.parquet (variant_id, allele_freq) -- identical schema to the main
gnomAD parquet -- plus an optional dedup-merge onto gnomad_v4_exomes.parquet so the existing --gnomad
join fills Y/MT with no connector change.

Author: Monzia Moodie
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

GNOMAD_API = "https://gnomad.broadinstitute.org/api"

_Y_QUERY = (
    "query($s:String!,$ds:DatasetId!){ gene(gene_symbol:$s, reference_genome:GRCh38){"
    " variants(dataset:$ds){ variant_id exome{af} genome{af} } } }"
)
_MT_REGION_QUERY = (
    "query($ds:DatasetId!){ region(chrom:\"M\", start:1, stop:16569, reference_genome:GRCh38){"
    " mitochondrial_variants(dataset:$ds){ variant_id an ac_hom ac_het } } }"
)


# ---------------------------------------------------------------- pure helpers (unit-tested)
def norm_key(gnomad_vid: str) -> str:
    """gnomAD 'Y-12904862-A-G' / 'M-3308-T-C' -> bare 'chrom:pos:ref:alt'.
    chrom: strip a leading 'chr', and map mitochondrial 'M' -> 'MT' to match the ClinVar cohort."""
    parts = str(gnomad_vid).split("-")
    if len(parts) < 4:
        return str(gnomad_vid)
    chrom = parts[0][3:] if parts[0].startswith("chr") else parts[0]
    if chrom == "M":
        chrom = "MT"
    pos, ref = parts[1], parts[2]
    alt = "-".join(parts[3:])  # alt should not contain '-', but be safe
    return f"{chrom}:{pos}:{ref}:{alt}"


def parse_y_af(payload: dict) -> dict:
    """gene.variants payload -> {bare_key: af}. af = exome.af if present else genome.af; skip null AF."""
    out: dict[str, float] = {}
    gene = (payload.get("data") or {}).get("gene") or {}
    for v in gene.get("variants") or []:
        vid = v.get("variant_id")
        if not vid:
            continue
        af = (v.get("exome") or {}).get("af")
        if af is None:
            af = (v.get("genome") or {}).get("af")
        if af is None:
            continue
        out[norm_key(vid)] = float(af)
    return out


def parse_mt_af(payload: dict) -> dict:
    """region.mitochondrial_variants payload -> {bare_key: af_hom}. af_hom = ac_hom/an (an>0)."""
    out: dict[str, float] = {}
    region = (payload.get("data") or {}).get("region") or {}
    for v in region.get("mitochondrial_variants") or []:
        vid = v.get("variant_id")
        an = v.get("an")
        ac_hom = v.get("ac_hom") or 0
        if not vid or not an:
            continue
        out[norm_key(vid)] = float(ac_hom) / float(an)
    return out


def build_ymt_frame(cohort_keys: set, y_af: dict, mt_af: dict) -> pd.DataFrame:
    """Rows (variant_id='gnomad:'+key, allele_freq) for keys present in BOTH gnomAD and the cohort.
    MT wins over Y on the impossible event of a key collision (disjoint chroms -> never happens)."""
    af_map: dict[str, float] = {}
    for k, v in y_af.items():
        if k in cohort_keys:
            af_map[k] = v
    for k, v in mt_af.items():
        if k in cohort_keys:
            af_map[k] = v
    rows = [{"variant_id": f"gnomad:{k}", "allele_freq": float(v)} for k, v in sorted(af_map.items())]
    return pd.DataFrame(rows, columns=["variant_id", "allele_freq"])


def merge_into_gnomad(ymt_df: pd.DataFrame, base_path: str, out_path: str) -> tuple:
    """Concat ymt rows onto the main gnomAD parquet, dedup on variant_id (ymt wins), write out_path."""
    base = pd.read_parquet(base_path)
    combined = pd.concat([base, ymt_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["variant_id"], keep="last").reset_index(drop=True)
    combined.to_parquet(out_path, index=False)
    return len(base), len(ymt_df), len(combined)


def cohort_ymt(clinvar_path: str) -> tuple:
    """-> (set of bare Y/MT keys, sorted list of distinct Y gene_symbols). Reads variant_id + gene_symbol."""
    cols = pd.read_parquet(clinvar_path, columns=["variant_id", "gene_symbol"])
    vid = cols["variant_id"].astype(str)
    toks = vid.str.split(":", expand=True)
    SRC = {"gnomad", "clinvar", "kg", "1kg", "1kgp"}
    off = 1 if str(toks.iloc[0, 0]) in SRC else 0
    chrom = toks[off].astype(str).str.replace(r"^chr", "", regex=True)
    key = chrom + ":" + toks[off + 1].astype(str) + ":" + toks[off + 2].astype(str) + ":" + toks[off + 3].astype(str)
    is_y = chrom == "Y"
    is_mt = chrom.isin(["MT", "M"])
    keys = set(key[is_y | is_mt])
    y_genes = sorted({g for g in cols.loc[is_y, "gene_symbol"].dropna().astype(str) if g and g.lower() != "nan"})
    return keys, y_genes


# ---------------------------------------------------------------- network (thin; lazy requests import)
def _post(session, query: str, variables: dict, timeout: int = 120) -> dict:
    r = session.post(GNOMAD_API, json={"query": query, "variables": variables}, timeout=timeout)
    j = r.json()
    if j.get("errors"):
        raise RuntimeError(f"gnomAD GraphQL errors: {j['errors']}")
    return j


def fetch_y_af(y_genes, dataset: str, pause: float = 0.5) -> dict:
    import requests
    s = requests.Session()
    out: dict[str, float] = {}
    for i, g in enumerate(y_genes, 1):
        try:
            out.update(parse_y_af(_post(s, _Y_QUERY, {"s": g, "ds": dataset})))
        except Exception as e:  # noqa: BLE001 -- loud, never silent
            print(f"  [WARN] Y gene {g}: {e}", file=sys.stderr)
        if pause:
            time.sleep(pause)
        if i % 5 == 0:
            print(f"  ...{i}/{len(y_genes)} Y genes, {len(out)} variants so far")
    return out


def fetch_mt_af(dataset: str) -> dict:
    import requests
    s = requests.Session()
    return parse_mt_af(_post(s, _MT_REGION_QUERY, {"ds": dataset}))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build gnomAD Y/MT allele-frequency parquet for the cohort")
    ap.add_argument("--clinvar", required=True, help="cohort parquet (variant_id + gene_symbol)")
    ap.add_argument("--gnomad", default="data/processed/gnomad_v4_exomes.parquet",
                    help="existing gnomAD parquet to merge onto (variant_id, allele_freq)")
    ap.add_argument("--out", default="data/processed/gnomad_ymt_af.parquet")
    ap.add_argument("--merged-out", default=None,
                    help="if set, write base+Y/MT dedup-merged parquet here (e.g. gnomad_v4_exomes_ymt.parquet)")
    ap.add_argument("--dataset", default="gnomad_r4")
    ap.add_argument("--pause", type=float, default=0.5, help="seconds between Y-gene calls (rate-limit courtesy)")
    a = ap.parse_args(argv)

    keys, y_genes = cohort_ymt(a.clinvar)
    n_y = sum(1 for k in keys if k.split(":", 1)[0] == "Y")
    n_mt = len(keys) - n_y
    print(f"[cohort] {len(keys)} Y/MT variants ({n_y} Y / {n_mt} MT); {len(y_genes)} distinct Y genes")

    print("[fetch] gnomAD mtDNA region (one call) ...")
    mt_af = fetch_mt_af(a.dataset)
    print(f"  mtDNA: {len(mt_af)} gnomAD variants returned")
    print(f"[fetch] gnomAD Y genes ({len(y_genes)}) ...")
    y_af = fetch_y_af(y_genes, a.dataset, a.pause)
    print(f"  Y: {len(y_af)} gnomAD variants returned")

    df = build_ymt_frame(keys, y_af, mt_af)
    cov_y = sum(1 for v in df["variant_id"] if v.startswith("gnomad:Y:"))
    cov_mt = len(df) - cov_y
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"[write] {a.out}: {len(df)} cohort Y/MT variants matched gnomAD ({cov_y} Y / {cov_mt} MT)")
    if len(df) == 0:
        print("[ERROR] zero matches -- check dataset id / chrom naming before merging", file=sys.stderr)
        return 1

    if a.merged_out:
        nb, ny, nc = merge_into_gnomad(df, a.gnomad, a.merged_out)
        print(f"[merge] {a.gnomad} ({nb}) + Y/MT ({ny}) -> {a.merged_out} ({nc} unique)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
