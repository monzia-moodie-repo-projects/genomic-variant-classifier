"""build_gnomad_ymt_af.py -- Monzia Moodie

Fetch gnomAD population allele frequencies for the cohort's chrY and chrMT variants -- absent from both
gnomad_v4_exomes.parquet (its gene list omitted Y/MT) and the 1000G high-coverage panel. Without this,
~6,300 cohort variants carry a silent allele_freq=0 from every source.

  Y  : gene(gene_symbol){ variants } -> exome.af (genome fallback). The cohort gene_symbol is dirty for Y
       (semicolon-joined multi-gene strings + free-text), so symbols are CLEANED (split on ;/, drop
       free-text) and queried in ALIASED BATCHES with retry+backoff -- gnomAD rate-limits rapid serial
       calls and returns non-JSON, which must be retried, never silently skipped.
  MT : region(chrom:"M"){ mitochondrial_variants } in ONE call -> af_hom = ac_hom/an (the ACMG-standard
       mtDNA population frequency; af_hom/af_het are NOT API fields, so computed from counts).

Output data/processed/gnomad_ymt_af.parquet (variant_id, allele_freq) + optional dedup-merge onto
gnomad_v4_exomes.parquet so the existing --gnomad join fills Y/MT with no connector change.

Author: Monzia Moodie
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

GNOMAD_API = "https://gnomad.broadinstitute.org/api"

_MT_REGION_QUERY = (
    "query($ds:DatasetId!){ region(chrom:\"M\", start:1, stop:16569, reference_genome:GRCh38){"
    " mitochondrial_variants(dataset:$ds){ variant_id an ac_hom ac_het } } }"
)
_VALID_SYM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


# ---------------------------------------------------------------- pure helpers (unit-tested)
def norm_key(gnomad_vid: str) -> str:
    """gnomAD 'Y-12904862-A-G' / 'M-3308-T-C' -> bare 'chrom:pos:ref:alt' (strip 'chr'; map 'M' -> 'MT')."""
    parts = str(gnomad_vid).split("-")
    if len(parts) < 4:
        return str(gnomad_vid)
    chrom = parts[0][3:] if parts[0].startswith("chr") else parts[0]
    if chrom == "M":
        chrom = "MT"
    return f"{chrom}:{parts[1]}:{parts[2]}:{'-'.join(parts[3:])}"


def clean_y_genes(raw_symbols) -> list:
    """Cohort Y gene_symbol is dirty (';'-joined multi-gene + free-text). Split on ;/, keep only valid
    single symbols (no spaces; alnum/._-), drop '-'/'nan'/free-text. -> sorted unique gene symbols."""
    genes: set = set()
    for gs in raw_symbols:
        gs = str(gs).strip()
        if not gs or gs == "-" or gs.lower() == "nan":
            continue
        for tok in re.split(r"[;,]", gs):
            tok = tok.strip()
            if tok and " " not in tok and _VALID_SYM.match(tok):
                genes.add(tok)
    return sorted(genes)


def build_y_batch_query(genes) -> str:
    """Aliased multi-gene query (one HTTP call per ~8 genes -> avoids rate-limiting). Symbols are embedded
    inline (GraphQL aliases can't be variables); callers MUST pass clean_y_genes output (validated safe)."""
    parts = [
        f'a{i}: gene(gene_symbol: "{g}", reference_genome: GRCh38) '
        f"{{ variants(dataset: $ds) {{ variant_id exome {{ af }} genome {{ af }} }} }}"
        for i, g in enumerate(genes)
    ]
    return "query($ds: DatasetId!) { " + " ".join(parts) + " }"


def _af_from_variant(v) -> float:
    af = (v.get("exome") or {}).get("af")
    if af is None:
        af = (v.get("genome") or {}).get("af")
    return None if af is None else float(af)


def parse_y_batch(payload: dict) -> dict:
    """Aliased gene-batch payload -> {bare_key: af}. Null aliases (gene-not-found) are skipped, so a
    partial response with per-alias errors still yields every gene that DID resolve."""
    out: dict = {}
    for _alias, gene in (payload.get("data") or {}).items():
        if not gene:
            continue
        for v in gene.get("variants") or []:
            vid = v.get("variant_id")
            if not vid:
                continue
            af = _af_from_variant(v)
            if af is None:
                continue
            out[norm_key(vid)] = af
    return out


def parse_mt_af(payload: dict) -> dict:
    """region.mitochondrial_variants payload -> {bare_key: af_hom}. af_hom = ac_hom/an (an>0)."""
    out: dict = {}
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
    """Rows (variant_id='gnomad:'+key, allele_freq) for keys present in BOTH gnomAD and the cohort."""
    af_map: dict = {}
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
    """-> (set of bare Y/MT keys, sorted distinct raw Y gene_symbols). Reads variant_id + gene_symbol."""
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
    y_raw = sorted({str(g) for g in cols.loc[is_y, "gene_symbol"].dropna()})
    return keys, y_raw


# ---------------------------------------------------------------- network (thin; lazy requests import)
def _post_retry(session, query: str, variables: dict, max_retries: int = 6, base_pause: float = 2.0) -> dict:
    """POST with retry+exponential backoff. gnomAD rate-limits -> non-200 or non-JSON 200; both are RETRIED
    (never silently skipped). Returns parsed JSON on the first 200-with-valid-JSON (per-alias 'errors' in
    the body are tolerated -- partial data is used). Raises only after exhausting retries."""
    last = "?"
    for attempt in range(max_retries):
        try:
            r = session.post(GNOMAD_API, json={"query": query, "variables": variables}, timeout=180)
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    last = "non-JSON 200 (throttled)"
            else:
                last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(base_pause * (2 ** attempt))  # 2,4,8,16,32,64s
    raise RuntimeError(f"gnomAD throttled/failed after {max_retries} retries: {last}")


def fetch_y_af(clean_genes, dataset: str, batch_size: int = 8, pause: float = 3.0) -> dict:
    import requests
    s = requests.Session()
    out: dict = {}
    batches = [clean_genes[i:i + batch_size] for i in range(0, len(clean_genes), batch_size)]
    for bi, batch in enumerate(batches, 1):
        try:
            out.update(parse_y_batch(_post_retry(s, build_y_batch_query(batch), {"ds": dataset})))
        except Exception as e:  # noqa: BLE001 -- loud after retries, never silent
            print(f"  [WARN] Y batch {bi} ({batch[0]}..{batch[-1]}): {e}", file=sys.stderr)
        print(f"  ...Y batch {bi}/{len(batches)} ({len(batch)} genes), {len(out)} variants so far")
        if pause:
            time.sleep(pause)
    return out


def fetch_mt_af(dataset: str) -> dict:
    import requests
    s = requests.Session()
    return parse_mt_af(_post_retry(s, _MT_REGION_QUERY, {"ds": dataset}))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build gnomAD Y/MT allele-frequency parquet for the cohort")
    ap.add_argument("--clinvar", required=True, help="cohort parquet (variant_id + gene_symbol)")
    ap.add_argument("--gnomad", default="data/processed/gnomad_v4_exomes.parquet",
                    help="existing gnomAD parquet to merge onto (variant_id, allele_freq)")
    ap.add_argument("--out", default="data/processed/gnomad_ymt_af.parquet")
    ap.add_argument("--merged-out", default=None,
                    help="if set, write base+Y/MT dedup-merged parquet here")
    ap.add_argument("--dataset", default="gnomad_r4")
    ap.add_argument("--batch-size", type=int, default=8, help="Y genes per aliased request")
    ap.add_argument("--pause", type=float, default=3.0, help="seconds between Y batches (rate-limit courtesy)")
    ap.add_argument("--min-y-cover", type=float, default=0.5,
                    help="warn if matched Y fraction < this (catches a silent throttle/clean regression)")
    a = ap.parse_args(argv)

    keys, y_raw = cohort_ymt(a.clinvar)
    y_keys = {k for k in keys if k.split(":", 1)[0] == "Y"}
    mt_keys = keys - y_keys
    y_genes = clean_y_genes(y_raw)
    print(f"[cohort] {len(keys)} Y/MT keys ({len(y_keys)} Y / {len(mt_keys)} MT); "
          f"{len(y_raw)} raw Y gene strings -> {len(y_genes)} clean Y genes")

    print("[fetch] gnomAD mtDNA region (one call) ...")
    mt_af = fetch_mt_af(a.dataset)
    print(f"  mtDNA: {len(mt_af)} gnomAD variants")
    print(f"[fetch] gnomAD Y genes in batches of {a.batch_size} ...")
    y_af = fetch_y_af(y_genes, a.dataset, a.batch_size, a.pause)
    print(f"  Y: {len(y_af)} gnomAD variants")

    df = build_ymt_frame(keys, y_af, mt_af)
    cov_y = sum(1 for v in df["variant_id"] if v.startswith("gnomad:Y:"))
    cov_mt = len(df) - cov_y
    print(f"[coverage] Y {cov_y}/{len(y_keys)} ({cov_y / max(len(y_keys),1):.0%}) | "
          f"MT {cov_mt}/{len(mt_keys)} ({cov_mt / max(len(mt_keys),1):.0%})")
    if len(df) == 0:
        print("[ERROR] zero matches -- check dataset/chrom before merging", file=sys.stderr)
        return 1
    if cov_y / max(len(y_keys), 1) < a.min_y_cover:
        print(f"[WARN] Y coverage below {a.min_y_cover:.0%} -- possible throttle or gene-clean regression; "
              "inspect before merging.", file=sys.stderr)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"[write] {a.out}: {len(df)} rows")
    if a.merged_out:
        nb, ny, nc = merge_into_gnomad(df, a.gnomad, a.merged_out)
        print(f"[merge] {a.gnomad} ({nb}) + Y/MT ({ny}) -> {a.merged_out} ({nc} unique)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
