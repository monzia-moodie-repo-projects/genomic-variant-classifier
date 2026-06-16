"""build_gnomad_ymt_af.py -- Monzia Moodie

Fetch gnomAD population allele frequencies for the cohort's chrY and chrMT variants -- absent from both
gnomad_v4_exomes.parquet (its gene list omitted Y/MT) and the 1000G high-coverage panel. Without this,
~6,300 cohort variants carry a silent allele_freq=0 from every source.

  Y  : gene(gene_symbol){ variants } -> exome.af (genome fallback), ONE gene per request. The cohort
       gene_symbol is dirty for Y (';'-joined multi-gene + free-text), so symbols are CLEANED first.
       Requests are SERIAL (not aliased-batched): gnomAD enforces a per-query COST ceiling and rejects
       multi-'variants' queries with HTTP 400, so one gene per request is the cost-safe unit. Pacing +
       retry/backoff absorb gnomAD's request-rate throttle (non-JSON 200 / 429), which must be retried,
       never silently skipped.
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

_Y_QUERY = (
    "query($s:String!,$ds:DatasetId!){ gene(gene_symbol:$s, reference_genome:GRCh38){"
    " variants(dataset:$ds){ variant_id exome{af} genome{af} } } }"
)
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


# GRCh38 pseudoautosomal regions (1-based). gnomAD canonicalises PAR variants to chromosome X, but
# ClinVar annotates them on Y, so a PAR gene query returns 'X-pos-ref-alt' that must be re-keyed to Y to
# match the cohort. PAR1 is identical on X and Y; PAR2 is shifted. Coords: GRC / UCSC hg38 / Wikipedia.
_PAR1 = (10001, 2781479)             # X == Y on PAR1
_PAR2_X = (155701383, 156030895)     # X coordinates of PAR2
_PAR2_SHIFT = 98813480               # Y_pos = X_pos - shift  (155701383 - 56887903)


def y_key(gnomad_vid: str):
    """Map a gnomAD variant_id to a bare 'Y:pos:ref:alt' key matching the ClinVar cohort. Handles gnomAD
    reporting PAR variants on chromosome X: PAR1 keeps the coordinate (X==Y), PAR2 shifts X->Y. Returns
    None for a genuine non-PAR X variant (not a Y variant) or a malformed id, so callers skip it."""
    parts = str(gnomad_vid).split("-")
    if len(parts) < 4:
        return None
    chrom = parts[0][3:] if parts[0].startswith("chr") else parts[0]
    pos, ref, alt = parts[1], parts[2], "-".join(parts[3:])
    if chrom == "Y":
        return f"Y:{pos}:{ref}:{alt}"
    if chrom == "X":
        try:
            ip = int(pos)
        except (TypeError, ValueError):
            return None
        if _PAR1[0] <= ip <= _PAR1[1]:
            return f"Y:{ip}:{ref}:{alt}"                     # PAR1: identical coordinate on X and Y
        if _PAR2_X[0] <= ip <= _PAR2_X[1]:
            return f"Y:{ip - _PAR2_SHIFT}:{ref}:{alt}"       # PAR2: shift X coordinate to Y
    return None


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


def _af_from_variant(v) -> float:
    af = (v.get("exome") or {}).get("af")
    if af is None:
        af = (v.get("genome") or {}).get("af")
    return None if af is None else float(af)


def parse_y_af(payload: dict) -> dict:
    """Single-gene gene.variants payload -> {Y-key: af}. exome.af preferred, genome fallback; null AF, a
    null gene (gene-not-found), and genuine non-PAR X variants are skipped. PAR variants that gnomAD
    reports on X are re-keyed to Y via y_key so they match the cohort's Y-annotated PAR variants."""
    out: dict = {}
    gene = (payload.get("data") or {}).get("gene") or {}
    for v in gene.get("variants") or []:
        vid = v.get("variant_id")
        if not vid:
            continue
        af = _af_from_variant(v)
        if af is None:
            continue
        k = y_key(vid)
        if k is None:
            continue
        out[k] = af
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
def _post_retry(session, query: str, variables: dict, max_retries: int = 8, base_pause: float = 2.0) -> dict:
    """POST with retry+exponential backoff (capped 60s). RETRYABLE: non-JSON 200 (throttle), HTTP 429,
    HTTP 5xx, connection errors -- never silently skipped. NON-RETRYABLE: other 4xx (e.g. 400 cost-limit,
    404) -> raise immediately with a body snippet (retrying a deterministic client error is futile). Per-
    alias/per-gene 'errors' inside a 200 body are tolerated (partial data used). Raises after exhausting."""
    last = "?"
    for attempt in range(max_retries):
        try:
            r = session.post(GNOMAD_API, json={"query": query, "variables": variables}, timeout=180)
            sc = r.status_code
            if sc == 200:
                try:
                    return r.json()
                except ValueError:
                    last = "non-JSON 200 (throttled)"
            elif sc == 429 or 500 <= sc < 600:
                last = f"HTTP {sc} (retryable)"
            else:
                snippet = (r.text or "")[:160].replace("\n", " ")
                raise RuntimeError(f"HTTP {sc} (non-retryable): {snippet}")
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001 -- network errors are retryable
            last = repr(e)
        time.sleep(min(base_pause * (2 ** attempt), 60.0))
    raise RuntimeError(f"gnomAD throttled/failed after {max_retries} retries: {last}")


def fetch_y_af(clean_genes, dataset: str, pause: float = 6.0) -> dict:
    """SERIAL one-gene-per-request (cost-safe). Pacing keeps us under gnomAD's request-rate throttle;
    _post_retry absorbs any throttle that still occurs. A gene-not-found is a 200 with a null gene and
    contributes {} (no raise); a persistent throttle on a gene raises -> logged loud, not silent."""
    import requests
    s = requests.Session()
    out: dict = {}
    n = len(clean_genes)
    for i, g in enumerate(clean_genes, 1):
        try:
            out.update(parse_y_af(_post_retry(s, _Y_QUERY, {"s": g, "ds": dataset})))
        except Exception as e:  # noqa: BLE001 -- loud after retries, never silent
            print(f"  [WARN] Y gene {g}: {e}", file=sys.stderr)
        if i % 10 == 0 or i == n:
            print(f"  ...{i}/{n} Y genes, {len(out)} variants so far")
        if pause and i < n:
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
    ap.add_argument("--pause", type=float, default=6.0,
                    help="seconds between serial Y-gene calls (rate-limit courtesy; ~10/min at 6s)")
    ap.add_argument("--min-y-cover", type=float, default=0.10,
                    help="warn if matched Y fraction < this. chrY's honest overlap is modest (gnomAD "
                         "callability + no-allele cohort entries); this is a regression tripwire, not a target")
    a = ap.parse_args(argv)

    keys, y_raw = cohort_ymt(a.clinvar)
    y_keys = {k for k in keys if k.split(":", 1)[0] == "Y"}
    mt_keys = keys - y_keys
    y_genes = clean_y_genes(y_raw)
    eta = len(y_genes) * a.pause / 60.0
    print(f"[cohort] {len(keys)} Y/MT keys ({len(y_keys)} Y / {len(mt_keys)} MT); "
          f"{len(y_raw)} raw Y gene strings -> {len(y_genes)} clean Y genes (~{eta:.0f} min at {a.pause:.0f}s/gene)")

    print("[fetch] gnomAD mtDNA region (one call) ...")
    mt_af = fetch_mt_af(a.dataset)
    print(f"  mtDNA: {len(mt_af)} gnomAD variants")
    print(f"[fetch] gnomAD Y genes serially ({len(y_genes)} genes @ {a.pause:.0f}s) ...")
    y_af = fetch_y_af(y_genes, a.dataset, a.pause)
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
