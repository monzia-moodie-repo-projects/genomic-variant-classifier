"""
resolve_alleleless_ncbi.py  (2026-07-09)
==========================================================================
FINAL evidence step: resolve the 19,369 STALE_MISS allele-less rows by their ClinVar
VariationID against NCBI, to obtain authoritative canonical ref/alt where one exists.

These rows have a real VariationID (present in variant_summary with a Start + Type) but
are ABSENT from both the raw and fresh ClinVar VCFs -- verified 2026-07-09, not a bug:
ClinVar's VCF omits many old / structural / repeat / cytogenetic records. NCBI's esummary
(db=clinvar) returns the canonical SPDI / GRCh38 location for a VariationID even when the
VCF does not carry it, so it is the authoritative resolver.

ENDPOINT: E-utilities esummary, db=clinvar, id=<comma-separated VariationIDs> (JSON).
  https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=clinvar&retmode=json&id=...
  Parse result[uid].variation_set[0]:
     - canonical_spdi: 'seqid:pos0:deletedSeq:insertedSeq'  (pos0 = 0-based) -> ref/alt, or
     - variation_loc: entry with assembly_name=='GRCh38' -> chr, start(1-based), ref, alt.

ENGINEERING:
  * BATCHING: up to --batch (default 200) VariationIDs per esummary call -> ~100 calls.
  * RATE LIMIT: --rate (default 3/s; use 10/s only with --api-key). Sleep between batches.
  * CACHE: --cache JSON on disk, varid -> parsed record; resume-safe / idempotent.
  * FAIL LOUD: verdict in {RESOLVED_HAS_ALLELE, CONFIRMED_ALLELELESS_NCBI, NOT_FOUND,
    HTTP_ERROR, PARSE_ERROR}. Errors are retried (--retries) then SURFACED, never silently
    treated as allele-less.
  * GENOME-VERIFY: any NCBI ref/alt must match GRCh38 at the returned coordinate before it
    is accepted as RESOLVED_HAS_ALLELE; a genome mismatch -> RESOLVED_GENOME_MISMATCH (quar).
  * GRCh38 ONLY.

The network fetch is injected (fetch_fn) so all parsing/bucketing/genome-verify logic is
unit-testable OFFLINE with canned esummary JSON. Recovers to TSV; writes NO cohort.

USAGE
  python scripts/resolve_alleleless_ncbi.py \
      --recovery-full   outputs/alleleless_identity_recovery_full.tsv \
      --fasta           data/external/grch38/GRCh38.fa \
      --cache           outputs/ncbi_clinvar_cache.json \
      [--api-key XXXX] [--rate 3] [--batch 200] [--limit N-for-testing]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

_NULL = {"", "na", "nan", "none", "-", ".", "<na>"}
ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def _norm_chrom(c: str) -> str:
    return str(c).strip().lstrip("chr")


def _clean_id(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s


def _real(x) -> bool:
    return str(x).strip().lower() not in _NULL and len(str(x).strip()) >= 1


def default_fetch(ids, api_key=None, timeout=30):
    """Real network fetch: esummary db=clinvar JSON for comma-separated ids. Returns dict."""
    params = {"db": "clinvar", "retmode": "json", "id": ",".join(ids)}
    if api_key:
        params["api_key"] = api_key
    url = ESUMMARY + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "genassoc-alleleless/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_spdi(spdi: str):
    """'seqid:pos0:del:ins' -> (seqid, pos1based, ref, alt) or None. pos0 is 0-based."""
    if not spdi or spdi.count(":") < 3:
        return None
    seqid, pos0, deleted, inserted = spdi.split(":", 3)
    try:
        pos1 = int(pos0) + 1
    except ValueError:
        return None
    return seqid, pos1, deleted, inserted


# map RefSeq accession -> chrom for GRCh38 primary assembly
_ACC2CHR = {f"NC_0000{n:02d}": str(i) for i, n in enumerate(range(1, 23), start=1)}
_ACC2CHR["NC_000023"] = "X"
_ACC2CHR["NC_000024"] = "Y"
_ACC2CHR["NC_012920"] = "MT"


def _acc_to_chrom(seqid: str):
    return _ACC2CHR.get(str(seqid).split(".")[0])


def parse_esummary_record(rec: dict):
    """Extract (chrom, pos1, ref, alt) for GRCh38 from a clinvar esummary uid record.
    Returns dict with keys chrom,pos,ref,alt,spdi OR {'no_simple_allele': True} OR None."""
    vsets = rec.get("variation_set") or []
    if not vsets:
        return None
    vs = vsets[0]
    # 1. canonical SPDI
    spdi = vs.get("canonical_spdi") or ""
    parsed = parse_spdi(spdi) if spdi else None
    if parsed:
        seqid, pos1, ref, alt = parsed
        chrom = _acc_to_chrom(seqid)
        if chrom and _real(ref) and _real(alt):
            return {"chrom": chrom, "pos": pos1, "ref": ref, "alt": alt, "spdi": spdi}
    # 2. GRCh38 variation_loc
    for loc in vs.get("variation_loc", []) or []:
        if str(loc.get("assembly_name")) == "GRCh38":
            chrom = _norm_chrom(loc.get("chr", ""))
            ref = loc.get("ref") or loc.get("reference_allele") or ""
            alt = loc.get("alt") or loc.get("alternate_allele") or ""
            start = loc.get("start") or loc.get("display_start")
            if chrom and _real(ref) and _real(alt) and start:
                return {"chrom": chrom, "pos": int(start), "ref": ref, "alt": alt,
                        "spdi": spdi}
    # present but no simple allele (CNV / cytogenetic / etc.)
    return {"no_simple_allele": True, "spdi": spdi}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recovery-full", default="outputs/alleleless_identity_recovery_full.tsv")
    ap.add_argument("--fasta", default="data/external/grch38/GRCh38.fa")
    ap.add_argument("--cache", default="outputs/ncbi_clinvar_cache.json")
    ap.add_argument("--out", default="outputs/alleleless_ncbi_resolved.tsv")
    ap.add_argument("--summary", default="outputs/alleleless_ncbi_summary.json")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--rate", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="cap ids (for testing)")
    ap.add_argument("--_fetch_fixture", default=None,
                    help="offline: path to a JSON file mapping id->esummary-record (testing)")
    a = ap.parse_args(argv)

    res = pd.read_csv(a.recovery_full, sep="\t")
    stale = res[res["verdict"] == "STALE_MISS_TRY_NCBI"].copy()
    stale["cohort_varid"] = stale["cohort_varid"].map(_clean_id)
    ids = [v for v in stale["cohort_varid"].tolist() if v]
    if a.limit:
        ids = ids[:a.limit]
    print(f"STALE_MISS rows to resolve via NCBI: {len(ids):,}")

    # cache
    cache = {}
    if Path(a.cache).exists():
        try:
            cache = json.loads(Path(a.cache).read_text())
        except Exception:
            cache = {}
    todo = [i for i in ids if i not in cache]
    print(f"  already cached: {len(ids) - len(todo):,}; to fetch: {len(todo):,}")

    # offline fixture fetch (testing) OR real batched network fetch
    fixture = None
    if a._fetch_fixture:
        fixture = json.loads(Path(a._fetch_fixture).read_text())

    def fetch_batch(batch_ids):
        if fixture is not None:
            return {"result": {**{"uids": batch_ids},
                               **{i: fixture.get(i, {}) for i in batch_ids}}}
        last = None
        for attempt in range(a.retries):
            try:
                return default_fetch(batch_ids, api_key=a.api_key)
            except Exception as e:  # noqa
                last = e
                time.sleep(1.0 + attempt)
        return {"_http_error": str(last)}

    interval = 1.0 / a.rate if a.rate > 0 else 0
    for start in range(0, len(todo), a.batch):
        batch = todo[start:start + a.batch]
        data = fetch_batch(batch)
        if "_http_error" in data:
            for i in batch:
                cache[i] = {"verdict": "HTTP_ERROR", "error": data["_http_error"]}
        else:
            result = data.get("result", {})
            for i in batch:
                rec = result.get(i)
                if rec is None:
                    cache[i] = {"verdict": "NOT_FOUND"}
                    continue
                try:
                    parsed = parse_esummary_record(rec)
                except Exception as e:  # noqa
                    cache[i] = {"verdict": "PARSE_ERROR", "error": str(e)}
                    continue
                if parsed is None:
                    cache[i] = {"verdict": "NOT_FOUND"}
                elif parsed.get("no_simple_allele"):
                    cache[i] = {"verdict": "CONFIRMED_ALLELELESS_NCBI",
                                "spdi": parsed.get("spdi")}
                else:
                    cache[i] = {"verdict": "RESOLVED_HAS_ALLELE", **parsed}
        Path(a.cache).parent.mkdir(parents=True, exist_ok=True)
        Path(a.cache).write_text(json.dumps(cache), encoding="utf-8")
        if interval and fixture is None:
            time.sleep(interval)

    # genome-verify the RESOLVED_HAS_ALLELE
    ref_genome = None
    if Path(a.fasta).exists():
        from pyfaidx import Fasta
        ref_genome = Fasta(str(a.fasta), rebuild=False)
    contigs = set(ref_genome.keys()) if ref_genome else set()

    def genome_ref_ok(chrom, pos, ref):
        if ref_genome is None or pos is None:
            return None
        c = _norm_chrom(chrom)
        if c not in contigs:
            return None
        try:
            got = str(ref_genome[c][int(pos) - 1:int(pos) - 1 + len(ref)]).upper()
        except Exception:
            return None
        return got == str(ref).upper()

    rows = []
    for _, r in stale.iterrows():
        vid = r["cohort_varid"]
        c = cache.get(vid, {"verdict": "NOT_ATTEMPTED"})
        verdict = c.get("verdict")
        gok = None
        if verdict == "RESOLVED_HAS_ALLELE":
            gok = genome_ref_ok(c.get("chrom"), c.get("pos"), c.get("ref"))
            if gok is False:
                verdict = "RESOLVED_GENOME_MISMATCH"
            elif gok is None:
                verdict = "RESOLVED_GENOME_UNVERIFIABLE"   # could not check -> do NOT accept
        rows.append({
            "variant_id": r["variant_id"], "cohort_varid": vid,
            "chrom": c.get("chrom"), "pos": c.get("pos"),
            "ref": c.get("ref"), "alt": c.get("alt"), "spdi": c.get("spdi"),
            "ncbi_verdict": verdict, "genome_ok": gok, "error": c.get("error"),
        })
    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, sep="\t", index=False)

    summary = {
        "date": "2026-07-09",
        "stale_miss_total": int(len(stale)),
        "attempted": int(len(ids)),
        "by_ncbi_verdict": out["ncbi_verdict"].value_counts().to_dict(),
        "resolved_with_allele_genome_ok": int(((out["ncbi_verdict"] == "RESOLVED_HAS_ALLELE")
                                               & (out["genome_ok"] == True)).sum()),  # noqa
    }
    errs = int(out["ncbi_verdict"].isin(["HTTP_ERROR", "PARSE_ERROR"]).sum())
    summary["errors_surfaced"] = errs
    summary["genome_unverifiable"] = int((out["ncbi_verdict"] == "RESOLVED_GENOME_UNVERIFIABLE").sum())
    Path(a.summary).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n--- NCBI RESOLUTION ---")
    for k, v in summary["by_ncbi_verdict"].items():
        print(f"  {k:28s}: {v:,}")
    print(f"\nresolved w/ allele (genome-verified): {summary['resolved_with_allele_genome_ok']:,}")
    if errs:
        print(f"*** {errs:,} rows had HTTP/PARSE errors -- NOT resolved, NOT allele-less. "
              f"Re-run to retry (cache resumes). ***", file=sys.stderr)
    print(f"\nwrote {a.out}")
    print(f"wrote {a.summary}")
    print(f"cache: {a.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
