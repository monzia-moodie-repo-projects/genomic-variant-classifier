#!/usr/bin/env python3
"""
probe_1kg_superpop_info.py  --  Monzia Moodie

Verify, BEFORE a multi-chromosome streaming build, that a 1000G panel actually carries the
five per-superpopulation AFs in its INFO field (the thing build_1kg_parquet.py needs to fill
af_1kg_afr/eur/eas/sas/amr). Streams ONLY the header + a few hundred data lines of the
smallest panel (chr22) over https; never downloads the whole file. Also HEAD-checks every URL
in the list resolves. Exit 0 only if all five super-pop keys are present AND populated.

Usage:
  python scripts/probe_1kg_superpop_info.py --url-list data/external/1kgp/grch38_panel_urls.txt
  python scripts/probe_1kg_superpop_info.py --url  https://.../chr22...vcf.gz
"""
from __future__ import annotations
import argparse, gzip, io, sys, urllib.request
from pathlib import Path

# accepted INFO spellings per super-pop (mirrors build_1kg_parquet._POP_SRC)
SUPERPOP = {
    "AFR": ("AF_AFR", "AFR_AF", "AF_afr"),
    "EUR": ("AF_EUR", "EUR_AF", "AF_eur"),
    "EAS": ("AF_EAS", "EAS_AF", "AF_eas"),
    "SAS": ("AF_SAS", "SAS_AF", "AF_sas"),
    "AMR": ("AF_AMR", "AMR_AF", "AF_amr"),
}


def _info_ids_and_first_record(stream, max_data=300):
    """Return (set of ##INFO ids, first parsed data-line INFO dict)."""
    info_ids, first_info = set(), None
    n_data = 0
    for raw in stream:
        line = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else raw
        if line.startswith("##INFO=<ID="):
            info_ids.add(line.split("ID=", 1)[1].split(",", 1)[0])
        elif line.startswith("#"):
            continue
        else:
            if first_info is None:
                cols = line.rstrip("\n").split("\t")
                if len(cols) >= 8:
                    kv = {}
                    for field in cols[7].split(";"):
                        if "=" in field:
                            k, v = field.split("=", 1)
                            kv[k] = v
                    first_info = kv
            n_data += 1
            if n_data >= max_data:
                break
    return info_ids, (first_info or {})


def probe_url(url: str) -> int:
    print(f"[probe] streaming header of {url.rsplit('/', 1)[-1]} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "kg-probe"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            gz = gzip.GzipFile(fileobj=resp)
            info_ids, first = _info_ids_and_first_record(gz)
    except Exception as e:
        print(f"  [FAIL] could not stream: {e}"); return 2

    rc = 0
    for pop, cands in SUPERPOP.items():
        hit = next((c for c in cands if c in info_ids), None)
        if not hit:
            print(f"  [FAIL] {pop}: none of {cands} present in INFO header"); rc = 1
            continue
        val = first.get(hit, "")
        ok_val = val not in ("", ".") and all(_is_floatish(x) for x in val.split(","))
        print(f"  {pop}: header key '{hit}' present; first-record value={val!r} "
              f"[{'ok' if ok_val else 'EMPTY/non-numeric'}]")
        if not ok_val:
            rc = 1
    return rc


def _is_floatish(x: str) -> bool:
    try:
        float(x); return True
    except ValueError:
        return False


def head_check(urls: list[str]) -> int:
    rc = 0
    for u in urls:
        try:
            req = urllib.request.Request(u, method="HEAD", headers={"User-Agent": "kg-probe"})
            with urllib.request.urlopen(req, timeout=30) as r:
                code = r.status
            print(f"  {code}  {u.rsplit('/', 1)[-1]}")
            if code != 200:
                rc = 1
        except Exception as e:
            print(f"  ERR  {u.rsplit('/', 1)[-1]}  ({e})"); rc = 1
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url-list", type=Path)
    ap.add_argument("--url", type=str)
    ap.add_argument("--skip-head", action="store_true", help="skip per-URL HEAD checks")
    args = ap.parse_args(argv)

    urls = []
    if args.url_list:
        urls = [u.strip() for u in args.url_list.read_text().splitlines()
                if u.strip() and not u.startswith("#")]
    probe = args.url or (next((u for u in urls if "chr22." in u), urls[-1]) if urls else None)
    if not probe:
        print("ABORT: provide --url-list or --url"); return 2

    rc = probe_url(probe)
    if urls and not args.skip_head:
        print("[head] checking every URL resolves ...")
        rc = head_check(urls) or rc
    print("\n" + ("[PASS] per-superpop AFs present + populated; URLs resolve -> safe to build"
                  if rc == 0 else
                  "[FAIL] do NOT run the full build; this panel/URL set is wrong for af_1kg_*"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
