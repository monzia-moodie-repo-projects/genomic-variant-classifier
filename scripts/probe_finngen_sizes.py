#!/usr/bin/env python3
"""FINNGEN R12 vs R13 SIZE-GAP PROBE (v2: PROBE_V2_INTEGRITY).

Single-pass, streaming, no full in-RAM load. Explains the R13<R12 file-size gap with
EVIDENCE and validates the RUN_17_PLAN B.D2 'same variants, same coords' assertion.

In ONE 28-30GB read per file, computes:
  (A) GZIP INTEGRITY VERDICT  -- full decompress (multi-member/bgzip aware): CLEAN vs
      TRUNCATED vs CORRUPT. Directly answers "is my downloaded file broken?" (the
      truncation/corruption explanation for a smaller file) WITHOUT re-downloading.
  (B) SHA-256 of the raw bytes -- compare to a published checksum or a re-download to
      prove the local file == the source byte-for-byte.
  (C) EXACT ROW COUNT          -- the dominant size driver; the real size-gap answer.

Plus (separate small reads):
  (D) Column count + which columns differ (confirm 1017 vs 1025, show the deltas).
  (E) Variant-KEY overlap on a HEAD sample (quick 'same variants?' signal).

Why single-pass zlib instead of gzip.open: we need integrity + hash + row count from the
SAME disk read. gzip.open can't give the raw-byte hash or a clean truncation verdict in one
pass. zlib.decompressobj(wbits=31) reads gzip (incl. bgzip/BGZF multi-member) and validates
CRC; reaching end-of-stream cleanly => not truncated; a zlib.error => corrupt.

Usage:
  python probe_finngen_sizes.py --r12 <path> --r13 <path> [--sample 200000] [--no-keys]
  python probe_finngen_sizes.py --one <path>     # integrity+hash+rows for a single file
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import sys
import time
import zlib

CHUNK = 8 * 1024 * 1024  # 8 MiB


def combined_pass(path, label):
    """ONE binary pass: sha256(raw) + full multi-member gzip decompress + newline count +
    integrity verdict. Returns dict(sha256, raw_bytes, lines, members, integrity, error)."""
    sha = hashlib.sha256()
    newlines = 0
    raw_bytes = 0
    members = 0
    pending = False            # current decompressor has unfinished input
    d = zlib.decompressobj(wbits=31)
    integrity = "UNKNOWN"
    error = None
    t0 = time.time()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(CHUNK)
                if not chunk:
                    break
                sha.update(chunk)
                raw_bytes += len(chunk)
                buf = chunk
                while buf:
                    out = d.decompress(buf)
                    newlines += out.count(b"\n")
                    if d.eof:
                        members += 1
                        pending = False
                        rest = d.unused_data
                        if rest:
                            d = zlib.decompressobj(wbits=31)
                            buf = rest          # feed next member's bytes
                        else:
                            buf = b""
                    else:
                        pending = True
                        buf = b""
                if raw_bytes % (1024 * 1024 * 1024) < CHUNK:  # ~each GB
                    gb = raw_bytes / (1024**3)
                    print(f"  [{label}] {gb:,.1f} GB read, {newlines:,} lines, "
                          f"{members} members ({time.time()-t0:,.0f}s)...", flush=True)
            tail = d.flush()
            newlines += tail.count(b"\n")
        if pending and not d.eof:
            integrity = "TRUNCATED"
        else:
            integrity = "CLEAN"
    except zlib.error as e:
        integrity = "CORRUPT"
        error = f"zlib.error: {e}"
    except EOFError as e:
        integrity = "TRUNCATED"
        error = f"EOFError: {e}"
    except Exception as e:  # noqa
        integrity = "ERROR"
        error = f"{type(e).__name__}: {e}"
    return {
        "sha256": sha.hexdigest(),
        "raw_bytes": raw_bytes,
        "lines": newlines,
        "members": members,
        "integrity": integrity,
        "error": error,
        "seconds": time.time() - t0,
    }


def sniff_header(path):
    """Small read: first (header) line via gzip.open (handles multi-member).
    Corruption-tolerant: returns ([], None) if the stream can't be read (the
    integrity verdict from combined_pass is the authoritative signal)."""
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            header = fh.readline().rstrip("\r\n")
    except (OSError, EOFError, zlib.error) as e:
        print(f"  (header unreadable: {type(e).__name__}: {e})")
        return [], None
    delim = "\t" if "\t" in header else None
    fields = header.split(delim) if delim else header.split()
    return fields, delim


def guess_key_columns(fields):
    lower = [f.lower() for f in fields]
    def find(cands):
        for c in cands:
            if c in lower:
                return lower.index(c)
        return None
    return (find(["#chrom", "chrom", "chr", "chromosome", "#chr"]),
            find(["pos", "position", "bp", "base_pair_location"]),
            find(["ref", "reference", "a1", "allele1"]),
            find(["alt", "alternate", "a2", "allele2"]))


def sample_keys(path, key_idx, delim, limit):
    chrom, pos, ref, alt = key_idx
    keys = set()
    if None in key_idx:
        return keys, "key columns not all found"
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        fh.readline()
        for i, line in enumerate(fh):
            if i >= limit:
                break
            parts = line.rstrip("\r\n").split(delim) if delim else line.split()
            try:
                keys.add(f"{parts[chrom]}:{parts[pos]}:{parts[ref]}:{parts[alt]}")
            except IndexError:
                continue
    return keys, None


def report_one(path, label, do_keys, sample):
    print(f"\n{'='*72}\n[{label}] {path}\n{'='*72}")
    # combined_pass FIRST -- it is the authoritative integrity verdict and never crashes
    res = combined_pass(path, label)
    # header sniff is best-effort (corruption-tolerant); integrity above is the real signal
    fields, delim = sniff_header(path)
    print(f"  columns: {len(fields) if fields else 'UNREADABLE'}"
          f"  (delim={'TAB' if delim else 'whitespace'})")
    data_rows = max(res["lines"] - 1, 0)  # minus header
    print(f"  integrity : {res['integrity']}" + (f"  ({res['error']})" if res['error'] else ""))
    print(f"  sha256    : {res['sha256']}")
    print(f"  raw bytes : {res['raw_bytes']:,}")
    print(f"  gzip members: {res['members']}  (>1 => bgzip/BGZF multi-member)")
    print(f"  lines     : {res['lines']:,}  => data rows (minus header): {data_rows:,}")
    print(f"  elapsed   : {res['seconds']:,.0f}s")
    res["fields"], res["delim"], res["data_rows"] = fields, delim, data_rows
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r12")
    ap.add_argument("--r13")
    ap.add_argument("--one", help="single-file mode: integrity+hash+rows")
    ap.add_argument("--sample", type=int, default=200_000)
    ap.add_argument("--no-keys", action="store_true")
    args = ap.parse_args()

    print("="*72)
    print("FINNGEN SIZE-GAP PROBE v2 (PROBE_V2_INTEGRITY) -- streaming, single-pass")
    print("="*72)

    if args.one:
        report_one(args.one, "FILE", not args.no_keys, args.sample)
        return

    if not (args.r12 and args.r13):
        print("ERROR: provide --r12 and --r13 (or --one <path>).")
        return 2

    r12 = report_one(args.r12, "R12", not args.no_keys, args.sample)
    r13 = report_one(args.r13, "R13", not args.no_keys, args.sample)

    print(f"\n{'='*72}\nCOMPARISON\n{'='*72}")
    # column delta
    s12, s13 = set(r12["fields"]), set(r13["fields"])
    only12 = [c for c in r12["fields"] if c not in s13]
    only13 = [c for c in r13["fields"] if c not in s12]
    print(f"  columns: R12={len(r12['fields'])}  R13={len(r13['fields'])}")
    print(f"  only in R12 ({len(only12)}): {only12[:25]}{' ...' if len(only12)>25 else ''}")
    print(f"  only in R13 ({len(only13)}): {only13[:25]}{' ...' if len(only13)>25 else ''}")

    # integrity gate
    print(f"\n  INTEGRITY: R12={r12['integrity']}  R13={r13['integrity']}")
    if r12["integrity"] != "CLEAN" or r13["integrity"] != "CLEAN":
        print("  *** AT LEAST ONE FILE IS NOT CLEAN -- do NOT use for a paid run. ***")
        print("  *** A TRUNCATED/CORRUPT R13 would itself explain the smaller size. ***")

    # row-count verdict (the size answer)
    n12, n13 = r12["data_rows"], r13["data_rows"]
    print(f"\n  ROW-COUNT VERDICT (the size-gap answer):")
    print(f"    R12 data rows = {n12:,}")
    print(f"    R13 data rows = {n13:,}")
    if n12:
        pct = 100.0 * (n13 - n12) / n12
        print(f"    R13 - R12 = {n13 - n12:,}  ({pct:+.2f}% vs R12)")
        if n13 < n12:
            print("    => R13 has FEWER variant rows -> primary cause of the smaller file.")
            print("       R12/R13 are NOT an identical variant set; B.D2 line 30 'same")
            print("       variants' needs a CAVEAT (compare on the INTERSECTION).")
        elif n13 > n12:
            print("    => R13 has MORE rows but a SMALLER file -> size gap is per-row")
            print("       encoding/precision/compression, NOT truncation. Still verify overlap.")
        else:
            print("    => identical row counts -> size gap is purely encoding/compression.")

    # key overlap (head sample)
    if not args.no_keys:
        print(f"\n  VARIANT-KEY OVERLAP (HEAD sample {args.sample:,} rows each):")
        k12 = guess_key_columns(r12["fields"]); k13 = guess_key_columns(r13["fields"])
        print(f"    R12 key idx (chrom,pos,ref,alt): {k12}")
        print(f"    R13 key idx (chrom,pos,ref,alt): {k13}")
        ks12, e12 = sample_keys(args.r12, k12, r12["delim"], args.sample)
        ks13, e13 = sample_keys(args.r13, k13, r13["delim"], args.sample)
        if e12 or e13:
            print(f"    overlap NOT computed: R12={e12} R13={e13}")
        else:
            inter = ks12 & ks13
            uni = ks12 | ks13
            jac = len(inter)/len(uni) if uni else 0.0
            print(f"    R12 sample keys={len(ks12):,}  R13 sample keys={len(ks13):,}")
            print(f"    intersection={len(inter):,}  Jaccard(head)={jac:.4f}")
            print("    (HEAD sample: high overlap = strong 'same variants' evidence; low")
            print("     overlap may be different sort order -> warrants a full keyed pass.)")

    print("\nDONE.")


if __name__ == "__main__":
    sys.exit(main())
