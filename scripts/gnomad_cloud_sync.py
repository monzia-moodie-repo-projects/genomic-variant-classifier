#!/usr/bin/env python3
r"""gnomad_cloud_sync.py -- Author: Monzia Moodie

No-gaps verification + transfer planning for the gnomAD v4.1 exome raw VCFs, with the
DESTINATION being Google Drive (via rclone) rather than a paid GCS bucket.

Context:
  * The pipeline reads the BUILT gnomad_v4_exomes.parquet, NOT these raw VCFs. The raw set
    is build-time/archival only. Storing all 24 contigs in Drive is optional completeness.
  * Source = the FREE public mirror gs://gcp-public-data--gnomad (no egress charge):
        gs://gcp-public-data--gnomad/release/4.1/vcf/exomes/gnomad.exomes.v4.1.sites.{contig}.vcf.bgz
    (verified vs gnomad_methods release_vcf_path; contig range chr1..chrY).
  * GCS(public) -> Drive has NO server-side path: bytes must pass through the machine.
    `rclone copyurl <https-url> <drive-remote>:<dest>` streams with minimal local footprint.

This tool never calls gcloud/rclone itself -- it consumes their LISTING output, so its logic
is fully testable and it can never half-run a transfer. parse_listing accepts BOTH:
  * `gcloud storage ls -l`  ->  "<bytes>  <iso-ts>  gs://.../file.vcf.bgz"
  * `rclone lsl`            ->  "<bytes> <YYYY-MM-DD> <HH:MM:SS.fff> path/file.vcf.bgz"

Workflow (Drive destination):
  gcloud storage ls -l gs://gcp-public-data--gnomad/release/4.1/vcf/exomes/ > source.txt
  rclone lsl genvarcla:genomic-variant-data/external/gnomad                  > dest.txt   2>$null
  python scripts/gnomad_cloud_sync.py plan   --source-listing source.txt --dest-listing dest.txt \
         --dest genvarcla:genomic-variant-data/external/gnomad --tool rclone
  # run the emitted commands (dry-run first!), re-capture dest.txt, then:
  python scripts/gnomad_cloud_sync.py verify --source-listing source.txt --dest-listing dest.txt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

EXPECTED_CONTIGS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
CONTIG_RE = re.compile(r"\.sites\.(chr(?:[0-9]{1,2}|X|Y))\.vcf\.bgz$", re.IGNORECASE)

def _require_nonempty_source(src: dict) -> None:
    """A source listing that parses to 0 gnomAD contigs means the capture FAILED (e.g. gcloud
    wrote an error to stderr and `>` only caught stdout). Refuse loudly rather than reporting
    a misleading 'complete'/'GO' that is only true because the source vanished."""
    if len(src) == 0:
        raise SystemExit(
            "ABORT: source listing parsed 0 gnomAD contigs -- the capture is empty or invalid.\n"
            "  Re-capture with BOTH streams and verify, e.g.:\n"
            "    gcloud storage ls -l gs://gcp-public-data--gnomad/release/4.1/vcf/exomes/ *> source.txt\n"
            "  then confirm it holds 24 .bgz lines and no ERROR before re-running."
        )

GS_PUBLIC = "gs://gcp-public-data--gnomad/"
HTTPS_PUBLIC = "https://storage.googleapis.com/gcp-public-data--gnomad/"


def _norm_contig(c: str) -> str:
    c = c.lower()
    return "chrX" if c == "chrx" else "chrY" if c == "chry" else c


def parse_listing(text: str) -> dict[str, dict]:
    """Parse `gcloud storage ls -l` OR `rclone lsl` -> {contig: {size, path}}.
    Format-agnostic: size = first all-digit token; path = last token; the gnomAD filename
    pattern is the real filter, so date/time/TOTAL/noise tokens can't false-match."""
    out: dict[str, dict] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.upper().startswith("TOTAL"):
            continue
        parts = line.split()
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        path = parts[-1]
        m = CONTIG_RE.search(path)
        if not m:
            continue
        contig = _norm_contig(m.group(1))
        size = int(parts[0])
        if contig not in out or size > out[contig]["size"]:
            out[contig] = {"size": size, "path": path}
    return out


def evaluate(src: dict[str, dict], dst: dict[str, dict]) -> list[dict]:
    rows = []
    for c in EXPECTED_CONTIGS:
        s, d = src.get(c), dst.get(c)
        if d is None:
            status = "MISSING_FROM_DEST" if s else "MISSING_FROM_BOTH"
        elif s is None:
            status = "PRESENT_DEST_NO_SRC"
        elif d["size"] != s["size"]:
            status = "SIZE_MISMATCH"
        else:
            status = "OK"
        rows.append({"contig": c, "status": status,
                     "src_size": s["size"] if s else None,
                     "dst_size": d["size"] if d else None,
                     "src_path": s["path"] if s else None})
    return rows


def _read(path_str: str) -> str:
    p = Path(path_str)
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def cmd_verify(args) -> int:
    src = parse_listing(_read(args.source_listing))
    _require_nonempty_source(src)
    dst = parse_listing(_read(args.dest_listing))
    rows = evaluate(src, dst)
    bad = [r for r in rows if r["status"] != "OK"]
    print(f"{'contig':7} {'status':20} {'src_bytes':>14} {'dst_bytes':>14}")
    for r in rows:
        print(f"{r['contig']:7} {r['status']:20} "
              f"{('' if r['src_size'] is None else r['src_size']):>14} "
              f"{('' if r['dst_size'] is None else r['dst_size']):>14}")
    print(f"\nexpected {len(EXPECTED_CONTIGS)} | OK {len(rows)-len(bad)} | problems {len(bad)}")
    if bad:
        print("NO_GO -- gaps remain:", ", ".join(f"{r['contig']}:{r['status']}" for r in bad))
        return 2
    print("GO -- all 24 contigs present in destination with byte sizes matching the public source")
    return 0


def _https_from_gs(gs_path: str) -> str:
    return gs_path.replace(GS_PUBLIC, HTTPS_PUBLIC, 1) if gs_path.startswith(GS_PUBLIC) else gs_path


def cmd_plan(args) -> int:
    src = parse_listing(_read(args.source_listing))
    _require_nonempty_source(src)
    dst = parse_listing(_read(args.dest_listing))
    rows = evaluate(src, dst)
    if len(src) < len(EXPECTED_CONTIGS):
        print(f"# WARNING: source listing has only {len(src)}/24 contigs -- capture may be partial", file=sys.stderr)
    need = [r for r in rows if r["status"] in ("MISSING_FROM_DEST", "SIZE_MISMATCH")]
    hard = [r for r in rows if r["status"] == "MISSING_FROM_BOTH"]

    if hard:
        print("# ERROR: absent from BOTH source and dest -- cannot fetch:",
              ", ".join(r["contig"] for r in hard), file=sys.stderr)

    total = sum(r["src_size"] for r in need if r["src_size"])
    print(f"# {len(need)} contig(s) to fetch, {total/1e9:.1f} GB total into {args.dest}")
    for r in need:
        gb = (r["src_size"] or 0) / 1e9
        print(f"#   {r['contig']:6} {gb:6.2f} GB  {r['status']}")
    if not need:
        print("# nothing to do -- destination already complete")
        return 1 if hard else 0

    print("#")
    dest = args.dest.rstrip("/")
    if args.tool == "report":
        print("# (report only; re-run with --tool gcloud or --tool rclone to emit commands)")
    for r in need:
        fname = r["src_path"].rstrip("/").split("/")[-1]
        if args.tool == "gcloud":
            print(f"gcloud storage cp {r['src_path']} {dest}/   # {r['contig']}")
        elif args.tool == "rclone":
            url = _https_from_gs(r["src_path"])
            print(f"rclone copyurl {url} {dest}/{fname} --progress   # {r['contig']}")
    if args.tool != "report":
        print("#")
        print("# DRY-RUN FIRST. For rclone, add --dry-run to confirm before the real run.")
        print("# After fetching, re-capture the dest listing and run `verify` (must exit 0).")
    return 1 if hard else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="prove all 24 contigs present in dest with matching sizes")
    v.add_argument("--source-listing", required=True)
    v.add_argument("--dest-listing", required=True)
    v.set_defaults(func=cmd_verify)

    p = sub.add_parser("plan", help="report missing contigs + emit fetch commands")
    p.add_argument("--source-listing", required=True)
    p.add_argument("--dest-listing", required=True)
    p.add_argument("--dest", required=True, help="destination prefix (gs://... or rclone remote:path)")
    p.add_argument("--tool", choices=["report", "gcloud", "rclone"], default="report")
    p.set_defaults(func=cmd_plan)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
