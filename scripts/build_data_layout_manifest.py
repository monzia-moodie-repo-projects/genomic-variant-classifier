#!/usr/bin/env python3
r"""build_data_layout_manifest.py -- Author: Monzia Moodie

READ-ONLY inventory of the genomic-variant-data layout after the storage migration.
Records reality so nothing is inferred: what lives under external/ vs datasets/, which
datasets have DIVERGED across the two trees, the gnomAD raw-chromosome completeness,
and which Run-17 working-set parquets are present on real-disk C:.

Safety / DriveFS notes:
  * Metadata only (name, size, mtime via os.scandir). NEVER opens/hashes file CONTENT,
    so it will not stream multi-GB files through Google Drive Desktop or evict cache.
  * Writes only the small manifest (.json + .csv). Default output is on C: (the repo),
    not G:, because G: is a ~cache-backed DriveFS mount with little real local space.

Usage:
  python scripts/build_data_layout_manifest.py \
    --g-root "G:\My Drive\genomic-variant-data" \
    --c-data "C:\Projects\genomic-variant-classifier\data" \
    --out    manifests\data_layout_manifest
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

GNOMAD_CHROM_RE = re.compile(r"\.sites\.(chr[0-9XYM]+)\.", re.IGNORECASE)
GNOMAD_EXPECTED = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}

# Run-17 working set (relative to --c-data); the local pipeline reads these.
WORKING_SET = [
    "processed/clinvar_grch38_clean.parquet", "processed/gnomad_v4_exomes.parquet",
    "external/spliceai/spliceai_index.parquet", "external/alphamissense/AlphaMissense_hg38.tsv.gz",
    "external/gnomad/gnomad.v4.1.constraint_metrics.tsv", "external/dbnsfp/dbnsfp_clinvar_index.parquet",
    "external/gtex_gene_expression.parquet", "external/reactome_gene_pathways.parquet",
    "external/rnaseq_gene_expression.parquet", "external/1kgp/kg_grch38_af.parquet",
    "external/lovd/lovd_all_variants.parquet",
]


def scan_dataset_dir(top: Path) -> dict:
    """Return {dataset_name: {n_files, bytes, files:[{relpath,bytes,mtime}]}} for the
    immediate child dirs of `top`. Loose files directly under `top` go under '<root>'.
    os.scandir = metadata only, no content read."""
    out: dict[str, dict] = {}

    def walk(base: Path):
        files = []
        stack = [base]
        while stack:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    for e in it:
                        try:
                            if e.is_dir(follow_symlinks=False):
                                stack.append(Path(e.path))
                            elif e.is_file(follow_symlinks=False):
                                st = e.stat()
                                files.append({
                                    "relpath": os.path.relpath(e.path, base),
                                    "bytes": st.st_size,
                                    "mtime": int(st.st_mtime),
                                })
                        except OSError:
                            continue
            except OSError:
                continue
        return files

    if not top.exists():
        return out
    loose = []
    with os.scandir(top) as it:
        entries = list(it)
    for e in entries:
        p = Path(e.path)
        if p.is_dir():
            fl = walk(p)
            out[p.name] = {"n_files": len(fl), "bytes": sum(f["bytes"] for f in fl), "files": fl}
        elif p.is_file():
            st = p.stat()
            loose.append({"relpath": p.name, "bytes": st.st_size, "mtime": int(st.st_mtime)})
    if loose:
        out["<root>"] = {"n_files": len(loose), "bytes": sum(f["bytes"] for f in loose), "files": loose}
    return out


def classify_divergence(ext: dict, dat: dict) -> list[dict]:
    """Per dataset name, classify presence/divergence across external vs datasets."""
    rows = []
    for name in sorted(set(ext) | set(dat)):
        e = ext.get(name)
        d = dat.get(name)
        e_names = {f["relpath"] for f in e["files"]} if e else set()
        d_names = {f["relpath"] for f in d["files"]} if d else set()
        if e and d:
            if e_names & d_names:
                status = "BOTH_OVERLAP"     # same filenames in both -> dup or conflict; inspect
            else:
                status = "BOTH_SPLIT"       # disjoint file sets across the two trees
        elif e:
            status = "EXTERNAL_ONLY"        # canonical location only -> good
        else:
            status = "DATASETS_ONLY"        # stranded -> belongs in external per config
        rows.append({
            "dataset": name, "status": status,
            "external_files": len(e_names), "external_gb": round((e["bytes"] if e else 0) / 1e9, 2),
            "datasets_files": len(d_names), "datasets_gb": round((d["bytes"] if d else 0) / 1e9, 2),
            "only_in_external": sorted(e_names - d_names)[:50],
            "only_in_datasets": sorted(d_names - e_names)[:50],
        })
    return rows


def gnomad_completeness(ext: dict, dat: dict) -> dict:
    observed: dict[str, str] = {}
    for loc, tree in (("external", ext), ("datasets", dat)):
        g = tree.get("gnomad")
        if not g:
            continue
        for f in g["files"]:
            m = GNOMAD_CHROM_RE.search(f["relpath"])
            if m:
                observed[m.group(1).replace("chrx", "chrX").replace("chry", "chrY")
                         if m.group(1).islower() else m.group(1)] = loc
    obs = set(observed)
    missing = sorted(GNOMAD_EXPECTED - obs, key=lambda c: (len(c), c))
    return {
        "expected": sorted(GNOMAD_EXPECTED, key=lambda c: (len(c), c)),
        "observed": {c: observed[c] for c in sorted(obs, key=lambda c: (len(c), c))},
        "missing_from_both": missing,
        "split_across_trees": sorted({c for c in obs} & set(observed)) if obs else [],
        "complete": not missing,
    }


def working_set_status(c_data: Path) -> list[dict]:
    rows = []
    for rel in WORKING_SET:
        p = c_data / rel
        ok = p.exists()
        rows.append({"path": rel, "present": ok,
                     "mb": round(p.stat().st_size / 1e6, 1) if ok else 0.0})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g-root", required=True, help="G: data root (contains external/ and datasets/)")
    ap.add_argument("--c-data", default=None, help="C: repo data dir (working-set check)")
    ap.add_argument("--out", default="manifests/data_layout_manifest",
                    help="output basename (writes .json and .csv); default on C:")
    args = ap.parse_args(argv)

    g = Path(args.g_root)
    ext = scan_dataset_dir(g / "external")
    dat = scan_dataset_dir(g / "datasets")
    divergence = classify_divergence(ext, dat)
    gnomad = gnomad_completeness(ext, dat)
    work = working_set_status(Path(args.c_data)) if args.c_data else []

    flags = []
    for r in divergence:
        if r["status"] in ("BOTH_SPLIT", "BOTH_OVERLAP", "DATASETS_ONLY"):
            flags.append(f"{r['dataset']}: {r['status']}")
    if not gnomad["complete"]:
        flags.append(f"gnomad raw incomplete -- missing {gnomad['missing_from_both']}")
    for w in work:
        if not w["present"]:
            flags.append(f"WORKING-SET MISSING: {w['path']}")

    manifest = {
        "g_root": str(g),
        "external_datasets": {k: {"n_files": v["n_files"], "gb": round(v["bytes"] / 1e9, 2)}
                              for k, v in sorted(ext.items())},
        "datasets_datasets": {k: {"n_files": v["n_files"], "gb": round(v["bytes"] / 1e9, 2)}
                              for k, v in sorted(dat.items())},
        "divergence": divergence,
        "gnomad_completeness": gnomad,
        "working_set": work,
        "flags": flags,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "status", "external_files", "external_gb",
                    "datasets_files", "datasets_gb"])
        for r in divergence:
            w.writerow([r["dataset"], r["status"], r["external_files"], r["external_gb"],
                        r["datasets_files"], r["datasets_gb"]])

    print(f"[manifest] wrote {out.with_suffix('.json')} and {out.with_suffix('.csv')}")
    print(f"[manifest] gnomAD complete: {gnomad['complete']}  missing: {gnomad['missing_from_both']}")
    if flags:
        print("[manifest] FLAGS:")
        for f in flags:
            print(f"   - {f}")
    else:
        print("[manifest] no divergence/missing flags")
    return 0


if __name__ == "__main__":
    sys.exit(main())
