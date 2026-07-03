"""
scripts/build_alphafold_parquet.py
==================================
Build the AlphaFold cohort structural-feature parquet -- Phase D.

Resolves the cohort's missense-bearing genes to canonical UniProt accessions (via the
local reviewed-UniProt parquet), downloads each gene's AlphaFold mmCIF structure once
(cached to disk; raw CIFs KEPT), extracts four per-residue structural features, and
writes a parquet keyed on (uniprot_accession, residue_pos):

    uniprot_accession  str
    residue_pos        int    1-based (matches AlphaMissense/HGVSp protein_pos)
    plddt              float  0-100
    rsa                float  0-1 (Shrake-Rupley SASA / Tien max-ASA, clamped)
    ss                 int    0=loop, 1=helix, 2=sheet
    dist_active        float  3-D C-alpha Angstrom to nearest UniProt active/binding site

Feature extraction is delegated to
``genomic_variant_classifier.data.alphafold_features`` (validated against the real
AF-E7ENB7 structure). Secondary structure is parsed from the file's DSSP ``_struct_conf``
records when present, else computed from coordinates. RSA is clamped to [0,1]; a raw RSA
beyond the fail-loud bound raises rather than being silently clamped.

Design notes
------------
* KEEP raw CIFs: downloaded structures are retained under the cache dir so features can
  be re-extracted later without re-downloading (Drive-backed).
* Mid-run disk guard: aborts cleanly if free space on the cache volume drops below a
  threshold, so a long download can never fill the disk.
* Resumable: an already-cached CIF is reused; re-running continues where it stopped.
* Hard-error on zero matches (mirrors build_dbsnp_parquet): a run that resolves no
  structures is a failure, not an empty success.

Usage
-----
    python scripts/build_alphafold_parquet.py \
        --cohort data/processed/clinvar_grch38_clean.parquet \
        --uniprot-index data/external/uniprot/uniprot_human_reviewed.parquet \
        --out data/external/alphafold/alphafold_cohort.parquet

    # bounded / resumable runs
    python scripts/build_alphafold_parquet.py --max-genes 500 ...
    python scripts/build_alphafold_parquet.py --genes-from genes.txt ...
    python scripts/build_alphafold_parquet.py --audit ...   # report only, no download
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# The script adds src/ to sys.path so it can import the package when run directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from genomic_variant_classifier.data import alphafold_features as aff  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_alphafold_parquet")

ALPHAFOLD_PREDICTION_API = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
UNIPROT_FEATURES_API = "https://www.ebi.ac.uk/proteins/api/features/{accession}"
_REQUEST_TIMEOUT = 30
_MIN_FREE_GB = 5.0            # mid-run disk guard threshold
_POLITE_DELAY_S = 0.5        # between network downloads (rate-limit courtesy)


def _free_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024 ** 3)


def _load_cohort_genes(cohort_path: Path) -> list[str]:
    """Distinct missense-bearing gene symbols from the cohort parquet."""
    df = pd.read_parquet(cohort_path, columns=["gene_symbol", "consequence"])
    mis = df[df["consequence"].fillna("").astype(str).str.contains("missense", case=False, na=False)]
    genes = sorted(set(mis["gene_symbol"].dropna().astype(str)) - {""})
    return genes


def _resolve_accessions(genes: list[str], uniprot_index_path: Path) -> dict[str, str]:
    """gene_symbol -> canonical UniProt accession, via the local reviewed parquet (1:1)."""
    up = pd.read_parquet(uniprot_index_path, columns=["gene_symbol", "uniprot_id"])
    up = up.dropna(subset=["gene_symbol", "uniprot_id"])
    gmap = dict(zip(up["gene_symbol"].astype(str), up["uniprot_id"].astype(str)))
    return {g: gmap[g] for g in genes if g in gmap}


def _load_acc_sequences(uniprot_index_path: Path) -> dict[str, str]:
    """accession -> canonical UniProt sequence, from the local reviewed parquet.

    Used to select the AlphaFold record whose sequence matches canonical (see
    _resolve_cif_url); isoform structures have divergent residue numbering and
    must never be attached to canonical protein_pos.
    """
    up = pd.read_parquet(uniprot_index_path, columns=["uniprot_id", "sequence"])
    up = up.dropna(subset=["uniprot_id", "sequence"])
    return dict(zip(up["uniprot_id"].astype(str), up["sequence"].astype(str)))


def _resolve_cif_url(accession: str, canonical_seq: str) -> Optional[str]:
    """Resolve the current-version AlphaFold cif URL via the prediction API.

    The per-file version suffix (model_v4 -> model_v6 -> ...) changes with each
    AlphaFold DB release, so hard-coding it silently breaks fetching. The
    prediction API always reports the current cifUrl for an accession; we read it
    rather than templating a version we would have to chase. Returns None on any
    failure (unreachable API, non-200, unparseable JSON, missing field) so the
    caller counts the gene as a miss and the coverage gate flags systematic loss
    loudly instead of papering over it with a guessed version.
    """
    try:
        resp = requests.get(
            ALPHAFOLD_PREDICTION_API.format(accession=accession),
            timeout=_REQUEST_TIMEOUT,
        )
    except Exception as exc:
        logger.debug("prediction API failed for %s: %s", accession, exc)
        return None
    if not resp.ok:
        return None
    try:
        data = resp.json()
    except Exception as exc:
        logger.debug("prediction API non-JSON for %s: %s", accession, exc)
        return None
    recs = data if isinstance(data, list) else [data]
    # Select the record whose UniProt sequence EXACTLY equals our canonical index
    # sequence. AlphaFold DB returns one record per isoform (entryId AF-{acc}-{N}-F1)
    # whose residue numbering follows THAT isoform; attaching an isoform structure to
    # canonical protein_pos would silently mis-number features. If no record matches
    # the canonical sequence (giants over the AFDB length ceiling, isoform-only
    # entries), return None so the gene is a documented coverage miss -- never a
    # partial/isoform substitute. Verified on AARS1/ABCB1 (base) and DYST/SYNE1 (None).
    if not canonical_seq:
        return None
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        if (rec.get("uniprotSequence") or "") == canonical_seq:
            cif_url = rec.get("cifUrl")
            if isinstance(cif_url, str) and cif_url:
                return cif_url
    return None


def _download_cif(accession: str, cache_dir: Path, canonical_seq: str) -> Optional[Path]:
    """Download+cache the current-version AlphaFold CIF; reuse if present.

    Resolves the versioned cif URL from the prediction API, saves under the
    server's own filename (so a v6 payload is never stored under a v4 name), and
    rejects any non-CIF payload so an error page can never be parsed as a
    structure. Returns the cached path, or None on any failure.
    """
    cached = [f for f in cache_dir.glob(f"AF-{accession}-F1-model_v*.cif")
              if f.stat().st_size > 0]
    if cached:
        def _ver(p: Path) -> int:
            stem = p.stem.rsplit("_v", 1)
            return int(stem[1]) if len(stem) == 2 and stem[1].isdigit() else -1
        return max(cached, key=_ver)
    cif_url = _resolve_cif_url(accession, canonical_seq)
    if not cif_url:
        return None
    try:
        resp = requests.get(cif_url, timeout=_REQUEST_TIMEOUT)
    except Exception as exc:
        logger.debug("download failed for %s: %s", accession, exc)
        return None
    if not resp.ok or not resp.text:
        return None
    text = resp.text
    if not text.lstrip().startswith("data_") or "_atom_site" not in text:
        logger.debug("payload for %s is not a CIF (%d bytes); rejecting",
                     accession, len(text))
        return None
    cache_file = cache_dir / cif_url.rsplit("/", 1)[-1]
    cache_file.write_text(text, encoding="utf-8")
    time.sleep(_POLITE_DELAY_S)
    return cache_file


def _fetch_active_sites(accession: str, cache_dir: Path) -> list[int]:
    """UniProt active/binding-site residue positions (cached JSON). [] on failure."""
    cache_file = cache_dir / f"uniprot_features_{accession}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text()).get("active_sites", [])
        except Exception:
            pass
    sites: list[int] = []
    try:
        resp = requests.get(
            UNIPROT_FEATURES_API.format(accession=accession),
            timeout=_REQUEST_TIMEOUT, headers={"Accept": "application/json"},
        )
        if resp.ok:
            for feat in resp.json().get("features", []):
                if feat.get("type", "").upper() in ("ACT_SITE", "BINDING"):
                    pos = feat.get("begin") or feat.get("location", {}).get("start", {}).get("value")
                    if pos:
                        try:
                            sites.append(int(pos))
                        except (ValueError, TypeError):
                            pass
        cache_file.write_text(json.dumps({"active_sites": sites}))
        time.sleep(_POLITE_DELAY_S)
    except Exception as exc:
        logger.debug("feature fetch failed for %s: %s", accession, exc)
    return sites


def _dist_to_active(ca: dict, active_sites: list[int]) -> dict[int, float]:
    """3-D C-alpha distance (Angstrom) from each residue to the nearest active site."""
    import math
    out: dict[int, float] = {}
    site_xyz = [ca[s] for s in active_sites if s in ca]
    for seq, (x, y, z) in ca.items():
        if not site_xyz:
            out[seq] = aff.DEFAULT_DIST_ACTIVE
            continue
        best = min(
            math.sqrt((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2)
            for (sx, sy, sz) in site_xyz
        )
        out[seq] = round(best, 3)
    return out


def _extract_one(accession: str, cif_path: Path, active_sites: list[int]) -> list[dict]:
    """Extract per-residue feature rows for one structure. [] if unparseable."""
    cif_text = cif_path.read_text(encoding="utf-8", errors="replace")
    try:
        atoms = aff.parse_atom_site(cif_text)
    except aff.CIFParseError as exc:
        logger.warning("%s: unparseable CIF (%s) -- skipped.", accession, exc)
        return []
    plddt = aff.per_residue_plddt(atoms)
    ca = aff.per_residue_ca(atoms)
    ss = aff.per_residue_secondary_structure(cif_text, atoms)
    try:
        rsa = aff.per_residue_rsa(atoms)
    except aff.CIFParseError as exc:
        # RSA fail-loud guard tripped: real geometry failure. Skip this structure
        # loudly rather than write suspect RSA values.
        logger.warning("%s: RSA geometry failure (%s) -- structure skipped.", accession, exc)
        return []
    dist = _dist_to_active(ca, active_sites)

    rows = []
    for seq in sorted(plddt):
        rows.append({
            "uniprot_accession": accession,
            "residue_pos": int(seq),
            "plddt": float(plddt[seq]),
            "rsa": float(rsa.get(seq, aff.DEFAULT_RSA)),
            "ss": int(ss.get(seq, 0)),  # residues not in _struct_conf -> loop
            "dist_active": float(dist.get(seq, aff.DEFAULT_DIST_ACTIVE)),
        })
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build the AlphaFold cohort feature parquet.")
    ap.add_argument("--cohort", default="data/processed/clinvar_grch38_clean.parquet")
    ap.add_argument("--uniprot-index", default="data/external/uniprot/uniprot_human_reviewed.parquet")
    ap.add_argument("--out", default="data/external/alphafold/alphafold_cohort.parquet")
    ap.add_argument("--cache-dir", default="data/raw/cache/alphafold")
    ap.add_argument("--max-genes", type=int, default=None, help="cap number of genes (resumable chunks)")
    ap.add_argument("--genes-from", default=None, help="file with one gene symbol per line")
    ap.add_argument("--audit", action="store_true", help="report resolution coverage only; no download")
    args = ap.parse_args(argv)

    cohort_path = Path(args.cohort)
    uniprot_index_path = Path(args.uniprot_index)
    out_path = Path(args.out)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.genes_from:
        genes = [g.strip() for g in Path(args.genes_from).read_text().splitlines() if g.strip()]
    else:
        genes = _load_cohort_genes(cohort_path)
    logger.info("cohort missense genes: %d", len(genes))

    acc_map = _resolve_accessions(genes, uniprot_index_path)
    acc_to_seq = _load_acc_sequences(uniprot_index_path)
    coverage: dict[str, dict] = {}
    logger.info("resolved to accession: %d ; unresolvable: %d", len(acc_map), len(genes) - len(acc_map))

    if args.max_genes is not None:
        limited = dict(list(acc_map.items())[: args.max_genes])
        logger.info("--max-genes: limiting to %d genes", len(limited))
        acc_map = limited

    if args.audit:
        logger.info("AUDIT MODE: %d genes resolve to %d distinct accessions; no download performed.",
                    len(acc_map), len(set(acc_map.values())))
        return 0

    all_rows: list[dict] = []
    n_done = 0
    n_struct = 0
    n_total = len(acc_map)
    for _i_gene, (gene, acc) in enumerate(acc_map.items(), 1):
        logger.info("gene %d/%d: %s (%s)", _i_gene, n_total, gene, acc)
        # mid-run disk guard
        free = _free_gb(cache_dir)
        if free < _MIN_FREE_GB:
            logger.error("ABORT: free space %.1f GB < %.1f GB threshold on cache volume. "
                         "Cached progress retained; re-run to resume.", free, _MIN_FREE_GB)
            return 3
        canonical_seq = acc_to_seq.get(acc, "")
        cif_path = _download_cif(acc, cache_dir, canonical_seq)
        n_done += 1
        if cif_path is None:
            coverage[gene] = {"accession": acc, "status": "no_canonical_structure",
                              "n_variants": None}
            continue
        coverage[gene] = {"accession": acc, "status": "ok",
                          "cif": cif_path.name}
        active_sites = _fetch_active_sites(acc, cache_dir)
        rows = _extract_one(acc, cif_path, active_sites)
        if rows:
            all_rows.extend(rows)
            n_struct += 1
        if n_done % 250 == 0:
            logger.info("progress: %d/%d genes processed, %d structures, %d residue rows, free=%.1fGB",
                        n_done, len(acc_map), n_struct, len(all_rows), _free_gb(cache_dir))

    if not all_rows:
        logger.error("ABORT: zero residue features extracted from %d genes. "
                     "This is a failure, not an empty success.", len(acc_map))
        return 4

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["uniprot_accession", "residue_pos"])
    # final integrity checks before writing
    assert df["plddt"].between(0, 100).all(), "pLDDT out of [0,100]"
    assert df["rsa"].between(0, 1).all(), "RSA out of [0,1] (should be clamped)"
    assert df["ss"].isin([0, 1, 2]).all(), "ss not in {0,1,2}"
    df.to_parquet(out_path, index=False)
    # Coverage report: which genes got a canonical structure vs. were unusable
    # (giants over the AFDB length ceiling, isoform-only entries, or absent). Written
    # next to the cohort parquet so any researcher can audit the structural gap.
    import json as _json
    n_ok = sum(1 for v in coverage.values() if v.get("status") == "ok")
    n_miss = len(coverage) - n_ok
    coverage_path = out_path.parent / "alphafold_coverage.json"
    with open(coverage_path, "w", encoding="utf-8", newline="\n") as _fh:
        _json.dump({"n_genes": len(coverage), "n_canonical_ok": n_ok,
                    "n_unusable": n_miss, "genes": coverage}, _fh, indent=2)
    usable_frac = n_ok / max(1, len(acc_map))
    logger.info("WROTE %s : %d residue rows across %d structures (%d/%d genes usable, %.1f%%).",
                out_path, len(df), n_struct, n_ok, len(acc_map), 100.0 * usable_frac)
    logger.info("WROTE %s : coverage report (%d unusable genes -> sentinel).",
                coverage_path, n_miss)
    # Hard gate: dormant in normal operation (~98% usable observed). Trips only if
    # canonical selection catastrophically regresses (e.g. an API/schema change makes
    # matching fail wholesale), so a broken run cannot masquerade as success.
    _MIN_USABLE_FRACTION = 0.90
    if usable_frac < _MIN_USABLE_FRACTION:
        logger.error("ABORT: usable-gene fraction %.3f < %.2f threshold -- canonical "
                     "selection may be broken. Parquet + coverage written for inspection.",
                     usable_frac, _MIN_USABLE_FRACTION)
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
