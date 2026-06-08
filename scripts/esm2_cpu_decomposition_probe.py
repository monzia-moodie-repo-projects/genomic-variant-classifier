"""
scripts/esm2_cpu_decomposition_probe.py
=======================================
Local CPU decomposition probe for ESM-2 annotation cost. No GPU. No cloud. No training.

WHY THIS EXISTS
---------------
The ESM-2 connector (src/genomic_variant_classifier/data/esm2.py) is CPU-only:
ESM2Connector.__init__ has no `device`/`batch_size`, the model is never moved
to a GPU, and the inputs are never moved either. A "GPU rate probe" of the
current code would rent a GPU and run the model on the host CPU with the GPU
idle -- paying to measure the wrong thing.

The connector's cost model (from the code): each missense variant is two forward
passes over a +/-21-residue context window (<= 43 tokens) through an 8M-param
model -- tiny -- plus one UniProt REST fetch per *new gene* (network) and a
SQLite cache lookup per window. The 1213.9 ms/variant first-pass figure is
therefore almost certainly network + per-call overhead, not matmul. This probe
MEASURES that, so we know whether the lever is GPU forward-compute or
(more likely) UniProt pre-fetch + batching on CPU.

WHAT IT MEASURES (three quantities, all on CPU)
-----------------------------------------------
  [1] t_fetch_per_gene   UniProt REST latency, against a FRESH cache (real network)
  [2] t_cold_per_variant annotate with sequences cached but embeddings EMPTY
                         (forces forward passes) -> forward + overhead
  [3] t_warm_per_variant annotate again, embeddings now cached (no forward)
                         -> pure overhead + cache read
      forward compute per variant ~= [2] - [3]

Then extrapolates to the full missense cohort using the ACTUAL distinct-gene
count read from the cohort (not a guess).

SAFETY / ISOLATION
------------------
- Uses private temp SQLite caches; NEVER touches data/raw/cache/esm2_cache.sqlite.
- Read-only on the ClinVar parquet and AlphaMissense file.
- Model load + first forward is warmed up ONCE and excluded from every timed section.
- Dry-run sanity on three known missense variants (BRCA1/TP53/PTEN) BEFORE any
  cohort timing; if the connector cannot produce signal here (no backend, no
  network, weights unavailable) it exits with a clear message and reports NO
  misleading timings.
- Temp dirs are left on disk (printed) rather than deleted, to avoid Windows
  file-lock errors on open SQLite handles; they live under the OS temp dir.

USAGE (run from the repo root with .venv312 active)
---------------------------------------------------
  python scripts/esm2_cpu_decomposition_probe.py \
      --alphamissense data/external/alphamissense/AlphaMissense_hg38.tsv.gz \
      --clinvar       data/processed/clinvar_grch38.parquet \
      --n 300

EXIT CODES
----------
  0  probe completed and reported a decomposition
  1  cohort/sanity data problem (no missense, no candidates, all-zero signal)
  2  environment cannot run the real backend (stub mode / weights unavailable)
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Cohort-audit constant; used only as a fallback label. The real extrapolation
# below is grounded in counts read from the cohort at run time.
_TOTAL_MISSENSE_FALLBACK = 2_488_903


def _ms(seconds: float) -> str:
    return f"{1000.0 * seconds:.1f} ms"


def _hours(seconds: float) -> str:
    return f"{seconds / 3600.0:.2f} h"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ESM-2 CPU cost decomposition probe")
    ap.add_argument("--alphamissense", required=True)
    ap.add_argument("--clinvar", required=True)
    ap.add_argument("--n", type=int, default=300,
                    help="missense rows to sample for timing (>=200 recommended; default 300)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    # Imports against the VERIFIED API (esm2.py lines 268/294/345/375/387;
    # proof-of-life ProteinCoordConnector(alphamissense_file=, cache_dir=)).
    from genomic_variant_classifier.data import esm2 as esm2_mod
    from genomic_variant_classifier.data.esm2 import ESM2Connector
    from genomic_variant_classifier.data.protein_coords import ProteinCoordConnector

    if esm2_mod._BACKEND is None:
        print("ESM-2 backend is None (transformers+torch / fair-esm not importable).")
        print("This probe needs the real backend to measure forward cost. STOP.")
        return 2

    ctx = esm2_mod._CONTEXT_WINDOW
    model_name = os.environ.get("ESM2_MODEL_NAME", "esm2_t6_8M_UR50D")
    print(f"ESM-2 backend : {esm2_mod._BACKEND}")
    print(f"model         : {model_name}")
    print(f"context window: +/-{ctx} residues (<= {2 * ctx + 1} tokens/forward, 2 forwards/variant)")

    # ---- STEP 0: warm up model + dry-run sanity on 3 known variants --------
    sanity_cache = Path(tempfile.mkdtemp(prefix="esm2_probe_sanity_")) / "cache.sqlite"
    sanity = ESM2Connector(cache_path=sanity_cache)

    t0 = time.time()
    warm_emb = esm2_mod._embed_sequence(
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", sanity.model_name, sanity._get_conn()
    )
    t_model_load = time.time() - t0
    if warm_emb is None:
        print(f"model warmup FAILED (weights could not load/run here). STOP. "
              f"({t_model_load:.1f}s)")
        return 2
    print(f"\nmodel load + first forward: {t_model_load:.1f}s "
          f"(one-time; excluded from all per-variant timings)")

    known = pd.DataFrame({
        "variant_id": ["v1", "v2", "v3"],
        "gene_symbol": ["BRCA1", "TP53", "PTEN"],
        "protein_pos": [1699, 175, 129],
        "wt_aa": ["R", "R", "G"],
        "mut_aa": ["Q", "H", "R"],
        "is_missense": [True, True, True],
    })
    d_known = sanity.annotate_dataframe(known)["esm2_delta_norm"].to_numpy(dtype=float)
    print(f"sanity deltas (BRCA1 R1699Q / TP53 R175H / PTEN G129R): "
          f"{np.round(d_known, 4).tolist()}")
    if not (d_known > 0).any():
        print("SANITY FAIL: all-zero on known variants (network/UniProt/weights issue). "
              "Refusing to report timings on a broken setup. STOP.")
        return 1

    # ---- cohort sample -----------------------------------------------------
    df = pd.read_parquet(
        args.clinvar, columns=["chrom", "pos", "ref", "alt", "gene_symbol", "consequence"]
    )
    miss = df[df["consequence"].fillna("").str.contains("missense", case=False)]
    if miss.empty:
        print("no missense rows in cohort -- STOP.")
        return 1
    n_missense_full = int(len(miss))
    n_genes_full = int(miss["gene_symbol"].dropna().nunique())
    print(f"\ncohort missense: {n_missense_full:,} rows across {n_genes_full:,} distinct genes")

    sample = miss.sample(n=min(args.n, len(miss)), random_state=args.seed).reset_index(drop=True)
    pc = ProteinCoordConnector(alphamissense_file=args.alphamissense, cache_dir=None)
    sample = pc.annotate_dataframe(sample)
    sample["is_missense"] = 1
    # Match esm2.py's internal candidate filter (line 412) exactly, so the
    # per-variant denominator equals the count the connector actually processes.
    have = (
        sample["protein_pos"].notna()
        & sample["wt_aa"].notna()
        & sample["mut_aa"].notna()
        & sample["gene_symbol"].notna()
    )
    cand = sample[have].copy()
    if cand.empty:
        print("no candidates with protein coords + gene_symbol -- STOP.")
        return 1
    n_var = len(cand)
    n_gene_sample = int(cand["gene_symbol"].dropna().nunique())
    print(f"timing sample  : {n_var} candidate variants across {n_gene_sample} distinct genes")
    if n_var < 200:
        print("  NOTE: <200 variants -- per-variant timings will be noisier; "
              "rerun with a larger --n for a stable estimate.")

    # ---- STEP 1: UniProt fetch cost (FRESH cache => real network) ----------
    fetch_cache = Path(tempfile.mkdtemp(prefix="esm2_probe_fetch_")) / "cache.sqlite"
    fetcher = ESM2Connector(cache_path=fetch_cache)
    genes = sorted(cand["gene_symbol"].dropna().astype(str).unique())
    t0 = time.time()
    n_ok = sum(1 for g in genes if fetcher._get_sequence(g) is not None)
    t_fetch_total = time.time() - t0
    t_fetch_gene = t_fetch_total / max(len(genes), 1)
    print(f"\n[1] UniProt fetch : {n_ok}/{len(genes)} genes resolved in {t_fetch_total:.1f}s "
          f"-> {_ms(t_fetch_gene)}/gene (network, once per gene)")
    if n_ok == 0:
        print("    no genes resolved from UniProt -- network/UniProt issue. STOP.")
        return 1

    # ---- STEP 2: COLD annotate (seqs cached in fetch_cache, embeddings empty)
    cold = ESM2Connector(cache_path=fetch_cache)
    t0 = time.time()
    out_cold = cold.annotate_dataframe(cand)
    t_cold_total = time.time() - t0
    nz = int((out_cold["esm2_delta_norm"] > 0).sum())
    t_cold_var = t_cold_total / n_var
    print(f"[2] COLD annotate : {t_cold_total:.1f}s for {n_var} vars "
          f"-> {_ms(t_cold_var)}/variant (forward + overhead; seqs pre-cached) | nz={nz}/{n_var}")

    # ---- STEP 3: WARM annotate (embeddings now cached => no forward) --------
    warm = ESM2Connector(cache_path=fetch_cache)
    t0 = time.time()
    warm.annotate_dataframe(cand)
    t_warm_total = time.time() - t0
    t_warm_var = t_warm_total / n_var
    print(f"[3] WARM annotate : {t_warm_total:.1f}s for {n_var} vars "
          f"-> {_ms(t_warm_var)}/variant (overhead + cache read; no forward)")

    t_forward_var = max(t_cold_var - t_warm_var, 0.0)
    print(f"\n    => forward compute alone ~= {_ms(t_forward_var)}/variant  (cold - warm)")

    # ---- extrapolation (grounded in measured cohort counts) ----------------
    full_fetch = n_genes_full * t_fetch_gene
    full_cold = n_missense_full * t_cold_var
    full_forward = n_missense_full * t_forward_var
    full_total = full_fetch + full_cold
    print("\n--- full-cohort one-time annotate estimate ---")
    print(f"  UniProt fetch  (all {n_genes_full:,} genes, once) : {_hours(full_fetch)}")
    print(f"  annotate       (all {n_missense_full:,} variants) : {_hours(full_cold)}  "
          f"(forward + overhead; fetch excluded)")
    print(f"    of which forward compute only                 : {_hours(full_forward)}")
    print(f"  TOTAL one-time regen estimate                   : {_hours(full_total)}")

    # ---- interpretation guard ----------------------------------------------
    share_fwd = (t_forward_var / t_cold_var) if t_cold_var > 0 else 0.0
    print("\n--- interpretation ---")
    if share_fwd < 0.5:
        print(f"  Forward compute is ~{100 * share_fwd:.0f}% of per-variant cost: a GPU "
              "would help little. Lever = pre-fetch every UniProt sequence once + batch "
              "the forwards; the full regen is likely viable on CPU (no Vast.ai spend).")
    else:
        print(f"  Forward compute is ~{100 * share_fwd:.0f}% of per-variant cost: adding a "
              "device + batch path to esm2.py (with a numeric-equivalence test) is worth "
              "it; THEN a real GPU probe becomes meaningful.")
    print("  All timings are single-process CPU. Batching cuts per-call overhead; a warm "
          "SQLite embedding cache eliminates repeat forwards across reruns.")
    print(f"\n  (temp caches left at: {sanity_cache.parent} ; {fetch_cache.parent})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
