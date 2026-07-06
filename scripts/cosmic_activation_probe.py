#!/usr/bin/env python
"""cosmic_activation_probe.py -- activate + verify the COSMIC connector on REAL data.

First run parses the full CMC TSV (~315 MB gz) and writes the parquet sidecar cache
next to it (one-time, minutes + a few GB RAM for the 8-column stream); later runs load
the cache in seconds. Reports match COVERAGE against a cohort slice + recurrence/tier
spread. COSMIC is somatic, so partial overlap with germline ClinVar is EXPECTED -- this
probe proves the join works and produces real values, not that coverage is high.

    python scripts/cosmic_activation_probe.py            # 50k cohort slice
    python scripts/cosmic_activation_probe.py 200000     # larger slice
"""
from __future__ import annotations

import collections
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

COHORT = Path("data/processed/clinvar_grch38_clean.parquet")
CMC = Path("data/external/cosmic/CancerMutationCensus_AllData_v104_GRCh37.tsv.gz")


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    if not COHORT.exists():
        print(f"FAIL: cohort not found: {COHORT}"); return 2
    if not CMC.exists():
        print(f"FAIL: CMC TSV not found: {CMC}"); return 2

    from genomic_variant_classifier.data.real_data_prep import AnnotationConfig
    from genomic_variant_classifier.data.connectors.connector_cosmic import CosmicCmcConnector

    ac = AnnotationConfig(cosmic_path=CMC)
    print("AnnotationConfig.cosmic_path =", ac.cosmic_path)
    conn = CosmicCmcConnector(cosmic_path=ac.cosmic_path)

    df = pd.read_parquet(COHORT, columns=["chrom", "pos", "ref", "alt"]).head(n)
    print(f"cohort slice: {len(df)} variants")

    t = time.time()
    out = conn.annotate_dataframe(df)          # first call: parse + cache CMC
    dt = time.time() - t
    print(f"annotated in {dt:.1f}s (includes one-time CMC parse+cache on first run)")

    rec = pd.to_numeric(out["cosmic_recurrence"], errors="coerce").to_numpy()
    tier = pd.to_numeric(out["cosmic_sig_tier"], errors="coerce").to_numpy()
    hit = rec > 0
    print(f"coverage: {int(hit.sum())}/{len(df)} ({100*hit.mean():.2f}%) have cosmic_recurrence > 0")
    if hit.any():
        v = rec[hit]
        print(f"recurrence (nonzero) min/median/max: {v.min():.5f} / {np.median(v):.5f} / {v.max():.5f}")
    print("cosmic_sig_tier distribution:", dict(collections.Counter(np.round(tier, 3).tolist())))

    cache = CMC.parent / "cosmic_cmc_grch38_index.parquet"
    print("cache written:", cache.exists(), f"({cache.stat().st_size/1e6:.1f} MB)" if cache.exists() else "")
    print("NOTE: COSMIC is somatic; modest overlap with germline ClinVar is EXPECTED, not a bug.")
    print("VERDICT: connector active" if hit.any() else
          "VERDICT: 0 matches -- check GRCh38 key / that the CMC file has GRCh38 coords populated")
    return 0 if hit.any() else 1


if __name__ == "__main__":
    sys.exit(main())
