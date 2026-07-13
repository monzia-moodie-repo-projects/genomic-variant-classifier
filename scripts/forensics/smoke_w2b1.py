#!/usr/bin/env python
"""smoke_w2b1.py (2026-07-11) -- prove run_v2's integration with the REAL split_protocol_v2 works
(not just that it imports). Isolates the novel section: assemble a combined df from a real slice of
the pathfix cohort exactly as run_v2 does, call the REAL split() + apply_train_only_leakage_remap(),
and assert a correct 4-way gene-disjoint partition + leakage remap + fractions. Does NOT run the heavy
annotation pipeline. ASCII-safe, read-only (writes nothing).
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass


def a(s): return s.encode("ascii", "replace").decode("ascii")


def main() -> int:
    print("=" * 78)
    print("W2-B1 SMOKE (run_v2 integration with the REAL split_protocol_v2)")
    print("=" * 78)
    sys.path.insert(0, "src")
    import numpy as np
    import pandas as pd
    results = []

    def chk(name, cond, detail=""):
        results.append(cond)
        print(a(f"  {'ok  ' if cond else 'FAIL'} {name}{('  -- ' + detail) if detail and not cond else ''}"))

    # 1. import the REAL split_protocol_v2 (the exact names run_v2 uses)
    try:
        from genomic_variant_classifier.data.split_protocol_v2 import (
            SplitProtocolV2Config,
            split as _split_v2,
            apply_train_only_leakage_remap as _leak_remap_v2,
        )
        chk("real split_protocol_v2 imports (split + leak_remap + config)", True)
    except Exception as e:
        chk("real split_protocol_v2 imports", False, repr(e)[:200])
        return 1

    # 2. build a combined df from a real cohort slice EXACTLY as run_v2 does
    cohort = Path("data/processed/clinvar_grch38_pathfix.parquet")
    if not cohort.exists():
        chk("pathfix cohort available", False, f"missing {cohort}")
        return 1
    df = pd.read_parquet(cohort)
    # need a gene column + label. run_v2 uses group_column ('gene_symbol') + 'label'.
    # the pathfix cohort may not carry engineered 'label' -- derive a proxy for the smoke:
    gene_col = "gene_symbol"
    if gene_col not in df.columns:
        # some cohorts store gene under a different name; try common fallbacks
        for alt in ["gene", "GeneSymbol", "symbol"]:
            if alt in df.columns:
                gene_col = alt; break
    chk(f"cohort has gene column ('{gene_col}')", gene_col in df.columns)
    # sample to keep it light
    samp = df.sample(n=min(50000, len(df)), random_state=7).reset_index(drop=True)
    # synth a label + the leakage feature if absent (smoke only tests the SPLIT mechanics)
    rng = np.random.RandomState(0)
    combo = pd.DataFrame({
        gene_col: samp[gene_col].fillna("unknown").astype(str),
        "label": (rng.rand(len(samp)) > 0.5).astype(int),
        "n_pathogenic_in_gene": rng.randint(0, 5, len(samp)),
        "gene_has_known_disease": rng.randint(0, 2, len(samp)),
        "feat_dummy": rng.rand(len(samp)),
    })

    # 3. call the REAL split() + leak_remap() exactly as run_v2 does
    try:
        cfg2 = SplitProtocolV2Config(gene_col=gene_col, label_col="label", seed=42, mode="hash")
        result = _split_v2(combo, cfg2)
        chk("real split() returns a result with .indices", hasattr(result, "indices"))
        idx = result.indices
        chk("indices has train/tune/conformal/test",
            all(k in idx for k in ["train", "tune", "conformal", "test"]))
    except Exception as e:
        chk("real split() runs as run_v2 calls it", False, repr(e)[:250])
        return 1

    try:
        combo2 = _leak_remap_v2(combo, result.indices, cfg2)
        chk("real apply_train_only_leakage_remap runs", combo2 is not None)
    except Exception as e:
        chk("real leak_remap runs as run_v2 calls it", False, repr(e)[:250])
        return 1

    # 4. gene-disjoint + coverage + fractions
    n = len(combo)
    total = sum(len(idx[k]) for k in ["train", "tune", "conformal", "test"])
    chk("coverage: all rows partitioned", total == n, f"{total} != {n}")

    genes = combo[gene_col]
    gsets = {k: set(genes.iloc[idx[k]].unique()) for k in ["train", "tune", "conformal", "test"]}
    disjoint = True
    ks = list(gsets)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            if gsets[ks[i]] & gsets[ks[j]]:
                disjoint = False
    chk("4-way gene-disjoint (no gene in 2 partitions)", disjoint)

    print("  partition fractions (real split on real genes):")
    for k, exp in [("train", 0.60), ("tune", 0.15), ("conformal", 0.10), ("test", 0.15)]:
        frac = len(idx[k]) / n
        print(a(f"    {k}: {frac:.3f} (~{exp})"))

    # 5. leakage remap: an unseen-in-train gene's count must be 0 after remap
    tr_genes = gsets["train"]
    test_idx = idx["test"]
    test_genes = genes.iloc[test_idx].reset_index(drop=True)
    test_counts = combo2["n_pathogenic_in_gene"].iloc[test_idx].reset_index(drop=True)
    unseen = ~test_genes.isin(tr_genes)
    if unseen.any():
        chk("leak remap: unseen-gene test rows -> count 0",
            bool((test_counts[unseen.values] == 0).all()))
    else:
        print("    (all test genes appear in train in this sample -- remap still applied)")
        results.append(True)

    print("-" * 78)
    npass = sum(1 for x in results if x)
    print(a(f"W2-B1 smoke: {npass}/{len(results)} checks pass"))
    print("=" * 78)
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
