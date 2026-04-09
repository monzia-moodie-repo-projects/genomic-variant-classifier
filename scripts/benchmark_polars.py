import pandas as pd
import polars as pl
import time
import os

# Benchmark the gnomAD constraint join — the heaviest pandas operation
# Uses actual processed data if available, synthetic otherwise

def make_synthetic(n=500_000):
    import numpy as np
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "gene_symbol": rng.choice([f"GENE{i}" for i in range(5000)], n),
        "chrom": rng.choice([str(i) for i in range(1,23)]+["X","Y"], n),
        "pos": rng.integers(1, 250_000_000, n),
        "af_raw": rng.random(n).astype("float32"),
    })

def make_constraint(n=20_000):
    import numpy as np
    rng = np.random.default_rng(0)
    genes = [f"GENE{i}" for i in range(n)]
    return pd.DataFrame({
        "gene": genes,
        "loeuf": rng.random(n).astype("float32"),
        "pli_score": rng.random(n).astype("float32"),
        "syn_z": rng.uniform(-5,5,n).astype("float32"),
        "mis_z": rng.uniform(-5,5,n).astype("float32"),
    })

print("Generating synthetic data (500K variants, 20K genes)...")
variants_pd = make_synthetic()
constraint_pd = make_constraint()

# --- Pandas benchmark ---
t0 = time.perf_counter()
for _ in range(3):
    result_pd = variants_pd.merge(
        constraint_pd, left_on="gene_symbol", right_on="gene", how="left"
    )
pandas_time = (time.perf_counter() - t0) / 3
print(f"Pandas merge (3-run avg): {pandas_time*1000:.1f} ms | {len(result_pd):,} rows")

# --- Polars benchmark ---
try:
    variants_pl = pl.from_pandas(variants_pd)
    constraint_pl = pl.from_pandas(constraint_pd)

    t0 = time.perf_counter()
    for _ in range(3):
        result_pl = variants_pl.join(
            constraint_pl, left_on="gene_symbol", right_on="gene", how="left"
        )
    polars_time = (time.perf_counter() - t0) / 3
    print(f"Polars join  (3-run avg): {polars_time*1000:.1f} ms | {len(result_pl):,} rows")
    print(f"Speedup: {pandas_time/polars_time:.1f}x")
except ImportError:
    print("Polars not installed — skipping Polars benchmark")
    print("Install with: pip install polars")
