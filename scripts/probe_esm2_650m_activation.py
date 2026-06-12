#!/usr/bin/env python3
"""probe_esm2_650m_activation.py -- CPU plumbing proof for ESM-2 650M activation.

Proves, WITHOUT GPU, that the protein-coords -> ESM-2 path produces non-zero
esm2_delta_norm and esm2_llr on a real cohort sample, BEFORE committing a Run-16
regen to the 4090. Read-only w.r.t. the repo: the only thing written is an ephemeral
SQLite cache under --out-dir.

Reports, in order:
  1. that the 650M snapshot blob is real (resolves the HF symlink 0-byte ambiguity);
  2. protein-coord triple coverage (protein_pos/wt_aa/mut_aa populated);
  3. esm2_delta_norm and esm2_llr non-zero fractions + ranges;
  4. the LLR wt-vs-sequence mismatch count.

Exit 0 = GREEN (both features non-zero; regen justified), 2 = RED (still zero/absent),
3 = usage/environment error.

Author: Monzia Moodie.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _fail(msg: str) -> "SystemExit":
    print(f"ABORT: {msg}", file=sys.stderr)
    raise SystemExit(3)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CPU proof that ESM-2 650M activates non-zero on a cohort sample (no GPU)."
    )
    p.add_argument("--cohort", type=Path, default=Path("data/processed/clinvar_smoke.parquet"),
                   help="Cohort parquet with chrom/pos/ref/alt/gene_symbol (and ideally consequence/is_missense).")
    p.add_argument("--alphamissense-file", type=Path,
                   default=Path("data/external/alphamissense/AlphaMissense_hg38.tsv.gz"))
    p.add_argument("--am-cache-dir", type=Path, default=Path("data/external/alphamissense"))
    p.add_argument("--uniprot-index", type=Path,
                   default=Path("data/external/uniprot/uniprot_human_reviewed.parquet"))
    p.add_argument("--esm2-model", type=str, default="esm2_t33_650M_UR50D")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-sample", type=int, default=300,
                   help="Missense variants to score (CPU; keep small -- 650M forward passes are slow).")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/esm2_probe"))
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import pandas as pd

    # path pre-checks (fail loud, never silent)
    if not args.cohort.exists():
        _fail(f"cohort not found: {args.cohort}")
    if not args.uniprot_index.exists():
        _fail(f"uniprot index not found: {args.uniprot_index}")
    cache_idx = args.am_cache_dir / "alphamissense_protein_index.parquet"
    if not cache_idx.exists() and not args.alphamissense_file.exists():
        _fail(f"neither AlphaMissense cache ({cache_idx}) nor raw file ({args.alphamissense_file}) found")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Confirm the 650M snapshot blob is real (HF snapshot entries are symlinks;
    #         a 0-byte directory listing is the symlink, not the weights) ----
    print(f"[1/5] Verifying {args.esm2_model} snapshot ...")
    hub = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
    snapdir = hub / f"models--facebook--{args.esm2_model}" / "snapshots"
    safes = list(snapdir.glob("*/model.safetensors")) if snapdir.exists() else []
    if not safes:
        print(f"      WARN: no model.safetensors under {snapdir}; the connector will try to download.")
    else:
        real = Path(os.path.realpath(safes[0]))
        mb = real.stat().st_size / 1e6
        print(f"      model.safetensors blob = {mb:.0f} MB (expect ~2400 MB for 650M)")
        if mb < 100:
            _fail(f"model.safetensors blob only {mb:.0f} MB -- incomplete pull; "
                  f"run: huggingface-cli download facebook/{args.esm2_model}")

    # ---- 2. Load + sample the cohort (missense only) ----
    print(f"[2/5] Loading cohort {args.cohort} ...")
    df = pd.read_parquet(args.cohort)
    need = {"chrom", "pos", "ref", "alt", "gene_symbol"}
    missing = need - set(df.columns)
    if missing:
        _fail(f"cohort missing columns {sorted(missing)}; present (first 40): {sorted(df.columns)[:40]}")
    if "is_missense" in df.columns:
        miss = df[df["is_missense"].astype("float").fillna(0).astype(bool)]
    elif "consequence" in df.columns:
        miss = df[df["consequence"].astype(str).str.contains("missense", case=False, na=False)]
    else:
        print("      WARN: no is_missense/consequence column; treating all rows as candidates.")
        miss = df
    if len(miss) == 0:
        _fail("no missense candidates in cohort")
    sample = miss.sample(n=min(args.n_sample, len(miss)), random_state=args.seed).reset_index(drop=True)
    print(f"      cohort rows={len(df)}  missense={len(miss)}  sampled={len(sample)}")

    # ---- 3. protein-coords: populate protein_pos / wt_aa / mut_aa ----
    print("[3/5] ProteinCoordConnector.annotate_dataframe ...")
    from genomic_variant_classifier.data.protein_coords import ProteinCoordConnector
    pc = ProteinCoordConnector(alphamissense_file=args.alphamissense_file, cache_dir=args.am_cache_dir)
    sample = pc.annotate_dataframe(sample)
    for col in ("protein_pos", "wt_aa", "mut_aa"):
        if col not in sample.columns:
            _fail(f"protein_coords did not add '{col}' (warn-and-stub? verify AlphaMissense source/cache)")
    n_cov = int(sample["protein_pos"].notna().sum())
    cov = n_cov / len(sample)
    print(f"      protein-coord triple coverage: {cov:.3f}  ({n_cov}/{len(sample)} sampled missense)")
    if n_cov == 0:
        _fail("zero protein-coord coverage -- ESM-2 cannot score. "
              "Check cohort (chrom/pos/ref/alt) key normalisation vs the AlphaMissense cache.")

    # ---- 4. ESM-2: delta_norm + signed LLR (this loads the model; the real completeness test) ----
    print(f"[4/5] ESM2Connector (delta_norm + LLR), model={args.esm2_model}, device={args.device} ...")
    from genomic_variant_classifier.data.esm2 import ESM2Connector
    try:
        esm2 = ESM2Connector(
            model_name=args.esm2_model,
            cache_path=args.out_dir / "esm2_probe_cache.sqlite",
            uniprot_index_path=args.uniprot_index,
            device=args.device,
        )
        sample = esm2.annotate_dataframe(sample)
        sample = esm2.annotate_llr(sample)
    except Exception as exc:  # noqa: BLE001 -- surface the real failure to the operator
        _fail(f"ESM-2 scoring failed ({type(exc).__name__}: {exc})")

    # ---- 5. Report ----
    print("[5/5] Results")

    def stats(col):
        if col not in sample.columns:
            return None
        s = pd.to_numeric(sample[col], errors="coerce").fillna(0.0)
        return float((s != 0).mean()), float(s.min()), float(s.max())

    dn = stats("esm2_delta_norm")
    llr = stats("esm2_llr")
    mism = getattr(esm2, "_llr_n_mismatch", "n/a")

    if dn:
        print(f"      esm2_delta_norm: nonzero_frac={dn[0]:.3f}  min={dn[1]:.3f}  max={dn[2]:.3f}")
    else:
        print("      esm2_delta_norm: ABSENT")
    if llr:
        print(f"      esm2_llr:        nonzero_frac={llr[0]:.3f}  min={llr[1]:.3f}  max={llr[2]:.3f}")
    else:
        print("      esm2_llr:        ABSENT")
    print(f"      LLR wt-vs-sequence mismatches: {mism}")

    ok = bool(dn) and dn[0] > 0 and bool(llr) and llr[0] > 0
    print("\nRESULT:", "GREEN -- ESM-2 650M activates non-zero on real data; the regen is justified."
          if ok else
          "RED -- esm2 features still zero/absent; investigate coverage / sequence index / model before regen.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
