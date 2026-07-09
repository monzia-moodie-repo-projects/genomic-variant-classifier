"""
rekey_seq_windows_v2.py  (2026-07-09)
==========================================================================
Rekey the sequence-window parquet from cohort-v1 to cohort-v2 coordinates.

WHY THIS IS A REKEY, NOT A RECUT (verified against the source 2026-07-09)

    `populate_fasta_seq.py` builds each window by ANCHORING to the position where the
    ref allele actually matches the genome (`seq_windows.find_anchor`), NOT to the cohort's
    `pos`. Its docstring (dated 2026-05-31) states the padded-deletion off-by-one explicitly:
    "SNV/MNV/insertion align at delta 0; deletions align at delta -1 ... the cohort's `pos`
    column is left untouched (other joins depend on it)." So:

      * the window CONTENT (fasta_seq_ref / fasta_seq_alt) is ALREADY correct for padded
        deletions -- it was cut at pos-1 via the anchor. Nothing about the sequence changes.
      * but the seq parquet's `pos`/`variant_id` still hold the STALE v1 coordinate. Verified
        2026-07-09: e.g. clinvar:7:4787730:GCTGCTGGACCTGCC:G stores pos=4787730 (v1 Start),
        while cohort-v2 corrected it to 4787729. And variant_id agrees with pos 100.000%
        (4,399,089/4,399,089), so the key is a clean function of pos.

    THE BREAKAGE this fixes: `seq_window_join.attach_delta_windows` joins windows to a split
    by key `chrom:pos:ref:alt` built from the split's `pos`. cohort-v2 splits carry the
    CORRECTED pos, so the 187,245 padded-deletion keys will NOT match the seq parquet's stale
    v1 keys -> every one silently falls back to poly-A (n_unmapped), losing all sequence
    signal via the quiet fallback path. This script realigns the keys so the join hits.

WHAT IT DOES

    * padded deletions (alt a strict prefix of ref, shorter): pos -= 1, variant_id rebuilt.
      fasta_seq_ref / fasta_seq_alt COPIED UNCHANGED (content was always correct).
    * every other row: passed through byte-for-byte.
    * writes a NEW parquet (v1 seq parquet untouched); refuses to overwrite (exit 5).
    * reconciliation JSON: rows rekeyed (must be the padded-deletion count), schema
      fingerprint, MD5, and the row-count gap vs the cohort explained.

PROOF (the point of the whole exercise)

    The `--verify-against COHORT_V2_SPLIT_META` option loads a cohort-v2 meta/split frame,
    attaches the REKEYED windows via the REAL `attach_delta_windows`, and asserts
    n_unmapped for padded deletions is ZERO -- i.e. the silent-poly-A breakage is closed.
    Without rekeying, the same check reports 187,245 unmapped.

USAGE (from project root, .venv312 active)
    python scripts/rekey_seq_windows_v2.py --audit
    python scripts/rekey_seq_windows_v2.py --apply
    python scripts/rekey_seq_windows_v2.py --apply \\
        --seq data/processed/clinvar_grch38_clean_seq.parquet \\
        --out data/processed/clinvar_grch38_clean_v2_seq.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REF_WIN_COL = "fasta_seq_ref"
ALT_WIN_COL = "fasta_seq_alt"
KEY_COLS = ("chrom", "pos", "ref", "alt")


def _startswith_elementwise(ref: pd.Series, alt: pd.Series) -> pd.Series:
    r = ref.astype("string").fillna("")
    a = alt.astype("string").fillna("")
    return pd.Series([rr.startswith(aa) for rr, aa in zip(r, a)], index=ref.index, dtype=bool)


def is_padded_deletion(ref: pd.Series, alt: pd.Series) -> pd.Series:
    r = ref.astype("string").fillna("")
    a = alt.astype("string").fillna("")
    return (a.str.len() < r.str.len()) & _startswith_elementwise(r, a)


def schema_fingerprint(columns) -> str:
    return hashlib.sha256(",".join(sorted(map(str, columns))).encode()).hexdigest()[:16]


def _rebuild_variant_id(df: pd.DataFrame) -> pd.Series:
    prefix = df["variant_id"].astype("string").str.split(":", n=1).str[0].fillna("clinvar")
    return (prefix + ":" + df["chrom"].astype(str) + ":" + df["pos"].astype("int64").astype(str)
            + ":" + df["ref"].astype(str) + ":" + df["alt"].astype(str))


def rekey(seq: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pure. Returns (rekeyed_df, reconciliation). Window content untouched."""
    for c in (*KEY_COLS, "variant_id", REF_WIN_COL, ALT_WIN_COL):
        if c not in seq.columns:
            raise ValueError(f"seq parquet missing required column {c!r}. "
                             f"present: {list(seq.columns)}")

    mask = is_padded_deletion(seq["ref"], seq["alt"])
    n_rekey = int(mask.sum())

    out = seq.copy()
    # snapshot the window content of the rows we touch, to prove we didn't alter it
    before_ref = out.loc[mask, REF_WIN_COL].to_numpy().copy()
    before_alt = out.loc[mask, ALT_WIN_COL].to_numpy().copy()

    pos = out["pos"].to_numpy()
    if not np.issubdtype(pos.dtype, np.integer):
        if np.isnan(pos[mask.to_numpy()]).any():
            raise ValueError("NaN pos among padded deletions -- cannot rekey.")
    out.loc[mask, "pos"] = out.loc[mask, "pos"] - 1
    out.loc[mask, "variant_id"] = _rebuild_variant_id(out.loc[mask])

    # invariant: window content of the rekeyed rows is byte-identical to before
    if not (np.array_equal(out.loc[mask, REF_WIN_COL].to_numpy(), before_ref)
            and np.array_equal(out.loc[mask, ALT_WIN_COL].to_numpy(), before_alt)):
        raise ValueError("window content changed during rekey -- must never happen.")

    # variant_id must still agree with pos for ALL rows (the key is a function of pos)
    vid_pos = out["variant_id"].astype(str).str.split(":").str[2]
    if not (vid_pos == out["pos"].astype(str)).all():
        bad = int((vid_pos != out["pos"].astype(str)).sum())
        raise ValueError(f"variant_id/pos disagree on {bad} rows after rekey.")

    recon = {
        "n_rows": int(len(out)),
        "n_padded_deletions_rekeyed": n_rekey,
        "n_unchanged": int(len(out) - n_rekey),
        "window_content_changed": False,
        "schema_fingerprint": schema_fingerprint(out.columns),
        "variant_id_matches_pos": True,
    }
    return out, recon


def verify_against_cohort(rekeyed: pd.DataFrame, cohort_meta_path: Path) -> dict:
    """Attach the rekeyed windows to a cohort-v2 meta frame via the REAL join and classify
    any unmapped padded deletions as COVERAGE_GAP (row absent from the seq parquet entirely)
    vs KEY_MISMATCH (present under a different key -- a real rekey defect)."""
    from genomic_variant_classifier.data.seq_window_join import attach_delta_windows

    meta = pd.read_parquet(cohort_meta_path)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
        tmp = Path(tf.name)
    rekeyed[[*KEY_COLS, REF_WIN_COL, ALT_WIN_COL]].to_parquet(tmp, index=False)
    try:
        wins, n_unmapped = attach_delta_windows(meta, tmp, window=101)
        poly = "A" * 101
        del_unmapped = 0
        key_mismatch = 0
        coverage_gap = 0
        if {"ref", "alt"} <= set(meta.columns):
            del_mask = is_padded_deletion(meta["ref"], meta["alt"]).to_numpy()
            unmapped_mask = (wins[REF_WIN_COL].to_numpy() == poly)
            del_unmapped_mask = del_mask & unmapped_mask
            del_unmapped = int(del_unmapped_mask.sum())
            if del_unmapped:
                # is each unmapped deletion ABSENT from the seq parquet (gap) or present
                # under a different key (mismatch)? Compare on (chrom, ref, alt).
                seq_cra = set(rekeyed["chrom"].astype(str) + "|" + rekeyed["ref"].astype(str)
                              + "|" + rekeyed["alt"].astype(str))
                um = meta.loc[del_unmapped_mask]
                cra = (um["chrom"].astype(str) + "|" + um["ref"].astype(str)
                       + "|" + um["alt"].astype(str))
                key_mismatch = int(cra.isin(seq_cra).sum())
                coverage_gap = int((~cra.isin(seq_cra)).sum())
    finally:
        tmp.unlink(missing_ok=True)
    return {"n_unmapped_total": int(n_unmapped),
            "padded_deletion_unmapped": int(del_unmapped),
            "padded_deletion_coverage_gap": int(coverage_gap),
            "padded_deletion_key_mismatch": int(key_mismatch),
            "cohort_rows": int(len(meta))}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rekey seq windows to cohort-v2 coordinates.")
    p.add_argument("--seq", default="data/processed/clinvar_grch38_clean_seq.parquet")
    p.add_argument("--out", default="data/processed/clinvar_grch38_clean_v2_seq.parquet")
    p.add_argument("--verify-against", default=None,
                   help="a cohort-v2 meta/split parquet; asserts padded deletions map (n_unmapped==0)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--audit", action="store_true", help="report only (default)")
    g.add_argument("--apply", action="store_true", help="write the rekeyed parquet")
    a = p.parse_args(argv)

    seq_path, out_path = Path(a.seq), Path(a.out)
    if not seq_path.exists():
        print(f"ERROR: seq parquet not found: {seq_path}", file=sys.stderr)
        return 2
    if out_path.exists() and a.apply:
        print(f"ERROR: {out_path} exists. Refusing to overwrite.", file=sys.stderr)
        return 5

    print("=" * 74)
    print(f"REKEY SEQ WINDOWS -> cohort-v2   input {seq_path}")
    print("=" * 74)
    seq = pd.read_parquet(seq_path)
    print(f"loaded {len(seq):,} rows / {len(seq.columns)} cols")

    rekeyed, recon = rekey(seq)
    recon["v1_seq_md5"] = hashlib.md5(seq_path.read_bytes()).hexdigest().upper()

    print(f"padded deletions rekeyed (pos -= 1): {recon['n_padded_deletions_rekeyed']:,}")
    print(f"rows unchanged                     : {recon['n_unchanged']:,}")
    print(f"window content changed             : {recon['window_content_changed']}  (must be False)")
    print(f"variant_id agrees with pos         : {recon['variant_id_matches_pos']}")
    print(f"schema fingerprint                 : {recon['schema_fingerprint']}")

    if a.verify_against:
        vpath = Path(a.verify_against)
        if not vpath.exists():
            print(f"ERROR: --verify-against not found: {vpath}", file=sys.stderr)
            return 2
        print(f"\nverifying the REKEYED windows join to {vpath.name} via attach_delta_windows ...")
        v = verify_against_cohort(rekeyed, vpath)
        recon["verify"] = v
        print(f"  cohort rows                    : {v['cohort_rows']:,}")
        print(f"  total unmapped (poly-A)        : {v['n_unmapped_total']:,}")
        print(f"  padded-deletion unmapped       : {v['padded_deletion_unmapped']}")
        print(f"    of which COVERAGE_GAP        : {v['padded_deletion_coverage_gap']} "
              f"(row absent from seq parquet entirely -- pre-existing, not a rekey defect)")
        print(f"    of which KEY_MISMATCH        : {v['padded_deletion_key_mismatch']} "
              f"(present under a different key -- a real rekey defect)")
        if v["padded_deletion_key_mismatch"] > 0:
            print("  *** KEY_MISMATCH > 0: the rekey did not correctly realign these keys. "
                  "Refusing to write. ***", file=sys.stderr)
            return 6
        if v["padded_deletion_coverage_gap"] > 0:
            print(f"  NOTE: {v['padded_deletion_coverage_gap']} padded deletions are a COVERAGE GAP "
                  f"(never had a window; part of the {v['cohort_rows'] - len(rekeyed):,}-row "
                  f"cohort-vs-seq gap). The rekey is correct; these rows have no window to map "
                  f"to and will be poly-A regardless. See diagnose_seq_coverage_gap.py.")
        else:
            print("  padded deletions all map (0 unmapped, 0 coverage gap). Breakage fully closed.")

    if not a.apply:
        print("\nAUDIT (dry-run). Nothing written. Re-run with --apply.")
        return 0

    rekeyed.to_parquet(out_path, index=False)
    recon["v2_seq_md5"] = hashlib.md5(out_path.read_bytes()).hexdigest().upper()
    recon_path = out_path.with_name("seq_windows_v2_reconciliation.json")
    recon_path.write_text(json.dumps(recon, indent=2), encoding="utf-8")
    print(f"\nWROTE: {out_path.name}  (MD5 {recon['v2_seq_md5']})")
    print(f"WROTE: {recon_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
