#!/usr/bin/env python3
"""
kg_semantic_hash.py  --  Monzia Moodie

Semantic content hash for the 1000G af_1kg parquet, plus a write-if-changed guard. The hash is taken
over the SORTED variant key + allele-frequency columns ONLY -- never raw parquet bytes -- so parquet
row order, compression, and metadata timestamps do not perturb it. Re-deriving an identical AF table
yields an identical hash, which lets the build/merge step SKIP rewriting (and thus avoid a redundant
~6 MB binary re-commit, as happened 26342e9 -> 988439c).

Canonical semantic columns (fixed order):
    chrom, pos, ref, alt, af, af_afr, af_amr, af_eas, af_eur, af_sas

The on-disk parquet stores these under different names; they are resolved case-insensitively from:
    key   : (chrom,pos,ref,alt) explicit, OR split from a single 'variant_id' = "chrom:pos:ref:alt"
    af    : allele_freq | af
    af_afr: AFR_AF | af_afr        af_amr: AMR_AF | af_amr        af_eas: EAS_AF | af_eas
    af_eur: EUR_AF | af_eur        af_sas: SAS_AF | af_sas
If any semantic field cannot be resolved, the functions raise KGSchemaError listing the columns that
ARE present, so the failure is loud and diagnosable.

CLI:
    python scripts/kg_semantic_hash.py <parquet>              # print semantic hash
    python scripts/kg_semantic_hash.py <parquet> <parquet2>   # print both + MATCH/DIFFER
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Union

import pandas as pd

HASH_VERSION = "kg_semantic_v1"
SEMANTIC_ORDER = ["chrom", "pos", "ref", "alt", "af", "af_afr", "af_amr", "af_eas", "af_eur", "af_sas"]
KEY_FIELDS = ["chrom", "pos", "ref", "alt"]
AF_FIELDS = ["af", "af_afr", "af_amr", "af_eas", "af_eur", "af_sas"]

# case-insensitive candidate source names for each AF semantic field
_AF_CANDIDATES = {
    "af": ["allele_freq", "af"],
    "af_afr": ["afr_af", "af_afr"],
    "af_amr": ["amr_af", "af_amr"],
    "af_eas": ["eas_af", "af_eas"],
    "af_eur": ["eur_af", "af_eur"],
    "af_sas": ["sas_af", "af_sas"],
}
_FLOAT_FMT = "{:.9f}"


class KGSchemaError(ValueError):
    """Raised when the kg parquet does not expose the required semantic columns."""


def _lc_map(cols) -> dict[str, str]:
    """lowercased-name -> actual-name (last wins on dup-lower, which we don't expect)."""
    return {c.lower(): c for c in cols}


def _resolve_key(df: pd.DataFrame, lc: dict[str, str]) -> pd.DataFrame:
    if all(k in lc for k in KEY_FIELDS):
        out = pd.DataFrame({k: df[lc[k]].astype(str) for k in KEY_FIELDS})
        return out
    if "variant_id" in lc:
        parts = df[lc["variant_id"]].astype(str).str.split(":", n=3, expand=True)
        if parts.shape[1] != 4:
            raise KGSchemaError(
                f"variant_id did not split into 4 ':'-fields (got {parts.shape[1]}); "
                f"sample={df[lc['variant_id']].astype(str).head(3).tolist()}"
            )
        bad = parts.isna().any(axis=1)
        if bool(bad.any()):
            ex = df[lc["variant_id"]].astype(str)[bad].head(3).tolist()
            raise KGSchemaError(f"{int(bad.sum())} variant_id value(s) are not 'chrom:pos:ref:alt'; e.g. {ex}")
        parts.columns = KEY_FIELDS
        return parts.astype(str)
    raise KGSchemaError(
        "cannot resolve variant key: need explicit chrom/pos/ref/alt OR a 'variant_id' column. "
        f"available columns = {sorted(df.columns)}"
    )


def _resolve_af(df: pd.DataFrame, lc: dict[str, str]) -> pd.DataFrame:
    out, missing = {}, []
    for field in AF_FIELDS:
        src = next((lc[c] for c in _AF_CANDIDATES[field] if c in lc), None)
        if src is None:
            missing.append(field)
        else:
            out[field] = pd.to_numeric(df[src], errors="coerce")
    if missing:
        raise KGSchemaError(
            f"cannot resolve AF field(s) {missing} from candidates "
            f"{ {m: _AF_CANDIDATES[m] for m in missing} }. available columns = {sorted(df.columns)}"
        )
    return pd.DataFrame(out)


def _canonical_frame(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    df = source if isinstance(source, pd.DataFrame) else pd.read_parquet(source)
    if df.shape[0] == 0:
        raise KGSchemaError("kg frame has 0 rows")
    lc = _lc_map(df.columns)
    key = _resolve_key(df, lc)
    af = _resolve_af(df, lc)
    frame = pd.concat([key.reset_index(drop=True), af.reset_index(drop=True)], axis=1)
    frame = frame[SEMANTIC_ORDER]
    frame = frame.sort_values(KEY_FIELDS, kind="mergesort").reset_index(drop=True)
    return frame


def semantic_hash(source: Union[str, Path, pd.DataFrame]) -> str:
    frame = _canonical_frame(source)
    h = hashlib.sha256()
    h.update((HASH_VERSION + "\t" + ",".join(SEMANTIC_ORDER) + f"\trows={len(frame)}\n").encode())
    key_vals = [frame[k].to_numpy() for k in KEY_FIELDS]
    af_vals = [frame[k].to_numpy() for k in AF_FIELDS]
    for i in range(len(frame)):
        rec = [str(col[i]) for col in key_vals]
        for col in af_vals:
            v = col[i]
            rec.append("nan" if pd.isna(v) else _FLOAT_FMT.format(float(v)))
        h.update(("\t".join(rec) + "\n").encode())
    return h.hexdigest()


def write_parquet_if_changed(df: pd.DataFrame, out_path: Union[str, Path], **to_parquet_kw) -> bool:
    """Write df to out_path UNLESS an existing file there is semantically identical.
    Returns True if written, False if skipped. Prints the skip notice."""
    out_path = Path(out_path)
    to_parquet_kw.setdefault("index", False)
    if out_path.exists():
        try:
            if semantic_hash(out_path) == semantic_hash(df):
                print("1KGP AF semantic hash unchanged; not rewriting parquet")
                return False
        except KGSchemaError as e:
            # existing file is unreadable/odd -> do not block the write; warn and overwrite
            print(f"[kg-guard] existing parquet not comparable ({e}); rewriting", file=sys.stderr)
    df.to_parquet(out_path, **to_parquet_kw)
    return True


def _main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(__doc__); return 2
    h1 = semantic_hash(argv[1])
    print(f"{h1}  {argv[1]}")
    if len(argv) == 3:
        h2 = semantic_hash(argv[2])
        print(f"{h2}  {argv[2]}")
        print("MATCH" if h1 == h2 else "DIFFER")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
