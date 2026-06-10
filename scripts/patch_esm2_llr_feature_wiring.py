#!/usr/bin/env python3
"""patch_esm2_llr_feature_wiring.py -- wire esm2_llr into the feature matrix (Phase 1).

Two files, kept in lockstep:
  variant_ensemble.py
    * EXPECTED_TABULAR_FEATURE_COUNT 79 -> 80
    * TABULAR_FEATURES: add "esm2_llr" after "esm2_delta_norm" (comment (1)->(2))
    * engineer_features: assemble feats["esm2_llr"] after esm2_delta_norm -- NO CLIP
      (esm2_llr is SIGNED; negative = damaging; clipping would zero the signal)
  real_data_prep.py
    * step 16: call esm2.annotate_llr(df) on the same connector + log (prints model)
    * feature assembly: feats["esm2_llr"] after esm2_delta_norm -- NO CLIP

INFERENCE_FEATURE_COLUMNS is list(TABULAR_FEATURES) in api/pipeline.py -- derived,
so it is NOT edited here; the contract test verifies the propagation.

Count-guarded, backup-first, idempotent, py_compile-gated. Assembly inserts are
indentation-detecting (4-space module fn vs 8-space method). Author: Monzia Moodie.
"""
from __future__ import annotations

import datetime as _dt
import py_compile
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VE = REPO / "src/genomic_variant_classifier/models/variant_ensemble.py"
RDP = REPO / "src/genomic_variant_classifier/data/real_data_prep.py"

LLR_COMMENT = "ESM-2 LLR (1) -- SIGNED feature (negative => damaging); NO clip"

# --- literal, count-guarded replaces (marker = skip if already applied) ---
LITERAL = {
    "ve": [
        ("EXPECTED_TABULAR_FEATURE_COUNT = 79",
         "EXPECTED_TABULAR_FEATURE_COUNT = 80",
         "EXPECTED_TABULAR_FEATURE_COUNT = 80",
         "count 79 -> 80"),
        ('    # ESM-2 (1)\n    "esm2_delta_norm",',
         '    # ESM-2 (2)\n    "esm2_delta_norm",\n    "esm2_llr",',
         '    "esm2_llr",',
         "TABULAR_FEATURES += esm2_llr"),
    ],
    "rdp": [
        ("        df = esm2.annotate_dataframe(df)",
         "        df = esm2.annotate_dataframe(df)\n"
         "        df = esm2.annotate_llr(df)\n"
         "        logger.info(\n"
         "            \"Score annotation 16b (ESM-2 LLR, model=%s): %d missense \"\n"
         "            \"variants scored (esm2_llr != 0).\",\n"
         "            ac.esm2_model_name,\n"
         "            int((df.get(\"esm2_llr\", pd.Series([0.0] * len(df), index=df.index)) != 0).sum()),\n"
         "        )",
         "esm2.annotate_llr(df)",
         "step-16b annotate_llr call + log"),
    ],
}


def _insert_esm2_llr_assembly(text: str) -> tuple[str, str]:
    """Insert feats[\"esm2_llr\"] right after the feats[\"esm2_delta_norm\"] block,
    matching the surrounding indentation. Returns (new_text, status)."""
    if 'feats["esm2_llr"]' in text:
        return text, "skip (already applied): esm2_llr assembly"
    lines = text.split("\n")
    anchor = 'feats["esm2_delta_norm"] = ('
    idxs = [i for i, ln in enumerate(lines) if ln.strip() == anchor]
    if len(idxs) != 1:
        return text, f"ABORT: feats[\"esm2_delta_norm\"] anchor found {len(idxs)}x (expected 1)"
    i = idxs[0]
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    # find the block's closing ')' at the SAME indent as the assignment
    j = None
    for k in range(i + 1, len(lines)):
        if lines[k] == indent + ")":
            j = k
            break
    if j is None:
        return text, "ABORT: could not find esm2_delta_norm block closing paren"
    block = [
        f"{indent}# {LLR_COMMENT}",
        f'{indent}feats["esm2_llr"] = (',
        f'{indent}    df.get("esm2_llr", pd.Series([0.0] * len(df), index=df.index))',
        f"{indent}    .fillna(0.0)",
        f"{indent}    .astype(float)",
        f"{indent})",
    ]
    lines[j + 1 : j + 1] = [""] + block
    return "\n".join(lines), "ok: esm2_llr assembly (no clip)"


def _apply(path: Path, key: str) -> int:
    text = path.read_text(encoding="utf-8")
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(path, f"{path}.bak_{ts}")
    for old, new, marker, label in LITERAL.get(key, []):
        if marker in text:
            print(f"  skip (already applied): {label}")
            continue
        n = text.count(old)
        if n != 1:
            print(f"ABORT [{path.name}]: anchor '{label}' found {n}x (expected 1); nothing written")
            return 3
        text = text.replace(old, new, 1)
        print(f"  ok: {label}")
    text, status = _insert_esm2_llr_assembly(text)
    print(f"  {status}")
    if status.startswith("ABORT"):
        return 4
    path.write_text(text, encoding="utf-8")
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        print(f"ABORT: py_compile failed for {path.name}: {exc}")
        return 5
    print(f"py_compile clean: {path.name}  (backup -> {path.name}.bak_{ts})")
    return 0


def main() -> int:
    for path, key in ((VE, "ve"), (RDP, "rdp")):
        if not path.exists():
            print(f"ABORT: missing {path}")
            return 2
        rc = _apply(path, key)
        if rc:
            return rc
    print("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
