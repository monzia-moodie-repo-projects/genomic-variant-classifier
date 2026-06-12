#!/usr/bin/env python3
"""patch_train_cnn_activation.py -- activate the 1D-CNN on the live 2-column
[fasta_seq_ref, fasta_seq_alt] delta windows.

Replaces train.py's CNN block: gate on the real ref/alt columns (not the
deprecated empty single 'fasta_seq'); build test-side windows from meta_test
and train-side windows from the already-persisted meta_train.parquet (gene-split
aligned to X_train); remove the NotImplementedError. No DataPrepPipeline.run()
signature change.

Idempotent, backup-first, py_compile-gated, ASCII-only, newline-preserving.
Author: Monzia Moodie.
"""
from __future__ import annotations
import py_compile, shutil, sys
from pathlib import Path

TARGET = Path("scripts/train.py")
MARKER = "attach_delta_windows"

OLD = (
    "    has_sequences = (\n"
    "        \"fasta_seq\" in meta_test.columns\n"
    "        and meta_test[\"fasta_seq\"].notna().sum() > 100\n"
    "    )\n"
    "    if not has_sequences:\n"
    "        logger.info(\"No usable sequence data -- removing CNN from ensemble.\")\n"
    "        ensemble.base_estimators.pop(\"cnn_1d\", None)\n"
    "        # CNN is the only sequence consumer; with it removed these series are\n"
    "        # inert placeholders that satisfy the seq-aware fit/evaluate/predict\n"
    "        # signatures but are never used for any prediction.\n"
    "        X_seq_train = pd.Series([\"A\" * 101] * len(y_train))\n"
    "        X_seq_test  = pd.Series([\"A\" * 101] * len(y_test))\n"
    "    else:\n"
    "        # Test side: meta_test[\"fasta_seq\"] is split-aligned by construction.\n"
    "        X_seq_test = meta_test[\"fasta_seq\"].reset_index(drop=True)\n"
    "        # Train side: run() does not return meta_train and X_train carries no\n"
    "        # variant_id key, so there is NO signature-free way to realign train\n"
    "        # sequences here. Rather than silently misalign (the PM11d defect),\n"
    "        # fail loudly. Enabling real training sequences requires plumbing\n"
    "        # meta_train out of DataPrepPipeline.run() first (Option-B-wide).\n"
    "        raise NotImplementedError(\n"
    "            \"Real training sequences detected, but train-side sequence \"\n"
    "            \"alignment requires meta_train, which DataPrepPipeline.run() \"\n"
    "            \"does not currently return. Plumb meta_train through run() \"\n"
    "            \"before enabling CNN training on real sequences. See \"\n"
    "            \"INCIDENT_2026-05-30_train-sequence-misalignment.md.\"\n"
    "        )\n"
)

NEW = (
    "    # CNN sequence input: the live 2-column [fasta_seq_ref, fasta_seq_alt] delta\n"
    "    # windows. The legacy single 'fasta_seq' column is deprecated and empty.\n"
    "    from genomic_variant_classifier.data.seq_window_join import (\n"
    "        attach_delta_windows,\n"
    "        REF_WIN_COL,\n"
    "        ALT_WIN_COL,\n"
    "    )\n"
    "\n"
    "    has_sequences = (\n"
    "        REF_WIN_COL in meta_test.columns\n"
    "        and ALT_WIN_COL in meta_test.columns\n"
    "        and meta_test[REF_WIN_COL].notna().sum() > 100\n"
    "    )\n"
    "    if not has_sequences:\n"
    "        logger.info(\"No usable ref/alt sequence windows -- removing CNN from ensemble.\")\n"
    "        ensemble.base_estimators.pop(\"cnn_1d\", None)\n"
    "        # CNN is the only sequence consumer; with it removed these placeholders\n"
    "        # satisfy the seq-aware fit/evaluate/predict signatures but are unused.\n"
    "        X_seq_train = pd.Series([\"A\" * 101] * len(y_train))\n"
    "        X_seq_test  = pd.Series([\"A\" * 101] * len(y_test))\n"
    "    else:\n"
    "        # Test side: meta_test carries ref/alt, structurally split-aligned to X_test.\n"
    "        X_seq_test, _n_unmapped_test = attach_delta_windows(meta_test)\n"
    "        # Train side: meta_train is persisted by _save_splits, gene-split-aligned to\n"
    "        # X_train (both df.iloc[train_idx].reset_index). Read it -- no run() change.\n"
    "        meta_train_path = config.output_dir / \"meta_train.parquet\"\n"
    "        if not meta_train_path.exists():\n"
    "            raise FileNotFoundError(\n"
    "                f\"meta_train.parquet not found at {meta_train_path}; required for \"\n"
    "                \"CNN train-side sequences (DataPrepPipeline._save_splits writes it).\"\n"
    "            )\n"
    "        _meta_train = pd.read_parquet(meta_train_path)\n"
    "        if len(_meta_train) != len(y_train):\n"
    "            raise ValueError(\n"
    "                f\"meta_train rows ({len(_meta_train)}) != y_train ({len(y_train)}); \"\n"
    "                \"split misalignment -- aborting to avoid PM11d-style label mismatch.\"\n"
    "            )\n"
    "        X_seq_train, _n_unmapped_train = attach_delta_windows(_meta_train)\n"
    "        logger.info(\n"
    "            \"CNN sequences active (delta mode): train=%d (unmapped=%d), \"\n"
    "            \"test=%d (unmapped=%d).\",\n"
    "            len(X_seq_train), _n_unmapped_train, len(X_seq_test), _n_unmapped_test,\n"
    "        )\n"
)

def main() -> int:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found"); return 1
    raw = TARGET.read_text(encoding="utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    if MARKER in text:
        print("already applied (attach_delta_windows present); no change."); return 0
    c = text.count(OLD)
    if c != 1:
        print(f"ABORT: CNN block anchor found {c} times (expected 1); no change."); return 1
    text = text.replace(OLD, NEW, 1)
    bak = TARGET.with_suffix(TARGET.suffix + ".bak")
    shutil.copy2(TARGET, bak)
    TARGET.write_text(text.replace("\n", nl), encoding="utf-8", newline="")
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as exc:
        shutil.copy2(bak, TARGET); print(f"ABORT: py_compile failed, restored:\n{exc}"); return 1
    print(f"OK: CNN activation applied; backup at {bak}"); return 0

if __name__ == "__main__":
    sys.exit(main())
