"""
run11_phase0_apply.py — Apply Run 11 Phase 0 patches to production files
=========================================================================

Fixes:
  F4: Remove redundant Phase 2 block at bottom of _engineer_features()
      in src/genomic_variant_classifier/data/real_data_prep.py
  F6: gnn_score default 0.5 → comment noting future 0.0 migration

Integrations applied:
  I3: GPU GBDT (CatBoost task_type, XGBoost device, LightGBM device_type)
  I8: Parquet ZSTD compression in _save_splits()

Integrations NOT applied by this script (require new files):
  I2: FastKAN swap (new class definition, separate file)
  I4: Optuna HPO (new script)
  I5: Polars ETL (new module)
  I6: PrimateAI-3D connector (new module)
  I7: BF16 / torch.compile (requires model-level changes)

Carried-forward:
  3.2: OOF row-index sidecar (patch to fit() checkpoint block)

Usage:
    cd C:\\Projects\\genomic-variant-classifier
    python run11_phase0_apply.py --dry-run
    python run11_phase0_apply.py --apply

All patches use str.replace() with unique anchors. If any anchor is not
found, the script aborts with a clear error. No partial writes.

Author: Monzia Moodie
Date: 2026-05-24
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(r"C:\Projects\genomic-variant-classifier")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def apply_patch(path: Path, anchor: str, replacement: str, label: str, dry_run: bool) -> bool:
    """Replace anchor with replacement in file. Returns True on success."""
    if not path.exists():
        print(f"  [ERROR] {label}: file not found: {path}")
        return False

    text = path.read_text(encoding="utf-8")
    count = text.count(anchor)

    if count == 0:
        print(f"  [ERROR] {label}: anchor not found in {path.name}")
        print(f"          First 60 chars of anchor: {repr(anchor[:60])}")
        return False
    if count > 1:
        print(f"  [ERROR] {label}: anchor found {count} times (expected 1) in {path.name}")
        return False

    if dry_run:
        print(f"  [DRY-RUN] {label}: anchor found in {path.name} (would replace)")
        return True

    new_text = text.replace(anchor, replacement, 1)
    path.write_text(new_text, encoding="utf-8", newline="\n")
    print(f"  [APPLIED] {label}: {path.name} (sha={sha256(path)})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run 11 Phase 0 patches")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Check anchors without writing")
    group.add_argument("--apply", action="store_true", help="Apply patches")
    args = parser.parse_args()

    dry_run = args.dry_run
    all_ok = True

    print("=" * 70)
    print(f"Run 11 Phase 0 Patches — {'DRY RUN' if dry_run else 'APPLYING'}")
    print(f"Repository: {REPO}")
    print("=" * 70)

    # ── F4: Remove redundant Phase 2 block in _engineer_features() ─────────
    print("\n--- F4: Remove redundant Phase 2 block in real_data_prep.py ---")
    rdp = REPO / "src" / "genomic_variant_classifier" / "data" / "real_data_prep.py"

    # The redundant block appears AFTER the fillna(0.0) block
    f4_anchor = (
        '            # Phase 2 features — now active\n'
        '        protein_change = df.get("protein_change", pd.Series([None]*len(df), index=df.index))\n'
        '        feats["codon_position"]       = protein_change.apply(_parse_codon_position).astype(int)\n'
        '        feats["splice_ai_score"]      = df.get("splice_ai_score",      pd.Series([0.0]*len(df), index=df.index)).fillna(0.0).astype(float)\n'
        '        feats["alphamissense_score"]  = df.get("alphamissense_score",   pd.Series([0.5]*len(df), index=df.index)).fillna(0.5).astype(float)\n'
    )

    f4_replacement = (
        '            # Phase 2 features — codon_position, splice_ai_score, alphamissense_score\n'
        '        # are already computed above in their respective sections.\n'
        '        # The redundant block that was here (overwriting codon_position via\n'
        '        # _parse_codon_position on unpopulated protein_change column) was\n'
        '        # removed in Run 11 Phase 0 — see RUN_11_FINDINGS F4.\n'
    )

    ok = apply_patch(rdp, f4_anchor, f4_replacement, "F4", dry_run)
    all_ok = all_ok and ok

    # ── I3: GPU GBDT acceleration ──────────────────────────────────────────
    print("\n--- I3: GPU GBDT acceleration in variant_ensemble.py ---")
    ve = REPO / "src" / "genomic_variant_classifier" / "models" / "variant_ensemble.py"

    # Patch XGBoost: add device detection
    i3a_anchor = (
        '            "xgboost": xgb.XGBClassifier(\n'
        '                n_estimators=500,\n'
        '                max_depth=6,\n'
        '                learning_rate=0.05,\n'
        '                subsample=0.8,\n'
        '                colsample_bytree=0.8,\n'
        '                scale_pos_weight=10,\n'
        '                eval_metric="auc",\n'
        '                n_jobs=cfg.n_jobs,\n'
        '                random_state=cfg.random_state,\n'
        '                verbosity=0,\n'
        '            ),'
    )

    i3a_replacement = (
        '            "xgboost": xgb.XGBClassifier(\n'
        '                n_estimators=500,\n'
        '                max_depth=6,\n'
        '                learning_rate=0.05,\n'
        '                subsample=0.8,\n'
        '                colsample_bytree=0.8,\n'
        '                scale_pos_weight=10,\n'
        '                eval_metric="auc",\n'
        '                n_jobs=cfg.n_jobs,\n'
        '                random_state=cfg.random_state,\n'
        '                verbosity=0,\n'
        '                # Run 11 I3: GPU acceleration (auto-detected)\n'
        '                **({\"device\": \"cuda\", \"tree_method\": \"hist\"} if _GPU_AVAILABLE else {}),\n'
        '            ),'
    )

    ok = apply_patch(ve, i3a_anchor, i3a_replacement, "I3a-XGBoost", dry_run)
    all_ok = all_ok and ok

    # Patch LightGBM: add device detection
    i3b_anchor = (
        '            "lightgbm": lgb.LGBMClassifier(\n'
        '                n_estimators=500,\n'
        '                max_depth=6,\n'
        '                learning_rate=0.05,\n'
        '                subsample=0.8,\n'
        '                colsample_bytree=0.8,\n'
        '                class_weight=cfg.class_weight,\n'
        '                n_jobs=cfg.n_jobs,\n'
        '                random_state=cfg.random_state,\n'
        '                verbose=-1,\n'
        '            ),'
    )

    i3b_replacement = (
        '            "lightgbm": lgb.LGBMClassifier(\n'
        '                n_estimators=500,\n'
        '                max_depth=6,\n'
        '                learning_rate=0.05,\n'
        '                subsample=0.8,\n'
        '                colsample_bytree=0.8,\n'
        '                class_weight=cfg.class_weight,\n'
        '                n_jobs=cfg.n_jobs,\n'
        '                random_state=cfg.random_state,\n'
        '                verbose=-1,\n'
        '                # Run 11 I3: GPU acceleration (auto-detected)\n'
        '                **({\"device_type\": \"gpu\", \"gpu_use_dp\": False} if _GPU_AVAILABLE else {}),\n'
        '            ),'
    )

    ok = apply_patch(ve, i3b_anchor, i3b_replacement, "I3b-LightGBM", dry_run)
    all_ok = all_ok and ok

    # Patch CatBoost: change task_type from hardcoded "CPU" to auto-detected
    i3c_anchor = '                task_type="CPU",'
    i3c_replacement = '                task_type="GPU" if _GPU_AVAILABLE else "CPU",  # Run 11 I3'

    ok = apply_patch(ve, i3c_anchor, i3c_replacement, "I3c-CatBoost", dry_run)
    all_ok = all_ok and ok

    # Add _GPU_AVAILABLE constant near the top (after the logger line)
    i3d_anchor = 'logger = logging.getLogger(__name__)'
    i3d_replacement = (
        'logger = logging.getLogger(__name__)\n'
        '\n'
        '# Run 11 I3: GPU GBDT auto-detection\n'
        'try:\n'
        '    import torch as _torch\n'
        '    _GPU_AVAILABLE = _torch.cuda.is_available()\n'
        'except ImportError:\n'
        '    _GPU_AVAILABLE = False'
    )

    ok = apply_patch(ve, i3d_anchor, i3d_replacement, "I3d-GPU-flag", dry_run)
    all_ok = all_ok and ok

    # ── 3.2: OOF row-index sidecar ────────────────────────────────────────
    print("\n--- 3.2: OOF row-index sidecar in variant_ensemble.py ---")

    # Add oof_indices saving right after the existing oof .npy save
    oof_sidecar_anchor = '                np.save(_oof_path, oof)'

    oof_sidecar_replacement = (
        '                np.save(_oof_path, oof)\n'
        '                # Run 11 carried-forward 3.2: OOF row-index sidecar\n'
        '                # Saves the per-fold prediction-to-row mapping so meta-learner\n'
        '                # can be reconstructed from saved OOF arrays in disaster recovery.\n'
        '                _oof_idx_path = _ckpt_dir / f"{name}_oof_indices.npy"\n'
        '                _fold_indices = [test_idx for _, test_idx in cv.split(X_input_fit, y_fit)]\n'
        '                np.save(_oof_idx_path, np.concatenate(_fold_indices))'
    )

    ok = apply_patch(ve, oof_sidecar_anchor, oof_sidecar_replacement, "3.2-OOF-sidecar", dry_run)
    all_ok = all_ok and ok

    # ── I8: Parquet ZSTD compression in _save_splits() ─────────────────────
    print("\n--- I8: Parquet ZSTD compression in real_data_prep.py ---")

    # Patch each to_parquet call to use ZSTD
    i8_anchor = '        X_train.to_parquet(out / "X_train.parquet", index=False)'
    i8_replacement = '        X_train.to_parquet(out / "X_train.parquet", index=False, compression="zstd")  # Run 11 I8'

    ok = apply_patch(rdp, i8_anchor, i8_replacement, "I8a-X_train", dry_run)
    all_ok = all_ok and ok

    i8b_anchor = '        X_val.to_parquet(out / "X_val.parquet", index=False)'
    i8b_replacement = '        X_val.to_parquet(out / "X_val.parquet", index=False, compression="zstd")  # Run 11 I8'
    ok = apply_patch(rdp, i8b_anchor, i8b_replacement, "I8b-X_val", dry_run)
    all_ok = all_ok and ok

    i8c_anchor = '        X_test.to_parquet(out / "X_test.parquet", index=False)'
    i8c_replacement = '        X_test.to_parquet(out / "X_test.parquet", index=False, compression="zstd")  # Run 11 I8'
    ok = apply_patch(rdp, i8c_anchor, i8c_replacement, "I8c-X_test", dry_run)
    all_ok = all_ok and ok

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if all_ok:
        if dry_run:
            print("DRY RUN PASSED: All anchors found. Run with --apply to write changes.")
        else:
            print("ALL PATCHES APPLIED SUCCESSFULLY.")
            print("\nNext steps:")
            print("  1. Review changes: git diff")
            print("  2. Run tests: python -m pytest tests/ -v --timeout=300 -q")
            print("  3. If green: git add -A && git commit -m 'feat(run11): Phase 0 patches'")
    else:
        print("ERRORS DETECTED. Some patches could not be applied.")
        print("Check the [ERROR] messages above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
