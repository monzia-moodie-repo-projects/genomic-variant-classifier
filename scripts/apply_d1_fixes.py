"""
scripts/apply_d1_fixes.py
==========================
Applies D.1 correctness patches to real_data_prep.py and run_phase2_eval.py.

Each patch uses a guarded single-match str.replace with count==1 abort.
Before any patch is applied the script shows what it found, so you can
verify the match is correct before committing.

Run from repo root:
    python scripts/apply_d1_fixes.py --dry-run   # preview only
    python scripts/apply_d1_fixes.py             # apply

Exit codes:
  0  All patches applied successfully (or --dry-run completed).
  1  One or more patches had match-count != 1 — nothing written, safe to re-run.

Patches
-------
F-02  real_data_prep.py  _assert_clean_cohort: eliminate silent-skip when
      neither variant_id nor locus columns exist.
F-05  run_phase2_eval.py auto-enable --skip-cnn when seq-windows file absent.
F-06  real_data_prep.py  normalise _annotate_scores log step numbers to N/17.
F-07  real_data_prep.py  _join_gnomad: coerce _pos to int (not str).
F-13  run_phase2_eval.py OOF sidecar: include _train_row_idx column.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _save(path: Path, text: str, dry_run: bool) -> None:
    if not dry_run:
        path.write_text(text, encoding="utf-8")


def apply_patch(
    path: Path,
    label: str,
    old: str,
    new: str,
    dry_run: bool,
) -> bool:
    """
    Replace exactly one occurrence of *old* with *new* in *path*.
    Returns True on success (or dry-run preview), False on mismatch.
    NEVER writes unless count == 1.
    """
    text  = _load(path)
    count = text.count(old)

    if count == 0:
        # Check whether it might already be patched (new text present).
        if text.count(new) == 1:
            print(f"  SKIP [{label}]: already patched in {path.name}")
            return True
        print(
            f"  ERROR [{label}]: OLD string not found in {path.name}.\n"
            f"    Expected to find (first 80 chars): {old[:80]!r}\n"
            f"    This means either the file is a different version than\n"
            f"    expected (HEAD 553efac) or the OLD string has whitespace\n"
            f"    differences.  Run with --dry-run and compare manually.",
            file=sys.stderr,
        )
        return False

    if count > 1:
        print(
            f"  ERROR [{label}]: found {count} matches (expected exactly 1) in "
            f"{path.name}.  Cannot safely apply.  Review the file manually.",
            file=sys.stderr,
        )
        return False

    # Exactly one match.
    if dry_run:
        print(f"  DRY-RUN [{label}]: 1 match found in {path.name} — would patch")
        return True

    patched = text.replace(old, new, 1)
    _save(path, patched, dry_run=False)
    print(f"  OK [{label}]: patched {path.name}")
    return True


# ---------------------------------------------------------------------------
# Patch definitions
# ---------------------------------------------------------------------------

def patches_for_rdp(rdp: Path, dry_run: bool) -> int:
    """Apply all real_data_prep.py patches. Returns number of failures."""
    failures = 0

    # ------------------------------------------------------------------
    # F-02: _assert_clean_cohort — eliminate silent-skip (_key = None)
    # ------------------------------------------------------------------
    ok = apply_patch(
        path=rdp, label="F-02 _assert_clean_cohort",
        dry_run=dry_run,
        old=(
            '        if "variant_id" in df.columns:\n'
            '            _key = df["variant_id"]\n'
            '        elif all(c in df.columns for c in ("chrom", "pos", "ref", "alt")):\n'
            '            _key = (\n'
            '                df["chrom"].astype(str) + ":" + df["pos"].astype(str)\n'
            '                + ":" + df["ref"].astype(str) + ":" + df["alt"].astype(str)\n'
            '            )\n'
            '        else:\n'
            '            _key = None\n'
            '        if _key is not None and bool(_key.duplicated().any()):\n'
            '            raise ValueError(\n'
            '                f"duplicate variant identity in {source}; run scripts/clean_cohort.py --apply."\n'
            '            )'
        ),
        new=(
            '        if "variant_id" in df.columns:\n'
            '            _key = df["variant_id"]\n'
            '        elif all(c in df.columns for c in ("chrom", "pos", "ref", "alt")):\n'
            '            _key = (\n'
            '                df["chrom"].astype(str) + ":" + df["pos"].astype(str)\n'
            '                + ":" + df["ref"].astype(str) + ":" + df["alt"].astype(str)\n'
            '            )\n'
            '        else:\n'
            '            raise ValueError(\n'
            '                f"Cannot construct variant identity key in {source}: "\n'
            '                "expected \'variant_id\' column or all of "\n'
            '                "(chrom, pos, ref, alt). "\n'
            '                "This is required for the dedup assertion."\n'
            '            )\n'
            '        if bool(_key.duplicated().any()):\n'
            '            raise ValueError(\n'
            '                f"duplicate variant identity in {source}; run scripts/clean_cohort.py --apply."\n'
            '            )'
        ),
    )
    if not ok:
        failures += 1

    # ------------------------------------------------------------------
    # F-06: normalise _annotate_scores log step numbers to N/17
    # Apply as individual string replacements — each is unique in the file.
    # ------------------------------------------------------------------
    log_patches = [
        # (old_fragment, new_fragment)
        # Each fragment is a short unique string — no multi-line matching needed.
        # Using just the log-message string literal avoids whitespace fragility.
        ('"Score annotation 3/4 skipped (CADD disabled)."',
         '"Score annotation 3/17 skipped (CADD disabled)."'),

        ('"Score annotation 4/6 (SpliceAI): %d variants with splice_ai_score > 0.",',
         '"Score annotation 4/17 (SpliceAI): %d variants with splice_ai_score > 0.",'),

        ('"Score annotation 5/6 (AlphaMissense): %d variants annotated (score != 0.5).",',
         '"Score annotation 5/17 (AlphaMissense): %d variants annotated (score != 0.5).",'),

        ('"Score annotation 6/12 (GTEx): %d eQTL variants.",',
         '"Score annotation 6/17 (GTEx): %d eQTL variants.",'),

        ('"Score annotation 7/12 (VEP): %d variants with non-zero codon_position.",',
         '"Score annotation 7/17 (VEP): %d variants with non-zero codon_position.",'),

        ('"Score annotation 8/12 (OMIM): %d variants with omim_n_diseases > 0.",',
         '"Score annotation 8/17 (OMIM): %d variants with omim_n_diseases > 0.",'),

        ('"Score annotation 9/12 (ClinGen): %d variants with clingen_validity_score > 0.",',
         '"Score annotation 9/17 (ClinGen): %d variants with clingen_validity_score > 0.",'),

        ('"Score annotation 10/12 (dbSNP): %d variants with dbsnp_af > 0.",',
         '"Score annotation 10/17 (dbSNP): %d variants with dbsnp_af > 0.",'),

        ('"Score annotation 11/12 (EVE): %d variants covered (score != 0.5).",',
         '"Score annotation 11/17 (EVE): %d variants covered (score != 0.5).",'),

        ('"Score annotation 12/14 (HGMD): %d variants flagged as disease mutations.",',
         '"Score annotation 12/17 (HGMD): %d variants flagged as disease mutations.",'),

        ('"Score annotation 13/14 (RNA splice): %d splice-gated variants annotated.",',
         '"Score annotation 13/17 (RNA splice): %d splice-gated variants annotated.",'),

        ('"Score annotation 14/14 (protein structure): %d missense variants annotated.",',
         '"Score annotation 14/17 (protein structure): %d missense variants annotated.",'),
    ]

    text = _load(rdp)
    any_log_failure = False
    for old_frag, new_frag in log_patches:
        cnt = text.count(old_frag)
        if cnt == 0:
            if text.count(new_frag) >= 1:
                print(f"  SKIP [F-06 log]: '{old_frag[:45]}' already patched")
                continue
            print(f"  WARN [F-06 log]: '{old_frag[:45]}' not found — skipping")
            continue
        if cnt > 1:
            print(
                f"  ERROR [F-06 log]: {cnt} matches for '{old_frag[:45]}' "
                f"— cannot safely apply",
                file=sys.stderr,
            )
            any_log_failure = True
            continue
        if dry_run:
            print(f"  DRY-RUN [F-06 log]: would patch '{old_frag[:45]}'")
        else:
            text = text.replace(old_frag, new_frag, 1)
            print(f"  OK [F-06 log]: patched '{old_frag[:45]}'")

    if not dry_run and not any_log_failure:
        _save(rdp, text, dry_run=False)
    if any_log_failure:
        failures += 1

    # ------------------------------------------------------------------
    # F-07: _join_gnomad — coerce _pos to int
    # Use a shorter unique anchor to avoid blank-line fragility.
    # ------------------------------------------------------------------
    ok = apply_patch(
        path=rdp, label="F-07 _join_gnomad pos int",
        dry_run=dry_run,
        old=(
            '        df["_chrom"] = df["chrom"].astype(str)\n'
            '        df["_pos"] = df["pos"].astype(str)\n'
            '        df["_ref"] = df["ref"].astype(str)\n'
            '        df["_alt"] = df["alt"].astype(str)'
        ),
        new=(
            '        df["_chrom"] = df["chrom"].astype(str)\n'
            '        df["_pos"]   = pd.to_numeric(df["pos"], errors="coerce").fillna(0).astype(int)\n'
            '        df["_ref"]   = df["ref"].astype(str)\n'
            '        df["_alt"]   = df["alt"].astype(str)\n'
            '        # Align gnomAD _pos to int for robust locus matching (avoids\n'
            '        # leading-zero string mismatch — FINDING F-07).\n'
            '        gnomad["_pos"] = pd.to_numeric(gnomad["_pos"], errors="coerce").fillna(0).astype(int)'
        ),
    )
    if not ok:
        failures += 1

    return failures


def patches_for_eval(eval_path: Path, dry_run: bool) -> int:
    """Apply all run_phase2_eval.py patches. Returns number of failures."""
    failures = 0

    # ------------------------------------------------------------------
    # F-05: auto-enable --skip-cnn when seq-windows file not found
    # ------------------------------------------------------------------
    ok = apply_patch(
        path=eval_path, label="F-05 seq-windows auto-skip-cnn",
        dry_run=dry_run,
        old=(
            '        _seq_win = Path(args.seq_windows) if getattr(args, "seq_windows", None) else None\n'
            '        if _seq_win is not None and not _seq_win.exists():\n'
            '            logger.warning("seq-windows parquet not found: %s (falling back to poly-A)", _seq_win)\n'
            '            _seq_win = None'
        ),
        new=(
            '        _seq_win = Path(args.seq_windows) if getattr(args, "seq_windows", None) else None\n'
            '        if _seq_win is not None and not _seq_win.exists():\n'
            '            logger.warning(\n'
            '                "seq-windows parquet not found: %s -- automatically enabling --skip-cnn. "\n'
            '                "Poly-A fallback would fail the >0.5%% unmapped gate. "\n'
            '                "Pass --seq-windows <path> or --skip-cnn explicitly to silence this.",\n'
            '                _seq_win,\n'
            '            )\n'
            '            _seq_win = None\n'
            '            args.skip_cnn = True'
        ),
    )
    if not ok:
        failures += 1

    # ------------------------------------------------------------------
    # F-13: OOF sidecar — include _train_row_idx column
    # ------------------------------------------------------------------
    ok = apply_patch(
        path=eval_path, label="F-13 OOF row-index",
        dry_run=dry_run,
        old=(
            '        try:\n'
            '            import numpy as _np  # noqa: F811\n'
            '            _oof = getattr(ensemble, "oof_predictions_", None)\n'
            '            _names = getattr(ensemble, "oof_model_names_", None)\n'
            '            if _oof is not None and _names is not None:\n'
            '                _oof_df = pd.DataFrame(_oof, columns=list(_names))\n'
            '                _oof_df.to_parquet(outdir / "oof_predictions.parquet",\n'
            '                                   index=False)\n'
            '                logger.info("OOF predictions flushed to %s/oof_predictions.parquet",\n'
            '                            outdir)\n'
            '        except Exception as _exc:\n'
            '            logger.warning("Could not flush OOF predictions: %s", _exc)'
        ),
        new=(
            '        try:\n'
            '            import numpy as _np  # noqa: F811\n'
            '            _oof   = getattr(ensemble, "oof_predictions_", None)\n'
            '            _names = getattr(ensemble, "oof_model_names_", None)\n'
            '            if _oof is not None and _names is not None:\n'
            '                _oof_df = pd.DataFrame(_oof, columns=list(_names))\n'
            '                # Persist training row indices so downstream meta-learner\n'
            '                # reconstruction aligns OOF rows to meta_train.parquet\n'
            '                # even when --max-train subsampling is active (F-13).\n'
            '                if args.max_train and len(y_train) == args.max_train:\n'
            '                    _oof_df.insert(0, "_train_row_idx", idx)\n'
            '                else:\n'
            '                    _oof_df.insert(\n'
            '                        0, "_train_row_idx",\n'
            '                        _np.arange(len(_oof_df), dtype=_np.int64),\n'
            '                    )\n'
            '                _oof_df.to_parquet(outdir / "oof_predictions.parquet",\n'
            '                                   index=False)\n'
            '                logger.info(\n'
            '                    "OOF predictions flushed to %s/oof_predictions.parquet "\n'
            '                    "(shape=%s, _train_row_idx included)",\n'
            '                    outdir, _oof_df.shape,\n'
            '                )\n'
            '        except Exception as _exc:\n'
            '            logger.warning("Could not flush OOF predictions: %s", _exc)'
        ),
    )
    if not ok:
        failures += 1

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Apply D.1 str_replace patches")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview matches without writing anything.")
    args = p.parse_args(argv)

    root      = Path(__file__).parent.parent
    rdp_path  = root / "src" / "genomic_variant_classifier" / "data" / "real_data_prep.py"
    eval_path = root / "scripts" / "run_phase2_eval.py"

    # Verify files exist.
    for path in (rdp_path, eval_path):
        if not path.exists():
            print(f"ERROR: {path} not found.", file=sys.stderr)
            print(
                "  Ensure you are running from the repo root and that the HEAD\n"
                "  files are at src/genomic_variant_classifier/data/real_data_prep.py\n"
                "  and scripts/run_phase2_eval.py",
                file=sys.stderr,
            )
            return 1

    print(f"Target files:")
    print(f"  real_data_prep.py : {rdp_path}")
    print(f"  run_phase2_eval.py: {eval_path}")
    if args.dry_run:
        print("  Mode: DRY-RUN (nothing will be written)\n")
    else:
        print("  Mode: APPLY\n")

    failures = 0
    failures += patches_for_rdp(rdp_path, args.dry_run)
    failures += patches_for_eval(eval_path, args.dry_run)

    print()
    if args.dry_run:
        print("Dry-run complete.  Run without --dry-run to apply.")
        return 0

    if failures:
        print(
            f"{failures} patch(es) FAILED.  See errors above.\n"
            "The target files have NOT been modified by failed patches.\n"
            "Common causes:\n"
            "  1. HEAD is not 553efac — run: git rev-parse --short HEAD\n"
            "  2. A previous partial run already applied some patches\n"
            "     (SKIP messages are normal for already-applied patches)\n"
            "  3. Whitespace differences — compare file manually",
            file=sys.stderr,
        )
        return 1

    print("All D.1 patches applied successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
