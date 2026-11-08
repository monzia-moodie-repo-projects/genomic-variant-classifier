"""
scripts/export_model.py
========================
Serialise the fitted ensemble of a COMPLETED run into an ``InferencePipeline`` joblib artefact.

Author: Monzia Moodie

What this script does (package 1, set 1c, 2026-11-08 -- gate G7.2)
------------------------------------------------------------------
1. Verifies the run (prediction_artifacts.verify_completed_run): an export is a deployment candidate, so its source
   must be complete evidence. An incomplete run is refused (exit 1).
2. Loads ``models/ensemble.joblib`` through the ONE admission route (``VariantEnsemble.load``) after checking that the
   file is the one the completion manifest lists, with the bytes it recorded; the ensemble's declared combiner must be
   the predictor the manifest names (the run's predictions came from it).
3. Loads ``scaler.joblib`` (when the run has one) from bytes verified against the manifest -- never a bare
   ``joblib.load`` of an unlisted file.
4. Builds the pipeline with metadata bound to the selected predictor: ``evaluation_evidence`` recomputed from the run's
   bound split predictions (prediction_artifacts.selected_predictor_evaluation_evidence) and ``fit_input_rows`` from
   the ensemble. The old ``val_auroc`` / ``n_train`` copied from an unbound metrics.json (defaulting to 0.0 and 0) are
   gone; the feature list is the one the pipeline renders, never a silent fallback.
5. Runs a smoke test on three known ClinVar variants, then writes ``--output``.

``--exclude-models`` is refused (owner ruling 2026-09-23, point 3): an ensemble with members removed is a different
model and must be trained, evaluated and registered as one.

Usage
-----
  python scripts/export_model.py export --input outputs/run16/full --output models/phase2_pipeline.joblib
  python scripts/export_model.py verify models/phase2_pipeline.joblib

Exit codes
----------
    0  artefact written (and smoke test passed)
    1  input not a verified completed run, required file missing, or smoke test failed
    2  unexpected error
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

from genomic_variant_classifier.containment import ContainmentError

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("export_model")

# ---------------------------------------------------------------------------
# Smoke-test variants (ClinVar-reviewed, deterministic)
# ---------------------------------------------------------------------------

# Smoke-test variants use only features with real training-time data:
#   allele_freq      — gnomAD v4 (real data used in training run)
#   alphamissense_score — real data for missense variants; 0.5 default for others
#   gene_constraint_oe  — gnomAD pLoF oe (real)
#   n_pathogenic_in_gene — computed from ClinVar training labels
#   consequence, chrom — from ClinVar
#
# NOTE: cadd_phred, sift_score, polyphen2_score, revel_score, phylop_score, gerp_score
# were NOT annotated in the training run (connectors ran in stub mode → constant
# defaults).  Providing non-default values would be out-of-distribution for the scaler
# and produce garbage predictions.  engineer_features() fills them with the same
# defaults used at training time.
_SMOKE_VARIANTS = [
    # BRCA1 c.68_69del — Pathogenic (ClinVar RCV000031443)
    {
        "chrom": "17", "pos": 43115726, "ref": "AAC", "alt": "A",
        "consequence": "frameshift_variant",
        "gene_symbol": "BRCA1",
        "allele_freq": 0.0,
        "gene_constraint_oe": 0.08,
        "n_pathogenic_in_gene": 2800,   # key signal: many known pathogenic variants in BRCA1
        "_expected_class": "pathogenic",
    },
    # rs1801133 (MTHFR p.Ala222Val) — Benign (common variant, AF ~0.33)
    {
        "chrom": "1", "pos": 11796321, "ref": "G", "alt": "A",
        "consequence": "missense_variant",
        "gene_symbol": "MTHFR",
        "allele_freq": 0.334,           # key signal: common in population → benign
        "gene_constraint_oe": 0.85,
        "alphamissense_score": 0.08,    # AlphaMissense: benign
        "n_pathogenic_in_gene": 5,
        "_expected_class": "benign",
    },
    # TP53 R175H (missense hotspot) — Pathogenic
    {
        "chrom": "17", "pos": 7675088, "ref": "C", "alt": "T",
        "consequence": "missense_variant",
        "gene_symbol": "TP53",
        "allele_freq": 0.0,
        "gene_constraint_oe": 0.05,
        "alphamissense_score": 0.98,    # AlphaMissense: strongly pathogenic
        "n_pathogenic_in_gene": 3200,
        "_expected_class": "pathogenic",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_listed(run_dir: Path, name: str, manifest: dict) -> bytes:
    """The bytes of a file the completion manifest lists, read once and compared with what it recorded."""
    import hashlib

    entry = (manifest.get("artefacts") or {}).get(name)
    if not isinstance(entry, dict):
        raise FileNotFoundError(f"{run_dir}: the completion manifest does not list {name}")
    data = (run_dir / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != entry.get("sha256") or len(data) != entry.get("bytes"):
        raise FileNotFoundError(f"{run_dir}: {name} is not the file the completed run recorded")
    return data


def _load_ensemble_from_run(run_dir: Path, manifest: dict):
    """The run's ``models/ensemble.joblib``, checked against the completion manifest, through the admission route."""
    name = "models/ensemble.joblib"
    _require_listed(run_dir, name, manifest)
    logger.info("Loading model from %s", run_dir / name)
    # Through the ONE admission route: refused before deserialization without a trusted binding.
    from genomic_variant_classifier.model_admission import repository_registry_path
    from genomic_variant_classifier.models.variant_ensemble import VariantEnsemble
    return VariantEnsemble.load(run_dir / name, consumer="scripts/export_model.py",
                                registry_path=repository_registry_path())


def _load_scaler(run_dir: Path, manifest: dict):
    """The run's fitted StandardScaler from bytes verified against the completion manifest, or None when the run
    recorded none (DataPrepConfig.scale_features=False)."""
    import io

    if "scaler.joblib" not in (manifest.get("artefacts") or {}):
        logger.warning("The completed run lists no scaler.joblib: the InferencePipeline will not scale (correct only "
                       "if DataPrepConfig.scale_features=False was used at training).")
        return None
    data = _require_listed(run_dir, "scaler.joblib", manifest)
    logger.info("Loading scaler from %s (bytes verified against the completion manifest)", run_dir / "scaler.joblib")
    return joblib.load(io.BytesIO(data))


def _smoke_test(pipeline) -> bool:
    """
    Run three known variants through the pipeline and check score ordering.

    Asserts:
      * P(pathogenic | BRCA1 frameshift) > 0.7
      * P(pathogenic | MTHFR common)     < 0.3
      * P(pathogenic | TP53 R175H)       > 0.7
    """
    logger.info("Running smoke test on 3 known ClinVar variants ...")

    rows_in = [{k: v for k, v in row.items() if k != "_expected_class"}
               for row in _SMOKE_VARIANTS]
    expected = [row["_expected_class"] for row in _SMOKE_VARIANTS]

    try:
        scores = pipeline.predict_proba(pd.DataFrame(rows_in))
    except ContainmentError:
        raise   # a refusal is not a smoke-test failure
    except Exception as exc:
        logger.error("Smoke test: predict_proba raised %s: %s", type(exc).__name__, exc)
        return False

    thresholds = {"pathogenic": (0.7, "above 0.70"), "benign": (0.3, "below 0.30")}
    all_pass = True

    for score, exp_class in zip(scores, expected):
        thresh, label = thresholds[exp_class]
        passed = (score > thresh) if exp_class == "pathogenic" else (score < thresh)
        status = "PASS" if passed else "FAIL"
        logger.info(
            "  [%s] expected=%-11s  score=%.4f  threshold=%s",
            status, exp_class, float(score), label,
        )
        if not passed:
            all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_export(args: argparse.Namespace) -> int:
    from genomic_variant_classifier.api.pipeline import SERVING_SCOPE, InferencePipeline
    from genomic_variant_classifier.evaluation.prediction_artifacts import (
        IncompleteRunError,
        selected_predictor_evaluation_evidence,
        verify_completed_run,
    )
    from genomic_variant_classifier.prediction_contract import ContractError

    run_dir = Path(args.input)
    if not run_dir.exists():
        logger.error("Input directory not found: %s", run_dir)
        return 1

    if args.exclude_models:
        # Owner ruling 2026-09-23, point 3: an ensemble with members removed is a DIFFERENT model, never
        # constructed at export. Its blend weights were fitted on the full roster.
        raise ContainmentError(
            f"--exclude-models {sorted(args.exclude_models)} refused: an ensemble with members removed is a "
            "different model and must be trained, evaluated and registered as one")

    try:
        manifest = verify_completed_run(run_dir)
    except IncompleteRunError as exc:
        logger.error("REFUSED: %s. An exported pipeline is a deployment candidate; it is built only from a completed "
                     "run (a run directory with a verified completion_manifest.json).", exc)
        return 1

    try:
        ensemble = _load_ensemble_from_run(run_dir, manifest)
        scaler = _load_scaler(run_dir, manifest)
        spec = ensemble.convex_blend_spec(SERVING_SCOPE)
        evidence = selected_predictor_evaluation_evidence(run_dir, spec.digest)
        pipeline = InferencePipeline.from_variant_ensemble(ensemble, scaler=scaler, evaluation_evidence=evidence)
        for role, split in evidence["splits"].items():
            logger.info("Bound evidence (%s): AUROC %.17g on %d rows (%d positive); predictions sha256 %s",
                        role, split["auroc"], split["n_rows"], split["n_positive"], split["predictions_sha256"])
        logger.info("fit_input_rows=%s  features=%d  combiner=%s", pipeline.metadata.fit_input_rows,
                    pipeline.metadata.n_features, spec.digest)

        # Smoke test before writing
        if not args.skip_smoke_test:
            passed = _smoke_test(pipeline)
            if not passed:
                logger.error(
                    "Smoke test FAILED — artefact NOT written. "
                    "Investigate model quality before deploying."
                )
                return 1
            logger.info("Smoke test passed.")

        pipeline.save(args.output)
        logger.info("InferencePipeline written to %s", args.output)
        return 0

    except ContainmentError:
        raise   # a refusal, not an export failure
    except (FileNotFoundError, IncompleteRunError, ContractError) as exc:
        logger.error("REFUSED: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Export failed: %s", exc)
        return 2


def cmd_verify(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        logger.error("Artefact not found: %s", path)
        return 1

    logger.info("Loading %s ...", path)
    from genomic_variant_classifier.api.pipeline import InferencePipeline
    from genomic_variant_classifier.model_admission import repository_registry_path
    pipeline = InferencePipeline.load(path, consumer="scripts/export_model.py",
                                      registry_path=repository_registry_path())
    passed = _smoke_test(pipeline)
    return 0 if passed else 1


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained pipeline to an InferencePipeline joblib artefact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    exp = sub.add_parser("export", help="Wrap and export a trained model.")
    exp.add_argument(
        "--input", "-i", type=Path, required=True,
        help="a COMPLETED run directory (verified completion_manifest.json listing models/ensemble.joblib).",
    )
    exp.add_argument(
        "--output", "-o", type=Path, default=Path("models/phase2_pipeline.joblib"),
        help="Destination path for the InferencePipeline artefact.",
    )
    exp.add_argument(
        "--exclude-models", nargs="*", metavar="MODEL",
        default=[],
        help=(
            "REFUSED (owner ruling 2026-09-23, point 3): an ensemble with members removed is a different model and "
            "must be trained, evaluated and registered as one. Kept only so a caller that passes it is refused "
            "explicitly instead of failing on an unknown argument."
        ),
    )
    exp.add_argument(
        "--skip-smoke-test", action="store_true",
        help="Skip the 3-variant smoke test after export.",
    )

    ver = sub.add_parser("verify", help="Smoke-test an existing artefact.")
    ver.add_argument("path", type=Path, help="Path to an InferencePipeline joblib.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    # Default to 'export' when no sub-command supplied (legacy usage).
    # Must be done before parse_args() — argparse exits on unrecognised top-level
    # arguments before we can inspect args.command.
    _subcommands = {"export", "verify", "-h", "--help"}
    if len(sys.argv) < 2 or sys.argv[1] not in _subcommands:
        sys.argv.insert(1, "export")

    args = parse_args()

    if args.command == "export":
        return cmd_export(args)
    if args.command == "verify":
        return cmd_verify(args)

    logger.error("Unknown command: %s", args.command)
    return 1


if __name__ == "__main__":
    sys.exit(main())
