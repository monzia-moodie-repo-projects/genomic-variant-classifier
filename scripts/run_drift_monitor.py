"""
scripts/run_drift_monitor.py
==============================
Scheduled drift monitoring CLI for the Genomic Variant Classifier.

Run this script:
  - Monthly (on each ClinVar release)
  - On-demand after any upstream data source update
  - Via cron / GitHub Actions scheduled workflow

What it does:
  1. Checks feature drift (PSI / KS / MMD) against the training reference
  2. Checks label drift (ClinVar reclassifications) against old release
  3. Writes a structured JSON report
  4. Optionally triggers retraining if drift exceeds thresholds
  5. Optionally exports an Evidently AI HTML dashboard
  6. Exits with code 0 (no action) or 2 (retraining recommended) so
     CI/CD can gate on the exit code

Usage:
    python scripts/run_drift_monitor.py \\
        --reference-splits  outputs/phase2_with_gnomad/splits/ \\
        --new-clinvar       data/processed/clinvar_grch38_2024_07.parquet \\
        --old-clinvar       data/processed/clinvar_grch38_2024_01.parquet \\
        --current-model     models/phase2_pipeline.joblib \\
        --output-dir        outputs/drift_reports/2024_07/ \\
        --registry          models/registry.json \\
        --release-name      2024_07 \\
        --auto-retrain

    # Check features only (no label drift, no retraining):
    python scripts/run_drift_monitor.py \\
        --reference-splits outputs/phase2_with_gnomad/splits/ \\
        --new-data         data/processed/gnomad_v5_exomes.parquet \\
        --features-only

Exit codes:
    0 = no drift detected, no action required
    1 = monitoring recommended (PSI in yellow zone)
    2 = retraining recommended (PSI in red zone or label drift)
    3 = urgent retraining (severe drift, high weighted flip rate)
    4 = NOT CHECKED -- see EXIT_NOT_CHECKED below. This is NOT a drift verdict.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

#: "I could not look" -- distinct from every code that means "I looked".
#:
#: Added 2026-07-13 (roadmap 6.20). The original defect in this subsystem was that a run which
#: measured NOTHING exited 0, and 0 means "no drift" -- so the scheduled monitor reported a
#: clean bill of health every month, with a green tick, having never read a row of data.
#:
#: The obvious fix -- make the not-checked paths exit 3 -- is the SAME BUG WEARING THE OPPOSITE
#: COSTUME. Exit 3 means `urgent_retrain`, and the workflow maps it to drift_level=severe. A
#: run that checked nothing would then fire a SEVERE DRIFT alarm on a model that may be
#: perfectly healthy. "Not checked" reported as catastrophe is exactly as false as "not
#: checked" reported as clean, and it is the faster route to a monitor everyone learns to
#: ignore.
#:
#: So it gets its own code. A measurement that did not happen is not a measurement, in EITHER
#: direction, and the workflow maps 4 to drift_level=UNKNOWN.
EXIT_NOT_CHECKED = 4

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("run_drift_monitor")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Drift monitor for the Genomic Variant Classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── The reference distribution: EXACTLY ONE of these (roadmap 6.20) ──────────────
    #
    # --reference-splits : the raw matrix. Full fidelity, including the joint Maximum Mean
    #                      Discrepancy and Székely-Rizzo energy tests. Needs the 23.8 MB
    #                      cohort matrix, so it runs where that data lives.
    #
    # --reference-profile: the aggregate profile (histograms + quantile grids). 1.4 MB, no
    #                      variant rows, committed to git -- so it works on a hosted runner
    #                      with no credentials and no cloud fetch. Population Stability Index
    #                      is IDENTICAL; the joint tests are reported as NOT COMPUTED.
    #
    # `required=True` on the group: a drift monitor with no reference is not a drift monitor.
    # It must not be possible to invoke this script such that it silently checks nothing --
    # that is the whole of roadmap 6.20.
    ref = p.add_mutually_exclusive_group(required=True)
    ref.add_argument("--reference-splits", type=Path, default=None,
                     help="Directory containing X_train.parquet (raw reference distribution). "
                          "Full fidelity, including the joint MMD/energy tests. Requires the "
                          "cohort matrix, so it runs where that data lives.")
    ref.add_argument("--reference-profile", type=Path, default=None,
                     help="Aggregate-only reference profile JSON (see "
                          "scripts/build_drift_reference_profile.py). Exact PSI, no raw data, "
                          "safe for hosted CI. Joint MMD/energy tests are NOT COMPUTED.")
    p.add_argument("--new-clinvar",  type=Path, default=None,
                   help="New ClinVar parquet to check for label drift.")
    p.add_argument("--old-clinvar",  type=Path, default=None,
                   help="Previous ClinVar parquet (for reclassification comparison).")
    p.add_argument("--new-data",     type=Path, default=None,
                   help="Any new feature parquet for covariate drift check.")
    p.add_argument("--current-model", type=Path, default=Path("models/phase2_pipeline.joblib"),
                   help="Path to the current production InferencePipeline.")
    p.add_argument("--gnomad",        type=Path, default=None)
    p.add_argument("--alphamissense", type=Path, default=None)
    p.add_argument("--output-dir",    type=Path, default=Path("outputs/drift_reports"),
                   help="Directory for reports and artefacts.")
    p.add_argument("--registry",      type=Path, default=Path("models/registry.json"),
                   help="Path to the model registry JSON.")
    p.add_argument("--release-name",  type=str,  default="latest",
                   help="Label for this release (e.g. '2024_07').")
    p.add_argument("--old-release-name", type=str, default="previous")
    p.add_argument("--auto-retrain",  action="store_true",
                   help="Automatically trigger retraining if drift is detected.")
    p.add_argument("--features-only", action="store_true",
                   help="Run feature drift check only (skip label drift).")
    p.add_argument("--evidently",     action="store_true",
                   help="Generate an Evidently AI HTML drift dashboard.")

    # --- Confidence-Based Performance Estimation (CBPE) -- added 2026-07-13, roadmap 6.19 ---
    #
    # Estimate the model's performance on the NEW, UNLABELLED release. A new ClinVar release
    # has no adjudicated labels and will not for months; CBPE estimates ROC AUC / accuracy
    # from the PREDICTED PROBABILITIES alone, calibrated against a labelled reference period.
    # If the estimate collapses, the model is degrading on real data and we know BEFORE anyone
    # gets a wrong variant call.
    #
    # This is the capability requirements.in has claimed since 2026-05 ("nannyml (CBPE)") and
    # which, until today, HAD NEVER BEEN BUILT -- nannyml was imported by exactly one file, a
    # script that printed its version number.
    #
    # REQUIRES THE ISOLATED DRIFT ENVIRONMENT (requirements-drift.txt): nannyml demands
    # lightgbm<4.6 while the ensemble TRAINS on lightgbm 4.6.0. See roadmap 6.19.
    p.add_argument("--estimate-performance", action="store_true",
                   help="Estimate performance on the UNLABELLED new release (nannyml CBPE). "
                        "Requires the isolated drift environment (requirements-drift.txt).")
    p.add_argument("--reference-predictions", type=Path, default=None,
                   help="Parquet with the LABELLED reference period: columns y_true and "
                        "y_pred_proba. Use the ensemble's OUT-OF-FOLD probabilities "
                        "(VariantEnsemble.oof_predictions_), never its in-sample ones -- "
                        "in-sample probabilities are optimistic and would mis-calibrate every "
                        "future estimate.")
    p.add_argument("--analysis-predictions", type=Path, default=None,
                   help="Parquet with the UNLABELLED new release: column y_pred_proba. Must "
                        "NOT contain y_true -- if you have labels, MEASURE the metric; an "
                        "estimate is strictly worse than a measurement.")
    p.add_argument("--cbpe-chunk-size", type=int, default=None,
                   help="Rows per CBPE chunk. Default: sized for ~10 chunks, below which "
                        "nannyml's sampling-error estimate degrades and the confidence bands "
                        "stop meaning anything.")

    p.add_argument("--psi-threshold", type=float, default=0.25,
                   help="PSI threshold for retraining trigger.")
    p.add_argument("--flip-rate-threshold", type=float, default=0.010,
                   help="ClinVar flip rate threshold for retraining trigger.")
    return p


def run_performance_estimation(args: argparse.Namespace) -> int:
    """Estimate performance on the UNLABELLED release. Returns an exit-code fragment.

    Exit codes:
        0 = no alert -- estimated performance is within the reference band
        2 = ALERT    -- estimated performance has crossed nannyml's threshold on data with no
                        labels. The model may be degrading on the real world. Investigate.
        3 = the estimation could not be run (missing inputs, or the drift environment is not
            active). NEVER silently skipped: a performance estimate that did not happen must
            not be mistaken for one that came back clean.
    """
    import json

    import pandas as pd

    from genomic_variant_classifier.monitoring.performance_estimator import (
        NannyMLUnavailableError,
        build_analysis_frame,
        build_reference_frame,
        estimate_performance,
    )

    if not args.reference_predictions or not args.reference_predictions.exists():
        logger.error(
            "--estimate-performance requires --reference-predictions (a parquet with y_true "
            "and y_pred_proba). Got: %s. Refusing to skip silently.",
            args.reference_predictions,
        )
        return 3
    if not args.analysis_predictions or not args.analysis_predictions.exists():
        logger.error(
            "--estimate-performance requires --analysis-predictions (a parquet with "
            "y_pred_proba, and NO y_true). Got: %s. Refusing to skip silently.",
            args.analysis_predictions,
        )
        return 3

    ref_raw = pd.read_parquet(args.reference_predictions)
    ana_raw = pd.read_parquet(args.analysis_predictions)
    logger.info(
        "CBPE: reference %d rows (labelled), analysis %d rows (UNLABELLED).",
        len(ref_raw), len(ana_raw),
    )

    for col in ("y_true", "y_pred_proba"):
        if col not in ref_raw.columns:
            logger.error("--reference-predictions is missing column %r.", col)
            return 3
    if "y_pred_proba" not in ana_raw.columns:
        logger.error("--analysis-predictions is missing column 'y_pred_proba'.")
        return 3

    feature_cols = [
        c for c in ref_raw.columns
        if c not in {"y_true", "y_pred", "y_pred_proba"} and c in ana_raw.columns
    ]

    reference = build_reference_frame(
        y_true=ref_raw["y_true"].to_numpy(),
        y_pred_proba=ref_raw["y_pred_proba"].to_numpy(),
        features=ref_raw[feature_cols] if feature_cols else None,
    )
    analysis = build_analysis_frame(
        y_pred_proba=ana_raw["y_pred_proba"].to_numpy(),
        features=ana_raw[feature_cols] if feature_cols else None,
    )

    try:
        est = estimate_performance(reference, analysis, chunk_size=args.cbpe_chunk_size)
    except NannyMLUnavailableError as exc:
        # FAIL LOUD. The old drift code caught ImportError and logged "not installed" while the
        # package WAS installed -- the report was silently never produced for months. Never
        # again: say exactly what is wrong and exit non-zero.
        logger.error("Confidence-Based Performance Estimation could not run:\n%s", exc)
        return 3

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"performance_estimate_{args.release_name}.json"
    payload = {
        "release": args.release_name,
        "n_reference": est.n_reference,
        "n_analysis": est.n_analysis,
        "chunk_size": est.chunk_size,
        "estimated_on_unlabelled_data": est.analysis_summary,
        "alerts": est.alerts,
        "any_alert": est.any_alert,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    est.estimates.to_csv(
        args.output_dir / f"performance_estimate_{args.release_name}.csv", index=False
    )
    logger.info("CBPE report -> %s", out)

    if est.any_alert:
        flagged = sorted(m for m, a in est.alerts.items() if a)
        logger.error(
            "CBPE ALERT on %s: estimated performance crossed the threshold for %s WITHOUT any "
            "ground-truth labels. Estimates: %s",
            args.release_name, ", ".join(flagged),
            {m: round(v, 4) for m, v in est.analysis_summary.items()},
        )
        return 2
    return 0


def run_feature_drift(args: argparse.Namespace) -> int:
    """Returns exit code fragment from feature drift check."""
    import pandas as pd
    from genomic_variant_classifier.monitoring.drift_detector import DriftDetector

    # ── The reference: raw matrix, or the aggregate profile (roadmap 6.20) ────────────
    #
    # EXACTLY ONE of these is supplied; argparse enforces it. The profile is how the hosted
    # monthly run works: it is a 1.4 MB committed histogram, versus a 23.8 MB cohort matrix
    # that would need cloud storage and credentials on every run. Population Stability Index
    # is IDENTICAL either way; only the joint Maximum Mean Discrepancy / energy tests are
    # lost, and those are reported as NOT COMPUTED, never as passing.
    X_ref = None
    if args.reference_profile:
        detector = DriftDetector.from_profile(args.reference_profile)
        ref_cols = detector.feature_names
        logger.info(
            "Reference: AGGREGATE PROFILE, %d features, %d rows summarised.",
            len(ref_cols), detector.profile.n_ref_samples,
        )
    else:
        splits_dir = args.reference_splits
        if not (splits_dir / "X_train.parquet").exists():
            logger.error(
                "X_train.parquet not found in %s. This is NOT 'no drift' -- it is 'no "
                "reference', and nothing can be compared. Exiting %d = NOT CHECKED.",
                splits_dir, EXIT_NOT_CHECKED,
            )
            return EXIT_NOT_CHECKED

        X_ref = pd.read_parquet(splits_dir / "X_train.parquet")
        ref_cols = list(X_ref.columns)
        logger.info("Reference: %d samples × %d features", *X_ref.shape)

        detector = DriftDetector.from_reference(
            X_ref=X_ref,
            save_path=args.output_dir / "drift_reference.pkl",
        )

    # ── The new data ─────────────────────────────────────────────────────────────────
    if args.new_data and args.new_data.exists():
        X_new = pd.read_parquet(args.new_data)
    elif args.new_clinvar and args.new_clinvar.exists():
        logger.info("Building feature matrix from new ClinVar …")
        clinvar = pd.read_parquet(args.new_clinvar)
        try:
            from genomic_variant_classifier.api.pipeline import engineer_features
            X_new = engineer_features(clinvar)
        except Exception as e:
            logger.error(
                "Feature engineering failed: %s. Drift was NOT CHECKED -- this is not a drift "
                "verdict. Exiting %d.", e, EXIT_NOT_CHECKED,
            )
            return EXIT_NOT_CHECKED
    else:
        # THIS USED TO `return 0`.
        #
        # Exit 0 means "checked, clean". There was no new data; nothing was checked; and the
        # workflow invokes this script with `--features-only` and NO `--new-data` -- so the
        # monthly drift monitor took this branch EVERY TIME and reported a clean bill of
        # health, for its entire life, having never compared a single distribution.
        #
        # That is the same defect as the placeholder Google Drive download, hiding one level
        # further down, and it would have survived the workflow fix completely. "I had nothing
        # to look at" is not "I looked and saw nothing." (roadmap 6.20)
        logger.error(
            "NO NEW DATA PROVIDED. Feature drift was NOT CHECKED.\n"
            "\n"
            "Pass --new-data (a feature matrix) or --new-clinvar (a ClinVar release to\n"
            "engineer features from). Exiting %d = NOT CHECKED.\n"
            "\n"
            "This branch previously returned 0 (= 'no drift'), which is why the scheduled\n"
            "monitor reported green every month without ever performing a comparison.\n"
            "\n"
            "Note it does NOT exit 3 either: 3 means urgent_retrain, and firing a SEVERE\n"
            "DRIFT alarm for a check that never ran is the same lie in the other direction.",
            EXIT_NOT_CHECKED,
        )
        return EXIT_NOT_CHECKED

    # ── Column alignment: NEVER fabricate a feature ──────────────────────────────────
    #
    # This used to be `X_new.reindex(columns=X_ref.columns, fill_value=0.0)`.
    #
    # If the new release was MISSING a feature the reference has, reindex INVENTED it as a
    # column of zeros and drift-checked it as though it had been measured. Depending on where
    # 0.0 falls in that feature's reference range, the fabricated column reads either as
    # catastrophic drift or as perfect stability -- and BOTH are pure fiction, fed into a
    # monitor whose output can trigger the retraining of a clinical pathogenicity classifier.
    #
    # A feature that was not measured must be reported as not measured.
    missing = [c for c in ref_cols if c not in X_new.columns]
    if missing:
        logger.error(
            "The new data is MISSING %d feature(s) the reference expects: %s%s\n"
            "\n"
            "Refusing to fill them with zeros. A fabricated column is not a measurement, and\n"
            "the resulting Population Stability Index would be meaningless in either\n"
            "direction. Fix the feature-engineering inputs. Exiting %d = NOT CHECKED.",
            len(missing), missing[:15], " ..." if len(missing) > 15 else "",
            EXIT_NOT_CHECKED,
        )
        return EXIT_NOT_CHECKED

    extra = [c for c in X_new.columns if c not in ref_cols]
    if extra:
        logger.warning(
            "%d feature(s) in the new data are absent from the reference and will NOT be "
            "drift-checked: %s%s. The reference is stale relative to the current feature "
            "contract -- rebuild the profile.",
            len(extra), extra[:15], " ..." if len(extra) > 15 else "",
        )
    X_new = X_new[ref_cols]

    logger.info("New data: %d samples", len(X_new))
    report = detector.check(X_new, timestamp=args.release_name)
    report.to_json(args.output_dir / f"feature_drift_{args.release_name}.json")

    # Optional Evidently AI export. Evidently compares two RAW frames -- it cannot work from
    # histograms. In profile mode there is no reference frame to give it, so the export is
    # skipped LOUDLY rather than handed an empty DataFrame that would render a beautiful,
    # meaningless report showing no drift.
    if args.evidently:
        if X_ref is None:
            logger.warning(
                "--evidently requested but the reference is an AGGREGATE PROFILE, which has "
                "no rows for Evidently to compare against. The HTML export is SKIPPED. The "
                "Population Stability Index results above are unaffected and exact. To get an "
                "Evidently report, run against --reference-splits where the raw matrix lives."
            )
        else:
            _export_evidently(X_ref, X_new, report, args.output_dir, args.release_name)

    # Map to exit code. See EXIT_NOT_CHECKED at the top of this module: a drift VERDICT and a
    # failure to MEASURE must never share a code. 3 means "I looked and it is severe". 4 means
    # "I could not look".
    if report.recommended_action == "urgent_retrain":
        return 3
    if report.recommended_action == "retrain":
        return 2
    if report.recommended_action == "monitor":
        return 1
    return 0


def run_label_drift(args: argparse.Namespace) -> tuple[int, object]:
    """Returns (exit_code, label_report)."""
    import pandas as pd
    from genomic_variant_classifier.monitoring.clinvar_tracker import ClinVarTracker

    # Label drift needs meta_test.parquet -- variant identifiers, to detect ClinVar
    # RECLASSIFICATIONS of variants the model was trained on. The aggregate profile is a
    # histogram; it has no identifiers and never will. So this check is simply not available
    # in profile mode, and says so. It does not quietly return "no label drift".
    if args.reference_splits is None:
        logger.error(
            "LABEL DRIFT WAS NOT CHECKED. It requires meta_test.parquet (variant identifiers, "
            "to spot ClinVar reclassifications of trained-on variants), and the run was given "
            "an aggregate reference profile, which contains histograms -- no identifiers. "
            "Run with --reference-splits where the cohort lives. Reporting NOT CHECKED "
            "(exit %d) -- neither clean nor drifted.", EXIT_NOT_CHECKED,
        )
        return EXIT_NOT_CHECKED, None

    splits_dir = args.reference_splits
    meta_path = splits_dir / "meta_test.parquet"
    training_ids: set[str] = set()
    if meta_path.exists():
        meta = pd.read_parquet(meta_path)
        if "variant_id" in meta.columns:
            training_ids = set(meta["variant_id"].astype(str))
    logger.info("Label drift check: tracking %d training variant IDs.", len(training_ids))

    tracker = ClinVarTracker(training_variant_ids=training_ids)
    report  = tracker.compare(
        old_path     = args.old_clinvar,
        new_path     = args.new_clinvar,
        output_dir   = args.output_dir / "temporal_cohorts",
        old_release  = args.old_release_name,
        new_release  = args.release_name,
    )
    report.to_json(args.output_dir / f"label_drift_{args.release_name}.json")

    if report.urgency == "urgent":
        return 3, report
    if report.urgency == "retrain":
        return 2, report
    if report.urgency == "monitor":
        return 1, report
    return 0, report


class EvidentlyUnavailableError(RuntimeError):
    """Evidently could not be used. RAISED, never swallowed into a logger.warning."""


def _export_evidently(
    X_ref, X_new, drift_report, output_dir: Path, release_name: str
) -> Path:
    """Export the Evidently tabular distribution-drift report. PORTED 2026-07-13 (roadmap 6.19).

    WHAT THIS USED TO BE, AND WHY IT NEVER RAN
    ------------------------------------------
    This function called the Evidently 0.4.x API:

        from evidently.report import Report                              # DELETED
        from evidently.metric_preset import DataDriftPreset,             # DELETED
                                            DataQualityPreset            # DELETED
        from evidently.pipeline.column_mapping import ColumnMapping      # DELETED

    ALL FOUR of those names were removed from Evidently two major versions ago. Verified
    2026-07-13 against the installed evidently 0.7.6: `evidently.report`,
    `evidently.metric_preset` and `evidently.pipeline.column_mapping` all raise
    ModuleNotFoundError.

    The old code wrapped that in `except ImportError` and logged:

        "Evidently AI not installed. Run: pip install evidently"

    **Evidently WAS installed.** The message is FALSE. The code misdiagnosed its own failure
    and told the operator to install a package they already had -- who would then see
    "requirement already satisfied" and be none the wiser. And a second `except Exception`
    swallowed every runtime failure into another warning. So the drift report was silently
    never produced, for months, and the only trace was a log line that lied about why.

    THE 0.7 API, VERIFIED BY EXECUTION (not by recollection)
    -------------------------------------------------------
        from evidently import Report, Dataset, DataDefinition
        from evidently.presets import DataDriftPreset, DataSummaryPreset

        dd   = DataDefinition(numerical_columns=[...], categorical_columns=[...])
        dref = Dataset.from_pandas(X_ref, data_definition=dd)
        dcur = Dataset.from_pandas(X_new, data_definition=dd)

        snapshot = Report(metrics=[DataDriftPreset(), DataSummaryPreset()]).run(
            current_data=dcur, reference_data=dref)
        snapshot.save_html(path)

    Two traps, both confirmed on 2026-07-13:

      * `DataQualityPreset` is GONE. Its counterpart is `DataSummaryPreset`.
      * `Report.run()` takes `current_data` FIRST and `reference_data` SECOND -- the REVERSE
        of the old signature. Passing them positionally would silently produce a BACKWARDS
        drift report: it would compare the reference against the new data and call the new
        data the baseline. ALWAYS pass them by keyword. (They are, below.)

    Exercised end-to-end before this port was written: a 3,684,706-byte HTML report was
    produced in the isolated drift environment (plotly 5.24.1).

    THIS FUNCTION NOW FAILS LOUD
    ----------------------------
    It RAISES. It does not degrade into a warning. A drift report that silently does not exist
    is worse than one that errors, because the operator believes they have one.
    """
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset, DataSummaryPreset
    except ImportError as exc:
        raise EvidentlyUnavailableError(
            f"Evidently is not importable in this environment: {exc}\n"
            "\n"
            "The drift monitor runs in a SEPARATE environment (requirements-drift.txt),\n"
            "because evidently requires plotly<6 while this project runs plotly 6.6.0.\n"
            "\n"
            "  python -m venv .venv-drift\n"
            "  .venv-drift/Scripts/pip install -r requirements-drift.txt\n"
            "  .venv-drift/Scripts/pip install -e . --no-deps\n"
            "\n"
            "If the import failed on `evidently.report` / `evidently.metric_preset` /\n"
            "`evidently.pipeline.column_mapping`, you are looking at the OLD 0.4 API. Those\n"
            "modules were DELETED. See this function's docstring, and roadmap 6.19."
        ) from exc

    # Evidently needs to know which columns are numeric and which are categorical. Derive it
    # from the reference frame's dtypes -- the reference IS the schema, by definition.
    numerical = [c for c in X_ref.columns if _is_numeric(X_ref[c])]
    categorical = [c for c in X_ref.columns if c not in numerical]
    logger.info(
        "Evidently: %d numerical, %d categorical columns (derived from the reference schema).",
        len(numerical), len(categorical),
    )

    definition = DataDefinition(
        numerical_columns=numerical or None,
        categorical_columns=categorical or None,
    )
    reference = Dataset.from_pandas(X_ref, data_definition=definition)
    current = Dataset.from_pandas(X_new, data_definition=definition)

    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    # KEYWORDS, DELIBERATELY. `run()` takes current FIRST in 0.7; positional args here would
    # silently invert the comparison and call the NEW data the baseline.
    snapshot = report.run(current_data=current, reference_data=reference)

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"evidently_drift_{release_name}.html"
    snapshot.save_html(str(html_path))

    json_path = output_dir / f"evidently_drift_{release_name}.json"
    snapshot.save_json(str(json_path))

    size = html_path.stat().st_size
    if size < 1024:
        raise EvidentlyUnavailableError(
            f"Evidently wrote {html_path} but it is only {size} bytes -- that is not a report. "
            f"Failing loud rather than leaving a plausible-looking empty artifact behind."
        )
    logger.info("Evidently report -> %s (%.1f MB) and %s", html_path, size / 1e6, json_path)
    return html_path


def _is_numeric(series) -> bool:
    """True if the column is numeric. Kept explicit so the schema derivation is auditable."""
    import pandas as pd

    return pd.api.types.is_numeric_dtype(series)


def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exit_codes = [0]

    # ── Feature drift ─────────────────────────────────────────────────────
    feature_code = run_feature_drift(args)
    exit_codes.append(feature_code)
    logger.info("Feature drift exit code: %d", feature_code)

    # ── Label drift ───────────────────────────────────────────────────────
    label_report = None
    if (
        not args.features_only
        and args.new_clinvar and args.new_clinvar.exists()
        and args.old_clinvar and args.old_clinvar.exists()
    ):
        label_code, label_report = run_label_drift(args)
        exit_codes.append(label_code)
        logger.info("Label drift exit code: %d", label_code)
    else:
        logger.info("Label drift check skipped (--features-only or missing paths).")

    # ── Confidence-Based Performance Estimation ───────────────────────────
    # Added 2026-07-13 (roadmap 6.19). Estimates ROC AUC / accuracy on the NEW release
    # WITHOUT any labels -- months before the ClinVar reclassifications are adjudicated.
    #
    # Feature drift tells you the INPUTS moved. Label drift tells you the LABELS moved. Neither
    # tells you whether the MODEL IS STILL RIGHT. This does, and it does it before anyone gets
    # a wrong variant call.
    if args.estimate_performance:
        perf_code = run_performance_estimation(args)
        exit_codes.append(perf_code)
        logger.info("Performance-estimation exit code: %d", perf_code)
    else:
        logger.info(
            "Performance estimation not requested (--estimate-performance). Feature and label "
            "drift tell you the DATA moved; only this tells you whether the MODEL is still "
            "right on it."
        )

    # ── Overall decision ──────────────────────────────────────────────────
    final_code = max(exit_codes)

    if final_code >= 2 and args.auto_retrain:
        # ------------------------------------------------------------------------------
        # HARD BOUNDARY (roadmap 6.19). Retraining loads the trained ensemble. The ISOLATED
        # DRIFT ENVIRONMENT (requirements-drift.txt) holds LightGBM 4.5.0 and XGBoost 2.1.4 --
        # NOT the 4.6.0 / 3.2.0 the ensemble is trained with -- because nannyml requires
        # lightgbm<4.6. Unpickling a 4.6.0 booster into a 4.5.0 runtime is either a warning
        # nobody reads or silently wrong deserialisation: root pattern (d), a green result from
        # a mutated environment.
        #
        # So: --auto-retrain MUST NOT run in the drift environment. Detect it and refuse,
        # loudly, rather than produce a quietly-corrupt retrain.
        # ------------------------------------------------------------------------------
        try:
            import importlib.metadata as _md
            _lgbm = _md.version("lightgbm")
        except Exception:  # pragma: no cover - lightgbm always present in both envs
            _lgbm = "unknown"
        try:
            _md.version("nannyml")
            _in_drift_env = True
        except Exception:
            _in_drift_env = False

        if _in_drift_env:
            logger.error(
                "REFUSING to --auto-retrain from the ISOLATED DRIFT ENVIRONMENT.\n"
                "  This environment has nannyml installed, which means lightgbm is %s -- NOT\n"
                "  the 4.6.0 the ensemble is trained with (nannyml requires lightgbm<4.6).\n"
                "  Unpickling a 4.6.0 booster here would be silently wrong.\n"
                "\n"
                "  Run the drift monitor here to DETECT. Run --auto-retrain from the TRAINING\n"
                "  environment (.venv312 / requirements.txt) to ACT. See roadmap 6.19.",
                _lgbm,
            )
            return 3

        logger.info("Drift detected. Triggering continual learning pipeline …")
        if not args.current_model.exists():
            logger.error("Current model not found: %s", args.current_model)
            return 3

        # Retraining needs the actual training rows. A histogram cannot be trained on.
        # Without this, ContinualLearner.run() would receive reference_splits_dir=None and
        # fail somewhere deep inside, with a message about something else entirely.
        if args.reference_splits is None:
            logger.error(
                "REFUSING to --auto-retrain from an AGGREGATE REFERENCE PROFILE.\n"
                "  Retraining requires the reference SPLITS -- the actual training rows. The\n"
                "  profile is a histogram: it has bin counts and quantiles, and no variants.\n"
                "  You cannot fit a model to a histogram.\n"
                "\n"
                "  The profile is for DETECTION on a hosted runner. Re-run with\n"
                "  --reference-splits, in the training environment, to ACT."
            )
            return 3

        from genomic_variant_classifier.training.continual_trainer import ContinualLearner, ContinualLearningConfig
        cl_config = ContinualLearningConfig(
            psi_retrain_threshold = args.psi_threshold,
            flip_rate_retrain     = args.flip_rate_threshold,
            auto_retrain          = True,
            output_dir            = str(args.output_dir / "retrain"),
            registry_path         = str(args.registry),
        )
        learner = ContinualLearner(cl_config)
        decision = learner.run(
            reference_splits_dir  = args.reference_splits,
            new_clinvar_path      = args.new_clinvar,
            old_clinvar_path      = args.old_clinvar,
            current_model_path    = args.current_model,
            gnomad_path           = args.gnomad,
            alphamissense_path    = args.alphamissense,
            release_name          = args.release_name,
            old_release_name      = args.old_release_name,
        )
        if decision.get("new_model_path"):
            logger.info("New model artefact: %s", decision["new_model_path"])
            logger.info(
                "Shadow deployment initiated. Promote to production after burn-in with:\n"
                "  python -c \"\n"
                "  from genomic_variant_classifier.monitoring.registry import ModelRegistry\n"
                "  r = ModelRegistry.load('%s')\n"
                "  r.print_summary()\n"
                "  # r.promote('v?.0.0', 'production')\n"
                "  \"", args.registry,
            )
    elif final_code >= 2:
        logger.warning(
            "Drift detected but auto_retrain=False. "
            "Re-run with --auto-retrain or review the reports in %s",
            args.output_dir,
        )

    exit_messages = {
        0: "No significant drift. No action required.",
        1: "Minor drift detected. Increase monitoring frequency.",
        2: "Significant drift detected. Retraining recommended.",
        3: "Severe drift detected. Urgent retraining required.",
    }
    logger.info("EXIT %d: %s", final_code, exit_messages.get(final_code, ""))
    return final_code


if __name__ == "__main__":
    sys.exit(main())