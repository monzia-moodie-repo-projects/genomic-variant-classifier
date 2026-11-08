#!/usr/bin/env python
"""run_conformal_calibration.py -- CLI for conformal calibration on a verified substrate.

Runs the non-bypassable alignment gate, gene-disjoint calibration, and coverage diagnostics, then
writes a JSON + markdown report. All outputs are labeled 'pre-correction, methodological only'.

Package 1, set 1c (2026-11-08): the default score is `selected_score` (the stored convex blend a completed run writes);
`--run-dir` reads a completed run's BOUND split predictions and the report names the predictor, run, file and rows.
`--substrate` remains for a pre-1b file, whose score column (`ensemble_prob`) must then be named, and the report says
the substrate is unbound.

Examples:
  python scripts/run_conformal_calibration.py --run-dir outputs/run16/full --role test \
      --alpha 0.1 --cal-frac 0.5 --seed 42 --outdir outputs/conformal_run16

  python scripts/run_conformal_calibration.py \
      --substrate outputs/ablation_run15/full/test_predictions.parquet \
      --score-col ensemble_prob --label-col label --group-col gene_symbol \
      --stratum-col consequence --alpha 0.1 --cal-frac 0.5 --seed 42 \
      --outdir outputs/conformal_demo
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

# Support both package and direct-script execution.
try:
    from genomic_variant_classifier.conformal.calibrate import (
        CalibrationConfig, calibrate, calibrate_completed_run, AlignmentError, DISCLAIMER)
    from genomic_variant_classifier.evaluation.alignment import (
        DEFAULT_SCORE_LABEL_ALIGNMENT_POLICY, ScoreLabelAlignmentPolicy)
    from genomic_variant_classifier.evaluation.prediction_artifacts import IncompleteRunError
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from genomic_variant_classifier.conformal.calibrate import (
        CalibrationConfig, calibrate, calibrate_completed_run, AlignmentError, DISCLAIMER)
    from genomic_variant_classifier.evaluation.alignment import (
        DEFAULT_SCORE_LABEL_ALIGNMENT_POLICY, ScoreLabelAlignmentPolicy)
    from genomic_variant_classifier.evaluation.prediction_artifacts import IncompleteRunError


def _md_report(res, args) -> str:
    c = res.coverage
    lines = [
        "# Conformal calibration report",
        "",
        f"**Disclaimer:** {DISCLAIMER}.",
        "",
        f"- Substrate: `{args.substrate or args.run_dir}`"
        + (f" (bound: run `{res.substrate['run_id']}/{res.substrate['ablation']}`, {res.substrate['role']} "
           f"predictions sha256 `{res.substrate['predictions_sha256']}`, predictor "
           f"`{res.substrate['combiner_sha256']}`)" if res.substrate.get("bound")
           else f" (UNBOUND file, sha256 `{res.substrate.get('sha256')}`)"),
        f"- Score column: `{args.score_col}`  Label: `{args.label_col}`  "
        f"Group: `{args.group_col}`  Stratum: `{args.stratum_col}`",
        f"- alpha = {args.alpha} (target coverage {1 - args.alpha:.2f})",
        f"- Rows: {res.n_total:,} total; {res.n_cal:,} calibration / {res.n_eval:,} evaluation "
        f"(gene-disjoint)",
        f"- Substrate AUROC (gate): {res.auroc:.4f}",
        f"- LAC threshold q_hat: {res.q_hat_lac:.4f}",
        "",
        "## Coverage (evaluation split, held-out genes)",
        f"- Marginal coverage: {c['marginal_coverage']:.4f} "
        f"(target {c['target']:.2f}, within tolerance: {c['marginal_ok']})",
        f"- Per-class coverage: {c['per_class_coverage']}",
        f"- Mean set size: {c['set_size']['mean']:.3f}",
        f"- Abstention (empty-set) rate: {c['abstention']['empty_rate']:.4f}",
    ]
    if "per_stratum_coverage" in c:
        lines.append("")
        lines.append("## Per-stratum coverage (by consequence)")
        for k, v in c["per_stratum_coverage"].items():
            lines.append(f"- {k}: n={v['n']:,} coverage={v['coverage']:.4f}")
    if "group_coverage_any" in c:
        lines.append("")
        lines.append(f"## Gene-level coverage: any={c['group_coverage_any']:.4f} "
                     f"all={c['group_coverage_all']:.4f}")
    lines.append("")
    lines.append("## Mondrian (class-conditional) comparison")
    lines.append(f"- Marginal: {res.mondrian_coverage['marginal']:.4f}")
    lines.append(f"- Per-class: {res.mondrian_coverage['per_class']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Conformal calibration on a verified substrate.")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", help="a COMPLETED run directory: its bound split predictions are calibrated")
    source.add_argument("--substrate", help="an unbound predictions parquet (pre-1b files: name --score-col)")
    ap.add_argument("--role", default="test", help="with --run-dir: the split whose predictions are used")
    ap.add_argument("--score-col", default="selected_score")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--group-col", default="gene_symbol")
    ap.add_argument("--stratum-col", default="consequence")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--cal-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    # ALIGNMENT-1. The old spelling is kept as an ALIAS because external
    # callers may exist; the destination and the default now come from
    # the shared policy, so there is one number, not three.
    ap.add_argument("--alignment-sanity-floor", "--auroc-floor",
                    dest="alignment_sanity_floor", type=float,
                    default=DEFAULT_SCORE_LABEL_ALIGNMENT_POLICY.minimum_auroc)
    ap.add_argument("--outdir", default="outputs/conformal_demo")
    a = ap.parse_args()

    cfg = CalibrationConfig(
        score_col=a.score_col, label_col=a.label_col, group_col=a.group_col,
        stratum_col=a.stratum_col, alpha=a.alpha, cal_frac=a.cal_frac,
        seed=a.seed,
        score_label_alignment_policy=ScoreLabelAlignmentPolicy(
            minimum_auroc=a.alignment_sanity_floor))

    print(f"Calibrating on {a.substrate or a.run_dir} "
          f"(score<->label alignment minimum AUROC "
          f"{a.alignment_sanity_floor}) ...")
    try:
        res = (calibrate_completed_run(a.run_dir, a.role, cfg) if a.run_dir else calibrate(a.substrate, cfg))
    except AlignmentError as e:
        print(f"ABORT: {e}")
        return 1
    except IncompleteRunError as e:
        print(f"NOT A VERIFIED COMPLETED RUN: {e}")
        return 2
    except (ValueError, RuntimeError) as e:
        print(f"ERROR: {e}")
        return 2

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "conformal_calibration_summary.json").write_text(
        json.dumps(asdict(res), indent=2, default=str))
    (outdir / "conformal_calibration_report.md").write_text(_md_report(res, a))
    print(f"WROTE: {outdir}/conformal_calibration_summary.json")
    print(f"WROTE: {outdir}/conformal_calibration_report.md")
    print(f"Marginal coverage {res.coverage['marginal_coverage']:.4f} "
          f"(target {1 - a.alpha:.2f}); {DISCLAIMER}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
