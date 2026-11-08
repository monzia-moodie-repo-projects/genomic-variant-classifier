"""scripts/derive_run15_oof_identity.py

Author: Monzia Moodie

Write the separately identified DERIVED record of Run 15's out-of-fold row identity (package 1, set 1b, gate G2.3),
and -- with --git-dir -- the code provenance of Run 15's evaluation rows (gate G7.3).

    python scripts/derive_run15_oof_identity.py --run-dir outputs/run15_rerun_report/full \
        --record docs/package1/RUN15_OOF_IDENTITY_DERIVED_RECORD.json \
        --mapping-out outputs/run15_rerun_report/derived/run15_oof_identity_mapping.parquet \
        --git-dir PRESERVED_REPOSITORY --code-commit 4b9de1c319739824955bb165c295e56e444626ce

Reads only; the original OOF file is never rewritten. The orchestrator is read with the inert reader
(genomic_variant_classifier.inert_pickle), so no model is deserialized and no class it names is imported. Exit 0 when the
record is derived and verified, 2 when any check fails (nothing is written then).

CODE PROVENANCE (--git-dir/--code-commit). The run's master log records `HEAD: 4b9de1c`; history was rewritten after the
run, and the pre-rewrite commits are preserved under refs/preserved/pre-rewrite/ and in the bundle pre_rewrite_commits.bundle
(sha256 c9b29cb6b75bd1b9611d3a013c9c3bf1eeb469a0ec11a8592aa2e2a46a73c0b7). PRESERVED_REPOSITORY is any repository holding that
commit, for example a new bare repository after `git fetch BUNDLE 'refs/preserved/pre-rewrite/*:refs/preserved/pre-rewrite/*'`
(what the set-1b installer does); git's output is decoded as UTF-8 on every system. The script reads the run's own
variant_ensemble.py and run_phase2_eval.py from that commit and records, as text found in that code (not inferred):
that predict_proba returned the convex blend `base_preds @ self.blend_weights_` whenever blend weights were stored, that
evaluate() labelled that blend's row ENSEMBLE_STACKER and called it with a strict `> 0.5` threshold, and that the run
evaluated the in-memory fitted ensemble before saving it. Together with the stored, non-null blend weights this is the
provenance that attributes Run 15's ENSEMBLE_STACKER rows to the stored blend under the historical `@` kernel.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("derive_run15_oof_identity")

_ENSEMBLE = "src/genomic_variant_classifier/models/variant_ensemble.py"
_DRIVER = "scripts/run_phase2_eval.py"
_FACTS = {
    _ENSEMBLE: {
        "predict_proba_returns_the_stored_blend": "blend = base_preds @ self.blend_weights_",
        "blend_used_whenever_weights_are_stored": "if self.blend_weights_ is not None:",
        "evaluate_row_label": 'results["ENSEMBLE_STACKER"] = {',
        "evaluate_row_scores_predict_proba": "ens_proba = self.predict_proba(X_tab, X_seq)[:, 1]",
        "evaluate_row_threshold_strictly_greater": "ens_preds = (ens_proba > 0.5).astype(int)",
        "legacy_carve": "idx_fit, idx_cal = _tts(",
    },
    _DRIVER: {
        "test_metrics_from_in_memory_evaluate": "results = ensemble.evaluate(X_test, seq_te, y_test)",
        "oof_train_row_idx_sequential": '"_train_row_idx",',
    },
}


def _git(git_dir: Path, *args: str) -> str:
    # UTF-8, strictly, on every system: never the locale's code page (cp1252 on Windows would decode these files differently).
    r = subprocess.run(["git", "-C", str(git_dir), *args], capture_output=True, encoding="utf-8", errors="strict",
                       check=False)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed ({r.returncode}): {r.stderr.strip()}")
    return r.stdout


def code_provenance(git_dir: Path, commit: str) -> dict:
    full = _git(git_dir, "rev-parse", "--verify", f"{commit}^{{commit}}").strip()
    record = {"commit": full, "commit_subject": _git(git_dir, "log", "-1", "--format=%s", full).strip(),
              "commit_date": _git(git_dir, "log", "-1", "--format=%cI", full).strip(), "files": {}}
    for path, facts in _FACTS.items():
        text = _git(git_dir, "show", f"{full}:{path}")
        found = {name: (needle in text) for name, needle in facts.items()}
        missing = [k for k, v in found.items() if not v]
        if missing:
            raise RuntimeError(f"{path} at {full} lacks the expected code {missing}; the attribution is not proven")
        record["files"][path] = {"git_blob": _git(git_dir, "rev-parse", f"{full}:{path}").strip(),
                                 "facts_found_verbatim": {k: facts[k] for k in found}}
    record["attribution"] = (
        "ENSEMBLE_STACKER rows written by this run are the stored convex blend computed with the historical kernel "
        "numpy_float64_matmul_stored_weights_v1 (P @ w) over the in-memory fitted members, with model calls at p > 0.5 "
        "(not the current positive_if_p_ge_0.5); they are not the logistic stacker")
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--record", required=True, type=Path)
    p.add_argument("--mapping-out", type=Path, default=None)
    p.add_argument("--git-dir", type=Path, default=None)
    p.add_argument("--code-commit", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    if (args.git_dir is None) != (args.code_commit is None):
        logger.error("--git-dir and --code-commit go together")
        return 2

    from genomic_variant_classifier.evaluation.historical_oof_identity import derive_legacy_oof_identity
    from genomic_variant_classifier.evaluation.serialization import dump_strict_json
    from genomic_variant_classifier.prediction_contract import ContractError

    try:
        provenance = code_provenance(args.git_dir, args.code_commit) if args.git_dir else None
        record = derive_legacy_oof_identity(args.run_dir, mapping_out=args.mapping_out, code_provenance=provenance)
    except (ContractError, RuntimeError, FileNotFoundError) as exc:
        logger.error("NOT derived: %s", exc)
        return 2
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(dump_strict_json(record, artifact=str(args.record)) + "\n", encoding="utf-8",
                           newline="\n")
    logger.info("Record written: %s (record_sha256 %s)", args.record, record["record_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
