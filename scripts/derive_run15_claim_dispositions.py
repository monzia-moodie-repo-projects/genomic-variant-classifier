"""scripts/derive_run15_claim_dispositions.py

Author: Monzia Moodie

Write (or, with --check, verify) the three claim-disposition records of Run 15 in docs/claims/ from the run's own files
(package 1D, stage 1D-0; owner ruling 2026-11-08). Every figure in the records is measured here; none is typed.

    python scripts/derive_run15_claim_dispositions.py --run-dir outputs/run15_rerun_report/full \
        --recorded-utc 2026-11-08T15:00:00Z --git-dir <pre-rewrite repository> [--check]

--git-dir names a repository holding Run 15's PRE-REWRITE commit 4b9de1c3 (the history was rewritten on 2026-07-11; the
owner's preserved bundle holds it). It is used to record the code facts: the gene count computed before the split, the
label-stratified inner folds, and that both leakage fixes descend from the run's code. Without it, --check reuses the code
provenance of the committed first record and says so; writing new records requires it.
Exit 0: written, or identical to the committed records. Exit 2: a refusal, nothing written.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("derive_run15_claim_dispositions")

RUN_CODE_COMMIT = "4b9de1c319739824955bb165c295e56e444626ce"
#: (pre-rewrite identifier, rewritten identifier on main) of the two 2026-06-13 leakage fixes.
FIXES = (("689787f93a0ca081965c755c80b0f08a89d0846b", "070ea735e7c2056172211a9bba4c680cf7adf1b1"),
         ("6b38985d61faa45b774af127f9a7939c062e455d", "6cd310a1fc15b9cb8050dcb6a68439a7901c0fda"))
CODE_FACTS = (
    ("src/genomic_variant_classifier/data/real_data_prep.py", "count_computed_before_split", "df = enrich_gene_counts(df)"),
    ("src/genomic_variant_classifier/data/real_data_prep.py", "gene_symbol_string_split", "splitter = GroupShuffleSplit("),
    ("src/genomic_variant_classifier/models/variant_ensemble.py", "label_stratified_inner_folds", "cv = StratifiedKFold("),
)


def _git(git_dir: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", f"--git-dir={git_dir}", *args], capture_output=True, text=True, encoding="utf-8")
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} exit {proc.returncode}: {proc.stderr.strip()}")
    return proc


def code_provenance(pre_git: Path, rewritten_git: Path) -> dict:
    out: dict = {"run_code_commit_pre_rewrite": RUN_CODE_COMMIT}
    subject, date = _git(pre_git, "log", "-1", "--format=%s%x00%cI", RUN_CODE_COMMIT).stdout.strip().split("\x00")
    out.update(commit_subject=subject, commit_date=date)
    facts = {}
    for path, key, needle in CODE_FACTS:
        text = _git(pre_git, "show", f"{RUN_CODE_COMMIT}:{path}").stdout.splitlines()
        lines = [i + 1 for i, ln in enumerate(text) if ln.strip() == needle]
        if len(lines) != 1:
            raise RuntimeError(f"{path} at {RUN_CODE_COMMIT[:8]}: {len(lines)} line(s) read exactly {needle!r}")
        blob = _git(pre_git, "rev-parse", f"{RUN_CODE_COMMIT}:{path}").stdout.strip()
        facts[key] = {"file": path, "git_blob": blob, "line": lines[0], "text": needle}
    if not facts["count_computed_before_split"]["line"] < facts["gene_symbol_string_split"]["line"]:
        raise RuntimeError("at the run's commit the gene count is not computed before the split")
    out["facts"] = facts
    fixes = []
    for pre, new in FIXES:
        descends = _git(pre_git, "merge-base", "--is-ancestor", RUN_CODE_COMMIT, pre, check=False).returncode == 0
        precedes = _git(pre_git, "merge-base", "--is-ancestor", pre, RUN_CODE_COMMIT, check=False).returncode == 0
        tree_pre = _git(pre_git, "rev-parse", f"{pre}^{{tree}}").stdout.strip()
        tree_new = _git(rewritten_git, "rev-parse", f"{new}^{{tree}}").stdout.strip()
        subj = _git(pre_git, "log", "-1", "--format=%s", pre).stdout.strip()
        if not descends or precedes or tree_pre != tree_new:
            raise RuntimeError(f"fix {pre[:8]}: descends from the run's code {descends}, precedes it {precedes}, "
                               f"tree {tree_pre} vs the rewritten {new[:8]}'s {tree_new}")
        fixes.append({"pre_rewrite": pre, "rewritten": new, "tree": tree_pre, "subject": subj,
                      "descends_from_run_code": True})
    out["leakage_fixes_after_the_run"] = fixes
    out["history_note"] = ("main's history was rewritten on 2026-07-11 (git filter-repo); pre-rewrite objects are read from "
                           "the owner's preserved repository; each fix's tree equals its rewritten counterpart's")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Derive Run 15's claim-disposition records from its files.")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--recorded-utc", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("docs/claims"))
    p.add_argument("--git-dir", type=Path, default=None, help="repository holding the pre-rewrite commit 4b9de1c3")
    p.add_argument("--rewritten-git-dir", type=Path, default=Path(".git"))
    p.add_argument("--check", action="store_true", help="compare with the committed records; write nothing")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    from genomic_variant_classifier.evaluation.claim_disposition import validate_record
    from genomic_variant_classifier.evaluation.run15_claim_evidence import (
        Run15EvidenceError, build_records, measure_run15, render_record)

    names = ["CLAIM-0001_run15.c3.unseen_gene_generalization.json", "CLAIM-0002_run15.test.ensemble_evaluation.json",
             "CLAIM-0003_run15.validation.ensemble_evaluation.json"]
    try:
        if args.git_dir is not None:
            provenance = code_provenance(args.git_dir, args.rewritten_git_dir)
        elif args.check:
            provenance = json.loads((args.out_dir / names[0]).read_text(encoding="utf-8"))[
                "historical_observation"]["code_provenance"]
            logger.warning("code provenance NOT re-measured (no --git-dir): taken from the committed %s", names[0])
        else:
            logger.error("writing records requires --git-dir (the pre-rewrite repository)")
            return 2
        measured = measure_run15(args.run_dir)
        records = build_records(measured, recorded_utc=args.recorded_utc, code_provenance=provenance)
        for name, rec in zip(names, records):
            validate_record(rec, where=name)
    except (Run15EvidenceError, RuntimeError, OSError, ValueError, KeyError) as exc:
        logger.error("NOT derived: %s", exc)
        return 2
    rendered = [render_record(r) for r in records]
    if args.check:
        differ = [n for n, text in zip(names, rendered)
                  if not (args.out_dir / n).is_file() or (args.out_dir / n).read_bytes() != text.encode("ascii")]
        if differ:
            logger.error("re-derived records DIFFER from the committed ones: %s", differ)
            return 2
        logger.info("all %d records re-derived byte-identical; chain head %s", len(names), records[-1]["record_sha256"])
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in zip(names, rendered):
        (args.out_dir / name).write_bytes(text.encode("ascii"))
        logger.info("written %s", args.out_dir / name)
    logger.info("chain head %s", records[-1]["record_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
