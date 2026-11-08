"""The measured evidence behind Run 15's claim dispositions, derived from the run's own files.

Author: Monzia Moodie

Package 1D, stage 1D-0 (owner ruling 2026-11-08). The three records in docs/claims/ are WRITTEN by this module from Run 15's
artifacts -- never typed -- so that every figure in them can be re-derived: scripts/derive_run15_claim_dispositions.py runs
it on `outputs/run15_rerun_report/full` and, with --check, compares its output with the committed records byte for byte.

What is measured, per evaluation (validation, test, and the C3 unseen-gene holdout):

  GENE_TARGET_COUNT_OWN_LABEL_LEAKAGE  the gene-count feature `n_pathogenic_in_gene` of each evaluated row equals the number
      of pathogenic labels of the EVALUATED rows of its gene (the row's own label included). Run 15 ran at commit 4b9de1c3
      (pre-rewrite), where enrich_gene_counts() counted labels corpus-wide BEFORE the gene-grouped split; the train-only
      recount landed on 2026-06-13. The record holds the row count, the mismatches (0 means the identity holds on every
      row) and the AUROC of that single feature on the evaluated rows.
  GENE_IDENTITY_SPLIT_LEAKAGE  the split grouped by gene-symbol STRING, so a string naming several genes and a string naming
      one of them land in different groups. Measured: evaluated rows whose string shares at least one gene token
      (separators ; , | and whitespace) with a training string, and the positives among them.
  VARIANT_LEVEL_CLINICAL_CLASS_FEATURE  `lovd_variant_class` is LOVD's own clinical classification of the variant. In the
      scaled matrix an unannotated row carries one value; rows carrying any other value are annotated. Listed only where
      the count is above zero.
  GNN_HOLDOUT_LABEL_LEAKAGE (C3 only)  the graph neural network was trained on the whole training split and its scores
      injected into it BEFORE the holdout was carved from it; read from the run's log by line and time.

The holdout is reconstructed with Run 15's own rule (data/splits.py at 4b9de1c3: a gene is held out when the MD5 of
"42:<gene_symbol>" read as a fraction of 2**128 is below 0.20) and must reproduce the counts the run logged and wrote.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from genomic_variant_classifier.evaluation.claim_disposition import SCHEMA, record_digest
from genomic_variant_classifier.evaluation.historical_oof_identity import file_identity
from genomic_variant_classifier.evaluation.metrics import auroc

logger = logging.getLogger(__name__)

RULING_DECISION_FILE_SHA256 = "56ca0eff9078e933faabe958dddb62f1017ee67c3a6036b8d5573a634f7476a0"
AUTHOR = "Monzia Moodie"
HOLDOUT_SEED = 42
HOLDOUT_FRACTION = 0.20
_TOKEN_SPLIT = re.compile(r"[;,|\s]+")
_LOG_STAMP = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

#: The artifacts each record binds, relative to the run directory (the log sits one level up, as copied by the owner).
ARTIFACTS = (
    "metrics.json", "unseen_gene_holdout_metrics.json", "unseen_gene_holdout_per_model.csv", "per_model_metrics.csv",
    "per_model_metrics_val.csv", "splits/meta_train.parquet", "splits/y_train.parquet", "splits/X_train.parquet",
    "splits/meta_val.parquet", "splits/y_val.parquet", "splits/X_val.parquet", "splits/meta_test.parquet",
    "splits/y_test.parquet", "splits/X_test.parquet", "../run15_baseline_master.log",
)


class Run15EvidenceError(RuntimeError):
    """The run's files do not reproduce what the derivation requires. Nothing is written."""


def gene_tokens(symbol: str) -> frozenset[str]:
    return frozenset(t for t in _TOKEN_SPLIT.split(str(symbol)) if t)


def run15_holdout_mask(genes: pd.Series, *, seed: int = HOLDOUT_SEED, fraction: float = HOLDOUT_FRACTION) -> np.ndarray:
    """Run 15's unseen-gene holdout rule, verbatim in effect (missing symbols read as "unknown")."""
    g = genes.fillna("unknown").astype(str)
    held = {s for s in g.unique()
            if int(hashlib.md5(f"{seed}:{s}".encode()).hexdigest(), 16) / (2 ** 128) < fraction}
    return g.isin(held).to_numpy()


def own_label_count_identity(genes: pd.Series, y: np.ndarray, npig: np.ndarray) -> dict:
    """Does each row's gene count equal the pathogenic labels of the EVALUATED rows of its gene (its own included)?"""
    y = np.asarray(y).astype(int)
    npig = np.asarray(npig)
    own = pd.Series(y).groupby(genes.fillna("unknown").astype(str).to_numpy()).transform("sum").to_numpy()
    return {"rows": int(len(y)), "positives": int(y.sum()), "mismatches": int((npig != own).sum()),
            "feature_auroc_alone": round(float(auroc(y, npig.astype(float))), 4)}


def gene_token_overlap(reference: pd.Series, genes: pd.Series, y: np.ndarray) -> dict:
    """Evaluated rows whose gene-symbol string shares a gene token with any reference (training) string."""
    ref_strings = set(reference.fillna("unknown").astype(str).unique())
    ref_tokens = set().union(*(gene_tokens(s) for s in ref_strings))
    g = genes.fillna("unknown").astype(str)
    strings = g.unique()
    shared = sorted(s for s in strings if gene_tokens(s) & ref_tokens)
    rows = g.isin(set(shared)).to_numpy()
    y = np.asarray(y).astype(int)
    return {"evaluated_strings": int(len(strings)), "strings_identical_to_a_training_string": int(
        sum(s in ref_strings for s in strings)),
            "strings_sharing_a_gene_token": len(shared), "rows_sharing_a_gene_token": int(rows.sum()),
            "rows": int(len(g)), "row_fraction": round(float(rows.mean()), 4),
            "positives_among_them": int(y[rows].sum()), "positives": int(y.sum())}


def unannotated_lovd_value(training_scaled_lovd: np.ndarray) -> float:
    """The scaled value of an UNANNOTATED row: the training split's most frequent value (362 of 1,490,014 rows are
    annotated in Run 15, so the mode is the unannotated value; the scaler was fitted on training, so every split maps a
    raw 0 to this same value)."""
    values, counts = np.unique(np.asarray(training_scaled_lovd), return_counts=True)
    if not len(values):
        raise Run15EvidenceError("the training split has no lovd_variant_class values")
    return float(values[counts.argmax()])


def lovd_annotated_rows(scaled_lovd: np.ndarray, unannotated: float) -> int:
    return int((np.asarray(scaled_lovd) != unannotated).sum())


def _log_line(lines: Sequence[str], needle: str) -> dict:
    hits = [(i + 1, ln) for i, ln in enumerate(lines) if needle in ln]
    if len(hits) != 1:
        raise Run15EvidenceError(f"the log holds {len(hits)} line(s) containing {needle!r}; exactly one is required")
    no, ln = hits[0]
    stamped = _LOG_STAMP.match(ln)
    # The driver's closing table is printed, not logged: its lines carry no timestamp, and none is invented for them.
    return {"line": no, "timestamp": stamped.group(0) if stamped else None,
            "text": (ln[stamped.end():] if stamped else ln).strip()[:160]}


def _read(run_dir: Path, split: str, *, x_cols: Sequence[str]) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    meta = pd.read_parquet(run_dir / "splits" / f"meta_{split}.parquet", columns=["gene_symbol", "n_pathogenic_in_gene"])
    y = pd.read_parquet(run_dir / "splits" / f"y_{split}.parquet").iloc[:, 0].to_numpy().astype(int)
    x = pd.read_parquet(run_dir / "splits" / f"X_{split}.parquet", columns=list(x_cols))
    if not (len(meta) == len(y) == len(x)):
        raise Run15EvidenceError(f"{split}: meta {len(meta)}, y {len(y)}, X {len(x)} rows")
    return meta, y, x


def measure_run15(run_dir: Path) -> dict:
    """Every measurement the three records cite. Raises on any disagreement with the run's own records."""
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    ugh = json.loads((run_dir / "unseen_gene_holdout_metrics.json").read_text(encoding="utf-8"))
    # Split on LF only: the log carries carriage returns inside progress output, and line numbers must be the ones grep and
    # an editor show (str.splitlines() also splits at CR and gave 353 where the file's line is 285).
    log = (run_dir.parent / "run15_baseline_master.log").read_bytes().decode("utf-8").split("\n")
    artifacts = [{"path": f"outputs/run15_rerun_report/{Path(p).as_posix().replace('../', '')}"
                  if p.startswith("../") else f"outputs/run15_rerun_report/full/{p}",
                  **file_identity((run_dir / p).resolve())} for p in ARTIFACTS]

    tr_meta, tr_y, tr_x = _read(run_dir, "train", x_cols=["lovd_variant_class"])
    lovd0 = unannotated_lovd_value(tr_x["lovd_variant_class"].to_numpy())
    out: dict[str, Any] = {"artifacts": artifacts, "metrics_json": metrics, "unseen_gene_holdout_json": ugh}
    for split, n_key in (("validation", "n_val"), ("test", "n_test")):
        meta, y, x = _read(run_dir, "val" if split == "validation" else "test", x_cols=["lovd_variant_class"])
        if len(y) != metrics[n_key]:
            raise Run15EvidenceError(f"{split}: {len(y)} rows, metrics.json {n_key} = {metrics[n_key]}")
        out[split] = {
            "own_label_count": own_label_count_identity(meta["gene_symbol"], y, meta["n_pathogenic_in_gene"].to_numpy()),
            "gene_token_overlap": gene_token_overlap(tr_meta["gene_symbol"], meta["gene_symbol"], y),
            "lovd_annotated_rows": lovd_annotated_rows(x["lovd_variant_class"].to_numpy(), lovd0),
        }
    ho = run15_holdout_mask(tr_meta["gene_symbol"])
    n_ho, n_sub = int(ho.sum()), int((~ho).sum())
    genes_ho = tr_meta["gene_symbol"].fillna("unknown")[ho]
    if (n_ho, n_sub, int(genes_ho.nunique())) != (ugh["n_holdout"], ugh["n_train_subset"], ugh["n_holdout_genes"]):
        raise Run15EvidenceError(f"the reconstructed holdout ({n_ho} rows, {genes_ho.nunique()} genes, {n_sub} training "
                                 f"rows) does not reproduce the run's record ({ugh['n_holdout']}, "
                                 f"{ugh['n_holdout_genes']}, {ugh['n_train_subset']})")
    out["c3_holdout"] = {
        "reconstruction": {"rule": "md5('42:' + gene_symbol) / 2**128 < 0.20 (data/splits.py at 4b9de1c3)",
                           "holdout_rows": n_ho, "holdout_genes": int(genes_ho.nunique()), "training_rows": n_sub,
                           "reproduces_unseen_gene_holdout_metrics_json": True},
        "own_label_count": own_label_count_identity(tr_meta["gene_symbol"][ho].reset_index(drop=True), tr_y[ho],
                                                    tr_meta["n_pathogenic_in_gene"].to_numpy()[ho]),
        "gene_token_overlap": gene_token_overlap(tr_meta["gene_symbol"][~ho], tr_meta["gene_symbol"][ho]
                                                 .reset_index(drop=True), tr_y[ho]),
        "lovd_annotated_rows": lovd_annotated_rows(tr_x["lovd_variant_class"].to_numpy()[ho], lovd0),
        "gnn_order_in_log": {
            "gnn_scores_injected_into_training_split": _log_line(log, "GNN scores injected into train split"),
            "holdout_carved_from_training_split": _log_line(log, "[UGH] split:"),
            "historical_verdict_line": _log_line(log, "[UGH] C3 falsifier (b)"),
        },
    }
    out["lovd_log_line"] = _log_line(log, "variants annotated (lovd_variant_class > 0)")
    out["validation_verdict_line"] = _log_line(log, "PASS  (target >= 0.9)")
    g = out["c3_holdout"]["gnn_order_in_log"]
    if not g["gnn_scores_injected_into_training_split"]["line"] < g["holdout_carved_from_training_split"]["line"]:
        raise Run15EvidenceError("the log does not show the GNN injection before the holdout carve")
    return out


_COMMON_REPLACEMENT_TEXT = ("A leakage-free, gene-component-disjoint evaluation of a predictor whose every supervised input "
                            "excludes the evaluated genes' labels (Package 1D stages 1D-1 to 1D-3), on a newly locked "
                            "population (ruling section 15.6).")


def _defects(m: dict, *, c3: bool, lovd_log_line: dict) -> list[dict]:
    defects: list[dict] = []
    if c3:
        defects.append({
            "defect_id": "GNN_HOLDOUT_LABEL_LEAKAGE",
            "statement": ("The graph neural network was trained on the labels of the whole training split, holdout genes "
                          "included, and its gene scores were written into the training matrix before the holdout was "
                          "carved from it; the holdout ensemble trained and was evaluated on those scores."),
            "evidence": m["gnn_order_in_log"]})
    defects.append({
        "defect_id": "GENE_TARGET_COUNT_OWN_LABEL_LEAKAGE",
        "statement": ("Each evaluated row's n_pathogenic_in_gene equals the number of pathogenic labels among the evaluated "
                      "rows of its gene, its own label included (mismatches counts the rows where it does not)."),
        "evidence": m["own_label_count"]})
    defects.append({
        "defect_id": "GENE_IDENTITY_SPLIT_LEAKAGE",
        "statement": ("The split grouped rows by gene-symbol string, not by gene: evaluated rows whose string shares at "
                      "least one gene with a training string were not held out from that gene."),
        "evidence": m["gene_token_overlap"]})
    if m["lovd_annotated_rows"] > 0:
        defects.append({
            "defect_id": "VARIANT_LEVEL_CLINICAL_CLASS_FEATURE",
            "statement": ("lovd_variant_class, LOVD's own clinical classification of the variant, was an input; evaluated "
                          "rows carrying an annotation are counted."),
            "evidence": {"annotated_evaluated_rows": m["lovd_annotated_rows"], "run_log": lovd_log_line}})
    return defects


def _population(name: str, m: dict, how: str) -> str:
    o = m["gene_token_overlap"]
    return f"{name}: {o['rows']:,} rows of {o['evaluated_strings']:,} gene-symbol strings, {how}"


def _criterion(rule: str, line: dict) -> str:
    when = f" ({line['timestamp']})" if line["timestamp"] else " (printed, no timestamp)"
    return f"{rule}; the run's log, line {line['line']}{when}: {line['text']}"


def build_records(measured: dict, *, recorded_utc: str, code_provenance: dict) -> list[dict]:
    """The three records, in order, hash-chained. Deterministic for fixed inputs."""
    metrics, ugh = measured["metrics_json"], measured["unseen_gene_holdout_json"]
    ablation = {
        "description": ("docs/ABLATION_n_pathogenic_in_gene.md, arm no_gene_level (2026-06-07): n_pathogenic_in_gene, "
                        "gene_has_known_disease and two constant constraint columns zeroed in every split, ensemble "
                        "retrained on a 150,000-row subsample of these splits (3 folds, 7 tree models, seed 42)."),
        "metrics": {"test_auroc": 0.99783, "test_auroc_ci95_low": 0.99761, "test_auroc_ci95_high": 0.99804,
                    "matched_full_arm_test_auroc": 0.99817},
        "source": "outputs/ablation_run15/ablation_results.parquet (as recorded in the ablation document)",
    }
    specs = [
        ("run15.c3.unseen_gene_generalization", "gene_disjoint_generalization",
         "The Run 15 ensemble generalizes to genes absent from its training data (C3 falsifier: unseen-gene AUROC >= 0.95).",
         _population("Run 15 C3 unseen-gene holdout", measured["c3_holdout"], "carved from the training split"),
         {k: ugh[k] for k in ("auroc", "auprc", "f1", "mcc", "brier")}, "PASS",
         _criterion("auroc >= 0.95", measured["c3_holdout"]["gnn_order_in_log"]["historical_verdict_line"]),
         measured["c3_holdout"], True, ["0.9988", "0.9945", "0.9847", "0.9695", "0.0059"], []),
        ("run15.test.ensemble_evaluation", "held_out_performance",
         "The Run 15 ensemble's test-split performance measures pathogenicity prediction on genes absent from training.",
         _population("Run 15 test split", measured["test"], "grouped apart from training by string"),
         {k: metrics[k] for k in ("auroc", "auprc", "f1", "mcc", "brier")}, "NOT_RECORDED", None,
         measured["test"], False, ["0.9984", "0.9936", "0.9826", "0.9652", "0.0069"], [ablation]),
        ("run15.validation.ensemble_evaluation", "held_out_performance",
         "The Run 15 ensemble's validation-split performance measures pathogenicity prediction on genes absent from "
         "training (and passed the run's AUROC >= 0.9 target).",
         _population("Run 15 validation split", measured["validation"], "grouped apart from training by string"),
         {k.replace("val_", ""): metrics[k] for k in ("val_auroc", "val_auprc", "val_f1", "val_mcc", "val_brier")},
         "PASS", _criterion("auroc >= 0.9 (run_phase2_eval --auroc-target)", measured["validation_verdict_line"]),
         measured["validation"], False, ["0.9984", "0.9924", "0.9801", "0.9603", "0.0072"], []),
    ]
    records: list[dict] = []
    previous: Optional[str] = None
    for seq, (claim_id, ctype, statement, population, obs_metrics, verdict, criterion, m, c3, figures,
              replacements) in enumerate(specs, start=1):
        defects = _defects(m, c3=c3, lovd_log_line=measured["lovd_log_line"])
        ids = [d["defect_id"] for d in defects]
        reps = []
        for rep in replacements:
            reps.append({**rep, "addresses_defects": [i for i in ids if i == "GENE_TARGET_COUNT_OWN_LABEL_LEAKAGE"],
                         "does_not_address_defects": [i for i in ids if i != "GENE_TARGET_COUNT_OWN_LABEL_LEAKAGE"]})
        record = {
            "schema": SCHEMA, "record_sequence": seq, "previous_record_sha256": previous, "recorded_utc": recorded_utc,
            "author": AUTHOR,
            "ruling": {"date": "2026-11-08", "decision_file_sha256": RULING_DECISION_FILE_SHA256,
                       "sections": ["2.1", "2.2", "2.3", "21"]},
            "claim": {"claim_id": claim_id, "claim_type": ctype, "statement": statement,
                      "evaluation_population": population},
            "historical_observation": {"run": "Run 15 (outputs/run15_rerun_report/full; launched 2026-06-05 22:46 UTC)",
                                       "metrics": obs_metrics, "verdict": verdict, "criterion": criterion,
                                       "code_provenance": code_provenance, "source_artifacts": measured["artifacts"]},
            "disposition": "INVALIDATED_EVALUATION", "scientific_claim_eligible": False,
            "invalidating_defects": defects, "inflation_estimate": None, "partial_replacements": reps,
            "hypothesis_status": "NOT_RESOLVED_BY_THIS_EXPERIMENT", "required_replacement": _COMMON_REPLACEMENT_TEXT,
            "replacement_claim_id": None, "cited_figures": figures,
        }
        record["record_sha256"] = record_digest(record)
        previous = record["record_sha256"]
        records.append(record)
    return records


def render_record(record: dict) -> str:
    """The committed bytes: the record as indented JSON in its own key order, LF, one trailing newline, ASCII."""
    return json.dumps(record, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
