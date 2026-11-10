"""Stage 1D-1b.2, step 1: the source-absence census -- what every one of the 91 features becomes when its source is absent,
MEASURED BY EXECUTION, not read from the code (owner ruling 7f4b69d0, section 5, step 1; ABSENCE-AS-EVIDENCE-1).

Two layers, because a value can be fabricated in either:

* FEATURE LAYER -- the repository's ``engineer_features`` executed on frames that differ only in their sources:
    absent   no raw input column at all (the source was never joined);
    nan      every raw input column present and NaN (the source was joined; this variant is not in it);
    varied   every raw input drawn at random over its range (seeded), to MEASURE which outputs are 0/1 indicators and, by
             perturbing one raw input at a time, which raw inputs each feature reads (its measured lineage).
  The raw inputs are themselves measured by execution (a recording frame logs every column the builder asks for) and
  cross-checked against the builder's source text. For each feature: the value under absence and under NaN; whether
  missingness survives; whether ABSENCE ALONE sets an indicator to 1 (absence turned into positive evidence); whether an
  OBSERVED input equal to the fill gives the same output (the fill is indistinguishable from a measurement); whether the
  builder logged or warned (silent substitution or not).
* CONNECTOR LAYER -- the production annotation chain (``DataPrepPipeline._annotate_scores``, every source unconfigured)
  followed by ``engineer_features``, executed in a HERMETIC child process: a fresh empty working folder (every connector
  default path and cache is relative to it, so nothing warm is read and nothing is written into a checkout), Hugging Face
  offline, ESM-2 network lookups off, and a socket guard that counts and refuses every connection attempt. It records
  which raw inputs the chain writes with which value, and the feature values a model sees when EVERY source is absent.

The lineage check: for every feature the census declares as derived, the raw inputs it is measured to read (other than a
same-named model output) must be read by its declared dependencies -- a derived feature may not draw on an undeclared input.

Shadow mode: this module changes nothing. ``--out`` writes the census (refusing to overwrite); ``--check`` re-derives it
and compares byte for byte. Needs pandas (it runs the feature builder).
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

SCHEMA = "gvc.source-absence-census.1d1b2/1"
REPORT_PATH = "docs/census/SOURCE_ABSENCE_1D1b2.json"
CONTRACT_FILE = "src/genomic_variant_classifier/models/variant_ensemble.py"
N_ROWS = 3
N_VARIED = 400
SEED = 20261110
CHILD_TIMEOUT_S = 900

#: Raw inputs drawn as non-negative integers in the varied frame (counts, ordinals, positions).
_INTEGER_INPUTS = ("n_pathogenic_in_gene", "omim_n_diseases", "omim_n_diseases_molecular", "reactome_pathway_count",
                   "exon_number", "dist_to_splice_site", "codon_position", "gtex_n_tissues_expressed",
                   "lovd_variant_class", "n_known_pathogenic_protein_variants", "hgmd_n_reports", "kegg_pathway_count")
_STRING_INPUTS: Mapping[str, tuple[str, ...]] = {
    "ref": ("A", "AT", "G", "CTT", "C"),
    "alt": ("C", "A", "GTA", "T", "TG"),
    "consequence": ("missense_variant", "stop_gained", "synonymous_variant", "intron_variant",
                    "splice_donor_variant&frameshift_variant", "inframe_deletion", "5_prime_UTR_variant"),
    "chrom": ("1", "7", "X", "MT", "22"),
    "gene_symbol": ("BRCA1", "TP53", "PTEN", "TTN", "AGRN"),
}


# ------------------------------------------------------------------------------------------------ the feature builder
def _builder(repo: Path) -> tuple[Callable, tuple[str, ...]]:
    """The REPOSITORY's engineer_features and contract; refuses a module loaded from anywhere else."""
    import importlib

    module = importlib.import_module("genomic_variant_classifier.models.variant_ensemble")
    loaded, expected = Path(module.__file__).resolve(), (repo / CONTRACT_FILE).resolve()
    if loaded != expected:
        raise ValueError(f"engineer_features was imported from {loaded}, not from the measured repository {expected}")
    return module.engineer_features, tuple(module.TABULAR_FEATURES)


def source_text_inputs(repo: Path) -> list[str]:
    """Raw inputs named in engineer_features' source: df.get("x") calls and the string literals its loops iterate."""
    tree = ast.parse((repo / CONTRACT_FILE).read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "engineer_features")
    found: set[str] = set()
    dicts: dict[str, ast.Dict] = {}
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    dicts[t.id] = n.value
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and n.args and \
                isinstance(n.func.value, ast.Name) and n.func.value.id == "df" and \
                isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
            found.add(n.args[0].value)
        if isinstance(n, ast.For):
            it = n.iter
            if isinstance(it, (ast.Tuple, ast.List)):
                for e in it.elts:
                    if isinstance(e, ast.Constant) and isinstance(e.value, str):
                        found.add(e.value)
                    elif isinstance(e, ast.Tuple) and e.elts and isinstance(e.elts[0], ast.Constant):
                        found.add(e.elts[0].value)
            elif isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr == "items" and \
                    isinstance(it.func.value, ast.Name) and it.func.value.id in dicts:
                found.update(k.value for k in dicts[it.func.value.id].keys if isinstance(k, ast.Constant))
    return sorted(found)


def _recording_frame(n: int):
    """A DataFrame that records every column name the builder asks for (get, [], in)."""
    import pandas as pd

    seen: set[str] = set()

    class Recording(pd.DataFrame):
        @property
        def _constructor(self):
            return pd.DataFrame

        def get(self, key, default=None):          # noqa: D401 -- pandas signature
            if isinstance(key, str):
                seen.add(key)
            return super().get(key, default)

        def __getitem__(self, key):
            if isinstance(key, str):
                seen.add(key)
            return super().__getitem__(key)

        def __contains__(self, key):
            if isinstance(key, str):
                seen.add(key)
            return super().__contains__(key)

    return Recording(index=pd.RangeIndex(n)), seen


def executed_inputs(engineer: Callable) -> list[str]:
    frame, seen = _recording_frame(N_ROWS)
    engineer(frame)
    return sorted(seen)


def _varied_frame(inputs: Sequence[str], rng, m: int):
    import numpy as np
    import pandas as pd

    data: dict[str, Any] = {c: rng.uniform(0, 1, m) for c in inputs}
    for c in _INTEGER_INPUTS:
        if c in data:
            data[c] = rng.integers(0, 40, m).astype(float)
    special = {"cadd_phred": rng.uniform(0, 50, m), "loeuf": rng.uniform(0, 2, m),
               "allele_freq": 10 ** rng.uniform(-7, -0.3, m), "phylop_score": rng.uniform(-10, 10, m),
               "gerp_score": rng.uniform(-12, 6, m), "esm2_llr": rng.uniform(-15, 3, m),
               "genomiclm_llr": rng.uniform(-10, 3, m), "mis_z": rng.uniform(-5, 8, m), "syn_z": rng.uniform(-5, 5, m),
               "gtex_max_tpm": rng.uniform(0, 500, m), "gtex_min_eqtl_pval": 10 ** rng.uniform(-30, 0, m),
               "maxentscan_score": rng.uniform(-20, 15, m), "maxentscan_delta": rng.uniform(-15, 15, m)}
    for c, v in special.items():
        if c in data:
            data[c] = v
    for c in list(data):                 # exact 0 and 1 in every numeric input, so thresholds AT the fills are crossed
        cut = rng.uniform(0, 1, m)
        data[c] = np.where(cut < 0.2, 0.0, np.where(cut < 0.3, 1.0, data[c]))
    for c, choices in _STRING_INPUTS.items():
        if c in data or c in ("ref", "alt", "consequence"):
            data[c] = rng.choice(np.array(choices, dtype=object), m)
    return pd.DataFrame(data)


def _value(series) -> float | str | None:
    import numpy as np

    v = series.iloc[0]
    if not (series.isna().all() or (series == v).all()):
        raise AssertionError(f"{series.name}: not constant over identical rows")
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return float(bool(v))
    return float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v)


def _changed(a, b) -> bool:
    import numpy as np

    av, bv = a.to_numpy(), b.to_numpy()
    if av.dtype.kind in "fciub" and bv.dtype.kind in "fciub":
        af, bf = av.astype(float), bv.astype(float)
        same = (np.isnan(af) & np.isnan(bf)) | (af == bf)
        return bool((~same).any())
    return bool((av != bv).any())


_SIGNALS_CHILD = r"""
import json, logging, sys, warnings
import pandas as pd
import genomic_variant_classifier.models.variant_ensemble as ve
records = []
class Capture(logging.Handler):
    def emit(self, record):
        records.append([record.name, record.levelno, record.getMessage()])
root = logging.getLogger()
root.addHandler(Capture(logging.DEBUG))
root.setLevel(logging.DEBUG)
with warnings.catch_warnings(record=True) as warned:          # -W always (the interpreter option) shows every warning
    ve.engineer_features(pd.DataFrame(index=pd.RangeIndex(int(sys.argv[1]))))
print(json.dumps({"module": ve.__file__, "records": records,
                  "warnings": [f"{w.category.__name__}: {w.message}" for w in warned]}))
"""


def builder_signals(repo: Path) -> tuple[list[tuple[str, int, str]], list[str]]:
    """What engineer_features logs and warns on an all-absent frame, measured in a CHILD process: the logging handler and the
    interpreter's -W always option belong to that process only, so the caller's logging and warning filters are never
    touched (the package rule: no module changes the process-wide warning filters)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update({"PYTHONPATH": str((repo / "src").resolve()), "PYTHONDONTWRITEBYTECODE": "1"})
    done = subprocess.run([sys.executable, "-B", "-W", "always", "-c", _SIGNALS_CHILD, str(N_ROWS)], cwd=str(repo),
                          env=env, capture_output=True, text=True, timeout=CHILD_TIMEOUT_S)
    if done.returncode != 0:
        raise RuntimeError(f"builder-signals child failed ({done.returncode}):\n{done.stderr[-4000:]}")
    result = json.loads(done.stdout.strip().splitlines()[-1])
    if Path(result["module"]).resolve() != (repo / CONTRACT_FILE).resolve():
        raise ValueError(f"the child measured {result['module']}, not the repository's builder")
    return [tuple(r) for r in result["records"]], list(result["warnings"])


def feature_layer(engineer: Callable, contract: Sequence[str], inputs: Sequence[str],
                  signals_source: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``signals_source``: builder_signals' result, or a callable returning one for ``engineer`` (tests)."""
    import numpy as np
    import pandas as pd

    absent = engineer(pd.DataFrame(index=pd.RangeIndex(N_ROWS)))
    nan = engineer(pd.DataFrame({c: [np.nan] * N_ROWS for c in inputs}))
    if list(absent.columns) != list(contract) or list(nan.columns) != list(contract):
        raise ValueError("engineer_features output does not follow the contract")
    rng = np.random.default_rng(SEED)
    base_frame = _varied_frame(inputs, rng, N_VARIED)
    base = engineer(base_frame)
    reads: dict[str, list[str]] = {c: [] for c in contract}
    for c in inputs:                       # one raw input re-drawn at a time; what changes reads it
        other = _varied_frame(inputs, np.random.default_rng(SEED + 1 + list(inputs).index(c)), N_VARIED)
        perturbed = base_frame.copy()
        perturbed[c] = other[c]
        out = engineer(perturbed)
        for f in contract:
            if _changed(base[f], out[f]):
                reads[f].append(c)
    unread = [f for f in contract if not reads[f]]
    if unread:
        raise ValueError(f"no raw input was measured to change {unread}; the perturbation is too weak to call them leaves")
    rows = []
    for f in contract:
        a, b = _value(absent[f]), _value(nan[f])
        values = pd.concat([absent[f], nan[f], base[f]]).dropna()
        indicator = bool(set(pd.unique(values)) <= {0, 1} and base[f].nunique() == 2)
        observable = None
        if f in inputs and a is not None and not isinstance(a, str):
            probe = engineer(pd.DataFrame({f: [a] * N_ROWS}))
            observable = _value(probe[f]) == a
        rows.append({"feature": f, "absent_value": a, "nan_value": b,
                     "missingness_survives": a is None and b is None, "absent_equals_nan": a == b,
                     "indicator": indicator, "indicator_set_by_absence": bool(indicator and (a == 1.0 or b == 1.0)),
                     "fill_equals_an_observed_value": observable, "raw_inputs_read": sorted(reads[f])})
    survivors = {r["feature"] for r in rows if r["missingness_survives"]}
    caught, warned = signals_source(engineer) if callable(signals_source) else signals_source
    signals = []
    for name, level, message in caught:
        named = sorted(f for f in contract if f in message)
        signals.append({"logger": name, "level": logging.getLevelName(level), "features_named": named})
        if level >= logging.WARNING or set(named) - survivors:
            raise ValueError(f"engineer_features signalled about a fabricated value ({name}, {logging.getLevelName(level)}"
                             f", {named}); re-measure what it says before recording the substitution as silent")
    if warned:
        raise ValueError(f"engineer_features raised {len(warned)} warnings on an all-absent frame: {warned[:3]}")
    return rows, signals


# ----------------------------------------------------------------------------------------------- the connector layer
_FIXTURE = {
    "variant_id": ["census:17:43071077:G:T", "census:1:925952:G:A", "census:7:117559590:ATC:A"],
    "chrom": ["17", "1", "7"], "pos": [43071077, 925952, 117559590], "ref": ["G", "G", "ATC"], "alt": ["T", "A", "A"],
    "consequence": ["missense_variant", "stop_gained", "inframe_deletion"], "gene_symbol": ["BRCA1", "AGRN", "CFTR"],
    "protein_change": ["p.Arg20Trp", "p.Gln12Ter", None], "clinical_sig": ["Pathogenic", "Benign", "Pathogenic"],
    "source_db": ["clinvar", "clinvar", "clinvar"], "source_id": ["1", "2", "3"],
}


def _connector_child(out_path: Path) -> int:
    """Run in a fresh working folder by connector_layer(); writes JSON to out_path."""
    import socket

    attempts: list[str] = []

    def refuse(*args, **kwargs):
        attempts.append(repr(args[:2]))
        raise OSError("source-absence census: network access refused")

    socket.socket.connect = refuse          # type: ignore[assignment]
    socket.socket.connect_ex = refuse       # type: ignore[assignment]
    socket.getaddrinfo = refuse             # type: ignore[assignment]
    socket.create_connection = refuse       # type: ignore[assignment]
    import numpy as np
    import pandas as pd

    from genomic_variant_classifier.data.real_data_prep import AnnotationConfig, DataPrepConfig, DataPrepPipeline
    from genomic_variant_classifier.models.variant_ensemble import TABULAR_FEATURES, engineer_features

    work = Path.cwd()
    before = {p.relative_to(work).as_posix() for p in work.rglob("*")}
    frame = pd.DataFrame(_FIXTURE)
    pipe = DataPrepPipeline(DataPrepConfig(output_dir=work / "splits"), AnnotationConfig(esm2_allow_network=False))
    annotated = pipe._annotate_scores(frame.copy())
    feats = engineer_features(annotated)
    written = {}
    for c in sorted(set(annotated.columns) - set(frame.columns)):
        vals = annotated[c].tolist()
        written[c] = [None if (v is None or (isinstance(v, float) and math.isnan(v))) else
                      (float(v) if isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_)) else str(v))
                      for v in vals]
    features = {c: [None if (isinstance(v, float) and math.isnan(v)) else float(v) for v in feats[c].tolist()]
                for c in TABULAR_FEATURES}
    created = sorted({p.relative_to(work).as_posix() for p in work.rglob("*")} - before - {out_path.name})
    out_path.write_text(json.dumps({"written": written, "features": features, "network_attempts": len(attempts),
                                    "created": created}, sort_keys=True), encoding="utf-8")
    return 0


def connector_layer(repo: Path) -> dict[str, Any]:
    """The production annotation chain with every source unconfigured, in a hermetic child process."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ESM2_", "HF_", "TRANSFORMERS_", "PYTHON"))}
    env.update({"PYTHONPATH": str((repo / "src").resolve()), "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    with tempfile.TemporaryDirectory(prefix="gvc_absence_") as tmp:
        work = Path(tmp)
        out = work / "connector_layer.json"
        done = subprocess.run([sys.executable, "-B", "-m", "genomic_variant_classifier.information_sources.absence_census",
                               "--connector-child", str(out)], cwd=str(work), env=env, capture_output=True, text=True,
                              timeout=CHILD_TIMEOUT_S)
        if done.returncode != 0:
            raise RuntimeError(f"connector-layer child failed ({done.returncode}):\n{done.stderr[-4000:]}")
        result = json.loads(out.read_text(encoding="utf-8"))
    if result["network_attempts"]:
        raise RuntimeError(f"the annotation chain attempted {result['network_attempts']} network connections")
    return result


# --------------------------------------------------------------------------------------------------------- the census
def lineage_check(ledger: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """For each declared-derived feature: measured raw inputs (bar a same-named model output) must be read by its
    declared dependencies. Returns one record per derived feature; refuses an unexplained input."""
    reads = {r["feature"]: set(r["raw_inputs_read"]) for r in rows}
    deps = {e["name"]: e["depends_on"] for e in ledger["entries"]}
    out = []
    for e in ledger["entries"]:
        if not e["depends_on"]:
            continue
        own = reads[e["name"]] - {e["name"]}
        explained = set().union(*(reads.get(d, set()) for d in e["depends_on"]))
        unexplained = sorted(own - explained)
        if unexplained:
            raise ValueError(f"{e['name']}: reads raw inputs its declared dependencies do not: {unexplained}")
        out.append({"feature": e["name"], "declared_dependencies": len(deps[e["name"]]),
                    "raw_inputs_read": sorted(reads[e["name"]]), "explained_by_dependencies": True})
    return out


def build(repo: Path, *, connector: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from genomic_variant_classifier.information_sources import census as _census

    engineer, contract = _builder(repo)
    text_inputs = source_text_inputs(repo)
    run_inputs = executed_inputs(engineer)
    if text_inputs != run_inputs:
        raise ValueError(f"raw inputs measured by execution {sorted(set(run_inputs) - set(text_inputs))} and named in the "
                         f"source {sorted(set(text_inputs) - set(run_inputs))} disagree")
    rows, signals = feature_layer(engineer, contract, run_inputs, builder_signals(repo))
    ledger = _census.strict_load(repo / _census.LEDGER_PATH)
    lineage = lineage_check(ledger, rows)
    layer = dict(connector) if connector is not None else connector_layer(repo)
    for r in rows:
        r["all_sources_absent_through_the_annotation_chain"] = layer["features"][r["feature"]]
    written = {c: v for c, v in layer["written"].items() if c in run_inputs}
    return {
        "schema": SCHEMA,
        "mode": "shadow: executes the repository's feature builder and annotation chain; changes nothing",
        "census_ledger_sha256": _census.sha256_bytes((repo / _census.LEDGER_PATH).read_bytes()),
        "builder_sha256": _census.symbol_closure_digest(repo, f"{CONTRACT_FILE}::engineer_features"),
        "raw_inputs": {"count": len(run_inputs), "names": run_inputs,
                       "measured_by": "execution (a recording frame), equal to the builder's source text"},
        "summary": {
            "features": len(rows),
            "missingness_survives": [r["feature"] for r in rows if r["missingness_survives"]],
            "fabricated_under_absence": sum(not r["missingness_survives"] for r in rows),
            "absent_differs_from_nan": [r["feature"] for r in rows if not r["absent_equals_nan"]],
            "indicators": sum(r["indicator"] for r in rows),
            "indicators_set_by_absence": [r["feature"] for r in rows if r["indicator_set_by_absence"]],
            "fill_equals_an_observed_value": [r["feature"] for r in rows if r["fill_equals_an_observed_value"]],
            "fabrication_is_silent": True,
            "builder_signals_under_absence": signals,
            "fill_precedes_scaling": True,
            "same_builder_at_inference": True,
        },
        "notes": [
            "fabrication_is_silent: on an all-absent frame engineer_features raised no warning and logged nothing at "
            "WARNING or above, and no record named a fabricated feature (measured; any would have refused the census); "
            "its only records are listed in builder_signals_under_absence",
            "fill_precedes_scaling: the scaler is fitted on engineer_features output (census surface scaler_fit), so every "
            "fill above is a training value before standardisation",
            "same_builder_at_inference: api/pipeline.py renders the contract with the same engineer_features; the two "
            "NaN-preserving features are then imputed with TRAINING-fold medians by the fitted missing-value policy",
        ],
        "connector_layer": {
            "fixture_rows": len(_FIXTURE["variant_id"]),
            "network_attempts": layer["network_attempts"],
            "files_created_in_the_hermetic_folder": layer["created"],
            "raw_inputs_written_by_the_chain": written,
            "raw_inputs_never_written_by_the_chain": sorted(set(run_inputs) - set(layer["written"]) - set(_FIXTURE)),
        },
        "lineage_check": lineage,
        "rows": rows,
    }


def report_bytes(report: Mapping[str, Any]) -> bytes:
    from genomic_variant_classifier.information_sources import census as _census

    return _census.ledger_bytes(report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 1D-1b.2 source-absence census (executes; changes nothing).")
    parser.add_argument("--repo", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path, help="write the census here (refuses to overwrite)")
    group.add_argument("--check", type=Path, help="exit 1 unless this file is byte-identical to a fresh derivation")
    group.add_argument("--connector-child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.connector_child is not None:
        return _connector_child(args.connector_child)
    if args.repo is None:
        parser.error("--repo is required")
    report = build(args.repo.resolve())
    data = report_bytes(report)
    if args.out is not None:
        if args.out.exists():
            raise SystemExit(f"refusing to overwrite {args.out}")
        args.out.write_bytes(data)
    elif args.check.read_bytes() != data:
        print(f"{args.check}: DIFFERS from the census derived now")
        return 1
    s = report["summary"]
    print(f"{s['features']} features; {s['fabricated_under_absence']} fabricate a value when their source is absent; "
          f"missingness survives in {len(s['missingness_survives'])}")
    print(f"indicators set to 1 by absence alone: {', '.join(s['indicators_set_by_absence'])}")
    print(f"raw inputs: {report['raw_inputs']['count']} (execution = source text); connector layer: "
          f"{report['connector_layer']['network_attempts']} network attempts")
    print("mode: SHADOW -- nothing changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
