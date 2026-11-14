"""Stage 1D-1b.4 real-data measurement: docs/census/POPULATION_ISOLATION_1D1b4.json.

Every number in the record is measured here from the verified bytes of the Run 15 split manifests (meta_{train,val,test} and
y_{train,val,test}) and the pinned HGNC complete set, each file read through one VerifiedReader pass with its SHA-256
recorded: how the cohort gene fields resolve to canonical gene SETS (and which hold prose), variant identity, the three-way
split audit under a VRS protocol and under a declared string-identity protocol, the STRICT and CONDITIONAL_COMPONENT fit
authorities (population.authorize_fit) against the gene-direct rule and the 1D-1b.3 one-gene rule, the label mix, tempered
component weights, the component-disjoint stratified checkpoint split against the legacy GNN early-stopping call
(GNN-EARLY-STOP-ROW-SPLIT-1), and one TrainingLabels receipt.

Run (the files are the owner's; a clone does not hold them):
    python scripts/measure_1d1b4_population.py --hgnc H --splits DIR --out docs/census/POPULATION_ISOLATION_1D1b4.json
    python scripts/measure_1d1b4_population.py --hgnc H --splits DIR --check docs/census/POPULATION_ISOLATION_1D1b4.json

``--check`` re-measures and compares everything but the run section (timings, peak memory, platform) byte for byte.
Peak memory is read through the ``resource`` module where it exists (Linux, macOS); elsewhere it is recorded as null.
Author: Monzia Moodie.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from genomic_variant_classifier.information_sources import census as C  # noqa: E402
from genomic_variant_classifier.information_sources import graph_identity as G  # noqa: E402
from genomic_variant_classifier.information_sources import labels as L  # noqa: E402
from genomic_variant_classifier.information_sources import population as P  # noqa: E402
from genomic_variant_classifier.information_sources.materialization import read_verified_bytes  # noqa: E402

logger = logging.getLogger(__name__)
SCRIPT = "scripts/measure_1d1b4_population.py"
SCHEMA = "gvc.population-isolation-record.1d1b4/1"
PARTITIONS = (("train", P.Partition.TRAIN), ("val", P.Partition.VALIDATION), ("test", P.Partition.TEST))
META_COLUMNS = ["variant_id", "gene_symbol", "chrom", "pos", "ref", "alt", "consequence"]
ALPHAS = (0, 0.25, 0.5, 1.0)
#: The legacy GNN early-stopping call: models/gnn.py train_gnn_pipeline, test_split as the run driver passes it.
LEGACY_EARLY_STOP = {"call": "train_test_split(np.arange(n), test_size=0.15, random_state=42, stratify=y)",
                     "code": "src/genomic_variant_classifier/models/gnn.py::train_gnn_pipeline",
                     "test_split_from": "scripts/run_phase2_eval.py (test_split=0.15)"}
BOUND_SYMBOLS = (
    "src/genomic_variant_classifier/information_sources/graph_identity.py::read_hgnc",
    "src/genomic_variant_classifier/information_sources/population.py::resolve_gene_field",
    "src/genomic_variant_classifier/information_sources/population.py::PopulationRow",
    "src/genomic_variant_classifier/information_sources/population.py::split_components",
    "src/genomic_variant_classifier/information_sources/population.py::audit_split",
    "src/genomic_variant_classifier/information_sources/population.py::authorize_fit",
    "src/genomic_variant_classifier/information_sources/population.py::FitAuthorization",
    "src/genomic_variant_classifier/information_sources/population.py::tempered_component_weights",
    "src/genomic_variant_classifier/information_sources/population.py::component_checkpoint_split",
    "src/genomic_variant_classifier/information_sources/labels.py::TrainingLabels",
    f"{SCRIPT}::measure",
)


def peak_mib() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 4) if d else 0.0


def measure(hgnc_path: Path, splits: Path, log: list) -> dict:
    """The record without its run section; deterministic for the same files and code."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    t0 = time.perf_counter()

    def step(name: str) -> None:
        log.append({"step": name, "seconds": round(time.perf_counter() - t0, 1), "peak_rss_mib": peak_mib()})
        print(f"[{name}] {log[-1]['seconds']} s, peak {log[-1]['peak_rss_mib']} MiB", flush=True)

    rec: dict = {"schema": SCHEMA, "author": "Monzia Moodie", "inputs": {}}
    hgnc = G.read_hgnc(hgnc_path)
    rec["inputs"]["hgnc"] = {"file": hgnc_path.name, "sha256": hgnc.source_sha256, "bytes": hgnc_path.stat().st_size,
                             "logical_source": "dataset:hgnc"}
    cache: dict[str, P.GeneField] = {}
    rows: list[P.PopulationRow] = []
    labels: dict[str, int] = {}
    raw_symbol: dict[str, str] = {}
    resolution: dict = {}
    keys: set = set()
    variant_ids: set = set()
    prose_consequence: Counter = Counter()
    unresolved_tokens: set = set()
    for name, part in PARTITIONS:
        frames = {}
        for kind in ("meta", "y"):
            path = splits / f"{kind}_{name}.parquet"
            blob = read_verified_bytes(path)
            rec["inputs"][f"{kind}_{name}"] = {"file": path.name, "sha256": blob.sha256, "bytes": path.stat().st_size,
                                               "logical_source": "dataset:clinvar_cohort"}
            frames[kind] = pd.read_parquet(io.BytesIO(blob.value), columns=META_COLUMNS if kind == "meta" else ["label"])
        meta, y = frames["meta"], frames["y"]["label"]
        if len(meta) != len(y) or y.dtype.kind not in "iu" or not y.isin([0, 1]).all():
            raise ValueError(f"{name}: the manifest and its labels disagree in length, or a label is not 0/1")
        counts = Counter()
        fields = meta["gene_symbol"].fillna("").astype(str).tolist()
        for i, (vid, field, label, chrom, pos, ref, alt, csq) in enumerate(zip(
                meta["variant_id"].astype(str), fields, y.tolist(), meta["chrom"].astype(str), meta["pos"].astype(str),
                meta["ref"].astype(str), meta["alt"].astype(str), meta["consequence"].fillna("").astype(str))):
            gf = cache.get(field)
            if gf is None:
                gf = cache[field] = P.resolve_gene_field(field, hgnc)
            rid = f"{name}:{i}"
            rows.append(P.PopulationRow(rid, vid, P.VariantIdentity.UNNORMALIZED, gf.gene_ids, gf.unresolved_tokens, part))
            labels[rid] = int(label)
            raw_symbol[rid] = field
            keys.add((chrom, pos, ref, alt))
            variant_ids.add(vid)
            unresolved_tokens.update(gf.unresolved_tokens)
            counts["rows"] += 1
            counts["no_resolved_gene"] += not gf.gene_ids
            counts["partially_resolved"] += bool(gf.gene_ids) and bool(gf.unresolved_tokens)
            counts["multi_gene_complete"] += len(gf.gene_ids) > 1 and not gf.unresolved_tokens
            counts["prose"] += gf.prose
            counts["pathogenic"] += int(label)
            if gf.prose:
                prose_consequence[csq or "(none)"] += 1
        resolution[name] = dict(sorted(counts.items()))
    n = len(rows)
    rec["gene_fields"] = {
        "rule": "split on ';' ',' '|' '/', strip, resolve by approved symbol else unique previous symbol (aliases are "
                "not identity); a token holding whitespace is prose",
        "by_partition": resolution, "distinct_fields": len(cache),
        "distinct_unresolved_tokens": len(unresolved_tokens),
        "unresolved_tokens_loc_placeholders": sum(1 for t in unresolved_tokens if t.startswith("LOC")),
        "prose_rows": sum(r["prose"] for r in resolution.values()),
        "prose_distinct_fields": sum(1 for gf in cache.values() if gf.prose),
        "prose_rows_by_consequence": dict(sorted(prose_consequence.items()))}
    rec["variant_identity"] = {"rows": n, "distinct_variant_ids": len(variant_ids),
                               "distinct_chrom_pos_ref_alt": len(keys), "qualification": "unnormalized (no VRS authority)"}
    del keys, variant_ids
    step("rows")

    vrs = P.audit_split(rows)
    declared = P.SplitProtocol(require_vrs=False)
    audit = P.audit_split(rows, declared)
    rec["audit"] = {"vrs_protocol_verdict": vrs["verdict"], "string_protocol": audit}
    step("audit")
    try:
        P.authorize_fit(rows, P.FitAuthority.STRICT, declared)
        rec["strict"] = {"outcome": "authorized"}
    except PermissionError as exc:
        rec["strict"] = {"outcome": "refused", "reason": str(exc)}
    fa = P.authorize_fit(rows, P.FitAuthority.CONDITIONAL_COMPONENT, declared)
    rep = dict(fa.report)
    closes = rep["training_rows"] - rep["excluded_gene_incomplete"] - rep["excluded_component_shares_heldout"]
    if closes != rep["authorized_rows"] or len(fa.authorized_rows) != rep["authorized_rows"]:
        raise AssertionError("the conditional authority's arithmetic does not close")
    rec["conditional"] = {"report": rep, "digest": fa.digest, "population_digest": fa.population_digest,
                          "arithmetic": "training - gene-incomplete - in a component holding a held-out row = authorized"}
    step("authorize")

    train = [r for r in rows if r.partition is P.Partition.TRAIN]
    held = [r for r in rows if r.partition is not P.Partition.TRAIN]
    held_genes = set().union(*(r.gene_ids for r in held))
    direct = {r.row_id for r in train if r.gene_complete and not (r.gene_ids & held_genes)}
    if not fa.authorized_rows <= direct:
        raise AssertionError("the component authority admits a row the gene-direct rule refuses")
    only_direct = direct - fa.authorized_rows
    # The 1D-1b.3 rule: ONE gene identity per row (the whole field resolved, else 'unresolved:<field>').
    one_gene = {}
    for r in rows:
        hid, _ = hgnc.resolve(raw_symbol[r.row_id]) if raw_symbol[r.row_id] else (None, None)
        one_gene[r.row_id] = hid or f"unresolved:{raw_symbol[r.row_id]}"
    held_one = {one_gene[r.row_id] for r in held}
    refused_1d1b3 = [r.row_id for r in train if one_gene[r.row_id] in held_one]
    refused_genes = sorted({one_gene[x] for x in refused_1d1b3})
    rec["comparison"] = {
        "gene_direct_rule": {"rule": "a gene-complete training row sharing no HGNC id with any held-out row",
                             "authorized": len(direct), "superset_of_component_authorization": True,
                             "admitted_only_by_gene_direct": len(only_direct),
                             "their_pathogenic_percent": pct(sum(labels[x] for x in only_direct), len(only_direct))},
        "one_gene_rule_1d1b3": {"rule": "one canonical gene per row (whole field), refuse a training row whose gene has a "
                                        "held-out row (the 1D-1b.3 LabelAuthority)",
                                "authorized": len(train) - len(refused_1d1b3), "refused_training_rows": len(refused_1d1b3),
                                "refused_hgnc_genes": [g for g in refused_genes if g.startswith("HGNC:")],
                                "refused_unresolved_identities": sum(1 for g in refused_genes if not g.startswith("HGNC:")),
                                "refused_rows_also_excluded_by_components": sum(
                                    1 for x in refused_1d1b3 if x not in fa.authorized_rows),
                                "authorized_by_it_but_excluded_by_components": sum(
                                    1 for r in train if one_gene[r.row_id] not in held_one
                                    and r.row_id not in fa.authorized_rows)}}
    del one_gene, held_one
    auth_sorted = sorted(fa.authorized_rows)
    excluded = [r.row_id for r in train if r.row_id not in fa.authorized_rows]
    rec["label_mix_pathogenic_percent"] = {
        "training": pct(sum(labels[r.row_id] for r in train), len(train)),
        "authorized": pct(sum(labels[x] for x in auth_sorted), len(auth_sorted)),
        "excluded": pct(sum(labels[x] for x in excluded), len(excluded)),
        "held_out": pct(sum(labels[r.row_id] for r in held), len(held))}
    step("compare")

    comp = P.split_components([r for r in rows if r.partition is not P.Partition.TRAIN or r.gene_complete])
    auth_comp = [comp[x] for x in auth_sorted]
    rec["weights"] = {}
    for alpha in ALPHAS:
        w = P.tempered_component_weights(auth_comp, alpha)
        rec["weights"][str(alpha)] = {"effective_sample_size": round(w["effective_sample_size"], 3),
                                      "ess_percent": round(100 * w["ess_fraction"], 3),
                                      "max_to_min_weight": round(w["max_to_min_weight"], 3),
                                      "rows": w["rows"], "components": w["components"]}
    y_auth = np.array([labels[x] for x in auth_sorted], dtype=np.int64)
    _, _, ck = P.component_checkpoint_split(y_auth, np.array(auth_comp), fraction=0.15, seed=42)
    rec["checkpoint"] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in ck.items()}
    step("weights_and_checkpoint")

    # GNN-EARLY-STOP-ROW-SPLIT-1: the legacy call on every training row, in persisted order.
    y_train = np.array([labels[r.row_id] for r in train], dtype=np.int64)
    tr, va = train_test_split(np.arange(len(train)), test_size=0.15, random_state=42, stratify=y_train)
    tr_symbols = {raw_symbol[train[i].row_id] for i in tr} - {""}
    tr_genes = set().union(*(train[i].gene_ids for i in tr))
    tr_comp = {comp[train[i].row_id] for i in tr if train[i].row_id in comp}
    rec["legacy_early_stop"] = {
        **LEGACY_EARLY_STOP, "training_rows": len(train), "early_stopping_rows": int(len(va)),
        "early_stopping_rows_sharing_a_gene_with_inner_training": sum(1 for i in va if train[i].gene_ids & tr_genes),
        "early_stopping_rows_in_an_inner_training_component": sum(
            1 for i in va if train[i].row_id in comp and comp[train[i].row_id] in tr_comp),
        "early_stopping_rows_sharing_a_symbol_with_inner_training": sum(
            1 for i in va if raw_symbol[train[i].row_id] in tr_symbols),
        "early_stopping_rows_without_a_resolved_gene": sum(1 for i in va if not train[i].gene_ids),
        "early_stopping_symbols": len({raw_symbol[train[i].row_id] for i in va} - {""}),
        "early_stopping_symbols_also_inner_training_symbols": len(
            ({raw_symbol[train[i].row_id] for i in va} - {""}) & tr_symbols)}
    del raw_symbol, tr_symbols
    step("legacy_early_stop")

    tl = L.TrainingLabels(fa, labels, fit_scope={"model": "measurement-probe", "population": "run15",
                                                 "split": audit["partition_digest"]})
    tl.read(auth_sorted[0])
    for row in (held[0].row_id, "absent:0"):
        try:
            tl.read(row)
        except L.LabelAccessError:
            pass
    tl.masked([auth_sorted[1], held[1].row_id, "absent:1"])
    receipt = tl.receipt()
    rec["training_labels_receipt"] = {k: receipt[k] for k in ("authorized", "denied", "unknown", "masked_calls",
                                                               "masked_authorized", "masked_denied", "masked_unknown")}
    rec["bound_code"] = {s: C.symbol_closure_digest(REPO, s) for s in BOUND_SYMBOLS}
    step("labels")
    return rec


def render(rec: dict) -> bytes:
    return (json.dumps(rec, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hgnc", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path)
    group.add_argument("--check", type=Path)
    a = ap.parse_args(argv)
    log: list = []
    rec = measure(a.hgnc, a.splits, log)
    if a.check:
        committed = json.loads(a.check.read_bytes())
        committed.pop("run", None)
        if render(committed) != render(rec):
            differ = sorted(k for k in set(rec) | set(committed) if rec.get(k) != committed.get(k))
            print(f"POPULATION RECORD DIFFERS from the re-measurement in {differ}", file=sys.stderr)
            return 1
        print(f"population record: {a.check} equals the re-measurement (run section excepted)")
        return 0
    rec["run"] = {"steps": log, "platform": {"system": platform.system(), "python": platform.python_version()}}
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}; delete it deliberately first")
    a.out.write_bytes(render(rec))
    print(f"wrote {a.out} sha256 {hashlib.sha256(a.out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
