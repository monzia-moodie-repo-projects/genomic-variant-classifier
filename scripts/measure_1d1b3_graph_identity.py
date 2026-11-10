"""Stage 1D-1b.3 real-data measurement: docs/census/GRAPH_IDENTITY_1D1b3.json.

Every number in the record is measured here from the source files' verified bytes (each file read through one
VerifiedReader pass, its SHA-256 recorded): STRING v12 exact-pair accounting, the quantization-model consistency, the
four-way threshold cross-tabulation, the canonical HGNC mapping and its collisions, the G1 / threshold-robust G1 / G2 gene
graphs and their computed identities, what the legacy StringDBGraph builds from the same files, the legacy cache key,
Reactome relations under the legacy and the global-size projections (induced-subgraph consistency), and, on the Run 15
cohort and splits, the canonical gene identity of the cohort and the label authority's scope.

Run (the files are the owner's; a clone does not hold them):
    python scripts/measure_1d1b3_graph_identity.py --links L --info I --hgnc H --reactome R --cohort C \
        --meta-val V --meta-test T --scratch DIR --out docs/census/GRAPH_IDENTITY_1D1b3.json

Peak memory is read through the ``resource`` module where it exists (Linux, macOS); elsewhere it is recorded as null.
Author: Monzia Moodie.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import logging
import platform
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from genomic_variant_classifier.data import kg_edges as K  # noqa: E402
from genomic_variant_classifier.information_sources import census as C  # noqa: E402
from genomic_variant_classifier.information_sources import graph_identity as G  # noqa: E402
from genomic_variant_classifier.information_sources import labels as L  # noqa: E402
from genomic_variant_classifier.information_sources.materialization import read_verified_bytes  # noqa: E402

logger = logging.getLogger(__name__)
SCRIPT = "scripts/measure_1d1b3_graph_identity.py"
G1_CHANNELS = ("neighborhood", "fusion", "cooccurence", "coexpression", "experimental")
THRESHOLD_MILLI = 700
LEGACY_MAX_SET_SIZE = 200
BOUND_SYMBOLS = (
    "src/genomic_variant_classifier/information_sources/graph_identity.py::read_string_pairs",
    "src/genomic_variant_classifier/information_sources/graph_identity.py::read_string_info",
    "src/genomic_variant_classifier/information_sources/graph_identity.py::read_hgnc",
    "src/genomic_variant_classifier/information_sources/graph_identity.py::model_consistency",
    "src/genomic_variant_classifier/information_sources/graph_identity.py::classify_threshold",
    "src/genomic_variant_classifier/information_sources/graph_identity.py::map_string_proteins",
    "src/genomic_variant_classifier/information_sources/graph_identity.py::GraphPolicy",
    "src/genomic_variant_classifier/information_sources/graph_identity.py::materialize_string_graph",
    "src/genomic_variant_classifier/information_sources/graph_identity.py::accept_graph",
    "src/genomic_variant_classifier/information_sources/graph_identity.py::read_gene_sets",
    "src/genomic_variant_classifier/information_sources/graph_identity.py::project_pathways",
    "src/genomic_variant_classifier/information_sources/graph_identity.py::induced_subgraph_consistency",
    "src/genomic_variant_classifier/information_sources/labels.py::LabelAuthority",
    f"{SCRIPT}::run",
)


def peak_mib() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def step(log: list, name: str, t0: float) -> None:
    log.append({"step": name, "seconds": round(time.perf_counter() - t0, 1), "peak_rss_mib": peak_mib()})
    print(f"[{name}] {log[-1]['seconds']} s, peak {log[-1]['peak_rss_mib']} MiB", flush=True)


def file_entry(path: Path, sha: str, logical: str) -> dict:
    return {"file": path.name, "sha256": sha, "bytes": path.stat().st_size, "logical_source": logical}


def legacy_string_graph(links: Path, info: Path, scratch: Path) -> dict:
    """What models/gnn.py StringDBGraph builds from the same files, and what its cache then returns for another source."""
    from genomic_variant_classifier.models.gnn import StringDBGraph

    with tempfile.TemporaryDirectory(dir=scratch) as cache:
        g = StringDBGraph(cache_dir=Path(cache), combined_score_threshold=THRESHOLD_MILLI, local_links_path=links,
                          local_info_path=info).build()
        nodes = list(g.nodes)
        out = {"nodes": len(nodes), "edges": g.number_of_edges(),
               "nodes_kept_as_raw_protein_ids": sum(1 for n in nodes if str(n).startswith("9606.")),
               "self_loops": sum(1 for u, v in g.edges() if u == v)}
        other = Path(cache) / "other_links.txt"
        other.write_text(" ".join(G.STRING_HEADER) + "\n9606.X 9606.Y 0 0 0 0 0 0 0 999\n9606.Y 9606.X 0 0 0 0 0 0 0 999\n")
        again = StringDBGraph(cache_dir=Path(cache), combined_score_threshold=THRESHOLD_MILLI, local_links_path=other,
                              local_info_path=info).build()
        out["cache_returned_for_a_different_links_file"] = {"edges": again.number_of_edges(),
                                                            "same_graph": again.number_of_edges() == out["edges"]}
        del g, again
    gc.collect()
    return out


def multigene_split_identity(sym, heldout_syms: set, hgnc) -> dict:
    """VARIANT-MULTIGENE-ISOLATION-1 (owner ruling 2026-11-10, section 7), measured: a ClinVar gene field may list several
    genes ('ABCC8;KCNJ11'). Read as a SET of canonical genes, which rows touch a gene of the other partition, and which
    groups of genes linked by such lists (union-find components) straddle the split."""
    import re

    def gene_set(s: str) -> frozenset:
        return frozenset(h for p in re.split(r"[;,|/]", s) if p.strip() for h in [hgnc.resolve(p.strip())[0]] if h)

    counts = sym.value_counts()
    sets = {s: gene_set(s) for s in counts.index}
    multi = {s for s in counts.index if re.search(r"[;,|/]", s)}
    train_genes = set().union(*(g for s, g in sets.items() if s not in heldout_syms))
    held_genes = set().union(*(g for s, g in sets.items() if s in heldout_syms))
    touch_t = {s for s, g in sets.items() if s not in heldout_syms and g & held_genes}
    touch_h = {s for s, g in sets.items() if s in heldout_syms and g & train_genes}
    parent: dict = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for g in sets.values():
        g = sorted(g)
        for x in g:
            find(x)
        for x in g[1:]:
            parent[find(x)] = find(g[0])
    comp_genes = Counter(find(x) for x in parent)
    comp_rows: Counter = Counter()
    comp_parts: dict = {}
    for s, n in counts.items():
        if sets[s]:
            c = find(next(iter(sets[s])))
            comp_rows[c] += int(n)
            comp_parts.setdefault(c, set()).add("heldout" if s in heldout_syms else "train")
    straddle = [c for c, p in comp_parts.items() if len(p) == 2]
    return {"multi_gene_symbols": len(multi), "rows_with_multi_gene_symbol": int(counts[list(multi)].sum()),
            "multi_gene_symbols_all_parts_resolve": sum(1 for s in multi if all(
                hgnc.resolve(p.strip())[0] for p in re.split(r"[;,|/]", s) if p.strip())),
            "rows_with_no_resolvable_gene": int(sum(int(n) for s, n in counts.items() if not sets[s])),
            "canonical_genes_on_both_sides": len(train_genes & held_genes),
            "training_rows_touching_a_heldout_gene": int(counts[list(touch_t)].sum()) if touch_t else 0,
            "heldout_rows_touching_a_training_gene": int(counts[list(touch_h)].sum()) if touch_h else 0,
            "gene_components": len(comp_genes), "largest_component_genes": max(comp_genes.values()),
            "largest_component_rows": max(comp_rows.values()),
            "components_straddling_split": len(straddle),
            "genes_in_straddling_components": int(sum(comp_genes[c] for c in straddle)),
            "rows_in_straddling_components": int(sum(comp_rows[c] for c in straddle))}


def run(args: argparse.Namespace) -> int:
    log: list = []
    t0 = time.perf_counter()
    rec: dict = {"schema": "gvc.graph-identity-record.1d1b3/1", "author": "Monzia Moodie", "inputs": {},
                 "platform": {"system": platform.system(), "python": platform.python_version()}}

    # ---- legacy StringDBGraph first (its pandas frame is the largest object; it is freed before the rest)
    rec["legacy_string_graph"] = legacy_string_graph(args.links, args.info, args.scratch)
    step(log, "legacy_string_graph", t0)

    # ---- STRING exact pairs
    t = G.read_string_pairs(args.links, scratch_dir=args.scratch, partitions=64)
    rec["inputs"]["string_links"] = file_entry(args.links, t.source_sha256, "dataset:string")
    proteins_in_links = set(t.proteins)
    rec["string_pairs"] = {"records": t.n_records, "pairs": len(t), "proteins": len(t.proteins),
                           "content_sha256": t.content_sha256, "partitions": t.partitions,
                           "max_partition_records": t.max_partition_records, "scratch_bytes": t.scratch_bytes,
                           "record_bytes": t.scratch_bytes // t.n_records}
    step(log, "read_string_pairs", t0)
    rec["quantization"] = G.model_consistency(t, G.QuantizationModel.ROUNDED_CHANNELS_FLOORED_COMBINED)
    cls = G.classify_threshold(t, G1_CHANNELS, THRESHOLD_MILLI)
    rec["threshold"] = {"channels": list(G1_CHANNELS), "threshold_milli": THRESHOLD_MILLI,
                        "model": cls.model.value, "crosstab": cls.crosstab(), "g1_selected": int(cls.selected.sum()),
                        "decided_exactly": dict(cls.decided_exactly),
                        "g2_published": int(cls.published_passes.sum())}
    step(log, "threshold", t0)

    # ---- canonical mapping
    hgnc = G.read_hgnc(args.hgnc)
    rec["inputs"]["hgnc"] = file_entry(args.hgnc, hgnc.source_sha256, "dataset:hgnc")
    info, info_sha = G.read_string_info(args.info)
    rec["inputs"]["string_info"] = file_entry(args.info, info_sha, "dataset:string")
    mapping, mrep = G.map_string_proteins(info, info_sha, hgnc)
    mrep["mapping_sha256"] = mapping.mapping_sha256
    mrep["info_proteins_without_links"] = len(set(info) - proteins_in_links)
    mrep["link_proteins_without_info"] = len(proteins_in_links - set(info))
    rec["mapping"] = mrep
    step(log, "mapping", t0)

    # ---- the three graphs, accepted by recomputation
    def policy(**kw) -> G.GraphPolicy:
        base = dict(source="dataset:string", release="12.0", source_sha256=t.source_sha256, info_sha256=info_sha,
                    mapping_source="dataset:hgnc", mapping_source_sha256=hgnc.source_sha256, regime=G.Regime.G1,
                    selection_channels=G1_CHANNELS, attribute_channels=("experimental", "coexpression"),
                    threshold_milli=THRESHOLD_MILLI, quantization=G.QuantizationModel.ROUNDED_CHANNELS_FLOORED_COMBINED)
        base.update(kw)
        return G.GraphPolicy(**base)

    policies = {"G1": policy(), "G1_threshold_robust": policy(threshold_robust_only=True),
                "G2": policy(regime=G.Regime.G2, selection_channels=G.STRING_CHANNELS, quantization=None,
                             attribute_channels=("experimental", "database", "coexpression"))}
    ledger = C.strict_load(REPO / C.LEDGER_PATH)
    rec["graphs"] = {}
    string_nodes_hgnc: set[str] = set()
    for name, pol in policies.items():
        g = G.materialize_string_graph(t, mapping, pol)
        accepted = G.accept_graph(pol, g)
        rep = dict(g.report)
        rep.pop("threshold_crosstab", None)
        rec["graphs"][name] = {"policy": pol.as_dict(), "policy_sha256": pol.policy_sha256,
                               "materialization": accepted.as_dict(), "report": rep,
                               "census": G.census_admission(pol, ledger)}
        string_nodes_hgnc = set(g.nodes)
        del g
    rec["legacy_cache"] = G.legacy_cache_collisions(REPO, list(policies.values()))
    step(log, "graphs", t0)
    protein_gene = mapping.gene_of()
    gene_of_name = {info[p]: protein_gene.get(p) for p in info}
    del t, cls
    gc.collect()

    # ---- cohort, splits and the label authority
    import pandas as pd

    pilots = _pilots_module()
    from genomic_variant_classifier.data.real_data_prep import DataPrepConfig, DataPrepPipeline

    cohort = read_verified_bytes(args.cohort)
    rec["inputs"]["cohort"] = file_entry(args.cohort, cohort.sha256, "dataset:clinvar_cohort")
    with tempfile.TemporaryDirectory(dir=args.scratch) as out_dir:
        frame, unread = pilots.load_population(DataPrepPipeline(DataPrepConfig(output_dir=Path(out_dir))), cohort.value)
    del cohort
    frame = frame[["variant_id", "gene_symbol", "label"]].reset_index(drop=True)
    gc.collect()
    vals = {}
    for key, path in (("meta_val", args.meta_val), ("meta_test", args.meta_test)):
        p = read_verified_bytes(path)
        rec["inputs"][key] = file_entry(path, p.sha256, "dataset:clinvar_cohort")
        vals[key] = set(pd.read_parquet(io.BytesIO(p.value), columns=["gene_symbol"])["gene_symbol"].dropna().astype(str))
    sym = frame["gene_symbol"].fillna("").astype(str)
    all_syms = set(sym) - {""}
    heldout_syms = vals["meta_val"] | vals["meta_test"]
    train_syms = all_syms - heldout_syms
    resolved = {s: hgnc.resolve(s) for s in all_syms}
    canon = {s: hid for s, (hid, _) in resolved.items() if hid is not None}
    by_id: dict[str, set[str]] = {}
    for s, hid in canon.items():
        by_id.setdefault(hid, set()).add(s)
    two_symbol = {hid: sorted(ss) for hid, ss in by_id.items() if len(ss) > 1}
    crossing = {hid: ss for hid, ss in two_symbol.items()
                if {("heldout" if s in heldout_syms else "train") for s in ss} == {"heldout", "train"}}
    string_names = {n for n, hid in gene_of_name.items() if hid is not None}
    preferred = set(info.values())
    by_id_only = {s for s in all_syms if canon.get(s) in string_nodes_hgnc and s not in preferred}
    rec["cohort"] = {"rows": int(len(frame)), "columns_not_read": unread, "symbols": len(all_syms),
                     "rows_without_symbol": int((sym == "").sum()),
                     "symbol_resolution": dict(sorted(Counter(st.value for _, st in resolved.values()).items())),
                     "val_symbols": len(vals["meta_val"]), "test_symbols": len(vals["meta_test"]),
                     "val_test_symbol_overlap": len(vals["meta_val"] & vals["meta_test"]),
                     "split_symbols_not_in_cohort": len(heldout_syms - all_syms),
                     "hgnc_ids_under_two_or_more_symbols": len(two_symbol),
                     "hgnc_ids_whose_symbols_straddle_train_and_heldout": len(crossing),
                     "straddling_examples": [[hid, ss] for hid, ss in sorted(crossing.items())][:20],
                     "symbols_matching_a_string_preferred_name": len(all_syms & preferred),
                     "symbols_whose_hgnc_id_is_a_string_node": sum(1 for s in all_syms if canon.get(s) in string_nodes_hgnc),
                     "symbols_with_a_string_name_but_not_mapped": len(all_syms & (preferred - string_names)),
                     "symbols_reaching_string_only_by_hgnc_id": len(by_id_only),
                     "rows_of_those_symbols": int(sym.isin(by_id_only).sum()),
                     "symbols_reaching_string_only_by_name": sum(
                         1 for s in all_syms if s in preferred and canon.get(s) not in string_nodes_hgnc)}
    rec["cohort"]["canonical_gene_sets"] = multigene_split_identity(sym, heldout_syms, hgnc)
    if frame["variant_id"].duplicated().any():
        raise ValueError("cohort variant ids are not unique; the label authority needs one record per row")
    part = np.where(sym.isin(heldout_syms).to_numpy(), "heldout", "train")
    gene_ids = [canon.get(s) or f"unresolved:{s}" for s in sym]
    authority = L.LabelAuthority(L.LabelRecord(str(v), g, int(y), p) for v, g, y, p in
                                 zip(frame["variant_id"], gene_ids, frame["label"], part))
    a = authority.authorize()
    rec["label_authority"] = {"authorized_rows": a.authorized_rows, "authorized_genes": a.authorized_genes,
                              "heldout_rows": a.heldout_rows, "heldout_genes": a.heldout_genes,
                              "refused_training_rows": a.refused_training_rows, "refused_genes": list(a.refused_genes),
                              "gene_identity": "HGNC id where the symbol resolves (approved or unique previous), "
                                               "else 'unresolved:<symbol>'"}
    step(log, "cohort_and_labels", t0)
    del authority, frame
    gc.collect()

    # ---- Reactome: legacy projection versus the global-size projection
    legacy_sets = K.parse_gmt(args.reactome)
    universes_sym = {"train": train_syms, "train_plus_heldout": all_syms}
    old = {k: set(K.co_membership_edges(legacy_sets, max_set_size=LEGACY_MAX_SET_SIZE, restrict_to=u))
           for k, u in universes_sym.items()}
    old_train_lost = sum(1 for e in old["train"] if e not in old["train_plus_heldout"])
    old_train_added = sum(1 for e in old["train_plus_heldout"] if e[0] in train_syms and e[1] in train_syms
                          and e not in old["train"])
    gs = G.read_gene_sets(args.reactome, hgnc, source_id="reactome", relation="shares_pathway")
    rec["inputs"]["reactome"] = file_entry(args.reactome, gs.source_sha256, "dataset:reactome")
    train_ids = {canon[s] for s in train_syms if s in canon}
    all_ids = {canon[s] for s in all_syms if s in canon}
    new_train = G.project_pathways(gs, train_ids, max_set_size=LEGACY_MAX_SET_SIZE)
    new_all = G.project_pathways(gs, all_ids, max_set_size=LEGACY_MAX_SET_SIZE)
    consistency = G.induced_subgraph_consistency(new_train.edges, train_ids, new_all.edges)
    global_ids = set(hgnc.approved.values())
    new_global = G.project_pathways(gs, global_ids, max_set_size=LEGACY_MAX_SET_SIZE)
    sizes = sorted((len(m) for m in gs.sets.values()), reverse=True)
    rng = np.random.default_rng(20261110)
    x = {g: float(v) for g, v in zip(sorted(all_ids), rng.normal(size=len(all_ids)))}
    capped_sets = {k: v for k, v in gs.sets.items() if 2 <= len(v) <= LEGACY_MAX_SET_SIZE}
    implicit = G.hypergraph_matvec(capped_sets, x)
    explicit = dict.fromkeys(x, 0.0)
    for (u, v), w in new_all.edges.items():
        explicit[u] += w * x[v]
        explicit[v] += w * x[u]
    rec["reactome"] = {
        "sets": len(gs.sets), "legacy_sets_after_name_merge": len(legacy_sets), "duplicate_names": gs.duplicate_names,
        "member_resolution": dict(gs.member_status_counts), "unresolved_members": gs.unresolved_members,
        "largest_canonical_sets": sizes[:5], "sets_over_cap_globally": new_global.sets_over_cap,
        "uncapped_clique_pairs": int(sum(m * (m - 1) // 2 for m in sizes)),
        "legacy": {"max_set_size_applied_to": "cohort-restricted size", "edges_train_universe": len(old["train"]),
                   "edges_train_plus_heldout_universe": len(old["train_plus_heldout"]),
                   "train_edges_removed_by_adding_heldout_genes": old_train_lost,
                   "train_edges_added_by_adding_heldout_genes": old_train_added},
        "global_size_projection": {"max_set_size_applied_to": "global canonical size",
                                   "edges_train_universe": len(new_train.edges),
                                   "edges_train_plus_heldout_universe": len(new_all.edges),
                                   "edges_global_universe": len(new_global.edges),
                                   "induced_subgraph_consistency": consistency},
        "hypergraph_equivalence_max_abs_difference": max(abs(implicit[g] - explicit[g]) for g in x),
    }
    step(log, "reactome", t0)

    rec["bound_code"] = {s: C.symbol_closure_digest(REPO, s) for s in BOUND_SYMBOLS}
    rec["steps"] = log
    args.out.write_bytes((json.dumps(rec, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii"))
    print(f"wrote {args.out} sha256 {hashlib.sha256(args.out.read_bytes()).hexdigest()}")
    return 0


def _pilots_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("measure_1d1b2_pilots", REPO / "scripts/measure_1d1b2_pilots.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("links", "info", "hgnc", "reactome", "cohort", "meta-val", "meta-test", "scratch", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
