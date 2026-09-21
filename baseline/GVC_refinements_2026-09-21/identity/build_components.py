"""Gene components from the RESOLVED ClinVar gene relation, and the old registry's leakage.

canonical_components is the reviewed builder from scientific_contracts.py with ONE change:
the identifier pattern is a parameter, so resolved NCBI GeneIDs ("NCBIGene:n") are accepted.
A test proves byte-identical output to the reference on HGNC input.

Components are built twice: from every resolved record, and excluding records whose NCBI
type_of_gene is in --exclude-types (default: biological-region, i.e. regulatory features,
not genes). Variants left with no gene after exclusion become declared singleton units.
The old registry (gene_symbol-keyed) is then audited: how many validation rows share a
component with training rows. Read-only on inputs; writes parquet + JSON.
"""
import argparse, collections, gzip, hashlib, json, re, sys
from datetime import datetime, timezone
from pathlib import Path


class ContractError(ValueError):
    pass


def stable_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def canonical_components(records, id_pattern=r"HGNC:[1-9][0-9]*"):
    relation, parent = {}, {}
    for variant, genes in records:
        if not isinstance(variant, str) or not variant or variant != variant.strip():
            raise ContractError("Invalid variant identity")
        if variant in relation:
            raise ContractError(f"Duplicate variant identity: {variant}")
        if isinstance(genes, (str, bytes)):
            raise ContractError("Gene IDs must be a collection, not a delimited string")
        genes = tuple(genes)
        if not genes or any(not isinstance(g, str) or not re.fullmatch(id_pattern, g) for g in genes):
            raise ContractError(f"Unresolved or invalid gene IDs for {variant}")
        genes = tuple(sorted(set(genes)))
        relation[variant] = genes
        for gene in genes:
            parent.setdefault(gene, gene)
    if not relation:
        raise ContractError("Empty relation")

    def root(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    for genes in relation.values():
        for gene in genes[1:]:
            a, b = root(genes[0]), root(gene)
            if a != b:
                parent[max(a, b)] = min(a, b)
    members = {}
    for gene in sorted(parent):
        members.setdefault(root(gene), []).append(gene)
    by_root = {r: "gene-component:" + stable_digest(gs) for r, gs in members.items()}
    components = {by_root[r]: tuple(gs) for r, gs in members.items()}
    variants = {v: by_root[root(gs[0])] for v, gs in relation.items()}
    return variants, components


NCBI = r"NCBIGene:[1-9][0-9]*"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("relation", "resolution", "gene-info", "cohort", "membership", "output-dir"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--exclude-types", nargs="*", default=["biological-region"])
    ap.add_argument("--restrict-to", default=None,
                    help="parquet with variant_id: ALSO report leakage over exactly these rows "
                         "(e.g. the evaluated validation predictions), for a like-for-like comparison")
    a = ap.parse_args()
    out = Path(a.output_dir)
    if out.exists(): sys.exit(f"ABORT: {out} exists; refusing to overwrite")
    import pandas as pd

    res = pd.read_parquet(a.resolution)
    ok = res[res["state"].isin(["current", "migrated"])]
    m = ok.groupby("source_gene_id")["resolved_gene_id"].nunique()
    if (m > 1).any(): sys.exit(f"ABORT: source GeneIDs resolving to >1 target: {list(m[m > 1].index)[:10]}")
    resolved = dict(zip(ok["source_gene_id"].astype(str), ok["resolved_gene_id"].astype(str)))
    unresolved_ids = set(res.loc[~res["state"].isin(["current", "migrated"]), "source_gene_id"].astype(str))

    gi = pd.read_csv(a.gene_info, sep="\t", dtype=str, keep_default_na=False, usecols=[0, 1, 2, 9])
    gi.columns = [c.lstrip("#") for c in gi.columns]
    if list(gi.columns) != ["tax_id", "GeneID", "Symbol", "type_of_gene"]:
        sys.exit(f"ABORT: unexpected gene_info columns {list(gi.columns)}")
    gtype = dict(zip(gi["GeneID"], gi["type_of_gene"])); gsym = dict(zip(gi["GeneID"], gi["Symbol"]))

    rel = pd.read_parquet(a.relation, columns=["variation_id", "source_gene_id"])
    rel["source_gene_id"] = rel["source_gene_id"].astype(str); rel["variation_id"] = rel["variation_id"].astype(str)
    rel["resolved"] = rel["source_gene_id"].map(resolved)
    rel["type"] = rel["resolved"].map(gtype).fillna("UNRESOLVED")
    type_rows = rel["type"].value_counts().to_dict()

    all_sets, gene_sets = collections.defaultdict(set), collections.defaultdict(set)
    for v, r, t in zip(rel["variation_id"], rel["resolved"], rel["type"]):
        if isinstance(r, str):
            all_sets[v].add(f"NCBIGene:{r}")
            if t not in a.exclude_types:
                gene_sets[v].add(f"NCBIGene:{r}")
    variants_all = set(rel["variation_id"])
    only_unresolved = sorted(variants_all - set(all_sets))
    bridged = sum(1 for v, s in all_sets.items() if len(s) > len(gene_sets.get(v, ())) and gene_sets.get(v))

    comp_all, members_all = canonical_components(all_sets.items(), NCBI)
    comp_gene, members_gene = canonical_components(((v, s) for v, s in gene_sets.items() if s), NCBI)
    no_gene = sorted(set(all_sets) - set(comp_gene))
    for v in no_gene + only_unresolved:            # declared singleton units, never silently dropped
        comp_gene[v] = f"variant-singleton:{v}"
    for v in only_unresolved:
        comp_all[v] = f"variant-singleton:{v}"

    def describe(members):
        sizes = sorted((len(g) for g in members.values()), reverse=True)
        top = sorted(members.items(), key=lambda kv: -len(kv[1]))[:8]
        return {"components": len(members), "largest_gene_counts": sizes[:10],
                "largest_members": [[gsym.get(g.split(":")[1], g) for g in gs][:25] for _, gs in top]}

    cohort = pd.read_parquet(a.cohort, columns=["variant_id", "source_id"])
    cohort["source_id"] = cohort["source_id"].astype(str)
    mem = pd.read_parquet(a.membership)
    need = {"variant_id", "partition"}
    if need - set(mem.columns): sys.exit(f"ABORT: membership lacks {sorted(need - set(mem.columns))}; has {list(mem.columns)}")
    df = cohort.merge(mem[["variant_id", "partition"]], on="variant_id", how="inner", validate="many_to_one")

    restrict = None
    if a.restrict_to:
        restrict = set(pd.read_parquet(a.restrict_to, columns=["variant_id"])["variant_id"])

    def leakage(comp, only=None):
        # A component's partition SPAN comes from the FULL membership. Only the rows being
        # counted are restricted: computing the span from restricted rows alone would hide
        # every training row and report zero leakage.
        full = df.assign(component=df["source_id"].map(comp))
        parts = full.dropna(subset=["component"]).groupby("component")["partition"].agg(lambda s: frozenset(s))
        d = full if only is None else full[full["variant_id"].isin(only)]
        missing = int(d["component"].isna().sum())
        d = d.dropna(subset=["component"]).copy()
        d["parts"] = d["component"].map(parts)
        val = d[d["partition"] == "validation"]; test = d[d["partition"] == "test"]
        return {"rows_with_partition": int(len(d)), "rows_without_component": missing,
                "partition_counts": d["partition"].value_counts().to_dict(),
                "components_spanning_2plus_partitions": int((parts.map(len) >= 2).sum()),
                "components_spanning_3_partitions": int((parts.map(len) == 3).sum()),
                "validation_rows_sharing_component_with_train":
                    int(val["parts"].map(lambda p: "train" in p).sum()),
                "validation_rows": int(len(val)),
                "test_rows_sharing_component_with_train": int(test["parts"].map(lambda p: "train" in p).sum()),
                "test_rows": int(len(test))}

    summary = {
        "exclude_types": a.exclude_types,
        "relation_rows_by_type_of_gene": type_rows,
        "variants": len(variants_all), "variants_only_unresolved_genes": len(only_unresolved),
        "variants_left_without_gene_after_exclusion": len(no_gene),
        "variants_where_excluded_types_were_extra_links": bridged,
        "all_resolved": {**describe(members_all), "leakage_vs_old_registry": leakage(comp_all),
                         "leakage_restricted": leakage(comp_all, restrict) if restrict else None},
        "genes_only": {**describe(members_gene), "leakage_vs_old_registry": leakage(comp_gene),
                       "leakage_restricted": leakage(comp_gene, restrict) if restrict else None},
        "restricted_rows_requested": len(restrict) if restrict else None,
        "run_utc": datetime.now(timezone.utc).isoformat()}
    out.mkdir(parents=True)
    pd.DataFrame({"variation_id": list(comp_gene), "component_genes_only": list(comp_gene.values()),
                  "component_all_resolved": [comp_all.get(v) for v in comp_gene]}
                 ).to_parquet(out / "variant_components.parquet", index=False)
    (out / "components_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("relation rows by type_of_gene:", type_rows)
    print(f"variants {len(variants_all):,} | only-unresolved {len(only_unresolved):,} | "
          f"no gene after excluding {a.exclude_types}: {len(no_gene):,} | excluded types were extra links: {bridged:,}")
    for k in ("all_resolved", "genes_only"):
        s = summary[k]; L = s["leakage_vs_old_registry"]
        print(f"\n[{k}] components {s['components']:,} | largest gene counts {s['largest_gene_counts']}")
        for mm in s["largest_members"][:3]: print("    ", mm)
        print(f"  old registry: components spanning >=2 partitions {L['components_spanning_2plus_partitions']:,} "
              f"(all 3: {L['components_spanning_3_partitions']:,}) | rows without component {L['rows_without_component']:,}")
        print(f"  validation rows sharing a component with train: {L['validation_rows_sharing_component_with_train']:,} "
              f"/ {L['validation_rows']:,} = {L['validation_rows_sharing_component_with_train']/max(L['validation_rows'],1):.2%}")
        print(f"  test rows sharing a component with train: {L['test_rows_sharing_component_with_train']:,} "
              f"/ {L['test_rows']:,} = {L['test_rows_sharing_component_with_train']/max(L['test_rows'],1):.2%}")
        R = s.get("leakage_restricted")
        if R:
            print(f"  RESTRICTED to {R['rows_with_partition']:,} requested rows: validation sharing with train "
                  f"{R['validation_rows_sharing_component_with_train']:,} / {R['validation_rows']:,} = "
                  f"{R['validation_rows_sharing_component_with_train']/max(R['validation_rows'],1):.2%}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
