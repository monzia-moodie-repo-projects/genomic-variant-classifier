"""Resolve every source GeneID in the ClinVar gene relation against a frozen identity bundle.

Per source GeneID the state is one of:
  current                      in human gene_info
  migrated                     discontinued; history chain ends at a current human GeneID
  discontinued_no_replacement  history chain ends at a discontinuation without replacement
  history_chain_problem        cycle, excessive depth, or a chain ending outside gene_info
  unknown                      in neither gene_info nor human gene_history
Every migration path is preserved. Then, for resolved IDs, each source SYMBOL is checked
through the reviewer's gate_a.GeneResolver, which refuses ID/symbol conflicts. HGNC
cross-references are checked in BOTH directions: gene_info dbXrefs against the HGNC
complete set's entrez_id. File formats are read from headers and reported, not assumed.
Read-only on inputs; writes a resolution parquet and a summary JSON.
"""
import argparse, collections, gzip, importlib.util, json, re, sys
from datetime import datetime, timezone
from pathlib import Path

HGNC_RE = re.compile(r"HGNC:[1-9][0-9]*")


def need(cols, required, label):
    missing = [c for c in required if c not in cols]
    if missing:
        sys.exit(f"ABORT: {label} lacks required columns {missing}; header was {list(cols)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True); ap.add_argument("--relation", required=True)
    ap.add_argument("--gate-a", required=True, help="path to gate_a.py")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    out = Path(a.output_dir)
    if out.exists(): sys.exit(f"ABORT: {out} exists; refusing to overwrite")
    import pandas as pd
    b = Path(a.bundle)
    spec = importlib.util.spec_from_file_location("gate_a", a.gate_a)
    ga = importlib.util.module_from_spec(spec); sys.modules["gate_a"] = ga; spec.loader.exec_module(ga)
    fmt = {}

    # --- NCBI gene_info (human) ---
    gi = pd.read_csv(b / "Homo_sapiens.gene_info.gz", sep="\t", dtype=str, keep_default_na=False)
    gi.columns = [c.lstrip("#") for c in gi.columns]
    need(gi.columns, ["tax_id", "GeneID", "Symbol", "Synonyms", "dbXrefs"], "gene_info")
    fmt["gene_info_rows"] = len(gi); fmt["gene_info_tax_ids"] = gi["tax_id"].value_counts().to_dict()
    gi = gi[gi["tax_id"] == "9606"]
    if gi["GeneID"].duplicated().any(): sys.exit("ABORT: duplicate GeneID in human gene_info")
    xref_forms = collections.Counter(); gi_hgnc = {}
    for gid, x in zip(gi["GeneID"], gi["dbXrefs"]):
        toks = [t for t in x.split("|") if t.startswith("HGNC:")]
        for t in toks:
            rest = t[len("HGNC:"):]
            if HGNC_RE.fullmatch(rest): xref_forms["HGNC:HGNC:n"] += 1; gi_hgnc.setdefault(gid, []).append(rest)
            elif re.fullmatch(r"[1-9][0-9]*", rest): xref_forms["HGNC:n"] += 1; gi_hgnc.setdefault(gid, []).append("HGNC:" + rest)
            else: xref_forms[f"unrecognised:{t[:30]}"] += 1
    fmt["gene_info_hgnc_xref_forms"] = dict(xref_forms)
    multi_hgnc = {g: v for g, v in gi_hgnc.items() if len(set(v)) > 1}

    # --- HGNC complete set ---
    hg = pd.read_csv(b / "hgnc_complete_set.txt", sep="\t", dtype=str, keep_default_na=False)
    need(hg.columns, ["hgnc_id", "symbol", "status", "entrez_id"], "hgnc_complete_set")
    fmt["hgnc_rows"] = len(hg); fmt["hgnc_status_counts"] = hg["status"].value_counts().to_dict()
    hg_by_entrez = collections.defaultdict(set)
    for h, e in zip(hg["hgnc_id"], hg["entrez_id"]):
        if e: hg_by_entrez[e].add(h)

    # Two-way HGNC cross-reference audit
    xref = collections.Counter(); xref_conflicts = []
    for gid, hs in gi_hgnc.items():
        h = set(hs); e = hg_by_entrez.get(gid, set())
        if not e: xref["gene_info_hgnc_but_hgnc_lacks_entrez"] += 1
        elif h == e: xref["agree"] += 1
        else:
            xref["disagree"] += 1
            if len(xref_conflicts) < 25: xref_conflicts.append({"GeneID": gid, "gene_info": sorted(h), "hgnc": sorted(e)})
    xref["hgnc_entrez_absent_from_gene_info"] = sum(1 for e in hg_by_entrez if e not in set(gi["GeneID"]))
    owner = collections.defaultdict(set)
    for gid, hs in gi_hgnc.items():
        for h in set(hs): owner[h].add(gid)
    hgnc_shared = {h: sorted(g) for h, g in owner.items() if len(g) > 1}

    # --- gene_history (human rows only), streamed ---
    hist = {}; gh_forms = collections.Counter(); gh_header = None
    with gzip.open(b / "gene_history.gz", "rt", encoding="utf-8") as f:
        gh_header = f.readline().rstrip("\n").lstrip("#").split("\t")
        need(gh_header, ["tax_id", "GeneID", "Discontinued_GeneID", "Discontinued_Symbol"], "gene_history")
        it, ig, idg = gh_header.index("tax_id"), gh_header.index("GeneID"), gh_header.index("Discontinued_GeneID")
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c[it] != "9606": continue
            new = c[ig]
            gh_forms["numeric" if re.fullmatch(r"[1-9][0-9]*", new) else f"value:{new[:12]}"] += 1
            if c[idg] in hist and hist[c[idg]] != new:
                sys.exit(f"ABORT: Discontinued_GeneID {c[idg]} has conflicting replacements")
            hist[c[idg]] = new
    fmt["gene_history_header"] = gh_header; fmt["gene_history_human_rows"] = len(hist)
    fmt["gene_history_replacement_value_forms"] = dict(gh_forms)

    current = set(gi["GeneID"])

    def follow(g):
        path = [g]
        for _ in range(25):
            if g in current: return ("current" if len(path) == 1 else "migrated"), g, path
            if g not in hist: return ("unknown" if len(path) == 1 else "history_chain_problem"), None, path
            nxt = hist[g]
            if not re.fullmatch(r"[1-9][0-9]*", nxt): return "discontinued_no_replacement", None, path + [nxt]
            if nxt in path: return "history_chain_problem", None, path + [nxt]
            path.append(nxt); g = nxt
        return "history_chain_problem", None, path

    # --- Build the reviewer's resolver from current human records ---
    sym_of = dict(zip(gi["GeneID"], gi["Symbol"]))
    shared_ids = {g for gs in hgnc_shared.values() for g in gs}
    genes = []
    for gid, sym, syn in zip(gi["GeneID"], gi["Symbol"], gi["Synonyms"]):
        hs = sorted(set(gi_hgnc.get(gid, [])))
        # A cross-reference that is not one-to-one is WITHHELD and recorded, never guessed.
        h = hs[0] if len(hs) == 1 and gid not in shared_ids else None
        aliases = tuple(s for s in syn.split("|") if s and s != "-")
        genes.append(ga.Gene(f"NCBIGene:{gid}", 9606, sym, aliases, h))
    resolver = ga.GeneResolver(genes)

    rel = pd.read_parquet(a.relation)
    need(rel.columns, ["variation_id", "source_symbol", "source_gene_id"], "relation")
    rel["source_gene_id"] = rel["source_gene_id"].astype(str)
    rows = []
    state_ct, sym_ct = collections.Counter(), collections.Counter()
    pairs = rel.groupby(["source_gene_id", "source_symbol"]).size().reset_index(name="relation_rows")
    for gid, sym, nrows in zip(pairs["source_gene_id"], pairs["source_symbol"], pairs["relation_rows"]):
        state, resolved, path = follow(gid)
        sstate = None
        if resolved is not None:
            try:
                r = resolver.resolve(source_gene_id=f"NCBIGene:{resolved}", symbol=sym)
                cur = sym_of[resolved]
                sstate = ("symbol_matches_current" if sym == cur else
                          "symbol_not_current_no_conflict" if r.get("symbol_unconfirmed") else
                          "unexpected_state")   # unreachable if the resolver behaves as read
            except ga.GateError as e:
                sstate = f"conflict:{str(e)[:60]}"
        state_ct[state] += int(nrows)
        if sstate: sym_ct[sstate] += int(nrows)
        rows.append({"source_gene_id": gid, "source_symbol": sym, "relation_rows": int(nrows),
                     "state": state, "resolved_gene_id": resolved,
                     "current_symbol": sym_of.get(resolved) if resolved else None,
                     "history_path": "->".join(path), "symbol_state": sstate})
    res = pd.DataFrame(rows)
    out.mkdir(parents=True)
    res.to_parquet(out / "gene_id_resolution.parquet", index=False)
    by_id = res.drop_duplicates("source_gene_id")["state"].value_counts().to_dict()
    summary = {
        "formats_observed": fmt,
        "distinct_source_gene_ids": int(res["source_gene_id"].nunique()),
        "state_by_distinct_gene_id": by_id,
        "state_by_relation_rows": dict(state_ct),
        "symbol_state_by_relation_rows": dict(sym_ct),
        "hgnc_crossref_two_way": dict(xref),
        "hgnc_crossref_conflict_examples": xref_conflicts,
        "gene_info_ids_with_multiple_hgnc": {k: sorted(set(v)) for k, v in list(multi_hgnc.items())[:25]},
        "hgnc_ids_shared_by_multiple_gene_ids": hgnc_shared,
        "non_current_examples": res[res["state"] != "current"].head(30).to_dict("records"),
        "run_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "resolution_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("formats:", json.dumps({k: fmt[k] for k in ("gene_info_hgnc_xref_forms",
          "gene_history_replacement_value_forms", "gene_history_human_rows")}, default=str))
    print(f"distinct source GeneIDs: {summary['distinct_source_gene_ids']:,}")
    print("state by distinct GeneID:", by_id)
    print(f"state by relation rows (sum {sum(state_ct.values()):,}):", dict(state_ct))
    print("symbol state by relation rows:", dict(sym_ct))
    print("HGNC cross-reference, two-way:", dict(xref))
    print(f"HGNC IDs shared by >1 GeneID: {len(hgnc_shared)} | GeneIDs with >1 HGNC: {len(multi_hgnc)}")
    for r in summary["non_current_examples"][:15]: print("   ", r)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
