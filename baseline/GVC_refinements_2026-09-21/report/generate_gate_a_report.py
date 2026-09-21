"""Generate the Gate A measurement record FROM ITS ARTIFACTS.

No number or verdict in the output is typed by hand: every figure, and every word that
depends on a figure ("detectable", "not detectable", "not assessed"), is computed from the
JSON artifacts named on the command line, whose SHA-256 digests are recorded in the document.
A missing key stops generation with its exact path. Refuses to overwrite.
"""
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

ARTS = ["mc_census", "geneinfo_census", "provenance", "authentication", "residuals",
        "resolution", "components", "bundle", "strata_constraint", "strata_representation"]


class MissingKey(KeyError):
    pass


class Inconsistent(ValueError):
    pass


USED = set()   # (artifact, key path) read through need(); listed in the provenance appendix


def need(d, *path, name="?"):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            raise MissingKey(f"{name}: missing key {'/'.join(map(str, path))} (stopped at {k!r})")
        cur = cur[k]
    USED.add((name, "/".join(map(str, path))))
    return cur


def consistency(A):
    """Independent outputs that describe the SAME rows must agree, or nothing is generated.
    Ported from the retired build_measurement_doc.py (which flagged; this refuses)."""
    fails = []
    au, rs, cp, gi, rv = (A[k] for k in ("authentication", "residuals", "components", "geneinfo_census", "resolution"))
    sc, sr = A["strata_constraint"], A["strata_representation"]
    # Counter-derived state maps carry only OBSERVED states; they are proven complete by the
    # states-sum-to-rows check in build(), so an absent "disagree" means zero, not a lookup miss.
    dis = need(au, "clinical_sig", name="authentication").get("disagree", 0)
    kinds = sum(need(rs, "clinical_sig_disagreement_kinds", name="residuals").values())
    if dis != kinds:
        fails.append(f"authentication found {dis} clinical_sig disagreements; residuals classified {kinds}")
    rel = need(gi, "provenance", "relation_rows", name="geneinfo_census")
    res = sum(need(rv, "state_by_relation_rows", name="resolution").values())
    if rel != res:
        fails.append(f"census wrote {rel} relation rows; resolution accounts for {res}")
    col = need(sc, "component_column", name="strata_constraint")
    comp_key = {"component_genes_only": "genes_only", "component_all_resolved": "all_resolved"}.get(col)
    if comp_key is None:
        fails.append(f"unknown component column {col!r}")
    else:
        L = need(cp, comp_key, "leakage_restricted", name="components")
        if need(sc, "strata", "all", "rows", name="strata_constraint") != L["validation_rows"]:
            fails.append("stratified rows differ from the component build's evaluated validation rows")
        if need(sc, "strata", "leaked", "rows", name="strata_constraint") != L["validation_rows_sharing_component_with_train"]:
            fails.append("leaked-stratum rows differ from the component build's leaked validation rows")
    if need(sr, "component_column", name="strata_representation") != col:
        fails.append("the two stratified runs used different component columns")
    for s in ("unseen", "leaked", "all"):
        a_, b_ = (need(x, "strata", s, "rows", name=n_) for x, n_ in ((sc, "strata_constraint"), (sr, "strata_representation")))
        if a_ != b_:
            fails.append(f"stratum {s}: constraint run has {a_} rows, representation run has {b_}")
    if fails:
        raise Inconsistent("; ".join(fails))
    return ["authentication disagreements == residual classifications",
            "census relation rows == resolution relation rows",
            "stratified rows == component build's evaluated and leaked validation rows",
            "constraint and representation runs share component column and stratum sizes"]


def n(x):   return f"{x:,}"
def pct(a, b): return f"{a / b:.2%}" if b else "undefined"
def f6(x):  return f"{x:+.6f}"


def verdict(c):
    if not c["interval_reliable"]:
        return "not assessed (too few components)"
    return "detectable" if c["excludes_zero"] else "not detectable"


def build(A):
    L = []; w = L.append
    mc, gi, pv, au, rs = A["mc_census"], A["geneinfo_census"], A["provenance"], A["authentication"], A["residuals"]
    rv, cp, bd = A["resolution"], A["components"], A["bundle"]

    w("## 1. Source recovery from the pinned ClinVar VCF\n")
    epr = need(mc, "mc_entries_per_record", name="mc_census")
    multi = sum(v for k, v in epr.items() if int(k) > 1)
    dropped = sum((int(k) - 1) * v for k, v in epr.items())
    wmc = need(mc, "records_with_mc", name="mc_census")
    w(f"Molecular consequence: {n(need(mc, 'records', name='mc_census'))} records, {n(wmc)} with an MC field, "
      f"{n(need(mc, 'records_with_malformed_mc', name='mc_census'))} malformed. {n(multi)} records "
      f"({pct(multi, wmc)} of annotated) list more than one consequence; the former first-term parser discarded "
      f"{n(dropped)} consequence entries. Labels mapping to more than one accession: "
      f"{len(need(mc, 'labels_with_more_than_one_accession', name='mc_census'))}. "
      f"Unknown accessions: {len(need(mc, 'so_validation', 'unknown_accessions', name='mc_census'))}; "
      f"obsolete: {len(need(mc, 'so_validation', 'obsolete_accessions', name='mc_census'))}.\n")
    for row in need(mc, "so_validation", "label_differs_from_so_name", name="mc_census"):
        w(f"- label differs from SO name: `{row}`")
    gpr = need(gi, "genes_per_record", name="geneinfo_census")
    wgi = need(gi, "records_with_geneinfo", name="geneinfo_census")
    mg = sum(v for k, v in gpr.items() if int(k) > 1)
    w(f"\nGene associations (GENEINFO): {n(wgi)} records with gene information, "
      f"{n(need(gi, 'records_with_malformed_geneinfo', name='geneinfo_census'))} malformed, "
      f"{n(mg)} ({pct(mg, wgi)}) associated with more than one gene; "
      f"{n(need(gi, 'provenance', 'relation_rows', name='geneinfo_census'))} variant-gene relation rows; "
      f"{n(need(gi, 'distinct_gene_ids', name='geneinfo_census'))} distinct source GeneIDs. "
      f"Records at the maximum gene count ({max(map(int, gpr))}): {n(gpr[str(max(map(int, gpr)))])} "
      f"- whether this is a cap is NOT established.\n")
    for c in need(gi, "symbols_containing_colon", name="geneinfo_census"):
        w(f"- symbol containing ':' : `{c['symbol']}` GeneID {c['gene_id']} ({n(c['entries'])} entries)")

    w("\n## 2. Release provenance\n")
    w(f"VCF header: {', '.join(need(pv, 'vcf_meta', name='provenance'))}. variant_summary maximum LastEvaluated: "
      f"{need(pv, 'vs_last_evaluated_max', name='provenance')} (a lower bound on its release date). "
      f"Cohort VariationIDs present in the VCF: {n(need(pv, 'cohort_in_vcf', name='provenance'))} of "
      f"{n(need(pv, 'cohort_source_ids', name='provenance'))}; absent from variant_summary: "
      f"{n(need(pv, 'cohort_in_vcf_not_vs', name='provenance'))}.\n")

    w("## 3. Same-release authentication, keyed by VariationID\n")
    rows = need(au, "cohort_rows", name="authentication")
    for fld in ("review_status", "clinical_sig"):
        st = need(au, fld, name="authentication")
        if sum(st.values()) != rows:
            raise SystemExit(f"ABORT: {fld} states sum {sum(st.values())} != cohort rows {rows}")
        w(f"- {fld}: " + ", ".join(f"{k} {n(v)}" for k, v in st.items()) + f" (sum {n(rows)})")
    kinds = need(rs, "clinical_sig_disagreement_kinds", name="residuals")
    w(f"- clinical_sig disagreement kinds over all rows: " + (", ".join(f"{k} {n(v)}" for k, v in kinds.items()) or "none"))
    chk = need(rs, "shared_id_checks", name="residuals")
    w(f"- shared-VariationID groups: {n(need(rs, 'shared_id_ids', name='residuals'))}; "
      + ", ".join(f"{k} {n(v)}" for k, v in sorted(chk.items())) + "\n")

    w("## 4. Identity bundle (retrospective harmonisation snapshot)\n")
    w("| Resource | Bytes | SHA-256 | Last-Modified |\n|---|---:|---|---|")
    for name, m in need(bd, "resources", name="bundle").items():
        w(f"| {name} | {n(m['bytes_received'])} | `{m['sha256'][:16]}` | {m['headers'].get('Last-Modified')} |")
    if bd.get("failures"):
        w(f"\nFailed acquisitions recorded: {', '.join(bd['failures'])}")

    w("\n## 5. GeneID resolution\n")
    for key in ("state_by_distinct_gene_id", "state_by_relation_rows", "symbol_state_by_relation_rows", "hgnc_crossref_two_way"):
        w(f"- {key}: " + ", ".join(f"{k} {n(v)}" for k, v in need(rv, key, name="resolution").items()))

    w("\n## 6. Gene components and the old registry's leakage\n")
    w("Relation rows by NCBI type_of_gene: " + ", ".join(
        f"{k} {n(v)}" for k, v in need(cp, "relation_rows_by_type_of_gene", name="components").items()) + "\n")
    w("| Relation | Components | Evaluated validation rows sharing a component with training |\n|---|---:|---:|")
    for k, label in (("all_resolved", "all resolved records"), ("genes_only", f"excluding {', '.join(need(cp, 'exclude_types', name='components'))}")):
        r = need(cp, k, "leakage_restricted", name="components")
        a_, b_ = r["validation_rows_sharing_component_with_train"], r["validation_rows"]
        w(f"| {label} | {n(need(cp, k, 'components', name='components'))} | {n(a_)} / {n(b_)} = {pct(a_, b_)} |")

    for art, title in (("strata_constraint", "7. Constraint features, split by leakage"),
                       ("strata_representation", "8. Representation arms, split by leakage")):
        S = A[art]
        w(f"\n## {title}\n")
        w(f"Component column: `{need(S, 'component_column', name=art)}`; model selection: "
          f"{S.get('model_selection', 'not recorded (pre-r11 artifact)')}.\n")
        w("| Stratum | Rows | Components | Prevalence |\n|---|---:|---:|---:|")
        for s in ("unseen", "leaked", "all"):
            e = need(S, "strata", s, name=art)
            w(f"| {s} | {n(e['rows'])} | {n(e['components'])} | {e['prevalence']:.4f} |")
        contrasts = need(S, "strata", "all", "contrasts", name=art)
        for cname in contrasts:
            w(f"\n**{cname}** (Brier difference; negative favours the first)\n")
            w("| Stratum | Delta | 95% component interval | Verdict | Top-3 share of signed delta | Without top 3 (point) |")
            w("|---|---:|---|---|---:|---:|")
            for s in ("unseen", "leaked", "all"):
                c = need(S, "strata", s, "contrasts", cname, name=art)
                inf = c.get("influence")
                if inf:
                    top3 = sum(t["contribution"] for t in inf["top"][:3])
                    # A share of a SIGNED delta is unbounded near zero (e.g. -50%, -320%), so it is shown
                    # only where the delta is detectable; elsewhere it would read as meaning something.
                    share = (f"{top3 / c['brier_delta']:.1%}" if verdict(c) == "detectable"
                             else "omitted: delta not detectable")
                    wo = inf["leave_out_point_estimates"].get("without_top_3")
                    wo = f6(wo) if wo is not None else "nothing left"
                else:
                    share, wo = "not computed", "not computed"
                lo, hi = c["ci95_component"]
                w(f"| {s} | {f6(c['brier_delta'])} | [{f6(lo)}, {f6(hi)}] | {verdict(c)} | {share} | {wo} |")
            lk = need(S, "strata", "leaked", "contrasts", cname, name=art).get("influence")
            if lk:
                w("\nLargest leaked-stratum contributors, each component labelled by the registry "
                  "`gene_symbol` strings of its rows (not resolved genes): " + "; ".join(
                    f"{'/'.join(t['genes'][:3])} ({n(t['rows'])} rows, {f6(t['contribution'])})" for t in lk["top"][:5]))
    w("\nTop-3 shares are shown only where the delta is detectable: a share of a signed delta is "
      "unbounded near zero. Leave-out values are point estimates: the components removed were chosen "
      "after seeing the data, so no interval is attached.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for a_ in ARTS:
        ap.add_argument(f"--{a_.replace('_', '-')}", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    out = Path(a.output)
    if out.exists(): sys.exit(f"ABORT: {out} exists; refusing to overwrite")
    A, src = {}, []
    for k in ARTS:
        p = Path(getattr(a, k))
        b = p.read_bytes()
        A[k] = json.loads(b.decode("utf-8"))
        src.append((k, str(p), len(b), hashlib.sha256(b).hexdigest()))
    try:
        checks = consistency(A)
        body = build(A)
    except MissingKey as e:
        sys.exit(f"ABORT, nothing written: {e}")
    except Inconsistent as e:
        sys.exit(f"ABORT, nothing written -- the artifacts do not describe the same rows: {e}")
    digest = {k: h for k, _, _, h in src}
    body += ("\n\n## Cross-output consistency\n\nGeneration refuses unless every check passes. Passed:\n\n"
             + "\n".join(f"- {c}" for c in checks)
             + "\n\n## Appendix: provenance of every figure\n\nEach row is a key path read from an artifact; "
             "values nested inside that object are covered by its row.\n\n"
             "| Artifact | Key path | Artifact SHA-256 |\n|---|---|---|\n"
             + "\n".join(f"| {a_} | `{p_}` | `{digest[a_][:16]}` |" for a_, p_ in sorted(USED)))
    head = ["# MEASUREMENT 2026-09-21: Gate A identity, authentication and leakage", "",
            "Generated from artifacts by `generate_gate_a_report.py`; no figure or verdict below was typed by hand.",
            f"Generated {datetime.now(timezone.utc).isoformat()}.", "", "## Sources", "",
            "| Artifact | Path | Bytes | SHA-256 |", "|---|---|---:|---|"]
    head += [f"| {k} | `{p}` | {n(s)} | `{h}` |" for k, p, s, h in src]
    out.write_text("\n".join(head) + "\n\n" + body + "\n", encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size:,} bytes) from {len(src)} artifacts")


if __name__ == "__main__":
    main()
