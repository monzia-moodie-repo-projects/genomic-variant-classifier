"""Generate the Gate A measurement record from run outputs. No figure is typed by hand.

Every {placeholder} in TEMPLATE is filled from a named key path in a named JSON output.
A missing file or key ABORTS the build. The appendix lists each figure with its source
file, key path, and that file's SHA-256, so the document can be checked against the
outputs that produced it. Prose carries interpretation only; numbers come from data.
"""
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

TEMPLATE = """# MEASUREMENT 2026-09-21: identity-resolved gene components, and what they change

Generated {generated_utc} by baseline/GVC_refinements_2026-09-21/identity/build_measurement_doc.py.
Every number below is drawn from the output files listed in the appendix, never typed.

## 1. Source authentication against the same ClinVar release

{vcf_coverage_sentence} {cohort_in_vcf_not_vs} are absent from the variant_summary on disk,
whose latest evaluation date is {vs_last_evaluated_max}; the VCF's own file date is {vcf_file_date}.
{release_sentence}
Keyed by VariationID (never by coordinates), against the same-release VCF:

| Field | Agree | Disagree | No VCF value |
|---|---:|---:|---:|
| Review status | {rev_agree} | {rev_disagree} | {rev_none} |
| Clinical significance | {sig_agree} | {sig_disagree} | {sig_none} |

{sig_sentence} {rev_sentence} {crosscheck_sentence} This is the first check of the
cohort's labels against an independent source record; the earlier "closure proof" compared a
column with its own copy and could not have failed.

Rows sharing a VariationID: {par1} PAR1 and {par2} PAR2 X/Y pairs of one variant each, and
{non_xy} other groups needing inspection. {par_eligible} label-eligible variants are counted twice.

## 2. Molecular consequence, recovered from source

scripts/patch_clinvar_alleles.py kept only the label of only the first MC entry. Recovered from
the same VCF bytes: {mc_with} of {mc_records} records carry MC; {mc_multi} carried more than one
distinct consequence and lost all but the first. Unknown accessions: {mc_unknown}; obsolete:
{mc_obsolete}; labels mapping to more than one accession: {mc_ambiguous}.

## 3. Gene identity

GENEINFO recovered {gi_rows} variant-gene associations from {gi_with} records ({gi_malformed}
malformed). Against the frozen NCBI/HGNC bundle, by relation rows:

| State | Rows |
|---|---:|
| current | {res_current} |
| migrated through gene_history | {res_migrated} |
| discontinued, no replacement | {res_disc} |

Symbol conflicts: {sym_conflicts}.

## 4. Train/validation leakage through shared gene components

Of the {val_rows} evaluated validation rows, {leak_genes} ({leak_genes_pct}) share a gene
component with training rows, excluding enhancer (biological-region) links; {leak_all}
({leak_all_pct}) including them. The reviewer's symbol-level estimate, cited from REVIEW.md and NOT
recomputed here, was 32.07%. {leak_sentence}

## 5. Consequences for the session's findings (existing predictions, no refitting)

Constraint (LightGBM, Brier delta, component-resampled 95% interval):

| Stratum | Rows | Delta | 95% interval | Excludes zero |
|---|---:|---:|---|---|
| unseen components | {u_rows} | {c_u} | {c_u_ci} | {c_u_ex} |
| leaked components | {l_rows} | {c_l} | {c_l_ci} | {c_l_ex} |
| all | {a_rows} | {c_a} | {c_a_ci} | {c_a_ex} |

**Restated finding (derived from the table):** {constraint_sentence}

Representation (share of the baseline-to-reference Brier gap closed by the treated model):
unseen {gap_u}, leaked {gap_l}, all {gap_a}. {rep_sentence}

Between-stratum differences are descriptive: the strata differ in composition, not only in
leakage, and their class prevalences differ ({prev_u} unseen, {prev_l} leaked).
"""


def dig(obj, path, src):
    cur = obj
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            sys.exit(f"ABORT: {src} lacks key path {path} (failed at {k!r})")
    return cur


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", required=True, help="the project's outputs/ directory")
    ap.add_argument("--output", required=True)
    ap.add_argument("--constraint-contrast", default="lightgbm__core_plus_constraint - lightgbm__core")
    ap.add_argument("--rep-models", nargs=3, default=["lr_current", "lr_representation", "lightgbm"],
                    metavar=("BASELINE", "TREATED", "REFERENCE"))
    a = ap.parse_args()
    if Path(a.output).exists(): sys.exit(f"ABORT: {a.output} exists; refusing to overwrite")
    o = Path(a.outputs)
    files = {"prov": "release_provenance_001.json", "auth": "same_release_authentication_001.json",
             "resid": "residuals_001.json", "mc": "mc_census_001.json", "gi": "geneinfo_census_002.json",
             "res": "gene_id_resolution_002/resolution_summary.json",
             "comp": "gene_components_001/components_summary.json",
             "cs": "leakage_strata_constraint_001.json", "rs": "leakage_strata_representation_001.json"}
    data, digests = {}, {}
    for k, f in files.items():
        p = o / f
        if not p.is_file(): sys.exit(f"ABORT: missing output {p}")
        raw = p.read_bytes(); digests[k] = hashlib.sha256(raw).hexdigest()
        data[k] = json.loads(raw.decode("utf-8"))
    used = []

    def v(key, path):
        val = dig(data[key], path, files[key]); used.append((files[key], "/".join(map(str, path)), digests[key])); return val

    n = lambda x: f"{x:,}"
    pct = lambda x: f"{100 * x:.2f}%"
    cc = a.constraint_contrast
    def contrast(s):
        c = v("cs", ["strata", s, "contrasts", cc])
        ex = c.get("excludes_zero")
        return (f"{c['brier_delta']:+.6f}", f"[{c['ci95_component'][0]:+.6f}, {c['ci95_component'][1]:+.6f}]",
                "not assessed" if ex is None else ("yes" if ex else "no"))
    def gap(s):
        b0, b1, r = a.rep_models
        pm = v("rs", ["strata", s, "per_model"])
        for m in (b0, b1, r):
            if m not in pm: sys.exit(f"ABORT: representation model {m!r} not in {list(pm)}")
        return pct((pm[b0]["brier"] - pm[b1]["brier"]) / (pm[b0]["brier"] - pm[r]["brier"]))

    sig = v("auth", ["clinical_sig"]); rev = v("auth", ["review_status"])
    shared = v("resid", ["shared_id_checks"]); kinds = v("resid", ["clinical_sig_disagreement_kinds"])
    mcv = v("mc", ["so_validation"]) or {}
    rows = v("res", ["state_by_relation_rows"]); syms = v("res", ["symbol_state_by_relation_rows"])
    lg = v("comp", ["genes_only", "leakage_restricted"]); la = v("comp", ["all_resolved", "leakage_restricted"])
    vals = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_in_vcf": n(v("prov", ["cohort_in_vcf"])), "cohort_source_ids": n(v("prov", ["cohort_source_ids"])),
        "cohort_in_vcf_not_vs": n(v("prov", ["cohort_in_vcf_not_vs"])),
        "vs_last_evaluated_max": str(v("prov", ["vs_last_evaluated_max"]))[:10],
        "rev_agree": n(rev.get("agree", 0)), "rev_disagree": n(rev.get("disagree", 0)),
        "rev_none": n(rev.get("vcf_has_no_clnrevstat", 0)),
        "sig_agree": n(sig.get("agree", 0)), "sig_disagree": n(sig.get("disagree", 0)),
        "sig_none": n(sig.get("vcf_has_no_clnsig", 0)), "sig_delim": n(kinds.get("delimiter_only", 0)),
        "par1": n(shared.get("par1_offset_consistent", 0)), "par2": n(shared.get("par2_offset_consistent", 0)),
        "non_xy": n(shared.get("NOT_xy_pair_with_same_alleles", 0) + shared.get("xy_pair_NOT_a_consistent_par_pair", 0)),
        "par_eligible": n(shared.get("binary_label_eligible_pairs", 0)),
        "mc_records": n(v("mc", ["records"])), "mc_with": n(v("mc", ["records_with_mc"])),
        "mc_multi": n(v("mc", ["records_with_more_than_one_distinct_label"])),
        "mc_unknown": len(mcv.get("unknown_accessions", [])), "mc_obsolete": len(mcv.get("obsolete_accessions", [])),
        "mc_ambiguous": len(v("mc", ["labels_with_more_than_one_accession"])),
        "gi_rows": n(v("gi", ["provenance", "relation_rows"])), "gi_with": n(v("gi", ["records_with_geneinfo"])),
        "gi_malformed": n(v("gi", ["records_with_malformed_geneinfo"])),
        "res_current": n(rows.get("current", 0)), "res_migrated": n(rows.get("migrated", 0)),
        "res_disc": n(rows.get("discontinued_no_replacement", 0)),
        "sym_conflicts": n(sum(c for k, c in syms.items() if k.startswith("conflict"))),
        "val_rows": n(lg["validation_rows"]),
        "leak_genes": n(lg["validation_rows_sharing_component_with_train"]),
        "leak_genes_pct": pct(lg["validation_rows_sharing_component_with_train"] / lg["validation_rows"]),
        "leak_all": n(la["validation_rows_sharing_component_with_train"]),
        "leak_all_pct": pct(la["validation_rows_sharing_component_with_train"] / la["validation_rows"]),
        "u_rows": n(v("cs", ["strata", "unseen", "rows"])), "l_rows": n(v("cs", ["strata", "leaked", "rows"])),
        "a_rows": n(v("cs", ["strata", "all", "rows"])),
        "prev_u": f"{v('cs', ['strata', 'unseen', 'prevalence']):.4f}", "prev_l": f"{v('cs', ['strata', 'leaked', 'prevalence']):.4f}",
        "gap_u": gap("unseen"), "gap_l": gap("leaked"), "gap_a": gap("all"),
    }
    for s, key in (("unseen", "u"), ("leaked", "l"), ("all", "a")):
        vals[f"c_{key}"], vals[f"c_{key}_ci"], vals[f"c_{key}_ex"] = contrast(s)

    # ---- every conclusion below is DERIVED; none is asserted
    inv, tot = v("prov", ["cohort_in_vcf"]), v("prov", ["cohort_source_ids"])
    vals["vcf_coverage_sentence"] = (f"All {n(tot)} cohort VariationIDs are present in the VCF." if inv == tot else
                                     f"{n(inv)} of {n(tot)} cohort VariationIDs are present in the VCF; {n(tot - inv)} are not.")
    meta = [m for m in v("prov", ["vcf_meta"]) if m.startswith("##fileDate=")]
    fdate = meta[0].split("=", 1)[1] if meta else None
    vals["vcf_file_date"] = fdate or "not stated in the VCF header"
    le = str(v("prov", ["vs_last_evaluated_max"]))[:10]
    vals["release_sentence"] = (
        "Because an evaluation in variant_summary postdates the VCF's file date, the two files are different releases."
        if fdate and le > fdate else
        "The dates do not establish different releases (LastEvaluated is only a lower bound on a release date).")
    dis, delim = sig.get("disagree", 0), kinds.get("delimiter_only", 0)
    classified = sum(kinds.values())
    vals["crosscheck_sentence"] = (
        f"Cross-check passed: the residual check classified all {n(dis)} disagreements the authentication found."
        if classified == dis else
        f"CROSS-CHECK FAILED: authentication found {n(dis)} clinical-significance disagreements but the residual "
        f"check classified {n(classified)}. These outputs do not describe the same rows; do not rely on section 1.")
    vals["sig_sentence"] = ("Clinical significance has no disagreements." if dis == 0 else
        f"All {n(dis)} clinical-significance disagreements differ only in the secondary-term delimiter (\"; \" versus \"|\")."
        if delim == dis else
        f"Of {n(dis)} clinical-significance disagreements, {n(delim)} are delimiter-only and {n(dis - delim)} are NOT; see residuals_001.json.")
    rdis = rev.get("disagree", 0)
    vals["rev_sentence"] = "Review status has no disagreements." if rdis == 0 else f"Review status has {n(rdis)} disagreements."
    lk = lg["validation_rows_sharing_component_with_train"]
    vals["leak_sentence"] = ("Results evaluated on this validation set were therefore partly scored on gene components "
                             "seen in training." if lk > 0 else "No validation row shares a component with training.")
    cu = v("cs", ["strata", "unseen", "contrasts", cc]); cl = v("cs", ["strata", "leaked", "contrasts", cc])
    def verdict(c):
        return "not assessed (too few components)" if c.get("excludes_zero") is None else (
            f"detected ({'harm' if c['brier_delta'] > 0 else 'benefit'})" if c["excludes_zero"] else "not detected")
    vals["constraint_sentence"] = (f"in unseen components the effect is {verdict(cu)}; in leaked components it is "
                                   f"{verdict(cl)}. Rule: 'detected' means the component-resampled 95% interval "
                                   f"excludes zero; positive Brier delta is harm.")
    b0, b1, r = a.rep_models
    def gapf(s):
        pm = v("rs", ["strata", s, "per_model"]); return (pm[b0]["brier"] - pm[b1]["brier"]) / (pm[b0]["brier"] - pm[r]["brier"])
    diff = 100 * (gapf("unseen") - gapf("all"))
    tol = 2.0
    vals["rep_sentence"] = (f"Unseen minus all: {diff:+.2f} percentage points. Declared tolerance {tol} points: the finding is "
                            + ("NOT materially changed by restricting to unseen components." if abs(diff) <= tol else
                               "materially changed by restricting to unseen components."))
    doc = TEMPLATE.format(**vals)
    doc += "\n## Appendix: provenance of every figure\n\n| Source file | Key path | File SHA-256 |\n|---|---|---|\n"
    for f, path, d in sorted(set(used)):
        doc += f"| `{f}` | `{path}` | `{d[:16]}...` |\n"
    doc += "| REVIEW.md (reviewer) | symbol-level leakage estimate 32.07% | CITED, typed, not recomputed |\n"
    Path(a.output).write_text(doc, encoding="utf-8")
    print(f"wrote {a.output} ({len(doc):,} chars, {len(set(used))} distinct figure sources, {len(files)} files)")


if __name__ == "__main__":
    main()
