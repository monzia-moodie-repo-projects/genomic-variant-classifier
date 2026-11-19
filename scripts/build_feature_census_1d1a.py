"""Build docs/census/FEATURE_CENSUS_1D1a.json -- the stage 1D-1a information-source census (set "ao").

The REVIEWED content (meaning, sources, supervision, overlap, timing, label-criterion use, disposition, reason,
evidence, dependencies) is written below by hand, from the code at 7cb70ec2 and from the cited literature. The MEASURED
content (contract digest, surface inventory, consumer sites, graph inventory, each entry's consumers, producer digests)
is computed here from the repository, never typed. information_sources.census.verify_census re-measures all of it.

Usage, from the repository root:
    python scripts/build_feature_census_1d1a.py --repo . --out docs/census/FEATURE_CENSUS_1D1a.json
    python scripts/build_feature_census_1d1a.py --repo . --check      # byte-compare with the committed ledger
Author: Monzia Moodie.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from genomic_variant_classifier.information_sources import census as C
from genomic_variant_classifier.information_sources import consumers as K

VE = "src/genomic_variant_classifier/models/variant_ensemble.py"
RDP = "src/genomic_variant_classifier/data/real_data_prep.py"
DRV = "scripts/run_phase2_eval.py"
D = "src/genomic_variant_classifier/data/"
PRODUCER = f"{VE}::engineer_features"

A, X, R, CO, W = "admit_strict", "exclude_strict", "retire_predictive", "conditional_only", "withhold_pending_evidence"

# Literature, each identifier confirmed by search on 2026-11-08 (see docs/census/README.md).
LIT = {
    "acmg": "doi:10.1038/gim.2015.30",                                    # Richards et al. 2015, ACMG/AMP criteria
    "pejaver": "url:https://pubmed.ncbi.nlm.nih.gov/36413997/",           # PP3/BP4 calibration (ClinGen SVI 2022)
    "walker": "url:https://pubmed.ncbi.nlm.nih.gov/37352859/",            # splicing PP3/BP7 (ClinGen SVI 2023)
    "grimm": "doi:10.1002/humu.22768",                                    # two types of circularity (2015)
    "cadd": "url:https://pubmed.ncbi.nlm.nih.gov/24487276/",              # Kircher et al. 2014
    "revel": "url:https://pmc.ncbi.nlm.nih.gov/articles/PMC5065685",      # Ioannidis et al. 2016
    "polyphen": "url:https://pubmed.ncbi.nlm.nih.gov/20354512/",          # Adzhubei et al. 2010
    "alphamissense": "url:https://pubmed.ncbi.nlm.nih.gov/37733863/",     # Cheng et al. 2023
    "spliceai": "url:https://pubmed.ncbi.nlm.nih.gov/30661751/",          # Jaganathan et al. 2019
    "eve": "doi:10.1038/s41586-021-04043-8",                              # Frazer et al. 2021
    "nt": "doi:10.1038/s41592-024-02523-z",                               # Dalla-Torre et al. 2024
    "string": "url:https://string-db.org/help/scores/",                   # STRING evidence channels
}


def code(path: str, anchor: str) -> str:
    return f"code:{path}#{anchor}"


def ve(anchor: str) -> str:
    return code(VE, anchor)


DATASETS = {
    "clinvar_cohort": ("ClinVar GRCh38 clean cohort: variant identity, consequence, clinical significance (the label)",
                       "public", [code(RDP, 'df.loc[df["clinical_sig"].isin(PATHOGENIC_TERMS), "label"] = 1')]),
    "gnomad_af": ("gnomAD allele frequencies joined by chrom:pos:ref:alt", "public",
                  [code(RDP, 'gnomad_path, columns=["variant_id", "allele_freq"]')]),
    "gnomad_ymt_api": ("gnomAD Y and mitochondrial frequencies fetched from the live API (dataset gnomad_r4) and appended "
                       "to the gnomAD extract on 2026-06-16 as 'gnomad:'-prefixed records: Y exome AF else genome AF, MT "
                       "ac_hom / an (stage 1D-1c.1)", "public",
                       [code("scripts/build_gnomad_ymt_af.py", "def parse_mt_af(payload: dict) -> dict:")]),
    "grch38_reference_check": ("the per-row GRCh38 reference-allele check of the 101 bp window build "
                               "(delta_window_builder: ok and reason per cohort row, data/processed/"
                               "clinvar_grch38_clean_seq.parquet) -- reference identity for claim eligibility, stage 1D-1c.1",
                               "project-built",
                               [code(D + "delta_window_builder.py", 'return WindowResult(poly, poly, False, "non_acgt_allele")')]),
    "kg_1000g": ("1000 Genomes Phase 3 (release 20130502) global and super-population allele frequencies", "public",
                 [code(D + "thousandgenomes.py", "ALL.chr*.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz")]),
    "finngen": ("FinnGen annotated-variant allele frequencies (Finnish and non-Finnish-Swedish-Estonian)", "public",
                [code(D + "finngen.py", "# Data source: https://r10.finngen.fi/")]),
    "dbsnp": ("dbSNP build 156 allele frequency", "public",
              [code(D + "dbsnp.py", "Build 156 VCF")]),
    "dbnsfp": ("dbNSFP index of precomputed SIFT, PolyPhen-2 HDIV, REVEL, CADD, phyloP100way and GERP++ scores",
               "academic", [code(D + "dbnsfp.py", "Delivers six pre-computed pathogenicity and conservation scores for all")]),
    "revel_v1_3": ("REVEL v1.3 (2021-05-03), the publisher's transcript-aware release: member revel_with_transcript_ids of "
                   "revel-v1.3_all_chromosomes.zip (Zenodo 10.5281/zenodo.7072866; MD5 3ea2bc33e6b5455fc7e9899da863b5fe as "
                   "published, SHA-256 815801d832e4ab76922dddfd901d7f24a755df57078a7138aa14f615da86a8a2 as measured): 82,100,677 "
                   "rows of chromosome, GRCh37 and lifted GRCh38 position, alleles, amino-acid change, score and unversioned "
                   "Ensembl transcript identifiers. Registered at stage 1D-1c.2A; no producer consumes it yet (stage 1D-1c.2B)",
                   "academic (the publisher's terms: free for non-commercial use; Zenodo labels the record ODbL 1.0 -- the "
                   "stricter publisher term is applied; nothing is redistributed. The project is strictly research and "
                   "non-commercial, confirmed by the owner on 2026-11-18, so the term is met)",
                   ["record:docs/census/PREDICTOR_SOURCES_1D1c2.json", "url:https://sites.google.com/site/revelgenomics/downloads"]),
    "gencode_v50": ("GENCODE release 50 comprehensive annotation, GRCh38 (gencode.v50.annotation.gtf.gz, SHA-256 "
                    "83fba3e9b03f0b8c958f3595c6c350adc55f468abf8b0e47b6d5284cfe13a453): the MANE Select and Ensembl-canonical "
                    "transcript of each gene, measured at stage 1D-1c.2A", "public", ["record:docs/census/PREDICTOR_SOURCES_1D1c2.json"]),
    "clinvar_variant_summary": ("ClinVar variant_summary.txt.gz (SHA-256 5fe4fe10783391d01dc414dc5583a3e63487b67f8cd3c8429d59227cd5f4f37f, "
                                "8,992,046 rows): the release-level table the cohort is built from. Its Name declares the "
                                "RefSeq transcript (with version) and the protein change, which the cohort connector drops "
                                "(COHORT-TRANSCRIPT-UNDECLARED-1)", "public",
                                [code(D + "database_connectors.py", 'df["transcript_id"] = None'), "record:docs/census/PREDICTOR_SOURCES_1D1c2.json"]),
    "phylop_bigwig": ("PhyloP conservation BigWig track (second phyloP producer)", "public",
                      [code(RDP, "PHYLOPLOG-1 (2026-08-12).")]),
    "spliceai": ("SpliceAI precomputed masked SNV delta scores (hg38)", "public",
                 [code(D + "spliceai.py", "File: spliceai_scores.masked.snv.hg38.vcf.gz")]),
    "alphamissense": ("AlphaMissense hg38 precomputed missense pathogenicity", "public (CC BY-NC-SA)",
                      [code(RDP, 'df["alphamissense_score"] = merged["alphamissense_score"].fillna(DEFAULT).values')]),
    "eve": ("EVE per-protein variant scores (evemodel.org)", "public",
            [code(D + "eve.py", "DEFAULT_SCORE = 0.5")]),
    "gnomad_constraint": ("gnomAD v4.1 gene constraint metrics (pLI, LOEUF, lof.oe, syn_z, mis_z)", "public",
                          [code(D + "connectors/connector_gnomad_constraint.py", "gnomAD v4.1 gene constraint connector.")]),
    "uniprot_variants": ("UniProt humsavar-style variant table grouped by gene symbol", "public",
                         [code(RDP, 'lambda x: (x == "pathogenic").sum(),')]),
    "gtex": ("GTEx gene median TPM per tissue (bulk parquet); single-tissue eQTL via the per-gene API path", "public",
             [code(D + "gtex.py", "gtex_min_eqtl_pval        max -log10(p) across tissues")]),
    "omim": ("OMIM mim2gene and genemap2 (registration and licence)", "licensed",
             [code(D + "omim.py", "MIM_type values that count as phenotype:")]),
    "clingen": ("ClinGen Gene-Disease Validity classifications", "public",
                [code(D + "clingen.py", "When a gene appears in multiple disease-gene pairs, the MAXIMUM score")]),
    "lovd": ("LOVD variant classifications", "public",
             [code(D + "lovd.py", "This ordinal encoding preserves the clinical severity ranking and lets")]),
    "cosmic_cmc": ("COSMIC Cancer Mutation Census", "licensed",
                   [code(D + "connectors/connector_cosmic.py", '_C_TIER = "MUTATION_SIGNIFICANCE_TIER"')]),
    "kegg": ("KEGG human pathway memberships (REST)", "academic",
             [code(D + "kegg.py", "hsa05xxx: cancers, neurodegenerative, immune,")]),
    "reactome": ("Reactome gene-to-pathway membership (all levels) and ReactomePathways.gmt", "public (CC0)",
                 [code(D + "reactome.py", "Reactome pathway-membership connector")]),
    "rnaseq_matrix": ("quantified RNA-seq expression matrix with optional case/control contrast", "project-built",
                      [code(D + "rnaseq.py", "rnaseq_log2fc          float  log2 fold-change case/control (0 if no DE)")]),
    "esm2": ("ESM-2 protein language model and a UniProt sequence index", "public",
             [code(D + "esm2.py", "L2 distance between the wildtype and mutant residue embeddings at the")]),
    "nucleotide_transformer": ("Nucleotide Transformer v2 DNA language model and the 101 bp sequence windows",
                               "public", [code(D + "genomic_lm.py", "masked-LM log-likelihood ratio at the")]),
    "protein_coordinates": ("AlphaMissense-derived protein coordinates, with an HGVSp fallback", "public",
                            [code(RDP, "pc = ProteinCoordConnector(alphamissense_file=_am_tsv)")]),
    "uniprot_index": ("UniProt reviewed human index: accession, entry name, HGNC symbol and sequence (EVE entry-name "
                      "resolution; the ESM-2 sequence index)", "public",
                      [code(D + "eve.py", "def load_eve_entry_map(parquet_path: object) -> dict[str, str]:")]),
    "string": ("STRING v12 human protein links (detailed channels) and protein info", "public (CC BY)",
               [code("src/genomic_variant_classifier/models/gnn.py",
                     '"""Builds a NetworkX protein-interaction graph from STRING DB."""')]),
    "kg_gene_sets": ("gene-set files for knowledge-graph relations: Reactome GMT, KEGG GMT, GO GMT, ClinGen CSV, "
                     "OMIM table", "mixed", [code(D + "kg_edges.py", 'KG_SOURCES = {')]),
    "hgnc": ("HGNC complete set: approved symbols, previous symbols and stable HGNC identifiers (canonical gene "
             "identity for graph nodes and gene sets, stage 1D-1b.3, and for the gene sets and split components of a "
             "population, stage 1D-1b.4)", "public (CC0)",
             ["record:docs/census/GRAPH_IDENTITY_1D1b3.json", "record:docs/census/POPULATION_ISOLATION_1D1b4.json",
              code("src/genomic_variant_classifier/information_sources/graph_identity.py",
                   'HGNC_REQUIRED = ("hgnc_id", "symbol", "status", "prev_symbol", "alias_symbol")')]),
}


def F(name, disp, meaning, sources, sup, thr, overlap, time, crit, reason, evidence, depends=(), state="verified"):
    return {"name": name, "disposition": disp, "producer": PRODUCER, "meaning": meaning, "source_ids": list(sources),
            "source_state": state, "consumers": [], "evidence_refs": list(evidence), "reason": reason,
            "upstream_supervision": sup, "threshold_supervision": thr, "upstream_overlap": overlap,
            "source_time_status": time, "depends_on": list(depends), "label_criterion_use": crit}


AF_SRC = ["dataset:gnomad_af", "dataset:kg_1000g", f"code:{RDP}::DataPrepPipeline._join_gnomad"]
AF_REASON = ("Population allele frequency is measured, unsupervised evidence of purifying selection. It is also the "
             "evidence of ACMG/AMP BA1, BS1 and PM2, so the label was partly assigned from it (annotation circularity, "
             "recorded in label_criterion_use); 1D-2 must report a criterion-independent evaluation. Absence is not "
             "separable from a failed join (MISSINGNESS-SEMANTICS-1); the representation, not the information, is "
             "defective.")
VT_REASON = ("Derived from the variant's own reference and alternate alleles; no external source, no supervision. "
             "In-frame length change is PM4 evidence, recorded as label-criterion use.")
CQ_SRC = ["dataset:clinvar_cohort", f"code:{RDP}::DataPrepPipeline._load_and_label"]
CQ_REASON = ("The consequence class is a property of the variant on its transcript; it is the evidence of PVS1 "
             "(null variants), PM4 and BP7, recorded as label-criterion use. Its annotation source and transcript "
             "choice are inherited from the cohort build.")
DBNSFP = ["dataset:dbnsfp", f"code:{D}dbnsfp.py::DbNSFPConnector"]
CONSTRAINT = ["dataset:gnomad_constraint", f"code:{D}connectors/connector_gnomad_constraint.py::GnomADConstraintConnector"]
GTEX = ["dataset:gtex", f"code:{D}gtex.py::annotate_gtex_expression_from_parquet"]
SPLICE = ["dataset:clinvar_cohort", "code:src/genomic_variant_classifier/pipelines/rna_pipeline.py::RNASpliceIsoformPipeline"]
SPLICE_REASON = ("Splice-site context is non-clinical sequence biology, admissible in principle; but the producer gates "
                 "on is_splice and consequence_severity, which do not exist until engineer_features runs AFTER "
                 "annotation, so every row receives the default (RNA-SPLICE-GATE-ORDER-1; Run 15: constant). Withheld "
                 "until a run measures real coverage.")
CONSTRAINT_REASON = ("Gene-level mutational constraint from a population mutation-rate model: no clinical labels, "
                     "transferable to unseen genes. Defaults for genes without a measurement are values "
                     "(MISSINGNESS-SEMANTICS-1).")
FG_SRC = ["dataset:finngen", f"code:{D}finngen.py::FinnGenConnector"]
FG_REASON = ("Measured allele frequency in a population biobank; no supervision. Population-data evidence (BA1/BS1/"
             "PM2). The R12 and R13 releases are two measurements of overlapping participants, not independent "
             "evidence.")

GNN_DEPS_COMMON = ["string_relation:combined_score_threshold"]
KG_SOURCES_ORDER = ("clingen", "go", "kegg", "omim", "reactome")

ENTRIES = [
    F("af_raw", A, "Allele frequency clipped at 0: gnomAD AF (cohort column, else gnomAD join), else the 1000 Genomes "
      "global AF; a variant with no frequency becomes 0.0. (finngen.py's comment calls FinnGen a third AF fallback; "
      "FinnGenConnector does not write allele_freq.)", AF_SRC, "none", "none", "not_applicable", "current_only",
      "acmg_population_data", AF_REASON, [ve('feats["af_raw"] = af'), LIT["acmg"]]),
    F("af_log10", A, "log10(af_raw + 1e-8); absent AF maps to -8.", AF_SRC, "none", "none", "not_applicable",
      "current_only", "acmg_population_data", AF_REASON, [ve('feats["af_log10"] = np.log10(af + 1e-8)'), LIT["acmg"]],
      ["af_raw"]),
    F("af_is_absent", A, "1 when af_raw == 0: absent from every joined frequency source OR never joined.", AF_SRC,
      "none", "none", "not_applicable", "current_only", "acmg_population_data", AF_REASON,
      [ve('feats["af_is_absent"] = (af == 0).astype(int)'), LIT["acmg"]], ["af_raw"]),
    F("af_is_ultra_rare", A, "1 when af_raw < 1e-4 (absent variants included).", AF_SRC, "none", "none",
      "not_applicable", "current_only", "acmg_population_data", AF_REASON,
      [ve('feats["af_is_ultra_rare"] = (af < 0.0001).astype(int)'), LIT["acmg"]], ["af_raw"]),
    F("af_is_rare", A, "1 when 1e-4 <= af_raw < 1e-3.", AF_SRC, "none", "none", "not_applicable", "current_only",
      "acmg_population_data", AF_REASON,
      [ve('feats["af_is_rare"] = ((af >= 0.0001) & (af < 0.001)).astype(int)'), LIT["acmg"]], ["af_raw"]),
    F("af_is_common", A, "1 when af_raw >= 0.01 (a project cut, not the BA1 0.05).", AF_SRC, "none", "none",
      "not_applicable", "current_only", "acmg_population_data", AF_REASON,
      [ve('feats["af_is_common"] = (af >= 0.01).astype(int)'), LIT["acmg"]], ["af_raw"]),
    F("ref_len", A, "Length of the reference allele (minimum 1).", CQ_SRC, "none", "none", "not_applicable",
      "not_applicable", "acmg_variant_type", VT_REASON, [ve('feats["ref_len"] = ref_len')]),
    F("alt_len", A, "Length of the alternate allele (minimum 1).", CQ_SRC, "none", "none", "not_applicable",
      "not_applicable", "acmg_variant_type", VT_REASON, [ve('feats["alt_len"] = alt_len')]),
    F("len_diff", A, "|alt_len - ref_len|.", CQ_SRC, "none", "none", "not_applicable", "not_applicable",
      "acmg_variant_type", VT_REASON, [ve('feats["len_diff"] = (alt_len - ref_len).abs()')]),
    F("is_snv", A, "1 when both alleles have length 1.", CQ_SRC, "none", "none", "not_applicable", "not_applicable",
      "acmg_variant_type", VT_REASON, [ve('feats["is_snv"] = ((ref_len == 1) & (alt_len == 1)).astype(int)')]),
    F("is_insertion", A, "1 when the alternate allele is longer.", CQ_SRC, "none", "none", "not_applicable",
      "not_applicable", "acmg_variant_type", VT_REASON, [ve('feats["is_insertion"] = (alt_len > ref_len).astype(int)')]),
    F("is_deletion", A, "1 when the reference allele is longer.", CQ_SRC, "none", "none", "not_applicable",
      "not_applicable", "acmg_variant_type", VT_REASON, [ve('feats["is_deletion"] = (ref_len > alt_len).astype(int)')]),
    F("is_indel", A, "is_insertion OR is_deletion (equal-length multi-base substitutions are 0).", CQ_SRC, "none",
      "none", "not_applicable", "not_applicable", "acmg_variant_type", VT_REASON,
      [ve('feats["is_indel"] = (feats["is_insertion"] | feats["is_deletion"]).astype(int)')],
      ["is_insertion", "is_deletion"]),
    F("consequence_severity", A, "Maximum project ordinal (CONSEQUENCE_SEVERITY) over the '&'-separated consequence "
      "terms; unknown terms 0.", CQ_SRC, "none", "none", "not_applicable", "not_applicable", "acmg_variant_type",
      CQ_REASON, [ve('feats["consequence_severity"] = consequence.map(')]),
    F("is_loss_of_function", A, "1 when the consequence names stop_gained, frameshift, splice donor/acceptor, "
      "start_lost or stop_lost.", CQ_SRC, "none", "none", "not_applicable", "not_applicable", "acmg_variant_type",
      CQ_REASON, [ve('feats["is_loss_of_function"] = consequence.str.contains('), LIT["acmg"]]),
    F("is_missense", A, "1 when the consequence contains 'missense'.", CQ_SRC, "none", "none", "not_applicable",
      "not_applicable", "acmg_variant_type", CQ_REASON, [ve('feats["is_missense"] = consequence.str.contains(')]),
    F("is_synonymous", A, "1 when the consequence contains 'synonymous'.", CQ_SRC, "none", "none", "not_applicable",
      "not_applicable", "acmg_variant_type", CQ_REASON, [ve('feats["is_synonymous"] = consequence.str.contains(')]),
    F("is_splice", A, "1 when the consequence contains 'splice' (region and canonical alike).", CQ_SRC, "none",
      "none", "not_applicable", "not_applicable", "acmg_variant_type", CQ_REASON,
      [ve('feats["is_splice"] = consequence.str.contains(')]),
    F("in_coding", A, "1 when the consequence names a coding or splice class.", CQ_SRC, "none", "none",
      "not_applicable", "not_applicable", "acmg_variant_type", CQ_REASON,
      [ve('feats["in_coding"] = consequence.str.contains(')]),
    F("cadd_phred", A, "CADD PHRED-scaled score from dbNSFP (or the optional REST connector); absent 15.0.",
      DBNSFP + [f"code:{D}cadd.py::CADDConnector"], "proxy_nonclinical", "none", "not_applicable", "current_only",
      "acmg_computational_evidence", "Trained to separate simulated variants from human-derived alleles: a "
      "non-clinical proxy objective. It is PP3/BP4 evidence (ClinGen calibration), recorded as label-criterion use."
      " Strict in principle; its MATERIALISED values are not yet a qualified observation (DBNSFP-INDEX-DEFAULT-BAKED-1): a scientific fit must withhold it, through a versioned roster, until its source is qualified (owner ruling e755519d, section 7.1).",
      [ve('"cadd_phred": 15.0,'), LIT["cadd"], LIT["pejaver"]]),
    F("sift_score", A, "SIFT score in [0,1] from dbNSFP; absent 0.5 (the code's neutral fill).", DBNSFP, "none",
      "none", "not_applicable", "current_only", "acmg_computational_evidence",
      "Homology-based and unsupervised. PP3/BP4 evidence, recorded as label-criterion use."
      " Strict in principle; its MATERIALISED values are not yet a qualified observation (DBNSFP-INDEX-DEFAULT-BAKED-1, PREDICTOR-TRANSCRIPT-COLLAPSE-1): a scientific fit must withhold it, through a versioned roster, until its source is qualified (owner ruling e755519d, section 7.1).",
      [ve('"sift_score": 0.5,'), LIT["pejaver"]]),
    F("polyphen2_score", CO, "PolyPhen-2 HDIV probability from dbNSFP; absent 0.5.", DBNSFP, "clinical", "none",
      "unknown", "current_only", "acmg_computational_evidence",
      "Trained on HumDiv: UniProtKB damaging alleles of Mendelian disease against divergence from close mammalian "
      "homologs -- clinically supervised, with unknown overlap with the ClinVar-labelled cohort (type-1 "
      "circularity, Grimm et al. 2015). Admissible only in an evaluation cohort verified disjoint from its training "
      "data (for example a time split after the tool's release), as a separately versioned claim.",
      [ve('"polyphen2_score": 0.5,'), LIT["polyphen"], LIT["grimm"]]),
    F("revel_score", CO, "REVEL ensemble score in [0,1] from dbNSFP; absent 0.5.", DBNSFP, "clinical", "none",
      "unknown", "current_only", "acmg_computational_evidence",
      "Trained on HGMD disease mutations against rare putatively neutral variants: clinically supervised, overlap "
      "with the ClinVar label set unknown and plausibly large (type-1 circularity). Conditional on a verified-"
      "disjoint evaluation cohort.", [ve('"revel_score": 0.5,'), LIT["revel"], LIT["grimm"], LIT["pejaver"]]),
    F("phylop_score", A, "phyloP conservation: the dbNSFP phyloP100way value where dbNSFP covers the variant, the "
      "PhyloP BigWig connector otherwise; the two producers are not reconciled (PHYLOP-RECONCILE-1). Absent 0.0.",
      DBNSFP + ["dataset:phylop_bigwig", f"code:{D}phylop.py::PhyloPConnector"], "none", "none", "not_applicable",
      "current_only", "acmg_computational_evidence",
      "Unsupervised evolutionary conservation; admissible. The unreconciled double producer is a representation "
      "defect already registered as PHYLOP-RECONCILE-1.", [ve('"phylop_score": 0.0,'), code(RDP, "Sources are NOT reconciled.")]),
    F("gerp_score", A, "GERP++ rejected-substitution score from dbNSFP; absent 0.0.", DBNSFP, "none", "none",
      "not_applicable", "current_only", "acmg_computational_evidence", "Unsupervised evolutionary constraint.",
      [ve('"gerp_score": 0.0,')]),
    F("alphamissense_score", A, "AlphaMissense pathogenicity in [0,1] joined by chrom:pos:ref:alt; absent 0.5.",
      ["dataset:alphamissense", f"code:{RDP}::DataPrepPipeline._join_alphamissense",
       f"code:{D}alphamissense.py::AlphaMissenseConnector"], "proxy_nonclinical", "none", "not_applicable",
      "current_only", "acmg_computational_evidence",
      "Fine-tuned on weak labels from population frequency (common human and primate variants as benign); its "
      "published class thresholds were set on ClinVar but no threshold is used here. Admissible; PP3/BP4-type "
      "evidence recorded as label-criterion use.", [ve('"alphamissense_score": 0.5,'), LIT["alphamissense"]]),
    F("splice_ai_score", A, "max(DS_AG, DS_AL, DS_DG, DS_DL) over overlapping genes; absent 0.0.",
      ["dataset:spliceai", f"code:{RDP}::DataPrepPipeline._join_spliceai", f"code:{D}spliceai.py::SpliceAIConnector"],
      "proxy_nonclinical", "none", "not_applicable", "current_only", "acmg_computational_evidence",
      "Trained on annotated GENCODE splice junctions, not on clinical labels. PP3/BP7 evidence for splicing "
      "(ClinGen 2023), recorded as label-criterion use.", [ve('"splice_ai_score": 0.0,'), LIT["spliceai"], LIT["walker"]]),
    F("eve_score", CO, "EVE score in [0,1]; the connector fills unmatched variants with 0.5 and counts coverage as "
      "score != 0.5, so a genuine 0.5 is counted uncovered. Run 15: connector unconfigured, every row the default.",
      ["dataset:eve", f"code:{D}eve.py::EVEConnector"], "none", "none", "not_applicable", "current_only", "none_known",
      "Pinned CONDITIONAL_ONLY (ruling c34eb717): admissible only within a frozen exact-variant-matched stratum "
      "(protein accession and isoform, reference residue, position, source release), coverage determined by join "
      "membership, never by value; scores are normalised within protein (EVE-COVERAGE-IDENTITY-1).",
      [ve('"eve_score": 0.5,'), code(D + "eve.py", 'n_covered = (result["eve_score"] != DEFAULT_SCORE).sum()'),
       LIT["eve"], "record:docs/census/measure_1d1_correction.json"]),
    F("cadd_high", A, "1 when cadd_phred >= 20 (top 1% of all possible substitutions by construction).", DBNSFP,
      "proxy_nonclinical", "none", "not_applicable", "current_only", "acmg_computational_evidence",
      "A fixed cut of an admitted score; the cut is rank-based, not fitted to clinical labels. Withheld with "
      "cadd_phred until its source is qualified (owner ruling e755519d, section 5 of the refined ruling).",
      [ve('feats["cadd_high"] = (feats["cadd_phred"] >= 20).astype(int)')], ["cadd_phred"]),
    F("sift_deleterious", A, "1 when sift_score < 0.05 (absent variants, filled 0.5, are 0).", DBNSFP, "none",
      "unknown", "not_applicable", "current_only", "acmg_computational_evidence",
      "A fixed cut of an admitted score. The provenance of the conventional 0.05 cut was not verified in this "
      "review (recorded as unknown); a single fixed cut-point can carry at most one fitted scalar, so it adds no "
      "label information beyond its admitted score. Withheld with sift_score until its source is qualified (owner "
      "ruling e755519d, section 7.1).",
      [ve('feats["sift_deleterious"] = (feats["sift_score"] < 0.05).astype(int)')], ["sift_score"]),
    F("polyphen_probably_damaging", CO, "1 when polyphen2_score >= 0.908 (the HDIV 'probably damaging' cut).",
      DBNSFP, "clinical", "clinical", "unknown", "current_only", "acmg_computational_evidence",
      "Inherits polyphen2_score's clinical supervision; its cut was itself set on the HumDiv disease set.",
      [ve('feats["polyphen_probably_damaging"] = (feats["polyphen2_score"] >= 0.908).astype('), LIT["polyphen"]],
      ["polyphen2_score"]),
    F("revel_pathogenic", CO, "1 when revel_score >= 0.5.", DBNSFP, "clinical", "unknown", "unknown", "current_only",
      "acmg_computational_evidence", "Inherits revel_score's clinical supervision and conditional admission.",
      [ve('feats["revel_pathogenic"] = (feats["revel_score"] >= 0.5).astype(int)')], ["revel_score"]),
    F("n_tools_pathogenic", CO, "Sum of cadd_high, sift_deleterious, polyphen_probably_damaging and revel_pathogenic.",
      DBNSFP, "mixed", "mixed", "unknown", "current_only", "acmg_computational_evidence",
      "Two of its four terms are clinically supervised; a sum cannot be more admissible than its least admissible "
      "term (monotone admission).", [ve('feats["n_tools_pathogenic"] = (')],
      ["cadd_high", "sift_deleterious", "polyphen_probably_damaging", "revel_pathogenic"]),
    F("gene_constraint_oe", A, "gnomAD lof.oe, the observed/expected loss-of-function ratio (NaN preserved when "
      "unmeasured).", CONSTRAINT, "none", "none", "not_applicable", "current_only", "none_known", CONSTRAINT_REASON,
      [ve('feats["gene_constraint_oe"] = pd.to_numeric(')]),
    F("gene_is_constrained", A, "1.0 when LOEUF < 0.35 (retained project policy; gnomAD v4.1.1 recommends 0.45), "
      "NaN when LOEUF is unmeasured.", CONSTRAINT, "none", "none", "not_applicable", "current_only", "none_known",
      CONSTRAINT_REASON, [ve("LEGACY_GENE_CONSTRAINT_LOEUF_THRESHOLD = 0.35")], ["loeuf"]),
    F("n_pathogenic_in_gene", R, "Count of label-1 rows in the variant's gene: corpus-wide by enrich_gene_counts, "
      "overwritten with train-only counts after the split and per fold inside the ensemble; unseen genes 0.",
      ["dataset:clinvar_cohort", f"code:{RDP}::enrich_gene_counts", f"code:{RDP}::DataPrepPipeline._gene_aware_split"],
      "clinical", "none", "verified_overlap", "not_applicable", "clinical_classification",
      "Pinned RETIRE_PREDICTIVE: a count of the target labels themselves. Within-gene AUROC exactly 0.5 and "
      "between-gene 0.7189 on Run 15 test (measured). enrich_gene_counts also returns early when the column "
      "already exists in the input, so a pre-enriched parquet's provenance is unbound.",
      [ve('feats["n_pathogenic_in_gene"] = ('), code(RDP, "already present in enriched parquet -- skip duplicate merge"),
       "record:docs/claims/README.md"]),
    F("gene_has_known_disease", R, "1 when n_pathogenic_in_gene > 0.", ["dataset:clinvar_cohort",
      f"code:{RDP}::enrich_gene_counts"], "clinical", "none", "verified_overlap", "not_applicable",
      "clinical_classification", "Pinned RETIRE_PREDICTIVE: derived from the label count.",
      [ve('feats["gene_has_known_disease"] = (feats["n_pathogenic_in_gene"] > 0).astype(int)')], ["n_pathogenic_in_gene"]),
    F("has_uniprot_annotation", X, "1 when the gene has any row in the UniProt variant table (gene-symbol join); "
      "never wired in the Run 17 launchers (constant 0 in Run 15).",
      ["dataset:uniprot_variants", f"code:{RDP}::DataPrepPipeline._join_uniprot"], "none", "none", "not_applicable",
      "current_only", "clinical_classification",
      "Pinned EXCLUDE_STRICT: annotation availability encodes ascertainment of disease-studied genes.",
      [ve('feats["has_uniprot_annotation"] = (')]),
    F("n_known_pathogenic_protein_variants", X, "Count of UniProt variant rows labelled 'pathogenic' in the gene.",
      ["dataset:uniprot_variants", f"code:{RDP}::DataPrepPipeline._join_uniprot"], "clinical", "none", "unknown",
      "current_only", "clinical_classification", "Pinned EXCLUDE_STRICT: a curated clinical pathogenic-variant count.",
      [ve('feats["n_known_pathogenic_protein_variants"] = ('), code(RDP, 'lambda x: (x == "pathogenic").sum(),')]),
    F("gtex_max_tpm", A, "Maximum across tissues of the gene's median TPM (bulk GTEx parquet).", GTEX, "none",
      "none", "not_applicable", "current_only", "none_known",
      "Measured tissue expression of the gene: non-clinical biology, transferable to unseen genes.",
      [ve('"gtex_max_tpm": 0.0,')]),
    F("gtex_n_tissues_expressed", A, "Number of tissues with median TPM >= 1.", GTEX, "none", "none",
      "not_applicable", "current_only", "none_known", "Measured tissue expression breadth.",
      [ve('"gtex_n_tissues_expressed": 0,')]),
    F("gtex_tissue_specificity", A, "1 - mean/max median TPM across tissues.", GTEX, "none", "none",
      "not_applicable", "current_only", "none_known", "Measured tissue specificity.",
      [ve('"gtex_tissue_specificity": 0.0,')]),
    F("gtex_is_eqtl", W, "1 when the variant has a GTEx single-tissue eQTL association (per-gene API path).",
      ["dataset:gtex", f"code:{D}gtex.py::build_gtex_feature_df"], "none", "none", "not_applicable", "current_only",
      "none_known", "eQTL membership is non-clinical biology, but the production bulk path writes the default for "
      "every row (GTEX-EQTL-PRODUCTION-DEFAULT-1) and the API path is not exercised by any launcher.",
      [ve('"gtex_is_eqtl": 0,'), code(RDP, "eQTL trio stays at defaults.")], state="unavailable"),
    F("gtex_min_eqtl_pval", W, "Despite its name: the maximum -log10(p) of the variant's eQTL associations (0 = none).",
      ["dataset:gtex", f"code:{D}gtex.py::build_gtex_feature_df"], "none", "none", "not_applicable", "current_only",
      "none_known", "As gtex_is_eqtl.", [ve('"gtex_min_eqtl_pval": 0.0,'),
                                          code(D + "gtex.py", "gtex_min_eqtl_pval        max -log10(p) across tissues")],
      state="unavailable"),
    F("gtex_max_abs_effect", W, "Maximum |beta| of the variant's eQTL associations.",
      ["dataset:gtex", f"code:{D}gtex.py::build_gtex_feature_df"], "none", "none", "not_applicable", "current_only",
      "none_known", "As gtex_is_eqtl.", [ve('"gtex_max_abs_effect": 0.0,')], state="unavailable"),
    F("codon_position", X, "Two incompatible quantities in one column: VEPConnector writes (aa_pos - 1) % 3 + 1 from "
      "the FIRST digit run of protein_change (an accession's digits can match), then the annotation stage overwrites "
      "it with protein_pos, an unbounded residue index, wherever protein coordinates exist; neither is the "
      "nucleotide position within a codon.",
      ["dataset:protein_coordinates", f"code:{D}vep.py::VEPConnector", f"code:{RDP}::DataPrepPipeline._annotate_scores"],
      "none", "none", "not_applicable", "not_applicable", "none_known",
      "Excluded: the column has no single meaning (CODON-POSITION-SEMANTICS-1). A corrected codon-phase or "
      "residue-position feature is a NEW identity with its own census entry.",
      [ve('feats["codon_position"] = ('), code(D + "vep.py", "return (aa_pos - 1) % 3 + 1  # 1-indexed codon position"),
       code(RDP, "pc = ProteinCoordConnector(alphamissense_file=_am_tsv)")]),
    F("dbsnp_af", A, "dbSNP build 156 allele frequency clipped at 0; absent 0.0.",
      ["dataset:dbsnp", f"code:{D}dbsnp.py::DbSNPConnector"], "none", "none", "not_applicable", "current_only",
      "acmg_population_data", "Measured allele frequency (population-data evidence).", [ve('feats["dbsnp_af"] = (')]),
    F("omim_n_diseases", X, "Number of OMIM phenotype entries linked to the gene.",
      ["dataset:omim", f"code:{D}omim.py::OMIMConnector"], "clinical", "none", "unknown", "current_only",
      "acmg_gene_disease_curation", "Pinned EXCLUDE_STRICT: curated gene-disease evidence.",
      [ve('feats["omim_n_diseases"] = (')]),
    F("omim_n_diseases_molecular", X, "Number of the gene's OMIM phenotypes with a known molecular basis (genemap2).",
      ["dataset:omim", f"code:{D}omim.py::OMIMConnector"], "clinical", "none", "unknown", "current_only",
      "acmg_gene_disease_curation", "Pinned EXCLUDE_STRICT: curated gene-disease evidence.",
      [ve('feats["omim_n_diseases_molecular"] = (')]),
    F("omim_is_autosomal_dominant", X, "1 when any of the gene's OMIM phenotypes is autosomal dominant.",
      ["dataset:omim", f"code:{D}omim.py::OMIMConnector"], "clinical", "none", "unknown", "current_only",
      "acmg_gene_disease_curation", "Pinned EXCLUDE_STRICT: gene-disease mode of inheritance.",
      [ve('feats["omim_is_autosomal_dominant"] = (')]),
    F("clingen_validity_score", X, "Maximum ClinGen gene-disease validity class over the gene's disease pairs "
      "(Definitive 5 ... No Known Disease Relationship 1; absent 0).",
      ["dataset:clingen", f"code:{D}clingen.py::ClinGenConnector"], "clinical", "none", "unknown", "current_only",
      "acmg_gene_disease_curation", "Pinned EXCLUDE_STRICT: curated gene-disease validity; the maximum discards "
      "which disease.", [ve('feats["clingen_validity_score"] = (')]),
    F("lovd_variant_class", R, "LOVD classification ordinal: 0 absent, 1 benign/likely benign, 2 VUS, 3 likely "
      "pathogenic, 4 pathogenic.", ["dataset:lovd", f"code:{D}lovd.py::LOVDConnector"], "clinical", "none",
      "unknown", "current_only", "clinical_classification",
      "Pinned RETIRE_PREDICTIVE: another resource's clinical classification of the same variant.",
      [ve('feats["lovd_variant_class"] = (')]),
    F("is_autosome", A, "1 for chromosomes 1-22.", CQ_SRC, "none", "none", "not_applicable", "not_applicable",
      "none_known", "Genomic location of the variant; no source beyond the variant itself.",
      [ve('feats["is_autosome"] = chrom.isin([str(i) for i in range(1, 23)]).astype(int)')]),
    F("is_sex_chrom", A, "1 for X or Y.", CQ_SRC, "none", "none", "not_applicable", "not_applicable", "none_known",
      "Genomic location of the variant.", [ve('feats["is_sex_chrom"] = chrom.isin(["X", "Y"]).astype(int)')]),
    F("is_mitochondrial", A, "1 for MT or M.", CQ_SRC, "none", "none", "not_applicable", "not_applicable",
      "none_known", "Genomic location of the variant.",
      [ve('feats["is_mitochondrial"] = chrom.isin(["MT", "M"]).astype(int)')]),
    F("gnn_score", CO, "Gene-level STRING-GNN probability injected by the run driver AFTER the ensemble was fitted on "
      "the default 0.5 (GNN-FEATURE-ORDER-1), unscaled into a standardised matrix in training and evaluation but "
      "scaled at inference (GNN-SCALE-SKEW-1); 0.5 also stands for 'gene not scored'.",
      ["dataset:string", "dataset:clinvar_cohort", "code:src/genomic_variant_classifier/models/gnn.py::GNNScorer",
       "code:src/genomic_variant_classifier/models/gnn.py::build_pyg_dataset",
       "code:src/genomic_variant_classifier/models/gnn.py::StringDBGraph"],
      "clinical", "none", "verified_overlap", "not_applicable", "clinical_classification",
      "Pinned CONDITIONAL_ONLY pending its move to a first-class ensemble member (1D-3). Trained on the training "
      "labels, with node attributes that include the retired label count and the excluded curated features "
      "(GNN-NODE-LABEL-COUNT-1) and edges selected by a mixed-origin STRING score (STRING-CHANNEL-ORIGIN-1).",
      [ve('feats["gnn_score"] = ('), code(DRV, 'X_split["gnn_score"] = ('), code(DRV, 'gnn_df["acmg_label"] = y_train.values')]),
    F("hetero_gnn_score", CO, "Gene-level heterogeneous knowledge-graph GNN probability (STRING interacts_with plus "
      "--kg-edges relations), injected after fit; never computed at inference (the default 0.5 is rendered).",
      ["dataset:string", "dataset:kg_gene_sets", "dataset:clinvar_cohort",
       "code:src/genomic_variant_classifier/models/hetero_gnn_scorer.py::HeteroGNNScorer",
       "code:src/genomic_variant_classifier/models/hetero_gnn_scorer.py::build_hetero_focal_graph",
       "code:src/genomic_variant_classifier/models/hetero_gnn_scorer.py::load_kg_edge_specs"],
      "clinical", "none", "verified_overlap", "not_applicable", "clinical_classification",
      "Pinned CONDITIONAL_ONLY. Its relation table accepts ClinGen and OMIM shares_disease edges and KEGG/Reactome "
      "shares_pathway edges (KG-DISEASE-RELATION-1); the Run 17 launchers pass Reactome only.",
      [ve('feats["hetero_gnn_score"] = ('), code(DRV, 'hetero_df["acmg_label"] = y_train.values'),
       code("scripts/launch_run17_baseline.sh", 'ARGS="$ARGS --hetero-gnn --kg-edges reactome:$REACTOME_GMT"')]),
    F("maxentscan_score", W, "MaxEntScan splice-site strength of the 9-mer donor or 23-mer acceptor context; "
      "default 0.0.", SPLICE, "proxy_nonclinical", "none", "not_applicable", "not_applicable", "acmg_computational_evidence",
      SPLICE_REASON, [ve('feats["maxentscan_score"] = ('),
                      code("src/genomic_variant_classifier/pipelines/rna_pipeline.py",
                           "splice_mask    = (is_splice.astype(int) == 1) | (consev.astype(int) >= 7)")], state="unavailable"),
    F("maxentscan_delta", W, "Change in MaxEntScan strength caused by the variant; default 0.0.", SPLICE,
      "proxy_nonclinical", "none", "not_applicable", "not_applicable", "acmg_computational_evidence", SPLICE_REASON,
      [ve('feats["maxentscan_delta"] = (')], state="unavailable"),
    F("dist_to_splice_site", W, "Distance in bp to the nearest canonical donor or acceptor; default 50.", SPLICE,
      "none", "none", "not_applicable", "not_applicable", "acmg_variant_type", SPLICE_REASON,
      [ve('feats["dist_to_splice_site"] = (')], state="unavailable"),
    F("exon_number", W, "Exon number parsed from the VEP exon annotation; 0 for non-exonic OR unknown.", SPLICE,
      "none", "none", "not_applicable", "not_applicable", "none_known", SPLICE_REASON,
      [ve('feats["exon_number"] = (')], state="unavailable"),
    F("is_canonical_splice", W, "1 inside the canonical GT-AG dinucleotides (+1/+2 donor, -2/-1 acceptor).", SPLICE,
      "none", "none", "not_applicable", "not_applicable", "acmg_variant_type", SPLICE_REASON,
      [ve('feats["is_canonical_splice"] = (')], state="unavailable"),
] + [
    F(f"af_1kg_{pop}", A, f"1000 Genomes Phase 3 {pop.upper()} super-population allele frequency; absent 0.0.",
      ["dataset:kg_1000g", f"code:{D}thousandgenomes.py::ThousandGenomesConnector"], "none", "none", "not_applicable",
      "verified_as_of", "acmg_population_data", "Measured frequency from a fixed 2013 release (population-data "
      "evidence); absence is not separable from a failed join.",
      [ve('for col in ("af_1kg_afr", "af_1kg_eur", "af_1kg_eas", "af_1kg_sas", "af_1kg_amr"):'),
       code(D + "thousandgenomes.py", "ALL.chr*.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz")])
    for pop in ("afr", "eur", "eas", "sas", "amr")
] + [
    F("finngen_af_fin", A, "FinnGen Finnish allele frequency; absent 0.0.", FG_SRC, "none", "none", "not_applicable",
      "current_only", "acmg_population_data", FG_REASON, [ve('("finngen_af_fin", 0.0),')]),
    F("finngen_af_nfsee", A, "FinnGen non-Finnish (Swedish-Estonian) allele frequency; absent 0.0.", FG_SRC, "none",
      "none", "not_applicable", "current_only", "acmg_population_data", FG_REASON, [ve('("finngen_af_nfsee", 0.0),')]),
    F("finngen_enrichment", A, "af_fin / (af_nfsee + 1e-9); absent 1.0.", FG_SRC, "none", "none", "not_applicable",
      "current_only", "acmg_population_data", FG_REASON, [ve('("finngen_enrichment", 1.0),')],
      ["finngen_af_fin", "finngen_af_nfsee"]),
    F("finngen_r13_af_fin", A, "FinnGen R13 Finnish allele frequency; absent 0.0.", FG_SRC, "none", "none",
      "not_applicable", "current_only", "acmg_population_data", FG_REASON, [ve('("finngen_r13_af_fin", 0.0),')]),
    F("finngen_r13_af_nfsee", A, "FinnGen R13 non-Finnish allele frequency; absent 0.0.", FG_SRC, "none", "none",
      "not_applicable", "current_only", "acmg_population_data", FG_REASON, [ve('("finngen_r13_af_nfsee", 0.0),')]),
    F("finngen_r13_enrichment", A, "R13 af_fin / (af_nfsee + 1e-9); absent 1.0.", FG_SRC, "none", "none",
      "not_applicable", "current_only", "acmg_population_data", FG_REASON, [ve('("finngen_r13_enrichment", 1.0),')],
      ["finngen_r13_af_fin", "finngen_r13_af_nfsee"]),
    F("esm2_delta_norm", A, "L2 distance between wild-type and mutant ESM-2 residue embeddings at the variant; 0.0 "
      "for non-missense or unavailable.", ["dataset:esm2", f"code:{D}esm2.py::ESM2Connector"], "none", "none",
      "not_applicable", "current_only", "none_known",
      "Self-supervised protein language model; no clinical labels. 0.0 conflates 'no change' with 'not computed'.",
      [ve('feats["esm2_delta_norm"] = (')]),
    F("esm2_llr", A, "ESM-2 masked-marginal log-likelihood ratio (mutant minus wild type; negative = damaging); "
      "0.0 unavailable.", ["dataset:esm2", f"code:{D}esm2.py::ESM2Connector"], "none", "none", "not_applicable",
      "current_only", "none_known", "Self-supervised; 0.0 is also the neutral value.", [ve('feats["esm2_llr"] = (')]),
    F("genomiclm_delta_norm", A, "L2 norm of the mean-pooled alt minus ref Nucleotide Transformer embedding over "
      "the 101 bp window; 0.0 unavailable.", ["dataset:nucleotide_transformer",
      f"code:{D}genomic_lm.py::NucleotideTransformerConnector"], "none", "none", "not_applicable", "current_only",
      "none_known", "Self-supervised DNA language model.", [ve('feats["genomiclm_delta_norm"] = ('), LIT["nt"]]),
    F("genomiclm_llr", A, "Masked-LM log-likelihood ratio at the variant-centre token; 0.0 unavailable.",
      ["dataset:nucleotide_transformer", f"code:{D}genomic_lm.py::NucleotideTransformerConnector"], "none", "none",
      "not_applicable", "current_only", "none_known", "Self-supervised DNA language model.",
      [ve('feats["genomiclm_llr"] = (  # SIGNED feature; NO clip'), LIT["nt"]]),
    F("cosmic_recurrence", X, "COSMIC CMC recurrence count; 0.0 for untested, absent or non-substitution.",
      ["dataset:cosmic_cmc", f"code:{D}connectors/connector_cosmic.py::CosmicCmcConnector"], "none", "none",
      "not_applicable", "current_only", "none_known",
      "Pinned EXCLUDE_STRICT: untested, unobserved and observed-zero are one value; a redesign keeps mutated and "
      "tested counts with an observation status under a new identity.",
      [ve('feats["cosmic_recurrence"] = ('), code(D + "connectors/connector_cosmic.py", "(0.0 when untested / absent / non-substitution).")]),
    F("cosmic_sig_tier", R, "COSMIC CMC MUTATION_SIGNIFICANCE_TIER ordinal (1 high, 2 medium, 3 low; Other and "
      "absent 0).", ["dataset:cosmic_cmc", f"code:{D}connectors/connector_cosmic.py::CosmicCmcConnector"], "clinical",
      "clinical", "unknown", "current_only", "clinical_classification",
      "Pinned RETIRE_PREDICTIVE: CMC Tier 1 draws on ClinVar pathogenic classification; the annotation is kept as "
      "interpretive evidence only.", [ve('feats["cosmic_sig_tier"] = (')]),
    F("kegg_pathway_count", X, "Number of distinct KEGG pathways containing the gene, disease maps (hsa05xxx) "
      "included.", ["dataset:kegg", f"code:{D}kegg.py::KEGGConnector"], "mixed", "none", "unknown", "current_only",
      "acmg_gene_disease_curation", "Pinned EXCLUDE_STRICT: counts disease maps; a disease-free count is a new "
      "identity requiring a verified pathway_is_disease classification.", [ve('feats["kegg_pathway_count"] = (')]),
    F("kegg_disease_pathway_flag", X, "1 when the gene is in any KEGG Human Diseases map (hsa05xxx).",
      ["dataset:kegg", f"code:{D}kegg.py::KEGGConnector"], "clinical", "none", "unknown", "current_only",
      "acmg_gene_disease_curation", "Pinned EXCLUDE_STRICT: curated disease-map membership.",
      [ve('feats["kegg_disease_pathway_flag"] = (')]),
    F("pli_score", A, "gnomAD v4.1 probability of loss-of-function intolerance; absent 0.0.", CONSTRAINT, "none",
      "none", "not_applicable", "current_only", "none_known", CONSTRAINT_REASON, [ve('feats["pli_score"] = (')]),
    F("loeuf", A, "gnomAD v4.1 LOEUF, the upper 90% bound of lof.oe, clipped to [0, 5]; absent 1.0 (a value).",
      CONSTRAINT, "none", "none", "not_applicable", "current_only", "none_known", CONSTRAINT_REASON,
      [ve('feats["loeuf"] = (')]),
    F("syn_z", A, "gnomAD synonymous Z-score; absent 0.0.", CONSTRAINT, "none", "none", "not_applicable",
      "current_only", "none_known", CONSTRAINT_REASON, [ve('feats["syn_z"] = (')]),
    F("mis_z", A, "gnomAD missense Z-score; absent 0.0.", CONSTRAINT, "none", "none", "not_applicable",
      "current_only", "unknown", CONSTRAINT_REASON + " Some laboratories cite missense constraint for PP2; whether "
      "the cohort's submitters did is unknown.", [ve('feats["mis_z"] = (')]),
    F("reactome_pathway_count", W, "Number of distinct Reactome pathways (all hierarchy levels) containing the "
      "gene; the Disease hierarchy is not excluded.", ["dataset:reactome", f"code:{D}reactome.py::ReactomeConnector"],
      "mixed", "none", "unknown", "current_only", "acmg_gene_disease_curation",
      "Pinned WITHHOLD_PENDING_EVIDENCE: Reactome's Disease hierarchy (isInDisease) is counted; the same source "
      "supplies the hetero-GNN shares_pathway relation.", [ve('feats["reactome_pathway_count"] = (')]),
    F("rnaseq_mean_log_tpm", W, "Mean log1p(TPM) of the gene across the RNA-seq matrix's samples.",
      ["dataset:rnaseq_matrix", f"code:{D}rnaseq.py::annotate_rnaseq_from_parquet"], "none", "none", "unknown",
      "unknown", "none_known", "The ruling admits the three summary features only after their source populations "
      "are reviewed; the matrix's sample population is not recorded in the repository.",
      [code(D + "rnaseq.py", "rnaseq_mean_log_tpm    float  mean log1p(TPM) across samples")], state="unknown"),
    F("rnaseq_detection_rate", W, "Fraction of samples expressing the gene.",
      ["dataset:rnaseq_matrix", f"code:{D}rnaseq.py::annotate_rnaseq_from_parquet"], "none", "none", "unknown",
      "unknown", "none_known", "As rnaseq_mean_log_tpm.",
      [code(D + "rnaseq.py", "rnaseq_detection_rate  float  fraction of samples expressing the gene")], state="unknown"),
    F("rnaseq_log2_cv", W, "log2(1 + coefficient of variation) of TPM across samples.",
      ["dataset:rnaseq_matrix", f"code:{D}rnaseq.py::annotate_rnaseq_from_parquet"], "none", "none", "unknown",
      "unknown", "none_known", "As rnaseq_mean_log_tpm.",
      [code(D + "rnaseq.py", "rnaseq_log2_cv         float  log2(1 + CV) of TPM across samples (dispersion)")],
      state="unknown"),
    F("rnaseq_log2fc", X, "log2 fold change, case against control (0 without a contrast).",
      ["dataset:rnaseq_matrix", f"code:{D}rnaseq.py::annotate_rnaseq_from_parquet"], "clinical", "none", "unknown",
      "unknown", "none_known", "Pinned EXCLUDE_STRICT: a disease contrast, not an intrinsic gene property.",
      [code(D + "rnaseq.py", "rnaseq_log2fc          float  log2 fold-change case/control (0 if no DE)")]),
    F("rnaseq_de_neglog10p", X, "-log10(p) of the case-control test (0 without a contrast).",
      ["dataset:rnaseq_matrix", f"code:{D}rnaseq.py::annotate_rnaseq_from_parquet"], "clinical", "none", "unknown",
      "unknown", "none_known", "Pinned EXCLUDE_STRICT: a disease contrast.",
      [code(D + "rnaseq.py", "rnaseq_de_neglog10p    float  -log10(p) case-vs-control test (0 if no DE)")]),
]

KG = "src/genomic_variant_classifier/data/kg_edges.py"
HG = "src/genomic_variant_classifier/models/hetero_gnn_scorer.py::load_kg_edge_specs"
GNN = "src/genomic_variant_classifier/models/gnn.py"


def S(name, kind, origin, disp, producer, meaning, sources, consumers, evidence, reason, sup, overlap):
    return {"name": name, "kind": kind, "origin": origin, "disposition": disp, "producer": producer,
            "meaning": meaning, "source_ids": list(sources), "consumers": list(consumers),
            "evidence_refs": list(evidence), "reason": reason, "upstream_supervision": sup, "upstream_overlap": overlap}


SOURCES = [
    S("kg_relation:clingen", "graph_relation", "clinical_disease", X, HG,
      "shares_disease edges between genes curated to the same ClinGen disease label.",
      ["dataset:kg_gene_sets", "dataset:clingen", f"code:{KG}::clingen_edges"], ["hetero_gnn_relations"],
      [code(KG, '"url": "https://search.clinicalgenome.org/kb/gene-validity/download",')],
      "Clinical gene-disease curation: the same information as the excluded clingen_validity_score, re-entering as "
      "topology (KG-DISEASE-RELATION-1). Not used by the Run 17 launchers; the code accepts it.", "clinical", "unknown"),
    S("kg_relation:go", "graph_relation", "unknown", W, HG,
      "shares_function edges between genes co-annotated to a GO term (MSigDB GMT).",
      ["dataset:kg_gene_sets", f"code:{KG}::go_edges"], ["hetero_gnn_relations"],
      [code(KG, '"url": "MSigDB c5.go.bp.*.symbols.gmt (or goa_human.gaf)",')],
      "GO is functional biology, but annotations include inferred-from-mutant-phenotype evidence from disease "
      "studies and a GMT carries no evidence codes to filter them; withheld until an evidence-code-filtered source "
      "is used.", "unknown", "unknown"),
    S("kg_relation:kegg", "graph_relation", "mixed", X, HG,
      "shares_pathway edges between genes in the same KEGG gene set.",
      ["dataset:kg_gene_sets", "dataset:kegg", f"code:{KG}::kegg_edges"], ["hetero_gnn_relations"],
      [code(KG, '"url": "MSigDB c2.cp.kegg_medicus.*.symbols.gmt (registration)",'),
       code(D + "kegg.py", "hsa05xxx: cancers, neurodegenerative, immune,")],
      "The same source class as the excluded KEGG features, disease maps included.", "mixed", "unknown"),
    S("kg_relation:omim", "graph_relation", "clinical_disease", X, HG,
      "shares_disease edges between genes linked to the same OMIM phenotype.",
      ["dataset:kg_gene_sets", "dataset:omim", f"code:{KG}::omim_edges"], ["hetero_gnn_relations"],
      [code(KG, '"url": "OMIM genemap2.txt (license required)",')],
      "Clinical gene-disease curation re-entering as topology (KG-DISEASE-RELATION-1).", "clinical", "unknown"),
    S("kg_relation:reactome", "graph_relation", "mixed", W, HG,
      "shares_pathway edges between genes in the same ReactomePathways.gmt pathway (Disease hierarchy included).",
      ["dataset:kg_gene_sets", "dataset:reactome", f"code:{KG}::reactome_edges"], ["hetero_gnn_relations"],
      [code(KG, '"url": "https://reactome.org/download/current/ReactomePathways.gmt",'),
       code("scripts/launch_run17_baseline.sh", 'ARGS="$ARGS --hetero-gnn --kg-edges reactome:$REACTOME_GMT"')],
      "Withheld with reactome_pathway_count (ruling c34eb717): the Disease hierarchy is not excluded. This is the "
      "relation the Run 17 launchers pass.", "mixed", "unknown"),
    S("string_edge_attribute:coexpression", "edge_attribute", "independent_biology", CO,
      f"{GNN}::build_pyg_dataset", "STRING co-expression channel score / 1000 as a GNN edge attribute.",
      ["dataset:string", f"code:{GNN}::StringDBGraph"], ["gnn_edge_attributes"], [LIT["string"], code(GNN, "_CHANNELS = [")],
      "Co-expression is measured biology, but it exists only on edges chosen by the mixed-origin combined score, so "
      "it is conditional on a channel-restricted edge selection (1D-3).", "none", "not_applicable"),
    S("string_edge_attribute:database", "edge_attribute", "mixed", X, f"{GNN}::build_pyg_dataset",
      "STRING curated-database channel score / 1000 as a GNN edge attribute.",
      ["dataset:string", f"code:{GNN}::StringDBGraph"], ["gnn_edge_attributes"], [LIT["string"], code(GNN, "_CHANNELS = [")],
      "STRING: 'associations derived from curated pathway and complex databases', including KEGG and Reactome -- the "
      "excluded and withheld pathway sources, disease maps included (STRING-CHANNEL-ORIGIN-1).", "mixed", "unknown"),
    S("string_edge_attribute:experimental", "edge_attribute", "independent_biology", CO,
      f"{GNN}::build_pyg_dataset", "STRING experimental channel score / 1000 as a GNN edge attribute.",
      ["dataset:string", f"code:{GNN}::StringDBGraph"], ["gnn_edge_attributes"], [LIT["string"], code(GNN, "_CHANNELS = [")],
      "Measured interaction evidence; conditional on a channel-restricted edge selection, as coexpression.",
      "none", "not_applicable"),
    S("string_relation:combined_score_threshold", "graph_relation", "mixed", X, f"{GNN}::StringDBGraph",
      "Gene pairs whose STRING combined score >= the threshold (700 by default): the GNN edges and the hetero-GNN "
      "interacts_with relation.", ["dataset:string", f"code:{GNN}::StringDBGraph"],
      ["gnn_edges", "hetero_gnn_relations"], [LIT["string"], code(GNN, 'links_df = links_df[links_df["combined_score"] >= self.threshold]')],
      "The combined score integrates the curated-database channel (KEGG, Reactome) and literature text mining with "
      "measured channels, so edge existence is of mixed origin. A channel-restricted combined score is a new identity "
      "(STRING-CHANNEL-ORIGIN-1).", "mixed", "unknown"),
    S("supervision:gnn_label", "supervision_label", "clinical_disease", CO, f"{GNN}::train_gnn_pipeline",
      "The STRING-GNN's training target: the ClinVar-derived label of the training rows.",
      ["dataset:clinvar_cohort", f"code:{GNN}::train_gnn_pipeline"], ["gnn_supervision"],
      [code(DRV, 'gnn_df["acmg_label"] = y_train.values')],
      "A supervision label is legitimate for a first-class member trained out-of-fold (1D-3); as an input feature's "
      "producer it is the clinical label itself.", "clinical", "verified_overlap"),
    S("supervision:hetero_gnn_label", "supervision_label", "clinical_disease", CO,
      "src/genomic_variant_classifier/models/hetero_gnn_scorer.py::HeteroGNNTrainer",
      "The hetero-GNN's training target: the ClinVar-derived label of the training rows.",
      ["dataset:clinvar_cohort", "code:src/genomic_variant_classifier/models/hetero_gnn_scorer.py::HeteroGNNTrainer"],
      ["hetero_gnn_supervision"], [code(DRV, 'hetero_df["acmg_label"] = y_train.values')],
      "As supervision:gnn_label.", "clinical", "verified_overlap"),
]

SITES = {
    "scripts/ablate_gnn.py": (["experiment"], "GNN timing and ablation harness; its scores are not model inputs"),
    "scripts/build_schema_baseline.py": (["monitoring"], "builds the drift-monitoring schema baseline"),
    "scripts/diff_engineer_features.py": (["diagnostic"], "compares feature builders"),
    "scripts/measure_1d1b2_pilots.py": (["diagnostic"], "stage 1D-1b.2 pilot measurement: the current builder's flags on "
                                        "the legacy joined values, for comparison; no model is fitted"),
    "scripts/measure_1d1b3_graph_identity.py": (["diagnostic"], "stage 1D-1b.3 measurement: builds the legacy "
                                                "StringDBGraph from the same files to compare it with the source-bound "
                                                "graphs and to show its cache key; no model is fitted"),
    "scripts/measure_qualified_preparation_1d1c1c.py": (["diagnostic"], "set \"av\" measurement: the qualified "
                                                        "preparation on the real files, compared with one builder call "
                                                        "on the first rows; fits nothing"),
    "scripts/preflight_run16_inputs.py": (["diagnostic"], "pre-flight input check"),
    "scripts/prove_engineer_features_equivalence.py": (["diagnostic"], "historical equivalence proof"),
    "scripts/run_benchmark.py": (["surface:evaluation"], "benchmarks learners on the contract's columns"),
    "scripts/run_drift_monitor.py": (["monitoring"], "drift monitoring of engineered features"),
    "scripts/run_phase2_eval.py": (["surface:gnn_node_attributes", "surface:hetero_gnn_node_attributes",
                                    "surface:split_persistence", "surface:evaluation"],
                                   "the run driver: GNN and hetero-GNN node attributes, split re-persistence, evaluation"),
    "src/genomic_variant_classifier/agent_layer/harness/correctness_harness.py": (["diagnostic"], "correctness harness"),
    "src/genomic_variant_classifier/api/pipeline.py": (["surface:inference"], "inference rendering of the contract"),
    "src/genomic_variant_classifier/data/real_data_prep.py": (["surface:tabular_fit", "surface:scaler_fit",
                                                               "surface:split_persistence"],
                                                              "builds, guards, scales and writes the training matrix"),
    "src/genomic_variant_classifier/evaluation/benchmark.py": (["surface:evaluation"], "benchmark runner"),
    "src/genomic_variant_classifier/information_sources/absence_census.py": (["diagnostic"], "the stage 1D-1b.2 "
                                                                             "source-absence census: executes the builder "
                                                                             "on absent, NaN and varied frames"),
    "src/genomic_variant_classifier/information_sources/consumers.py": (["diagnostic"], "this census's measurement"),
    "src/genomic_variant_classifier/information_sources/qualified_preparation.py": (
        ["surface:feature_engineering"], "set \"av\": the qualified preparation builds the canonical epoch-2 matrix "
        "through the single builder (engineer_features); it fits nothing, and training from it is refused"),
    "src/genomic_variant_classifier/models/feature_semantics.py": (["definition"], "defines the feature-semantics "
                                                                   "epochs from the contract (epoch 2 is epoch 1 with the "
                                                                   "registered rename); fits and serves nothing itself"),
    "src/genomic_variant_classifier/models/gnn.py": (["definition"], "defines the GNN graph and dataset builders"),
    "src/genomic_variant_classifier/models/hetero_gnn_scorer.py": (["definition"], "defines the hetero-GNN builders"),
    "src/genomic_variant_classifier/models/variant_ensemble.py": (["definition", "surface:feature_engineering",
                                                                   "surface:evaluation"],
                                                                  "defines the contract and the builder; prediction "
                                                                  "consumes the columns recorded at fit"),
    "src/genomic_variant_classifier/training/continual_trainer.py": (["surface:continual_training"],
                                                                     "continual refits on engineer_features output"),
}

DEFECTS = [
    ("EVE-COVERAGE-IDENTITY-1", "EVE coverage is inferred from the value: the connector fills unmatched variants with "
     "0.5 and counts coverage as score != 0.5, so a genuine 0.5 is uncovered; the 1D-1 audit's measurement compared "
     "standardised matrices with raw defaults (withdrawn, docs/census/measure_1d1_correction.json).",
     ["eve_score"], [code(D + "eve.py", 'n_covered = (result["eve_score"] != DEFAULT_SCORE).sum()'),
                     "record:docs/census/measure_1d1_correction.json"]),
    ("MISSINGNESS-SEMANTICS-1", "(Registered by the owner, decision file 25691d66.) engineer_features fills an absent source with a value that is also a valid "
     "observation (0.5 for SIFT, PolyPhen-2, REVEL, AlphaMissense, EVE and both GNN scores; 15.0 for CADD; 0.0 for "
     "conservation, splicing, language-model, frequency and count features; 1.0 for LOEUF and FinnGen enrichment), "
     "so no model can tell 'not observed' from 'observed at the default'. The observation contract (value, status, "
     "source identity) is stage 1D-1b.",
     ["af_raw", "af_is_absent", "sift_score", "polyphen2_score", "revel_score", "alphamissense_score", "eve_score",
      "cadd_phred", "phylop_score", "gerp_score", "splice_ai_score", "esm2_delta_norm", "esm2_llr",
      "genomiclm_delta_norm", "genomiclm_llr", "loeuf", "pli_score", "syn_z", "mis_z", "dbsnp_af", "gnn_score",
      "hetero_gnn_score", "finngen_enrichment", "finngen_r13_enrichment"],
     [ve("All missing columns are filled with safe defaults (df.get)")]),
    ("GNN-SCALE-SKEW-1", "The run driver injects RAW gnn_score and hetero_gnn_score into matrices whose other "
     "columns are StandardScaler outputs, after the scaler was fitted on the constant default; inference instead "
     "adds gnn_score BEFORE scaling (a scaler fitted on a constant 0.5 has scale 1 and mean 0.5, so inference sees "
     "g - 0.5 where evaluation saw g) and never computes hetero_gnn_score. Measured on Run 15: persisted gnn_score "
     "spans [0.0012, 0.5] while cadd_phred has mean 0 and standard deviation 1.",
     ["gnn_score", "hetero_gnn_score"], [code(DRV, 'X_split["gnn_score"] = ('),
                                         code("src/genomic_variant_classifier/api/pipeline.py", 'enriched["gnn_score"] = gene_symbols.map('),
                                         "record:docs/census/measure_1d1_correction.json"]),
    ("GNN-NODE-LABEL-COUNT-1", "GNN node attributes are every training column but gnn_score -- including the "
     "retired label count n_pathogenic_in_gene, lovd_variant_class and cosmic_sig_tier and the excluded curated "
     "gene-disease features -- and the GNN is trained on the same training labels, so gnn_score inherits every "
     "retired and excluded dependency (measured_dependencies).",
     ["gnn_score", "hetero_gnn_score"], [code(DRV, 'c for c in X_train.columns if c != "gnn_score"')]),
    ("KG-DISEASE-RELATION-1", "The hetero-GNN relation table accepts ClinGen and OMIM shares_disease and KEGG and "
     "Reactome shares_pathway sources: curated disease information excluded from the tabular census can re-enter "
     "as graph topology.", ["kg_relation:clingen", "kg_relation:omim", "kg_relation:kegg", "kg_relation:reactome",
                            "hetero_gnn_score"],
     [code("src/genomic_variant_classifier/models/hetero_gnn_scorer.py", '"clingen": (_kg.clingen_edges, "shares_disease"),')]),
    ("STRING-CHANNEL-ORIGIN-1", "GNN edges are selected on STRING's combined score, which integrates the "
     "curated-database channel (KEGG, Reactome, BioCyc) and literature text mining, and the database channel is "
     "itself an edge attribute.", ["string_relation:combined_score_threshold", "string_edge_attribute:database",
                                   "gnn_score", "hetero_gnn_score"],
     [LIT["string"], code(GNN, 'links_df = links_df[links_df["combined_score"] >= self.threshold]')]),
    ("CODON-POSITION-SEMANTICS-1", "codon_position holds (aa_pos - 1) % 3 + 1 from the first digit run of "
     "protein_change, overwritten by the unbounded residue index protein_pos wherever protein coordinates exist.",
     ["codon_position"], [code(D + "vep.py", "match = re.search(r'(\\d+)', protein_change)")]),
    ("RNA-SPLICE-GATE-ORDER-1", "The splice-context pipeline runs during annotation but gates on is_splice and "
     "consequence_severity, which engineer_features creates afterwards; every row receives the defaults (Run 15's "
     "annotated rows carry neither column; four of its five outputs were constant).",
     ["maxentscan_score", "maxentscan_delta", "dist_to_splice_site", "exon_number", "is_canonical_splice"],
     [code("src/genomic_variant_classifier/pipelines/rna_pipeline.py",
           'is_splice      = result.get("is_splice", pd.Series(0, index=result.index))')]),
    ("ABSENCE-AS-EVIDENCE-1", "An absent source becomes positive evidence (stage 1D-1b.2, measured by execution). "
     "engineer_features fills an absent REVEL with 0.5 and flags revel_score >= 0.5, so revel_pathogenic is 1 and "
     "n_tools_pathogenic counts it; fills an absent allele frequency with 0.0, which sets af_is_absent and "
     "af_is_ultra_rare (PM2-like rarity from missing data); fills an absent ref or alt with 'A', which sets is_snv "
     "(docs/census/SOURCE_ABSENCE_1D1b2.json). On the Run 15 cohort (1,490,014 rows, docs/census/PILOTS_1D1b2.json): "
     "1,169,388 rows (78.5%) have no REVEL score and are all flagged pathogenic, so 90.1% of benign rows carry "
     "revel_pathogenic = 1; 497,970 rows (33.4%) have no gnomAD frequency and are all 'ultra-rare', and the missing "
     "frequency alone separates the labels (67.0% of pathogenic against 27.9% of benign rows; AUROC 0.695). No cohort "
     "row lacks ref or alt, so is_snv is latent there. Fix: stage 1D-1c (observed-only derivation, columns.DERIVED_RULES). "
     "CORRECTED at stage 1D-1c.2A and made precise by owner ruling ace92e09 (docs/census/PREDICTOR_SOURCES_1D1c2.json, "
     "absence_as_evidence_correction): 1,169,388 is a LOWER bound -- the join's 'observed' rows include 129,779 that hold "
     "exactly 0.5, the dbNSFP index's own fill (DBNSFP-INDEX-DEFAULT-BAKED-1) -- so 1,299,167 rows carry "
     "revel_pathogenic = 1 from no index score or an index 0.5, and 1,299,101 of them from no index score or an index 0.5 "
     "without a publisher 0.5 at the allele (the publisher has a 0.5 at 66; whether it is the transcript the index "
     "described is unverified). That is NOT the number without a publisher score: 1,242,065 have no publisher score in the "
     "completely scanned GRCh38 view and 57,036 have one (42,861 without an index value, 14,175 with another score than "
     "the index's 0.5). Of the 1,242,065, 1,238,371 are outside REVEL's domain by the cohort's own fields (74,708 not "
     "single-nucleotide, 1,163,663 single-nucleotide with a non-missense ClinVar consequence) -- a value fabricated where "
     "REVEL does not apply, not a missing one -- and 3,694 are missense; 352 (326 missense) have a scored publisher row at "
     "the variant's ClinVar GRCh37 allele whose GRCh38 position the publisher left unresolved.",
     ["af_is_absent", "af_is_ultra_rare", "is_snv", "revel_pathogenic", "n_tools_pathogenic"],
     [ve('feats["revel_pathogenic"] = (feats["revel_score"] >= 0.5).astype(int)'),
      ve('feats["af_is_ultra_rare"] = (af < 0.0001).astype(int)'),
      ve('ref = df.get("ref", pd.Series(["A"] * len(df), index=df.index)).fillna("A")'),
      "record:docs/census/SOURCE_ABSENCE_1D1b2.json", "record:docs/census/PILOTS_1D1b2.json", "record:docs/census/PREDICTOR_SOURCES_1D1c2.json"])
    ,
    ("GNOMAD-CHRX-LOCUS-PARSE-1", "DataPrepPipeline._join_gnomad's locus parse drops a first field that is not digits "
     "once X, Y and M are removed, as a prefix: 'X' becomes '' and ''.isdigit() is False, so every chromosome-X gnomAD "
     "record (91,771 of 2,951,148 in gnomad_v4_exomes.parquet) is discarded and all 63,929 chromosome-X cohort rows get "
     "no frequency -- then 0.0 and 'ultra-rare' (ABSENCE-AS-EVIDENCE-1). Autosomes join: 990,350 of 1,424,307 observed, "
     "Run 15's logged count exactly (docs/census/PILOTS_1D1b2.json). Fix: stage 1D-1c.",
     ["af_raw", "af_log10", "af_is_absent", "af_is_ultra_rare", "af_is_rare", "af_is_common"],
     [code(RDP, '.replace("X", "")'), "record:docs/census/PILOTS_1D1b2.json"]),
    ("EVE-DUPLICATE-KEY-MULTIPLIES-ROWS-1", "41 of the 3,211 EVE files do not resolve entry name -> HGNC and are keyed "
     "by their filename prefix (BRCA1_BRCT_HUMAN and BRCA1_RING_HUMAN both become BRCA1); 10,982 (gene, substitution) "
     "keys are then scored more than once, always with different scores, and EVEConnector._annotate's left join "
     "multiplies every matching cohort row: with EVE configured, 280 Run 15 cohort rows would each appear twice "
     "(1,490,294 rows for 1,490,014). Latent: Run 15 ran EVE unconfigured. The shadow materialiser records these rows "
     "as bad_mapping (ambiguous_source_record) instead (docs/census/PILOTS_1D1b2.json).", ["eve_score"],
     [code(D + "eve.py", 'return normalize_gene_symbol(str(stem).split("_")[0]), False'),
      code(D + "eve.py", 'on=["_gene_symbol", "_aa_change"],'), "record:docs/census/PILOTS_1D1b2.json"]),
    ("GTEX-EQTL-PRODUCTION-DEFAULT-1", "The production GTEx path (bulk parquet) writes the defaults for the three "
     "variant-level eQTL features; only the per-gene API path, which no launcher uses, computes them.",
     ["gtex_is_eqtl", "gtex_min_eqtl_pval", "gtex_max_abs_effect"], [code(RDP, "eQTL trio stays at defaults.")]),
    ("STRING-PAIR-ACCOUNTING-1", "StringDBGraph.build adds one nx.Graph edge per DIRECTED record, so the reverse "
     "record of a pair silently overwrites the forward one's attributes, and no check finds a missing reciprocal, a "
     "contradictory direction or a duplicate; the 1D-1b.1 order-independent digest accepted a pair duplicated in both "
     "directions. Stage 1D-1b.3 reads the file in ONE verified pass and matches the two directions of every pair "
     "EXACTLY in disk-backed partitions, refusing every deviation (graph_identity.read_string_pairs). Measured on STRING "
     "v12 human (sha256 28a2ea3b): 13,715,404 records = 6,857,702 pairs x 2, every pair symmetric, 19,622 proteins; "
     "content identical for 16, 64 and 256 partitions; 61.0-64.1 s, 621-819 MiB peak (re-measured with the stage 1D-1b.4 read-only buffers), 342,885,100 bytes of scratch on a "
     "2-core container (docs/census/STRING_READER_BENCHMARK_1D1b3.json).",
     ["string_relation:combined_score_threshold", "string_edge_attribute:experimental", "string_edge_attribute:database",
      "string_edge_attribute:coexpression", "gnn_score", "hetero_gnn_score"],
     [code(GNN, "G.add_edge("), "record:docs/census/GRAPH_IDENTITY_1D1b3.json",
      "record:docs/census/STRING_READER_BENCHMARK_1D1b3.json"]),
    ("STRING-THRESHOLD-RECONSTRUCTION-1", "A channel-restricted STRING graph rebuilt from the exported integers is not "
     "exactly defined by them: rounded channels with a floored combined score is the best-supported storage model (fits "
     "6,857,369 of 6,857,702 pairs; 333 miss by one thousandth), the restricted reconstruction has 3 pairs outside the "
     "published graph, and near the threshold the selection depends on rounding. Stage 1D-1b.3 defines G1 = "
     "reconstruction >= 0.700 INTERSECTED with the published graph, compares in integer thousandths, decides every "
     "comparison within floating-point error of the threshold in exact rational arithmetic (52 pairs reconstruct to "
     "exactly 0.700), and records the four-way status (owner ruling 7f4b69d0, section 7): selected robust 84,874, "
     "selected sensitive 113, excluded sensitive 64 (published passes) + 9 (fails), excluded robust 151,879 + "
     "6,620,763; G1 84,987 pairs, G2 236,930 (graph_identity.classify_threshold).",
     ["string_relation:combined_score_threshold", "gnn_score", "hetero_gnn_score"],
     [code(GNN, 'links_df = links_df[links_df["combined_score"] >= self.threshold]'),
      "record:docs/census/GRAPH_IDENTITY_1D1b3.json"]),
    ("STRING-GENE-MAPPING-1", "StringDBGraph maps a protein to its STRING preferred name, keeps an unmapped protein "
     "under its protein identifier and merges proteins that share a name; its graph cache is keyed by the threshold "
     "alone (string_graph_{threshold}.pkl), so it returned the cached 236,930-edge graph when given a different links "
     "file, and its names and links caches carry no identity at all. Preferred names are not canonical identity "
     "(ruling 7f4b69d0, section 9.1): against HGNC (sha256 69bb5722) 18,837 proteins resolve by approved symbol and 479 "
     "by a unique previous symbol, 1 ambiguously, 382 not at all, and 4 HGNC identifiers are claimed by two proteins "
     "AFTER canonicalisation (resolved by approved-symbol precedence, graph_identity.map_string_proteins); 250 Run 15 "
     "cohort symbols (8,324 rows) reach a STRING node only through their HGNC identifier. Stage 1D-1b.3 binds a graph to a policy "
     "and to its computed materialisation (source, mapping, node universe, relation policy, edge content); stage 1D-1b.4 "
     "accepts it only when it is exactly the graph re-derived from the pinned files (graph_identity.accept_graph).",
     ["string_relation:combined_score_threshold", "string_edge_attribute:experimental", "string_edge_attribute:database",
      "string_edge_attribute:coexpression", "gnn_score", "hetero_gnn_score"],
     [code(GNN, 'p1 = self._protein_to_gene.get(row["protein1"], row["protein1"])'),
      code(GNN, 'cache_path = self.cache_dir / f"string_graph_{self.threshold}.pkl"'),
      "record:docs/census/GRAPH_IDENTITY_1D1b3.json"]),
    ("KG-COHORT-DEPENDENT-PROJECTION-1", "kg_edges.co_membership_edges removes out-of-cohort genes BEFORE applying "
     "max_set_size, so a pathway's edges exist or vanish with the cohort's composition. Measured on ReactomePathways.gmt "
     "(sha256 cf17607f) and the Run 15 splits: the training-gene universe gives 732,550 edges; adding the validation "
     "and test genes, as the run driver does, DELETES 180,843 train-train edges and adds none. Stage 1D-1b.3 caps and "
     "weights by the GLOBAL canonical set size, w = 1/(|S_s| - 1), so growing the universe never removes or reweights "
     "an edge (graph_identity.project_pathways; induced-subgraph consistency measured on the same data: 0 removed, 0 "
     "added, 0 reweighted).",
     ["kg_relation:reactome", "kg_relation:kegg", "kg_relation:go", "kg_relation:clingen", "kg_relation:omim",
      "hetero_gnn_score"],
     [code(D + "kg_edges.py", "gs = list(dict.fromkeys(g for g in genes if (allow is None or g in allow)))"),
      "record:docs/census/GRAPH_IDENTITY_1D1b3.json"]),
    ("KG-RELATION-SOURCE-MERGE-1", "load_kg_edge_specs merges every source mapped to one relation into one edge set "
     "(Reactome and KEGG into shares_pathway; ClinGen and OMIM into shares_disease), so a withheld and an excluded "
     "source cannot be told apart per edge. Stage 1D-1b.3 keys relations by (source, relation) and pathways by "
     "(source, stable pathway id) (graph_identity.GeneSetSource).",
     ["kg_relation:reactome", "kg_relation:kegg", "kg_relation:clingen", "kg_relation:omim", "hetero_gnn_score"],
     [code("src/genomic_variant_classifier/models/hetero_gnn_scorer.py",
           "by_rel.setdefault(relation, set()).update(edges)")]),
    ("GRAPH-NODE-UNIVERSE-SPLIT-DEPENDENT-1", "The run driver builds the hetero-GNN node universe from the union of the "
     "training, validation and test genes read from the persisted splits, and restricts every relation to it, so the "
     "test split's composition shapes the training graph's nodes and, through KG-COHORT-DEPENDENT-PROJECTION-1, "
     "deletes 180,843 Reactome train-train edges (no labels move, but split identity reaches training). Stage 1D-1b.3's "
     "graphs use the source-defined universe (GraphPolicy.node_universe = global_source); a cohort is an induced view.",
     ["hetero_gnn_score", "kg_relation:reactome", "kg_relation:kegg", "kg_relation:go", "kg_relation:clingen",
      "kg_relation:omim", "string_relation:combined_score_threshold"],
     [code(DRV, "cohort_genes = sorted(_train_genes | _val_genes | _test_genes)"),
      "record:docs/census/GRAPH_IDENTITY_1D1b3.json"]),
    ("GENE-SYMBOL-SPLIT-IDENTITY-1", "The gene-disjoint split groups rows by gene SYMBOL, so two symbols of one gene can "
     "fall on both sides. Measured on the Run 15 cohort and splits: 26 HGNC identifiers appear under two symbols and 14 "
     "of them straddle training and held-out (e.g. HGNC:4166 as GAS8 in training and DRC4 in test; HGNC:1152 as BVES and "
     "POPDC1); the 1D-1b.3 one-gene rule refuses the 206 training rows of those 14 genes, and the stage 1D-1b.4 "
     "component authority excludes all 206 among the 128,435 training rows whose component holds a held-out row "
     "(docs/census/POPULATION_ISOLATION_1D1b4.json). Owner: stage 1D-1d (canonical gene and split identities).",
     ["supervision:gnn_label", "supervision:hetero_gnn_label", "gnn_score", "hetero_gnn_score", "n_pathogenic_in_gene"],
     [code(RDP, "trainval_idx, test_idx = next(splitter.split(X, y, groups=groups))"),
      "record:docs/census/GRAPH_IDENTITY_1D1b3.json", "record:docs/census/POPULATION_ISOLATION_1D1b4.json"]),
    # ---- owner ruling 68ca322f (2026-11-10), registered by set "ar" revision B. Its four reproduced defects
    # (LABEL-TYPE-IDENTITY-1, LABEL-DENIED-READ-AUDIT-1, GRAPH-MAPPING-RECEIPT-AUTHORITY-1, GRAPH-MUTABLE-MATERIALIZATION-1)
    # were repaired by set "as" (stage 1D-1b.4) and removed here with their witnesses.
    ("VARIANT-MULTIGENE-ISOLATION-1", "One gene identity per row cannot represent a variant listed under several genes. "
     "Measured on the Run 15 cohort: 729 gene fields are multi-gene lists (1,838 rows, e.g. 'ABCC8;KCNJ11'); read as "
     "canonical gene SETS, 419 HGNC genes lie on both sides of the split, 113,568 training rows touch a held-out gene and "
     "82,999 held-out rows touch a training gene; union-find over genes linked by such lists gives 16,599 components "
     "(largest 11 genes, 19,985 rows: no giant component), of which 309 straddle the split -- 660 genes, 219,128 rows. "
     "On the three split manifests, with variant identity included (docs/census/POPULATION_ISOLATION_1D1b4.json): "
     "16,964 components (16,599 gene groups and 365 rows with no resolved gene), 327 spanning partitions; shared genes "
     "and components training-validation 145 and 115, training-test 291 and 229, validation-test 54 and 53; no "
     "variant identifier and no (chrom, pos, ref, alt) repeats. Stage 1D-1b.4 represents a row by its gene SET "
     "(population.PopulationRow) and authorizes a fit by component (population.authorize_fit); the production split is "
     "unchanged. Owner: 1D-1d (split groups built from components; which loci bind a group is the owner's decision).",
     ["supervision:gnn_label", "supervision:hetero_gnn_label", "gnn_score", "hetero_gnn_score", "n_pathogenic_in_gene"],
     [code(RDP, "trainval_idx, test_idx = next(splitter.split(X, y, groups=groups))"),
      "record:docs/census/GRAPH_IDENTITY_1D1b3.json", "record:docs/census/POPULATION_ISOLATION_1D1b4.json"]),
    ("LABEL-ACCESS-DISCOVERY-SCOPE-1", "labels.discover_label_sites finds only the graph builders' label column and "
     "label_col/label_column parameters (12 reviewed sites); it does not discover labels passed as y arrays, positional "
     "arguments or other names, so it is a regression detector, never a complete label-access audit. Stage 1D-1b.4 adds "
     "the claims and evidence interfaces (claims.ClaimsAuthority) and per-fit read receipts (labels.TrainingLabels), "
     "in shadow: no legacy consumer uses them. Owner: 1D-3, where the legacy consumers are replaced by fold-scoped "
     "capabilities.",
     ["supervision:gnn_label", "supervision:hetero_gnn_label"],
     [code("src/genomic_variant_classifier/information_sources/labels.py", 'LABEL_PARAMETERS = frozenset({"label_col", "label_column"})'),
      "record:docs/census/LABEL_SITES_1D1b3.json"]),
    ("STRING-PROTEIN-GENE-REDUCTION-1", "The canonical STRING mapping resolves the 4 HGNC identifiers claimed by two "
     "proteins by keeping the approved-symbol protein and discarding the other, which may remove real protein-level "
     "evidence (isoforms, distinct products). Owner: 1D-3: keep the protein graph intact and derive gene-level edges by "
     "a declared, evidence-preserving aggregation.",
     ["string_relation:combined_score_threshold", "gnn_score", "hetero_gnn_score"],
     [code("src/genomic_variant_classifier/information_sources/graph_identity.py", "        winners = [p for p in ps if first[p][1] is MappingStatus.APPROVED_SYMBOL]"),
      "record:docs/census/GRAPH_IDENTITY_1D1b3.json"]),
    # ---- registered by set "as" (stage 1D-1b.4), measured 2026-11-10 and 2026-11-14, each with an executable witness
    ("GNN-EARLY-STOP-ROW-SPLIT-1", "train_gnn_pipeline early-stops on train_test_split(np.arange(n), "
     "test_size=test_split, random_state=42, stratify=y): a ROW-level split of the focal training rows with no groups "
     "(the run driver passes test_split=0.15). The same call on Run 15's 1,038,974 training rows: 155,488 of the "
     "155,847 early-stopping rows (99.77%) share a gene symbol with the inner training rows and 155,510 share a split "
     "component, and 7,543 of their 7,860 symbols are inner-training symbols, so checkpoint selection rewards memorised "
     "genes. Repair (owner 1D-3): population.component_checkpoint_split, component-disjoint and class-stratified (on the "
     "910,273 component-authorized rows: inner validation 13.319% positive against 13.320% overall) "
     "(docs/census/POPULATION_ISOLATION_1D1b4.json).",
     ["gnn_score", "supervision:gnn_label"],
     [code(GNN, "    train_pos, val_pos = train_test_split("), code(DRV, "test_split=0.15,"),
      code("tests/unit/test_registered_defect_witnesses.py", "def test_witness_gnn_early_stop_row_split_1():"),
      "record:docs/census/POPULATION_ISOLATION_1D1b4.json"]),
    ("GENE-FIELD-PROSE-1", "32 Run 15 rows (29 training, 1 validation, 2 test; 10 distinct fields) carry ClinVar's "
     "description of a large variant in gene_symbol instead of gene symbols (e.g. 'covers 26 genes, none of which "
     "curated to show dosage sensitivity'). The split groups by the raw field (gene_symbol, missing values filled with "
     "'unknown'), so each description is one 'gene' group and the genes it covers bind nothing; the stage 1D-1b.4 "
     "resolver (population.resolve_gene_field) flags them as prose and leaves the rows gene-incomplete, never "
     "authorized. Their consequence values (13 intronic, 9 missense, 4 3' UTR, 2 synonymous, 1 non-coding transcript, "
     "1 splice donor, 2 none) are to be checked against the ClinVar records. Owner: 1D-1d (recover the covered genes "
     "from the ClinVar record, or exclude the rows by a declared rule).",
     ["supervision:gnn_label", "supervision:hetero_gnn_label", "gnn_score", "hetero_gnn_score", "n_pathogenic_in_gene"],
     [code(RDP, 'group_column: str = "gene_symbol"'),
      code(RDP, 'groups = df[self.config.group_column].fillna("unknown").reset_index(drop=True)\n\n'),
      code("tests/unit/test_registered_defect_witnesses.py", "def test_witness_gene_field_prose_1():"),
      "record:docs/census/POPULATION_ISOLATION_1D1b4.json"]),
    ("GNOMAD-SOURCE-REVISED-AFTER-RUN15-1", "data/processed/gnomad_v4_exomes.parquet was rewritten on 2026-06-16 16:45 UTC, "
     "after Run 15 (2026-06-05), when build_gnomad_ymt_af.py appended 3,778 'gnomad:'-prefixed Y and MT records (1,047 Y, "
     "2,731 MT; merge_into_gnomad rewrites the whole file). No digest of the bytes Run 15 read was recorded, so Run 15's "
     "allele-frequency input cannot be byte-certified from the current file: Run 15 logged 990,350 rows with a frequency; "
     "the current join gives 992,044 (990,350 + 20 Y + 1,674 MT). Run 15 stays a historical record with this "
     "qualification (stage 1D-1c.1, owner ruling 4bef3910).",
     ['af_raw', 'af_log10', 'af_is_absent', 'af_is_ultra_rare', 'af_is_rare', 'af_is_common'],
     [code("scripts/build_gnomad_ymt_af.py", "def merge_into_gnomad(ymt_df: pd.DataFrame, base_path: str, out_path: str) -> tuple:"),
      "record:docs/census/PILOTS_1D1b2.json", "record:docs/census/AF_REPAIR_1D1c1.json"]),
    ("GNOMAD-YMT-STATISTIC-MIXING-1", "The appended API records carry statistics other than the extract's: Y is exome AF "
     "else genome AF per variant (fallback unrecorded) and MT is the homoplasmic fraction ac_hom / an. The CURRENT join "
     "parses them ('gnomad:MT:100:T:C' -> MT) and writes them into allele_freq, so the next production run would "
     "train on 1,674 MT homoplasmic fractions and 20 Y values as if they were the primary frequency. Stage 1D-1c.1 types "
     "them as their own logical source (dataset:gnomad_ymt_api) and never selects them (owner rulings 4bef3910, "
     "b937681e).",
     ['af_raw', 'af_log10', 'af_is_absent', 'af_is_ultra_rare', 'af_is_rare', 'af_is_common'],
     [code("scripts/build_gnomad_ymt_af.py", "out[norm_key(vid)] = float(ac_hom) / float(an)"), code("scripts/build_gnomad_ymt_af.py", 'af = (v.get("genome") or {}).get("af")'),
      code(RDP, 'df["allele_freq"] = _af.fillna(_gf)'), "record:docs/census/AF_REPAIR_1D1c1.json"]),
    ("GNOMAD-DUPLICATE-LOCUS-FIRST-WINS-1", "DataPrepPipeline._join_gnomad resolves a locus that occurs more than once by "
     "keeping the first record, whatever the values; build_gnomad_parquet.py does the same on variant_id. Latent today: "
     "the current file has no duplicated normalised locus (measured, stage 1D-1c.1). The 1D-1c.1 materialiser refuses "
     "a duplicated key instead.",
     ['af_raw', 'af_log10', 'af_is_absent', 'af_is_ultra_rare', 'af_is_rare', 'af_is_common'],
     [code(RDP, '.drop_duplicates(subset=["_chrom", "_pos", "_ref", "_alt"])['), "record:docs/census/AF_REPAIR_1D1c1.json"]),
    ("GNOMAD-PRIMARY-EXOME-GENOME-FALLBACK-1", "build_gnomad_parquet.py takes the INFO field AF and, where it is absent "
     "or empty, AF_genome, and records nothing about which it used: the primary statistic is exome-else-genome with "
     "fallback events unrecorded (named gnomad_preprocessed_mixed_af_unfiltered_v1). Whether the fallback ever fired "
     "cannot be measured without the source VCFs. A reported zero cannot be read as AC0 for the same reason.",
     ['af_raw', 'af_log10', 'af_is_absent', 'af_is_ultra_rare', 'af_is_rare', 'af_is_common'],
     [code("scripts/build_gnomad_parquet.py", "af_str = info[_EXOME_AF_FIELD] or info[_GENOME_AF_FIELD]"), "record:docs/census/AF_REPAIR_1D1c1.json"]),
    ("GNOMAD-FILTER-NOT-RECORDED-1", "build_gnomad_parquet.py never reads the VCF FILTER column, so records that failed "
     "gnomAD quality control (AC0, RF, InbreedingCoeff) are included indistinguishably; 130,341 of the extract's "
     "2,947,370 records are exactly 0, and 42,571 Run 15 rows select one. Their quality is unknown, not PASS and not AC0. "
     "Measurement owner: a FILTER-aware, allele-specific source reconstruction (a separately identified source).",
     ['af_raw', 'af_log10', 'af_is_absent', 'af_is_ultra_rare', 'af_is_rare', 'af_is_common'],
     [code("scripts/build_gnomad_parquet.py", "chrom, pos, _id, ref, alt_field, _qual, _filter, info_str = ("), "record:docs/census/AF_REPAIR_1D1c1.json"]),
    ("GNOMAD-YPAR-REKEY-DUPLICATES-1", "build_gnomad_ymt_af.py re-keys gnomAD's X-PAR records to Y coordinates (PAR1 "
     "identical, PAR2 minus 98,813,480): 913 of the 956 Y-PAR API records have an X twin in the primary extract with the "
     "same REF and ALT, equal within six-significant-digit rounding (relative difference <= 4.75e-6). The same "
     "biological variant is represented twice; exact equivalence is not certified. No Run 15 row is a Y-PAR "
     "representation (measured).",
     ["af_raw"],
     [code("scripts/build_gnomad_ymt_af.py", 'return f"Y:{ip - _PAR2_SHIFT}:{ref}:{alt}"'), "record:docs/census/AF_REPAIR_1D1c1.json"]),
    ("GNOMAD-BUILD-FILTER-IDENTITY-UNRECORDED-1", "build_gnomad_parquet.py keeps only VCF lines whose locus is in the "
     "--clinvar locus set, and records neither that file's identity nor its digest. Every record of the extract and "
     "every Run 15 locus lies in data/processed/clinvar_grch38.parquet's locus set (measured), consistent with that "
     "file (or a superset) having been the filter; it is not proven.",
     ['af_raw', 'af_log10', 'af_is_absent', 'af_is_ultra_rare', 'af_is_rare', 'af_is_common'],
     [code("scripts/build_gnomad_parquet.py", "if loci_filter is not None and locus not in loci_filter:"), "record:docs/census/AF_REPAIR_1D1c1.json"]),
    ("GNOMAD-QUERY-UNIVERSE-UNRECORDED-1", "(Registered by the owner, ruling b937681e.) The complete set of attempted "
     "source queries and their outcomes was not recorded -- the ClinVar-ascertained VCF scan for the primary extract, "
     "and for the API append one request per cleaned Y gene symbol (rows of uncleanable symbols were never queried) "
     "and one mitochondrial region call. A row missing from the extract is therefore 'no row in the extracted table', "
     "never 'not in gnomAD'.",
     ['af_raw', 'af_log10', 'af_is_absent', 'af_is_ultra_rare', 'af_is_rare', 'af_is_common'],
     [code("scripts/build_gnomad_ymt_af.py", "def fetch_y_af(clean_genes, dataset: str, pause: float = 6.0) -> dict:"),
      code("scripts/build_gnomad_parquet.py", "if loci_filter is not None and locus not in loci_filter:"), "record:docs/census/AF_REPAIR_1D1c1.json"]),
    ("LOEUF-SENTINEL-1", "(Registered at stage 1D-1c.1C, owner ruling aa999b96, audit R2.) The model feature loeuf is "
     "filled with 1.0 -- an assertion of loss-of-function tolerance -- when its source is absent, while "
     "gene_is_constrained, a STRICT function of the same LOEUF, is three-valued (NaN): one derivation family with a "
     "fabricated root and an unknown dependent, so it cannot be imputed jointly. Measured 2026-11-16 on Run 15's "
     "meta_train.parquet: loeuf has no NaN but 57,556 rows at exactly 1.0 against 34,688 in the whole (0.98, 1.02) "
     "neighbourhood (1,866 distinct values); rows at 1.0 are 24.1% pathogenic against 13.5% elsewhere -- an upstream "
     "sentinel carrying ascertainment. Repair: the constraint stage (1D-1c.2-1c.5).",
     ["loeuf", "gene_is_constrained"],
     [ve('feats["loeuf"] = ('),
      code("tests/unit/test_registered_defect_witnesses.py", "def test_witness_loeuf_sentinel_1():")]),
    ("GNOMAD-MT-AC-HOM-NULL-AS-ZERO-1", "build_gnomad_ymt_af.py computes the MT value as (ac_hom or 0) / an, so a "
     "missing homoplasmic count becomes a homoplasmic fraction of 0.0, indistinguishable from a genuine zero (ac_het "
     "was not kept). 660 of the 2,731 MT records are exactly 0; 111 Run 15 MT rows match one, 75 of them pathogenic "
     "(measured, stage 1D-1c.1). Not selected into any model input; a future mitochondrial source must keep ac_hom, "
     "ac_het and an.",
     ["af_raw"],
     [code("scripts/build_gnomad_ymt_af.py", 'ac_hom = v.get("ac_hom") or 0'), "record:docs/census/AF_REPAIR_1D1c1.json"]),
    # ---- registered by set "aw" (stage 1D-1c.2A, owner ruling e755519d), each with an executable witness
    ("DBNSFP-INDEX-DEFAULT-BAKED-1", "data/dbnsfp.py's raw parse fills an absent SIFT, PolyPhen-2, REVEL or CADD score with "
     "the connector default (0.5; 15.0 for CADD) BEFORE writing the parquet cache, so the cached index holds no missing value "
     "and a stored 0.5 is a score or a fill, irreversibly; replacing 0.5 by NaN cannot repair it (a genuine 0.5 is the same "
     "value). Measured (docs/census/PREDICTOR_SOURCES_1D1c2.json): dbnsfp_full_index.parquet, 85,299,339 rows, no null; "
     "exactly at the default REVEL 15,505,294 rows against about 44,000 at each neighbouring value, SIFT 16,413,824 (about "
     "17,400), PolyPhen-2 18,787,223 (about 13,000), CADD 44,611 (about 19,000); the masks differ (all three 12,828,878; "
     "REVEL alone 308,674). On the pilot population's 320,626 joined rows, 129,779 REVEL, 132,603 SIFT and 137,944 "
     "PolyPhen-2 values are exactly 0.5, mostly nonsense and canonical-splice variants (REVEL: 65,096, 20,525 donor, 17,077 "
     "acceptor), so the value encodes the consequence class and separates the labels BETWEEN classes (REVEL at 0.5: 49.6% "
     "of pathogenic against 2.0% of benign rows; AUROC 0.738 over all pilot rows, 0.748 over the joined rows) but hardly "
     "WITHIN them (joined rows: missense 0.489, nonsense 0.500, intron 0.492): an invalid representation of REVEL that "
     "carries the consequence, not established target leakage (owner ruling ace92e09, section 15). Against the publisher's REVEL v1.3 at those alleles: 66 genuine 0.5, "
     "115,538 without a publisher score, 14,175 with another publisher score the index lost; 42,861 further alleles have a "
     "publisher score and no index row. SIFT and CADD are strict-admitted: withheld until qualified (owner ruling e755519d).",
     ["sift_score", "polyphen2_score", "revel_score", "cadd_phred", "cadd_high", "sift_deleterious", "polyphen_probably_damaging", "revel_pathogenic", "n_tools_pathogenic"],
     [code(D + "dbnsfp.py", "chunk[out_col] = chunk[out_col].fillna(default)"), "record:docs/census/PREDICTOR_SOURCES_1D1c2.json",
      code("tests/unit/test_registered_defect_witnesses.py", "def test_witness_dbnsfp_index_default_baked_1(tmp_path):")]),
    ("PREDICTOR-TRANSCRIPT-COLLAPSE-1", "data/dbnsfp.py reduces each per-transcript score list to one number -- SIFT its "
     "minimum, PolyPhen-2, REVEL and CADD their maximum -- and keeps the first record of an allele, so which transcript, "
     "protein context and amino-acid change the stored value describes is lost. Measured against the publisher's REVEL "
     "v1.3 (docs/census/PREDICTOR_SOURCES_1D1c2.json): 10,584 of the pilot population's 247,947 alleles with publisher rows "
     "have several distinct scores and 11,188 several amino-acid changes; the index value equals the publisher maximum at "
     "188,525 alleles and differs at 2,320. A maximum is a research feature of its own identity, never the observation.",
     ["sift_score", "polyphen2_score", "revel_score", "sift_deleterious", "polyphen_probably_damaging", "revel_pathogenic",
      "n_tools_pathogenic"],
     [code(D + "dbnsfp.py", '_COL_REVEL:  "max",'), code(D + "dbnsfp.py", 'full = full.drop_duplicates(subset=["chrom", "pos", "ref", "alt"])'),
      "record:docs/census/PREDICTOR_SOURCES_1D1c2.json", code("tests/unit/test_registered_defect_witnesses.py", "def test_witness_predictor_transcript_collapse_1(tmp_path):")]),
    ("DBNSFP-PARSE-FAILURE-SILENT-DEFAULTS-1", "data/dbnsfp.py catches every exception of the raw parse, logs it and returns "
     "an EMPTY index; annotate_dataframe then writes all six defaults to every row, so an unreadable or renamed source "
     "becomes plausible scores with only a log line (no refusal).",
     ["sift_score", "polyphen2_score", "revel_score", "cadd_phred", "phylop_score", "gerp_score"],
     [code(D + "dbnsfp.py", 'logger.error("DbNSFP: failed to parse %s: %s", self._path, exc)'),
      code("tests/unit/test_registered_defect_witnesses.py", "def test_witness_dbnsfp_parse_failure_silent_defaults_1(tmp_path):")]),
    ("COHORT-TRANSCRIPT-UNDECLARED-1", "No prediction unit declares a transcript: the ClinVar connector writes transcript_id "
     "= None and keeps no Name, and the cohort's transcript_id and protein_change are null in every row, while ClinVar's own Name declares the "
     "RefSeq transcript and protein change (NM_014630.3(ZNF592):c.3136G>A (p.Gly1046Arg)). Measured "
     "(docs/census/PREDICTOR_SOURCES_1D1c2.json): both null in all 1,490,014 pilot rows; 1,489,881 rows have one "
     "GRCh38 variant_summary record by VariationID and allele. Of the 201,130 missense rows, ClinVar's preferred name, "
     "matched to the publisher's rows by the allele and the amino-acid pair only (the RefSeq accession and the residue "
     "position are not matched: the publisher has neither), gives a CANDIDATE score for 191,427 (38 have publisher rows of "
     "another amino-acid change only, 1,568 no publisher row, 6,926 no single-residue declaration or no unique record, "
     "1,171 are not single-nucleotide); the GENCODE v50 MANE Select transcript of the cohort's gene_symbol, matched by "
     "stable identifier with no version or coding-sequence check, gives a candidate for 160,973 (34,284 with publisher rows "
     "on other transcripts only: the 2021 transcript set against GENCODE v50). Neither is a qualified transcript context "
     "(stage 1D-1c.2B; owner ruling ace92e09). The MANE policy is not independent of the ClinVar record: it reads the "
     "cohort's gene_symbol, and at 848 single-nucleotide missense positions the one MANE coding sequence covering the "
     "position is another gene's (75 are covered by several). Both give a candidate for 160,886 rows, the same score at "
     "160,723; the other 163 are policy disagreements, not source conflicts -- in all 163 the MANE-selected row has another "
     "amino-acid change than ClinVar's declared one (15 cross 0.5).",
     ["revel_score", "sift_score", "polyphen2_score"],
     [code(D + "database_connectors.py", 'df["transcript_id"] = None'), "record:docs/census/PREDICTOR_SOURCES_1D1c2.json",
      code("tests/unit/test_registered_defect_witnesses.py", "def test_witness_cohort_transcript_undeclared_1(tmp_path):")]),
]


def build(repo: Path) -> dict:
    contract = C.source_contract(repo)
    names = list(contract.names)
    entries = {e["name"]: e for e in ENTRIES}
    if sorted(entries) != sorted(names) or len(entries) != len(ENTRIES):
        raise SystemExit(f"reviewed entries differ from the contract: missing {sorted(set(names) - set(entries))}, "
                         f"extra {sorted(set(entries) - set(names))}")
    surfaces = K.observe_surfaces(repo, names)
    graph = K.observe_graph_sources(repo)
    ordered = []
    for n in names:
        e = dict(entries[n])
        e["consumers"] = sorted(s for s, rec in surfaces.items() if n in rec["observed"])
        ordered.append(e)
    measured = {
        "gnn_score": ["surface:gnn_node_attributes", "string_relation:combined_score_threshold",
                      "string_edge_attribute:experimental", "string_edge_attribute:database",
                      "string_edge_attribute:coexpression", "supervision:gnn_label"],
        "hetero_gnn_score": ["surface:hetero_gnn_node_attributes", "string_relation:combined_score_threshold"]
        + [f"kg_relation:{s}" for s in graph["kg_relation_sources"]] + ["supervision:hetero_gnn_label"],
    }
    for name, rule in measured.items():
        need: set[str] = set()
        for item in rule:
            need |= set(surfaces[item[8:]]["observed"]) if item.startswith("surface:") else {item}
        need.discard(name)
        entry = next(e for e in ordered if e["name"] == name)
        entry["depends_on"] = sorted(need)
    sites = K.discover_sites(repo)
    if sorted(sites) != sorted(SITES):
        raise SystemExit(f"consumer sites changed: discovered {sorted(set(sites) - set(SITES))} unreviewed; "
                         f"reviewed but gone {sorted(set(SITES) - set(sites))}")
    symbols = set()
    for r in [*ordered, *SOURCES]:
        symbols.add(r["producer"])
        symbols.update(s[5:] for s in r["source_ids"] if s.startswith("code:"))
    return {
        "schema": C.SCHEMA,
        "authority": {"path": C.CONTRACT_FILE, "contract_digest": contract.contract_digest,
                      "file_sha256_at_capture": contract.file_sha256},
        "expected_feature_count": len(names),
        "entries": ordered,
        "information_sources": sorted(SOURCES, key=lambda s: s["name"]),
        "datasets": {k: {"description": d, "access": a, "evidence_refs": ev} for k, (d, a, ev) in sorted(DATASETS.items())},
        "producer_digests": {s: C.symbol_closure_digest(repo, s) for s in sorted(symbols)},
        "surfaces": surfaces,
        "consumer_sites": {k: {"symbols": sites[k], "classes": SITES[k][0], "reason": SITES[k][1]} for k in sorted(sites)},
        "measured_dependencies": measured,
        "registered_defects": [{"id": i, "statement": s, "affects": a, "evidence_refs": ev} for i, s, a, ev in DEFECTS],
        "graph_inventory": graph,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path)
    group.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    repo = a.repo.resolve()
    ledger = build(repo)
    raw = C.ledger_bytes(ledger)
    report = C.verify_census(repo, C.strict_loads(raw.decode("ascii")))
    if a.check:
        committed = (repo / C.LEDGER_PATH).read_bytes()
        if committed != raw:
            print("FEATURE CENSUS DIFFERS from what the reviewed builder writes", file=sys.stderr)
            return 1
        print("feature census: committed ledger is byte-identical to the build")
    else:
        if a.out.exists():
            raise SystemExit(f"refusing to overwrite {a.out}; delete it deliberately first")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_bytes(raw)
        print(f"wrote {a.out} ({len(raw)} bytes)")
    for line in report.lines():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
