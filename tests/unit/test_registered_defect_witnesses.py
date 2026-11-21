"""Executable witnesses of registered defects (owner ruling 68ca322f, 2026-11-10; set "ar" revision B onward).

Each test PASSES while its defect still exists, so the registration in the census ledger is evidence, not prose. A stage that
repairs a defect will see its witness FAIL: the repair must then remove the witness and the registration together, in the
same change. Set "as" (stage 1D-1b.4) repaired the four defects set "ar" registered with witnesses -- LABEL-TYPE-IDENTITY-1
and LABEL-DENIED-READ-AUDIT-1 (labels.TrainingLabels), GRAPH-MAPPING-RECEIPT-AUTHORITY-1 (graph_identity.accept_graph
re-derives from the pinned files) and GRAPH-MUTABLE-MATERIALIZATION-1 (bytes-backed read-only buffers, read-only mappings)
-- and their witnesses now FAIL: they were removed with their registrations, and their inverted forms are tests in
test_label_authority.py and test_graph_trust_boundary.py. The two defects below are in production code (models/gnn.py,
data/real_data_prep.py); registering them changes no behaviour. Set "at" (stage 1D-1c.1) adds witnesses for the gnomAD
extraction and join defects it registered: they execute the builders (scripts/build_gnomad_parquet.py in a subprocess, because
it configures logging when imported; scripts/build_gnomad_ymt_af.py by path) and the CURRENT join through the pilot's
legacy_af_values. Author: Monzia Moodie."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GNN = REPO / "src/genomic_variant_classifier/models/gnn.py"
RDP = REPO / "src/genomic_variant_classifier/data/real_data_prep.py"


def _registered(defect_id: str) -> None:
    from genomic_variant_classifier.information_sources import census as C

    ids = {d["id"] for d in C.strict_load(REPO / C.LEDGER_PATH)["registered_defects"]}
    assert defect_id in ids, f"{defect_id} is witnessed here but not registered in the census ledger"


def _function(path: Path, name: str, cls: str | None = None) -> ast.FunctionDef:
    tree = ast.parse(path.read_bytes())
    body = tree.body
    if cls is not None:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    return next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_witness_gnn_early_stop_row_split_1():
    """GNN-EARLY-STOP-ROW-SPLIT-1: train_gnn_pipeline early-stops on a ROW-level split of the training rows
    (train_test_split over np.arange(n), stratified by label, no groups), so rows of one gene sit on both sides of the
    checkpoint split. Executed here on 4 genes of 10 rows each: every gene lands on both sides."""
    _registered("GNN-EARLY-STOP-ROW-SPLIT-1")
    fn = _function(GNN, "train_gnn_pipeline")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "train_test_split"]
    assert len(calls) == 1
    call = calls[0]
    assert ast.unparse(call.args[0]) == "np.arange(n)"
    assert {k.arg for k in call.keywords} == {"test_size", "random_state", "stratify"}       # no groups
    from sklearn.model_selection import train_test_split

    genes = np.repeat(np.arange(4), 10)
    y = np.tile(np.array([0, 1] * 5), 4)
    tr, va = train_test_split(np.arange(40), test_size=0.15, random_state=42, stratify=y)
    assert set(genes[tr]) & set(genes[va])


def test_witness_gene_field_prose_1():
    """GENE-FIELD-PROSE-1: the legacy split's group is the raw gene field, so a ClinVar description ("covers 26 genes,
    ...") is one 'gene' group and the genes it covers bind nothing; the shadow resolver flags it as prose and leaves the row
    gene-incomplete."""
    _registered("GENE-FIELD-PROSE-1")
    tree = ast.parse(RDP.read_bytes())
    groups = [ast.unparse(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "groups"]
    assert groups.count("df[self.config.group_column].fillna('unknown').reset_index(drop=True)") == 2
    from genomic_variant_classifier.information_sources import graph_identity as G
    from genomic_variant_classifier.information_sources import population as P

    hgnc = G.HgncIndex({"AAA": "HGNC:1"}, {}, {}, "a" * 64, 1)
    got = P.resolve_gene_field("covers 26 genes, none of which curated to show dosage sensitivity", hgnc)
    assert got.prose and not got.gene_ids


# ------------------------------------------------------------------------------- stage 1D-1c.1 (set "at"): gnomAD defects
def _ymt_builder():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("build_gnomad_ymt_af", REPO / "scripts" / "build_gnomad_ymt_af.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_gnomad_ymt_af"] = mod
    spec.loader.exec_module(mod)
    return mod


def _parse_vcf(tmp_path, lines, loci=None):
    """scripts/build_gnomad_parquet.parse_vcf in a fresh interpreter (the module calls logging.basicConfig at import)."""
    import json
    import subprocess
    import sys

    vcf = tmp_path / "x.vcf"
    vcf.write_bytes(("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                     + "".join(l + "\n" for l in lines)).encode("ascii"))
    code = ("import importlib.util, json, sys\n"
            "spec = importlib.util.spec_from_file_location('b', sys.argv[1]); m = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(m)\n"
            "loci = None if sys.argv[3] == 'null' else set(json.loads(sys.argv[3]))\n"
            "print(json.dumps(m.parse_vcf(sys.argv[2], loci)))\n")
    done = subprocess.run([sys.executable, "-c", code, str(REPO / "scripts" / "build_gnomad_parquet.py"), str(vcf),
                           "null" if loci is None else json.dumps(sorted(loci))], capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def _legacy_join(tmp_path, gnomad_rows, cohort_rows):
    import pandas as pd

    from genomic_variant_classifier.information_sources import materialization as M

    path = tmp_path / "g.parquet"
    pd.DataFrame(gnomad_rows, columns=["variant_id", "allele_freq"]).to_parquet(path, index=False)
    frame = pd.DataFrame(cohort_rows, columns=["variant_id", "chrom", "pos", "ref", "alt"])
    frame["allele_freq"] = float("nan")
    return M.legacy_af_values(frame, path)


def test_witness_gnomad_ymt_statistic_mixing_1(tmp_path):
    """GNOMAD-YMT-STATISTIC-MIXING-1: the CURRENT join parses an appended API identifier ('gnomad:MT:...') and writes its
    value -- an MT homoplasmic fraction -- into allele_freq, the primary frequency."""
    _registered("GNOMAD-YMT-STATISTIC-MIXING-1")
    out = _legacy_join(tmp_path, [("gnomad:MT:100:T:C", 0.375), ("1:5:A:G", 0.1)],
                       [("clinvar:MT:100:T:C", "MT", 100, "T", "C"), ("clinvar:1:5:A:G", "1", 5, "A", "G")])
    assert out.tolist() == [0.375, 0.1]


def test_witness_gnomad_duplicate_locus_first_wins_1(tmp_path):
    """GNOMAD-DUPLICATE-LOCUS-FIRST-WINS-1: of two records of one locus with different values, the join keeps the first."""
    _registered("GNOMAD-DUPLICATE-LOCUS-FIRST-WINS-1")
    out = _legacy_join(tmp_path, [("1:5:A:G", 0.1), ("1:5:A:G", 0.9)], [("clinvar:1:5:A:G", "1", 5, "A", "G")])
    assert out.tolist() == [0.1]


def test_witness_gnomad_primary_exome_genome_fallback_1(tmp_path):
    """GNOMAD-PRIMARY-EXOME-GENOME-FALLBACK-1: without an AF field the builder takes AF_genome, and the record says nothing of
    which field it used."""
    _registered("GNOMAD-PRIMARY-EXOME-GENOME-FALLBACK-1")
    rows = _parse_vcf(tmp_path, ["chr1\t5\t.\tA\tG\t.\tPASS\tAF_genome=0.3", "chr1\t6\t.\tA\tG\t.\tPASS\tAF=0.2;AF_genome=0.3"])
    assert rows == [{"variant_id": "1:5:A:G", "allele_freq": 0.3}, {"variant_id": "1:6:A:G", "allele_freq": 0.2}]


def test_witness_gnomad_filter_not_recorded_1(tmp_path):
    """GNOMAD-FILTER-NOT-RECORDED-1: a record that failed gnomAD quality control (FILTER AC0) is kept, and nothing records it."""
    _registered("GNOMAD-FILTER-NOT-RECORDED-1")
    rows = _parse_vcf(tmp_path, ["chr1\t5\t.\tA\tG\t.\tAC0\tAF=0", "chr1\t6\t.\tA\tG\t.\tRF\tAF=0.25"])
    assert rows == [{"variant_id": "1:5:A:G", "allele_freq": 0.0}, {"variant_id": "1:6:A:G", "allele_freq": 0.25}]


def test_witness_gnomad_build_filter_identity_unrecorded_1(tmp_path):
    """GNOMAD-BUILD-FILTER-IDENTITY-UNRECORDED-1: the ClinVar locus filter decides which records exist, and the output
    records neither the filter nor its identity."""
    _registered("GNOMAD-BUILD-FILTER-IDENTITY-UNRECORDED-1")
    lines = ["chr1\t5\t.\tA\tG\t.\tPASS\tAF=0.2", "chr1\t6\t.\tA\tG\t.\tPASS\tAF=0.3"]
    assert _parse_vcf(tmp_path, lines, {"1:5:A:G"}) == [{"variant_id": "1:5:A:G", "allele_freq": 0.2}]


def test_witness_gnomad_ypar_rekey_duplicates_1():
    """GNOMAD-YPAR-REKEY-DUPLICATES-1: an X-PAR record is re-keyed to Y coordinates (PAR2 shifted by 98,813,480), so one
    biological variant gets a second representation."""
    _registered("GNOMAD-YPAR-REKEY-DUPLICATES-1")
    b = _ymt_builder()
    assert b.y_key("X-155701383-A-G") == "Y:56887903:A:G" and b.y_key("X-10001-A-G") == "Y:10001:A:G"


def test_witness_gnomad_query_universe_unrecorded_1():
    """GNOMAD-QUERY-UNIVERSE-UNRECORDED-1: rows whose gene field cannot be cleaned are never queried, and the output keeps
    only matched records -- no attempted query, no outcome."""
    _registered("GNOMAD-QUERY-UNIVERSE-UNRECORDED-1")
    b = _ymt_builder()
    assert b.clean_y_genes(["SRY", "covers 26 genes, none of which", "-", "TBL1Y;SRY"]) == ["SRY", "TBL1Y"]
    frame = b.build_ymt_frame({"Y:5:A:G", "Y:6:A:G"}, {"Y:5:A:G": 0.1}, {})
    assert list(frame.columns) == ["variant_id", "allele_freq"] and frame["variant_id"].tolist() == ["gnomad:Y:5:A:G"]


def test_witness_gnomad_mt_ac_hom_null_as_zero_1():
    """GNOMAD-MT-AC-HOM-NULL-AS-ZERO-1: a missing homoplasmic count becomes a homoplasmic fraction of 0.0."""
    _registered("GNOMAD-MT-AC-HOM-NULL-AS-ZERO-1")
    b = _ymt_builder()
    payload = {"data": {"region": {"mitochondrial_variants": [
        {"variant_id": "M-100-T-C", "an": 200, "ac_hom": None, "ac_het": 7},
        {"variant_id": "M-200-A-G", "an": 200, "ac_hom": 0, "ac_het": 0}]}}}
    assert b.parse_mt_af(payload) == {"MT:100:T:C": 0.0, "MT:200:A:G": 0.0}


def test_witness_gnomad_source_revised_after_run15_1(tmp_path):
    """GNOMAD-SOURCE-REVISED-AFTER-RUN15-1: the merge rewrites the whole extract and returns counts only -- no digest of the
    bytes it replaced."""
    import pandas as pd

    _registered("GNOMAD-SOURCE-REVISED-AFTER-RUN15-1")
    b = _ymt_builder()
    base = tmp_path / "g.parquet"
    pd.DataFrame({"variant_id": ["1:5:A:G"], "allele_freq": [0.1]}).to_parquet(base, index=False)
    before = base.read_bytes()
    out = b.merge_into_gnomad(pd.DataFrame({"variant_id": ["gnomad:MT:1:A:G"], "allele_freq": [0.5]}), str(base), str(base))
    assert out == (1, 1, 2) and base.read_bytes() != before


# --------------------------------------------------------------------------- stage 1D-1c.1C (set "au"): gene constraint
def test_witness_loeuf_sentinel_1():
    """LOEUF-SENTINEL-1: with no LOEUF the builder writes loeuf = 1.0 (tolerant) and gene_is_constrained = NaN (unknown)
    -- in either epoch -- so the family's root is fabricated while its dependent is three-valued."""
    _registered("LOEUF-SENTINEL-1")
    import pandas as pd

    from genomic_variant_classifier.models.feature_semantics import FeatureSemantics
    from genomic_variant_classifier.models.variant_ensemble import engineer_features

    raw = pd.DataFrame({"allele_freq": [0.1, float("nan")], "loeuf": [float("nan"), 0.2]})
    for semantics in FeatureSemantics:
        out = engineer_features(raw, semantics=semantics)
        assert out.loc[0, "loeuf"] == 1.0 and np.isnan(out.loc[0, "gene_is_constrained"])
        assert out.loc[1, "loeuf"] == 0.2 and out.loc[1, "gene_is_constrained"] == 1.0


# ------------------------------------------------------------- stage 1D-1c.2A (set "aw"): computational-predictor sources
_DBNSFP_HEADER = ("#chr", "pos(1-based)", "ref", "alt", "SIFT_score", "Polyphen2_HDIV_score", "REVEL_score", "CADD_phred",
                  "phyloP100way_vertebrate", "GERP++_RS")


def _dbnsfp_source(tmp_path, rows, header=_DBNSFP_HEADER):
    raw = tmp_path / "dbNSFP_test_variant.tsv"
    raw.write_text("\t".join(header) + "\n" + "".join("\t".join(r) + "\n" for r in rows), encoding="ascii",
                   newline="\n")
    return raw


def test_witness_dbnsfp_index_default_baked_1(tmp_path):
    """Legacy projection maps a missing score and an observed default-valued score to the same number."""
    from genomic_variant_classifier.data.dbnsfp import DbNSFPConnector

    _registered("DBNSFP-INDEX-DEFAULT-BAKED-1")
    raw = _dbnsfp_source(tmp_path, [("1", "100", "A", "G", ".", ".", ".", ".", "1.0", "2.0"),
                                    ("1", "101", "A", "G", "0.5", "0.5", "0.5", "15.0", "1.0", "2.0")])
    source = DbNSFPConnector(raw)
    assert source.get_scores("1", 100, "A", "G") == source.get_scores("1", 101, "A", "G")


def test_witness_predictor_transcript_collapse_1(tmp_path):
    """The legacy projection still collapses lists within one row to extrema."""
    from genomic_variant_classifier.data.dbnsfp import DbNSFPConnector

    _registered("PREDICTOR-TRANSCRIPT-COLLAPSE-1")
    raw = _dbnsfp_source(tmp_path, [("1", "100", "A", "G", "0.3;0.01", "0.2;0.9", "0.2;0.8", "20", "1", "2")])
    scores = DbNSFPConnector(raw).get_scores("1", 100, "A", "G")
    assert (scores.sift_score, scores.polyphen2_score, scores.revel_score) == (0.01, 0.9, 0.8)


def test_dbnsfp_corrupt_source_refuses_scalar_and_dataframe(tmp_path):
    import gzip

    import pandas as pd
    import pytest

    from genomic_variant_classifier.data.dbnsfp import DbNSFPConnector, DbNSFPSourceError

    row = ("1", "100", "A", "G", "0.01", "0.99", "0.95", "30", "1", "2")
    raw = _dbnsfp_source(tmp_path, [row])
    path = tmp_path / "raw.bgz"
    data = gzip.compress(raw.read_bytes())
    path.write_bytes(data[:-7])
    source = DbNSFPConnector(path)
    frame = pd.DataFrame({"chrom": ["1"], "pos": [100], "ref": ["A"], "alt": ["G"]})
    with pytest.raises(DbNSFPSourceError):
        source.get_scores("1", 100, "A", "G")
    with pytest.raises(DbNSFPSourceError):
        source.annotate_dataframe(frame)
    path.write_bytes(data)
    assert source.get_scores("1", 100, "A", "G").revel_score == 0.95


def test_witness_cohort_transcript_undeclared_1(tmp_path):
    """COHORT-TRANSCRIPT-UNDECLARED-1: ClinVar's own Name declares the transcript and the protein change
    (NM_...(GENE):c... (p....)), but the ClinVar connector writes transcript_id = None and keeps no Name, so no prediction
    unit of the cohort declares a transcript."""
    _registered("COHORT-TRANSCRIPT-UNDECLARED-1")
    from genomic_variant_classifier.data.database_connectors import ClinVarConnector, FetchConfig

    header = ["#AlleleID", "Type", "Name", "GeneSymbol", "ClinicalSignificance", "RS# (dbSNP)", "Assembly", "Chromosome",
              "Start", "ReferenceAllele", "AlternateAllele", "ReviewStatus", "VariationID"]
    row = ["1", "single nucleotide variant", "NM_014630.3(ZNF592):c.3136G>A (p.Gly1046Arg)", "ZNF592",
           "Uncertain significance", "150829393", "GRCh38", "15", "84799209", "G", "A",
           "no assertion criteria provided", "4"]
    src = tmp_path / "variant_summary.txt"
    src.write_text("\t".join(header) + "\n" + "\t".join(row) + "\n", encoding="ascii", newline="\n")
    out = ClinVarConnector(FetchConfig(cache_dir=tmp_path / "cache")).fetch(local_path=str(src), force_refresh=True)
    assert len(out) == 1 and out.loc[0, "transcript_id"] is None and out.loc[0, "protein_change"] is None
    assert "Name" not in out.columns
