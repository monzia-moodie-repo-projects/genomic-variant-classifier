"""test_split_health_gate.py -- Author: Monzia Moodie
Validates the recalibrated split-health gate: hard/soft degeneracy, the buckets,
GNN-stage placeholder exemption under prep_only, train-only frame scoping, and the
corrected EXPECTED_ZERO.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts"))
import split_health_gate as G  # noqa: E402


def test_hard_vs_soft_degeneracy():
    assert G.is_hard_degenerate("ALL_ZERO") is True
    assert G.is_hard_degenerate("CONSTANT;ALL_ZERO") is True
    assert G.is_hard_degenerate("NEAR_CONSTANT(0.9994)") is False
    assert G.is_hard_degenerate("") is False


def test_reason_from_health_extracts_string():
    assert G.reason_from_health({"degenerate": "ALL_ZERO"}) == "ALL_ZERO"
    assert G.reason_from_health({"degenerate": ""}) == ""


def test_reason_from_health_rejects_bad_contract():
    import pytest
    with pytest.raises(TypeError):
        G.reason_from_health("ALL_ZERO")
    with pytest.raises(TypeError):
        G.reason_from_health({"oops": 1})


def test_core_degenerate_is_no_go():
    res = G.classify({"alphamissense_score": "ALL_ZERO"})
    assert res["verdict"] == "NO_GO" and res["core_degenerate"] == ["alphamissense_score"]


def test_unexpected_degenerate_is_no_go():
    res = G.classify({"gtex_max_tpm": "ALL_ZERO"})
    assert res["verdict"] == "NO_GO" and res["unexpected_degenerate"] == ["gtex_max_tpm"]


def test_only_expected_stubs_is_go():
    res = G.classify({"eve_score": "CONSTANT", "phylop_score": "ALL_ZERO",
                      "esm2_llr": "ALL_ZERO", "dbsnp_af": "ALL_ZERO",
                      "codon_position": "ALL_ZERO"})
    assert res["verdict"] == "GO"
    assert set(res["expected_degenerate"]) == {"eve_score", "phylop_score", "esm2_llr",
                                               "dbsnp_af", "codon_position"}


def test_new_expected_zero_members():
    for c in ("esm2_llr", "dbsnp_af", "codon_position"):
        assert c in G.EXPECTED_ZERO, c


def test_gnn_stage_placeholder_exempt_under_prep_only():
    # gnn_score/hetero_gnn_score are 0.5 placeholders in prep output
    deg = {"gnn_score": "CONSTANT", "hetero_gnn_score": "CONSTANT"}
    assert G.classify(deg, prep_only=True)["verdict"] == "GO"          # exempt
    full = G.classify(deg, prep_only=False)
    assert full["verdict"] == "NO_GO"                                  # enforced post-GNN
    assert set(full["core_degenerate"]) == {"gnn_score", "hetero_gnn_score"}


def test_missing_core_absent_is_no_go_with_prep_only_gnn_exempt():
    present = G.CORE_FEATURES - G.GNN_STAGE_FEATURES
    assert G.classify({}, present=present, prep_only=True)["verdict"] == "GO"
    full = G.classify({}, present=present, prep_only=False)
    assert full["verdict"] == "NO_GO"
    assert set(full["missing_core"]) == G.GNN_STAGE_FEATURES


def test_near_constant_is_warning_not_failure():
    res = G.classify({"is_mitochondrial": "NEAR_CONSTANT(0.9994)"})
    assert res["verdict"] == "GO"
    assert res["near_constant_warnings"] == ["is_mitochondrial"]


def test_curated_sets_disjoint():
    assert not (G.EXPECTED_ZERO & G.CORE_FEATURES)


def test_score_splits_returns_strings_and_present(tmp_path):
    import pandas as pd
    n = 40
    df = pd.DataFrame({"alphamissense_score": [i / n for i in range(n)],
                       "gtex_max_tpm": [0.0] * n, "eve_score": [1.0] * n})
    sdir = tmp_path / "splits"; sdir.mkdir()
    df.to_parquet(sdir / "X_train.parquet")
    deg, present = G._score_splits(sdir, near_constant_frac=0.999)
    assert all(isinstance(v, str) for v in deg.values())
    assert present == {"alphamissense_score", "gtex_max_tpm", "eve_score"}


def test_score_splits_no_files_is_loud(tmp_path):
    import pytest
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        G._score_splits(tmp_path / "empty", near_constant_frac=0.999)


def test_train_only_feature_dead_in_val_test_is_ok(tmp_path):
    # n_pathogenic_in_gene: healthy in train, zero in val/test (gene-disjoint) -> NOT flagged
    import pandas as pd
    n = 40
    base = {c: [i / n for i in range(n)] for c in
            ["alphamissense_score", "revel_score", "revel_pathogenic", "sift_score",
             "sift_deleterious", "cadd_phred", "cadd_high", "splice_ai_score", "is_splice",
             "loeuf", "gerp_score", "pli_score", "consequence_severity"]}
    train = dict(base); train["n_pathogenic_in_gene"] = [float(i % 5) for i in range(n)]
    train["gene_has_known_disease"] = [float(i % 2) for i in range(n)]
    val = dict(base); val["n_pathogenic_in_gene"] = [0.0] * n          # zero by design
    val["gene_has_known_disease"] = [0.0] * n
    res = G.gate_frames({"X_train": pd.DataFrame(train), "X_val": pd.DataFrame(val)},
                        prep_only=True)
    assert res["verdict"] == "GO", res
    assert "n_pathogenic_in_gene" not in res["core_degenerate"]
    assert "gene_has_known_disease" not in res["unexpected_degenerate"]


def test_train_only_feature_dead_in_train_is_flagged(tmp_path):
    # if it's dead in TRAIN too, that IS a failure
    import pandas as pd
    n = 40
    train = {c: [i / n for i in range(n)] for c in
             ["alphamissense_score", "revel_score", "revel_pathogenic", "sift_score",
              "sift_deleterious", "cadd_phred", "cadd_high", "splice_ai_score", "is_splice",
              "loeuf", "gerp_score", "pli_score", "consequence_severity"]}
    train["n_pathogenic_in_gene"] = [0.0] * n                          # dead in TRAIN
    res = G.gate_frames({"X_train": pd.DataFrame(train)}, prep_only=True)
    assert res["verdict"] == "NO_GO"
    assert "n_pathogenic_in_gene" in res["core_degenerate"]


def test_gate_frames_healthy_prep_only_go():
    import pandas as pd
    n = 40
    healthy = {c: [i / n for i in range(n)] for c in
               ["alphamissense_score", "revel_score", "revel_pathogenic", "sift_score",
                "sift_deleterious", "cadd_phred", "cadd_high", "splice_ai_score",
                "is_splice", "loeuf", "gerp_score", "pli_score", "consequence_severity",
                "gtex_max_tpm", "af_1kg_eur", "finngen_af_fin", "n_pathogenic_in_gene"]}
    healthy["eve_score"] = [0.0] * n        # expected stub
    healthy["gnn_score"] = [0.5] * n        # placeholder, exempt under prep_only
    res = G.gate_frames({"X_train": pd.DataFrame(healthy)}, prep_only=True)
    assert res["verdict"] == "GO", res


def test_observed_prep_only_scenario_reduces_to_finngen(tmp_path):
    """Reconstruct the real --prep-only output (3 core + 16 unexpected) and confirm the
    recalibrated gate leaves ONLY the 3 finngen_* columns unexpected (dead locally only
    because --finngen-path was not passed); everything else is correctly reclassified."""
    import pandas as pd
    n = 60
    def varied(): return [i / n for i in range(n)]
    def zeros():  return [0.0] * n

    core_healthy = ["alphamissense_score", "revel_score", "revel_pathogenic", "sift_score",
                    "sift_deleterious", "cadd_phred", "cadd_high", "splice_ai_score",
                    "is_splice", "loeuf", "gerp_score", "pli_score", "consequence_severity"]
    revived = ["gtex_max_tpm", "gtex_n_tissues_expressed", "gtex_tissue_specificity",
               "af_1kg_eur", "gene_constraint_oe", "gene_is_constrained"]
    now_expected = ["dbsnp_af", "esm2_llr", "codon_position", "exon_number",
                    "dist_to_splice_site", "is_canonical_splice", "maxentscan_score",
                    "maxentscan_delta", "gtex_is_eqtl", "gtex_max_abs_effect",
                    "gtex_min_eqtl_pval", "has_uniprot_annotation",
                    "n_known_pathogenic_protein_variants",
                    # genuine stubs:
                    "alphafold_plddt", "clingen_validity_score", "dist_to_active_site",
                    "esm2_delta_norm", "eve_score", "hgmd_is_disease_mutation",
                    "hgmd_n_reports", "omim_is_autosomal_dominant", "omim_n_diseases",
                    "phylop_score", "secondary_structure_context", "solvent_accessibility"]
    finngen = ["finngen_af_fin", "finngen_af_nfsee", "finngen_enrichment"]

    def frame(is_train):
        d = {c: varied() for c in core_healthy}
        d["gnn_score"] = [0.5] * n            # prep placeholder
        d["hetero_gnn_score"] = [0.5] * n
        # train-only: healthy in train, zero in val/test (gene-disjoint)
        d["n_pathogenic_in_gene"] = [float(i % 7) for i in range(n)] if is_train else zeros()
        d["gene_has_known_disease"] = [float(i % 2) for i in range(n)] if is_train else zeros()
        for c in revived:      d[c] = varied()
        for c in now_expected: d[c] = zeros()
        for c in finngen:      d[c] = zeros()   # dead because --finngen-path not passed
        return pd.DataFrame(d)

    res = G.gate_frames({"X_train": frame(True), "X_val": frame(False)}, prep_only=True)
    assert res["verdict"] == "NO_GO"
    assert res["unexpected_degenerate"] == sorted(finngen), res["unexpected_degenerate"]
    assert res["core_degenerate"] == []           # gnn exempt; n_pathogenic train-healthy
    assert res["missing_core"] == []

    # and with FinnGen wired (revived), the same splits gate GO
    def frame_fg(is_train):
        f = frame(is_train)
        for c in finngen: f[c] = varied()
        return f
    res2 = G.gate_frames({"X_train": frame_fg(True), "X_val": frame_fg(False)}, prep_only=True)
    assert res2["verdict"] == "GO", res2
