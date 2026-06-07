"""Tests for genomic_variant_classifier.data.protein_coords."""

from __future__ import annotations

import gzip

import pandas as pd
import pytest

from genomic_variant_classifier.data.protein_coords import (
    ProteinCoordConnector,
    _norm_chrom,
)

_HEADER = (
    "#CHROM\tPOS\tREF\tALT\tgenome\tuniprot_id\ttranscript_id\t"
    "protein_variant\tam_pathogenicity\tam_class"
)


def _write_am(path, body_rows):
    rows = ["# Copyright", "#", "# license", _HEADER] + body_rows
    with gzip.open(path, "wt") as f:
        f.write("\n".join(rows) + "\n")


def _cohort():
    return pd.DataFrame(
        {
            "chrom": ["15", "11", "14", "7"],
            "pos": [84799209, 126277517, 31562125, 55249071],
            "ref": ["G", "A", "G", "C"],
            "alt": ["A", "G", "A", "T"],
            "consequence": ["missense_variant"] * 4,
        }
    )


def test_norm_chrom():
    got = list(_norm_chrom(pd.Series(["chr15", "15", "chrX", "MT", "CHR7"])))
    assert got == ["15", "15", "X", "MT", "7"]


def test_build_join_cache(tmp_path):
    am = tmp_path / "AlphaMissense_hg38.tsv.gz"
    _write_am(
        am,
        [
            "chr15\t84799209\tG\tA\thg38\tQ1\tENST1\tV2L\t0.29\tlikely_benign",
            "chr11\t126277517\tA\tG\thg38\tQ2\tENST2\tR45Q\t0.88\tpathogenic",
            "chr14\t31562125\tG\tA\thg38\tQ3\tENST3\tG12V\t0.91\tpathogenic",
            "chr1\t69094\tG\tT\thg38\tQ4\tENST4\tV2V\t0.10\tbenign",  # synonymous
        ],
    )
    pc = ProteinCoordConnector(alphamissense_file=str(am), cache_dir=str(tmp_path))
    out = pc.annotate_dataframe(_cohort())
    assert (out["protein_pos"].iloc[0], out["wt_aa"].iloc[0], out["mut_aa"].iloc[0]) == (2, "V", "L")
    assert (out["protein_pos"].iloc[1], out["wt_aa"].iloc[1], out["mut_aa"].iloc[1]) == (45, "R", "Q")
    assert out["protein_pos"].iloc[2] == 12
    assert pd.isna(out["protein_pos"].iloc[3])  # chr7 not in AM -> NA via left-merge
    assert pc.cache_path.exists()
    # cache reload path
    pc2 = ProteinCoordConnector(alphamissense_file=str(am), cache_dir=str(tmp_path))
    assert pc2.annotate_dataframe(_cohort())["protein_pos"].iloc[0] == 2


def test_missing_source_degrades_gracefully(tmp_path):
    # No AM file + no cache -> NO data source at all -> warn-and-stub (must NOT raise),
    # matching the sibling connectors (spliceai/dbnsfp warn-and-stub on a missing path).
    # Fail-loud is reserved for a *present* file that parses to garbage
    # (see test_parse_rate_guard). Returning the frame unchanged keeps the pre-10b
    # shape so the whole annotation pipeline cannot crash on boxes without the TSV.
    pc = ProteinCoordConnector(
        alphamissense_file=str(tmp_path / "nope.tsv.gz"), cache_dir=str(tmp_path)
    )
    out = pc.annotate_dataframe(_cohort())  # must not raise
    assert len(out) == 4
    # no data source -> connector leaves the frame's columns unchanged
    assert "protein_pos" not in out.columns


def test_missing_source_none_path_degrades_gracefully(tmp_path):
    # alphamissense_file=None (the AnnotationConfig() default used by the pipeline and
    # every unit test) must also warn-and-stub rather than raise.
    pc = ProteinCoordConnector(alphamissense_file=None, cache_dir=str(tmp_path))
    out = pc.annotate_dataframe(_cohort())  # must not raise
    assert len(out) == 4
    assert "protein_pos" not in out.columns


def test_parse_rate_guard(tmp_path):
    am = tmp_path / "AlphaMissense_hg38.tsv.gz"
    # A *present* file whose matched protein_variants are all garbage -> parse rate 0
    # -> fail loud (this is the genuine silent-zero / format-drift case).
    _write_am(
        am,
        [
            "chr15\t84799209\tG\tA\thg38\tQ1\tENST1\t???\t0.1\tx",
            "chr11\t126277517\tA\tG\thg38\tQ2\tENST2\tjunk\t0.1\tx",
            "chr14\t31562125\tG\tA\thg38\tQ3\tENST3\t..\t0.1\tx",
        ],
    )
    pc = ProteinCoordConnector(alphamissense_file=str(am), cache_dir=str(tmp_path))
    with pytest.raises(ValueError):
        pc.annotate_dataframe(_cohort())
