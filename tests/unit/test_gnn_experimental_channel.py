"""Regression test for the STRING experimental edge-channel fix in gnn.py.

Bug (fixed in commit 63a2fb7): StringDBGraph.build() referenced the edge channel
as "experiments" in both the _CHANNELS guard list and the add_edge() call, but
STRING's protein.links.detailed.v12.0 names that column "experimental". The guard
therefore zero-filled a phantom "experiments" column and add_edge read those zeros,
silently nulling the experimental channel (1 of 3 STRING confidence channels).

This test builds a tiny graph from synthetic *local* STRING files (the same code
path Run 15 exercises after download caches them as local parquet) and asserts the
experimental edge attribute is populated from the real column -- i.e. non-zero and
equal to experimental/1000. On pre-63a2fb7 code this assertion fails (0.0).

It also exercises the protein.info TSV parse (sep="\t") implicitly: the edge can
only be keyed on GENEA/GENEB if the protein->gene mapping was read correctly.

gnn.py imports torch_geometric at module scope, which segfaults (0xc0000139) on the
local CPU box; like tests/unit/test_ablate_gnn.py this test importorskips so it SKIPS
locally and RUNS on the GPU/VM where torch_geometric imports cleanly.

Author: Monzia Moodie
"""
from __future__ import annotations

import gzip
from pathlib import Path

import pytest

# gnn.py imports torch_geometric at module scope -> skip where it can't import.
pytest.importorskip("torch_geometric")

from genomic_variant_classifier.models.gnn import StringDBGraph  # noqa: E402


def _write_gz(path: Path, text: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(text)


# Real STRING protein.info is TAB-delimited with a space-containing annotation column.
_INFO = (
    "#string_protein_id\tpreferred_name\tprotein_size\tannotation\n"
    "9606.ENSP00000000001\tGENEA\t100\tsome protein with spaces in annotation\n"
    "9606.ENSP00000000002\tGENEB\t200\tanother annotated protein here\n"
)

# Real STRING protein.links.detailed is SPACE-delimited and the channel is
# "experimental" (NOT "experiments"). combined_score 950 >= threshold 700 -> kept.
_LINKS = (
    "protein1 protein2 neighborhood fusion cooccurence coexpression "
    "experimental database textmining combined_score\n"
    "9606.ENSP00000000001 9606.ENSP00000000002 0 0 0 700 900 800 0 950\n"
)


def _build_graph(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    info_path = tmp_path / "9606.protein.info.v12.0.txt.gz"
    links_path = tmp_path / "9606.protein.links.detailed.v12.0.txt.gz"
    _write_gz(info_path, _INFO)
    _write_gz(links_path, _LINKS)
    return StringDBGraph(
        cache_dir=cache_dir,
        combined_score_threshold=700,
        local_links_path=links_path,
        local_info_path=info_path,
    ).build()


def test_experimental_channel_populated_from_real_column(tmp_path: Path) -> None:
    g = _build_graph(tmp_path)

    # protein.info parsed as TSV -> proteins mapped to gene symbols (else nodes
    # would be raw 9606.ENSP* ids and this edge lookup would fail).
    assert g.has_edge("GENEA", "GENEB"), f"nodes seen: {sorted(g.nodes())}"

    attrs = g.get_edge_data("GENEA", "GENEB")

    # The whole point of 63a2fb7: experimental must reflect the real column (0.9),
    # not the zero-filled phantom "experiments" channel of the pre-fix code.
    assert attrs["experimental"] == pytest.approx(0.9), attrs
    assert attrs["database"] == pytest.approx(0.8), attrs
    assert attrs["coexpression"] == pytest.approx(0.7), attrs
    assert attrs["weight"] == pytest.approx(0.95), attrs


def test_no_phantom_experiments_attribute(tmp_path: Path) -> None:
    # Guards against a half-fix that adds "experimental" but leaves a stray
    # "experiments" edge attribute behind.
    g = _build_graph(tmp_path)
    attrs = g.get_edge_data("GENEA", "GENEB")
    assert "experiments" not in attrs, attrs
