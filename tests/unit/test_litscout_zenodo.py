"""test_litscout_zenodo.py  --  Monzia Moodie

Zenodo fetch parse, journal allow-list relevance boost, and the broadened
PubMed/keyword/Zenodo config scope.
"""
from genomic_variant_classifier.agent_layer.agents.literature_scout_agent import (
    _journal_relevance_boost,
    _parse_zenodo_hit,
    _strip_html,
)
from genomic_variant_classifier.agent_layer import config as cfg

_HIT = {
    "id": 10876543,
    "doi": "10.5281/zenodo.10876543",
    "links": {"html": "https://zenodo.org/records/10876543"},
    "metadata": {
        "title": "DeepVariantNet",
        "publication_date": "2025-04-02",
        "description": "<p>Improved AUROC &amp; calibration.</p>",
        "creators": [{"name": "Moodie, Monzia"}, {"name": "Lee, K."}],
    },
}


def test_parse_zenodo_hit_full():
    p = _parse_zenodo_hit(_HIT)
    assert p["source"] == "Zenodo" and p["journal"] == "Zenodo"
    assert p["doi"] == "10.5281/zenodo.10876543"
    assert p["url"] == "https://zenodo.org/records/10876543"
    assert p["authors"] == "Moodie, Monzia; Lee, K."
    assert p["publication_date"] == "2025-04-02"
    assert p["abstract"] == "Improved AUROC & calibration."   # HTML stripped, entity decoded


def test_parse_zenodo_hit_degenerate():
    p = _parse_zenodo_hit({"id": 42})
    assert p["doi"] == "zenodo:42" and p["url"] == "" and p["authors"] == ""
    assert p["source"] == "Zenodo"


def test_strip_html_order_and_entities():
    assert _strip_html("<p>scores &lt;0.5&gt; here</p>") == "scores <0.5> here"
    assert _strip_html("<p>X.</p><p>Y.</p>") == "X. Y."


def test_journal_boost_allowlisted_only():
    b = cfg.LITERATURE_JOURNAL_BOOST
    assert _journal_relevance_boost("Nature Methods") == b
    assert _journal_relevance_boost("Briefings in Bioinformatics") == b
    assert _journal_relevance_boost("Journal of Natural Products") == 0.0   # 'natural' != 'nature'
    assert _journal_relevance_boost("bioRxiv") == 0.0
    assert _journal_relevance_boost("") == 0.0


def test_config_scope_broadened():
    assert len(cfg.LITERATURE_PUBMED_QUERIES) >= 19
    assert {"graph neural network", "knowledge graph", "self-supervised"} <= set(cfg.LITERATURE_RELEVANCE_KEYWORDS)
    assert len(cfg.LITERATURE_ZENODO_QUERIES) == 4
    assert cfg.ZENODO_API_BASE.startswith("https://zenodo.org")
    assert isinstance(cfg.LITERATURE_JOURNAL_ALLOWLIST, set) and len(cfg.LITERATURE_JOURNAL_ALLOWLIST) >= 15
