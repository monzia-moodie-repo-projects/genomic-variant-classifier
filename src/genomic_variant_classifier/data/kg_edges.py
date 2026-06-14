"""
kg_edges.py - Knowledge-graph gene-gene edge connectors
=======================================================
Monzia Moodie

Produces per-relation (gene_a, gene_b) edge lists for the heterogeneous GNN
(models/hetero_gnn.py). Every knowledge graph we integrate has the same shape:
genes belong to SETS -- Reactome/KEGG pathways, GO terms, OMIM/ClinGen diseases
-- and two genes share an edge in that relation iff they co-occur in a set. So
the core is ONE co-membership primitive plus thin per-source parsers, rather than
five bespoke connectors.

Edge model: the co-membership PROJECTION (a clique per set) onto a single "gene"
node type, matching hetero_gnn.py's one-node-type / many-relation design. (A
bipartite gene+pathway graph with two node types is a future extension; the
projection is the clean v1.)

Robustness lives in co_membership_edges:
  - set-size explosion guard (a 1000-gene pathway is ~500k edges and low
    specificity) -> skip sets larger than max_set_size, with a loud warning;
  - restrict_to (cohort genes) drops out-of-cohort genes BEFORE pairing, keeping
    the graph relevant and small;
  - self-pairs excluded, duplicate undirected pairs de-duplicated (canonical
    a<b order), deterministic sorted output.

These read LOCAL files (GMT or a gene/group table). Data acquisition (downloading
the GMT/CSV) is a separate step; see KG_SOURCES for provenance.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def parse_gmt(path) -> dict[str, list[str]]:
    """Parse a GMT file (Reactome / KEGG / GO gene sets).

    Each line: set_name <TAB> description <TAB> gene1 <TAB> gene2 ...
    Returns {set_name: [unique genes]}. Blank/short (<3 field) lines are skipped
    with a warning; duplicate set names are merged; genes de-duplicated in order.
    """
    sets: dict[str, list[str]] = {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for ln, raw in enumerate(fh, 1):
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                logger.warning("GMT %s line %d has <3 tab fields; skipped.", p.name, ln)
                continue
            name = parts[0].strip()
            genes = [g.strip() for g in parts[2:] if g.strip()]
            if not name or not genes:
                continue
            sets.setdefault(name, [])
            sets[name].extend(genes)
    for name, genes in sets.items():
        sets[name] = list(dict.fromkeys(genes))  # de-dup, preserve order
    logger.info("parse_gmt %s: %d sets.", p.name, len(sets))
    return sets


def parse_gene_group_table(
    path,
    gene_col: str,
    group_col: str,
    *,
    delimiter: str = ",",
) -> dict[str, list[str]]:
    """Parse a delimited table mapping genes to groups (e.g. ClinGen gene->disease,
    OMIM gene->phenotype). Returns {group: [unique genes]}.

    Tolerant of leading metadata/preamble rows: the header is auto-located as the
    first row that contains BOTH gene_col and group_col (case-insensitive). This
    handles ClinGen's multi-row CSV preamble without a hand-counted skiprows.
    """
    p = Path(path)
    want_g = gene_col.lower().strip()
    want_grp = group_col.lower().strip()
    with p.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh, delimiter=delimiter))
    header_idx = None
    for i, row in enumerate(rows):
        lc = [c.lower().strip() for c in row]
        if want_g in lc and want_grp in lc:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(
            f"Could not find a header row containing both {gene_col!r} and "
            f"{group_col!r} in {p.name}."
        )
    header = [c.strip() for c in rows[header_idx]]
    lc = [c.lower().strip() for c in header]
    gi = lc.index(want_g)
    gri = lc.index(want_grp)
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows[header_idx + 1:]:
        if len(row) <= max(gi, gri):
            continue
        g = row[gi].strip()
        grp = row[gri].strip()
        if g and grp:
            groups[grp].append(g)
    out = {grp: list(dict.fromkeys(gl)) for grp, gl in groups.items()}
    logger.info("parse_gene_group_table %s: %d groups.", p.name, len(out))
    return out


# ---------------------------------------------------------------------------
# Co-membership primitive
# ---------------------------------------------------------------------------
def co_membership_edges(
    set_to_genes: Mapping[str, Iterable[str]],
    *,
    max_set_size: int = 200,
    restrict_to: Iterable[str] | None = None,
) -> list[tuple[str, str]]:
    """Undirected gene-gene edges for genes sharing a set.

    max_set_size  sets with more surviving genes than this are SKIPPED (warned):
                  they scale as n^2 and are usually low-specificity.
    restrict_to   if given, genes outside this set are dropped BEFORE pairing.

    Returns a sorted list of canonical (a, b) tuples with a < b, no self-pairs,
    no duplicates.
    """
    allow = set(restrict_to) if restrict_to is not None else None
    edges: set[tuple[str, str]] = set()
    n_big = 0
    skipped_edges = 0
    n_sets = 0
    for name, genes in set_to_genes.items():
        n_sets += 1
        gs = list(dict.fromkeys(g for g in genes if (allow is None or g in allow)))
        if len(gs) < 2:
            continue
        if len(gs) > max_set_size:
            n_big += 1
            skipped_edges += len(gs) * (len(gs) - 1) // 2
            continue
        for a, b in combinations(gs, 2):
            if a == b:
                continue
            edges.add((a, b) if a < b else (b, a))
    if n_big:
        logger.warning(
            "co_membership: skipped %d set(s) larger than max_set_size=%d "
            "(~%d edges suppressed).", n_big, max_set_size, skipped_edges,
        )
    logger.info(
        "co_membership: %d unique gene-gene edges from %d set(s).",
        len(edges), n_sets,
    )
    return sorted(edges)


# ---------------------------------------------------------------------------
# Per-source adapters
# ---------------------------------------------------------------------------
def reactome_edges(gmt_path, **kw) -> list[tuple[str, str]]:
    """Reactome shares_pathway edges from ReactomePathways.gmt."""
    return co_membership_edges(parse_gmt(gmt_path), **kw)


def kegg_edges(gmt_path, **kw) -> list[tuple[str, str]]:
    """KEGG shares_pathway edges from a KEGG pathway GMT (e.g. MSigDB c2.cp.kegg)."""
    return co_membership_edges(parse_gmt(gmt_path), **kw)


def go_edges(gmt_path, **kw) -> list[tuple[str, str]]:
    """GO shares_function edges from a GO GMT (e.g. MSigDB c5.go.bp)."""
    return co_membership_edges(parse_gmt(gmt_path), **kw)


def clingen_edges(
    csv_path,
    *,
    gene_col: str = "GENE SYMBOL",
    group_col: str = "DISEASE LABEL",
    **kw,
) -> list[tuple[str, str]]:
    """ClinGen shares_disease edges from the Gene-Disease Validity CSV."""
    return co_membership_edges(
        parse_gene_group_table(csv_path, gene_col, group_col), **kw
    )


def omim_edges(
    table_path,
    *,
    gene_col: str = "gene_symbol",
    group_col: str = "phenotype",
    delimiter: str = "\t",
    **kw,
) -> list[tuple[str, str]]:
    """OMIM shares_disease edges from a gene->phenotype table (license-restricted)."""
    return co_membership_edges(
        parse_gene_group_table(table_path, gene_col, group_col, delimiter=delimiter),
        **kw,
    )


# ---------------------------------------------------------------------------
# Provenance registry (for the wiring step + docs; not used at runtime)
# ---------------------------------------------------------------------------
KG_SOURCES = {
    "shares_pathway_reactome": {
        "relation": "shares_pathway",
        "adapter": "reactome_edges",
        "format": "gmt",
        "url": "https://reactome.org/download/current/ReactomePathways.gmt",
        "license": "CC0 (public)",
    },
    "shares_pathway_kegg": {
        "relation": "shares_pathway",
        "adapter": "kegg_edges",
        "format": "gmt",
        "url": "MSigDB c2.cp.kegg_medicus.*.symbols.gmt (registration)",
        "license": "KEGG academic; MSigDB terms apply",
    },
    "shares_function_go": {
        "relation": "shares_function",
        "adapter": "go_edges",
        "format": "gmt",
        "url": "MSigDB c5.go.bp.*.symbols.gmt (or goa_human.gaf)",
        "license": "GO CC-BY 4.0",
    },
    "shares_disease_clingen": {
        "relation": "shares_disease",
        "adapter": "clingen_edges",
        "format": "csv",
        "url": "https://search.clinicalgenome.org/kb/gene-validity/download",
        "license": "ClinGen (public)",
    },
    "shares_disease_omim": {
        "relation": "shares_disease",
        "adapter": "omim_edges",
        "format": "tsv",
        "url": "OMIM genemap2.txt (license required)",
        "license": "OMIM (restricted)",
    },
}
