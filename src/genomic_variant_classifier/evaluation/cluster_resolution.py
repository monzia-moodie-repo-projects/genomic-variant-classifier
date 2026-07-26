"""Resolve the gene-cluster labels a certified bootstrap requires.

WHY THIS MODULE EXISTS
======================
The certified confidence interval in this project resamples WHOLE GENES, not
individual variants, because variants within a gene share the gene's constraint,
its network position and its curation history. Resampling variants independently
understates variance; the gene-cluster design effect was measured at 2.935 times
on the real cohort (suite-size ratchet entry 2055) and at 2.807 times on a
synthetic fixture, meaning every previously published interval was roughly that
factor too narrow.

To resample genes, the evaluator needs a per-row gene label. Two different
columns carry one in this repository, and THEY ARE DIFFERENT NAMESPACES:

  * 'gene_id'      an Ensembl gene identifier (ENSG...), arriving from the
                   Genome Aggregation Database (gnomAD) GraphQL application
                   programming interface and from Genotype-Tissue Expression
                   (GTEx) matrices, whose 'Name' column is Ensembl.
  * 'gene_symbol'  a HUGO Gene Nomenclature Committee (HGNC) symbol such as
                   BRCA1, used by 413 references across roughly 110 files and
                   by gene-stratified splitting and the graph neural network.

Measured on 2026-07-26 against HEAD 2e04bd9:
  scripts/build_gtex_de_features.py:100-101 writes 'gene_id' from the GTEx
  matrix 'Name' column and 'gene_symbol' from its 'Description' column, in the
  same DataFrame constructor; and scripts/build_rnaseq_canonical_real.py:82
  explicitly EXCLUDES rows whose 'gene_symbol' begins with "ENSG", a filter that
  exists only because the project knows Ensembl identifiers are a different
  namespace that must not leak into the symbol column.

COMPARING THEM AS STRINGS IS THEREFORE INVALID. "ENSG00000012048" does not equal
"BRCA1", and a resolver that tested raw equality would refuse a perfectly valid
certified interval on every frame that supplied both columns.

WHAT IS COMPARED INSTEAD
------------------------
The bootstrap consumes only the GROUPING. 'cluster_bootstrap_ci' builds
'{label: row_positions}' and never interprets a label. Two labelings are
therefore interchangeable for this purpose if and only if they induce the SAME
PARTITION of rows -- a namespace-free property. This module tests exactly that
and nothing weaker.

NO SILENT PREFERENCE
--------------------
When both legacy columns are present the resolver does NOT quietly pick one. It
verifies partition equivalence and refuses when the partitions differ, because a
frame whose two gene columns disagree about which rows belong together does not
have one well-defined cluster structure, and choosing either would silently
select an inferential design.

Author: written for Monzia Moodie, 2026-07-26.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from genomic_variant_classifier.evaluation.capabilities import MetricStatus

logger = logging.getLogger(__name__)

__all__ = [
    "CLUSTER_ID_COLUMN",
    "LEGACY_CLUSTER_COLUMNS",
    "FINDING_IDENTIFIER_REQUIRED",
    "FINDING_IDENTIFIER_INCOMPLETE",
    "FINDING_PARTITIONS_DISAGREE",
    "SOURCE_CLUSTER_ID",
    "SOURCE_GENE_ID",
    "SOURCE_GENE_SYMBOL",
    "SOURCE_VERIFIED_GENE_PARTITION",
    "ClusterResolution",
    "partitions_equivalent",
    "resolve_gene_clusters",
]

# The canonical column an adapter should eventually supply directly. When it is
# present it wins outright: an explicit declaration is never second-guessed
# against a legacy column.
CLUSTER_ID_COLUMN = "cluster_id"

# The two legacy columns, in the order they are reported. Order does NOT encode
# precedence when both are present -- see resolve_gene_clusters.
LEGACY_CLUSTER_COLUMNS = ("gene_id", "gene_symbol")

# Machine-readable findings. Strings, matching the convention in capabilities.py,
# so they can be grepped and compared without importing this module.
FINDING_IDENTIFIER_REQUIRED = "gene_cluster_identifier_required"
FINDING_IDENTIFIER_INCOMPLETE = "gene_cluster_identifier_incomplete"
FINDING_PARTITIONS_DISAGREE = "gene_cluster_partitions_disagree"

# Provenance tokens. These distinguish an explicit canonical column, a single
# legacy column, and two independently supplied columns proven equivalent --
# a distinction that would be lost if dual-column provenance reused
# SOURCE_CLUSTER_ID.
SOURCE_CLUSTER_ID = "cluster_id"
SOURCE_GENE_ID = "gene_id"
SOURCE_GENE_SYMBOL = "gene_symbol"
SOURCE_VERIFIED_GENE_PARTITION = "gene_id+gene_symbol"


@dataclass(frozen=True)
class ClusterResolution:
    """The cluster labels, where they came from, and whether they are usable.

    'values' is None whenever 'status' is not OK. There is no partially usable
    state: either a well-defined cluster structure was resolved, or the caller
    is told why one was not, and no certified interval is produced.
    """

    values: Optional[np.ndarray]
    source: Optional[str]
    partition_verified: bool
    status: MetricStatus
    finding: Optional[str]
    n_rows: int = 0
    n_clusters: Optional[int] = None
    n_missing: int = 0

    @property
    def usable(self) -> bool:
        """Whether these labels may drive a gene-cluster bootstrap."""
        return self.status is MetricStatus.OK and self.values is not None


def _as_object_series(values: Sequence) -> pd.Series:
    """Normalise any sequence to an object-dtype Series with a clean index.

    Object dtype is deliberate: gene labels are strings, and coercing them
    through a numeric dtype would silently mangle identifiers that happen to
    look numeric. The index is reset so two Series built from differently
    indexed frames still align positionally.
    """
    s = pd.Series(list(values), dtype="object")
    return s.reset_index(drop=True)


def _is_missing(s: pd.Series) -> np.ndarray:
    """Missingness mask that also treats the empty string as absent.

    pandas.isna does not consider "" missing, but an empty gene label carries no
    grouping information and must not become a cluster of its own. Treating it
    as present would silently pool every unlabelled row into one enormous
    pseudo-gene, which is the exact opposite of a conservative design.
    """
    na = pd.isna(s).to_numpy()
    empty = s.map(lambda v: isinstance(v, str) and v.strip() == "").to_numpy(dtype=bool)
    return na | empty


def partitions_equivalent(a: Sequence, b: Sequence) -> bool:
    """Whether two labelings induce the SAME partition of rows.

    This is namespace-free. It asks whether the map from an 'a' label to a 'b'
    label is a bijection over the observed rows, not whether the labels match.
    So these are equivalent:

        a: ENSG1 ENSG1 ENSG2 ENSG2
        b: BRCA1 BRCA1 TP53  TP53

    and these are not, because one 'a' label spans two 'b' groups:

        a: ENSG1 ENSG1
        b: BRCA1 TP53

    Missingness is significant: the two labelings must be missing on exactly the
    same rows, because a row one column can cluster and the other cannot is a
    genuine disagreement about the cluster structure.

    Complexity is O(n) in factorisation plus one sort, against O(n log n) for
    the equivalent groupby-nunique formulation. 'test_cluster_resolution.py'
    cross-checks this implementation against that reference formulation on
    randomised inputs, so the faster path cannot drift from the clearer one.
    """
    sa = _as_object_series(a)
    sb = _as_object_series(b)
    if len(sa) != len(sb):
        return False
    if len(sa) == 0:
        return True

    miss_a = _is_missing(sa)
    miss_b = _is_missing(sb)
    if not np.array_equal(miss_a, miss_b):
        return False

    keep = ~miss_a
    if not keep.any():
        # Both are missing everywhere. They agree, vacuously, that there is no
        # cluster structure at all. resolve_gene_clusters refuses this case
        # separately; equivalence itself is true.
        return True

    codes_a, uniq_a = pd.factorize(sa[keep], use_na_sentinel=False)
    codes_b, uniq_b = pd.factorize(sb[keep], use_na_sentinel=False)
    n_a = len(uniq_a)
    n_b = len(uniq_b)

    # Encode each (a, b) pair as a single integer, then count distinct pairs.
    # A bijection exists exactly when the number of distinct pairs equals the
    # number of distinct labels on BOTH sides: more pairs than 'a' labels means
    # some 'a' label was split; more pairs than 'b' labels means some 'b' label
    # was split.
    pair_codes = codes_a.astype(np.int64) * np.int64(n_b) + codes_b.astype(np.int64)
    n_pairs = int(np.unique(pair_codes).size)
    return n_pairs == n_a and n_pairs == n_b


def _resolved(values: pd.Series, source: str, *, partition_verified: bool) -> ClusterResolution:
    """Build a resolution from one already-chosen label column, refusing gaps."""
    missing = _is_missing(values)
    n_missing = int(missing.sum())
    n_rows = int(len(values))

    if n_missing == n_rows:
        return ClusterResolution(
            values=None, source=source, partition_verified=False,
            status=MetricStatus.INSUFFICIENT_SUPPORT,
            finding=FINDING_IDENTIFIER_REQUIRED,
            n_rows=n_rows, n_clusters=None, n_missing=n_missing,
        )

    if n_missing:
        # A row whose gene is unknown cannot be assigned to a cluster. Pooling
        # such rows under a single missing label would resample them together as
        # one pseudo-gene; giving each its own label would resample them
        # individually, which is the anti-conservative row-level design this
        # commit exists to stop happening by accident. Both are silent changes
        # to the inferential design, so neither is taken.
        return ClusterResolution(
            values=None, source=source, partition_verified=False,
            status=MetricStatus.INSUFFICIENT_SUPPORT,
            finding=FINDING_IDENTIFIER_INCOMPLETE,
            n_rows=n_rows, n_clusters=None, n_missing=n_missing,
        )

    arr = np.asarray(values.to_numpy(), dtype=object)
    return ClusterResolution(
        values=arr, source=source, partition_verified=partition_verified,
        status=MetricStatus.OK, finding=None,
        n_rows=n_rows, n_clusters=int(np.unique(arr).size), n_missing=0,
    )


def resolve_gene_clusters(meta: Optional[pd.DataFrame]) -> ClusterResolution:
    """Resolve per-row gene-cluster labels from an evaluator metadata frame.

    Contract, in order:

      'cluster_id' present
          used directly; source is 'cluster_id'; partition_verified is False,
          because nothing was cross-checked -- the column was declared, not
          derived.

      exactly one of 'gene_id' / 'gene_symbol'
          used; source names the column; partition_verified is False.

      both legacy columns
          their induced partitions are compared. Equivalent: the resolution
          succeeds with source 'gene_id+gene_symbol' and partition_verified
          True. Divergent: MetricStatus.FAILED with
          'gene_cluster_partitions_disagree', and NO interval is produced.

      neither
          MetricStatus.INSUFFICIENT_SUPPORT with
          'gene_cluster_identifier_required'. Point metrics are unaffected;
          only the certified interval is withheld.
    """
    if meta is None or len(getattr(meta, "columns", ())) == 0:
        return ClusterResolution(
            values=None, source=None, partition_verified=False,
            status=MetricStatus.INSUFFICIENT_SUPPORT,
            finding=FINDING_IDENTIFIER_REQUIRED,
            n_rows=0 if meta is None else int(len(meta)),
        )

    columns = set(meta.columns)

    if CLUSTER_ID_COLUMN in columns:
        return _resolved(_as_object_series(meta[CLUSTER_ID_COLUMN]),
                         SOURCE_CLUSTER_ID, partition_verified=False)

    has_id = LEGACY_CLUSTER_COLUMNS[0] in columns
    has_symbol = LEGACY_CLUSTER_COLUMNS[1] in columns

    if not has_id and not has_symbol:
        return ClusterResolution(
            values=None, source=None, partition_verified=False,
            status=MetricStatus.INSUFFICIENT_SUPPORT,
            finding=FINDING_IDENTIFIER_REQUIRED,
            n_rows=int(len(meta)),
        )

    if has_id and not has_symbol:
        return _resolved(_as_object_series(meta[LEGACY_CLUSTER_COLUMNS[0]]),
                         SOURCE_GENE_ID, partition_verified=False)

    if has_symbol and not has_id:
        return _resolved(_as_object_series(meta[LEGACY_CLUSTER_COLUMNS[1]]),
                         SOURCE_GENE_SYMBOL, partition_verified=False)

    # Both present. Compare the partitions they induce, never the strings.
    gene_id = _as_object_series(meta[LEGACY_CLUSTER_COLUMNS[0]])
    gene_symbol = _as_object_series(meta[LEGACY_CLUSTER_COLUMNS[1]])

    if not partitions_equivalent(gene_id, gene_symbol):
        logger.warning(
            "gene_id and gene_symbol induce different row partitions; the "
            "certified bootstrap is withheld rather than silently choosing one."
        )
        return ClusterResolution(
            values=None, source=None, partition_verified=False,
            status=MetricStatus.FAILED,
            finding=FINDING_PARTITIONS_DISAGREE,
            n_rows=int(len(meta)),
        )

    # Equivalent. Either column produces the same clusters; gene_id is carried
    # because it is the finer-grained identifier, but the provenance records
    # that BOTH were supplied and verified rather than that one was chosen.
    return _resolved(gene_id, SOURCE_VERIFIED_GENE_PARTITION, partition_verified=True)
