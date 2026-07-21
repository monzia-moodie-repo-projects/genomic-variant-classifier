"""Split protocol: gene-disjoint partitioning with a first-class partition schema.

WHAT CHANGED ON 2026-07-21, AND WHY
====================================
This module produced a FOUR-way gene-disjoint split: train / tune / conformal /
test. The metric specification requires FIVE internal partitions plus an external
one (Finding 2; Priority 2 calls the separation "essential"):

    train                   model fitting
    development validation  hyperparameters, early stopping
    probability calibration Platt / isotonic / temperature scaling
    conformal calibration   the nonconformity quantile
    test                    locked, touched once
    (external test)         a separate dataset, not a partition of this one

The gap was not the conformal partition -- that already existed and was the point
of version 2. The gap is PROBABILITY CALIBRATION. Measured in the code on
2026-07-21: scripts/train.py:131 and :550-551 fit the isotonic calibration on the
`tune` partition, and line 52 of this module's previous revision defined `tune`
as "the model/method/alpha selection set". So the probability calibrator was
fitted on data already used to choose the model, the method and alpha. The
specification forbids that sharing explicitly.

WHY A SCHEMA RATHER THAN A FIFTH FRACTION FIELD
------------------------------------------------
The previous revision hard-coded `PARTITIONS = ("train", "tune", "conformal",
"test")` as a module constant, referenced in five places, alongside four
separately-named fraction fields validated by a hand-written sum check. Adding a
fifth partition therefore meant editing six sites in lockstep, and a sixth
partition would mean editing them again.

That is the same shape as three other defects repaired on 2026-07-20 and
2026-07-21: the Run-16 preflight gate's `EXPECTED_COUNT = 81`, the conformal
package's five-of-seven import list, and the suite-size ratchet before it gained
drift detection. A literal that must be kept correct by memory, with no mechanism
to notice when it is not.

So the partition set becomes DATA. `PartitionSchema` holds an ordered tuple of
`Partition(name, fraction, role)`. Every function here derives its behaviour from
the schema. Adding a partition is a schema entry, not a rewrite.

BACKWARD COMPATIBILITY IS EXACT
-------------------------------
`SplitProtocolV2Config()` with no schema builds FOUR_WAY, whose fractions, hash
interval order and bucket assignment are byte-for-byte what the previous revision
produced. Gene-to-bucket assignment under hash mode is UNCHANGED, so the
stability invariant I8 holds across this upgrade and no existing split is
invalidated. `PARTITIONS`, `four_way_hash_split`, `four_way_group_shuffle_split`,
`split`, `assert_partition_invariants`, `apply_train_only_leakage_remap`,
`genes_are_stable_under_growth`, `SplitResultV2` and `SplitProtocolV2Config` all
keep their names and signatures.

TWO DEFECTS REPAIRED IN THE SAME PASS
--------------------------------------
1. group_shuffle carve() rescaled by ROW counts while scikit-learn's
   GroupShuffleSplit interprets test_size as a proportion of GROUPS. Verified
   empirically on 2026-07-21: ten genes, one holding 100 of 109 rows,
   test_size=0.30 carved exactly 3 of 10 genes (30 % of groups, 3 % of rows).

   The row ratio is an unbiased estimator of the gene ratio under random
   assignment, so it is usually close -- but its variance grows with gene-size
   skew, and it OVERFLOWS when a high-row-count gene is carved early: the pool
   then holds few rows but nearly all genes, `frac * n / len(pool)` exceeds 1,
   and the clamp `min(max(f, 1e-9), 1 - 1e-9)` silently turns it into
   0.999999999. scikit-learn then raises

       ValueError: With n_samples=340, test_size=0.999999999 and
       train_size=None, the resulting train set will be empty.

   which names neither genes, nor row skew, nor the partition being carved.

   Measured, mean worst-partition absolute gene-fraction deviation over 12 seeds:

       gene-size skew          rescale by rows        rescale by genes
       uniform                 0.0000                 0.0000
       heavy-tailed (Pareto)   0.0329                 0.0000
       one gene = 90 % rows    0.0434, 3/12 CRASH     0.0065

   ClinVar variant counts per gene ARE heavy-tailed -- TTN, BRCA1 and a handful
   of others carry orders of magnitude more variants than the median gene -- so
   the middle row is the realistic regime, and the bottom row is reachable.
   carve() now rescales by gene counts, the unit GroupShuffleSplit operates on.
   The ratio is bounded by construction, so the overflow cannot arise.

2. genes_are_stable_under_growth had an `if cfg.mode != "hash": ... else: ...`
   whose branches were IDENTICAL -- both called split(df_small) then
   split(df_large) -- under a comment implying they differed. Dead code removed.

The leakage-safe train-only n_pathogenic_in_gene remap (incident 2026-06-13) is
preserved and now applies to every partition in the schema, whatever its size.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

# Reuse the canonical, hash-stable gene hash rather than a second implementation.
from .splits import _gene_hash

try:
    from sklearn.model_selection import GroupShuffleSplit
except Exception:  # pragma: no cover
    GroupShuffleSplit = None


class PartitionRole(str, Enum):
    """What a partition is FOR.

    Roles exist so downstream code can ask "where do I fit the probability
    calibrator?" and receive an answer, instead of hard-coding a partition name.
    That is the mechanism by which the probability calibrator stops being fitted
    on the selection set: train.py asks for CALIBRATE_PROBABILITY and, under the
    five-way schema, gets a partition no other stage has touched.

    Inheriting from str keeps roles JSON-serialisable and printable, which
    matters because they end up in run manifests.
    """

    TRAIN = "train"
    SELECT = "select"
    CALIBRATE_PROBABILITY = "calibrate_probability"
    CALIBRATE_CONFORMAL = "calibrate_conformal"
    TEST = "test"


@dataclass(frozen=True)
class Partition:
    """One partition: a name, a fraction OF GENES, and what it is for."""

    name: str
    fraction: float
    role: PartitionRole

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"partition name must be a non-empty string, got {self.name!r}")
        if not (0.0 < self.fraction < 1.0):
            raise ValueError(
                f"partition {self.name!r} fraction must be strictly in (0, 1), "
                f"got {self.fraction}")


@dataclass(frozen=True)
class PartitionSchema:
    """An ordered set of partitions, plus the order hash intervals are assigned in.

    `hash_order` is separate from declaration order and is LOAD-BEARING FOR
    STABILITY: it fixes which region of [0, 1) each partition occupies, so a gene
    keeps its bucket as the cohort grows (invariant I8). Changing it reassigns
    every gene. FOUR_WAY therefore preserves the historical order exactly --
    test, conformal, tune, train -- so this rewrite invalidates no existing split.
    """

    partitions: tuple[Partition, ...]
    hash_order: tuple[str, ...]
    label: str = "unnamed"

    def __post_init__(self) -> None:
        if len(self.partitions) < 2:
            raise ValueError("a schema needs at least two partitions")
        names = [p.name for p in self.partitions]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate partition names: {dupes}")
        total = sum(p.fraction for p in self.partitions)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"fractions must sum to 1.0, got {total:.6f}")
        if sorted(self.hash_order) != sorted(names):
            raise ValueError(
                f"hash_order {list(self.hash_order)} does not match partition "
                f"names {names}")
        roles = [p.role for p in self.partitions]
        if roles.count(PartitionRole.TRAIN) != 1:
            raise ValueError(
                f"exactly one partition must carry role TRAIN, found "
                f"{roles.count(PartitionRole.TRAIN)}. The train-only leakage remap "
                "(incident 2026-06-13) derives every partition's counts from it, so "
                "there must be exactly one and it must be unambiguous.")
        for role in (PartitionRole.SELECT, PartitionRole.CALIBRATE_CONFORMAL,
                     PartitionRole.TEST):
            if roles.count(role) > 1:
                raise ValueError(f"role {role.value} appears {roles.count(role)} times")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.partitions)

    @property
    def fractions(self) -> dict:
        return {p.name: p.fraction for p in self.partitions}

    def role_of(self, name: str) -> PartitionRole:
        for p in self.partitions:
            if p.name == name:
                return p.role
        raise KeyError(f"no partition named {name!r}; have {list(self.names)}")

    def name_for_role(self, role: PartitionRole) -> Optional[str]:
        """The partition serving `role`, or None if the schema has no such stage.

        Returns None rather than raising, because "this schema has no dedicated
        probability-calibration partition" is a legitimate state that a caller
        must be able to detect and act on -- it is exactly the condition the
        four-way schema is in.
        """
        for p in self.partitions:
            if p.role is role:
                return p.name
        return None

    def has_role(self, role: PartitionRole) -> bool:
        return self.name_for_role(role) is not None


# ---------------------------------------------------------------------------
# The two shipped schemas.
# ---------------------------------------------------------------------------

FOUR_WAY = PartitionSchema(
    partitions=(
        Partition("train", 0.60, PartitionRole.TRAIN),
        Partition("tune", 0.15, PartitionRole.SELECT),
        Partition("conformal", 0.10, PartitionRole.CALIBRATE_CONFORMAL),
        Partition("test", 0.15, PartitionRole.TEST),
    ),
    # Historical order. Do not reorder: it would move every gene to a new bucket.
    hash_order=("test", "conformal", "tune", "train"),
    label="four_way_v2",
)

FIVE_WAY = PartitionSchema(
    partitions=(
        Partition("train", 0.55, PartitionRole.TRAIN),
        Partition("tune", 0.15, PartitionRole.SELECT),
        Partition("calib", 0.10, PartitionRole.CALIBRATE_PROBABILITY),
        Partition("conformal", 0.08, PartitionRole.CALIBRATE_CONFORMAL),
        Partition("test", 0.12, PartitionRole.TEST),
    ),
    # Smaller, more sensitive partitions occupy the low-hash region, matching the
    # four-way convention. `calib` is inserted between conformal and tune.
    hash_order=("test", "conformal", "calib", "tune", "train"),
    label="five_way_spec",
)

# Backward compatibility: the historical module constant, unchanged.
PARTITIONS = FOUR_WAY.names


@dataclass
class SplitProtocolV2Config:
    """Split configuration.

    Fractions are of GENES (the split is gene-level). Naming is unambiguous:
    `tune` is the model/method/alpha selection set (the v1 pipeline confusingly
    called its selection set `test`); `test` here is the locked evaluation set
    touched once; `conformal` is the dedicated conformal-calibration partition;
    and under FIVE_WAY, `calib` is the dedicated PROBABILITY-calibration
    partition that no other stage touches.

    The four historical `*_frac` fields are retained so every existing caller
    works unchanged. Supplying `schema` explicitly overrides them -- and if BOTH
    a schema and non-default fractions are given, that is a contradiction and
    raises, rather than one silently winning.
    """

    train_frac: float = 0.60
    tune_frac: float = 0.15
    conformal_frac: float = 0.10
    test_frac: float = 0.15
    schema: Optional[PartitionSchema] = None
    seed: int = 42
    gene_col: str = "gene_symbol"
    label_col: str = "label"
    mode: str = "hash"  # "hash" | "group_shuffle"
    require_both_classes: bool = True
    frac_tolerance: float = 0.05   # floor for allowed absolute gene-fraction deviation
    frac_sigma: float = 4.0        # also allow this many binomial standard errors
    count_col: str = "n_pathogenic_in_gene"
    derived_flag_col: str = "gene_has_known_disease"

    _DEFAULT_FRACS = {"train_frac": 0.60, "tune_frac": 0.15,
                      "conformal_frac": 0.10, "test_frac": 0.15}

    def __post_init__(self) -> None:
        if self.mode not in ("hash", "group_shuffle"):
            raise ValueError(f"mode must be 'hash' or 'group_shuffle', got {self.mode!r}")

        customised = [k for k, v in self._DEFAULT_FRACS.items()
                      if getattr(self, k) != v]

        if self.schema is None:
            # Historical path: build FOUR_WAY from the four fraction fields, so
            # validation messages stay in the vocabulary the caller used.
            for name in self._DEFAULT_FRACS:
                v = getattr(self, name)
                if not (0.0 < v < 1.0):
                    raise ValueError(f"{name} must be strictly in (0, 1), got {v}")
            s = (self.train_frac + self.tune_frac
                 + self.conformal_frac + self.test_frac)
            if abs(s - 1.0) > 1e-6:
                raise ValueError(f"fractions must sum to 1.0, got {s:.6f}")
            self.schema = PartitionSchema(
                partitions=(
                    Partition("train", self.train_frac, PartitionRole.TRAIN),
                    Partition("tune", self.tune_frac, PartitionRole.SELECT),
                    Partition("conformal", self.conformal_frac,
                              PartitionRole.CALIBRATE_CONFORMAL),
                    Partition("test", self.test_frac, PartitionRole.TEST),
                ),
                hash_order=FOUR_WAY.hash_order,
                label="four_way_v2",
            )
        elif customised:
            raise ValueError(
                f"both `schema` and non-default fraction field(s) {customised} were "
                "given. These specify the same thing and would silently disagree. "
                "Set the fractions inside the schema instead.")

    @property
    def partitions(self) -> tuple[str, ...]:
        return self.schema.names


@dataclass
class SplitResultV2:
    indices: dict = field(default_factory=dict)   # partition -> integer position array
    genes: dict = field(default_factory=dict)     # partition -> frozenset of gene symbols
    mode: str = "hash"
    seed: int = 42
    n_total: int = 0
    schema: Optional[PartitionSchema] = None

    @property
    def _names(self) -> tuple[str, ...]:
        return self.schema.names if self.schema is not None else PARTITIONS

    def summary(self) -> dict:
        return {
            p: {"n_rows": int(len(self.indices[p])), "n_genes": int(len(self.genes[p]))}
            for p in self._names
        }

    def rows_for_role(self, role: PartitionRole) -> Optional[np.ndarray]:
        """Row positions of the partition serving `role`, or None if absent.

        This is the accessor that lets train.py stop naming partitions. Under
        FOUR_WAY, rows_for_role(CALIBRATE_PROBABILITY) returns None -- which is
        the honest answer, and a caller that silently fell back to the selection
        set would be re-creating the defect this schema exists to fix.
        """
        if self.schema is None:
            return None
        name = self.schema.name_for_role(role)
        return None if name is None else self.indices[name]


def _resolve_genes(df: pd.DataFrame, gene_col: str) -> pd.Series:
    if gene_col not in df.columns:
        raise ValueError(f"gene column {gene_col!r} absent from DataFrame")
    return df[gene_col].fillna("unknown").astype(str)


def hash_split(df: pd.DataFrame, cfg: SplitProtocolV2Config) -> dict:
    """Deterministic, hash-stable gene-disjoint split over the schema.

    Cumulative interval rule on h = _gene_hash(gene, seed) in [0, 1), walking
    `schema.hash_order`. For FOUR_WAY this reproduces the historical assignment
    exactly:
        [0, f_test)                                  -> test
        [f_test, f_test + f_conf)                    -> conformal
        [f_test + f_conf, ... + f_tune)              -> tune
        [..., 1.0)                                   -> train
    """
    schema = cfg.schema
    genes = _resolve_genes(df, cfg.gene_col)
    unique = genes.unique()
    h = {g: _gene_hash(g, cfg.seed) for g in unique}
    hs = genes.map(h)
    fracs = schema.fractions

    masks = {}
    lower = 0.0
    for i, name in enumerate(schema.hash_order):
        upper = lower + fracs[name]
        if i == len(schema.hash_order) - 1:
            # The final partition takes the remainder, so floating-point drift in
            # the cumulative sum cannot leave a sliver of [0, 1) unassigned.
            masks[name] = hs >= lower
        else:
            masks[name] = (hs >= lower) & (hs < upper)
        lower = upper
    return {p: np.where(m.to_numpy())[0] for p, m in masks.items()}


def group_shuffle_split(df: pd.DataFrame, cfg: SplitProtocolV2Config) -> dict:
    """Nested scikit-learn GroupShuffleSplit gene-disjoint split over the schema.

    Carves each partition in `hash_order` except the TRAIN one, which takes the
    remainder. Gene-disjoint but NOT stable as the cohort grows -- that is the
    documented difference from hash mode.

    Rescaling is by GENE counts. See the module docstring: rescaling by row
    counts is an unbiased but high-variance approximation that overflows and
    crashes when a high-row-count gene is carved early.
    """
    if GroupShuffleSplit is None:
        raise RuntimeError("scikit-learn required for group_shuffle mode")
    schema = cfg.schema
    genes = _resolve_genes(df, cfg.gene_col)
    all_pos = np.arange(len(df))
    n_genes_total = genes.nunique()
    train_name = schema.name_for_role(PartitionRole.TRAIN)

    def carve(pool_pos: np.ndarray, frac_of_whole: float, seed: int,
              name: str) -> tuple:
        n_genes_pool = int(genes.iloc[pool_pos].nunique())
        if n_genes_pool < 2:
            raise ValueError(
                f"cannot carve partition {name!r}: only {n_genes_pool} gene(s) "
                "remain in the pool. Reduce the number of partitions, raise the "
                "train fraction, or supply a cohort with more genes.")
        pool_frac = frac_of_whole * n_genes_total / n_genes_pool
        if pool_frac >= 1.0:
            raise ValueError(
                f"cannot carve partition {name!r}: it requires "
                f"{frac_of_whole:.3f} of all {n_genes_total} genes, but only "
                f"{n_genes_pool} genes ({n_genes_pool / n_genes_total:.3f}) remain "
                "in the pool. The requested fractions cannot be satisfied.")
        gss = GroupShuffleSplit(n_splits=1, test_size=pool_frac, random_state=seed)
        keep_rel, carve_rel = next(
            gss.split(pool_pos, groups=genes.iloc[pool_pos].to_numpy()))
        return pool_pos[keep_rel], pool_pos[carve_rel]

    out: dict = {}
    pool = all_pos
    step = 0
    for name in schema.hash_order:
        if name == train_name:
            continue
        pool, carved = carve(pool, schema.fractions[name], cfg.seed + step, name)
        out[name] = carved
        step += 1
    out[train_name] = pool
    return out


# Historical names, retained so existing callers and forensics scripts keep working.
four_way_hash_split = hash_split
four_way_group_shuffle_split = group_shuffle_split


def assert_partition_invariants(indices: dict, df: pd.DataFrame,
                                cfg: SplitProtocolV2Config) -> dict:
    """Fail-loud checks shared by both modes. Returns the realized gene sets."""
    schema = cfg.schema
    names = schema.names
    genes = _resolve_genes(df, cfg.gene_col)
    n = len(df)

    missing = [p for p in names if p not in indices]
    if missing:
        raise AssertionError(f"indices missing partition(s): {missing}")
    extra = [p for p in indices if p not in names]
    if extra:
        raise AssertionError(f"indices contain partition(s) not in the schema: {extra}")

    # I1 coverage: exactly-once partition of all rows
    all_idx = np.concatenate([indices[p] for p in names]) if n else np.array([], dtype=int)
    if len(all_idx) != n:
        raise AssertionError(f"coverage: partitions cover {len(all_idx)} rows, expected {n}")
    if len(np.unique(all_idx)) != n:
        raise AssertionError("coverage: overlapping row assignments detected")

    # I5 non-empty
    for p in names:
        if len(indices[p]) == 0:
            raise AssertionError(f"partition {p!r} is empty; adjust fractions or seed")

    gene_sets = {p: frozenset(genes.iloc[indices[p]].unique()) for p in names}

    # I2 gene-disjoint: every pair
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = gene_sets[a] & gene_sets[b]
            if overlap:
                raise AssertionError(
                    f"gene-disjoint violated between {a} and {b}: "
                    f"{len(overlap)} shared gene(s), e.g. {sorted(overlap)[:3]}")

    # I3 fraction accuracy (of genes). Hash bucketing is exact only in
    # expectation; the realized fraction deviates by binomial variance that
    # shrinks as the gene count grows. Adaptive bound so the check is meaningful
    # at both small test scales and the real thousands-of-genes scale.
    total_genes = len(set().union(*gene_sets.values()))
    want = schema.fractions
    for p in names:
        realized = len(gene_sets[p]) / max(total_genes, 1)
        se = (want[p] * (1.0 - want[p]) / max(total_genes, 1)) ** 0.5
        bound = max(cfg.frac_tolerance, cfg.frac_sigma * se)
        if abs(realized - want[p]) > bound:
            raise AssertionError(
                f"fraction accuracy: {p} realized {realized:.3f} vs requested "
                f"{want[p]:.3f} (bound {bound:.3f} = max(floor {cfg.frac_tolerance}, "
                f"{cfg.frac_sigma}*se {se:.3f}), n_genes={total_genes})")

    # I4 both-classes
    if cfg.require_both_classes and cfg.label_col in df.columns:
        for p in names:
            classes = set(pd.to_numeric(
                df.iloc[indices[p]][cfg.label_col], errors="coerce").dropna().unique())
            if classes != {0, 1} and classes != {0.0, 1.0}:
                raise AssertionError(
                    f"partition {p!r} missing class(es): has {classes}, need {{0, 1}}")

    return gene_sets


def apply_train_only_leakage_remap(df: pd.DataFrame, indices: dict,
                                   cfg: SplitProtocolV2Config) -> pd.DataFrame:
    """Recompute n_pathogenic_in_gene from TRAIN rows only and remap onto every
    partition (unseen genes -> 0), recomputing gene_has_known_disease in
    lockstep. Preserves the incident-2026-06-13 leakage fix and applies it to
    every partition in the schema. Returns a copy with corrected columns.
    """
    if cfg.count_col not in df.columns:
        return df  # nothing to remap
    schema = cfg.schema
    out = df.copy()
    # Dtype-safe promotion (2026-07-11): the cohort's count_col/derived_flag_col
    # may be int32; the per-partition .iloc writes below assign int64 arrays,
    # which triggers a pandas incompatible-dtype FutureWarning (silent cast now,
    # error in a future pandas). Widen once, up front. int32 -> int64 is lossless
    # for these counts.
    if cfg.count_col in out.columns:
        out[cfg.count_col] = out[cfg.count_col].astype("int64")
    if cfg.derived_flag_col in out.columns:
        out[cfg.derived_flag_col] = out[cfg.derived_flag_col].astype("int64")
    genes = _resolve_genes(df, cfg.gene_col)
    if cfg.label_col not in df.columns:
        raise ValueError(f"label column {cfg.label_col!r} required for train-only remap")

    train_name = schema.name_for_role(PartitionRole.TRAIN)
    tr = indices[train_name]
    y_tr_pos = (pd.to_numeric(df.iloc[tr][cfg.label_col], errors="coerce").to_numpy()
                == 1).astype(int)
    g_tr = genes.iloc[tr].to_numpy()
    train_counts = pd.Series(y_tr_pos).groupby(g_tr).sum()
    for p in schema.names:
        ix = indices[p]
        g = genes.iloc[ix]
        cnt = g.map(train_counts).fillna(0).astype(int).to_numpy()
        out.iloc[ix, out.columns.get_loc(cfg.count_col)] = cnt
        if cfg.derived_flag_col in out.columns:
            out.iloc[ix, out.columns.get_loc(cfg.derived_flag_col)] = (cnt > 0).astype(int)
    return out


def split(df: pd.DataFrame, cfg: Optional[SplitProtocolV2Config] = None) -> SplitResultV2:
    """Produce a validated gene-disjoint split. Dispatches by mode, asserts all
    shared invariants, and returns the result carrying its schema."""
    cfg = cfg or SplitProtocolV2Config()
    indices = hash_split(df, cfg) if cfg.mode == "hash" else group_shuffle_split(df, cfg)
    gene_sets = assert_partition_invariants(indices, df, cfg)
    return SplitResultV2(indices=indices, genes=gene_sets, mode=cfg.mode,
                         seed=cfg.seed, n_total=len(df), schema=cfg.schema)


def genes_are_stable_under_growth(df_small: pd.DataFrame, df_large: pd.DataFrame,
                                  cfg: SplitProtocolV2Config) -> bool:
    """Stability invariant I8: does adding genes leave existing genes' bucket
    unchanged? True for hash mode, generally False for group_shuffle. Used by the
    equivalence check to document the difference. df_large must be a superset (by
    gene) of df_small.

    (The previous revision branched on cfg.mode here with two IDENTICAL bodies,
    under a comment implying they differed. Removed 2026-07-21 -- the honest
    reporting the comment described comes from running the same procedure for
    both modes and letting the RESULT differ, which is what this does.)
    """
    r_s = split(df_small, cfg)
    r_l = split(df_large, cfg)
    names = cfg.schema.names
    gene_to_part_small = {g: p for p in names for g in r_s.genes[p]}
    gene_to_part_large = {g: p for p in names for g in r_l.genes[p]}
    for g, p in gene_to_part_small.items():
        if gene_to_part_large.get(g) != p:
            return False
    return True


def five_way_config(**overrides) -> SplitProtocolV2Config:
    """A configuration using the specification-compliant FIVE_WAY schema.

    Convenience so callers write `five_way_config(seed=7)` rather than
    assembling a schema and remembering not to touch the fraction fields.
    """
    return SplitProtocolV2Config(schema=FIVE_WAY, **overrides)
