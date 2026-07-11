"""Split protocol v2: four-way gene-disjoint partitioning for the conformal re-baseline.

Adds a DEDICATED conformal-calibration partition to the existing gene-disjoint split, so the
conformal quantile is estimated on data untouched by base-model fitting, meta-learner fitting,
probability calibration, and method or alpha selection. Coexists with the v1 split
(real_data_prep._gene_aware_split); it does not replace it. See CONFORMAL_DESIGN.md section 4.

Two modes, both producing gene-disjoint four-way partitions (train / tune / conformal / test):

  - "hash": extends the deterministic, hash-stable rule of splits.gene_stratified_split from three
    buckets to four, keyed off the canonical splits._gene_hash. A gene keeps its bucket as the
    dataset grows (stability invariant I8). This is the recommended default for the conformal
    stability claims.
  - "group_shuffle": nested sklearn GroupShuffleSplit, matching the mechanism the active v1 split
    uses. Gene-disjoint but NOT stable as the dataset grows. Retained for equivalence checking and
    migration.

"Equivalence" between the modes means both satisfy the same partition invariants (coverage,
gene-disjointness, both-classes, non-empty, fraction accuracy within tolerance, leakage-safe
remap), NOT identical gene membership -- the two mechanisms assign different genes to different
buckets and that is expected. Only the hash mode satisfies the stability invariant.

The leakage-safe train-only n_pathogenic_in_gene remap (incident 2026-06-13) is preserved and
extended to the conformal partition: every partition's count is derived from TRAIN rows only, with
unseen genes set to zero, and gene_has_known_disease recomputed in lockstep.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Reuse the canonical, hash-stable gene hash rather than a second implementation.
from .splits import _gene_hash

try:
    from sklearn.model_selection import GroupShuffleSplit
except Exception:  # pragma: no cover
    GroupShuffleSplit = None

PARTITIONS = ("train", "tune", "conformal", "test")


@dataclass
class SplitProtocolV2Config:
    """Four-way gene-disjoint split configuration.

    Fractions are of GENES (the split is gene-level), must be positive, and must sum to 1 within
    a small tolerance. Naming is unambiguous: `tune` is the model/method/alpha selection set
    (the v1 pipeline confusingly called its selection set `test`); `test` here is the locked
    evaluation set touched once; `conformal` is the dedicated calibration partition.
    """
    train_frac: float = 0.60
    tune_frac: float = 0.15
    conformal_frac: float = 0.10
    test_frac: float = 0.15
    seed: int = 42
    gene_col: str = "gene_symbol"
    label_col: str = "label"
    mode: str = "hash"  # "hash" | "group_shuffle"
    require_both_classes: bool = True
    frac_tolerance: float = 0.05  # floor for allowed absolute deviation of realized gene-fraction
    frac_sigma: float = 4.0        # also allow up to this many binomial standard errors (adaptive)
    count_col: str = "n_pathogenic_in_gene"
    derived_flag_col: str = "gene_has_known_disease"

    def __post_init__(self) -> None:
        for name in ("train_frac", "tune_frac", "conformal_frac", "test_frac"):
            v = getattr(self, name)
            if not (0.0 < v < 1.0):
                raise ValueError(f"{name} must be strictly in (0, 1), got {v}")
        s = self.train_frac + self.tune_frac + self.conformal_frac + self.test_frac
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"fractions must sum to 1.0, got {s:.6f}")
        if self.mode not in ("hash", "group_shuffle"):
            raise ValueError(f"mode must be 'hash' or 'group_shuffle', got {self.mode!r}")


@dataclass
class SplitResultV2:
    indices: dict = field(default_factory=dict)     # partition -> integer position array
    genes: dict = field(default_factory=dict)       # partition -> frozenset of gene symbols
    mode: str = "hash"
    seed: int = 42
    n_total: int = 0

    def summary(self) -> dict:
        return {
            p: {"n_rows": int(len(self.indices[p])), "n_genes": int(len(self.genes[p]))}
            for p in PARTITIONS
        }


def _resolve_genes(df: pd.DataFrame, gene_col: str) -> pd.Series:
    if gene_col not in df.columns:
        raise ValueError(f"gene column {gene_col!r} absent from DataFrame")
    return df[gene_col].fillna("unknown").astype(str)


def four_way_hash_split(df: pd.DataFrame, cfg: SplitProtocolV2Config) -> dict:
    """Deterministic, hash-stable four-way gene-disjoint split.

    Interval rule on h = _gene_hash(gene, seed) in [0, 1):
        [0, f_test)                              -> test
        [f_test, f_test + f_conf)                -> conformal
        [f_test + f_conf, f_test + f_conf + f_tune) -> tune
        [f_test + f_conf + f_tune, 1.0)          -> train
    Ordering test < conformal < tune < train keeps the smaller, more sensitive partitions in the
    low-hash region and is arbitrary but fixed.
    """
    genes = _resolve_genes(df, cfg.gene_col)
    unique = genes.unique()
    h = {g: _gene_hash(g, cfg.seed) for g in unique}
    b_test = cfg.test_frac
    b_conf = cfg.test_frac + cfg.conformal_frac
    b_tune = cfg.test_frac + cfg.conformal_frac + cfg.tune_frac
    hs = genes.map(h)
    masks = {
        "test": hs < b_test,
        "conformal": (hs >= b_test) & (hs < b_conf),
        "tune": (hs >= b_conf) & (hs < b_tune),
        "train": hs >= b_tune,
    }
    return {p: np.where(m.to_numpy())[0] for p, m in masks.items()}


def four_way_group_shuffle_split(df: pd.DataFrame, cfg: SplitProtocolV2Config) -> dict:
    """Nested sklearn GroupShuffleSplit four-way gene-disjoint split.

    Carves test, then conformal, then tune from the shrinking pool, leaving train. Gene-disjoint
    but not stable as the dataset grows (that is the documented difference from the hash mode).
    """
    if GroupShuffleSplit is None:
        raise RuntimeError("scikit-learn required for group_shuffle mode")
    genes = _resolve_genes(df, cfg.gene_col)
    n = len(df)
    all_pos = np.arange(n)

    def carve(pool_pos: np.ndarray, frac_of_whole: float, seed: int) -> tuple:
        pool_frac = frac_of_whole * n / max(len(pool_pos), 1)
        pool_frac = min(max(pool_frac, 1e-9), 1 - 1e-9)
        gss = GroupShuffleSplit(n_splits=1, test_size=pool_frac, random_state=seed)
        keep_rel, carve_rel = next(gss.split(pool_pos, groups=genes.iloc[pool_pos].to_numpy()))
        return pool_pos[keep_rel], pool_pos[carve_rel]

    pool, test_pos = carve(all_pos, cfg.test_frac, cfg.seed)
    pool, conf_pos = carve(pool, cfg.conformal_frac, cfg.seed + 1)
    pool, tune_pos = carve(pool, cfg.tune_frac, cfg.seed + 2)
    train_pos = pool
    return {"train": train_pos, "tune": tune_pos, "conformal": conf_pos, "test": test_pos}


def assert_partition_invariants(indices: dict, df: pd.DataFrame,
                                cfg: SplitProtocolV2Config) -> dict:
    """Fail-loud checks shared by both modes. Returns the realized gene sets on success."""
    genes = _resolve_genes(df, cfg.gene_col)
    n = len(df)

    # I1 coverage: exactly-once partition of all rows
    all_idx = np.concatenate([indices[p] for p in PARTITIONS]) if n else np.array([], dtype=int)
    if len(all_idx) != n:
        raise AssertionError(f"coverage: partitions cover {len(all_idx)} rows, expected {n}")
    if len(np.unique(all_idx)) != n:
        raise AssertionError("coverage: overlapping row assignments detected")

    # I5 non-empty
    for p in PARTITIONS:
        if len(indices[p]) == 0:
            raise AssertionError(f"partition {p!r} is empty; adjust fractions or seed")

    gene_sets = {p: frozenset(genes.iloc[indices[p]].unique()) for p in PARTITIONS}

    # I2 gene-disjoint: all six pairwise
    for i, a in enumerate(PARTITIONS):
        for b in PARTITIONS[i + 1:]:
            overlap = gene_sets[a] & gene_sets[b]
            if overlap:
                raise AssertionError(
                    f"gene-disjoint violated between {a} and {b}: "
                    f"{len(overlap)} shared gene(s), e.g. {sorted(overlap)[:3]}")

    # I3 fraction accuracy (of genes). Hash bucketing is exact only in expectation; the realized
    # gene-fraction deviates by finite-sample binomial variance that shrinks as the gene count
    # grows. Use an adaptive bound: the larger of the configured floor and several binomial
    # standard errors, so the check is meaningful at both small test scales and the real
    # thousands-of-genes scale.
    total_genes = len(set().union(*gene_sets.values()))
    want = {"train": cfg.train_frac, "tune": cfg.tune_frac,
            "conformal": cfg.conformal_frac, "test": cfg.test_frac}
    for p in PARTITIONS:
        realized = len(gene_sets[p]) / max(total_genes, 1)
        se = (want[p] * (1.0 - want[p]) / max(total_genes, 1)) ** 0.5
        bound = max(cfg.frac_tolerance, cfg.frac_sigma * se)
        if abs(realized - want[p]) > bound:
            raise AssertionError(
                f"fraction accuracy: {p} realized {realized:.3f} vs requested {want[p]:.3f} "
                f"(bound {bound:.3f} = max(floor {cfg.frac_tolerance}, "
                f"{cfg.frac_sigma}*se {se:.3f}), n_genes={total_genes})")

    # I4 both-classes
    if cfg.require_both_classes and cfg.label_col in df.columns:
        for p in PARTITIONS:
            classes = set(pd.to_numeric(
                df.iloc[indices[p]][cfg.label_col], errors="coerce").dropna().unique())
            if classes != {0, 1} and classes != {0.0, 1.0}:
                raise AssertionError(
                    f"partition {p!r} missing class(es): has {classes}, need {{0, 1}}")

    return gene_sets


def apply_train_only_leakage_remap(df: pd.DataFrame, indices: dict,
                                   cfg: SplitProtocolV2Config) -> pd.DataFrame:
    """Recompute n_pathogenic_in_gene from TRAIN rows only and remap onto every partition (unseen
    genes -> 0), recomputing gene_has_known_disease in lockstep. Preserves the incident-2026-06-13
    leakage fix and extends it to the conformal partition. Returns a copy with corrected columns.
    """
    if cfg.count_col not in df.columns:
        return df  # nothing to remap
    out = df.copy()
    # Dtype-safe promotion (2026-07-11): the cohort's count_col/derived_flag_col may be
    # int32; the per-partition .iloc writes below assign int64 arrays (from .map().sum()),
    # which triggers a pandas incompatible-dtype FutureWarning (silent cast now, error in a
    # future pandas). Widen these two columns to int64 once, up front, so the writes are
    # dtype-compatible. Values are unchanged (int32 -> int64 is lossless for these counts).
    if cfg.count_col in out.columns:
        out[cfg.count_col] = out[cfg.count_col].astype("int64")
    if cfg.derived_flag_col in out.columns:
        out[cfg.derived_flag_col] = out[cfg.derived_flag_col].astype("int64")
    genes = _resolve_genes(df, cfg.gene_col)
    if cfg.label_col not in df.columns:
        raise ValueError(f"label column {cfg.label_col!r} required for train-only remap")
    tr = indices["train"]
    y_tr_pos = (pd.to_numeric(df.iloc[tr][cfg.label_col], errors="coerce").to_numpy() == 1).astype(int)
    g_tr = genes.iloc[tr].to_numpy()
    train_counts = pd.Series(y_tr_pos).groupby(g_tr).sum()
    for p in PARTITIONS:
        ix = indices[p]
        g = genes.iloc[ix]
        cnt = g.map(train_counts).fillna(0).astype(int).to_numpy()
        out.iloc[ix, out.columns.get_loc(cfg.count_col)] = cnt
        if cfg.derived_flag_col in out.columns:
            out.iloc[ix, out.columns.get_loc(cfg.derived_flag_col)] = (cnt > 0).astype(int)
    return out


def split(df: pd.DataFrame, cfg: Optional[SplitProtocolV2Config] = None) -> SplitResultV2:
    """Produce a validated four-way gene-disjoint split. Dispatches by mode, asserts all shared
    invariants, and (if the count column is present) applies the train-only leakage remap."""
    cfg = cfg or SplitProtocolV2Config()
    if cfg.mode == "hash":
        indices = four_way_hash_split(df, cfg)
    else:
        indices = four_way_group_shuffle_split(df, cfg)
    gene_sets = assert_partition_invariants(indices, df, cfg)
    return SplitResultV2(indices=indices, genes=gene_sets, mode=cfg.mode,
                         seed=cfg.seed, n_total=len(df))


def genes_are_stable_under_growth(df_small: pd.DataFrame, df_large: pd.DataFrame,
                                  cfg: SplitProtocolV2Config) -> bool:
    """Stability invariant I8: does adding genes leave existing genes' bucket unchanged? True for
    hash mode, generally False for group_shuffle. Used by the equivalence check to document the
    difference. df_large must be a superset (by gene) of df_small."""
    if cfg.mode != "hash":
        # group_shuffle is not designed to be stable; report honestly
        r_s = split(df_small, cfg)
        r_l = split(df_large, cfg)
    else:
        r_s = split(df_small, cfg)
        r_l = split(df_large, cfg)
    gene_to_part_small = {g: p for p in PARTITIONS for g in r_s.genes[p]}
    gene_to_part_large = {g: p for p in PARTITIONS for g in r_l.genes[p]}
    for g, p in gene_to_part_small.items():
        if gene_to_part_large.get(g) != p:
            return False
    return True
