"""Stage 1D-1c.1C, set "av": the QUALIFIED data preparation -- verified sources in, one canonical epoch-2 matrix out.

Owner rulings aa999b96 (sections 13-15 and revised sections 10 and 15) and ff77015c (sections 8, 11 and 13); audit
AUDIT_OF_RULING_ff77015c_2026-11-16.md. No model is fitted here and scientific training stays refused.

THE MODE IS EXPLICIT. ``PreparationMode`` names the two preparations. HISTORICAL_REPLAY is ``DataPrepPipeline.run`` /
``run_v2``, unchanged and sealed by the census and the pilot records; this module prepares QUALIFIED only, and refuses
any other mode -- nothing is inferred from a missing argument or a missing path.

THE FLOW (revised section 10, points 1-10):
  1. the caller reads every source through materialization.read_verified_bytes and DECLARES its digest; a source whose
     bytes differ from the declaration is refused by name (no substitution: a different file is a different source);
  2. the GRCh38 reference status is a verified ``reference_status.ReferenceStatus`` for exactly this population;
  3. the typed AF materializer (allele_frequency.materialize_allele_frequency) and 4. the registered strict policy
     (select_strict) produce the model value -- NaN wherever nothing was selected;
  5. the full population is preserved (the 500 alleles with unknown bases stay in it) and 6. the sequence-resolved
     primary eligibility is a recorded projection (``claim``), never a deletion;
  7. ``allele_freq`` becomes the selected value and the single builder runs (engineer_features, gvc.features/2);
  8. the canonical matrix is validated (roster, AF family bit for bit against feature_semantics.derive_family, af_log10
     certified) and persisted content-addressed with its receipt; 9. the receipt records the source, policy, code and population
     identities; 10. training is refused (no scientific FitContext can be constructed: models/fit_context.py).

Refused, never repaired: a non-missing ``allele_freq`` in the incoming frame (the cohort's own column is an
unqualified source; measured all-missing in the owner's cohort), any column the legacy gnomAD join writes (1000 Genomes
fallback and population columns, FinnGen columns: unqualified population-frequency sources), a frame whose rows are not
the population in order, a reference status for another population or another rule, and a census that differs from a
declared expectation. The LABEL-DERIVED columns ``n_pathogenic_in_gene`` and ``gene_has_known_disease`` are not part of
the label-blind canonical matrix: they are computed from labels and materialised per split (train-only) by the split
protocol; both are ``retire_predictive`` in the census.

A fitted ensemble is BOUND to the canonical matrix it was fitted on (``bind_fit``); ``score_canonical`` is EXACT REPLAY
and refuses any other matrix -- another source, policy, population, semantics, roster or representation (the exact bits,
af_log10's included). It is not a general inference interface: scoring a newly qualified population under the same
model-input contract is stage 1D-1c.5's (owner ruling 6fa240c0, section 8), and a model is never scored on a numerically
equivalent but different representation until prediction equivalence is qualified.

THE IDENTITY IS THE SAME ON EVERY MACHINE (CANONICAL-IDENTITY-ENVIRONMENT-1, measured 2026-11-17). The float64 bits of a
transcendental function depend on the numerical environment: on the real 1,028,224 observed values, numpy's log10 gave
different bits on 27,950 rows between its AVX-512 (Advanced Vector Extensions, 512-bit) path and its baseline path (at most
2 ULP, units in the last place), and the owner's Windows machine produced a third result -- so a digest of those bits was
a digest of the machine. The canonical identity therefore seals every column bit for bit EXCEPT the columns
``environment_dependent_columns`` proves to pass through a transcendental function (today exactly ``af_log10``); each of
those is bound by its exact input (``af_raw``, sealed), its formula (``PROCEDURE``) and an accuracy certificate
(``certify_log10``: within ``LOG10_ULP_BOUND`` of log10 computed to ``REFERENCE_PRECISION_BITS`` bits), checked when the
matrix is prepared and again whenever a stored matrix is reused. The stored table keeps the preparing machine's bits.

THREE IDENTITIES, NEVER CONFLATED (set "av" revision c; owner ruling 6fa240c0 sections 5-7, 9-11):
  * the LOGICAL identity (``CanonicalMatrix.identity``): the digest of the receipt -- what the matrix means;
  * the REPRESENTATION (``CanonicalMatrix.representation``): the digest of every column's exact bits, af_log10's
    included, and of the selection and claim codes -- what a model was fitted on, bound by ``bind_fit``;
  * the PAYLOAD: the SHA-256 of a stored file's bytes. The store keeps ``<logical>/<payload>.parquet`` with its record
    (integrity.store), so two certified representations of one logical identity never collide, and a record whose
    payload is missing or differs is refused.
A ``CanonicalMatrix`` is VALID BY CONSTRUCTION and IMMUTABLE: its constructor verifies the receipt's population, roster,
exact-column digest, code digests and af_log10 certificate against the arrays it is given and stores them bytes-backed
(NumPy refuses to make them writable again); ``features`` is a fresh zero-copy read-only view and ``receipt`` a fresh
copy, so an edit of either reaches nothing; ``verify`` re-measures the arrays at every consumer (bind, score, write).

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, NamedTuple

import numpy as np
import pandas as pd

from genomic_variant_classifier.information_sources import allele_frequency as A
from genomic_variant_classifier.information_sources import integrity as I
from genomic_variant_classifier.information_sources import reference_status as R
from genomic_variant_classifier.information_sources.census import HEX64, canonical_bytes, sha256_bytes, symbol_closure_digest
from genomic_variant_classifier.information_sources.columns import Population, readonly_array
from genomic_variant_classifier.information_sources.materialization import Parsed, check_registered

logger = logging.getLogger(__name__)

__all__ = ["BINDING_SCHEMA", "BINDING_SCOPE", "CanonicalMatrix", "COHORT", "ENVIRONMENT_DEPENDENT", "LEGACY_JOIN_COLUMNS",
           "LOG10_ULP_BOUND", "PROCEDURE", "PreparationMode", "QUALIFICATION_SCOPE", "RECEIPT_SCHEMA",
           "REFERENCE_PRECISION_BITS", "REPRESENTATION_SCHEMA", "SCHEMA", "SPLIT_DEPENDENT", "STORE_SCHEMA",
           "StoredCanonical", "TRAINING", "TRANSCENDENTAL", "bind_fit", "build_in_chunks", "builder_input_names",
           "canonical_roster", "certify_log10", "environment_dependent_columns", "exact_column_digests", "fit_roster",
           "load_canonical", "payload_for", "prepare_qualified", "require_source_binding", "row_local_proof",
           "score_canonical", "write_canonical"]

SCHEMA = "gvc.qualified-preparation.1d1c1c/1"
RECEIPT_SCHEMA = "gvc.canonical-matrix-receipt.1d1c1c/3"
BINDING_SCHEMA = "gvc.source-binding.1d1c1c/2"
STORE_SCHEMA = "gvc.canonical-matrix-store.1d1c1c/2"
REPRESENTATION_SCHEMA = "gvc.canonical-matrix-representation.1d1c1c/1"
#: What a source binding licenses (owner ruling 6fa240c0, section 8): replay on exactly the fitted representation.
BINDING_SCOPE = ("exact_replay: scoring only the matrix the ensemble was fitted on (its sources, policy, population, code, "
                 "roster and exact bits); general inference on a newly qualified population is stage 1D-1c.5's")
#: What this matrix is qualified for, stated in its receipt (owner ruling 6fa240c0, section 12).
QUALIFICATION_SCOPE = ("allele-frequency family only (QUALIFIED-FRAME-PROVENANCE-1): af_raw and its family come from the "
                       "verified, declared gnomAD bytes through the registered strict policy; every other column is the "
                       "single builder's value from the caller's frame, whose derivation from the declared cohort bytes is "
                       "the caller's and is not re-established here -- not a source-qualified biological feature matrix")
COHORT = "dataset:clinvar_cohort"
_THIS_FILE = "src/genomic_variant_classifier/information_sources/qualified_preparation.py"
_AF_FILE = "src/genomic_variant_classifier/information_sources/allele_frequency.py"
_RS_FILE = "src/genomic_variant_classifier/information_sources/reference_status.py"
#: The code a canonical matrix's identity binds (semantic-closure digests, measured at preparation).
PROCEDURE = (f"{_THIS_FILE}::prepare_qualified", f"{_AF_FILE}::materialize_allele_frequency", f"{_AF_FILE}::select_strict",
             f"{_RS_FILE}::materialize_reference_status",
             "src/genomic_variant_classifier/models/variant_ensemble.py::engineer_features",
             "src/genomic_variant_classifier/models/feature_semantics.py::derive_family",
             "src/genomic_variant_classifier/models/feature_semantics.py::formula_af_log10")
#: The canonical columns whose float64 bits depend on the numerical environment (proven: environment_dependent_columns).
ENVIRONMENT_DEPENDENT = ("af_log10",)
#: The project's REGISTERED PERMISSIBLE ERROR for epoch 2's af_log10 (owner ruling 6fa240c0, sections 3.2 and 6): a policy
#: bound, not a guarantee of numpy's for every platform and input. Its scale was chosen with the worst case numpy's
#: developers stated for the SVML-vectorised functions (Short Vector Math Library, "up to 4 ULP", numpy-discussion, the
#: SVML merge announcement) as context. MEASURED 2026-11-17 against the reference on the real 454,351 distinct
#: arguments: at most 0.575 ULP on the AVX-512 path and 1.552 ULP on the baseline path. A path beyond it is refused.
LOG10_ULP_BOUND = 4.0
#: The reference: log10 to 113 bits (mpmath, arbitrary precision in exact integer arithmetic), so its own error (about
#: 2**-112 relative) is negligible against one float64 ULP (2**-52 relative).
REFERENCE_PRECISION_BITS = 113
#: Names whose float64 result is not fixed by IEEE 754 basic arithmetic (a library approximation, or numexpr's evaluation).
TRANSCENDENTAL = frozenset({"log", "log10", "log2", "log1p", "exp", "exp2", "expm1", "power", "float_power", "pow", "rpow",
                            "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2", "asin", "acos", "atan", "atan2",
                            "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "asinh", "acosh", "atanh", "cbrt",
                            "hypot", "logaddexp", "logaddexp2", "erf", "erfc", "gamma", "lgamma", "eval"})
#: Label-derived: computed from labels, so materialised per split by the split protocol, never here.
SPLIT_DEPENDENT = ("n_pathogenic_in_gene", "gene_has_known_disease")
#: Columns DataPrepPipeline._join_gnomad writes besides allele_freq (1000 Genomes and FinnGen): unqualified sources here.
LEGACY_JOIN_COLUMNS = ("gnomad_af", "af_1kg_afr", "af_1kg_eur", "af_1kg_eas", "af_1kg_sas", "af_1kg_amr",
                       "finngen_af_fin", "finngen_af_nfsee", "finngen_enrichment", "r13_finngen_af_fin",
                       "r13_finngen_af_nfsee", "finngen_r13_enrichment")
TRAINING = ("refused: no scientific fit context can be constructed until the readiness gates are certified "
            "(models/fit_context.py; execution.scientific_fit_context names them)")


class PreparationMode(str, Enum):
    HISTORICAL_REPLAY = "historical_replay"     # DataPrepPipeline.run / run_v2: the sealed historical preparation
    QUALIFIED = "qualified"                     # this module


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def canonical_roster() -> tuple[str, ...]:
    """The canonical matrix's columns: the epoch-2 contract without the label-derived columns, in contract order."""
    from genomic_variant_classifier.models.feature_semantics import FeatureSemantics, tabular_features

    return tuple(n for n in tabular_features(FeatureSemantics.V2_OBSERVED_AF) if n not in SPLIT_DEPENDENT)


def builder_input_names() -> frozenset[str]:
    """Every input column the single builder CAN read, measured from its running source (inspect + ast): it reads its
    frame ``df`` only through ``df.get(<name>)``, ``df[<name>]``, ``<name> in df.columns``, ``df.index`` and ``len(df)``, so a
    column whose name is no string constant of its body cannot influence its output. Refused, never assumed: any other
    use of ``df`` (another attribute, ``df`` passed to a call, a non-constant subscript) -- the bound would not hold."""
    import ast
    import inspect
    import textwrap

    from genomic_variant_classifier.models.variant_ensemble import engineer_features

    tree = ast.parse(textwrap.dedent(inspect.getsource(engineer_features)))
    fn = tree.body[0]
    if not isinstance(fn, ast.FunctionDef) or fn.args.args[0].arg != "df":
        raise AssertionError("engineer_features's first parameter is not df")
    membership = {id(c.comparators[0]) for c in ast.walk(fn) if isinstance(c, ast.Compare) and len(c.ops) == 1
                  and isinstance(c.ops[0], (ast.In, ast.NotIn)) and isinstance(c.left, ast.Constant)
                  and isinstance(c.left.value, str)}
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "df" \
                and node.attr == "columns" and id(node) not in membership:
            raise AssertionError("engineer_features reads df.columns other than as '<name> in df.columns'")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "df" \
                and node.attr not in {"get", "columns", "index"}:
            raise AssertionError(f"engineer_features uses df.{node.attr}; its input columns cannot be bounded")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id == "len"):
            args = [*node.args, *(k.value for k in node.keywords)]          # len(df) reads no column
            if any(isinstance(a, ast.Name) and a.id == "df" for a in args):
                raise AssertionError("engineer_features passes df to a call; its input columns cannot be bounded")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "df" \
                and not (isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
            raise AssertionError("engineer_features subscripts df with a non-constant; its input columns cannot be bounded")
    # Every column name the builder asks for is a string literal of its own body: df.get(<x>) and df[<x>] take a constant,
    # or a loop variable bound by a for loop over a literal tuple or list of constants (or of tuples whose first item is
    # one). A name from a module constant or an argument would escape the bound, so it is refused.
    literal_loop: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.For) and isinstance(node.iter, (ast.Tuple, ast.List)):
            items = node.iter.elts
            if all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in items) and isinstance(node.target,
                                                                                                         ast.Name):
                literal_loop.add(node.target.id)
            elif all(isinstance(e, ast.Tuple) and e.elts and isinstance(e.elts[0], ast.Constant)
                     and isinstance(e.elts[0].value, str) for e in items) and isinstance(node.target, ast.Tuple) \
                    and isinstance(node.target.elts[0], ast.Name):
                literal_loop.add(node.target.elts[0].id)
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "df":
            key = node.args[0] if node.args else None
            if not ((isinstance(key, ast.Constant) and isinstance(key.value, str))
                    or (isinstance(key, ast.Name) and key.id in literal_loop)):
                raise AssertionError(f"engineer_features reads df.get({ast.unparse(key) if key else ''}); its input "
                                     "columns cannot be bounded")
    return frozenset(n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, str))


#: Methods that combine rows. The builder must use none of them (``row_local_proof``) except its one diagnostic count.
CROSS_ROW = frozenset({"groupby", "rank", "quantile", "mean", "median", "sum", "std", "var", "min", "max", "shift",
                       "diff", "cumsum", "cumprod", "cummax", "cummin", "value_counts", "unique", "nunique", "duplicated",
                       "drop_duplicates", "transform", "rolling", "expanding", "ewm", "merge", "join", "sort_values",
                       "sort_index", "agg", "aggregate", "describe", "corr", "cov", "pct_change", "interpolate",
                       "ffill", "bfill", "resample", "pivot", "pivot_table", "melt", "explode", "sample", "nlargest",
                       "nsmallest", "idxmax", "idxmin", "mode", "prod", "count", "any", "all"})


def row_local_proof() -> str:
    """The builder is ROW-LOCAL -- output row i depends on input row i alone -- by its running source: no method that
    combines rows is called anywhere in it, except ``_null_counts = feats.isnull().sum()``, which feeds a log line only
    (asserted: the name is read by nothing but the log call and its own filter); its only non-builtin callee,
    ``derive_family``, combines rows only to refuse. Returns the digest of the sources the proof was made on; refused,
    never assumed, when either holds anything else."""
    import ast
    import inspect
    import textwrap

    from genomic_variant_classifier.models.variant_ensemble import engineer_features

    source = textwrap.dedent(inspect.getsource(engineer_features))
    fn = ast.parse(source).body[0]
    allowed = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == ["_null_counts"]:
            allowed.add(id(node.value))
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in CROSS_ROW \
                and id(node) not in allowed:
            raise AssertionError(f"engineer_features calls .{node.func.attr}(), which combines rows; chunking is not proven "
                                 "output-identical")
    # The diagnostic: "_nulls = _null_counts[_null_counts > 0]" and "if len(_nulls): logger.info(...)" -- nothing else may
    # read either name, so the counts reach no feature.
    permitted: set[int] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == ["_nulls"]:
            permitted.update(id(n) for n in ast.walk(node.value))
        if isinstance(node, ast.If) and ast.unparse(node.test) == "len(_nulls)" and len(node.body) == 1 \
                and not node.orelse and ast.unparse(node.body[0]).startswith("logger.info("):
            permitted.update(id(n) for n in ast.walk(node))
    stray = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id in {"_null_counts", "_nulls"}
             and isinstance(n.ctx, ast.Load) and id(n) not in permitted]
    if stray:
        raise AssertionError(f"the builder's null counts are read beyond their diagnostic log line (lines {stray})")
    # Its one non-builtin callee, feature_semantics.derive_family: elementwise on observed rows; a cross-row call there
    # may only be the test of an "if" whose body raises (a refusal holds for the whole iff it holds for some chunk).
    from genomic_variant_classifier.models import feature_semantics

    called = {n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    if called - {"TypeError", "ValueError", "derive_family", "dict", "isinstance", "len", "max", "range", "str", "type"}:
        raise AssertionError(f"engineer_features calls {sorted(called)}; only the measured callees are proven")
    dsource = textwrap.dedent(inspect.getsource(feature_semantics.derive_family))
    dfn = ast.parse(dsource).body[0]
    refusals: set[int] = set()
    for node in ast.walk(dfn):
        if isinstance(node, ast.If) and all(isinstance(s, ast.Raise) for s in node.body) and not node.orelse:
            refusals.update(id(n) for n in ast.walk(node))                  # the test and the refusal's message
        # "if observed.any(): values[observed] = f(a[observed])": a guard that skips an empty selection; the result is
        # the same either way, so it is row-local too.
        if isinstance(node, ast.If) and ast.unparse(node.test) == "observed.any()" and not node.orelse \
                and len(node.body) == 1 and isinstance(node.body[0], ast.Assign) \
                and ast.unparse(node.body[0].targets[0]) == "values[observed]":
            refusals.update(id(n) for n in ast.walk(node.test))
    for node in ast.walk(dfn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in CROSS_ROW \
                and id(node) not in refusals:
            raise AssertionError(f"derive_family calls .{node.func.attr}() outside a refusal; chunking is not proven")
    source = source + dsource
    return sha256_bytes(source.encode("utf-8"))


def build_in_chunks(work: pd.DataFrame, roster: tuple[str, ...], *, chunk_rows: int) -> pd.DataFrame:
    """The canonical columns of ``engineer_features(work)`` (gvc.features/2), built chunk by chunk into float64 columns.
    Output-identical to one call because the builder is row-local (``row_local_proof``); the suite also compares the
    two bit for bit, and so does the real-data record on its first rows. Measured: one call over the 1,490,014-row
    cohort peaks about 4.5 GB above its input; chunking bounds the builder's temporaries by the chunk."""
    return pd.DataFrame(_build_columns(work, roster, chunk_rows=chunk_rows), copy=False)


def _build_columns(work: pd.DataFrame, roster: tuple[str, ...], *, chunk_rows: int) -> dict[str, np.ndarray]:
    """``build_in_chunks``'s columns as float64 arrays, in roster order."""
    from genomic_variant_classifier.models.feature_semantics import FeatureSemantics
    from genomic_variant_classifier.models.variant_ensemble import engineer_features

    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("chunk_rows is a positive int")
    row_local_proof()
    n = len(work)
    out = {c: np.empty(n, dtype=np.float64) for c in roster}
    for start in range(0, n, chunk_rows):
        stop = min(n, start + chunk_rows)
        part = engineer_features(work.iloc[start:stop], semantics=FeatureSemantics.V2_OBSERVED_AF)
        if [c for c in part.columns if c not in SPLIT_DEPENDENT] != list(roster):
            raise ValueError("engineer_features did not produce the epoch-2 contract")
        for c in roster:
            col = part[c]
            if not (pd.api.types.is_float_dtype(col.dtype) or pd.api.types.is_integer_dtype(col.dtype)
                    or pd.api.types.is_bool_dtype(col.dtype)):
                raise TypeError(f"{c}: the builder produced dtype {col.dtype}; the canonical matrix holds numbers only")
            out[c][start:stop] = col.to_numpy(dtype=np.float64)
        del part
    return out


def fit_roster(compiled: Mapping[str, Any]) -> tuple[str, ...]:
    """The epoch-2 column names of a roster compiled by the admission authority (admission.compiled_roster, whose names
    are census identities), in contract order. Every name must be a canonical column: a roster naming a label-derived
    column is refused (both are retire_predictive, so no regime admits them today)."""
    from genomic_variant_classifier.models.feature_semantics import RENAMED_IN_V2

    if not isinstance(compiled, Mapping) or compiled.get("schema") != "gvc.compiled-roster.1d1c1c/1":
        raise ValueError("a fit roster comes from admission.compiled_roster")
    to_v2 = {v: k for k, v in RENAMED_IN_V2.items()}
    names = tuple(to_v2.get(n, n) for n in compiled["names"])
    canonical = canonical_roster()
    outside = [n for n in names if n not in canonical]
    if outside:
        raise ValueError(f"the compiled roster names columns the canonical matrix does not hold: {outside}")
    return names


def _transcendental_sites(fn) -> list:
    """Every node of ``fn`` whose float64 result is not fixed by IEEE 754 basic arithmetic: a reference to a TRANSCENDENTAL
    name (called or passed as a value, e.g. ``.apply(np.log)``) and every power (``**``, whatever its exponent)."""
    import ast

    sites = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and node.attr in TRANSCENDENTAL:
            sites.append(node)
        elif isinstance(node, ast.Name) and node.id in TRANSCENDENTAL:
            sites.append(node)
        elif isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.Pow):
            sites.append(node)
    return sites


def environment_dependent_columns() -> frozenset[str]:
    """The canonical columns whose bits can depend on the numerical environment, PROVEN from the running sources (inspect
    + ast): every transcendental site of the builder must lie in the value of an assignment ``feats["<name>"] = ...``, and
    every transcendental site of an AF-family formula is attributed to the member that formula computes. Everything else
    is IEEE 754 basic arithmetic, comparison, casting or string length -- the same bits on every machine. Refused, never
    assumed: a site anywhere else (it would make an unnamed column environment-dependent)."""
    import ast
    import inspect
    import textwrap

    from genomic_variant_classifier.models import feature_semantics
    from genomic_variant_classifier.models.variant_ensemble import engineer_features

    fn = ast.parse(textwrap.dedent(inspect.getsource(engineer_features))).body[0]
    owner: dict[int, str] = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript) \
                and isinstance(node.targets[0].value, ast.Name) and node.targets[0].value.id == "feats" \
                and isinstance(node.targets[0].slice, ast.Constant) and isinstance(node.targets[0].slice.value, str):
            for sub in ast.walk(node.value):
                owner[id(sub)] = node.targets[0].slice.value
    found: set[str] = set()
    for site in _transcendental_sites(fn):
        if id(site) not in owner:
            raise AssertionError(f"engineer_features line {site.lineno}: a transcendental operation outside a "
                                 "'feats[<name>] = ...' assignment; the environment-dependent columns cannot be named")
        found.add(owner[id(site)])
    for dependent, formula in feature_semantics.AF_FAMILY.implementation_names().items():
        ffn = ast.parse(textwrap.dedent(inspect.getsource(getattr(feature_semantics, formula)))).body[0]
        if _transcendental_sites(ffn):
            found.add(dependent)
    dfn = ast.parse(textwrap.dedent(inspect.getsource(feature_semantics.derive_family))).body[0]
    if _transcendental_sites(dfn):
        raise AssertionError("derive_family itself uses a transcendental operation")
    return frozenset(found)


def certify_log10(af_raw: np.ndarray, af_log10: np.ndarray) -> float:
    """The largest distance, in units in the last place of the computed value, between ``af_log10`` and log10(af_raw +
    1e-8) computed to REFERENCE_PRECISION_BITS bits; refused when it exceeds LOG10_ULP_BOUND, or when af_log10 is not
    missing exactly where af_raw is. The argument is the formula's own (an IEEE addition: the same on every machine), and
    the reference is computed once per distinct argument (measured: 454,351 of 1,028,224 observed values, 4 seconds)."""
    import mpmath

    a = np.asarray(af_raw, dtype=np.float64)
    y = np.asarray(af_log10, dtype=np.float64)
    observed = ~np.isnan(a)
    if a.shape != y.shape or not np.array_equal(np.isnan(y), ~observed):
        raise ValueError("af_log10 is not missing exactly where af_raw is missing; refused")
    if not observed.any():
        return 0.0
    x = a[observed] + 1e-8
    distinct, inverse = np.unique(x, return_inverse=True)
    head, tail = np.empty(len(distinct)), np.empty(len(distinct))
    with mpmath.workprec(REFERENCE_PRECISION_BITS):
        for i, v in enumerate(distinct.tolist()):
            ref = mpmath.log10(mpmath.mpf(v))
            head[i] = float(ref)                                    # the reference rounded to float64 ...
            tail[i] = float(ref - mpmath.mpf(head[i]))              # ... and the remainder
    got = y[observed]
    ref_head, ref_tail = head[inverse], tail[inverse]
    exact_zero = (ref_head == 0.0) & (ref_tail == 0.0)              # log10(1) = 0 exactly: only 0 is within any bound
    if (got[exact_zero] != 0.0).any():
        raise ValueError("af_log10 is not 0 where its argument is exactly 1; refused")
    keep = ~exact_zero
    # got and ref_head are within a few ULP of each other and of one sign, so their difference is exact (Sterbenz).
    distance = np.abs((got[keep] - ref_head[keep]) - ref_tail[keep]) / np.spacing(np.abs(got[keep]))
    worst = float(distance.max()) if distance.size else 0.0
    if not worst <= LOG10_ULP_BOUND:
        raise ValueError(f"af_log10 is {worst:.3f} ULP from log10 computed to {REFERENCE_PRECISION_BITS} bits, beyond the "
                         f"certified bound of {LOG10_ULP_BOUND} ULP; refused")
    return worst


def _column_sha256(values: np.ndarray) -> str:
    """SHA-256 of a column's little-endian float64 bytes, every NaN written as the one canonical NaN."""
    v = np.array(values, dtype="<f8", copy=True)
    v[np.isnan(v)] = np.nan
    return hashlib.sha256(v.tobytes()).hexdigest()


def exact_column_digests(features: pd.DataFrame) -> dict[str, str]:
    """Per-column SHA-256 of every column whose bits are the same on every machine (all but ENVIRONMENT_DEPENDENT), in
    roster order; the receipt seals their canonical digest (``_exact_digest``)."""
    return {str(c): _column_sha256(features[c].to_numpy(dtype=np.float64)) for c in features.columns
            if c not in ENVIRONMENT_DEPENDENT}


def _exact_digest(digests: Mapping[str, str]) -> str:
    """The canonical digest of per-column digests, independent of their order (the roster's order is sealed separately,
    by the receipt's ``roster``); a record's sorted keys reproduce it."""
    return sha256_bytes(canonical_bytes({"columns": sorted([str(n), str(d)] for n, d in digests.items())}))


def _environment_dependent_entry() -> dict[str, Any]:
    """What the receipt states about the environment-dependent columns: how each is bound, never its bits."""
    return {"af_log10": {"formula": "feature_semantics.formula_af_log10: numpy log10(af_raw + 1e-8) on observed rows",
                         "input": "af_raw, sealed bit for bit with the exact columns",
                         "missing": "exactly where af_raw is missing",
                         "certificate": {"reference": f"log10 to {REFERENCE_PRECISION_BITS} bits (mpmath)",
                                         "bound_ulp": LOG10_ULP_BOUND,
                                         "bound_basis": "the project's registered permissible error for epoch 2 (owner "
                                                        "ruling 6fa240c0), not a guarantee of numpy's for every "
                                                        "platform and input"}}}


def _bytes_backed(values: np.ndarray) -> bool:
    """True when ``values`` is read-only down to an immutable ``bytes`` buffer (graph_identity.readonly_array's form), so
    nothing can make it writable again."""
    base: Any = values
    while isinstance(base, np.ndarray):
        if base.flags.writeable:
            return False
        base = base.base
    return isinstance(base, bytes)


def _immutable_vector(values: Any, dtype: Any, n: int, what: str) -> np.ndarray:
    if not isinstance(values, np.ndarray) or values.ndim != 1 or values.dtype != np.dtype(dtype) or len(values) != n:
        raise ValueError(f"{what}: a one-dimensional {np.dtype(dtype)} array of {n} rows is required")
    return values if _bytes_backed(values) else readonly_array(values)


def _codes_sha256(codes: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(codes, np.uint8).tobytes()).hexdigest()


@dataclass(frozen=True, eq=False)
class CanonicalMatrix:
    """The canonical epoch-2 matrix of one population. VALID BY CONSTRUCTION and IMMUTABLE: the constructor refuses
    unless the receipt (canonical bytes) is a schema-``RECEIPT_SCHEMA`` receipt for ``population``, the columns are its
    roster in order (one float64 value per row), the exact columns reproduce its exact-column digest, the selection and
    claim codes reproduce their digests, the environment-dependent entry is the current accuracy contract and af_log10 is
    within it (``certify_log10``). Every array is stored bytes-backed; ``features`` and ``receipt`` hand out fresh
    read-only views and copies. ``representation`` is the digest of every column's exact bits and both code vectors."""
    population: Population
    columns: Mapping[str, np.ndarray]     # float64, canonical_roster() order, population order; bytes-backed
    selection: np.ndarray                 # uint8 codes into allele_frequency.SELECTION_STATES; bytes-backed
    claim: np.ndarray                     # uint8 codes into allele_frequency.CLAIM_STATES; bytes-backed
    receipt_bytes: bytes                  # canonical_bytes(receipt)
    representation: str = field(init=False)
    certified_ulp: float = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.population, Population):
            raise TypeError("population must be a columns.Population")
        if type(self.receipt_bytes) is not bytes:
            raise TypeError("receipt_bytes are the receipt's canonical bytes")
        receipt = json.loads(self.receipt_bytes.decode("utf-8"))
        if not isinstance(receipt, dict) or canonical_bytes(receipt) != self.receipt_bytes:
            raise ValueError("the receipt is not in canonical form")
        if receipt.get("schema") != RECEIPT_SCHEMA:
            raise ValueError(f"the receipt's schema is {receipt.get('schema')!r}, not {RECEIPT_SCHEMA!r}")
        n = len(self.population)
        if receipt.get("population") != {"population_id": self.population.population_id, "rows": n,
                                         "sha256": self.population.sha256}:
            raise ValueError("the receipt is for another population")
        roster = receipt.get("roster")
        if not isinstance(self.columns, Mapping) or list(self.columns) != roster:
            raise ValueError("the columns are not the receipt's roster, in its order")
        columns = {c: _immutable_vector(self.columns[c], np.float64, n, f"column {c}") for c in roster}
        object.__setattr__(self, "columns", MappingProxyType(columns))
        object.__setattr__(self, "selection", _immutable_vector(self.selection, np.uint8, n, "the selection codes"))
        object.__setattr__(self, "claim", _immutable_vector(self.claim, np.uint8, n, "the claim codes"))
        digests = {c: _column_sha256(v) for c, v in columns.items()}
        if _exact_digest({c: d for c, d in digests.items() if c not in ENVIRONMENT_DEPENDENT}) != \
                receipt.get("exact_columns_sha256"):
            raise ValueError("the column values do not reproduce the receipt's exact-column digest")
        for codes, key in ((self.selection, "selection_codes_sha256"), (self.claim, "claim_codes_sha256")):
            if _codes_sha256(codes) != receipt.get(key):
                raise ValueError(f"the {key[:-len('_codes_sha256')]} codes do not reproduce the receipt's digest")
        if receipt.get("environment_dependent") != _environment_dependent_entry():
            raise ValueError("the matrix was certified under another accuracy contract")
        object.__setattr__(self, "certified_ulp", certify_log10(columns["af_raw"], columns["af_log10"]))
        object.__setattr__(self, "representation", self._representation(digests))

    def _representation(self, digests: Mapping[str, str]) -> str:
        return sha256_bytes(canonical_bytes({"schema": REPRESENTATION_SCHEMA, "canonical_identity": self.identity,
                                             "columns": [[c, digests[c]] for c in self.columns],
                                             "selection_codes_sha256": _codes_sha256(self.selection),
                                             "claim_codes_sha256": _codes_sha256(self.claim)}))

    def verify(self) -> None:
        """Re-measure every array and the code now, at a consumer's boundary (bind, score, write): refused unless the
        columns are still the receipt's roster with its exact-column digest, both code vectors still reproduce the
        receipt's digests, the representation fixed when af_log10 was certified still reproduces (so af_log10's bits are
        the certified ones), and the code that produced the matrix is unchanged."""
        receipt = self.receipt
        digests = {c: _column_sha256(v) for c, v in self.columns.items()}
        moved = "refused: the canonical matrix's arrays are not the ones verified at its construction"
        if list(digests) != receipt["roster"] or _exact_digest(
                {c: d for c, d in digests.items() if c not in ENVIRONMENT_DEPENDENT}) != receipt["exact_columns_sha256"]:
            raise ValueError(f"{moved} (the exact columns)")
        if _codes_sha256(self.selection) != receipt["selection_codes_sha256"] or \
                _codes_sha256(self.claim) != receipt["claim_codes_sha256"]:
            raise ValueError(f"{moved} (the selection or claim codes)")
        if self._representation(digests) != self.representation:
            raise ValueError(f"{moved} (the representation)")
        repo = _repo_root()
        stale = sorted(s for s, d in receipt["procedure"].items() if symbol_closure_digest(repo, s) != d)
        if stale:
            raise ValueError(f"refused: the code that produced this matrix has changed ({stale})")

    @property
    def identity(self) -> str:
        return sha256_bytes(self.receipt_bytes)

    @property
    def receipt(self) -> dict[str, Any]:
        """A fresh copy of the receipt: editing it changes nothing."""
        return json.loads(self.receipt_bytes.decode("utf-8"))

    @property
    def features(self) -> pd.DataFrame:
        """A fresh zero-copy view of the columns: an in-place edit is refused (the arrays are read-only), and replacing a
        column replaces it in this view only."""
        return pd.DataFrame(dict(self.columns), copy=False)

    @property
    def sequence_resolved_primary(self) -> np.ndarray:
        return self.claim == A.CLAIM_STATES.index("eligible")

    def binding(self, roster: Iterable[str]) -> dict[str, Any]:
        roster = list(roster)
        unknown = [n for n in roster if n not in self.columns]
        if unknown or len(set(roster)) != len(roster) or not roster:
            raise ValueError(f"a fitted roster is a non-empty, duplicate-free subset of the canonical columns ({unknown})")
        receipt = self.receipt
        body = {"schema": BINDING_SCHEMA, "scope": BINDING_SCOPE, "canonical_identity": self.identity,
                "representation_sha256": self.representation, "semantics": receipt["semantics"],
                "selection_identity": receipt["selection"]["identity"], "policy": receipt["selection"]["policy"],
                "population_sha256": self.population.sha256, "roster": roster}
        return {**body, "digest": sha256_bytes(canonical_bytes(body))}


def prepare_qualified(frame: pd.DataFrame, population: Population, *, mode: PreparationMode, cohort_sha256: str,
                      gnomad: Parsed, reference: R.ReferenceStatus, reference_pin: R.ReferencePin,
                      policy: A.AFSelectionPolicy, registered_datasets: Iterable[str],
                      declared_sources: Mapping[str, str], expected_census: Mapping[str, int] | None = None,
                      chunk_rows: int = 262_144) -> CanonicalMatrix:
    """The canonical epoch-2 matrix of ``population`` from verified sources. ``frame`` is the labelled cohort frame in
    population order (DataPrepPipeline._load_and_label, optionally followed by the NON-allele-frequency annotation
    steps); its labels are never read. ``reference_pin`` is the reference the experiment committed to, declared, never
    defaulted: a reference status produced under any other pin is refused. The gnomAD bytes are hashed again here
    (integrity.require_verified_bytes); the cohort is declared by its digest, and the frame's derivation from those bytes
    is the caller's (QUALIFIED-FRAME-PROVENANCE-1, stated in the receipt's ``scope``)."""
    from genomic_variant_classifier.models.feature_semantics import AF_FAMILY, FeatureSemantics, derive_family
    from genomic_variant_classifier.models.variant_ensemble import engineer_features

    if mode is not PreparationMode.QUALIFIED:
        raise ValueError(f"prepare_qualified prepares mode {PreparationMode.QUALIFIED.value!r} only (got {mode!r}); "
                         "historical replay is DataPrepPipeline.run / run_v2, and no mode is inferred")
    registered = tuple(registered_datasets)
    for logical in (COHORT, A.PRIMARY, A.YMT_API, R.EVIDENCE_DATASET):
        check_registered(logical, registered)
    if not isinstance(population, Population):
        raise TypeError("population must be a columns.Population")
    if not isinstance(policy, A.AFSelectionPolicy):
        raise TypeError("policy must be an allele_frequency.AFSelectionPolicy")
    if type(reference) is not R.ReferenceStatus:
        raise TypeError("reference must be a verified reference_status.ReferenceStatus")
    if type(reference_pin) is not R.ReferencePin:
        raise TypeError("reference_pin must be the declared reference_status.ReferencePin")
    # 1. No substitution: each source is exactly the declared one -- gnomAD's bytes measured here, not asserted.
    expected_keys = {COHORT, A.PRIMARY, R.EVIDENCE_DATASET}
    if not isinstance(declared_sources, Mapping) or set(declared_sources) != expected_keys:
        raise ValueError(f"declare exactly the sources {sorted(expected_keys)}, each by its SHA-256")
    if type(cohort_sha256) is not str or not HEX64.match(cohort_sha256):
        raise ValueError("cohort_sha256 is the cohort file's SHA-256")
    actual = {COHORT: cohort_sha256, A.PRIMARY: I.require_verified_bytes(gnomad, what=A.PRIMARY),
              R.EVIDENCE_DATASET: reference.source_sha256}
    for logical, sha in actual.items():
        declared = declared_sources[logical]
        if type(declared) is not str or not HEX64.match(declared):
            raise ValueError(f"{logical}: the declared digest is not a SHA-256")
        if sha != declared:
            raise ValueError(f"{logical}: the bytes supplied have SHA-256 {sha}, not the declared {declared}; "
                             "an undeclared source is refused, never substituted")
    # 2. The reference status is for this population and the rule as it is now.
    if reference.population.sha256 != population.sha256:
        raise ValueError("the reference status is aligned to another population")
    if dict(reference.rule) != R.rule_identity():
        raise ValueError("the reference status was produced by another validation rule; it is invalid here")
    if dict(reference.reference) != reference_pin.as_dict():
        raise ValueError(f"the reference status was produced under another reference identity than the declared pin "
                         f"{reference_pin.digest}; it is invalid here")
    if R.result_digest(reference.ok.tolist(), reference.reason) != reference.result_sha256:
        raise ValueError("the reference status does not reproduce its result digest")
    # The frame: the population in order, no unqualified frequency source.
    if len(frame) != len(population) or "variant_id" not in frame.columns or \
            not (frame["variant_id"].astype(str).to_numpy() == np.asarray(population.row_ids, dtype=object)).all():
        raise ValueError("the frame's rows are not the population's rows in population order")
    present = [c for c in LEGACY_JOIN_COLUMNS if c in frame.columns]
    if present:
        raise ValueError(f"the frame carries columns the legacy gnomAD join writes {present}: 1000 Genomes and FinnGen "
                         "are unqualified population-frequency sources here; prepare from the labelled cohort frame")
    if "allele_freq" in frame.columns and bool(frame["allele_freq"].notna().any()):
        raise ValueError("the frame's allele_freq is not entirely missing: a value there comes from an unqualified "
                         "source (the cohort's own column or a fallback); only the strict selection supplies it")
    # 3-6. Materialise, select; the full population is kept.
    mat = A.materialize_allele_frequency(frame[["variant_id", "chrom", "pos", "ref", "alt"]], population, gnomad,
                                         registered_datasets=registered)
    sel = A.select_strict(mat, policy, reference_verified=np.asarray(reference.ok, dtype=bool),
                          reference_evidence_sha256=reference.result_sha256)
    census = sel.census()
    if expected_census is not None and dict(census) != dict(expected_census):
        raise ValueError(f"the population census differs from the declared expectation: {census} != "
                         f"{dict(expected_census)}; a difference is investigated, never absorbed")
    # 7. The single builder, on the selected value.
    # The builder receives only the columns it can read (builder_input_names: output-identical by construction), never
    # the labels, never a label-derived column, and the selected value as allele_freq. The caller's frame is not modified.
    readable = builder_input_names()
    keep = [c for c in frame.columns if c in readable and c not in ("label", "allele_freq", *SPLIT_DEPENDENT)]
    work = frame[keep].copy(deep=False)
    work["allele_freq"] = np.asarray(sel.af_model, dtype=np.float64)
    roster = canonical_roster()
    built = _build_columns(work, roster, chunk_rows=chunk_rows)
    del work
    columns = {c: readonly_array(built.pop(c)) for c in roster}       # bytes-backed, one column at a time
    # 8. Validate the AF family bit for bit against the one formula set.
    expected_af = derive_family(AF_FAMILY, np.asarray(sel.af_model, dtype=np.float64))
    for name, values in expected_af.items():
        got = columns[name]
        if not (np.array_equal(np.isnan(got), np.isnan(values)) and
                np.array_equal(got[~np.isnan(got)].view(np.uint64), values[~np.isnan(values)].view(np.uint64))):
            raise AssertionError(f"{name}: the engineered value is not derive_family's, bit for bit")
    # The environment-dependent columns: proven to be exactly the declared ones (af_log10 is certified by the constructor).
    if environment_dependent_columns() != frozenset(ENVIRONMENT_DEPENDENT):
        raise AssertionError(f"the environment-dependent columns are {sorted(environment_dependent_columns())}, not the "
                             f"declared {list(ENVIRONMENT_DEPENDENT)}")
    repo = _repo_root()
    receipt = {
        "schema": RECEIPT_SCHEMA, "mode": PreparationMode.QUALIFIED.value, "semantics": FeatureSemantics.V2_OBSERVED_AF.value,
        "population": {"population_id": population.population_id, "rows": len(population), "sha256": population.sha256},
        "sources": {k: actual[k] for k in sorted(actual)},
        "reference_status": {"identity": reference.identity, "rule": dict(reference.rule),
                             "reference_assembly": dict(reference.reference), "reference_pin_sha256": reference_pin.digest,
                             "result_sha256": reference.result_sha256},
        "allele_frequency": json.loads(json.dumps(mat.receipt)),
        "selection": {"identity": sel.identity, "policy": dict(sel.policy), "census": census},
        "roster": list(roster),
        "excluded": {"split_dependent": list(SPLIT_DEPENDENT),
                     "reason": "label-derived; materialised per split (train-only) by the split protocol"},
        "exact_columns_sha256": _exact_digest({c: _column_sha256(v) for c, v in columns.items()
                                               if c not in ENVIRONMENT_DEPENDENT}),
        "environment_dependent": _environment_dependent_entry(),
        "selection_codes_sha256": _codes_sha256(sel.selection),
        "claim_codes_sha256": _codes_sha256(sel.claim),
        "procedure": {s: symbol_closure_digest(repo, s) for s in PROCEDURE},
        "scope": QUALIFICATION_SCOPE,
        "training": TRAINING,
    }
    return CanonicalMatrix(population, columns, readonly_array(np.asarray(sel.selection, np.uint8)),
                           readonly_array(np.asarray(sel.claim, np.uint8)), canonical_bytes(receipt))


# ----------------------------------------------------------------------------------------------------- persistence
class StoredCanonical(NamedTuple):
    identity: str            # the logical identity (the receipt's digest)
    payload_sha256: str      # the stored file's SHA-256


def _table_bytes(cm: CanonicalMatrix) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = {"variant_id": pa.array(cm.population.row_ids, type=pa.string()),
            "_selection": pa.array(cm.selection, type=pa.uint8()), "_claim": pa.array(cm.claim, type=pa.uint8())}
    cols.update({c: pa.array(v, type=pa.float64()) for c, v in cm.columns.items()})
    buf = io.BytesIO()
    pq.write_table(pa.table(cols), buf, compression="zstd")
    return buf.getvalue()


def _certificate(cm: CanonicalMatrix) -> dict[str, Any]:
    return {"af_log10_max_ulp": cm.certified_ulp, "bound_ulp": LOG10_ULP_BOUND, "reference_bits": REFERENCE_PRECISION_BITS}


def write_canonical(cm: CanonicalMatrix, directory: Path) -> StoredCanonical:
    """Store ``cm`` as ``<identity>/<payload>.parquet`` with its record ``<identity>/<payload>.receipt.json``
    (integrity.store): the record states the logical receipt, the representation, this payload's af_log10 certificate and
    the payload's SHA-256 and size. ``cm`` is verified first. The same bytes again are a no-op only when the stored payload
    is present and intact; a second certified representation of the same identity is stored beside the first; a record
    without its payload, or with another payload, is refused -- never replaced, never repaired."""
    if type(cm) is not CanonicalMatrix:
        raise TypeError("cm must be a CanonicalMatrix")
    cm.verify()
    table = _table_bytes(cm)
    record = {"schema": STORE_SCHEMA, "identity": cm.identity, "representation_sha256": cm.representation,
              "certificate": _certificate(cm), "receipt": cm.receipt, "payload": I.payload_entry(table)}
    data = (json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    return StoredCanonical(cm.identity, I.store(Path(directory), cm.identity, table, data, what="canonical matrix"))


def payload_for(directory: Path, identity: str, representation_sha256: str) -> str:
    """The stored payload of ``identity`` whose record states ``representation_sha256`` (a source binding's): the
    deterministic selector for exact replay. Refused when none or more than one does."""
    if type(representation_sha256) is not str or not HEX64.match(representation_sha256):
        raise ValueError("representation_sha256 is a lower-case SHA-256")
    folder = Path(directory) / identity
    found = [p for p in I.stored_payloads(Path(directory), identity)
             if json.loads((folder / f"{p}{I.RECEIPT_SUFFIX}").read_text(encoding="ascii")).get("representation_sha256")
             == representation_sha256]
    if len(found) != 1:
        raise ValueError(f"{len(found)} stored payloads of {identity} have the representation {representation_sha256}; "
                         "exactly one is required")
    return found[0]


def load_canonical(directory: Path, identity: str, payload_sha256: str, *,
                   population: Population | None = None) -> CanonicalMatrix:
    """One stored payload of a canonical matrix, reused only after verifying the payload bytes against its record and its
    name, the receipt against the identity, the code identities against the code NOW, the population (when given), the
    table's columns, and -- by the constructor -- every exact column, both code vectors and af_log10's certificate
    (recomputed here, for this payload); the representation and the certificate must be the record's."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if type(identity) is not str or not HEX64.match(identity):
        raise ValueError("identity is a lower-case SHA-256")
    raw_record, raw = I.read_stored(Path(directory), identity, payload_sha256, what="canonical matrix")
    record = json.loads(raw_record.decode("ascii"))
    if record.get("schema") != STORE_SCHEMA or record.get("identity") != identity:
        raise ValueError("not a canonical-matrix record for this identity")
    receipt = record["receipt"]
    receipt_bytes = canonical_bytes(receipt)
    if sha256_bytes(receipt_bytes) != identity:
        raise ValueError("the stored receipt does not reproduce its identity")
    repo = _repo_root()
    stale = sorted(s for s, d in receipt["procedure"].items() if symbol_closure_digest(repo, s) != d)
    if stale:
        raise ValueError(f"reuse refused: the code that produced this matrix has changed ({stale})")
    table = pq.read_table(pa.BufferReader(raw))
    pop = Population(receipt["population"]["population_id"], tuple(table["variant_id"].to_pylist()))
    if pop.sha256 != receipt["population"]["sha256"]:
        raise ValueError("the stored rows are not the receipt's population")
    if population is not None and population.sha256 != pop.sha256:
        raise ValueError("reuse refused: the stored matrix is for another population")
    roster = receipt["roster"]
    if table.column_names != ["variant_id", "_selection", "_claim", *roster]:
        raise ValueError("the stored columns are not the receipt's roster")
    if any(table.schema.field(c).type != pa.float64() for c in roster) or \
            any(table.schema.field(c).type != pa.uint8() for c in ("_selection", "_claim")):
        raise ValueError("the stored columns are not float64 values and uint8 codes")
    columns = {c: readonly_array(table[c].to_numpy(zero_copy_only=False)) for c in roster}
    selection = readonly_array(table["_selection"].to_numpy(zero_copy_only=False))
    claim = readonly_array(table["_claim"].to_numpy(zero_copy_only=False))
    del table
    cm = CanonicalMatrix(pop, columns, selection, claim, receipt_bytes)          # verified and re-certified here
    if cm.representation != record.get("representation_sha256"):
        raise ValueError("the stored values do not reproduce the record's representation")
    if record.get("certificate") != _certificate(cm):
        raise ValueError("the stored payload's af_log10 certificate does not reproduce")
    return cm


# ------------------------------------------------------------------------------------------------ fit and serving
def bind_fit(ensemble: Any, cm: CanonicalMatrix) -> dict[str, Any]:
    """Record on a fitted ensemble the canonical matrix it was fitted on -- its logical identity AND its representation
    (the exact bits, af_log10's included); its roster is the ensemble's fitted columns. ``cm`` is verified first. The
    ensemble saves the binding (VariantEnsemble.save) and replay verifies it (``require_source_binding``)."""
    if type(cm) is not CanonicalMatrix:
        raise TypeError("cm must be a CanonicalMatrix")
    cm.verify()
    roster = list(getattr(ensemble, "tabular_columns_", None) or ())
    binding = cm.binding(roster)
    if getattr(ensemble, "source_binding_", None) is not None:
        raise ValueError("this ensemble is already bound to a source; a binding is never replaced")
    ensemble.source_binding_ = binding
    return binding


def require_source_binding(ensemble: Any, cm: CanonicalMatrix) -> None:
    """Refuse unless ``ensemble`` was bound to exactly ``cm``: the same source, policy, population, semantics, roster and
    representation (EXACT REPLAY, ``BINDING_SCOPE``)."""
    binding = getattr(ensemble, "source_binding_", None)
    if not isinstance(binding, Mapping):
        raise PermissionError("the ensemble records no source binding; it cannot score a qualified canonical matrix")
    body = {k: v for k, v in binding.items() if k != "digest"}
    if sha256_bytes(canonical_bytes(body)) != binding.get("digest"):
        raise PermissionError("the ensemble's source binding does not reproduce its digest")
    if binding.get("schema") != BINDING_SCHEMA or binding.get("scope") != BINDING_SCOPE:
        raise PermissionError(f"the ensemble's source binding is not a {BINDING_SCHEMA} exact-replay binding")
    roster = list(getattr(ensemble, "tabular_columns_", None) or ())
    if roster != list(binding["roster"]):
        raise PermissionError("the ensemble's fitted columns are not the roster it was bound with")
    if cm.binding(binding["roster"]) != dict(binding):
        expected = cm.binding(binding["roster"])
        differs = sorted(k for k in expected if k != "digest" and expected[k] != binding.get(k))
        raise PermissionError(f"refused: the canonical matrix differs from the one the ensemble was fitted on ({differs})")


def score_canonical(ensemble: Any, cm: CanonicalMatrix) -> np.ndarray:
    """EXACT REPLAY: the ensemble's probabilities on ``cm``, only after ``cm.verify()`` and ``require_source_binding``.
    Not general inference (``BINDING_SCOPE``)."""
    if type(cm) is not CanonicalMatrix:
        raise TypeError("cm must be a CanonicalMatrix")
    cm.verify()
    require_source_binding(ensemble, cm)
    return ensemble.predict_proba(cm.features[list(ensemble.tabular_columns_)])
