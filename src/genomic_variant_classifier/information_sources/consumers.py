"""Observed consumption of the tabular feature contract, measured from the repository's own code (stage 1D-1a).

The census must cover what the code CONSUMES, not only what it DECLARES (owner ruling 2026-11-09, section 7):

    F_observed(surface)  must be a subset of  F_reviewed      for every predictive surface.

Two measurements make that checkable rather than asserted:

1. Surface rules. Each surface names the exact code text that decides which columns reach it, and the set is computed
   from that text:
     engineered     run the repository's engineer_features on an empty frame and read its columns;
     contract       the anchored statement consumes the declared contract verbatim (list(TABULAR_FEATURES), the fail-loud
                    order guard, the scaler over the guarded matrix); the observed set is the contract;
     comprehension  the anchored text is a list comprehension over ``X_train.columns``; it is parsed, its shape is checked
                    (names, attributes, constants, comparisons only) and it is evaluated against the contract;
     none           the surface consumes no tabular feature; discovery (below) must classify no site to it.
   Every anchor must occur exactly once in its file, so an edit that moves the decision elsewhere fails loudly.

2. Discovery closure. Every module under src/ and scripts/ that references a feature-contract or graph-construction
   symbol (by name, attribute or import -- aliases included) is found by parsing; the census must classify every such
   site, and must not classify a site that no longer exists. A new consumer therefore cannot appear unreviewed.

Nothing here imports a model library except the "engineered" rule, which imports the repository's own feature builder and
refuses to measure a module loaded from anywhere but the repository being measured.
"""
from __future__ import annotations

import ast
import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

logger = logging.getLogger(__name__)

CONTRACT_FILE = "src/genomic_variant_classifier/models/variant_ensemble.py"
RUN_DRIVER = "scripts/run_phase2_eval.py"

#: The closed surface vocabulary. The first ten are the owner's list (ruling 2026-11-09, section 7); continual_training
#: is the eleventh because training/continual_trainer.py refits on engineer_features output.
SURFACES: tuple[str, ...] = (
    "feature_engineering", "tabular_fit", "scaler_fit", "gnn_node_attributes", "hetero_gnn_node_attributes",
    "split_persistence", "inference", "calibration", "export", "evaluation", "continual_training",
)

#: Consumers of non-tabular information (graph structure and supervision), the vocabulary of information-source entries.
GRAPH_CONSUMERS: tuple[str, ...] = (
    "gnn_edges", "gnn_edge_attributes", "gnn_supervision", "hetero_gnn_relations", "hetero_gnn_supervision",
)
HETERO_SCORER = "src/genomic_variant_classifier/models/hetero_gnn_scorer.py"
GNN_MODULE = "src/genomic_variant_classifier/models/gnn.py"

#: Symbols whose use marks a consumer of the feature contract or of graph construction.
DISCOVERY_SYMBOLS: frozenset[str] = frozenset({
    "TABULAR_FEATURES", "INFERENCE_FEATURE_COLUMNS", "engineer_features", "_engineer_features",
    "build_pyg_dataset", "train_gnn_pipeline", "build_hetero_focal_graph", "HeteroGNNTrainer",
    "load_kg_edge_specs", "StringDBGraph", "string_graph_to_edges",
})

#: The closed site classification vocabulary (a site may carry several).
SITE_CLASSES: frozenset[str] = frozenset(
    {f"surface:{s}" for s in SURFACES} | {"definition", "diagnostic", "monitoring", "experiment"})


@dataclass(frozen=True)
class SurfaceRule:
    surface: str
    kind: str            # engineered | contract | comprehension | none
    path: str | None
    anchor: str | None
    note: str

    def __post_init__(self) -> None:
        if self.surface not in SURFACES:
            raise ValueError(f"unknown surface {self.surface!r}")
        if self.kind not in {"engineered", "contract", "comprehension", "none"}:
            raise ValueError(f"{self.surface}: unknown rule kind {self.kind!r}")
        if (self.kind == "none") != (self.path is None and self.anchor is None):
            raise ValueError(f"{self.surface}: a 'none' rule has no anchor, every other rule has one")


#: The measured rules. The anchors are the code that decides each surface's columns at 7cb70ec2 (main after PR #71).
SURFACE_RULES: tuple[SurfaceRule, ...] = (
    SurfaceRule("feature_engineering", "engineered", CONTRACT_FILE,
                "def engineer_features(df: pd.DataFrame) -> pd.DataFrame:",
                "the single feature builder; its output columns, measured by running it"),
    SurfaceRule("tabular_fit", "contract", "src/genomic_variant_classifier/data/real_data_prep.py",
                "engineer_features output violates the TABULAR_FEATURES contract -- ",
                "the training matrix is refused unless its columns equal the contract in order; VariantEnsemble.fit "
                "records X_tab.columns as tabular_columns_"),
    SurfaceRule("scaler_fit", "contract", "src/genomic_variant_classifier/data/real_data_prep.py",
                "X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=cols)",
                "StandardScaler fitted on every column of the guarded training matrix"),
    SurfaceRule("gnn_node_attributes", "comprehension", RUN_DRIVER,
                'c for c in X_train.columns if c != "gnn_score"',
                "gene-level means of these columns over the training rows are the STRING-GNN node attributes"),
    SurfaceRule("hetero_gnn_node_attributes", "comprehension", RUN_DRIVER,
                'c for c in X_train.columns if c not in ("gnn_score", "hetero_gnn_score")',
                "node attributes of the heterogeneous knowledge-graph GNN"),
    SurfaceRule("split_persistence", "contract", "src/genomic_variant_classifier/data/real_data_prep.py",
                'X_train.to_parquet(out / "X_train.parquet", index=False, compression="zstd")',
                "every column of the scaled matrices is written; the run driver re-writes them after GNN injection"),
    SurfaceRule("inference", "contract", "src/genomic_variant_classifier/api/pipeline.py",
                "INFERENCE_FEATURE_COLUMNS: list[str] = list(TABULAR_FEATURES)",
                "the inference pipeline renders the contract (or the fitted scaler's feature_names_in_)"),
    SurfaceRule("calibration", "none", None, None,
                "calibration consumes member and blend scores, never feature columns; no discovered site"),
    SurfaceRule("export", "none", None, None,
                "no exporter reads feature columns; no discovered site"),
    SurfaceRule("evaluation", "contract", CONTRACT_FILE,
                "self.tabular_columns_ = tuple(X_tab.columns)",
                "prediction and evaluation consume exactly the columns recorded at fit"),
    SurfaceRule("continual_training", "engineered", "src/genomic_variant_classifier/training/continual_trainer.py",
                "X_new = engineer_features(new_clinvar)",
                "continual refits build their matrix with engineer_features"),
)


def _read(repo: Path, rel: str) -> str:
    path = repo / rel
    if not path.is_file():
        raise ValueError(f"surface anchor file is missing: {rel}")
    return path.read_text(encoding="utf-8")


def require_unique_anchor(repo: Path, rel: str, anchor: str) -> int:
    """Return the 1-based line of `anchor`; refuse unless it occurs exactly once in the file."""
    text = _read(repo, rel)
    count = text.count(anchor)
    if count != 1:
        raise ValueError(f"anchor occurs {count} times (must be exactly once) in {rel}: {anchor!r}")
    return text[: text.index(anchor)].count("\n") + 1


_ALLOWED_COMPREHENSION_NODES = (
    ast.Expression, ast.ListComp, ast.comprehension, ast.Name, ast.Attribute, ast.Load, ast.Store, ast.Compare,
    ast.Constant, ast.Tuple, ast.NotEq, ast.Eq, ast.In, ast.NotIn, ast.BoolOp, ast.And, ast.Or,
)


def evaluate_comprehension(anchor: str, contract: Sequence[str]) -> list[str]:
    """Evaluate `[<anchor>]` with X_train.columns bound to the contract. Only a pure selection expression is accepted."""
    tree = ast.parse(f"[{anchor}]", mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_COMPREHENSION_NODES):
            raise ValueError(f"comprehension rule contains a disallowed construct {type(node).__name__}: {anchor!r}")
    comp = tree.body
    if not isinstance(comp, ast.ListComp) or len(comp.generators) != 1:
        raise ValueError(f"not a single-generator list comprehension: {anchor!r}")
    it = comp.generators[0].iter
    if not (isinstance(it, ast.Attribute) and it.attr == "columns" and isinstance(it.value, ast.Name)
            and it.value.id == "X_train"):
        raise ValueError(f"the comprehension must iterate X_train.columns: {anchor!r}")
    namespace = {"X_train": SimpleNamespace(columns=list(contract))}
    result = eval(compile(tree, "<surface-rule>", "eval"), {"__builtins__": {}}, namespace)  # noqa: S307 -- shape-checked
    if not all(isinstance(c, str) for c in result):
        raise ValueError("comprehension rule produced a non-string column")
    return list(result)


def engineered_columns(repo: Path) -> list[str]:
    """Run the REPOSITORY's engineer_features on an empty one-row frame; refuse a module loaded from elsewhere."""
    import pandas as pd

    module = importlib.import_module("genomic_variant_classifier.models.variant_ensemble")
    loaded = Path(module.__file__).resolve()
    expected = (repo / CONTRACT_FILE).resolve()
    if loaded != expected:
        raise ValueError(f"engineer_features was imported from {loaded}, not from the measured repository {expected}")
    frame = module.engineer_features(pd.DataFrame(index=pd.RangeIndex(1)))
    return [str(c) for c in frame.columns]


def observe_surfaces(repo: Path, contract: Sequence[str], *,
                     engineered: Sequence[str] | None = None) -> dict[str, dict]:
    """Measure every surface. `engineered` may be supplied by a caller that already ran the builder (tests)."""
    contract = list(contract)
    out: dict[str, dict] = {}
    built = list(engineered) if engineered is not None else None
    for rule in SURFACE_RULES:
        line = None
        if rule.anchor is not None:
            line = require_unique_anchor(repo, rule.path, rule.anchor)
        if rule.kind == "engineered":
            if built is None:
                built = engineered_columns(repo)
            observed = list(built)
        elif rule.kind == "contract":
            observed = list(contract)
        elif rule.kind == "comprehension":
            observed = evaluate_comprehension(rule.anchor, contract)
        else:
            observed = []
        out[rule.surface] = {"rule": rule.kind, "path": rule.path, "anchor": rule.anchor, "line": line,
                             "note": rule.note, "observed": sorted(observed)}
    if tuple(out) != SURFACES:
        raise ValueError("surface rules do not cover the surface vocabulary exactly once, in order")
    return out


def discover_sites(repo: Path, roots: Sequence[str] = ("src", "scripts")) -> dict[str, list[str]]:
    """Every module that references a discovery symbol: {relative path: sorted symbols}. Aliased imports count."""
    sites: dict[str, set[str]] = {}
    for root in roots:
        base = repo / root
        if not base.is_dir():
            raise ValueError(f"discovery root is missing: {root}")
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(repo).as_posix()
            try:
                tree = ast.parse(path.read_bytes(), filename=rel)
            except SyntaxError as exc:
                raise ValueError(f"cannot parse {rel} for consumer discovery: {exc}") from exc
            found: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    found.update(a.name for a in node.names if a.name in DISCOVERY_SYMBOLS)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in DISCOVERY_SYMBOLS:
                    found.add(node.id)
                elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and \
                        node.attr in DISCOVERY_SYMBOLS:
                    found.add(node.attr)
                elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in DISCOVERY_SYMBOLS:
                    found.add(node.name)
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    found.update(t.id for t in targets if isinstance(t, ast.Name) and t.id in DISCOVERY_SYMBOLS)
            if found:
                sites[rel] = found
    return {k: sorted(v) for k, v in sorted(sites.items())}


def check_site_classification(discovered: Mapping[str, Sequence[str]], recorded: Mapping[str, Mapping]) -> None:
    """The recorded site table must equal discovery exactly, and every classification must be in the vocabulary."""
    missing = sorted(set(discovered) - set(recorded))
    stale = sorted(set(recorded) - set(discovered))
    if missing or stale:
        raise ValueError(f"consumer sites unaccounted for: unclassified {missing}; classified but absent {stale}")
    for rel, record in recorded.items():
        if list(record.get("symbols", [])) != list(discovered[rel]):
            raise ValueError(f"{rel}: recorded symbols {record.get('symbols')} differ from discovered {discovered[rel]}")
        classes = record.get("classes")
        if not isinstance(classes, list) or not classes or len(set(classes)) != len(classes):
            raise ValueError(f"{rel}: classes must be a non-empty list without duplicates")
        unknown = [c for c in classes if c not in SITE_CLASSES]
        if unknown:
            raise ValueError(f"{rel}: unknown site classes {unknown}")
        if not isinstance(record.get("reason"), str) or not record["reason"].strip():
            raise ValueError(f"{rel}: classification needs a reason")
    rule_paths = {r.surface: r.path for r in SURFACE_RULES}
    for surface, path in rule_paths.items():
        claimed = sorted(rel for rel, rec in recorded.items() if f"surface:{surface}" in rec["classes"])
        if path is None and claimed:
            raise ValueError(f"surface {surface} is measured as consuming nothing, but sites claim it: {claimed}")
        if path is not None and path not in claimed:
            raise ValueError(f"surface {surface}: its rule's file {path} is not classified to it")


def _function(tree: ast.Module, name: str, *, cls: str | None = None) -> ast.FunctionDef:
    scope = tree.body
    if cls is not None:
        owner = next((n for n in scope if isinstance(n, ast.ClassDef) and n.name == cls), None)
        if owner is None:
            raise ValueError(f"class {cls} is missing")
        scope = owner.body
    fn = next((n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if fn is None:
        raise ValueError(f"function {name} is missing")
    return fn


def _single_assignment(fn: ast.FunctionDef, target: str) -> ast.expr:
    values = [n.value for n in ast.walk(fn) if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == target]
    if len(values) != 1:
        raise ValueError(f"{fn.name}: expected exactly one assignment to {target}, found {len(values)}")
    return values[0]


def observe_graph_sources(repo: Path) -> dict:
    """Measure the graph information sources from the code: the knowledge-graph adapter table that maps a --kg-edges
    source to its relation, and the STRING evidence channels that become GNN edge attributes."""
    tree = ast.parse(_read(repo, HETERO_SCORER), filename=HETERO_SCORER)
    table = _single_assignment(_function(tree, "load_kg_edge_specs"), "adapters")
    if not isinstance(table, ast.Dict):
        raise ValueError("load_kg_edge_specs.adapters is not a dict literal")
    kg: dict[str, str] = {}
    for key, value in zip(table.keys, table.values):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str) and isinstance(value, ast.Tuple)
                and len(value.elts) == 2 and isinstance(value.elts[1], ast.Constant)
                and isinstance(value.elts[1].value, str)):
            raise ValueError("load_kg_edge_specs.adapters has an entry that is not (adapter, 'relation')")
        kg[key.value] = value.elts[1].value
    gtree = ast.parse(_read(repo, GNN_MODULE), filename=GNN_MODULE)
    channels = ast.literal_eval(_single_assignment(_function(gtree, "build", cls="StringDBGraph"), "_CHANNELS"))
    if not isinstance(channels, list) or not all(isinstance(c, str) for c in channels):
        raise ValueError("StringDBGraph.build._CHANNELS is not a list of names")
    threshold_anchor = 'links_df = links_df[links_df["combined_score"] >= self.threshold]'
    require_unique_anchor(repo, GNN_MODULE, threshold_anchor)
    return {"kg_relation_sources": dict(sorted(kg.items())), "string_edge_attribute_channels": list(channels),
            "string_edge_selection": threshold_anchor}


def required_information_sources(graph: Mapping) -> list[str]:
    """The information-source entries the census must hold, derived from the measured graph inventory."""
    names = [f"kg_relation:{s}" for s in graph["kg_relation_sources"]]
    names += [f"string_edge_attribute:{c}" for c in graph["string_edge_attribute_channels"]]
    names += ["string_relation:combined_score_threshold", "supervision:gnn_label", "supervision:hetero_gnn_label"]
    return names
