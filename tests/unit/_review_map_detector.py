"""Find every ClinVar review-status tier map in the repository, by content.

WHY THIS IS CONTENT-BASED AND NOT NAME-BASED
============================================
The first version of the single-definition guard, written 2026-07-24, matched
only assignments whose target was the `ast.Name` `REVIEW_STATUS_TIER`. It missed
`src/genomic_variant_classifier/monitoring/clinvar_tracker.py:160`, which defines
`REVIEW_TIER` -- a six-key class attribute on `ClinVarTracker`, imported live by
`training/continual_trainer.py:126`,
`agent_layer/agents/reclassification_sentinel_agent.py:86` and
`scripts/run_drift_monitor.py:421`. A guard that can be evaded by renaming the
variable does not pin the class of defect; it pins one spelling of it.

This detector keys on what a tier map IS rather than what it is called:

    a dictionary literal
    whose keys are constant strings, at least two of which are recognised
      ClinVar review-status vocabulary after normalisation
    and whose values are all constant integers

Under that rule the repository contains EIGHT definitions, not seven. The eighth
is the one the name-scoped guard could not see.

WHAT IT DELIBERATELY DOES NOT FLAG
----------------------------------
  * An IMPORT of a map. `from ... import REVIEW_STATUS_TIER as T` is a reference,
    not a definition. Flagging imports would make every legitimate consumer a
    violation, and a guard that fires on correct code is disabled within a week.
  * Review-status strings inside a docstring, comment or plain string. The
    modules that discuss this defect quote the vocabulary constantly.
  * A dictionary with review-status keys and NON-integer values -- a semantics
    map, a rationale map, a display-label map. Those are not tier maps and the
    project deliberately keeps one alongside the tier map.
  * A dictionary with fewer than three matching keys, which is a threshold
    chosen so that an incidental two-entry lookup is not mistaken for a map.

WHY LINE NUMBERS ARE NOT PART OF THE IDENTITY
---------------------------------------------
The inventory that consumes this detector keys on `relative/path.py::NAME`. Line
numbers shift whenever anything above them is edited, so an inventory keyed on
them would turn red on unrelated changes and train the reader to ignore it --
the lesson recorded from the Run-16 preflight gate on 2026-07-20. Line numbers
are reported in failure messages for locatability and nowhere else.

Acronyms on first use. AST = abstract syntax tree. ClinVar = the National Center
for Biotechnology Information's Clinical Variation archive.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

__all__ = [
    "REVIEW_STATUS_VOCABULARY",
    "MIN_MATCHING_KEYS",
    "SEARCH_ROOTS",
    "find_tier_map_definitions",
    "find_with_line_numbers",
]

#: Every ClinVar review-status string this project has observed or supported,
#: including the spellings ClinVar has since renamed. Membership here decides
#: whether a dictionary looks like a tier map; it does NOT decide what tier
#: anything gets, which is owned solely by
#: src/genomic_variant_classifier/data/review_status.py.
REVIEW_STATUS_VOCABULARY: frozenset[str] = frozenset({
    "practice guideline",
    "reviewed by expert panel",
    "criteria provided, multiple submitters, no conflicts",
    "criteria provided, single submitter",
    "criteria provided, conflicting classifications",
    "criteria provided, conflicting interpretations",
    "no assertion criteria provided",
    "no classification provided",
    "no classification for the single variant",
    "no classification for the individual variant",
    "no classifications from unflagged records",
})

#: A dictionary needs at least this many recognised keys to count as a tier map.
#: Two would catch an incidental pair; four would miss a partial map. Three is
#: the smallest threshold that has never produced a false positive in this tree.
MIN_MATCHING_KEYS = 3

SEARCH_ROOTS = ("src", "scripts")

_WHITESPACE = re.compile(r"\s+")


def _normalise(value: str) -> str:
    """The same transformation review_status.normalise applies.

    Duplicated rather than imported so the detector has no dependency on the
    module it is auditing. A guard that imports its subject cannot report on a
    tree where its subject is broken.
    """
    return _WHITESPACE.sub(" ", value.lower().replace("_", " ")).strip()


def _looks_like_a_tier_map(node: ast.Dict) -> bool:
    if not node.keys:
        return False
    keys: list[str] = []
    for key in node.keys:
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            return False                      # a non-literal key: not a static map
        keys.append(key.value)
    for value in node.values:
        # bool is a subclass of int; a flag map is not a tier map.
        if not (isinstance(value, ast.Constant)
                and isinstance(value.value, int)
                and not isinstance(value.value, bool)):
            return False
    matching = sum(1 for k in keys if _normalise(k) in REVIEW_STATUS_VOCABULARY)
    return matching >= MIN_MATCHING_KEYS


def _binding_name(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    """The name a dictionary literal is bound to, walking up the syntax tree.

    Handles module-level assignment, annotated assignment, and assignment inside
    a class body, which is how ClinVarTracker.REVIEW_TIER is written.
    """
    current: ast.AST | None = node
    for _ in range(8):                        # bounded: a binding is never deep
        parent = parents.get(current)         # type: ignore[arg-type]
        if parent is None:
            return "<unbound>"
        if isinstance(parent, ast.Assign):
            for target in parent.targets:
                if isinstance(target, ast.Name):
                    return target.id
                if isinstance(target, ast.Attribute):
                    return target.attr
            return "<unbound>"
        if isinstance(parent, ast.AnnAssign):
            target = parent.target
            if isinstance(target, ast.Name):
                return target.id
            if isinstance(target, ast.Attribute):
                return target.attr
            return "<unbound>"
        current = parent
    return "<unbound>"


def _scan_file(path: Path, root: Path) -> list[tuple[str, str, int]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        # A file that cannot be parsed is reported as unscannable by the caller
        # rather than silently counted as containing nothing.
        return []
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    found: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and _looks_like_a_tier_map(node):
            rel = path.relative_to(root).as_posix()
            found.append((rel, _binding_name(node, parents), node.lineno))
    return found


def find_with_line_numbers(root: Path) -> list[tuple[str, str, int]]:
    """Every definition as (relative_path, binding_name, line_number), sorted."""
    out: list[tuple[str, str, int]] = []
    for sub in SEARCH_ROOTS:
        base = root / sub
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            out.extend(_scan_file(path, root))
    return sorted(out)


def find_tier_map_definitions(root: Path) -> tuple[str, ...]:
    """Every definition as 'relative/path.py::BINDING_NAME', sorted.

    Line numbers are deliberately absent from the identity. See the module
    docstring.
    """
    return tuple(f"{rel}::{name}" for rel, name, _ in find_with_line_numbers(root))


def unparseable_files(root: Path) -> tuple[str, ...]:
    """Files the detector could not parse, so a scan gap is never silent."""
    bad: list[str] = []
    for sub in SEARCH_ROOTS:
        base = root / sub
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (SyntaxError, UnicodeDecodeError):
                bad.append(path.relative_to(root).as_posix())
    return tuple(bad)
