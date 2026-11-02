"""nannyml is imported in exactly one place in src/ and scripts/: performance_estimator.import_nannyml. Author: Monzia Moodie

That function switches nannyml's usage telemetry off and suppresses the implicit .env search its import performs (revision b of
WARNINGS-AND-RESOURCES-1, 2026-11-02). Any other import of nannyml skips both, so a second route is refused here by an AST census of
every Python file under src/ and scripts/ -- `import nannyml`, `from nannyml... import`, and importlib.import_module / __import__ with
the literal name. tests_drift/ is outside this census on purpose: its negative controls import nannyml directly to prove the guard
is necessary.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
GUARD = Path("src/genomic_variant_classifier/monitoring/performance_estimator.py")


def _is_nannyml(name) -> bool:
    return isinstance(name, str) and (name == "nannyml" or name.startswith("nannyml."))


def _imports(tree: ast.AST) -> list:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, a.name) for a in node.names if _is_nannyml(a.name)]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and _is_nannyml(node.module):
            found.append((node.lineno, node.module))
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
            if name in ("import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant) and _is_nannyml(node.args[0].value):
                found.append((node.lineno, node.args[0].value))
    return found


def _census() -> dict:
    out = {}
    files = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").rglob("*.py"))
    assert len(files) > 100, len(files)
    for path in files:
        found = _imports(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        if found:
            out[str(path.relative_to(ROOT)).replace("\\", "/")] = found
    return out


def test_nannyml_is_imported_only_by_the_guard():
    census = _census()
    assert set(census) == {GUARD.as_posix()}, census
    assert [name for _, name in census[GUARD.as_posix()]] == ["nannyml"]


def test_the_census_sees_every_form_of_import():
    # Negative controls: each form is found, so the positive result above is not vacuous.
    for source in ("import nannyml", "import nannyml.usage_logging as u", "from nannyml import CBPE", "from nannyml.chunk import Chunker",
                   "import importlib\nimportlib.import_module('nannyml')", "__import__('nannyml')"):
        assert _imports(ast.parse(source)), source
    for source in ("import nannymlx", "from . import nannyml", "importlib.import_module(name)"):
        assert not _imports(ast.parse(source)), source
