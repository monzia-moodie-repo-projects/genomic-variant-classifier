"""Code identity: the canonical bytes, their SHA-256, and the comment-insensitive symbol-closure digest of a function or
class -- its code plus every module-level binding it transitively references.

Moved here VERBATIM from information_sources/census.py (stage 1D-1c.1C, set "au"); census.py re-exports every name, so
each existing caller, and the import line of each sealed module, is unchanged. The reason is measured: the models layer
binds its fitted projections to code closures (refinement R7), and the models layer is in the inference service's
static import closure. census.py reaches consumers.py and lineage.py, and through them the data layer, so importing it
from models/ put 48 modules into the API image (tests/unit/test_docker_image_covers_the_api.py). This module imports the
standard library only.

Set "au.1" (owner ruling ff77015c, section 6) split ``symbol_closure_digest`` into the file read and
``closure_digest_of_text`` so a caller can hash the bytes it read and compute the closure from exactly those bytes; every
digest is unchanged (tests/unit/test_corrections_1d1c1c_r1.py compares them for every symbol a committed record binds).

Author: Monzia Moodie
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import logging
import textwrap
import tokenize
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

__all__ = ["canonical_bytes", "closure_digest_of_text", "parse_symbol", "sha256_bytes", "source_text",
           "symbol_closure_digest"]


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _strip_comments(segment: str) -> str:
    """Remove comment tokens, trailing blanks and blank lines; dedent. Stable across Python 3.11 and 3.12 (f-string
    tokenisation changed in 3.12, but COMMENT tokens did not)."""
    text = textwrap.dedent(segment)
    lines = text.splitlines(keepends=True)
    cuts: dict[int, int] = {}
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            row, col = tok.start
            cuts[row] = min(cuts.get(row, col), col)
    kept = []
    for i, line in enumerate(lines, start=1):
        line = line[: cuts[i]] if i in cuts else line
        line = line.rstrip()
        if line.strip():
            kept.append(line)
    return "\n".join(kept)


def _module_table(tree: ast.Module) -> dict[str, ast.AST]:
    """Top-level bindings, including those made inside top-level try/if blocks."""
    table: dict[str, ast.AST] = {}

    def visit(body: Sequence[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                table.setdefault(node.name, node)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            table.setdefault(n.id, node)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                table.setdefault(node.target.id, node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    table.setdefault((alias.asname or alias.name).split(".")[0], node)
            elif isinstance(node, ast.Try):
                visit(node.body)
                for h in node.handlers:
                    visit(h.body)
                visit(node.orelse)
                visit(node.finalbody)
            elif isinstance(node, ast.If):
                visit(node.body)
                visit(node.orelse)
    visit(tree.body)
    return table


def _resolve(tree: ast.Module, qualname: str) -> ast.AST:
    scope: Sequence[ast.stmt] = tree.body
    node: ast.AST | None = None
    for part in qualname.split("."):
        node = next((n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and n.name == part), None)
        if node is None:
            raise ValueError(f"symbol {qualname!r}: {part!r} is not defined at that scope")
        scope = node.body
    assert node is not None
    return node


def parse_symbol(symbol: str) -> tuple[str, str]:
    if symbol.count("::") != 1:
        raise ValueError(f"code symbol must be '<path>::<qualname>': {symbol!r}")
    rel, qual = symbol.split("::")
    if not rel.endswith(".py") or not qual or rel.startswith("/") or ".." in Path(rel).parts:
        raise ValueError(f"malformed code symbol: {symbol!r}")
    return rel, qual


def source_text(raw: bytes) -> str:
    """Source bytes decoded exactly as ``Path.read_text(encoding="utf-8")`` decodes them (universal newlines), so a
    digest computed from bytes already read equals the digest ``symbol_closure_digest`` computes from the file."""
    return io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", newline=None).read()


def symbol_closure_digest(repo: Path, symbol: str) -> str:
    """Digest of a function or class plus every module-level binding it transitively references (comment-insensitive)."""
    rel, _qual = parse_symbol(symbol)
    path = repo / rel
    if not path.is_file():
        raise ValueError(f"symbol {symbol!r}: file is missing")
    return closure_digest_of_text(symbol, path.read_text(encoding="utf-8"))


def closure_digest_of_text(symbol: str, text: str) -> str:
    """``symbol_closure_digest`` of ``symbol`` computed from ``text``, the source of its file (stage 1D-1c.1C set
    "au.1"): a caller that has read and hashed the bytes itself computes the closure from those same bytes, so the
    digest it binds cannot describe a file other than the one it hashed."""
    rel, qual = parse_symbol(symbol)
    tree = ast.parse(text, filename=rel)
    table = _module_table(tree)
    root = _resolve(tree, qual)
    segments: list[list[str]] = []
    seen: set[int] = set()

    def segment_of(node: ast.AST) -> str:
        parts = [ast.get_source_segment(text, d) or "" for d in getattr(node, "decorator_list", [])]
        parts.append(ast.get_source_segment(text, node) or "")
        return "\n".join(_strip_comments(p) for p in parts)

    def include(label: str, node: ast.AST) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        segments.append([label, segment_of(node)])
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return
        names = sorted({n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)})
        for d in getattr(node, "decorator_list", []):
            names.extend(n.id for n in ast.walk(d) if isinstance(n, ast.Name))
        for name in sorted(set(names)):
            if name in table:
                include(name, table[name])

    include(qual, root)
    return sha256_bytes(canonical_bytes({"symbol": symbol, "segments": segments}))
