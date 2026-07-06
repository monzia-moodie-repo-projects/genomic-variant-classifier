"""patch_orchestrator_phase1.py -- Monzia Moodie

Phase 1 of the orchestrator redesign: make agent registration LAZY.

Transforms ``Orchestrator._register_agents`` from the eager form (22 ``from ... import Agent`` statements
followed by ``self._agent_registry = { "Name": Name, ... }``) into a lazy form:

    self._agent_registry = {
        "Name": _Lazy("module.path:Class"),
        ...
    }

so the Orchestrator imports ZERO agent modules at construction. Each agent's import (and its transitive
heavy dependencies -- torch, sklearn, the detector modules) is deferred to that agent's first use in
run_pipeline. This resolves the Data Freshness CI failure at the root: the eager registration pulled
sklearn (via ModelInsightsAgent -> model_insights_detector) and torch (via the EWC chain) at
construction, crashing in the minimal CI environment that has neither.

What this script does NOT touch:
  - the run_pipeline call site (``agent_cls = self._agent_registry.get(name)`` then
    ``agent_cls.from_default_baseline(state)`` / ``agent_cls(state)``). _Lazy is callable-with-state and
    delegates attribute access to the resolved class, so that site works UNCHANGED, including the
    ``hasattr(agent_cls, "from_default_baseline")`` drift-agent routing.
  - the registry being a dict literal of STRING KEYS (the AST liveness checker depends on this).

Discipline: READ-FIRST, anchor-based (no line numbers), idempotent (sentinel), validates the eager
anchor occurs exactly once, writes a .bak, ast.parse syntax-guard with rollback, ABORTS on any anchor
mismatch. Builds the full new text and writes ONCE.

Usage:
    python patch_orchestrator_phase1.py [--orchestrator PATH] [--lazy-import IMPORT_LINE] \
        [--agent-base MODULE_PREFIX]

Defaults target the real repo layout.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


SENTINEL = "# >>> PHASE1_LAZY_REGISTRY <<<"


def _die(msg: str) -> None:
    print(f"ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--orchestrator",
        default="src/genomic_variant_classifier/agent_layer/orchestrator.py",
    )
    ap.add_argument(
        "--lazy-import",
        default="from genomic_variant_classifier.agent_layer._lazy_agent import _Lazy",
    )
    ap.add_argument(
        "--agent-base",
        default="genomic_variant_classifier.agent_layer.agents",
        help="dotted module prefix the eager 'from <base>.<mod> import <Class>' lines use",
    )
    args = ap.parse_args()

    path = Path(args.orchestrator)
    if not path.is_file():
        _die(f"orchestrator not found: {path}")
    src = path.read_text(encoding="utf-8")

    # Idempotency: already applied?
    if SENTINEL in src:
        print("[skip] sentinel present -- Phase 1 already applied; nothing to do.")
        return

    # --- Locate the _register_agents method body ---
    m_def = re.search(r"\n(?P<indent> *)def _register_agents\(self\)[^\n]*:\n", src)
    if not m_def:
        _die("could not find 'def _register_agents(self)'")
    method_indent = m_def.group("indent")
    body_start = m_def.end()

    # The eager body = consecutive 'from <base>.<mod> import <Class>' lines, then the dict literal
    # 'self._agent_registry = { "Name": Class, ... }'. Capture from body_start to the dict's closing brace.
    dict_open = src.find("self._agent_registry = {", body_start)
    if dict_open == -1:
        _die("could not find 'self._agent_registry = {' in _register_agents")

    # Verify it occurs exactly once in the whole file (the AST checker depends on a single literal).
    if src.count("self._agent_registry = {") != 1:
        _die("'self._agent_registry = {' must occur exactly once; found "
             f"{src.count('self._agent_registry = {')}")

    # Find the matching closing brace of the dict literal (brace counting from dict_open).
    brace_scan_start = src.find("{", dict_open)
    depth = 0
    close_idx = -1
    for i in range(brace_scan_start, len(src)):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                close_idx = i
                break
    if close_idx == -1:
        _die("could not find the closing brace of the _agent_registry dict literal")
    dict_block = src[dict_open:close_idx + 1]

    # --- Parse the eager imports: 'from <base>.<mod> import <Class>' between body_start and dict_open ---
    imports_region = src[body_start:dict_open]
    base = re.escape(args.agent_base)
    import_re = re.compile(
        rf"from\s+(?P<mod>{base}\.[A-Za-z_][\w.]*)\s+import\s+(?P<cls>[A-Za-z_]\w*)"
    )
    import_map = {}  # Class -> module
    for mm in import_re.finditer(imports_region):
        import_map[mm.group("cls")] = mm.group("mod")
    if not import_map:
        _die("found no eager 'from <base>.<mod> import <Class>' lines in _register_agents")

    # --- Parse the dict entries: '"Name": Class,' -- map registry key -> Class symbol ---
    entry_re = re.compile(r'"(?P<key>[A-Za-z_]\w*)"\s*:\s*(?P<cls>[A-Za-z_]\w*)\s*,?')
    entries = entry_re.findall(dict_block)
    if not entries:
        _die("found no '\"Name\": Class,' entries in the _agent_registry dict literal")

    # Every dict value Class must have a matching import (so we know its module).
    new_lines = []
    for key, cls in entries:
        if cls not in import_map:
            _die(f"dict entry '{key}': {cls} has no matching eager import; cannot derive module")
        module = import_map[cls]
        new_lines.append(f'{method_indent}        "{key}": _Lazy("{module}:{cls}"),')

    n_entries = len(entries)
    n_imports = len(import_map)
    if n_entries != n_imports:
        # Not necessarily fatal (an import could be unused), but surface it loudly.
        print(f"[warn] {n_imports} eager imports but {n_entries} dict entries "
              "(an import may be unused, or a value lacks an import).")

    # --- Build the new _register_agents body ---
    new_body = (
        f"{method_indent}    {SENTINEL}\n"
        f"{method_indent}    # Lazy registry: values are _Lazy(\"module:Class\") -- NO agent module is\n"
        f"{method_indent}    # imported at construction. Each agent (and its heavy transitive deps) is\n"
        f"{method_indent}    # imported on first use in run_pipeline. Keys stay string literals so the\n"
        f"{method_indent}    # AST liveness checker (scripts/check_agents_active.py) still finds all agents.\n"
        f"{method_indent}    self._agent_registry = {{\n"
        + "\n".join(new_lines) + "\n"
        f"{method_indent}    }}\n"
    )

    # Replace from body_start through the dict's closing brace (+ trailing newline if present).
    end_replace = close_idx + 1
    if end_replace < len(src) and src[end_replace] == "\n":
        end_replace += 1
    new_src = src[:body_start] + new_body + src[end_replace:]

    # --- REGION 2: widen the per-agent guard to wrap construction (graceful failure, criterion #4) ---
    # With the lazy registry, hasattr/from_default_baseline/ctor trigger the agent's import on first use.
    # In the eager code the per-agent try: wraps ONLY agent.run(), so a lazy-import failure at construction
    # would ESCAPE run_pipeline and crash the whole pipeline. We move the construction INSIDE the try and
    # hoist _t0/_err above it (so telemetry timing also covers construction). Behavior is identical for the
    # eager case (construction never raised there); for the lazy case it isolates per-agent import failures.
    guard_sentinel = "# >>> PHASE1_GUARDED_CONSTRUCTION <<<"
    if guard_sentinel not in new_src:
        # Anchor on the exact eager guard region. We locate the 'if hasattr(agent_cls, "from_default_baseline"):'
        # construction block followed by '_t0 = time.monotonic()', '_err = None', 'try:', 'result = agent.run('.
        guard_re = re.compile(
            r"(?P<lead>\n(?P<ind> +)if hasattr\(agent_cls, \"from_default_baseline\"\):\n)"
            r"(?P<between>.*?)"
            r"(?P<run>\n(?P=ind)_t0 = time\.monotonic\(\)\n"
            r"(?P=ind)_err = None\n"
            r"(?P=ind)try:\n"
            r"(?P=ind)    result = agent\.run\(dry_run=self\._dry_run\)\n)",
            re.DOTALL,
        )
        gm = guard_re.search(new_src)
        if not gm:
            _die("could not find the eager per-agent guard region to widen "
                 "(hasattr ... _t0/_err/try/result=agent.run pattern)")
        ind = gm.group("ind")
        # Reconstruct the construction body (the hasattr/else lines + their comments) from the matched text.
        construct_block = gm.group("lead") + gm.group("between")
        # Re-indent the construction block by 4 spaces (it moves inside the try:).
        construct_lines = construct_block.split("\n")
        reindented = []
        for ln in construct_lines:
            if ln.strip() == "":
                reindented.append("")
            else:
                reindented.append("    " + ln)
        reindented_block = "\n".join(reindented)
        new_guard = (
            f"\n{ind}{guard_sentinel}\n"
            f"{ind}_t0 = time.monotonic()\n"
            f"{ind}_err = None\n"
            f"{ind}try:\n"
            f"{ind}    # Construction may import the agent module on first use (lazy registry),\n"
            f"{ind}    # so it is inside the guard: a missing optional dependency or a broken\n"
            f"{ind}    # agent import is isolated to THIS agent and never crashes the pipeline.\n"
            + reindented_block.rstrip("\n") + "\n"
            f"{ind}    result = agent.run(dry_run=self._dry_run)\n"
        )
        new_src = new_src[:gm.start()] + new_guard + new_src[gm.end():]

    # --- Add the _Lazy import near the top (after the module docstring / first import block) ---
    if args.lazy_import not in new_src:
        # Insert after the first 'from __future__' if present, else after the first import line,
        # else after the module docstring.
        lines = new_src.splitlines(keepends=True)
        insert_at = None
        for idx, line in enumerate(lines):
            if line.startswith("from __future__"):
                insert_at = idx + 1
                break
        if insert_at is None:
            for idx, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from "):
                    insert_at = idx
                    break
        if insert_at is None:
            insert_at = 0
        lines.insert(insert_at, args.lazy_import + "\n")
        new_src = "".join(lines)

    # --- Syntax guard ---
    try:
        ast.parse(new_src)
    except SyntaxError as e:
        _die(f"patched source fails to parse: {e}")

    # --- Verify the AST checker contract: self._agent_registry is still a Dict of string keys ---
    tree = ast.parse(new_src)
    found_registry_dict = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Attribute) and tgt.attr == "_agent_registry"
                        and isinstance(node.value, ast.Dict)):
                    keys = node.value.keys
                    if all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in keys):
                        found_registry_dict = True
                        if len(keys) != n_entries:
                            _die(f"post-transform registry has {len(keys)} keys, expected {n_entries}")
    if not found_registry_dict:
        _die("post-transform: self._agent_registry is no longer a Dict of string keys (AST contract broken)")

    # --- Write .bak + new file ---
    bak = path.with_suffix(path.suffix + ".bak")
    bak.write_text(src, encoding="utf-8")
    path.write_text(new_src, encoding="utf-8")
    print(f"[ok] sentinel installed; {n_entries} agents now lazy; backup at {bak}")
    print(f"[ok] _Lazy import present: {args.lazy_import in new_src}")
    print(f"[ok] eager 'from {args.agent_base}' imports removed from _register_agents: "
          f"{('from ' + args.agent_base) not in new_src.split('def _register_agents')[1].split('def ')[0]}")


if __name__ == "__main__":
    main()
