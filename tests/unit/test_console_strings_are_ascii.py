"""Every string that reaches a console must be ASCII. All of src/, not one file.

Created 2026-07-14 (roadmap 6.24).

WHY
---
On 2026-07-14 an audit found **160 call sites across 40 files in `src/`** passing non-ASCII
characters to `logger.*`, `print()`, `warnings.warn()` or an exception constructor: em-dashes,
arrows, box-drawing, ellipses, Greek letters, and emoji.

    logger.error("Unknown agent '%s' — skipping.", name)
    logger.info("📨  %s → %s  [%s]  id=%s", ...)
    logger.warning("⚠  %d message(s) awaiting approval ...", n)
    print(f"  DRIFT REPORT — {self.timestamp}")

On a Windows console running the cp1252 code page those bytes are decoded as cp1252 and render
as mojibake. **This is not hypothetical.** `docs/runs/RUN_16_results.md:84`, written during
Run 16:

    "Cosmetic: Reactome warning string uses an em-dash that mojibakes (ΓÇö) in PowerShell --
     switch to ASCII `--` in reactome.py."

It was recorded, and then nothing happened. `reactome.py` still carried the em-dash eight weeks
later, on 2026-07-14, when this test was written.

    A FINDING IN A LOG IS A COMMENT.
    A FINDING IN A DOCUMENT IS A COMMENT.
    A FINDING THAT FAILS A TEST IS A GATE.

WHY NOT RECONFIGURE STDOUT TO UTF-8 INSTEAD?
--------------------------------------------
Because `scripts/train.py:61` ALREADY DOES -- `sys.stdout.reconfigure(encoding="utf-8",
errors="replace")` -- and it does not solve this. Python then writes correct UTF-8 *bytes*, and
a cp1252 console *decodes* them as cp1252, producing the mojibake. Fixing it that way would
require every operator to also run `chcp 65001`, on every machine, in every terminal, forever.

A fix whose correctness depends on the terminal's code page is green on one box and broken on
another. That is roadmap section 7, root pattern (d): **a green result from a mutated
environment is evidence about the ENVIRONMENT, not the code.** It is the exact shape of the
defect that silently dropped the Kolmogorov-Arnold Network from every Continuous Integration
run for two months.

ASCII in console-bound strings has no environmental dependency at all. It is correct on
Windows, Linux, macOS, Docker, a hosted runner, and a rented graphics-processing-unit box,
without configuration.

WHAT THIS DOES AND DOES NOT POLICE
----------------------------------
POLICED:    string literals (including f-string parts) passed to logger.debug / info / warning /
            warn / error / exception / critical, to print(), to warnings.warn(), or to an
            exception constructor inside a `raise`.
NOT POLICED: docstrings -- they are never printed to a console, and `"p.Arg175His" -> "R175H"`
            style arrows genuinely aid comprehension there. Comments -- not in the AST.
            Strings written to FILES with an explicit encoding: a generated report may carry
            Unicode legitimately, because the file declares its own encoding and no console is
            involved.

The check is done over the ABSTRACT SYNTAX TREE, not with a regular expression over the file,
so a docstring containing the same characters cannot be flagged by accident -- and a console
string hidden inside an f-string cannot be missed.

PREDECESSOR
-----------
`tests/unit/test_variant_ensemble_ascii.py` asserts exactly this rule -- for ONE file. It was
written after a real incident and it works. It was simply never generalised, so 40 other modules
carried the identical hazard, ungated, for months. This file is that generalisation.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path("src/genomic_variant_classifier")

#: Anything whose output can land on a terminal.
CONSOLE_CALLS = {
    "debug", "info", "warning", "warn", "error", "exception", "critical", "print",
}


class _ConsoleStringVisitor(ast.NodeVisitor):
    """Find every non-ASCII string literal that reaches a console -- DIRECTLY OR INDIRECTLY.

    THE INDIRECT PATH IS THE ONE THAT NEARLY GOT AWAY.
    -------------------------------------------------
    The first version of this visitor only looked INSIDE `logger.*(...)` / `print(...)` calls.
    It would therefore have passed, with a clean bill of health, a file containing:

        _DIVIDER = "=" * 60            # was U+2550, a box-drawing double horizontal
        ...
        logger.info(_DIVIDER)          # <- reaches the console. Invisible to the old check.

    Measured on 2026-07-14: **39** non-ASCII literals in `src/` sit outside a console call and
    outside a docstring. Several of them reach a terminal anyway:

        orchestrator.py:69    _DIVIDER = "═" * 60          -> logger.info(_DIVIDER)
        orchestrator.py:396   approval_tag = " [⏳ ... ]"   -> appended to a log line
        message_bus.py:352    verb = "✅ approved" if ...   -> logged
        evaluator.py:534      sep = "─" * 60               -> print(sep)
        run_agents.py:95      "... — Agent Layer CLI"      -> argparse description, printed by --help
        drift_detector.py:457 "...Székely-Rizzo..."        -> logged at WARNING

    A gate that misses the indirect path is a gate that certifies the hazard.

    BUT NOT EVERY NON-ASCII STRING IS A HAZARD, AND BANNING THEM ALL WOULD BE WRONG.
    -------------------------------------------------------------------------------
    Of those 39, a substantial number are FILE output and must STAY Unicode:

        reports/report_generator.py     -- HTML report body; matplotlib chart titles (-> PNG)
        agents/literature_scout_agent.py-- HTML digest
        agents/label_shift_agent.py     -- markdown report containing "chi-squared" as U+03C7 U+00B2
        api/schemas.py                  -- Pydantic Field descriptions (-> OpenAPI JSON)
        provisioning/provisioning_docs.py -- markdown provisioning record

    Those are written to files and to JSON, which declare their own encoding. No console is
    involved. A blanket "no non-ASCII anywhere in src/" would force them into ASCII for no
    reason and would be exactly the kind of over-broad rule that gets switched off.

    So the visitor tracks REACHABILITY, not mere presence:
      1. literals directly inside a console call or a `raise`   -- as before; and
      2. names assigned a non-ASCII string literal, where that NAME is later passed to a
         console call within the same scope (module-level constants and locals alike).
    """

    def __init__(self) -> None:
        self.offenders: list[tuple[int, str, str]] = []   # (lineno, offending chars, text)
        self._depth = 0
        #: name -> [(lineno, value)] for every non-ASCII string bound to a name
        self._bindings: dict[str, list[tuple[int, str]]] = {}
        #: names that appear anywhere inside a console call
        self._names_reaching_console: set[str] = set()

    # -- direct path ---------------------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        name = None
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr          # logger.info(...) / self.logger.info(...)
        elif isinstance(node.func, ast.Name):
            name = node.func.id            # print(...)

        inside = name in CONSOLE_CALLS
        if inside:
            self._depth += 1
            # Record every NAME referenced inside this call -- that is the indirect path.
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name):
                    self._names_reaching_console.add(sub.id)
        self.generic_visit(node)
        if inside:
            self._depth -= 1

    def visit_Raise(self, node: ast.Raise) -> None:
        # A raised exception's message reaches the console via the traceback.
        self._depth += 1
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                self._names_reaching_console.add(sub.id)
        self.generic_visit(node)
        self._depth -= 1

    # -- indirect path: remember non-ASCII strings bound to a name ------------------------
    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            for sub in ast.walk(node.value):
                if (
                    isinstance(sub, ast.Constant)
                    and isinstance(sub.value, str)
                    and any(ord(c) > 127 for c in sub.value)
                ):
                    self._bindings.setdefault(target.id, []).append((sub.lineno, sub.value))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if self._depth > 0 and isinstance(node.value, str):
            bad = sorted({c for c in node.value if ord(c) > 127})
            if bad:
                self.offenders.append(
                    (node.lineno, "".join(bad), node.value.replace("\n", "\\n")[:70])
                )
        self.generic_visit(node)

    def finalise(self) -> None:
        """Fold in the indirect offenders once the whole module has been walked.

        Deferred to the end because an assignment can appear AFTER the console call that uses
        it (a module-level constant used in a function defined above it, for instance).
        """
        seen = {(ln, txt) for ln, _c, txt in self.offenders}
        for name, bindings in self._bindings.items():
            if name not in self._names_reaching_console:
                continue                       # bound, but never printed -- not our problem
            for lineno, value in bindings:
                bad = sorted({c for c in value if ord(c) > 127})
                key = (lineno, value.replace("\n", "\\n")[:70])
                if bad and key not in seen:
                    self.offenders.append((lineno, "".join(bad), key[1]))
                    seen.add(key)


def _scan(path: Path) -> list[tuple[int, str, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    v = _ConsoleStringVisitor()
    v.visit(tree)
    v.finalise()
    return sorted(v.offenders)


def test_no_console_bound_string_in_src_contains_non_ascii():
    """The whole rule, in one assertion, over the whole package."""
    assert SRC.is_dir(), f"source tree not found at {SRC.resolve()}"

    files = [p for p in sorted(SRC.rglob("*.py")) if "__pycache__" not in p.parts]

    # Guard the guard: if the tree moves or the glob breaks, this test must not silently
    # become vacuous -- a check that inspects nothing is the defect this project keeps finding.
    assert len(files) > 50, (
        f"only {len(files)} Python files found under {SRC} -- expected the full package. "
        f"Either the layout changed or this locator is broken. A test that scans nothing "
        f"passes for free."
    )

    offenders: dict[str, list[tuple[int, str, str]]] = {}
    for f in files:
        hits = _scan(f)
        if hits:
            offenders[str(f)] = hits

    if offenders:
        n_sites = sum(len(v) for v in offenders.values())
        report = [
            f"{n_sites} console-bound string(s) in {len(offenders)} file(s) contain non-ASCII "
            f"characters.",
            "",
            "On a Windows cp1252 console these render as MOJIBAKE. This was observed and",
            "recorded in Run 16 (docs/runs/RUN_16_results.md:84) and never fixed -- which is why",
            "it is now a test.",
            "",
            "Reconfiguring stdout to UTF-8 does NOT fix it (scripts/train.py:61 already does):",
            "Python then writes UTF-8 bytes and the cp1252 console decodes them as cp1252. A fix",
            "that depends on the terminal's code page is root pattern (d).",
            "",
            "REPAIR:",
            "    python scripts/maintenance/fix_console_ascii.py --dry-run   # see the changes",
            "    python scripts/maintenance/fix_console_ascii.py             # apply them",
            "",
            "Docstrings and comments are NOT policed -- only strings that reach a console.",
            "",
        ]
        for f, hits in sorted(offenders.items()):
            report.append(f"  {f}")
            for lineno, chars, text in hits:
                codes = " ".join(f"U+{ord(c):04X}" for c in chars)
                report.append(f"      L{lineno:<5} [{codes}]  {text!r}")
        pytest.fail("\n".join(report))


def test_the_scanner_actually_detects_a_planted_offender(tmp_path):
    """Negative test: prove the visitor FIRES. A guard nobody has watched fail is a rumour.

    Without this, a refactor of _ConsoleStringVisitor could silently stop matching and the test
    above would pass forever, guarding nothing -- which is precisely how the schema gate, the
    drift monitor and the correctness harness all came to be checking nothing at all.
    """
    planted = tmp_path / "planted.py"
    planted.write_text(
        'import logging\n'
        'logger = logging.getLogger(__name__)\n'
        '\n'
        'def f():\n'
        '    """A docstring with an em-dash -- this must NOT be flagged."""\n'
        '    x = "a plain string with an em-dash \\u2014 also not flagged"\n'
        '    logger.info("this one IS flagged \\u2014 it reaches a console")\n'
        '    print("and this \\u2192 too")\n'
        '    raise ValueError("and this \\u2026 as well")\n',
        encoding="utf-8",
    )

    hits = _scan(planted)
    linenos = sorted(h[0] for h in hits)

    assert linenos == [7, 8, 9], (
        f"the scanner found offenders on lines {linenos}; expected exactly [7, 8, 9] -- the "
        f"logger.info, the print and the raise.\n"
        f"  If it found FEWER: it has stopped detecting a real hazard.\n"
        f"  If it found MORE: it is flagging the docstring (line 5) or the plain assignment "
        f"(line 6), neither of which reaches a console, and it will generate false failures "
        f"that get it switched off."
    )
