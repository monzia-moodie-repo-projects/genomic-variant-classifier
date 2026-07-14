#!/usr/bin/env python3
"""Make every CONSOLE-BOUND string in src/ pure ASCII. AST-precise; docstrings untouched.

Created 2026-07-14 (roadmap 6.24).

THE PROBLEM
-----------
160 call sites across 40 files in `src/` pass non-ASCII characters to `logger.*`, `print()`,
`warnings.warn()` or an exception constructor -- em-dashes, arrows, box-drawing, ellipses, and
emoji:

    logger.error("Unknown agent '%s' -- skipping.", name)      <- was an em-dash
    logger.info("[MSG]  %s -> %s  [%s]  id=%s", ...)           <- was an emoji and an arrow
    logger.warning("[!]  %d message(s) awaiting approval", n)  <- was a warning sign
    print(f"  DRIFT REPORT -- {self.timestamp}")               <- was an em-dash

On a Windows console running the cp1252 code page, those bytes are decoded as cp1252 and render
as mojibake. This is not hypothetical: `docs/runs/RUN_16_results.md:84` recorded it in Run 16 --

    "Cosmetic: Reactome warning string uses an em-dash that mojibakes (mojibake) in PowerShell
     -- switch to ASCII `--` in reactome.py."

-- and it was never fixed. `reactome.py` still carried the em-dash on 2026-07-14, eight weeks
later. **A finding in a document is a comment. A finding that fails a test is a gate.**

WHY NOT JUST RECONFIGURE STDOUT TO UTF-8?
-----------------------------------------
`scripts/train.py:61` already does exactly that (`sys.stdout.reconfigure(encoding="utf-8",
errors="replace")`), and it does NOT solve this. Python then writes correct UTF-8 *bytes*, and a
cp1252 console *decodes* them as cp1252 -- producing the mojibake. To actually fix it that way,
every operator would also have to run `chcp 65001`, on every machine, in every terminal, forever.

A fix whose correctness depends on the terminal's code page is a fix that is green on one box
and broken on another. That is `docs/ROADMAP.md` section 7, root pattern (d): **a green result
from a mutated environment is evidence about the ENVIRONMENT, not the code.** The KAN model was
silently dropped from every Continuous Integration run for two months for exactly this reason.

So: console-bound strings are ASCII. No dependency on code page, PYTHONIOENCODING, terminal
emulator, Docker base image, or CI runner. Deterministic everywhere, forever.

The project had ALREADY made this decision -- `tests/unit/test_variant_ensemble_ascii.py`
asserts it -- but enforced it on exactly ONE file out of the 41 that violate it. This script
plus `tests/unit/test_console_strings_are_ascii.py` generalise the rule and make forgetting fail.

WHAT IS AND IS NOT TOUCHED
--------------------------
TOUCHED:   string literals (including f-string parts) that are arguments to
           logger.debug/info/warning/warn/error/exception/critical, print(),
           warnings.warn(), or an exception constructor in a `raise`.
UNTOUCHED: docstrings (module/class/function). Comments (not in the AST at all).
           Strings written to FILES with an explicit encoding -- a report or a
           parquet header may legitimately carry Unicode; the file declares its own encoding
           and no console is involved.

The edit is made by byte-span, using each literal's (lineno, col_offset, end_lineno,
end_col_offset) from the AST -- never by regex over the file -- so a docstring containing the
same characters cannot be caught by accident.

AFTER REWRITING, IT RE-PARSES THE FILE AND REFUSES TO WRITE IF THE ABSTRACT SYNTAX TREE
CHANGED SHAPE. A source-rewriting script that has not proven it preserved the parse is a
source-rewriting script that will one day silently corrupt a module.

USAGE
-----
    python scripts/maintenance/fix_console_ascii.py --dry-run     # report only, changes nothing
    python scripts/maintenance/fix_console_ascii.py               # rewrite in place
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

#: Console-bound call names. `warn` covers warnings.warn; the logger methods cover
#: logger.* and self.logger.* alike (matched on the attribute name).
CONSOLE_CALLS = {
    "debug", "info", "warning", "warn", "error", "exception", "critical", "print",
}

#: Deliberate, reviewed replacements. NOT a generic "strip to ASCII": a silent `?` or a dropped
#: character would make a log message WORSE than the mojibake it replaces. Every substitution
#: below preserves the meaning the author intended.
REPLACEMENTS: dict[str, str] = {
    "—": "--",       # — em dash
    "–": "-",        # – en dash
    "→": "->",       # → rightwards arrow
    "←": "<-",       # ← leftwards arrow
    "↳": "->",       # ↳ downwards-then-right arrow (used as an indent marker)
    "…": "...",      # … horizontal ellipsis
    "─": "-",        # ─ box drawing light horizontal
    "═": "=",        # ═ box drawing double horizontal
    "•": "*",        # • bullet
    "▶": ">",        # ▶ black right-pointing triangle
    "✓": "[OK]",     # ✓ check mark
    "✅": "[OK]",     # ✅ white heavy check mark
    "❌": "[X]",      # ❌ cross mark
    "⚠": "[!]",      # ⚠ warning sign
    "✉": "[NEW]",    # ✉ envelope -- used as the "unread message" marker
    "·": "*",        # · middle dot (U+00B7). NOTE: U+2022 bullet is mapped separately above.
    "\U0001f4e8": "[MSG]",    # 📨 incoming envelope
    "\U0001f4ec": "[INBOX]",  # 📬 open mailbox with raised flag
    "⏳": "[WAIT]",   # ⏳ hourglass
    "\U0001f4dc": "[LOG]",    # 📜 scroll
    "σ": "sigma",    # σ
    "Δ": "delta",    # Δ
    "ε": "epsilon",  # ε
    "μ": "mu",       # μ
    "χ": "chi",      # χ
    "≥": ">=",       # ≥
    "≤": "<=",       # ≤
    "×": "x",        # ×
    "²": "^2",       # ²
    "°": " deg",     # °
    " ": " ",        # non-breaking space -- invisible, and a real hazard
}


def _ascii_ise(text: str) -> tuple[str, list[str]]:
    """Return (ascii_text, unmapped_chars). Never drops a character silently.

    Three tiers, in order:

      1. ASCII already            -> keep.
      2. In REPLACEMENTS          -> a deliberate, reviewed substitution (arrows, box-drawing,
                                     emoji, mathematical symbols). These have no meaningful
                                     ASCII decomposition, so a human chose the mapping.
      3. A LATIN LETTER WITH A DIACRITIC -> decompose (Unicode Normalization Form KD) and drop
                                     the combining marks: e-acute -> e, o-umlaut -> o.
                                     This is lossless in MEANING for a log line and it means
                                     "Szekely-Rizzo" does not need a hand-written entry, nor
                                     does the next accented surname somebody types.
      4. Anything else            -> LEFT IN PLACE and REPORTED. Never guessed at.

    Tier 4 is the important one. A silently-dropped or `?`-substituted character makes a log
    message WORSE than the mojibake it was meant to replace: the reader now has neither the
    original glyph nor a clue that something was lost.

    An unmapped character is left EXACTLY where it was. The per-file residual check in
    `process()` then sees it is still non-ASCII, ABORTS that file, and writes nothing; `main()`
    exits 2 and names the character so a deliberate mapping can be added. Nothing is guessed at
    and no half-repaired file reaches disk.
    """
    import unicodedata

    out: list[str] = []
    unmapped: list[str] = []
    for ch in text:
        if ord(ch) < 128:
            out.append(ch)
            continue
        if ch in REPLACEMENTS:
            out.append(REPLACEMENTS[ch])
            continue

        # Tier 3: strip diacritics from Latin letters (e-acute -> e, o-umlaut -> o, ...).
        decomposed = unicodedata.normalize("NFKD", ch)
        stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
        if stripped and stripped.isascii():
            out.append(stripped)
            continue

        unmapped.append(ch)
        out.append(ch)              # leave it -- and REPORT it. Never mangle blindly.
    return "".join(out), unmapped


class _ConsoleStringFinder(ast.NodeVisitor):
    """Collect the source spans of every string literal that reaches a console.

    DIRECTLY *AND* INDIRECTLY -- and the indirect half is not optional.
    -------------------------------------------------------------------
    The first version of this finder only looked INSIDE `logger.*(...)` / `print(...)`. So did
    the first version of tests/unit/test_console_strings_are_ascii.py. When the GATE was later
    widened to catch the indirect path and the FIXER was not, the result was a split brain: the
    gate flagged 21 strings across 5 files that the fixer was structurally incapable of
    repairing --

        orchestrator.py:69    _DIVIDER = "═" * 60            -> logger.info(_DIVIDER)
        message_bus.py:352    verb = "✅ approved" if ...     -> logged
        evaluator.py:534      sep = "─" * 60                 -> print(sep)
        version_monitor_agent.py:177  msg = "... — ..."      -> logged

    A repair tool that cannot fix what its own gate reports is worse than no repair tool: it
    turns a red test into a manual chore, and manual chores are what rot. ONE detector, shared
    by both. This is that detector; the test file mirrors it exactly.

    Reachability, in two hops:
      1. a literal directly inside a console call or a `raise`; and
      2. a NAME bound to a non-ASCII literal, where that name is later passed to a console call
         anywhere in the module.
    """

    def __init__(self) -> None:
        self.spans: list[tuple[int, int, int, int, str]] = []
        self._depth = 0
        #: name -> spans of the non-ASCII literals bound to it
        self._bindings: dict[str, list[tuple[int, int, int, int, str]]] = {}
        #: names referenced anywhere inside a console call
        self._names_reaching_console: set[str] = set()

    @staticmethod
    def _span(node: ast.Constant) -> tuple[int, int, int, int, str]:
        return (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, node.value)

    def visit_Call(self, node: ast.Call) -> None:
        name = None
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id

        console = name in CONSOLE_CALLS
        if console:
            self._depth += 1
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name):
                    self._names_reaching_console.add(sub.id)
        self.generic_visit(node)
        if console:
            self._depth -= 1

    def visit_Raise(self, node: ast.Raise) -> None:
        # `raise ValueError("... -- ...")` reaches a console via the traceback.
        self._depth += 1
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                self._names_reaching_console.add(sub.id)
        self.generic_visit(node)
        self._depth -= 1

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            for sub in ast.walk(node.value):
                if (
                    isinstance(sub, ast.Constant)
                    and isinstance(sub.value, str)
                    and any(ord(c) > 127 for c in sub.value)
                    and sub.end_lineno is not None
                ):
                    self._bindings.setdefault(target.id, []).append(self._span(sub))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if (
            self._depth > 0
            and isinstance(node.value, str)
            and any(ord(c) > 127 for c in node.value)
            and node.end_lineno is not None
        ):
            self.spans.append(self._span(node))
        self.generic_visit(node)

    def finalise(self) -> None:
        """Fold in the indirect spans once the whole module has been walked.

        Deferred to the end because a name can be BOUND after the console call that uses it --
        a module-level constant referenced by a function defined above it, for instance.
        """
        known = set(self.spans)
        for name, spans in self._bindings.items():
            if name not in self._names_reaching_console:
                continue                     # bound, but never printed -- not our problem
            for s in spans:
                if s not in known:
                    self.spans.append(s)
                    known.add(s)


def _structural_shape(tree: ast.AST) -> str:
    """The AST with every string literal blanked -- i.e. the CODE STRUCTURE, without content.

    A rewrite must change string CONTENT and nothing else. Comparing `ast.dump()` directly
    cannot prove that: it changes when content changes, so it tells you nothing about whether
    the structure also changed. Blanking every string first makes the comparison mean exactly
    what we need it to mean -- "did I alter anything other than the text inside strings?"

    Without this, a mis-spliced edit that still PARSES would sail straight through. The first
    version of this script had precisely that hole, and it had already written to 36 files
    before the bug was found (see the byte-offset note in _rewrite_span).
    """
    import copy
    t = copy.deepcopy(tree)
    for node in ast.walk(t):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    return ast.dump(t)


def _rewrite_span(
    lines: list[str], lineno: int, col: int, end_lineno: int, end_col: int
) -> list[str]:
    """ASCII-ise the literal at this span. Returns unmapped chars. Mutates `lines`.

    ========================================================================================
    col_offset AND end_col_offset ARE UTF-8 *BYTE* OFFSETS -- NOT CHARACTER OFFSETS.
    ========================================================================================
    This is documented in the `ast` module and it is easy to miss, because for pure-ASCII
    lines the two are identical -- so the bug is INVISIBLE on exactly the lines that do not
    need fixing, and appears only on the lines that do.

    The first version of this script sliced a Python `str` with those offsets:

        literal = line[col:end_col]        # WRONG: str slicing is by CHARACTER

    `'—'` is ONE character but THREE bytes, so every offset after a non-ASCII character on the
    same line is shifted by +2 per character. Measured on the real source:

        print(f"  {'-'*22} {'─'*7} {'─'*7} {'─'*8} {'-'*8}")   # evaluator.py:574
            Constant '─'  col=20 end_col=25  ->  line[20:25] = "'─'*7"   correct
            Constant '─'  col=30 end_col=35  ->  line[30:35] = "'*7} "   GARBAGE
            Constant '─'  col=40 end_col=45  ->  line[40:45] = "8} {'"   GARBAGE

    Many rewrites survived by luck -- the mis-slice usually overshoots into trailing ASCII
    (`")`, `, x`), which _ascii_ise passes through unchanged and reinserts verbatim. But
    "correct by luck" is not correct, and a multi-literal line whose offsets go stale after a
    length-changing edit can genuinely lose characters.

    THE FIX: encode the line to bytes, slice by BYTE offset, decode. Then the offsets mean what
    the AST says they mean.
    ========================================================================================
    """
    unmapped: list[str] = []

    if lineno == end_lineno:
        raw = lines[lineno - 1].encode("utf-8")
        literal = raw[col:end_col].decode("utf-8")
        fixed, um = _ascii_ise(literal)
        unmapped += um
        if fixed != literal:
            lines[lineno - 1] = (
                raw[:col] + fixed.encode("utf-8") + raw[end_col:]
            ).decode("utf-8")
        return unmapped

    # ---- MULTI-LINE literal (implicit concatenation across lines is ONE Constant node) -----
    # The AST gives exact boundaries, so this is safe: the first line from `col` to its end,
    # every whole line between, and the last line up to `end_col`. All by BYTE offset.
    first_raw = lines[lineno - 1].encode("utf-8")
    last_raw = lines[end_lineno - 1].encode("utf-8")

    head = first_raw[:col]
    tail = last_raw[end_col:]

    body_first, um = _ascii_ise(first_raw[col:].decode("utf-8"));  unmapped += um
    body_last, um = _ascii_ise(last_raw[:end_col].decode("utf-8")); unmapped += um

    lines[lineno - 1] = (head + body_first.encode("utf-8")).decode("utf-8")
    for i in range(lineno, end_lineno - 1):          # the whole lines strictly between
        mid, um = _ascii_ise(lines[i])
        unmapped += um
        lines[i] = mid
    lines[end_lineno - 1] = (body_last.encode("utf-8") + tail).decode("utf-8")

    return unmapped


def process(path: Path, dry_run: bool) -> tuple[int, list[str]]:
    """Returns (n_literals_rewritten, unmapped_chars_found)."""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        print(f"  SKIP (does not parse): {path} -- {exc}")
        return 0, []

    finder = _ConsoleStringFinder()
    finder.visit(tree)
    finder.finalise()          # <- folds in the INDIRECT spans. Without this the widening
                               #    is inert and the fixer silently under-repairs.
    if not finder.spans:
        return 0, []

    lines = source.splitlines(keepends=True)
    all_unmapped: list[str] = []

    # BOTTOM-UP, RIGHT-TO-LEFT. Every edit invalidates the offsets of everything AFTER it on
    # the same line, so we must never edit left-of something we have not yet handled.
    for lineno, col, end_lineno, end_col, _value in sorted(finder.spans, reverse=True):
        all_unmapped += _rewrite_span(lines, lineno, col, end_lineno, end_col)

    new_source = "".join(lines)
    if new_source == source:
        return 0, all_unmapped

    # ---- REFUSE TO WRITE A FILE WE HAVE NOT PROVEN WE DID NOT BREAK ----------------------
    try:
        new_tree = ast.parse(new_source)
    except SyntaxError as exc:
        print(f"  ABORT: rewriting {path} produced INVALID PYTHON ({exc}). NOT written.")
        return 0, all_unmapped

    # THE REAL GUARD. Parsing successfully proves almost nothing -- a mis-spliced edit can still
    # be valid Python. This proves the only things that changed are the CONTENTS of string
    # literals: identical structure, identical everything else.
    if _structural_shape(new_tree) != _structural_shape(tree):
        print(
            f"  ABORT: rewriting {path} CHANGED THE CODE STRUCTURE, not just string contents. "
            f"NOT written. This is a bug in the rewriter -- do not work around it."
        )
        return 0, all_unmapped

    # And the result must actually be ASCII-clean in the spans we targeted, or we have quietly
    # done nothing while reporting success. Re-run the SAME detector -- including the indirect
    # path -- over the rewritten tree.
    residual = _ConsoleStringFinder()
    residual.visit(new_tree)
    residual.finalise()
    if residual.spans:
        print(
            f"  ABORT: {path} -- {len(residual.spans)} console string(s) are STILL non-ASCII "
            f"after the rewrite. Refusing to write a half-done file."
        )
        return 0, all_unmapped

    if not dry_run:
        path.write_text(new_source, encoding="utf-8", newline="")

    return len(finder.spans), all_unmapped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("src"))
    ap.add_argument("--dry-run", action="store_true",
                    help="report what WOULD change; write nothing")
    args = ap.parse_args()

    files = sorted(args.root.rglob("*.py"))
    if not files:
        print(f"ABORT: no .py files under {args.root}")
        return 1

    total_files = 0
    total_literals = 0
    unmapped_all: list[str] = []

    for f in files:
        if "__pycache__" in f.parts:
            continue
        n, unmapped = process(f, args.dry_run)
        unmapped_all += unmapped
        if n:
            total_files += 1
            total_literals += n
            verb = "would fix" if args.dry_run else "FIXED"
            print(f"  {verb:9s} {n:3d} console string(s)  {f}")

    print()
    print(f"{'DRY RUN -- nothing written' if args.dry_run else 'WRITTEN'}")
    print(f"  files touched   : {total_files}")
    print(f"  console strings : {total_literals}")

    if unmapped_all:
        uniq = sorted(set(unmapped_all))
        print()
        print(f"  *** {len(uniq)} CHARACTER(S) HAVE NO MAPPING AND WERE LEFT IN PLACE ***")
        for c in uniq:
            print(f"      U+{ord(c):04X}  {c!r}")
        print()
        print("  They were NOT stripped and NOT guessed at. Add a deliberate entry to")
        print("  REPLACEMENTS -- a silently-dropped character makes a log message worse than")
        print("  the mojibake it replaced. Then re-run.")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
