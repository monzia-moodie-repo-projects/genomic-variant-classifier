"""Repo-wide tripwire: nothing may infer window PROVENANCE from window CONTENT.

WHY THIS FILE EXISTS -- 2026-07-15 (roadmap 6.28)
================================================
Four independent poly-window detectors existed in this repository. Every one of them
compared a window's CONTENT against the literal `"A" * 101`, and every one was blind to
the same 21,814 rows:

    scripts/train.py:436/485        `_POLY_WIN = "A" * 101`; decided has_sequences
    data/genomic_lm.py:201/250      `self._poly = "A" * self.window`; `_mapped_mask`
    data/seq_window_join.py         the fallback that created the string in the first place
    scripts/rekey_seq_windows_v2.py:146  `poly = "A" * 101`; wired into a WRITE gate

They were blind because there are TWO fabrication paths with DIFFERENT fill characters.
`delta_window_builder.POLY = "N"` is written INTO the parquet and flagged `ok=False`
(the live 2026-07-10 artifact: n_poly 21,814 of 4,420,180 rows, 0.494%); the join's own
fallback used `"A"`. Four detectors, one blind spot, and the rows flowed into training as
though they were real sequence.

THE DEEPER ERROR, AND THE REASON THIS IS A REPO-WIDE BAN RATHER THAN FOUR FIXES:
**A window that reads "A"*101 MAY BE REAL DATA.** Poly-A tracts are real biology. Content
can never distinguish "the reference genome genuinely says AAAA..." from "the builder gave
up and typed A". A content check is not merely incomplete here -- it is asking a question
that has no answer. Only PROVENANCE answers it, and `build_seq_windows.py:154` has been
writing an `ok` column into the artifact the entire time.

WHAT PROVED THE POINT: when PLACEHOLDER_BASE moved from "A" to "N" on 2026-07-15, two of
those four detectors did not start failing -- they started PASSING UNCONDITIONALLY.
rekey_seq_windows_v2's gate (which returns 6 and REFUSES TO WRITE) silently became a
rubber stamp, and train.py's `has_sequences` became permanently True, which would have
kept cnn_1d in the ensemble training on 4.4 million fabricated windows. The full suite was
GREEN in both cases. A content check does not fail loudly when it rots; it goes quiet.

So the rule is not "match N as well as A". The rule is: **ask the artifact, never the
string.** Read `WindowAttachment.usable`.
"""
from __future__ import annotations

import ast
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_SCRIPTS = _REPO / "scripts"

#: Modules permitted to build a placeholder window literal, with the reason.
#: NOTHING is added here to make a failure go away -- a new entry means a new module has
#: been given permission to fabricate sequence, which is a design decision, not a fix.
_ALLOWED: dict[str, str] = {
    "seq_window_join.py":
        "Defines PLACEHOLDER_BASE and is the ONLY module allowed to CREATE a placeholder "
        "window. It never compares against one -- it emits `usable` instead, so that no "
        "other module has to ask the question. "
        "(This entry originally read '...so it does not trip this check anyway -- listed "
        "for intent, not necessity.' That was the gate's blind spot written down as a "
        "reassurance: `PLACEHOLDER_BASE * window` is Name*Name, which the first version "
        "of _poly_literals could not see, which is how populate_fasta_seq.py's "
        "`_POLY_A = sw.PAD_CHAR * sw.WINDOW` -- a live content comparison at line 154 -- "
        "walked straight through. The exemption is now NECESSARY, which is the honest "
        "state: this module is exempt because it is the sanctioned constructor, not "
        "because the check happens to miss it.)",
    "delta_window_builder.py":
        "Defines POLY = 'N' and writes placeholder windows into the artifact ALONGSIDE "
        "ok=False. It is the source of the provenance this ban exists to protect; it "
        "records the fabrication rather than concealing it.",
}

#: Archived / historical trees. scripts/forensics/ was archived on 2026-07-12 (roadmap
#: 6.8, 68 files) and patch_*.py are committed project history -- the 2026-07-11 cleanup
#: deliberately SKIPPED 109 tracked patch_*.py because "a blind rm patch_*.py would have
#: destroyed them". They are a record of what was done, not code that runs.
_SKIP_DIRS = {"forensics"}
_SKIP_PREFIXES = ("patch_", "install_", "apply_", "dump_")

_BASES = set("ACGTNacgtn")

#: Identifiers that ARE a placeholder base, wherever they come from. Needed because the
#: first version of this gate matched only `Constant * X` -- the literal `"A" * 101` --
#: and therefore could not see:
#:
#:     src/genomic_variant_classifier/data/populate_fasta_seq.py:59
#:         _POLY_A = sw.PAD_CHAR * sw.WINDOW     # Attribute * Attribute
#:     ...:154
#:         if rw == _POLY_A:                     # a SIXTH content-based poly detector
#:
#: A named constant walked straight through the ban. Worse, the ban's own _ALLOWED entry
#: described that hole AS A SAFETY PROPERTY: "it builds it as PLACEHOLDER_BASE * window
#: (Name * Name), so it does not trip this check anyway". The blind spot was written down
#: as a reassurance.
#:
#: That is roadmap 7c -- a gate that checks a PROXY instead of the thing it protects --
#: committed inside the gate built to prevent 7c. Banning `"A" * 101` bans a SPELLING.
#: The thing to ban is the IDEA: constructing a run of one base in order to compare
#: against it. So the gate now resolves names too.
_PLACEHOLDER_IDENTS = {
    "PAD_CHAR", "POLY", "PLACEHOLDER_BASE", "POLY_A", "POLY_N",
    "_POLY", "_POLY_A", "_POLY_N", "_POLY_WIN", "_PAD",
}


def _live_python_files() -> list[Path]:
    out: list[Path] = []
    for root in (_SRC, _SCRIPTS):
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if set(part for part in p.parts) & _SKIP_DIRS:
                continue
            if p.name.startswith(_SKIP_PREFIXES):
                continue
            out.append(p)
    return sorted(out)


def _poly_literals(path: Path) -> list[tuple[int, str]]:
    """Every `<single-base-literal> * <anything>` expression in the file.

    Uses `ast`, not a text search, so that the extensive PROSE in this repository which
    quotes `"A" * 101` while explaining this incident does not trip the gate. That
    distinction has been learned three times the hard way: a ban on "1,926" fired on its
    own correction note; a feature-count sweep matched "36 of its 78 features"; and this
    session's first pragma guard matched the comment describing the pragma it banned.
    A rule that cannot tell itself apart from a description of itself is not a rule.
    """
    # `utf-8-sig`, NOT `utf-8` -- and this distinction was learned by this test crashing
    # on its first run (2026-07-15). scripts/build_spliceai_index.py begins with a UTF-8
    # byte-order mark (U+FEFF). `read_text(encoding="utf-8")` keeps that mark as a real
    # character, and `ast.parse` then dies with "invalid non-printable character U+FEFF"
    # -- so the gate ERRORED instead of reporting, which is a gate that fails closed by
    # accident rather than by design. Python's own tokeniser strips the mark (PEP 263),
    # so the file imports perfectly and nothing ever complained. `utf-8-sig` reads the
    # file the way Python reads it, which is the only reading a source-analysis gate may
    # use: a tool that parses differently from the interpreter is measuring a different
    # program. The mark itself is a real latent hazard and is now its own gate below,
    # rather than being silently swallowed by the encoding that hides it.
    # `filename=str(path)` is NOT cosmetic. Without it, ast.parse reports every
    # diagnostic against "<unknown>", and this gate's first clean run duly emitted
    #
    #     <unknown>:6: SyntaxWarning: invalid escape sequence '\M'
    #
    # -- a real defect (invalid escape sequences are deprecated and become a SyntaxError
    # in a future Python) reported against a file it refused to name, across a scan of
    # ~250 files. A tool that finds a problem and will not say where is the same failure
    # this whole session has been chasing.
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    # Pass 1: which names in THIS module are bound to a single base literal?
    # `PAD_CHAR = "A"` makes `PAD_CHAR * WINDOW` exactly as much a poly constructor as
    # `"A" * 101`, and the first version of this gate was blind to it.
    local_base_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            v = node.value.value
            if isinstance(v, str) and len(v) == 1 and v in _BASES:
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        local_base_names.add(tgt.id)

    def _describe(side: ast.expr) -> str | None:
        """Name this operand if it IS a placeholder base, else None."""
        if (
            isinstance(side, ast.Constant)
            and isinstance(side.value, str)
            and len(side.value) == 1
            and side.value in _BASES
        ):
            return f"{side.value!r} * ..."
        if isinstance(side, ast.Name):
            if side.id in local_base_names:
                return f"{side.id} * ...   (= a single base assigned in this module)"
            if side.id in _PLACEHOLDER_IDENTS:
                return f"{side.id} * ...   (a known placeholder identifier)"
        if isinstance(side, ast.Attribute) and side.attr in _PLACEHOLDER_IDENTS:
            return f"....{side.attr} * ...   (a known placeholder identifier)"
        return None

    # Pass 2: any `<placeholder base> * <anything>` construction.
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
            continue
        for side in (node.left, node.right):
            desc = _describe(side)
            if desc is not None:
                hits.append((node.lineno, desc))
                break
    return hits


def test_no_live_module_fabricates_a_poly_window_literal():
    """No module outside the two sanctioned ones may construct `"A" * n` / `"N" * n`.

    This is the gate that would have caught all four detectors at once, and it is what
    stops a fifth. It fires on CONSTRUCTION, because construction is the tell: a module
    that builds the placeholder string is a module that intends to compare against it or
    fill with it, and both are provenance questions it must not answer itself.
    """
    offenders: list[str] = []
    for path in _live_python_files():
        if path.name in _ALLOWED:
            continue
        for lineno, expr in _poly_literals(path):
            offenders.append(f"    {path.relative_to(_REPO)}:{lineno}  {expr}")

    assert not offenders, (
        "These modules construct a placeholder-window literal:\n"
        + "\n".join(offenders)
        + "\n\n"
        "A window whose content equals 'A'*101 MAY BE REAL -- poly-A tracts are real "
        "biology. Content cannot distinguish 'the reference genuinely says A' from 'the "
        "builder gave up and typed A'. Ask the artifact instead:\n\n"
        "    att = attach_delta_windows(meta, seq_windows_path=...)\n"
        "    usable = att.usable          # from the builder's own `ok` column\n"
        "    att.n_usable, att.n_unmapped, att.n_placeholder, att.summary()\n\n"
        "Four detectors did it the other way and all four were blind to the same 21,814 "
        "rows. When PLACEHOLDER_BASE changed, two of them did not start FAILING -- they "
        "started passing UNCONDITIONALLY, one of them a gate that refuses to write. "
        "Content checks go quiet when they rot. If you genuinely need to fabricate a "
        "window, add the module to _ALLOWED with the reason, and understand that you are "
        "deciding to manufacture sequence data."
    )


def test_this_ban_can_actually_fail():
    """Negative test -- CLAUDE.md 6.4: 'A guard is not real until you have watched it
    FAIL.' The AST filter above is narrow by design; prove it is not so narrow that it
    catches nothing."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "offender.py"
        p.write_text('POLY = "A" * 101\nx = 2 * 3\n', encoding="utf-8")
        assert _poly_literals(p) == [(1, "'A' * ...")], "the ban no longer catches 'A'*101"

        p.write_text('poly = "N" * window\n', encoding="utf-8")
        assert len(_poly_literals(p)) == 1, "the ban does not catch 'N'*window"

        # THE HOLE THAT populate_fasta_seq.py WALKED THROUGH -- these MUST now fire.
        # The first version of this gate matched only `Constant * X`, so every one of the
        # following was invisible to it. They are the same idea in different spellings,
        # and a gate that bans a spelling is not a gate.
        p.write_text('PAD_CHAR = "A"\n_POLY = PAD_CHAR * WINDOW\n', encoding="utf-8")
        assert len(_poly_literals(p)) == 1, (
            "the ban misses `PAD_CHAR * WINDOW` where PAD_CHAR is a single base assigned "
            "in the same module -- this is exactly how populate_fasta_seq.py evaded it"
        )

        p.write_text('_POLY_A = sw.PAD_CHAR * sw.WINDOW\n', encoding="utf-8")
        assert len(_poly_literals(p)) == 1, (
            "the ban misses `sw.PAD_CHAR * sw.WINDOW` -- the literal line from "
            "populate_fasta_seq.py:59 that this gate was supposed to catch"
        )

        p.write_text('x = PLACEHOLDER_BASE * window\n', encoding="utf-8")
        assert len(_poly_literals(p)) == 1, "the ban misses a known placeholder identifier"

        # ... but NOT on a multi-character literal (test fixtures build 'ACGT' * 25) ...
        p.write_text('win = "ACGT" * 25 + "A"\n', encoding="utf-8")
        assert _poly_literals(p) == [], "the ban fires on a multi-base fixture literal"

        # ... nor on a non-base single character ...
        p.write_text('bar = "-" * 80\n', encoding="utf-8")
        assert _poly_literals(p) == [], "the ban fires on an unrelated single-char repeat"

        # ... nor on PROSE quoting the banned form, which is the mistake this project
        # has now made three times.
        p.write_text('# the old code said: poly = "A" * 101\nx = 1\n', encoding="utf-8")
        assert _poly_literals(p) == [], "the ban fires on a comment describing the ban"

        p.write_text('"""Docstring mentioning `poly = "A" * 101` for context."""\n',
                     encoding="utf-8")
        assert _poly_literals(p) == [], "the ban fires on a docstring describing the ban"


def test_no_source_file_carries_a_byte_order_mark():
    """No Python source file may begin with a UTF-8 byte-order mark (U+FEFF).

    FOUND BY ACCIDENT, 2026-07-15, and kept because accidents are evidence.
    The poly ban above crashed on its first run:

        SyntaxError: invalid non-printable character U+FEFF
        scripts/build_spliceai_index.py

    The mark is INVISIBLE and HARMLESS-LOOKING. Python's tokeniser strips it (PEP 263),
    so the module imports, runs, and tests green -- it had been there long enough that
    nobody knows when it arrived. It only surfaced because a tool read the file with
    `utf-8` instead of `utf-8-sig` and got a different program than the interpreter does.

    That is the whole reason this is a gate. A byte-order mark is a silent divergence
    between "what Python sees" and "what everything else sees", and this project's entire
    method is that nothing may fail silently:

      * `ast.parse(f.read())` -- any source-analysis tool, including several in this
        repository's own test suite -- raises rather than analysing the file. A repo-wide
        gate that silently skips a file is worse than no gate, because it reports
        coverage it does not have.
      * `git diff` and merge tooling treat the mark as content on line 1.
      * A shebang after a byte-order mark is not a shebang: the kernel reads the raw
        bytes, sees `\\xef\\xbb\\xbf#!`, and the file is not executable as a script even
        though it looks like it is.

    The fix is to strip the mark and save as UTF-8 without a signature. It costs three
    bytes and removes an entire class of "why does the linter say that" from the project.
    """
    offenders: list[str] = []
    for root in (_SRC, _SCRIPTS, _REPO / "tests"):
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.py")):
            with p.open("rb") as fh:
                if fh.read(3) == b"\xef\xbb\xbf":
                    offenders.append(f"    {p.relative_to(_REPO)}")

    assert not offenders, (
        "These Python source files begin with a UTF-8 byte-order mark (U+FEFF):\n"
        + "\n".join(offenders)
        + "\n\n"
        "Python's tokeniser strips it, so they import and test green -- which is exactly "
        "why this went unnoticed. But `ast.parse(path.read_text(encoding='utf-8'))` "
        "raises `SyntaxError: invalid non-printable character U+FEFF`, so every "
        "source-analysis gate in this suite either dies or must silently skip the file. "
        "A gate that skips a file reports coverage it does not have.\n\n"
        "Fix, per file (Windows-side; the sandbox must not write tracked files):\n"
        "    $p = 'scripts/build_spliceai_index.py'\n"
        "    $t = [IO.File]::ReadAllText($p)              # strips the mark on read\n"
        "    [IO.File]::WriteAllText($p, $t, (New-Object Text.UTF8Encoding $false))\n"
    )


def test_the_sanctioned_modules_still_exist():
    """_ALLOWED must not silently accumulate names for files that are gone -- a stale
    allowlist entry is a permission nobody is watching."""
    live = {p.name for p in _live_python_files()} | {
        p.name for root in (_SRC, _SCRIPTS) if root.exists() for p in root.rglob("*.py")
    }
    missing = sorted(name for name in _ALLOWED if name not in live)
    assert not missing, (
        f"_ALLOWED names modules that no longer exist: {missing}. Delete the entries; an "
        f"allowlist that outlives its subject is dead weight, and dead weight is how "
        f"gene_is_constrained sat in KNOWN_ZERO_DEFAULT swallowing a regression."
    )
