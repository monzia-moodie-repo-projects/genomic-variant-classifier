"""The README's package list must name exactly the packages and top-level modules the code has.

Author: Monzia Moodie
Created 2026-11-27 (the research-scope revision of the README).

WHY
---
The README's repository section listed every subpackage of src/genomic_variant_classifier/ as of 2026-11-04. Nothing
checked it. Stage 1D-1a added information_sources/ -- the census, observation, absence, lineage, graph-identity, label,
population and claims authorities -- and the list never named it: measured on main b7983ab6, the code had 25 package
directories and the README listed 24. A reader saw the model roster and could not find the package that decides what
information a model may use.

The list now lives in one fenced block under "### Every package", and this file re-derives the agreement on every run:
add or remove a package or a top-level module, or a package's __init__.py, without changing the README, and the suite
fails, naming the difference. Every relative link in the README must also resolve.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
README = REPO / "README.md"
PACKAGE = REPO / "src" / "genomic_variant_classifier"
HEADING = "### Every package"
PACKAGE_LINE = re.compile(r"^  ([a-z][a-z0-9_]*)/\s+- \S")
MODULE_LINE = re.compile(r"^  ([a-z][a-z0-9_]*\.py)\s+- \S")
NAMESPACE_LINE = re.compile(r"^  \(no __init__\.py: (.+)\)$")
ROOT_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*/?)\s+- \S")
CONTINUATION = re.compile(r"^ {29,}\S")  # a description wrapped onto the next line, aligned under its text


def _block() -> list[str]:
    """The fenced block that follows the inventory heading, as lines (refuses if the heading or the fence is not unique)."""
    text = README.read_text(encoding="utf-8")
    assert text.count(HEADING + "\n") == 1, f"README must carry exactly one {HEADING!r} heading"
    after = text.split(HEADING + "\n", 1)[1]
    fences = [i for i, line in enumerate(after.splitlines()) if line.startswith("```")]
    assert len(fences) >= 2, f"no fenced block after {HEADING!r}"
    lines = after.splitlines()[fences[0] + 1:fences[1]]
    assert lines and lines[0].startswith("src/genomic_variant_classifier/"), \
        "the inventory block must open with src/genomic_variant_classifier/"
    return lines


def _package_lines() -> list[str]:
    lines = _block()
    body = []
    for line in lines[1:]:
        if not line.startswith("  "):
            break
        body.append(line)
    return body


def _listed(pattern: re.Pattern[str]) -> list[str]:
    return [m.group(1) for m in map(pattern.match, _package_lines()) if m]


def _directories() -> set[str]:
    return {p.name for p in PACKAGE.iterdir() if p.is_dir() and p.name != "__pycache__" and not p.name.startswith(".")}


def test_every_line_of_the_package_list_is_one_package_one_module_or_the_namespace_note():
    lines = _package_lines()
    unparsed = [line for line in lines
                if not (PACKAGE_LINE.match(line) or MODULE_LINE.match(line) or NAMESPACE_LINE.match(line)
                        or CONTINUATION.match(line))]
    assert unparsed == [], f"package-list lines that name neither a package, a module nor the namespace note: {unparsed}"
    assert lines and not CONTINUATION.match(lines[0]), "the package list cannot open with a continuation line"


def test_the_listed_packages_are_exactly_the_package_directories():
    listed = _listed(PACKAGE_LINE)
    assert len(listed) == len(set(listed)), f"a package is listed twice: {sorted(listed)}"
    actual = _directories()
    assert set(listed) == actual, (f"in the code, not in the README: {sorted(actual - set(listed))}; "
                                   f"in the README, not in the code: {sorted(set(listed) - actual)}")


def test_the_listed_top_level_modules_are_exactly_the_modules():
    listed = _listed(MODULE_LINE)
    assert len(listed) == len(set(listed)), f"a module is listed twice: {sorted(listed)}"
    actual = {p.name for p in PACKAGE.glob("*.py") if p.name != "__init__.py"}
    assert set(listed) == actual, (f"in the code, not in the README: {sorted(actual - set(listed))}; "
                                   f"in the README, not in the code: {sorted(set(listed) - actual)}")


def test_the_namespace_note_names_exactly_the_directories_without_an_init_file():
    notes = [m.group(1) for m in map(NAMESPACE_LINE.match, _package_lines()) if m]
    assert len(notes) == 1, f"the package list must carry exactly one '(no __init__.py: ...)' line, found {len(notes)}"
    named = {part.strip().rstrip("/") for part in notes[0].split(",")}
    actual = {d for d in _directories() if not (PACKAGE / d / "__init__.py").is_file()}
    assert named == actual, f"without __init__.py in the code: {sorted(actual)}; named in the README: {sorted(named)}"


def test_every_repository_path_the_block_names_exists():
    roots = [m.group(1) for m in map(ROOT_LINE.match, _block()[1 + len(_package_lines()):]) if m]
    assert roots, "the block names no repository path after the package list"
    missing = [r for r in roots if not (REPO / r.rstrip("/")).exists()]
    assert missing == [], f"repository paths named in the README that do not exist: {missing}"


def test_every_relative_link_in_the_readme_resolves():
    # One test, not one per link: the suite-size ratchet must not move when the README gains a link.
    targets = sorted({t for t in re.findall(r"\]\(([^)\s]+)\)", README.read_text(encoding="utf-8"))
                      if not re.match(r"^[a-z]+:", t) and not t.startswith("#")})
    assert targets, "the README has no relative link"
    missing = [t for t in targets if not (REPO / t.split("#", 1)[0]).exists()]
    assert missing == [], f"README links that resolve to nothing: {missing}"
