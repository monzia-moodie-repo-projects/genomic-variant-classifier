"""Tests for the AST import census -- DEPENDENCY-SCOPE-CENSUS.

Author: Monzia Moodie
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import pytest

from genomic_variant_classifier.deps.dependency_census import (
    CensusAudit, CensusError, ImportRequirement, ImportSite, census, report,
)


def _tree(files: dict) -> Path:
    root = Path(tempfile.mkdtemp())
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        io.open(p, "w", encoding="utf-8", newline="\n").write(text)
    return root


# ---- THE REASON THIS IS AST AND NOT GREP -------------------------------
def test_a_package_named_only_in_COMMENTS_is_not_a_consumer():
    """Eleven times this session a text search matched its own documentation.
    A comment mentioning a package is not an import of it."""
    root = _tree({"src/a.py": "# pyfaidx would be needed for random access\nx = 1\n"})
    sites, _ = census(["src"], ["pyfaidx"], repo_root=root)
    assert sites["pyfaidx"] == ()


def test_a_package_named_only_in_a_DOCSTRING_is_not_a_consumer():
    root = _tree({"src/a.py": '"""Uses pyfaidx for random access."""\nx = 1\n'})
    sites, _ = census(["src"], ["pyfaidx"], repo_root=root)
    assert sites["pyfaidx"] == ()


def test_a_package_named_only_in_a_STRING_is_not_a_consumer():
    root = _tree({"src/a.py": 'raise ImportError("install pyfaidx first")\n'})
    sites, _ = census(["src"], ["pyfaidx"], repo_root=root)
    assert sites["pyfaidx"] == ()


def test_a_real_import_IS_a_consumer():
    root = _tree({"src/a.py": "import pyfaidx\n"})
    sites, _ = census(["src"], ["pyfaidx"], repo_root=root)
    assert len(sites["pyfaidx"]) == 1
    assert sites["pyfaidx"][0].lineno == 1
    assert sites["pyfaidx"][0].path == "src/a.py"


def test_a_from_import_IS_a_consumer():
    root = _tree({"src/a.py": "from pyfaidx import Fasta\n"})
    sites, _ = census(["src"], ["pyfaidx"], repo_root=root)
    assert len(sites["pyfaidx"]) == 1


def test_a_submodule_import_counts_under_its_top_level_package():
    root = _tree({"src/a.py": "import matplotlib.pyplot as plt\n"})
    sites, _ = census(["src"], ["matplotlib"], repo_root=root)
    assert len(sites["matplotlib"]) == 1


def test_a_RELATIVE_import_is_never_an_external_consumer():
    """`from .seaborn import x` is a local module, not the package."""
    root = _tree({"src/a.py": "from .seaborn import helper\n"})
    sites, _ = census(["src"], ["seaborn"], repo_root=root)
    assert sites["seaborn"] == ()


# ---- try-guarded imports are a DIFFERENT claim --------------------------
@pytest.mark.parametrize("handler,expected", [
    ("except ImportError:",               ImportRequirement.IMPORTERROR_GUARDED),
    ("except ModuleNotFoundError:",       ImportRequirement.MODULENOTFOUND_GUARDED),
    ("except Exception:",                 ImportRequirement.BROAD_EXCEPTION_GUARDED),
    ("except:",                           ImportRequirement.BROAD_EXCEPTION_GUARDED),
    ("except (ValueError, ImportError):", ImportRequirement.IMPORTERROR_GUARDED),
    ("except ValueError:",                ImportRequirement.HARD),
    ("except KeyError:",                  ImportRequirement.HARD),
    ("except (ValueError, KeyError):",    ImportRequirement.HARD),
])
def test_an_import_is_classified_by_WHAT_THE_HANDLER_CATCHES(handler, expected):
    """The measured bug. An earlier version tested only that a handler EXISTED,
    so `except ValueError` was reported optional -- but an unavailable package
    raises ImportError, which escapes it, and the import is HARD.

    Three of these eight shapes were classified wrongly before this test.
    """
    root = _tree({"src/a.py":
                  "try:\n    import seaborn\n{}\n    pass\n".format(handler)})
    sites, _ = census(["src"], ["seaborn"], repo_root=root)
    assert sites["seaborn"][0].requirement is expected


def test_ModuleNotFoundError_is_a_WEAKER_guard_than_ImportError():
    """Measured: ModuleNotFoundError subclasses ImportError, so catching
    ImportError catches a missing package -- but catching ModuleNotFoundError
    does NOT catch a plain ImportError from a partially broken package."""
    assert issubclass(ModuleNotFoundError, ImportError)
    assert not issubclass(ImportError, ModuleNotFoundError)
    assert (ImportRequirement.MODULENOTFOUND_GUARDED
            is not ImportRequirement.IMPORTERROR_GUARDED)


def test_an_unguarded_import_is_HARD():
    root = _tree({"src/a.py": "import seaborn\n"})
    sites, _ = census(["src"], ["seaborn"], repo_root=root)
    assert sites["seaborn"][0].requirement is ImportRequirement.HARD
    assert sites["seaborn"][0].guarded is False


def test_a_try_WITHOUT_a_handler_is_HARD():
    """try/finally has no except at all -- the code cannot proceed without it."""
    root = _tree({"src/a.py": "try:\n    import seaborn\nfinally:\n    pass\n"})
    sites, _ = census(["src"], ["seaborn"], repo_root=root)
    assert sites["seaborn"][0].requirement is ImportRequirement.HARD


def test_a_PARTIAL_census_RAISES_unless_explicitly_allowed():
    """A partially measured topology must not present itself as a complete one.

    Previously only walked-and-parsed-NONE raised, so 940 parsed with one
    critical file failing returned a "successful" census.
    """
    root = _tree({"src/ok.py": "import httpx\n", "src/broken.py": "def (((\n"})
    with pytest.raises(CensusError) as exc:
        census(["src"], ["httpx"], repo_root=root)
    assert "partially measured" in str(exc.value)

    sites, audit = census(["src"], ["httpx"], repo_root=root, allow_partial=True)
    assert len(sites["httpx"]) == 1
    assert len(audit.parse_failures) == 1






# ---- scope, which is what step 5 needs ---------------------------------
def test_imports_are_located_by_root_so_scope_can_be_judged():
    """A package imported in src/ while declared development-only is the
    torch_geometric shape. The census reports WHERE; a person decides."""
    root = _tree({
        "src/prod.py":    "import pyfaidx\n",
        "tests/t.py":     "import pyfaidx\nimport httpx\n",
        "scripts/run.py": "import httpx\n",
    })
    sites, _ = census(["src", "scripts", "tests"], ["pyfaidx", "httpx"],
                      repo_root=root)
    assert {s.path for s in sites["pyfaidx"]} == {"src/prod.py", "tests/t.py"}
    assert {s.path for s in sites["httpx"]} == {"tests/t.py", "scripts/run.py"}


def test_a_package_with_NO_importer_is_reported_as_measured_and_absent():
    """Present in the result with an empty tuple -- so "measured and absent" is
    distinguishable from "never asked"."""
    root = _tree({"src/a.py": "import os\n"})
    sites, _ = census(["src"], ["jinja2"], repo_root=root)
    assert "jinja2" in sites and sites["jinja2"] == ()


def test_results_are_keyed_by_CANONICAL_DISTRIBUTION_name():
    """The key is the distribution, not the module it happens to provide.

    An earlier version keyed on `.lower().replace("-", "_")` -- a
    distribution-to-module TRANSFORMATION, which is the string surgery the
    shared model forbids. MEASURED against installed metadata, four of seven
    sampled distributions disagree with that guess, including pyBigWig, whose
    real module keeps its capitals while Python imports are case-sensitive.
    """
    root = _tree({"src/a.py": "import pytest_cov\n"})
    sites, _ = census(["src"], ["pytest-cov"], repo_root=root,
                      installed={"pytest_cov": ["pytest-cov"]})
    assert "pytest-cov" in sites, sorted(sites)
    assert len(sites["pytest-cov"]) == 1
    assert sites["pytest-cov"][0].module == "pytest_cov"


def test_a_module_that_does_NOT_match_the_distribution_name_is_found():
    """beautifulsoup4 imports as bs4. No hyphen rule produces that, so the
    mapping must come from metadata rather than from a transformation."""
    root = _tree({"src/a.py": "import bs4\n"})
    sites, _ = census(["src"], ["beautifulsoup4"], repo_root=root,
                      installed={"bs4": ["beautifulsoup4"]})
    assert len(sites["beautifulsoup4"]) == 1


def test_an_uninstalled_distribution_records_its_guess_as_an_ASSUMPTION():
    """There is no alternative to guessing when metadata is absent -- but the
    guess is RECORDED as ASSUMED_IDENTICAL rather than hidden in a replace()."""
    from genomic_variant_classifier.deps.dependency_census import resolve_modules
    from genomic_variant_classifier.deps.model import MappingSource
    m = resolve_modules("not-installed-anywhere", installed={})
    assert m.source is MappingSource.ASSUMED_IDENTICAL
    assert any(str(x) == "not_installed_anywhere" for x in m.modules)


def test_metadata_beats_the_guess_when_available():
    from genomic_variant_classifier.deps.dependency_census import resolve_modules
    from genomic_variant_classifier.deps.model import MappingSource
    m = resolve_modules("beautifulsoup4", installed={"bs4": ["beautifulsoup4"]})
    assert m.source is MappingSource.PACKAGE_METADATA
    assert [str(x) for x in m.modules] == ["bs4"]


# ---- the census must not under-report silently -------------------------
def test_an_unparseable_file_is_NAMED_not_skipped():
    root = _tree({"src/ok.py": "import httpx\n",
                  "src/broken.py": "def (((\n"})
    sites, audit = census(["src"], ["httpx"], repo_root=root,
                                     allow_partial=True)
    assert len(sites["httpx"]) == 1
    assert len(audit.parse_failures) == 1
    assert "broken.py" in audit.parse_failures[0]
    assert audit.reconciles()


def test_a_missing_root_RAISES_rather_than_reporting_nothing():
    """A root that is not walked is not a root that found nothing -- the same
    distinction as absence versus unmeasured."""
    root = _tree({"src/a.py": "import httpx\n"})
    with pytest.raises(CensusError) as exc:
        census(["src", "nonexistent"], ["httpx"], repo_root=root)
    assert "does not exist" in str(exc.value)


def test_walking_files_but_parsing_NONE_is_an_instrument_failure():
    root = _tree({"src/a.py": "def (((\n", "src/b.py": "class ###\n"})
    with pytest.raises(CensusError) as exc:
        census(["src"], ["httpx"], repo_root=root,
                                     allow_partial=True)
    assert "instrument failure" in str(exc.value)


def test_the_audit_counts_walked_parsed_and_failed_separately():
    """The counts are reported; the equality between them is TRUE BY
    CONSTRUCTION and is not a guard.

    An earlier version RAISED on `not audit.reconciles()`, which sabotage
    removed with no test failure -- because every file takes exactly one
    branch, so the equality can never be false. The same tautology was written
    into requirements_parse.py twice. There it was replaced by a physical-line
    identity; here there is no independent quantity, so the false guard was
    deleted instead of disguised.
    """
    root = _tree({"src/a.py": "import httpx\n", "src/b.py": "x = 1\n",
                  "src/c.py": "def (((\n"})
    _, audit = census(["src"], ["httpx"], repo_root=root,
                                     allow_partial=True)
    assert audit.files_walked == 3
    assert audit.files_parsed == 2
    assert len(audit.parse_failures) == 1
    assert audit.reconciles(), "construction-true; asserted as documentation"


def test_a_virtual_environment_is_excluded_from_the_walk():
    """Counting a package's own internal imports would report every dependency
    as its own consumer."""
    root = _tree({"src/a.py": "x = 1\n",
                  "src/.venv312/lib/httpx/__init__.py": "import httpx\n"})
    sites, audit = census(["src"], ["httpx"], repo_root=root)
    assert sites["httpx"] == ()
    assert audit.files_walked == 1


def test_the_audit_is_immutable():
    import dataclasses
    root = _tree({"src/a.py": "import httpx\n"})
    _, audit = census(["src"], ["httpx"], repo_root=root)
    try:
        audit.files_walked = 999
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("the census audit was mutable")


def test_an_import_site_is_immutable():
    import dataclasses
    root = _tree({"src/a.py": "import httpx\n"})
    sites, _ = census(["src"], ["httpx"], repo_root=root)
    try:
        sites["httpx"][0].lineno = 999
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("an import site was mutable")


def test_the_report_names_a_package_with_no_importer():
    root = _tree({"src/a.py": "import os\n"})
    sites, audit = census(["src"], ["jinja2"], repo_root=root)
    text = report(sites, audit)
    assert "NO IMPORT ANYWHERE" in text
    assert "Verify before removing" in text


def main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = []
    for name, fn in tests:
        try:
            fn(); print("  PASS  {}".format(name))
        except Exception as exc:                        # noqa: BLE001
            failures.append(name); print("  FAIL  {}  {}".format(name, exc))
    print("\n  {} passed, {} failed, {} total".format(
        len(tests) - len(failures), len(failures), len(tests)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
