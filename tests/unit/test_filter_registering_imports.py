"""WARNING-FILTER-REORDER-AT-FIRST-IMPORT-1 (2026-11-05): no test's filters depend on which tests ran with it.

Some libraries register warning filters when first imported (mlxtend's frequent_patterns/fpcommon.py:8 calls
simplefilter("always", DeprecationWarning) at module level; spaCy registers numpy's two "size changed" entries again). pytest
collects inside warnings.catch_warnings() and discards what collection registered, so in a full run no test runs under those
entries; in a run of fewer files a library could be imported for the first time inside a test, and what it registered went to the
front of that test's filters, ahead of pyproject.toml's. MEASURED 2026-11-04: tests/unit/test_1kgp.py alone gave 46 passed, 1
error (the warning-filter guard, at the teardown of the test whose body first imported the ensemble).

The repository root's conftest.py now imports every module in _FILTER_REGISTERING_MODULES at session start, inside
catch_warnings(), so what they register is discarded as collection discards it; checks around collection that nothing collection
imported registered a filter; and fails a test that registers one. Pinned here, in this session and in child sessions running a
copy of the root conftest.py whose list names planted modules instead:
* this session imported every listed module, and none is absent from this environment (the training-and-test one);
* a listed module's registrations are discarded, a warning it emits at import reaches pytest's report, and a listed module this
  environment lacks is named, not imported;
* a module that registers filters, imported for the first time inside a test and NOT listed, fails that test at teardown, naming
  the entry it added and the identical one it moved ahead, each with where it stands now;
* the same module imported at collection and not listed -- or a test module that registers a filter at module level -- stops the
  session before any test runs, naming the entries; a listed module that is installed and fails to import stops it too;
* pyproject.toml's two entries for torch_geometric hide exactly the three warnings its import emits about its own code, and this
  project calls neither deprecated interface (parsed).

Author: Monzia Moodie
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT_CONFTEST = REPO / "conftest.py"
LIST_NAME = "_FILTER_REGISTERING_MODULES"

# --------------------------------------------------------------------------- in this session


def _root_conftest_module(config):
    found = [plugin for plugin in config.pluginmanager.get_plugins()
             if Path(getattr(plugin, "__file__", None) or "-").resolve() == ROOT_CONFTEST]
    assert len(found) == 1, found
    return found[0]


def _declared() -> tuple:
    tree = ast.parse(ROOT_CONFTEST.read_text(encoding="utf-8"), filename=str(ROOT_CONFTEST))
    values = [node for node in tree.body if isinstance(node, ast.Assign)
              and [getattr(t, "id", "") for t in node.targets] == [LIST_NAME]]
    assert len(values) == 1, values
    return values[0]


def test_this_session_imported_every_listed_module_at_session_start_and_none_is_absent(request):
    module = _root_conftest_module(request.config)
    declared = ast.literal_eval(_declared().value)
    assert declared == module._FILTER_REGISTERING_MODULES
    imported, absent, seconds = request.config.stash[module._WARNING_GUARD_TALLY]["imported"]
    # tests/ runs in the training-and-test environment, which has every one of them (both locks, read 2026-11-04).
    assert imported == declared and absent == (), (imported, absent)
    assert all(name in sys.modules for name in declared)
    assert 0 < seconds < 120, seconds


def test_every_shape_of_difference_is_found_and_each_entry_placed_exactly(request):
    """The comparison the guard and the collection check share, on every shape a list can take (the independent review of
    2026-11-05 found the first version's "now at the front" wrong for an appended entry, one inserted in the middle and a swap)."""
    differences = _root_conftest_module(request.config)._filter_differences
    a, b, c, d, x = (("ignore", None, cls, None, 0) for cls in (UserWarning, DeprecationWarning, FutureWarning, RuntimeWarning,
                                                                  SyntaxWarning))
    start, bound = [a, b, c, d], object()

    def found(after, same_list=True):
        return differences(bound, tuple(start), bound if same_list else object(), tuple(after), "before setup", "this test")

    assert found([a, b, c, d]) == []
    appended = found([a, b, c, d, x])
    assert appended[1].endswith("(position 5 of 5; ahead of 0 of the 4 entries this test started with)") and len(appended) == 2
    middle = found([a, b, x, c, d])
    assert middle[1].endswith("(position 3 of 5; ahead of 2 of the 4 entries this test started with)") and len(middle) == 2
    front = found([x, a, b, c, d])
    assert front[1].endswith("(position 1 of 5; ahead of 4 of the 4 entries this test started with)") and len(front) == 2
    swap = found([a, c, b, d])                               # only c moved ahead (of b); b moved nowhere ahead
    assert swap[0].endswith("no longer in their original order (order is precedence):") and len(swap) == 2
    assert "SyntaxWarning" not in swap[1] and "FutureWarning" in swap[1]
    assert swap[1].endswith("(now ahead of 1 entry it used to follow)")
    moved = found([d, a, b, c])
    assert len(moved) == 2 and "RuntimeWarning" in moved[1] and moved[1].endswith("(now ahead of 3 entries it used to follow)")
    inserted_again = found([b, a, b, c, d])                  # a direct insert of an existing entry: b now precedes a
    assert len(inserted_again) == 2 and "DeprecationWarning" in inserted_again[1] and inserted_again[1].endswith(
        "(now ahead of 1 entry it used to follow)")
    duplicated = found([a, b, c, d, d])                      # nothing new, gone or out of order -- and still not the same list
    assert "the list still differs: duplicated" in duplicated[0] and duplicated[1].endswith("(now 2 times, 1 before)")
    removed = found([a, c, d])
    assert removed[0] == "the filter entries changed: 1 gone, 0 new" and "DeprecationWarning" in removed[1]
    rebound = found([a, b, c, d], same_list=False)
    assert rebound == ["warnings.filters is a DIFFERENT list from the one bound before setup (4 entries, now 4)"]


# --------------------------------------------------------------------------- in child sessions

REGISTERS = '''
import warnings

warnings.filterwarnings("ignore", message="planted at import", category=UserWarning)
warnings.simplefilter("always", DeprecationWarning)     # identical to an entry pytest registers: it is MOVED to the front
'''

BROKEN = '''
import warnings
from pathlib import Path

warnings.warn("planted warning before the failure", UserWarning)
Path("written_by_a_broken_import.txt").write_text("x", encoding="utf-8")
raise RuntimeError("planted failure at import")
'''

WARNS_AN_ERROR = 'import warnings\n\nwarnings.warn("planted resource warning", ResourceWarning)\n'

WARNS = 'import warnings\n\nwarnings.warn("planted warning at import", UserWarning)\n'

FIRST_IMPORTED_INSIDE_A_TEST = '''
import warnings


def test_imports_a_module_that_registers_filters():
    import gvc_planted_registers  # noqa: F401
    planted = [f for f in warnings.filters if getattr(f[1], "pattern", None) == "planted at import"]
    assert len(planted) == int(__import__("os").environ["GVC_EXPECTED_PLANTED"]), warnings.filters


def test_after_it():
    pass
'''

IMPORTED_AT_COLLECTION = '''
import gvc_planted_registers  # noqa: F401


def test_never_runs():
    pass
'''

REPLACES_SHOWWARNING_AT_MODULE_LEVEL = '''
import warnings

warnings.showwarning = lambda *args, **kwargs: None


def test_never_runs():
    pass
'''

REGISTERS_AT_MODULE_LEVEL = '''
import warnings

warnings.filterwarnings("ignore", message="planted by a test module", category=UserWarning)


def test_never_runs():
    pass
'''

CHILD_INI = "[pytest]\nfilterwarnings =\n    error::ResourceWarning\n"


def _child(tmp_path: Path, environment, conftest, listed: tuple, tests: str, expected_planted: int = 0) -> tuple:
    project = tmp_path / "project"
    (project / "tests").mkdir(parents=True)
    (project / "pytest.ini").write_text(CHILD_INI, encoding="utf-8")
    (project / "conftest.py").write_bytes(conftest(*listed))
    (project / "tests" / "test_child.py").write_text(tests, encoding="utf-8")
    library = tmp_path / "library"
    library.mkdir()
    (library / "gvc_planted_registers.py").write_text(REGISTERS, encoding="utf-8")
    (library / "gvc_planted_broken.py").write_text(BROKEN, encoding="utf-8")
    (library / "gvc_planted_warns.py").write_text(WARNS, encoding="utf-8")
    (library / "gvc_planted_warns_an_error.py").write_text(WARNS_AN_ERROR, encoding="utf-8")
    env = environment(PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(library)] +
                                                 ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])),
                      GVC_EXPECTED_PLANTED=str(expected_planted))
    run = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "--confcutdir", str(project)],
                         cwd=str(project), env=env, capture_output=True, text=True, timeout=600)
    return run.returncode, run.stdout + run.stderr


def test_a_child_projects_conftest_differs_from_the_root_one_in_the_list_alone(child_conftest):
    """The root conftest's child_conftest fixture -- the one source for every child session's conftest.py (four test files)."""
    original = ROOT_CONFTEST.read_text(encoding="utf-8").splitlines()
    node = _declared()
    for names, line in (((), "_FILTER_REGISTERING_MODULES = ()"),
                        (("gvc_planted_registers",), "_FILTER_REGISTERING_MODULES = ('gvc_planted_registers',)")):
        copy = child_conftest(*names).decode("utf-8").splitlines()
        assert copy[:node.lineno - 1] == original[:node.lineno - 1]
        assert copy[node.lineno - 1] == line
        assert copy[node.lineno:] == original[node.end_lineno:]
        assert len(copy) == len(original) - (node.end_lineno - node.lineno)


def test_a_module_first_imported_inside_a_test_and_not_listed_fails_that_test(tmp_path, child_pytest_environment, child_conftest):
    code, out = _child(tmp_path, child_pytest_environment, child_conftest, (), FIRST_IMPORTED_INSIDE_A_TEST, expected_planted=1)
    assert code == 1, out
    assert "2 passed, 1 error" in out, out
    assert out.count(" ERROR at teardown of ") == 1, out
    assert " ERROR at teardown of test_imports_a_module_that_registers_filters " in out, out
    assert "1 filter entry was registered" in out, out
    assert "    new:  ignore  message='planted at import'  category=UserWarning  module=None  line=0" in out, out
    assert "the entries this test started with are no longer in their original order (order is precedence)" in out, out
    new = re.search(r"    new:  ignore  message='planted at import'  category=UserWarning  module=None  line=0  "
                    r"\(position 2 of (\d+); ahead of (\d+) of the (\d+) entries this test started with\)", out)
    assert new and int(new.group(1)) == int(new.group(3)) + 1 and int(new.group(2)) == int(new.group(3)) - 1, out
    assert re.search(r"    moved: always  message=None  category=DeprecationWarning  module=None  line=0  "
                     r"\(now ahead of [1-9]\d* entr(y|ies) it used to follow\)", out), out
    assert "2 tests checked, 1 changed the warning state" in out, out
    assert "imported at session start, what they register discarded: none in " in out, out


def test_a_listed_module_registers_nothing_any_test_sees_and_a_listed_absent_one_is_named(tmp_path, child_pytest_environment,
                                                                                          child_conftest):
    """A listed module's warning at import reaches pytest's own report, as one emitted during collection would."""
    listed = ("gvc_planted_absent_module", "gvc_planted_registers", "gvc_planted_warns")
    code, out = _child(tmp_path, child_pytest_environment, child_conftest, listed, FIRST_IMPORTED_INSIDE_A_TEST, expected_planted=0)
    assert code == 0, out
    assert "2 passed, 1 warning in " in out and "error" not in out, out
    assert "warnings summary" in out and "UserWarning: planted warning at import" in out, out
    assert "2 tests checked, 0 changed the warning state" in out, out
    assert ("imported at session start, what they register discarded: gvc_planted_registers, gvc_planted_warns in " in out and
            "; absent in this environment: gvc_planted_absent_module" in out), out


def test_a_module_imported_at_collection_and_not_listed_stops_the_session(tmp_path, child_pytest_environment, child_conftest):
    code, out = _child(tmp_path, child_pytest_environment, child_conftest, (), IMPORTED_AT_COLLECTION)
    assert code == 4, out                                                           # pytest.ExitCode.USAGE_ERROR
    assert "THE WARNING STATE CHANGED WHILE TESTS WERE COLLECTED (WARNING-FILTER-REORDER-AT-FIRST-IMPORT-1)" in out, out
    assert "    new:  ignore  message='planted at import'  category=UserWarning  module=None  line=0" in out, out
    assert "the entries collection started with are no longer in their original order" in out, out
    assert "    moved: always  message=None  category=DeprecationWarning  module=None  line=0  (now ahead of " in out, out
    assert "add the module that registers them (or one that imports it) to _FILTER_REGISTERING_MODULES" in out, out
    assert "passed" not in out and "0 tests checked" in out, out


def test_a_test_module_that_registers_a_filter_at_module_level_stops_the_session(tmp_path, child_pytest_environment, child_conftest):
    code, out = _child(tmp_path, child_pytest_environment, child_conftest, (), REGISTERS_AT_MODULE_LEVEL)
    assert code == 4, out
    assert "THE WARNING STATE CHANGED WHILE TESTS WERE COLLECTED" in out, out
    assert "    new:  ignore  message='planted by a test module'  category=UserWarning  module=None  line=0" in out, out
    assert "pytestmark = pytest.mark.filterwarnings(...)" in out and "passed" not in out, out


def test_a_test_module_that_replaces_showwarning_at_module_level_stops_the_session(tmp_path, child_pytest_environment,
                                                                                   child_conftest):
    """The collection check compares the whole warning state the per-test guard compares, not only the filters."""
    code, out = _child(tmp_path, child_pytest_environment, child_conftest, (), REPLACES_SHOWWARNING_AT_MODULE_LEVEL)
    assert code == 4, out
    assert "THE WARNING STATE CHANGED WHILE TESTS WERE COLLECTED" in out and "warnings.showwarning was replaced" in out, out
    assert "passed" not in out, out


def test_a_listed_module_that_is_installed_and_fails_to_import_stops_the_session_and_says_everything(tmp_path,
                                                                                                   child_pytest_environment,
                                                                                                   child_conftest):
    """pytest runs neither the terminal summary nor sessionfinish after session start raises, so the usage error itself carries
    the warning the import emitted and the file it wrote into the checkout (the independent review of 2026-11-05 measured both
    lost)."""
    code, out = _child(tmp_path, child_pytest_environment, child_conftest, ("gvc_planted_broken",),
                       "def test_never_runs():\n    pass\n")
    assert code == 4, out
    assert "imports 'gvc_planted_broken' at session start" in out, out
    assert "It is installed and failed to import: RuntimeError: planted failure at import" in out, out
    assert "  warned while importing: UserWarning: planted warning before the failure (" in out, out
    assert "  THE IMPORT CHANGED THE WORKING TREE:" in out and "written_by_a_broken_import.txt" in out, out
    assert "passed" not in out and "a configured `error::` filter" not in out, out


def test_a_listed_module_whose_import_warning_is_an_error_entry_stops_the_session_and_says_so(tmp_path, child_pytest_environment,
                                                                                            child_conftest):
    """The child's configuration has error::ResourceWarning: the import's ResourceWarning is raised, and the message says why."""
    code, out = _child(tmp_path, child_pytest_environment, child_conftest, ("gvc_planted_warns_an_error",),
                       "def test_never_runs():\n    pass\n")
    assert code == 4, out
    assert "It is installed and failed to import: ResourceWarning: planted resource warning" in out, out
    assert "That exception is a warning: a configured `error::` filter turned it into one" in out, out
    assert "  The working tree is unchanged." in out and "passed" not in out, out


# --------------------------------------------------------------------------- the two torch_geometric entries in pyproject.toml
PREMISE = r'''
import json
import sys
import warnings

from _pytest.config import parse_warning_filter

with warnings.catch_warnings(record=True) as log:
    warnings.simplefilter("always")
    for entry in json.loads(sys.argv[1]):
        warnings.filterwarnings(*parse_warning_filter(entry, escape=False))
    import torch_geometric  # noqa: F401
print(json.dumps([[w.category.__name__, w.filename.replace("\\", "/").split("site-packages/")[-1], w.lineno, str(w.message)]
                  for w in log]))
'''
TORCH_GEOMETRIC_WARNINGS = [
    ["DeprecationWarning", "torch_geometric/llm/utils/backend_utils.py", 26,
     "`torch_geometric.distributed` has been deprecated since 2.7.0 and will no longer be maintained."],
    ["DeprecationWarning", "torch/jit/_script.py", 1488,
     "`torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`."],
    ["DeprecationWarning", "torch/jit/_script.py", 1488,
     "`torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`."],
]


def _pyproject_entries() -> list:
    import tomllib

    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"]["filterwarnings"]


def _emitted_by_a_fresh_import_of_torch_geometric(entries: list) -> list:
    import json

    done = subprocess.run([sys.executable, "-I", "-c", PREMISE, json.dumps(entries)], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    return json.loads(done.stdout)


def test_a_fresh_import_of_torch_geometric_warns_exactly_three_times_and_the_two_entries_hide_exactly_those():
    entries = [e for e in _pyproject_entries() if "torch_geometric" in e or "torch\\.jit" in e]
    assert entries == [
        "ignore:`torch_geometric\\.distributed` has been deprecated:DeprecationWarning:"
        "torch_geometric\\.llm\\.utils\\.backend_utils",
        "ignore:`torch\\.jit\\.script` is deprecated:DeprecationWarning:torch\\.jit\\._script"], entries
    emitted = _emitted_by_a_fresh_import_of_torch_geometric([])
    assert [w[:3] for w in emitted] == [w[:3] for w in TORCH_GEOMETRIC_WARNINGS], emitted
    assert all(w[3].startswith(expected[3]) for w, expected in zip(emitted, TORCH_GEOMETRIC_WARNINGS)), emitted
    assert _emitted_by_a_fresh_import_of_torch_geometric(entries) == []


def _uses_of_the_deprecated_apis(path: Path) -> list:
    """torch.jit.script (as an attribute chain or imported from torch.jit) and torch_geometric.distributed, in one file. Parsed; a
    file that cannot be parsed raises here rather than being skipped."""
    import ast as _ast

    tree = _ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Attribute) and node.attr == "script" and isinstance(node.value, _ast.Attribute) \
                and node.value.attr == "jit":
            found.append("{}:{} .jit.script".format(path.name, node.lineno))
        elif isinstance(node, _ast.ImportFrom) and (node.module or "").startswith("torch.jit") and \
                any(a.name == "script" for a in node.names):
            found.append("{}:{} from {} import script".format(path.name, node.lineno, node.module))
        elif isinstance(node, _ast.ImportFrom) and ((node.module or "").startswith("torch_geometric.distributed") or (
                node.module == "torch_geometric" and any(a.name == "distributed" for a in node.names))):
            found.append("{}:{} from {}".format(path.name, node.lineno, node.module))
        elif isinstance(node, _ast.Import) and any(a.name.startswith("torch_geometric.distributed") for a in node.names):
            found.append("{}:{} import torch_geometric.distributed".format(path.name, node.lineno))
    return found


def test_this_project_calls_neither_deprecated_api_so_the_two_entries_hide_nothing_of_its_own(tmp_path):
    files = sorted((REPO / "src").rglob("*.py")) + sorted((REPO / "scripts").rglob("*.py"))
    assert len(files) > 300, len(files)                                             # the scan reached both trees
    found = [use for path in files for use in _uses_of_the_deprecated_apis(path)]
    assert found == [], found
    planted = tmp_path / "planted.py"
    planted.write_text("import torch\nfrom torch.jit import script\nfrom torch_geometric import distributed\n"
                       "import torch_geometric.distributed\nf = torch.jit.script(len)\n", encoding="utf-8")
    assert _uses_of_the_deprecated_apis(planted) == [
        "planted.py:2 from torch.jit import script", "planted.py:3 from torch_geometric",
        "planted.py:4 import torch_geometric.distributed", "planted.py:5 .jit.script"], _uses_of_the_deprecated_apis(planted)
