"""Guard the resume path in scripts/run_phase2_eval.py: it MUST reconstruct the ensemble via
VariantEnsemble.load() (format_version=2 dict -> object), never a raw joblib.load() that returns
the orchestrator dict and crashes the subsequent .evaluate(). Author: Monzia Moodie.

2026-09-23: read STRUCTURALLY. The previous version inspected a fixed 400-character window after a log
message; three comment lines pushed the call out of it, so the positive test failed and the negative test
passed VACUOUSLY over a window holding only comments. The resume block is now found by its code.
"""
import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "scripts" / "run_phase2_eval.py"


def _resume_block() -> list:
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    blocks = [n for n in ast.walk(tree) if isinstance(n, ast.If)
              and ast.unparse(n.test) == "_ensemble_path.exists()"]
    assert len(blocks) == 1, f"expected exactly one `if _ensemble_path.exists():` resume block, found {len(blocks)}"
    return blocks[0].body


def _calls(body) -> list:
    return [n for stmt in body for n in ast.walk(stmt) if isinstance(n, ast.Call)]


def test_resume_uses_classmethod_load():
    loads = [c for c in _calls(_resume_block()) if ast.unparse(c.func) == "VariantEnsemble.load"]
    assert len(loads) == 1, "resume path must call VariantEnsemble.load (reconstructs object with .evaluate)"
    call = loads[0]
    assert [ast.unparse(a) for a in call.args] == ["_ensemble_path"]
    keywords = {k.arg for k in call.keywords}
    assert {"consumer", "registry_path"} <= keywords, "the resume must go through the admission route"


def test_resume_does_not_raw_joblib_load_ensemble():
    calls = _calls(_resume_block())
    assert calls, "the resume block contains no calls at all -- this test would pass vacuously"
    raw = [ast.unparse(c) for c in calls if ast.unparse(c.func) in ("joblib.load", "_jl.load", "jl.load", "pickle.load")]
    assert not raw, f"resume must not raw-load the orchestrator dict (it has no .evaluate()): {raw}"
