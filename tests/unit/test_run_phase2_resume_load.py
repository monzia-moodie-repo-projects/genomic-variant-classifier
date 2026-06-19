"""Guard the resume path in scripts/run_phase2_eval.py: it MUST reconstruct the ensemble via
VariantEnsemble.load() (format_version=2 dict -> object), never a raw joblib.load() that returns
the orchestrator dict and crashes the subsequent .evaluate(). Author: Monzia Moodie."""
import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "scripts" / "run_phase2_eval.py"


def _resume_block() -> str:
    text = _SRC.read_text()
    i = text.index("Resuming: loading existing ensemble")
    return text[i:i + 400]


def test_resume_uses_classmethod_load():
    block = _resume_block()
    assert "VariantEnsemble.load(_ensemble_path)" in block, \
        "resume path must call VariantEnsemble.load (reconstructs object with .evaluate)"


def test_resume_does_not_raw_joblib_load_ensemble():
    block = _resume_block()
    # neither joblib.load(_ensemble_path) nor an aliased _jl.load(_ensemble_path)
    assert not re.search(r"\b(joblib|_jl)\.load\(_ensemble_path\)", block), \
        "resume must not raw-load the orchestrator dict (it has no .evaluate())"
