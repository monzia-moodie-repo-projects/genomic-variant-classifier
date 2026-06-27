"""Unit battery for scripts/launch_run17_baseline.sh -- asserts every Run-17 activation flag and
guarded block is present, no model-diminishing flag leaked, and the script is syntactically valid.
Author: Monzia Moodie."""
import shutil
import subprocess
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
_LAUNCHER = _SCRIPTS / "launch_run17_baseline.sh"


@pytest.fixture(scope="module")
def text():
    assert _LAUNCHER.exists(), f"missing {_LAUNCHER}"
    return _LAUNCHER.read_text()


@pytest.mark.parametrize("needle", [
    "--rnaseq-path", "--kg ", "--hetero-gnn", "--kg-edges reactome:",
])
def test_run17_activation_flags_present(text, needle):
    assert needle in text, f"Run-17 activation flag absent: {needle!r}"


@pytest.mark.parametrize("needle", [
    "--clinvar ", "--seq-windows ", "--gnomad ", "--spliceai ", "--alphamissense ",
    "--gnomad-constraint ", "--dbnsfp-path ", "--gtex-path ", "--reactome-path ",
    "--string-db auto", "--min-review-tier 3", "--n-folds 5", "--skip-svm",
    "--unseen-gene-holdout", "--output ",
])
def test_required_flags_present(text, needle):
    assert needle in text, f"required flag absent: {needle!r}"


@pytest.mark.parametrize("needle", ["--skip-nn", "--skip-cnn", "--skip-kan"])
def test_no_model_diminishing_flags(text, needle):
    arg_lines = [ln for ln in text.splitlines()
                 if ("ARGS=" in ln or "run_phase2_eval.py" in ln) and not ln.strip().startswith("#")]
    assert all(needle not in ln for ln in arg_lines), f"diminishing flag wired: {needle!r}"


def test_kg_and_rnaseq_are_required_inputs(text):
    assert 'KG_PARQUET="$DATA/external/1kgp/kg_grch38_af.parquet"' in text
    assert 'RNASEQ_PARQUET="$DATA/external/rnaseq_gene_expression.parquet"' in text
    assert '"$KG_PARQUET" \\' in text, "kg not in the hard-fail required loop"
    assert '"$RNASEQ_PARQUET" \\' in text, "rnaseq not in the hard-fail required loop"
    assert 'ABORT (exit 2): missing required inputs' in text


def test_esm2_uniprot_index_wired(text):
    # ESM-2 is now deliberately wired (HGVSp parser delivered -> ESM-2 carries real signal).
    # launch_run17_baseline.sh appends --esm2-uniprot-index to ARGS; assert its PRESENCE.
    arg_lines = [ln for ln in text.splitlines()
                 if "ARGS=" in ln and not ln.strip().startswith("#")]
    assert any("--esm2-uniprot-index" in ln for ln in arg_lines), \
        "--esm2-uniprot-index expected in ARGS (ESM-2 wired this session)"


def test_outdir_pinned(text):
    assert "outputs/run17_baseline/full" in text


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")
def test_bash_syntax_valid():
    # Syntax-check the CONTENT via `bash -n -c`, never a path. Git-Bash/MSYS on Windows
    # resolves neither backslash paths (C:\\ -> escape mangling) nor C:/ drive paths
    # (it expects /c/... mount form) as argv; passing the script text sidesteps all of it.
    content = _LAUNCHER.read_text()
    r = subprocess.run(["bash", "-n", "-c", content], capture_output=True, text=True)
    assert r.returncode == 0, f"bash -n failed:\n{r.stderr}"
