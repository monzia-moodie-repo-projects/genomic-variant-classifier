"""deps/environment_observation.py: the observer the pytest job and the owner's blocks run BEFORE the suite.

It had no test of its own (measured 2026-11-01: 0 of its 91 statements covered by the suite, although the workflow runs it on every
pull request), so a regression would surface only as a refused contract. Its two re-implementations of packaging's rules are
checked against packaging itself, and the program is run as the workflow runs it -- a child interpreter with -I -B -- including its
three refusals. Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import logging
import subprocess
import sys
from pathlib import Path

from packaging.markers import default_environment
from packaging.utils import canonicalize_name

from genomic_variant_classifier.deps import environment_observation as eo
from genomic_variant_classifier.deps.environment_contract import read_verified_json

logger = logging.getLogger(__name__)

PROGRAM = Path(eo.__file__)


def _run(*args, flags=("-I", "-B"), cwd=None):
    # Run as a FILE, not with -m: the program imports only the standard library, and -I ignores PYTHONPATH, so a file path works
    # whether the project is installed editable (the workflow) or only on PYTHONPATH.
    return subprocess.run([sys.executable, *flags, str(PROGRAM), *args], capture_output=True, text=True, timeout=900, cwd=cwd)


def test_the_marker_environment_is_packagings_default_environment():
    assert eo.marker_environment() == default_environment()


def test_the_name_normalisation_is_packagings():
    for name in ("Pillow", "pyBigWig", "typing_extensions", "zope.interface", "A--B__C..D", "ruamel.yaml.clib"):
        assert eo.canonical_name(name) == canonicalize_name(name)


def test_a_distribution_row_carries_the_installed_metadata():
    row = eo.distribution_row(md.distribution("pytest"))
    assert (row["canonical_name"], row["version"]) == ("pytest", md.version("pytest"))
    assert row["record_present"] and row["problems"] == [] and row["requires_dist"] == (md.requires("pytest") or [])


def test_the_program_writes_one_observation_whose_printed_digest_is_its_bytes(tmp_path):
    out = tmp_path / "observation.json"
    done = _run(str(out))
    assert done.returncode == 0, done.stdout + done.stderr
    digest = next(l.split()[1] for l in done.stdout.splitlines() if l.startswith("sha256 "))
    assert digest == hashlib.sha256(out.read_bytes()).hexdigest()
    doc = read_verified_json(out, digest)                  # the contract's own reader accepts it (no duplicate keys, finite)
    assert doc["schema"] == eo.SCHEMA and doc["marker_environment"] == default_environment()
    assert doc["interpreter"]["flags"] == {"isolated": 1, "dont_write_bytecode": True}
    names = {r["canonical_name"]: r["version"] for r in doc["distributions"]}
    assert names["pytest"] == md.version("pytest") and doc["summary"]["distributions"] == len(doc["distributions"])
    assert doc["pip_version"]["exit_code"] == 0 and doc["pip_freeze_all"]["exit_code"] == 0
    assert out.read_bytes().isascii() and b"\r\n" not in out.read_bytes()


def test_the_program_refuses_an_existing_output_and_leaves_it_alone(tmp_path):
    out = tmp_path / "observation.json"
    out.write_bytes(b"keep")
    done = _run(str(out))
    assert done.returncode == 2 and "already exists" in done.stderr and out.read_bytes() == b"keep"


def test_the_program_refuses_a_relative_path_and_a_run_without_isolation(tmp_path):
    relative = _run("observation.json", cwd=tmp_path)
    assert relative.returncode == 2 and "must be absolute" in relative.stderr and not (tmp_path / "observation.json").exists()
    plain = _run(str(tmp_path / "o.json"), flags=())
    assert plain.returncode == 2 and "-I -B" in plain.stderr and not (tmp_path / "o.json").exists()
