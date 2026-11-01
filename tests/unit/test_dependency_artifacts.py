"""Every dependency file and every install path is declared, and each declaration is true of the repository. Author: Monzia Moodie"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from genomic_variant_classifier.deps.governance import ARTIFACTS, INSTALL_CONSUMERS, artifact_for
from genomic_variant_classifier.deps.model import ArtifactAuthority, ArtifactPurpose

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PATTERN = re.compile(r"^(requirements[^/]*\.(txt|in|lock)|(build-)?constraints[^/]*\.txt)$")


def _install_lines(path: Path) -> list:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        body = line.strip()
        if body.startswith("#") or body.startswith("echo"):
            continue
        if re.search(r"\bpip install\b", body):
            out.append(body)
    return out


def _consumer_files() -> set:
    candidates = list((ROOT / ".github" / "workflows").glob("*.yml")) + [ROOT / "Dockerfile"] + list((ROOT / "scripts").glob("*.sh"))
    return {str(p.relative_to(ROOT)).replace("\\", "/") for p in candidates if p.is_file() and _install_lines(p)}


def test_every_dependency_file_at_the_root_is_declared_exactly_once():
    on_disk = {p.name for p in ROOT.iterdir() if p.is_file() and PATTERN.match(p.name)}
    declared = [a.path for a in ARTIFACTS]
    assert len(declared) == len(set(declared))
    assert on_disk == set(declared), {"undeclared": sorted(on_disk - set(declared)), "missing": sorted(set(declared) - on_disk)}


def test_a_generated_artifact_names_declared_sources_and_an_installation_lock_is_hashed_throughout():
    for a in ARTIFACTS:
        assert all(artifact_for(s) is not None for s in a.generated_from), a.path
        if a.authority is ArtifactAuthority.GENERATED and a.purpose in (ArtifactPurpose.INSTALLATION, ArtifactPurpose.GENERATOR_TOOLCHAIN):
            text = (ROOT / a.path).read_text(encoding="utf-8")
            records = re.findall(r"(?m)^[A-Za-z0-9][^\n]*==[^\n]*$", text)
            assert records and all(r.rstrip().endswith("\\") for r in records), a.path       # every record continues into its hashes


def test_no_install_path_installs_from_a_reference_resolution():
    references = [a.path for a in ARTIFACTS if a.purpose is ArtifactPurpose.REFERENCE_RESOLUTION]
    for consumer in _consumer_files():
        for line in _install_lines(ROOT / consumer):
            assert not any(re.search(r"(?<![\w-])" + re.escape(ref) + r"\b", line) for ref in references), (consumer, line)


def test_every_install_path_has_a_declared_migration_status():
    declared = {c.path for c in INSTALL_CONSUMERS}
    found = _consumer_files()
    assert found == declared, {"undeclared": sorted(found - declared), "declared but installs nothing": sorted(declared - found)}
    assert all(c.status for c in INSTALL_CONSUMERS)
