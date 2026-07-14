"""requirements-api.txt and requirements.txt MUST agree on every package they share.

Added 2026-07-13 (roadmap 6.19).

WHY
---
Continuous Integration installs BOTH files into ONE environment:

    pip install -r requirements-api.lock
    pip install -r requirements.txt
    pip install -r requirements-dev.txt

Each file is valid on its own. `lockfile-check` proved the lock INSTALLS. The test suite
proved requirements.txt works. **Nothing ever checked the COMBINATION** -- and the combination
is what actually gets built.

pip DID check it. On every single one of 508 Continuous Integration runs it printed:

    ERROR: pip's dependency resolver does not currently take into account all the packages
    that are installed. This behaviour is the source of the following dependency conflicts.
    prometheus-fastapi-instrumentator 7.1.0 requires starlette<1.0.0,>=0.30.0,
        but you have starlette 1.0.0 which is incompatible.

...and then exited 0. Nothing ran `pip check`. The machine said it plainly, every time, and
nothing failed. That is the sixth instance of this exact shape in a single day.

WHAT THE AUDIT FOUND (2026-07-13)
---------------------------------
Thirteen packages appear in both files. NINE disagreed:

    package        requirements-api.txt     requirements.txt
    catboost       >=1.2,<2.0               ==1.2.10
    joblib         >=1.3,<2.0               ==1.5.3
    lightgbm       >=4.3,<5.0               ==4.6.0
    numpy          >=2.0,<3.0               ==2.4.4
    pandas         >=2.2,<3.0               ==2.3.3
    pyarrow        >=15.0,<20.0             ==23.0.1    <-- the RANGE EXCLUDES the PIN
    requests       >=2.31,<3.0              ==2.33.0
    scikit-learn   >=1.4,<2.0               ==1.8.0
    xgboost        >=2.0,<3.0               ==3.2.0     <-- the RANGE EXCLUDES the PIN

`pyarrow` and `xgboost` were outright ResolutionImpossible. The other seven admitted the pin
but let pip-compile resolve to the LATEST in range, which then collided.

Also found: `fastapi>=0.111,<0.120` in the API file resolved to FastAPI **0.119.1**, while
requirements.txt pins **0.135.2**. So the Docker API image was building on a web framework
FIFTEEN MINOR VERSIONS BEHIND the one src/genomic_variant_classifier/api/ is developed and
tested against -- and FastAPI 0.119.1's `starlette<0.49` constraint is what created the
508-run conflict in the first place. The instrumentator was the symptom, not the cause.

WHY THIS TEST EXISTS RATHER THAN A COMMENT
------------------------------------------
Pinning the nine to match duplicates NINE NUMBERS across two files. That is root pattern (a)
-- "a number written down once and never re-derived becomes a lie on a schedule" -- and it is
the single most repeated defect in this project's history (KNOWN_ZERO_DEFAULT 27 vs 25; a "65
features" comment against a 97-feature contract; a G1 pytest floor 330 tests below reality,
which then rotted five more times in two days beneath an all-capitals comment ordering the
next person to raise it).

    A COMMENT DOES NOT ENFORCE ITSELF. A number that is duplicated must be RE-DERIVED by a
    gate on every run, or it WILL drift.

So the duplication is permitted, and guarded. This test re-derives the agreement every time
the suite runs. The permanent fix -- ONE source of truth for shared pins -- is roadmap 6.18.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_API = _REPO / "requirements-api.txt"
_MAIN = _REPO / "requirements.txt"


def _parse(path: Path) -> dict[str, str]:
    """package -> version specifier. Comments and blank lines ignored. Extras stripped."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#")[0].strip()
        if not stripped or stripped.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(?:\[[^\]]*\])?\s*(.*)$", stripped)
        if not m:
            continue
        name = m.group(1).lower().replace("_", "-")
        out[name] = (m.group(2) or "").strip()
    return out


def test_both_requirements_files_exist():
    assert _API.is_file(), f"{_API} is missing"
    assert _MAIN.is_file(), f"{_MAIN} is missing"


def test_every_shared_package_is_pinned_to_the_SAME_exact_version():
    """THE GATE. Any divergence is a ResolutionImpossible waiting to happen.

    Continuous Integration installs BOTH files into ONE environment. A package that appears in
    both and resolves differently is a HARD conflict -- and pip reports it in an ERROR line and
    then exits 0, so it survives indefinitely unless something FAILS on it.
    """
    api = _parse(_API)
    main = _parse(_MAIN)

    shared = sorted(set(api) & set(main))
    assert shared, "no shared packages found -- the parser is broken, not the requirements"

    mismatches: list[str] = []
    for pkg in shared:
        api_spec, main_spec = api[pkg], main[pkg]
        if api_spec != main_spec:
            mismatches.append(f"    {pkg:<16} api={api_spec!r:<18} main={main_spec!r}")

    assert not mismatches, (
        "requirements-api.txt and requirements.txt DISAGREE on packages they SHARE.\n"
        "Continuous Integration installs BOTH into ONE environment, so every one of these is a\n"
        "dependency conflict -- and pip reports conflicts in an ERROR line and then EXITS 0.\n"
        "That is how a starlette conflict survived 508 Continuous Integration runs.\n\n"
        + "\n".join(mismatches)
        + "\n\nPin them to the SAME EXACT version in both files, then regenerate the lock:\n"
        "    pip-compile --strip-extras requirements-api.txt -o requirements-api.lock\n\n"
        "DO NOT 'fix' this by loosening one side into a range. A range guarantees the two\n"
        "files resolve differently and collide. See roadmap 6.19."
    )


def test_no_shared_package_uses_a_RANGE_in_the_api_file():
    """A range in requirements-api.txt for a SHARED package is a bug, by construction.

    requirements.txt pins exactly. If the API file ranges, pip-compile resolves to the LATEST
    in that range -- which will not be the pinned version, and the two collide when installed
    together. Ranges are fine ONLY for packages the API file alone declares (gunicorn,
    slowapi, python-json-logger, prometheus-fastapi-instrumentator).
    """
    api = _parse(_API)
    main = _parse(_MAIN)

    ranged = [
        f"    {pkg:<16} {api[pkg]}"
        for pkg in sorted(set(api) & set(main))
        if not api[pkg].startswith("==")
    ]
    assert not ranged, (
        "these packages are SHARED with requirements.txt but RANGED in requirements-api.txt:\n"
        + "\n".join(ranged)
        + "\n\nA range resolves to the LATEST version in it, which will not equal the exact pin\n"
        "in requirements.txt. Installing both -> ResolutionImpossible. This is exactly how\n"
        "`pyarrow>=15.0,<20.0` (resolving to 19.0.1) collided with `pyarrow==23.0.1`, and how\n"
        "`xgboost>=2.0,<3.0` collided with `xgboost==3.2.0`. Pin them exactly."
    )


@pytest.mark.parametrize(
    "pkg, why",
    [
        ("torch", "the neural base models -- the API image does inference from a saved artifact"),
        ("torch-geometric", "the graph-neural-network branch"),
        ("pyspark", "the Spark extract-transform-load path"),
        ("transformers", "ESM-2 / the Nucleotide Transformer"),
        ("nannyml", "Confidence-Based Performance Estimation -- the ISOLATED drift environment"),
        ("evidently", "the drift report -- the ISOLATED drift environment"),
    ],
)
def test_the_api_image_stays_MINIMAL(pkg, why):
    """The API file subsets by WHICH PACKAGES, not by WHICH VERSIONS.

    requirements-api.txt's whole purpose is a lean inference image ("No PySpark, no
    TensorFlow, no training-only libs"). Pinning the shared packages to the same versions as
    the training stack does NOT make the image fat -- it makes it CONSISTENT. These must stay
    out.
    """
    api = _parse(_API)
    assert pkg not in api, (
        f"{pkg} has appeared in requirements-api.txt. It is training-only ({why}). The API "
        f"image is an INFERENCE image and must stay minimal; adding this would bloat it and, "
        f"in the case of nannyml/evidently, drag in constraints (lightgbm<4.6, plotly<6) that "
        f"are incompatible with the training stack."
    )
