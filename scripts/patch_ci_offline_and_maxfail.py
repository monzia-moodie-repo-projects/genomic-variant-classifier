#!/usr/bin/env python3
"""patch_ci_offline_and_maxfail.py -- harden .github/workflows/ci.yml after the
ESM-2 HF-Hub CI flake (INCIDENT_2026-06-12).

Two edits to the 'Run unit tests' step:
  1. Add HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE env so CI never reaches HuggingFace
     Hub -- any model-needing test fails fast with OSError (caught -> skip) instead
     of flaking on 429 rate-limits. Safe across the whole suite (the local offline
     run is 898 passed / exit 0).
  2. Replace pytest -x with --maxfail=5 so a future break surfaces several failures
     instead of halting at (and hiding everything after) the first.

Idempotent, YAML-validated, newline-preserving, ASCII. Author: Monzia Moodie."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET = Path(".github/workflows/ci.yml")
MARKER = "HF_HUB_OFFLINE"

ANCHOR_STEP = (
    "      - name: Run unit tests\n"
    "        run: |\n"
    "          pytest tests/unit/ -x -q \\\n"
)
REPLACEMENT_STEP = (
    "      - name: Run unit tests\n"
    "        env:\n"
    "          # ESM-2 weights are never downloaded in CI: forcing offline makes any\n"
    "          # model-needing test fail fast with OSError (caught -> skip) instead of\n"
    "          # flaking on HuggingFace Hub 429 rate-limits. See INCIDENT_2026-06-12.\n"
    "          HF_HUB_OFFLINE: \"1\"\n"
    "          TRANSFORMERS_OFFLINE: \"1\"\n"
    "        run: |\n"
    "          pytest tests/unit/ --maxfail=5 -q \\\n"
)


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found (run from repo root).")
        return 2
    with TARGET.open("r", encoding="utf-8", newline="") as f:
        raw = f.read()
    if MARKER in raw:
        print("already patched (marker present); no change.")
        return 0
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    if text.count(ANCHOR_STEP) != 1:
        print(f"ERROR: expected exactly 1 occurrence of the step anchor, found {text.count(ANCHOR_STEP)}.")
        print("       (the 'Run unit tests' step / '-x' flag may have changed) -- not patching.")
        return 2
    text = text.replace(ANCHOR_STEP, REPLACEMENT_STEP, 1)

    bak = TARGET.with_suffix(TARGET.suffix + ".bak")
    shutil.copy2(TARGET, bak)
    with TARGET.open("w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\n", nl))

    try:
        import yaml
    except ImportError:
        print(f"patched {TARGET} (backup at {bak.name}); pyyaml absent -- skipped re-parse.")
        print("       validate with: python -c \"import yaml,io; yaml.safe_load(io.open(r'.github/workflows/ci.yml',encoding='utf-8'))\"")
        return 0
    try:
        yaml.safe_load(TARGET.read_text(encoding="utf-8"))
    except Exception as exc:
        shutil.copy2(bak, TARGET)
        print(f"ERROR: YAML no longer parses, reverted: {exc}")
        return 2
    print(f"patched {TARGET} (backup at {bak.name}); YAML OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
