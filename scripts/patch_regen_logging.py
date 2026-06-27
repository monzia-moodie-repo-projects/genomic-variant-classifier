#!/usr/bin/env python3
r"""patch_regen_logging.py

Make DataPrepPipeline's per-step coverage logs VISIBLE when run via
regen_splits_local.py. The library emits coverage via logging.getLogger(__name__)
at INFO level (e.g. "Score annotation 11/17 (EVE): N variants covered",
"Score annotation 10b (protein coords): N", "EVEConnector: N/M covered"), but
regen_splits_local never calls logging.basicConfig, so Python's last-resort handler
emits only WARNING+ -- every INFO coverage line is silently discarded. This forced
us to reverse-engineer EVE/coords coverage from parquet columns all through the EVE
arc. (Connector WARNINGs like "DbSNPConnector: parquet_path not set" survived,
which is why those showed but coverage lines did not.)

Fix: add logging.basicConfig(level=logging.INFO, ...) as the FIRST statement in
main(), so the script (not the library) configures logging -- consistent with the
"keep logging out of library modules" convention and the "Issue L: module-level
basicConfig removed" note in real_data_prep. Verified: real_data_prep only does
getLogger(__name__) (no competing basicConfig), so no double-config, no force= needed.

Target (anchors verified against the live file):
  scripts/regen_splits_local.py
    - `import argparse` block (add `import logging` if absent)
    - `def main(argv=None) -> int:` ... `    args = parse_args(argv)` (inject after)

ANCHOR-BASED, IDEMPOTENT.

  python scripts/patch_regen_logging.py --check
  python scripts/patch_regen_logging.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

TARGET = Path("scripts/regen_splits_local.py")

# main() opens with these two lines (verified live: 108-109).
MAIN_ANCHOR = (
    "def main(argv=None) -> int:\n"
    "    args = parse_args(argv)\n"
)
MAIN_INSERT = (
    "def main(argv=None) -> int:\n"
    "    # Stream the library's per-step coverage logs (every \"Score annotation N/17\"\n"
    "    # line, ProteinCoord/EVE coverage, etc.) to stderr. DataPrepPipeline emits these\n"
    "    # via logging.getLogger(__name__) at INFO; without a basicConfig here Python's\n"
    "    # last-resort handler drops INFO (keeping only WARNING+), so prep coverage was\n"
    "    # invisible and had to be reverse-engineered from parquet columns. The script\n"
    "    # (not the library) owns logging config, per the 'logging out of library\n"
    "    # modules' convention. Honour an existing root config if one is already set.\n"
    "    if not logging.getLogger().handlers:\n"
    "        logging.basicConfig(\n"
    "            level=logging.INFO,\n"
    '            format="%(asctime)s %(levelname)s %(name)s: %(message)s",\n'
    "        )\n"
    "    args = parse_args(argv)\n"
)

# import logging -- add next to `import argparse` if the module lacks it.
IMPORT_ANCHOR = "import argparse\n"
IMPORT_INSERT = "import argparse\nimport logging\n"

MARKER = "Stream the library's per-step coverage logs"  # idempotency sentinel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()

    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found.")
        return 2
    src = TARGET.read_text(encoding="utf-8")

    if MARKER in src:
        print("OK (idempotent): regen_splits_local.py already patched.")
        return 0

    # Anchor checks
    n_main = src.count(MAIN_ANCHOR)
    if n_main != 1:
        print(f"FAIL: main() anchor occurs {n_main}x (need 1). Not patching blind.")
        return 3

    has_logging_import = ("import logging\n" in src)
    if not has_logging_import:
        n_imp = src.count(IMPORT_ANCHOR)
        if n_imp != 1:
            print(f"FAIL: 'import argparse' anchor occurs {n_imp}x (need 1) and 'import logging' "
                  "is absent; cannot place the import safely.")
            return 4

    if ns.check:
        print("CHECK: main() anchor found once; "
              + ("logging already imported." if has_logging_import
                 else "'import argparse' anchor found once (will add 'import logging')."))
        print("RESULT: PASS (check)")
        return 0

    patched = src
    if not has_logging_import:
        patched = patched.replace(IMPORT_ANCHOR, IMPORT_INSERT, 1)
    patched = patched.replace(MAIN_ANCHOR, MAIN_INSERT, 1)

    backup = TARGET.with_suffix(".py.pre_logging.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline="")
        print(f"OK: backup -> {backup}")
    TARGET.write_text(patched, encoding="utf-8", newline="")

    # Post-checks
    after = TARGET.read_text(encoding="utf-8")
    checks = [
        ("logging imported", "import logging\n" in after),
        ("basicConfig in main", "logging.basicConfig(" in after),
        ("guarded by handler check", "if not logging.getLogger().handlers:" in after),
        ("INFO level", "level=logging.INFO," in after),
        ("injected before parse_args", after.index("logging.basicConfig(") < after.index("args = parse_args(argv)")),
    ]
    ok = True
    for label, present in checks:
        print(f"  {'OK' if present else 'MISSING'}  {label}")
        ok &= present
    import py_compile
    try:
        py_compile.compile(str(TARGET), doraise=True)
        print("  OK  regen_splits_local.py compiles")
    except py_compile.PyCompileError as exc:
        print(f"  FAIL  does not compile: {exc}")
        ok = False
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
