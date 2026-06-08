#!/usr/bin/env python3
"""
patch_fetchconfig_lazy_mkdir.py
===============================
Make FetchConfig construction side-effect-free and convert the cryptic
WinError-183 directory-shadow failure into a clear, actionable error.

Root cause it fixes
-------------------
FetchConfig.__post_init__ eagerly called

    self.cache_dir.mkdir(parents=True, exist_ok=True)

on a CWD-relative default (data/raw/cache). Merely *constructing* any
connector (including stub-mode and every unit test) therefore performed
filesystem I/O. When a stray file -- or a dangling symlink/junction -- named
'data' shadowed the data/ directory, pathlib's recursive mkdir hit
os.mkdir('data') -> FileExistsError [WinError 183], and because 'data' was
not a directory the exist_ok branch re-raised. Result: ~79 opaque test
failures at construction time, all masking one underlying cause.

What this does (idempotent; guarded; count==1 abort; .bak backup; AST verify)
-----------------------------------------------------------------------------
1. Inserts a module-level ensure_dir() helper (clear error on non-dir parent).
2. Removes the eager mkdir from FetchConfig.__post_init__. Caching is
   UNAFFECTED because _save_cache already creates the cache dir lazily right
   before the first write.
3. Routes _save_cache through ensure_dir() so any future shadow yields a clear
   message instead of WinError 183.

Target: src/genomic_variant_classifier/data/database_connectors.py
Run from the repo root. Safe to re-run (detects post-patch markers, no-ops).
"""
from __future__ import annotations

import ast
import datetime as _dt
import sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/data/database_connectors.py")

ANCHOR_LOGGER = "logger = logging.getLogger(__name__)\n"

ENSURE_DIR_HELPER = '''

def ensure_dir(directory: Path) -> Path:
    """Create *directory* (and parents) idempotently, with an actionable error
    if a path component exists as a non-directory.

    pathlib's ``mkdir(parents=True, exist_ok=True)`` raises a cryptic
    ``FileExistsError: [WinError 183]`` when an ancestor (e.g. a stray file
    named 'data' shadowing the data/ tree, or a dangling symlink/junction) is
    not a directory. We surface a clear message so the failure is never silent.
    """
    directory = Path(directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except FileExistsError as exc:  # an ancestor exists but is not a directory
        offender = next(
            (p for p in [directory, *directory.parents]
             if p.exists() and not p.is_dir()),
            directory,
        )
        raise NotADirectoryError(
            f"Cannot create directory {directory!s}: path component "
            f"{offender!s} exists but is not a directory. Remove or rename it "
            f"(and restore the real directory from git: "
            f"`git checkout HEAD -- data`), then retry."
        ) from exc
    return directory

'''

OLD_POSTINIT = (
    "    def __post_init__(self) -> None:\n"
    "        self.cache_dir = Path(self.cache_dir)\n"
    "        self.cache_dir.mkdir(parents=True, exist_ok=True)\n"
)
NEW_POSTINIT = (
    "    def __post_init__(self) -> None:\n"
    "        # Side-effect-free construction: do NOT mkdir here. The cache dir\n"
    "        # is created lazily in _save_cache right before the first write.\n"
    "        # Eager mkdir made every connector construction (incl. stub-mode\n"
    "        # and unit tests) perform CWD-relative filesystem I/O, which\n"
    "        # detonated the suite when a stray file named 'data' shadowed the\n"
    "        # data/ directory (WinError 183). See\n"
    "        # tests/unit/test_data_dir_not_shadowed.py.\n"
    "        self.cache_dir = Path(self.cache_dir)\n"
)

OLD_SAVE = (
    "    def _save_cache(self, key: str, df: pd.DataFrame) -> None:\n"
    "        self._cache_path(key).parent.mkdir(parents=True, exist_ok=True)\n"
    "        df.to_parquet(self._cache_path(key), index=False)\n"
)
NEW_SAVE = (
    "    def _save_cache(self, key: str, df: pd.DataFrame) -> None:\n"
    "        path = self._cache_path(key)\n"
    "        ensure_dir(path.parent)\n"
    "        df.to_parquet(path, index=False)\n"
)


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text and old not in text:
        print(f"  SKIP  {label}: already patched")
        return text
    n = text.count(old)
    if n == 0:
        print(f"  ABORT {label}: anchor not found (file drifted from expected).")
        sys.exit(2)
    if n > 1:
        print(f"  ABORT {label}: anchor found {n}x (expected 1). Manual review.")
        sys.exit(2)
    print(f"  OK    {label}: 1 replacement")
    return text.replace(old, new, 1)


def main() -> None:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found. Run from the repo root.")
        sys.exit(2)

    original = TARGET.read_text(encoding="utf-8")
    text = original

    # 1. ensure_dir helper -- insert once, right after the module logger line
    if "def ensure_dir(" in text:
        print("  SKIP  ensure_dir helper: already present")
    else:
        if text.count(ANCHOR_LOGGER) != 1:
            print("  ABORT ensure_dir: logger anchor not unique. Manual review.")
            sys.exit(2)
        text = text.replace(ANCHOR_LOGGER, ANCHOR_LOGGER + ENSURE_DIR_HELPER, 1)
        print("  OK    ensure_dir helper inserted")

    # 2. FetchConfig.__post_init__ -- drop the eager mkdir
    text = _replace_once(text, OLD_POSTINIT, NEW_POSTINIT, "FetchConfig.__post_init__")

    # 3. _save_cache -- route through ensure_dir for the clear error
    text = _replace_once(text, OLD_SAVE, NEW_SAVE, "_save_cache")

    if text == original:
        print("No changes needed (already fully patched).")
        return

    try:
        ast.parse(text)
    except SyntaxError as exc:
        print(f"ABORT: patched file fails AST parse: {exc}")
        sys.exit(3)

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_suffix(TARGET.suffix + f".bak_{stamp}")
    backup.write_text(original, encoding="utf-8")
    TARGET.write_text(text, encoding="utf-8")
    print(f"PATCHED {TARGET}  (backup {backup.name})")
    print("AST parse: OK")


if __name__ == "__main__":
    main()
