#!/usr/bin/env python3
"""
patch_suppress_downcast_version_aware.py -- make _suppress_fillna_downcast a no-op on pandas >= 3.0.

WHY: the decorator wraps a builder in pd.option_context("future.no_silent_downcasting", True).
On pandas 2.x that opts into the future (no-silent-downcasting) behavior and silences the
FutureWarning. On pandas 3.0 that behavior is already the DEFAULT, and the option itself is
deprecated toward pandas 4.0 -- so entering the context now emits a Pandas4Warning (observed:
3 warnings from contextlib in the real_pandas304 bundle). The decorator's own docstring already
says "No-op on pandas >= 3"; this makes the code honor that.

The fix: compute the pandas major version once at import; on >= 3.0 the wrapper calls the function
directly (no option_context); on < 3.0 it keeps the existing behavior. Value-identical on both
(the equivalence harness re-run on 3.0.4 must still show feature_hash 49e98393... and warnings empty).

Anchored on the exact wrapper body (not line numbers). Idempotent (sentinel). .bak backup. Aborts
if the anchor is absent or non-unique.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/data/real_data_prep.py")

OLD = '''    @functools.wraps(_fn)
    def _wrapper(*args, **kwargs):
        with pd.option_context("future.no_silent_downcasting", True):
            return _fn(*args, **kwargs)

    return _wrapper'''

NEW = '''    @functools.wraps(_fn)
    def _wrapper(*args, **kwargs):
        # pandas >= 3.0: no-silent-downcasting is the default AND the option is deprecated
        # toward pandas 4.0 (entering the context emits a Pandas4Warning). Call directly.
        if _PANDAS_MAJOR >= 3:
            return _fn(*args, **kwargs)
        with pd.option_context("future.no_silent_downcasting", True):
            return _fn(*args, **kwargs)

    return _wrapper'''

# The module-level constant we insert just above the decorator def.
ANCHOR_DEF = "def _suppress_fillna_downcast(_fn):"
CONST_BLOCK = (
    "# pandas major version, used to no-op _suppress_fillna_downcast on pandas >= 3.0\n"
    "# (where no-silent-downcasting is the default and the option_context is deprecated).\n"
    "_PANDAS_MAJOR = int(pd.__version__.split(\".\")[0])\n\n\n"
)

SENTINEL = "_PANDAS_MAJOR >= 3:"


def main() -> int:
    if not TARGET.exists():
        print(f"[FAIL] {TARGET} not found (run from repo root)")
        return 2
    text = TARGET.read_text(encoding="utf-8")

    if SENTINEL in text:
        print("[idempotent] version-aware guard already present; no change.")
        return 0

    # 1) Insert the _PANDAS_MAJOR constant just above the decorator def (once).
    if text.count(ANCHOR_DEF) != 1:
        print(f"[FAIL] expected exactly 1 '{ANCHOR_DEF}', found {text.count(ANCHOR_DEF)}.")
        return 3
    if "_PANDAS_MAJOR" in text:
        print("[FAIL] _PANDAS_MAJOR already defined unexpectedly; aborting to avoid dup.")
        return 7

    # 2) Replace the wrapper body (once).
    if text.count(OLD) != 1:
        print(f"[FAIL] wrapper-body anchor found {text.count(OLD)} times -- expected 1.")
        return 4

    bak = TARGET.with_suffix(TARGET.suffix + ".bak")
    shutil.copy2(TARGET, bak)

    new_text = text.replace(ANCHOR_DEF, CONST_BLOCK + ANCHOR_DEF, 1).replace(OLD, NEW, 1)
    TARGET.write_text(new_text, encoding="utf-8")

    after = TARGET.read_text(encoding="utf-8")
    import ast
    try:
        ast.parse(after)
    except SyntaxError as e:
        shutil.copy2(bak, TARGET)
        print(f"[FAIL] post-patch syntax error ({e}); restored from .bak.")
        return 5

    ok = (SENTINEL in after) and ("_PANDAS_MAJOR = int(pd.__version__" in after)
    print(f"[ok] version-aware guard patched + compiles; sentinel present: {ok}")
    print(f"[ok] backup at {bak} (remove before committing)")
    return 0 if ok else 6


if __name__ == "__main__":
    sys.exit(main())
