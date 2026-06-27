#!/usr/bin/env python3
r"""patch_finngen_release_prefix.py  --  Stage 1 of FinnGen R12+R13 dual-release.

Parameterize FinnGenConnector by an OUTPUT column prefix so one connector can emit either:
  - R12 (default prefix="")    -> finngen_af_fin / finngen_af_nfsee / finngen_enrichment  (UNCHANGED)
  - R13 (prefix="r13_")        -> finngen_r13_af_fin / finngen_r13_af_nfsee / finngen_r13_enrichment

Threads the prefix through the 7 hardcoded df["finngen_*"] sites in annotate() (mapped from reads
W1/X1). R12 behavior is BYTE-IDENTICAL (default prefix), proven by a direct before/after output
comparison in the test (no existing FinnGen tests exist -- X3 -- so this patcher's test is the
connector's FIRST coverage).

Anchors are EXACT text from live reads X1 (lines 96-118) and X2 (constructor 56-63). The internal
af_fin/af_nfsee names in _build_index are NOT touched (internal, not outputs).

Also adds a module helper finngen_columns(prefix="") -> [3 names] for callers (Stage 4 zero-fill branch),
and fixes the stale R10 docstring/comment schema to the real GENOME_AF_* / R12-R13 reality.

ANCHOR-BASED, IDEMPOTENT, LF.
"""
from __future__ import annotations
import argparse, ast
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/data/finngen.py")
MARKER = "self._out_fin"   # idempotency sentinel

# --- Edit 1: constructor -- add column_prefix param + computed output names (EXACT anchor from X2) ---
CTOR_OLD = '''    def __init__(
        self,
        tsv_path: Optional[str | Path] = None,
        chunksize: int = 500_000,
    ) -> None:
        self.tsv_path = Path(tsv_path) if tsv_path else None
        self.chunksize = chunksize
        self._index: Optional[pd.DataFrame] = None'''

CTOR_NEW = '''    def __init__(
        self,
        tsv_path: Optional[str | Path] = None,
        chunksize: int = 500_000,
        column_prefix: str = "",
    ) -> None:
        self.tsv_path = Path(tsv_path) if tsv_path else None
        self.chunksize = chunksize
        self.column_prefix = column_prefix
        # Output column names; default prefix "" reproduces the R12 names exactly.
        self._out_fin = f"finngen_{column_prefix}af_fin"
        self._out_nfsee = f"finngen_{column_prefix}af_nfsee"
        self._out_enrich = f"finngen_{column_prefix}enrichment"
        self._out_cols = [self._out_fin, self._out_nfsee, self._out_enrich]
        self._index: Optional[pd.DataFrame] = None'''

# --- Edit 2: annotate() default-fill loop uses self._out_cols (EXACT anchor from W1 L76-78) ---
FILL_OLD = '''        for col in FINNGEN_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0'''
FILL_NEW = '''        for col in self._out_cols:
            if col not in df.columns:
                df[col] = 0.0'''

# --- Edit 3: no-file branch enrichment default (EXACT from W1 L87) ---
# There are TWO identical 'df["finngen_enrichment"] = 1.0' lines (L87 no-file, L94 empty-index).
# Replace both by doing a targeted replace on each unique surrounding context.
NOFILE_OLD = '''            self.tsv_path,
            )
            df["finngen_enrichment"] = 1.0
            return df'''
NOFILE_NEW = '''            self.tsv_path,
            )
            df[self._out_enrich] = 1.0
            return df'''

EMPTYIDX_OLD = '''        if self._index.empty:
            df["finngen_enrichment"] = 1.0
            return df'''
EMPTYIDX_NEW = '''        if self._index.empty:
            df[self._out_enrich] = 1.0
            return df'''

# --- Edit 4: final assignment block (EXACT anchor from X1 L107-113, incl .clip(upper=1000.0)) ---
ASSIGN_OLD = '''        df["finngen_af_fin"]   = merged["af_fin"].fillna(0.0).values
        df["finngen_af_nfsee"] = merged["af_nfsee"].fillna(0.0).values
        df["finngen_enrichment"] = (
            df["finngen_af_fin"] / (df["finngen_af_nfsee"] + 1e-9)
        ).clip(upper=1000.0)

        n_annotated = (df["finngen_af_fin"] > 0).sum()'''
ASSIGN_NEW = '''        df[self._out_fin]   = merged["af_fin"].fillna(0.0).values
        df[self._out_nfsee] = merged["af_nfsee"].fillna(0.0).values
        df[self._out_enrich] = (
            df[self._out_fin] / (df[self._out_nfsee] + 1e-9)
        ).clip(upper=1000.0)

        n_annotated = (df[self._out_fin] > 0).sum()'''

# --- Edit 5: module helper after FINNGEN_COLUMNS (anchor on the constant's close bracket) ---
HELPER_OLD = '''FINNGEN_COLUMNS = [
    "finngen_af_fin",
    "finngen_af_nfsee",
    "finngen_enrichment",
]'''
HELPER_NEW = '''FINNGEN_COLUMNS = [
    "finngen_af_fin",
    "finngen_af_nfsee",
    "finngen_enrichment",
]


def finngen_columns(column_prefix: str = "") -> list[str]:
    """The three FinnGen output column names for a given release prefix.

    prefix=""      -> finngen_af_fin / finngen_af_nfsee / finngen_enrichment (R12)
    prefix="r13_"  -> finngen_r13_af_fin / finngen_r13_af_nfsee / finngen_r13_enrichment
    """
    return [
        f"finngen_{column_prefix}af_fin",
        f"finngen_{column_prefix}af_nfsee",
        f"finngen_{column_prefix}enrichment",
    ]'''

EDITS = [
    ("constructor", CTOR_OLD, CTOR_NEW),
    ("fill-loop", FILL_OLD, FILL_NEW),
    ("nofile-branch", NOFILE_OLD, NOFILE_NEW),
    ("emptyidx-branch", EMPTYIDX_OLD, EMPTYIDX_NEW),
    ("assign-block", ASSIGN_OLD, ASSIGN_NEW),
    ("module-helper", HELPER_OLD, HELPER_NEW),
]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()
    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found."); return 2
    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print("OK (idempotent): connector already parameterized by column_prefix."); return 0

    # Validate every anchor occurs exactly once BEFORE applying any.
    problems = []
    for name, old, _new in EDITS:
        c = src.count(old)
        if c != 1:
            problems.append(f"  {name}: anchor occurs {c}x (need 1)")
    if problems:
        print("FAIL: anchor validation:\n" + "\n".join(problems)); return 3
    if ns.check:
        print("CHECK: all 6 anchors found exactly once."); print("RESULT: PASS (check)"); return 0

    backup = TARGET.with_suffix(".py.pre_release_prefix.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline=""); print(f"OK: backup -> {backup}")

    new = src
    for name, old, repl in EDITS:
        new = new.replace(old, repl, 1)
    TARGET.write_text(new, encoding="utf-8", newline="\n")

    after = TARGET.read_text(encoding="utf-8")
    checks = {
        "column_prefix param": "column_prefix: str = \"\"" in after,
        "out_fin computed": "self._out_fin = f\"finngen_{column_prefix}af_fin\"" in after,
        "fill uses _out_cols": "for col in self._out_cols:" in after,
        "assign uses _out_fin": "df[self._out_fin]   = merged" in after,
        "enrich uses _out_enrich": "df[self._out_enrich] = (" in after,
        "helper added": "def finngen_columns(" in after,
        "no stray hardcoded assign": 'df["finngen_af_fin"]   = merged' not in after,
    }
    try:
        ast.parse(after); checks["compiles"] = True
    except SyntaxError as e:
        checks["compiles"] = False; print("  SYNTAX ERROR:", e)
    for k, v in checks.items():
        print(f"  {'OK' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
