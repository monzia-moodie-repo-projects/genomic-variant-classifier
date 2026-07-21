#!/usr/bin/env python
"""verify_dtype.py -- verify the leak-remap dtype fix in split_protocol_v2.py.

HISTORY
-------
Written 2026-07-11 to confirm the int64 promotions added to
apply_train_only_leakage_remap. The cohort's n_pathogenic_in_gene and
gene_has_known_disease columns may be int32; the per-partition .iloc writes
assign int64 arrays, which makes pandas emit an incompatible-dtype FutureWarning
(a silent cast today, an error in a future pandas). The fix widens both columns
to int64 once, up front.

REVISED 2026-07-21. The original version checked the fix by SEARCHING THE SOURCE
TEXT, including:

    chk("partition loop still present (unchanged)", "for p in PARTITIONS:" in src
        and "out.iloc[ix, out.columns.get_loc(cfg.count_col)] = cnt" in src)

That check broke when split_protocol_v2 moved to a schema-driven partition set on
2026-07-21 and the loop became `for p in schema.names:`. It broke for a good
reason -- the module was correctly refactored -- which is exactly the problem
with the check: it verified that a line of code LOOKED a certain way, not that
the dtype promotion WORKED. A refactor that preserved the behaviour perfectly
would still have failed it, and a change that silently broke the promotion while
leaving the loop text intact would still have passed it.

Both failure directions are wrong, so the text check is replaced with a
BEHAVIOURAL one: construct a frame whose columns are genuinely int32, escalate
FutureWarning to an error, run the real split and the real remap, and assert the
result is int64 with no warning raised. That check survives any refactor of the
loop and fails if the promotion is ever removed -- which is the only thing worth
knowing.

The remaining text checks are retained. They are weak on their own but they pin
the specific lines, and a marker comment is legitimately a text artefact.

Read-only. Compiles, imports, and exercises the module against synthetic data.
ASCII-safe.
"""
from __future__ import annotations

import io
import importlib
import py_compile
import sys
import warnings
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass


def a(s):
    return s.encode("ascii", "replace").decode("ascii")


def main() -> int:
    print("=" * 78)
    print("LEAK-REMAP DTYPE-FIX VERIFICATION")
    print("=" * 78)
    sp = Path("src/genomic_variant_classifier/data/split_protocol_v2.py")
    if not sp.exists():
        print("ABORT: split_protocol_v2.py missing")
        return 2
    src = sp.read_text(encoding="utf-8", errors="replace")

    checks = []

    def chk(n, c, detail=""):
        checks.append((n, c))
        print(a(f"  {'ok  ' if c else 'FAIL'} {n}"))
        if detail:
            print(a(f"       {detail}"))

    # ---- text markers: weak, but they pin the specific lines ---------------
    chk("dtype-fix marker present", "Dtype-safe promotion (2026-07-11)" in src)
    chk("count_col int64 promotion",
        'out[cfg.count_col] = out[cfg.count_col].astype("int64")' in src)
    chk("derived_flag_col int64 promotion",
        'out[cfg.derived_flag_col] = out[cfg.derived_flag_col].astype("int64")' in src)
    chk("promotion guarded on column presence", "if cfg.count_col in out.columns:" in src)
    chk("per-partition write intact",
        "out.iloc[ix, out.columns.get_loc(cfg.count_col)] = cnt" in src)
    chk("out = df.copy() intact", "out = df.copy()" in src)

    print("-" * 78)
    try:
        py_compile.compile(str(sp), doraise=True)
        chk("split_protocol_v2 compiles", True)
    except Exception as e:
        chk("split_protocol_v2 compiles", False, str(e))

    sys.path.insert(0, "src")
    V2 = None
    try:
        V2 = importlib.import_module("genomic_variant_classifier.data.split_protocol_v2")
        chk("split_protocol_v2 imports", True)
    except Exception as e:
        chk("split_protocol_v2 imports", False, str(e))

    # ---- the behavioural check that replaced the source-text proxy --------
    print("-" * 78)
    print("  BEHAVIOURAL CHECK (replaced the 'for p in PARTITIONS:' text check)")
    if V2 is None:
        chk("int32 columns are promoted to int64 without FutureWarning", False,
            "module did not import; cannot exercise")
    else:
        try:
            import numpy as np
            import pandas as pd

            rng = np.random.default_rng(0)
            genes, labels = [], []
            for i in range(900):
                c = int(rng.integers(2, 12))
                genes += [f"GENE{i:05d}"] * c
                labels += [0, 1] + list(rng.integers(0, 2, size=c - 2))
            df = pd.DataFrame({"gene_symbol": genes, "label": labels})
            df["n_pathogenic_in_gene"] = np.int32(0)
            df["gene_has_known_disease"] = np.int32(0)

            pre_ok = (df["n_pathogenic_in_gene"].dtype == np.int32
                      and df["gene_has_known_disease"].dtype == np.int32)
            chk("fixture columns really are int32 before the remap", pre_ok,
                f"{df['n_pathogenic_in_gene'].dtype}, {df['gene_has_known_disease'].dtype}")

            cfg = V2.SplitProtocolV2Config()
            res = V2.split(df, cfg)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out = V2.apply_train_only_leakage_remap(df, res.indices, cfg)
            future = [w for w in caught if issubclass(w.category, FutureWarning)]
            chk("no FutureWarning raised during the remap", not future,
                "; ".join(str(w.message)[:90] for w in future))

            chk("count_col is int64 after the remap",
                out["n_pathogenic_in_gene"].dtype == np.int64,
                str(out["n_pathogenic_in_gene"].dtype))
            chk("derived_flag_col is int64 after the remap",
                out["gene_has_known_disease"].dtype == np.int64,
                str(out["gene_has_known_disease"].dtype))

            # Every partition in the SCHEMA must be remapped -- not a fixed list,
            # so this keeps working if the protocol gains a partition.
            names = cfg.schema.names if hasattr(cfg, "schema") else V2.PARTITIONS
            covered = all(
                (out.iloc[res.indices[p]]["n_pathogenic_in_gene"].notna()).all()
                for p in names)
            chk(f"every partition remapped ({len(names)}: {', '.join(names)})", covered)

            # The leakage property itself: a gene absent from train scores zero.
            train_genes = set(df.iloc[res.indices["train"]]["gene_symbol"])
            bad = 0
            for p in names:
                sub = out.iloc[res.indices[p]]
                unseen = sub[~sub["gene_symbol"].isin(train_genes)]
                if len(unseen):
                    bad += int((unseen["n_pathogenic_in_gene"] != 0).sum())
            chk("genes unseen in train score zero everywhere (incident 2026-06-13)",
                bad == 0, f"{bad} violation(s)")
        except Exception as e:
            chk("behavioural dtype check ran", False, f"{type(e).__name__}: {e}")

    npass = sum(1 for _, c in checks if c)
    print("-" * 78)
    print(a(f"dtype-fix verification: {npass}/{len(checks)} checks pass"))
    print("=" * 78)
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
