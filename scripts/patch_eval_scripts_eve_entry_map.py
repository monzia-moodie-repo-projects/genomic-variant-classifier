#!/usr/bin/env python3
"""patch_eval_scripts_eve_entry_map.py

Wire the independent `--eve-entry-map` flag (UniProt index parquet with the
entry_name column) through BOTH eval entry points, and restore the full EVE
mirror in regen_splits_local (option A) so a prep-only split check exercises the
same EVE entry-name -> HGNC resolution the full run uses.

run_phase2_eval.py  (already has --eve-path + eve_path wiring):
  - add argparse `--eve-entry-map` (plain, default=None) after --eve-path
  - thread `eve_entry_map_path=Path(args.eve_entry_map) if args.eve_entry_map else None`
    into AnnotationConfig(...) right after the existing eve_path= line

regen_splits_local.py  (currently wires NEITHER --eve-path NOR eve_path):
  - add argparse `--eve-path` and `--eve-entry-map` (default=None) after --esm2-uniprot-index
  - thread both eve_path= and eve_entry_map_path= into its AnnotationConfig(...)
    after the finngen_path= line (restoring the true mirror)

INDEPENDENT in code: no reference to _esm2_index in the EVE threading; the two
flags only ever share a value via $UNIPROT_INDEX in the launch script.

ANCHOR-BASED, IDEMPOTENT, LF-SAFE. Run from repo root.
  python scripts/patch_eval_scripts_eve_entry_map.py            # apply
  python scripts/patch_eval_scripts_eve_entry_map.py --check    # report only
"""
from __future__ import annotations

import argparse
from pathlib import Path

RUN = Path("scripts/run_phase2_eval.py")
REGEN = Path("scripts/regen_splits_local.py")

# ---------------------------------------------------------------- run_phase2_eval
# argparse: add --eve-entry-map right AFTER the full --eve-path block closes.
# Adder is `p` (p = argparse.ArgumentParser(...)). Anchor on the exact closing tail
# of the existing --eve-path help block so we insert a clean, separate call.
RUN_ARG_ANCHOR = (
    '        "the HGVSp-parsed protein_change. When omitted, eve_score defaults to 0.5 "\n'
    '        "(silent stub). Covers missense substitutions only.",\n'
    "    )\n"
)
RUN_ARG_INSERT = (
    '        "the HGVSp-parsed protein_change. When omitted, eve_score defaults to 0.5 "\n'
    '        "(silent stub). Covers missense substitutions only.",\n'
    "    )\n"
    "    p.add_argument(\n"
    '        "--eve-entry-map",\n'
    "        default=None,\n"
    '        help="UniProt index parquet (entry_name column) resolving EVE per-protein "\n'
    '        "filenames (UniProt entry names, e.g. 1433G_HUMAN) to HGNC symbols (YWHAG). "\n'
    '        "Without it EVE keys on the entry-name prefix and misses an HGNC-keyed cohort "\n'
    '        "(eve_score stuck at 0.5). Independent of --esm2-uniprot-index; the launch "\n'
    '        "script points both at the same UniProt index.",\n'
    "    )\n"
)
# AnnotationConfig: thread eve_entry_map_path right after the eve_path= line.
RUN_CFG_ANCHOR = "            eve_path=Path(args.eve_path) if args.eve_path else None,\n"
RUN_CFG_INSERT = (
    "            eve_path=Path(args.eve_path) if args.eve_path else None,\n"
    "            eve_entry_map_path=Path(args.eve_entry_map) if args.eve_entry_map else None,\n"
)

# ---------------------------------------------------------------- regen_splits_local
# argparse: add --eve-path + --eve-entry-map after --esm2-uniprot-index.
REGEN_ARG_ANCHOR = '    p.add_argument("--esm2-uniprot-index", default=None)\n'
REGEN_ARG_INSERT = (
    '    p.add_argument("--esm2-uniprot-index", default=None)\n'
    '    p.add_argument("--eve-path", default=None)\n'
    '    p.add_argument("--eve-entry-map", default=None)\n'
)
# AnnotationConfig: add eve_path + eve_entry_map_path after the finngen_path= line.
REGEN_CFG_ANCHOR = "        finngen_path=Path(args.finngen_path) if args.finngen_path else None,\n    )\n"
REGEN_CFG_INSERT = (
    "        finngen_path=Path(args.finngen_path) if args.finngen_path else None,\n"
    "        eve_path=Path(args.eve_path) if args.eve_path else None,\n"
    "        eve_entry_map_path=Path(args.eve_entry_map) if args.eve_entry_map else None,\n"
    "    )\n"
)


def _patch_file(target: Path, edits: list[tuple[str, str, str]], marker: str,
                check: bool, suffix: str) -> int:
    if not target.exists():
        print(f"FAIL: {target} not found.")
        return 2
    src = target.read_text(encoding="utf-8")
    if marker in src:
        print(f"OK (idempotent): {target.name} already wired.")
        return 0
    problems = []
    for name, anc, _new in edits:
        n = src.count(anc)
        if n != 1:
            problems.append(f"{target.name}/{name}: anchor occurs {n}x (need 1)")
    if problems:
        print("FAIL: cannot safely anchor:")
        for p in problems:
            print(f"  - {p}")
        return 3
    patched = src
    for _name, anc, new in edits:
        patched = patched.replace(anc, new, 1)
    if check:
        print(f"CHECK: {target.name} all {len(edits)} anchors found.")
        return 0
    backup = target.with_suffix(target.suffix + suffix)
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline="\n")
        print(f"OK: backup -> {backup}")
    target.write_text(patched, encoding="utf-8", newline="\n")
    if b"\r\n" in target.read_bytes():
        print(f"FAIL: CRLF in {target.name}.")
        return 5
    try:
        compile(patched, str(target), "exec")
    except SyntaxError as e:
        print(f"FAIL: py-compile {target.name}: {e}")
        return 4
    print(f"OK: patched {target}  (py-compile OK)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()

    rc1 = _patch_file(
        RUN,
        [("argparse --eve-entry-map", RUN_ARG_ANCHOR, RUN_ARG_INSERT),
         ("AnnotationConfig thread", RUN_CFG_ANCHOR, RUN_CFG_INSERT)],
        marker="eve_entry_map_path=Path(args.eve_entry_map)",
        check=ns.check, suffix=".pre_eve_entry_map.bak",
    )
    rc2 = _patch_file(
        REGEN,
        [("argparse --eve-path/--eve-entry-map", REGEN_ARG_ANCHOR, REGEN_ARG_INSERT),
         ("AnnotationConfig thread", REGEN_CFG_ANCHOR, REGEN_CFG_INSERT)],
        marker="eve_entry_map_path=Path(args.eve_entry_map)",
        check=ns.check, suffix=".pre_eve_entry_map.bak",
    )
    rc = rc1 or rc2
    print("RESULT:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
