#!/usr/bin/env python3
"""patch_eve_entry_name_resolution.py

Fix the EVE silent-zero on HGNC-keyed cohorts. EVE per-protein files are named by
UniProt ENTRY NAME (1433G_HUMAN.csv), so the connector keyed its lookup on the
entry-name prefix ("1433G"); the cohort keys on the HGNC symbol ("YWHAG"), so the
join matched ~nothing and eve_score silently stayed 0.5 (empirically 0/2 -> fixed 2/2).

This patch (explicit-thread design, option b):
  1. Adds module constants + three pure resolver functions to eve.py:
       _EVE_MIN_RESOLVED_FRACTION, _eve_stem_to_entry_name, load_eve_entry_map,
       resolve_eve_gene
  2. Adds an `entry_map_path` parameter to EVEConnector.__init__ (loads + caches the
     {ENTRY_NAME: HGNC} map from the UniProt index parquet's entry_name column).
  3. Rewrites the `else` (filename) branch of _parse_single_csv to resolve the stem
     to HGNC via the map (was: csv_file.stem.split("_")[0]).
  4. The _parse_single_csv signature gains an injected `entry_map` arg, and
     _parse_csv_directory builds the map once, passes it in, tracks resolved vs
     fallback counts, and FAILS LOUD (logger.warning) if the resolved fraction is
     below threshold or the map is empty -- so a stale/missing index is never silent.

ANCHOR-BASED, IDEMPOTENT, LF-SAFE (CRLF guard). Run from repo root.

  python scripts/patch_eve_entry_name_resolution.py            # apply
  python scripts/patch_eve_entry_name_resolution.py --check    # report only
"""
from __future__ import annotations

import argparse
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/data/eve.py")
MARKER = "resolve_eve_gene"

# ---- 1. module-level constants + resolver functions, inserted after DEFAULT_SCORE ----
CONST_ANCHOR = "DEFAULT_SCORE = 0.5\n"
CONST_INSERT = '''DEFAULT_SCORE = 0.5

# EVE per-protein files are named by UniProt ENTRY NAME (e.g. "1433G_HUMAN"), not by
# HGNC symbol. The cohort keys on HGNC ("YWHAG"), so the filename stem must be resolved
# entry-name -> HGNC before keying the lookup, else the join silently misses (eve_score
# stays 0.5 everywhere). The map comes from the UniProt index parquet's entry_name column
# (built by scripts/build_uniprot_index.py). Below this fraction of files resolving via
# the map, the connector logs a LOUD warning rather than silently keying on entry names.
_EVE_MIN_RESOLVED_FRACTION = 0.80


def _eve_stem_to_entry_name(stem: object) -> str:
    """EVE filename stem -> UniProt entry name.
    "1433G_HUMAN" -> "1433G_HUMAN"; "TP53_HUMAN_singles_scores" -> "TP53_HUMAN".
    Returns "" for null-ish input."""
    s = "" if stem is None else str(stem).strip().upper()
    if not s:
        return ""
    i = s.find("_HUMAN")
    return s[: i + len("_HUMAN")] if i != -1 else s


def load_eve_entry_map(parquet_path: object) -> dict[str, str]:
    """Build {ENTRY_NAME_UPPER: HGNC_UPPER} from the UniProt index parquet.
    Returns {} if the path is missing or the parquet lacks an entry_name column
    (e.g. an index built before entry_name was added); the caller logs that loudly."""
    if parquet_path is None:
        return {}
    p = Path(parquet_path)
    if not p.exists():
        return {}
    try:
        df = pd.read_parquet(p)
    except Exception as exc:  # pragma: no cover - I/O guard
        logger.warning("EVEConnector: failed to read entry-name map %s: %s", p, exc)
        return {}
    if "entry_name" not in df.columns or "gene_symbol" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for en, g in zip(df["entry_name"].astype(str), df["gene_symbol"].astype(str)):
        en = en.strip().upper()
        g = g.strip().upper()
        if en and g and en not in out:
            out[en] = g
    return out


def resolve_eve_gene(stem: object, entry_map: dict[str, str]) -> tuple[str, bool]:
    """Resolve an EVE filename stem to (HGNC_symbol, resolved_via_map).
    Falls back to the legacy prefix-before-"_" (so HGNC-named files still work),
    with resolved_via_map=False so callers can count + fail loud on mass misses."""
    entry = _eve_stem_to_entry_name(stem)
    hgnc = entry_map.get(entry) if entry else None
    if hgnc:
        return normalize_gene_symbol(hgnc), True
    return normalize_gene_symbol(str(stem).split("_")[0]), False

'''

# ---- 2. __init__ signature + body ----
INIT_SIG_ANCHOR = (
    "    def __init__(\n"
    "        self,\n"
    "        eve_path: Optional[str | Path] = None,\n"
    "        config: Optional[FetchConfig] = None,\n"
    "    ) -> None:\n"
    "        super().__init__(config)\n"
    "        self.eve_path: Optional[Path] = (\n"
    "            Path(eve_path) if eve_path is not None else None\n"
    "        )\n"
)
INIT_SIG_INSERT = (
    "    def __init__(\n"
    "        self,\n"
    "        eve_path: Optional[str | Path] = None,\n"
    "        config: Optional[FetchConfig] = None,\n"
    "        entry_map_path: Optional[str | Path] = None,\n"
    "    ) -> None:\n"
    "        super().__init__(config)\n"
    "        self.eve_path: Optional[Path] = (\n"
    "            Path(eve_path) if eve_path is not None else None\n"
    "        )\n"
    "        # UniProt index parquet (entry_name column) for entry-name -> HGNC resolution.\n"
    "        self.entry_map_path: Optional[Path] = (\n"
    "            Path(entry_map_path) if entry_map_path is not None else None\n"
    "        )\n"
    "        self._last_csv_resolved: bool = False\n"
)

# ---- 3. _parse_single_csv: signature gains entry_map, else-branch uses the resolver ----
PARSE_SINGLE_SIG_ANCHOR = "    def _parse_single_csv(self, csv_file: Path) -> pd.DataFrame:\n"
PARSE_SINGLE_SIG_INSERT = (
    "    def _parse_single_csv(self, csv_file: Path, entry_map: Optional[dict] = None) -> pd.DataFrame:\n"
)

GENEKEY_ANCHOR = (
    "        # Extract gene symbol from protein name when present, otherwise from filename.\n"
    '        if "mutations_protein_name" in raw.columns:\n'
    '            raw["gene_symbol"] = raw["mutations_protein_name"].astype(str).str.split("_").str[0]\n'
    "        else:\n"
    '            raw["gene_symbol"] = csv_file.stem.split("_")[0]\n'
)
GENEKEY_INSERT = (
    "        # Extract gene symbol from protein name when present, otherwise resolve the\n"
    "        # filename entry-name (e.g. 1433G_HUMAN) to its HGNC symbol via the UniProt map.\n"
    '        if "mutations_protein_name" in raw.columns:\n'
    '            raw["gene_symbol"] = raw["mutations_protein_name"].astype(str).str.split("_").str[0]\n'
    "            self._last_csv_resolved = True\n"
    "        else:\n"
    "            _hgnc, _resolved = resolve_eve_gene(csv_file.stem, entry_map or {})\n"
    '            raw["gene_symbol"] = _hgnc\n'
    "            self._last_csv_resolved = _resolved\n"
)


# ---- 4. _parse_csv_directory: build map once, pass to each parse, fail-loud guard ----
PARSE_DIR_ANCHOR = (
    "        parts = []\n"
    "        for csv_file in csv_files:\n"
    "            try:\n"
    "                part = self._parse_single_csv(csv_file)\n"
    "                if not part.empty:\n"
    "                    parts.append(part)\n"
    "            except Exception as exc:\n"
    "                logger.warning(\n"
    '                    "EVEConnector: failed to parse %s: %s", csv_file.name, exc\n'
    "                )\n"
)
PARSE_DIR_INSERT = (
    "        # Build the entry-name -> HGNC map once; pass it to every per-file parse so\n"
    "        # filenames (UniProt entry names) resolve to the cohort's HGNC symbols.\n"
    "        entry_map = load_eve_entry_map(self.entry_map_path)\n"
    "        n_resolved = 0\n"
    "        n_fallback = 0\n"
    "        unresolved_sample: list[str] = []\n"
    "        parts = []\n"
    "        for csv_file in csv_files:\n"
    "            try:\n"
    "                self._last_csv_resolved = False\n"
    "                part = self._parse_single_csv(csv_file, entry_map)\n"
    "                if self._last_csv_resolved:\n"
    "                    n_resolved += 1\n"
    "                else:\n"
    "                    n_fallback += 1\n"
    "                    if len(unresolved_sample) < 5:\n"
    "                        unresolved_sample.append(csv_file.stem)\n"
    "                if not part.empty:\n"
    "                    parts.append(part)\n"
    "            except Exception as exc:\n"
    "                logger.warning(\n"
    '                    "EVEConnector: failed to parse %s: %s", csv_file.name, exc\n'
    "                )\n"
    "        # Fail loud if too few files resolved entry-name -> HGNC (stale/missing index).\n"
    "        _total = n_resolved + n_fallback\n"
    "        _frac = (n_resolved / _total) if _total else 0.0\n"
    "        if not entry_map:\n"
    "            logger.warning(\n"
    '                "EVEConnector: entry-name map empty (entry_map_path=%s); keying EVE by "\n'
    '                "filename prefix -> near-zero HGNC coverage. Rebuild the UniProt index "\n'
    '                "with the entry_name column (scripts/build_uniprot_index.py).",\n'
    "                self.entry_map_path,\n"
    "            )\n"
    "        elif _frac < _EVE_MIN_RESOLVED_FRACTION:\n"
    "            logger.warning(\n"
    '                "EVEConnector: only %d/%d (%.1f%%) EVE files resolved entry-name -> HGNC "\n'
    '                "(min %.0f%%). Sample unresolved: %s. Check the UniProt index entry_name column.",\n'
    "                n_resolved, _total, 100 * _frac, 100 * _EVE_MIN_RESOLVED_FRACTION, unresolved_sample,\n"
    "            )\n"
    "        else:\n"
    "            logger.info(\n"
    '                "EVEConnector: resolved %d/%d (%.1f%%) EVE files entry-name -> HGNC.",\n'
    "                n_resolved, _total, 100 * _frac,\n"
    "            )\n"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()

    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found (run from repo root).")
        return 2

    src = TARGET.read_text(encoding="utf-8")

    if MARKER in src:
        print("OK (idempotent): entry-name resolver already present in eve.py.")
        return 0

    anchors = [
        ("module constants", CONST_ANCHOR),
        ("__init__ signature/body", INIT_SIG_ANCHOR),
        ("_parse_single_csv signature", PARSE_SINGLE_SIG_ANCHOR),
        ("gene-key branch", GENEKEY_ANCHOR),
        ("_parse_csv_directory loop", PARSE_DIR_ANCHOR),
    ]
    problems = []
    for name, anc in anchors:
        n = src.count(anc)
        if n != 1:
            problems.append(f"{name}: anchor occurs {n}x (need exactly 1)")
    if problems:
        print("FAIL: cannot safely anchor:")
        for p in problems:
            print(f"  - {p}")
        return 3

    patched = src
    patched = patched.replace(CONST_ANCHOR, CONST_INSERT, 1)
    patched = patched.replace(INIT_SIG_ANCHOR, INIT_SIG_INSERT, 1)
    patched = patched.replace(PARSE_SINGLE_SIG_ANCHOR, PARSE_SINGLE_SIG_INSERT, 1)
    patched = patched.replace(GENEKEY_ANCHOR, GENEKEY_INSERT, 1)
    patched = patched.replace(PARSE_DIR_ANCHOR, PARSE_DIR_INSERT, 1)

    if ns.check:
        print("CHECK: all 5 anchors found; would inject resolver + entry_map_path + map-build + fail-loud guard.")
        return _verify(patched, applied=False)

    backup = TARGET.with_suffix(TARGET.suffix + ".pre_entry_resolver.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline="\n")
        print(f"OK: backup -> {backup}")
    TARGET.write_text(patched, encoding="utf-8", newline="\n")
    if b"\r\n" in TARGET.read_bytes():
        print("FAIL: CRLF detected in written file.")
        return 5
    print(f"OK: patched {TARGET}")
    return _verify(patched, applied=True)


def _verify(text: str, applied: bool) -> int:
    ok = True
    for needle in ["def resolve_eve_gene", "def load_eve_entry_map",
                   "entry_map_path: Optional[str | Path] = None",
                   "resolve_eve_gene(csv_file.stem, entry_map or {})",
                   "entry_map = load_eve_entry_map(self.entry_map_path)",
                   "_EVE_MIN_RESOLVED_FRACTION"]:
        present = needle in text
        print(f"  {'OK' if present else 'MISSING'}  {needle[:52]}")
        ok &= present
    try:
        compile(text, str(TARGET), "exec")
        print("  py-compile OK")
    except SyntaxError as e:
        print(f"  py-compile FAIL: {e}")
        ok = False
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
