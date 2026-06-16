#!/usr/bin/env python3
"""scripts/preflight_run17.py -- the SINGLE Run-17 pre-flight gate (RUN17_SCOPE Gate F).

Run 17 activates two present-but-constant feature groups WITHOUT changing the schema:
  * gnn_score        via --string-db auto   (STRING-DB v12 GNN over the PPI graph)
  * af_1kg_{afr,eur,eas,sas,amr}  via --kg <1000G Phase-3 AF parquet>

This is the ONE script that fills/validates every launch variable so nothing is hand-typed
and nothing fails silently. It COMPOSES the validated command-level gate (preflight_gate.py)
and adds the three Run-17-specific checks the older gates miss:

  1. KG gate -- af_1kg_* must be EITHER activated (a --kg parquet that actually carries the
     five per-superpopulation AF source columns) OR consciously deferred (--defer-kg). The
     existing preflight_gate treats --kg as merely optional, so a silently-omitted --kg (or a
     --kg pointed at the EMPTY 1kgp/1000genomes dirs the registry warns about) would leave
     af_1kg_* constant with no signal. locate_1kg.py only checks the combined `allele_freq`
     column, not the per-superpop columns -- so it cannot validate a Run-17 kg parquet either.
  2. Schema gate -- the 81-column baseline (data/reference/schema/schema_baseline.json,
     n_columns must be 81). Guards the build_schema_baseline.py DEFAULT_MATRIX footgun that
     would silently regress the baseline 81 -> 78.
  3. Hard-gate scripts present -- verify_gnn_score.py / run_schema_drift_check.py /
     smoke_all_models.py must exist (they back Gates B and D).

Exit 0 = GO (all gates pass); non-zero = NO-GO. STRICTLY READ-ONLY; no training, no spend.

Usage:
  python scripts/preflight_run17.py --check "<full run_phase2_eval.py command>" [--data-root data]
  python scripts/preflight_run17.py --emit-kg   <kg_parquet> --output outputs\\run17   # activate af_1kg_*
  python scripts/preflight_run17.py --emit-defer --output outputs\\run17                # gnn_score only
Author: Monzia Moodie."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Compose the validated command-level gate (same dir; stdlib-only, main-guarded).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import preflight_gate as gate  # noqa: E402

# af_1kg_* target -> acceptable source-column names in the kg parquet (mirrors
# src/genomic_variant_classifier/data/thousandgenomes.py::_POP_TARGETS).
POP_TARGET_CANDIDATES = {
    "af_1kg_afr": ("af_afr", "AFR_AF", "afr_af", "af_1kg_afr"),
    "af_1kg_eur": ("af_eur", "EUR_AF", "eur_af", "af_1kg_eur"),
    "af_1kg_eas": ("af_eas", "EAS_AF", "eas_af", "af_1kg_eas"),
    "af_1kg_sas": ("af_sas", "SAS_AF", "sas_af", "af_1kg_sas"),
    "af_1kg_amr": ("af_amr", "AMR_AF", "amr_af", "af_1kg_amr"),
}
EXPECTED_SCHEMA_COLS = 82
SCHEMA_BASELINE_REL = "data/reference/schema/schema_baseline.json"
HARD_GATE_SCRIPTS = ("verify_gnn_score.py", "run_schema_drift_check.py", "smoke_all_models.py")

# gnn_score (THE Run-17 deliverable) source chain for --string-db auto. run_phase2_eval maps
# 'auto' -> threshold 700; gnn.py StringDBGraph.build() resolves a STRING source in this order:
#   cache_dir/string_graph_<thr>.pkl  ->  cache_dir/string_links.parquet  ->  local .txt.gz  ->  DOWNLOAD.
# If all local sources are absent the GNN downloads STRING v12 from stringdb-downloads.org ON THE GPU
# BOX (network + time dependency on a paid instance). The preflight verifies a local source exists.
STRING_CACHE_DIR = "data/raw/cache"
STRING_LOCAL_LINKS = "data/external/string/9606.protein.links.detailed.v12.0.txt.gz"
STRING_DEFAULT_THRESHOLD = 700


def _parquet_columns(path: Path) -> list[str]:
    """Read-only parquet column names from the footer schema (no row read)."""
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(str(path)).schema_arrow.names)


def kg_gate(ns, defer_kg: bool, data_root: str) -> list[tuple[str, str]]:
    """af_1kg_* must be activated-with-a-healthy-parquet XOR consciously deferred."""
    rows: list[tuple[str, str]] = []
    kg = vars(ns).get("kg")
    if defer_kg:
        if kg:
            rows.append(("FAIL", "--defer-kg set AND --kg supplied; choose ONE (activate or defer af_1kg_*)"))
        else:
            rows.append(("WARN", "af_1kg_* DEFERRED to Run 18: 5 cols stay CONSTANT this run "
                                 "(conscious deferral, not a silent stub). Run is gnn_score-only."))
        return rows
    if not kg:
        rows.append(("FAIL", "--kg absent and --defer-kg not set -> af_1kg_* would silently stay constant. "
                             "Registry warns 1kgp/1000genomes dirs are EMPTY (kg_path silent-zero). "
                             "Supply --kg <parquet>, or pass --defer-kg to omit on purpose."))
        return rows
    p = Path(kg)
    if not p.is_absolute():
        p = Path(data_root).parent / kg if Path(data_root).name == "data" else Path(kg)
    p = Path(kg) if Path(kg).exists() else p
    if not p.exists():
        rows.append(("FAIL", f"--kg parquet not found: {kg}"))
        return rows
    try:
        cols = set(_parquet_columns(p))
    except Exception as e:  # noqa: BLE001
        rows.append(("FAIL", f"--kg parquet unreadable ({p}): {e}"))
        return rows
    if "variant_id" not in cols or "allele_freq" not in cols:
        rows.append(("WARN", f"--kg parquet lacks variant_id/allele_freq -> combined-AF fallback won't fill ({p})"))
    missing = {t: POP_TARGET_CANDIDATES[t] for t in POP_TARGET_CANDIDATES
               if not (set(POP_TARGET_CANDIDATES[t]) & cols)}
    if missing:
        rows.append(("FAIL", f"--kg parquet missing per-superpop AF columns for {sorted(missing)}; "
                             f"af_1kg_* would stay CONSTANT (silent-zero). Need one of each: {missing}"))
    else:
        rows.append(("OK", f"--kg parquet carries all 5 per-superpop AF source columns ({p})"))
    return rows


def schema_gate(baseline_path: str | Path = SCHEMA_BASELINE_REL,
                expected: int = EXPECTED_SCHEMA_COLS) -> list[tuple[str, str]]:
    """The 82-col baseline must be intact (guards the build_schema_baseline DEFAULT_MATRIX footgun)."""
    p = Path(baseline_path)
    if not p.exists():
        return [("FAIL", f"schema baseline not found: {p}")]
    try:
        d = json.loads(p.read_bytes())  # bytes -> encoding-robust (BOM-safe), same lesson as finops
    except Exception as e:  # noqa: BLE001
        return [("FAIL", f"schema baseline unreadable: {e}")]
    n = d.get("n_columns")
    if n != expected:
        return [("FAIL", f"schema baseline n_columns={n} (expected {expected}); "
                         f"build_schema_baseline.py DEFAULT_MATRIX footgun may have regressed it -- "
                         f"rebuild with an explicit --matrix or restore.")]
    return [("OK", f"schema baseline intact: n_columns={n}, run_label={d.get('run_label')!r}, "
                   f"hash={str(d.get('expected_schema_hash'))[:12]}...")]


def scripts_gate(scripts_dir: str | Path = "scripts") -> list[tuple[str, str]]:
    rows = []
    for s in HARD_GATE_SCRIPTS:
        ok = (Path(scripts_dir) / s).exists()
        rows.append(("OK", f"hard-gate script present: {s}") if ok
                    else ("FAIL", f"hard-gate script MISSING: scripts/{s}"))
    return rows


def _string_threshold_from_ns(ns) -> int:
    """Mirror run_phase2_eval: 'auto' or non-digit --string-db -> 700, else the int value."""
    sd = vars(ns).get("string_db")
    if not sd or sd == "auto" or not str(sd).lstrip("-").isdigit():
        return STRING_DEFAULT_THRESHOLD
    return int(sd)


def string_db_gate(threshold: int = STRING_DEFAULT_THRESHOLD,
                   cache_dir: str | Path = STRING_CACHE_DIR,
                   local_links: str | Path = STRING_LOCAL_LINKS) -> list[tuple[str, str]]:
    """gnn_score is THE Run-17 deliverable. Verify a LOCAL STRING source exists for the run's threshold
    so the GNN doesn't fall back to a mid-run download from stringdb-downloads.org on the paid GPU box.
    Resolution order mirrors gnn.py StringDBGraph.build(): cached graph pkl -> cached links parquet ->
    local .txt.gz -> download."""
    graph_pkl = Path(cache_dir) / f"string_graph_{threshold}.pkl"
    links_pq = Path(cache_dir) / "string_links.parquet"
    local = Path(local_links)
    if graph_pkl.exists():
        return [("OK", f"STRING: cached graph present ({graph_pkl}) -> gnn_score uses it directly, no download")]
    if links_pq.exists():
        return [("OK", f"STRING: cached links parquet present ({links_pq}) -> graph rebuilt locally, no download")]
    if local.exists():
        return [("OK", f"STRING: local links file present ({local}) -> parsed locally, no download")]
    return [("WARN", f"STRING: no local source for threshold {threshold} ("
                     f"{graph_pkl}, {links_pq}, {local} all absent) -> gnn_score will DOWNLOAD STRING v12 from "
                     f"stringdb-downloads.org ON THE GPU BOX (network + time dependency). Pre-stage one of those "
                     f"files, or confirm the box has outbound network, before launch.")]


REACTOME_GMT_REL = "external/reactome/ReactomePathways.gmt"


def hetero_gate(ns, data_root: str) -> list[tuple[str, str]]:
    """hetero_gnn_score is a Run-17 no-defer deliverable. --hetero-gnn must be set AND --kg-edges must carry
    a reactome:<gmt> whose file exists, else the hetero-GNN has nothing to build and the column silently
    stays at its 0.5 default. Mirrors kg_gate (loud FAIL, never silent)."""
    rows: list[tuple[str, str]] = []
    if not getattr(ns, "hetero_gnn", False):
        rows.append(("FAIL", "--hetero-gnn absent -> hetero_gnn_score stays the 0.5 default (Run-17 "
                             "no-defer deliverable). Add --hetero-gnn."))
    edges = getattr(ns, "kg_edges", None) or []
    reactome = [e for e in edges if str(e).startswith("reactome:")]
    if not reactome:
        rows.append(("FAIL", "--kg-edges reactome:<gmt> absent -> the hetero-GNN has no KG relations to "
                             "overwrite hetero_gnn_score. Add --kg-edges reactome:<path>."))
    else:
        path = str(reactome[0]).split(":", 1)[1]
        if not Path(path).exists():
            rows.append(("FAIL", f"--kg-edges reactome path not found: {path}"))
        else:
            rows.append(("OK", f"--hetero-gnn + --kg-edges reactome present ({path})"))
    return rows


def run_all(command: str, data_root: str, n_train: int, defer_kg: bool,
            baseline_path: str | Path = SCHEMA_BASELINE_REL,
            scripts_dir: str | Path = "scripts",
            cache_dir: str | Path = STRING_CACHE_DIR,
            local_links: str | Path = STRING_LOCAL_LINKS) -> list[tuple[str, str]]:
    """All Run-17 gates, composed. Pure -> unit-testable."""
    ns = gate._parse_candidate(command)
    # let preflight_gate own command-structure + data-path existence; ack kg/finngen (kg_gate is authority on kg)
    rows = list(gate.validate(ns, data_root, n_train, ack={"kg", "finngen"}))
    rows += kg_gate(ns, defer_kg, data_root)
    rows += hetero_gate(ns, data_root)
    rows += string_db_gate(_string_threshold_from_ns(ns), cache_dir, local_links)
    rows += schema_gate(baseline_path)
    rows += scripts_gate(scripts_dir)
    return rows


# ---- command builders (emit the exact launch line; no hand-typed vars) ----
# Data + value flags are DERIVED from preflight_gate's contract so the emitted command can never
# drift from what the gate validates (single source of truth). This is what caught the original
# missing --lovd-path: REQUIRED_PATHS carries it, so the derived command always includes it.
def _data_flags(data_root: str = "data") -> list[str]:
    return [f"{flag} {Path(data_root).joinpath(rel).as_posix()}"
            for flag, rel in gate.REQUIRED_PATHS.items()]


def _fixed_flags() -> list[str]:
    return ([f"{f} {v}" for f, v in gate.REQUIRED_VALUES.items()]
            + [gate.SCALE_SKIP] + list(gate.REQUIRED_FLAGS))


def emit_command(kg_parquet: str | None, output: str, max_train: int | None,
                 data_root: str = "data") -> str:
    parts = ["python scripts/run_phase2_eval.py"] + _data_flags(data_root) + _fixed_flags()
    parts.append("--hetero-gnn")
    parts.append("--kg-edges reactome:" + Path(data_root).joinpath(REACTOME_GMT_REL).as_posix())
    if kg_parquet:
        parts.append(f"--kg {kg_parquet}")
    if max_train:
        parts.append(f"--max-train {max_train}")
    parts.append(f"--output {output}")
    return " ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run-17 single pre-flight gate (RUN17_SCOPE Gate F)")
    ap.add_argument("--check", help="full run_phase2_eval.py command to validate")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--n-train", type=int, default=1_200_000)
    ap.add_argument("--defer-kg", action="store_true", help="af_1kg_* deferred to Run 18 (gnn_score-only)")
    ap.add_argument("--baseline", default=SCHEMA_BASELINE_REL)
    ap.add_argument("--emit-kg", help="emit the full launch command activating af_1kg_* from this kg parquet")
    ap.add_argument("--emit-defer", action="store_true", help="emit the gnn_score-only launch command")
    ap.add_argument("--output", default="outputs/run17")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--string-cache-dir", default=STRING_CACHE_DIR,
                    help="dir holding string_graph_<thr>.pkl / string_links.parquet")
    ap.add_argument("--string-links", default=STRING_LOCAL_LINKS,
                    help="local STRING links .txt.gz path")
    a = ap.parse_args()

    if a.emit_kg or a.emit_defer:
        print(emit_command(a.emit_kg if a.emit_kg else None, a.output, a.max_train))
        return 0
    if not a.check:
        print("nothing to do: pass --check \"<command>\", or --emit-kg <parquet> / --emit-defer", file=sys.stderr)
        return 2

    rows = run_all(a.check, a.data_root, a.n_train, a.defer_kg, a.baseline,
                   cache_dir=a.string_cache_dir, local_links=a.string_links)
    order = {"FAIL": 0, "WARN": 1, "ACK-NEEDED": 1, "OK": 2}
    for level, msg in sorted(rows, key=lambda r: order.get(r[0], 1)):
        print(f"  {level:<10} {msg}")
    fails = [r for r in rows if r[0] in ("FAIL", "ACK-NEEDED")]
    print(f"\n{'RUN17 PREFLIGHT: NO-GO' if fails else 'RUN17 PREFLIGHT: GO'} -- "
          f"{sum(1 for r in rows if r[0]=='FAIL')} fail, "
          f"{sum(1 for r in rows if r[0]=='WARN')} warn, "
          f"{sum(1 for r in rows if r[0]=='OK')} ok.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
