#!/usr/bin/env python3
"""install_docs_incident_protein_coord.py -- record the protein-coord index
corruption + repair (2026-06-12). Creates the incident doc, appends CHANGELOG and a
ROADMAP delta. Append-only, idempotent (marker-guarded), no-BOM, CRLF, ASCII.
Author: Monzia Moodie."""
from __future__ import annotations

import sys
from pathlib import Path

DOCS = Path("docs")
INCIDENT = DOCS / "incidents" / "INCIDENT_2026-06-12_protein-coord-index-corruption.md"
CHANGELOG = DOCS / "CHANGELOG.md"
ROADMAP = DOCS / "ROADMAP.md"

CL_MARKER = "<!-- incident: protein-coord-index-corruption 2026-06-12 -->"
RM_MARKER = "<!-- roadmap-delta: protein-coord-rebuild 2026-06-12 -->"

INCIDENT_BODY = """# INCIDENT 2026-06-12 -- protein-coord index corrupted by sample rebuild

## Status: RESOLVED

## Summary
During Run-16 AlphaMissense verification, the protein-coord index
`data/external/alphamissense/alphamissense_protein_index.parquet` (17.8 MB, full
cohort) was overwritten with a 50k-sample-only index (0.29 MB). A coverage probe was
re-run after the cache file was deleted; `ProteinCoordConnector._build_index` filters
to the cohort passed to `annotate_dataframe` and writes the result to the canonical
cache path, so passing only a 50k sample produced a sample-sized cache in place of the
full one.

## Detection
Cache file size: 0.29 MB vs the expected ~18 MB full index. (Coverage on the same
`random_state=0` sample still read 0.9672, a false pass -- the tiny index covered
exactly the sample it was built from.)

## Impact (averted before launch)
A Run-16 regen would have loaded the tiny cache as-is (the connector never rebuilds
when a cache file exists), yielding ~1% `protein_pos` coverage on the full 2.49M-
missense cohort -> the protein-coord coverage gate would have aborted the regen. This
is the same silent-ESM-2-zero class that capped Run 15 at 3,451 of ~2.49M.

## Resolution
- Hardened the probe (`scripts/probe_protein_coord_coverage.py`, v2): default mode is
  READ-ONLY and refuses to build from a sample; size-checks the cache (full ~18 MB vs
  sample <1 MB) so a corrupt cache FAILS even when the reused sample would match; full
  rebuild is an explicit `--rebuild-full` that reads the entire cohort.
- Rebuilt the FULL index from the full cohort + TSV: 4,399,089 cohort rows ->
  18.64 MB cache, full-cohort coverage 0.9665 (2,405,448 / 2,488,889 missense).
- Read-only verify: 18.64 MB, coverage 0.9672, exit 0.

## Standing lessons
1. The protein-coord index MUST be built from the FULL cohort, never a sample.
2. Any diagnostic that calls `annotate_dataframe` on a cache-miss WILL rebuild and
   overwrite the canonical cache -- diagnostics must be read-only or use a temp dir.
3. Validate the cache by SIZE (full ~18 MB), not existence or coverage-on-the-same
   sample.
4. Run-16 `--alphamissense` = the TSV `data/external/alphamissense/AlphaMissense_hg38.tsv.gz`,
   NOT the scores parquet. `train.py` help points at `alphamissense_scores_hg38.parquet`,
   but that parquet's directory lacks the protein-index, so `ProteinCoordConnector`
   would deadzone ESM-2. (The connector reads the TSV for scores via `_parse_tsv`.)
5. Ship the rebuilt 18.64 MB index to the Vast.ai box, co-located with the
   `--alphamissense` source dir, so the regen loads it instead of re-scanning the
   613 MB TSV.
"""

CL_ENTRY = f"""
{CL_MARKER}
### 2026-06-12 -- protein-coord index corruption + repair
- Failed: probe v1 re-run after `Remove-Item` of the cache rebuilt the protein-coord
  index from a 50k sample, overwriting the full 17.8 MB index with a 0.29 MB one.
- Learned: `ProteinCoordConnector._build_index` filters to the passed cohort and writes
  the canonical cache; diagnostics must be read-only. Validate the cache by size.
- Fixed: probe v2 (read-only default + size guard + explicit `--rebuild-full`); full
  rebuild -> 18.64 MB, full-cohort coverage 0.9665 (2,405,448/2,488,889 missense).
- Confirmed: Run-16 `--alphamissense` = TSV (not the scores parquet); 96.65% full-cohort
  protein-coord coverage means ESM-2 will populate.
"""

RM_ENTRY = f"""
{RM_MARKER}
## ROADMAP delta -- 2026-06-12 (protein-coord index repair + Run-16 input contract)
- Protein-coord index rebuilt full-cohort: 18.64 MB, 0.9665 coverage (ESM-2 ready).
- Run-16 launch contract (mandatory flags): `--clinvar` clean_seq cohort (ref/alt);
  `--esm2-model esm2_t33_650M_UR50D`; `--esm2-uniprot-index uniprot_human_reviewed.parquet`;
  `--alphamissense data/external/alphamissense/AlphaMissense_hg38.tsv.gz` (TSV, NOT the
  scores parquet); `--gnomad-constraint data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv`.
- Data staging: ship the 18.64 MB protein-coord index to Vast.ai co-located with the
  `--alphamissense` dir so the regen loads it (no 613 MB TSV re-scan).
- Open: decouple the protein-coord source from `--alphamissense` (elegant fix removing
  the scores-parquet-vs-TSV trap); fold a coverage/size check into the preflight.
"""


def _append(path: Path, marker: str, entry: str) -> str:
    if not path.exists():
        return f"SKIP (missing): {path}"
    with path.open("r", encoding="utf-8", newline="") as f:
        raw = f.read()
    if marker in raw:
        return f"already present: {path.name}"
    nl = "\r\n" if "\r\n" in raw else "\n"
    body = raw.replace("\r\n", "\n").rstrip("\n") + "\n" + entry
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(body.replace("\n", nl))
    return f"appended: {path.name}"


def main() -> int:
    if not DOCS.exists():
        print("ERROR: run from repo root (docs/ not found).")
        return 2
    INCIDENT.parent.mkdir(parents=True, exist_ok=True)
    if INCIDENT.exists():
        print(f"already present: {INCIDENT.name}")
    else:
        with INCIDENT.open("w", encoding="utf-8", newline="") as f:
            f.write(INCIDENT_BODY.replace("\n", "\r\n"))
        print(f"created: {INCIDENT.name}")
    print(_append(CHANGELOG, CL_MARKER, CL_ENTRY))
    print(_append(ROADMAP, RM_MARKER, RM_ENTRY))
    print("NEXT: python scripts/make_roadmap_docx.py ; then review git diff and commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
