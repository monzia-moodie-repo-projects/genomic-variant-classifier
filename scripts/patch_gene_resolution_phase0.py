#!/usr/bin/env python3
"""patch_gene_resolution_phase0.py -- wire the shared gene-symbol helper.

Phase 0 (part 1 of 2): esm2.py + protein_pipeline.py. eve.py follows once its
import region is confirmed.

Count-guarded (each old-string must appear EXACTLY once or the script aborts),
backup-first, idempotent (skips an edit whose new-string marker is already
present), py_compile-gated per file. Author: Monzia Moodie.

Requires src/genomic_variant_classifier/data/gene_symbols.py to be in place.
"""
from __future__ import annotations

import datetime as _dt
import py_compile
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ESM2 = REPO / "src/genomic_variant_classifier/data/esm2.py"
PROT = REPO / "src/genomic_variant_classifier/pipelines/protein_pipeline.py"

# (path, old, new, idempotency-marker, label)
EDITS = [
    # ---- esm2.py ----
    (ESM2,
     '_DEFAULT_MODEL = os.environ.get("ESM2_MODEL_NAME", "esm2_t6_8M_UR50D")',
     'from genomic_variant_classifier.data.gene_symbols import (\n'
     '    gene_symbol_candidates,\n'
     '    normalize_gene_symbol,\n'
     ')\n\n'
     '_DEFAULT_MODEL = os.environ.get("ESM2_MODEL_NAME", "esm2_t6_8M_UR50D")',
     'from genomic_variant_classifier.data.gene_symbols import (',
     'esm2: import shared gene-symbol helper'),

    (ESM2,
     '        self._conn: Optional[sqlite3.Connection] = None\n'
     '        self._uniprot_index: Optional[dict] = None\n'
     '        self._warned_missing = False',
     '        self._conn: Optional[sqlite3.Connection] = None\n'
     '        self._uniprot_index: Optional[dict] = None\n'
     '        self._warned_missing = False\n'
     '        self._missing_genes: set = set()',
     'self._missing_genes: set = set()',
     'esm2: init _missing_genes accumulator'),

    (ESM2,
     '            seq = self._uniprot_index.get(str(gene).strip().upper())\n'
     '            if seq:\n'
     '                _cache_put_sequence(conn, gene, "", seq)\n'
     '                return seq\n'
     '            if not self.allow_network:\n'
     '                if not self._warned_missing:\n'
     '                    logger.warning(\n'
     '                        "ESM-2: gene(s) absent from the UniProt index and network "\n'
     '                        "disabled -- those variants get esm2_delta_norm=0.0 "\n'
     '                        "(first missing: %s).", gene,\n'
     '                    )\n'
     '                    self._warned_missing = True\n'
     '                return None',
     '            seq = None\n'
     '            for _cand in gene_symbol_candidates(gene):\n'
     '                seq = self._uniprot_index.get(_cand)\n'
     '                if seq:\n'
     '                    break\n'
     '            if seq:\n'
     '                _cache_put_sequence(conn, gene, "", seq)\n'
     '                return seq\n'
     '            if not self.allow_network:\n'
     '                self._missing_genes.add(normalize_gene_symbol(gene))\n'
     '                if not self._warned_missing:\n'
     '                    logger.warning(\n'
     '                        "ESM-2: one or more gene symbols are absent from the "\n'
     '                        "UniProt index and network is disabled -- those variants "\n'
     '                        "get esm2_delta_norm=0.0 (first: %s). Aggregate count is "\n'
     '                        "logged at the end of annotate_dataframe.", gene,\n'
     '                    )\n'
     '                    self._warned_missing = True\n'
     '                return None',
     'for _cand in gene_symbol_candidates(gene):',
     'esm2: _get_sequence candidate loop + miss accumulation'),

    (ESM2,
     '        n_scored = sum(1 for v in scores.values() if v > 0.0)\n'
     '        logger.info("ESM-2: %d/%d variants scored (>0).", n_scored, len(candidates))\n'
     '        return df',
     '        n_scored = sum(1 for v in scores.values() if v > 0.0)\n'
     '        logger.info("ESM-2: %d/%d variants scored (>0).", n_scored, len(candidates))\n'
     '        if self._missing_genes:\n'
     '            _missing_norm = candidates["gene_symbol"].map(normalize_gene_symbol)\n'
     '            _n_missing_var = int(_missing_norm.isin(self._missing_genes).sum())\n'
     '            logger.warning(\n'
     '                "ESM-2: %d gene symbol(s) absent from the UniProt index -> %d "\n'
     '                "candidate missense variant(s) scored 0.0. Examples: %s",\n'
     '                len(self._missing_genes), _n_missing_var,\n'
     '                ", ".join(sorted(self._missing_genes)[:10]),\n'
     '            )\n'
     '        return df',
     'gene symbol(s) absent from the UniProt index -> %d',
     'esm2: annotate_dataframe aggregate missing-gene log'),

    # ---- protein_pipeline.py ----
    (PROT,
     'import requests\n\nlogger = logging.getLogger(__name__)',
     'import requests\n\n'
     'from genomic_variant_classifier.data.gene_symbols import gene_symbol_candidates\n\n'
     'logger = logging.getLogger(__name__)',
     'from genomic_variant_classifier.data.gene_symbols import gene_symbol_candidates',
     'protein_pipeline: import shared gene-symbol helper'),

    (PROT,
     '        accession: Optional[str] = None\n'
     '        try:\n'
     '            url = UNIPROT_LOOKUP.format(symbol=gene_symbol)\n'
     '            resp = requests.get(url, timeout=_REQUEST_TIMEOUT)\n'
     '            if resp.ok:\n'
     '                lines = resp.text.strip().splitlines()\n'
     '                if len(lines) > 1:   # header + at least one result\n'
     '                    accession = lines[1].strip()\n'
     '        except Exception as exc:\n'
     '            logger.debug("UniProt lookup failed for %s: %s", gene_symbol, exc)',
     '        accession: Optional[str] = None\n'
     '        for _cand in gene_symbol_candidates(gene_symbol):\n'
     '            try:\n'
     '                url = UNIPROT_LOOKUP.format(symbol=_cand)\n'
     '                resp = requests.get(url, timeout=_REQUEST_TIMEOUT)\n'
     '                if resp.ok:\n'
     '                    lines = resp.text.strip().splitlines()\n'
     '                    if len(lines) > 1:   # header + at least one result\n'
     '                        accession = lines[1].strip()\n'
     '                        if accession:\n'
     '                            break\n'
     '            except Exception as exc:\n'
     '                logger.debug("UniProt lookup failed for %s: %s", _cand, exc)',
     '        for _cand in gene_symbol_candidates(gene_symbol):',
     'protein_pipeline: get_accession candidate loop'),
]


def main() -> int:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    by_file: dict = {}
    for path, old, new, marker, label in EDITS:
        by_file.setdefault(path, []).append((old, new, marker, label))

    for path, edits in by_file.items():
        if not path.exists():
            print(f"ABORT: missing {path}")
            return 2
        text = path.read_text(encoding="utf-8")
        shutil.copy2(path, f"{path}.bak_{ts}")
        for old, new, marker, label in edits:
            if marker in text:
                print(f"  skip (already applied): {label}")
                continue
            n = text.count(old)
            if n != 1:
                print(f"ABORT: anchor for '{label}' found {n}x (expected 1); no changes written to {path.name}")
                return 3
            text = text.replace(old, new, 1)
            print(f"  ok: {label}")
        path.write_text(text, encoding="utf-8")
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            print(f"ABORT: py_compile failed for {path.name}: {exc}")
            return 4
        print(f"py_compile clean: {path.name}  (backup -> {path.name}.bak_{ts})")
    print("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
