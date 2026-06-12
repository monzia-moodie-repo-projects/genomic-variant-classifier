#!/usr/bin/env python3
"""patch_rna_maxentscan_delta.py -- option-b RNA splice: emit maxentscan_delta
= score(alt) - score(ref) from the ref/alt context windows.
Idempotent, backup-first, py_compile-gated, ASCII, newline-preserving.
Author: Monzia Moodie."""
from __future__ import annotations
import py_compile, shutil, sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/pipelines/rna_pipeline.py")
MARKER = "maxentscan_delta"
EM = "\u2014"

EDITS = [
    (
        "        pd.DataFrame with four new columns.\n",
        "        pd.DataFrame with five new columns.\n",
        "docstring four->five",
    ),
    (
        '        result["maxentscan_score"]    = self.DEFAULT_MAXENTSCAN\n'
        '        result["dist_to_splice_site"] = self.DEFAULT_DIST_TO_SPLICE\n',
        '        result["maxentscan_score"]    = self.DEFAULT_MAXENTSCAN\n'
        '        result["maxentscan_delta"]    = self.DEFAULT_MAXENTSCAN\n'
        '        result["dist_to_splice_site"] = self.DEFAULT_DIST_TO_SPLICE\n',
        "init maxentscan_delta",
    ),
    (
        '        # --- MaxEntScan from fasta_seq context ---\n'
        '        if "fasta_seq" in df.columns:\n'
        '            fasta_col = df["fasta_seq"].fillna("")\n'
        '            splice_idx = result.index[splice_mask]\n'
        '\n'
        '            scores = []\n'
        '            for i in splice_idx:\n'
        '                seq = str(fasta_col.iloc[i] if isinstance(i, int) else fasta_col.loc[i])\n'
        '                center = len(seq) // 2   # variant position in 101-bp window\n'
        '\n'
        '                # Try donor score (variant at position +1 of GT)\n'
        '                donor_start  = center - 3\n'
        '                donor_end    = center + 6\n'
        '                acceptor_start = center - 20\n'
        '                acceptor_end   = center + 3\n'
        '\n'
        '                score = self.DEFAULT_MAXENTSCAN\n'
        '                if donor_start >= 0 and donor_end <= len(seq):\n'
        '                    seq9  = seq[donor_start:donor_end]\n'
        '                    score = _score_donor(seq9)\n'
        '                elif acceptor_start >= 0 and acceptor_end <= len(seq):\n'
        '                    seq23 = seq[acceptor_start:acceptor_end]\n'
        '                    score = _score_acceptor(seq23)\n'
        '                scores.append(score)\n'
        '\n'
        '            result.loc[splice_idx, "maxentscan_score"] = scores\n'
        '\n'
        '        else:\n'
        '            logger.warning(\n'
        '                "RNASpliceIsoformPipeline: \'fasta_seq\' column absent ' + EM + ' "\n'
        '                "maxentscan_score will use default 0.0 for %d splice variants.",\n'
        '                n_splice,\n'
        '            )\n',

        '        # --- MaxEntScan delta from ref/alt context windows ---\n'
        '        # Prefer variant-resolved [fasta_seq_ref, fasta_seq_alt] windows; the\n'
        '        # delta = score(alt) - score(ref) is the splice-disruption signal.\n'
        '        # Fall back to the legacy single fasta_seq (ref == alt -> delta 0),\n'
        '        # then to defaults. NOTE: donor/acceptor selection is bounds-based\n'
        '        # (always donor for a 101-bp window); fixing that is tracked separately.\n'
        '        def _window_score(seq):\n'
        '            center = len(seq) // 2\n'
        '            donor_start, donor_end = center - 3, center + 6\n'
        '            acceptor_start, acceptor_end = center - 20, center + 3\n'
        '            if donor_start >= 0 and donor_end <= len(seq):\n'
        '                return _score_donor(seq[donor_start:donor_end])\n'
        '            if acceptor_start >= 0 and acceptor_end <= len(seq):\n'
        '                return _score_acceptor(seq[acceptor_start:acceptor_end])\n'
        '            return self.DEFAULT_MAXENTSCAN\n'
        '\n'
        '        if "fasta_seq_ref" in df.columns and "fasta_seq_alt" in df.columns:\n'
        '            ref_col = df["fasta_seq_ref"].fillna("")\n'
        '            alt_col = df["fasta_seq_alt"].fillna("")\n'
        '        elif "fasta_seq" in df.columns:\n'
        '            ref_col = alt_col = df["fasta_seq"].fillna("")\n'
        '        else:\n'
        '            ref_col = None\n'
        '\n'
        '        if ref_col is not None:\n'
        '            splice_idx = result.index[splice_mask]\n'
        '            ref_scores = []\n'
        '            delta_scores = []\n'
        '            for i in splice_idx:\n'
        '                rseq = str(ref_col.iloc[i] if isinstance(i, int) else ref_col.loc[i])\n'
        '                aseq = str(alt_col.iloc[i] if isinstance(i, int) else alt_col.loc[i])\n'
        '                rs = _window_score(rseq) if rseq else self.DEFAULT_MAXENTSCAN\n'
        '                alt_s = _window_score(aseq) if aseq else rs\n'
        '                ref_scores.append(rs)\n'
        '                delta_scores.append(alt_s - rs)\n'
        '            result.loc[splice_idx, "maxentscan_score"] = ref_scores\n'
        '            result.loc[splice_idx, "maxentscan_delta"] = delta_scores\n'
        '        else:\n'
        '            logger.warning(\n'
        '                "RNASpliceIsoformPipeline: no fasta_seq_ref/fasta_seq_alt or "\n'
        '                "fasta_seq column -- maxentscan_score/delta default to 0.0 "\n'
        '                "for %d splice variants.",\n'
        '                n_splice,\n'
        '            )\n',
        "maxentscan delta block",
    ),
    (
        '        logger.info(\n'
        '            "RNASpliceIsoformPipeline: annotated %d / %d splice variants "\n'
        '            "(mean maxentscan=%.2f).",\n'
        '            n_splice,\n'
        '            n,\n'
        '            float(result.loc[splice_mask, "maxentscan_score"].mean()),\n'
        '        )\n',
        '        logger.info(\n'
        '            "RNASpliceIsoformPipeline: annotated %d / %d splice variants "\n'
        '            "(mean maxentscan=%.2f, mean |delta|=%.3f).",\n'
        '            n_splice,\n'
        '            n,\n'
        '            float(result.loc[splice_mask, "maxentscan_score"].mean()),\n'
        '            float(result.loc[splice_mask, "maxentscan_delta"].abs().mean()),\n'
        '        )\n',
        "logger mean delta",
    ),
]

def main() -> int:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found"); return 1
    raw = TARGET.read_text(encoding="utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    if MARKER in text:
        print("already applied (maxentscan_delta present); no change."); return 0
    for old, new, label in EDITS:
        c = text.count(old)
        if c != 1:
            print(f"ABORT: [{label}] anchor found {c} times (expected 1); no change."); return 1
        text = text.replace(old, new, 1)
    bak = TARGET.with_suffix(TARGET.suffix + ".bak")
    shutil.copy2(TARGET, bak)
    TARGET.write_text(text.replace("\n", nl), encoding="utf-8", newline="")
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as exc:
        shutil.copy2(bak, TARGET); print(f"ABORT: py_compile failed, restored:\n{exc}"); return 1
    print(f"OK: rna_pipeline {len(EDITS)} edits applied; backup at {bak}"); return 0

if __name__ == "__main__":
    sys.exit(main())
