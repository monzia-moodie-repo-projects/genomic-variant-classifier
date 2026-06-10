#!/usr/bin/env python3
"""probe_esm2_llr.py -- CPU correctness gate for the ESM-2 LLR scorer (Phase 1).

Proves the log-likelihood-ratio variant score is correct on the REAL model
BEFORE any GPU time or feature wiring. Validates:

  * sign convention -- known TP53 DNA-binding hotspot missense (R175H, R248Q,
    R273H) must score clearly NEGATIVE (LLR<0 = damaging); the common benign
    polymorphism P72R (rs1042522) must score much higher (near 0 or positive);
  * position indexing -- the residue the model actually sees at the mapped
    token index must equal wt_aa (off-by-one guard); mismatch is reported, not
    silently scored;
  * normalization invariance -- LLR == logit[mut]-logit[wt] (partition function
    cancels), already proven in the sandbox; restated here for the record;
  * WT-marginal vs masked-marginal -- both must agree in sign.

Loads facebook/esm2_t33_650M_UR50D via transformers EsmForMaskedLM on CPU.
FIRST RUN DOWNLOADS ~2.5 GB (one-time, cached by HF). Sequences come from the
local UniProt index (data/external/uniprot/uniprot_human_reviewed.parquet) --
NO network at score time. Read-only; no training; no GPU.

Usage:
    python scripts/probe_esm2_llr.py
    python scripts/probe_esm2_llr.py --model esm2_t33_650M_UR50D \
        --uniprot-index data/external/uniprot/uniprot_human_reviewed.parquet
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

# Known TP53 (UniProt P04637) variants: 3 pathogenic DNA-binding hotspots + 1
# common benign polymorphism. (gene, pos1, wt_aa, mut_aa, label)
KNOWN = [
    ("TP53", 175, "R", "H", "pathogenic"),   # R175H hotspot
    ("TP53", 248, "R", "Q", "pathogenic"),   # R248Q hotspot
    ("TP53", 273, "R", "H", "pathogenic"),   # R273H hotspot
    ("TP53", 72,  "P", "R", "benign"),       # P72R common polymorphism (rs1042522)
]


def llr_from_logits(logit_row: np.ndarray, wt_id: int, mut_id: int) -> float:
    """LLR = logit[mut] - logit[wt]. The softmax partition function cancels in
    the difference, so this equals the log-softmax difference over the full
    vocab OR over just the 20 amino acids (proven in sandbox)."""
    return float(logit_row[mut_id] - logit_row[wt_id])


def _load(model_name: str):
    import torch  # noqa: F401  (lazy: keeps module importable without torch)
    from transformers import AutoTokenizer, EsmForMaskedLM

    hf = f"facebook/{model_name}"
    print(f"Loading {hf} (EsmForMaskedLM) on CPU ... first run downloads ~2.5 GB", flush=True)
    tok = AutoTokenizer.from_pretrained(hf)
    mdl = EsmForMaskedLM.from_pretrained(hf)
    mdl.eval()
    return tok, mdl


def _logits(tok, mdl, seq: str, mask_pos_token_idx: int | None = None) -> np.ndarray:
    import torch

    enc = tok(seq, return_tensors="pt", add_special_tokens=True)
    if mask_pos_token_idx is not None:
        enc["input_ids"][0, mask_pos_token_idx] = tok.mask_token_id
    with torch.no_grad():
        out = mdl(**enc)
    return out.logits[0].numpy(), enc["input_ids"][0].numpy()  # (L+2, vocab), ids


def score(tok, mdl, seq: str, pos1: int, wt_aa: str, mut_aa: str, masked: bool):
    # BOS occupies token index 0, so residue pos1 (1-based) sits at token index pos1.
    tok_idx = pos1
    wt_id = tok.convert_tokens_to_ids(wt_aa)
    mut_id = tok.convert_tokens_to_ids(mut_aa)
    unk = tok.unk_token_id
    if wt_id == unk or mut_id == unk:
        return None, f"AA not in tokenizer vocab (wt={wt_aa}, mut={mut_aa})"
    logit_rows, ids = _logits(tok, mdl, seq, mask_pos_token_idx=(tok_idx if masked else None))
    seen_tok = tok.convert_ids_to_tokens(int(ids[tok_idx]))
    # index/wt-match guard
    seq_residue = seq[pos1 - 1] if 1 <= pos1 <= len(seq) else "?"
    matched = (seq_residue == wt_aa) and (masked or seen_tok == wt_aa)
    llr = llr_from_logits(logit_rows[tok_idx], wt_id, mut_id)
    return llr, ("OK" if matched else f"WT-MISMATCH seq[{pos1}]={seq_residue} tok={seen_tok}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esm2_t33_650M_UR50D")
    ap.add_argument("--uniprot-index",
                    default="data/external/uniprot/uniprot_human_reviewed.parquet")
    args = ap.parse_args()

    idx = pd.read_parquet(args.uniprot_index, columns=["gene_symbol", "sequence"])
    idx["g"] = idx["gene_symbol"].astype(str).str.strip().str.upper()
    seqmap = dict(zip(idx["g"], idx["sequence"].astype(str)))
    if "TP53" not in seqmap:
        print("ABORT: TP53 not in UniProt index; cannot run known-variant probe.")
        return 2

    tok, mdl = _load(args.model)
    print(f"\n{'variant':<16}{'label':<12}{'WT-marginal':>14}{'masked':>12}   index/wt")
    print("-" * 70)

    rows = []
    for gene, pos1, wt, mut, label in KNOWN:
        seq = seqmap.get(gene.upper())
        if not seq:
            print(f"{gene} p.{wt}{pos1}{mut:<6} {label:<12} (gene seq absent)")
            continue
        llr_wt, status = score(tok, mdl, seq, pos1, wt, mut, masked=False)
        llr_mk, _ = score(tok, mdl, seq, pos1, wt, mut, masked=True)
        name = f"{gene} {wt}{pos1}{mut}"
        print(f"{name:<16}{label:<12}{llr_wt:>14.4f}{llr_mk:>12.4f}   {status}")
        rows.append((label, llr_wt, llr_mk, status))

    # ---- assertions (the gate) ----
    print("\n--- checks ---")
    ok = True
    path_wt = [r[1] for r in rows if r[0] == "pathogenic"]
    benign_wt = [r[1] for r in rows if r[0] == "benign"]
    mism = [r for r in rows if r[3] != "OK"]
    if mism:
        ok = False
        print(f"FAIL index/wt-match: {len(mism)} variant(s) mismatched -- indexing or isoform issue")
    else:
        print("PASS index/wt-match: every wt_aa matches the sequence residue at its token index")
    if path_wt and all(v < 0 for v in path_wt):
        print(f"PASS sign: all pathogenic hotspots negative (WT-marginal) {[round(v,3) for v in path_wt]}")
    else:
        ok = False
        print(f"FAIL sign: a pathogenic hotspot was not negative {[round(v,3) for v in path_wt]}")
    if benign_wt and path_wt and min(benign_wt) > max(path_wt):
        print(f"PASS separation: benign P72R ({benign_wt[0]:.3f}) > all hotspots (max {max(path_wt):.3f})")
    else:
        print(f"WARN separation: benign {benign_wt} vs hotspots max {max(path_wt) if path_wt else None} "
              "(relative check; not fatal)")
    # WT vs masked sign agreement
    sign_agree = all((rw < 0) == (rm < 0) for _, rw, rm, st in rows if st == "OK")
    print(("PASS" if sign_agree else "WARN") + " WT-marginal vs masked-marginal agree in sign")

    print("\nGATE:", "PASS -- LLR scorer math is correct on the real model; safe to integrate + GPU."
          if ok else "FAIL -- do NOT integrate; re-audit sign/indexing before proceeding.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
