#!/usr/bin/env python3
"""patch_esm2_llr_windowing.py -- bound the ESM-2 LLR forward pass to the model
context so long proteins (e.g. TTN ~34k aa) no longer OOM the O(L^2) attention.

Adds _MLM_MAX_RESIDUES + _windowed_logit_row, and routes _score_llr's WT- and
masked-marginal reads through the window ONLY for proteins longer than the cap;
short proteins keep the existing one-pass-per-protein fast path unchanged.

Idempotent, backup-first, py_compile-gated, ASCII-only, newline-preserving.
Author: Monzia Moodie.
"""
from __future__ import annotations
import py_compile, shutil, sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/data/esm2.py")
MARKER = "_windowed_logit_row"

A1_OLD = "_CONTEXT_WINDOW = 21  # residues either side of the mutation\n"
A1_NEW = (
    "_CONTEXT_WINDOW = 21  # residues either side of the mutation\n"
    "_MLM_MAX_RESIDUES = 1022  # ESM-2 context (1024 tokens) minus BOS/EOS; bounds LLR forward-pass length\n"
)

# Anchored on the UNIQUE return line of _mlm_logit_matrix (no dependence on the
# exact dash count of the section-separator comments). Inserts the helper right
# after that return, with two blank lines on each side (the original two blank
# lines that preceded the Embedding section remain, giving 2 below the helper).
A2_OLD = "    return out.logits[0].float().cpu().numpy()\n"
A2_NEW = (
    "    return out.logits[0].float().cpu().numpy()\n"
    "\n"
    "\n"
    "def _windowed_logit_row(tok, mdl, seq: str, pos1: int, mask: bool = False,\n"
    "                        max_residues: int = _MLM_MAX_RESIDUES):\n"
    "    \"\"\"Logit row for 1-based residue ``pos1``, scored within a window of at most\n"
    "    ``max_residues`` residues centered on ``pos1``.\n"
    "\n"
    "    A full-sequence pass on a long protein both exceeds ESM-2's ~1024-token\n"
    "    positional range and explodes the O(L^2) attention (TTN ~34k aa -> ~94 GB).\n"
    "    Windowing bounds the pass and keeps the residue inside the model range. For\n"
    "    ``len(seq) <= max_residues`` the window is the whole sequence and the read\n"
    "    index equals ``pos1`` -- i.e. identical to ``_mlm_logit_matrix(...)[pos1]``.\n"
    "    \"\"\"\n"
    "    idx0 = pos1 - 1\n"
    "    half = max_residues // 2\n"
    "    lo = max(0, idx0 - half)\n"
    "    hi = min(len(seq), lo + max_residues)\n"
    "    lo = max(0, hi - max_residues)            # re-anchor near the C-terminus\n"
    "    window = seq[lo:hi]\n"
    "    local_pos1 = idx0 - lo + 1                 # 1-based pos within window (BOS at token 0)\n"
    "    mask_idx = local_pos1 if mask else None\n"
    "    return _mlm_logit_matrix(tok, mdl, window, mask_token_idx=mask_idx)[local_pos1]\n"
)

A3_OLD = (
    "            seqlen = len(seq)\n"
    "            wt_mat = _mlm_logit_matrix(tok, mdl, seq) if method == \"wt\" else None\n"
    "            masked_cache: dict = {}\n"
)
A3_NEW = (
    "            seqlen = len(seq)\n"
    "            long_protein = seqlen > _MLM_MAX_RESIDUES\n"
    "            wt_mat = (\n"
    "                _mlm_logit_matrix(tok, mdl, seq)\n"
    "                if method == \"wt\" and not long_protein\n"
    "                else None\n"
    "            )\n"
    "            masked_cache: dict = {}\n"
)

A4_OLD = (
    "                if method == \"wt\":\n"
    "                    logit_row = wt_mat[pos1]\n"
    "                else:\n"
    "                    if pos1 not in masked_cache:\n"
    "                        masked_cache[pos1] = _mlm_logit_matrix(\n"
    "                            tok, mdl, seq, mask_token_idx=pos1\n"
    "                        )[pos1]\n"
    "                    logit_row = masked_cache[pos1]\n"
)
A4_NEW = (
    "                if method == \"wt\":\n"
    "                    logit_row = (\n"
    "                        _windowed_logit_row(tok, mdl, seq, pos1)\n"
    "                        if long_protein\n"
    "                        else wt_mat[pos1]\n"
    "                    )\n"
    "                else:\n"
    "                    if pos1 not in masked_cache:\n"
    "                        if long_protein:\n"
    "                            masked_cache[pos1] = _windowed_logit_row(\n"
    "                                tok, mdl, seq, pos1, mask=True\n"
    "                            )\n"
    "                        else:\n"
    "                            masked_cache[pos1] = _mlm_logit_matrix(\n"
    "                                tok, mdl, seq, mask_token_idx=pos1\n"
    "                            )[pos1]\n"
    "                    logit_row = masked_cache[pos1]\n"
)

EDITS = [("constant", A1_OLD, A1_NEW), ("helper", A2_OLD, A2_NEW),
         ("wt_mat", A3_OLD, A3_NEW), ("read", A4_OLD, A4_NEW)]

def main() -> int:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found"); return 1
    raw = TARGET.read_text(encoding="utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    if MARKER in text:
        print("already applied (_windowed_logit_row present); no change."); return 0
    for name, old, _ in EDITS:
        c = text.count(old)
        if c != 1:
            print(f"ABORT: anchor '{name}' found {c} times (expected 1); no change."); return 1
    for _, old, new in EDITS:
        text = text.replace(old, new, 1)
    bak = TARGET.with_suffix(TARGET.suffix + ".bak")
    shutil.copy2(TARGET, bak)
    TARGET.write_text(text.replace("\n", nl), encoding="utf-8", newline="")
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as exc:
        shutil.copy2(bak, TARGET)
        print(f"ABORT: py_compile failed, restored backup:\n{exc}"); return 1
    print(f"OK: LLR windowing applied; backup at {bak}"); return 0

if __name__ == "__main__":
    sys.exit(main())
