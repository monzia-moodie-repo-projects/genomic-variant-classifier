#!/usr/bin/env python3
"""patch_esm2_llr_scorer.py -- add the LLR scoring path to ESM2Connector (Phase 1).

Adds (does NOT modify the existing embedding-delta path):
  * _load_transformers_mlm  -- loads EsmForMaskedLM (per-position AA logits),
    distinct from _load_transformers_model (EsmModel embeddings);
  * _llr_from_logit_row     -- pure LLR = logit[mut]-logit[wt] (negative=damaging);
  * _mlm_logit_matrix       -- one forward pass -> (L+2, vocab) numpy logits;
  * ESM2Connector._score_llr    -- WT-marginal (1 pass/protein) or masked-marginal
                                   (opt-in); skips wt_aa-vs-seq mismatches (counted);
  * ESM2Connector.annotate_llr  -- adds esm2_llr (0.0 = neutral/unscored).

Reuses the Phase-0 hardened _get_sequence (so ;-join recovery + _missing_genes
aggregation apply to LLR too). Count-guarded, backup-first, py_compile-gated,
idempotent. Author: Monzia Moodie.
"""
from __future__ import annotations

import datetime as _dt
import py_compile
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ESM2 = REPO / "src/genomic_variant_classifier/data/esm2.py"

_MOD_ANCHOR = (
    "# ---------------------------------------------------------------------------\n"
    "# Embedding computation\n"
    "# ---------------------------------------------------------------------------"
)
_MOD_NEW = (
    "# ---------------------------------------------------------------------------\n"
    "# LLR (log-likelihood-ratio) scoring -- masked-LM head (Phase 1)\n"
    "# ---------------------------------------------------------------------------\n"
    "def _load_transformers_mlm(model_name: str, device: str = \"cpu\"):\n"
    "    \"\"\"Load EsmForMaskedLM (per-position logits over the AA vocab) for LLR\n"
    "    scoring. Distinct from _load_transformers_model, which loads EsmModel\n"
    "    (embeddings) for the delta feature.\"\"\"\n"
    "    from transformers import AutoTokenizer, EsmForMaskedLM\n"
    "\n"
    "    key = f\"hf_mlm_{model_name}_{device}\"\n"
    "    if key not in _model_cache:\n"
    "        hf_name = f\"facebook/{model_name}\"\n"
    "        logger.info(\"Loading ESM-2 MLM (%s) via HuggingFace on %s ...\", model_name, device)\n"
    "        tok = AutoTokenizer.from_pretrained(hf_name)\n"
    "        mdl = EsmForMaskedLM.from_pretrained(hf_name)\n"
    "        mdl.eval()\n"
    "        if device != \"cpu\":\n"
    "            mdl = mdl.to(device)\n"
    "        _model_cache[key] = (tok, mdl)\n"
    "        logger.info(\"ESM-2 MLM loaded.\")\n"
    "    return _model_cache[key]\n"
    "\n"
    "\n"
    "def _llr_from_logit_row(logit_row, wt_id: int, mut_id: int) -> float:\n"
    "    \"\"\"LLR = logit[mut] - logit[wt]. The softmax partition function cancels in\n"
    "    the difference, so this equals the log-softmax difference over the full\n"
    "    vocab or over the 20 amino acids. Negative => mut less likely than wt =>\n"
    "    damaging.\"\"\"\n"
    "    return float(logit_row[mut_id] - logit_row[wt_id])\n"
    "\n"
    "\n"
    "def _mlm_logit_matrix(tok, mdl, seq: str, mask_token_idx: Optional[int] = None):\n"
    "    \"\"\"Single forward pass; returns the (L+2, vocab) logit matrix as numpy.\n"
    "    If mask_token_idx is given, that token is masked first (masked-marginal).\"\"\"\n"
    "    import torch\n"
    "\n"
    "    enc = tok(seq, return_tensors=\"pt\", add_special_tokens=True)\n"
    "    if mask_token_idx is not None:\n"
    "        enc[\"input_ids\"][0, mask_token_idx] = tok.mask_token_id\n"
    "    enc = {k: v.to(mdl.device) for k, v in enc.items()}\n"
    "    with torch.no_grad():\n"
    "        out = mdl(**enc)\n"
    "    return out.logits[0].float().cpu().numpy()\n"
    "\n"
    "\n"
    "# ---------------------------------------------------------------------------\n"
    "# Embedding computation\n"
    "# ---------------------------------------------------------------------------"
)

_CLS_ANCHOR = (
    "        scores: dict = {}\n"
    "        for idx, wt_ctx, mut_ctx, local_idx in plan:\n"
    "            ew = emap.get(wt_ctx)\n"
    "            em = emap.get(mut_ctx)\n"
    "            if ew is None or em is None or ew.ndim == 1 or em.ndim == 1:\n"
    "                continue\n"
    "            scores[idx] = float(np.linalg.norm(em[local_idx] - ew[local_idx]))\n"
    "        return scores"
)
_CLS_NEW = _CLS_ANCHOR + "\n\n" + (
    "    def _score_llr(self, candidates: pd.DataFrame, method: str = \"wt\") -> dict:\n"
    "        \"\"\"LLR per variant. WT-marginal (method=\"wt\", default): ONE forward\n"
    "        pass per protein scores all its variants. Masked-marginal\n"
    "        (method=\"masked\"): one pass per unique masked position (rigorous,\n"
    "        ~L x more passes). Skips wt_aa-vs-sequence mismatches (counted in\n"
    "        self._llr_n_mismatch).\"\"\"\n"
    "        tok, mdl = _load_transformers_mlm(self.model_name, device=self.device)\n"
    "        unk_id = tok.unk_token_id\n"
    "        scores: dict = {}\n"
    "        self._llr_n_mismatch = 0\n"
    "\n"
    "        by_gene: dict = {}\n"
    "        for row in candidates.itertuples():\n"
    "            gene = str(row.gene_symbol) if getattr(row, \"gene_symbol\", \"\") else \"\"\n"
    "            if gene:\n"
    "                by_gene.setdefault(gene, []).append(row)\n"
    "\n"
    "        for gene, rows in by_gene.items():\n"
    "            seq = self._get_sequence(gene)\n"
    "            if seq is None:\n"
    "                continue\n"
    "            seqlen = len(seq)\n"
    "            wt_mat = _mlm_logit_matrix(tok, mdl, seq) if method == \"wt\" else None\n"
    "            masked_cache: dict = {}\n"
    "            for row in rows:\n"
    "                pos1 = int(row.protein_pos)\n"
    "                wt_aa = str(row.wt_aa)\n"
    "                mut_aa = str(row.mut_aa)\n"
    "                # BOS at token index 0 -> residue pos1 (1-based) at token index pos1.\n"
    "                if not (1 <= pos1 <= seqlen) or seq[pos1 - 1] != wt_aa:\n"
    "                    self._llr_n_mismatch += 1\n"
    "                    continue\n"
    "                wt_id = tok.convert_tokens_to_ids(wt_aa)\n"
    "                mut_id = tok.convert_tokens_to_ids(mut_aa)\n"
    "                if wt_id == unk_id or mut_id == unk_id:\n"
    "                    self._llr_n_mismatch += 1\n"
    "                    continue\n"
    "                if method == \"wt\":\n"
    "                    logit_row = wt_mat[pos1]\n"
    "                else:\n"
    "                    if pos1 not in masked_cache:\n"
    "                        masked_cache[pos1] = _mlm_logit_matrix(\n"
    "                            tok, mdl, seq, mask_token_idx=pos1\n"
    "                        )[pos1]\n"
    "                    logit_row = masked_cache[pos1]\n"
    "                scores[row.Index] = _llr_from_logit_row(logit_row, wt_id, mut_id)\n"
    "        return scores\n"
    "\n"
    "    def annotate_llr(self, df: pd.DataFrame, method: str = \"wt\") -> pd.DataFrame:\n"
    "        \"\"\"Add ``esm2_llr`` (log-likelihood-ratio; NEGATIVE => damaging) to\n"
    "        *df*. 0.0 = neutral / unscored. esm2_llr is a CONTINUOUS feature: even\n"
    "        benign variants score negative, so the downstream model must learn the\n"
    "        threshold -- never treat sign as a class label. Requires the\n"
    "        transformers backend.\"\"\"\n"
    "        df = df.copy()\n"
    "        df[\"esm2_llr\"] = 0.0\n"
    "        if _BACKEND != \"transformers\":\n"
    "            logger.warning(\"ESM-2 LLR needs the transformers backend; esm2_llr all 0.0 (neutral).\")\n"
    "            return df\n"
    "        required = {\"gene_symbol\", \"protein_pos\", \"wt_aa\", \"mut_aa\"}\n"
    "        miss = required - set(df.columns)\n"
    "        if miss:\n"
    "            logger.info(\"ESM-2 LLR: columns %s absent -- esm2_llr=0.0.\", miss)\n"
    "            return df\n"
    "        is_missense = df.get(\"is_missense\", pd.Series(1, index=df.index)).astype(bool)\n"
    "        candidates = df[is_missense & df[\"protein_pos\"].notna() & df[\"wt_aa\"].notna() & df[\"mut_aa\"].notna()]\n"
    "        if candidates.empty:\n"
    "            return df\n"
    "        logger.info(\"Computing ESM-2 LLR (%s-marginal) for %d missense variants ...\", method, len(candidates))\n"
    "        scores = self._score_llr(candidates, method=method)\n"
    "        for idx, s in scores.items():\n"
    "            df.at[idx, \"esm2_llr\"] = s\n"
    "        logger.info(\"ESM-2 LLR: %d/%d variants scored.\", len(scores), len(candidates))\n"
    "        if getattr(self, \"_llr_n_mismatch\", 0):\n"
    "            logger.warning(\"ESM-2 LLR: %d variant(s) skipped on wt_aa-vs-sequence mismatch.\", self._llr_n_mismatch)\n"
    "        if self._missing_genes:\n"
    "            logger.warning(\n"
    "                \"ESM-2 LLR: %d gene(s) unresolved -> esm2_llr=0.0. Examples: %s\",\n"
    "                len(self._missing_genes), \", \".join(sorted(self._missing_genes)[:10]),\n"
    "            )\n"
    "        return df"
)

EDITS = [
    (_MOD_ANCHOR, _MOD_NEW, "def _load_transformers_mlm(", "module-level LLR loader + helpers"),
    (_CLS_ANCHOR, _CLS_NEW, "def annotate_llr(", "ESM2Connector._score_llr + annotate_llr"),
]


def main() -> int:
    if not ESM2.exists():
        print(f"ABORT: missing {ESM2}")
        return 2
    text = ESM2.read_text(encoding="utf-8")
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(ESM2, f"{ESM2}.bak_{ts}")
    for old, new, marker, label in EDITS:
        if marker in text:
            print(f"  skip (already applied): {label}")
            continue
        n = text.count(old)
        if n != 1:
            print(f"ABORT: anchor for '{label}' found {n}x (expected 1); no changes written")
            return 3
        text = text.replace(old, new, 1)
        print(f"  ok: {label}")
    ESM2.write_text(text, encoding="utf-8")
    try:
        py_compile.compile(str(ESM2), doraise=True)
    except py_compile.PyCompileError as exc:
        print(f"ABORT: py_compile failed: {exc}")
        return 4
    print(f"py_compile clean: esm2.py  (backup -> esm2.py.bak_{ts})")
    print("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
