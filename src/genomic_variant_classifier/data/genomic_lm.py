"""
src/genomic_variant_classifier/data/genomic_lm.py
=================================================
Nucleotide Transformer (DNA language model) connector -- Phase 2 (2026-07-05).

The DNA-sequence analogue of the ESM-2 (protein) connector: it scores the local
genomic neighbourhood of a variant with a nucleotide foundation model, giving the
whole ensemble a sequence signal the 1D-CNN could not extract on its own.

Features produced (both real, computed values -- NOT PHASE_2_FEATURES placeholders):
  genomiclm_delta_norm  float >= 0.0   L2 norm of (mean-pooled alt embedding
                                        - mean-pooled ref embedding) over the
                                        variant window. 0.0 = window unavailable
                                        / model unavailable (stub).
  genomiclm_llr         float          masked-LM log-likelihood ratio at the
                                        variant-centre token: logp(alt | context)
                                        - logp(ref | context). 0.0 = unavailable.

Window source (verified 2026-07-05): the ref/alt windows are NOT constructed here.
They are the pre-built 101 bp windows in the seq-window parquet
(clinvar_grch38_clean_seq.parquet, fasta_seq_ref/fasta_seq_alt, 0.0% null, valid
for SNVs *and* indels), resolved through the SAME
``seq_window_join.attach_delta_windows`` the sequence-CNN uses -- so NT and the
CNN see identical windows, and no per-variant window construction (which would be
wrong: the cohort's ``fasta_seq`` column is null) is attempted here.

Backends:
  HuggingFace ``transformers`` + ``torch`` -- preferred (Nucleotide Transformer v2
  is ESM-architecture based, plain transformers, CPU-friendly). If neither is
  installed the connector runs in STUB mode: features default to 0.0, logged once,
  never silently.

Score cache (mirrors the ESM-2 Fix-7 score cache, coordinate-keyed):
  (chrom, pos, ref, alt, model_name) -> (genomiclm_delta_norm, genomiclm_llr)
  at data/raw/cache/genomiclm_scores.parquet. Cache-hit == recompute.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
DEFAULT_WINDOW = 101
# `_POLY_A = "A" * DEFAULT_WINDOW` was DELETED 2026-07-15 (roadmap 6.28). It was already
# dead -- zero references in this module -- because its consumers (`self._poly` and
# `_mapped_mask`) were removed earlier the same day. I MISSED IT: I deleted the two uses
# and left the constant standing, which is exactly the "dead code left behind by a
# partial cleanup" that CLAUDE.md 4 forbids. It was found by
# tests/unit/test_no_content_based_poly_detection.py -- the repo-wide ban written that
# same hour, catching its own author's incomplete work on its first clean run.

_DEFAULT_SCORE_CACHE = os.environ.get(
    "GENOMICLM_SCORE_CACHE", "data/raw/cache/genomiclm_scores.parquet"
)

# -- backend detection (stub-safe; import must never require torch/transformers) --
try:
    import torch  # noqa: F401
    from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: F401

    _BACKEND: Optional[str] = "transformers"
except Exception:  # pragma: no cover - environment-dependent
    _BACKEND = None
    logger.debug(
        "Nucleotide Transformer: transformers+torch not installed. Stub mode "
        "(genomiclm_delta_norm/llr = 0.0). Install: pip install transformers torch"
    )

_model_cache: dict = {}


def _resolve_device(device: Optional[str]) -> str:
    """None/'auto' -> 'cuda' if available else 'cpu'; explicit value passes through."""
    if device in (None, "auto"):
        if _BACKEND == "transformers":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return "cpu"
    return device


def _load_model(model_name: str, device: str = "cpu"):
    """Load + cache the NT tokenizer + masked-LM (gives both hidden states and MLM logits)."""
    key = (model_name, device)
    if key not in _model_cache:
        import torch  # noqa: F401
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        mdl = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        mdl.eval().to(device)
        _model_cache[key] = (tok, mdl)
    return _model_cache[key]


# ---------------------------------------------------------------------------
# NT-specific numerics. These are the only functions that touch the model; the
# connector calls them so tests can substitute a deterministic fake embedder.
# Real forward-pass behaviour is exercised on the training box (huggingface.co
# is unreachable from the build sandbox), the framework below on both.
# ---------------------------------------------------------------------------
def _embed_mean_pooled(windows: list, model_name: str, device: str = "cpu") -> np.ndarray:
    """Return (N, hidden) mean-pooled last-hidden-state embeddings for DNA windows."""
    import torch

    tok, mdl = _load_model(model_name, device)
    out = np.empty((len(windows), int(mdl.config.hidden_size)), dtype=np.float32)
    with torch.no_grad():
        for i, w in enumerate(windows):
            enc = tok(w, return_tensors="pt").to(device)
            hid = mdl(**enc, output_hidden_states=True).hidden_states[-1]  # (1, T, H)
            mask = enc.get("attention_mask")
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                vec = (hid * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            else:
                vec = hid.mean(dim=1)
            out[i] = vec.squeeze(0).cpu().numpy()
    return out


def centre_token_index(tok, win: str, ids: list) -> Optional[int]:
    """Index of the token spanning the window's centre base, WITHOUT offset mapping.

    WHY THIS FUNCTION EXISTS -- 2026-07-15, roadmap 6.27
    ----------------------------------------------------
    It replaces one line, which was the single most expensive line in this module:

        off = tok(win, return_offsets_mapping=True).get("offset_mapping")

    Nucleotide Transformer's tokeniser is ``EsmTokenizer``, and ``EsmTokenizer.is_fast``
    is **False** -- a pure-Python ("slow") tokeniser. HuggingFace raises
    ``NotImplementedError`` for ``return_offsets_mapping`` on EVERY slow tokeniser;
    only ``PreTrainedTokenizerFast`` supports it. So that call raised on EVERY window,
    a bare ``except Exception`` two frames up swallowed it into a ``logger.debug``
    (below the default level -> printed NOTHING), and ``genomiclm_llr`` came back
    **identically 0.0 for all 4,420,180 cohort rows**, silently, since the connector
    was written. Measured on the owner's box, 2026-07-15::

        is_fast: False | class: EsmTokenizer
        OFFSET RAISES -> NotImplementedError: return_offset_mapping is not available
        when using Python tokenizers.

    It was invisible for two reasons at once. ``genomiclm_delta_norm`` never touches
    offset mapping, so the sibling feature stayed ALIVE and the connector looked
    healthy; and ``build_reference_slice`` FEEDS ``genomiclm_llr`` a synthetic
    ``rng.uniform(-12, 4)``, so the stage-5 zero-audit graded the fixture, never the
    connector (roadmap 7c: a gate that checks a PROXY is not a gate).

    HOW THIS WORKS, AND WHAT IT REFUSES TO ASSUME
    ---------------------------------------------
    Offsets are reconstructed from the TOKEN STRINGS the tokeniser actually produced.
    It therefore assumes nothing about k-mer size, how a non-multiple-of-k remainder is
    split, or how many special tokens are prepended -- all of which would be guesses.
    It reads what happened.

    The one assumption it DOES make -- that the non-special tokens concatenate back to
    the input -- is **asserted, not trusted**. A mismatch raises. Silently mislocating
    the centre would mask the WRONG base and yield a plausible, wrong log-likelihood
    ratio for every variant in the cohort: worse than zeros, because zeros are at least
    visibly dead.

    Returns the token index, or None if the centre base is not covered by any
    non-special token (counted and reported by the caller; never silently zeroed).
    """
    toks = tok.convert_ids_to_tokens(ids)
    special = set(tok.all_special_tokens or ())
    centre = len(win) // 2
    cursor = 0
    hit: Optional[int] = None
    for j, t in enumerate(toks):
        if t in special:
            continue
        span = len(t)
        if span == 0:
            continue
        if hit is None and cursor <= centre < cursor + span:
            hit = j
        cursor += span
    if cursor != len(win):
        raise RuntimeError(
            f"tokeniser round-trip mismatch: the non-special tokens span {cursor} "
            f"characters but the window is {len(win)}. The centre-token index would be "
            f"WRONG, so every log-likelihood ratio would score the wrong base. "
            f"tokeniser={type(tok).__name__}, is_fast={getattr(tok, 'is_fast', '?')}. "
            f"Refusing to guess."
        )
    return hit


def _masked_centre_logratio(ref_windows: list, alt_windows: list,
                            model_name: str, device: str = "cpu") -> np.ndarray:
    """Masked-LM log-likelihood ratio at the variant-centre token, per window pair.

    For each pair: mask the token covering the window centre in the ref window and
    in the alt window, and return logp(alt-centre-token) - logp(ref-centre-token).
    Non-overlapping k-mer tokenisation -> one token covers the centre base. Returns
    0.0 for a pair whenever the centre token cannot be located (logged, not silent).
    """
    import torch

    tok, mdl = _load_model(model_name, device)
    mask_id = tok.mask_token_id
    out = np.zeros(len(ref_windows), dtype=np.float32)
    if mask_id is None:
        logger.warning("NT tokenizer has no mask token; genomiclm_llr -> 0.0")
        return out

    def _logp_centre(win: str) -> Optional[float]:
        enc = tok(win, return_tensors="pt")
        ids = enc["input_ids"][0]
        tok_idx = centre_token_index(tok, win, ids.tolist())
        if tok_idx is None:
            return None
        true_id = int(ids[tok_idx])
        masked = ids.clone()
        masked[tok_idx] = mask_id
        with torch.no_grad():
            logits = mdl(input_ids=masked.unsqueeze(0).to(device)).logits[0, tok_idx]
        logp = torch.log_softmax(logits, dim=-1)[true_id]
        return float(logp)

    # NO bare `except Exception` HERE. This loop used to read:
    #
    #     try:
    #         ...
    #     except Exception as exc:  # pragma: no cover
    #         logger.debug("NT LLR failed for pair %d: %s", i, exc)
    #
    # and that is precisely what hid the offset-mapping NotImplementedError for the
    # entire life of this connector (see centre_token_index). Two compounding faults:
    # `logger.debug` is BELOW the default level, so nothing printed; and
    # `# pragma: no cover` told the coverage tool to stop looking. CLAUDE.md 4:
    # "A bare `except Exception` that logs and continues is a defect, not robustness --
    # it is exactly what erased a base model from the ensemble."
    #
    # Exceptions now propagate. The only tolerated outcome is a centre token that
    # genuinely cannot be located, which is COUNTED and reported -- and if it happens
    # to EVERY pair, that is a dead connector and it RAISES. An all-zero return is no
    # longer reachable in silence.
    n_unlocatable = 0
    for i, (rw, aw) in enumerate(zip(ref_windows, alt_windows)):
        lr = _logp_centre(rw)
        la = _logp_centre(aw)
        if lr is None or la is None:
            n_unlocatable += 1
            continue
        out[i] = la - lr

    n_pairs = len(ref_windows)
    if n_pairs and n_unlocatable == n_pairs:
        raise RuntimeError(
            f"genomiclm_llr: the centre token could not be located for ANY of "
            f"{n_pairs} window pairs (model={model_name!r}). Every score would be "
            f"0.0, which is indistinguishable from 'no effect'. This is a dead "
            f"connector, not a result. Refusing to return a column of zeros."
        )
    if n_unlocatable:
        logger.warning(
            "genomiclm_llr: centre token unlocatable for %d/%d pairs -> 0.0 for those "
            "rows (model=%s)", n_unlocatable, n_pairs, model_name,
        )
    return out


class NucleotideTransformerConnector:
    """DNA language-model connector. Adds genomiclm_delta_norm + genomiclm_llr."""

    _SCORE_CACHE_COLS = [
        "chrom", "pos", "ref", "alt", "model_name",
        "genomiclm_delta_norm", "genomiclm_llr",
    ]

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_path: Optional[Path] = None,
        seq_windows_path: Optional[Path] = None,
        device: Optional[str] = None,
        window: int = DEFAULT_WINDOW,
        min_coverage_warn: float = 0.5,
    ):
        self.model_name = model_name
        self.cache_path = Path(cache_path) if cache_path else Path(_DEFAULT_SCORE_CACHE)
        self.seq_windows_path = Path(seq_windows_path) if seq_windows_path else None
        self.device = _resolve_device(device)
        self.window = int(window)
        self.min_coverage_warn = float(min_coverage_warn)
        # `self._poly = "A" * self.window` was REMOVED 2026-07-15 (roadmap 6.28) along
        # with `_mapped_mask`, its only consumer. Window provenance now arrives as
        # WindowAttachment.usable, read from the builder's `ok` column. Nothing in this
        # module may infer "is this sequence real?" from the sequence's own content.

    # -- score cache (coordinate-keyed; mirrors the ESM-2 score cache) ---------
    def _score_cache_path(self) -> Path:
        return self.cache_path

    @staticmethod
    def _keys(frame: pd.DataFrame, model_name: str) -> list:
        return (
            frame["chrom"].astype(str) + "|" + frame["pos"].astype(str) + "|"
            + frame["ref"].astype(str) + "|" + frame["alt"].astype(str) + "|" + model_name
        ).tolist()

    def _cache_load(self) -> pd.DataFrame:
        p = self._score_cache_path()
        if p.is_file():
            try:
                return pd.read_parquet(p)
            except Exception:  # pragma: no cover
                logger.warning("genomiclm score cache unreadable, ignoring: %s", p)
        return pd.DataFrame(columns=self._SCORE_CACHE_COLS)

    def _cache_lookup(self, cache_df: pd.DataFrame, keys: list, col: str) -> dict:
        if cache_df.empty:
            return {}
        ck = self._keys(cache_df, self.model_name)
        return dict(zip(ck, cache_df[col]))

    def _cache_append(self, rows: list) -> None:
        if not rows:
            return
        p = self._score_cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        new = pd.DataFrame(rows, columns=self._SCORE_CACHE_COLS)
        existing = self._cache_load()
        combined = new if existing.empty else pd.concat([existing, new], ignore_index=True)
        combined["_k"] = self._keys(combined, self.model_name)
        combined = combined.drop_duplicates("_k", keep="last").drop(columns="_k")
        combined.to_parquet(p, index=False)

    # -- window resolution (the verified path; identical to the CNN's) --------
    def _resolve_windows(self, df: pd.DataFrame):
        """Return the seq_window_join.WindowAttachment for `df`.

        2026-07-15 (roadmap 6.28): returns the WindowAttachment itself, not a
        `(windows, n_unmapped)` pair. Callers must mask on `.usable`.
        """
        from genomic_variant_classifier.data.seq_window_join import attach_delta_windows

        return attach_delta_windows(df, self.seq_windows_path, self.window)

    # `_mapped_mask` IS GONE. It read:
    #
    #     def _mapped_mask(self, wins):
    #         """A row is 'mapped' iff its ref window is not the poly-A fallback."""
    #         return (wins["fasta_seq_ref"].astype(str) != self._poly).to_numpy()
    #
    # It was WRONG TWICE OVER, and both faults are now structurally impossible:
    #
    # 1. It tested for the JOIN's poly-A fallback and was blind to the BUILDER's
    #    poly-N placeholders -- 21,814 rows (0.494%) of the live 2026-07-10 artifact.
    #    For those rows ref and alt are BOTH poly-N, so ||alt_emb - ref_emb|| is
    #    exactly 0.0 -- the same value this module documents as "window unavailable /
    #    model unavailable (stub)". Three distinct conditions, one indistinguishable
    #    number.
    # 2. More fundamentally, it inferred provenance from CONTENT. A window that reads
    #    "A"*101 may be real: poly-A tracts are real biology. Content can never
    #    separate "the reference genuinely says A" from "we gave up and typed A".
    #
    # `WindowAttachment.usable` carries the builder's own `ok` verdict out of the
    # parquet, which is provenance rather than a guess about a string.

    # -- public API -----------------------------------------------------------
    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if _BACKEND is None:
            logger.info("Nucleotide Transformer stub mode -> genomiclm_delta_norm = 0.0")
            return df
        required = {"chrom", "pos", "ref", "alt"}
        if not required.issubset(df.columns):
            logger.warning("genomiclm: missing %s; skipping", required - set(df.columns))
            return df

        att = self._resolve_windows(df)
        wins, mapped = att.windows, att.usable
        cov = att.usable_fraction
        if cov < self.min_coverage_warn:
            logger.warning(
                "genomiclm: only %.1f%% of variants have a USABLE window "
                "(%d unmapped + %d builder-placeholder of %d -> "
                "genomiclm_delta_norm=0.0 for those rows). Check --seq-windows. [%s]",
                100.0 * cov, att.n_unmapped, att.n_placeholder, len(df), att.summary(),
            )

        out = np.zeros(len(df), dtype=np.float32)
        cand = df.loc[mapped, ["chrom", "pos", "ref", "alt"]].reset_index(drop=True)
        if len(cand):
            keys = self._keys(cand, self.model_name)
            hits = self._cache_lookup(self._cache_load(), keys, "genomiclm_delta_norm")
            need = [i for i, k in enumerate(keys) if k not in hits]
            vals = np.array([hits.get(k, np.nan) for k in keys], dtype=np.float32)
            if need:
                ref_w = wins.loc[mapped, "fasta_seq_ref"].to_numpy()[need].tolist()
                alt_w = wins.loc[mapped, "fasta_seq_alt"].to_numpy()[need].tolist()
                ref_emb = _embed_mean_pooled(ref_w, self.model_name, self.device)
                alt_emb = _embed_mean_pooled(alt_w, self.model_name, self.device)
                delta = np.linalg.norm(alt_emb - ref_emb, axis=1).astype(np.float32)
                for j, idx in enumerate(need):
                    vals[idx] = delta[j]
                self._cache_append([
                    [cand.chrom[i], cand.pos[i], cand.ref[i], cand.alt[i],
                     self.model_name, float(delta[j]), 0.0]
                    for j, i in enumerate(need)
                ])
            out[np.where(mapped)[0]] = vals
        df = df.copy()
        df["genomiclm_delta_norm"] = out
        logger.info(
            "genomiclm_delta_norm: %d/%d variants scored (>0), coverage %.1f%%",
            int((out > 0).sum()), len(df), 100.0 * cov,
        )
        return df

    def annotate_llr(self, df: pd.DataFrame) -> pd.DataFrame:
        if _BACKEND is None:
            return df
        required = {"chrom", "pos", "ref", "alt"}
        if not required.issubset(df.columns):
            return df
        att = self._resolve_windows(df)
        wins, mapped = att.windows, att.usable
        out = np.zeros(len(df), dtype=np.float32)
        idx_mapped = np.where(mapped)[0]
        if len(idx_mapped):
            ref_w = wins.loc[mapped, "fasta_seq_ref"].tolist()
            alt_w = wins.loc[mapped, "fasta_seq_alt"].tolist()
            llr = _masked_centre_logratio(ref_w, alt_w, self.model_name, self.device)
            out[idx_mapped] = llr
        df = df.copy()
        df["genomiclm_llr"] = out
        logger.info("genomiclm_llr: %d/%d variants scored (!=0)", int((out != 0).sum()), len(df))
        return df
