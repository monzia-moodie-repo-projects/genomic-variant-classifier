"""
src/genomic_variant_classifier/data/esm2.py
================
ESM-2 protein language model connector -- Phase 4A.

Computes the L2 norm of the per-residue embedding delta between the wildtype
and mutant protein sequences using Meta's ESM-2 model.  This
conservation-independent signal is one of the strongest available for
missense pathogenicity prediction.

Feature produced
----------------
  esm2_delta_norm  float >= 0.0
    L2 distance between the wildtype and mutant residue embeddings at the
    mutated position.  0.0 = not a missense variant / model unavailable.
    Higher values indicate greater structural/functional disruption.

Input columns consumed (all optional)
--------------------------------------
  gene_symbol     HGNC gene symbol -- used for UniProt sequence lookup
  protein_pos     1-based residue position within the canonical protein
  wt_aa           wildtype amino acid (single-letter code, e.g. 'A')
  mut_aa          mutant amino acid (single-letter code, e.g. 'V')
  is_missense     flag from engineer_features(); non-missense variants get 0.0

If protein_pos / wt_aa / mut_aa are absent (common when VEP HGVSp is not
available) the connector falls back to 0.0 for that variant.

Backends (tried in order)
--------------------------
1. HuggingFace ``transformers`` + ``torch`` -- preferred; supports all ESM-2 sizes
2. Meta ``fair-esm`` library -- original implementation
3. Stub mode -- returns 0.0 for every variant; zero dependencies

Install:
  pip install transformers torch      # HuggingFace backend
  # or
  pip install fair-esm                # Meta backend

Model size
----------
esm2_t6_8M_UR50D   (8M params,  ~32 MB) -- fast, good quality; default
esm2_t12_35M_UR50D (35M params, ~140 MB) -- higher quality, 4x slower
Override with ESM2_MODEL_NAME env var.

Caching
-------
UniProt sequences and computed embeddings are cached in SQLite so each
(uniprot_id, wt_seq) pair is only fetched/computed once.
Default cache: data/raw/cache/esm2_cache.sqlite
Override with ESM2_CACHE_PATH env var.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_MODEL = os.environ.get("ESM2_MODEL_NAME", "esm2_t6_8M_UR50D")
_DEFAULT_CACHE = Path(
    os.environ.get("ESM2_CACHE_PATH", "data/raw/cache/esm2_cache.sqlite")
)
_UNIPROT_REST = "https://rest.uniprot.org/uniprotkb"
_UNIPROT_GENE_SEARCH = (
    "https://rest.uniprot.org/uniprotkb/search"
    "?query=gene_exact:{gene}+AND+organism_id:9606+AND+reviewed:true"
    "&fields=accession,sequence&format=json&size=1"
)
_CONTEXT_WINDOW = 21  # residues either side of the mutation
_REQUEST_TIMEOUT = 10

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
_BACKEND: Optional[str] = None

try:
    import torch
    from transformers import AutoTokenizer, EsmModel

    _BACKEND = "transformers"
    logger.debug("ESM-2 backend: HuggingFace transformers + torch")
except ImportError:
    pass

if _BACKEND is None:
    try:
        import esm as _esm_lib  # fair-esm

        _BACKEND = "fair-esm"
        logger.debug("ESM-2 backend: fair-esm")
    except ImportError:
        logger.info(
            "ESM-2: neither 'transformers+torch' nor 'fair-esm' installed. "
            "Running in stub mode (esm2_delta_norm = 0.0). "
            "Install: pip install transformers torch"
        )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _open_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sequences "
        "(gene TEXT PRIMARY KEY, uniprot_id TEXT, sequence TEXT, fetched_at REAL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embeddings "
        "(seq_hash TEXT PRIMARY KEY, embedding BLOB, model TEXT, computed_at REAL)"
    )
    conn.commit()
    return conn


def _hash_seq(seq: str) -> str:
    return hashlib.sha1(seq.encode()).hexdigest()


def _cache_get_sequence(conn: sqlite3.Connection, gene: str) -> Optional[tuple[str, str]]:
    row = conn.execute(
        "SELECT uniprot_id, sequence FROM sequences WHERE gene = ?", (gene,)
    ).fetchone()
    return (row[0], row[1]) if row else None


def _cache_put_sequence(
    conn: sqlite3.Connection, gene: str, uniprot_id: str, sequence: str
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO sequences VALUES (?, ?, ?, ?)",
        (gene, uniprot_id, sequence, time.time()),
    )
    conn.commit()


def _cache_get_embedding(
    conn: sqlite3.Connection, seq_hash: str, model_name: str
) -> Optional[np.ndarray]:
    row = conn.execute(
        "SELECT embedding FROM embeddings WHERE seq_hash = ? AND model = ?",
        (seq_hash, model_name),
    ).fetchone()
    if row:
        return np.frombuffer(row[0], dtype=np.float32)
    return None


def _cache_put_embedding(
    conn: sqlite3.Connection,
    seq_hash: str,
    model_name: str,
    embedding: np.ndarray,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)",
        (seq_hash, embedding.astype(np.float32).tobytes(), model_name, time.time()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# UniProt sequence lookup
# ---------------------------------------------------------------------------

def _fetch_uniprot_sequence(gene: str, timeout: int = _REQUEST_TIMEOUT) -> Optional[tuple[str, str]]:
    """Return (uniprot_id, amino_acid_sequence) for a human gene or None."""
    try:
        url = _UNIPROT_GENE_SEARCH.format(gene=gene)
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None
        entry = results[0]
        uid = entry.get("primaryAccession", "")
        seq = entry.get("sequence", {}).get("value", "")
        if uid and seq:
            return uid, seq
    except Exception as exc:
        logger.debug("UniProt lookup failed for %s: %s", gene, exc)
    return None


# ---------------------------------------------------------------------------
# Model loading (lazy, module-level singleton)
# ---------------------------------------------------------------------------
_model_cache: dict[str, object] = {}


def _load_transformers_model(model_name: str, device: str = "cpu") -> tuple:
    """Load and cache ESM-2 tokenizer + model via HuggingFace.

    device="cpu" reproduces the original behavior exactly (model stays on CPU).
    """
    key = f"hf_{model_name}_{device}"
    if key not in _model_cache:
        logger.info("Loading ESM-2 (%s) via HuggingFace on %s ...", model_name, device)
        hf_name = f"facebook/{model_name}"
        tok = AutoTokenizer.from_pretrained(hf_name)
        mdl = EsmModel.from_pretrained(hf_name)
        mdl.eval()
        if device != "cpu":
            mdl = mdl.to(device)
        _model_cache[key] = (tok, mdl)
        logger.info("ESM-2 loaded.")
    return _model_cache[key]


def _load_fairesm_model(model_name: str) -> tuple:
    key = f"fairesm_{model_name}"
    if key not in _model_cache:
        logger.info("Loading ESM-2 (%s) via fair-esm ...", model_name)
        import esm as _esm_lib

        model, alphabet = _esm_lib.pretrained.load_model_and_alphabet(model_name)
        model.eval()
        batch_converter = alphabet.get_batch_converter()
        _model_cache[key] = (model, alphabet, batch_converter)
        logger.info("ESM-2 loaded.")
    return _model_cache[key]


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------

def _embed_sequence_transformers(seq: str, model_name: str) -> np.ndarray:
    """Return per-residue embeddings, shape (len(seq), hidden_dim)."""
    import torch

    tokenizer, model = _load_transformers_model(model_name)
    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # last_hidden_state: (1, seq_len+2, hidden) -- strip BOS/EOS tokens
    emb = outputs.last_hidden_state[0, 1:-1, :].cpu().numpy()
    return emb.astype(np.float32)


def _embed_sequences_transformers(seqs: list, model_name: str, device: str = "cpu") -> list:
    """Batched per-residue embeddings for a list of sequences.

    Returns a list of (len_i, hidden) arrays aligned to *seqs*. Pads the batch
    and passes attention_mask to the model; extracts each sequence's residues
    with a LENGTH-AWARE strip ([1 : real_len - 1] from attention_mask.sum()),
    not a fixed [1:-1] which is wrong under right-padding.
    """
    import torch

    tokenizer, model = _load_transformers_model(model_name, device=device)
    enc = tokenizer(list(seqs), return_tensors="pt", add_special_tokens=True, padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc)
    hidden = outputs.last_hidden_state  # (B, Lmax, H)
    mask = enc["attention_mask"]
    out = []
    for i in range(hidden.shape[0]):
        real_len = int(mask[i].sum().item())  # includes BOS + EOS
        emb = hidden[i, 1:real_len - 1, :].cpu().numpy().astype(np.float32)
        out.append(emb)
    return out


def _embed_sequence_fairesm(seq: str, model_name: str) -> np.ndarray:
    """Return per-residue embeddings, shape (len(seq), hidden_dim)."""
    import torch

    model, alphabet, batch_converter = _load_fairesm_model(model_name)
    data = [("variant", seq)]
    _, _, tokens = batch_converter(data)
    with torch.no_grad():
        results = model(tokens, repr_layers=[model.num_layers])
    # shape (1, seq_len+2, hidden) -- strip BOS/EOS
    emb = results["representations"][model.num_layers][0, 1:-1, :].cpu().numpy()
    return emb.astype(np.float32)


def _embed_sequence(seq: str, model_name: str, conn: sqlite3.Connection) -> Optional[np.ndarray]:
    h = _hash_seq(seq)
    cached = _cache_get_embedding(conn, h, model_name)
    if cached is not None:
        # reshape: stored flat
        return cached.reshape(-1, cached.shape[0] // len(seq)) if len(cached) % len(seq) == 0 else cached

    try:
        if _BACKEND == "transformers":
            emb = _embed_sequence_transformers(seq, model_name)
        elif _BACKEND == "fair-esm":
            emb = _embed_sequence_fairesm(seq, model_name)
        else:
            return None

        _cache_put_embedding(conn, h, model_name, emb.flatten())
        return emb
    except Exception as exc:
        logger.debug("ESM-2 embedding failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def _make_windows(full_sequence: str, protein_pos: int, wt_aa: str, mut_aa: str):
    """Build (wt_ctx, mut_ctx, local_idx) for the +/-_CONTEXT_WINDOW window.

    Mirrors _compute_delta's window construction exactly. Returns None for an
    out-of-range position (matching _compute_delta's 0.0). Like _compute_delta,
    it does NOT zero on a wt/UniProt mismatch (that is patch #2's concern).
    """
    seq_len = len(full_sequence)
    idx = protein_pos - 1
    if idx < 0 or idx >= seq_len:
        return None
    lo = max(0, idx - _CONTEXT_WINDOW)
    hi = min(seq_len, idx + _CONTEXT_WINDOW + 1)
    wt_ctx = full_sequence[lo:hi]
    mut_ctx = wt_ctx[: idx - lo] + mut_aa + wt_ctx[idx - lo + 1 :]
    return wt_ctx, mut_ctx, idx - lo


def _embed_sequences(seqs: list, model_name: str, conn: sqlite3.Connection,
                     device: str = "cpu", batch_size: int = 64) -> list:
    """Cache-aware BATCHED embedding. Returns embeddings aligned to *seqs*.

    Dedupes internally; SQLite cache hits (same flat layout as _embed_sequence)
    are served without a forward; misses are embedded in batches of batch_size.
    Falls back to per-sequence _embed_sequence for the fair-esm backend.
    """
    uniq = list(dict.fromkeys(seqs))
    result: dict = {}
    misses: list = []
    for s in uniq:
        cached = _cache_get_embedding(conn, _hash_seq(s), model_name)
        if cached is not None and len(s) > 0 and len(cached) % len(s) == 0:
            result[s] = cached.reshape(-1, len(cached) // len(s))
        else:
            misses.append(s)
    if misses:
        if _BACKEND == "transformers":
            for i in range(0, len(misses), batch_size):
                chunk = misses[i:i + batch_size]
                try:
                    embs = _embed_sequences_transformers(chunk, model_name, device=device)
                except Exception as exc:
                    logger.debug("ESM-2 batched embedding failed: %s", exc)
                    embs = [None] * len(chunk)
                for s, e in zip(chunk, embs):
                    if e is not None:
                        _cache_put_embedding(conn, _hash_seq(s), model_name, e.flatten())
                    result[s] = e
        else:
            for s in misses:
                result[s] = _embed_sequence(s, model_name, conn)
    return [result.get(s) for s in seqs]


def _compute_delta(
    full_sequence: str,
    protein_pos: int,   # 1-based
    wt_aa: str,
    mut_aa: str,
    model_name: str,
    conn: sqlite3.Connection,
) -> float:
    """
    Return ||embedding_mut[pos] - embedding_wt[pos]||_2.

    Uses a context window of +/- _CONTEXT_WINDOW residues to keep sequences
    short enough for CPU inference while capturing local structural context.
    """
    seq_len = len(full_sequence)
    # Convert to 0-based
    idx = protein_pos - 1

    if idx < 0 or idx >= seq_len:
        return 0.0
    if full_sequence[idx].upper() != wt_aa.upper():
        logger.debug(
            "Sequence mismatch at pos %d: expected %s got %s",
            protein_pos, wt_aa, full_sequence[idx],
        )
        # Still compute -- annotation may be off by one or use alt transcript

    lo = max(0, idx - _CONTEXT_WINDOW)
    hi = min(seq_len, idx + _CONTEXT_WINDOW + 1)
    wt_ctx = full_sequence[lo:hi]
    mut_ctx = wt_ctx[: idx - lo] + mut_aa + wt_ctx[idx - lo + 1 :]
    local_idx = idx - lo

    emb_wt = _embed_sequence(wt_ctx, model_name, conn)
    emb_mut = _embed_sequence(mut_ctx, model_name, conn)

    if emb_wt is None or emb_mut is None:
        return 0.0

    if emb_wt.ndim == 1 or emb_mut.ndim == 1:
        # Flat storage fallback
        return 0.0

    delta = emb_mut[local_idx] - emb_wt[local_idx]
    return float(np.linalg.norm(delta))


# ---------------------------------------------------------------------------
# Public connector
# ---------------------------------------------------------------------------

class ESM2Connector:
    """
    Annotates a variant DataFrame with ``esm2_delta_norm``.

    Parameters
    ----------
    model_name : str
        ESM-2 model variant (default: esm2_t6_8M_UR50D).
    cache_path : Path or str
        SQLite cache for sequences and embeddings.
    request_timeout : int
        Seconds for UniProt REST calls.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        cache_path: Path | str | None = None,
        request_timeout: int = _REQUEST_TIMEOUT,
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.cache_path = Path(cache_path) if cache_path else _DEFAULT_CACHE
        self.request_timeout = request_timeout
        self.device = device
        self.batch_size = batch_size
        self._conn: Optional[sqlite3.Connection] = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = _open_cache(self.cache_path)
        return self._conn

    def _get_sequence(self, gene: str) -> Optional[str]:
        conn = self._get_conn()
        cached = _cache_get_sequence(conn, gene)
        if cached:
            return cached[1]
        result = _fetch_uniprot_sequence(gene, self.request_timeout)
        if result:
            uid, seq = result
            _cache_put_sequence(conn, gene, uid, seq)
            return seq
        return None

    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ``esm2_delta_norm`` column to *df* in-place and return it.

        Only missense variants with gene_symbol, protein_pos, wt_aa, mut_aa
        present receive non-zero scores.  All others default to 0.0.
        """
        df = df.copy()
        df["esm2_delta_norm"] = 0.0

        if _BACKEND is None:
            logger.warning("ESM-2 stub mode -- all esm2_delta_norm values = 0.0")
            return df

        required = {"gene_symbol", "protein_pos", "wt_aa", "mut_aa"}
        missing = required - set(df.columns)
        if missing:
            logger.info(
                "ESM-2: columns %s absent -- defaulting to 0.0 (add VEP HGVSp parsing "
                "to populate these columns for missense variants).",
                missing,
            )
            return df

        is_missense = df.get("is_missense", pd.Series(1, index=df.index)).astype(bool)
        candidates = df[is_missense & df["protein_pos"].notna() & df["wt_aa"].notna() & df["mut_aa"].notna()]

        if candidates.empty:
            return df

        logger.info("Computing ESM-2 delta for %d missense variants ...", len(candidates))

        if _BACKEND == "transformers":
            scores = self._score_batched(candidates)
        else:
            scores = self._score_per_variant(candidates)

        for idx, score in scores.items():
            df.at[idx, "esm2_delta_norm"] = score

        n_scored = sum(1 for v in scores.values() if v > 0.0)
        logger.info("ESM-2: %d/%d variants scored (>0).", n_scored, len(candidates))
        return df

    def _score_per_variant(self, candidates: pd.DataFrame) -> dict:
        """Oracle scorer: one (wt, mut) embedding pair per variant (unbatched).

        This is the exact loop annotate_dataframe used before patch #1; it is the
        reference the batched path is checked against.
        """
        seq_cache: dict = {}
        scores: dict = {}
        for row in candidates.itertuples():
            gene = str(row.gene_symbol) if hasattr(row, "gene_symbol") and row.gene_symbol else ""
            if not gene:
                continue

            if gene not in seq_cache:
                seq_cache[gene] = self._get_sequence(gene)

            seq = seq_cache[gene]
            if seq is None:
                continue

            try:
                delta = _compute_delta(
                    full_sequence=seq,
                    protein_pos=int(row.protein_pos),
                    wt_aa=str(row.wt_aa),
                    mut_aa=str(row.mut_aa),
                    model_name=self.model_name,
                    conn=self._get_conn(),
                )
                scores[row.Index] = delta
            except Exception as exc:
                logger.debug("ESM-2 delta failed for %s: %s", getattr(row, "gene_symbol", "?"), exc)
        return scores

    def _score_batched(self, candidates: pd.DataFrame) -> dict:
        """Batched scorer: build + dedupe every (wt, mut) context window across
        candidates, embed the unique windows in batches on self.device, then
        compute deltas. Numerically equivalent to _score_per_variant
        (tests/unit/test_esm2_batched_equivalence.py).
        """
        seq_cache: dict = {}
        plan: list = []          # (row_index, wt_ctx, mut_ctx, local_idx)
        windows: dict = {}       # ordered set of unique context windows
        for row in candidates.itertuples():
            gene = str(row.gene_symbol) if hasattr(row, "gene_symbol") and row.gene_symbol else ""
            if not gene:
                continue

            if gene not in seq_cache:
                seq_cache[gene] = self._get_sequence(gene)

            seq = seq_cache[gene]
            if seq is None:
                continue

            try:
                w = _make_windows(seq, int(row.protein_pos), str(row.wt_aa), str(row.mut_aa))
            except Exception as exc:
                logger.debug("ESM-2 window build failed for %s: %s", getattr(row, "gene_symbol", "?"), exc)
                continue
            if w is None:
                continue
            wt_ctx, mut_ctx, local_idx = w
            windows.setdefault(wt_ctx, None)
            windows.setdefault(mut_ctx, None)
            plan.append((row.Index, wt_ctx, mut_ctx, local_idx))

        if not plan:
            return {}

        uniq = list(windows.keys())
        embs = _embed_sequences(
            uniq, self.model_name, self._get_conn(),
            device=self.device, batch_size=self.batch_size,
        )
        emap = {u: e for u, e in zip(uniq, embs)}

        scores: dict = {}
        for idx, wt_ctx, mut_ctx, local_idx in plan:
            ew = emap.get(wt_ctx)
            em = emap.get(mut_ctx)
            if ew is None or em is None or ew.ndim == 1 or em.ndim == 1:
                continue
            scores[idx] = float(np.linalg.norm(em[local_idx] - ew[local_idx]))
        return scores
