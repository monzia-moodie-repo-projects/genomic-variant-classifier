"""
scripts/patch_esm2_batched.py
=============================
Patch #1: add a BATCHED + device embedding path to
src/genomic_variant_classifier/data/esm2.py, leaving the per-variant oracle
(_compute_delta / _embed_sequence) byte-for-byte untouched.

Strictly semantics-preserving: annotate_dataframe now dispatches to a batched
scorer for the transformers backend, but the batched scorer is numerically
equivalent to the per-variant loop (gated by tests/unit/test_esm2_batched_equivalence.py).

Discipline: reads/writes BYTES (LF preserved, no BOM); each anchor must match
exactly once or the patch aborts; a timestamped .bak is written first; rerun is
a no-op once patched; py_compile is run on the result.

Usage:
  python scripts/patch_esm2_batched.py
"""

from __future__ import annotations

import datetime as _dt
import py_compile
import sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/data/esm2.py")

# --------------------------------------------------------------------------
# Edit 1: device-aware model loader
# --------------------------------------------------------------------------
OLD1 = '''def _load_transformers_model(model_name: str) -> tuple:
    """Load and cache ESM-2 tokenizer + model via HuggingFace."""
    key = f"hf_{model_name}"
    if key not in _model_cache:
        logger.info("Loading ESM-2 (%s) via HuggingFace ...", model_name)
        hf_name = f"facebook/{model_name}"
        tok = AutoTokenizer.from_pretrained(hf_name)
        mdl = EsmModel.from_pretrained(hf_name)
        mdl.eval()
        _model_cache[key] = (tok, mdl)
        logger.info("ESM-2 loaded.")
    return _model_cache[key]'''

NEW1 = '''def _load_transformers_model(model_name: str, device: str = "cpu") -> tuple:
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
    return _model_cache[key]'''

# --------------------------------------------------------------------------
# Edit 2: insert batched transformers embedder before the fair-esm one
# --------------------------------------------------------------------------
EMBED_BATCHED = '''def _embed_sequences_transformers(seqs: list, model_name: str, device: str = "cpu") -> list:
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
    return out'''

OLD2 = '''def _embed_sequence_fairesm(seq: str, model_name: str) -> np.ndarray:'''
NEW2 = EMBED_BATCHED + "\n\n\n" + OLD2

# --------------------------------------------------------------------------
# Edit 3: insert _make_windows + _embed_sequences before _compute_delta
# --------------------------------------------------------------------------
MAKE_WINDOWS = '''def _make_windows(full_sequence: str, protein_pos: int, wt_aa: str, mut_aa: str):
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
    return wt_ctx, mut_ctx, idx - lo'''

EMBED_SEQUENCES = '''def _embed_sequences(seqs: list, model_name: str, conn: sqlite3.Connection,
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
    return [result.get(s) for s in seqs]'''

OLD3 = '''def _compute_delta(
    full_sequence: str,
    protein_pos: int,   # 1-based'''
NEW3 = MAKE_WINDOWS + "\n\n\n" + EMBED_SEQUENCES + "\n\n\n" + OLD3

# --------------------------------------------------------------------------
# Edit 4: device + batch_size on the connector
# --------------------------------------------------------------------------
OLD4 = '''    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        cache_path: Path | str | None = None,
        request_timeout: int = _REQUEST_TIMEOUT,
    ) -> None:
        self.model_name = model_name
        self.cache_path = Path(cache_path) if cache_path else _DEFAULT_CACHE
        self.request_timeout = request_timeout
        self._conn: Optional[sqlite3.Connection] = None'''

NEW4 = '''    def __init__(
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
        self._conn: Optional[sqlite3.Connection] = None'''

# --------------------------------------------------------------------------
# Edit 5: dispatch in annotate_dataframe (replace the inline per-variant loop)
# --------------------------------------------------------------------------
OLD5 = '''        seq_cache: dict[str, Optional[str]] = {}
        scores: dict[int, float] = {}

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
                logger.debug("ESM-2 delta failed for %s: %s", getattr(row, "gene_symbol", "?"), exc)'''

NEW5 = '''        if _BACKEND == "transformers":
            scores = self._score_batched(candidates)
        else:
            scores = self._score_per_variant(candidates)'''

# --------------------------------------------------------------------------
# Edit 6: append the two scorer methods at the end of the class
# --------------------------------------------------------------------------
OLD6 = '''        n_scored = sum(1 for v in scores.values() if v > 0.0)
        logger.info("ESM-2: %d/%d variants scored (>0).", n_scored, len(candidates))
        return df'''

SCORERS = '''    def _score_per_variant(self, candidates: pd.DataFrame) -> dict:
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
        return scores'''

NEW6 = OLD6 + "\n\n" + SCORERS

EDITS = [
    ("E1 device-aware loader", OLD1, NEW1),
    ("E2 batched transformers embedder", OLD2, NEW2),
    ("E3 _make_windows + _embed_sequences", OLD3, NEW3),
    ("E4 device/batch_size on connector", OLD4, NEW4),
    ("E5 batched dispatch in annotate", OLD5, NEW5),
    ("E6 scorer methods", OLD6, NEW6),
]

# A marker that exists only AFTER patching, for idempotence detection.
PATCHED_MARKER = "def _score_batched(self, candidates: pd.DataFrame) -> dict:"


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found. Run from the repo root.")
        return 2

    raw = TARGET.read_bytes()
    text = raw.decode("utf-8")

    if PATCHED_MARKER in text:
        print("Already patched (found _score_batched). No-op.")
        return 0

    # Count-guard every anchor BEFORE writing anything.
    for name, old, _new in EDITS:
        n = text.count(old)
        if n != 1:
            print(f"ABORT: anchor for {name} found {n} times (expected exactly 1). "
                  "No changes written.")
            return 1

    # Backup first.
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = TARGET.with_suffix(TARGET.suffix + f".bak_{stamp}")
    bak.write_bytes(raw)
    print(f"backup written: {bak}")

    patched = text
    for name, old, new in EDITS:
        patched = patched.replace(old, new, 1)
        print(f"applied: {name}")

    TARGET.write_bytes(patched.encode("utf-8"))

    # Post-checks.
    if PATCHED_MARKER not in patched:
        print("POST-CHECK FAIL: marker absent after patch.")
        return 1
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as exc:
        print(f"POST-CHECK FAIL: py_compile error:\n{exc}")
        return 1

    print("POST-CHECK: PASS (markers present, py_compile clean)")
    print(f"\nReview with:  git diff -- {TARGET}")
    print("Then run the gate:  python -m pytest tests/unit/test_esm2_batched_equivalence.py "
          "tests/unit/test_esm2_activation.py -q")
    return 0


if __name__ == "__main__":
    sys.exit(main())
