"""
Regression test for the train.py sequence/label alignment defect (PM11c/PM11d).

Proves:
 1. The OLD logic -- raw_df["fasta_seq"].iloc[:len(y_test)] -- misaligns
    sequences with labels after a gene-aware (shuffling) split.
 2. The NEW logic -- meta_test["fasta_seq"] -- stays aligned.

The synthetic fixture encodes the label INTO the sequence (first base A=benign,
T=pathogenic) so misalignment is detectable as a collapse in seq<->label
agreement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.model_selection import GroupShuffleSplit


def _make_label_encoding_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = (rng.uniform(0, 1, n) < 0.4).astype(int)
    first = np.where(label == 1, "T", "A")
    rest = np.array(["".join(rng.choice(list("ACGT"), 100)) for _ in range(n)])
    fasta_seq = np.char.add(first, rest)
    gene = rng.choice([f"GENE{i}" for i in range(12)], n)
    return pd.DataFrame(
        {
            "variant_id": [f"syn:{i}" for i in range(n)],
            "gene_symbol": gene,
            "fasta_seq": fasta_seq,
            "label": label,
        }
    )


def _seq_label_agreement(seqs: pd.Series, labels: pd.Series) -> float:
    decoded = (seqs.str[0] == "T").astype(int).reset_index(drop=True)
    lab = labels.reset_index(drop=True)
    return float((decoded == lab).mean())


def test_old_iloc_logic_misaligns_sequences():
    df = _make_label_encoding_frame()
    X = df[["gene_symbol"]].reset_index(drop=True)
    y = df["label"].reset_index(drop=True)
    groups = df["gene_symbol"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    y_test = y.iloc[test_idx].reset_index(drop=True)

    old_seq_test = df["fasta_seq"].iloc[: len(y_test)].reset_index(drop=True)
    agreement = _seq_label_agreement(old_seq_test, y_test)

    assert agreement < 0.85, (
        f"Expected misalignment (<0.85) from iloc slice, got {agreement:.3f}. "
        "If this fails, the split may not be permuting rows; strengthen the fixture."
    )


def test_meta_test_source_stays_aligned():
    df = _make_label_encoding_frame()
    X = df[["gene_symbol"]].reset_index(drop=True)
    y = df["label"].reset_index(drop=True)
    groups = df["gene_symbol"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    y_test = y.iloc[test_idx].reset_index(drop=True)

    meta_test = df.iloc[test_idx].reset_index(drop=True)
    new_seq_test = meta_test["fasta_seq"].reset_index(drop=True)
    agreement = _seq_label_agreement(new_seq_test, y_test)

    assert agreement == pytest.approx(1.0), (
        f"meta_test sequence source must stay perfectly aligned, got {agreement:.3f}."
    )