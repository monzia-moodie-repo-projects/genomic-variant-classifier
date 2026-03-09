"""
Priority 2: Ensemble Model Framework
======================================
Implements 8 base classifiers + 1 stacking meta-learner for
variant pathogenicity prediction.

Base classifiers:
  1.  Random Forest        (sklearn)
  2.  XGBoost              (xgboost)
  3.  LightGBM             (lightgbm)
  4.  SVM (RBF kernel)     (sklearn)
  5.  Logistic Regression  (sklearn)
  6.  Gradient Boosting    (sklearn)
  7.  1D-CNN               (TensorFlow/Keras)  ← sequence-based
  8.  Feedforward NN       (TensorFlow/Keras)  ← tabular features

Meta-learner:
  - Logistic Regression stacker trained on OOF predictions

Evaluation metrics per model:
  - AUROC, AUPRC, F1 (macro/weighted), MCC, Brier score, calibration curve
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, matthews_corrcoef,
    roc_auc_score, brier_score_loss
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
TABULAR_FEATURES = [
    # Allele / variant properties
    "allele_freq",          # population AF from gnomAD
    "ref_len",              # length of reference allele
    "alt_len",              # length of alternate allele
    "is_snv",               # bool: single nucleotide variant
    "is_indel",             # bool: insertion or deletion
    # Functional scores (pre-computed or from external tools)
    "cadd_phred",           # CADD PHRED score (higher = more deleterious)
    "sift_score",           # SIFT score (lower = more deleterious)
    "polyphen2_score",      # PolyPhen-2 HDIV score
    "revel_score",          # REVEL ensemble score
    "phylop_score",         # phyloP conservation score
    # Coding context
    "in_coding_region",     # bool
    "in_splice_site",       # bool: within 2bp of exon boundary
    "codon_position",       # 1, 2, or 3
    "is_missense",          # bool
    "is_nonsense",          # bool
    # Gene-level
    "gene_constraint_oe",   # gnomAD pLoF observed/expected ratio
    "num_pathogenic_in_gene",  # from ClinVar
    # Protein features (from UniProt)
    "in_active_site",       # bool: overlaps active site annotation
    "in_domain",            # bool: overlaps annotated domain
]

SEQUENCE_FEATURES = ["fasta_seq"]   # 101 bp context window, k-mer encoded


@dataclass
class EnsembleConfig:
    n_folds: int = 5
    random_state: int = 42
    calibrate: bool = True         # Platt scaling on all models
    class_weight: str = "balanced" # handles class imbalance
    n_jobs: int = -1
    model_dir: Path = Path("models/ensemble")

    def __post_init__(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive tabular features from canonical variant DataFrame.
    Input: canonical schema from database_connectors.py
    Output: feature matrix with columns = TABULAR_FEATURES
    """
    feats = pd.DataFrame(index=df.index)

    feats["allele_freq"] = df.get("allele_freq", 0.0).fillna(0.0)
    feats["ref_len"] = df.get("ref", "").str.len().fillna(1)
    feats["alt_len"] = df.get("alt", "").str.len().fillna(1)
    feats["is_snv"] = ((feats["ref_len"] == 1) & (feats["alt_len"] == 1)).astype(int)
    feats["is_indel"] = (feats["is_snv"] == 0).astype(int)

    # Consequence-based booleans
    consequence = df.get("consequence", "").fillna("")
    feats["is_missense"] = consequence.str.contains("missense", case=False).astype(int)
    feats["is_nonsense"] = consequence.str.contains("stop_gained|nonsense", case=False).astype(int)
    feats["in_splice_site"] = consequence.str.contains("splice", case=False).astype(int)
    feats["in_coding_region"] = consequence.str.contains(
        "missense|synonymous|stop|frameshift|inframe", case=False
    ).astype(int)
    feats["codon_position"] = 0  # placeholder; populated by VEP annotation

    # Scores — fill with population median if missing
    for score_col, default in [
        ("cadd_phred", 15.0), ("sift_score", 0.5),
        ("polyphen2_score", 0.5), ("revel_score", 0.5), ("phylop_score", 0.0),
    ]:
        feats[score_col] = df.get(score_col, default).fillna(default)

    # Gene constraint (gnomAD pLoF oe ratio)
    feats["gene_constraint_oe"] = df.get("gene_constraint_oe", 1.0).fillna(1.0)
    feats["num_pathogenic_in_gene"] = df.get("num_pathogenic_in_gene", 0).fillna(0)

    # Protein annotations
    feats["in_active_site"] = df.get("in_active_site", 0).fillna(0).astype(int)
    feats["in_domain"] = df.get("in_domain", 0).fillna(0).astype(int)

    return feats[TABULAR_FEATURES]


def encode_sequence(seq: str, k: int = 3, window: int = 101) -> np.ndarray:
    """
    Encode a nucleotide sequence as a k-mer frequency vector.
    Also returns the raw one-hot array for CNN input.
    """
    BASES = "ACGT"
    base_map = {b: i for i, b in enumerate(BASES)}

    # Pad/trim to window size
    seq = seq.upper().replace("N", "A")[:window].ljust(window, "A")

    # One-hot encoding: shape (window, 4)
    one_hot = np.zeros((window, len(BASES)), dtype=np.float32)
    for i, nuc in enumerate(seq):
        if nuc in base_map:
            one_hot[i, base_map[nuc]] = 1.0

    return one_hot  # shape: (101, 4)


# ---------------------------------------------------------------------------
# Sklearn-compatible CNN wrapper
# ---------------------------------------------------------------------------
class CNN1DClassifier(BaseEstimator, ClassifierMixin):
    """
    1D Convolutional neural network for variant sequence classification.
    Wraps Keras for sklearn compatibility.
    Input: batch of one-hot arrays (n_samples, window, 4)
    """
    def __init__(
        self,
        window: int = 101,
        filters: int = 64,
        kernel_size: int = 7,
        dropout: float = 0.3,
        epochs: int = 30,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ):
        self.window = window
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model_ = None
        self.classes_ = np.array([0, 1])

    def _build_model(self):
        import tensorflow as tf
        from tensorflow.keras import layers, models

        tf.random.set_seed(self.random_state)
        inp = layers.Input(shape=(self.window, 4))
        x = layers.Conv1D(self.filters, self.kernel_size, activation="relu", padding="same")(inp)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(self.filters * 2, self.kernel_size, activation="relu", padding="same")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        out = layers.Dense(1, activation="sigmoid")(x)

        model = models.Model(inp, out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="binary_crossentropy",
            metrics=["AUC"]
        )
        return model

    def _encode_X(self, X):
        """X can be a pd.Series of sequences or a list of sequences."""
        if isinstance(X, pd.DataFrame) and "fasta_seq" in X.columns:
            seqs = X["fasta_seq"].fillna("A" * self.window)
        elif isinstance(X, pd.Series):
            seqs = X.fillna("A" * self.window)
        else:
            seqs = pd.Series(X).fillna("A" * self.window)
        return np.stack([encode_sequence(s, window=self.window) for s in seqs])

    def fit(self, X, y):
        import tensorflow as tf
        self.model_ = self._build_model()
        X_enc = self._encode_X(X)
        self.model_.fit(
            X_enc, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ],
            verbose=0,
        )
        return self

    def predict_proba(self, X):
        X_enc = self._encode_X(X)
        proba = self.model_.predict(X_enc, verbose=0).flatten()
        return np.column_stack([1 - proba, proba])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# ---------------------------------------------------------------------------
# Feedforward NN for tabular features
# ---------------------------------------------------------------------------
class TabularNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_dims=(256, 128, 64),
        dropout: float = 0.3,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model_ = None
        self.scaler_ = StandardScaler()
        self.classes_ = np.array([0, 1])

    def _build_model(self, input_dim: int):
        import tensorflow as tf
        from tensorflow.keras import layers, models, regularizers

        tf.random.set_seed(self.random_state)
        inp = layers.Input(shape=(input_dim,))
        x = inp
        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation="relu",
                             kernel_regularizer=regularizers.l2(1e-4))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        model = models.Model(inp, out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="binary_crossentropy",
            metrics=["AUC"]
        )
        return model

    def fit(self, X, y):
        import tensorflow as tf
        X_scaled = self.scaler_.fit_transform(X)
        self.model_ = self._build_model(X_scaled.shape[1])
        self.model_.fit(
            X_scaled, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
            ],
            verbose=0,
        )
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler_.transform(X)
        proba = self.model_.predict(X_scaled, verbose=0).flatten()
        return np.column_stack([1 - proba, proba])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# ---------------------------------------------------------------------------
# Ensemble orchestrator
# ---------------------------------------------------------------------------
class VariantEnsemble:
    """
    Orchestrates training and evaluation of all base classifiers
    plus a stacking meta-learner.
    """
    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self._build_estimators()

    def _build_estimators(self):
        cfg = self.config
        self.base_estimators = {
            "random_forest": RandomForestClassifier(
                n_estimators=500, max_features="sqrt",
                class_weight=cfg.class_weight,
                n_jobs=cfg.n_jobs, random_state=cfg.random_state,
            ),
            "xgboost": xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=10,  # approximate imbalance ratio
                eval_metric="auc", random_state=cfg.random_state,
                n_jobs=cfg.n_jobs, verbosity=0,
            ),
            "lightgbm": lgb.LGBMClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                class_weight=cfg.class_weight,
                n_jobs=cfg.n_jobs, random_state=cfg.random_state,
                verbose=-1,
            ),
            "svm": CalibratedClassifierCV(
                SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
                cv=3,
            ),
            "logistic_regression": LogisticRegression(
                C=0.1, max_iter=1000, class_weight=cfg.class_weight,
                n_jobs=cfg.n_jobs, random_state=cfg.random_state,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=cfg.random_state,
            ),
            "tabular_nn": TabularNNClassifier(random_state=cfg.random_state),
            "cnn_1d": CNN1DClassifier(random_state=cfg.random_state),
        }

        self.meta_learner = LogisticRegression(
            C=0.1, max_iter=1000, random_state=cfg.random_state
        )

    def fit(self, X_tab: pd.DataFrame, X_seq: pd.Series, y: pd.Series):
        """
        Two-phase training:
          Phase 1: Train all base models with cross-validation → OOF predictions
          Phase 2: Train meta-learner on OOF predictions
        """
        logger.info(f"Training ensemble on {len(y):,} samples, {y.sum():,} pathogenic.")
        cv = StratifiedKFold(n_splits=self.config.n_folds, shuffle=True,
                             random_state=self.config.random_state)

        oof_preds = np.zeros((len(y), len(self.base_estimators)))
        self.trained_models_ = {}

        for model_idx, (name, model) in enumerate(self.base_estimators.items()):
            logger.info(f"  Training {name}...")

            if name == "cnn_1d":
                X_input = X_seq
            else:
                X_input = X_tab.values

            oof = cross_val_predict(
                model, X_input, y.values,
                cv=cv, method="predict_proba", n_jobs=1
            )[:, 1]
            oof_preds[:, model_idx] = oof

            # Fit final model on full data
            model.fit(X_input, y.values)
            self.trained_models_[name] = model
            logger.info(f"  {name} OOF AUROC: {roc_auc_score(y, oof):.4f}")

        # Meta-learner on OOF
        logger.info("Training meta-learner...")
        self.meta_learner.fit(oof_preds, y.values)
        self.feature_names_ = list(self.base_estimators.keys())
        return self

    def predict_proba(self, X_tab: pd.DataFrame, X_seq: pd.Series) -> np.ndarray:
        base_preds = np.zeros((len(X_tab), len(self.trained_models_)))
        for i, (name, model) in enumerate(self.trained_models_.items()):
            X_input = X_seq if name == "cnn_1d" else X_tab.values
            base_preds[:, i] = model.predict_proba(X_input)[:, 1]
        return self.meta_learner.predict_proba(base_preds)

    def predict(self, X_tab, X_seq):
        return (self.predict_proba(X_tab, X_seq)[:, 1] > 0.5).astype(int)

    def evaluate(self, X_tab, X_seq, y) -> dict:
        """Full evaluation of all models + ensemble."""
        results = {}

        for name, model in self.trained_models_.items():
            X_input = X_seq if name == "cnn_1d" else X_tab.values
            proba = model.predict_proba(X_input)[:, 1]
            preds = (proba > 0.5).astype(int)
            results[name] = {
                "auroc": roc_auc_score(y, proba),
                "auprc": average_precision_score(y, proba),
                "f1_macro": f1_score(y, preds, average="macro"),
                "f1_weighted": f1_score(y, preds, average="weighted"),
                "mcc": matthews_corrcoef(y, preds),
                "brier": brier_score_loss(y, proba),
            }

        # Ensemble
        ens_proba = self.predict_proba(X_tab, X_seq)[:, 1]
        ens_preds = (ens_proba > 0.5).astype(int)
        results["ENSEMBLE_STACKER"] = {
            "auroc": roc_auc_score(y, ens_proba),
            "auprc": average_precision_score(y, ens_proba),
            "f1_macro": f1_score(y, ens_preds, average="macro"),
            "f1_weighted": f1_score(y, ens_preds, average="weighted"),
            "mcc": matthews_corrcoef(y, ens_preds),
            "brier": brier_score_loss(y, ens_proba),
        }

        df_results = pd.DataFrame(results).T.round(4)
        df_results = df_results.sort_values("auroc", ascending=False)
        logger.info("\n" + df_results.to_string())
        return df_results

    def save(self, path: Optional[Path] = None):
        import joblib
        path = path or self.config.model_dir / "ensemble.joblib"
        joblib.dump(self, path)
        logger.info(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "VariantEnsemble":
        import joblib
        return joblib.load(path)
