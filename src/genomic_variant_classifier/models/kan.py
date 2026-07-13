"""
src/genomic_variant_classifier/models/kan.py
=================
KAN (Kolmogorov-Arnold Network) classifier -- Run 11 Integration 2.

Run 12 change: imodelsx (efficient-kan based) is the primary backend; pykan is a fallback.
FastKAN is 3.7x faster in benchmarks with identical API shape.
pykan caused Run 10a KAN runaway (19h 22m, $14.72 wasted).

Backend priority (Run 11):
1. fastkan  -- github.com/ZiyaoLi/fast-kan (MIT license, 3.7x faster)
   Install: pip install fastkan
2. pykan   -- original MIT CSAIL implementation (fallback)
   Install: pip install pykan
3. efficient-kan -- faster GPU-friendly re-implementation (fallback)
   Install: pip install efficient-kan
4. MLP fallback -- sklearn MLPClassifier; no splines but same interface

Reference: Liu et al., 2024 -- "KAN: Kolmogorov-Arnold Networks"
           https://arxiv.org/abs/2404.19756

Usage:
    from genomic_variant_classifier.models.kan import KANClassifier

    clf = KANClassifier(hidden_sizes=[64, 32], max_iter=200)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection -- Run 12: imodelsx (efficient-kan based) first
# ---------------------------------------------------------------------------
_KAN_BACKEND: Optional[str] = None
_ImodelsxKAN = None  # type: ignore[assignment]

def _repair_imodelsx_kan_bare_names() -> str:
    """Make imodelsx 1.0.13's KANClassifier.fit() actually runnable. IN-PROCESS.

    THE UPSTREAM BUG -- it is in __init__, not in fit
    -------------------------------------------------
    imodelsx 1.0.13 (the LATEST release; there is no 1.0.14) declares:

        def __init__(self, ..., test_size=0.2, random_state=42, shuffle=True, ...):
            self.hidden_layer_sizes = ...
            self.device = device
            self.regularize_activation = ...
            self.regularize_entropy = ...
            self.regularize_ridge = ...
            self.kwargs = kwargs
            # test_size / random_state / shuffle are ACCEPTED AND THROWN AWAY.

    Verified empirically 2026-07-13: after construction, `hasattr(m, "test_size")` is False,
    likewise `random_state` and `shuffle`. Every symptom below follows from that one defect.

    `fit` then does:

        X_train, X_tune, y_train, y_tune = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

    -- BARE NAMES. Not parameters of `fit`, not locals. Python resolves them as MODULE
    GLOBALS of `imodelsx.kan.kan_sklearn`, where they do not exist:

        NameError: name 'test_size' is not defined

    TWO SOURCE FORMS EXIST IN THE WILD, AND THEY FAIL DIFFERENTLY
    -------------------------------------------------------------
    Since 2026-05 the launch scripts (`launch_run11_vm.sh`, `launch_run16_vm.sh`,
    `launch_run16.py`) have run a `sed -i` over the INSTALLED site-packages file:

        sed -i 's/test_size=test_size/test_size=self.test_size/g' "$IMODELSX_KAN"

    So a given machine holds one of two different source forms:

        form                       fit() reads      raises            repaired by
        -------------------------  ---------------  ----------------  -------------------
        PRISTINE (PyPI 1.0.13)     bare test_size   NameError         module globals
        SED-PATCHED (dev laptop,   self.test_size   AttributeError    INSTANCE ATTRIBUTES
          Run 11 / Run 16 hosts)                    (__init__ never
                                                     set them)

    Which means the `sed` and the instance-attribute assignments in `_fit_imodelsx` are TWO
    HALVES OF ONE MECHANISM: the `sed` redirects the lookup onto `self`, and `_fit_imodelsx`
    puts the value on `self` because `__init__` refused to. NEITHER WORKS ALONE. The
    2026-05-28 KAN audit's note that the bug was "handled twice" was CORRECT.

    (On 2026-07-13 those instance-attribute lines were briefly deleted here as "dead code",
    on the reasoning that a NameError cannot be fixed by setting an attribute. That
    reasoning was drawn from the __init__ SIGNATURE without reading its BODY. It broke the
    local path instantly. Recorded so the mistake is not repeated: the signature says
    test_size is a parameter; only the body says whether it is ever stored.)

    WHERE THE SED IS NOT
    --------------------
    It was NEVER applied in Continuous Integration, NEVER in Docker, and -- critically -- it
    is NOT in `scripts/vm_bootstrap_run.sh`, the RUN 17 path.

    CONSEQUENCE, MEASURED 2026-07-13
    --------------------------------
    KAN raised NameError in EVERY Continuous Integration run. The old bare `except
    Exception` in `VariantEnsemble.fit` swallowed it, set KAN's out-of-fold column to 0.5,
    and `continue`d past `model.fit()`. **The ensemble silently trained TWELVE models
    instead of thirteen and reported normal metrics.** It surfaced only once that handler
    was made to fail loud (commit 7d42409) -- and it surfaced as a RED build, one day
    before Run 17.

    Run 17's own pre-flight would not have caught it: `vm_bootstrap_run.sh` checks that
    `imodelsx` and `KANClassifier` **import**. The bug is in `fit()`. Importing succeeds.
    Every gate would have gone green and the run would have published a 12-model algorithm
    comparison with KAN silently absent -- in a project whose stated first-class goal is
    comparing exactly these algorithms.

    THE FIX -- supply BOTH bindings, in-process, in our own code
    ------------------------------------------------------------
    `_fit_imodelsx` sets the INSTANCE ATTRIBUTES (which repairs the sed-patched form), and
    this function injects the MODULE GLOBALS (which repairs the pristine form). Together
    they are correct on either installed form, with no detection, no `sed`, no writing into
    `site-packages`, and no environment-dependent behaviour in a scientific pipeline.

    Injecting the module globals is not a hack layered on a hack: because the broken names
    resolve as module globals, that IS the binding Python is looking for -- exactly it, and
    nothing else. The function is guarded (it acts only if the names are genuinely absent)
    and idempotent (safe across repeated imports).

    Per-estimator `random_state` is preserved on both forms: `_fit_imodelsx` re-binds it
    immediately before each fit. The out-of-fold path is strictly sequential
    (`cross_val_predict(..., n_jobs=1)`; `_leakfree_oof`'s fold loop), so there is no
    cross-fit race. Asserted by `tests/unit/test_kan_actually_fits.py`.

    THE DIVERGENCE ITSELF IS A DEFECT, AND IS BEING CLOSED SEPARATELY
    ----------------------------------------------------------------
    That two source forms exist at all is the deeper problem: the developer's `.venv312`
    holds a MUTATED `site-packages`, so local tests have been exercising a code path no
    clean machine has, and "it passes on my machine" has been load-bearing since May. The
    `sed` is being removed from the launch scripts and the local install restored to
    pristine, so that every environment converges on ONE form. This function keeps both
    bindings regardless -- a stale VM image with a patched file must not silently lose KAN.

    Returns a short status string for logging: "repaired", "already-sane", or "absent".
    """
    try:
        import imodelsx.kan.kan_sklearn as _ks  # type: ignore[import]
    except Exception:  # pragma: no cover - imodelsx not installed
        return "absent"

    # Guard: only act if the bare names are genuinely missing from the module namespace.
    # If a future imodelsx fixes fit() to read self.test_size, or a launch script's sed has
    # already rewritten the file, these injections are harmless -- but we still report which.
    needed = {"test_size": 0.2, "random_state": 42, "shuffle": True}
    missing = [n for n in needed if not hasattr(_ks, n)]
    if not missing:
        return "already-sane"

    for name in missing:
        setattr(_ks, name, needed[name])
    logger.warning(
        "imodelsx 1.0.13 KAN repair applied IN-PROCESS: injected module globals %s into "
        "%s. Upstream KANClassifier.fit() references these as BARE NAMES and raises "
        "NameError without them. See _repair_imodelsx_kan_bare_names() for the full "
        "history -- this replaces a sed-into-site-packages that ran on some machines and "
        "not others, and silently cost the ensemble its KAN model in every Continuous "
        "Integration run.",
        missing,
        getattr(_ks, "__file__", "<unknown>"),
    )
    return "repaired"


_IMODELSX_KAN_REPAIR: Optional[str] = None

try:
    from imodelsx import KANClassifier as _ImodelsxKAN  # type: ignore[import]
    _IMODELSX_KAN_REPAIR = _repair_imodelsx_kan_bare_names()
    _KAN_BACKEND = "imodelsx"
    logger.debug(
        "KAN backend: imodelsx (efficient-kan, Run 12 primary); repair=%s",
        _IMODELSX_KAN_REPAIR,
    )
except ImportError:
    pass

if _KAN_BACKEND is None:
    try:
        from kan import KAN as _PyKAN  # type: ignore[import]
        _KAN_BACKEND = "pykan"
        logger.debug("KAN backend: pykan (fallback)")
    except ImportError:
        pass

if _KAN_BACKEND is None:
    logger.info(
        "KAN: no KAN backend installed (tried imodelsx, pykan). "
        "Falling back to sklearn MLPClassifier. "
        "Install: pip install imodelsx"
    )


# ---------------------------------------------------------------------------
# KANClassifier
# ---------------------------------------------------------------------------
class KANClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible KAN classifier for tabular genomic data.

    Run 11 changes:
    - imodelsx (efficient-kan based) is the primary backend (pykan and efficient-kan are fallbacks)
    - max_fit_samples=100_000 maintained (Run 10a OOM safeguard)
    - Module-level class definition for pickle safety (Run 10b lesson)

    Parameters
    ----------
    hidden_sizes : list[int]
        Hidden layer widths, e.g. [64, 32].
    spline_degree : int
        Polynomial degree of the B-spline basis functions (default: 3).
    grid_size : int
        Number of spline grid intervals per edge (default: 5).
    max_iter : int
        Maximum training epochs (default: 200).
    learning_rate : float
        Adam learning rate (default: 1e-3).
    batch_size : int
        Mini-batch size (default: 256).
    random_state : int
    scale : bool
        Standardise features before fitting (default: True).
    max_fit_samples : int
        Cap training set size to avoid OOM (default: 100_000).
        Run 10a: pykan allocated 17.9 GB at 1.2M samples.
        Commit 2389ee2 added this safeguard.
    """

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        spline_degree: int = 3,
        grid_size: int = 5,
        max_iter: int = 200,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        random_state: int = 42,
        scale: bool = True,
        max_fit_samples: int = 100_000,
    ) -> None:
        self.hidden_sizes = hidden_sizes or [64, 32]
        self.spline_degree = spline_degree
        self.grid_size = grid_size
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.random_state = random_state
        self.scale = scale
        self.max_fit_samples = max_fit_samples

    def _subsample_if_needed(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Stratified subsample to max_fit_samples. Returns (X, y) possibly smaller."""
        if X.shape[0] <= self.max_fit_samples:
            return X, y
        rng = np.random.default_rng(self.random_state)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_pos = int(self.max_fit_samples * len(pos_idx) / len(y))
        n_neg = self.max_fit_samples - n_pos
        chosen = np.concatenate([
            rng.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False),
            rng.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False),
        ])
        rng.shuffle(chosen)
        logger.info(
            "KAN: subsampled %d -> %d (max_fit_samples=%d, preserving class balance)",
            X.shape[0], len(chosen), self.max_fit_samples,
        )
        return X[chosen], y[chosen]

    # ------------------------------------------------------------------
    # Backend-specific fit
    # ------------------------------------------------------------------
    def _fit_imodelsx(self, X: np.ndarray, y: np.ndarray) -> None:
        """imodelsx backend -- Run 12 primary. Uses efficient-kan internally."""
        self._backend_used = "imodelsx"

        # Subsample gate (inherited from Run 10a safeguard)
        if len(X) > self.max_fit_samples:
            logger.info(
                "KAN (imodelsx): subsampling %d -> %d for training",
                len(X), self.max_fit_samples,
            )
            from sklearn.utils import resample
            X, y = resample(
                X, y,
                n_samples=self.max_fit_samples,
                stratify=y,
                random_state=getattr(self, "random_state", 42),
            )

        # Map our params to imodelsx params
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._imodelsx_model = _ImodelsxKAN(
            hidden_layer_sizes=self.hidden_sizes,
            device=device,
            regularize_activation=1.0,
            regularize_entropy=1.0,
        )
        # ------------------------------------------------------------------
        # imodelsx 1.0.13 repair -- BOTH bindings. See _repair_imodelsx_kan_bare_names()
        # for the full history; the short version is that TWO different source forms of
        # kan_sklearn.py exist in the wild and they fail in DIFFERENT ways.
        #
        # The upstream defect is in __init__, which ACCEPTS test_size / random_state /
        # shuffle and then THROWS THEM AWAY (verified 2026-07-13: the body assigns only
        # hidden_layer_sizes, device, regularize_*, kwargs). Every downstream symptom
        # follows from that.
        #
        #   form                     fit() reads      fails with        fixed by
        #   -----------------------  ---------------  ----------------  -------------------
        #   pristine (PyPI 1.0.13)   bare test_size   NameError         module globals
        #   sed-patched launch/local self.test_size   AttributeError    INSTANCE ATTRIBUTES
        #
        # The instance attributes below are NOT redundant and are NOT dead code. On a
        # sed-patched install they are the ONLY thing that makes fit() work, because
        # __init__ never set them. (They were briefly deleted on 2026-07-13 in the mistaken
        # belief that they were inert -- that inference was made from the __init__ SIGNATURE
        # without reading its BODY, and it broke the local path immediately.)
        #
        # Setting BOTH bindings makes this correct on either installed form, with no
        # detection and no environment-dependent behaviour. Both are cheap. Both are tested
        # (tests/unit/test_kan_actually_fits.py).
        # ------------------------------------------------------------------
        self._imodelsx_model.test_size = 0.2
        self._imodelsx_model.random_state = getattr(self, "random_state", 42)
        self._imodelsx_model.shuffle = True

        if _IMODELSX_KAN_REPAIR == "repaired":
            import imodelsx.kan.kan_sklearn as _ks  # type: ignore[import]
            # Pristine form only: fit() resolves the bare names as MODULE globals, so the
            # per-estimator seed must be re-bound here or every KAN silently shares seed 42.
            # Safe because our out-of-fold path is strictly sequential
            # (cross_val_predict(n_jobs=1); _leakfree_oof's fold loop).
            _ks.random_state = getattr(self, "random_state", 42)

        self._imodelsx_model.fit(X, y)
        logger.info(
            "KAN (imodelsx/efficient-kan): trained on %d samples, device=%s",
            len(X), device,
        )

    def _fit_pykan(self, X: np.ndarray, y: np.ndarray) -> None:
        """pykan backend -- fallback."""
        import torch

        self._backend_used = "pykan"
        X, y = self._subsample_if_needed(X, y)

        n_features = X.shape[1]
        widths = [n_features] + list(self.hidden_sizes) + [1]

        self._kan = _PyKAN(
            width=widths,
            grid=self.grid_size,
            k=self.spline_degree,
            seed=self.random_state,
        )

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

        dataset = {
            "train_input": X_t,
            "train_label": y_t,
            "test_input": X_t,
            "test_label": y_t,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._kan.fit(
                dataset,
                opt="Adam",
                lr=self.learning_rate,
                steps=self.max_iter,
                loss_fn=torch.nn.BCEWithLogitsLoss(),
                metrics=None,
            )

    def _fit_efficient_kan(self, X: np.ndarray, y: np.ndarray) -> None:
        """efficient-kan backend -- fallback."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        self._backend_used = "efficient-kan"
        n_features = X.shape[1]
        widths = [n_features] + list(self.hidden_sizes) + [1]

        self._kan = _EfficientKAN(widths, grid_size=self.grid_size, spline_order=self.spline_degree)
        optimizer = optim.Adam(self._kan.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        self._kan.train()
        for _ in range(self.max_iter):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self._kan(xb), yb)
                loss.backward()
                optimizer.step()
        self._kan.eval()

    def _fit_mlp(self, X: np.ndarray, y: np.ndarray) -> None:
        """MLP fallback -- no KAN backend available."""
        self._backend_used = "mlp"
        self._mlp = MLPClassifier(
            hidden_layer_sizes=tuple(self.hidden_sizes),
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )
        self._mlp.fit(X, y)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "KANClassifier":
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        if self.scale:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None

        if _KAN_BACKEND == "imodelsx":
            self._fit_imodelsx(X, y)
        elif _KAN_BACKEND == "pykan":
            self._fit_pykan(X, y)
        else:
            self._fit_mlp(X, y)

        logger.info("KANClassifier fitted via %s backend.", getattr(self, "_backend_used", "mlp"))
        return self

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Return probabilities, shape (n_samples,)."""
        if self.scale and self.scaler_ is not None:
            X = self.scaler_.transform(X)

        backend = getattr(self, "_backend_used", "mlp")

        if backend == "mlp":
            return self._mlp.predict_proba(X)[:, 1]

        if backend == "imodelsx":
            return self._imodelsx_model.predict_proba(X)[:, 1]

        import torch

        self._kan.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            logits = self._kan(X_t).squeeze(-1).numpy()
        return 1.0 / (1.0 + np.exp(-logits))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "classes_")
        p = self._predict_raw(X)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def plot_edge_functions(self, **kwargs) -> None:
        """Visualise learned spline functions (pykan backend only)."""
        check_is_fitted(self, "classes_")
        if getattr(self, "_backend_used", "mlp") != "pykan":
            logger.warning("plot_edge_functions() requires pykan backend.")
            return
        self._kan.plot(**kwargs)
