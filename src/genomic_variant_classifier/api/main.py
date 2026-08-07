"""
src/genomic_variant_classifier/api/main.py
===============
FastAPI REST API for the Genomic Variant Pathogenicity Classifier.

Endpoints
---------
  GET  /health           Liveness + readiness check
  GET  /info             Model metadata and feature list
  GET  /metrics          Prometheus metrics
  GET  /gene/{symbol}    Gene-level features for request enrichment
  POST /predict          Classify a single variant
  POST /batch            Classify up to 1 000 variants

Usage
-----
  # Development
  uvicorn genomic_variant_classifier.api.main:app --reload --port 8000

  # Production (inside Docker)
  gunicorn genomic_variant_classifier.api.main:app -k uvicorn.workers.UvicornWorker \
      --bind 0.0.0.0:8000 --workers 2

Environment variables
---------------------
  MODEL_PATH          Path to serialised InferencePipeline artifact
                      (default: models/phase2_pipeline.joblib)
  GNOMAD_INDEX_PATH   Optional path to gnomAD parquet for live AF lookup
                      (default: None — callers must supply allele_freq)
  GENE_SUMMARY_PATH   Path to gene summary parquet
                      (default: data/processed/gene_summary.parquet)
  API_KEYS            Comma-separated list of valid X-API-Key tokens.
                      When empty or unset auth is disabled (dev mode).
  LOG_LEVEL           Python logging level (default: INFO)
  LOG_FORMAT          "json" for structured JSON output, "text" otherwise
                      (default: "json")
  DBSNP_INDEX_PATH    Optional path to dbSNP parquet (columns: rs_id, chrom,
                      pos, ref, alt) for /rsid/{rs_id} lookups
                      (default: data/processed/dbsnp_index.parquet)

Implementation notes
--------------------
* The model is loaded once at startup into a module-level ``_PIPELINE``
  singleton.  Concurrent requests share it read-only (joblib artifacts are
  thread-safe after load).
* Feature engineering is delegated to the InferencePipeline wrapper
  (``src/genomic_variant_classifier/api/pipeline.py``), which replicates the 64-feature logic from
  ``DataPrepPipeline._engineer_features`` without any I/O side-effects.
* Auth is HTTPBearer.  When VALID_API_KEYS is empty, all requests are
  allowed (development mode).  /health is always public.
* Rate limits: /predict 1000/min per key, /batch 100/min per key.
* /metrics is served by prometheus-fastapi-instrumentator.

Phase 7 additions:
  7.1 — API key authentication (X-API-Key header + API_KEYS env var)
  7.2 — Rate limiting (slowapi; /predict 1000/min, /batch 100/min)
  7.4 — Structured JSON logging (python-json-logger)
  7.5 — Prometheus /metrics endpoint
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from genomic_variant_classifier.api.auth import require_api_key
from genomic_variant_classifier.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    GeneSummaryResponse,
    HealthResponse,
    InfoResponse,
    PredictResponse,
    RsidLookupResponse,
    VariantPrediction,
    VariantRequest,
    score_to_classification,
)

# ---------------------------------------------------------------------------
# NOTE: Auth dependency is in src/genomic_variant_classifier/api/auth.py (X-API-Key header, API_KEYS env var)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Logging — structured JSON when LOG_FORMAT=json (default in Docker)
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
    fmt   = os.environ.get("LOG_FORMAT", "json").lower()

    if fmt == "json":
        try:
            from pythonjsonlogger.jsonlogger import JsonFormatter
            handler = logging.StreamHandler()
            handler.setFormatter(
                JsonFormatter(
                    "%(asctime)s %(name)s %(levelname)s %(message)s "
                    "%(pathname)s %(lineno)d"
                )
            )
            logging.root.handlers = []
            logging.root.addHandler(handler)
            logging.root.setLevel(level)
            return
        except ImportError:
            pass  # fall through to text formatter

    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (from environment)
# ---------------------------------------------------------------------------

MODEL_PATH: Path = Path(os.environ.get("MODEL_PATH", "models/phase2_pipeline.joblib"))
GNOMAD_INDEX_PATH: Optional[Path] = (
    Path(p) if (p := os.environ.get("GNOMAD_INDEX_PATH")) else None
)
GENE_SUMMARY_PATH: Optional[Path] = Path(
    os.environ.get("GENE_SUMMARY_PATH", "data/processed/gene_summary.parquet")
)
DBSNP_INDEX_PATH: Optional[Path] = Path(
    os.environ.get("DBSNP_INDEX_PATH", "data/processed/dbsnp_index.parquet")
)

# Filled at startup
_PIPELINE: Any = None
#: PROD-1 (2026-08-07). What this process is serving and what it may
#: claim about it. Computed ONCE in the lifespan and immutable after:
#: re-measuring the path per request would describe whatever bytes are
#: on disk now, which need not be the bytes that were deserialised.
_RUNTIME_MODEL_BINDING: Any = None
_GNOMAD_INDEX: Optional[pd.DataFrame] = None
_GENE_SUMMARY: Optional[pd.DataFrame] = None
_DBSNP_INDEX: Optional[pd.DataFrame] = None   # rs_id → chrom/pos/ref/alt
_START_TIME: float = time.monotonic()

# PROD-1 (2026-08-07). THE FIVE CONSTANTS THAT STOOD HERE ARE GONE.
#
# They were written in ae1853b on 2026-03-25 under the comment "update
# after each training run", and were never updated -- through Runs 9 to
# 16. `HOLDOUT_AUROC = 0.9847` fused a Run-8 sixty-four-feature figure
# with 154,404, the validation split size of the Runs 10-14 cohort: two
# measurements, two eras, one line. The same digits are Run 15's
# unseen-gene F1, so a reader reconciling them lands on a third
# quantity. `/info` published all of it regardless of which artifact was
# loaded, and four of the five were pinned by literal in the suite.
#
# Nothing hand-maintained replaces them. Model identity now comes only
# from `_RUNTIME_MODEL_BINDING`, which is derived from the digest of the
# bytes that were actually loaded.

#: The version of the HTTP contract and the serving software. NOT a
#: model fact. `PIPELINE_VERSION` was retired rather than narrowed,
#: because one symbol was serving as both the OpenAPI version and
#: prediction provenance, and that conflation is why "1.0.0" sat
#: unexamined for four and a half months.
API_VERSION = "2.0.0"

#: The committed deployment declaration. See deployments/README.md for
#: what a registry in version control can and cannot claim.
DEPLOYMENT_REGISTRY_PATH: Path = Path(
    os.environ.get("DEPLOYMENT_REGISTRY_PATH",
                   "deployments/registry.v1.json"))

from genomic_variant_classifier.api.attribution import (  # noqa: E402
    ArtifactChangedDuringLoadError,
    RuntimeModelBinding,
    load_pipeline_with_identity,
    resolve_runtime_binding,
)

# `/info` constructs this, and referencing it without importing it raised
# NameError on every call to that endpoint. Placed beside the attribution
# import rather than in the file's main import block so the anchor is
# text this project wrote and can verify, not text transcribed from a
# terminal -- which is how an anchor drifted earlier today.
from genomic_variant_classifier.api.schemas import (  # noqa: E402
    ModelAttributionResponse,
)


# ---------------------------------------------------------------------------
# Rate limiting (7.2)
# ---------------------------------------------------------------------------

try:
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    def _rate_key(request: Request) -> str:
        """Use the API key as rate-limit identity; fall back to IP."""
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            key = auth[7:].strip()
            if key:
                return key
        return get_remote_address(request)

    _limiter = Limiter(key_func=_rate_key)
    _SLOWAPI_AVAILABLE = True

except ImportError:
    _limiter = None  # type: ignore[assignment]
    _SLOWAPI_AVAILABLE = False


def _rate_limit(rate: str):
    """Apply slowapi rate limiting when available; no-op decorator otherwise."""
    if _SLOWAPI_AVAILABLE and _limiter is not None:
        return _limiter.limit(rate)
    def _noop(f):
        return f
    return _noop


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure logging, load model and auxiliary tables at startup."""
    global _PIPELINE, _GNOMAD_INDEX, _GENE_SUMMARY, _DBSNP_INDEX
    global _RUNTIME_MODEL_BINDING

    _configure_logging()

    # --- Load inference pipeline, and bind it to a declared identity ---
    #
    # PROD-1. The digest is taken BEFORE and AFTER the load and the two
    # are compared. A digest taken only before describes bytes that may
    # have been replaced during the load; one taken only after describes
    # bytes that may not be what was deserialised. Exactly two
    # measurements: registry resolution reuses the second rather than
    # reading a large artifact a third time.
    _artifact_identity = None
    if MODEL_PATH.exists():
        try:
            from genomic_variant_classifier.api.pipeline import InferencePipeline
            _PIPELINE, _artifact_identity = load_pipeline_with_identity(
                MODEL_PATH, InferencePipeline.load)
            logger.info("Loaded inference pipeline from %s (sha256 %s)",
                        MODEL_PATH, _artifact_identity.sha256[:16])
        except ArtifactChangedDuringLoadError:
            # Deliberately NOT swallowed into a generic warning. If the
            # bytes moved mid-load, no binding can describe the object.
            _PIPELINE = None
            _artifact_identity = None
            logger.exception(
                "Refusing to serve: %s changed while being loaded",
                MODEL_PATH)
        except Exception as exc:
            logger.error("Failed to load pipeline from %s: %s", MODEL_PATH, exc)
    else:
        logger.warning(
            "MODEL_PATH %s does not exist.  "
            "Run scripts/export_model.py to serialise the trained pipeline.",
            MODEL_PATH,
        )

    # --- Load gnomAD AF index (optional) ---
    if GNOMAD_INDEX_PATH and GNOMAD_INDEX_PATH.exists():
        try:
            _GNOMAD_INDEX = pd.read_parquet(
                GNOMAD_INDEX_PATH, columns=["variant_id", "allele_freq"]
            )
            _GNOMAD_INDEX = _GNOMAD_INDEX.set_index("variant_id")
            logger.info("Loaded gnomAD AF index: %d loci", len(_GNOMAD_INDEX))
        except Exception as exc:
            logger.warning("Could not load gnomAD index: %s", exc)

    # --- Load gene summary table ---
    if GENE_SUMMARY_PATH and GENE_SUMMARY_PATH.exists():
        try:
            _GENE_SUMMARY = pd.read_parquet(GENE_SUMMARY_PATH).set_index("gene_symbol")
            logger.info("Loaded gene summary: %d genes", len(_GENE_SUMMARY))
        except Exception as exc:
            logger.warning("Could not load gene summary: %s", exc)
    else:
        logger.warning(
            "GENE_SUMMARY_PATH %s not found -- gene features will use defaults "
            "when callers omit them.  Build with scripts/build_gene_summary.py.",
            GENE_SUMMARY_PATH,
        )

    # --- Load dbSNP rs-ID index (optional) ---
    if DBSNP_INDEX_PATH and DBSNP_INDEX_PATH.exists():
        try:
            _DBSNP_INDEX = pd.read_parquet(
                DBSNP_INDEX_PATH, columns=["rs_id", "chrom", "pos", "ref", "alt"]
            ).set_index("rs_id")
            logger.info("Loaded dbSNP index: %d rs-IDs", len(_DBSNP_INDEX))
        except Exception as exc:
            logger.warning("Could not load dbSNP index: %s", exc)
    else:
        logger.info(
            "DBSNP_INDEX_PATH %s not found -- /rsid lookups will return known=false. "
            "Build with: python scripts/build_dbsnp_index.py",
            DBSNP_INDEX_PATH,
        )

    _RUNTIME_MODEL_BINDING = resolve_runtime_binding(
        _PIPELINE, _artifact_identity, DEPLOYMENT_REGISTRY_PATH)
    logger.info(
        "Runtime model binding: resolution=%s alignment=%s "
        "roster=%s evidence=%s",
        _RUNTIME_MODEL_BINDING.resolution_status.value,
        _RUNTIME_MODEL_BINDING.deployment_alignment.value,
        _RUNTIME_MODEL_BINDING.roster_alignment.value,
        _RUNTIME_MODEL_BINDING.evaluation_applicability.value)
    if not _RUNTIME_MODEL_BINDING.is_ready:
        logger.warning(
            "NOT READY for inference traffic: %s",
            _RUNTIME_MODEL_BINDING.detail or "the loaded model cannot be "
            "attributed to a declared production record")

    yield  # app is running

    logger.info("API shutting down.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Genomic Variant Pathogenicity Classifier",
    version=API_VERSION,
    # DESC-1 (2026-08-07). DEPLOYMENT-INDEPENDENT BY DESIGN. This string
    # is built at module-import time, before any artifact is loaded, so
    # any model claim in it is a claim about nothing in particular. The
    # previous text asserted a holdout AUROC and a 154,404-row cohort,
    # both stale, and the OpenAPI document is application documentation
    # rather than deployment telemetry. Model facts are structured, and
    # live at /info.
    description=(
        "Application programming interface for genomic-variant pathogenicity "
        "inference. Model identity, deployment attribution, feature "
        "provenance and roster alignment are available from /info."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Attach rate limiter state and exception handler
if _SLOWAPI_AVAILABLE:
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware

    app.state.limiter = _limiter
    app.add_middleware(SlowAPIMiddleware)

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": f"Rate limit exceeded: {exc.detail}"},
        )

# Prometheus metrics (7.5)
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator(
        should_group_status_codes=True,
        excluded_handlers=["/metrics", "/health"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
except ImportError:
    pass  # prometheus-fastapi-instrumentator not installed; /metrics silently absent


# ---------------------------------------------------------------------------
# Request logging middleware (7.4)
# ---------------------------------------------------------------------------

@app.middleware("http")
async def _log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    latency_ms = round((time.monotonic() - start) * 1000, 1)
    logger.info(
        "request",
        extra={
            "method":     request.method,
            "path":       request.url.path,
            "status":     response.status_code,
            "latency_ms": latency_ms,
        },
    )
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lookup_rsid(rs_id: str) -> Optional[dict]:
    """Return chrom/pos/ref/alt for an rs-ID, or None if not in the dbSNP index."""
    if _DBSNP_INDEX is None:
        return None
    # Normalise: ensure lowercase "rs" prefix and strip whitespace
    key = rs_id.strip().lower()
    if not key.startswith("rs"):
        key = "rs" + key
    try:
        row = _DBSNP_INDEX.loc[key]
        return {
            "chrom": str(row["chrom"]),
            "pos":   int(row["pos"]),
            "ref":   str(row["ref"]),
            "alt":   str(row["alt"]),
        }
    except KeyError:
        return None


def _lookup_gene_count(gene_symbol: str) -> Optional[int]:
    if _GENE_SUMMARY is None or not gene_symbol:
        return None
    try:
        return int(_GENE_SUMMARY.loc[gene_symbol, "n_pathogenic_in_gene"])
    except KeyError:
        return 0


def _lookup_gnomad_af(variant_id: str) -> Optional[float]:
    if _GNOMAD_INDEX is None:
        return None
    try:
        return float(_GNOMAD_INDEX.loc[variant_id, "allele_freq"])
    except KeyError:
        return 0.0
    except Exception:
        return None


def _variant_to_row(req: VariantRequest) -> dict:
    variant_id = f"{req.chrom}:{req.pos}:{req.ref}:{req.alt}"

    af = req.allele_freq
    if af is None:
        af = _lookup_gnomad_af(variant_id)
    if af is None:
        af = 0.0

    return {
        "variant_id":             variant_id,
        "chrom":                  req.chrom,
        "pos":                    req.pos,
        "ref":                    req.ref,
        "alt":                    req.alt,
        "allele_freq":            af,
        "consequence":            req.consequence or "",
        "gene_symbol":            req.gene_symbol or "",
        "cadd_phred":             req.cadd_phred,
        "sift_score":             req.sift_score,
        "polyphen2_score":        req.polyphen2_score,
        "revel_score":            req.revel_score,
        "phylop_score":           req.phylop_score,
        "gerp_score":             req.gerp_score,
        "alphamissense_score":    req.alphamissense_score,
        "splice_ai_score":            req.splice_ai_score,
        "eve_score":                  req.eve_score,
        "codon_position":             req.codon_position,
        "dbsnp_af":                   req.dbsnp_af,
        "omim_n_diseases":            req.omim_n_diseases,
        "omim_n_diseases_molecular":  req.omim_n_diseases_molecular,
        "omim_is_autosomal_dominant": req.omim_is_autosomal_dominant,
        "clingen_validity_score":     req.clingen_validity_score,
        "hgmd_is_disease_mutation":   req.hgmd_is_disease_mutation,
        "hgmd_n_reports":             req.hgmd_n_reports,
        "gene_constraint_oe":     req.gene_constraint_oe,
        "n_pathogenic_in_gene":   (
            req.n_pathogenic_in_gene
            if req.n_pathogenic_in_gene is not None
            else _lookup_gene_count(req.gene_symbol or "")
        ),
        "has_uniprot_annotation": req.has_uniprot_annotation or 0,
        "n_known_pathogenic_protein_variants": req.n_known_pathogenic_protein_variants or 0,
    }


def _make_prediction(row: dict) -> VariantPrediction:
    result = _PIPELINE.predict_single(row)
    score  = float(np.clip(result["pathogenicity_score"], 0.0, 1.0))
    classification, confidence = score_to_classification(score)

    top_features: Optional[dict[str, float]] = None
    if hasattr(_PIPELINE, "feature_importances"):
        try:
            top_features = _PIPELINE.feature_importances(pd.DataFrame([row]), top_n=5)
        except Exception:
            pass

    return VariantPrediction(
        variant_id          = row["variant_id"],
        pathogenicity_score = round(score, 4),
        classification      = classification,
        confidence          = confidence,
        top_features        = top_features,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and readiness check",
    tags=["ops"],
)
async def health() -> HealthResponse:
    # /health is intentionally unauthenticated for load-balancer probes.
    #
    # PROD-1. LIVENESS and READINESS are separated. A process that loaded
    # bytes it cannot identify is ALIVE and NOT READY: an orchestrator
    # should stop routing inference traffic to it, not restart it. For a
    # clinical-facing service, "I loaded a model but cannot establish
    # what it is" must not be operationally green. HEALTHSEM-1 tracks
    # splitting this into /health/live and /health/ready.
    binding = _RUNTIME_MODEL_BINDING
    ready = binding is not None and binding.is_ready
    return HealthResponse(
        status              = "ok" if ready else "degraded",
        live                = True,
        ready               = ready,
        model_loaded        = _PIPELINE is not None,
        model_attributed    = ready,
        gnomad_index_loaded = _GNOMAD_INDEX is not None,
        gene_counts_loaded  = _GENE_SUMMARY is not None,
        uptime_seconds      = round(time.monotonic() - _START_TIME, 1),
    )


@app.get(
    "/info",
    response_model=InfoResponse,
    summary="Model metadata, version, and feature list",
    tags=["ops"],
)
async def info(_key: str = Depends(require_api_key)) -> InfoResponse:
    feature_names: list[str] = []
    n_features = 0

    if _PIPELINE is not None and hasattr(_PIPELINE, "metadata"):
        feature_names = list(_PIPELINE.metadata.feature_names)
        n_features    = _PIPELINE.metadata.n_features

    # PIPEMETA-1. `_PIPELINE.metadata.val_auroc` is deliberately NOT read
    # here and must never be. It is an unqualified scalar embedded in the
    # artifact format, with no population and no protocol beside it --
    # exactly the shape of the constant this endpoint just stopped
    # publishing. A fallback to it would recreate PROD-1 on purpose.
    binding = _RUNTIME_MODEL_BINDING or RuntimeModelBinding.no_model_loaded(
        detail="the application lifespan has not run")

    return InfoResponse(
        api_version               = API_VERSION,
        model_loaded              = _PIPELINE is not None,
        attribution               = ModelAttributionResponse(
            resolution_status        = binding.resolution_status.value,
            deployment_alignment     = binding.deployment_alignment.value,
            roster_alignment         = binding.roster_alignment.value,
            evaluation_applicability = (
                binding.evaluation_applicability.value),
            record_id                = binding.record_id,
            model_version            = binding.model_version,
            artifact_sha256          = binding.artifact_sha256,
            registry_stage           = (
                binding.registry_stage.value
                if binding.registry_stage is not None else None),
            registered_model_roster  = (
                list(binding.registered_model_roster)
                if binding.registered_model_roster is not None else None),
            served_model_roster      = list(binding.served_model_roster),
            served_roster_fingerprint = binding.served_roster_fingerprint,
            detail                   = binding.detail,
        ),
        n_features                = n_features,
        feature_names             = feature_names,
        phase2_features_remaining = [],
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify a single genomic variant",
    tags=["inference"],
)
@_rate_limit("1000/minute")
async def predict(
    request: Request,
    body: VariantRequest,
    _key: str = Depends(require_api_key),
) -> PredictResponse:
    if _PIPELINE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model not loaded.  "
                "Check MODEL_PATH and run scripts/export_model.py."
            ),
        )
    try:
        row  = _variant_to_row(body)
        pred = _make_prediction(row)
    except Exception as exc:
        logger.exception("Prediction failed for %s", body)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {exc}",
        ) from exc

    _binding = _RUNTIME_MODEL_BINDING
    return PredictResponse(
        prediction      = pred,
        model_record_id = _binding.record_id if _binding else None,
        model_version   = _binding.model_version if _binding else None,
    )


@app.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="Classify up to 1 000 variants",
    tags=["inference"],
)
@_rate_limit("100/minute")
async def batch_predict(
    request: Request,
    body: BatchPredictRequest,
    _key: str = Depends(require_api_key),
) -> BatchPredictResponse:
    if _PIPELINE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )
    try:
        rows        = [_variant_to_row(v) for v in body.variants]
        raw_results = _PIPELINE.predict_batch(rows)

        predictions: list[VariantPrediction] = []
        for row, result in zip(rows, raw_results):
            score          = float(np.clip(result["pathogenicity_score"], 0.0, 1.0))
            classification, confidence = score_to_classification(score)
            predictions.append(VariantPrediction(
                variant_id          = row["variant_id"],
                pathogenicity_score = round(score, 4),
                classification      = classification,
                confidence          = confidence,
            ))

    except Exception as exc:
        logger.exception("Batch prediction failed (%d variants)", len(body.variants))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {exc}",
        ) from exc

    classes = [p.classification for p in predictions]
    _binding = _RUNTIME_MODEL_BINDING
    return BatchPredictResponse(
        predictions     = predictions,
        n_pathogenic    = sum(1 for c in classes if "Pathogenic" in c),
        n_benign        = sum(1 for c in classes if "Benign" in c),
        n_uncertain     = sum(1 for c in classes if "Uncertain" in c),
        model_record_id = _binding.record_id if _binding else None,
        model_version   = _binding.model_version if _binding else None,
    )


@app.get(
    "/gene/{gene_symbol}",
    response_model=GeneSummaryResponse,
    summary="Gene-level features for request enrichment",
    tags=["reference"],
)
async def gene_summary(
    gene_symbol: str,
    _key: str = Depends(require_api_key),
) -> GeneSummaryResponse:
    """
    Return the three gene-level features used by the model for a given HGNC symbol.

    Callers can use this to auto-enrich /predict or /batch requests rather than
    looking up counts themselves.
    """
    if _GENE_SUMMARY is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gene summary table not loaded.  Check GENE_SUMMARY_PATH.",
        )
    try:
        row = _GENE_SUMMARY.loc[gene_symbol]
    except KeyError:
        return GeneSummaryResponse(
            gene_symbol            = gene_symbol,
            n_pathogenic_in_gene   = 0,
            gene_constraint_oe     = None,
            has_uniprot_annotation = 0,
        )

    oe = row["gene_constraint_oe"]
    return GeneSummaryResponse(
        gene_symbol            = gene_symbol,
        n_pathogenic_in_gene   = int(row["n_pathogenic_in_gene"]),
        gene_constraint_oe     = None if pd.isna(oe) else float(oe),
        has_uniprot_annotation = int(row["has_uniprot_annotation"]),
    )


@app.get(
    "/rsid/{rs_id}",
    response_model=RsidLookupResponse,
    summary="Resolve an rs-ID to a genomic locus and classify",
    tags=["lookup"],
)
async def rsid_lookup(
    rs_id: str,
    _key: str = Depends(require_api_key),
) -> RsidLookupResponse:
    """
    Resolve an NCBI dbSNP rs-ID to chrom:pos:ref:alt (GRCh38) and optionally
    classify the resolved variant.

    Requires the dbSNP index parquet to be present at DBSNP_INDEX_PATH.
    When the index is absent or the rs-ID is unknown, ``known=false`` is
    returned with no prediction.

    The rs-ID is normalised: leading 'rs' prefix is case-insensitive and
    optional (both 'rs12345678' and '12345678' are accepted).
    """
    normalised_id = rs_id.strip().lower()
    if not normalised_id.startswith("rs"):
        normalised_id = "rs" + normalised_id

    locus = _lookup_rsid(rs_id)

    if locus is None:
        return RsidLookupResponse(rs_id=normalised_id, known=False)

    # Attempt prediction if the model is loaded
    prediction: Optional[VariantPrediction] = None
    if _PIPELINE is not None:
        try:
            row = _variant_to_row(
                VariantRequest(
                    chrom=locus["chrom"],
                    pos=locus["pos"],
                    ref=locus["ref"],
                    alt=locus["alt"],
                )
            )
            prediction = _make_prediction(row)
        except Exception as exc:
            logger.warning("rsid_lookup: prediction failed for %s: %s", rs_id, exc)

    return RsidLookupResponse(
        rs_id=normalised_id,
        known=True,
        chrom=locus["chrom"],
        pos=locus["pos"],
        ref=locus["ref"],
        alt=locus["alt"],
        prediction=prediction,
    )


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _global_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )
