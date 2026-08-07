"""
src/genomic_variant_classifier/api/schemas.py
==================
Pydantic request / response schemas for the variant pathogenicity API.

Design notes
------------
* All optional fields default to None / sentinel values so that callers
  can submit a minimal {chrom, pos, ref, alt} payload and the inference
  pipeline will impute the rest (using population-mean defaults where safe,
  or conservative "unknown" fills where not).
* ``VariantRequest`` mirrors the raw-input columns accepted by
  ``DataPrepPipeline._engineer_features``; any additional columns in the
  parquet schema are simply ignored by the pipeline.
* ``BatchPredictRequest`` caps at MAX_BATCH_SIZE variants to bound memory.
  Larger jobs should use the offline ``scripts/run_phase2_eval.py`` path.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Hard cap enforced in /batch endpoint.
MAX_BATCH_SIZE: int = 1_000


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class VariantRequest(BaseModel):
    """A single genomic variant to classify."""

    # --- Required: locus + alleles -------------------------------------------
    chrom: str = Field(
        ...,
        description="Chromosome (e.g. '1', 'X', 'MT').  'chr' prefix accepted.",
        examples=["1", "17", "X"],
    )
    pos: int = Field(..., gt=0, description="1-based genomic position (GRCh38).")
    ref: str = Field(..., min_length=1, description="Reference allele (ACGT).")
    alt: str = Field(..., min_length=1, description="Alternate allele (ACGT).")

    # --- Optional: population allele frequency --------------------------------
    allele_freq: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "gnomAD v4.1 allele frequency.  If absent the pipeline will "
            "attempt a lookup against the in-memory gnomAD index; if the "
            "variant is not found, AF is treated as 0 (absent from gnomAD)."
        ),
    )

    # --- Optional: functional annotation ------------------------------------
    consequence: Optional[str] = Field(
        default=None,
        description=(
            "VEP consequence term or '&'-delimited list "
            "(e.g. 'missense_variant', 'stop_gained&splice_region_variant')."
        ),
    )
    gene_symbol: Optional[str] = Field(
        default=None,
        description="HGNC gene symbol (e.g. 'BRCA1').  Used for gene-level features.",
    )

    # --- Optional: pre-computed tool scores ---------------------------------
    cadd_phred:          Optional[float] = Field(default=None, ge=0.0)
    sift_score:          Optional[float] = Field(default=None, ge=0.0, le=1.0)
    polyphen2_score:     Optional[float] = Field(default=None, ge=0.0, le=1.0)
    revel_score:         Optional[float] = Field(default=None, ge=0.0, le=1.0)
    phylop_score:        Optional[float] = Field(default=None)
    gerp_score:          Optional[float] = Field(default=None)
    alphamissense_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="AlphaMissense pathogenicity score (0 = benign, 1 = pathogenic).",
    )
    splice_ai_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="SpliceAI max delta score (0 = no splice disruption, 1 = high confidence).",
    )
    eve_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="EVE evolutionary model pathogenicity score (0=benign, 1=pathogenic; 0.5=not covered).",
    )
    codon_position: Optional[int] = Field(
        default=None,
        ge=0,
        le=3,
        description="Position within codon (1, 2, or 3); 0 for non-coding variants.",
    )
    dbsnp_af: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="dbSNP population allele frequency (supplement for variants absent from gnomAD).",
    )
    omim_n_diseases: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of OMIM disease phenotypes for this gene.",
    )
    omim_n_diseases_molecular: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of OMIM phenotypes with a confirmed molecular basis (mapping key '(3)').",
    )
    omim_is_autosomal_dominant: Optional[int] = Field(
        default=None,
        ge=0,
        le=1,
        description="1 if gene has any autosomal dominant OMIM phenotype.",
    )
    clingen_validity_score: Optional[int] = Field(
        default=None,
        ge=0,
        le=5,
        description="ClinGen Gene Validity score (0=no evidence, 5=Definitive).",
    )
    hgmd_is_disease_mutation: Optional[int] = Field(
        default=None,
        ge=0,
        le=1,
        description="1 if variant is classified as DM in HGMD.",
    )
    hgmd_n_reports: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of HGMD reports for this variant.",
    )

    # --- Optional: gene-level constraint and ClinVar gene reputation --------
    gene_constraint_oe: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="gnomAD pLoF observed/expected ratio for this gene.",
    )
    n_pathogenic_in_gene: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Number of ClinVar pathogenic variants in this gene. "
            "Top feature by importance (1448 vs next at 417). "
            "Defaults to 0 when absent — conservative but will underestimate "
            "pathogenicity for known disease genes (BRCA1, TP53, LDLR, etc.). "
            "Callers should supply this from a ClinVar gene summary lookup."
        ),
    )

    # --- Optional: UniProt protein features ---------------------------------
    has_uniprot_annotation: Optional[int] = Field(
        default=None,
        ge=0,
        le=1,
        description="1 if the gene has any UniProt functional annotation.",
    )
    n_known_pathogenic_protein_variants: Optional[int] = Field(
        default=None,
        ge=0,
        description="Pathogenic variant count for this gene from UniProt.",
    )

    @field_validator("chrom")
    @classmethod
    def _strip_chr_prefix(cls, v: str) -> str:
        """Accept 'chr1' / 'chrM' as well as '1' / 'MT' for user convenience."""
        v = v.strip()
        if v.lower().startswith("chr"):
            v = v[3:]
        if v == "M":
            v = "MT"
        return v

    @field_validator("ref", "alt")
    @classmethod
    def _allele_uppercase(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def _derive_variant_id(self) -> VariantRequest:
        """Attach a canonical variant_id used by the feature pipeline."""
        object.__setattr__(
            self,
            "_variant_id",
            f"{self.chrom}:{self.pos}:{self.ref}:{self.alt}",
        )
        return self

    model_config = {"populate_by_name": True}


class BatchPredictRequest(BaseModel):
    """Up to MAX_BATCH_SIZE variants in a single request."""

    variants: list[VariantRequest] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"List of variants to classify (max {MAX_BATCH_SIZE}).",
    )


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

class VariantPrediction(BaseModel):
    """Per-variant prediction result."""

    variant_id: str = Field(description="chrom:pos:ref:alt")
    pathogenicity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Ensemble probability of pathogenicity (0 = benign, 1 = pathogenic).",
    )
    classification: str = Field(
        description=(
            "Categorical call: 'Pathogenic', 'Likely pathogenic', "
            "'Uncertain significance', 'Likely benign', or 'Benign'."
        )
    )
    confidence: str = Field(
        description="'high' | 'medium' | 'low' — based on score distance from thresholds."
    )
    # Feature contributions — top-5 only, to keep payload small.
    top_features: Optional[dict[str, float]] = Field(
        default=None,
        description="SHAP values or model importances for the top-5 features.",
    )
    # Uncertainty decomposition
    uncertainty_epistemic: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Variance across base model predictions (epistemic / model uncertainty). "
            "High values indicate ensemble members disagree — collect additional "
            "functional evidence before reporting."
        ),
    )
    uncertainty_aleatoric: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Binary entropy of the final pathogenicity probability (aleatoric / data "
            "uncertainty). Peaks near score=0.5; irreducible by further training."
        ),
    )
    # Conformal prediction interval
    coverage_interval: Optional[list[float]] = Field(
        default=None,
        description=(
            "[lo, hi] conformal prediction interval at 90% coverage. "
            "None when conformal calibration config is not loaded."
        ),
    )


class PredictResponse(BaseModel):
    """Response for /predict (single variant).

    PROD-1 (2026-08-07), API 2.0.0. `pipeline_version` is gone: it was
    the OpenAPI version doubling as prediction provenance. `model_version`
    is now the registry record's label and is None when the serving
    artifact cannot be attributed -- which is honest, and was previously
    a constant reading "phase2-v1" whatever was loaded.
    """

    prediction: VariantPrediction
    model_record_id: Optional[str] = None
    model_version: Optional[str] = None


class BatchPredictResponse(BaseModel):
    """Response for /batch. Every result shares one serving identity."""

    predictions: list[VariantPrediction]
    n_pathogenic: int
    n_benign: int
    n_uncertain: int
    model_record_id: Optional[str] = None
    model_version: Optional[str] = None


class HealthResponse(BaseModel):
    """Liveness and readiness, which are different questions.

    `live` says the process is responsive. `ready` says it may take
    inference traffic: a model is loaded, its bytes are attributable to a
    declared production record, and its executable roster is coherent
    with that record. `status` remains for compatibility and mirrors
    `ready`.
    """

    status: str  # "ok" | "degraded"
    live: bool = True
    ready: bool = False
    model_loaded: bool
    model_attributed: bool = False
    gnomad_index_loaded: bool
    gene_counts_loaded: bool
    uptime_seconds: float


class GeneSummaryResponse(BaseModel):
    gene_symbol: str
    n_pathogenic_in_gene: int
    gene_constraint_oe: Optional[float] = Field(
        default=None,
        description=(
            "gnomAD pLoF observed/expected ratio.  None = not available; "
            "engineer_features() defaults to 1.0 (unconstrained) when absent."
        ),
    )
    has_uniprot_annotation: int = Field(
        default=0,
        description="1 if the gene has any UniProt functional annotation.",
    )
    source: str = "ClinVar (training set)"


class RsidLookupResponse(BaseModel):
    """Response for GET /rsid/{rs_id}."""

    rs_id: str = Field(description="Normalised rs-ID (e.g. 'rs12345678').")
    known: bool = Field(description="True if the rs-ID was found in the dbSNP index.")
    chrom: Optional[str] = None
    pos: Optional[int] = None
    ref: Optional[str] = None
    alt: Optional[str] = None
    prediction: Optional[VariantPrediction] = Field(
        default=None,
        description="Pathogenicity prediction for the resolved locus, if the model is loaded.",
    )


class ModelAttributionResponse(BaseModel):
    """What the process is serving, and what it may claim about it.

    Four independent axes, because they answer four different questions,
    and collapsing any two is what allowed a metric from one model to be
    advertised for another. Consumers branch on the enums; `detail` is
    supplementary human diagnostics and is never a machine contract.
    """

    resolution_status: str
    deployment_alignment: str
    roster_alignment: str
    evaluation_applicability: str

    record_id: Optional[str] = None
    model_version: Optional[str] = None
    artifact_sha256: Optional[str] = None
    registry_stage: Optional[str] = None

    #: SERVEROSTER-1. The trained roster and the roster this artifact can
    #: actually execute are different facts. The REST pipeline excludes
    #: `cnn_1d`, which needs a FASTA context window unavailable here, so
    #: a twelve-model projection of a thirteen-model ensemble is served.
    #: Reporting one as the other would advertise a property of something
    #: other than the loaded object.
    registered_model_roster: Optional[list[str]] = None
    served_model_roster: list[str] = []
    served_roster_fingerprint: Optional[str] = None

    detail: Optional[str] = None

    @model_validator(mode="after")
    def _coherent(self) -> "ModelAttributionResponse":
        """A resolved attribution must carry identity; an unresolved one
        must not, and neither may carry evidence -- there is nowhere in
        this model to put any."""
        resolved = self.resolution_status == "registered"
        if resolved and not all((self.record_id, self.model_version,
                                 self.artifact_sha256)):
            raise ValueError(
                "a resolved attribution requires record_id, model_version "
                "and artifact_sha256")
        if not resolved and self.record_id is not None:
            raise ValueError(
                "an unresolved attribution must not carry registry identity")
        return self


class InfoResponse(BaseModel):
    """Model metadata. API 2.0.0.

    BREAKING: `pipeline_version`, `training_auroc`, `training_auprc`,
    `holdout_auroc` and the free-text `description` are removed. The
    first was a software version masquerading as model provenance; the
    three metrics were 2026-03-25 constants published irrespective of
    what was loaded; the description asserted a five-model ensemble
    against a roster of thirteen and a 1.2 M cohort against 1.49 M.

    NO METRIC APPEARS HERE. The serving artifact is a projection of the
    evaluated ensemble, so evidence measured on the record does not
    automatically describe these bytes. `evaluation_applicability` states
    why, and a sealed evaluation naming this digest and this roster
    fingerprint is what will eventually authorise publication.
    """

    api_version: str
    model_loaded: bool
    attribution: ModelAttributionResponse

    n_features: int
    feature_names: list[str]
    phase2_features_remaining: list[str]


# ---------------------------------------------------------------------------
# Classification thresholds
# ---------------------------------------------------------------------------

# ACMGish five-tier mapping based on calibrated probability.
# These defaults apply when no calibrated threshold file is found.
# Run scripts/calibrate_thresholds.py to produce models/classification_thresholds.json
# and override these defaults with empirically-calibrated boundaries.
_DEFAULT_THRESHOLDS: dict[str, tuple[float, float]] = {
    "Pathogenic":             (0.90, 1.01),
    "Likely pathogenic":      (0.70, 0.90),
    "Uncertain significance": (0.30, 0.70),
    "Likely benign":          (0.10, 0.30),
    "Benign":                 (-0.01, 0.10),
}


def _load_thresholds() -> dict[str, tuple[float, float]]:
    """
    Attempt to load calibrated thresholds from models/classification_thresholds.json.
    Falls back to _DEFAULT_THRESHOLDS silently if the file is absent or malformed.
    """
    import json
    import os
    from pathlib import Path

    candidates = [
        Path(os.environ.get("THRESHOLDS_PATH", "")),
        Path("models/classification_thresholds.json"),
    ]
    for path in candidates:
        if path and path.exists():
            try:
                data = json.loads(path.read_text())
                raw = data.get("thresholds", {})
                parsed: dict[str, tuple[float, float]] = {}
                for label, bounds in raw.items():
                    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                        parsed[label] = (float(bounds[0]), float(bounds[1]))
                if set(parsed.keys()) == set(_DEFAULT_THRESHOLDS.keys()):
                    return parsed
            except Exception:
                pass   # malformed file — use defaults
    return dict(_DEFAULT_THRESHOLDS)


CLASSIFICATION_THRESHOLDS: dict[str, tuple[float, float]] = _load_thresholds()


def score_to_classification(score: float) -> tuple[str, str]:
    """Return (classification, confidence) for a raw probability score."""
    for label, (lo, hi) in CLASSIFICATION_THRESHOLDS.items():
        if lo < score <= hi:
            dist = min(score - lo, hi - score)
            if dist >= 0.15:
                confidence = "high"
            elif dist >= 0.05:
                confidence = "medium"
            else:
                confidence = "low"
            return label, confidence
    # Fallback — should not occur for valid scores in [0, 1]
    return "Uncertain significance", "low"
