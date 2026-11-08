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

class QuantityModel(BaseModel):
    """A value with its unit and state. `unavailable` and `unresolved` carry null and a reason; a computed zero is 0."""

    model_config = {"extra": "forbid"}

    value: Optional[float]
    unit: str
    status: str
    reason: Optional[str] = None


class StatusModel(BaseModel):
    """Something this record does not provide, and why."""

    model_config = {"extra": "forbid"}

    status: str
    reason: Optional[str] = None


class CombinerReference(BaseModel):
    model_config = {"extra": "forbid"}

    kind: str
    combiner_sha256: str
    kernel: str
    members: list[str]


class RosterResearchView(BaseModel):
    """A DESCRIPTIVE uniform view of the fitted members. Not the selected blend, and not its uncertainty."""

    model_config = {"extra": "forbid"}

    interpretation: str
    weighting: str
    ddof: int
    between_member_spread_variance: QuantityModel
    member_disagreement: QuantityModel
    disagreement_status: str
    method: Optional[str] = None


class VariantPrediction(BaseModel):
    """Per-variant prediction record, schema 3.0.0 (package 1, gate G6).

    BREAKING relative to 2.0.0. Removed: `classification` (the five-tier relabelling), `confidence` (distance from
    an unqualified threshold), `uncertainty_epistemic` and `uncertainty_aleatoric` (a variance and an entropy under
    retired names, the second of them served from a different predictor), `top_features` (never populated: no
    attribution method existed) and `coverage_interval` (never populated). The score is the selected predictor's --
    the stored convex blend -- at full binary64 precision, never rounded, clipped or relabelled. `model_call` is a
    descriptive binary call under one tie rule; no decision policy is qualified, and the record says so.
    """

    model_config = {"extra": "forbid"}

    prediction_schema_version: str
    variant_id: str = Field(description="chrom:pos:ref:alt as submitted (after normalisation)")
    source_row_key: str = Field(description="Position of this variant within the submitted request")
    source_observation_id: str
    prediction_unit_id: str
    ordered_rows_sha256: str
    result_block_sha256: str
    combiner: CombinerReference
    positive_class: int
    pathogenicity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="The selected predictor's probability of the positive class (full precision; validated, "
                    "never clipped).",
    )
    model_call: int = Field(description="1 when pathogenicity_score >= 0.5 (tie_rule), else 0. Descriptive only.")
    tie_rule: str
    predictive_entropy_nats: QuantityModel
    roster_research_view: RosterResearchView
    weighted_blend_diagnostic: StatusModel
    preprocessing_status: str
    calibration_status: str
    decision_policy: StatusModel
    feature_attribution: StatusModel
    coverage_interval: StatusModel

    @field_validator("prediction_schema_version")
    @classmethod
    def _only_this_schema(cls, v: str) -> str:
        """A reader of 3.0.0 refuses any other schema version rather than guessing what its fields mean."""
        if v != "3.0.0":
            raise ValueError(f"unsupported prediction schema version {v!r}; this reader understands only 3.0.0")
        return v


class PredictResponse(BaseModel):
    """Response for /predict (single variant). API 3.0.0.

    `model_version` is the registry record's label and is None when the serving artifact cannot be attributed.
    """

    prediction: VariantPrediction
    model_record_id: Optional[str] = None
    model_version: Optional[str] = None


class BatchPredictResponse(BaseModel):
    """Response for /batch. Every result shares one serving identity.

    The 2.0.0 counts (n_pathogenic, n_benign, n_uncertain) are removed: matched by substring on the five-tier
    labels, they omitted every "Likely ..." result and did not partition the batch. The counts here partition it:
    n_model_call_positive + n_model_call_negative == n_predictions.
    """

    predictions: list[VariantPrediction]
    n_predictions: int
    n_model_call_positive: int
    n_model_call_negative: int
    model_record_id: Optional[str] = None
    model_version: Optional[str] = None

    @model_validator(mode="after")
    def _counts_partition(self) -> "BatchPredictResponse":
        if self.n_predictions != len(self.predictions):
            raise ValueError("n_predictions differs from the number of predictions")
        if self.n_model_call_positive + self.n_model_call_negative != self.n_predictions:
            raise ValueError("the model-call counts do not partition the batch")
        return self


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
        description="Pathogenicity prediction for the resolved locus, when a READY model is loaded.",
    )
    prediction_unavailable_reason: Optional[str] = Field(
        default=None,
        description="Why no prediction accompanies a known locus (null when one does).",
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
    predictor_contract_alignment: str = "unknown"

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
    """Model metadata. API 3.0.0 (2.0.0 removed the fields below; 3.0.0 changed the prediction schema).

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
# Classification thresholds: REMOVED in package 1, set 1c (2026-11-08)
# ---------------------------------------------------------------------------
#
# `_load_thresholds`, `CLASSIFICATION_THRESHOLDS` and `score_to_classification` lived here. Serving stopped using
# them in 3.0.0 (gate G6.2), and their last consumer, scripts/validate_external.py, now reads an explicit,
# predictor-bound decision policy (evaluation/decision_policy.py) passed by argument. The loader ran at IMPORT, read
# a path relative to the current directory (`Path("")` is "." -- a directory, so THRESHOLDS_PATH unset never
# failed), and swallowed every error into five fixed "ACMG-ish" tiers -- so what a script classified with depended on
# the directory it was started from, silently. No schema module reads a file.
