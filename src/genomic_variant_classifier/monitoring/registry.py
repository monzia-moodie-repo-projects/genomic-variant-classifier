"""
monitoring/registry.py -- Monzia Moodie

Declarative single-source-of-truth for every external data source the classifier consumes. Populated to
replace the former 0-byte stub. Everything that needs to reason about "the databases" -- the freshness
monitor, the documented freshness report, the launch preflight's critical-asset list, the data-source
audit -- reads THIS, so adding/retiring a source is one row here, not edits scattered across DataFreshnessAgent
(which hardcoded 4 polls) and LiteratureScout (which hardcoded a few schema hashes).

Grounding (no guessing): local_path comes from AnnotationConfig + the on-disk robocopy manifest; upstream_url
+ check_method come from the agent code's CONFIRMED probes (ClinVar FTP, gnomAD GCS, AlphaMissense GCS, LOVD
REST). Sources whose upstream probe is not yet confirmed are check=MANUAL with upstream_url=None and a TODO --
they are listed (so coverage is complete) but never probed against a fabricated URL.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Category(str, Enum):
    CORE = "core"              # the label/cohort source
    ANNOTATION = "annotation"  # per-variant functional scores
    POPULATION = "population"  # allele-frequency / constraint
    NETWORK = "network"        # gene/protein interaction graphs
    SEQUENCE = "sequence"      # protein sequence / structure / LM
    SOFTWARE = "software"      # a tool/model whose *version* matters (not a corpus)


class Check(str, Enum):
    FTP_LISTING = "ftp_listing"      # newest file in an FTP directory
    HTTP_ETAG = "http_etag"          # HEAD -> ETag / Last-Modified
    HTTP_HASH = "http_hash"          # GET -> hash of the body
    GITHUB_RELEASE = "github_release"  # latest GitHub release tag
    MANUAL = "manual"                # no confirmed automated probe yet (TODO)


class Verdict(str, Enum):
    ACTIVE = "active"    # data present on disk AND feeding the feature matrix
    CACHE = "cache"      # only a small lookup/cache present (partial)
    STUB = "stub"        # connector exists, no data -> silent-zero
    BLOCKED = "blocked"  # access-controlled / licensed; acquisition blocked
    PLANNED = "planned"  # new connector code required


@dataclass(frozen=True)
class Source:
    key: str                  # canonical lowercase key (unique)
    name: str                 # display name
    category: Category
    verdict: Verdict
    check: Check
    local_path: str | None    # repo-relative primary asset (None if N/A)
    upstream_url: str | None   # probe URL (None iff check == MANUAL)
    version: str | None        # current pinned version, if known
    acquire: str | None        # how a HUMAN re-acquires it (connector/script) -- HITL
    notes: str = ""


# CONFIRMED upstream probes (mirror agent_layer/config.py + the agent code).
_CLINVAR_FTP = "ftp://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/"
_CLINVAR_SUMMARY = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
# NOTE: DataFreshnessAgent pins v4.0 here; the corpus is v4.1 -> probe URL is STALE (tracked in notes).
_GNOMAD_V41_TBI = ("https://storage.googleapis.com/gcp-public-data--gnomad/"
                   "release/4.1/vcf/exomes/gnomad.exomes.v4.1.sites.chr1.vcf.bgz.tbi")
_ALPHAMISSENSE_GCS = "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz"
_LOVD_API = "https://databases.lovd.nl/shared/api/rest.php/variants/"


REGISTRY: tuple[Source, ...] = (
    # ---- core ----
    Source("clinvar", "ClinVar", Category.CORE, Verdict.ACTIVE, Check.FTP_LISTING,
           "data/processed/clinvar_grch38.parquet", _CLINVAR_FTP, "weekly GRCh38 VCF",
           "scripts data-prep clinvar ingest (HITL)",
           "raw VCF data/raw/clinvar/clinvar_GRCh38.vcf.gz; the cohort label source."),
    # ---- annotation (functional scores) ----
    Source("dbnsfp", "dbNSFP", Category.ANNOTATION, Verdict.ACTIVE, Check.MANUAL,
           "data/external/dbnsfp/dbnsfp_clinvar_index.parquet", None, "4.x",
           "DbNSFPConnector(dbnsfp_path)",
           "TODO confirm upstream (Box/Google). dbnsfp_full_index.parquet.OOMbak (896MB) is stale cruft."),
    Source("spliceai", "SpliceAI", Category.ANNOTATION, Verdict.ACTIVE, Check.MANUAL,
           "data/external/spliceai/spliceai_index.parquet", None, "Illumina precomputed",
           "SpliceAIConnector(spliceai_path)",
           "TODO confirm upstream (Illumina BaseSpace, login). DUP copy at data/processed/spliceai_index.parquet."),
    Source("alphamissense", "AlphaMissense", Category.ANNOTATION, Verdict.ACTIVE, Check.HTTP_ETAG,
           "data/external/alphamissense/alphamissense_protein_index.parquet", _ALPHAMISSENSE_GCS, "hg38",
           "AlphaMissense connector",
           "HEAD ETag/Last-Modified on the GCS TSV; also data/processed/alphamissense_index.parquet."),
    Source("phylop", "PhyloP", Category.ANNOTATION, Verdict.STUB, Check.MANUAL,
           None, None, None, "PhyloPConnector(phylop_path)",
           "TODO confirm upstream (UCSC bigWig). No local data -> silent-zero."),
    Source("vep", "Ensembl VEP", Category.ANNOTATION, Verdict.STUB, Check.MANUAL,
           None, None, None, "VEP CLI / cache",
           "TODO. vep codon feature currently stub."),
    Source("eve", "EVE", Category.ANNOTATION, Verdict.CACHE, Check.MANUAL,
           "data/raw/cache/eve_eve_lookup.parquet", None, None, "EVE connector",
           "TODO confirm upstream. Only a small lookup cache present; needs HGVSp parser to activate."),
    Source("hgmd", "HGMD Professional", Category.ANNOTATION, Verdict.BLOCKED, Check.MANUAL,
           "data/raw/cache/hgmd_variant_lookup.parquet", None, None, "QIAGEN licensed seat",
           "Procurement-blocked. Only a tiny lookup cache present."),
    # ---- population (allele frequency / constraint) ----
    Source("gnomad", "gnomAD v4 exomes", Category.POPULATION, Verdict.ACTIVE, Check.HTTP_ETAG,
           "data/processed/gnomad_v4_exomes.parquet", _GNOMAD_V41_TBI, "v4.1",
           "gnomAD connector",
           "DataFreshnessAgent probe pins v4.0 (STALE) -- corpus is v4.1; probe URL corrected here to v4.1."),
    Source("gnomad_constraint", "gnomAD constraint", Category.POPULATION, Verdict.ACTIVE, Check.HTTP_ETAG,
           "data/external/gnomad/gnomad.v4.1.constraint_metrics.constraint_index.parquet", _GNOMAD_V41_TBI,
           "v4.1", "connector_gnomad_constraint.py",
           "constraint_metrics.tsv -> loeuf-derived oe (see patch_constraint_oe_from_loeuf)."),
    Source("kgp_1000", "1000 Genomes Phase 3", Category.POPULATION, Verdict.STUB, Check.MANUAL,
           None, None, "Phase 3", "connector_1kgp.py",
           "TODO. data/external/1kgp + 1000genomes dirs are EMPTY on disk -> kg_path silent-zero."),
    Source("finngen", "FinnGen", Category.POPULATION, Verdict.ACTIVE, Check.MANUAL,
           "data/external/finngen/finnge_R12_annotated_variants_v1.gz", None, "R12",
           "FinnGen connector",
           "29.9GB (74% of corpus). FILENAME TYPO 'finnge'; memory said R10 -> actual R12. TODO confirm upstream."),
    Source("dbsnp", "dbSNP", Category.POPULATION, Verdict.ACTIVE, Check.MANUAL,
           "data/processed/dbsnp_index.parquet", None, None, "dbSNP connector (stub step 10)",
           "TODO confirm upstream (NCBI). index parquet present + data/raw/cache/dbsnp_af_lookup.parquet."),
    Source("gtex", "GTEx", Category.POPULATION, Verdict.CACHE, Check.MANUAL,
           "data/raw/cache", None, "v8/v10?", "GTEx connector",
           "TODO confirm version + upstream. Only per-gene eqtl/expr caches present."),
    # ---- network ----
    Source("string", "STRING-DB", Category.NETWORK, Verdict.ACTIVE, Check.MANUAL,
           "data/external/string/9606.protein.links.detailed.v12.0.txt.gz", None, "v12.0",
           "STRING GNN builder",
           "TODO confirm upstream (stringdb-downloads.org). v12.0 pinned; string_graph_700.pkl cached."),
    Source("reactome", "Reactome", Category.NETWORK, Verdict.PLANNED, Check.MANUAL,
           None, None, None, "reactome connector (NEW code)",
           "Phase D. reactome_path connector not yet written."),
    # ---- sequence / structure / LM ----
    Source("uniprot", "UniProt (human reviewed)", Category.SEQUENCE, Verdict.ACTIVE, Check.MANUAL,
           "data/external/uniprot/uniprot_human_reviewed.parquet", None, "reviewed/SwissProt",
           "UniProt index", "TODO confirm upstream release cadence."),
    Source("alphafold", "AlphaFold DB", Category.SEQUENCE, Verdict.CACHE, Check.HTTP_ETAG,
           "data/raw/cache/alphafold", "https://alphafold.ebi.ac.uk/api/prediction/", "v4",
           "EBI REST /api/prediction/{UNIPROT_ACC} (stub step 14)",
           "Per-residue pLDDT/RSA. Only a few .cif cached; AF-E7ENB7 cache is LOCAL-ONLY (not on Drive)."),
    Source("esm2", "ESM-2", Category.SOFTWARE, Verdict.STUB, Check.GITHUB_RELEASE,
           "data/raw/cache", "https://api.github.com/repos/facebookresearch/esm/releases/latest", None,
           "GPU regen (CPU ~31h)", "esm2_delta_norm silent-zero until HGVSp parser lands."),
    # ---- annotation: clinical-curation ----
    Source("omim", "OMIM", Category.ANNOTATION, Verdict.CACHE, Check.MANUAL,
           "data/raw/cache/omim_gene_table.parquet", None, None, "OMIM connector (licensed)",
           "TODO. Only a small gene table cache present."),
    Source("clingen", "ClinGen", Category.ANNOTATION, Verdict.CACHE, Check.MANUAL,
           "data/raw/cache/clingen_gene_scores.parquet", None, None, "ClinGen connector",
           "TODO confirm upstream (clinicalgenome.org). Only a gene-score cache present."),
    Source("lovd", "LOVD", Category.CORE, Verdict.ACTIVE, Check.HTTP_HASH,
           "data/external/lovd/lovd_all_variants.parquet", _LOVD_API, "shared API",
           "LOVDConnector(lovd_path)",
           "REST probe may 401/403 without LOVD_API_KEY -> poll skipped honestly."),
    # ---- blocked / planned somatic ----
    Source("cosmic", "COSMIC", Category.ANNOTATION, Verdict.BLOCKED, Check.MANUAL,
           None, None, None, "COSMIC connector (NEW code; licensed)",
           "Empty dir. Licensed; NEW connector required."),
    Source("tcga", "TCGA", Category.ANNOTATION, Verdict.BLOCKED, Check.MANUAL,
           None, None, None, "TCGA (dbGaP controlled)",
           "Empty dir. Controlled-access; dbGaP/PI sponsorship blocked."),
)


def all_sources() -> tuple[Source, ...]:
    return REGISTRY


def by_key(key: str) -> Source:
    for s in REGISTRY:
        if s.key == key:
            return s
    raise KeyError(f"no registry source with key {key!r}")


def probeable() -> list[Source]:
    """Sources with a confirmed automated upstream probe (check != MANUAL and a URL present)."""
    return [s for s in REGISTRY if s.check is not Check.MANUAL and s.upstream_url]


def by_verdict(v: Verdict) -> list[Source]:
    return [s for s in REGISTRY if s.verdict is v]


def critical_assets() -> list[str]:
    """Local paths whose absence would silent-stub an ACTIVE source -- the preflight --asset list."""
    return [s.local_path for s in REGISTRY if s.verdict is Verdict.ACTIVE and s.local_path]
