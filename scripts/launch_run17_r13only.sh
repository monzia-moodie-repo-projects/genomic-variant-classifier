#!/usr/bin/env bash
# [GENERATED from launch_run17_baseline.sh by gen_run17_ablation_launchers.py -- DO NOT EDIT BY HAND]
# Config: r13only. If the baseline changes, RE-GENERATE (do not hand-patch) to avoid drift.
# launch_run17_r13only.sh -- Run 17 R13-only ablation (FinnGen R12 excluded; generated from baseline).
#
# Activates, vs Run 15, FOUR present-but-previously-constant signal groups WITHOUT a schema change:
#   - af_1kg_{afr,eur,eas,sas,amr}  via --kg  (1000G 30x GRCh38 per-superpop AF parquet)
#   - rnaseq_{mean_log_tpm,detection_rate,log2_cv,log2fc,de_neglog10p}  via --rnaseq-path
#   - hetero_gnn_score              via --hetero-gnn + --kg-edges reactome:<gmt>
#   - gtex_* / reactome_pathway_count (carried from Run 15, kept REQUIRED here)
#
# CONTRACT vs launch_run15_baseline.sh (every difference fixes a Run-17 silent-zero risk):
#   - --rnaseq-path is REQUIRED + wired. The preflight_run17 --emit-kg template OMITS it, so a
#     hand-copied launch would zero the now-real rnaseq_* features. This script hard-fails if absent.
#   - --kg is REQUIRED + wired (af_1kg_*). Hard-fail if the parquet is missing (no silent constant).
#   - --hetero-gnn + --kg-edges reactome:<gmt> wired (hetero_gnn_score; else stays 0.5 default).
#   - --skip-svm ON (run_phase2_eval --help: required >100k; n_train ~1.2M, RBF is O(n^2)).
#   - gtex/reactome/kg/rnaseq ELEVATED to REQUIRED (hard-fail), matching the no-zero-feature directive
#     (Run 15 treated gtex/reactome as if-present). LOVD stays if-present (low-coverage, optional).
#   - --esm2-uniprot-index AND --eve-entry-map both wired to $UNIPROT_INDEX (HGVSp parser delivered);
#     EVE resolves per-protein entry-name filenames (1433G_HUMAN) to HGNC (YWHAG) via the index.
#     lands (roadmap; INCIDENT_2026-04-17). Wiring it now would add a KNOWN-zero feature, not signal.
#   - OUTDIR pinned to outputs/run17_r13only/full.

set -euo pipefail

if [ -d /venv/main/bin ] && ! echo "$PATH" | grep -q "/venv/main/bin"; then
    export PATH="/venv/main/bin:$PATH"
fi
PY="$(command -v python || command -v python3 || echo /venv/main/bin/python)"

REPO=/workspace/genomic-variant-classifier
DATA="$REPO/data"
OUTDIR="$REPO/outputs/run17_r13only/full"
LOG=/workspace/run17_r13only_master.log

cleanup() {
    rc=$?
    echo "============================================================" | tee -a "$LOG"
    if [ "$rc" -eq 0 ]; then
        echo "==> run_phase2_eval.py exit 0 (success) @ $(date -u +'%F %T') UTC" | tee -a "$LOG"
    else
        echo "==> run_phase2_eval.py exit $rc -- ABORT @ $(date -u +'%F %T') UTC" | tee -a "$LOG"
    fi
    echo "==> Checkpoints under $OUTDIR/models/:" | tee -a "$LOG"
    ls -la "$OUTDIR/models/" 2>&1 | tee -a "$LOG" || true
    echo "============================================================" | tee -a "$LOG"
}
trap cleanup EXIT

echo "==> Run 17 baseline launch @ $(date -u +'%F %T') UTC" | tee "$LOG"
echo "==> Python: $PY ($($PY --version 2>&1))" | tee -a "$LOG"

# -- 1. Required data preflight (Run-17 full input set; hard-fail = no silent zero) -----------
echo "==> [1/6] Data preflight" | tee -a "$LOG"
FAIL=0
KG_PARQUET="$DATA/external/1kgp/kg_grch38_af.parquet"
RNASEQ_PARQUET="$DATA/external/rnaseq_gene_expression.parquet"
COSMIC_TSV="$DATA/external/cosmic/CancerMutationCensus_AllData_v104_GRCh37.tsv.gz"  # Phase 2: COSMIC CMC (GRCh38 col inside the GRCh37 release)
KEGG_PARQUET="$DATA/external/kegg_gene_pathways.parquet"  # Phase 2: KEGG gene->pathway mapping
GTEX_PARQUET="$DATA/external/gtex_gene_expression.parquet"
REACTOME_PARQUET="$DATA/external/reactome_gene_pathways.parquet"
REACTOME_GMT="$DATA/external/reactome/ReactomePathways.gmt"
for f in \
    "$DATA/processed/clinvar_grch38_clean.parquet" \
    "$DATA/processed/clinvar_grch38_clean_seq.parquet" \
    "$DATA/processed/gnomad_v4_exomes.parquet" \
    "$DATA/external/spliceai/spliceai_index.parquet" \
    "$DATA/external/alphamissense/AlphaMissense_hg38.tsv.gz" \
    "$DATA/external/gnomad/gnomad.v4.1.constraint_metrics.tsv" \
    "$DATA/external/dbnsfp/dbnsfp_clinvar_index.parquet" \
    "$GTEX_PARQUET" \
    "$REACTOME_PARQUET" \
    "$REACTOME_GMT" \
    "$KG_PARQUET" \
    "$RNASEQ_PARQUET" \
    "$COSMIC_TSV" \
    "$KEGG_PARQUET" \
; do
    if [ ! -f "$f" ]; then
        echo "==> MISSING (required): $f" | tee -a "$LOG"; FAIL=1
    else
        SZ=$(stat -c%s "$f" 2>/dev/null || echo 0)
        echo "==> OK: $f ($(( SZ / 1048576 )) MB)" | tee -a "$LOG"
    fi
done

# STRING required for the GNN path (gnn_score std > 0)
STRING_LINKS="$DATA/external/string/9606.protein.links.detailed.v12.0.txt.gz"
STRING_INFO="$DATA/external/string/9606.protein.info.v12.0.txt.gz"
for f in "$STRING_LINKS" "$STRING_INFO"; do
    if [ ! -f "$f" ]; then
        echo "==> MISSING (required for GNN): $f" | tee -a "$LOG"; FAIL=1
    else
        SZ=$(stat -c%s "$f" 2>/dev/null || echo 0)
        echo "==> OK: $f ($(( SZ / 1048576 )) MB)" | tee -a "$LOG"
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "==> ABORT (exit 2): missing required inputs" | tee -a "$LOG"; exit 2
fi

cd "$REPO"

# -- 2. Smoke import ----------------------------------------------------------
echo "==> [2/6] Smoke import" | tee -a "$LOG"
if ! python -c "from genomic_variant_classifier.models.variant_ensemble import VariantEnsemble; from genomic_variant_classifier.data.real_data_prep import DataPrepPipeline; from genomic_variant_classifier.data.rnaseq import annotate_rnaseq_from_parquet; from genomic_variant_classifier.data.thousandgenomes import ThousandGenomesConnector; print('import OK')" 2>&1 | tee -a "$LOG"; then
    echo "==> ABORT (exit 3): import failed" | tee -a "$LOG"; exit 3
fi
echo "==> HEAD: $(git rev-parse --short HEAD)" | tee -a "$LOG"

# -- 2b. imodelsx KAN package patch (idempotent) ------------------------------
# patch_imodelsx_kan.py docstring: "the smoke gate invokes it first, and the launcher should too."
# Unpatched, imodelsx KANClassifier.fit raises NameError -> KAN drops from BOTH ensemble fits, an
# invalid model-comparison run. The full launcher has no other KAN guard (only the smoke gate does).
echo "==> [2b/6] imodelsx KAN patch" | tee -a "$LOG"
if ! python scripts/patch_imodelsx_kan.py 2>&1 | tee -a "$LOG"; then
    echo "==> ABORT (exit 3): imodelsx KAN patch failed (KAN would drop from the ensemble)" | tee -a "$LOG"; exit 3
fi

# -- 3. Stale .pyc cleanup ----------------------------------------------------
echo "==> [3/6] Clear stale .pyc" | tee -a "$LOG"
find "$REPO/src" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# -- 4. GPU / dependency check ------------------------------------------------
echo "==> [4/6] GPU / dependency check" | tee -a "$LOG"
python -c "
import torch, sklearn, lightgbm
print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU', torch.cuda.get_device_name(0))
else:
    print('WARNING: no CUDA -- GBDT/NN on CPU (much slower)')
from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
LGBMClassifier(n_estimators=10, verbose=-1).fit(X, y)
print('lightgbm smoke fit OK')
" 2>&1 | tee -a "$LOG"

# -- 4b. pybigtools (PhyloP BigWig reader): install-if-missing + HARD verify.
#     launch activates a prebuilt /venv/main that may predate this dep; without it
#     PhyloPConnector ImportErrors -> silent phylop_score=0.0. Idempotent install,
#     then a hard import gate (exit 4) so a failed install is LOUD, never silent.
echo "==> [4b/6] pybigtools (PhyloP) install + verify" | tee -a "$LOG"
$PY -m pip install 'pybigtools>=0.3.0' --quiet 2>&1 | tail -3 | tee -a "$LOG" || true
if ! python -c "import pybigtools; print('pybigtools', pybigtools.__version__ if hasattr(pybigtools,'__version__') else 'OK')" 2>&1 | tee -a "$LOG"; then
    echo "==> ABORT (exit 4): pybigtools import failed -- PhyloP would silent-zero. Install pybigtools>=0.3.0 on the VM." | tee -a "$LOG"; exit 4
fi

# -- 5. KG + rnaseq + STRING wiring sanity (read-only column probes) -----------
echo "==> [5/6] KG + rnaseq column probes" | tee -a "$LOG"
mkdir -p "$OUTDIR"
python -c "
import pyarrow.parquet as pq, sys
kg = pq.ParquetFile(r'$KG_PARQUET').schema_arrow.names
need = ['AFR_AF','EUR_AF','EAS_AF','SAS_AF','AMR_AF']
miss = [c for c in need if c not in kg]
print('kg cols:', kg)
if miss:
    print('ABORT: kg parquet missing per-superpop AF cols', miss); sys.exit(4)
rs = pq.ParquetFile(r'$RNASEQ_PARQUET').schema_arrow.names
need_rs = ['gene_symbol','rnaseq_mean_log_tpm','rnaseq_detection_rate','rnaseq_log2_cv','rnaseq_log2fc','rnaseq_de_neglog10p']
miss_rs = [c for c in need_rs if c not in rs]
print('rnaseq cols:', rs)
if miss_rs:
    print('ABORT: rnaseq parquet missing cols', miss_rs); sys.exit(5)
print('kg + rnaseq column contracts OK')
kegg = pq.ParquetFile(r'$KEGG_PARQUET').schema_arrow.names
need_kegg = ['gene_symbol','kegg_pathway_count','kegg_disease_pathway_flag']
miss_kegg = [c for c in need_kegg if c not in kegg]
print('kegg cols:', kegg)
if miss_kegg:
    print('ABORT: kegg parquet missing cols', miss_kegg); sys.exit(6)
print('kegg column contract OK')
" 2>&1 | tee -a "$LOG"
echo "==> STRING links $(stat -c%s "$STRING_LINKS") bytes; info $(stat -c%s "$STRING_INFO") bytes" | tee -a "$LOG"

# -- 6. Build CLI + launch ----------------------------------------------------
echo "==> [6/6] Launch" | tee -a "$LOG"
ARGS="--clinvar $DATA/processed/clinvar_grch38_clean.parquet"
ARGS="$ARGS --seq-windows $DATA/processed/clinvar_grch38_clean_seq.parquet"
ARGS="$ARGS --gnomad $DATA/processed/gnomad_v4_exomes.parquet"
ARGS="$ARGS --spliceai $DATA/external/spliceai/spliceai_index.parquet"
ARGS="$ARGS --alphamissense $DATA/external/alphamissense/AlphaMissense_hg38.tsv.gz"
ARGS="$ARGS --gnomad-constraint $DATA/external/gnomad/gnomad.v4.1.constraint_metrics.tsv"
ARGS="$ARGS --dbnsfp-path $DATA/external/dbnsfp/dbnsfp_clinvar_index.parquet"
ARGS="$ARGS --gtex-path $GTEX_PARQUET"
ARGS="$ARGS --reactome-path $REACTOME_PARQUET"
ARGS="$ARGS --rnaseq-path $RNASEQ_PARQUET"
ARGS="$ARGS --kg $KG_PARQUET"
ARGS="$ARGS --cosmic-path $COSMIC_TSV"
ARGS="$ARGS --kegg-path $KEGG_PARQUET"
# R12 INTENTIONALLY EXCLUDED (r13only ablation): no --finngen-path passed.
# run_phase2_eval.py else-branch constant-fills finngen_af_fin/finngen_af_nfsee (0.0) +
# finngen_enrichment (1.0). The 91-feature contract holds; R12 columns carry no signal.
FINNGEN_R13_FILE="$DATA/external/finngen/finngen_R13_annotated_variants_v0.gz"  # R13 dual-release (correct spelling, _v0)
if [ -f "$FINNGEN_R13_FILE" ]; then ARGS="$ARGS --finngen-r13-path $FINNGEN_R13_FILE"; echo "==> FinnGen R13 wired: $FINNGEN_R13_FILE" | tee -a "$LOG"; else echo "==> ABORT: FinnGen R13 file missing: $FINNGEN_R13_FILE" | tee -a "$LOG"; exit 7; fi

# --- Run 17 EVE/ESM-2 wiring (HGVSp parser delivered -> EVE/ESM-2 now carry REAL
#     signal). Plus omim/phylop/dbsnp/clingen, whose CLI flags exist but the launch
#     script never passed (silent-zero). Hard-fail if a configured source is missing
#     on the VM; each echoes the exact file picked (a wrong pick is LOUD, not silent).
# EVE: directory of per-protein score CSVs (gene_symbol + HGVSp-derived aa_change).
# The 3,211 score CSVs live in EVE_all_data/variant_files (NOT the bundle root,
# which has 0 top-level CSVs). EVE's glob is non-recursive, so point at the leaf
# dir and ABORT on 0 CSVs -- the old `ls -A` check passed on the CSV-less bundle
# root and would have silently scored every variant 0.5.
EVE_DIR="$DATA/external/eve/EVE_all_data/variant_files"
_EVE_CSVN=$(ls "$EVE_DIR"/*.csv 2>/dev/null | wc -l)
if [ -d "$EVE_DIR" ] && [ "$_EVE_CSVN" -gt 0 ]; then
    ARGS="$ARGS --eve-path $EVE_DIR"; echo "==> EVE wired: $EVE_DIR ($_EVE_CSVN CSVs)" | tee -a "$LOG"
else
    echo "==> ABORT: EVE variant_files missing or no CSVs: $EVE_DIR ($_EVE_CSVN found; expected ~3211). Stage variant_files to the VM." | tee -a "$LOG"; exit 8
fi
# ESM-2 UniProt sequence index (offline; else slow live REST per gene).
UNIPROT_INDEX="$DATA/external/uniprot/uniprot_human_reviewed.parquet"
if [ -f "$UNIPROT_INDEX" ]; then
    ARGS="$ARGS --esm2-uniprot-index $UNIPROT_INDEX"; echo "==> ESM-2 UniProt index wired: $UNIPROT_INDEX" | tee -a "$LOG"
    ARGS="$ARGS --eve-entry-map $UNIPROT_INDEX"; echo "==> EVE entry-name map wired: $UNIPROT_INDEX (resolves 1433G_HUMAN -> YWHAG)" | tee -a "$LOG"
else
    echo "==> ABORT: UniProt index missing: $UNIPROT_INDEX" | tee -a "$LOG"; exit 8
fi
# OMIM: prefer a mim2gene file (OMIMConnector(mim2gene_path=...)); else first file.
OMIM_FILE="$(ls "$DATA"/external/omim/*mim2gene* 2>/dev/null | head -n1 || true)"
if [ -z "$OMIM_FILE" ]; then OMIM_FILE="$(ls "$DATA"/external/omim/* 2>/dev/null | grep -v -i 'readme\|checksum\|md5' | head -n1 || true)"; fi
if [ -n "$OMIM_FILE" ] && [ -f "$OMIM_FILE" ]; then
    ARGS="$ARGS --omim-path $OMIM_FILE"; echo "==> OMIM wired: $OMIM_FILE" | tee -a "$LOG"
else
    echo "==> ABORT: OMIM file missing under $DATA/external/omim/" | tee -a "$LOG"; exit 8
fi
# OMIM genemap2: the SOLE source for omim_n_diseases / omim_n_diseases_molecular /
# omim_is_autosomal_dominant after the connector rewrite (mim2gene is inert now).
# Without --omim-genemap2-path, all three OMIM columns silent-zero across the cohort.
OMIM_GENEMAP2_FILE="$(ls "$DATA"/external/omim/*genemap2* 2>/dev/null | head -n1 || true)"
if [ -n "$OMIM_GENEMAP2_FILE" ] && [ -f "$OMIM_GENEMAP2_FILE" ]; then
    ARGS="$ARGS --omim-genemap2-path $OMIM_GENEMAP2_FILE"; echo "==> OMIM genemap2 wired: $OMIM_GENEMAP2_FILE" | tee -a "$LOG"
else
    echo "==> ABORT: OMIM genemap2.txt missing under $DATA/external/omim/ (omim_n_diseases/omim_n_diseases_molecular/omim_is_autosomal_dominant would silent-zero)" | tee -a "$LOG"; exit 8
fi
# PhyloP: single source file.
PHYLOP_FILE="$(ls "$DATA"/external/phylop/* 2>/dev/null | grep -v -i 'readme\|checksum\|md5' | head -n1 || true)"
if [ -n "$PHYLOP_FILE" ] && [ -f "$PHYLOP_FILE" ]; then
    ARGS="$ARGS --phylop-path $PHYLOP_FILE"; echo "==> PhyloP wired: $PHYLOP_FILE" | tee -a "$LOG"
else
    echo "==> ABORT: PhyloP file missing under $DATA/external/phylop/" | tee -a "$LOG"; exit 8
fi
# dbSNP: DbSNPConnector(parquet_path=...) wants a parquet.
DBSNP_FILE="$(ls "$DATA"/external/dbsnp/*.parquet 2>/dev/null | head -n1 || true)"
if [ -n "$DBSNP_FILE" ] && [ -f "$DBSNP_FILE" ]; then
    ARGS="$ARGS --dbsnp-path $DBSNP_FILE"; echo "==> dbSNP wired: $DBSNP_FILE" | tee -a "$LOG"
else
    echo "==> ABORT: dbSNP parquet missing under $DATA/external/dbsnp/" | tee -a "$LOG"; exit 8
fi
# AlphaFold (Phase D): cohort structural-feature parquet + UniProt index for the
# gene->accession map and wt_aa cross-check. Flag exists in run_phase2_eval.py; the
# connector defaults to sentinel stubs if unpassed, so wire it explicitly and ABORT
# loudly if the built parquet is missing (a wrong/absent pick must be LOUD, not silent).
ALPHAFOLD_FILE="$(ls "$DATA"/external/alphafold/*.parquet 2>/dev/null | head -n1 || true)"
if [ -n "$ALPHAFOLD_FILE" ] && [ -f "$ALPHAFOLD_FILE" ]; then
    ARGS="$ARGS --alphafold-path $ALPHAFOLD_FILE"; echo "==> AlphaFold wired: $ALPHAFOLD_FILE" | tee -a "$LOG"
    ARGS="$ARGS --alphafold-uniprot-index $UNIPROT_INDEX"; echo "==> AlphaFold UniProt index wired: $UNIPROT_INDEX" | tee -a "$LOG"
else
    echo "==> ABORT: AlphaFold parquet missing under $DATA/external/alphafold/ (build with scripts/build_alphafold_parquet.py)" | tee -a "$LOG"; exit 8
fi
# ClinGen: Gene-Disease Validity CSV (flag existed but launch never passed it -> silent 0).
CLINGEN_FILE="$(ls "$DATA"/external/clingen/*.csv 2>/dev/null | head -n1 || true)"
if [ -n "$CLINGEN_FILE" ] && [ -f "$CLINGEN_FILE" ]; then
    ARGS="$ARGS --clingen-path $CLINGEN_FILE"; echo "==> ClinGen wired: $CLINGEN_FILE" | tee -a "$LOG"
else
    echo "==> ABORT: ClinGen CSV missing under $DATA/external/clingen/" | tee -a "$LOG"; exit 8
fi
# end Run 17 EVE/ESM-2 wiring

ARGS="$ARGS --string-db auto"
ARGS="$ARGS --hetero-gnn --kg-edges reactome:$REACTOME_GMT"
ARGS="$ARGS --min-review-tier 3 --n-folds 5"
ARGS="$ARGS --skip-svm"
# LOVD is ON-if-present (B9): absence must never silently zero it, but it is optional (low coverage).
LOVD_PARQUET="$DATA/external/lovd/lovd_all_variants.parquet"
if [ -f "$LOVD_PARQUET" ]; then
    ARGS="$ARGS --lovd-path $LOVD_PARQUET"
    echo "==> LOVD wired: $LOVD_PARQUET" | tee -a "$LOG"
else
    echo "==> LOVD absent ($LOVD_PARQUET); proceeding without it (B9 if-present)" | tee -a "$LOG"
fi
ARGS="$ARGS --unseen-gene-holdout"
ARGS="$ARGS --output $OUTDIR"
echo "==> rnaseq wired: $RNASEQ_PARQUET" | tee -a "$LOG"
echo "==> kg wired: $KG_PARQUET" | tee -a "$LOG"
echo "==> hetero-GNN edges: reactome:$REACTOME_GMT" | tee -a "$LOG"
echo "==> NOTE: ESM-2/EVE ACTIVE (HGVSp parser delivered; protein_pos/wt_aa/mut_aa populated for missense)" | tee -a "$LOG"
echo "==> ARGS: $ARGS" | tee -a "$LOG"

( sleep 2700
  if ls "$OUTDIR"/models/*.joblib >/dev/null 2>&1; then
      echo "==> CHECKPOINT SENTINEL @ T+45min: checkpoints present -- OK." | tee -a "$LOG"
  else
      echo "==> CHECKPOINT SENTINEL @ T+45min: NO $OUTDIR/models/*.joblib yet -- investigate." | tee -a "$LOG"
  fi
) &
SENTINEL_PID=$!

echo "==> ALL PREFLIGHT PASSED. Launching @ $(date -u +'%F %T') UTC" | tee -a "$LOG"
set +e
python scripts/run_phase2_eval.py $ARGS 2>&1 | tee -a "$LOG"
RUN_RC=${PIPESTATUS[0]}
set -e
kill "$SENTINEL_PID" 2>/dev/null || true

echo "==> run_phase2_eval.py rc=$RUN_RC @ $(date -u +'%F %T') UTC" | tee -a "$LOG"

echo "==> Post-run artifact check" | tee -a "$LOG"
for f in \
    "$OUTDIR/metrics.json" \
    "$OUTDIR/per_model_metrics.csv" \
    "$OUTDIR/per_model_metrics_val.csv" \
    "$OUTDIR/oof_predictions.parquet" \
    "$OUTDIR/feature_importance.csv" \
    "$OUTDIR/models/ensemble.joblib" \
    "$OUTDIR/models/ensemble.manifest.json" \
; do
    if [ -f "$f" ]; then echo "==> VERIFIED: $f" | tee -a "$LOG"; else echo "==> MISSING: $f" | tee -a "$LOG"; fi
done
N_MODELS=$(ls "$OUTDIR/models/"*.joblib 2>/dev/null | wc -l || echo 0)
echo "==> model .joblib count: $N_MODELS" | tee -a "$LOG"
exit $RUN_RC
