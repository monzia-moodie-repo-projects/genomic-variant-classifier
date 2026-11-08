"""The reviewed TRANSITION policy of the prediction contract (package 1) -- which entrypoints are refused, and why,
while their migration is pending.

Author: Monzia Moodie

Owner ruling of 2026-11-07 (decision 38f15c7a): package 1 is delivered as three sequential guarded installers, and
"each installer must leave every affected, reachable operation either correct or explicitly refused". Set 1a removed the
logistic meta-learner from InferencePipeline (it is not the selected predictor) and made the stored convex blend, under
the exact once-rounded kernel, the only authoritative score. The entrypoints below still depend on what was removed, or
would publish results attributed to a different predictor, so they REFUSE until the named set migrates them.

Dependency-free on purpose (like quarantine_policy.py): each refused script imports this module and
containment.require_execution_permitted as its FIRST statements, before any model library, artifact, data file or
output is touched. Kept separate from quarantine_policy.HISTORICAL_EXECUTION_DENIED so the quarantine policy -- and the
digest admission decisions bind to (model_admission.quarantine_policy_sha256) -- is unchanged by a transition that has
nothing to do with the quarantine.

Removing an entry is the enabling package's job, done together with the migration and its tests; never an edit here
alone.
"""
from __future__ import annotations

from types import MappingProxyType

#: Script repository path -> "REASON_CODE (enabling set): why".
PREDICTION_CONTRACT_TRANSITION_DENIED = MappingProxyType({
    "scripts/calibrate_thresholds.py":
        "THRESHOLD_CONSUMER_MIGRATION_REQUIRED (enabled by package 1, set 1c): this threshold workflow does not yet "
        "consume the selected prediction contract -- it rebuilt member predictions itself, scored them with the "
        "logistic meta-learner (removed from InferencePipeline in set 1a: not the selected predictor) and fitted a "
        "probability transformation whose score stage the thresholds would silently inherit",
    "scripts/run9_ablations.py":
        "ABLATION_WRITER_MIGRATION_REQUIRED (enabled by package 1, set 1b): this harness re-predicts every member "
        "independently of the selected predictor's bound member capture, writes predictions without unit identities, "
        "and its explanation writer (RunArtifactWriter.save_shap_values) explained the logistic meta-learner rather "
        "than the selected blend; a run could not be represented faithfully as complete evidence",
    "scripts/export_phase2_predictions_from_saved_models.py":
        "SCORE_RECONSTRUCTION_REFUSED (disposition decided in package 1, set 1c): this exporter reconstructs member "
        "scores from saved models, DIVIDES THE STORED WEIGHTS BY THEIR SUM and writes the result as prediction files "
        "-- a different predictor from the selected blend (no normalisation; exact kernel) -- and its AUROC "
        "agreement check is not prediction identity. It re-predicts the quarantine-era run "
        "outputs/rnaseq_pred_write_smoke, the same run whose diagnostic is already historical-execution denied",
})

#: Operations inside a module that refuse while the module itself stays importable (reason codes as above).
EXPLANATION_TARGET_MIGRATION_REQUIRED = (
    "EXPLANATION_TARGET_MIGRATION_REQUIRED (enabled by package 1, set 1b): RunArtifactWriter.save_shap_values "
    "explained ensemble.meta_learner -- the logistic research candidate -- not the selected convex blend. An "
    "explanation of another predictor is refused, not written; a valid blend explanation (member-score accounting "
    "phi_ij = w_j (p_ij - mu_j) with a recorded closure residual) replaces it in set 1b.")
