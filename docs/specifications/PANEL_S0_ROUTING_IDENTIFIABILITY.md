Panel S0 -- Expert Identity, Routing Relevance, and Mechanistic Interpretive Admissibility

This panel governs whether any quantity produced by a Mixture-of-Experts (MoE)
component of the classifier may be read as evidence about biology. It is the first
panel in the Expert-Systems series (S0 through S8) and it governs INTERPRETATION,
not execution: the operational panels S1 through S8 still compute their metrics on
whatever MoE exists, but no mechanistic statement derived from those metrics is
admissible unless Panel S0 is satisfied for the specific expert and the specific
claim. Panel S0 does not measure predictive accuracy, and a strong S0 result does
not make a better classifier. The final MoE prediction may be developed, deployed,
and clinically evaluated while every expert remains mechanistically inadmissible
under S0. Those two tracks -- predictive deployment and mechanistic interpretation
-- carry separate validation states and are never collapsed into one.

The single most important boundary in this panel is between model reliance and
biological causation. Every intervention Panel S0 can perform is an intervention on
the MODEL -- removing an expert, changing an allocation, replacing an expert
output, masking an input. These establish that the model relies on a component.
They do not, and cannot, establish that a biological mechanism causally mediates
pathogenicity, because no model-component intervention intervenes on splicing, on
protein stability, on expression, or on disease. Biological mediation is therefore
outside Panel S0 entirely and is deferred to a future Panel T (see S0.17). Panel S0
establishes association and computational necessity; it never establishes
biological causation.

S0.0 Scope, non-claims, and terminology

This is a specification. No production MoE model exists in the codebase at the time
of writing, and none is authorised by this document. The data and representation
prerequisites for a biological expert -- a populated data/external tree, a
Human-Genome-Variation-Society protein-consequence (HGVSp) parser, activated
Evolutionary-Scale-Modeling protein language model (ESM-2) and Nucleotide
Transformer encoders, exported graph-neural-network embeddings, a Joint-Embedding
Predictive Architecture (JEPA) representation, and acquired experimental anchor
targets -- do not yet exist. Panel S0 is written now so that the interpretive gate
is designed before any expert is built, exactly as the Panel R matched-null
protocol was designed and calibrated before any real genomic representation was
admitted. Every routing and expert-identity capability defined here begins its life
in the capability state NOT_IMPLEMENTED or OUTPUT_AVAILABLE, with mechanistic
admissibility NOT_ADMISSIBLE.

Panel S0 does not claim that any expert corresponds to a biological mechanism; it
specifies the conditions under which an associational or computational claim about
an expert could later be made. It does not claim that whitening, JEPA, or any
foundation model improves biology. It does not license a mechanistic report label
for any expert at the time of writing, because no expert exists and no anchor has
been acquired. It does not treat predictive success as mechanistic evidence: an MoE
can be predictively excellent and mechanistically mute. And it does not establish
biological mediation under any circumstances, which is the province of Panel T.

S0.1 Equivalence class and inferential target

The inferential goal of this panel is precisely stated. It is not identifiability
in the strict statistical sense, which learned-router mixtures do not possess. It
is EMPIRICAL INTERPRETIVE ADMISSIBILITY under a declared equivalence class: given an
explicit account of the symmetries and gauge freedoms the model admits, and given
held-out, matched-null, confounder-controlled evidence, may a named associational
or computational reading of a specific expert be reported. The panel makes the
conditions for that reading explicit and refuses the reading until they are met.

Four non-identifiabilities motivate the whole design. Label switching: any common
permutation of the expert outputs and the router coordinates leaves the prediction
unchanged, so the model identifies an unordered collection of components, and the
biological name attached to an expert index is imposed by the developer, not
learned. Gauge freedom: deep expert parameterisations and router normalisations
admit multiple internal configurations that produce the same predictions, so a
routing "confidence" is not a scale-identified quantity on its own. Shortcut
routing: ordinary training does not force experts to specialise along biologically
meaningful axes, and a router can reach an apparently biological partition through a
non-causal correlate -- the project's own n_pathogenic_in_gene history, where a top
feature encoded gene prevalence and needed a permutation ablation to establish
variant-level content, is the precedent. Compositional dependence: a softmax
allocation is simplex-constrained, so raising one expert's weight necessarily lowers
the others even when two mechanisms are simultaneously present, which means the
allocation weight conflates "this mechanism is strongly relevant" with "this
mechanism is more relevant than the others."

The consequence is a direct analogy to Panel R stage R3. There, an angular recovery
statistic looked like representation recovery but was alignment-blind, and a
separate alignment-sensitive estimand had to be constructed to ask the intended
question. Here, the softmax allocation weight looks like a mechanism attribution but
is the wrong object, and separate, purpose-built quantities must be constructed
instead. The same discipline -- a held-out, matched-null, confounder-controlled
admissibility gate on the correct estimand -- resolves both.

S0.2 Four quantities: allocation, mechanism evidence, expert utility, reliability

The central structural commitment of this panel is that a single routing number
must be decomposed into four quantities that answer four different questions and
carry different validation statuses. Collapsing any two of them is the root of the
interpretability hazard.

Allocation a_e(x) is the softmax mixing weight. It is an engineering quantity: it
governs how the prediction is composed and how compute is spent. It is
simplex-constrained and compositionally dependent, and it is never, on its own,
mechanistic evidence.

Mechanism evidence m_e(x) is an independent, unconstrained per-expert quantity in
the unit interval, produced by a dedicated head rather than a shared softmax, that
estimates whether mechanism e is active in or compatible with the observation. Two
mechanisms can both have high mechanism evidence for the same variant without
competing for a fixed budget. Mechanism evidence is the scientific quantity, and
only a calibrated m_e is eligible for a mechanistic reading.

Expert utility u_e(x) is the expected marginal predictive benefit of using expert e
for the observation. It is distinct from mechanism evidence: a mechanism can be
genuinely present while its expert is poorly trained (high m_e, low u_e), and an
expert can improve prediction through a proxy while the mechanism is absent (low
m_e, high u_e). The 2x2 of m_e and u_e distinguishes mechanism-supported-and-useful,
mechanism-supported-but-expert-inadequate, expert-exploiting-a-proxy, and
neither-supported -- distinctions a single "relevance" score cannot make.

Reliability q_e(x) is the trustworthiness of the evidence and representation
supporting expert e for the observation, and it is a RUNTIME quantity: it uses only
information available at inference. It is defined in S0.4 with a declared estimand
and an explicit exclusion of anchor availability.

A principled allocation rule may combine utility and reliability with the base
router logit, for example a_e(x) = softmax over experts of the router logit plus a
weighted logit of u_e(x) plus a weighted logit of q_e(x), with the weights frozen
on the tuning partition. Mechanism evidence m_e must NOT enter the allocation, so
that the scientific quantity never competes for mixing mass. The gates are
independent: allocation utility (S0.9), mechanism-evidence validity (S0.8), and
expert identity (S0.7) each stand alone, and a mechanistic reading requires all that
apply. An implementation that reports only the softmax allocation has produced no
mechanism-evidence quantity at all and is categorically ineligible for mechanistic
interpretation, however strong its predictions.

S0.3 Expert identity classes

Not every expert is a candidate mechanism, and forcing every expert to carry a named
biological target is itself an error -- it presses free experts into imitating
incomplete or noisy prior knowledge. Panel S0 recognises four identity classes, and
the mechanistic-admissibility gate applies only to the first.

ANCHORED_MECHANISM. An expert with an externally specified mechanistic estimand -- a
measured splice outcome, a measured protein abundance or stability, a measured
functional activity, a measured regulatory or binding change. Only this class is
eligible for a mechanism-compatible report label, and only after the full gate.

PREDICTIVE_UNNAMED. An expert permitted to discover a useful partition and improve
prediction, reported as latent expert A, latent expert B, and so on. It receives no
biological name until Panel S0 establishes a reproducible biological meaning for it.

GENERALIST. Always eligible, responsible for shared predictive structure common
across variants. It absorbs common capability and reduces routed-expert redundancy;
it is never given a mechanism name.

RESIDUAL. Always eligible, representing unsupported mechanisms, poor anchor
coverage, multimodal disagreement, out-of-distribution biology, and technical
failure not yet distinguished from novelty. It must never be called a
novel-mechanism expert. High routing to the residual expert marks a variant as a
candidate for later investigation, not as a discovery.

S0.4 Anchor tier, independence, provenance, and the reliability estimand

A mechanistic name is only as strong as the independence of the evidence anchoring
it, and evidence tier and independence are two different axes that must not be
conflated. A high-tier assay used both to train and to validate an expert is not
independent evidence for that expert; a lower-tier source from a genuinely separate
process may be more probative.

The evidence tiers, strongest first: DIRECT_EXPERIMENTAL, a measurement of the
mechanism itself; ORTHOGONAL_EXPERIMENTAL, an independent readout through different
assay physics; HUMAN_ASSOCIATION, segregation or case-control or other genetic
association; CURATED_MECHANISM, expert-curated annotation; COMPUTATIONAL_TEACHER,
the prediction of another computational model; HEURISTIC_PRIOR, a hand-built prior.

The independence profile, recorded per anchor and per expert, asks separately
whether the anchor or a direct proxy was part of the expert's training signal or the
router's input; whether it shares a data-generating process, submitter, curation
pipeline, or computational ancestor with the training labels; and whether it covers
the genes and variant classes on which the claim is to be made. A provenance audit
is transitive: a derived feature -- an aggregate, a graph statistic, an embedding --
computed from the anchor counts as the anchor for leakage purposes, and the audit
must trace it. An anchor high in tier but low in independence supports at most a
provisional claim.

The critical rule concerns computational teachers. A predicted score such as
SpliceAI is a COMPUTATIONAL_TEACHER, not an independent mechanism anchor. SpliceAI
may be an input feature, a distillation teacher, weak supervision, a routing prior,
or a synthetic development target, but anchoring an expert to it establishes only
that the expert reproduces SpliceAI's decision surface, including its errors. Such
an expert may be called a SpliceAI-distillation expert; it may not be called a
splice-mechanism expert unless separately validated against measured splicing.
Independent sources include multiplexed assays of variant effect, measured splicing
assays, abundance assays such as Variant-Abundance-by-Massively-Parallel-sequencing
(VAMP-seq), deep-mutational-scanning functional scores, reporter assays, and binding
or enzymatic assays; the Multiplexed-Assay-of-Variant-Effect database (MaveDB) is
the canonical store of such scores with experimental metadata. A computational
prediction may serve as a weak anchor, recorded as weak_computational with
mechanistic_admissibility false.

The reliability quantity q_e requires its own definition and its own estimand, and
it must be strictly separated into two non-overlapping groups. RUNTIME evidence
reliability uses only inference-time information: modality presence, sequence
mapping quality, transcript ambiguity, graph coverage, representation
out-of-distribution distance, assay-like feature quality, and missingness pattern.
EVALUATION support uses information available only for scientific validation: anchor
observed, anchor assay quality, anchor tier, anchor independence, and observation
propensity. Evaluation support must NEVER be fed into the runtime reliability head.
In particular, "whether the anchor is defined" is evaluation support, not runtime
reliability: a deployment variant has no anchor, and a runtime head keyed on anchor
availability would learn the assay-selection process -- the exact verification bias
of S0.5 -- and would suffer train-serve skew. Reliability is not an opaque learned
scalar; it has a declared estimand chosen from expected expert-specific error,
out-of-distribution probability, modality adequacy, and mapping confidence, and a
reliability output may carry several such components, each with its own target,
rather than one uninterpretable number.

S0.5 The anchor observation process and transportability

An anchor is not a random sample of the genome. Assays are performed on the genes
and variants investigators chose to study, for reasons correlated with prior belief
about pathogenicity, gene importance, and tractability. An expert trained against
such an anchor can learn which variants were selected for assay rather than which
use the mechanism, and a router can allocate on the same selection signal. This is
verification bias, and it produces a biological-looking result with no mechanistic
content.

Panel S0 therefore requires, for every ANCHORED_MECHANISM claim, an explicit model
of the anchor observation process: which genes and variant classes are covered, what
selection produced that coverage, and whether the coverage overlaps the population on
which the claim is made. A positivity diagnostic is mandatory -- there must be
adequate anchor coverage across the strata of the claim, and strata without coverage
are excluded from the claim rather than silently extrapolated into. Where selection
is informative, the analysis reweights toward the claim population through
inverse-probability weighting or an equivalent doubly-robust adjustment, and reports
the sensitivity of the conclusion to that reweighting.

Weighting does not solve everything, and the specification says so plainly.
Inverse-probability weighting identifies the target only under selection-on-observ
ables, positivity, and adequate model specification. Where anchor observation is
plausibly missing-not-at-random because of unmeasured scientific interest, weighting
on observed variables does not identify the target estimand; the panel then requires
a sensitivity analysis over plausible unmeasured selection, or restricts the claim to
the observed anchor-support population. The required diagnostics are propensity
overlap, maximum weight, effective sample size, weight-truncation sensitivity,
weighted covariate balance, a negative-control selection variable, and bounds under
plausible unmeasured selection. Often the honest claim population is "variants
resembling the assay-supported population," not all genomic variants.

S0.6 Claim population and partition and cross-fitting policy

Establishing expert identity, mechanism evidence, routing relevance, and utility is
a model-selection activity that chooses anchors, alignment weights, thresholds,
null constructions, and interpretations. Performing those selections on the locked
test partition is selection on test.

Therefore expert alignment, thresholding, null calibration, confounder-role
classification, and any interpretation occur on the STRUCTURE partition, which is
gene-disjoint from train, tune, probability calibration, conformal calibration, and
test. The test partition admits only a predeclared replication: freeze the experts,
router, alignment map, anchor set, null construction, and thresholds on STRUCTURE,
then evaluate the prespecified S0 statistics once on test. No alignment, threshold
selection, or interpretation occurs on test, and no method selection occurs on an
external cohort. The router must never receive the anchor target or a direct proxy
of it as an input; an expert may be trained against an anchor, but a router that can
read the anchor has copied the answer rather than learned to route by mechanism.

Cross-fitting is mandatory and covers every nuisance model, not only the final
estimator. The selection or propensity model, the calibration map, the relevance
and mechanism-evidence models, the anchor measurement model, the matching strata,
the confounder-adjustment model, and the utility estimator are each fit on rows
disjoint from those used to estimate the effect they feed. Folds operate at the gene
level: outer gene folds carry scientific effect estimation, inner gene folds carry
router and expert tuning and nuisance fitting. For external replication every
component is frozen.

S0.7 The Expert Identity Gate

The Expert Identity Gate (S0-I) is not satisfied by an expert's auxiliary head
fitting its anchor. Three tests are required, and each is stated in the four-quantity
vocabulary so that no test silently mixes allocation with mechanism evidence.

Anchor prediction: the expert representation predicts the anchor target on held-out
genes. Conditional comparative advantage: the expert outperforms the generalist, the
other experts, and a parameter-matched dense branch specifically on anchor-relevant
observations, which is the difference-in-differences estimand of S0.12 -- an expert
can pass anchor prediction yet fail this if every expert learned the same target.
Faithfulness: the expert's mechanism evidence must be coupled to what the expert
actually contributes, so that the mechanism-evidence head is an interpretation of
this expert rather than a free-standing classifier. The faithfulness statistic is
the cross-fitted, gene-cluster correlation between m_e(x) and the expert's marginal
utility Delta-L_e(x) = L-without-expert-e(x) minus L-with-the-full-model(x) on
anchor-relevant cases; mechanism evidence that does not predict the expert's own
contribution does not identify the expert.

S0.8 The Mechanism-Evidence Gate

The Mechanism-Evidence Gate (S0-R) asks whether high mechanism evidence m_e(x)
coincides specifically with the presence of mechanism e, rather than with a
confound. It requires that a calibrated m_e track independent mechanism measurements
on held-out genes, that its agreement with the biological anchor survive the
confounder comparison of S0.10, and that its calibration hold in the sense of S0.18
before any probabilistic language is used. Mechanism evidence is associational: it
supports statements that the mechanism is present or compatible, never that it
mediates pathogenicity.

S0.9 The Expert-Utility and Allocation Gates

Two engineering gates stand apart from the scientific ones. The expert-utility gate
(S0-U) asks whether u_e(x) predicts the observed marginal benefit of expert e --
whether retaining or upweighting the expert conditionally reduces loss on the cases
where utility is claimed. The allocation gate (S0-A) asks whether the allocation
a_e(x) improves the final task, a utilisation-and-capacity question. Neither gate is
mechanistic: passing them establishes that an expert earns its compute, not that it
represents biology. They are reported separately from S0-I and S0-R so that a useful
expert with a wrong mechanism label fails the scientific gates while passing the
engineering ones, and the discrepancy is visible rather than averaged away.

S0.10 Confounder roles and the conditional paired gate

Not every associated variable should be adjusted away, so the confounder analysis
begins with a role registry, not a flat list. Each covariate is classified as a
technical confounder, a design variable, biological context, a mediator candidate, a
collider risk, a target proxy, or a selection variable, and an unclassified variable
fails closed. A versioned covariate-role manifest, a causal-role diagram, precedes
STRUCTURE analysis, because adjusting for a mediator or conditioning on a collider
would itself bias the result. The enumerated project covariates -- gene identity,
ClinVar submitter, review status, annotation source, variant-consequence label,
modality missingness, transcript availability, graph degree, direct gene-disease
priors, and pathogenic-count aggregates -- are assigned roles in that manifest
rather than uniformly "adjusted."

The gate itself is a PAIRED, CONDITIONAL, gene-cluster comparison, not a comparison
of separately estimated marginal intervals. For the primary biological anchor B and
a technical covariate T, the quantities are computed on a common scale --
cross-validated predictive information gain, the reduction in a proper scoring loss
from adding the routing quantity, normalised by the null loss, so a continuous
anchor and a categorical laboratory label are comparable. The marginal paired
difference A(r,B) - A(r,T) is estimated within the same gene-cluster bootstrap
resamples, so the positive correlation between the two agreement statistics tightens
the difference rather than being ignored, and requiring separate intervals to be
disjoint would be needlessly conservative. Beyond the marginal difference, the gate
requires CONDITIONAL information: biological incremental information after technical
covariates, A(r, B given T), must be positive and clear the dominance margin, while
technical incremental information after the biological anchor, A(r, T given B), must
fall below a declared margin. Conditioning is what catches a relevance score that
tracks biology only because biology and the technical covariate are collinear. The
whole family of technical covariates is Holm-corrected, and if routing is continuous
the agreement statistic is a predeclared association measure fixed before the
analysis. No mechanistic routing claim is admissible if a technical covariate ties
or beats the biological anchor under this paired, conditional, corrected comparison.

S0.11 Invariance controls and matched-null families

A null for routing must match the exact claim, exactly as the R3 matched-spectrum
null preserved the gain spectrum while scrambling gain-to-direction alignment. Panel
S0 separates an invariance CONTROL, which is deterministic and structural, from an
inferential null, which is stochastic and calibrated.

S0-C1, the expert-index permutation invariance control, permutes expert identities
after predictions are fixed. It is not a biological null; it demonstrates a
structural symmetry. Its expected result is exact: predictions unchanged,
raw-expert-index interpretation invalid, aligned identity restored. It keeps the
label-switching symmetry conceptually separate from any biological evidence.

S0-N1b, the anchor-assignment null, is inferential. It builds an expert-anchor
evidence matrix S_ea whose components may include held-out anchor prediction,
conditional comparative advantage, intervention specificity, mechanism-evidence
calibration, and negative-control exclusion, then scores the intended assignment as
the sum over experts of S at the intended expert-anchor pairing and compares it
against alternative one-to-one assignments under an assignment null. If expert
identities were prespecified during training, the intended assignment must not be
scored by the same auxiliary training loss alone, or it wins by construction.

S0-N2, the blocked-route-reassignment null, reassigns whole-gene route vectors among
eligible gene blocks within matched strata. Because anchor availability is selective,
it must preserve or balance not only loads, entropy, eligibility, gene blocks, label
prevalence, and missingness, but also anchor-observation propensity, assay family,
assay laboratory, source, and claim-population propensity. Exact preservation is
frequently infeasible, so every draw carries a feasibility diagnostic reporting the
residual imbalance in each preserved quantity; a draw exceeding tolerance is
rejected and redrawn, and if no acceptable matched ensemble can be assembled the null
returns INSUFFICIENT_SUPPORT rather than a mis-calibrated p-value. S0-N2 is therefore
an honest constrained matched-reassignment null, not an exact permutation null, and
it additionally reports the number of unique feasible assignments, the effective
Monte Carlo support, the duplicate-null fraction, and the minimum attainable
p-value, so a tail probability is never reported at a resolution the support cannot
sustain. Every S0 statistic-and-null pairing is subject to the compatibility
contract from the alignment work: a statistic that responds only to a quantity the
null preserves cannot gain power from that null, and pairing them must raise.

S0.12 The specialisation estimand

An expert that is merely a strong generalist must not pass as a specialist. The
specialisation estimand is a difference in differences: the expert's held-out
advantage over the generalist on anchor-relevant observations, minus its advantage
over the generalist on matched non-anchor observations. A pure generalist has equal
advantage on both and a specialisation of zero. The estimand is computed with
cross-fitting so that the rows used to fit the expert and router are disjoint from
those used to estimate the advantage, and it must exceed the matched-null
distribution of the same difference-in-differences quantity with a gene-cluster
confidence interval excluding zero.

S0.13 The model-component intervention taxonomy

Enrichment is correlational; a computational-necessity claim additionally requires a
controlled intervention. Every intervention here is an intervention on the MODEL, and
it establishes model reliance, never biological causation. The taxonomy has four
distinct, non-interchangeable estimands.

S0-I1, allocation intervention: set the allocation a_e(x) to zero and redistribute
the removed allocation under a declared policy. This tests computational reliance on
the expert in the final mixture. Because the redistribution policy affects the
result, the intervention reports the sensitivity of its contrast to that policy; a
contrast that exists only under one renormalisation is not a necessity result.

S0-I2, expert-function intervention: hold allocation fixed and replace the expert
output f_e(x) with the generalist output, a matched null-expert output, a zero
residual, or another eligible expert. This tests whether the expert function carries
unique useful information.

S0-I3, relevance-head intervention: perturb or suppress the mechanism-evidence or
relevance head. If that head is purely observational and does not feed allocation or
reporting, suppressing it should not change predictions -- a desirable architectural
test that the scientific head is not silently steering the mixture.

S0-I4, input or modality intervention: alter the expert's inputs or evidence
availability. This tests modality dependence, which is not by itself expert
necessity.

For each intervention the reported quantity is the interaction contrast I_e, the
difference between anchor-case degradation and matched control-case degradation, with
a gene-cluster confidence interval excluding zero. The expected pattern for an expert
the model genuinely relies on for a mechanism is a larger degradation on
anchor-relevant cases than on matched control cases, stable across intervention types
and redistribution policies. These interventions establish computational necessity;
they are the evidence for the EXPERT_COMPUTATIONAL_NECESSITY and
PATHOGENICITY_RELEVANCE claims of S0.17, and they are never evidence for biological
mediation.

S0.14 Alignment, unmatched experts, and lineage

Because expert indices are not identifiable across runs, experts are aligned before
their stability is measured, but the signature used for alignment must exclude the
very external quantity whose stability is later claimed, or the conclusion is
circular. The signature is therefore tiered. A core signature, used for alignment, is
built only from quantities frozen on the tuning partition: anchor-target performance
on tune, the cross-fitted tune routing distribution, routing by eligibility class,
and output residual structure on tune. An extended signature, which includes external
enrichment, test-set routing, and intervention effects, is used only for reporting
after alignment, never to drive it; aligning experts on external enrichment and then
claiming external stability would leak the conclusion into its own premise.

Alignment is not assumed one-to-one. Across runs or releases an expert may split into
two, two may merge, one may die, and counts may differ, so the assignment permits
dummy experts with an unmatched penalty, a maximum acceptable alignment cost, an
explicit unmatched status, and, for split-and-merge analysis, unbalanced optimal
transport. A high-cost one-to-one match is never forced merely because a Hungarian
assignment always returns one.

Every expert carries an immutable identifier and a versioned lineage recording its
model release, parent experts, lineage event (created, continued, split, merged,
retired, identity-revised), and the hashes of its anchor manifest and expert
specification. Mechanistic admissibility does not silently transfer across a split, a
merge, an anchor change, or an architecture change; a lineage event resets the claim
ledger to require re-validation.

Only after alignment is routing stability measured, at the distributional,
observation, and gene levels, with the lower-tail and minimum-subgroup stability
reported alongside the median, so a stable average cannot hide a route unstable
specifically for splice or noncoding variants.

S0.15 Multi-anchor triangulation

A single target rarely defines a mechanism. Measured stability does not fully capture
protein destabilisation: assays are context-dependent, stability and abundance are
related but distinct, some pathogenic effects preserve stability, and gain-of-
function need not appear as destabilisation. A stronger identity triangulates across
partially independent anchors -- a primary measured target, a secondary related
measurement, an orthogonal assay, and a negative-control anchor from an unrelated
mechanism -- and is stronger when the primary and secondary agree, the negative
control does not, and the effect transfers across genes and assays.

The mature form of triangulation is a measurement model rather than a checklist. A
latent mechanism activity M_e is posited, and each measured anchor loads on it as
Y_ea = lambda_ea times M_e plus assay-specific error, which accommodates
assay-specific noise, differing scales, partial overlap, correlated assay errors,
anchor discordance, and uncertainty in the latent state, with the negative control
expected to load near zero. A hierarchical Bayesian or confirmatory latent-variable
model then estimates mechanism evidence without treating any one imperfect assay as
truth, and distinguishes genuine mechanism disagreement from assay noise, context
dependence, and laboratory effects. The measurement model is the preferred mature
implementation; it is an advanced option, not a Phase-1 blocker.

S0.16 Eligibility semantics and audit

Deterministic biological gating is more identifiable than a free router but is itself
a modelling assumption that can encode a mistaken prior as fact, so Panel S0 uses
graded eligibility with executable semantics. FORBIDDEN is a hard zero on the
allocation. ELIGIBLE applies no prior adjustment. PREFERRED applies a prespecified,
frozen-on-tune prior logit bonus. UNKNOWN applies no biological prior and routes
through the generalist, residual, or an exploratory path, and is treated as neither
forbidden nor preferred. The prior bonus is configured, frozen on tune, audited,
included in null construction, and sensitivity-tested. The eligibility audit checks,
on held-out data with anchors independently unavailable to the gating rule, whether
PREFERRED assignments actually carry anchor-relevant signal and whether FORBIDDEN
assignments would have; an eligibility prior the data contradict is a finding, not a
fixed input.

S0.17 The claim-specific ledger and the causal boundary

Expert-global admissibility over-generalises, so Panel S0 records admissibility per
claim type. The claim types are ANCHOR_PREDICTION, that the expert predicts its
anchor on held-out genes; ROUTING_RELEVANCE, that mechanism evidence tracks the
mechanism rather than a confound; EXPERT_COMPUTATIONAL_NECESSITY, that the model
relies on the expert for anchor-relevant cases under the S0.13 interventions;
PATHOGENICITY_RELEVANCE, that the expert's mechanism-compatible evidence is
specifically associated with, and computationally necessary for, pathogenicity
prediction in the declared population; CROSS_GENE_TRANSFER, that the identity holds on
unseen genes; CROSS_ASSAY_TRANSFER, that it holds under an independent assay; and
CLINICAL_ACTIONABILITY, that the reading supports a clinical decision. Each claim
carries its own method-validation and scientific-validation states, reusing the two
orthogonal axes already shipped for Panel R -- a method-validation axis over
not-evaluated, failed, passed-synthetic, passed-internal-empirical, and a
scientific-validation axis over not-evaluated, insufficient-support, failed,
passed-heldout, passed-external, passed-temporal.

BIOLOGICAL_MEDIATION is a claim type in the ledger, and it is explicitly and
permanently NOT admissible through Panel S0. Model-component interventions establish
that the model relies on an expert; they do not intervene on splicing, protein
stability, expression, or disease, and therefore cannot establish that a biological
mechanism causally mediates pathogenicity. A model ablation that specifically
degrades predictions for variants with measured splice defects is a statement about
the model, not about the splice defect's causal role. Biological mediation requires
independent intervention on the proposed molecular mechanism -- rescue experiments,
molecular perturbation, allele-specific functional intervention, a valid longitudinal
or instrumental-variable design, or a credible causal graph with defensible
identification assumptions -- and is deferred to a future Panel T (Causal Mechanism
and Mediation). Panel T does not exist at the time of writing and is named here only
to fix the boundary. Any request to establish BIOLOGICAL_MEDIATION through Panel S0
returns INSUFFICIENT_SUPPORT with a finding that independent causal evidence is
required; no S0 evidence, however strong, upgrades it. No causal word -- mediates,
causes, is responsible for -- may appear in any S0 report for a claim whose estimand
is not causal, and every admissible S0 claim type is associational or computational,
so causal language is barred from S0 output entirely.

S0.18 Support, power, positivity, and multiplicity

Every S0 verdict is guarded by four preconditions, each able to return
INSUFFICIENT_SUPPORT rather than a false verdict. Support: enough anchor-covered
observations in the relevant strata. Power: the matched null and the statistic
together can, in principle, detect a true effect of the prespecified size,
established by the same Type-I-and-power calibration used for the R3 alignment
estimand. Positivity: adequate overlap between the anchor population and the claim
population, from S0.5. Multiplicity: control across declared testing families.

Multiplicity is controlled by declared families, not a single blanket correction. The
confirmatory family -- one primary mechanism anchor per named expert, one primary
specialisation statistic, one primary intervention, across all candidate named
experts -- is Holm-corrected. The confounder family -- all prespecified technical
covariates for one expert and claim -- is Holm-corrected, because any confounder
failure blocks the claim. The secondary-anchor family for triangulation and the
enrichment family for ontology or pathway terms are Benjamini-Hochberg-controlled and
remain exploratory until external replication. A failed primary anchor cannot be
rescued by secondary analyses. Mechanistic admissibility is an intersection of
mandatory gates: all must pass, and evidence is never averaged or combined across
gates to let one strong gate compensate for one failed gate.

Before mechanism evidence or relevance is described in probabilistic language it must
be calibrated: a stated value must correspond to an observed rate of mechanism-
relevant outcomes on held-out data, with a reliability assessment, exactly as
predicted probabilities are calibrated in the classification stack. An uncalibrated
score is reported as a score, not a probability.

S0.19 Mechanistic admissibility states and reporting

Expert mechanistic admissibility is tracked on its own axis: NOT_EVALUATED,
NOT_ADMISSIBLE, PROVISIONAL, ADMISSIBLE_INTERNAL, ADMISSIBLE_EXTERNAL.
ADMISSIBLE_INTERNAL requires every internal gate to pass: output available, an
available and sufficiently independent anchor, permutation-aware alignment passed,
held-out identity passed with faithfulness, mechanism-evidence validity passed, the
matched-null specialisation passed, the confounder gate passed, and the intervention
contrast passed. ADMISSIBLE_EXTERNAL additionally requires external replication.

The expert-global state is not a single most-conservative minimum, which would hide
strong valid evidence behind one unevaluated claim. The claim ledger is authoritative,
and the expert summary reports the HIGHEST supported claim level together with a
separate list of stronger claims not yet established. An expert admissible for anchor
prediction and routing relevance but not evaluated for clinical actionability reports
its highest supported interpretation as mechanism-compatible anchor prediction and
lists clinical actionability and biological mediation as not established, rather than
collapsing to a single uninformative state.

The report-label invariant remains strict for external and clinical artifacts: below
ADMISSIBLE_EXTERNAL an expert's permitted external label is its opaque key, expert_003
in every clinical export and public artifact, never "the splice expert," and a
non-admissible claim carries at least one recorded finding. For internal scientific
reports after ADMISSIBLE_INTERNAL, a candidate label is permitted -- expert_003,
candidate splice-compatible expert -- provided the word candidate is mandatory, the
internal status is displayed, the absence of external replication is visible, and the
machine-readable admissibility travels with the label. Clinical and public outputs
still require ADMISSIBLE_EXTERNAL.

The routing capability itself begins as OUTPUT_AVAILABLE with method validation passed
only on synthetic data, scientific validation not evaluated, admissibility false, and
the reason routing_identifiability_not_established, its findings recording that
allocation weights are predictive-composition outputs only, that no calibrated
mechanism-evidence quantity has yet been validated, and that mechanistic
interpretation is forbidden. It reaches VALIDATED with ADMISSIBLE_INTERNAL only after
internal genomic validation and ADMISSIBLE_EXTERNAL only after external replication.
The MoE's predictive output may be clinically evaluated throughout.

S0.20 The programmatic reporting and serialisation contract, and the sabotage suite

Panel S0 produces a typed, serialisable evidence record per expert and per claim,
carrying the identity class, the anchor with its tier and independence and provenance,
the four routing quantities, the gate outcomes, the specialisation and intervention
statistics with gene-cluster intervals, the matched-null balance and support reports,
the calibration assessment, the support-power-positivity-multiplicity diagnostics, the
lineage, the claim ledger, the mechanistic-admissibility state, the highest supported
claim, the permitted report label, and the findings. A serialisation round-trip test
is part of the contract: the record must survive export and reload without loss, so no
downstream report can silently upgrade an expert's permitted label.

A Panel S0 implementation is not trusted until each of the following deliberately
constructed failures is caught by the gate that should catch it. S0-1 label switching:
permuting experts and router coordinates leaves predictions unchanged, raw-index
agreement fails, aligned comparison recovers the equivalence, biological labels do not
follow raw indices. S0-2 anchor-head success with shortcut routing: an expert predicts
its anchor while the router uses gene identity; identity may pass, the confounder gate
fails. S0-3 router success with interchangeable experts: duplicated expert weights
leave predictive output good while redundancy and intervention gates fail. S0-4
computational-teacher anchor: an expert anchored only to SpliceAI passes as a
distillation expert but fails the independent mechanistic-identity gate. S0-5
matched-null specialisation: a genuine anchor-specific expert beats the blocked
matched null while a generalist does not. S0-6 prevalence-only routing: routing all
common variant classes to one expert produces apparent enrichment the matched null and
conditional-advantage tests reject. S0-7 modality-missing shortcut: routing on whether
a protein embedding exists trips the missingness confounder gate. S0-8 non-transferable
specialisation: specialisation present only in train fails held-out stability or anchor
utility. S0-9 generalist renamed as mechanism: an expert equally good on every anchor
fails the difference-in-differences contrast. S0-10 residual enriched for batch: a
residual expert enriched for a technical batch is refused a novelty interpretation.
S0-11 softmax compositionality: raising one mechanism's allocation by lowering
another's is refused as evidence the second mechanism is absent, because allocation is
not mechanism evidence. S0-12 verification bias: an expert that predicts its anchor only
under the assay's gene selection and dissolves under reweighting is refused. S0-13
marginal-interval miscalibration: paired biological and technical associations with
strong within-bootstrap correlation make the separate-interval rule and the predeclared
paired-difference gate disagree, and only the paired-difference result is accepted,
without prespecifying which rule is more lenient. S0-14 infeasible null: a null draw
whose balance exceeds tolerance is rejected and, if no matched ensemble can be built,
returns INSUFFICIENT_SUPPORT rather than a p-value. S0-15 alignment circularity:
aligning experts on external enrichment and then claiming external stability is refused
by the tiered signature. S0-16 redistribution-policy artifact: an intervention contrast
that exists only under one renormalisation policy is refused. S0-17 uncalibrated
relevance: probabilistic language on an uncalibrated mechanism-evidence score is refused.
S0-18 multiplicity: an admissible-looking claim found by scanning many experts against
many anchors does not survive the family-wise correction. S0-19 eligibility prior wrong:
a PREFERRED assignment that carries no held-out anchor signal is flagged by the
eligibility audit. S0-20 claim conflation: an expert admissible for ANCHOR_PREDICTION is
refused a stronger report label until the evidence for that stronger claim exists. S0-21
model ablation mistaken for biological causality: an expert whose removal specifically
harms anchor cases passes computational necessity but is refused any biological-mediation
reading. S0-22 relevance head detached from expert: mechanism evidence predicts the
anchor while expert ablation effect is unrelated to it, so anchor prediction passes but
the faithfulness gate fails. S0-23 expert mechanism present but expert useless: an
independent assay confirms the mechanism while the expert adds no predictive value, so
mechanism evidence may pass but expert-utility and allocation gates fail. S0-24 useful
expert with wrong mechanism label: an expert improves prediction through a technical
proxy, so allocation utility passes but relevance, confounder, and identity gates fail.
S0-25 anchor missing not at random: selection depends on an unobserved variable, so
inverse-probability weighting on observed variables fails the sensitivity analysis and
the claim is restricted or refused. S0-26 derived-anchor leakage: the direct target is
removed but an aggregate or graph feature derived from it remains, so the transitive
provenance audit fails. S0-27 relevance shifts when a redundant expert is added:
mechanism evidence is unchanged but allocation shifts, and the allocation change is
allowed while the calibrated mechanism evidence must remain stable and is not
interpreted from the allocation shift. S0-28 one-to-many lineage: one expert splits into
two across releases, so a forced one-to-one continuation is rejected, the split lineage
is recorded, and prior admissibility does not transfer. S0-29 low unique null support:
thousands of draws yield only a few unique assignments, so effective support fails and a
tail probability is not reported at impossible resolution. S0-30 reliability learned from
anchor availability: the reliability head predicts whether the assay exists rather than
evidence quality, so the evaluation-support leakage gate fails.

S0.21 Relationship to S1 through S8, Panels Q and R, and the future causal panel

Panel S0 precedes the operational expert panels and gates their interpretation, not
their execution. The intended order is S0 identity and interpretive admissibility, S1
eligibility and routing integrity, S2 utilisation and capacity, S3 stability and
reproducibility, S4 specialisation and anchor validity, S5 intervention and necessity,
S6 redundancy and equivalence, S7 confounding and shortcut routing, S8 calibration,
uncertainty, and clinical utility. Utilisation entropy and load balance, the metrics
most associated with Mixture-of-Experts monitoring, live in S2 and are meaningless as
evidence of biology until S0 admits the reading.

The boundary with Panel R is explicit. Panel R global geometry -- effective rank, mean
resultant length, and the like -- is a checkpoint-level or expert-level qualification
gate, used to decide whether an expert's representation is fit to interpret at all.
Only per-observation local geometry -- local density, norm, leverage, out-of-distribution
distance -- may enter the runtime reliability quantity q_e, and only if it is trained and
frozen leakage-safely. A test-derived cohort-level Panel R finding is never fed into the
router or the reliability head. Panel S0 also applies the Panel Q confounder rule to
routing and inherits the leakage-safe, matched-null, held-out discipline calibrated for
the R3 recovery estimand, including the statistic-and-null compatibility contract, the
gene-cluster bootstrap, the two orthogonal validation axes, and the
non-admissible-requires-a-finding invariant. Biological mediation is reserved for the
future Panel T and is inadmissible here by construction.
