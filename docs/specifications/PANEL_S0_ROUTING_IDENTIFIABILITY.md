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
-- carry separate validation states and must never be collapsed into one.

The inferential goal of this panel is precisely stated. It is not identifiability
in the strict statistical sense, which learned-router mixtures do not possess. It
is EMPIRICAL INTERPRETIVE ADMISSIBILITY under a declared equivalence class: given
an explicit account of the symmetries and gauge freedoms the model admits, and
given held-out, matched-null, confounder-controlled evidence, may a named
mechanistic reading of a specific expert be reported. The panel makes the
conditions for that reading explicit and refuses the reading until they are met.

S0.0 Scope, non-claims, and status (2026-07-22)

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
admitted. Every routing and expert-identity capability defined here begins its
life in the capability state NOT_IMPLEMENTED or OUTPUT_AVAILABLE, with mechanistic
admissibility NOT_ADMISSIBLE.

Panel S0 does not claim that any expert corresponds to a biological mechanism; it
specifies the conditions under which such a claim could later be made. It does not
claim that whitening, JEPA, or any foundation model improves biology. It does not
license a mechanistic report label for any expert at the time of writing, because
no expert exists and no anchor has been acquired. And it does not treat predictive
success as mechanistic evidence: an MoE can be predictively excellent and
mechanistically mute, and this panel keeps those two verdicts apart.

S0.1 The problem Panel S0 exists to solve

A Mixture-of-Experts prediction has the form f(x) = sum_e a_e(x) f_e(x), where the
f_e are expert functions and the a_e are router allocation weights. The temptation
is to read a large allocation weight a_e(x) as a statement that mechanism e is
responsible for the pathogenicity of variant x. That reading is not licensed by
the fit, for four independent reasons.

First, label switching. Any common permutation of the expert outputs and the
router coordinates leaves f(x) unchanged. The model identifies an unordered
collection of components, not "splice expert number two." The biological name
attached to an expert index is imposed by the developer, not learned.

Second, gauge freedom. Deep expert parameterisations and router normalisations
admit multiple internal configurations that produce the same predictions. Learned
representations are in general not identifiable in raw parameter coordinates
without additional constraints or a declared equivalence class, so routing
"confidence" is not a scale-identified quantity on its own.

Third, shortcut routing. Ordinary MoE training does not guarantee that experts
specialise along biologically meaningful axes. A router can achieve an apparently
biological partition through a convenient non-causal correlate. The project's own
n_pathogenic_in_gene history is the precedent: a top feature encoded gene
prevalence and required a permutation ablation to establish that it carried genuine
variant-level biology rather than memorised gene identity.

Fourth, and specific to the routing weight itself, compositional dependence. A
softmax allocation is simplex-constrained: the a_e(x) sum to one, so raising one
expert's weight necessarily lowers the others, even when two mechanisms are
simultaneously present in the same variant. A variant that both disrupts splicing
and truncates the protein cannot have two allocation weights near one; the softmax
forces the mechanisms to compete for a fixed budget. An allocation weight therefore
conflates "this mechanism is strongly relevant" with "this mechanism is more
relevant than the others," which are different statements. The allocation weight is
the wrong object to interrogate for mechanism.

The consequence is a direct analogy to Panel R stage R3. There, the angular
recovery statistic looked like representation recovery but was shown to be an
alignment-blind artifact, and a separate alignment-sensitive estimand had to be
constructed to ask the intended question. Here, the softmax allocation weight looks
like a mechanism attribution but is the wrong object, and a separate,
unconstrained relevance quantity must be constructed to ask the intended question.
The same discipline -- a held-out, matched-null, confounder-controlled admissibility
gate on the correct estimand -- resolves both.

S0.2 Three separate quantities: allocation, relevance, reliability

The central structural commitment of this panel is that a single routing number
must be decomposed into three quantities that answer three different questions and
carry three different validation statuses.

Allocation a_e(x) is the softmax mixing weight. It is an engineering quantity: it
governs how the prediction is composed and how compute is spent. It is
simplex-constrained and compositionally dependent, and it is never, on its own,
mechanistic evidence. Its gate, S0-A, asks only whether the allocation improves the
final task -- a utilisation-and-capacity question.

Relevance r_e(x) is a separate, unconstrained, per-expert quantity in the unit
interval, produced by an independent head rather than a shared softmax, so that two
mechanisms can both be highly relevant to the same variant without competing for a
fixed budget. Relevance is the scientific quantity, and only a calibrated relevance
is eligible for mechanistic interpretation. Its gate, S0-R, asks whether high
relevance for expert e coincides specifically with the presence and usefulness of
mechanism e rather than a confound.

Reliability q_e(x) is a third quantity that records whether the inputs expert e
needs are present and trustworthy for variant x -- whether the required modality
exists, whether the anchor is defined, whether the representation is in
distribution. Reliability is neither allocation nor relevance; a variant may be
highly relevant to a mechanism whose measurement is unreliable for it. Reliability
gates whether any relevance claim about that variant is even eligible to enter the
S0-R analysis.

The three gates are independent. S0-A (allocation utility), S0-R (relevance
validity), and S0-I (expert identity) each stand alone, and a mechanistic reading
requires all of the ones that apply. A future implementation that reports only the
softmax allocation has not produced a relevance quantity at all and is therefore
categorically ineligible for mechanistic interpretation, no matter how strong its
predictions.

S0.3 Expert identity classes

Not every expert is a candidate mechanism, and forcing every expert to carry a
named biological target is itself an error -- it presses free experts into imitating
incomplete or noisy prior knowledge. Panel S0 recognises four identity classes, and
the mechanistic-admissibility gate applies only to the first.

ANCHORED_MECHANISM. An expert with an externally specified mechanistic estimand -- a
measured splice outcome, a measured protein abundance or stability, a measured
functional activity, a measured regulatory or binding change. Only this class is
eligible for a mechanistic report label, and only after the full gate below.

PREDICTIVE_UNNAMED. An expert permitted to discover a useful partition and improve
prediction, reported as latent expert A, latent expert B, and so on. It receives no
biological name until Panel S0 establishes a reproducible biological meaning for it.
This class preserves the value of latent specialisation without asserting an
ontology it has not earned.

GENERALIST. Always eligible, responsible for shared predictive structure common
across variants. A shared expert of this kind absorbs common capability and reduces
routed-expert redundancy; it is never given a mechanism name.

RESIDUAL. Always eligible, representing unsupported mechanisms, poor anchor
coverage, multimodal disagreement, out-of-distribution biology, and technical
failure not yet distinguished from novelty. It must never be called a
novel-mechanism expert. High routing probability to the residual expert marks a
variant as a candidate for later investigation, not as a discovery.

S0.4 The anchor evidence hierarchy and the independence profile

A mechanistic name is only as strong as the independence of the evidence anchoring
it. Panel S0 ranks anchor sources by evidence tier, and, separately, records how
independent each anchor is from the training signal. Tier and independence are two
different axes and must not be conflated: a high-tier experimental assay used both
to train and to validate an expert is not independent evidence for that expert, and
a lower-tier source drawn from a genuinely separate process may be more probative.

The evidence tiers, strongest first: DIRECT_EXPERIMENTAL, a measurement of the
mechanism itself for the variant or its protein product; ORTHOGONAL_EXPERIMENTAL,
an independent experimental readout correlated with the mechanism through different
assay physics; HUMAN_ASSOCIATION, segregation, case-control, or other human-genetic
association; CURATED_MECHANISM, expert-curated mechanistic annotation;
COMPUTATIONAL_TEACHER, the prediction of another computational model;
HEURISTIC_PRIOR, a rule-of-thumb or hand-built prior.

The independence profile, recorded per anchor and per expert, asks a separate
question: was this anchor, or a direct proxy of it, part of the expert's training
signal or the router's input; does it share a data-generating process, a
submitter, a curation pipeline, or a computational ancestor with the training
labels; and does it cover the genes and variant classes on which the mechanistic
claim is to be made. An anchor high in tier but low in independence supports at most
a provisional claim.

The critical rule concerns computational teachers. A predicted score such as
SpliceAI is a COMPUTATIONAL_TEACHER, not an independent mechanism anchor. SpliceAI
may be used as an input feature, a teacher for distillation, weak supervision, a
routing prior, or a synthetic development target, but anchoring an expert to it
establishes only that the expert reproduces SpliceAI's decision surface, including
its errors and inductive biases. An expert anchored solely to SpliceAI may be
called a SpliceAI-distillation expert; it may not be called a splice-mechanism
expert unless separately validated against measured splicing outcomes. Appropriate
independent sources include multiplexed assays of variant effect, measured splicing
assays, abundance assays such as Variant-Abundance-by-Massively-Parallel-sequencing
(VAMP-seq), deep-mutational-scanning functional scores, reporter assays, and
binding or enzymatic assays; the Multiplexed-Assay-of-Variant-Effect database
(MaveDB) is the canonical store of such functional scores with experimental
metadata. A computational prediction may still serve as a weak anchor, but its
status must be recorded explicitly as weak_computational with
mechanistic_admissibility false.

S0.5 The anchor observation process and verification bias

An anchor is not a random sample of the genome. Assays are performed on the genes
and variants that investigators chose to study, for reasons correlated with prior
belief about pathogenicity, gene importance, and tractability. An expert trained
against such an anchor can learn which variants were selected for assay rather than
which variants use the mechanism, and a router can then allocate on the same
selection signal. This is verification bias, and it produces a biological-looking
result with no mechanistic content.

Panel S0 therefore requires, for every ANCHORED_MECHANISM claim, an explicit model
of the anchor observation process: which genes and variant classes are covered,
what selection produced that coverage, and whether the coverage overlaps the
population on which the claim is made. A positivity diagnostic is mandatory -- there
must be adequate anchor coverage across the strata of the claim, and strata without
coverage are excluded from the claim rather than silently extrapolated into. Where
selection is informative, the analysis must reweight toward the claim population,
through inverse-probability weighting or an equivalent doubly-robust adjustment, and
must report the sensitivity of the conclusion to that reweighting. An anchored claim
that holds only under the assay's selection and dissolves under reweighting to the
claim population is not admissible.

S0.6 Two independent propositions: identity and routing relevance

Establishing what an expert is and establishing why the router selects it are two
different claims, and a mechanistic interpretation requires both.

Expert identity asks: does expert e contain information predictive of its anchor
target on held-out genes. Routing relevance asks: is the relevance r_e(x) high
specifically when expert e's mechanism is relevant and useful, rather than when some
confound is present. An expert can predict its anchor perfectly and still be routed
by gene identity, assay availability, or data source. Panel S0 therefore maintains
two independent gates, the Expert Identity Gate (S0-I) and the Router Relevance
Gate (S0-R), and neither substitutes for the other.

S0.7 The confounder gate for routing, as a paired difference

The Router Relevance Gate must actively exclude the confounds that could produce a
biological-looking routing pattern without biological routing. The enumerated
shortcuts for this project are gene identity, ClinVar submitter, review status,
annotation source, variant-consequence label, modality missingness, transcript
availability, graph degree, direct gene-disease priors, and pathogenic-count
aggregates. Each is treated as a technical or design covariate.

The gate is a PAIRED gene-cluster difference, not a comparison of two separately
estimated intervals. For the primary biological anchor B and a technical covariate
T, the quantity of interest is the difference in routing agreement, delta = A(r, B)
- A(r, T), estimated within the same gene-cluster bootstrap resamples so that the
positive correlation between the two agreement statistics is used rather than
ignored. Requiring the lower bound of a separate interval for A(r, B) to exceed the
upper bound of a separate interval for A(r, T) is needlessly conservative: because
the same resampled clusters drive both statistics, the difference has much smaller
variance than the two marginals suggest, and the separate-interval rule can refuse a
genuine biological-dominance signal that the paired difference detects. The gate
therefore requires the lower bound of the gene-cluster confidence interval for delta
to exceed a prespecified dominance margin delta_min greater than zero, with the
comparison Holm-corrected across the full family of technical covariates. If routing
is expressed as a continuous relevance rather than a hard assignment, the agreement
statistic A is a predeclared association measure appropriate to continuous routing,
fixed before the analysis, not chosen after seeing the data.

No mechanistic routing claim is admissible if a technical covariate ties or beats
the biological anchor under this paired, corrected comparison.

S0.8 The two matched-null families and their feasibility diagnostics

A null for routing must match the exact claim, exactly as the R3 matched-spectrum
null had to preserve the gain spectrum while scrambling gain-to-direction
alignment. Two null families are required, each preserving the nuisance structure
and randomising only the quantity under test.

Both families aim to preserve, by construction: the number of experts, the
per-expert loads, the biological eligibility masks, the route sparsity or top-k
pattern, the route entropy distribution, the pathogenic-label distribution, the
gene-block structure, the anchor-target prevalence, the missing-modality patterns,
and the variant-class strata.

S0-N1 is the expert-identity-permutation family, and it has two members. S0-N1a
permutes expert identity labels after predictions are fixed, testing whether a
proposed biological name is uniquely supported or whether an equally good account is
obtained by relabelling; it is the direct analogue of the R3 identity-orientation
control. S0-N1b permutes only the anchor-to-expert assignment while holding the
routing fixed, isolating whether the specific anchor claimed for an expert is the
one it supports, as distinct from supporting some anchor.

S0-N2 is the blocked-route-reassignment family. It reassigns whole-gene route
vectors among eligible gene blocks within matched strata, testing whether the
learned routing carries anchor-specific information beyond prevalence and
eligibility.

Exact preservation of every nuisance quantity is frequently infeasible on real
data: loads, entropy, eligibility, and gene-block structure cannot all be held
exactly fixed while randomising. Panel S0 therefore requires a feasibility
diagnostic for every null draw. Each draw reports the residual imbalance in every
preserved quantity; a draw whose imbalance exceeds a prespecified tolerance is
rejected and redrawn; and if an acceptable matched ensemble cannot be assembled, the
null returns INSUFFICIENT_SUPPORT rather than a mis-calibrated p-value. S0-N2 is
therefore described honestly as a constrained matched-reassignment null, not an
exact permutation null, and its balance report is part of its output.

The specialisation statistic must be compatible with the null in the sense
established for R3: a statistic that responds only to a quantity the null preserves
cannot gain power from that null, and pairing them must raise rather than silently
report zero power. The statistic-and-null compatibility contract from the alignment
work applies to every S0 statistic-and-null pairing.

S0.9 The specialisation estimand as a difference in differences

An expert that is merely a strong generalist must not pass as a specialist. The
specialisation estimand is therefore a difference in differences: the expert's
held-out advantage over the generalist on anchor-relevant observations, minus its
advantage over the generalist on matched non-anchor observations. A pure generalist
has equal advantage on both and a specialisation of zero. The estimand is computed
with cross-fitting so that the observations used to fit the expert and the router
are disjoint from those used to estimate the advantage, and it must exceed the
matched-null distribution of the same difference-in-differences quantity, with a
gene-cluster confidence interval excluding zero.

S0.10 The Expert Identity Gate

The Expert Identity Gate (S0-I) is not satisfied by an expert's auxiliary head
fitting its anchor. Three tests are required. Anchor prediction: the expert
representation predicts the anchor target on held-out genes. Conditional comparative
advantage: the expert outperforms the generalist, the other experts, and a
parameter-matched dense branch, specifically on anchor-relevant observations, which
is the difference-in-differences estimand of S0.9 -- an expert can pass anchor
prediction yet fail this if every expert learned the same target. Router allocation
validity: allocating more relevance to the expert improves both the anchor and the
final task conditionally, not unconditionally.

S0.11 Intervention is mandatory, with a taxonomy

Enrichment is correlational; mechanistic interpretation additionally requires a
controlled intervention. Panel S0 defines four intervention types. I1, route
ablation: zero the expert's relevance and renormalise the remaining router mass. I2,
expert replacement: replace the expert with the generalist. I3, expert swap:
substitute another expert in its place. I4, input perturbation: perturb the
expert's anchor-specific input or mask the relevant modality.

For each intervention the reported quantity is the interaction contrast I_e, the
difference between the anchor-case degradation and the matched control-case
degradation, carrying a gene-cluster confidence interval excluding zero. Because I1
redistributes the ablated mass across the remaining experts, its effect depends on
the redistribution policy; the intervention must therefore report the sensitivity of
I_e to the redistribution policy, so that a contrast which exists only under one
renormalisation is not reported as a mechanism. The expected pattern for a genuine
mechanism expert is a larger degradation on anchor-relevant cases than on matched
control cases, stable across intervention types and redistribution policies. Showing
that an expert receives many anchor-relevant variants is not sufficient; removing it
must specifically hurt anchor-relevant predictions.

S0.12 Symmetry-aware expert alignment before any stability claim, without
circularity

Because expert indices are not identifiable across training runs, experts must be
aligned across runs before their stability is measured; comparing expert one in one
run to expert one in another is meaningless. For each expert an identity signature
is assembled, but the signature used for alignment must not include the very
external quantity whose stability is later claimed, or the conclusion is circular.

The signature is therefore tiered. A core signature, used for alignment, is built
only from quantities frozen on the TUNE partition: anchor-target performance on
TUNE, the held-out routing distribution on TUNE, routing by eligibility class, and
output residual structure on TUNE. Experts are aligned between runs by minimising a
weighted core-signature distance under a Hungarian assignment, with the alignment
weights frozen on TUNE and not adjusted after seeing whether the resulting names
look stable. An extended signature, which includes external enrichment, TEST-set
routing, and intervention effects, is used only for reporting AFTER alignment, never
to drive the alignment itself. Using external or TEST enrichment to align experts
and then claiming external stability would leak the conclusion into its own premise;
the tiered signature forbids it.

Only after alignment is routing stability measured, at the distributional,
observation, and gene levels, with the lower-tail and minimum-subgroup stability
reported alongside the median so that a stable average cannot hide a route that is
unstable specifically for splice or noncoding variants.

S0.13 Multi-anchor triangulation

A single target rarely defines a mechanism. Measured stability, for instance, does
not fully capture protein destabilisation: assays are context-dependent, stability
and abundance are related but distinct, some pathogenic protein effects preserve
stability, and gain-of-function effects need not appear as destabilisation. A
stronger expert identity triangulates across partially independent anchors -- a
primary measured target, a secondary related measurement, an orthogonal assay, and a
negative-control anchor from an unrelated mechanism. Identity is stronger when the
primary and secondary anchors agree, the unrelated negative control does not, and
the effect transfers across genes and assays.

S0.14 Graded eligibility and the eligibility audit

Deterministic biological gating -- sending missense variants to a protein expert and
intronic variants to a splice expert -- is more identifiable than a free router, but
it is itself a modelling assumption that can be wrong, and a hard mask can encode a
mistaken biological prior as though it were fact. Panel S0 therefore uses graded
eligibility with four levels: FORBIDDEN, an expert must not be routed a variant
class; ELIGIBLE, it may be; PREFERRED, it is expected to be; UNKNOWN, eligibility is
not established. The eligibility assignment is itself auditable: an eligibility
audit checks, on held-out data, whether PREFERRED assignments actually carry
anchor-relevant signal and whether FORBIDDEN assignments would have. An eligibility
prior that the data contradict is a finding, not a fixed input.

S0.15 The claim-specific admissibility ledger

Expert-global admissibility over-generalises. An expert can predict a measured
exon-inclusion anchor yet fail to establish that its mechanism mediates
pathogenicity, or fail to transfer across assays or across genes. Panel S0 therefore
records admissibility per claim type, not per expert. The claim types are
ANCHOR_PREDICTION, that the expert predicts its anchor on held-out genes;
ROUTING_RELEVANCE, that relevance tracks the mechanism rather than a confound;
PATHOGENICITY_MEDIATION, that the mechanism the expert encodes mediates the
pathogenicity label; CROSS_GENE_TRANSFER, that the identity holds on genes unseen in
training; CROSS_ASSAY_TRANSFER, that it holds under an independent assay; and
CLINICAL_ACTIONABILITY, that the mechanistic reading supports a clinical decision.

Each claim carries its own method-validation and scientific-validation states,
using the same two orthogonal axes already shipped for Panel R (a method-validation
axis over not-evaluated, failed, passed-synthetic, passed-internal-empirical, and a
scientific-validation axis over not-evaluated, insufficient-support, failed,
passed-heldout, passed-external, passed-temporal). The expert-global mechanistic
admissibility is then the MOST CONSERVATIVE summary over its claim ledger, never a
headline that hides an inadmissible sub-claim. ANCHOR_PREDICTION admissible does not
imply PATHOGENICITY_MEDIATION admissible, and the ledger keeps them distinct exactly
as the R3a/R3b split kept method-validated-on-synthetic distinct from
scientifically-validated.

S0.16 Mechanistic admissibility states and the report-label invariant

Expert mechanistic admissibility is tracked on its own axis, orthogonal to the
capability state and to the method and scientific validation axes: NOT_EVALUATED,
NOT_ADMISSIBLE, PROVISIONAL, ADMISSIBLE_INTERNAL, ADMISSIBLE_EXTERNAL.
ADMISSIBLE_INTERNAL requires every internal identity and routing gate to pass:
output available, an available and sufficiently independent anchor, permutation-
aware alignment passed, held-out identity passed, routing relevance passed, the
matched-null specialisation passed, the confounder gate passed, and the intervention
contrast passed. ADMISSIBLE_EXTERNAL additionally requires external replication.

The report-label invariant is strict and is the operational heart of the panel:
below ADMISSIBLE_EXTERNAL an expert may carry an internal proposed anchor label, but
its permitted EXTERNAL report label remains its opaque key. An unvalidated expert is
expert_003 in every report, figure, and exported artifact, never "the splice
expert." A non-admissible expert must carry at least one recorded finding explaining
why, mirroring the non-admissible-requires-a-finding invariant already enforced in
Panel R.

The routing capability itself begins as OUTPUT_AVAILABLE with method validation
passed only on synthetic data, scientific validation not evaluated, admissibility
false, and the reason routing_identifiability_not_established, with findings
recording that allocation weights are predictive-composition outputs only, that no
calibrated relevance quantity has yet been validated, and that mechanistic
interpretation is forbidden. It reaches VALIDATED with ADMISSIBLE_INTERNAL only
after internal genomic validation, and ADMISSIBLE_EXTERNAL only after external
replication. The MoE's predictive output may be clinically evaluated throughout; the
two gates stay independent.

S0.17 Relevance calibration, support, power, positivity, multiplicity, and causal
language

Before relevance may be described in probabilistic language, it must be calibrated:
a stated relevance must correspond to an observed rate of mechanism-relevant
outcomes, checked on held-out data with a reliability assessment, exactly as
predicted probabilities are calibrated in the classification stack. An uncalibrated
relevance is reported as an uncalibrated score, not as a probability.

Every S0 verdict is additionally guarded by four preconditions, each of which can
return INSUFFICIENT_SUPPORT rather than a false verdict. Support: enough
anchor-covered observations in the relevant strata. Power: the matched null and the
statistic together can, in principle, detect a true effect of the prespecified size,
established by the same Type-I-and-power calibration used for the R3 alignment
estimand. Positivity: adequate overlap between the anchor population and the claim
population, from S0.5. Multiplicity: across experts, anchors, claim types, and
technical covariates, the panel controls the family-wise error or false-discovery
rate with a prespecified correction, so that scanning many experts against many
anchors does not manufacture an admissible claim by chance.

No causal word -- mediates, causes, is responsible for -- may appear in any report
for a claim whose estimand is not causal. ANCHOR_PREDICTION and ROUTING_RELEVANCE
are associational and must be described associationally; PATHOGENICITY_MEDIATION is
a causal claim and requires the intervention contrasts of S0.11, not enrichment
alone, before any mediating language is permitted.

S0.18 The programmatic reporting contract and the required sabotage tests

Panel S0 produces a typed, serialisable evidence record per expert and per claim,
carrying the identity class, the anchor and its tier and independence profile, the
three gate outcomes, the specialisation and intervention statistics with their
gene-cluster intervals, the matched-null balance reports, the calibration
assessment, the support-power-positivity-multiplicity diagnostics, the claim ledger,
the mechanistic-admissibility state, the permitted report label, and the findings. A
serialisation round-trip test is part of the contract: the record must survive
export and reload without loss, so that no downstream report can silently upgrade an
expert's permitted label.

A Panel S0 implementation is not trusted until each of the following deliberately
constructed failures is caught by the gate that should catch it. S0-1, label
switching: permuting experts and router coordinates leaves predictions unchanged,
raw expert-index agreement fails, aligned comparison recovers the equivalence, and
biological labels do not follow raw indices. S0-2, anchor-head success with shortcut
routing: an expert predicts its anchor while the router uses gene identity -- the
identity test may pass, the confounder gate fails, and mechanistic routing stays
inadmissible. S0-3, router success with interchangeable experts: duplicated expert
weights leave predictive output good while the redundancy and intervention gates
fail. S0-4, computational-teacher anchor: an expert anchored only to SpliceAI passes
as a distillation expert but fails the independent mechanistic-identity gate. S0-5,
matched-null specialisation: a genuine anchor-specific expert beats the blocked
matched null while a generalist does not. S0-6, prevalence-only routing: routing all
common variant classes to one expert produces apparent enrichment that the matched
null and conditional-advantage tests reject. S0-7, modality-missing shortcut:
routing on whether a protein embedding exists trips the missingness confounder gate.
S0-8, non-transferable specialisation: specialisation present only in train fails
held-out routing stability or anchor utility. S0-9, generalist renamed as mechanism:
an expert equally good on every anchor fails the difference-in-differences
anchor-specificity contrast. S0-10, residual enriched for batch: a residual expert
enriched for a technical batch is refused a novelty interpretation. S0-11, softmax
compositionality: raising one mechanism's allocation by lowering another's is
refused as evidence that the second mechanism is absent, because allocation is not
relevance. S0-12, verification bias: an expert that predicts its anchor only under
the assay's gene selection and dissolves under reweighting to the claim population is
refused. S0-13, separate-interval leniency: a confounder that would pass a
separate-interval rule but fails the paired gene-cluster difference is correctly
refused. S0-14, infeasible null: a null draw whose balance report exceeds tolerance
is rejected and, if no matched ensemble can be built, returns INSUFFICIENT_SUPPORT
rather than a p-value. S0-15, alignment circularity: aligning experts on external
enrichment and then claiming external stability is refused by the tiered signature.
S0-16, redistribution-policy artifact: an intervention contrast that exists only
under one renormalisation policy is refused. S0-17, uncalibrated relevance:
probabilistic language on an uncalibrated relevance is refused. S0-18,
multiplicity: an admissible-looking claim found by scanning many experts against
many anchors does not survive the family-wise correction. S0-19, eligibility prior
wrong: a PREFERRED eligibility assignment that carries no held-out anchor signal is
flagged by the eligibility audit. S0-20, claim conflation: an expert admissible for
ANCHOR_PREDICTION is refused a PATHOGENICITY_MEDIATION report label until the causal
intervention evidence exists.

S0.19 Relationship to the rest of the Expert-Systems series

Panel S0 precedes the operational expert panels and gates their interpretation, not
their execution. The intended order is S0 identity and interpretive admissibility,
S1 eligibility and routing integrity, S2 utilisation and capacity, S3 stability and
reproducibility, S4 specialisation and anchor validity, S5 intervention and
necessity, S6 redundancy and equivalence, S7 confounding and shortcut routing, S8
calibration, uncertainty, and clinical utility. Utilisation entropy and load
balance, the metrics most associated with Mixture-of-Experts monitoring, live in S2
and are meaningless as evidence of biology until S0 admits the mechanistic reading
in the first place. Panel S0 is also the point of contact with the existing
framework: it consumes Panel R representation geometry as a reliability signal on
expert modalities, it applies the Panel Q confounder rule to routing, and it
inherits the leakage-safe, matched-null, held-out discipline calibrated for the R3
recovery estimand -- including the statistic-and-null compatibility contract, the
gene-cluster bootstrap, the two orthogonal validation axes, and the
non-admissible-requires-a-finding invariant.
