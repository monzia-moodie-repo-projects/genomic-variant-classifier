Panel Q — Unsupervised structure, reproducibility, biological validity, and confounding

This panel evaluates whether a representation or cohort exhibits coherent structure. It does not evaluate whether predictions are correct. A strong result here does not make a better classifier; it supports a claim about how a representation is organized. Panel Q results must never be reported as evidence of clinical superiority.

Partition policy

Structure analysis is a model-selection activity. It chooses a representation, preprocessing, dimensionality, distance geometry, clustering algorithm, cluster count, noise handling, stability thresholds, and a biological interpretation. Performing those choices on the locked test partition is selection on test.

Therefore:

cluster discovery, cluster-count selection, geometry comparison, and biological interpretation occur on a dedicated STRUCTURE partition;
the STRUCTURE partition is gene-disjoint from train, tune, probability calibration, conformal calibration, and test;
the test partition admits only a predeclared replication analysis: freeze the representation, preprocessing, algorithm, distance and cluster count on STRUCTURE, assign test observations to the frozen solution, and evaluate prespecified replication metrics once;
no cluster interpretation, cluster-count selection, or method selection occurs on test;
no method selection occurs on an external cohort.

Q1 — Internal geometry

Always computed:

Davies-Bouldin index, Euclidean centroid;
Calinski-Harabasz index;
cluster-count;
cluster-size distribution;
smallest cluster size;
largest-cluster fraction;
singleton-cluster count;
degeneracy diagnostics.

Computed where the representation is L2-normalized and the intended geometry is angular:

Davies-Bouldin index, spherical cosine.

The Euclidean-centroid and spherical-cosine forms are different estimands and must be reported under different names. For unit-normalized vectors the squared Euclidean distance equals two minus twice the cosine similarity, so pairwise Euclidean and cosine distances are monotone; that equivalence does NOT make an ordinary Euclidean-centroid Davies-Bouldin index a cosine Davies-Bouldin index, because the Euclidean mean of points on a sphere does not lie on the sphere. A metric labelled only "Davies-Bouldin" is unreportable.

Estimated by deterministic stratified subsampling, never computed exactly at cohort scale:

silhouette coefficient.

The silhouette coefficient materializes an n-by-n distance matrix. Measured on 1280-dimensional embeddings: 42.4 seconds and 1.0 gibibyte at thirty thousand observations; approximately 27 gibibytes at sixty thousand; 16.4 tebibytes at 1.5 million. It is an estimated quantity and must be reported as one, with:

sampling method;
sample size;
sampling fraction;
sampling seed;
replicate count;
mean estimate;
standard deviation;
95% interval;
minimum per-cluster sampling floor;
distance metric.

An estimate is called stable only when its conclusion is insensitive to sample budgets of 2,500, 5,000, 10,000 and 20,000.

Conditional, behind a dependency gate:

density-based clustering validation.

Density-based clustering validation is not in scikit-learn. Its output must be labelled exactly one of: exact density-based clustering validation; HDBSCAN relative-validity approximation; or unavailable. The approximation must never be reported as the exact quantity.

Population accounting

Reported separately, never as a single "noise fraction":

input count;
non-finite exclusion count and fraction;
algorithm-noise count and fraction;
small-cluster exclusion count and fraction;
analyzed count and fraction.

For density-based methods, additionally:

fraction assigned to noise;
fraction with low membership probability;
membership-probability distribution;
cluster persistence where available.

An algorithm must not obtain a favourable internal score by discarding difficult observations as noise. A single aggregate exclusion figure conceals exactly that.

Q2 — Reproducibility

Resampling is fixed-rate subsampling without replacement, at the gene level. The ordinary bootstrap is unsuitable for partition agreement: observations repeat, some disappear, duplicates overweight, the shared-observation intersection varies, and the denominator changes across replicates. Sampling genes rather than variants prevents large genes from dominating the estimate.

For each replicate:

sample a fixed fraction of genes without replacement;
include their observations;
refit preprocessing where required;
rerun clustering;
compare assignments on shared observations only;
align clusters and compute per-cluster Jaccard;
record cluster dissolution and emergence.

Reported:

adjusted Rand index, median, interquartile range and 95% interval;
adjusted mutual information, median, interquartile range and 95% interval;
minimum per-cluster Jaccard;
median per-cluster Jaccard;
dissolved-cluster fraction;
subsample fraction, replicate count, and sampling unit.

The adjusted Rand index is primary for reproducibility; adjusted mutual information is its companion; per-cluster Jaccard is essential because a global agreement score can remain high while one cluster dissolves entirely. Where the shared-observation overlap falls below the prespecified minimum, the result is insufficient support, not a number.

Q3 — External biological agreement

Against each independent reference partition:

adjusted mutual information;
adjusted Rand index;
normalized mutual information;
homogeneity;
completeness;
V-measure;
pathway enrichment;
phenotype enrichment;
molecular-consequence enrichment;
functional-assay effect distribution;
tissue-specificity enrichment;
disease-category enrichment.

Adjusted mutual information is primary here because a discovered partition may legitimately refine a reference partition, which the adjusted Rand index penalizes. Normalized mutual information is a secondary descriptive statistic and is never reported alone, because it is not adjusted for agreement expected by chance and inflates when clusters are numerous or samples small.

Every reference must be declared as exactly one of:

used as model input;
not used as model input;
derived from training labels;
independent external validation.

Agreement with a reference that was supplied as an input is not discovery. The strongest evidence comes from annotations or assays not used during representation learning. Homogeneity and completeness are not symmetric in their arguments; the reference partition is the first argument and the discovered clustering the second.

Q4 — Confounding

Adjusted mutual information between the cluster assignment and each covariate, with a permutation null.

For variant-level clusters:

gene identity;
chromosome;
variant type;
coding status;
transcript;
source laboratory;
ClinVar submitter;
ancestry;
assay platform;
batch;
reference build;
missing-modality pattern;
data-source coverage;
review status;
label availability;
dataset release.

For patient-level clusters:

site;
scanner;
staining batch;
sequencing centre;
ancestry;
sex;
age;
sample-processing protocol;
missingness pattern.

Each covariate is classified as technical, design-related, biological-but-nuisance, or target biology.

The null distribution is generated by permutation respecting the dependence structure: for gene-dependent data, permute at the gene level or within prespecified strata, never across arbitrary variant rows. This follows the logic of the n_pathogenic_in_gene permutation ablation, which established that feature's contribution as genuine variant-level biology against a permuted null rather than by inspection.

Reported per covariate:

observed adjusted mutual information;
null mean;
null 95th percentile;
permutation p-value;
effect over null;
95% interval on the observed value.

Q5 — Scientific claim gate

A cluster solution supports a scientific claim only when every gate passes. Thresholds are prespecified in configuration and are not chosen after seeing results.

Geometry gate
at least two internal metrics are computable;
no serious geometric degeneracy;
the silhouette estimate is stable across sample budgets and seeds;
the analyzed fraction is above the prespecified floor.

Reproducibility gate
median subsample adjusted Rand index above the prespecified threshold;
per-cluster Jaccard shows no systematic cluster dissolution;
the solution survives multiple clustering seeds;
the solution survives reasonable preprocessing perturbations.

Biological gate
agreement evaluated against an independent variable, declared as such;
enrichment survives multiplicity correction;
effect sizes reported with confidence intervals;
cluster membership adds information beyond existing labels or taxonomies.

Confounder gate
hard refusal: for any technical covariate T and primary biological target B, if the upper 95% bound of adjusted mutual information with T is not below the lower 95% bound of adjusted mutual information with B, the solution is confounded and no scientific claim is permitted;
warning: for a biological-but-nuisance covariate meeting the same condition, the solution is potentially non-target biology and requires stratified analysis;
gene-identity dependence is explicitly tested;
missingness pattern does not dominate;
source laboratory and platform do not dominate;
permutation nulls are passed.

Comparison is on intervals, not point estimates. A margin smaller than the uncertainty is not a margin.

Replication gate
the frozen cluster definition replicates in an independent cohort;
external cluster proportions and geometry are reported;
clinical or molecular associations repeat directionally;
no method selection occurs on the external cohort.

The panel is fail-closed. Any gate that cannot be evaluated is a refusal, not a pass. Metrics that are undefined, unsupported, deferred, or failed carry an explicit status and reason, and the reporting layer must refuse to average across them without an explicit override.

The evaluator should be able to state:

On the STRUCTURE partition, representation R under geometry G yields K reproducible clusters with median subsample adjusted Rand index A, whose adjusted mutual information with independent biology B exceeds that with every technical covariate by a margin wider than the confidence intervals, and which replicate in cohort C.

That is a defensible representation claim. "Davies-Bouldin fell from 3.1 to 1.5" is not. A lower Davies-Bouldin index can be obtained by making clusters tighter around gene identity, laboratory, ancestry, sequencing platform, variant type, or missing-modality pattern, and a compact map of provenance drawn in latent ink is indistinguishable from biology by geometry alone.
