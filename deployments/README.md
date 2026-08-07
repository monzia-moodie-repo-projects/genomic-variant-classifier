# deployments/

**Author: Monzia Moodie**
**Created 2026-08-07, REGISTRY-1.**

Small, reviewable **control-plane declarations**. Not model artifacts, and not
scientific reference data.

```
models/          model artifacts -- large, machine-local, gitignored
data/reference/  scientific reference data and profiles
deployments/     control-plane declarations -- this directory
```

`registry.v1.json` is the **declared** deployment state, read and written by
`genomic_variant_classifier.monitoring.model_registry.ModelRegistry`.

## What a committed registry can and cannot claim

Continuous Integration reading this file can establish:

> the repository's declared production deployment is structurally coherent

It cannot establish:

> production is healthy

Those are different claims. A file in version control is a declaration; it is
not the process serving predictions. Closing that gap requires the serving
environment to attest that the artifact digest it loaded equals the digest
declared here, which is DEPLOY-1's work and is not done.

An empty `records` list is therefore honest and meaningful: **no deployment is
declared**. It is not the same statement as "no registry exists", and
`ModelRegistry.load` refuses to treat a missing file as an empty one.
