# INCIDENT: GPU Quota Denied — 2026-04-08

## Summary
GPU quota request (GPUS_ALL_REGIONS = 1) denied by Google on 2026-04-08.

## Root Cause
New GCP account with insufficient billing history. Google applies fraud
prevention controls on GPU quota for new accounts regardless of billing
account status. The CPU billing from Runs 6 & 7 (n2-highmem-32) begins
building the required history.

## Symptom
Every zone returned `ZONE_RESOURCE_POOL_EXHAUSTED` for T4 instances.
This was a red herring — the real error only appeared when the L4 was
attempted: `Quota GPUS_ALL_REGIONS exceeded. Limit: 0.0 globally`.

## Zones Tried (all exhausted or quota-blocked)
us-central1-a, us-central1-b, us-central1-c, us-central1-f,
us-east1-b, us-east1-c, us-west1-b, us-west2-b, us-east4-b, us-east4-c

## Resolution
- Reapply 2026-04-15 (allow billing history to accumulate)
- Justification to use: production ML training for genomic variant
  pathogenicity classification; 1 GPU (T4/L4) for 3-4 hour training runs;
  active billing history from CPU runs on 2026-04-08

## Status
OPEN — pending quota reapplication on 2026-04-15
