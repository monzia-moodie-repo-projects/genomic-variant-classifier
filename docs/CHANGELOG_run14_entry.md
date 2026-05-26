# CHANGELOG — Run 14 entry

> Append the block below to `docs/CHANGELOG.md` after the existing Run 13 entry.
> Edit TBD slots in place once results arrive.

## 2026-05-26 — Run 14

### Attempted

- First end-to-end run with all 10 base models trained (RF, XGB, LightGBM, GBM, LR, CatBoost, TabularNN, MC Dropout, Deep Ensemble, KAN).
- KAN remediation chain validation: imodelsx v1.0.13 package patch (sed) + attribute injection (kan.py) under real 100K-row training.
- Maximum-information capture via `scripts/run14_observability.py` for per-model timing, KAN backend confirmation, LightGBM device, feature non-zero rates, blend weights, and artifact inventory.

### Failed

- TBD

### Fixed

- TBD

### Learned

- TBD

### Commits in this session

```
TBD — append commits as they land
bf2f665 fix(run14): add imodelsx package patch to launch script for Vast.ai  (prior session)
```

### Run summary

- Instance: Vast.ai `<id>`, RTX 4090, `<region>`, $`<rate>`/hr
- Elapsed: TBD h
- Cost: ~$TBD
- Test AUROC: TBD
- AUPRC: TBD
- F1: TBD
- MCC: TBD
- Brier: TBD
- Models trained: TBD / 10
- KAN backend: TBD
- Dead features: TBD / 78
