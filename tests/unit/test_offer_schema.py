"""test_offer_schema.py -- Monzia Moodie
Provider-agnostic offer schema (agent_layer/provisioning/offer_schema.py): Vast/RunPod
normalization, cross-provider budget-capped selection, floors, and loud failures.
Imports from the installed package (src-layout) -- NOT a top-level module.
"""
from genomic_variant_classifier.agent_layer.provisioning import offer_schema as S

VAST = [
    {"id": 40057706, "gpu_name": "RTX_4090", "num_gpus": 1, "gpu_ram": 24576,
     "dph_total": 0.3347, "disk_space": 1365, "geolocation": "Quebec, CA",
     "verification": "verified", "reliability2": 0.994, "dlperf": 98.4,
     "pcie_bandwidth": 25.0},
    {"id": 12239065, "gpu_name": "RTX_4090", "num_gpus": 1, "gpu_ram": 24576,
     "dph_total": 0.4685, "disk_space": 1204, "geolocation": "Utah, US",
     "verification": "verified", "reliability2": 0.998, "dlperf": 96.7,
     "pcie_bandwidth": 22.1},
    {"id": 99999999, "gpu_name": "RTX_4090", "num_gpus": 1, "gpu_ram": 24576,
     "dph_total": 0.30, "disk_space": 100, "geolocation": "X",
     "verification": "unverified", "reliability2": 0.5},
]
RUNPOD = [
    {"id": "NVIDIA GeForce RTX 4090", "displayName": "RTX 4090",
     "memoryInGb": 24, "securePrice": 0.69, "communityPrice": 0.34},
    {"id": "NVIDIA A100 80GB", "displayName": "A100 80GB",
     "memoryInGb": 80, "securePrice": 1.89, "communityPrice": 1.19},
]

def test_normalize_vast():
    o = S.normalize_vast(VAST[0])
    assert o.provider == "vast" and o.offer_id == "40057706"
    assert abs(o.price_per_hr - 0.3347) < 1e-9
    assert abs(o.vram_gb - 24.0) < 0.1 and o.verified and o.disk_gb == 1365

def test_normalize_runpod_secure_vs_community():
    sec = S.normalize_runpod(RUNPOD[0], cloud="secure")
    com = S.normalize_runpod(RUNPOD[0], cloud="community")
    assert sec.price_per_hr == 0.69 and com.price_per_hr == 0.34
    assert sec.provider == "runpod" and sec.vram_gb == 24 and sec.verified

def test_cross_provider_pick_cheapest_within_cap():
    offers = [S.normalize_vast(v) for v in VAST] + \
             [S.normalize_runpod(r, cloud="community") for r in RUNPOD]
    best = S.pick_offer(offers, max_price_per_hr=0.50, min_vram_gb=20, require_verified=True)
    assert best.provider == "vast" and best.offer_id == "40057706"

def test_budget_cap_excludes_expensive():
    offers = [S.normalize_runpod(r, cloud="secure") for r in RUNPOD]
    best = S.pick_offer(offers, max_price_per_hr=0.80, min_vram_gb=20)
    assert best.offer_id == "NVIDIA GeForce RTX 4090"
    assert S.pick_offer(offers, max_price_per_hr=0.50, min_vram_gb=20) is None

def test_require_verified_excludes_unverified_cheaper():
    offers = [S.normalize_vast(v) for v in VAST]
    assert S.pick_offer(offers, max_price_per_hr=1.0, min_vram_gb=20, require_verified=True).offer_id == "40057706"
    assert S.pick_offer(offers, max_price_per_hr=1.0, min_vram_gb=20, require_verified=False).offer_id == "99999999"

def test_min_vram_floor():
    offers = [S.normalize_runpod(r, cloud="community") for r in RUNPOD]
    assert S.pick_offer(offers, max_price_per_hr=2.0, min_vram_gb=48).offer_id == "NVIDIA A100 80GB"

def test_loud_failure_on_missing_price():
    import pytest
    with pytest.raises(KeyError):
        S.normalize_vast({"id": 1, "gpu_name": "RTX_4090"})

def test_zero_cap_rejected():
    import pytest
    with pytest.raises(ValueError):
        S.pick_offer([], max_price_per_hr=0)

def test_no_fit_returns_none():
    assert S.pick_offer([], max_price_per_hr=1.0) is None
