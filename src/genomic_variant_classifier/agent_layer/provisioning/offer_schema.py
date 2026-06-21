"""offer_schema.py  --  Author: Monzia Moodie

Provider-agnostic GPU-offer schema for the ProvisioningAgent (design item B2 of
docs/design/GPU_FINOPS_DESIGN.md). Normalizes Vast.ai marketplace offers and
RunPod GPU-type pricing into ONE canonical shape, then selects the cheapest
offer that clears the run's floors and the budget cap.

Money-safety: this module performs NO provisioning, makes NO live account calls,
and SPENDS NOTHING. It only reads already-fetched offer dicts and returns a
selection. Live create/terminate lives in the per-provider backends, gated by
HITL approval in the agent. Keeping selection here (pure, deterministic) makes
it unit-testable without credentials or network.

Two provider shapes are flattened:
  * Vast.ai  -- a list of concrete host OFFERS from `vastai search offers --raw`
               (each a biddable listing with its own id, $/hr, DLP, PCIe, ...).
  * RunPod   -- a list of GPU TYPES (each a gpuTypeId + per-hour price for the
               Secure and/or Community cloud + VRAM); you request a type and
               RunPod assigns the machine. There is no per-host id to bid on.

Both collapse to CanonicalOffer. `provider`/`offer_id`/`extras` preserve enough
to build the real create call in the backend later.

NOTE ON SOURCE KEYS: the Vast `--raw` and RunPod field names are mapped with
documented fallbacks (`_first`). They MUST be validated against one real
`vastai search offers --raw` dump and one real RunPod GPU-types response before
the agent is allowed to spend; `normalize_*` raise loudly (KeyError) if no
recognized price/id field is present rather than silently defaulting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

VALID_PROVIDERS = ("vast", "runpod")


@dataclass(frozen=True)
class CanonicalOffer:
    provider: str            # "vast" | "runpod"
    offer_id: str            # vast offer id | runpod gpuTypeId
    gpu_name: str
    num_gpus: int
    vram_gb: float
    price_per_hr: float      # USD/hr for the whole offer (num_gpus included)
    disk_gb: float           # host/available disk (vast) or 0.0 if N/A (runpod)
    region: str
    verified: bool
    reliability: float       # 0..1 (vast reliability2; runpod -> 1.0 if unknown)
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.provider not in VALID_PROVIDERS:
            raise ValueError(f"unknown provider {self.provider!r}")
        if self.price_per_hr < 0:
            raise ValueError(f"negative price_per_hr {self.price_per_hr}")


def _first(d: dict, keys: Iterable[str], *, required: bool = True,
           default: Any = None) -> Any:
    """Return d[k] for the first present, non-None k in keys."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    if required:
        raise KeyError(f"none of {list(keys)} present in offer dict "
                       f"(have: {sorted(d)[:12]})")
    return default


def _as_verified(value: Any) -> bool:
    """Normalize provider verified fields from bool/string/numeric forms."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"verified", "true", "yes", "1"}
    return False


def normalize_vast(raw: dict) -> CanonicalOffer:
    """One row of `vastai search offers --raw` (JSON dict) -> CanonicalOffer.

    Vast reports VRAM in MB under gpu_ram/gpu_total_ram; price under
    dph_total (the field the prior Provision function sorted on)."""
    vram_mb = float(_first(raw, ("gpu_ram", "gpu_total_ram", "gpu_mem_bw"),
                           required=False, default=0.0) or 0.0)
    # If a *_ram field is actually already GB-scale (< 1000), treat as GB.
    vram_gb = vram_mb / 1024.0 if vram_mb >= 1000 else vram_mb
    return CanonicalOffer(
        provider="vast",
        offer_id=str(_first(raw, ("id", "ask_contract_id", "machine_id"))),
        gpu_name=str(_first(raw, ("gpu_name",), required=False, default="?")),
        num_gpus=int(_first(raw, ("num_gpus",), required=False, default=1) or 1),
        vram_gb=vram_gb,
        price_per_hr=float(_first(raw, ("dph_total", "dph", "min_bid"))),
        disk_gb=float(_first(raw, ("disk_space",), required=False, default=0.0)
                      or 0.0),
        region=str(_first(raw, ("geolocation", "country"), required=False,
                          default="?")),
        verified=_as_verified(
            _first(raw, ("verification", "verified", "is_verified"),
                   required=False, default=None)
        ),
        reliability=float(_first(raw, ("reliability2", "reliability"),
                                 required=False, default=0.0) or 0.0),
        extras={k: raw.get(k) for k in
                ("dlperf", "pcie_bandwidth", "cuda_max_good", "inet_down")
                if k in raw},
    )


def normalize_runpod(raw: dict, *, cloud: str = "secure") -> CanonicalOffer:
    """One RunPod GPU-type dict -> CanonicalOffer.

    cloud: "secure" or "community" selects which price column to use. RunPod
    exposes securePrice / communityPrice (USD/hr per GPU) and memoryInGb."""
    if cloud not in ("secure", "community"):
        raise ValueError(f"cloud must be secure|community, got {cloud!r}")
    price_keys = (("securePrice", "lowestPrice", "price") if cloud == "secure"
                  else ("communityPrice", "lowestPrice", "price"))
    return CanonicalOffer(
        provider="runpod",
        offer_id=str(_first(raw, ("id", "gpuTypeId", "displayName"))),
        gpu_name=str(_first(raw, ("displayName", "id"))),
        num_gpus=int(_first(raw, ("count", "gpuCount"), required=False,
                            default=1) or 1),
        vram_gb=float(_first(raw, ("memoryInGb", "vram", "memoryGb"),
                             required=False, default=0.0) or 0.0),
        price_per_hr=float(_first(raw, price_keys)),
        disk_gb=0.0,  # RunPod disk is requested at create-time, not part of offer
        region=str(_first(raw, ("dataCenterId", "region"), required=False,
                          default=cloud)),
        verified=True,   # RunPod Secure/Community are vetted; no per-host flag
        reliability=1.0,
        extras={"cloud": cloud,
                **{k: raw.get(k) for k in ("secureCloud", "communityCloud")
                   if k in raw}},
    )


def pick_offer(
    offers: list[CanonicalOffer], *,
    max_price_per_hr: float,            # the BUDGET CAP (budget_usd / est_hours)
    min_vram_gb: float = 20.0,
    min_disk_gb: float = 0.0,           # 0 => don't filter on disk (runpod)
    require_verified: bool = True,
    providers: tuple[str, ...] = VALID_PROVIDERS,
    min_reliability: float = 0.0,
) -> CanonicalOffer | None:
    """Cheapest offer clearing every floor AND the budget cap. None if no fit.

    Ties: lower price, then higher vram, then higher reliability."""
    if max_price_per_hr <= 0:
        raise ValueError(f"max_price_per_hr (budget cap) must be > 0, "
                         f"got {max_price_per_hr}")
    elig = [
        o for o in offers
        if o.provider in providers
        and o.price_per_hr <= max_price_per_hr
        and o.vram_gb >= min_vram_gb
        and (min_disk_gb <= 0 or o.disk_gb >= min_disk_gb)
        and (not require_verified or o.verified)
        and o.reliability >= min_reliability
    ]
    if not elig:
        return None
    return sorted(elig, key=lambda o: (o.price_per_hr, -o.vram_gb,
                                       -o.reliability))[0]


def summarize(offer: CanonicalOffer | None) -> str:
    if offer is None:
        return "no offer cleared the floors + budget cap"
    return (f"{offer.provider}:{offer.offer_id}  ${offer.price_per_hr:.4f}/hr  "
            f"{offer.gpu_name} x{offer.num_gpus}  {offer.vram_gb:.0f}GB VRAM  "
            f"{offer.region}  verified={offer.verified}")
