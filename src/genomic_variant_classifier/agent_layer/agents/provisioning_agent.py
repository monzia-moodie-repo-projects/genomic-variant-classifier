"""provisioning_agent.py -- Monzia Moodie

ProvisioningAgent (BaseAgent) -- the SEPARATE, platform-agnostic GPU provisioning
agent (Vast.ai + RunPod). This is distinct from FinOpsAdvisorAgent, which stays
strictly recommend-only; this agent owns the provisioning lifecycle behind
budget-cap + HITL-approve + confirm-on-terminate.

THIS INCREMENT is select-and-record only -- it enforces the budget cap, picks the
cheapest verified offer across providers via offer_schema, records a `provisioning`
SharedState section, and writes an auditable run-doc (provisioning_docs). It makes
NO live calls and NEVER spends: live create/destroy backends (with the HITL-approve
gate on provision and confirm-on-terminate gate on teardown) are the next increment,
mirroring how FinOps deferred autonomous provisioning as a deliberate, gated non-goal
(docs/design/GPU_FINOPS_DESIGN.md, option A-GATED).

Constructed by the orchestrator as cls(shared_state); all knobs are injectable for
hermetic tests. With no offers provided -> 'skipped' (never crashes, never calls out).
"""
from __future__ import annotations

from pathlib import Path

from genomic_variant_classifier.agent_layer.agents.base_agent import BaseAgent
from genomic_variant_classifier.agent_layer.provisioning import offer_schema as OS
from genomic_variant_classifier.agent_layer.provisioning import provisioning_docs as PD
from genomic_variant_classifier.agent_layer.config import PROJECT_ROOT

# Defaults aligned with project ops: 4090 typ $0.33-0.77/hr, >=20 GB VRAM, ~15h run.
DEFAULT_BUDGET_CAP_PER_HR = 0.77
DEFAULT_EST_HOURS = 15.0
DEFAULT_MIN_VRAM_GB = 20.0


class ProvisioningAgent(BaseAgent):
    def __init__(
        self,
        shared_state,
        *,
        vast_offers: list[dict] | None = None,
        runpod_offers: list[dict] | None = None,
        runpod_cloud: str = "community",
        budget_cap_per_hr: float = DEFAULT_BUDGET_CAP_PER_HR,
        est_hours: float = DEFAULT_EST_HOURS,
        min_vram_gb: float = DEFAULT_MIN_VRAM_GB,
        require_verified: bool = True,
        root: str | None = None,
    ) -> None:
        super().__init__(shared_state)
        self._vast_offers = vast_offers
        self._runpod_offers = runpod_offers
        self._runpod_cloud = runpod_cloud
        self._budget_cap = budget_cap_per_hr
        self._est_hours = est_hours
        self._min_vram_gb = min_vram_gb
        self._require_verified = require_verified
        self._root = root if root is not None else str(PROJECT_ROOT)

    # -- helpers ---------------------------------------------------------------
    def _normalize_all(self) -> list[OS.CanonicalOffer]:
        offers: list[OS.CanonicalOffer] = []
        for raw in (self._vast_offers or []):
            try:
                offers.append(OS.normalize_vast(raw))
            except Exception as exc:  # malformed single offer -> skip it, never crash
                self.logger.warning("Skipping unparseable Vast offer: %s", exc)
        for raw in (self._runpod_offers or []):
            try:
                offers.append(OS.normalize_runpod(raw, cloud=self._runpod_cloud))
            except Exception as exc:
                self.logger.warning("Skipping unparseable RunPod offer: %s", exc)
        return offers

    # -- main ------------------------------------------------------------------
    def run(self, dry_run: bool = False) -> dict:
        self._log_start(dry_run)

        offers = self._normalize_all()
        if not offers:
            result = {"action": "skipped",
                      "reason": "no offers provided (wire a Vast/RunPod offers snapshot)"}
            self._log_finish(result)
            return result

        filters = {"max_price_per_hr": self._budget_cap, "min_vram_gb": self._min_vram_gb,
                   "require_verified": self._require_verified}
        best = OS.pick_offer(offers,
                             max_price_per_hr=self._budget_cap,
                             min_vram_gb=self._min_vram_gb,
                             require_verified=self._require_verified)

        # Candidate summary for the audit doc (cheapest-first, capped).
        ranked = sorted(offers, key=lambda o: o.price_per_hr)[:10]
        candidates = [{"provider": o.provider, "offer_id": o.offer_id,
                       "gpu_name": o.gpu_name, "price_per_hr": o.price_per_hr} for o in ranked]

        if best is None:
            event = PD.new_event(
                phase="select", provider=(ranked[0].provider if ranked else "vast"),
                offer_id=(ranked[0].offer_id if ranked else "none"),
                price_per_hr=(ranked[0].price_per_hr if ranked else 0.0),
                budget_cap_per_hr=self._budget_cap, est_hours=self._est_hours,
                within_budget=False, dry_run=dry_run, candidate_offers=candidates,
                search_filters=filters,
                reason="no verified offer within budget cap")
            doc = PD.write_provisioning_doc(self._root, event)
            event["doc_path"] = str(doc)
            PD.record_provisioning(self._get_section, self._update_section, event)
            if not dry_run:
                self._state.add_review_item(
                    f"Provisioning: NO offer within ${self._budget_cap}/hr cap "
                    f"({len(offers)} considered). Raise the cap or wait for cheaper offers.")
            result = {"action": "no_offer_within_budget", "n_offers": len(offers),
                      "budget_cap_per_hr": self._budget_cap, "doc": str(doc)}
            self._log_finish(result)
            return result

        event = PD.new_event(
            phase="select", provider=best.provider, offer_id=best.offer_id,
            gpu_name=best.gpu_name, num_gpus=best.num_gpus, vram_gb=best.vram_gb,
            price_per_hr=best.price_per_hr, budget_cap_per_hr=self._budget_cap,
            est_hours=self._est_hours, within_budget=True, dry_run=dry_run,
            candidate_offers=candidates, search_filters=filters,
            reason=f"cheapest verified {best.gpu_name} within ${self._budget_cap}/hr cap")
        doc = PD.write_provisioning_doc(self._root, event)
        event["doc_path"] = str(doc)
        PD.record_provisioning(self._get_section, self._update_section, event)

        # NOTE: live provisioning is intentionally NOT performed here. When the
        # Vast/RunPod create backends land, the actual spend goes behind:
        #   approved = self._require_approval(f"Provision {best.provider}:{best.offer_id} "
        #                                     f"at ${best.price_per_hr}/hr", dry_run=dry_run)
        # and teardown behind a separate confirm-on-terminate gate.
        result = {"action": "provision_select", "provider": best.provider,
                  "offer_id": best.offer_id, "price_per_hr": best.price_per_hr,
                  "est_cost_usd": event["est_cost_usd"], "doc": str(doc)}
        self._log_finish(result)
        return result
