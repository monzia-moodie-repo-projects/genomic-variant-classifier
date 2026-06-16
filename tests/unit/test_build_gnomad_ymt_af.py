"""test_build_gnomad_ymt_af.py -- Monzia Moodie

Validates the pure (non-network) logic of build_gnomad_ymt_af plus the throttle-handling contract of
_post_retry. v3 reverts the Y path to SERIAL single-gene requests: gnomAD enforces a per-query cost
ceiling and rejects aliased multi-'variants' batches with HTTP 400, so one gene per request is the
cost-safe unit. _post_retry retries throttling (non-JSON 200 / 429 / 5xx) but FAILS FAST on a
non-retryable 4xx (e.g. 400 cost-limit), never silently.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_gnomad_ymt_af as B


# ---- norm_key / AF parsing -------------------------------------------------------------------------
def test_norm_key_y_mt_and_chr_prefix():
    assert B.norm_key("Y-12904862-A-G") == "Y:12904862:A:G"
    assert B.norm_key("M-8602-T-C") == "MT:8602:T:C"
    assert B.norm_key("chrY-100-A-G") == "Y:100:A:G"


def test_y_key_msy_passthrough_and_par_mapping():
    # MSY (male-specific Y) stays on Y
    assert B.y_key("Y-12709505-C-A") == "Y:12709505:C:A"
    assert B.y_key("chrY-100-A-G") == "Y:100:A:G"
    # PAR1: gnomAD reports on X, identical coordinate -> Y (the exact real-data failure case)
    assert B.y_key("X-1285848-G-A") == "Y:1285848:G:A"
    assert B.y_key("X-10001-A-G") == "Y:10001:A:G"        # PAR1 start (inclusive)
    assert B.y_key("X-2781479-A-G") == "Y:2781479:A:G"    # PAR1 end (inclusive)
    # PAR2: gnomAD reports on X, shifted by 98,813,480 -> Y
    assert B.y_key("X-155701383-A-G") == "Y:56887903:A:G"  # PAR2 start
    assert B.y_key("X-156030895-A-G") == "Y:57217415:A:G"  # PAR2 end


def test_y_key_rejects_non_par_x_and_malformed():
    assert B.y_key("X-10000-A-G") is None        # 1 before PAR1
    assert B.y_key("X-2781480-A-G") is None       # 1 after PAR1
    assert B.y_key("X-155701382-A-G") is None     # 1 before PAR2
    assert B.y_key("X-156030896-A-G") is None     # 1 after PAR2
    assert B.y_key("X-50000000-A-G") is None      # generic non-PAR X
    assert B.y_key("Y-100") is None               # malformed


def test_parse_y_af_maps_par_x_to_y_and_drops_nonpar_x():
    payload = {"data": {"gene": {"variants": [
        {"variant_id": "X-1285848-G-A", "exome": {"af": 0.003}, "genome": None},   # PAR1 X -> Y
        {"variant_id": "Y-100-A-G", "exome": {"af": 0.002}, "genome": {"af": 0.5}},  # MSY, exome wins
        {"variant_id": "Y-200-C-T", "exome": None, "genome": {"af": 0.01}},          # MSY genome fallback
        {"variant_id": "Y-300-G-A", "exome": {"af": None}, "genome": None},          # null AF -> skip
        {"variant_id": "X-50000000-A-G", "exome": {"af": 0.2}, "genome": None},      # non-PAR X -> drop
    ]}}}
    assert B.parse_y_af(payload) == {"Y:1285848:G:A": 0.003, "Y:100:A:G": 0.002, "Y:200:C:T": 0.01}


def test_parse_y_af_gene_not_found_is_empty_not_error():
    assert B.parse_y_af({"data": {"gene": None}, "errors": [{"message": "Gene not found"}]}) == {}


def test_parse_mt_af_computes_af_hom_and_skips_zero_an():
    payload = {"data": {"region": {"mitochondrial_variants": [
        {"variant_id": "M-8602-T-C", "an": 56000, "ac_hom": 56, "ac_het": 3},
        {"variant_id": "M-3308-T-C", "an": 56000, "ac_hom": 0, "ac_het": 9},
        {"variant_id": "M-9-G-A", "an": 0, "ac_hom": 1},
    ]}}}
    af = B.parse_mt_af(payload)
    assert af["MT:8602:T:C"] == pytest.approx(0.001)
    assert af["MT:3308:T:C"] == 0.0
    assert "MT:9:G:A" not in af


# ---- dirty gene-symbol cleaning --------------------------------------------------------------------
def test_clean_y_genes_splits_multigene_and_drops_freetext():
    raw = [
        "DDX3Y", "AKAP17A;ASMT;ASMTL;P2RY8", "DDX3Y;LOC108004538;USP9Y;UTY",
        "-", "nan",
        "covers 10 genes, none of which curated to show dosage sensitivity",
        "subset of 103 genes: SRY",
    ]
    genes = B.clean_y_genes(raw)
    assert genes == sorted(["DDX3Y", "AKAP17A", "ASMT", "ASMTL", "P2RY8", "LOC108004538", "USP9Y", "UTY"])
    assert all(" " not in g for g in genes)


def test_clean_y_genes_keeps_hyphenated_real_symbol_but_drops_bare_dash():
    assert B.clean_y_genes(["ZFY;ZFY-AS1", "-"]) == ["ZFY", "ZFY-AS1"]


# ---- _post_retry throttle contract (fake session, no network) --------------------------------------
class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status; self._payload = payload; self.text = text
    def json(self):
        if self._payload is None:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


class _Sess:
    def __init__(self, responses): self.responses = list(responses); self.calls = 0
    def post(self, *a, **k):
        r = self.responses[self.calls]; self.calls += 1; return r


def test_post_retry_retries_non_json_then_returns(monkeypatch):
    monkeypatch.setattr(B.time, "sleep", lambda *a, **k: None)
    s = _Sess([_Resp(200), _Resp(200), _Resp(200, payload={"data": {"gene": None}})])
    out = B._post_retry(s, "q", {}, max_retries=6, base_pause=0.0)
    assert out == {"data": {"gene": None}} and s.calls == 3


def test_post_retry_fails_fast_on_400_cost_limit(monkeypatch):
    # a 400 (cost-limit) must NOT be retried 8x -- raise on the first call
    monkeypatch.setattr(B.time, "sleep", lambda *a, **k: None)
    s = _Sess([_Resp(400, text="query cost exceeds limit")] + [_Resp(200, payload={})] * 8)
    with pytest.raises(RuntimeError, match="non-retryable"):
        B._post_retry(s, "q", {}, max_retries=8, base_pause=0.0)
    assert s.calls == 1  # failed fast, did not burn retries


def test_post_retry_retries_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(B.time, "sleep", lambda *a, **k: None)
    s = _Sess([_Resp(429), _Resp(200, payload={"data": {}})])
    assert B._post_retry(s, "q", {}, max_retries=4, base_pause=0.0) == {"data": {}} and s.calls == 2


def test_post_retry_raises_after_exhausting(monkeypatch):
    monkeypatch.setattr(B.time, "sleep", lambda *a, **k: None)
    s = _Sess([_Resp(200)] * 5)  # persistent non-JSON throttle
    with pytest.raises(RuntimeError, match="throttled/failed after"):
        B._post_retry(s, "q", {}, max_retries=5, base_pause=0.0)
    assert s.calls == 5


# ---- frame build / merge / cohort extraction --------------------------------------------------------
def test_build_ymt_frame_only_cohort_keys_and_gnomad_prefix():
    cohort = {"Y:100:A:G", "MT:8602:T:C"}
    df = B.build_ymt_frame(cohort, {"Y:100:A:G": 0.002, "Y:999:T:C": 0.3}, {"MT:8602:T:C": 0.001, "MT:5:X:Y": 0.4})
    assert set(df["variant_id"]) == {"gnomad:Y:100:A:G", "gnomad:MT:8602:T:C"}
    assert list(df.columns) == ["variant_id", "allele_freq"]


def test_merge_into_gnomad_dedups(tmp_path):
    base = pd.DataFrame({"variant_id": ["gnomad:1:5:A:G", "gnomad:2:9:C:T"], "allele_freq": [0.1, 0.2]})
    bp = tmp_path / "base.parquet"; base.to_parquet(bp)
    ymt = pd.DataFrame({"variant_id": ["gnomad:Y:100:A:G", "gnomad:MT:8602:T:C"], "allele_freq": [0.002, 0.001]})
    nb, ny, nc = B.merge_into_gnomad(ymt, str(bp), str(tmp_path / "merged.parquet"))
    assert (nb, ny, nc) == (2, 2, 4)
    assert len(pd.read_parquet(tmp_path / "merged.parquet")) == 4


def test_cohort_ymt_extracts_keys_and_raw_y_symbols(tmp_path):
    c = pd.DataFrame({
        "variant_id": ["clinvar:1:5:A:G", "clinvar:Y:100:A:G", "clinvar:Y:200:C:T", "clinvar:MT:8602:T:C"],
        "gene_symbol": ["BRCA1", "DDX3Y;USP9Y", "covers 10 genes, none curated", "MT-CO1"],
    })
    cp = tmp_path / "c.parquet"; c.to_parquet(cp)
    keys, y_raw = B.cohort_ymt(str(cp))
    assert keys == {"Y:100:A:G", "Y:200:C:T", "MT:8602:T:C"}
    assert B.clean_y_genes(y_raw) == ["DDX3Y", "USP9Y"]
