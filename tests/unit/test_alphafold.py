"""
tests/unit/test_alphafold.py
============================
Unit tests for the AlphaFold structural-feature connector, extractor library, and
cohort coverage gate.

Coverage:
  Extractor (alphafold_features):
    1.  parse_atom_site on a real-format CIF -> correct atom count, 1-based seq_id
    2.  per_residue_plddt -> B_iso values, correct residues
    3.  parse_struct_conf (DSSP parse-first) -> helix/sheet/loop codes
    4.  secondary_structure_from_coords (fallback) -> non-degenerate
    5.  per_residue_rsa -> all in [0,1] (clamp), core < terminus ordering
    6.  RSA fail-loud guard raises on a fabricated geometry failure
    7.  3-D C-alpha distance: adjacent residues ~3.8 A
    8.  residue indexing off-by-one tripwire (seq_id starts at 1, not 0)
  Connector (alphafold.AlphaFoldConnector):
    9.  wt_aa match -> real features attached
    10. wt_aa isoform MISMATCH -> fail-closed to sentinel default
    11. missing protein_pos -> default
    12. empty df -> columns present, no rows
    13. stub mode (no parquet) -> all defaults
    14. TABULAR_FEATURES membership -> the 4 AF features present
  Coverage gate:
    15. AF features non-constant on a real cohort sample (fraction-at-sentinel gate)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from genomic_variant_classifier.data import alphafold_features as aff
from genomic_variant_classifier.data.alphafold import (
    AlphaFoldConnector,
    DEFAULT_PLDDT,
    DEFAULT_RSA,
    DEFAULT_SECONDARY,
    DEFAULT_DIST_ACTIVE,
)

# The real AF-E7ENB7 structure (98-residue BRCA1 fragment, 101 KB, carries DSSP
# _struct_conf) is VENDORED as a COMMITTED test fixture.
#
# FIXED 2026-07-11. This used to read:
#
#     _CIF_DIR = Path("data/raw/cache/alphafold")     # CWD-relative, and GITIGNORED
#     requires_cif = pytest.mark.skipif(not _has_real_cif(),
#                                       reason="real cached CIF not present")
#
# with the comment "The two real cached CIFs live under the repo ... Skip gracefully if
# absent in CI." They do NOT live under the repo: data/raw/ is gitignored, so the CIF is
# absent from every clean checkout. The consequences, all measured:
#
#   * These SEVEN tests -- the guards for pLDDT parsing, relative solvent accessibility,
#     secondary structure, the 3-D C-alpha distance, and the residue OFF-BY-ONE tripwire
#     -- never ran in CI and never would. A fresh checkout skips them, and CI is always
#     fresh. They were dead everywhere except one laptop.
#
#   * The suite was NOT IDEMPOTENT. Run 1 on a clean clone: 1805 passed, 17 skipped.
#     Run 2 on the SAME clone: 1812 passed, 10 skipped. Something in run 1 DOWNLOADED the
#     CIF from https://alphafold.ebi.ac.uk (ProteinStructurePipeline defaults cache_dir to
#     a CWD-relative "data/raw/cache/alphafold", protein_pipeline.py:372) and wrote it into
#     the checkout; run 2's collection-time skipif then found it and the seven came alive.
#     A suite whose result depends on whether it has been run before is not a suite.
#
#   * git status could not see any of it -- data/raw/ is gitignored, so the tool that
#     would have flagged the pollution was blindfolded. cf.
#     docs/incidents/INCIDENT_2026-06-14_data-junction-dangling.md, which already recorded
#     that tests "write to the REAL data/".
#
# The fixture is now committed under tests/fixtures/, addressed RELATIVE TO THIS FILE (not
# the working directory). The seven tests therefore run everywhere, always, deterministically:
# no network, no ambient state, no skip.
_CIF_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "alphafold"
_E7ENB7 = _CIF_DIR / "AF-E7ENB7-F1-model_v4.cif"


def _has_real_cif() -> bool:
    return _E7ENB7.exists()


# Retained ONLY as a fail-loud tripwire. The fixture is committed, so this must never fire.
# If it ever does, the checkout is broken -- it is NOT a normal condition to skip past.
requires_cif = pytest.mark.skipif(
    not _has_real_cif(),
    reason=(
        f"VENDORED CIF FIXTURE MISSING at {_E7ENB7}. This should be impossible -- the "
        f"fixture is committed to the repository. A skip here means a BROKEN CHECKOUT, "
        f"not an absent optional artifact. Do not 'fix' this by restoring the old "
        f"data/raw/cache path: that is what made these seven tests dead in CI."
    ),
)


# ---------------------------------------------------------------------------
# Extractor tests (real structure)
# ---------------------------------------------------------------------------
@requires_cif
def test_parse_atom_site_real():
    atoms = aff.parse_atom_site(_E7ENB7.read_text(encoding="utf-8", errors="replace"))
    assert len(atoms) > 400
    ca = aff.per_residue_ca(atoms)
    assert min(ca) == 1, "auth_seq_id must be 1-based"
    assert max(ca) == 98


@requires_cif
def test_plddt_real():
    atoms = aff.parse_atom_site(_E7ENB7.read_text(errors="replace"))
    plddt = aff.per_residue_plddt(atoms)
    assert len(plddt) == 98
    assert plddt[1] == pytest.approx(62.97, abs=0.01)
    assert plddt[12] == pytest.approx(96.94, abs=0.01)
    assert all(0 <= v <= 100 for v in plddt.values())


@requires_cif
def test_struct_conf_dssp_parse_first():
    cif = _E7ENB7.read_text(errors="replace")
    ss = aff.parse_struct_conf(cif)
    assert ss  # non-empty: this file carries DSSP records
    assert ss[3] == 1 and ss[21] == 1, "helix 3-21"
    assert ss[23] == 2, "strand"
    assert ss.get(22, 0) == 0, "turn -> loop"


@requires_cif
def test_secondary_structure_coord_fallback_nondegenerate():
    atoms = aff.parse_atom_site(_E7ENB7.read_text(errors="replace"))
    ss = aff.secondary_structure_from_coords(atoms)
    codes = set(ss.values())
    assert 1 in codes, "helix-rich fragment must yield some helix from coords"
    assert len(ss) == 98


@requires_cif
def test_rsa_clamped_and_ordered():
    atoms = aff.parse_atom_site(_E7ENB7.read_text(errors="replace"))
    rsa = aff.per_residue_rsa(atoms)
    assert all(0.0 <= v <= 1.0 for v in rsa.values()), "RSA must be clamped to [0,1]"
    assert rsa[11] < rsa[98], "buried core (11) < exposed C-terminus (98)"


@requires_cif
def test_ca_distance_adjacent_real():
    atoms = aff.parse_atom_site(_E7ENB7.read_text(errors="replace"))
    ca = aff.per_residue_ca(atoms)
    (x1, y1, z1), (x2, y2, z2) = ca[1], ca[2]
    d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    assert 3.5 < d < 4.1, f"adjacent C-alpha spacing ~3.8 A, got {d:.2f}"


def test_rsa_fail_loud_guard():
    """
    A corrupt record with several well-separated atoms assigned to ONE residue yields
    RSA far above the fail-loud bound -> must raise, not clamp. (A single isolated atom
    only reaches ~1.2, within the terminal-artefact range that is correctly clamped;
    the guard fires on genuine geometry failures like this multi-atom pileup.)
    """
    atoms = [
        {"element": "C", "atom_id": "CA", "comp": "GLY", "seq_id": 1,
         "x": 0.0, "y": 0.0, "z": 0.0, "bfactor": 50.0},
        {"element": "C", "atom_id": "C", "comp": "GLY", "seq_id": 1,
         "x": 50.0, "y": 0.0, "z": 0.0, "bfactor": 50.0},
        {"element": "C", "atom_id": "N", "comp": "GLY", "seq_id": 1,
         "x": 0.0, "y": 50.0, "z": 0.0, "bfactor": 50.0},
    ]
    with pytest.raises(aff.CIFParseError):
        aff.per_residue_rsa(atoms)


def test_residue_indexing_off_by_one_tripwire():
    """Guard against a 0-based regression: a minimal 2-residue CIF must index 1,2."""
    cif = (
        "data_x\n#\n"
        "loop_\n_atom_site.group_PDB\n_atom_site.type_symbol\n_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n_atom_site.auth_seq_id\n_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.B_iso_or_equiv\n"
        "ATOM C CA ALA 1 0.0 0.0 0.0 90.0\n"
        "ATOM C CA GLY 2 3.8 0.0 0.0 80.0\n#\n"
    )
    plddt = aff.per_residue_plddt(aff.parse_atom_site(cif))
    assert set(plddt) == {1, 2}, "residues must be 1-based (1,2), not (0,1)"


# ---------------------------------------------------------------------------
# Connector tests
# ---------------------------------------------------------------------------
def _write_af_parquet(tmp_path: Path) -> Path:
    af = pd.DataFrame({
        "uniprot_accession": ["P38398", "P38398"],
        "residue_pos": [5, 11],
        "plddt": [86.06, 95.88], "rsa": [0.30, 0.48], "ss": [1, 1], "dist_active": [12.5, 20.1],
    })
    p = tmp_path / "af.parquet"
    af.to_parquet(p, index=False)
    return p


def _write_uniprot_index(tmp_path: Path) -> Path:
    seq = "MDLSALRVEEV" + "A" * 90  # pos5 = A (ALA), pos11 = V (VAL)
    up = pd.DataFrame({"gene_symbol": ["BRCA1"], "uniprot_id": ["P38398"], "sequence": [seq]})
    p = tmp_path / "up.parquet"
    up.to_parquet(p, index=False)
    return p


def test_connector_wt_match_attaches(tmp_path):
    c = AlphaFoldConnector(
        parquet_path=_write_af_parquet(tmp_path),
        uniprot_index_path=_write_uniprot_index(tmp_path),
    )
    df = pd.DataFrame({"gene_symbol": ["BRCA1"], "protein_pos": [5], "wt_aa": ["ALA"]})
    out = c.annotate_dataframe(df)
    assert out.loc[0, "alphafold_plddt"] == pytest.approx(86.06)
    assert out.loc[0, "secondary_structure_context"] == 1


def test_connector_wt_mismatch_fails_closed(tmp_path):
    c = AlphaFoldConnector(
        parquet_path=_write_af_parquet(tmp_path),
        uniprot_index_path=_write_uniprot_index(tmp_path),
    )
    # claims TRP at pos5 but the sequence has ALA -> isoform mismatch -> default
    df = pd.DataFrame({"gene_symbol": ["BRCA1"], "protein_pos": [5], "wt_aa": ["TRP"]})
    out = c.annotate_dataframe(df)
    assert out.loc[0, "alphafold_plddt"] == DEFAULT_PLDDT
    assert out.loc[0, "solvent_accessibility"] == DEFAULT_RSA


def test_connector_missing_protein_pos_defaults(tmp_path):
    c = AlphaFoldConnector(
        parquet_path=_write_af_parquet(tmp_path),
        uniprot_index_path=_write_uniprot_index(tmp_path),
    )
    df = pd.DataFrame({"gene_symbol": ["BRCA1"], "protein_pos": [None], "wt_aa": ["ALA"]})
    out = c.annotate_dataframe(df)
    assert out.loc[0, "alphafold_plddt"] == DEFAULT_PLDDT


def test_connector_empty_df():
    c = AlphaFoldConnector(parquet_path=None)
    out = c.annotate_dataframe(pd.DataFrame(columns=["gene_symbol", "protein_pos", "wt_aa"]))
    assert "alphafold_plddt" in out.columns
    assert len(out) == 0


def test_connector_stub_mode_all_defaults():
    c = AlphaFoldConnector(parquet_path=None)
    df = pd.DataFrame({"gene_symbol": ["BRCA1"], "protein_pos": [5], "wt_aa": ["ALA"]})
    out = c.annotate_dataframe(df)
    assert out.loc[0, "alphafold_plddt"] == DEFAULT_PLDDT
    assert out.loc[0, "secondary_structure_context"] == DEFAULT_SECONDARY
    assert out.loc[0, "dist_to_active_site"] == DEFAULT_DIST_ACTIVE


def test_af_features_in_tabular_features():
    from genomic_variant_classifier.models.variant_ensemble import TABULAR_FEATURES
    for feat in ("alphafold_plddt", "solvent_accessibility",
                 "secondary_structure_context", "dist_to_active_site"):
        assert feat in TABULAR_FEATURES


# ---------------------------------------------------------------------------
# Coverage gate
# ---------------------------------------------------------------------------
@requires_cif
def test_coverage_gate_features_nonconstant(tmp_path):
    """
    The gate that would have caught the stub-constant state: after building a small
    real parquet and annotating a cohort sample, the AF features must be non-constant
    (fraction at sentinel below threshold). Here we assert directly on a built parquet.
    """
    atoms = aff.parse_atom_site(_E7ENB7.read_text(errors="replace"))
    plddt = aff.per_residue_plddt(atoms)
    rsa = aff.per_residue_rsa(atoms)
    ss = aff.per_residue_secondary_structure(_E7ENB7.read_text(errors="replace"), atoms)
    # non-constant across residues
    assert len(set(round(v, 2) for v in plddt.values())) > 5
    assert len(set(round(v, 2) for v in rsa.values())) > 5
    assert len(set(ss.values())) >= 2  # at least loop + helix
    # fraction at sentinel default must be low (these are REAL values, not stubs)
    frac_plddt_sentinel = sum(1 for v in plddt.values() if v == DEFAULT_PLDDT) / len(plddt)
    assert frac_plddt_sentinel < 0.5, "most residues should have real (non-sentinel) pLDDT"


# ---------------------------------------------------------------------------
# Fetch-path tests (offline, mocked) -- added with the v6/API URL fix.
# These close the gap that let a stale v4 URL pass 15/15 while the live build
# fetched zero structures: no prior test exercised _download_cif.
# ---------------------------------------------------------------------------
import importlib.util as _ilu

_BUILDER = Path(__file__).resolve().parents[2] / "scripts" / "build_alphafold_parquet.py"


def _load_builder():
    spec = _ilu.spec_from_file_location("build_alphafold_parquet", _BUILDER)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeResp:
    def __init__(self, status=200, text="", payload=None):
        self.status_code = status
        self.ok = 200 <= status < 300
        self.text = text
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


_CIF_BODY = (
    "data_AF-P04637-F1\n#\nloop_\n_atom_site.group_PDB\n_atom_site.type_symbol\n"
    "ATOM C CA MET 1 0.0 0.0 0.0 90.0\n#\n"
)
_CIF_URL = "https://alphafold.ebi.ac.uk/files/AF-P04637-F1-model_v6.cif"
_CANON = "MEEPQSDPSV"  # stand-in canonical sequence for fetch-path mocks (post canonical-select fix)


def test_download_cif_resolves_api_and_writes_server_version(tmp_path, monkeypatch):
    bap = _load_builder()

    def fake_get(url, timeout=None, **kw):
        if "api/prediction" in url:
            return _FakeResp(200, payload=[{"uniprotSequence": _CANON, "cifUrl": _CIF_URL}])
        if url == _CIF_URL:
            return _FakeResp(200, text=_CIF_BODY)
        return _FakeResp(404)

    monkeypatch.setattr(bap.requests, "get", fake_get)
    out = bap._download_cif("P04637", tmp_path, _CANON)
    assert out is not None
    assert out.name == "AF-P04637-F1-model_v6.cif", "must save under server version, not v4"
    assert out.read_text().lstrip().startswith("data_")


def test_download_cif_rejects_non_cif_payload(tmp_path, monkeypatch):
    bap = _load_builder()

    def fake_get(url, timeout=None, **kw):
        if "api/prediction" in url:
            return _FakeResp(200, payload=[{"uniprotSequence": _CANON, "cifUrl": _CIF_URL}])
        if url == _CIF_URL:
            return _FakeResp(200, text="<html><body>404 Not Found</body></html>")
        return _FakeResp(404)

    monkeypatch.setattr(bap.requests, "get", fake_get)
    out = bap._download_cif("P04637", tmp_path, _CANON)
    assert out is None, "an HTML error page must never be accepted as a CIF"
    assert list(tmp_path.glob("*.cif")) == [], "nothing may be written on rejection"


def test_download_cif_api_miss_returns_none(tmp_path, monkeypatch):
    bap = _load_builder()
    monkeypatch.setattr(bap.requests, "get", lambda url, timeout=None, **kw: _FakeResp(404))
    assert bap._download_cif("NOSUCH", tmp_path, _CANON) is None
    assert list(tmp_path.glob("*.cif")) == []


def test_resolve_cif_url_reads_current_version(tmp_path, monkeypatch):
    bap = _load_builder()
    monkeypatch.setattr(
        bap.requests, "get",
        lambda url, timeout=None, **kw: _FakeResp(200, payload=[{"uniprotSequence": _CANON, "cifUrl": _CIF_URL}]),
    )
    assert bap._resolve_cif_url("P04637", _CANON) == _CIF_URL


# ---------------------------------------------------------------------------
# RSA vectorization guards (added with the O(n) Shrake-Rupley fix).
# The fast cKDTree+numpy path MUST equal the naive O(n^2) path bit-for-bit,
# and must be clearly faster (regression tripwire against reverting to O(n^2)).
# ---------------------------------------------------------------------------
import random as _random
import time as _time


def _synth_atoms(n_res, seed=7):
    _random.seed(seed)
    comps = list(aff._MAX_ASA)
    elems = ["C", "C", "C", "N", "O", "C", "C", "S"]
    atoms = []
    side = int(round(n_res ** (1 / 3))) + 1
    idx = 0
    for r in range(1, n_res + 1):
        cx = (idx % side) * 3.8
        cy = ((idx // side) % side) * 3.8
        cz = (idx // (side * side)) * 3.8
        idx += 1
        comp = comps[r % len(comps)]
        for k in range(8):
            atoms.append({
                "element": elems[k], "comp": comp, "seq_id": r,
                "x": cx + _random.uniform(-1.5, 1.5),
                "y": cy + _random.uniform(-1.5, 1.5),
                "z": cz + _random.uniform(-1.5, 1.5),
            })
    return atoms


def _rsa_naive(atoms):
    """Independent O(n^2) reference: the pre-optimization Shrake-Rupley, inline."""
    import math as _m
    n = len(atoms)
    radii = [aff._VDW_RADII.get(a["element"], aff._DEFAULT_VDW) + aff._WATER_PROBE for a in atoms]
    xs = [a["x"] for a in atoms]
    ys = [a["y"] for a in atoms]
    zs = [a["z"] for a in atoms]
    sasa_res = {}
    for i in range(n):
        ri = radii[i]
        xi, yi, zi = xs[i], ys[i], zs[i]
        neigh = []
        for j in range(n):
            if j == i:
                continue
            cutoff = ri + radii[j]
            dx = xi - xs[j]
            dy = yi - ys[j]
            dz = zi - zs[j]
            if dx * dx + dy * dy + dz * dz < cutoff * cutoff:
                neigh.append(j)
        accessible = 0
        for (sx, sy, sz) in aff._SPHERE:
            px = xi + ri * sx
            py = yi + ri * sy
            pz = zi + ri * sz
            buried = False
            for j in neigh:
                dx = px - xs[j]
                dy = py - ys[j]
                dz = pz - zs[j]
                if dx * dx + dy * dy + dz * dz < radii[j] * radii[j]:
                    buried = True
                    break
            if not buried:
                accessible += 1
        area = 4.0 * _m.pi * ri * ri * accessible / float(len(aff._SPHERE))
        seq = atoms[i]["seq_id"]
        sasa_res[seq] = sasa_res.get(seq, 0.0) + area
    comp = aff.per_residue_comp(atoms)
    rsa = {}
    for seq, sasa in sasa_res.items():
        max_asa = aff._MAX_ASA.get(comp.get(seq, ""), None)
        if not max_asa:
            rsa[seq] = aff.DEFAULT_RSA
            continue
        raw = sasa / max_asa
        if raw < 0.0 or raw > aff._RSA_FAIL_LOUD_MAX:
            raise aff.CIFParseError("naive reference geometry failure")
        rsa[seq] = min(1.0, max(0.0, raw))
    return rsa


def test_rsa_vectorized_matches_naive_reference():
    atoms = _synth_atoms(120)
    fast = aff.per_residue_rsa(atoms)
    slow = _rsa_naive(atoms)
    assert set(fast) == set(slow), "residue key sets diverge"
    maxdiff = max(abs(fast[k] - slow[k]) for k in fast)
    assert maxdiff < 1e-12, f"vectorized RSA diverges from naive by {maxdiff:.2e}"


def test_rsa_empty_input_returns_empty():
    assert aff.per_residue_rsa([]) == {}


def test_rsa_performance_beats_naive():
    """Regression tripwire: reverting to O(n^2) makes fast==naive (ratio ~1.0)."""
    atoms = _synth_atoms(500)
    t = _time.time(); aff.per_residue_rsa(atoms); t_fast = _time.time() - t
    t = _time.time(); _rsa_naive(atoms); t_slow = _time.time() - t
    assert t_fast < 0.80 * t_slow, f"fast={t_fast:.2f}s not < 0.80*naive={t_slow:.2f}s (O(n^2) regression?)"


# ---------------------------------------------------------------------------
# Canonical-record selection guards (added with the isoform-numbering fix).
# AlphaFold returns one record per isoform (entryId AF-{acc}-{N}-F1) with isoform
# residue numbering; the record whose sequence matches our canonical index MUST be
# chosen, and a giant/isoform-only entry with no canonical match MUST yield None --
# never a mis-numbered isoform structure attached to canonical protein_pos.
# ---------------------------------------------------------------------------
def test_resolve_cif_url_selects_canonical_record(monkeypatch):
    bap = _load_builder()
    canon = "M" * 968
    def fake_get(url, timeout=None, **kw):
        if "api/prediction" in url:
            return _FakeResp(200, payload=[
                {"entryId": "AF-P49588-F1", "uniprotSequence": canon,
                 "cifUrl": "https://x/AF-P49588-F1-model_v6.cif"},
                {"entryId": "AF-P49588-2-F1", "uniprotSequence": "M" * 992,
                 "cifUrl": "https://x/AF-P49588-2-F1-model_v6.cif"}])
        return _FakeResp(404)
    monkeypatch.setattr(bap.requests, "get", fake_get)
    got = bap._resolve_cif_url("P49588", canon)
    assert got == "https://x/AF-P49588-F1-model_v6.cif", "must pick canonical, not the longer isoform"


def test_resolve_cif_url_rejects_when_no_canonical_match(monkeypatch):
    bap = _load_builder()
    canon = "D" * 7570  # giant; only a 2649-residue isoform exists in AFDB
    def fake_get(url, timeout=None, **kw):
        if "api/prediction" in url:
            return _FakeResp(200, payload=[
                {"entryId": "AF-Q03001-3-F1", "uniprotSequence": "D" * 2649,
                 "cifUrl": "https://x/AF-Q03001-3-F1-model_v6.cif"}])
        return _FakeResp(404)
    monkeypatch.setattr(bap.requests, "get", fake_get)
    assert bap._resolve_cif_url("Q03001", canon) is None, "isoform must NOT substitute for canonical"


def test_resolve_cif_url_empty_canonical_returns_none(monkeypatch):
    bap = _load_builder()
    monkeypatch.setattr(
        bap.requests, "get",
        lambda url, timeout=None, **kw: _FakeResp(200, payload=[{"uniprotSequence": "X", "cifUrl": "u"}]),
    )
    assert bap._resolve_cif_url("ACC", "") is None