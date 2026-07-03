"""
src/genomic_variant_classifier/data/alphafold_features.py
=========================================================
Structure-feature extraction from AlphaFold mmCIF files -- Phase D, AlphaFold connector.

Pure, dependency-light extractors shared by the cohort parquet builder
(scripts/build_alphafold_parquet.py) and the unit tests. No pandas, no logging
(kept out of library modules per project convention); returns plain dicts.

Four per-residue features are produced (all validated against the real
AF-E7ENB7 structure, a 98-residue BRCA1 fragment):

  plddt                 float  Per-residue pLDDT confidence (0-100), read directly
                               from the mmCIF ``B_iso_or_equiv`` column, which for
                               AlphaFold models equals the per-residue pLDDT
                               (verified equal to ``_ma_qa_metric_local``).
  secondary_structure   int    0=loop/coil, 1=helix, 2=sheet.  Parsed from the
                               ``_struct_conf`` DSSP records when the file carries
                               them (AlphaFold ships DSSP-computed secondary
                               structure in current revisions); falls back to a
                               coordinate-derived backbone-geometry assignment when
                               ``_struct_conf`` is absent or empty (older revisions,
                               e.g. some full-length models, ship it empty).
  rsa                   float  Relative solvent accessibility (0-1).  Shrake-Rupley
                               solvent-accessible surface area over ALL atoms,
                               normalised by the Tien et al. (2013) theoretical
                               per-residue maximum ASA, then CLAMPED to [0, 1].
                               See ``_RSA_CLAMP_NOTE`` for the reasoning behind the
                               clamp.
  ca_xyz                tuple  (x, y, z) of the residue's C-alpha atom, in Angstrom.
                               Used by the builder to compute the real 3-D C-alpha
                               Euclidean distance to the nearest annotated active /
                               binding site (``dist_to_active_site``).

The residue key throughout is the mmCIF ``auth_seq_id`` (1-based UniProt residue
numbering), which matches AlphaMissense / HGVSp ``protein_pos`` (also 1-based).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

# --- physical constants -----------------------------------------------------

# Bondi van der Waals radii (Angstrom) by element symbol.
_VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}
_DEFAULT_VDW = 1.70
_WATER_PROBE = 1.40

# Tien et al. (2013) "theoretical" maximum accessible surface area (Angstrom^2)
# per residue, used to normalise absolute SASA into relative SASA (RSA).
_MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

# RSA clamp reasoning (documented deliberately; not a silent fix):
#   Relative solvent accessibility is DEFINED as a fraction in [0, 1] -- the ratio
#   of a residue's observed SASA to its maximum possible SASA.  The Tien et al.
#   normalisation constants are derived from residues in extended Gly-X-Gly
#   tripeptides.  Chain-terminal residues carry extra atoms (e.g. the C-terminal
#   OXT) and unusually high exposure, so their raw ratio can slightly exceed 1.0
#   (observed max 1.17 at the C-terminus of the validation structure).  This is a
#   normalisation artefact, not a geometry error: the underlying SASA is correct.
#   We therefore CLAMP RSA to [0, 1] (the definitional range).  A genuine geometry
#   failure would produce values far outside this range or negative SASA; those are
#   caught by ``_RSA_FAIL_LOUD_MAX`` and raise rather than being clamped.
_RSA_CLAMP_NOTE = "RSA clamped to [0,1]; terminal-residue normalisation artefact per Tien et al."
_RSA_FAIL_LOUD_MAX = 1.5  # raw RSA above this indicates a real geometry failure -> raise

# Shrake-Rupley sphere sampling density (points per atom).  192 is a good
# accuracy/speed trade-off for cohort-scale batch computation.
_SPHERE_N = 192

# Sentinel defaults (mirror ProteinStructurePipeline / real_data_prep).
DEFAULT_PLDDT = 50.0
DEFAULT_RSA = 0.5
DEFAULT_SECONDARY = 0
DEFAULT_DIST_ACTIVE = 100.0


class CIFParseError(ValueError):
    """Raised when an mmCIF file cannot be parsed into usable atom records."""


# --- mmCIF atom-site parsing ------------------------------------------------

def parse_atom_site(cif_text: str) -> list[dict]:
    """
    Parse the ``_atom_site`` loop of an mmCIF file into a list of atom dicts.

    Each atom dict has: element, atom_id, comp (3-letter residue), seq_id
    (int, auth_seq_id, 1-based), x, y, z (float, Angstrom), bfactor (float, pLDDT).

    Raises CIFParseError if no atoms are found.
    """
    lines = cif_text.splitlines()
    columns: list[str] = []
    in_header = False
    in_data = False
    atoms: list[dict] = []

    for raw in lines:
        s = raw.strip()
        if s.startswith("_atom_site."):
            columns.append(s.split(".", 1)[1])
            in_header = True
            continue
        if in_header and not s.startswith("_atom_site."):
            in_header = False
            in_data = True
        if in_data:
            if s.startswith("ATOM") or s.startswith("HETATM"):
                parts = s.split()
                if len(parts) < len(columns):
                    continue
                rec = dict(zip(columns, parts))
                try:
                    atom_id = rec["label_atom_id"]
                    type_symbol = rec.get("type_symbol", "")
                    element = _element_of(type_symbol, atom_id)
                    atoms.append({
                        "element": element,
                        "atom_id": atom_id,
                        "comp": rec["label_comp_id"],
                        "seq_id": int(rec["auth_seq_id"]),
                        "x": float(rec["Cartn_x"]),
                        "y": float(rec["Cartn_y"]),
                        "z": float(rec["Cartn_z"]),
                        "bfactor": float(rec["B_iso_or_equiv"]),
                    })
                except (KeyError, ValueError):
                    continue
            elif s == "#" or s.startswith("loop_") or s.startswith("_"):
                # end of the atom_site loop
                if atoms:
                    break
                in_data = False

    if not atoms:
        raise CIFParseError("no _atom_site ATOM records found")
    return atoms


def _element_of(type_symbol: str, atom_id: str) -> str:
    """Resolve the chemical element from the mmCIF type_symbol or atom name."""
    ts = (type_symbol or "").strip().upper()
    if ts in _VDW_RADII:
        return ts
    # fall back to the atom-name first letter (SD/SG -> S, otherwise C/N/O)
    a = atom_id.strip().upper()
    if a.startswith("S"):
        return "S"
    if a and a[0] in ("C", "N", "O"):
        return a[0]
    return "C"


# --- pLDDT ------------------------------------------------------------------

def per_residue_plddt(atoms: list[dict]) -> dict[int, float]:
    """Return {auth_seq_id: pLDDT} from the C-alpha atom B-factor of each residue."""
    out: dict[int, float] = {}
    for a in atoms:
        if a["atom_id"] == "CA":
            out[a["seq_id"]] = a["bfactor"]
    return out


# --- C-alpha coordinates ----------------------------------------------------

def per_residue_ca(atoms: list[dict]) -> dict[int, tuple[float, float, float]]:
    """Return {auth_seq_id: (x, y, z)} of each residue's C-alpha atom."""
    out: dict[int, tuple[float, float, float]] = {}
    for a in atoms:
        if a["atom_id"] == "CA":
            out[a["seq_id"]] = (a["x"], a["y"], a["z"])
    return out


def per_residue_comp(atoms: list[dict]) -> dict[int, str]:
    """Return {auth_seq_id: 3-letter residue name}."""
    out: dict[int, str] = {}
    for a in atoms:
        out.setdefault(a["seq_id"], a["comp"])
    return out


# --- secondary structure ----------------------------------------------------

def parse_struct_conf(cif_text: str) -> dict[int, int]:
    """
    Parse the ``_struct_conf`` loop into {auth_seq_id: ss_code}, where ss_code is
    0=loop, 1=helix, 2=sheet.  Returns an EMPTY dict when the loop is absent or
    carries no records (the caller then falls back to coordinate-derived SS).

    conf_type_id mapping (AlphaFold DSSP vocabulary):
      HELX_*  -> 1 (alpha RH, left-handed PP, 3-10, pi helices)
      STRN    -> 2 (beta strand)
      TURN_*, BEND, and anything else -> 0 (not regular secondary structure)
    """
    lines = cif_text.splitlines()
    columns: list[str] = []
    in_header = False
    in_data = False
    ss: dict[int, int] = {}

    for raw in lines:
        s = raw.strip()
        if s.startswith("_struct_conf."):
            columns.append(s.split(".", 1)[1])
            in_header = True
            continue
        if in_header and not s.startswith("_struct_conf."):
            in_header = False
            in_data = True
        if in_data:
            if not s or s.startswith("#") or s.startswith("loop_") or s.startswith("_"):
                break
            parts = _split_cif_row(s)
            if len(parts) < len(columns):
                continue
            rec = dict(zip(columns, parts))
            try:
                conf_type = rec["conf_type_id"]
                beg = int(rec["beg_auth_seq_id"])
                end = int(rec["end_auth_seq_id"])
            except (KeyError, ValueError):
                continue
            code = _ss_code(conf_type)
            for r in range(beg, end + 1):
                ss[r] = code
    return ss


def _ss_code(conf_type_id: str) -> int:
    c = (conf_type_id or "").upper()
    if c.startswith("HELX"):
        return 1
    if c.startswith("STRN"):
        return 2
    return 0


def _split_cif_row(s: str) -> list[str]:
    """Split an mmCIF data row, respecting double-quoted tokens."""
    out: list[str] = []
    cur = ""
    in_q = False
    for ch in s:
        if ch == '"':
            in_q = not in_q
            continue
        if ch == " " and not in_q:
            if cur:
                out.append(cur)
                cur = ""
            continue
        cur += ch
    if cur:
        out.append(cur)
    return out


def secondary_structure_from_coords(atoms: list[dict]) -> dict[int, int]:
    """
    Coordinate-derived secondary-structure fallback used only when ``_struct_conf``
    is absent/empty.  Assigns helix (1) from the characteristic i -> i+4 C-alpha
    spacing of alpha helices and sheet (2) from extended-backbone geometry, else
    loop (0).  This is a geometric heuristic, NOT full DSSP hydrogen-bond analysis;
    it is a documented approximation that applies only when the authoritative DSSP
    records are missing from the file.
    """
    ca = per_residue_ca(atoms)
    seqs = sorted(ca)
    ss: dict[int, int] = {s: 0 for s in seqs}
    if len(seqs) < 5:
        return ss

    def dist(a: int, b: int) -> Optional[float]:
        if a in ca and b in ca:
            (x1, y1, z1), (x2, y2, z2) = ca[a], ca[b]
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        return None

    # Alpha helix: C-alpha(i) to C-alpha(i+4) ~ 6.2 A (5.0-6.5 A window);
    # C-alpha(i) to C-alpha(i+3) ~ 5.0-5.5 A. Assign helix to residues whose
    # local i,i+3,i+4 spacing matches the helical signature.
    for i in seqs:
        d3 = dist(i, i + 3)
        d4 = dist(i, i + 4)
        if d4 is not None and 4.8 <= d4 <= 6.6 and d3 is not None and 4.6 <= d3 <= 6.3:
            for r in (i, i + 1, i + 2, i + 3, i + 4):
                if r in ss:
                    ss[r] = 1

    # Beta strand: extended chain, C-alpha(i) to C-alpha(i+2) ~ 6.5-7.0 A with
    # near-linear i,i+1,i+2 (not already assigned helix).
    for i in seqs:
        d2 = dist(i, i + 2)
        if d2 is not None and 6.0 <= d2 <= 7.2:
            if ss.get(i, 0) == 0 and ss.get(i + 2, 0) == 0:
                for r in (i, i + 1, i + 2):
                    if r in ss and ss[r] == 0:
                        ss[r] = 2
    return ss


# --- solvent accessibility (Shrake-Rupley) ----------------------------------

def _sphere_points(n: int) -> list[tuple[float, float, float]]:
    """Fibonacci-lattice unit-sphere points for Shrake-Rupley sampling."""
    pts: list[tuple[float, float, float]] = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (i / float(n - 1)) * 2.0 if n > 1 else 0.0
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        pts.append((math.cos(theta) * r, y, math.sin(theta) * r))
    return pts


_SPHERE = _sphere_points(_SPHERE_N)


def per_residue_rsa(atoms: list[dict]) -> dict[int, float]:
    """
    Compute per-residue relative solvent accessibility (RSA) in [0, 1] via a
    Shrake-Rupley SASA over ALL atoms, normalised by Tien et al. max-ASA and
    clamped to [0, 1] (see _RSA_CLAMP_NOTE).

    Raises CIFParseError if any residue's RAW RSA exceeds _RSA_FAIL_LOUD_MAX or is
    negative -- that indicates a real geometry failure, not a terminal artefact,
    and must fail loud rather than be silently clamped.
    """
    n = len(atoms)
    if n == 0:
        return {}
    # Vectorized Shrake-Rupley neighbour search (cKDTree) + numpy occlusion.
    # Numerically identical to the original O(n^2) scan: the ball query returns a
    # superset of neighbours and the exact per-pair test below (unchanged) filters
    # it to the same set; the occlusion test is the same strict inequality. Proven
    # bit-identical (max RSA diff 0.0) on real and synthetic structures. See the
    # test_rsa_vectorized_matches_naive_reference guard.
    radii = np.array(
        [_VDW_RADII.get(a["element"], _DEFAULT_VDW) + _WATER_PROBE for a in atoms],
        dtype=float,
    )
    coords = np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=float)
    _sphere = np.asarray(_SPHERE, dtype=float)
    n_sphere = len(_SPHERE)
    max_r = float(radii.max())
    tree = cKDTree(coords)

    sasa_res: dict[int, float] = {}
    for i in range(n):
        ri = radii[i]
        ci = coords[i]
        cand = tree.query_ball_point(ci, ri + max_r)
        if cand:
            cand_arr = np.fromiter((j for j in cand if j != i), dtype=int)
        else:
            cand_arr = np.empty(0, dtype=int)
        if cand_arr.size:
            d = coords[cand_arr] - ci
            cutoff = ri + radii[cand_arr]
            neigh = cand_arr[(d * d).sum(1) < cutoff * cutoff]
        else:
            neigh = cand_arr
        if neigh.size == 0:
            accessible = n_sphere
        else:
            pts = ci + ri * _sphere
            nc = coords[neigh]
            nr2 = radii[neigh] * radii[neigh]
            diff = pts[:, None, :] - nc[None, :, :]
            d2 = (diff * diff).sum(2)
            buried = (d2 < nr2[None, :]).any(1)
            accessible = int(n_sphere - buried.sum())
        area = 4.0 * math.pi * ri * ri * accessible / float(n_sphere)
        seq = atoms[i]["seq_id"]
        sasa_res[seq] = sasa_res.get(seq, 0.0) + area

    comp = per_residue_comp(atoms)
    rsa: dict[int, float] = {}
    for seq, sasa in sasa_res.items():
        max_asa = _MAX_ASA.get(comp.get(seq, ""), None)
        if not max_asa:
            rsa[seq] = DEFAULT_RSA
            continue
        raw = sasa / max_asa
        if raw < 0.0 or raw > _RSA_FAIL_LOUD_MAX:
            raise CIFParseError(
                f"residue {seq} RSA={raw:.3f} outside [0,{_RSA_FAIL_LOUD_MAX}] "
                f"-- geometry failure, not a terminal artefact"
            )
        rsa[seq] = min(1.0, max(0.0, raw))  # clamp; see _RSA_CLAMP_NOTE
    return rsa


# --- secondary structure dispatch (parse-first, coords-fallback) ------------

def per_residue_secondary_structure(cif_text: str, atoms: list[dict]) -> dict[int, int]:
    """
    Authoritative DSSP from ``_struct_conf`` when present; coordinate-derived
    fallback otherwise.  Returns {auth_seq_id: ss_code}.
    """
    ss = parse_struct_conf(cif_text)
    if ss:
        return ss
    return secondary_structure_from_coords(atoms)
