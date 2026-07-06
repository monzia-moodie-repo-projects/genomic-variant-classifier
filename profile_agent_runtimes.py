#!/usr/bin/env python3
"""profile_agent_runtimes.py (v2) -- Monzia Moodie

Per-agent runtime profile for setting principled execution timeouts (Phase 2).

v2 closes three defects found by cross-checking v1's suggestions against observed durations:
  (D1) the suggestion ignored OBSERVED runtime -- DataReadinessAgent ran 45.9s but was suggested 60s.
       Fix: suggested ceiling is now floored by observed_max * SAFETY_FACTOR, never below real runtime.
  (D2) the helper scan only looked for heavy-compute markers, not network/subprocess/config timeouts in
       helpers -- DatabaseFreshnessMonitorAgent's 4x20s detector calls were invisible.
       Fix: helper scan now extracts timeouts AND compute markers from helper files.
  (D3) the helper resolver missed `from package import submodule` imports -- every detector-backed agent
       (ModelInsights->sklearn, DataReadiness, AgentOps, FinOps) had its helper compute invisible.
       Fix: resolver now handles both `from pkg.sub import name` and `from pkg import sub` forms.

Sources per agent:
  STATIC  (own file + one level of LOCAL helper files, import-free):
    network timeouts, config timeouts (incl. annotated `name: int = 1800`), subprocess timeouts,
    heavy compute (torch/shap/sklearn/.fit/.predict/transformers/gudhi), network-call count.
  OBSERVED (agent_runs telemetry): max/last duration_ms over recorded executions.

SUGGESTED max_runtime_s = max(static_ceiling, observed_max * SAFETY_FACTOR). Agents with no static bound
AND no observed data are flagged NEEDS-EMPIRICAL (no guessed number). Flags mark observed-driven and
near-ceiling cases for review. You set the final per-agent numbers.

Pure stdlib, import-free w.r.t. the project. Run from the repo root:  python profile_agent_runtimes.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path.cwd()
SRC = REPO / "src"
ORCH = SRC / "genomic_variant_classifier" / "agent_layer" / "orchestrator.py"
STATE = SRC / "genomic_variant_classifier" / "agent_layer" / "agent_state.json"

SAFETY_FACTOR = 4.0  # observed_max * this = empirical floor for the ceiling (headroom for data growth)

_REG_RE = re.compile(r'"(\w+)"\s*:\s*_Lazy\(\s*["\']([^"\']+)["\']')
_TIMEOUT_RE = re.compile(r'\btimeout\s*=\s*(\d+(?:\.\d+)?)')
# config-field timeouts incl. type-annotated defaults: `install_timeout_s: int = 1800`
_CFG_TIMEOUT_RE = re.compile(r'\b(\w*timeout\w*)\s*(?::\s*[\w\[\], ]+?)?\s*=\s*(\d+(?:\.\d+)?)')
_SUBPROC_RE = re.compile(r'subprocess\.(run|check_output|check_call|Popen|call)\b')
_NET_RE = re.compile(r'\b(requests\.(?:get|post|head|put)|urllib\.request\.urlopen|urlopen|ftplib\.FTP)\b')
_HEAVY_RE = re.compile(r'(import\s+torch|import\s+shap|from\s+sklearn|\.fit\(|\.predict\(|import\s+transformers|import\s+gudhi)')
# captures: leading dots, module path, and the imported-names clause (single-line)
_LOCALIMP_RE = re.compile(r'^\s*from\s+(\.*)([\w.]*)\s+import\s+(.+)$', re.MULTILINE)


def registry_map(orch_text: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in _REG_RE.finditer(orch_text)}


def module_to_path(module: str) -> Path:
    return SRC / (module.replace(".", "/") + ".py")


def _extract_timeouts(src: str) -> tuple[list[float], list[tuple[str, float]], bool, int]:
    net_timeouts = [float(x) for x in _TIMEOUT_RE.findall(src)]
    cfg_timeouts = []
    for name, val in _CFG_TIMEOUT_RE.findall(src):
        if name == "timeout":
            continue  # bare inline kwarg already in net_timeouts
        cfg_timeouts.append((name, float(val)))
    has_subproc = bool(_SUBPROC_RE.search(src))
    n_net = len(_NET_RE.findall(src))
    return net_timeouts, cfg_timeouts, has_subproc, n_net


def _resolve(agent_module: str, dots: str, tail: str) -> str | None:
    if dots:  # relative
        pkg = agent_module.split(".")[:-1]
        up = len(dots) - 1
        base = pkg[: len(pkg) - up] if up > 0 else pkg
        return ".".join(base + (tail.split(".") if tail else []))
    if tail.startswith("genomic_variant_classifier"):
        return tail
    return None


def _imported_names(clause: str) -> list[str]:
    clause = clause.split("#", 1)[0].strip().strip("()")
    out = []
    for part in clause.split(","):
        nm = part.strip().split(" as ")[0].strip()
        if nm and nm != "*":
            out.append(nm)
    return out


def _helper_files(agent_module: str, src: str) -> list[tuple[str, Path]]:
    """Resolve one level of LOCAL helper imports to (module_name, path) for files that exist.
    Handles BOTH `from pkg.sub import name` (scan pkg.sub) and `from pkg import sub` (scan pkg.sub)."""
    found: dict[str, Path] = {}
    for dots, tail, clause in _LOCALIMP_RE.findall(src):
        mod = _resolve(agent_module, dots, tail)
        if not mod:
            continue
        # (a) the module itself: `from pkg.sub import name`
        p = module_to_path(mod)
        if p.exists():
            found[mod] = p
        # (b) each imported name as a submodule: `from pkg import sub`
        for nm in _imported_names(clause):
            sub = f"{mod}.{nm}" if mod else nm
            ps = module_to_path(sub)
            if ps.exists():
                found[sub] = ps
    return sorted(found.items())


def _scan_helpers(agent_module: str, src: str) -> dict:
    heavy: set[str] = set()
    net_timeouts: list[float] = []
    cfg_timeouts: list[tuple[str, float]] = []
    has_subproc = False
    n_net = 0
    vias: set[str] = set()
    for mod, path in _helper_files(agent_module, src):
        htext = path.read_text(encoding="utf-8", errors="replace")
        short = mod.rsplit(".", 1)[-1]
        for h in _HEAVY_RE.findall(htext):
            heavy.add(f"{h} (via {short})")
        nt, ct, sp, nn = _extract_timeouts(htext)
        if nt or ct or sp or _HEAVY_RE.search(htext):
            vias.add(short)
        net_timeouts += nt
        cfg_timeouts += ct
        has_subproc = has_subproc or sp
        n_net += nn
    return {
        "heavy": sorted(heavy),
        "net_timeouts": net_timeouts,
        "cfg_timeouts": cfg_timeouts,
        "has_subproc": has_subproc,
        "n_net": n_net,
        "vias": sorted(vias),
    }


def profile_one(agent_module: str, src: str) -> dict:
    own_net, own_cfg, own_sub, own_nnet = _extract_timeouts(src)
    own_heavy = sorted(set(_HEAVY_RE.findall(src)))
    helpers = _scan_helpers(agent_module, src)
    return {
        "own_net": own_net, "own_cfg": own_cfg, "own_sub": own_sub, "own_nnet": own_nnet,
        "own_heavy": own_heavy,
        "h_net": helpers["net_timeouts"], "h_cfg": helpers["cfg_timeouts"],
        "h_sub": helpers["has_subproc"], "h_nnet": helpers["n_net"],
        "h_heavy": helpers["heavy"], "vias": helpers["vias"],
    }


def suggest_timeout(p: dict, observed_max_s: float | None) -> tuple[float | None, str, list[str]]:
    flags: list[str] = []
    all_net = list(p["own_net"]) + list(p["h_net"])
    all_cfg = [v for _, v in p["own_cfg"]] + [v for _, v in p["h_cfg"]]
    has_subproc = p["own_sub"] or p["h_sub"]
    heavy = bool(p["own_heavy"]) or bool(p["h_heavy"])
    has_bound = bool(all_net or all_cfg or has_subproc)

    largest = max(all_net + all_cfg) if (all_net + all_cfg) else 0.0
    net_sum = sum(all_net)

    if not has_bound:
        static = None if heavy else 60.0
        static_note = ("heavy compute, no explicit wait" if heavy else "trivial (state-only); 60s floor")
    else:
        base = largest + net_sum
        margin = max(60.0, 0.25 * base)
        static = float(round(base + margin))
        static_note = "static bound (largest wait + sum network + 25% margin)"
        if heavy:
            static_note += "; +heavy compute present"

    obs_floor = (observed_max_s * SAFETY_FACTOR) if observed_max_s else 0.0

    if static is None and obs_floor <= 0:
        return None, "NEEDS-EMPIRICAL (" + static_note + "; no observed data)", flags
    if static is None:
        suggested = float(round(obs_floor))
        return suggested, f"empirical: observed_max x{SAFETY_FACTOR:g} (no static bound)", ["OBSERVED-DRIVEN"]

    suggested = max(static, float(round(obs_floor)))
    note = static_note
    if obs_floor > static:
        note = f"observed_max x{SAFETY_FACTOR:g} exceeds static -> observed-driven"
        flags.append("OBSERVED-DRIVEN")
    if observed_max_s and suggested and observed_max_s > 0.5 * suggested:
        flags.append("OBSERVED-NEAR-CEILING")
    return suggested, note, flags


def observed_durations() -> dict[str, dict]:
    if not STATE.exists():
        return {}
    try:
        data = json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    for name, recs in data.get("agent_runs", {}).items():
        ds = [r.get("duration_ms") for r in recs if isinstance(r, dict) and r.get("duration_ms") is not None]
        if ds:
            out[name] = {"max_ms": max(ds), "last_ms": ds[-1], "n": len(ds)}
    return out


def main() -> int:
    if not ORCH.exists():
        print(f"ERROR: orchestrator not found at {ORCH} -- run from repo root.", file=sys.stderr)
        return 2
    reg = registry_map(ORCH.read_text(encoding="utf-8"))
    if not reg:
        print("ERROR: no agents parsed from the lazy registry.", file=sys.stderr)
        return 2
    obs = observed_durations()

    print("=" * 112)
    print("PER-AGENT RUNTIME PROFILE v2  (own + one-level helper bounds, floored by observed duration)")
    print("=" * 112)
    flagged = []
    for name, spec in sorted(reg.items()):
        module = spec.rsplit(":", 1)[0]
        path = module_to_path(module)
        if not path.exists():
            print(f"\n{name}\n  MODULE FILE NOT FOUND: {path}")
            continue
        p = profile_one(module, path.read_text(encoding="utf-8", errors="replace"))
        o = obs.get(name)
        obs_max_s = (o["max_ms"] / 1000.0) if o else None
        suggested, note, flags = suggest_timeout(p, obs_max_s)

        net_all = sorted(set(p["own_net"] + p["h_net"]))
        nett = ",".join(str(int(x)) for x in net_all) or "-"
        cfg_all = p["own_cfg"] + p["h_cfg"]
        cfg = ",".join(f"{n}={int(v)}" for n, v in cfg_all) or "-"
        heavy_all = sorted(set(p["own_heavy"]) | set(p["h_heavy"]))
        heavy = ",".join(heavy_all) or "-"
        vias = ",".join(p["vias"]) or "-"
        obs_s = (f"max={o['max_ms']:.0f}ms last={o['last_ms']:.0f}ms n={o['n']}" if o else "no telemetry")
        sug = (f"{suggested:.0f}s" if suggested is not None else "NEEDS-EMPIRICAL")
        fl = ("  [" + ", ".join(flags) + "]") if flags else ""
        if flags:
            flagged.append((name, flags))

        print(f"\n{name}{fl}")
        print(f"  net_timeouts={nett}  n_net_calls={p['own_nnet']+p['h_nnet']}  subprocess={p['own_sub'] or p['h_sub']}")
        print(f"  cfg_timeouts={cfg}")
        print(f"  heavy_compute={heavy}   (helpers scanned: {vias})")
        print(f"  observed={obs_s}")
        print(f"  SUGGESTED max_runtime_s = {sug}   ({note})")

    print("\n" + "=" * 112)
    print(f"{len(reg)} agents profiled (SAFETY_FACTOR={SAFETY_FACTOR:g}).")
    if flagged:
        print("REVIEW these (observed runtime drives or approaches the ceiling -- confirm headroom):")
        for n, fs in flagged:
            print(f"  - {n}: {', '.join(fs)}")
    print("SUGGESTED values are conservative starting points; set final per-agent max_runtime_s yourself.")
    print("NOTE: helper scan is one level deep; 2nd-level helper compute relies on observed durations.")
    print("=" * 112)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
