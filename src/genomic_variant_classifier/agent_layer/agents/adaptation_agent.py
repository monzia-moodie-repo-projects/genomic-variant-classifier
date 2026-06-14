"""
adaptation_agent.py - Adaptation & Migration-Evaluation Agent
=============================================================
Monzia Moodie

Closes the loop that VersionMonitorAgent leaves open. VersionMonitorAgent only
DETECTS new upstream versions and writes them to the 'version_monitor' shared
state section; nothing consumes those alerts. AdaptationAgent consumes them and,
for each NEW candidate, either:

  - PLANS it (default, safe, fast): records the candidate to an append-only
    ledger and surfaces a review item, OR
  - EVALUATES it (opt-in via ADAPTATION_EVALUATE=1 / config.evaluate=True):
    builds a THROWAWAY virtual environment, installs the project plus the
    candidate version, runs the test suite in isolation, parses the result,
    and records pass/fail + the first failing test to the ledger.

Candidate sources (all read from the 'version_monitor' section in one go):
  - deps_major_bumps : list of "name installed -> latest"   (kind="deps_major")
  - python_alert     : non-empty string                      (kind="python")
  - pyg_abi_alert    : non-empty string                      (kind="pyg_abi")

Hard safety boundary (matches the project's action policy):
  AdaptationAgent NEVER mutates the live environment, requirements files, or
  configuration. It evaluates in an isolated venv and REPORTS. Migration stays a
  human decision; detection is never suppressed (the review-item alert always
  fires for new candidates).

Running record:
  Every candidate seen, planned, evaluated, validated, and verified is written
  as one JSON object per line to an append-only ledger
  (default logs/adaptation/adaptation_ledger.jsonl). The ledger is the durable
  audit trail; the 'adaptation' state section holds the latest-run summary.

dry_run contract (per BaseAgent): no external writes (no ledger file, no venv,
no evaluation, no review item); the internal 'adaptation' state section IS still
updated so the dry-run output is meaningful.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from genomic_variant_classifier.agent_layer.agents.base_agent import BaseAgent
from genomic_variant_classifier.agent_layer.shared_state import SharedState


# ---------------------------------------------------------------------------
# Configuration (safe defaults; overridable from the environment so the agent
# works unchanged under the orchestrator, which constructs it as cls(state)).
# ---------------------------------------------------------------------------
def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class AdaptationConfig:
    # Plan-only by default. Heavy venv build + install + test is OPT-IN.
    evaluate: bool = False
    # Cap heavy evaluations per run (each can take many minutes).
    max_candidates_per_run: int = 1
    ledger_path: Path = field(
        default_factory=lambda: Path("logs/adaptation/adaptation_ledger.jsonl")
    )
    venv_root: Path = field(default_factory=lambda: Path("logs/adaptation/venvs"))
    project_root: Path = field(default_factory=lambda: Path("."))
    # Args appended after the venv python; default is a fast, representative subset.
    test_command: tuple[str, ...] = ("-m", "pytest", "-x", "-q", "tests/unit")
    install_timeout_s: int = 1800
    test_timeout_s: int = 1800
    keep_venv: bool = False

    @classmethod
    def from_env(cls) -> "AdaptationConfig":
        c = cls(evaluate=_env_flag("ADAPTATION_EVALUATE"))
        _led = os.environ.get("ADAPTATION_LEDGER")
        if _led:
            c.ledger_path = Path(_led)
        _root = os.environ.get("ADAPTATION_PROJECT_ROOT")
        if _root:
            c.project_root = Path(_root)
        _max = os.environ.get("ADAPTATION_MAX_CANDIDATES")
        if _max and _max.isdigit():
            c.max_candidates_per_run = int(_max)
        if _env_flag("ADAPTATION_KEEP_VENV"):
            c.keep_venv = True
        return c


# ---------------------------------------------------------------------------
# Candidate model + parsing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    kind: str  # "deps_major" | "python" | "pyg_abi"
    name: str
    from_version: str
    to_version: str
    raw: str

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.name}:{self.to_version}"


_BUMP_RE = re.compile(r"^\s*(?P<name>\S+)\s+(?P<frm>\S+)\s*->\s*(?P<to>\S+)\s*$")


def parse_candidates(version_monitor: dict) -> list[Candidate]:
    """Extract integration candidates from the version_monitor section.
    Resilient to missing keys and malformed bump strings."""
    out: list[Candidate] = []
    for raw in version_monitor.get("deps_major_bumps", []) or []:
        m = _BUMP_RE.match(str(raw))
        if m:
            out.append(
                Candidate("deps_major", m.group("name"), m.group("frm"),
                          m.group("to"), str(raw))
            )
        else:
            # keep it as a candidate even if unparseable, so nothing is dropped
            out.append(Candidate("deps_major", str(raw).strip(), "", "", str(raw)))
    py = str(version_monitor.get("python_alert") or "").strip()
    if py:
        out.append(
            Candidate("python", "python",
                      str(version_monitor.get("python_running") or ""), "", py)
        )
    abi = str(version_monitor.get("pyg_abi_alert") or "").strip()
    if abi:
        out.append(Candidate("pyg_abi", "torch-geometric-abi", "", "", abi))
    return out


# ---------------------------------------------------------------------------
# pytest output parser (verdict extraction)
# ---------------------------------------------------------------------------
_COUNT_RE = re.compile(r"(?P<n>\d+)\s+(?P<kind>passed|failed|errors?|skipped)\b")
_FAIL_RE = re.compile(r"^(?:FAILED|ERROR)\s+(?P<node>\S+)", re.MULTILINE)


def parse_pytest_output(stdout: str, stderr: str = "") -> dict:
    """Parse pytest -q output into counts + the first failing node id.
    Counts are read from the LAST summary-style line (the one containing a
    duration 'in <n>s') to avoid matching incidental numbers in tracebacks."""
    text = (stdout or "") + "\n" + (stderr or "")
    counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}

    summary_line = ""
    for line in text.splitlines():
        low = line.lower()
        if (" in " in low and low.rstrip().endswith(("s", "s)", ")")) and
                any(k in low for k in ("passed", "failed", "error", "skipped"))):
            summary_line = line  # keep the last one
    scope = summary_line if summary_line else text
    for m in _COUNT_RE.finditer(scope):
        n = int(m.group("n"))
        k = m.group("kind")
        if k == "passed":
            counts["passed"] = n
        elif k == "failed":
            counts["failed"] = n
        elif k.startswith("error"):
            counts["error"] = n
        elif k == "skipped":
            counts["skipped"] = n

    fm = _FAIL_RE.search(text)
    counts["first_failure"] = fm.group("node") if fm else ""
    return counts


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class AdaptationAgent(BaseAgent):
    """Consumes VersionMonitorAgent alerts; plans or isolates-and-tests each new
    candidate; keeps an append-only ledger; always alerts. Report-only."""

    def __init__(self, shared_state: SharedState,
                 config: AdaptationConfig | None = None) -> None:
        super().__init__(shared_state)
        self.config = config or AdaptationConfig.from_env()

    # -- ledger I/O ------------------------------------------------------
    def _load_ledger(self) -> list[dict]:
        path = self.config.ledger_path
        if not path.exists():
            return []
        out: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a partially-written final line
        return out

    def _append_ledger(self, entries: list[dict]) -> None:
        path = self.config.ledger_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, default=str) + "\n")

    @staticmethod
    def _evaluated_keys(ledger: list[dict]) -> set[str]:
        # only terminal (evaluated) entries count as "done"; planned ones do not,
        # so a planned candidate is still picked up once evaluation is enabled.
        return {e.get("candidate_key", "") for e in ledger
                if e.get("action") == "evaluated"}

    # -- isolated evaluation (heavy; opt-in) -----------------------------
    def _evaluate_candidate(self, cand: Candidate) -> dict:
        """Build a throwaway venv, install project + candidate, run tests.
        Never touches the live environment. Returns a verdict record."""
        t0 = time.time()
        stamp = int(t0)
        venv_dir = self.config.venv_root / f"{cand.name}_{cand.to_version or 'x'}_{stamp}"
        rec: dict[str, Any] = {
            "venv_path": str(venv_dir), "install_ok": False, "install_error": "",
            "test_returncode": None, "passed": 0, "failed": 0, "error": 0,
            "skipped": 0, "first_failure": "", "duration_s": 0.0,
            "verdict": "inconclusive",
        }
        try:
            venv_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)],
                           check=True, capture_output=True, text=True, timeout=600)
            sub = "Scripts" if os.name == "nt" else "bin"
            exe = "python.exe" if os.name == "nt" else "python"
            vpy = str(venv_dir / sub / exe)
            subprocess.run([vpy, "-m", "pip", "install", "--upgrade", "pip"],
                           capture_output=True, text=True, timeout=600)
            install_cmd = [vpy, "-m", "pip", "install", "-e", str(self.config.project_root)]
            if cand.kind == "deps_major" and cand.to_version:
                install_cmd.append(f"{cand.name}=={cand.to_version}")
            ires = subprocess.run(install_cmd, capture_output=True, text=True,
                                  timeout=self.config.install_timeout_s)
            rec["install_ok"] = ires.returncode == 0
            if not rec["install_ok"]:
                rec["install_error"] = (ires.stderr or ires.stdout or "")[-2000:]
                return rec
            tcmd = [vpy, *self.config.test_command]
            tres = subprocess.run(tcmd, capture_output=True, text=True,
                                  cwd=str(self.config.project_root),
                                  timeout=self.config.test_timeout_s)
            rec["test_returncode"] = tres.returncode
            parsed = parse_pytest_output(tres.stdout, tres.stderr)
            for k in ("passed", "failed", "error", "skipped", "first_failure"):
                rec[k] = parsed[k]
            rec["verdict"] = (
                "compatible"
                if (tres.returncode == 0 and parsed["failed"] == 0 and parsed["error"] == 0)
                else "incompatible"
            )
        except subprocess.TimeoutExpired as exc:
            rec["install_error"] = f"timeout: {exc}"
        except Exception as exc:  # noqa: BLE001 - record, never crash the agent
            rec["install_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            rec["duration_s"] = round(time.time() - t0, 1)
            if not self.config.keep_venv:
                shutil.rmtree(venv_dir, ignore_errors=True)
        return rec

    # -- entry point -----------------------------------------------------
    def run(self, dry_run: bool = False) -> dict:
        self._log_start(dry_run)

        vm = self._get_section("version_monitor")
        candidates = parse_candidates(vm)
        ledger = self._load_ledger()
        done = self._evaluated_keys(ledger)
        new = [c for c in candidates if c.key not in done]
        self.logger.info(
            "Candidates: %d total, %d new (not yet evaluated).",
            len(candidates), len(new),
        )

        will_evaluate = self.config.evaluate and not dry_run
        new_entries: list[dict] = []
        n_eval = 0
        for cand in new:
            base = {
                "ts": self._now_iso(), "candidate_key": cand.key, "kind": cand.kind,
                "name": cand.name, "from_version": cand.from_version,
                "to_version": cand.to_version, "raw": cand.raw, "dry_run": dry_run,
            }
            if will_evaluate and n_eval < self.config.max_candidates_per_run:
                self.logger.info("Evaluating %s in an isolated venv ...", cand.key)
                ev = self._evaluate_candidate(cand)
                new_entries.append({**base, "action": "evaluated", **ev})
                n_eval += 1
            else:
                note = ("dry_run" if dry_run
                        else "plan-only (set ADAPTATION_EVALUATE=1 to evaluate)")
                new_entries.append({**base, "action": "planned",
                                    "verdict": "planned", "note": note})

        # External write (ledger): suppressed under dry_run.
        if new_entries and not dry_run:
            self._append_ledger(new_entries)

        incompatible = [e for e in new_entries if e.get("verdict") == "incompatible"]
        summary = {
            "last_run": self._now_iso(),
            "n_candidates": len(candidates),
            "n_new": len(new),
            "n_evaluated": n_eval,
            "n_incompatible": len(incompatible),
            "evaluate_mode": will_evaluate,
            "latest": [
                {"candidate": e["candidate_key"], "action": e["action"],
                 "verdict": e.get("verdict")}
                for e in new_entries
            ],
            "dry_run": dry_run,
        }
        # Internal state update is allowed in dry_run (keeps output meaningful).
        self._update_section("adaptation", summary)

        # Detection is never suppressed: alert the human about new candidates.
        if new and not dry_run:
            msg = f"AdaptationAgent: {len(new)} new version candidate(s) for integration"
            if n_eval:
                msg += f"; evaluated {n_eval} ({len(incompatible)} incompatible)"
            else:
                msg += " (plan-only; not yet evaluated)"
            self._state.add_review_item(msg)

        result = {"action": "evaluate" if will_evaluate else "plan", **summary}
        self._log_finish(result)
        return result
