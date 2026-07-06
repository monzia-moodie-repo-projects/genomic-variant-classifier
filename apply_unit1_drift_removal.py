#!/usr/bin/env python3
# apply_unit1_drift_removal.py
# Unit 1 edit-script: remove the vestigial, never-functional drift-detection
# path from training_lifecycle_agent.py.
#
# Root cause (investigated, evidence-backed this session):
#   _check_drift() imported a phantom `detect_drift` from ewc_utils that NEVER
#   existed (git -S empty; not anywhere in tree). Guarded by try/except, it
#   returned False on EVERY run since written -- a silent dead path. Retrain
#   triggering is owned by the inbox (DATA_UPDATED / FEATURE_INSTABILITY);
#   statistical drift is owned by the dedicated DriftMonitorBase agents.
#
# This script is anchored, idempotent, and ABORTS on any unexpected file state.
# Run from repo root:  python apply_unit1_drift_removal.py
from __future__ import annotations
import io
import sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/agent_layer/agents/training_lifecycle_agent.py")

# Each transform: (name, anchor, replacement, applied_signature)
# - If anchor present  -> replace.
# - elif applied_signature present (and anchor absent) -> already applied, skip.
# - else -> ABORT (file not in a state this script understands).
TRANSFORMS = [
    (
        "T1: remove dead 'import subprocess'",
        "import logging\nimport os\nimport subprocess\nfrom datetime import datetime, timezone\n",
        "import logging\nimport os\nfrom datetime import datetime, timezone\n",
        "import logging\nimport os\nfrom datetime import datetime, timezone\n",
    ),
    (
        "T2: correct docstring step 6 (drift logic never existed)",
        "  6. Run existing EWC / drift-detection logic.\n",
        "  6. Retrain triggering is inbox-driven; statistical drift detection is owned\n"
        "     by the dedicated DriftMonitorBase agents (no drift logic runs here).\n",
        "  6. Retrain triggering is inbox-driven; statistical drift detection is owned\n",
    ),
    (
        "T3: remove Step-2 drift block + renumber Step 3 -> Step 2",
        "        # ----------------------------------------------------------\n"
        "        # Step 2: Drift detection (existing EWC logic)\n"
        "        # ----------------------------------------------------------\n"
        "        self._log_section(\"Drift Detection\")\n"
        "        drift_detected = self._check_drift(dry_run)\n"
        "        if drift_detected and not self._retrain_flag:\n"
        "            self._retrain_flag = True\n"
        "            self._trigger_reason = \"drift_detected\"\n"
        "\n"
        "        # ----------------------------------------------------------\n"
        "        # Step 3: Decide whether to retrain\n"
        "        # ----------------------------------------------------------\n",
        "        # ----------------------------------------------------------\n"
        "        # Step 2: Decide whether to retrain\n"
        "        # ----------------------------------------------------------\n",
        "        # Step 2: Decide whether to retrain\n",
    ),
    (
        "T4: renumber 'Step 4 [NEW]' comment -> 'Step 3 [NEW]'",
        "# Step 4 [NEW]: Emit CHECKPOINT_READY to InterpretabilityAgent",
        "# Step 3 [NEW]: Emit CHECKPOINT_READY to InterpretabilityAgent",
        "# Step 3 [NEW]: Emit CHECKPOINT_READY to InterpretabilityAgent",
    ),
    (
        "T5: drop dead 'drift_detected' key from result dict",
        "            \"action\": \"ewc_lifecycle\",\n"
        "            \"drift_detected\": drift_detected,\n"
        "            \"retrain_triggered\": self._retrain_flag,\n",
        "            \"action\": \"ewc_lifecycle\",\n"
        "            \"retrain_triggered\": self._retrain_flag,\n",
        "            \"action\": \"ewc_lifecycle\",\n"
        "            \"retrain_triggered\": self._retrain_flag,\n",
    ),
    (
        "T6: delete vestigial _check_drift method",
        "    def _check_drift(self, dry_run: bool) -> bool:\n"
        "        \"\"\"\n"
        "        Run drift detection against the most recent variant batch.\n"
        "        Returns True if drift is detected above threshold.\n"
        "        \"\"\"\n"
        "        self.logger.info(\"Running drift detection \u2026\")\n"
        "        try:\n"
        "            from ewc_utils import detect_drift\n"
        "\n"
        "            drift = detect_drift(self._get_section(\"training\"))\n"
        "            if drift:\n"
        "                self.logger.info(\"Drift detected above threshold \u2014 retrain warranted.\")\n"
        "            else:\n"
        "                self.logger.info(\"Drift within acceptable bounds.\")\n"
        "            return drift\n"
        "        except Exception as exc:\n"
        "            self.logger.warning(\n"
        "                \"Drift detection failed: %s \u2014 treating as no drift.\", exc\n"
        "            )\n"
        "            return False\n"
        "\n",
        "",
        None,  # special: deletion. "applied" == anchor absent AND '_check_drift' absent
    ),
]


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else TARGET
    if not path.is_file():
        print(f"ABORT: target not found: {path}", file=sys.stderr)
        return 2
    with io.open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    original = text

    for name, anchor, repl, applied in TRANSFORMS:
        if anchor in text:
            count = text.count(anchor)
            if count != 1:
                print(f"ABORT [{name}]: anchor found {count} times (expected exactly 1).",
                      file=sys.stderr)
                return 3
            text = text.replace(anchor, repl)
            print(f"APPLIED  {name}")
        else:
            # anchor absent -> must already be applied, else abort
            if applied is None:
                # deletion transform: 'applied' means _check_drift fully gone
                if "_check_drift" not in text:
                    print(f"SKIP     {name} (already applied: method absent)")
                    continue
                print(f"ABORT [{name}]: anchor absent but '_check_drift' still present "
                      f"-> unexpected file state.", file=sys.stderr)
                return 4
            else:
                if applied in text:
                    print(f"SKIP     {name} (already applied)")
                    continue
                print(f"ABORT [{name}]: anchor absent and applied-signature absent "
                      f"-> unexpected file state.", file=sys.stderr)
                return 5

    # Hard post-conditions: nothing drift-vestigial may remain.
    for forbidden in ("_check_drift", "drift_detected", "detect_drift",
                      "import subprocess"):
        if forbidden in text:
            print(f"ABORT post-check: '{forbidden}' still present after transforms.",
                  file=sys.stderr)
            return 6

    if text == original:
        print("NO-OP: file already fully in target state.")
        return 0

    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"WROTE {path} ({len(text.encode('utf-8'))} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
