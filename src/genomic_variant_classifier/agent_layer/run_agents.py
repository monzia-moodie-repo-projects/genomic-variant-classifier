#!/usr/bin/env python3
"""
run_agents.py — Agent Layer CLI Entry Point
============================================
Run agent pipelines and inspect system state from the command line.

Usage
-----
  # Run a pipeline
  python run_agents.py --pipeline full
  python run_agents.py --pipeline data_freshness --dry-run
  python run_agents.py --pipeline training
  python run_agents.py --pipeline interpretability
  python run_agents.py --pipeline literature

  # Inspect system state
  python run_agents.py --status
  python run_agents.py --reviews
  python run_agents.py --candidates

  # Manage review items
  python run_agents.py --resolve 0

  # Manage feature candidates
  python run_agents.py --mark-candidate 0

  # ── NEW: Inter-agent message bus ──────────────────────────────────

  # Show full inbox for a specific agent
  python run_agents.py --inbox DataFreshnessAgent
  python run_agents.py --inbox TrainingLifecycleAgent
  python run_agents.py --inbox InterpretabilityAgent
  python run_agents.py --inbox LiteratureScoutAgent

  # List all messages currently awaiting human approval
  python run_agents.py --pending-msgs

  # Approve or reject a pending message by its 8-char ID prefix or full UUID
  python run_agents.py --approve-msg a1b2c3d4
  python run_agents.py --reject-msg  a1b2c3d4

  # View message history (all agents, or one specific agent)
  python run_agents.py --msg-history
  python run_agents.py --msg-history TrainingLifecycleAgent

Pipeline names
--------------
  data_freshness    DataFreshnessAgent only
  training          TrainingLifecycleAgent only
  interpretability  InterpretabilityAgent only
  literature        LiteratureScoutAgent only
  full              All four agents in sequence

Message approval workflow
--------------------------
  When an agent sends a message whose subject requires approval
  (DATA_UPDATED, CHECKPOINT_READY), the receiving agent will NOT act on it
  until you approve it:

    1. python run_agents.py --pending-msgs
       → Lists pending messages with their 8-char IDs

    2. python run_agents.py --approve-msg <id>
       → Approves the message

    3. python run_agents.py --pipeline training
       → TrainingLifecycleAgent now sees the approved DATA_UPDATED and
         factors it into its retrain decision

  To reject a message (prevent the receiving agent from ever acting on it):
    python run_agents.py --reject-msg <id>
"""

from __future__ import annotations

import argparse
import sys   # still required: sys.exit() is called 12 times in main() (lines ~295-351).
            # `from pathlib import Path` was removed with the sys.path hack below -- it had no
            # other use in this file. `sys` DOES have other uses, and deleting it with the hack
            # would have turned every CLI exit into a NameError. Blast radius measured, not
            # assumed (CLAUDE.md 6.2).

# =========================================================================================
# THE BARE IMPORT AND THE sys.path HACK ARE GONE. (2026-07-15, roadmap 6.26)
#
# This file used to read:
#
#     # Ensure agent_layer/ is on the path when run from the project root
#     sys.path.insert(0, str(Path(__file__).parent))
#
#     from orchestrator import Orchestrator, PIPELINE_DEFINITIONS
#     from genomic_variant_classifier.agent_layer.shared_state import SharedState
#
# Two lines apart, the SAME PACKAGE was imported by two different names. That is not a style
# nit -- Python keys sys.modules by the import path, so `orchestrator` and
# `genomic_variant_classifier.agent_layer.orchestrator` become TWO DISTINCT MODULE OBJECTS:
#
#   * two separate `Orchestrator` CLASSES -- `isinstance(x, Orchestrator)` then returns a
#     different answer depending on which path constructed x;
#   * two separate `PIPELINE_DEFINITIONS`, `_TELEMETRY_CAP` and `_DIVIDER`;
#   * two separate copies of every `_Lazy` wrapper and anything it has cached.
#
# Every other consumer in this repository -- 20+ files, including run_adaptation.py,
# run_data_freshness.py, and a dozen tests -- imports the full package path. This file was the
# only one that did not, so it was the only one that could hold a second copy.
#
# And `sys.path.insert(0, agent_layer/)` put this directory FIRST on the import path, ahead of
# the standard library, so any module in it shadows a stdlib module of the same name for the
# whole process. This project already carries tests/unit/test_data_dir_not_shadowed.py because
# shadowing has bitten it before.
#
# A remediation on 2026-07-01 was written to eliminate exactly this
# (apply_unit2_reactivate_message_bus.py; SESSION_2026-07-01 records "Repointed 4 bare `from
# orchestrator import Orchestrator` to the full package path") -- and
# Install_unit2_reactivate_message_bus.ps1:111 still carries the post-condition check
# `if "from orchestrator import Orchestrator" in s: errs.append(...)`. It reported four fixed
# and missed this one. The check existed, was correct, and was not run against this file.
#
# The package is installed editable (`pip install -e .`), so the full path resolves everywhere:
# from the project root, from any cwd, under `python -m`, in the Docker trainer image, and on
# the rented graphics-processing-unit box. The hack bought nothing and cost module identity.
# =========================================================================================
from genomic_variant_classifier.agent_layer.orchestrator import (
    PIPELINE_DEFINITIONS,
    Orchestrator,
)
from genomic_variant_classifier.agent_layer.shared_state import SharedState


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_agents",
        description="Genomic Variant Classifier — Agent Layer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Pipeline ──────────────────────────────────────────────────────
    parser.add_argument(
        "--pipeline",
        choices=list(PIPELINE_DEFINITIONS.keys()),
        metavar="PIPELINE",
        help=("Pipeline to run. Choices: " + ", ".join(PIPELINE_DEFINITIONS.keys())),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Execute in dry-run mode: poll sources and log decisions "
            "but do not trigger any external actions or send messages."
        ),
    )

    # ── System state inspection ───────────────────────────────────────
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print a one-line system status summary and exit.",
    )
    parser.add_argument(
        "--reviews",
        action="store_true",
        help="List all unresolved human-review items and exit.",
    )
    parser.add_argument(
        "--resolve",
        type=int,
        metavar="INDEX",
        help="Mark review item INDEX as resolved and exit.",
    )
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="List pending feature candidates and exit.",
    )
    parser.add_argument(
        "--mark-candidate",
        type=int,
        metavar="INDEX",
        help="Mark feature candidate INDEX as reviewed and exit.",
    )

    # ── NEW: Message bus inspection ───────────────────────────────────
    parser.add_argument(
        "--inbox",
        metavar="AGENT_NAME",
        help=(
            "Show the full inbox (all messages, read and unread) for the "
            "named agent and exit. "
            "Example: --inbox TrainingLifecycleAgent"
        ),
    )
    parser.add_argument(
        "--pending-msgs",
        action="store_true",
        help=(
            "List all inter-agent messages currently awaiting human approval "
            "and exit. Use --approve-msg or --reject-msg to act on them."
        ),
    )
    parser.add_argument(
        "--approve-msg",
        metavar="MSG_ID",
        help=(
            "Approve a pending inter-agent message by its ID (the 8-char "
            "prefix shown in --pending-msgs is sufficient). "
            "Example: --approve-msg a1b2c3d4"
        ),
    )
    parser.add_argument(
        "--reject-msg",
        metavar="MSG_ID",
        help=(
            "Reject a pending inter-agent message by its ID. The receiving "
            "agent will not act on it, but it remains in the history. "
            "Example: --reject-msg a1b2c3d4"
        ),
    )
    parser.add_argument(
        "--msg-history",
        nargs="?",
        const="__all__",
        metavar="AGENT_NAME",
        help=(
            "Show message history. With no argument, shows history for all "
            "agents. With an agent name, shows only that agent's inbox history. "
            "Example: --msg-history TrainingLifecycleAgent"
        ),
    )
    parser.add_argument(
        "--msg-history-limit",
        type=int,
        default=20,
        metavar="N",
        help=("Maximum number of messages to show with --msg-history. " "Default: 20."),
    )

    return parser


# ---------------------------------------------------------------------------
# Helpers for full-UUID message ID matching
# ---------------------------------------------------------------------------


def _resolve_msg_id(shared_state: SharedState, partial_id: str) -> str | None:
    """
    Resolve a partial message ID (e.g. 8-char prefix) to its full UUID.

    Returns the full UUID string if exactly one match is found, or None.
    This lets the user type the short prefix shown in --pending-msgs rather
    than the full 36-char UUID.
    """
    state = shared_state.load()
    matches: list[str] = []
    for inbox in state.get("agent_messages", {}).values():
        for msg in inbox:
            full_id = msg.get("id", "")
            if full_id == partial_id or full_id.startswith(partial_id):
                matches.append(full_id)

    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        print(
            f"No message found with ID starting with '{partial_id}'. "
            f"Run --pending-msgs to see available IDs."
        )
        return None
    # Multiple matches — ask for more characters
    print(
        f"Ambiguous ID prefix '{partial_id}' matches {len(matches)} messages. "
        f"Please provide more characters."
    )
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    shared_state = SharedState()
    orchestrator = Orchestrator(shared_state, dry_run=args.dry_run)

    # ── Existing: state inspection ────────────────────────────────────

    if args.status:
        orchestrator.print_status()
        sys.exit(0)

    if args.reviews:
        orchestrator.print_reviews()
        sys.exit(0)

    if args.resolve is not None:
        orchestrator.resolve_review_item(args.resolve)
        sys.exit(0)

    if args.candidates:
        orchestrator.print_candidates()
        sys.exit(0)

    if args.mark_candidate is not None:
        orchestrator.mark_candidate_reviewed(args.mark_candidate)
        sys.exit(0)

    # ── NEW: message bus commands ─────────────────────────────────────

    if args.inbox:
        orchestrator.print_inbox(args.inbox)
        sys.exit(0)

    if args.pending_msgs:
        orchestrator.print_all_pending_messages()
        sys.exit(0)

    if args.approve_msg:
        full_id = _resolve_msg_id(shared_state, args.approve_msg)
        if full_id:
            orchestrator.approve_message(full_id)
        sys.exit(0)

    if args.reject_msg:
        full_id = _resolve_msg_id(shared_state, args.reject_msg)
        if full_id:
            orchestrator.reject_message(full_id)
        sys.exit(0)

    if args.msg_history is not None:
        agent_name = None if args.msg_history == "__all__" else args.msg_history
        orchestrator.print_message_history(
            agent_name=agent_name,
            limit=args.msg_history_limit,
        )
        sys.exit(0)

    # ── Pipeline execution ────────────────────────────────────────────

    if args.pipeline:
        orchestrator.run_pipeline(args.pipeline)
        sys.exit(0)

    # No arguments given — print help
    parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    main()
