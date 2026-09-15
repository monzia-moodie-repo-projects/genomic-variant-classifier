"""Report that the run HAPPENED. Findings are a different question.

Author: Monzia Moodie

WHY A HEARTBEAT IS NOT OPTIONAL
==============================
MEASURED 2026-09-14: every module built today reports findings and none can
report its own silence. gnomAD (Genome Aggregation Database) version 4.1.1 was
released 2026-03-30 and found by hand on 2026-09-12 -- five and a half months
-- because the agent that should have reported it never ran, and nothing
noticed that it never ran.

A scheduled workflow cannot supervise its own absence. GitHub documents that
scheduled runs may be delayed, dropped under load, run only on the default
branch, and be disabled after repository inactivity. A workflow that never
starts produces no exit code, no red run, and no artifact. Its silence is
indistinguishable from success.

An external service that expects a signal by a deadline, and alerts when none
arrives, is the only thing that closes that gap. It must be EXTERNAL: a
watchdog inside the process it watches dies with it.

THE DISTINCTION THIS MODULE EXISTS TO KEEP
==========================================
    the heartbeat answers   "did the run happen and complete?"
    the exit code answers   "what did it find?"

These are different questions and conflating them breaks both.

    exit 0  qualified, nothing to review   -> the run HAPPENED    -> success
    exit 1  qualified, review required     -> the run HAPPENED    -> success
    exit 2  NOT qualified                  -> the run FAILED      -> failure

Exit 1 signals SUCCESS to the heartbeat. A monitor that finds something has
worked, not broken -- signalling failure there would train the reader to
ignore heartbeat alarms on exactly the runs that matter most.

Exit 2 signals failure, because a run that could not qualify its evidence did
not do its job even though the process ran to completion.

SIGNALLING MUST NEVER CHANGE THE RUN'S OUTCOME
==============================================
If the heartbeat endpoint is unreachable, the monitoring result stands. The
failure to signal is recorded in the report and the process still exits on its
own findings. A watchdog that can veto its subject's verdict is a second point
of failure, not a safeguard.

The converse also holds: a signalling failure must be VISIBLE, because an
unreachable heartbeat means the external service will alarm on a run that
actually happened -- a false alarm whose cause should already be in the report.

WHAT THIS DOES NOT ESTABLISH
============================
    * that anyone reads the alarm. An external service notifies; whether a
      human acts is outside this module.
    * that the signal was authentic. The URL is a bearer capability: anyone
      holding it can signal success. It must be treated as a secret, and a
      leaked URL lets a third party silence the alarm.
    * that the deadline is correct. A deadline longer than the interval
      between runs cannot detect a single missed run.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass


#: How long to wait for the heartbeat service. Deliberately short: this is a
#: notification, and a slow endpoint must not extend the monitoring run.
SIGNAL_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class SignalOutcome:
    """What happened when we tried to signal. Never changes the run's verdict."""

    attempted: bool
    delivered: bool
    endpoint_configured: bool
    detail: str = ""

    def as_document(self) -> dict:
        return {
            "attempted": self.attempted,
            "delivered": self.delivered,
            "endpoint_configured": self.endpoint_configured,
            "detail": self.detail,
            "does_not_establish": [
                "that anyone read the alarm",
                "that the signal was authentic; the URL is a bearer "
                "capability and a holder can silence the alarm",
            ],
        }


def signal_start(base_url) -> SignalOutcome:
    """Tell the service a run began, so it can detect one that never ends."""
    return _send(base_url, "/start")


def signal_outcome(base_url, exit_code: int) -> SignalOutcome:
    """Signal success or failure from the run's EXIT CODE.

    0 and 1 are both successes: the run happened and qualified its evidence.
    2 is a failure: the run could not qualify, whatever else it did.
    """
    if type(exit_code) is not int or type(exit_code) is bool:
        raise TypeError("exit_code must be an int, not {!r}".format(exit_code))
    if exit_code not in (0, 1, 2):
        # An unrecognised code must not become success by default.
        return _send(base_url, "/fail")
    return _send(base_url, "" if exit_code in (0, 1) else "/fail")


def _send(base_url, suffix: str) -> SignalOutcome:
    if not base_url:
        # NOT an error. An unconfigured heartbeat is a deployment state, and
        # the report says so rather than pretending a signal was sent.
        return SignalOutcome(False, False, False,
                             "no heartbeat endpoint configured")
    url = str(base_url).rstrip("/") + suffix
    try:
        req = urllib.request.Request(url, method="POST", data=b"")
        with urllib.request.urlopen(req, timeout=SIGNAL_TIMEOUT_SECONDS) as resp:
            code = resp.getcode()
        if 200 <= code < 300:
            return SignalOutcome(True, True, True, "http {}".format(code))
        return SignalOutcome(True, False, True, "http {}".format(code))
    except Exception as exc:
        # The monitoring verdict STANDS. Only the signal failed, and saying so
        # matters: the external service will alarm on a run that happened.
        return SignalOutcome(True, False, True,
                             "{}: {}".format(type(exc).__name__, exc))
