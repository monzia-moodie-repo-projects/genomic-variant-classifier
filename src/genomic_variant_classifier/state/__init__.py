"""Mutable operational state: atomic, identified, fail-closed.

Two JSON stores existed with different behaviour -- SharedState writes
atomically and logs corruption; version_monitor_agent wrote directly and
swallowed corruption into an empty mapping that the next save then persisted.

This package holds the mechanism both should use. SharedState's atomic write
was read before this was designed and is reproduced deliberately and
identically, with fsync added, so that a second subtly-different implementation
does not appear beside a correct one.

Author: Monzia Moodie
"""

from __future__ import annotations