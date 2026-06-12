#!/usr/bin/env python3
"""patch_ssh_url_support.py -- add --ssh-url to preflight_run16.py and stage_run16.py.

--ssh-url accepts the exact string `vastai ssh-url <id>` emits
(ssh://user@host:port) and overrides --ssh-host/--ssh-port/--remote-user, removing the
manual host/port split that invited placeholder mistakes. For stage_run16.py it also
relaxes --ssh-host from required=True and validates that a host arrived from either source.

Count-guarded, idempotent, backup-first, CRLF/LF-preserving, ASCII-only. Run from repo
root. Author: Monzia Moodie.
"""
from __future__ import annotations

import sys
from pathlib import Path

ARG_LINE = ('    ap.add_argument("--ssh-url", help="ssh://user@host:port from '
            '\'vastai ssh-url\'; overrides --ssh-host/--ssh-port/--remote-user")\n')

PARSE_BLOCK = (
    '    if getattr(args, "ssh_url", None):\n'
    '        _u = args.ssh_url.strip()\n'
    '        if _u.startswith("ssh://"):\n'
    '            _u = _u[6:]\n'
    '        if "@" in _u:\n'
    '            args.remote_user, _u = _u.split("@", 1)\n'
    '        _host, _sep, _port = _u.partition(":")\n'
    '        if _host:\n'
    '            args.ssh_host = _host\n'
    '        if _port:\n'
    '            args.ssh_port = _port\n'
)

STAGE_VALIDATE = (
    '    if not args.ssh_host:\n'
    '        ap.error("provide --ssh-host (with --ssh-port) or --ssh-url from '
    '\'vastai ssh-url\'")\n'
)

ANCHOR_HOST_PLAIN = '    ap.add_argument("--ssh-host")\n'
ANCHOR_HOST_REQ = '    ap.add_argument("--ssh-host", required=True)\n'
ANCHOR_PARSE = '    args = ap.parse_args()\n'

TARGETS = {
    Path("scripts/preflight_run16.py"): {"req_replace": False},
    Path("scripts/stage_run16.py"): {"req_replace": True},
}


def patch_one(path: Path, req_replace: bool) -> str:
    if not path.exists():
        return f"SKIP (not found): {path}"
    raw = path.open("r", encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")

    if "--ssh-url" in text:
        return f"already patched: {path.name}"

    if req_replace:
        if text.count(ANCHOR_HOST_REQ) != 1:
            return f"ABORT {path.name}: required=True host anchor not found exactly once"
        text = text.replace(ANCHOR_HOST_REQ, ANCHOR_HOST_PLAIN)

    if text.count(ANCHOR_HOST_PLAIN) != 1:
        return f"ABORT {path.name}: --ssh-host anchor not found exactly once"
    if text.count(ANCHOR_PARSE) != 1:
        return f"ABORT {path.name}: parse_args anchor not found exactly once"

    text = text.replace(ANCHOR_HOST_PLAIN, ANCHOR_HOST_PLAIN + ARG_LINE)
    block = PARSE_BLOCK + (STAGE_VALIDATE if req_replace else "")
    text = text.replace(ANCHOR_PARSE, ANCHOR_PARSE + block)

    backup = path.with_suffix(path.suffix + ".pre_sshurl.bak")
    backup.write_bytes(path.read_bytes())
    path.open("w", encoding="utf-8", newline="").write(text.replace("\n", nl))
    return f"OK: {path.name} (+--ssh-url{', host no longer required' if req_replace else ''})"


def main() -> int:
    rc = 0
    for path, cfg in TARGETS.items():
        msg = patch_one(path, cfg["req_replace"])
        print(" " + msg)
        if msg.startswith("ABORT"):
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
