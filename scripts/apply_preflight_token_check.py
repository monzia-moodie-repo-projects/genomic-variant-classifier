#!/usr/bin/env python3
"""apply_preflight_token_check.py -- Author: Monzia Moodie

PREFLIGHT-TOKEN-SUBSTRING-1: check 9 must confirm a CREDENTIAL, not a name.

THE DEFECT, MEASURED AT scripts/preflight_check.py:259

    if "GITHUB_TOKEN=" in content:
        return True, ".env has GITHUB_TOKEN"

A substring search over the WHOLE FILE. It returns True for a commented-out
line, an empty value, a placeholder, and an unrelated variable whose name
merely ENDS with GITHUB_TOKEN.

IT FAILED TWICE ON 2026-08-15, on the same day, in the same file. The literal
text `GITHUB_TOKEN=<the real token>` was written into .env on two separate
occasions and check 9 reported

    GITHUB_TOKEN available somewhere: True  (.env has GITHUB_TOKEN)

both times. EVERY CLOUD RUN IS GATED ON THIS CHECK, so a run would have
proceeded on a credential that did not exist.

The Windows User-environment branch at line 275 has ALWAYS applied a length
floor. Two of three branches disagreed about what "available" means, and the
weaker one is the branch people actually use.

THE FLOOR IS THIRTY, DERIVED FROM TWO MEASURED LENGTHS
    22  the fragments PowerShell Add-Content produced when it split a pasted
        token across two lines. It did this TWICE, and the second time the
        remainder landed on its own line -- where a display that assumed every
        line contains "=" printed it in full, exposing the credential.
    40  the real token: ghp_ + 36 alphanumeric characters.

Every current GitHub format is at least 40 (ghp_, gho_, ghu_, ghs_ are 40;
ghr_ refresh tokens 76; github_pat_ fine-grained 93; installation tokens moving
to ~520). Any floor in [22, 40) rejects the fragment and admits every real
format. Thirty leaves margin on both sides.

I FIRST ASSERTED A FLOOR OF 20, which does NOT reject a 22-character value. The
threshold was chosen by reasoning and corrected by arithmetic against a number
measured minutes earlier. Both wrong floors are now sabotage cases.

TWO EDITS, EACH ANCHOR BYTE-EXACT AND VERIFIED TO OCCUR ONCE
  1. A helper inserted above github_token_available: _MIN_TOKEN_LENGTH,
     _PLACEHOLDER_PREFIXES, and _env_token_value, which parses LINES, matches
     the name EXACTLY, strips surrounding quotes, and rejects placeholder
     shapes.
  2. The .env branch replaced: it now applies the same floor the Windows branch
     uses, reports a LENGTH rather than the value, and distinguishes "cannot
     read" from "no token" instead of `except Exception: pass`.

VERIFIED
  33 tests: 32 fail before the edits, all pass after. 11 of 11 sabotage
  mutations detected, including the substring search restored, the floor
  dropped, lowered to 10, lowered to 20, raised to 100, placeholder rejection
  dropped, the name matched by suffix, empty values accepted, quotes not
  stripped, read failure swallowed, and the detail string echoing the token.

Idempotent, ast-verifies before AND after writing, backs up to
.pre_pftoken.bak, and rolls back if any post-write check fails.

Usage:  python scripts/apply_preflight_token_check.py --repo-root . --check
        python scripts/apply_preflight_token_check.py --repo-root .
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

HELPER_ANCHOR = 'def github_token_available() -> tuple[bool, str]:\n'

HELPER_NEW = '#: A credential shorter than this is a placeholder or a FRAGMENT, not a\n#: token. The Windows User-environment branch has always applied a floor; the\n#: .env branch applied NOTHING, so the two disagreed about what "available"\n#: means -- and the weaker one is the branch people actually use.\n#:\n#: THIRTY, derived from two measured lengths on 2026-08-15:\n#:     22  the fragments PowerShell Add-Content produced when it split a\n#:         pasted token across two lines. It did this TWICE, and the second\n#:         time the remainder landed on its own line and was printed in full\n#:         by a display that assumed every line contains "=".\n#:     40  the real token, ghp_ + 36 alphanumeric characters.\n#:\n#: Every current GitHub credential format is at least 40: ghp_, gho_, ghu_ and\n#: ghs_ are 40; ghr_ refresh tokens are 76; github_pat_ fine-grained tokens are\n#: 93; and installation tokens are moving to roughly 520. So any floor between\n#: 22 and 40 rejects the fragment and admits every real format. Thirty leaves\n#: margin on both sides.\n#:\n#: A FLOOR and not an exact length, because the ~520-character format is\n#: already announced: an equality check would break on the next one.\n_MIN_TOKEN_LENGTH = 30\n\n#: Values that LOOK like a token but are not one. Both were written into .env\n#: during the 2026-08-15 session and both satisfied the previous substring\n#: check, which reported a usable credential when none existed.\n_PLACEHOLDER_PREFIXES = ("<", "$", "{", "your", "YOUR", "paste", "PASTE")\n\n\ndef _env_token_value(content: str) -> str | None:\n    """The GITHUB_TOKEN value from a .env body, or None.\n\n    THE DEFECT THIS REPLACES was `if "GITHUB_TOKEN=" in content` -- a substring\n    search over the WHOLE FILE. It returned True for a commented-out line, for\n    an empty value, for a placeholder such as GITHUB_TOKEN=<the real token>,\n    and for an unrelated variable whose name merely ENDS with GITHUB_TOKEN.\n\n    The first two of those happened during the 2026-08-15 session: check 9\n    reported a usable credential twice when the file held only placeholder\n    text, and a cloud run gated on that check would have proceeded.\n\n    This parses LINES, requires the name to match exactly, strips surrounding\n    quotes, and rejects placeholder shapes. It returns the VALUE so the caller\n    can apply a length floor -- the same floor the Windows branch has always\n    used.\n    """\n    for raw_line in content.splitlines():\n        line = raw_line.strip()\n        if not line or line.startswith("#"):\n            continue\n        name, sep, value = line.partition("=")\n        if not sep or name.strip() != "GITHUB_TOKEN":\n            continue\n        value = value.strip()\n        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\\"\'":\n            value = value[1:-1]\n        if not value:\n            continue\n        if value.startswith(_PLACEHOLDER_PREFIXES):\n            continue\n        return value\n    return None\n\n\ndef github_token_available() -> tuple[bool, str]:\n'

ENV_OLD = '    env_path = REPO / ".env"\n    if env_path.exists():\n        try:\n            content = env_path.read_text(encoding="utf-8")\n            if "GITHUB_TOKEN=" in content:\n                return True, ".env has GITHUB_TOKEN"\n        except Exception:\n            pass\n'

ENV_NEW = '    env_path = REPO / ".env"\n    if env_path.exists():\n        try:\n            content = env_path.read_text(encoding="utf-8")\n        except OSError as exc:\n            # A .env that cannot be READ is not a .env without a token. The\n            # previous `except Exception: pass` made those indistinguishable.\n            return False, f".env could not be read: {exc}"\n        value = _env_token_value(content)\n        if value is not None and len(value) > _MIN_TOKEN_LENGTH:\n            return True, f".env (length: {len(value)})"\n'

TARGET = "scripts/preflight_check.py"

EDITS = (
    (HELPER_ANCHOR, HELPER_NEW, "_env_token_value"),
    (ENV_OLD, ENV_NEW, "_MIN_TOKEN_LENGTH:"),
)


def _verify(source: str) -> tuple:
    """Structural checks by AST, per ROOTFIX-VERIFY-TEXTUAL-1.

    `if "GITHUB_TOKEN" in source` would be satisfied by the docstring, by the
    old code, and by any comment. These walk the tree.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, "syntax error after patch: {}".format(exc)

    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "_env_token_value" not in names:
        return False, "_env_token_value is missing"
    if "github_token_available" not in names:
        return False, "github_token_available is missing"

    assigns = {}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant):
                    assigns[t.id] = n.value.value
    if "_MIN_TOKEN_LENGTH" not in assigns:
        return False, "_MIN_TOKEN_LENGTH is missing"
    floor = assigns["_MIN_TOKEN_LENGTH"]
    if not isinstance(floor, int):
        return False, "_MIN_TOKEN_LENGTH is not an integer"
    # The floor must sit strictly between the observed fragment (22) and the
    # shortest current credential format (40). Outside that, it either accepts
    # the fragment or rejects real tokens.
    if not (22 <= floor < 40):
        return False, (
            "_MIN_TOKEN_LENGTH is {}; it must be at least 22 (the measured "
            "fragment length) and below 40 (the shortest current GitHub "
            "credential)".format(floor))

    # The substring search must be GONE from the function body.
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == "github_token_available":
            for sub in ast.walk(n):
                if (isinstance(sub, ast.Compare) and sub.ops
                        and isinstance(sub.ops[0], ast.In)
                        and isinstance(sub.left, ast.Constant)
                        and sub.left.value == "GITHUB_TOKEN="):
                    return False, ("the substring search survives at line "
                                   "{}".format(sub.lineno))
            # And the branch must call the parser.
            calls = [c for c in ast.walk(n) if isinstance(c, ast.Call)
                     and getattr(c.func, "id", None) == "_env_token_value"]
            if not calls:
                return False, "github_token_available does not call the parser"
    return True, "parser present; floor {}; substring search gone".format(floor)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    p = Path(args.repo_root) / TARGET
    if not p.exists():
        print("  ERROR: not found: {}".format(TARGET))
        return 2
    src = p.read_text(encoding="utf-8")

    pending = []
    already = 0
    for old, new, marker in EDITS:
        if marker in src:
            already += 1
            print("  {:<52} already applied".format(marker))
            continue
        n = src.count(old)
        if n != 1:
            print("  {:<52} ERROR: anchor occurs {} time(s), expected 1; "
                  "NOTHING written.".format(marker, n))
            return 1
        print("  {:<52} anchor OK".format(marker))
        pending.append((old, new))

    if args.check:
        print("\n  --check: {} pending, {} already applied. Nothing written."
              .format(len(pending), already))
        return 0
    if not pending:
        print("\n  All {} edit(s) already applied.".format(len(EDITS)))
        return 0

    patched = src
    for old, new in pending:
        out = patched.replace(old, new, 1)
        if out == patched:
            print("  ERROR: an edit changed nothing; NOTHING written.")
            return 1
        patched = out

    ok, msg = _verify(patched)
    if not ok:
        print("  ERROR: verification failed BEFORE writing ({}); "
              "NOTHING written.".format(msg))
        return 1
    print("  pre-write  {}".format(msg))

    backup = p.with_suffix(p.suffix + ".pre_pftoken.bak")
    if not backup.exists():
        backup.write_bytes(p.read_bytes())
    p.write_text(patched, encoding="utf-8", newline="\n")
    print("  wrote {}".format(TARGET))

    ok, msg = _verify(p.read_text(encoding="utf-8"))
    if not ok:
        p.write_bytes(backup.read_bytes())
        print("  ERROR: POST-WRITE verification failed ({}); ROLLED BACK."
              .format(msg))
        return 1
    print("  post-write {}".format(msg))
    print("\n  {} edit(s) applied; {} already were.".format(len(pending), already))
    return 0


if __name__ == "__main__":
    sys.exit(main())
