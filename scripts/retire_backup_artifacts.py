#!/usr/bin/env python3
"""retire_backup_artifacts.py -- Author: Monzia Moodie

INSTALLER-TRANSACTION-1, migration step 1: classify every `.bak_*` artefact in
the repository, record what it was, and retire it.

    A transaction may temporarily own a rollback state. The repository never
    does. Git owns history; the incident system owns evidence; secret material
    expires.

WHAT THESE ARTEFACTS ARE
Every installer in this project writes `<target>.bak_<timestamp>` before
editing, as its rollback path, and none removes it on success. What was
designed as a rollback IMPLEMENTATION DETAIL became a permanent archival
system by omission.

MEASURED 2026-08-19: 148 such files, 17,640,928 bytes, spanning 2026-08-10 to
2026-08-19. `README.md` alone has 31 and `tests/EXPECTED_SUITE_SIZE` has 29 --
one per installer run, since nearly every unit touches both.

They were invisible to `git status` because `.gitignore` carries `*.bak_*`.
That rule was sensible when backups were intentional siblings; under the
transaction architecture it makes installer leakage silent.

THREE CLASSIFICATIONS, AND WHY THE DISTINCTION MATTERS
    git_exact_preimage
        `git hash-object` of the backup matches a blob git already holds for
        that path in some commit. Git has these exact bytes. Redundant.

    superseded_uncommitted_preimage
        The bytes are NOT in git history -- a working-tree state captured
        mid-edit and superseded before any commit. MEASURED: eight such files
        across four tracked originals, each smaller than current, each with a
        committed successor within hours.

        This class is why "the original is tracked" was NOT sufficient grounds
        for deletion. A tracked original says git has SOME version; it does not
        say git has THESE bytes. The stronger check is the one that runs here.

    secret_bearing
        Matched by PATH SHAPE, not by content inspection: .env, *.pem, *.key,
        credentials*, token*, secrets*. These never receive a content excerpt
        in the manifest, only size, digest and structural counts.

WHAT THE MANIFEST RECORDS FOR A SECRET-BEARING ARTEFACT
    size, SHA-256 of the complete file, whether a GITHUB_TOKEN assignment is
    present, and how many bare non-comment lines it holds.

NOT the token, not a prefix long enough to identify it, not any line content.
The digest of a complete high-entropy credential file establishes that a
particular artefact existed without retaining it, and does not realistically
enable recovery.

    incident evidence != secret retention

WHY A MANIFEST AND NOT SIMPLY DELETION
148 opaque files become one explicit statement of what they were and why they
were retired. That is stronger provenance than the files themselves, and it
matches this project's documentation practice: preserve the RECORD of a state,
not indefinitely preserve every byte.

SAFETY
--check is the default posture in every sense that matters: nothing is deleted
unless --apply is passed. The manifest is written and re-read BEFORE any
deletion, and deletion aborts if the manifest does not account for every file
found. A file whose classification cannot be determined is never deleted.

Usage:
    python scripts/retire_backup_artifacts.py --repo-root . --check
    python scripts/retire_backup_artifacts.py --repo-root . --apply
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

#: Directories whose contents are not this repository's artefacts.
EXCLUDED_ROOTS = (".venv312", "renv", ".git", "node_modules")

#: Path shapes that may carry credential material. Matched on the ORIGINAL
#: path, so `.env.bak_2026-08-15_205854` is caught via `.env`.
SECRET_PATTERNS = (
    ".env", "*.env", "*.pem", "*.key", "*.p12", "*.pfx",
    "credentials*", "token*", "secrets*", "*_rsa", "*_ed25519",
)

CLASS_GIT = "git_exact_preimage"
CLASS_SUPERSEDED = "superseded_uncommitted_preimage"
CLASS_SECRET = "secret_bearing"
CLASS_UNKNOWN = "unclassified"


def _git(repo: Path, *args) -> str:
    return subprocess.run(("git", "-C", str(repo)) + args,
                          capture_output=True, text=True, timeout=300).stdout


#: Shapes that MUST be recognised as secret-bearing. Emptying or narrowing
#: SECRET_PATTERNS would silently reclassify a credential file as ordinary --
#: no leak into the manifest, because the shape reader would never run, but the
#: secret-handling decision would never be surfaced either, and the artefact
#: would become deletable as routine detritus.
#:
#: Sabotage E1 (2026-08-19) emptied SECRET_PATTERNS and produced NO leak,
#: which is exactly why an absence-of-leak check is not sufficient here.
SECRET_CANARIES = (".env", "id_rsa", "server.pem", "api.key",
                   "credentials.json", "token.txt", "secrets.yaml")


def _is_secret_path(original: str) -> bool:
    name = Path(original).name
    return any(fnmatch.fnmatch(name, pat) for pat in SECRET_PATTERNS)


def _assert_secret_detection_intact() -> None:
    """Refuse to run at all if the secret classifier has been weakened.

    A classifier that recognises nothing produces a clean manifest and a
    confident deletion. This is checked BEFORE any scanning, so the failure is
    a refusal rather than a silently permissive pass.
    """
    missed = [c for c in SECRET_CANARIES if not _is_secret_path(c)]
    if missed:
        raise SystemExit(
            "  REFUSING TO RUN: the secret-path classifier does not recognise "
            "{}. Emptying or narrowing SECRET_PATTERNS would let a "
            "credential-bearing artefact be classified as ordinary and "
            "deleted as routine detritus.".format(missed))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _historical_blobs(repo: Path, original: str) -> set:
    """Every blob git has ever held for this path, across all refs."""
    commits = _git(repo, "log", "--all", "--format=%H", "--", original).split()
    blobs = set()
    for c in commits:
        h = _git(repo, "rev-parse", "{}:{}".format(c, original)).strip()
        if h:
            blobs.add(h)
    return blobs


def _successor_commit(repo: Path, original: str) -> dict | None:
    """The most recent commit touching this path, for a superseded preimage."""
    out = _git(repo, "log", "-1", "--format=%H%x00%aI%x00%s", "--", original)
    if not out.strip():
        return None
    parts = out.strip().split("\x00")
    if len(parts) != 3:
        return None
    return {"sha": parts[0], "authored_at": parts[1], "subject": parts[2]}


def _secret_shape(path: Path) -> dict:
    """Structural facts only. NEVER content, never a value, never a prefix."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"readable": False, "error": str(exc)}
    lines = text.splitlines()
    live = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    assignments = [l for l in live if "=" in l]
    bare = [l for l in live if "=" not in l]
    names = sorted({l.split("=", 1)[0].strip() for l in assignments})
    return {
        "readable": True,
        "total_lines": len(lines),
        "assignment_lines": len(assignments),
        "bare_noncomment_lines": len(bare),
        "contains_GITHUB_TOKEN_assignment": any(
            n == "GITHUB_TOKEN" for n in names),
        # Variable NAMES are configuration structure, not secret values.
        "assignment_names": names,
    }


def collect(repo: Path) -> list:
    """Classify every backup artefact. Reads only; deletes nothing."""
    records = []
    for p in sorted(repo.rglob("*.bak_*")):
        rel = p.relative_to(repo).as_posix()
        if any(part in EXCLUDED_ROOTS for part in Path(rel).parts):
            continue
        if not p.is_file():
            continue
        original = rel.rsplit(".bak_", 1)[0]
        stamp = rel.rsplit(".bak_", 1)[1]
        orig_path = repo / original
        rec = {
            "backup": rel,
            "original": original,
            "backup_stamp": stamp,
            "size": p.stat().st_size,
            "sha256": _sha256(p),
            "original_exists": orig_path.exists(),
            "original_tracked": _git(
                repo, "ls-files", "--", original).strip() != "",
        }
        if _is_secret_path(original):
            rec["classification"] = CLASS_SECRET
            rec["secret_shape"] = _secret_shape(p)
            rec["git_blob"] = None
            rec["successor_commit"] = None
            rec["rationale"] = (
                "path shape matches a credential-bearing pattern; recorded by "
                "digest and structure only, never by content")
        else:
            blob = _git(repo, "hash-object", "--", str(p)).strip()
            rec["git_blob"] = blob or None
            known = _historical_blobs(repo, original) if rec["original_tracked"] else set()
            if blob and blob in known:
                rec["classification"] = CLASS_GIT
                rec["successor_commit"] = None
                rec["rationale"] = (
                    "git holds these exact bytes for this path in history")
            elif rec["original_tracked"]:
                rec["classification"] = CLASS_SUPERSEDED
                rec["successor_commit"] = _successor_commit(repo, original)
                rec["rationale"] = (
                    "a working-tree state captured mid-edit and superseded "
                    "before any commit; its successor is committed")
            else:
                rec["classification"] = CLASS_UNKNOWN
                rec["successor_commit"] = None
                rec["rationale"] = (
                    "the original is neither tracked nor secret-shaped; "
                    "REFUSING to classify, and therefore refusing to delete")
        records.append(rec)
    return records


def build_manifest(repo: Path, records: list) -> dict:
    counts = {}
    for r in records:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1
    return {
        "schema": "gvc.backup-retirement-manifest",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "unit": "INSTALLER-TRANSACTION-1",
        "repo_head": _git(repo, "rev-parse", "HEAD").strip(),
        "total_artifacts": len(records),
        "total_bytes": sum(r["size"] for r in records),
        "classification": counts,
        "policy": {
            "principle": (
                "A transaction may temporarily own a rollback state. The "
                "repository never does. Git owns history; the incident system "
                "owns evidence; secret material expires."),
            "secret_handling": (
                "size, digest and structural counts only; never content, "
                "never a value, never an identifying prefix"),
            "unclassified_handling": (
                "never deleted; requires a deliberate decision"),
        },
        "files": records,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--manifest",
                    default="docs/incidents/BACKUP_RETIREMENT_2026-08-19.json")
    ap.add_argument("--apply", action="store_true",
                    help="delete the classified artefacts; default is to "
                         "write the manifest and delete NOTHING")
    args = ap.parse_args(argv)
    repo = Path(args.repo_root).resolve()

    if not (repo / ".git").exists():
        print("  ERROR: {} is not a git working tree".format(repo))
        return 2

    _assert_secret_detection_intact()
    print("  secret classifier: {} canary shape(s) recognised".format(
        len(SECRET_CANARIES)))

    print("  scanning {} ...".format(repo))
    records = collect(repo)
    manifest = build_manifest(repo, records)

    print("  {} artefact(s), {:,} bytes".format(
        manifest["total_artifacts"], manifest["total_bytes"]))
    for k in sorted(manifest["classification"]):
        print("    {:<38} {}".format(k, manifest["classification"][k]))

    unknown = [r for r in records if r["classification"] == CLASS_UNKNOWN]
    if unknown:
        print("\n  {} UNCLASSIFIED artefact(s) -- these are NOT deleted:".format(
            len(unknown)))
        for r in unknown:
            print("    {}".format(r["backup"]))

    secrets = [r for r in records if r["classification"] == CLASS_SECRET]
    if secrets:
        print("\n  {} secret-bearing artefact(s), recorded by shape only:".format(
            len(secrets)))
        for r in secrets:
            sh = r.get("secret_shape", {})
            print("    {}".format(r["backup"]))
            print("        {} bytes, sha256 {}".format(r["size"], r["sha256"][:16]))
            print("        {} assignment line(s), {} bare non-comment line(s), "
                  "GITHUB_TOKEN present: {}".format(
                      sh.get("assignment_lines"), sh.get("bare_noncomment_lines"),
                      sh.get("contains_GITHUB_TOKEN_assignment")))

    mpath = repo / args.manifest
    mpath.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with open(mpath, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(payload)
    print("\n  manifest written: {} ({:,} bytes)".format(
        args.manifest, len(payload.encode("utf-8"))))

    # Re-read it. A manifest that cannot be read back is not a record.
    reread = json.loads(mpath.read_text(encoding="utf-8"))
    if reread["total_artifacts"] != len(records):
        print("  ERROR: the manifest does not account for every artefact found; "
              "NOTHING deleted.")
        return 1
    print("  manifest verified: {} artefact(s) accounted for".format(
        reread["total_artifacts"]))

    if not args.apply:
        print("\n  --check: nothing deleted. Re-run with --apply to retire "
              "{} artefact(s).".format(len(records) - len(unknown)))
        return 0

    deleted, failed = 0, []
    for r in records:
        if r["classification"] == CLASS_UNKNOWN:
            continue
        p = repo / r["backup"]
        try:
            p.unlink()
            deleted += 1
        except OSError as exc:
            failed.append((r["backup"], str(exc)))
    print("\n  deleted {} artefact(s); {} retained as unclassified".format(
        deleted, len(unknown)))
    if failed:
        print("  {} deletion(s) FAILED:".format(len(failed)))
        for path, err in failed:
            print("    {}: {}".format(path, err))
        return 1

    remaining = [p for p in repo.rglob("*.bak_*")
                 if p.is_file()
                 and not any(part in EXCLUDED_ROOTS for part in p.relative_to(repo).parts)]
    print("  remaining .bak_* artefact(s): {}".format(len(remaining)))
    for p in remaining:
        print("    {}".format(p.relative_to(repo).as_posix()))
    if len(remaining) != len(unknown):
        print("  ERROR: expected {} to remain, found {}".format(
            len(unknown), len(remaining)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
