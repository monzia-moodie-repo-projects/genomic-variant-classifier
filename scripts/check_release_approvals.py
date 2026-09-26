"""Admission check for release approvals, read ONLY from Git objects at a commit.

Created 2026-09-26 (change A). Run by CI before the test suite, after fetching the
evidence commits (`--list-evidence-commits`) and the base commit. Read-only: it
never fetches, checks out, writes or pushes; Git 2.45+ is required.

For the commit checked (default HEAD):
  1. the manifest at that commit selects each active approval ({record, sha256});
  2. each selected record is read as a Git blob, hashed and parsed from the SAME
     bytes; target, scope and dates are validated (release_approval.parse_approval);
  3. every evidence item is verified from Git objects: commit/path membership,
     then size, SHA-256 and blob identifier -- never a working-tree file;
  4. the verifier's independent pin (request_verifier) must agree at runtime;
  5. with --base: the COMPLETE docs/approvals directory at the base must survive
     byte-for-byte (append-only), and a changed active approval must supersede
     the base's active approval.
Without --base, it says so: append-only was NOT evaluated.

Exit 0 admitted; 1 refused (the reason is printed).

Author: Monzia Moodie
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from genomic_variant_classifier.data import release_approval as ra  # noqa: E402
from genomic_variant_classifier.data.source_registry import SourceRegistry, SourceRegistryError  # noqa: E402

MANIFEST = "configs/data_manifest.yaml"
APPROVALS_DIR = "docs/approvals"


def _approvals_at(git, commit):
    """Complete docs/approvals snapshot at a commit: {path: bytes}. Every entry must be a
    regular blob -- a symlink or submodule there is refused, not skipped."""
    listing = git("ls-tree", "-r", "-z", commit, "--", APPROVALS_DIR)
    out = {}
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        header, path = entry.split(b"\t", 1)
        mode, kind, oid = header.split(b" ")
        if mode not in {b"100644", b"100755"} or kind != b"blob":
            raise ra.PolicyError(f"{path.decode('utf-8', 'replace')}: not a regular file in {APPROVALS_DIR}")
        out[path.decode("utf-8")] = git("cat-file", "blob", oid.decode("ascii"))
    return out


def _registry_at(git, commit):
    _, raw = ra.read_blob_at(git, commit, MANIFEST, max_size=1024 * 1024)
    return SourceRegistry.from_text(raw.decode("utf-8"), f"{commit}:{MANIFEST}")


def _active(git, commit, registry):
    active = {}
    for ptr in registry.release_approvals:
        approval = ra.load_approval(ptr.target, ptr.record, ptr.sha256,
                                    lambda path: ra.read_blob_at(git, commit, path,
                                                                 max_size=ra.MAX_RECORD_BYTES)[1])
        active[ptr.target] = approval
    return active


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=str(_ROOT))
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--base", default=None, help="trusted base commit for the append-only check")
    parser.add_argument("--list-evidence-commits", action="store_true")
    args = parser.parse_args(argv)
    try:
        git = ra.git_reader(args.repo)
        commit = git("rev-parse", "--verify", args.commit + "^{commit}").strip().decode("ascii")
        registry = _registry_at(git, commit)
        active = _active(git, commit, registry)
        if args.list_evidence_commits:
            for c in sorted({e.commit for a in active.values() for e in a.evidence}):
                print(c)
            return 0
        from genomic_variant_classifier.source_monitor import request_verifier as rv
        for target, approval in active.items():
            for ref in approval.evidence:
                ra.verify_git_evidence(ref, git)
            if target == rv.APPROVAL_TARGET:
                ra.require_verifier_pin(approval, target=rv.APPROVAL_TARGET,
                                        approved_release=rv.APPROVED_BASELINE,
                                        record_sha256=rv.APPROVED_RECORD_SHA256)
            print(f"ADMITTED  {target}: release {approval.approved_release}, scope {approval.scope}, "
                  f"record {approval.record_sha256}, {len(approval.evidence)} evidence item(s) verified")
        if rv.APPROVAL_TARGET not in active:
            raise ra.PolicyError(f"the manifest at {commit} selects no approval for {rv.APPROVAL_TARGET}")
        if args.base is None:
            print("APPEND-ONLY NOT EVALUATED: no --base given")
        else:
            base = git("rev-parse", "--verify", args.base + "^{commit}").strip().decode("ascii")
            before, after = _approvals_at(git, base), _approvals_at(git, commit)
            ra.require_append_only(before, after)
            base_active = _active(git, base, _registry_at(git, base))
            for target, previous in base_active.items():
                if target not in active:
                    raise ra.PolicyError(f"active approval for {target} removed relative to {base}")
                ra.require_successor(previous, active[target])
            print(f"APPEND-ONLY OK against {base}: {len(before)} existing record(s) unchanged, "
                  f"{len(set(after) - set(before))} added")
    except (ra.PolicyError, SourceRegistryError, UnicodeError, ValueError) as exc:
        print(f"REFUSED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
