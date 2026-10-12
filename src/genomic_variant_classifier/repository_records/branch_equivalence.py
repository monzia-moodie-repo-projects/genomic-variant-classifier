"""Branch equivalence: the proof, kept as a record, that a branch holds no content the reference branch lacks -- before it is retired.

MEASURED 2026-10-10. The GitHub branch run9a-prep (tip c1c01920, 2026-04-30) showed "180 commits not in main". It is the April history
that pull request #1 merged at 2026-05-01 01:44 UTC (61c1b2b on main, the commit tagged run9a-baseline, whose second parent is the
counterpart of that tip) and kept "briefly for reference". The history rewrite of 2026-07-11 (git filter-repo, which
removed data/alphafold_cif_cache_2026-07-03.tar.gz and notebooks./gitkeep -- scripts/forensics/verify_postrewrite.py) gave every commit on
main a new identifier and renamed the identifiers quoted inside commit messages; the GitHub branch kept the old ones. A count of
"commits not in main" therefore says nothing about CONTENT. This module answers the content question, commit by commit:

    identical            a commit on the reference has the same tree, author, committer, both dates AND the same message
    identifier_rewrite   ... the same, except the message quotes commit identifiers that map EXACTLY to the counterparts' identifiers
    excised_paths_only   no counterpart, because the commit changes ONLY declared excised paths and, with them removed, its tree equals
                         its parent's counterpart's tree (the rewrite dropped it as empty)
    differs              a counterpart by tree and identity, but the message or the parents disagree
    ambiguous            more than one reference commit matches
    no_counterpart       nothing matches and the commit is not excised-paths-only

and accounts for every OBJECT the subject reaches that the reference does not: commits, trees and -- the decisive set -- blobs. The verdict
is "equivalent" only when every commit is identical, identifier_rewrite or excised_paths_only, and not one blob is missing from the
reference. Anything else is "not_equivalent"; nothing is skipped.

The record is AUTHORED and deterministic: `derive` + `render` reproduce it byte for byte from the same history, which is how the installer
binds it to the preserved copy of the branch (a git bundle in the artifact store): it fetches the bundle into its clone and requires
`render(derive(...))` to equal the committed bytes. `load` is strict and re-checks the record's internal consistency.

Author: Monzia Moodie
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from enum import Enum
from pathlib import Path, PurePosixPath

from .identity import RecordId
from .roles import ArtifactRole, RecordsOntologyError, canonical_root

logger = logging.getLogger(__name__)

__all__ = ["SCHEMA", "SCHEMA_VERSION", "ROLE", "FAMILY", "Outcome", "EQUIVALENT_OUTCOMES", "BranchEquivalenceError", "family_root",
           "derive", "render", "load"]

SCHEMA = "gvc.branch-equivalence"
SCHEMA_VERSION = 1
ROLE = ArtifactRole.VERIFICATION_RESULT
FAMILY = PurePosixPath("repository-history") / "branches"
QUESTION = "Does the subject branch hold any content that the reference branch does not?"


class Outcome(str, Enum):
    IDENTICAL = "identical"
    IDENTIFIER_REWRITE = "identifier_rewrite"
    EXCISED_PATHS_ONLY = "excised_paths_only"
    DIFFERS = "differs"
    AMBIGUOUS = "ambiguous"
    NO_COUNTERPART = "no_counterpart"


EQUIVALENT_OUTCOMES = frozenset({Outcome.IDENTICAL, Outcome.IDENTIFIER_REWRITE, Outcome.EXCISED_PATHS_ONLY})
_OID = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"(?<![0-9A-Za-z])([0-9a-f]{7,40})(?![0-9A-Za-z])")


class BranchEquivalenceError(RecordsOntologyError):
    """A branch-equivalence record, or the history it is derived from, does not satisfy this contract."""


def _require(condition, message):
    if not condition:
        raise BranchEquivalenceError(message)


def family_root() -> PurePosixPath:
    return canonical_root(ROLE) / FAMILY


# ------------------------------------------------------------------------------------------------------------------ git, read-only

def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=600)
    if result.returncode != 0:
        raise BranchEquivalenceError("git {} exit {}: {}".format(" ".join(args), result.returncode,
                                                                 result.stderr.decode("utf-8", "replace").strip()))
    return result.stdout.decode("utf-8")


def _commit(repo: Path, ref: str) -> str:
    oid = _git(repo, "rev-parse", "--verify", "--quiet", ref + "^{commit}").strip()
    _require(_OID.fullmatch(oid) is not None, "{!r} does not name a commit".format(ref))
    return oid


_META = "%H%x1f%T%x1f%P%x1f%an%x1f%ae%x1f%aI%x1f%cn%x1f%ce%x1f%cI%x1f%B%x1e"


def _log(repo: Path, *revs: str) -> list:
    rows = []
    for chunk in _git(repo, "log", "--format=" + _META, *revs).split("\x1e"):
        chunk = chunk.lstrip("\n")
        if not chunk:
            continue
        f = chunk.split("\x1f")
        _require(len(f) == 10, "unparseable log record: {!r}".format(chunk[:80]))
        rows.append({"commit": f[0], "tree": f[1], "parents": f[2].split(), "identity": tuple(f[3:9]), "message": f[9]})
    return rows


def _paths_changed(repo: Path, commit: str, parent: str) -> list:
    out = _git(repo, "diff-tree", "--no-commit-id", "--no-renames", "-r", "-z", "--name-only", parent, commit)
    return sorted(p for p in out.split("\0") if p)


def _listing(repo: Path, tree: str) -> dict:
    out = _git(repo, "ls-tree", "-r", "-z", "--full-tree", tree)
    listing = {}
    for entry in out.split("\0"):
        if entry:
            meta, path = entry.split("\t", 1)
            listing[path] = meta
    return listing


def _objects(repo: Path, *revs: str) -> set:
    return {line.split(" ", 1)[0] for line in _git(repo, "rev-list", "--objects", *revs).splitlines() if line}


def _types(repo: Path, oids: list) -> dict:
    if not oids:
        return {}
    result = subprocess.run(["git", "-C", str(repo), "cat-file", "--batch-check=%(objectname) %(objecttype)"],
                            input=("\n".join(oids) + "\n").encode("ascii"), capture_output=True, timeout=600)
    _require(result.returncode == 0, "git cat-file exit {}".format(result.returncode))
    types = {}
    for line in result.stdout.decode("ascii").splitlines():
        oid, kind = line.split(" ")
        types[oid] = kind
    _require(set(types) == set(oids), "git cat-file did not type every object")
    return types


# ------------------------------------------------------------------------------------------------------------------ derivation

def _rewrite(message: str, mapping: dict, branch_only: list):
    """Replace every identifier token in `message` that uniquely prefixes a branch-only commit with a counterpart by the same-length
    prefix of that counterpart. Returns (rewritten message, [[old, new], ...] in order of first appearance)."""
    pairs = []

    def swap(match):
        token = match.group(1)
        hits = [c for c in branch_only if c.startswith(token)]
        if len(hits) == 1 and mapping.get(hits[0]):
            new = mapping[hits[0]][:len(token)]
            if [token, new] not in pairs:
                pairs.append([token, new])
            return new
        return token
    return _TOKEN.sub(swap, message), pairs


def derive(repo, *, record_id: str, subject_name: str, subject_ref: str, reference_name: str, reference_ref: str,
           excised_paths, preserved_copy: dict) -> dict:
    """Measure the subject branch against the reference, read-only, and return the record (deterministic for the same history)."""
    repo = Path(repo)
    RecordId(record_id)
    excised = sorted(set(excised_paths))
    _require(excised == list(excised_paths) and all(p and not p.startswith("/") for p in excised), "excised paths: sorted, unique, relative")
    tip, ref = _commit(repo, subject_ref), _commit(repo, reference_ref)
    branch_only = [r for r in reversed(_log(repo, "--topo-order", tip, "--not", ref))]
    reference = _log(repo, ref)
    by_key = {}
    for r in reference:
        by_key.setdefault((r["tree"],) + r["identity"], []).append(r)
    on_reference = {r["commit"] for r in reference}
    only = [r["commit"] for r in branch_only]

    mapping, rows, claimed = {}, [], {}
    for r in branch_only:                                     # counterparts first: messages may quote LATER commits
        hits = by_key.get((r["tree"],) + r["identity"], [])
        r["hits"] = hits
        if len(hits) == 1:
            claimed.setdefault(hits[0]["commit"], []).append(r["commit"])
    for r in branch_only:                                     # a reference commit claimed by two subject commits matches neither
        r["ambiguous"] = len(r["hits"]) > 1 or (len(r["hits"]) == 1 and len(claimed[r["hits"][0]["commit"]]) > 1)
        if not r["ambiguous"] and len(r["hits"]) == 1:
            mapping[r["commit"]] = r["hits"][0]["commit"]

    def counterpart_of_parent(parent):
        if parent in on_reference:
            return parent
        if parent in mapping:
            return mapping[parent]
        prior = next((x for x in branch_only if x["commit"] == parent), None)
        if prior is not None and prior.get("excised_parent_counterpart"):
            return prior["excised_parent_counterpart"]
        return None

    for r in branch_only:
        row = {"commit": r["commit"], "outcome": None, "counterpart": None, "identifier_rewrites": [], "excised_paths": [],
               "subject_line": r["message"].split("\n", 1)[0]}
        hits = r["hits"]
        if r["ambiguous"]:
            row["outcome"] = Outcome.AMBIGUOUS.value
        elif len(hits) == 1:
            twin = hits[0]
            row["counterpart"] = twin["commit"]
            parents_ok = [counterpart_of_parent(p) for p in r["parents"]] == twin["parents"]
            if not parents_ok:
                row["outcome"] = Outcome.DIFFERS.value
            elif r["message"] == twin["message"]:
                row["outcome"] = Outcome.IDENTICAL.value
            else:
                rewritten, pairs = _rewrite(r["message"], mapping, only)
                if pairs and rewritten == twin["message"]:
                    row["outcome"], row["identifier_rewrites"] = Outcome.IDENTIFIER_REWRITE.value, pairs
                else:
                    row["outcome"] = Outcome.DIFFERS.value
        else:
            row["outcome"] = Outcome.NO_COUNTERPART.value
            if len(r["parents"]) == 1:
                parent_twin = counterpart_of_parent(r["parents"][0])
                changed = _paths_changed(repo, r["commit"], r["parents"][0])
                # The subset test is a short-circuit (no tree listings for commits that touch other paths); the tree comparison below
                # implies it, and is the condition that ALSO refuses a declared excision the reference still holds (its tree keeps it).
                if parent_twin and changed and set(changed) <= set(excised):
                    mine = {p: m for p, m in _listing(repo, r["tree"]).items() if p not in excised}
                    theirs = _listing(repo, _git(repo, "rev-parse", parent_twin + "^{tree}").strip())
                    if mine == theirs:
                        row["outcome"], row["excised_paths"] = Outcome.EXCISED_PATHS_ONLY.value, changed
                        r["excised_parent_counterpart"] = parent_twin
        rows.append(row)

    missing = sorted(_objects(repo, tip) - _objects(repo, ref))
    types = _types(repo, missing)
    kinds = {k: sorted(o for o in missing if types[o] == k) for k in ("commit", "tree", "blob", "tag")}
    _require(not kinds["tag"], "annotated tags reachable from the subject only: {}".format(kinds["tag"]))
    summary = {"branch_only_commits": len(rows)}
    for outcome in Outcome:
        summary[outcome.value] = sum(1 for row in rows if row["outcome"] == outcome.value)
    equivalent = all(Outcome(row["outcome"]) in EQUIVALENT_OUTCOMES for row in rows) and not kinds["blob"]
    tip_row = _log(repo, "-1", tip)[0]
    return {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "record_id": record_id, "question": QUESTION,
        "subject": {"name": subject_name, "tip": tip, "tree": tip_row["tree"],
                    "commits": int(_git(repo, "rev-list", "--count", tip).strip()),
                    "roots": sorted(_git(repo, "rev-list", "--max-parents=0", tip).split())},
        "reference": {"name": reference_name, "commit": ref, "tree": _git(repo, "rev-parse", ref + "^{tree}").strip()},
        "excised_paths": excised,
        "preserved_copy": preserved_copy,
        "branch_only_commits": rows,
        "summary": summary,
        "objects_not_in_reference": {"commits": len(kinds["commit"]), "trees": kinds["tree"], "blobs": kinds["blob"]},
        "verdict": "equivalent" if equivalent else "not_equivalent",
    }


def render(record: dict) -> bytes:
    """The authored form: two-space indentation, ASCII, a final newline. Key order is the order `derive` builds."""
    return (json.dumps(record, indent=2, ensure_ascii=True) + "\n").encode("ascii")


# ------------------------------------------------------------------------------------------------------------------ strict loading

_TOP = ["schema", "schema_version", "record_id", "question", "subject", "reference", "excised_paths", "preserved_copy",
        "branch_only_commits", "summary", "objects_not_in_reference", "verdict"]
_ROW = ["commit", "outcome", "counterpart", "identifier_rewrites", "excised_paths", "subject_line"]
_COPY = ["store", "path", "sha256", "size_bytes", "heads"]


def _keys(doc, required, what):
    _require(type(doc) is dict, "{}: must be an object".format(what))
    _require(list(doc) == required, "{}: keys {} (expected exactly, in order, {})".format(what, list(doc), required))


def _oid(value, what, nullable=False):
    if nullable and value is None:
        return
    _require(type(value) is str and _OID.fullmatch(value) is not None, "{}: not a 40-hex object identifier".format(what))


def load(raw: bytes) -> dict:
    """Parse and check a committed record: strict JSON, exact keys, formats, and every derived figure re-derived from the rows."""
    _require(type(raw) is bytes and raw.endswith(b"\n") and not raw.startswith(b"\xef\xbb\xbf"), "record: bytes with a final newline, no BOM")

    def pairs(items):
        out = {}
        for k, v in items:
            _require(k not in out, "record: duplicate key {!r}".format(k))
            out[k] = v
        return out

    def refuse(token):
        raise BranchEquivalenceError("record: non-integer number {!r}".format(token))
    try:
        doc = json.loads(raw.decode("ascii"), object_pairs_hook=pairs, parse_float=refuse, parse_constant=refuse)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BranchEquivalenceError("record: invalid JSON: {}".format(exc)) from None
    _keys(doc, _TOP, "record")
    _require(doc["schema"] == SCHEMA and doc["schema_version"] == SCHEMA_VERSION and doc["question"] == QUESTION, "record: schema")
    RecordId(doc["record_id"])
    _keys(doc["subject"], ["name", "tip", "tree", "commits", "roots"], "subject")
    _keys(doc["reference"], ["name", "commit", "tree"], "reference")
    for side in ("subject", "reference"):
        _require(type(doc[side]["name"]) is str and doc[side]["name"].strip() == doc[side]["name"] != "", side + ": name")
    _oid(doc["subject"]["tip"], "subject tip"); _oid(doc["subject"]["tree"], "subject tree")
    _oid(doc["reference"]["commit"], "reference commit"); _oid(doc["reference"]["tree"], "reference tree")
    _require(type(doc["subject"]["commits"]) is int and doc["subject"]["commits"] > 0, "subject: commits")
    _require(type(doc["subject"]["roots"]) is list and doc["subject"]["roots"] == sorted(doc["subject"]["roots"]) != [], "subject: roots")
    for r in doc["subject"]["roots"]:
        _oid(r, "subject root")
    ex = doc["excised_paths"]
    _require(type(ex) is list and ex == sorted(set(ex)) and all(type(p) is str and p for p in ex), "excised_paths: sorted unique text")
    _keys(doc["preserved_copy"], _COPY, "preserved_copy")
    pc = doc["preserved_copy"]
    _require(type(pc["store"]) is str and pc["store"] != "" and type(pc["path"]) is str and pc["path"] != "", "preserved_copy: store and path")
    _require("\\" not in pc["path"] and not PurePosixPath(pc["path"]).is_absolute(), "preserved_copy: a store-relative POSIX path")
    _require(type(pc["sha256"]) is str and _SHA256.fullmatch(pc["sha256"]) is not None, "preserved_copy: sha256")
    _require(type(pc["size_bytes"]) is int and pc["size_bytes"] > 0, "preserved_copy: size_bytes")
    _require(type(pc["heads"]) is list and pc["heads"] and all(type(h) is str for h in pc["heads"]), "preserved_copy: heads")
    _require(any(h.split(" ", 1)[0] == doc["subject"]["tip"] for h in pc["heads"]), "preserved_copy: no head is the subject tip")
    rows = doc["branch_only_commits"]
    _require(type(rows) is list and rows, "branch_only_commits: a non-empty list")
    seen, twins = set(), set()
    for i, row in enumerate(rows):
        _keys(row, _ROW, "row {}".format(i))
        _oid(row["commit"], "row {} commit".format(i)); _oid(row["counterpart"], "row {} counterpart".format(i), nullable=True)
        _require(row["commit"] not in seen, "row {}: duplicate commit".format(i)); seen.add(row["commit"])
        try:
            outcome = Outcome(row["outcome"])
        except ValueError:
            raise BranchEquivalenceError("row {}: unknown outcome {!r}".format(i, row["outcome"])) from None
        has_twin = outcome in (Outcome.IDENTICAL, Outcome.IDENTIFIER_REWRITE, Outcome.DIFFERS)
        _require((row["counterpart"] is not None) == has_twin, "row {}: counterpart present iff identical / identifier_rewrite / differs".format(i))
        if row["counterpart"] is not None:
            _require(row["counterpart"] not in twins, "row {}: counterpart used twice".format(i)); twins.add(row["counterpart"])
        rw = row["identifier_rewrites"]
        _require(type(rw) is list and (bool(rw) == (outcome is Outcome.IDENTIFIER_REWRITE)), "row {}: identifier_rewrites iff identifier_rewrite".format(i))
        for pair in rw:
            _require(type(pair) is list and len(pair) == 2 and all(type(t) is str and re.fullmatch(r"[0-9a-f]{7,40}", t) for t in pair)
                     and len(pair[0]) == len(pair[1]) and pair[0] != pair[1], "row {}: an identifier pair".format(i))
        xp = row["excised_paths"]
        _require(type(xp) is list and (bool(xp) == (outcome is Outcome.EXCISED_PATHS_ONLY)) and set(xp) <= set(ex), "row {}: excised_paths".format(i))
        _require(type(row["subject_line"]) is str, "row {}: subject_line".format(i))
    summary = {"branch_only_commits": len(rows)}
    for outcome in Outcome:
        summary[outcome.value] = sum(1 for row in rows if row["outcome"] == outcome.value)
    _require(doc["summary"] == summary and list(doc["summary"]) == list(summary), "summary: {} (re-derived {})".format(doc["summary"], summary))
    _keys(doc["objects_not_in_reference"], ["commits", "trees", "blobs"], "objects_not_in_reference")
    onr = doc["objects_not_in_reference"]
    _require(type(onr["commits"]) is int and onr["commits"] == len(rows), "objects_not_in_reference: commits must equal the branch-only rows")
    for kind in ("trees", "blobs"):
        _require(type(onr[kind]) is list and onr[kind] == sorted(set(onr[kind])), "objects_not_in_reference: {} sorted unique".format(kind))
        for o in onr[kind]:
            _oid(o, kind)
    equivalent = all(Outcome(row["outcome"]) in EQUIVALENT_OUTCOMES for row in rows) and not onr["blobs"]
    _require(doc["verdict"] == ("equivalent" if equivalent else "not_equivalent"), "verdict {!r} contradicts the rows".format(doc["verdict"]))
    _require(render(doc) == raw, "record: not in its authored form (re-rendering changes the bytes)")
    return doc


# ------------------------------------------------------------------------------------------------------------------ command line

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m genomic_variant_classifier.repository_records.branch_equivalence")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("derive", help="measure, and write the record to --out")
    d.add_argument("--repo", required=True); d.add_argument("--record-id", required=True)
    d.add_argument("--subject-name", required=True); d.add_argument("--subject-ref", required=True)
    d.add_argument("--reference-name", required=True); d.add_argument("--reference-ref", required=True)
    d.add_argument("--excise", action="append", default=[]); d.add_argument("--preserved-copy", required=True, help="a JSON object")
    d.add_argument("--out", required=True)
    v = sub.add_parser("verify", help="strictly load a committed record")
    v.add_argument("record")
    args = parser.parse_args(argv)
    if args.command == "derive":
        record = derive(args.repo, record_id=args.record_id, subject_name=args.subject_name, subject_ref=args.subject_ref,
                        reference_name=args.reference_name, reference_ref=args.reference_ref, excised_paths=sorted(args.excise),
                        preserved_copy=json.loads(args.preserved_copy))
        raw = render(record)
        load(raw)
        Path(args.out).write_bytes(raw)
        print("{} {} verdict {}".format(args.out, len(raw), record["verdict"]))
        return 0
    doc = load(Path(args.record).read_bytes())
    print("{} verdict {} ({} branch-only commits)".format(args.record, doc["verdict"], doc["summary"]["branch_only_commits"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
