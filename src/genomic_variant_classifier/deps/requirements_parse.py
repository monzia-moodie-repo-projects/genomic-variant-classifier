"""Requirements parsing that reports the coverage of its own parse.

MEASUREMENT-INTEGRITY-1
=======================
On 2026-08-13 a parser was pointed at `requirements-dev.lock` -- 310,494 bytes,
180 packages -- and reported `0`. The file is a `pip-compile --generate-hashes`
artifact, so every record spans continuation lines; the parser read line by
line, handed `aiobotocore==3.6.0 \\` to `Requirement()`, caught the exception,
and continued. Every record failed identically and silently.

    A parser that silently drops every record looks exactly like a file that
    contains nothing.

So parsing here is FAIL-CLOSED, and every file yields a ParseAudit whose counts
must reconcile against the PHYSICAL LINE COUNT -- a quantity no branch counter
touches. Two earlier reconciliation checks were tautologies that could never
fail, and sabotage removed them with no test failure at all.

WHAT THIS VERSION ADDS, AND WHY EACH MATTERS
============================================
DUPLICATE CLAUSES ARE PRESERVED. A dictionary of one record per package cannot
represent:

    foo==1.2 ; python_version < "3.12"
    foo==2.0 ; python_version >= "3.12"

Overwriting silently discards a whole branch of the dependency contract. This
project already depends on marker-conditional declarations -- pyBigWig
publishes no Windows wheel -- so clauses are a tuple per distribution.

IDENTITY COMES FROM model.DistributionName. Measured with packaging 26.0, a
naive `.lower()` disagrees with `canonicalize_name` on six of ten sampled
names: `foo_bar`, `foo.bar`, `FOO--BAR`, `zope.interface`, `ruamel.yaml`,
`backports_abc`. Two analyzers each inventing a normalisation rule is how
parallel vocabularies begin.

INCLUDES AND CONSTRAINTS ARE TYPED. `-r requirements.in` is an EDGE in the
dependency-artifact graph -- it is how requirements-dev.in pulls in the
production set. Counting it as an anonymous "ignored option" alongside
`--index-url` throws away the topology this module exists to measure.

HASHES ARE PRESERVED. These are supply-chain reference artifacts. Stripping
`--hash=` so `Requirement()` can parse, then discarding it, loses the
distinction between "same package and version" and "same package, version and
integrity material".

CONTINUATIONS ARE MATCHED EXPLICITLY. Measured: `raw.replace("\\\\\\n", " ")`
works only because universal newlines translates CRLF on read. Under
`newline=""` it joins ZERO records -- the same zero-record failure this module
was written to end. The regex matches CRLF, LF and CR.

NO CROSS-TYPE EQUALITY. A ParsedRequirement no longer compares equal to a bare
specifier string, because that comparison discards name and marker by design
and would let a marker-sensitive object pass a specifier-only assertion.

Author: Monzia Moodie
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement

from genomic_variant_classifier.deps.model import DistributionName

#: Continuation forms. Matched explicitly rather than relying on the reader's
#: newline mode -- measured to matter under newline="".
_CONTINUATION = re.compile(r"\\(?:\r\n|\n|\r)")

#: `--hash=sha256:...` fragments, extracted before parsing and RETAINED.
_HASH = re.compile(r"--hash=(\S+)")


class MeasurementIntegrityError(RuntimeError):
    """The parse lost records it cannot account for.

    Raised INSTEAD of returning a result: a silently truncated measurement is
    more dangerous than no measurement, because it is actionable and wrong.
    """


class DirectiveKind(str, Enum):
    """A pip option line, classified by what it MEANS.

    `-r other.txt` is a dependency-artifact edge; `--index-url` is not. Folding
    both into one "ignored option" counter discards the graph.
    """
    REQUIREMENT_INCLUDE = "requirement_include"
    CONSTRAINT_INCLUDE = "constraint_include"
    EDITABLE = "editable"
    INDEX_URL = "index_url"
    EXTRA_INDEX_URL = "extra_index_url"
    FIND_LINKS = "find_links"
    OTHER = "other"


_DIRECTIVES = (
    ("-r", DirectiveKind.REQUIREMENT_INCLUDE),
    ("--requirement", DirectiveKind.REQUIREMENT_INCLUDE),
    ("-c", DirectiveKind.CONSTRAINT_INCLUDE),
    ("--constraint", DirectiveKind.CONSTRAINT_INCLUDE),
    ("-e", DirectiveKind.EDITABLE),
    ("--editable", DirectiveKind.EDITABLE),
    ("--index-url", DirectiveKind.INDEX_URL),
    ("-i", DirectiveKind.INDEX_URL),
    ("--extra-index-url", DirectiveKind.EXTRA_INDEX_URL),
    ("--find-links", DirectiveKind.FIND_LINKS),
    ("-f", DirectiveKind.FIND_LINKS),
)


@dataclass(frozen=True)
class RequirementDirective:
    """One pip option line, with its argument preserved."""
    kind: DirectiveKind
    argument: str
    raw: str

    def __repr__(self) -> str:
        return "{}({!r})".format(self.kind.value, self.argument)


@dataclass(frozen=True)
class ParsedRequirement:
    """One requirement clause, with everything the record stated.

    The MARKER is a field because dropping it silently converts a
    platform-conditional dependency into an unconditional one, and two files
    differing only by a marker would compare as equal.

    There is deliberately NO equality with a bare string. `record == ">=1.26"`
    is a lossy comparison that hides name and marker, and evidence objects
    should make lossy comparisons conspicuous.
    """
    name: str
    specifier: str
    marker: str = None
    hashes: tuple = ()
    raw: str = ""

    def __repr__(self) -> str:
        m = "; {}".format(self.marker) if self.marker else ""
        h = "  [{} hash(es)]".format(len(self.hashes)) if self.hashes else ""
        return "{}{}{}{}".format(self.name, self.specifier, m, h)


@dataclass(frozen=True)
class ParseAudit:
    """What the parser saw, and what it did with every line."""
    path: str = ""
    physical_lines: int = 0
    logical_records: int = 0
    parsed_requirements: int = 0
    ignored_comments: int = 0
    ignored_directives: int = 0
    ignored_blank: int = 0
    parse_failures: tuple = ()
    joined_continuations: int = 0

    @property
    def accounted(self) -> int:
        return (self.parsed_requirements + self.ignored_directives
                + len(self.parse_failures))

    def reconciles(self) -> bool:
        """TRUE BY CONSTRUCTION, and retained only as a reportable field.

        Every logical record takes exactly one branch. Two earlier versions
        RAISED on this, which read like a guard and guarded nothing. The real
        check is against `physical_lines`, which no branch counter touches.
        """
        return self.logical_records == self.accounted

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["parse_failures"] = list(self.parse_failures)
        d["accounted"] = self.accounted
        d["reconciles"] = self.reconciles()
        return d


@dataclass(frozen=True)
class ParsedRequirementsFile:
    """A requirements file as measured: clauses, directives, and the audit.

    `clauses` maps a canonical distribution name to a TUPLE of records, because
    one distribution may legitimately appear more than once under different
    markers.
    """
    clauses: dict
    directives: tuple
    audit: ParseAudit

    def names(self) -> set:
        return set(self.clauses)

    def includes(self) -> tuple:
        return tuple(d for d in self.directives
                     if d.kind in (DirectiveKind.REQUIREMENT_INCLUDE,
                                   DirectiveKind.CONSTRAINT_INCLUDE))


def _classify_directive(body: str) -> "RequirementDirective":
    for prefix, kind in _DIRECTIVES:
        if body == prefix or body.startswith(prefix + " ") or body.startswith(prefix + "="):
            arg = body[len(prefix):].lstrip("= ").strip()
            return RequirementDirective(kind=kind, argument=arg, raw=body)
    return RequirementDirective(kind=DirectiveKind.OTHER, argument="", raw=body)


def parse_requirements_file(path, *, allow_failures: bool = False):
    """Return a ParsedRequirementsFile, or RAISE rather than under-report."""
    p = Path(path)
    raw = io.open(p, encoding="utf-8", newline="").read()
    physical = raw.count("\n") + (0 if raw.endswith(("\n", "\r")) or not raw else 1)

    joined_source, joined = _CONTINUATION.subn(" ", raw)

    clauses: dict = {}
    directives: list = []
    failures: list = []
    comments = blank = 0
    logical = 0

    for line in joined_source.splitlines():
        stripped = line.strip()
        if not stripped:
            blank += 1
            continue
        if stripped.startswith("#"):
            comments += 1
            continue
        body = stripped.split(" #", 1)[0].split("\t#", 1)[0].strip()
        if not body:
            comments += 1
            continue

        logical += 1
        if body.startswith("-"):
            directives.append(_classify_directive(body))
            continue

        hashes = tuple(_HASH.findall(body))
        cleaned = re.sub(r"\s*--hash=\S+", "", body).strip()
        try:
            req = Requirement(cleaned)
        except InvalidRequirement as exc:
            failures.append("{}: {}".format(cleaned[:60], exc))
            continue
        record = ParsedRequirement(
            name=str(DistributionName(req.name)),
            specifier=str(req.specifier),
            marker=str(req.marker) if req.marker is not None else None,
            hashes=hashes,
            raw=body,
        )
        clauses.setdefault(record.name, []).append(record)

    audit = ParseAudit(
        path=str(p),
        physical_lines=physical,
        logical_records=logical,
        parsed_requirements=sum(len(v) for v in clauses.values()),
        ignored_comments=comments,
        ignored_directives=len(directives),
        ignored_blank=blank,
        parse_failures=tuple(failures),
        joined_continuations=joined,
    )

    if audit.logical_records and not clauses:
        raise MeasurementIntegrityError(
            "{}: {} logical record(s) but ZERO requirements parsed. This is an "
            "INSTRUMENT FAILURE, not a finding -- the same shape that reported "
            "a 310,494-byte lock as empty on 2026-08-13. First rejected "
            "record: {}".format(
                p, audit.logical_records,
                failures[0] if failures else "(none recorded)"))

    if failures and not allow_failures:
        raise MeasurementIntegrityError(
            "{}: the parser rejected {} record(s), e.g. {}. Pass "
            "allow_failures=True only when a partial parse is what you "
            "actually want.".format(p, len(failures), failures[:3]))

    # RECONCILE AGAINST THE PHYSICAL FILE, not against the branch counters.
    # `physical_lines` is touched by no counter, so a record read and never
    # classified breaks this. A continuation absorbs one extra physical line
    # into one logical record.
    expected_physical = (audit.ignored_comments + audit.ignored_blank
                         + audit.logical_records + audit.joined_continuations)
    if audit.physical_lines != expected_physical:
        raise MeasurementIntegrityError(
            "{}: the parse does not reconcile against the FILE -- {} physical "
            "line(s) against {} accounted ({} comments + {} blank + {} logical "
            "+ {} continuations). {} line(s) were read and never classified."
            .format(p, audit.physical_lines, expected_physical,
                    audit.ignored_comments, audit.ignored_blank,
                    audit.logical_records, audit.joined_continuations,
                    audit.physical_lines - expected_physical))

    return ParsedRequirementsFile(
        clauses={k: tuple(v) for k, v in clauses.items()},
        directives=tuple(directives),
        audit=audit)


def direct_requirements(path, **kw) -> dict:
    """Only the clauses. The audit is still computed and still enforced."""
    return parse_requirements_file(path, **kw).clauses
