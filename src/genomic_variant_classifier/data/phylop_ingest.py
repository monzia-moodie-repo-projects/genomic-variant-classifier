"""Strict ingestion of a flat phyloP source. A2: PHYLOP-INGEST-INTEGRITY-1.

THREE DEFECTS, ALL MEASURED 2026-08-12
======================================

PHYLOPPARSE-1 -- `on_bad_lines="skip"`
    `pd.read_csv(..., on_bad_lines="skip")` discarded malformed rows with no
    count, no warning, and no record. The resulting index was smaller than the
    source and nothing said so. A conservation index silently missing an
    unknown number of positions is indistinguishable from one that is complete.

    THE REPAIR IS REFUSAL, NOT COUNTING. Counting malformed rows and continuing
    changes the scientific source: the index would then describe a subset
    nobody chose, admitted by a threshold nobody set. A malformed row means the
    build stops and no cache is published. Forensic counting belongs to a
    separate diagnostic tool, exactly as `measure_phylop_agreement` is separate
    from `admit_phylop_equivalence`.

PHYLOPHEADER-1 -- a heuristic that cannot fire
    `if str(chunk.iloc[0]["pos"]).lower() in ("pos", "position", "start")`
    tried to detect a header row by inspecting already-parsed data. `pos` is
    read as `Int64`, so a header row makes that cell `<NA>` -- never the string
    `"pos"`. The check could not fire, and with `on_bad_lines="skip"` the
    header was most likely dropped by the later `dropna` instead, silently.

    THE REPAIR IS A DECLARED CONTRACT. The caller states whether the source has
    a header; the reader VERIFIES that claim against the raw first line and
    refuses on mismatch. A property of the file is not inferred from a guess.

ROW ACCOUNTING
    Nothing reconciled rows read against rows indexed. This module makes the
    accounting an invariant: every row read is either ACCEPTED or REJECTED WITH
    A NAMED REASON, and the totals must sum. An audit that cannot reconcile is
    itself a defect, and `assert_ingest_reconciles` refuses.

DUPLICATE LOCI
    `d[(chrom, pos)] = score` means LAST ROW WINS, so a duplicated locus
    resolved by source row order. That is the identical failure as
    `drop_duplicates(keep="first")` in the gnomAD constraint connector, which
    disagreed with MANE Select for 5,468 of 17,473 genes -- 31.3% -- with 132
    crossing the constrained boundary. Here it is refused rather than resolved,
    because two different scores at one position is a source-integrity question
    and not something a loader may decide.

Author: Monzia Moodie
"""

from __future__ import annotations

import io as _io
import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

#: Columns a flat phyloP source must supply, in the order they are read.
REQUIRED_COLUMNS: tuple = ("chrom", "pos", "score")

#: Tokens accepted as a header cell for each required column. Compared against
#: the RAW first line, never against parsed values -- see PHYLOPHEADER-1.
HEADER_TOKENS: dict = {
    "chrom": ("chrom", "chromosome", "chr", "#chrom", "#chr"),
    "pos": ("pos", "position", "start"),
    "score": ("score", "phylop", "value"),
}


class PhyloPIngestError(RuntimeError):
    """The source violated the ingestion contract. The build must stop."""


class PhyloPHeaderContractError(PhyloPIngestError):
    """The declared header contract does not match the file.

    Separate from the general ingest error because the CAUSE is specific: the
    caller stated something about the file that the file contradicts, which is
    a configuration defect rather than a data defect.
    """


class PhyloPDuplicateLocusError(PhyloPIngestError):
    """One genomic position carries more than one score.

    Separate because it is a SCIENTIFIC question, not a parse failure. Two
    scores at one locus may mean overlapping intervals, a merged source, or a
    coordinate-convention error -- and a loader may not choose between them.
    """


@dataclass(frozen=True)
class PhyloPIngestAudit:
    """What the reader saw, and what it did with every row.

    Every field is a count of rows, and they must sum. `assert_reconciles`
    makes that an enforced invariant rather than an aspiration: a row that is
    neither accepted nor rejected for a named reason has vanished, and a loader
    that can lose rows without saying so is the defect this unit exists to end.
    """
    source_path: str = ""
    source_sha256: str = ""
    header_declared: bool = False
    header_observed: bool = False
    rows_read: int = 0
    rows_accepted: int = 0
    rows_rejected_missing_chrom: int = 0
    rows_rejected_missing_pos: int = 0
    rows_rejected_missing_score: int = 0
    n_distinct_loci: int = 0
    notes: tuple = ()

    @property
    def rows_rejected(self) -> int:
        return (self.rows_rejected_missing_chrom
                + self.rows_rejected_missing_pos
                + self.rows_rejected_missing_score)

    def reconciles(self) -> bool:
        return self.rows_read == self.rows_accepted + self.rows_rejected

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["notes"] = list(self.notes)
        d["rows_rejected"] = self.rows_rejected
        d["reconciles"] = self.reconciles()
        return d


def assert_ingest_reconciles(audit: PhyloPIngestAudit) -> None:
    """Refuse an audit whose rows do not sum.

    This is the accounting equivalent of the row-conservation assertion the
    gnomAD canonicaliser carries. It cannot be satisfied by a loader that
    quietly drops rows, which is precisely what `on_bad_lines="skip"` did.
    """
    if not audit.reconciles():
        raise PhyloPIngestError(
            "ingest accounting does not reconcile: {} row(s) read, {} accepted "
            "+ {} rejected = {}. {} row(s) are unaccounted for.".format(
                audit.rows_read, audit.rows_accepted, audit.rows_rejected,
                audit.rows_accepted + audit.rows_rejected,
                audit.rows_read - audit.rows_accepted - audit.rows_rejected))


def observe_header(first_line: str) -> bool:
    """Decide from the RAW first line whether this source carries a header.

    Operates on text before any typing, because PHYLOPHEADER-1 was a check on
    already-parsed data: `pos` is read as Int64, so a header row yields <NA>
    and the comparison against the string "pos" could never be true.
    """
    fields = [f.strip().lower() for f in first_line.rstrip("\n").split("\t")]
    if len(fields) < len(REQUIRED_COLUMNS):
        return False
    return any(fields[i] in HEADER_TOKENS[col]
               for i, col in enumerate(REQUIRED_COLUMNS))


def verify_header_contract(first_line: str, *, declared: bool,
                           source_path: str = "") -> bool:
    """Verify the caller's declared header contract against the file.

    The caller STATES whether a header is present; this confirms it. A
    mismatch raises rather than adapting, because a loader that silently
    accommodates either shape cannot tell a headed file from one whose first
    data row happens to be unparseable.
    """
    observed = observe_header(first_line)
    if observed != declared:
        raise PhyloPHeaderContractError(
            "header contract mismatch for {!r}: caller declared "
            "has_header={}, the first line {!r} indicates has_header={}. "
            "Declare the source's actual shape; do not let the reader "
            "guess.".format(source_path or "<source>", declared,
                            first_line.rstrip("\n")[:80], observed))
    return observed


def parse_phylop_frame(frame: pd.DataFrame, *, source_path: str = "",
                       source_sha256: str = "", header_declared: bool = False,
                       header_observed: bool = False) -> tuple:
    """Validate one already-read frame, returning (clean_frame, audit).

    MISSING is tolerated and COUNTED, by named reason. MALFORMED is not
    reachable here: the reader uses on_bad_lines="error", so a row that cannot
    be split into fields stops the build before this function is called.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise PhyloPIngestError(
            "phyloP source is missing required column(s) {}; it has "
            "{}".format(missing, list(frame.columns)[:12]))

    rows_read = int(len(frame))
    bad_chrom = frame["chrom"].isna() | (
        frame["chrom"].astype(str).str.strip() == "")
    bad_pos = frame["pos"].isna() & ~bad_chrom
    bad_score = frame["score"].isna() & ~bad_chrom & ~bad_pos

    keep = ~(bad_chrom | bad_pos | bad_score)
    clean = frame.loc[keep, list(REQUIRED_COLUMNS)].copy()

    audit = PhyloPIngestAudit(
        source_path=source_path,
        source_sha256=source_sha256,
        header_declared=bool(header_declared),
        header_observed=bool(header_observed),
        rows_read=rows_read,
        rows_accepted=int(len(clean)),
        rows_rejected_missing_chrom=int(bad_chrom.sum()),
        rows_rejected_missing_pos=int(bad_pos.sum()),
        rows_rejected_missing_score=int(bad_score.sum()),
        n_distinct_loci=int(len(clean.drop_duplicates(subset=["chrom", "pos"]))),
    )
    assert_ingest_reconciles(audit)

    if audit.rows_rejected:
        logger.warning(
            "phyloP source %r: %d of %d row(s) rejected -- %d missing chrom, "
            "%d missing pos, %d missing score. These are COUNTED, not "
            "silently dropped.",
            source_path or "<source>", audit.rows_rejected, audit.rows_read,
            audit.rows_rejected_missing_chrom, audit.rows_rejected_missing_pos,
            audit.rows_rejected_missing_score)
    return clean, audit


def assert_no_duplicate_loci(clean: pd.DataFrame, *,
                             source_path: str = "") -> None:
    """Refuse a source carrying two scores for one position.

    The dictionary substitute -- d[(chrom, pos)] = score -- resolved this by
    LAST ROW WINS, making the index depend on source row order. That is the
    same order-dependence as `drop_duplicates(keep="first")` in the gnomAD
    connector, which disagreed with MANE Select for 5,468 of 17,473 genes.

    Two scores at one locus is a source-integrity question -- overlapping
    intervals, a merged source, a coordinate-convention error -- and a loader
    may not choose between them.
    """
    dup = clean.duplicated(subset=["chrom", "pos"], keep=False)
    if not bool(dup.any()):
        return
    offending = clean.loc[dup].drop_duplicates(subset=["chrom", "pos"])
    n_loci = int(len(offending))
    sample = [
        "{}:{}".format(r.chrom, r.pos)
        for r in offending.head(5).itertuples(index=False)
    ]
    raise PhyloPDuplicateLocusError(
        "phyloP source {!r} carries more than one score at {} locus/loci, "
        "e.g. {}. Resolving by row order would make the index depend on file "
        "ordering; the source must be disambiguated before it is "
        "indexed.".format(source_path or "<source>", n_loci, sample))


def read_phylop_source(path, *, has_header: bool, chunk_size: int = 1_000_000,
                       source_sha256: str = "", _opener=None):
    """Read a flat phyloP source strictly, returning (clean_frame, audit).

    `on_bad_lines="error"`: a row that cannot be split into fields STOPS the
    build. It is not skipped, not counted, not tolerated. The previous
    behaviour published a cache describing a subset nobody chose.
    """
    opener = _opener or (lambda p: _io.open(p, "r", encoding="utf-8"))
    with opener(path) as fh:
        first_line = fh.readline()
    if not first_line:
        raise PhyloPIngestError(
            "phyloP source {!r} is empty; no index can be built".format(str(path)))
    observed = verify_header_contract(
        first_line, declared=has_header, source_path=str(path))

    frames, audits = [], []
    reader = pd.read_csv(
        path, sep="\t", header=0 if has_header else None,
        names=None if has_header else list(REQUIRED_COLUMNS),
        dtype={"chrom": "string", "pos": "Int64", "score": "float64"},
        chunksize=chunk_size, on_bad_lines="error",
    )
    for chunk in reader:
        clean, audit = parse_phylop_frame(
            chunk, source_path=str(path), source_sha256=source_sha256,
            header_declared=has_header, header_observed=observed)
        frames.append(clean)
        audits.append(audit)

    if not frames:
        raise PhyloPIngestError(
            "phyloP source {!r} yielded no rows".format(str(path)))

    combined = pd.concat(frames, ignore_index=True)
    total = PhyloPIngestAudit(
        source_path=str(path),
        source_sha256=source_sha256,
        header_declared=has_header,
        header_observed=observed,
        rows_read=sum(a.rows_read for a in audits),
        rows_accepted=sum(a.rows_accepted for a in audits),
        rows_rejected_missing_chrom=sum(a.rows_rejected_missing_chrom for a in audits),
        rows_rejected_missing_pos=sum(a.rows_rejected_missing_pos for a in audits),
        rows_rejected_missing_score=sum(a.rows_rejected_missing_score for a in audits),
        n_distinct_loci=int(len(combined.drop_duplicates(subset=["chrom", "pos"]))),
        notes=("chunks={}".format(len(audits)),),
    )
    assert_ingest_reconciles(total)
    assert_no_duplicate_loci(combined, source_path=str(path))
    return combined, total
