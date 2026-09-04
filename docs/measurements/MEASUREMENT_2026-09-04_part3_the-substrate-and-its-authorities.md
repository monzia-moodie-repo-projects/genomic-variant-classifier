# MEASUREMENT 2026-09-04 part 3 -- the substrate, its authorities, and a cycle population that was not there

**Author: Monzia Moodie**
**Measured at:** `fb53610`
**Status:** MEASUREMENT ONLY. Nothing is built.

---

## 0. Why this exists

Phase 3B.0 asks where this repository decides how its artifacts are
interpreted, decoded, decompressed, loaded, persisted or materialized. The
governing constraint, stated before any of it began, is that a closed
`MediaType` vocabulary must not be invented before that census exists, and
that a search finding nothing has not proven absence.

Three units ran on 2026-09-04, each inheriting the previous one's population
and refusing if it had moved. None of them adjudicated anything, followed no
representation semantics, and repaired nothing.

---

## 1. 3B.0a -- the substrate, frozen

MEASURED at `fb53610`: **1,720 tracked paths**, and the HEAD tree, the index
and the working tree measured as THREE separate populations rather than
assumed equal.

```
tracked paths       1720      1719 regular + 1 executable
symlinks 0          gitlinks 0          other modes 0
committed bytes     47,240,133 across 1,720 blobs, every size from git cat-file
extensionless       16        numeric suffix fragments 0
case-fold collisions 0        Unicode NFC collisions 0
core.autocrlf true   core.ignorecase true   core.symlinks true
object format sha1   git 2.55.0.windows.3
```

The population digests held identical across FOUR probe revisions while the
census bytes changed each time, which is the property that makes the substrate
citable: the instrument improved and the finding did not move.

### `.gitattributes` is a materialization authority, not housekeeping

MEASURED by ASKING git rather than parsing the file: a declared attribute
covers **1,720 of 1,720** paths.

```
text=set   eol=lf      1607
text=auto                48
text=set   eol=crlf      43     exactly the 43 PowerShell scripts
text=unset eol=lf        18
text=unset                4
```

`core.autocrlf` is true, so committed blob bytes are NOT worktree bytes for
text. Every subsequent unit therefore reads blobs from the object store; a
probe reading the worktree would measure a checkout transformation and call it
the artifact.

### Eighteen paths are declared binary AND given a line-ending directive

Seventeen are the preserved install attestations. `records/README.md` documents
the rule and calls its ordering load-bearing: `-text` unsets TEXT and does NOT
unset EOL, because they are separate attributes. The residue is an INERT
directive on a binary path -- no bytes change, and the preservation claim is
unaffected.

The eighteenth is `docs/archive/legacy/ROADMAP_2026-03_to_2026-08-22.md`, and
its cause is different: `docs/archive/legacy/** -text`, added 2026-08-23 to
freeze archived documents.

**That rule's comment carries an expired premise.** It states that
`docs/archive/legacy/` "holds nothing yet, so this cannot change the checkout
representation of any tracked file". True on 2026-08-23. MEASURED today: the
directory holds one tracked file and the rule governs it, exactly as intended.
The rule is correct; the comment asserts a fact that moved.
`ARCHIVE-LEGACY-COMMENT-PREMISE-EXPIRED-1`.

The same comment states that "the three artifacts already under
`docs/archive/` resolve to `text: auto`". MEASURED: the count is still three,
the attributes are not. `worktree_status_at_removal.txt` resolves to
`text: set, eol: lf` because `*.txt text eol=lf` on line 15 predates the
archive block. Two of three, not three of three.

---

## 2. 3B.0b -- where the repository decides, directly

MEASURED: **11,372 decision sites across 1,197 of 1,720 files**, zero parse
failures across 1,063 abstract-syntax-tree parses.

```
python_ast 6609    config 2268    lexical_backstop 2252    powershell 243
parsed 6609        lexical 4763
explicit 4905      IMPLICIT 522
```

Evidence strength is recorded because a parsed site is not a matched one.
PowerShell findings are LEXICAL and labelled so: there is no PowerShell parser
here, and pretending otherwise would weight a regular-expression hit equally
with a syntax-tree node.

Implicit decisions are counted separately. A default is an authority too, just
a poorly located one, and 261 sites open a text file without naming an
encoding.

### Two numbers that point outside the population

`parquet.read` 376 and `parquet.write` 255 against **four tracked `.parquet`
paths**. Parquet input and output overwhelmingly targets artifacts that are
not in the repository -- a materialization surface the substrate census cannot
see by construction.

### The backstop is noisy BY DESIGN, and its noise is reported

2,252 unresolved calls across 487 distinct names. The top three --
`argparse.ArgumentParser` 235, `ap.parse_args` 186, `ast.parse` 121 -- are 542
sites and none is input or output at all. Reporting the raw count as an
adjudication backlog would overstate the work by that share.

---

## 3. 3B.0c -- reachability, and a cycle population that was not there

The first attempt (v1, preserved at digest `173a824b`) produced numbers that
are NOT repository facts. Its resolver collapsed `parser.parse_args()` to the
bare name `parse_args` and bound it to a module-level function of that name.
MEASURED in its own census: of 48 origins it reported as `recursive_cycle`, 13
carry attribute-form origins -- `p.parse_args` 7, `ap.parse_args` 3,
`fasta.fetch`, `parser.parse_args`, `self._fetch_pubmed`. An attribute call
cannot recurse into a module function that merely shares its suffix.

v2 preserves the syntactic form of every call, so that binding is structurally
impossible. MEASURED at `fb53610`:

```
origins 2252        with an authority 1667        (v1 said 525)
components 11066    cyclic components 17          largest cyclic member 1
graph symbols 11066 resolved call sites 11,958 -> 9,308 unique edges
parse failures 0    sites unmapped to a region 0
```

**A cyclic component of size 1 is self-recursion.** All seventeen are size 1,
so this repository contains ZERO multi-function recursion cycles. Every one of
v1's forty-eight is accounted for:

```
unresolvable:recursive_cycle -> partial_open_boundary -> authority   24
unresolvable:recursive_cycle -> partial_open_boundary -> none        24
```

Twenty-four reach an authority once the false cycle is removed. v1 abandoned
all forty-eight.

### More authority AND less certainty, which is the correct direction

```
partial_open_boundary   2211   98.2%
complete                  41    1.8%
```

982 origins that v1 called unresolvable reach an authority in v2; 6 that v1
called resolved-to-authority reach none. v1 was wrong in both directions.

And v1 declared 525 `resolved_to_authority` plus 337 `resolved_no_authority`
-- 862 claims of completeness the evidence does not support. Completeness is
derived from open boundaries and traversal limits ONLY. A cycle is topology
and never makes an observation partial.

### The traversal bound is measured, not stipulated

```
ceiling   authority   limited   complete   maxdepth
   2           1652       176         41          2
   4           1667        14         41          4
   8           1667         0         41          7
  16           1667         0         41          7
  32           1667         0         41          7
saturated at ceiling 4
```

Monotonicity holds across all five. This replaces v1's claim that "a ceiling
of 3 would resolve every chain that resolves at all", which was tautologically
true only over chains already observed to resolve.

---

## 4. Two defects in the instruments, both measured

### `PRODUCER-IDENTITY-1`

Two v1 artifacts of identical length differed in exactly ONE leaf of 15,771:
`header.probe_sha256`. The program read `__file__` twice -- once for its
banner, once when building the header -- and the file changed between them.
A report and the artifact it produced disagreed about which program made it,
and nothing noticed.

v1 is BYTE-DETERMINISTIC; a hypothesis that this was an ordering defect was
refuted by the measurement. The affected artifact is renamed
`gvc_3b0c_delegation_PRODUCER-IDENTITY-INCONSISTENT_f8c4b7f1.json`, bytes
preserved and verified, with a sidecar recording the disposition rather than
editing the evidence.

v2 captures a content-addressed snapshot at IMPORT, before any work, and
re-verifies at exit. A mid-run replacement was reproduced deliberately and is
reported as evidence inside the result.

### `SITE-IDENTITY-LACKS-A-COLUMN-1`

3B.0b records path, line, operation and evidence, but no column offset. Two
calls on one line with the same expression are indistinguishable in the
inherited evidence. MEASURED: five such tuples, ten origins.

```
scripts/check_agents_active.py                  line 172  _parse_iso(...)  x2
scripts/diff_cohorts.py                         line 101  _load(...)       x2
tests/unit/test_phylop_bigwig.py                line 541  opener(...)      x2
tests/unit/test_provenance_migration_corpus.py  line 157  _load(...)       x2
tests/unit/test_provenance_migration_corpus.py  line 158  _load(...)       x2
```

v2 REFUSED the census rather than silently merging them. A canonical ordinal
now separates them, assigned over a sorted ordering so input order cannot
change the result, and the groups are listed rather than absorbed. Any later
unit keying on a 3B.0b site identity inherits this ambiguity.

---

## 5. What this does NOT decide

**What any of it means.** No `MediaType` vocabulary is named. No
representation is adjudicated. No conflict is resolved. The eighteen
binary-with-line-ending declarations, the parquet operations pointing outside
the population, and the 2,252 unresolved calls are observations awaiting the
adjudication unit.

**Whether the 3B.0b site schema should gain a column.** It should be decided
with the ambiguity measured, not assumed away.

**Where the raw census artifacts belong.** They remain outside the repository.
`records/measurements/` is declared and empty, so writing there is a first
write to a new record family, and `EVIDENCE-DISPOSITION-INCONSISTENT-1` rules
that legacy evidence is classified individually rather than moved because a
directory now exists.

---

## 6. Status

Nothing in the repository is changed by this record beyond its own creation
and the changelog entry accompanying it.

The substrate is frozen and citable. The direct decisions are enumerated. The
reachability graph is measured under an explicit, saturated bound with every
unresolved edge carrying a named reason. What none of it yet does is say what
any artifact IS -- which was the whole point of measuring first.
