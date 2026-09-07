# MEASUREMENT 2026-09-04 -- one genome, one name, and forty-two references to a path that no longer exists

**Author: Monzia Moodie**
**Measured at commit:** f44341b
**Data operation performed:** 2026-09-07, outside the repository (`data/` is ignored)

---

## 0. Why this happened

`configs/data_manifest.yaml`, in the `gencode` declaration, carried a standing
instruction that nothing had acted on:

> The same release directory also publishes GRCh38 genome FASTA, so the
> undeclared `data/external/grch38` may be GENCODE-sourced -- **measure its
> consumers before declaring or moving it.**

The instruction was followed. Its hypothesis was REFUTED, and the measurement
found a duplicate, a broken declaration, and forty-two references.

---

## 1. What was on disk

```
data/external/grch38                              4,033,396,532 B, UNDECLARED
  GRCh38.fa                                       3,151,425,851 B
  GRCh38.fa.fai                                           6,600 B
  Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz    881,964,081 B

data/external/reference                           DECLARED
  Homo_sapiens.GRCh38.dna.primary_assembly.fa.fai         6,600 B
  (the .fa it indexes)                            ABSENT LOCALLY
```

The declared read form was present on Google Drive at
`genvarcla:genomic-variant-classifier/data/external/reference/`, 3,151,425,851
bytes, and absent from the local working copy, which held only its orphaned
index.

**A WITHDRAWN CLAIM.** An earlier statement in this session read *the declared
reference FASTA does not exist; only its orphaned index remains.* That was
measured on the LOCAL tree alone and asserted repository-wide. Drive was not
checked until Monzia asked. The correct statement is
`REFERENCE-READ-FORM-IS-DRIVE-ONLY-LOCALLY-1`: present where declared, absent
where the code looks.

---

## 2. The two files were proven identical THREE ways

Not by size. Size equality is not content equality, and this project has paid
for that: `ALIAS-MERGE-VERIFIES-BY-SIZE-NOT-DIGEST-1` recorded two EVE score
files at exactly 612,501 bytes with different digests.

**FULL SHA-256 over every one of the 3,151,425,851 bytes, both sides:**

```
local  data/external/grch38/GRCh38.fa
       1e74081a49ceb9739cc14c812fbb8b3db978eb80ba8e5350beb80d8ad8dfef3b
Drive  .../reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa
       1e74081a49ceb9739cc14c812fbb8b3db978eb80ba8e5350beb80d8ad8dfef3b
```

Drive's digest came from `rclone hashsum sha256`, computed SERVER-SIDE at no
transfer cost. The local digest took 3.8 seconds.

**BOTH .fai INDEXES, identical:**

```
1411003e3a78242551943185c3a4158920270c7564fae336200384e9e411e813
```

A FASTA index records, per sequence, its NAME, its LENGTH, and its BYTE OFFSETS
into its own file. Two identical indexes mean identical sequence names,
identical lengths and identical byte layout -- character-for-character
equivalence proven on evidence the file digest does not cover.

**THE HEADER, read verbatim from the first bytes:**

```
>1 dna:chromosome chromosome:GRCh38:1:1:248956422:1 REF
```

Ensembl's own convention (`>1`, not UCSC's `>chr1`), and 248,956,422 is the
correct GRCh38 chromosome 1 length. The declaration's `acquire` field says
Ensembl; the bytes agree. The `.gz` in the same directory carries Ensembl's
filename, so the GENCODE hypothesis is REFUTED.

---

## 3. Why the descriptive name won, and why nothing is documented as an exception

`GRCh38.fa` states the assembly. `Homo_sapiens.GRCh38.dna.primary_assembly.fa`
states the species, the assembly, that it is DNA rather than complementary DNA
or protein, and that it is the PRIMARY ASSEMBLY -- excluding scaffolds, patches
and alternate haplotypes. That last distinction is scientifically load-bearing
and the short name omits it entirely.

It is also the publisher's own filename, the name the declaration specifies,
and the name Drive already carries. There is NO reason for two names for one
file, so no exception is documented: the descriptive name simply wins.

---

## 4. Why `consolidate_aliases.py` was the WRONG tool

The project has a consolidation tool and its 2026-08-29 digest repair is
present and correct. Traced against these directories it produces the wrong
result:

```
alias  data/external/grch38      3 files
canon  data/external/reference   1 file   -> canon NON-EMPTY -> the MERGE path
```

**The merge PRESERVES FILENAMES and never renames.** No name collides, so all
three files copy across as-is, leaving `GRCh38.fa` -- the less descriptive name
-- in the canonical directory, beside a SECOND index byte-identical to the one
already there. `CONSOLIDATE-ALIASES-CANNOT-RENAME-1`.

A purpose-built script performed the rename instead. Its parse tree carries
EXACTLY ONE mutation call site, `shutil.move`; no `rmtree`, no `unlink`, no
`remove`. It could not delete anything even if the judgement behind it were
wrong.

---

## 5. What was done, and what the auditor says now

```
RENAMED   data/external/grch38/GRCh38.fa
      ->  data/external/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa
          re-digested at the destination: 1e74081a... UNCHANGED
          source confirmed gone: a rename, not a copy

DELETED   GRCh38.fa.fai              6,600 B   proven byte-identical duplicate
DELETED   ...primary_assembly.fa.gz  881,964,081 B  no consumer in 42 sites,
                                     absent from Drive, and the declaration
                                     records its intended removal
DELETED   data/external/grch38/      now empty
          881,970,681 B reclaimed
```

Deletions were performed by Monzia in a SEPARATE block after reading the
verification, per the standing rule for irreversible commands. The destination
digest was confirmed BEFORE any removal.

`scripts/maintenance/audit_data_tree.py`, run afterwards:

```
ok    reference  external  public  public_redownloadable  sync=True  2.9GB  2f
[warn] external/eve_smoke: ORPHAN in external/ (not in manifest)
VERDICT: 1 warning(s)
```

`reference` is now POPULATED and RECOGNISED. Of the three orphans the auditor
first reported -- together 4,685,941,722 bytes -- `gencode` was declared at
`24bfb11` and `grch38` is folded. **Only `eve_smoke` remains.**

---

## 6. The forty-two references, and an arithmetic defect

MEASURED at `f44341b`: 42 occurrences across 27 files, in three classes.

```
GRCh38.fa   (plain)   36   a real file that MOVED
GRCh38.fasta           4   a CANDIDATE that never existed on disk
bare directory         2   prose: the manifest note and a test docstring
```

**AN ARITHMETIC DEFECT, AND WHY THE TOTAL HID IT.** The first decomposition
reported `bare directory: -2`. `data/external/grch38/GRCh38.fasta` CONTAINS
`data/external/grch38/GRCh38.fa` as a PREFIX, so the plain count included the
`.fasta` occurrences and the bare count then subtracted them TWICE. The TOTAL
was 42 either way -- the double-count of +4 and the double-subtraction of -4
cancelled exactly. **A correct total from a wrong decomposition is the shape
that hides a defect**, and the negative was the only reason it was found.
`PREFIX-CONTAINMENT-IN-A-SUBSTRING-COUNT-1`.

**THE SAME CONTAINMENT IS A CONSTRAINT ON THE REPAIR.** Replacing the shorter
string first turns `GRCh38.fasta` into `...primary_assembly.fasta`, a path that
does not exist -- one broken default traded for another. The longer strings are
replaced FIRST, and the ordering is asserted rather than assumed.

**THE FOUR `.fasta` SITES ARE FALLBACK CANDIDATE LISTS**, read in context:

```python
for c in ["data/external/grch38/GRCh38.fa", "data/external/grch38/GRCh38.fasta"]:
    if Path(c).exists():
        fa_path = Path(c); break
if fa_path is None:
    print("ABORT: reference not found."); return 2
```

Three files share that exact shape and now ABORT, because both candidates are
dead. `probe_seq_feasibility.py` differs: its literal list is followed by
`glob.glob("data/**/*.fa", recursive=True)`, which SELF-HEALS -- the recursive
glob finds the new path regardless.

The `.fasta` spelling never existed. Keeping it would preserve exactly the
two-names tolerance this consolidation removed, so the two entries collapse to
the one canonical path.

---

## 7. What is NOT claimed

That the other `data/external/` sources were audited. Only `grch38` and
`reference` were. `eve_smoke` remains an orphan and is untouched here.

That `data/external/GRCh38.fa` and `data/reference/GRCh38.fa` -- two further
dead candidates in `probe_seq_feasibility.py` -- were repaired. They predate
this work and are outside its scope; they are recorded, not fixed.

That the twenty-five scripts still RUN. Their defaults are repaired; whether
each script is live, historical, or superseded was not measured.
`seq_windows.py` and `populate_fasta_seq.py` ARE retired -- `tests/test_build_
seq_windows.py:5` states it -- so the `reference` declaration's
`CODE-REFERENCED -- do NOT rename` names a retired consumer.
`REFERENCE-DECLARATION-NAMES-A-RETIRED-CONSUMER-1`, recorded not repaired.

That any Drive copy was modified. Drive was READ ONLY, by `rclone lsl` and
`rclone hashsum`. It remained the durable third party throughout.
