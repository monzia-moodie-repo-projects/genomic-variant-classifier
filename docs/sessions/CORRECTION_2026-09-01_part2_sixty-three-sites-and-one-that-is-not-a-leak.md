# CORRECTION 2026-09-01 part 2 -- sixty-three sites, and one that is not a leak

**Author: Monzia Moodie**
**Applies to:** `11df0b5`
(`SESSION_2026-09-01_a-coordinate-the-evidence-required.md`, section 5)
**Status:** the corrected claims are recorded here; `11df0b5` is not amended.

---

## 0. What went wrong

`11df0b5` registers `RESOURCE-WARNING-FROM-UNCLOSED-READS-1` and describes it
as three named sites:

> `tests/unit/test_backup_artifacts.py:80` does
> `ast.parse(io.open(f, encoding="utf-8").read())` with no close, once per
> source module scanned, and `scripts/train.py:76` leaves `logs/train.log` open
> through `logging.basicConfig`. Also `tests/test_resolve_alleleless_ncbi.py`
> at 75, 119 and 122.

Two things are wrong. **The scope is far larger than three files**, and **one
of the three is not a leak.**

---

## 1. SIXTY-THREE sites across THIRTY-FIVE files

MEASURED 2026-09-01 at `e109de9`, by parsing every tracked Python file and
counting each call to `open` whose result is neither assigned to a name nor
bound by a `with` statement:

```
TOTAL unmanaged open() calls   63
files                          35
```

The heaviest are `tests/unit/test_phylop_bigwig.py` (9),
`tests/unit/test_json_state_store.py` (7), `tests/unit/test_runtime_paths.py`
(4) and `tests/test_resolve_alleleless_ncbi.py` (3).

### And the census MISSES the site that produces most of the warnings

`tests/unit/test_backup_artifacts.py:80` does not appear in it, because it
calls `io.open` and the detector matched the attribute name `open` on the
module `io` -- the same shape as `PROBE-FETCH-CALL-NAME-IS-AMBIGUOUS-1`, where
`dict.get` and `requests.get` were indistinguishable by callee name.

So the warning count (33) undercounts the SITES, and the site census (63)
misses the line that generates most of the WARNINGS. Neither number is the
scope on its own, and I quoted the smaller one as though it were.

---

## 2. TWO POPULATIONS INSIDE THE SIXTY-THREE, and a third OUTSIDE it

Counted exactly from the census, not approximated:

```
scripts, one-shot patchers   28 sites in 24 files
    apply_*.py, patch_*.py, install_*.py, build_*.py, migrate_*.py.
    They ran once, are tracked for the record of what they did, and will
    never run again. Repairing them changes nothing and touches files
    whose only value is that record.

tests                        35 sites in 11 files
    these run on every gate. This is the population worth repairing.
    Heaviest: test_phylop_bigwig.py (9), test_json_state_store.py (7),
    test_runtime_paths.py (4), test_resolve_alleleless_ncbi.py (3).

                             63 sites in 35 files   TOTAL
```

`scripts/train.py:76` IS NOT AMONG THEM. It is a `logging.basicConfig` call,
not a bare `open()`, so a detector looking for unmanaged `open` never saw it.
It is a SEPARATE finding, and section 3 shows it is not a defect.

An earlier draft of this record said "~22" and "~39" and placed `train.py`
inside the sixty-three. Both were wrong, and counting the census exactly is
what found it.

---

## 3. `scripts/train.py:76` IS NOT A DEFECT

```python
logging.basicConfig(
    level=logging.INFO,
    ...
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/train.log", mode="w"),
    ],
)
```

A `logging.FileHandler` OWNS its file for the process lifetime by design. That
is what a file handler is. The `ResourceWarning` is real -- Python emits it
because the handler is never closed at interpreter teardown -- but "leaves
`logs/train.log` open" describes normal logging behaviour, not a leak.

**Repairing it would break logging.** Closing the handler after
`basicConfig` would discard every subsequent log line; adding teardown changes
production behaviour in a training entry point.

`11df0b5` names it alongside two genuine unmanaged reads, which invites exactly
that repair. Recorded here so a later session does not make it -- and it is
outside the sixty-three, so a site count will never surface it either.

---

## 4. Why I proposed this as a unit, and why it is not one

I called the finding "decision-free" on the strength of ONE site I had read --
`test_backup_artifacts.py:80`, where
`ast.parse(io.open(f, encoding="utf-8").read())` becomes
`ast.parse(Path(f).read_text(encoding="utf-8"))` with identical behaviour.

That site IS decision-free. Sixty-three are not, three populations are not, and
one is not a defect at all.

The rulings already classify this correctly:

> Harden critical integration-test skip semantics and tracked-file mutation
> tests, but treat these as test-infrastructure debt rather than blockers.

Acting on it now, ahead of item 5, would be choosing volume over the priority
order.

---

## 5. What stands unchanged in `11df0b5`

The finding itself. Unmanaged reads exist, the warning count is stable at 33
across six consecutive gates, and on Windows an unclosed handle can block a
later rename -- which is what an installer's atomic replace performs.

Only the SCOPE and the `train.py` characterisation were wrong.

---

## 6. Status

`RESOURCE-WARNING-FROM-UNCLOSED-READS-1` remains OPEN and is NOT ready to be a
repair unit. If it becomes one, its scope is the tests population, its site
count must be measured with a detector that sees `io.open` as well as `open`,
and `scripts/train.py:76` must be excluded by name.

No file in the repository is changed by this record.
