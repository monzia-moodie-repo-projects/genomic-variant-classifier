# SESSION 2026-09-02 -- the substrate moves down

**Author: Monzia Moodie**
**Commits:** `accdf49`, `4805033`, `2d90c23`, `1ef4ca5`
**Ratchet:** 5732 -> 5905
**Preceding head:** `4eea19d` (via `e109de9`, `522ef1b`)
**Ending head:** `1ef4ca5`, pushed, `+0 -0`

---

## 0. What this covers

| commit | unit | transition | ratchet | gate |
|---|---|---|---|---|
| `accdf49` | ADMISSION-BOUNDARY | ADDITION +14 | 5732 -> 5746 | 5731p/15s |
| `4805033` | PROVENANCE-HASHING | ADDITION +17 | 5746 -> 5763 | 5748p/15s |
| `2d90c23` | MIGRATION-CORPUS | ADDITION +67 | 5763 -> 5830 | 5815p/15s |
| `1ef4ca5` | PROVENANCE-OWNERSHIP | +76 / -1 | 5830 -> 5905 | 5890p/15s |

Phase 1C units 1, 2, 3A.0 and 3A of the adopted design authority. Phase 0 --
repository and remote reconciled -- was confirmed at `accdf49` and has held
per unit since.

---

## 1. `accdf49` -- the factory stops stringifying whatever it is handed

`SourceArtifactKey.of` is the admission boundary for scientific evidence and
called `str()` on its arguments. MEASURED:

```
of(None, kind)            produced a source named "None"
of(3, kind)               produced "3"
of(Path("clinvar.vcf"))   produced a plausible name for a PATH
of("clinvar", "nonsense") propagated a bare ValueError from the enum
```

It now refuses a non-string source or product, an empty or whitespace-only
source, an empty product, and an unknown `ArtifactKind`. It ACCEPTS surrounding
whitespace and strips it, which the dataclass pattern used to reject outright.

`SourceIdentityError` SUBCLASSES `SourceError`, itself already a `ValueError`.
The design authority specified `ValueError`; subclassing the narrower existing
error satisfies that AND keeps every `pytest.raises(SourceError)` catching the
new one -- which is why fourteen identities were added and NONE removed.
Sabotaging the base back to `ValueError` fails THIRTEEN tests.

### Ten boundaries sabotaged, nine detected, one mutation a no-op

Two misses were REAL TEST WEAKNESSES. Disabling the factory's empty-source and
empty-product checks still raised, because `_SOURCE` and `_PRODUCT` reject the
empty string in `__post_init__`. Defence in depth is correct behaviour, and a
test that cannot say WHICH layer refused is weak: at an admission boundary the
factory owes the caller a reason, and a pattern-match failure is not one. Both
tests now assert the factory's own message.

---

## 2. `4805033` -- a digest that refuses when the file moves underneath it

MEASURED 2026-09-01 BY EXECUTION: three helpers hash a file and all three
produce the identical digest --
`constraint_canonicalize.py:325`, `phylop_cache.py:158`,
`science_claw/ledger.py:70`. Duplication, not disagreement.

MEASURED 2026-09-02: not one calls `stat()`, reads `st_mtime_ns`, or raises
when the file changes underneath the read. A digest over a file being
rewritten describes BYTES THAT NEVER EXISTED AS A WHOLE FILE. For a
636,522,106-byte GENCODE artifact the read is long enough for that to be a
real window.

`digest_file` stats before and after and refuses on either change.

THE THREE ARE NOT MIGRATED: seventeen call sites across eleven files use those
names, four of them test files that pin them by identity.
`test_all_four_implementations_AGREE` prevents drift by EXECUTING all four.

### Three sabotage gaps, all real

Comparing only `st_size` passed every test, because the fake stat changed the
size too -- a file rewritten IN PLACE at the same length moves mtime and NOT
size. A `Warning` base passed, because `Warning` subclasses `Exception` while
a `-W ignore` run would swallow it. And nothing asserted `FileDigest` was
frozen.

---

## 3. `2d90c23` -- freeze what an identity MEANS, before its owner moves

Three oracles that fail for different reasons: the existing suite,
`semantic.json`, and sixteen pickles. Plus `ownership.json`, recording
`__module__` -- the ONE thing 3A changes, kept OUT of the semantic oracle so
that oracle cannot fail by construction.

### The corpus was NOT REPRODUCIBLE, and `--check` could not see it

Three fixtures carry `frozenset` of a str-based enum, whose iteration order is
randomized per process. Three consecutive runs produced `68e9ee011ce2`,
`efaa2a1ed0b5`, `68e9ee011ce2` for one fixture OF CONSTANT LENGTH.

`--check` compares `loaded == original`, and set equality is order-independent.
It reported ZERO failures on genuinely different bytes.

What found it was comparing all SIXTY-FOUR digest characters instead of
sixteen -- after "byte-identical" had already been written. The generator now
REFUSES an unpinned seed and `main()` re-executes with `PYTHONHASHSEED=0`.

---

## 4. `1ef4ca5` -- one class object, two import paths

MEASURED: `CoordinateContext`, `ArtifactKind`, the source-evidence kernel and
`TransformationIdentity` all lived under `monitoring/drift/`, while the drift
package imported NOTHING outside `monitoring` -- a pure leaf holding the
scientific identity substrate. They were never monitoring concepts; they were
born there in Phase 1B, before `provenance` existed.

THREE OF THE FIVE CANONICAL MODULES ARE BYTE-IDENTICAL to their originals:
`serialization.py` = the old `_digest.py`, `coordinate.py` = the old
`coordinate.py`, `artifact.py` = the old `source_vocabulary.py`. The same file
at a new path. `source.py` differs by twenty-one bytes of imports and
`transformation.py` by the extracted comparison; every top-level definition was
unparsed before and after and proven identical.

Twenty-two names are EXACT aliases -- `drift.X is provenance.X`. Not a
subclass, which would be a different runtime type; not a copy, which a pickle
would resolve to the wrong authority. `__module__` is NOT forged: old pickles
load because a pickle resolves `module.Name` THROUGH the legacy module, which
hands back the canonical object.

`differing_components` moved to `transformation_delta.py`, body byte-identical.
Provenance defines states; monitoring compares them. The drift `__all__` is
UNCHANGED at 32 names.

---

## 5. Five refusals, and what each bought

| refused by | what it caught |
|---|---|
| `test_no_live_module_fabricates_a_poly_window_literal` | the corpus generator built `'a' * 64`; A and C are nucleotides, B and D are not |
| `str.format()` at PUBLICATION | `KeyError: 'SourceRole, '` -- any brace group is a replacement field |
| the frozen oracle | 37 failures: `representation.py` imported `differing_components` from a module I had emptied |
| the acceptance gate | Unit 2's test pinned `provenance.__all__` to a CLOSED list of three |
| the transition guard | a blanket `if removed: raise`, correct for an addition and wrong for a rename |

THE POLY-WINDOW GUARD WAS RIGHT REGARDLESS OF MY INTENT. My constants were
SHA-256 test digests, but content cannot distinguish a real poly-adenine tract
from a fabricated placeholder -- which is exactly why four earlier detectors
were blind to the same 21,814 rows. Exempting the file would have reopened the
hole it was built to close. Digests are now DERIVED, and the example forms are
deliberately absent from the docstring because I have not read whether the
guard inspects syntax or text.

THE PUBLICATION FAILURE left the repository recoverable but not clean: the gate
had passed, the journal was destroyed, 24 files were STAGED, and `HEAD` had not
moved. Committing them by hand would have produced a change with no
`plan_digest`, no suite transition and no acceptance evidence. The tree was
restored and the unit re-run.

---

## 6. Errors made

| # | error | how it surfaced |
|---|---|---|
| 1 | Compared digests at SIXTEEN characters, then wrote "byte-identical" | three of nineteen differed; only the full comparison found it |
| 2 | `--check` verified loaded equality, not bytes, and could not detect it | sabotage-by-repetition |
| 3 | Built `'a' * 64` in a live module | the poly-window guard |
| 4 | Wrote `frozenset({SourceRole, ...})` in a `.format()` template | `KeyError` at publication |
| 5 | My placeholder audit matched only `\w+` fields | it reported "placeholders == call: True" on an unformattable string |
| 6 | Measured `drift/__init__.py`'s imports and stopped | `representation.py` also imported the moved function |
| 7 | Assumed the drift facade exports everything relocated | it never exported `canonical_json`, `domain_digest`, `EVIDENCE_DOMAIN` or `SourceIdentityError` |
| 8 | Wrote a test that ALWAYS skips | the skip count moved 15 -> 16 |
| 9 | Computed the transition as 75/0 instead of measuring | it is 76/1; a rename removes an identity |
| 10 | Left `all_targets` referenced after deleting the block that defined it | my own unresolved-name audit |
| 11 | Left `expected_removed_nodeids=frozenset()` and a justification from another unit | reading the block I had just edited |

Errors 1 and 9 are one shape: **arithmetic instead of measurement**. Errors 5
and 7 are another: **a check that inspects only well-formed or expected inputs
cannot find the malformed or unexpected one.**

---

## 7. Findings

### Registered
`SOURCE-IDENTITY-ERROR-NOT-EXPORTED-1`. `SourceIdentityError` was introduced at
`accdf49` and is absent from `drift.__all__`, which still holds 32 names. It is
catchable as `SourceError`, so nothing is broken, but a caller wanting to
distinguish an admission refusal cannot import it from the package surface. NOT
repaired in 3A, which had to be semantic-zero; `provenance.__all__` does export
it.

`INSTALLER-HEADER-UNDERSTATES-A-MIXED-TRANSITION-1`. The banner reads
"ADDITION of 76" while the unit also retires one. Display only.

### Still open
`DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1` -- closes at Unit 14.
`FILE-DIGEST-HELPER-DEFINED-THREE-TIMES-1`, `HASHING-MIGRATION-PENDING`,
`SOURCE-ACQUISITION-KEY-ONLY-MATCH-1`,
`EVIDENCE-DOMAIN-V4-PAYLOAD-SCHEMA3-1`,
`RESOURCE-WARNING-FROM-UNCLOSED-READS-1`,
`CONFIG-DECLARES-A-SECOND-PATH-VOCABULARY-1`,
`VALIDATOR-CHECKS-A-LOCATION-THE-DATA-LEFT-1`,
`AUDITOR-TREATS-AN-EMPTY-DIRECTORY-AS-PRESENT-1`,
`CONNECTOR-SOURCE-NAMES-DISAGREE-WITH-THE-MANIFEST-1`,
`DATABASE-CONNECTORS-NOT-BYTE-EXACT-BY-TRANSCRIPT-1`.

---

## 8. Ending state

```
HEAD     1ef4ca5, pushed, +0 -0
ratchet  5905
gate     5890 passed, 15 skipped, 0 failed, 33 pre-existing warnings
suite    6fe8a6ed7514b48f -> 95949dac84de867c
```

## 9. Next intended action

Unit 3B: `MaterializationIdentity` and the derivation model. `FileDigest`
already supplies `sha256` and `size_bytes`; `media_type` and the
`DerivationEdge` are new.

The frozen corpus is now an ASSET rather than a formality: `ownership.json`
records the PRE-MOVE owners, so a later unit can prove the move happened by
comparison rather than assertion.
