# CORRECTION 2026-08-21 -- a comparison needs one decoding
**Author: Monzia Moodie**
**Applies to:** `db7c4b9` -- docs: session record for 2026-08-20 to 2026-08-21
**Placed beside that record, not inside it.**

---

## What the record claims

The commit message of `db7c4b9` states, under VERIFIED BEFORE COMMITTING:

```
506,875 -> 511,044 bytes; the previous content is a suffix of the new
```

**That claim is true.** This note does not retract it. What this note records is
that the check I ran *first* could not have established it, and that I came
close to accepting its output without noticing.

---

## What was actually run

Before committing, the changelog prepend was verified with:

```python
new = io.open("docs/CHANGELOG.md", encoding="utf-8").read()
old = subprocess.run(["git", "show", "HEAD:docs/CHANGELOG.md"],
                     capture_output=True, text=True).stdout
print("  new = entry + old: {}".format(new.endswith(old)))
```

It printed:

```
committed length : 506,875
working length   : 509,807
new = entry + old: False
prepended chars  : 2,932
```

Three figures disagree with the install step, which had reported the file
growing from **506,875 to 511,044 bytes** after prepending **4,169 bytes**.

---

## Why the comparison was invalid

`subprocess.run(..., text=True)` decodes the child's output using
`locale.getpreferredencoding()`. On this workstation that is the Windows
codepage, not UTF-8. The file was simultaneously read with an explicit
`encoding="utf-8"`.

So the two sides of `new.endswith(old)` were **two different decodings of the
same bytes**, and the result carried no information about the file at all.

The tell was visible in the numbers before the `False` was:

```
committed length : 506,875   <- equals the BYTE count exactly
```

A UTF-8 document containing 1,894 non-ASCII bytes cannot have a character count
equal to its byte count. That equality is the signature of a one-byte-per-
character decoding.

---

## What settled it

A byte-level comparison, with no decoding on either side:

```
HEAD~1 changelog : 506,875 bytes  sha facb8736dfaea875
working changelog: 511,044 bytes  sha bbd47b3e0a643037
entry            :   4,169 bytes

new == entry + old  : True
new ends with old   : True
new starts w/ entry : True
arithmetic          : 4,169 + 506,875 = 511,044  (actual 511,044)
non-ASCII bytes     : old 1894   new 1894        unchanged
BOM  : False   CRLF : False
decodes as UTF-8    : True, 509,807 characters
bytes - characters  : 1,237  (multi-byte sequence overhead)
```

And independently, from git rather than from any script of mine:

```
git diff HEAD~1 HEAD --numstat -- docs/CHANGELOG.md
67      0       docs/CHANGELOG.md
```

**Zero deletions.** Nothing existing was altered.

The 1,237 difference between bytes and characters is exactly the multi-byte
overhead of the 1,894 pre-existing non-ASCII bytes, most of them em-dashes,
which occupy three bytes each in UTF-8.

---

## The transferable rule

> A comparison is only evidence if both sides were measured the same way.

`text=True` is a decoding decision, not a convenience. On Windows it is not
UTF-8. When comparing file content:

- read **bytes** on both sides, or
- decode **both** sides explicitly with the same codec.

For subprocess output specifically, prefer `capture_output=True` without
`text=True` and decode deliberately, or pass `encoding="utf-8"` so the choice
is stated rather than inherited from the machine.

---

## Why this note exists at all

This is the thirteenth instance in two days of one recurring shape, named in
the session document as:

> I check for the presence of what I intend and not the absence of what I need.

It occurred **inside a verification written specifically to catch such things**,
one exchange after that pattern was named in a commit message. The instrument
that was supposed to check the work needed checking, and the only reason it was
caught is that its three printed figures contradicted figures printed minutes
earlier by the install step.

That is the argument for printing intermediate measurements rather than only
verdicts: a lone `False` is unfalsifiable, but a `False` sitting beside
`506,875` when the install said `511,044` is a contradiction anyone can see.

---

## Status

`db7c4b9` stands as committed. Its content is correct, its claim is true, and
the changelog is intact -- verified at byte level and confirmed independently by
`git diff --numstat`.

No file is changed by this note. It is a record of how a true claim was nearly
established by an invalid method.
