# SESSION 2026-07-30, part five — one constant, two meanings

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Commit:** `dfc9d74`, on top of `5c29ff7`
**Ratchet:** 4120 -> 4121 (+1), computed by collection
**Suite:** 4115 passed, 6 skipped, 0 failed, 766.22 s
**Python:** 3.12.10 in `.venv312`

---

## 1. The defect

`working_cache_gib` feeds a gate that decides **whether a run may start**, and
`configs/data_manifest.yaml` documented it as *"JEPA embedding cache built during
a full run"*. That claim was false in both directions:

* the JEPA embedding cache is a **one-time artifact** that no training run builds;
* **14.7 is the pooled-only figure the JEPA design explicitly forbids** — *"do
  not cache only one pooled vector"*, `ROADMAP.md:971-972`.

The figure had propagated into **three independent copies** — the manifest, the
census tool and the preflight guard — pinned together by two tests. And it was
labelled `GIB` while holding a `GB` value: 14.7 GB is 13.69 GiB.

---

## 2. What the fix is, and what it is not

**The name was wrong, not the value.** `working_cache_gib` keeps 14.7 as what it
actually is — general working space for a run — with its comment repaired. A
new `jepa_embedding_cache_gib: 55.2` carries the real figure.

Setting the old key to 55.2 instead would have made **every ordinary training run
demand 101.98 GiB free for a cache it is not building**, and
`data_manifest.yaml:46` says the three-band design exists precisely so the gate
does not *"cry wolf"*.

    pooled,      full cohort      14.70 GB =  13.69 GiB   DESIGN FORBIDS pooled-only
    pooled,      trainable rows    5.70 GB =   5.31 GiB   also pooled, also forbidden
    token-level, full cohort     154.00 GB = 143.42 GiB   the eventual requirement
    token-level, trainable        59.27 GB =  55.20 GiB   DECIDED, build this first

1,701,217 / 4,420,180 = 0.38488; 154.00 GB x 0.38488 = 59.27 GB = 55.20 GiB.
Cross-checked against the pooled pair: 5.7 / 14.7 = 0.38776, agreeing to 0.003.

### What deliberately changes meaning

`audit_disk_census.py` now answers *"can this volume hold the JEPA cache?"* and
reports **101.98 GiB** required. `preflight_data_guard.py` still answers *"may a
run start?"* and reports **61.48 GiB**. The two tools now give **different
verdicts on the same volume on the same day — correctly**.
`test_policy_matches_the_census_tool` said they must agree; its docstring now
says why they do not, and its pin follows the JEPA figure.

---

## 3. Three constraints the source imposed, each read rather than assumed

**One.** `preflight_data_guard.py:130-133` drives the manifest read from
`DEFAULT_POLICY`'s **keys**. A key in `DEFAULT_POLICY` but absent from the
manifest makes every `load()` raise `KeyError`, get caught, and fall back to
defaults — **silently changing which policy the gate enforces**. A key in the
manifest without the dataclass field makes `cls(**...)` raise `TypeError`. The
manifest entry, the `DEFAULT_POLICY` entry and the field are **one edit in three
files**.

**Two.** `test_storage_guard.py:113` splits the census line on `=` and `:119`
calls `float()`. The constant must be a **bare literal**: a trailing comment
would split to `"55.2  # ..."` and raise *inside the test*. The explanation goes
above the line, never beside it.

**Three.** `required_free_bytes` reads `working_cache_gib`, which does not move.
So `required` stays 61.48 GiB and **nothing in the run gate changes** — not the
61.48 assertions, not the ten parametrised bands, not the inclusivity checks, not
one `_at()` call.

---

## 4. Four reads, and each one found coupling the previous had missed

This took four rounds of reading before a line was written, and that was not
waste.

    read 1   found preflight_data_guard.py, a THIRD copy I did not know existed
    read 2   found test_storage_guard.py:165, a SECOND hard-coded 61.48
    read 3   found the file was at scripts/maintenance/, not scripts/
    read 4   found DEFAULT_POLICY's keys drive the manifest read

**Three claims I made between those reads were wrong**, every one of them from
reasoning ahead of the source:

* I said adding a constant to the census would break an exact-set assertion. The
  loop that builds that set only ever inserts keys from a three-name tuple, so it
  cannot.
* I said `_at(monkeypatch, 90.0)` would break. The severity does change; **the
  test does not assert severity there** — it asserts the ten-per-cent advisory,
  which fires on both OK and WARN.
* I built the path `scripts\preflight_data_guard.py` out of a bare leaf name,
  because `Select-String` prints `$_.Filename`, which is the leaf.

---

## 5. The fixture predicted every byte delta exactly

    data_manifest.yaml         +1537    fixture predicted +1537
    audit_disk_census.py       +1048    fixture predicted +1048
    preflight_data_guard.py     +405    fixture predicted  +405
    test_storage_guard.py       +740    fixture predicted  +740
    ROADMAP.md                 +1859    fixture predicted +1859

All five exact. The injected text is byte-identical between a fixture built from
the real policy section and the real repository, which is the strongest available
evidence that the anchors landed where they were meant to.

**`ROADMAP.md` at zero deletions** is the one that matters most: it confirms the
amendment **appended** and the dated 2026-07-20 measurement survives verbatim.
That is what an amendment means and a rewrite does not.

---

## 6. Defects in my own instruments

**A syntax error, caught by the first check that runs.** `\'` inside a
single-quoted Python string is an escaped backslash followed by a quote, and that
quote terminates the string. The opening parenthesis was never closed. The fix
was not a better escape; it was prose that needs none.

**A wrong prediction about the roadmap, corrected by the repository.** I asserted
`"JEPA V1 cannot cache embeddings locally"` would appear twice after the
amendment. It appears three times: the original at 1359, my amendment quoting it
at 1372, and **my own roadmap delta from two hours earlier at 3321**, which
quoted the same phrase. No duplication, no defect — I had forgotten my own
work.

---

## 7. A standing rule of mine was wrong, and Monzia corrected it

I was carrying a rule that read *"never add authorship, attribution, byline or
dedication lines to any delivered file"*. **That blanket was mine, not his.**

The real instruction had been narrow: never write *"Written FOR Monzia Moodie"*,
because it frames his own project as work done for him rather than by him. I took
that one correction and widened it into a prohibition on all forms.

Then I compounded it. I found his name in six files, matched them to my own
over-broad rule, and **queued a commit to strip his name off his own project**
without once asking whose name it was or how it got there.

The corrected rule, in his words:

* bylines should read **"Written by Monzia Moodie"** or **"Author: Monzia Moodie"**;
* never **"Written for Monzia Moodie"**;
* **stop extrapolating — always ask first.**

The removal was withdrawn. `preflight_data_guard.py:2` and the five RNA scripts
keep their bylines, which are his signature on his own work.

**Two extrapolations stacked, and the second was worse than the first.** This is
recorded here because a rule that produces a wrong commit is a project defect,
not a private one.

---

## 8. Figures

    ratchet            4120 -> 4121  (+1, predicted +1; second consecutive exact)
    README badge       4120 -> 4121  (derived; non-ascii 110, CRLF 0, LF 502, delta +0)
    test_storage_guard 46 passed  (45 before)
    full suite         4115 passed, 6 skipped, 0 failed, 766.22 s
                       4115 + 6 = 4121
    skip set           unchanged, THIRTEENTH consecutive run
    commit             7 files, 97 insertions, 10 deletions
    the run gate       61.48 GiB required, UNMOVED
    the census verdict 61.48 -> 101.98 GiB required, deliberately

---

*Written 2026-07-30. The carried-item register decides status; `tests/EXPECTED_SUITE_SIZE`
decides the count.*
