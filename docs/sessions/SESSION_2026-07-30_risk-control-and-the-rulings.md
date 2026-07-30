# SESSION 2026-07-30, part four — risk control, a red suite, and the rulings

**Repository:** github.com/monzia-moodie-repo-projects/genomic-variant-classifier
**Commit:** `c4f14fb`, on top of `408505b`
**Ratchet:** 3959 -> 4120 (+161), computed by collection
**Suite:** 4114 passed, 6 skipped, 0 failed, 986.58 s
**Python:** 3.12.10 in `.venv312`

---

## 1. What landed

`risk_control.py` — the first of conformal's five absent modules. Every other
module in that package bounds COVERAGE, which is the right guarantee for a
general classifier and the wrong one for a clinical screen, where the quantity
that matters is the rate at which a genuinely pathogenic variant is not flagged.
`project_metrics.txt` asks for exactly that at lines 909 and 912.

Five files, 790 insertions, 2 deletions. The conformal package goes from seven
modules to eight.

### What was measured before any of it was written

    the binomial tail vs a direct math.comb sum   1,967 comparisons, worst 2.184e-13
    three bounds, BY SIMULATION                   failure rates 0.0000 to 0.0440
                                                  at delta = 0.05, none exceeding it
    the whole procedure, end to end               3 violations in 2,000 trials
                                                  = 0.0015 against delta = 0.10

A confidence bound cannot be verified by inspection. The only honest test is to
draw from a known risk, compute the bound, and count how often it falls below
the truth.

---

## 2. One algorithmic defect, found and fixed before delivery

The first draft computed the binomial log-probability with
`np.vectorize(math.lgamma)`. **numpy's own documentation says `np.vectorize` "is
provided primarily for convenience, not for performance" and is "essentially a
for loop."** Every tail evaluation therefore made 2(k+1) Python-level calls,
every bisection made about fifty tail evaluations, and the end-to-end simulation
made 8,400 bisections. The test module took **67.05 seconds**, fifty of them in
two tests.

Replaced by a recurrence — `log_pmf[i] = log_pmf[i-1] + log((n-i+1)/i) +
log(p) - log(1-p)`, one cumulative sum, no gamma function — which is **9 times
faster at n=200, 19 times at n=1,000 and 31 times at n=5,000**, agrees with the
gamma form to 1.2e-11 over 6,410 comparisons including both degenerate values of
p, and brings the module to **20.14 seconds**. The exact-sum check and every
coverage simulation were re-run afterwards and are unchanged.

---

## 3. The suite went red, and the cause was an inconsistency in my own instructions

The three-metrics badge sync created `README.md.bak_2026-07-30_badge`. The
cleanup command for that commit filtered on `*.bak_2026-07-30_threemetrics`,
which does not match `*_badge`, so the backup survived. The risk-control badge
sync then **refused** — *"a backup already exists ... ABORT"* — and wrote
nothing. The badge stayed at 3959 while the ratchet moved to 4120.

    1 failed, 4113 passed, 6 skipped      4113 + 1 + 6 = 4120
    FAILED test_readme_claims: shields.io badge says 3959, ratchet says 4120

For the registry-commit-2 cleanup the filter had been the wildcard
`*.bak_2026-07-30_*`, which catches everything. Three commits later it was
narrowed to a specific suffix and that generality was lost.

**Every guard did its job.** The badge tool refused to clobber a backup it did
not create; `--assert-suite-size` confirmed the collection; and
`test_readme_claims` caught the stale badge with no tolerance. Nothing passed
silently. **From here the cleanup filter is always `*.bak_*`.**

---

## 4. The ratchet prediction was exact, for the first time today

**+161 predicted, +161 measured.** 159 nodes from the new module plus 2 from the
two package-export tests parametrised over the directory.

Five predictions before it were wrong. The difference is that this one was not a
hand count: the 159 was **measured** by running the module in a fixture, and the
2 was **read** from a contract stated in `conformal/__init__.py` rather than
inferred from decorators.

---

## 5. An external review, whose five criticisms are correct

A review of `risk_control.py` holds that:

* the module names one function `control_risk` and never identifies **which
  theorem**; it is Risk-Controlling Prediction Sets and should say so, with the
  method as a typed enumeration;
* the guarantee rests on the **population** risk being monotone, while the gate
  checks the **empirical** curve — a diagnostic presented as if it were the
  precondition;
* Clopper-Pearson must **mechanically refuse** a loss vector that is not exactly
  zeros and ones;
* `false_negative_risk` has three distinct estimands — all, accepted-only, and
  abstention-as-failure — and returns one without saying which;
* the **exchangeability unit** is unstated, and variants from one patient, gene,
  family or batch are not exchangeable.

Every one of those is correct. **None is fixed in this commit** and the commit
message says so.

Where the review is wrong, and it can be shown: it says the recurrence *"often
starts from (1-p)^n, which underflows for large n."* This one starts from
`n * log1p(-p)` — log space, no underflow — and the only exponentiation is a
single log-sum-exp at the end. Its **test matrix is excellent** and has not been
run: n to 10^6, p at `nextafter(0,1)` and `1-1e-12`, and the complement identity.

---

## 6. Defects in my own instruments, this session

1. **A cleanup filter narrowed from a wildcard to a suffix**, three commits
   apart, which left a backup that turned the suite red two hours later.

2. **A format specifier written into a string that is never formatted.** The
   ratchet line printed `%+d` literally. My self-check scans for arity mismatches
   in `%`-formatted literals and had nothing to match, because there was no
   formatting operation at all. The missing check now exists and found two more
   instances across twenty scripts.

3. **A path invented from a bare leaf name.** `Select-String` prints
   `$_.Filename`, the leaf; I wrote `scripts\preflight_data_guard.py` and the
   file is at `scripts\maintenance\`.

4. **A claim about a test that the test's own code disproves.** I stated that
   adding a constant to `audit_disk_census.py` would break an exact-set
   assertion. The loop that builds that set only ever inserts keys from a
   three-name tuple, so it cannot.

5. **An over-claimed blast radius.** I said `_at(monkeypatch, 90.0)` would break
   under one design. The severity does change; **the test does not assert
   severity there** — it asserts the ten-per-cent advisory, which fires on
   both OK and WARN.

6. **An unanchored digit search**, for the sixth time. `14\.7` matched
   `11,654.7`, `$14.72` five times, `(14.7%)`, `14.74 GB` and a timestamp.

---

## 7. Figures

    conformal package  7 -> 8 modules
    ratchet            3959 -> 4120  (+161, predicted exactly)
    README badge       3959 -> 4120  (derived; non-ascii 110, CRLF 0, LF 502, delta +0)
    the two files      181 passed
    full suite         4114 passed, 6 skipped, 0 failed, 986.58 s
                       4114 + 6 = 4120
    skip set           unchanged, ELEVENTH consecutive run
    commit             5 files, 790 insertions, 2 deletions
    test module        67.05 s -> 20.14 s after the recurrence

---

*Written 2026-07-30. The carried-item register decides status; `tests/EXPECTED_SUITE_SIZE`
decides the count.*
