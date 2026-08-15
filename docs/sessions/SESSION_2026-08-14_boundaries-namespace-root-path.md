# SESSION 2026-08-14 -- boundaries: namespace, root and path

**Author: Monzia Moodie**

**Commits:** `a7e576f`, `8f28a23`, `d3d3dbe`, `69a9597` (four)
**Ratchet:** 4853 -> 4910
**Preceding head:** `8cb0429`

---

## What this session was

Four units, each about a boundary the codebase had lost: between importing a
namespace and activating a capability; between where a process was launched and
where its output belongs; between a source package and an artifact directory;
and between repository identity, artifact destination and mutable state.

| commit | unit | files | lines |
|---|---|---|---|
| `a7e576f` | REPORTS-EAGER-IMPORT-1 | 4 | +243 -12 |
| `8f28a23` | AGENT-ROOT-ANCHOR-1 | 9 | +457 -12 |
| `d3d3dbe` | REPORTS-DIR-IGNORED-1 | 4 | +182 -3 |
| `69a9597` | RUNTIME-PATHS-1 | 5 | +587 -2 |

---

## 1. `a7e576f` -- importing a namespace must not activate a capability

`reports/__init__.py:14` imported `report_generator` eagerly. That module
imports seaborn and jinja2 **unguarded** at lines 45 and 46, and executes
`sns.set_style()` at **module scope** at line 57 -- so merely touching the
namespace required both packages and mutated process-wide plotting
configuration.

Neither is in `requirements-api.lock`, while matplotlib -- two lines above them
in the same import block -- **is** present at line 24.

**Latent, not firing.** `python -X importtime -c "import api.main"` shows no
seaborn, jinja2, reports or report_generator in the API's import graph. One
import statement away from breaking the image, and the Docker job builds
without importing, so it would not catch it.

### Measured before changing anything

All ten consumers import the **submodule** directly. **Zero** callers use a
re-exported name. The eight-name import block served no caller while
guaranteeing the activation. Replaced with a PEP 562 module `__getattr__`.

### A performance mistake of mine

I predicted eight subprocess tests would cost about 28 seconds, consolidated
them into one probe, and made the file **slower** -- 11.17 s against the 5.75 s
it replaced. Measured: the namespace import costs 0.03 s and the full
`report_generator` import 1.91 s, a **64x difference**, and my single probe
resolved `ReportGenerator` so every property paid the expensive path.

Split by cost rather than count -- one cheap probe, one expensive -- it runs in
3.63 s. **I optimised on the wrong axis, and only measuring the result revealed
it.**

### Deliberately not in this unit

`sns.set_style()` and `plt.rcParams.update()` remain at module scope.
`ReportGenerator.__init__` creates a directory and a Jinja environment; it does
**not** call the plotting functions, so applying style there would leave a
standalone `plot_roc_curves()` caller unstyled. And `matplotlib.use("Agg")` must
stay at line 41, before pyplot is imported at 42.

---

## 2. `8f28a23` -- an output location must not depend on the launch directory

Five agents defaulted `root: str = "."`, so `Path(self._root) / "reports" / ...`
resolved against the current working directory.

`scripts/apply_data_readiness_root_fix.py` had already diagnosed and repaired
this for a sixth agent, in these terms:

> *"it defaulted root='.' so it resolved registry.critical_assets()
> (repo-relative paths) against the CURRENT WORKING DIRECTORY. When the
> orchestrator is launched from src/.../agent_layer ... every asset read as
> missing -> spurious NO_GO"*

Three generated reports sat inside the source tree as its residue.

### Measured by construction, not read from source

Each agent instantiated with a stub shared state and its `_root` printed: five
reported `'.'`, `DataReadinessAgent` reported `PROJECT_ROOT`.

### My first census found four

I searched the agents already in hand. A search across **all** of `src` and
`scripts` found `provisioning_agent.py:45` as the fifth -- it writes via
`PD.write_provisioning_doc(self._root, event)` at lines 104 and 123. Shipping
the four-agent version would have looked complete and left one agent defective.

**That is the third time in three days I sized a defect from the set in hand
rather than the set it could inhabit:** one em-dash file when there were two,
one ignored directory when there were three, four agents when there are five.

### Deliberately excluded, and recorded so

Four detector function defaults are **not** defects:
`database_freshness_detector.py:96,115` and `data_readiness_detector.py:40,94`.
Pure functions taking `root` as a parameter, with every measured caller passing
it explicitly. Anchoring them to `PROJECT_ROOT` would import agent-layer
configuration into the evaluation layer -- a dependency inversion, and a worse
defect than the one it fixes.

### Why the test exists

All eight existing test call sites pass `root=str(tmp_path)`. The docstrings
call `root` "injectable for hermetic tests" and the tests do exactly that -- so
the **default was never exercised**, and changing it would have passed the whole
suite whether the change was right or wrong.

---

## 3. `d3d3dbe` -- a source package must never be silently ignored

`.gitignore:101` read `reports/`. A pattern with no leading slash matches at
**any depth**, so it caught four directories rather than the one it was written
for -- including `src/genomic_variant_classifier/reports/`, a **source package**.

**Proven by probe:** an untracked `.py` written into that package was reported
ignored by `git check-ignore` and never appeared in `git status`. Its two files
survive only because they predate the rule.

That is the `torch_geometric` shape, and this repository has one recorded
instance of it lasting 508 continuous-integration runs.

### The tests probe behaviour, not text

A test asserting `"/reports/" in gitignore_text` would pass against rules
reordered into uselessness and fail against a correct rewrite. Each of nine
sentinel cases asks git directly via `git check-ignore --no-index`.

`--no-index` is the right instrument: it evaluates the rules **without** the
index, so a tracked file's status cannot mask the answer -- which is exactly how
the original investigation was misled.

### Three artifacts, each classified by reading it

An inference I got wrong twice before measuring:

- **`FRESHNESS_2026-06-20.md`** -- routine. Probes remote sources, so the working
  directory never affected it, and the root series jumps 06-14 to 06-30.
- **`OPS_2026-06-20.md`** -- routine. Reads heartbeats from shared state, not the
  filesystem.
- **`READINESS_2026-06-20.md`** -- **evidence.** The 22:30 copy reports no asset
  findings; the 23:37 root copy reports all eleven present. Same evening, 67
  minutes apart, opposite verdicts. Preserved as
  `READINESS_2026-06-20_2230_wrong-cwd.md` rather than deleted.

### A finding I declined to widen

`.gitignore` carries 603 non-ASCII bytes, and I nearly logged that as a new
instance of `REQFILES-NONASCII-1`. Measured: **196 box-drawing characters**
forming section separators and 5 em dashes, on 9 of 287 lines, all in comments
and none in rules -- a deliberate visual convention. And 378 tracked files carry
non-ASCII bytes, mostly parquet and docx where the question is meaningless.

**I was about to widen a finding on a byte count without asking what the bytes
were.**

---

## 4. `69a9597` -- one authority for repository, artifact and state locations

Five independent conventions existed for the same question:

```
root: str = "."                 five agents, ambient working directory
PROJECT_ROOT                    config.py, a hard-coded Windows literal
ADAPTATION_PROJECT_ROOT         adaptation_agent, its own variable
Path(__file__).parent           shared_state.py, the only correct one
Path("data/agent_state.json")   version_monitor_agent, cwd-relative
```

`GVC_PROJECT_ROOT` is set **nowhere** -- not in continuous integration, not in
the Dockerfile, not in any script, not in the shell. So `config.py:17`'s
fallback, a literal Windows path, is the value every consumer receives.

Two of its derived constants point at directories that do not exist even on the
workstation: `AUDIT_LOG_DIR` and `SHARED_STATE_PATH` resolve to `PROJECT_ROOT /
"agent_layer"`, but `agent_layer` lives under `src/genomic_variant_classifier/`.
Nothing reads either.

### Three roots, not one

`project_root` is identity; `artifact_root` is destination; `state_root` is
mutable state. Conflating the first two is `OUTPUT-ROOT-CONFLATION-1`;
conflating the first and third put state inside `src/` twice.

### Discovery verifies identity, not existence

`(candidate / "src").exists()` would be a comfort assertion. Discovery requires
three sentinels in conjunction **and** the declared name from `pyproject.toml`,
measured to be `genomic-variant-classifier`.

**And there is no fallback:** explicit argument, then `GVC_PROJECT_ROOT`, then
discovery, then **raise**.

Verified against the real filesystem before the installer was written:

```
discover from a Downloads copy : None      anchored to __file__
looks_like_project_root(repo)  : True      conjunction + name, on Windows
C:\Users\monzi                 : refused
GVC_PROJECT_ROOT=C:\Windows    : refused
```

Both refusals fired on directories that **exist**, so identity rather than
existence is doing the work.

### Two sabotage gaps in my own tests

- **R1:** the no-developer-path check tested `s[1:3] == ":\\"` only, so a
  fallback reintroduced as `"C:/Projects/..."` with a **forward slash** went
  undetected.
- **S13:** the discovery test asserted `found is None or looks_like(found)`,
  which holds whether the walk starts from `__file__` or from `"."`.

Both were assertions that could not fail, in exactly the properties the module
exists to guarantee.

### A measurement that corrected my own claim

I called the suite's 17:07 -> 24:01 increase after `AGENT-ROOT-ANCHOR-1`
"disproportionate". Measured: that file alone runs in 7.93 s, a second run in
10.22 s, and the heaviest agent import costs 1.85 s. The 24-minute run was
**workstation load**, as an earlier 22-minute anomaly was. No optimisation was
warranted.

---

## Register at close

**Closed this session:** `REPORTS-EAGER-IMPORT-1`, `AGENT-ROOT-AMBIENT-1`,
`REPORTS-DIR-IGNORED-1`; `RUNTIME-PATHS-1` established.

**Open:**

| item | state |
|---|---|
| `PROJECT-ROOT-HARDCODED-1` | `config.py:17` still holds the literal |
| `CONFIG-DEAD-PATHS-1` | two constants, four environment variables, zero readers |
| `LITERATURE-STATE-CWD-RELATIVE-1` | `version_monitor_agent.py:58` reads and writes a cwd-relative path; live in two pipelines |
| `STATE-FILE-DUPLICATES-1` | the nested copy supersedes the root copy on all five differing values |
| `OUTPUT-ROOT-CONFLATION-1` | now addressable via `artifact_root` |
| `ROOTFIX-VERIFY-TEXTUAL-1` | `if "root" not in source` is satisfied by `outputs_root` alone |
| `PREFLIGHT-TOKEN-SUBSTRING-1` | check 9 tests for the substring `GITHUB_TOKEN=`, not a usable credential |
| SESSION_2026-06-19 item 5 | `run_agents.py` still has no chdir to `PROJECT_ROOT` |

---

## An operational error, recorded

I proposed removing the user-scope `GITHUB_TOKEN` environment variable -- which
was shadowing a valid `gh` keyring credential and causing repeated
`HTTP 401: Bad credentials` -- and only **afterwards** suggested scanning for
what depended on it. `scripts/preflight_check.py` check 9 reads exactly that
location, and the value was not recoverable.

**The ordering was backwards on an irreversible action.** The same shape as
sizing a defect from the set in hand, applied to a deletion.

The follow-on finding is real: check 9 tests for the **substring**
`GITHUB_TOKEN=` in `.env`, so a literal placeholder satisfied it and reported
`True`. Its three branches disagree about what "available" means -- the
user-environment branch checks `len(token) > 10`, the `.env` branch checks
nothing.
