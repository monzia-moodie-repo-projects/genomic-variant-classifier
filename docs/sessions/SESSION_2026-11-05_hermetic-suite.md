# Session 2026-11-05 -- TESTS-REACH-THE-NETWORK-1 and GIT-VERSION-REQUIREMENT-UNSTATED-1: no test reaches the network

(Revised the same day after the owner's first Windows validation failed: PROGRAM-START-RECORD-PLATFORM-FORM-1 and
GUARD-SYSTEM-PROXIES-1, in the section "Revision B" below.)

Author: Monzia Moodie

The session continued from set "ag" (WARNING-FILTER-REORDER-AT-FIRST-IMPORT-1, delivered 03:20 Coordinated Universal Time (UTC),
awaiting the owner's run) and took the two findings that work recorded next: tests whose outcome could depend on the network, and a
Git requirement that only Git's own usage text revealed.

## What was measured

The instrument (outside the repository; its files and digests are in the evidence bundle): an audit hook (PEP 578,
`sys.addaudithook`) installed in every Python process by a `sitecustomize.py` on PYTHONPATH -- the system's own sitecustomize is run
after it, so nothing else changes -- appending one JSON line per event with a single O_APPEND write as it happens (a container restart
keeps everything up to the cut): every name lookup, every Internet-family connection (a connection to the HTTPS proxy counted as
network, its real destination read from the calling frames' `_tunnel_host`), and every program started (subprocess.Popen, os.system,
os.posix_spawn, os.exec) with whether the program inherits the measurement. A pytest plugin keeps the phase and test node identifier
in a context file every measured process reads, and records each test's start and end. The HTTPS proxy's status (its refusals,
with host and time) was sampled every 5 seconds. Self-test before use, six planted cases: a lookup; a proxied HTTPS request
(destination huggingface.co read from the frames, the proxy's refusal in the same second); a measured child's connection; a child
with an emptied environment and a child started with -I, each named as unmeasured.

The run (2026-11-05 06:41:28 to 07:04:46 UTC; the validation clone of set "ag", tree 6621b6ba; its training-and-test candidate
environment; Git 2.47.1; the gate's options and its HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1): 8,635 passed, 26 skipped, 6
xfailed in 1,358 s -- the outcome without the instrument. Every test started and ended (8,667 of 8,667); 20,489 events; no hook
error.

- 15 tests reached the network: 28 connections through the proxy and 4 name lookups, every one refused there and every refusal
  swallowed by the code that made it.

| Tests | Destinations | Where in the project |
|---|---|---|
| 10 of tests/unit/test_core.py::TestAnnotationPipeline (1 each) | rest.uniprot.org | data/esm2.py:193 `_fetch_uniprot_sequence`, from `ESM2Connector._get_sequence` |
| test_esm2_activation.py::test_esm2_emits_delta_norm_column (3) | rest.uniprot.org | the same |
| test_orchestrator_agent_isolation, test_orchestrator_run_telemetry (dry run), test_science_claw_orchestrator_gate (4 each) | ftp.ncbi.nlm.nih.gov, storage.googleapis.com (2), databases.lovd.nl | agent_layer/agents/data_freshness_agent.py `_poll_clinvar`, `_poll_gnomad`, `_poll_lovd`, `_poll_alphamissense` |
| test_orchestrator_lazy_registry.py::test_ci_data_freshness_pipeline_runs_without_sklearn, in a child interpreter (7) | ftp.ncbi.nlm.nih.gov, storage.googleapis.com (3), alphafold.ebi.ac.uk, api.github.com, databases.lovd.nl | agent_layer/agents/database_freshness_detector.py `_default_probe` |

- Collection started `/usr/bin/Rscript` five times without --vanilla (06:42:44 to 06:43:19 UTC), from
  tests/unit/test_inference_backend_trace.py's five skip conditions, in the checkout. Measured alone at 07:05 UTC: one such start took
  7.72 s and was refused 24 connections to bioconductor.org and packagemanager.posit.co ("failed to bootstrap renv: the project will
  not be loaded"); with --vanilla, 0.18 s; from a folder outside the checkout, 0.22 s -- neither reached anything, and all three saw
  the same R libraries (/usr/local/lib/R/site-library, /usr/lib/R/site-library, /usr/lib/R/library). The run's other 30 starts of
  /usr/bin/Rscript, and 11 of tests' stand-in Rscripts, used --vanilla.
- Nothing else reached the network. 2,966 program starts were recorded; 4 Python children ran without the measurement in their
  environment and 8 with -I. Every proxy refusal of the run is one of the 28 measured connections or falls within the five R starts
  at collection.

Two errors of mine, recorded. The first run used the sandbox's Git 2.43.0 (the suite needs 2.45 or later): stopped by process
identifier after 1.5 minutes, before any test ran; the run script now refuses to start unless Git 2.47.1 is first on PATH. During the
run I ran other pytest sessions: the events are unaffected (only processes carrying the measurement write them), but three bursts of
R refusals between 06:44:56 and 06:47:38 UTC came from my concurrent full collections, not the measured run (which started no R
then); they are excluded. Later, a verification run was stopped 34 seconds in, because I edited README.md in the tree it measured;
the documentation was then finished before the run that counts. The rule: nothing that could reach the network, and no edit of the
measured tree, while a measurement runs. The proxy's log holds 20 entries and was sampled every 5 seconds: refusal counts from R are
lower bounds.

## The change

- **The network guard** (conftest.py, the repository root's, so both suites). The three Hugging Face offline variables are 1 for the
  session (set when the file is imported -- huggingface_hub reads them at its own import; an invoker's other value is named in the
  summary; restored at unconfigure). An audit hook, installed when the file is imported, refuses inside the call every name lookup of
  a host that is not the loopback and every connection or datagram to a non-loopback Internet address or to a proxy on the loopback
  (in revision B: read from every <scheme>_proxy variable and the operating system's own settings), raising NetworkRefused (a
  ConnectionRefusedError). A test that tried fails at teardown, also
  when its code swallowed the refusal; the finding names each attempt (event, target, the proxied destination, the phase, the test,
  the project's frames). An attempt while the session starts or tests are collected stops the session. The session fails whenever any
  test was found -- measured: pytest records the teardown failure of an xfail-marked test as a second xfail. A "network guard" section
  in the terminal summary. Not covered, stated: child processes (they inherit the offline variables) and programs that are not Python.
- **The measured sites.** TestAnnotationPipeline's configs say `esm2_allow_network=False`, a new AnnotationConfig field passed to
  `ESM2Connector(allow_network=...)` (None keeps the connector's default). The ESM-2 activation tests take sequences only from the
  local UniProt index (data/external/uniprot/uniprot_human_reviewed.parquet), the real-mode test skipping with its reason when the
  index is absent (after the weights check, which still xfails first), and the unused `_has_network()` (a live connection to
  rest.uniprot.org) is removed. `offline_freshness_sources` (tests/unit/conftest.py) replaces the four polls with "no change" and the
  registry probe with "unreachable"; the three orchestrator tests request it, and the dry-run test also requires that all four polls
  ran. The lazy-registry child replaces the detector's probe before the pipeline builds the agent, prints how many sources it probed,
  and the test requires the report to carry the offline answer. test_inference_backend_trace.py probes DANDELION once per session,
  when the first test needing it runs, from a folder outside the checkout, and runs every R process outside it -- which closes
  TEST-R-PROJECT-PROFILE-1 for that module. The choice of library is unchanged where measured; it remains the R on PATH with the
  user's own profile, not the project's renv library, whose qualification is separate work.
- **The Git requirement.** `release_approval.require_git_version()` reads `git --version` once, when a reader is created, with the
  reader's environment, and refuses below 2.45 or on unreadable output with the requirement named.

## Verification (the first delivery, superseded by revision B below)

- With the guard and before the fixes, the measured offenders: 14 of the 15 failed at teardown, each naming its destination and
  frames (the 15th is the child, outside the guard's reach by design, fixed in the test).
- The changed test files and the other guard test files: 421 passed, 8 skipped, 1 xfailed; the network guard 430 checked, 0 tried.
- Git: with Git 2.43.0 first on PATH, the same 8 tests fail, now with "Git reader requires Git 2.45 or later (--no-lazy-fetch);
  found 'git version 2.43.0'" (before: "Git reader refused (exit 129): unknown option: --no-lazy-fetch"); with 2.47.1, 135 passed in
  the two files.
- Node identities: +45 / -0 against 90e83128; 8,712 collected.
- Mutations, each caught (14):
  * the Git check removed (18 tests fail), the minimum lowered to 2.44 (10), no boundary after the minor number (1), the version read
    without the reader's environment (1);
  * the audit hook never installed (3 child-session tests fail); the loopback treated as network (9 fail, 1 error); the proxy not
    recognised (1); no teardown finding (2); no collection check (1); the offline variables not set (2); the session not failed by
    the network tally (1: the xfail-alone child exits 0);
  * the DANDELION probe evaluated at import again (1); the probe run in the checkout (1); the offline fixture dropped from the
    dry-run test (the guard fails it at teardown).
- The whole suite under the same measurement, on the changed tree (07:28:41 to 07:50 UTC; nothing else ran and the tree was not
  edited): 8,680 passed, 26 skipped, 6 xfailed, 5 subtests passed in 1,236.80 s, coverage 84%; the network guard 8,712 checked, 0
  tried, 0 attempts refused; the warning-filter guard 8,712 checked, 0 changed; the working tree unchanged. Every test started and
  ended (8,712). The instrument recorded 6 network attempts in all, every one a planted attempt inside the guard's own child
  sessions (tests/unit/test_network_guard.py), refused there; the HTTPS proxy recorded 0 refusals in the whole run (250 samples).
  No R started at collection. Against the measured run before the change: the same 8,635 passed, plus the 45 new ones.
- Forecast for the owner's validation on Windows: the same, except test_esm2_activation.py::test_esm2_not_in_stub_mode, which
  passed there on 2026-11-05 after a live UniProt lookup (the weights are cached on that machine) and will skip, because a fresh
  validation clone has no local UniProt index.

## Revision B (2026-11-05, after the owner's first Windows validation)

**What the owner's run measured.** Set "ah"'s V1 (2026-11-05, from 08:42 UTC; the pasted block equal to its file by SHA-256) ran the
dry run and -Validate. Every stage passed through the tests' temporary root; the targeted tests gave 1 failed, 883 passed, 14
skipped in 432.35 s, with the network guard reporting 898 tests checked and 0 that tried, the warning-filter guard 898 and 0, and the
working-tree guard unchanged. Preservation of the checkout: VERIFIED UNCHANGED; nothing was committed or published. The 14 skips are
the forecast one (test_esm2_not_in_stub_mode: no local UniProt index), eight "Rscript not on PATH", and five that need the right to
create symbolic links.

**The failure (PROGRAM-START-RECORD-PLATFORM-FORM-1).** test_the_dandelion_probe_starts_r_once_outside_the_checkout read the command of
a "subprocess.Popen" audit event as a list of arguments. That is CPython's form on POSIX. On Windows, `_execute_child` joins the list
with `subprocess.list2cmdline` and passes `executable=None` before it raises the event (Lib/subprocess.py, read for 3.11, 3.12 and
3.13). Ground truth: `list2cmdline` of the planted Rscript.BAT path, "-e" and the probe's expression equals, byte for byte, the command
line the owner's run recorded. The code under test was right; the test was wrong, and it had been verified only on Linux -- the error
class of this revision. The same review found a second defect: the test module installed its audit hook at import, so its own import
check, which executes the module a second time, left an inert second hook in the process.

**The change.** The recorder moved to tests/unit/conftest.py and is installed once. While a test holds the `program_starts` fixture
it records -- and by default blocks -- every Python-level program start (subprocess.Popen, os.system, os.exec, os.spawn,
os.posix_spawn, os.startfile, os.fork, os.forkpty) and every audit hook added. A subprocess.Popen record keeps what the platform raised;
`program_starts.subprocess_start(argv, cwd)` predicts it. tests/unit/test_program_start_recorder.py compares that prediction with a
REAL start on whichever machine runs the suite, so the owner's run checks the Windows form on Windows. The Windows prediction (os.name
forced to "nt") also equals the record from the owner's run. Measured and stated as not covered: `_posixsubprocess.fork_exec`, which
raises no audit event (CPython 3.11, 3.12 and 3.13), and `_winapi.CreateProcess` called directly.

**GUARD-SYSTEM-PROXIES-1**, found while re-examining the guard for Windows. It read six proxy variables. urllib.request.getproxies()
-- which requests, httpx and aiohttp use (read in the training-and-test lock's environment) -- also reads every other <scheme>_proxy
variable and, when none is set, the Windows Internet settings or macOS's System Configuration. A loopback proxy configured there would
have been let through. The guard now reads all of them, keeps only proxies on the loopback (any other address is refused anyway),
defaults a missing port by the scheme, never keeps credentials, and names each proxy and where it was read in its summary.

**Verification of revision B.**
- The R module, the recorder's tests and the guard tests: 52 passed, 8 skipped (the DANDELION package is not installed here).
- Node identities against main (3c01564a): +57 / -0; 8,724 collected.
- Mutations, each caught (14): the POSIX prediction made the Windows command line; a start blocked even when let through; an added
  audit hook not recorded; the command kept as a tuple; the test module adding an audit hook at import; the DANDELION probe run in
  the current folder; os.system not recorded; a proxy without a port ignored; the system settings not read; only six variables read;
  the proxies not named in the summary; a connection to a loopback proxy let through; no_proxy read as a proxy; credentials kept as
  the host.
- The whole suite under the measurement (10:20:30 to 10:42:41 UTC, commit aeac5948, tree 0a164a25; nothing else ran and the
  measured clone was not edited): 8,692 passed, 26 skipped, 6 xfailed, 5 subtests passed in 1,304.38 s, coverage 84%; the network
  guard 8,724 checked, 0 tried, "proxies on the loopback, refused as the network: 127.0.0.1:35717 (the environment)"; the
  warning-filter guard 8,724 checked, 0 changed; the working tree unchanged; every test started and ended. The instrument recorded
  the guard tests' 6 planted attempts, in their child sessions, and nothing else; the proxy recorded 0 refusals in 263 samples. The
  clone's ignored entries after the run are the same nine as after the first delivery's run (the gate's two coverage files, pytest's
  cache, and PYCACHE-FROM-ISOLATED-CHILDREN-1's three files). Against the first delivery's run: the same 8,680 passed, plus the 12
  new tests that pass here. Drift suite: 22 passed.
- The Windows prediction: `program_starts.subprocess_start` with os.name forced to "nt", given the paths of the owner's run, equals
  the command line that run recorded, and the recorder, given that event as CPython raises it on Windows, records exactly the
  prediction. What only Windows can prove -- the four calibration tests against a real start -- is proved by the owner's run.
- Forecast for the owner's validation on Windows: as the first delivery's targeted stage measured, with the failing test passing and
  the four calibration tests passing; test_esm2_not_in_stub_mode skips; eight R tests skip (Rscript is not on that machine's PATH).

## Open, recorded

- ESM2-REAL-MODE-FIXTURE-1: test_esm2_not_in_stub_mode needs the canonical sequences of BRCA1, TP53 and PTEN. Committing them as a
  fixture (from the owner's local UniProt index, with accession, UniProt release and SHA-256) lets it run, offline, wherever the
  model weights are cached -- as it did, online, on the owner's machine.
- PYCACHE-FROM-ISOLATED-CHILDREN-1: a child interpreter started with -I ignores PYTHONDONTWRITEBYTECODE and writes __pycache__
  into the checkout (measured: src/genomic_variant_classifier/__pycache__ and deps/__pycache__ after a full run). The working-tree
  guard and the installers exclude __pycache__ by declaration; the write itself is unaddressed.
- ESM2-LIVE-UNIPROT-DEFAULT-1: in production, ESM2Connector reaches UniProt by default when no local index is configured and logs a
  failure at DEBUG level: a variant's esm2_delta_norm can be 0.0 because the network failed, with nothing above DEBUG. Part of
  SILENT-FALLBACKS-1; the default is a production decision.
- FRESHNESS-DRY-RUN-1: DataFreshnessAgent's dry run polls the network and writes SharedState; a failed poll is recorded as
  "last_checked".
- AGENT-LOGGING-IN-LIBRARY-1: agent_layer/agents/data_freshness_agent.py configures its own handler and level and sets
  propagate=False (against the convention: no logging configuration in library modules), and its formatter writes local time
  labelled "Z" (measured: "2026-11-05T02:13:28Z" printed at 07:13 UTC on a machine five hours behind UTC).
