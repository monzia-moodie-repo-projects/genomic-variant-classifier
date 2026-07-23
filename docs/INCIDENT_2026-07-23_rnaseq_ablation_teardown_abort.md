# Incident -- SIGABRT at interpreter teardown in the RNA-sequencing ablation script

**Opened** 2026-07-22. **Root cause proven** 2026-07-23. **Fixed** 2026-07-23.

## Summary

Continuous Integration run 29962715186 (run number 585, commit `821a990`) failed one
test, on Python 3.12 only. A child process of
`tests/unit/test_rnaseq_ablation_tools.py::test_full_is_unchanged` returned -6
(SIGABRT) with the standard-error line `terminate called without an active
exception`, *after* printing its own success line. The parquet had been written and
the work had completed; the process aborted during interpreter finalisation.

Re-running the identical commit passed every job, proving the fault nondeterministic.
730 local reproduction attempts at the exact pinned library versions produced zero
aborts, so the culprit could not be identified locally.

## How it was found

A dispatch-only diagnostic workflow ran nine, then fourteen, arms of 5000 child
executions each on the Continuous Integration runner -- 115,000 executions in total --
bisecting by construction and by mitigation, with core dumps captured and backtraced.

| arm | aborts / 5000 | reading |
| --- | --- | --- |
| `numpy_only` | 0 | numpy alone is not sufficient |
| `pandas_no_parquet` | 0 | importing pandas is not sufficient |
| `pyarrow_read` (`pandas.read_parquet`) | 45, then 27 | **reproduces** |
| `pyarrow_write` (`DataFrame.to_parquet`) | 0 | the write path is clean |
| `pyarrow_direct` (`pq.read_table` + `pq.write_table`) | 0 | Arrow's own path is clean |
| `to_pandas_explicit` (`pq.read_table(...).to_pandas()`) | 0 | the conversion is **not** the trigger |
| `baseline_real_script` | 1, then 0 | the real script, far rarer |

## Root cause

27 core dumps carry an identical frame chain. `PyThread_exit_thread`,
`_Unwind_ForcedUnwind` and `std::terminate` each appear exactly once per core.

```
arrow::py::PyReadableFile::~PyReadableFile()   libarrow_python.so.2300
arrow::py::OwnedRefNoGIL::~OwnedRefNoGIL()     libarrow_python.so.2300
PyGILState_Ensure()                            Python/pystate.c:2240
PyEval_RestoreThread()                         Python/ceval_gil.c:708
take_gil()                                     Python/ceval_gil.c:353
PyThread_exit_thread()                         Python/thread_pthread.h:370
__GI___pthread_exit() -> __do_cancel()
_Unwind_ForcedUnwind()                         libgcc_s.so.1
std::terminate()                               libstdc++.so.6
__GI_abort()                                   -> SIGABRT
```

`pandas.read_parquet` hands Arrow a **Python file handle**, which Arrow wraps in
`arrow::py::PyReadableFile`. That wrapper holds a Python object reference, so its
destructor calls `PyGILState_Ensure` to release it. When the destructor runs on an
Arrow background thread after interpreter finalisation has begun, CPython's `take_gil`
kills the thread with `pthread_exit`. The forced unwind propagates through C++
destructor frames that cannot survive it, so libstdc++ calls `std::terminate`.

`pq.read_table` opens the file natively in C++. No Python object is wrapped, so the
aborting destructor is never constructed.

## Two hypotheses that were wrong, and why they are recorded

**OpenBLAS.** Parked OpenBLAS worker threads appear in the thread lists on
`thread_status` futex waits, which looked incriminating. Zero OpenBLAS frames appear
on any abort path. They are bystanders.

**The Arrow-to-pandas conversion.** Predicted as the trigger before round two.
`to_pandas_explicit` returned 0/5000. Refuted.

Both are recorded because the next person will form the same two hypotheses.

## Statistical caveat, recorded deliberately

The first round's three thread-constraint arms were referenced against the real
script, which aborts at 1/5000. A constrained arm showing zero was about 37 per cent
likely even if the constraint did nothing; Fisher exact on 1/5000 against 0/5000
returns 1.000. Those arms proved nothing, and the consolidator originally said they
did. It now refuses to call suppression when fewer than three events are expected.

Separately, `read_arrow_io_threads_1` still aborted 22/5000, so the `ARROW_IO_THREADS`
environment variable is not effective, and `read_cpu_count_1` is confounded because it
changes import order as well as thread counts. Neither is used as evidence.

## Fix

`scripts/make_rnaseq_ablation_parquet.py` now reads through
`pyarrow.parquet.read_table(...).to_pandas()`. This removes the faulting object rather
than suppressing its symptom. It is not a retry, not a loosened assertion, not
`os._exit`, and not a thread-count workaround.

`tests/unit/test_rnaseq_ablation_native_read.py` pins it: the syntax tree is walked to
prove `pandas.read_parquet` is not called, both readers are compared frame to frame to
prove the data is unchanged, and all four modes are run end to end. Reintroducing
`pandas.read_parquet` turns two tests red.

## Open

The same class of fault can reach any code passing a Python file handle to Arrow. The
six other test files that spawn subprocesses share the exposure. A repository-wide
audit of Python-handle reads into Arrow is not yet done.
