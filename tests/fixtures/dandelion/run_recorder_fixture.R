# Fixture runner for tests/unit/test_inference_backend_trace.py: the REAL installed DANDELION::safe_qvalues under the recorder.
# Usage: Rscript run_recorder_fixture.R <recorder.R> <output dir>. qvalue behaviour comes from the library path (absent, or the
# test double under GVC_QVALUE_DOUBLE_MODE). Calls resolve DANDELION:::safe_qvalues AT EACH CALL -- a reference captured before
# trace() would hold the untraced function and silently bypass the recorder (measured 2026-10-04).
args <- commandArgs(trailingOnly = TRUE)
source(args[1])
sq <- function(p) DANDELION:::safe_qvalues(p)
caller <- function(exposure.id, p) sq(p)
small <- c(0.01, 0.2, 0.5); few <- rep(c(0.1, 0.2, 0.3), 4); many <- seq(0.001, 0.9, length.out = 25)
untraced <- list(sq(small), sq(few), sq(many))
recorder_start(args[2], "f471153bfa3c0069cd68a67565000889c7cdf5d1")
traced <- list(caller("rs_small", small), caller("rs_few", few), caller("rs_many", many))
recorder_stop()
if (!identical(traced, untraced)) stop("traced output differs from untraced output")
if (inherits(get("safe_qvalues", envir = asNamespace("DANDELION")), "functionWithTrace")) stop("runtime not restored")
cat("fixture OK\n")
