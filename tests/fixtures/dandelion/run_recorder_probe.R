# Fixture for tests/unit/test_inference_backend_trace.py -- ONE scenario per FRESH R process (owner ruling 2026-10-04b).
# Usage: Rscript run_recorder_probe.R <recorder.R> <scenario> <trace dir> <result file>
#   off     -- no recording; writes the scientific outputs
#   on      -- recording; writes the scientific outputs (compared byte-for-byte with "off" by the test)
#   fork    -- recording, then two forked workers (non-Windows); writes how many worker calls were refused
#   badpath -- recorder_start on an unusable destination; writes the refusal message
# Outputs are exact hexadecimal values written through a BINARY connection (identical bytes on every platform).
args <- commandArgs(trailingOnly = TRUE)
source(args[1])
scenario <- args[2]; trace_dir <- args[3]; result <- args[4]
write_bin <- function(lines, path) { con <- file(path, open = "wb"); on.exit(close(con)); writeLines(lines, con, sep = "\n", useBytes = TRUE) }
inputs <- list(small = c(0.01, 0.2, 0.5), few = rep(c(0.1, 0.2, 0.3), 4), many = seq(0.001, 0.9, length.out = 25))
science <- function() unlist(lapply(names(inputs), function(n) c(paste0("# ", n), sprintf("%a", DANDELION:::safe_qvalues(inputs[[n]])))))
if (scenario == "off") {
  write_bin(science(), result)
} else if (scenario == "on") {
  recorder_start(trace_dir, "f471153bfa3c0069cd68a67565000889c7cdf5d1")
  out <- science(); recorder_stop(); write_bin(out, result)
} else if (scenario == "fork") {
  recorder_start(trace_dir, "f471153bfa3c0069cd68a67565000889c7cdf5d1")
  res <- suppressWarnings(parallel::mclapply(1:2, function(i) DANDELION:::safe_qvalues(inputs$many), mc.cores = 2))
  recorder_stop()
  refused <- sum(vapply(res, function(r) inherits(r, "try-error") && grepl("forked worker", conditionMessage(attr(r, "condition"))), logical(1)))
  write_bin(c(paste("refused", refused), paste("files", length(list.files(trace_dir, all.files = TRUE, no.. = TRUE)))), result)
} else if (scenario == "badpath") {
  msg <- tryCatch({ recorder_start(trace_dir, "f471153bfa3c0069cd68a67565000889c7cdf5d1"); "ACCEPTED" }, error = function(e) conditionMessage(e))
  traced <- inherits(get("safe_qvalues", envir = asNamespace("DANDELION")), "functionWithTrace")
  write_bin(c(msg, paste("traced", traced)), result)
} else stop("unknown scenario: ", scenario)
