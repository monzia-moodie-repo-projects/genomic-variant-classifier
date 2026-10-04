# dandelion_backend_recorder.R -- ACTUAL-CALL trace of DANDELION's q-value backend (owner rulings 2026-10-03b / 2026-10-04).
# Author: Monzia Moodie
#
# WHAT IT OBSERVES: which branch each REAL call of the installed DANDELION::safe_qvalues took. It instruments the installed
# RUNTIME with trace(); the pinned source checkout is not modified -- but the runtime IS instrumented, and every event says so.
# It never replays the computation: a branch is identified from what the call itself did.
#
#   installed safe_qvalues (measured, R's indexing): step 2 clamp; step 3 small/few-distinct -> BH; step 4 qvalue present ->
#   tryCatch(qvalue, error = BH, warning = BH); step 5 BH (qvalue absent). The tracer runs before steps 2..5 in order, so the
#   number of tracer hits identifies the last step REACHED.
#
# PER CALL: <dir>/call-NNNN.input.txt, .clamped.txt, .output.txt (exact hexadecimal floats, one per line, "%a" -- lossless),
# and one line in <dir>/events.jsonl. SHA-256 digests are computed from those exact files afterwards (base R has no SHA-256).
# Warning/error class is recorded when observed through base::warning / base::stop inside qvalue; a condition raised another
# way (for example from compiled code) is recorded as "unobserved", never guessed.

RECORDER_VERSION <- "gvc.dandelion-backend-recorder/1"
.rec <- new.env(parent = emptyenv())

.json_str <- function(x) {
  if (is.null(x) || length(x) == 0 || is.na(x)) return("null")
  x <- gsub("\\\\", "\\\\\\\\", as.character(x)); x <- gsub("\"", "\\\\\"", x)
  paste0("\"", x, "\"")
}

# BINARY connections only: on Windows, R's TEXT-mode connections translate "\n" into "\r\n", which would change the exact bytes
# (and every digest) and make the trace unreadable to backend_trace.py. "wb"/"ab" are never translated on any platform.
.write_values <- function(x, path) {
  con <- file(path, open = "wb"); on.exit(close(con))
  writeLines(sprintf("%a", as.numeric(x)), con, sep = "\n", useBytes = TRUE)
}

.append_event <- function(line, path) {
  con <- file(path, open = "ab"); on.exit(close(con))
  writeLines(line, con, sep = "\n", useBytes = TRUE)
}

recorder_start <- function(dir, method_commit) {
  stopifnot(is.character(dir), length(dir) == 1, is.character(method_commit), grepl("^[0-9a-f]{40}$", method_commit))
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  if (length(list.files(dir)) > 0) stop("recorder directory is not empty -- events are never appended to an old run: ", dir)
  .rec$dir <- dir; .rec$commit <- method_commit; .rec$n <- 0L; .rec$cur <- NULL
  ns <- asNamespace("DANDELION")
  trace("safe_qvalues", where = ns, print = FALSE, at = 2:5,
        tracer = quote(.dandelion_recorder_step(environment())),
        exit = quote(.dandelion_recorder_exit(returnValue(default = quote(.abnormal_exit)))))
  trace("p.adjust", where = asNamespace("stats"), print = FALSE,
        tracer = quote(.dandelion_recorder_padjust(method)))
  if (requireNamespace("qvalue", quietly = TRUE)) {
    trace("qvalue", where = asNamespace("qvalue"), print = FALSE,
          tracer = quote(.dandelion_recorder_qvalue_enter()),
          exit = quote(.dandelion_recorder_qvalue_exit(returnValue(default = quote(.abnormal_exit)))))
    for (fn in c("warning", "stop")) {
      trace(fn, where = baseenv(), print = FALSE,
            tracer = substitute(.dandelion_recorder_condition(K), list(K = if (fn == "warning") "warning" else "error")))
    }
  }
  invisible(TRUE)
}

recorder_stop <- function() {
  for (spec in list(list("safe_qvalues", asNamespace("DANDELION")), list("p.adjust", asNamespace("stats")))) {
    suppressMessages(untrace(spec[[1]], where = spec[[2]]))
  }
  if (requireNamespace("qvalue", quietly = TRUE)) {
    suppressMessages(untrace("qvalue", where = asNamespace("qvalue")))
    for (fn in c("warning", "stop")) suppressMessages(untrace(fn, where = baseenv()))
  }
  invisible(.rec$n)
}

.dandelion_recorder_padjust <- function(method) {
  # ATTRIBUTION BY OBSERVED CONTEXT: a p.adjust call made while qvalue is still executing (entered, not yet exited) belongs to
  # qvalue's internals, not to safe_qvalues' fallback -- whatever the real qvalue package does internally.
  cur <- .rec$cur
  if (is.null(cur)) return(invisible(NULL))
  if (isTRUE(cur$qvalue_entered) && is.na(cur$qvalue_normal_exit)) {
    .rec$cur$p_adjust_inside_qvalue <- c(cur$p_adjust_inside_qvalue, as.character(method)[1])
  } else {
    .rec$cur$p_adjust <- c(cur$p_adjust, as.character(method)[1])
  }
}

.dandelion_recorder_exposure <- function() {
  # OBSERVED, not assumed by frame depth: the innermost frame on the actual call stack that holds `exposure.id`
  # (run_dandelion_for_exposure's own frame). NA when safe_qvalues is called some other way.
  frames <- sys.frames()
  for (i in rev(seq_along(frames))) {
    if (exists("exposure.id", envir = frames[[i]], inherits = FALSE)) return(as.character(get("exposure.id", envir = frames[[i]]))[1])
  }
  NA_character_
}

.dandelion_recorder_step <- function(frame) {
  cur <- .rec$cur
  if (is.null(cur)) {                                   # first tracer hit of a call = before step 2
    .rec$n <- .rec$n + 1L
    cur <- list(id = sprintf("call-%04d", .rec$n), hits = 0L, p_adjust = character(0), qvalue_entered = FALSE,
                qvalue_normal_exit = NA, conditions = character(0),
                exposure = .dandelion_recorder_exposure())
    .write_values(get("p", envir = frame), file.path(.rec$dir, paste0(cur$id, ".input.txt")))
  }
  cur$hits <- cur$hits + 1L
  if (cur$hits == 2L) {                                 # before step 3: p is the ACTUAL post-clamp vector
    p <- get("p", envir = frame)
    .write_values(p, file.path(.rec$dir, paste0(cur$id, ".clamped.txt")))
    cur$n_values <- length(p); cur$n_distinct <- length(unique(p))
  }
  .rec$cur <- cur
}

.dandelion_recorder_qvalue_enter <- function() { if (!is.null(.rec$cur)) .rec$cur$qvalue_entered <- TRUE }

.dandelion_recorder_qvalue_exit <- function(value) {
  if (!is.null(.rec$cur)) .rec$cur$qvalue_normal_exit <- !identical(value, quote(.abnormal_exit))
}

.dandelion_recorder_condition <- function(kind) {
  cur <- .rec$cur
  if (!is.null(cur) && isTRUE(cur$qvalue_entered) && is.na(cur$qvalue_normal_exit)) .rec$cur$conditions <- c(cur$conditions, kind)
}

.dandelion_recorder_exit <- function(value) {
  cur <- .rec$cur; .rec$cur <- NULL
  if (is.null(cur)) return(invisible(NULL))
  last_step <- cur$hits + 1L
  normal <- !identical(value, quote(.abnormal_exit))
  if (normal) .write_values(value, file.path(.rec$dir, paste0(cur$id, ".output.txt")))
  bh <- "BH" %in% cur$p_adjust
  if (!normal) {
    backend <- "none"; reason <- "safe_qvalues_abnormal_exit"
  } else if (last_step == 3L && bh) {
    backend <- "BH"
    reason <- if (isTRUE(cur$n_values < 10)) "fewer_than_10_values" else "fewer_than_4_distinct_values"
  } else if (last_step == 5L && bh) {
    backend <- "BH"; reason <- "qvalue_not_installed"
  } else if (last_step == 4L && cur$qvalue_entered && isTRUE(cur$qvalue_normal_exit) && !bh) {
    backend <- "qvalue"; reason <- NA_character_
  } else if (last_step == 4L && cur$qvalue_entered && identical(cur$qvalue_normal_exit, FALSE) && bh) {
    backend <- "BH"
    kinds <- unique(cur$conditions)
    reason <- if (length(kinds) == 1) paste0("qvalue_", kinds) else if (length(kinds) == 0) "qvalue_abnormal_exit_class_unobserved" else "qvalue_abnormal_exit_multiple_conditions"
  } else {
    backend <- "unclassified"; reason <- "observation_inconsistent_with_measured_branches"
  }
  line <- paste0("{", paste(c(
    paste0("\"call\":", .json_str(cur$id)), paste0("\"exposure_id\":", .json_str(cur$exposure)),
    paste0("\"backend\":", .json_str(backend)), paste0("\"fallback_reason\":", .json_str(reason)),
    paste0("\"last_step_reached\":", last_step), paste0("\"n_values\":", if (is.null(cur$n_values)) "null" else cur$n_values),
    paste0("\"n_distinct\":", if (is.null(cur$n_distinct)) "null" else cur$n_distinct),
    paste0("\"qvalue_entered\":", tolower(cur$qvalue_entered)),
    paste0("\"p_adjust_inside_qvalue\":", length(cur$p_adjust_inside_qvalue)),
    paste0("\"observation_kind\":", .json_str("actual_call_trace")), paste0("\"runtime_instrumented\":true"),
    paste0("\"method_commit\":", .json_str(.rec$commit)), paste0("\"recorder_version\":", .json_str(RECORDER_VERSION))
  ), collapse = ","), "}")
  .append_event(line, file.path(.rec$dir, "events.jsonl"))
}
