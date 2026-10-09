# dandelion_exposure_recorder.R -- ACTUAL-CALL trace of DANDELION's per-exposure outcome (owner ruling 2026-10-08g).
# Author: Monzia Moodie
#
# WHY: run_dandelion_for_exposure returns NULL for FOUR different reasons and med_gene keeps no record of which. The ruling separates
# structural ineligibility (decidable before execution) from MIXTURE-ESTIMATION FAILURE (data-dependent, after execution): the latter
# must never be relabelled structural, and its estimated quantities must be preserved -- including invalid values.
#
# WHAT IT OBSERVES (installed DANDELION 0.1.0, body positions MEASURED with as.list(body(run_dandelion_for_exposure)), position 1 = `{`):
#   at 2  before `if (length(gene.trans) == 0) return(NULL)`       -> n_trans
#   at 9  before `if (length(gene.trans) < 2) return(NULL)`         -> n_valid (gene.trans after the missingness filter)
#   at 18 before `if (any(is.na(c(pi0a, pi0b))) || pi0a < 0 || ...)` -> pi0a, pi0b AS ESTIMATED (before min(., 1)), and (version 2,
#                                                                       owner ruling 2026-10-09) the EFFECTIVE BURDEN INPUT p_b: the
#                                                                       ordered gene identifiers names(p_b) and the exact post-clamp values
#   at 26 before `if (is.na(wg.sum) || wg.sum <= 0) return(NULL)`    -> wg1, wg2, wg3, wg.sum
#   exit: whether a result list was returned. The number of tracer hits identifies the last guard REACHED, so the outcome is read from
#   what the call did, never replayed. The positions are checked against the installed body before tracing (refuse on any mismatch).
# PER EXPOSURE: one line in <dir>/exposures.jsonl (created empty at start); numbers as exact hexadecimal binary64 strings ("%a"),
# NA as "NA", a quantity the call never reached as JSON null. Run DANDELION with n.cores = 1 (a forked worker is refused).
# BURDEN INPUTS (version 2): one file <dir>/burden-NNNN.tsv per DISTINCT effective burden input ("gene<TAB>%a value" lines, numbered in
# order of first occurrence); the exposure line names its file in "burden_input" (null when the pi0 guard was never reached). Distinct
# means not identical() with num.eq = FALSE -- bit-for-bit values AND names; a cheap bucket key only narrows the comparison and never
# decides it. Exposures sharing an input therefore share a file, which is what lets a reader see which exposures face the SAME
# burden-side estimation problem (inference/exposure_outcomes.py: burden_input_summary). Nothing here changes the computation.

EXPOSURE_RECORDER_VERSION <- "gvc.dandelion-exposure-recorder/2"   # 2: the effective burden input at the pi0 guard (ruling 2026-10-09)
.xrec <- new.env(parent = emptyenv())
.EXPECTED_GUARDS <- list(
  `2` = "if (length(gene.trans) == 0) {", `9` = "if (length(gene.trans) < 2) {",
  `18` = "if (any(is.na(c(pi0a, pi0b))) || pi0a < 0 || pi0b < 0) {", `26` = "if (is.na(wg.sum) || wg.sum <= 0) {")
# Version 2: the statements that MAKE the recorded burden input what inference/exposure_outcomes.py (BURDEN_PREPROCESSING,
# BURDEN_ESTIMATOR) says it is -- measured in the installed body 2026-10-09 and checked, like the guards, before any tracing.
.EXPECTED_BURDEN_STEPS <- list(
  `4` = "p_b <- p.wes.new[gene.trans]", `5` = "valid <- !is.na(p_a) & !is.na(p_b)", `7` = "p_b <- p_b[valid]",
  `8` = "gene.trans <- gene.trans[valid]", `11` = "p_b <- clamp_p(p_b)", `13` = "names(p_b) <- gene.trans",
  `15` = "Z_b <- stats::qnorm(p_b, lower.tail = FALSE)", `17` = "pi0b <- 1 - nonnullPropEst(Z_b, 0, 1)")

.xhex <- function(x) if (length(x) != 1) "\"invalid_length\"" else if (is.na(x)) "\"NA\"" else paste0("\"", sprintf("%a", as.numeric(x)), "\"")
.xstr <- function(x) { x <- gsub("\\\\", "\\\\\\\\", as.character(x)); paste0("\"", gsub("\"", "\\\\\"", x), "\"") }

exposure_recorder_start <- function(dir) {
  fn <- get("run_dandelion_for_exposure", envir = asNamespace("DANDELION"))
  body_list <- as.list(body(fn))
  expected <- c(.EXPECTED_GUARDS, .EXPECTED_BURDEN_STEPS)
  for (at in names(expected)) {
    seen <- deparse(body_list[[as.integer(at)]])[1]
    if (!identical(trimws(seen), expected[[at]])) stop("run_dandelion_for_exposure body differs at position ", at, ": ", seen,
                                                       " -- refusing to trace an unmeasured implementation")
  }
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  if (!dir.exists(dir)) stop("exposure recorder destination could not be created: ", dir)
  if (length(list.files(dir, all.files = TRUE, no.. = TRUE)) > 0) stop("exposure recorder directory is not empty: ", dir)
  probe <- file.path(dir, ".probe")
  ok <- tryCatch({ con <- file(probe, open = "wb"); writeBin(as.raw(1:3), con); close(con); identical(readBin(probe, "raw", 3L), as.raw(1:3)) },
                 warning = function(w) FALSE, error = function(e) FALSE)
  if (!isTRUE(ok) || !isTRUE(file.remove(probe))) stop("exposure recorder destination is not writable: ", dir)
  # created EMPTY now, so "no exposure entered run_dandelion_for_exposure" is distinguishable from "the recorder never ran"
  con <- file(file.path(dir, "exposures.jsonl"), open = "wb"); close(con)
  .xrec$dir <- dir; .xrec$pid <- Sys.getpid(); .xrec$cur <- NULL; .xrec$n <- 0L
  .xrec$burden <- list(); .xrec$bucket <- new.env(parent = emptyenv()); .xrec$nb <- 0L
  trace("run_dandelion_for_exposure", where = asNamespace("DANDELION"), print = FALSE, at = c(2L, 9L, 18L, 26L),
        tracer = quote(.dandelion_exposure_step(environment())),
        exit = quote(.dandelion_exposure_exit(environment(), returnValue(default = quote(.abnormal_exit)))))
  invisible(TRUE)
}

exposure_recorder_stop <- function() {
  suppressMessages(untrace("run_dandelion_for_exposure", where = asNamespace("DANDELION")))
  invisible(.xrec$n)
}

.dandelion_exposure_step <- function(frame) {
  if (!identical(Sys.getpid(), .xrec$pid)) stop("exposure recorder: traced call in a forked worker; run DANDELION with n.cores = 1", call. = FALSE)
  cur <- .xrec$cur
  if (is.null(cur)) cur <- list(hits = 0L, exposure = get("exposure.id", envir = frame))
  cur$hits <- cur$hits + 1L
  if (cur$hits == 1L) cur$n_trans <- length(get("gene.trans", envir = frame))
  if (cur$hits == 2L) cur$n_valid <- length(get("gene.trans", envir = frame))
  if (cur$hits == 3L) {
    cur$pi0a <- get("pi0a", envir = frame); cur$pi0b <- get("pi0b", envir = frame)
    cur$burden <- .dandelion_burden_input(get("p_b", envir = frame))
  }
  if (cur$hits == 4L) for (v in c("wg1", "wg2", "wg3", "wg.sum")) cur[[v]] <- get(v, envir = frame)
  .xrec$cur <- cur
}

.dandelion_burden_input <- function(p) {
  # The identity of the effective burden input; written ONCE per distinct input. identical(num.eq = FALSE) compares the doubles bit
  # for bit and the names attribute; the bucket key (length, first and last name) only narrows which earlier inputs are compared.
  if (!is.double(p) || length(p) < 1L || anyNA(p) || is.null(names(p)) || anyNA(names(p)) || any(names(p) == ""))
    stop("exposure recorder: the burden input at the pi0 guard is not a named, non-missing double vector", call. = FALSE)
  key <- paste(length(p), names(p)[1L], names(p)[length(p)], sep = "\t")
  seen <- if (exists(key, envir = .xrec$bucket, inherits = FALSE)) get(key, envir = .xrec$bucket, inherits = FALSE) else character(0)
  for (id in seen) if (identical(.xrec$burden[[id]], p, num.eq = FALSE)) return(id)
  .xrec$nb <- .xrec$nb + 1L
  id <- sprintf("burden-%04d", .xrec$nb)
  path <- file.path(.xrec$dir, paste0(id, ".tsv"))
  if (file.exists(path)) stop("exposure recorder: ", path, " already exists", call. = FALSE)
  con <- file(path, open = "wb"); on.exit(close(con))
  writeLines(paste(names(p), sprintf("%a", as.numeric(p)), sep = "\t"), con, sep = "\n", useBytes = TRUE)
  .xrec$burden[[id]] <- p
  assign(key, c(seen, id), envir = .xrec$bucket)
  id
}

.dandelion_exposure_exit <- function(frame, value) {
  cur <- .xrec$cur; .xrec$cur <- NULL
  if (is.null(cur)) return(invisible(NULL))
  .xrec$n <- .xrec$n + 1L
  abnormal <- identical(value, quote(.abnormal_exit))
  returned <- !abnormal && !is.null(value)
  outcome <- if (abnormal) "abnormal_exit" else if (returned && cur$hits == 4L) "scored" else if (!returned && cur$hits == 1L) "no_trans_genes" else
    if (!returned && cur$hits == 2L) "fewer_than_2_valid_genes" else if (!returned && cur$hits == 3L) "mixture_estimate_invalid" else
    if (!returned && cur$hits == 4L) "nonpositive_weight_sum" else "unclassified"
  num <- function(name) if (is.null(cur[[name]])) "null" else .xhex(cur[[name]])
  int <- function(name) if (is.null(cur[[name]])) "null" else as.character(as.integer(cur[[name]]))
  line <- paste0("{", paste(c(
    paste0("\"exposure_id\":", .xstr(cur$exposure)), paste0("\"outcome\":", .xstr(outcome)), paste0("\"last_guard_reached\":", cur$hits),
    paste0("\"n_trans\":", int("n_trans")), paste0("\"n_valid\":", int("n_valid")),
    paste0("\"pi0a\":", num("pi0a")), paste0("\"pi0b\":", num("pi0b")), paste0("\"wg1\":", num("wg1")), paste0("\"wg2\":", num("wg2")),
    paste0("\"wg3\":", num("wg3")), paste0("\"wg_sum\":", num("wg.sum")),
    paste0("\"burden_input\":", if (is.null(cur$burden)) "null" else .xstr(cur$burden)),
    paste0("\"observation_kind\":\"actual_call_trace\""), paste0("\"recorder_version\":", .xstr(EXPOSURE_RECORDER_VERSION))), collapse = ","), "}")
  con <- file(file.path(.xrec$dir, "exposures.jsonl"), open = "ab"); on.exit(close(con))
  writeLines(line, con, sep = "\n", useBytes = TRUE)
}
