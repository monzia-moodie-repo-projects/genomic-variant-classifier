# method_fixtures.R -- ONE predetermined DANDELION method fixture in ONE fresh R process (owner rulings 2026-10-08e section 6,
# 2026-10-08f section 6). Author: Monzia Moodie
#
# Usage: Rscript --vanilla method_fixtures.R <dandelion_backend_recorder.R> <dandelion_exposure_recorder.R> <scenario dir>
#        <mode: off | on> <output dir>
#
# The scenario directory is written by inference/method_trace.py (prepare_scenarios) from the frozen fixture specification
# tests/fixtures/dandelion/method_fixtures_v2.json; this runner never reads the specification or its predictions, so it cannot be
# tuned to them. It runs the INSTALLED DANDELION (and qvalue, when installed) exactly as a caller would, and writes every scientific
# output as exact hexadecimal binary64 text ("%a", lossless) through BINARY connections (identical bytes on every platform).
#
# route : DANDELION:::safe_qvalues on one vector, called from a frame holding `exposure.id` (as run_dandelion_for_exposure does), so the
#         recorder attributes the call. Independent values computed BEFORE any recording: p.adjust(clamp_p(p), "BH") and, when qvalue is
#         installed, qvalue(clamp_p(p), pi0 = 1, lfdr.out = FALSE)$qvalues.
# trace : DANDELION::med_gene(n.cores = 1) on a small gene-by-exposure matrix, then DANDELION::calc_pair.gene -- the package's own
#         nominations. mat.p (the raw DANDELION p-values the extended ranking consumes) and mat.sig (the q-value decisions) are written
#         separately, so a reader can see which layer an adjustment change reaches.
# Both  : set.seed(20261008) first; the random-number state before and after the science and a SECOND statistical call (runif(3))
#         are written, so recorder-off and recorder-on runs can be compared for side effects on the random stream.
# mode on: the actual-call backend recorder is started before the science and stopped after it; its trace goes to <output dir>/trace.
#         For a trace scenario the actual-call EXPOSURE recorder (owner ruling 2026-10-08g) is started too: one line per exposure that
#         entered run_dandelion_for_exposure, with the guard it reached and its mixture estimates, in <output dir>/trace_exposures.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) stop("usage: method_fixtures.R <backend recorder.R> <exposure recorder.R> <scenario dir> <off|on> <output dir>")
recorder_file <- args[1]; exposure_recorder_file <- args[2]; scenario_dir <- args[3]; mode <- args[4]; out_dir <- args[5]
if (!(mode %in% c("off", "on"))) stop("mode must be off or on: ", mode)
if (dir.exists(out_dir) && length(list.files(out_dir, all.files = TRUE, no.. = TRUE)) > 0) stop("output directory is not empty: ", out_dir)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
source(recorder_file)
source(exposure_recorder_file)
METHOD_COMMIT <- "f471153bfa3c0069cd68a67565000889c7cdf5d1"

write_lines_bin <- function(lines, name) {
  con <- file(file.path(out_dir, name), open = "wb"); on.exit(close(con))
  writeLines(as.character(lines), con, sep = "\n", useBytes = TRUE)
}
hex <- function(x) ifelse(is.na(x), "NA", sprintf("%a", as.numeric(x)))
read_hex <- function(name) {
  lines <- readLines(file.path(scenario_dir, name), warn = FALSE)
  x <- rep(NA_real_, length(lines)); ok <- lines != "NA"
  x[ok] <- as.numeric(lines[ok])
  if (any(is.na(x[ok]))) stop("unparseable value in ", name)
  x
}

scenario <- read.dcf(file.path(scenario_dir, "scenario.dcf"), all = TRUE)
if (nrow(scenario) != 1) stop("scenario.dcf must hold exactly one record")
kind <- scenario$Kind; fixture <- scenario$Fixture; target_fdr <- as.numeric(scenario$TargetFDR)

suppressPackageStartupMessages(library(DANDELION))
has_qvalue <- requireNamespace("qvalue", quietly = TRUE)
warnings_seen <- character(0)
science <- function(expr) withCallingHandlers(expr, warning = function(w) {
  warnings_seen <<- c(warnings_seen, conditionMessage(w)); invokeRestart("muffleWarning")
})

set.seed(20261008L)
seed_before <- .Random.seed

if (kind == "route") {
  p <- read_hex("input.txt")
  clamped <- DANDELION:::clamp_p(p)
  write_lines_bin(hex(stats::p.adjust(clamped, method = "BH")), "oracle_bh.txt")
  if (has_qvalue) write_lines_bin(hex(qvalue::qvalue(clamped, pi0 = 1, lfdr.out = FALSE)$qvalues), "oracle_qvalue_pi0_1.txt")
  caller <- function(exposure.id, values) DANDELION:::safe_qvalues(values)
  if (mode == "on") recorder_start(file.path(out_dir, "trace"), METHOD_COMMIT)
  q <- science(caller(fixture, p))
  if (mode == "on") recorder_stop()
  write_lines_bin(hex(q), "output.txt")
} else if (kind == "trace") {
  genes <- readLines(file.path(scenario_dir, "genes.txt"), warn = FALSE)
  exposures <- readLines(file.path(scenario_dir, "exposures.txt"), warn = FALSE)
  p.trans <- matrix(NA_real_, length(genes), length(exposures), dimnames = list(genes, exposures))
  for (e in exposures) p.trans[, e] <- read_hex(paste0("trans_", e, ".txt"))
  p.wes <- stats::setNames(read_hex("wes.txt"), genes)
  ref.table <- utils::read.delim(file.path(scenario_dir, "ref_table.tsv"), colClasses = c("character", "character", "character", "numeric", "numeric"))
  if (mode == "on") { exposure_recorder_start(file.path(out_dir, "trace_exposures")); recorder_start(file.path(out_dir, "trace"), METHOD_COMMIT) }
  res <- science(DANDELION::med_gene(p.trans = p.trans, p.wes = p.wes, ref.table = ref.table, gene1.list = exposures,
                                      target.fdr = target_fdr, gene1.type = "Gene", n.cores = 1))
  if (mode == "on") { recorder_stop(); exposure_recorder_stop() }
  # the same filtered annotation med_gene uses internally (its first three filters), for calc_pair.gene
  keep <- ref.table[ref.table$type %in% c("lincRNA", "protein_coding"), , drop = FALSE]
  keep <- keep[!(keep$Chromosome %in% c("chrM", "chrX", "chrY")), , drop = FALSE]
  keep <- keep[!duplicated(keep$gene_name), , drop = FALSE]
  pairs <- science(DANDELION::calc_pair.gene(res$mat.sig, res$mat.p, p.wes, res$gene1, keep))
  write_lines_bin(res$gene1, "gene1.txt")
  cells <- expand.grid(gene = rownames(res$mat.p), exposure = colnames(res$mat.p), stringsAsFactors = FALSE)
  write_lines_bin(paste(cells$gene, cells$exposure, hex(res$mat.p[cbind(cells$gene, cells$exposure)]), sep = "\t"), "mat_p.tsv")
  write_lines_bin(paste(cells$gene, cells$exposure, res$mat.sig[cbind(cells$gene, cells$exposure)], sep = "\t"), "mat_sig.tsv")
  pd <- pairs$pairs_dact
  write_lines_bin(if (nrow(pd)) paste(pd$gene1, pd$gene2, hex(pd$DANDELION_p), sep = "\t") else character(0), "nominations.tsv")
} else stop("unknown scenario kind: ", kind)

seed_after <- .Random.seed
second <- stats::runif(3)
write_lines_bin(c(paste(seed_before, collapse = " "), paste(seed_after, collapse = " "), hex(second)), "rng.txt")
write_lines_bin(warnings_seen, "warnings.txt")
desc <- function(pkg) if (pkg == "qvalue" && !has_qvalue) "absent" else
  paste(as.character(utils::packageVersion(pkg)), normalizePath(file.path(find.package(pkg), "DESCRIPTION"), winslash = "/"), sep = "\t")
write_lines_bin(c(paste0("R\t", R.version.string), paste0("platform\t", R.version$platform), paste0("rng\t", paste(RNGkind(), collapse = " ")),
                  paste0("libpaths\t", paste(normalizePath(.libPaths(), winslash = "/"), collapse = " | ")),
                  paste0("DANDELION\t", desc("DANDELION")), paste0("qvalue\t", desc("qvalue")), paste0("mode\t", mode)), "environment.tsv")
cat("method fixture", fixture, "mode", mode, "complete\n")
