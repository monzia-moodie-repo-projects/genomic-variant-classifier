# method_fixtures_v2_values.R -- PROVENANCE of the exact binary64 inputs in method_fixtures_v2.json (hexadecimal, lossless), from their
# decimal definitions. Usage: Rscript --vanilla method_fixtures_v2_values.R  (prints one tab-separated line per vector).
# R1-R6 and T1 are the inputs of the superseded, never-run v1 specification, unchanged; T2 is T1 extended (ruling 2026-10-08g).
# Author: Monzia Moodie
spread <- function(n, lo, hi) seq(lo, hi, length.out = n)
h <- function(x) ifelse(is.na(x), "NA", sprintf("%a", x))
emit <- function(id, x) cat(id, "\t", paste(h(x), collapse = " "), "\n", sep = "")
emit("R1", c(seq(0.001, 0.2, length.out = 30), 0.96))
emit("R2", seq(0.10, 0.89, length.out = 40))
emit("R3", c(0.001, 0.01, 0.02, 0.2, 0.5, 0.6, 0.7, 0.8, 0.99))
emit("R4", rep(c(0.1, 0.2, 0.3), 4))
emit("R5", c(0, 1, seq(0.01, 0.9, length.out = 10)))
emit("R6", seq(0.001, 0.9, length.out = 25))
e1 <- c(1e-9, 1e-7, 1e-5, spread(27, 0.02, 0.999)); e1[4] <- 0.97; e1[6] <- 0.02
e2 <- c(1e-6, 1e-4, spread(28, 0.05, 0.60)); e2[4] <- 0.58
e3 <- c(1e-8, 1e-6, spread(7, 0.1, 0.9), rep(NA, 21))
e4 <- c(0.5, rep(NA, 29))
wes <- c(1e-6, 1e-4, 1e-3, spread(27, 0.01, 0.98))
emit("T1.E1", e1); emit("T1.E2", e2); emit("T1.E3", e3); emit("T1.E4", e4)
emit("T1.wes", wes)
# T2: gene G31 (missing for E1-E4 and E6) and exposures E5 (every trans p-value 1 -> negative trans-side pi0) and E6 (unannotated)
emit("T2.E1", c(e1, NA)); emit("T2.E2", c(e2, NA)); emit("T2.E3", c(e3, NA)); emit("T2.E4", c(e4, NA))
emit("T2.E5", rep(1, 31)); emit("T2.E6", c(e1, NA))
emit("T2.wes", c(wes, 0.99))
