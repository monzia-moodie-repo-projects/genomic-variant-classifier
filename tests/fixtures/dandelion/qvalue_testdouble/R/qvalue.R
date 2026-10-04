qvalue <- function(p, ...) {
  mode <- Sys.getenv("GVC_QVALUE_DOUBLE_MODE", "success")
  if (mode == "warning") warning("test double: forced warning")
  if (mode == "error") stop("test double: forced error")
  list(qvalues = pmin(1, stats::p.adjust(p, method = "BH") / 2))   # deliberately NOT equal to BH
}
