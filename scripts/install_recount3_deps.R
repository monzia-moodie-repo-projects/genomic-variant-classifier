if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org")
}

pkgs <- c("recount3", "SummarizedExperiment", "HDF5Array", "rhdf5", "jsonlite")

for (pkg in pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    BiocManager::install(pkg, ask = FALSE, update = FALSE)
  }
}

cat("recount3 dependencies OK\n")
