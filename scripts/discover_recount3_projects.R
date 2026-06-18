suppressPackageStartupMessages({
  library(recount3)
  library(jsonlite)
})

out_dir <- "G:/My Drive/genomic-variant-data/external/rnaseq/recount3"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

projects <- available_projects()

human <- subset(projects, organism == "human")

write.csv(
  human,
  file = file.path(out_dir, "recount3_human_available_projects.csv"),
  row.names = FALSE
)

cat("Human projects:", nrow(human), "\n")
cat("Wrote:", file.path(out_dir, "recount3_human_available_projects.csv"), "\n")
