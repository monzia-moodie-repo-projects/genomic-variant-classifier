"""R-side package-metadata semantics for the pre-install gate (owner ruling 2026-10-07).

Python checks bytes and archive structure; the QUALIFIED R interprets R's own metadata: read.dcf for DESCRIPTION records (multiple
records, repeated fields, continuation lines) and package_version for version comparison. Measured 2026-10-07 against the merged
Python approximations: "1.2" vs "1.2.0" compared unequal (R: equal); two blank-line-separated records were merged (R: two records);
"stats (>= 999.0)" was accepted unchecked; empty coverage was accepted.

R_PACKAGE_SEMANTICS is the owner's reference R code (ruling generation d2dbb657: check_dependency_closure etc. lines 168-311 and
read_description_strict lines 415-450, taken by measured line range) with ONE refinement -- an empty DESCRIPTION is refused as
description.record_count, any other read.dcf failure is reported as description.unparseable, and a malformed Version as
description.version_syntax (measured under R 4.3.3: the reference leaked R's internal messages for both) -- plus two entry points. It is kept as TEXT so it ships with the Python
package and run_r_file preserves the exact program each run executed.

Author: Monzia Moodie
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["R_PACKAGE_SEMANTICS"]

R_PACKAGE_SEMANTICS = r'''read_one_description <- function(path) {
    records <- read.dcf(path)

    if (nrow(records) != 1L)
        stop("description.record_count:", path)

    required <- c("Package", "Version")
    if (!all(required %in% colnames(records)))
        stop("description.identity_missing:", path)

    if (anyNA(records[1L, required]))
        stop("description.identity_missing:", path)

    as.list(records[1L, ])
}


version_satisfies <- function(have, operator, need) {
    h <- package_version(have)
    n <- package_version(need)

    answer <- switch(
        operator,
        ">=" = h >= n,
        "<=" = h <= n,
        "==" = h == n,
        ">"  = h > n,
        "<"  = h < n,
        stop("dependency.operator:", operator)
    )

    isTRUE(answer)
}


check_dependency_closure <- function(
    descriptions,
    planned_versions,
    runtime_base_versions,
    runtime_version
) {
    # Inputs are named lists or named character vectors.
    valid_names <- function(x) {
        n <- names(x)
        !is.null(n) && !anyNA(n) &&
            !any(n == "") && !anyDuplicated(n)
    }

    if (!valid_names(descriptions) ||
        !valid_names(planned_versions) ||
        !valid_names(runtime_base_versions))
        stop("dependency.names")

    if (!setequal(names(descriptions), names(planned_versions)))
        stop("dependency.description_coverage")

    if (length(intersect(
        names(planned_versions), names(runtime_base_versions)
    )))
        stop("dependency.base_inventory_overlap")

    available <- c(
        as.list(planned_versions),
        as.list(runtime_base_versions),
        list(R = runtime_version)
    )

    pattern <- paste0(
        "^([A-Za-z][A-Za-z0-9.]*)[[:space:]]*",
        "(\\([[:space:]]*(>=|<=|==|>|<)[[:space:]]*",
        "([0-9]+([.-][0-9]+)*)[[:space:]]*\\))?$"
    )

    problems <- character()

    for (package in sort(names(descriptions))) {
        description <- descriptions[[package]]

        if (!identical(description[["Package"]], package))
            stop("dependency.package_identity:", package)

        if (!identical(
            description[["Version"]],
            unname(planned_versions[[package]])
        ))
            stop("dependency.package_version:", package)

        for (field in c("Depends", "Imports", "LinkingTo")) {
            value <- description[[field]]

            if (is.null(value) || is.na(value) ||
                !nzchar(trimws(value)))
                next

            entries <- trimws(strsplit(value, ",", fixed = TRUE)[[1L]])

            for (entry in entries) {
                match <- regmatches(
                    entry, regexec(pattern, entry)
                )[[1L]]

                if (!length(match)) {
                    problems <- c(
                        problems,
                        paste(package, field, "unparsed", entry, sep = ":")
                    )
                    next
                }

                dependency <- match[[2L]]
                operator <- match[[4L]]
                required <- match[[5L]]
                observed <- available[[dependency]]

                if (is.null(observed)) {
                    problems <- c(
                        problems,
                        paste(package, dependency, "missing", sep = ":")
                    )
                } else if (
                    nzchar(operator) &&
                    !version_satisfies(observed, operator, required)
                ) {
                    problems <- c(
                        problems,
                        paste(
                            package, dependency,
                            observed, operator, required,
                            sep = ":"
                        )
                    )
                }
            }
        }
    }

    if (length(problems))
        stop(
            "dependency.unsatisfied:\n",
            paste(problems, collapse = "\n")
        )

    invisible(TRUE)
}

read_description_strict <- function(path) {
    # REFINED 2026-10-07 (measured under R 4.3.3): read.dcf(all = TRUE) on an EMPTY file raised R's internal "missing value where
    # TRUE/FALSE needed" instead of a record-count refusal; any other parse failure is reported, never leaked uncategorised.
    raw <- readBin(path, "raw", file.size(path))
    if (!length(raw) || !nzchar(trimws(rawToChar(raw[raw != as.raw(0)]))))
        stop("description.record_count")
    records <- tryCatch(read.dcf(path, all = TRUE),
                        error = function(e) stop("description.unparseable: ", conditionMessage(e), call. = FALSE))

    if (nrow(records) != 1L)
        stop("description.record_count")

    fields <- names(records)
    result <- setNames(vector("list", length(fields)), fields)

    for (field in fields) {
        column <- records[[field]]

        # all=TRUE uses list columns for repeated fields and
        # character columns otherwise.
        value <- if (is.list(column)) column[[1L]] else column[1L]

        if (length(value) != 1L)
            stop("description.duplicate_field:", field)

        if (is.na(value))
            stop("description.missing_value:", field)

        result[[field]] <- unname(value)
    }

    for (field in c("Package", "Version")) {
        value <- result[[field]]
        if (is.null(value) || !nzchar(value))
            stop("description.identity_missing:", field)
    }

    # Validate package-version syntax using R's own implementation.
    tryCatch(package_version(result[["Version"]]),
             error = function(e) stop("description.version_syntax:", result[["Version"]], call. = FALSE))

    result
}

# ---------------------------------------------------------------------------------------------------------------- entry points
# Values are escaped so each result is ONE line: backslash -> \\, tab -> \t, newline -> \n (Python unescapes).
gvc_escape <- function(x) gsub("\n", "\\n", gsub("\t", "\\t", gsub("\\", "\\\\", x, fixed = TRUE), fixed = TRUE), fixed = TRUE)

# describe: for every line "<index>\t<path>" of `listing`, read that DESCRIPTION strictly and write "<index>\t<field>\t<value>" lines.
gvc_describe <- function(listing, out) {
    rows <- strsplit(readLines(listing, warn = FALSE), "\t", fixed = TRUE)
    lines <- character()
    for (row in rows) {
        fields <- read_description_strict(row[[2L]])
        for (name in names(fields)) lines <- c(lines, paste(row[[1L]], gvc_escape(name), gvc_escape(fields[[name]]), sep = "\t"))
    }
    writeLines(lines, out, useBytes = TRUE)
}

# closure: descriptions from the listing, planned versions from "<package>\t<version>" lines; the runtime's BASE inventory is MEASURED here.
gvc_closure <- function(listing, planned_file) {
    rows <- strsplit(readLines(listing, warn = FALSE), "\t", fixed = TRUE)
    descriptions <- setNames(list(), character())       # NAMED even when empty, so coverage is judged by the reference
    for (row in rows) { d <- read_description_strict(row[[2L]]); descriptions[[d[["Package"]]]] <- d }
    planned_rows <- strsplit(readLines(planned_file, warn = FALSE), "\t", fixed = TRUE)
    planned <- vapply(planned_rows, `[[`, "", 2L); names(planned) <- vapply(planned_rows, `[[`, "", 1L)
    base <- utils::installed.packages(priority = "base", noCache = TRUE)
    runtime_base <- base[, "Version"]; names(runtime_base) <- rownames(base)
    check_dependency_closure(descriptions, planned, runtime_base, as.character(getRversion()))
    cat("GVC_CLOSURE_OK\n")
}
'''
