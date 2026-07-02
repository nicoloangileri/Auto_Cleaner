#!/usr/bin/env Rscript
# =============================================================================
# auto_cleaner :: R EDA companion
# -----------------------------------------------------------------------------
# A teaching/cross-check companion to the Python/polars engine. It re-derives
# the core statistical EDA in idiomatic, functional R so results can be
# validated independently (useful for coursework and peer review).
#
# Usage:
#   Rscript r/eda_companion.R <clean_data.parquet|csv> [out_report.md]
#
# Optional packages (graceful fallback if missing):
#   * arrow    -> read Parquet directly (else falls back to a sibling CSV)
#   * moments  -> skewness / kurtosis (else computed from first principles)
# =============================================================================

suppressWarnings(suppressMessages({
  have_arrow   <- requireNamespace("arrow", quietly = TRUE)
  have_moments <- requireNamespace("moments", quietly = TRUE)
}))

# ---- args -------------------------------------------------------------------
args   <- commandArgs(trailingOnly = TRUE)
input  <- if (length(args) >= 1) args[[1]] else "clean_data.parquet"
out_md <- if (length(args) >= 2) args[[2]] else "eda_report_R.md"

# ---- IO ---------------------------------------------------------------------
read_data <- function(path) {
  is_parquet <- grepl("\\.parquet$|\\.pq$", path, ignore.case = TRUE)
  if (is_parquet && have_arrow) {
    return(as.data.frame(arrow::read_parquet(path)))
  }
  if (is_parquet && !have_arrow) {
    alt <- sub("\\.(parquet|pq)$", ".csv", path, ignore.case = TRUE)
    if (file.exists(alt)) {
      message("arrow not installed; reading sibling CSV: ", alt)
      return(utils::read.csv(alt, stringsAsFactors = FALSE))
    }
    stop("Parquet requested but 'arrow' is unavailable and no sibling CSV exists.")
  }
  utils::read.csv(path, stringsAsFactors = FALSE)
}

# ---- statistics (functional) ------------------------------------------------
skewness_ <- function(x) {
  if (have_moments) return(moments::skewness(x, na.rm = TRUE))
  x <- x[is.finite(x)]; n <- length(x)
  if (n < 3) return(NA_real_)
  m <- mean(x); s <- sd(x)
  if (s == 0) return(0)
  (sum((x - m)^3) / n) / (s^3)
}

kurtosis_ <- function(x) {
  if (have_moments) return(moments::kurtosis(x, na.rm = TRUE) - 3)  # excess
  x <- x[is.finite(x)]; n <- length(x)
  if (n < 4) return(NA_real_)
  m <- mean(x); s <- sd(x)
  if (s == 0) return(0)
  (sum((x - m)^4) / n) / (s^4) - 3
}

profile_numeric <- function(df) {
  num <- df[, vapply(df, is.numeric, logical(1)), drop = FALSE]
  if (ncol(num) == 0) return(NULL)
  stats <- lapply(names(num), function(nm) {
    x <- num[[nm]]
    data.frame(
      feature  = nm,
      n        = sum(!is.na(x)),
      missing  = sum(is.na(x)),
      mean     = mean(x, na.rm = TRUE),
      sd       = sd(x, na.rm = TRUE),
      min      = suppressWarnings(min(x, na.rm = TRUE)),
      median   = median(x, na.rm = TRUE),
      max      = suppressWarnings(max(x, na.rm = TRUE)),
      skewness = round(skewness_(x), 4),
      kurtosis = round(kurtosis_(x), 4),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, stats)
}

normality <- function(df, max_n = 5000L) {
  num <- df[, vapply(df, is.numeric, logical(1)), drop = FALSE]
  if (ncol(num) == 0) return(NULL)
  res <- lapply(names(num), function(nm) {
    x <- num[[nm]]; x <- x[is.finite(x)]
    if (length(x) < 3 || length(unique(x)) < 3) {
      return(data.frame(feature = nm, W = NA, p_value = NA, stringsAsFactors = FALSE))
    }
    if (length(x) > max_n) x <- sample(x, max_n)
    t <- tryCatch(shapiro.test(x), error = function(e) NULL)
    if (is.null(t)) return(data.frame(feature = nm, W = NA, p_value = NA, stringsAsFactors = FALSE))
    data.frame(feature = nm, W = round(unname(t$statistic), 4),
               p_value = signif(t$p.value, 4), stringsAsFactors = FALSE)
  })
  do.call(rbind, res)
}

collinear_pairs <- function(df, threshold = 0.9) {
  num <- df[, vapply(df, is.numeric, logical(1)), drop = FALSE]
  if (ncol(num) < 2) return(NULL)
  cm <- suppressWarnings(cor(num, use = "pairwise.complete.obs"))
  out <- list()
  cols <- colnames(cm)
  for (i in seq_len(ncol(cm) - 1)) {
    for (j in (i + 1):ncol(cm)) {
      r <- cm[i, j]
      if (!is.na(r) && abs(r) >= threshold) {
        out[[length(out) + 1]] <- data.frame(
          feature_a = cols[i], feature_b = cols[j], r = round(r, 3),
          stringsAsFactors = FALSE
        )
      }
    }
  }
  if (length(out)) do.call(rbind, out) else NULL
}

# ---- markdown emitter -------------------------------------------------------
md_table <- function(df) {
  if (is.null(df) || nrow(df) == 0) return("_none_\n")
  header <- paste0("| ", paste(colnames(df), collapse = " | "), " |")
  sep    <- paste0("|", paste(rep("---", ncol(df)), collapse = "|"), "|")
  body   <- apply(df, 1, function(r) paste0("| ", paste(r, collapse = " | "), " |"))
  paste(c(header, sep, body), collapse = "\n")
}

main <- function() {
  message("Reading: ", input)
  df <- read_data(input)
  message(sprintf("Loaded %d rows x %d cols", nrow(df), ncol(df)))

  numeric_profile <- profile_numeric(df)
  norm            <- normality(df)
  pairs           <- collinear_pairs(df)

  lines <- c(
    "# R EDA Companion Report",
    "",
    sprintf("*Source:* `%s`  •  *Rows:* %d  •  *Cols:* %d  •  *Generated:* %s",
            input, nrow(df), ncol(df), format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
    "",
    "## Numeric summary (incl. skewness & excess kurtosis)",
    "",
    md_table(numeric_profile),
    "",
    "## Shapiro-Wilk normality test",
    "",
    md_table(norm),
    "",
    "## High-collinearity pairs (|r| >= 0.9)",
    "",
    md_table(pairs),
    ""
  )
  writeLines(lines, out_md)
  message("Report written: ", out_md)

  cat("\n==== Numeric summary ====\n");  print(numeric_profile, row.names = FALSE)
  cat("\n==== Normality (Shapiro-Wilk) ====\n"); print(norm, row.names = FALSE)
  cat("\n==== Collinear pairs ====\n"); print(if (is.null(pairs)) "none" else pairs, row.names = FALSE)
}

main()
