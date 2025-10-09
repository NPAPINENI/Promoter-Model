# ==========================
# 📦 Load Packages
# ==========================
if (!requireNamespace("data.table")) install.packages("data.table", repos = "https://cloud.r-project.org/")
library(data.table)

# ====================
# 📂 Define Base Paths
# ====================
base_output_dir <- "/home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/proms"

# ====================
# 🔁 Process Each Mark
# ====================
marks <- list.dirs(base_output_dir, recursive = FALSE, full.names = FALSE)
marks <- marks[!grepl("^\\.", marks)]  # remove hidden/system dirs

# Set k-mer size
k <- 6

# Kmer generation function
generate_kmers_string <- function(seq, k) {
  n <- nchar(seq)
  if (n < k) return(NA_character_)
  kmers <- substring(seq, 1:(n-k+1), k:n)
  return(paste(kmers, collapse = " "))
}

for (mark in marks) {
  cat("\n===========================\n")
  cat("🔄 Starting Mark:", mark, "\n")

  # mark_output_inc <- file.path(base_output_dir, mark, "300bp_balanced", "inc")
  mark_output_inc <- file.path(base_output_dir, mark, "inc")

  files_to_process <- c(
    file.path(mark_output_inc, paste0("full_shape_median_abs_dev_", mark, ".csv")),
    file.path(mark_output_inc, paste0("shape_diffsum_", mark, ".csv")),
    file.path(mark_output_inc, paste0("shape_diffmean_", mark, ".csv"))
  )

  for (file_path in files_to_process) {
    if (!file.exists(file_path)) {
      cat("❌ Missing file:", basename(file_path), "- skipping.\n")
      next
    }

    # Define output file
    output_file <- sub("\\.csv$", "_kmers_label.tsv", file_path)

    # Skip if output already exists
    if (file.exists(output_file)) {
      cat("⏩ Already processed:", basename(output_file), "- skipping.\n")
      next
    }

    cat("🔍 Processing file:", basename(file_path), "\n")
    
    df <- fread(file_path)

    if (!"SeqName" %in% colnames(df)) {
      cat("❌ No SeqName column in", basename(file_path), "- skipping.\n")
      next
    }

    # 🧩 Split SeqName into SeqID and Label
    df$SeqID <- sapply(strsplit(df$SeqName, "_label_"), `[`, 1)
    df$Label <- as.integer(sapply(strsplit(df$SeqName, "_label_"), `[`, 2))

    # 🧬 Generate K-mers if Sequence exists
    if ("Sequence" %in% colnames(df)) {
      df$Kmers <- sapply(df$Sequence, generate_kmers_string, k = k)
    }

    # (Optional) Reorder columns: SeqName, SeqID, Label, Kmers first
    all_cols <- names(df)
    reorder_cols <- c("SeqName", "SeqID", "Label", "Kmers")
    reorder_cols <- reorder_cols[reorder_cols %in% all_cols]  # only those that exist
    other_cols <- setdiff(all_cols, reorder_cols)
    setcolorder(df, c(reorder_cols, other_cols))

    # Save to new _kmers_label.tsv file
    fwrite(df, output_file, sep = "\t")

    cat("✅ Saved new file:", basename(output_file), "\n")
  }
}
