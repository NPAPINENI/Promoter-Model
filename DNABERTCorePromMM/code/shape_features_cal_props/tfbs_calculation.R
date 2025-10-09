install.packages("data.table", repos = "https://cloud.r-project.org/")

library(DNAshapeR)
library(Biostrings)
library(data.table)



marks <- c("300bp_balanced")

base_dir <- "/home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/TFBS/CHD4_500bp/"

for (mark in marks) {
  cat("🔄 Running shape prediction for:", mark, "\n")
  fasta_path <- file.path(base_dir, mark, "main.fa")
  output_dir <- file.path(base_dir, mark)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  if (!file.exists(fasta_path)) {
    cat("❌ Skipping", mark, "- FASTA file not found at", fasta_path, "\n")
    next
  }

  seqs <- readDNAStringSet(fasta_path)
  seqs_char <- as.character(seqs)


  valid <- nchar(seqs_char) >= 11 & nchar(seqs_char) <= 1000 & grepl("^[ACGTacgt]+$", seqs_char)
  seqs <- seqs[valid]
  seqs_char <- as.character(seqs)
  names(seqs) <- paste0("seq", seq_along(seqs))

  temp_fa <- tempfile(fileext = ".fa")
  writeXStringSet(seqs, temp_fa)


  core_feats <- c("HelT", "MGW", "ProT", "Roll")
  extra_feats <- c("Rise", "Shift", "Slide", "Tilt", "Buckle", "Opening", "Shear", "Stagger", "Stretch")
  all_pred <- getShape(temp_fa)

  for (f in extra_feats) {
    all_pred[[f]] <- getShape(temp_fa, shapeType = f)[[f]]
  }

  final_df <- data.frame(SeqName = names(seqs), Sequence = seqs_char, stringsAsFactors = FALSE)

  for (feature in names(all_pred)) {
    shape <- all_pred[[feature]]
    if (is.matrix(shape)) {
      diff_sum <- rowSums(abs(t(apply(shape, 1, diff))), na.rm = TRUE)
      df_temp <- data.frame(SeqName = rownames(shape), diffsum = diff_sum)
      final_df <- merge(final_df, df_temp, by = "SeqName", all.x = TRUE)
      colnames(final_df)[ncol(final_df)] <- paste0(feature, "_diffsum")
    } else {
      cat("⚠️ Skipping", feature, "(not a matrix)\n")
    }
  }

  out_file <- file.path(output_dir, paste0(mark, "_shape_summary_diff.csv"))
  fwrite(final_df, out_file)

  missing <- setdiff(paste0("seq", seq_along(seqs)), rownames(all_pred[["MGW"]]))
  if (length(missing) > 0) {
    writeLines(missing, file.path(output_dir, "missing_sequences_due_to_shape_failure.txt"))
  }

  cat("✅ Completed:", mark, "\n")
}
