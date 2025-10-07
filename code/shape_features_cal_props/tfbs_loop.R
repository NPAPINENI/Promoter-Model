# Load necessary libraries
library(data.table)
library(stringr)
library(tools)

# Define histone marks
# histone_marks <- c("H3K27ac", "H3K27me3", "H3K4me2", "H3K4me3", "H3K9ac")
histone_marks <- c("CHD4")
base_input_dir <- "/home/shared/rdavuluri/TFBS_Pallavi/ramana.cewit/1_1_pos_neg/CHD4/300bp_balanced"
base_output_dir <- "/home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/TFBS/CHD4_500bp"

# Define paths
# base_input_dir <- "/home/shared/rdavuluri/TFBS_data_histone_marks"
# base_input_dir <- "/home/shared/rdavuluri/TFBS_Pallavi/ramana.cewit/1_1_pos_neg/CHD4/"

# base_output_dir <- "/home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/TFBS/"
# base_output_dir <- "/home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/TFBS/500bp"

# Function to reconstruct sequence from k-mers
# reconstruct_sequence_from_kmers <- function(kmers_str, k = 6) {
#   kmers <- unlist(strsplit(kmers_str, " "))
#   if (length(kmers) == 0) return("")
#   sequence <- kmers[1]
#   for (i in 2:length(kmers)) {
#     sequence <- paste0(sequence, substr(kmers[i], k, k))
#   }
#   return(sequence)
# }

# # Loop through histone marks
# for (mark in histone_marks) {
#   cat("🔄 Processing:", mark, "\n")
  
#   mark_dir <- file.path(base_output_dir, mark)
#   dir.create(mark_dir, showWarnings = FALSE, recursive = TRUE)
  
#   train_path <- file.path(base_input_dir, mark, "train.tsv")
#   dev_path <- file.path(base_input_dir, mark, "dev.tsv")
  
#   if (!file.exists(train_path) || !file.exists(dev_path)) {
#     cat("❌ Missing files for", mark, "- skipping.\n")
#     next
#   }
  
#   df1 <- fread(train_path, sep = "\t", col.names = c("Sequence", "Label"), header = FALSE)
#   df2 <- fread(dev_path, sep = "\t", col.names = c("Sequence", "Label"), header = FALSE)
#   df_combined <- rbind(df1, df2)

#   # Save merged file
#   merged_path <- file.path(mark_dir, paste0("merged_file_", mark, ".tsv"))
#   fwrite(df_combined, merged_path, sep = "\t")

#   # Reconstruct sequences
#   setnames(df_combined, "Sequence", "Kmers")
#   df_combined[, Sequence := sapply(Kmers, reconstruct_sequence_from_kmers)]

#   # Save reconstructed file
#   kmers_merged_path <- file.path(mark_dir, paste0("kmers_merged_file_", mark, ".tsv"))
#   fwrite(df_combined, kmers_merged_path, sep = "\t")

  # Clean and save final file
#   if ("Kmers" %in% names(df_combined)) df_combined[, Kmers := NULL]
#   df_combined <- df_combined[, !grepl("^Unnamed", names(df_combined)), with = FALSE]

#   cleaned_output_path <- file.path(mark_dir, paste0("rem_kmers_merged_file_", mark, ".tsv"))
#   fwrite(df_combined, cleaned_output_path, sep = "\t")

#   cat("✅ Done:", mark, "\n")
# }
# Load necessary libraries




# Function to reconstruct sequence from k-mers
reconstruct_sequence_from_kmers <- function(kmers_str, k = 6) {
  kmers <- unlist(strsplit(kmers_str, " "))
  if (length(kmers) == 0) return("")
  sequence <- kmers[1]
  for (i in 2:length(kmers)) {
    sequence <- paste0(sequence, substr(kmers[i], k, k))
  }
  return(sequence)
}

# Loop through histone marks
for (mark in histone_marks) {
  cat("🔄 Processing:", mark, "\n")
  
  mark_dir <- file.path(base_output_dir, mark)
  dir.create(mark_dir, showWarnings = FALSE, recursive = TRUE)
  
  train_path <- file.path(base_input_dir, mark, "train.tsv")
  dev_path <- file.path(base_input_dir, mark, "dev.tsv")
  
  if (!file.exists(train_path) || !file.exists(dev_path)) {
    cat("❌ Missing files for", mark, "- skipping.\n")
    next
  }
  
  df1 <- fread(train_path, sep = "\t", header = TRUE)  # Keep original headers
  df2 <- fread(dev_path, sep = "\t", header = TRUE)    # Keep original headers
  df_combined <- rbind(df1, df2)

  # Save merged file
  merged_path <- file.path(mark_dir, paste0("merged_file_", mark, ".tsv"))
  fwrite(df_combined, merged_path, sep = "\t")

  # Reconstruct sequences (assumes first column is k-mers)
  kmer_col <- names(df_combined)[1]  # Assume first column contains k-mers
  df_combined[, Sequence := sapply(get(kmer_col), reconstruct_sequence_from_kmers)]

  # Save reconstructed file
  kmers_merged_path <- file.path(mark_dir, paste0("kmers_merged_file_", mark, ".tsv"))
  fwrite(df_combined, kmers_merged_path, sep = "\t")

  # Clean and save final file
  df_combined[, (kmer_col) := NULL]  # Remove original k-mer column
  cleaned_output_path <- file.path(mark_dir, paste0("rem_kmers_merged_file_", mark, ".tsv"))
  fwrite(df_combined, cleaned_output_path, sep = "\t")

  cat("✅ Done:", mark, "\n")
}
