

# Generate FASTA for all histone marks
# install.packages("dplyr", repos = "https://cloud.r-project.org")
library(dplyr)

# Histone mark list
# marks <- c("H3K27ac", "H3K27me3", "H3K4me2", "H3K4me3", "H3K9ac")
# base_path <- "/home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/TFBS/500bp"
marks <- c("300bp_balanced")

base_path <- "/home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/TFBS/CHD4_500bp/"


# FASTA writer function
write_fasta <- function(df, filename) {
  fasta_content <- unlist(lapply(1:nrow(df), function(i) {
    paste0(">seq", i, "_label_", df$Label[i], "\n", df$Sequence[i])
  }))
  writeLines(fasta_content, filename)
}

# Loop through each mark
for (mark in marks) {
  input_file <- file.path(base_path, mark, paste0("kmers_merged_file_", mark, ".tsv"))
  output_file <- file.path(base_path, mark, "main.fa")

  cat("🔄 Processing:", mark, "\n")

  # Read input
  data <- read.csv(input_file, sep = "\t", header = TRUE, stringsAsFactors = FALSE)

  # Ensure required columns exist
  stopifnot(all(c("Sequence", "Label") %in% colnames(data)))

  # Write FASTA
  write_fasta(data, output_file)
  cat("✅ FASTA created at:", output_file, "\n")
}




# # Generate Fasta 
# # install.packages("dplyr", repos = "https://cloud.r-project.org")
# library(dplyr)


# # Read the input data
# data <- read.csv("/home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/TFBS/H3K27me3/kmers_merged_file_H3K27me3.tsv", sep = "\t", header = TRUE, stringsAsFactors = FALSE)

# # Confirm required columns exist
# stopifnot(all(c("Sequence", "Label") %in% colnames(data)))

# # ✅ FASTA writer for full dataset (Label is included in the header)
# write_fasta <- function(df, filename) {
#   fasta_content <- unlist(lapply(1:nrow(df), function(i) {
#     paste0(">seq", i, "_label_", df$Label[i], "\n", df$Sequence[i])
#   }))
#   writeLines(fasta_content, filename)
# }

# # ✅ Write one combined FASTA file
# output_file <- "/home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/TFBS/H3K27me3/main.fa"
# write_fasta(data, output_file)

# cat("✅ Combined FASTA file created at:", output_file, "\n")



#____________________________________________________________




# data <- read.csv("/home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/nontata/PB/new_merged_file_PB.tsv", 
#                  sep = "\t", header = TRUE, stringsAsFactors = FALSE)

# # Confirm required columns exist
# stopifnot(all(c("Sequence", "Label") %in% colnames(data)))

# # ✅ FASTA writer for full dataset (Label is included in the header)
# write_fasta <- function(df, filename) {
#   fasta_content <- unlist(lapply(1:nrow(df), function(i) {
#     paste0(">seq", i, "_label_", df$Label[i], "\n", df$Sequence[i])
#   }))
#   writeLines(fasta_content, filename)
# }

# # ✅ Write one combined FASTA file
# output_file <- "/home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/nontata/PB/main.fa"
# write_fasta(data, output_file)

# cat("✅ Combined FASTA file created at:", output_file, "\n")


# #____________________________________________________________



# data <- read.csv("/home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/nontata/PWM/new_merged_file_PWM.tsv", 
#                  sep = "\t", header = TRUE, stringsAsFactors = FALSE)

# # Confirm required columns exist
# stopifnot(all(c("Sequence", "Label") %in% colnames(data)))

# # ✅ FASTA writer for full dataset (Label is included in the header)
# write_fasta <- function(df, filename) {
#   fasta_content <- unlist(lapply(1:nrow(df), function(i) {
#     paste0(">seq", i, "_label_", df$Label[i], "\n", df$Sequence[i])
#   }))
#   writeLines(fasta_content, filename)
# }

# # ✅ Write one combined FASTA file
# output_file <- "/home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/nontata/PWM/main.fa"
# write_fasta(data, output_file)

# cat("✅ Combined FASTA file created at:", output_file, "\n")


# #____________________________________________________________

