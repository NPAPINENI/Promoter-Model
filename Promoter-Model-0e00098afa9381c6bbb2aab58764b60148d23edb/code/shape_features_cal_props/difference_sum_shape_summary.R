

install.packages("data.table", repos = "https://cloud.r-project.org/")

library(DNAshapeR)
library(Biostrings)
library(data.table)

# Input FASTA file
fn <- "/home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/TFBS/H3K27me3/main.fa"

# Read sequences
seqs <- readDNAStringSet(fn)
seqs_char <- as.character(seqs)
seq_names <- names(seqs)

# Core DNAshapeR features
core_features <- c("HelT", "MGW", "ProT", "Roll")
core_pred <- getShape(fn)

# Additional 9 structural features
extra_features <- c("Rise", "Shift", "Slide", "Tilt", "Buckle", 
                    "Opening", "Shear", "Stagger", "Stretch")

# Initialize prediction list with core
all_pred <- core_pred

# Add extra features
for (feature in extra_features) {
  all_pred[[feature]] <- getShape(fn, shapeType = feature)[[feature]]
}

# Initialize final data frame
final_df <- data.frame(
  SeqName = seq_names,
  Sequence = seqs_char,
  stringsAsFactors = FALSE
)

# Loop through all shape features and calculate per-nucleotide diff sum
for (feature in names(all_pred)) {
  shape_data <- all_pred[[feature]]
  
  if (is.matrix(shape_data)) {
    # Compute absolute differences between consecutive nucleotides
    diffs <- abs(t(apply(shape_data, 1, diff)))
    
    # Sum the differences per sequence
    diff_sums <- rowSums(diffs, na.rm = TRUE)
    
    # Add to final dataframe
    final_df[[paste0(feature, "_diffsum")]] <- diff_sums
  } else {
    message("Skipping feature: ", feature, " (not a matrix)")
  }
}

# Output directory
output_dir <- "/home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/TFBS/H3K27me3/fa"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Save the final summary CSV
fwrite(final_df, file.path(output_dir, "shape_summary_diff.csv"))

# Print dimensions
cat("DataFrame Shape: ", dim(final_df)[1], "rows x", dim(final_df)[2], "columns\n")
