# Install required packages (if not already installed)
install.packages("data.table", repos = "https://cloud.r-project.org/")
install.packages("matrixStats", repos = "https://cloud.r-project.org/")  # for rowMedians

# Load libraries
library(DNAshapeR)
library(Biostrings)
library(data.table)
library(matrixStats)

# Input FASTA file
full_fn <- "/home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/TFBS/CHD4/main.fa"

# Read the entire file
all_seqs <- readDNAStringSet(full_fn)
seq_names <- names(all_seqs)
seqs_char <- as.character(all_seqs)

# Compute all shape features at once
shape_features <- c("MGW", "Roll", "ProT", "HelT", "EP", 
                   "Rise", "Shift", "Slide", "Tilt", "Buckle",
                   "Opening", "Shear", "Stagger", "Stretch")

all_shapes <- getShape(full_fn, shapeType = shape_features)

# Initialize output dataframe
final_df <- data.frame(SeqName = seq_names, Sequence = seqs_char, stringsAsFactors = FALSE)

# Define summary type (median for entire sequence)
summary_type <- "median"

# Loop over all shape features and calculate median for entire sequence
for (feature in names(all_shapes)) {
  shape_data <- all_shapes[[feature]]
  
  if (is.matrix(shape_data)) {
    # Calculate median across all positions for each sequence
    summary_vals <- rowMedians(shape_data, na.rm = TRUE)
    
    # Create column name
    colname <- paste0(feature, "_", summary_type, "_fullseq")
    
    # Add to final dataframe
    final_df[[colname]] <- summary_vals
  } else {
    message("Skipping feature: ", feature, " (not a matrix)")
  }
}

# Output file
output_dir <- "/home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/TFBS/CHD4/inc"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

final_outfile <- file.path(output_dir, "full_shape_summary_CHD4_fullseq_median.csv")
fwrite(final_df, final_outfile)

# Done
cat("✅ All done. Final shape data saved!\n")
cat("Shape of final data: ", dim(final_df)[1], " rows x ", dim(final_df)[2], " columns\n")
cat("Output saved to:", final_outfile, "\n")