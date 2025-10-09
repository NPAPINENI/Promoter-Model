# Install required packages (if not already installed)
install.packages("data.table", repos = "https://cloud.r-project.org/")
install.packages("matrixStats", repos = "https://cloud.r-project.org/")  # for rowMedians

# Load libraries
library(DNAshapeR)
library(Biostrings)
library(data.table)
library(matrixStats)

# Input FASTA file
full_fn <- "/home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/TFBS/CHD4_500bp/300bp_balanced/main.fa"

# Read the entire file
all_seqs <- readDNAStringSet(full_fn)
seq_names <- names(all_seqs)
seqs_char <- as.character(all_seqs)

# Compute all shape features at once
shape_features <- c("MGW", "Roll", "ProT", "HelT", "EP", 
                   "Rise", "Shift", "Slide", "Tilt", "Buckle",
                   "Opening", "Shear", "Stagger", "Stretch")

# all_shapes <- getShape(full_fn, shapeType = shape_features)

# # Initialize output dataframe
# final_df <- data.frame(SeqName = seq_names, Sequence = seqs_char, stringsAsFactors = FALSE)

# # Define window parameters
# window_size <- 45
# slide_step <- 10
# summary_type <- "median"

# # Function to calculate sliding window medians
# calculate_sliding_medians <- function(shape_vector, window_size, slide_step) {
#   seq_length <- length(shape_vector)
#   if (seq_length < window_size) return(NA)
  
#   start_pos <- seq(1, seq_length - window_size + 1, by = slide_step)
#   end_pos <- start_pos + window_size - 1
  
#   sapply(seq_along(start_pos), function(i) {
#     window <- shape_vector[start_pos[i]:end_pos[i]]
#     median(window, na.rm = TRUE)
#   })
# }

# # Loop over all shape features and calculate sliding window medians
# for (feature in names(all_shapes)) {
#   shape_data <- all_shapes[[feature]]
  
#   if (is.matrix(shape_data)) {
#     # Process each sequence's shape data
#     window_medians <- t(apply(shape_data, 1, function(seq_shape) {
#       calculate_sliding_medians(seq_shape, window_size, slide_step)
#     }))
    
#     # Name columns appropriately
#     num_windows <- ncol(window_medians)
#     start_pos <- seq(1, by = slide_step, length.out = num_windows)
#     end_pos <- start_pos + window_size - 1
    
#     col_names <- paste0(feature, "_", summary_type, "_window", window_size,
#                        "_slide", slide_step, "_", start_pos, "-", end_pos)
    
#     # Add to final dataframe
#     final_df[col_names] <- window_medians
#   } else {
#     message("Skipping feature: ", feature, " (not a matrix)")
#   }
# }

# # Output file
# output_dir <- "/home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/TFBS/CHD4_500bp/300bp_balanced/inc"
# dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# final_outfile <- file.path(output_dir, "full_shape_summary_CHD4_sliding_window_45bp_10sw.csv")
# fwrite(final_df, final_outfile)

# # Done
# cat("✅ All done. Final shape data with sliding window medians saved!\n")
# cat("Shape of final data: ", dim(final_df)[1], " rows x ", dim(final_df)[2], " columns\n")
# cat("Output saved to:", final_outfile, "\n")




all_shapes <- getShape(full_fn, shapeType = shape_features)

# Initialize output dataframe
final_df <- data.frame(SeqName = seq_names, Sequence = seqs_char, stringsAsFactors = FALSE)

# Function to calculate median absolute deviation from median
calc_median_abs_dev <- function(shape_vector) {
  if (all(is.na(shape_vector))) return(NA)
  med <- median(shape_vector, na.rm = TRUE)
  mad <- median(abs(shape_vector - med), na.rm = TRUE)
  return(mad)
}

# Loop over shape features and apply MAD calculation
for (feature in names(all_shapes)) {
  shape_data <- all_shapes[[feature]]
  
  if (is.matrix(shape_data)) {
    # Compute MAD for each sequence's shape values
    mad_vec <- apply(shape_data, 1, calc_median_abs_dev)
    
    # Add to final dataframe
    final_df[[paste0(feature, "_median_abs_dev")]] <- mad_vec
  } else {
    message("Skipping feature: ", feature, " (not a matrix)")
  }
}

# Output file
output_dir <- "/home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/TFBS/CHD4_500bp/300bp_balanced/inc"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

final_outfile <- file.path(output_dir, "full_shape_median_abs_dev_CHD4.csv")
fwrite(final_df, final_outfile)

cat("✅ All done. Final shape data with MADs saved!\n")
cat("Shape of final data: ", dim(final_df)[1], " rows x ", dim(final_df)[2], " columns\n")
cat("Output saved to:", final_outfile, "\n")


