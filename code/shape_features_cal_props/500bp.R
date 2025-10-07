# ==========================
# 📦 Install + Load Packages
# ==========================
if (!requireNamespace("data.table")) install.packages("data.table", repos = "https://cloud.r-project.org/")
if (!requireNamespace("matrixStats")) install.packages("matrixStats", repos = "https://cloud.r-project.org/")
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
if (!requireNamespace("Biostrings", quietly = TRUE)) BiocManager::install("Biostrings")
if (!requireNamespace("DNAshapeR", quietly = TRUE)) BiocManager::install("DNAshapeR")

library(data.table)
library(matrixStats)
library(DNAshapeR)
library(Biostrings)

# ====================
# 📂 Define Base Paths
# ====================
mark <- "H3K27ac"
base_input_dir <- "/home/shared/rdavuluri/TFBS_data_histone_marks"
base_output_dir <- "/home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/500_TFBS"

mark_input <- file.path(base_input_dir, mark)
mark_output <- file.path(base_output_dir, mark, "500bp_balanced")
mark_output_inc <- file.path(mark_output, "inc")

# ===============================
# 🔧 Helper Functions
# ===============================
reconstruct_sequence_from_kmers <- function(kmers_str, k = 6) {
  if (is.na(kmers_str) || nchar(kmers_str) == 0) return(NA)
  kmers <- unlist(strsplit(kmers_str, " "))
  if (length(kmers) == 0) return(NA)
  sequence <- kmers[1]
  for (i in 2:length(kmers)) {
    if (nchar(kmers[i]) < k) return(NA)
    sequence <- paste0(sequence, substr(kmers[i], k, k))
  }
  return(sequence)
}

write_fasta <- function(df, filename) {
  fasta_lines <- c()
  for (i in 1:nrow(df)) {
    header <- paste0(">seq", i, "_label_", df$Label[i])
    seq <- toupper(as.character(df$Sequence[i]))
    if (isTRUE(!is.na(seq) && nchar(seq) > 0 && grepl("^[ACGT]+$", seq))) {
      fasta_lines <- c(fasta_lines, header, seq)
    } else {
      cat("⚠️ Skipping invalid or missing sequence at row", i, "\n")
    }
  }
  writeLines(fasta_lines, filename)
}

calc_median_abs_dev <- function(shape_vector) {
  if (all(is.na(shape_vector))) return(NA)
  med <- median(shape_vector, na.rm = TRUE)
  mad <- median(abs(shape_vector - med), na.rm = TRUE)
  return(mad)
}

# ======================
# 🔁 Process H3K27ac
# ======================

cat("\n===========================\n")
cat("🔄 Starting Mark:", mark, "\n")

# Check if already done
final_shape_files_exist <- all(file.exists(
  c(
    file.path(mark_output_inc, paste0("full_shape_median_abs_dev_", mark, ".csv")),
    file.path(mark_output_inc, paste0("shape_diffsum_", mark, ".csv")),
    file.path(mark_output_inc, paste0("shape_diffmean_", mark, ".csv"))
  )
))

if (final_shape_files_exist) {
  cat("⏩ Already processed", mark, "- skipping.\n")
} else {
  # Create output directories
  dir.create(mark_output, recursive = TRUE, showWarnings = FALSE)
  dir.create(mark_output_inc, recursive = TRUE, showWarnings = FALSE)

  train_path <- file.path(mark_input, "train.tsv")
  dev_path <- file.path(mark_input, "dev.tsv")

  if (!file.exists(train_path) || !file.exists(dev_path)) {
    cat("❌ Missing train/dev for", mark, "- skipping.\n")
  } else {
    # 📥 Read train/dev files assuming no header
    df1 <- fread(train_path, header = FALSE)
    df2 <- fread(dev_path, header = FALSE)

    if (ncol(df1) >= 2) {
      setnames(df1, c("Sequence", "Label"))
      setnames(df2, c("Sequence", "Label"))
    } else {
      cat("❌ Unexpected number of columns in", mark, "- skipping.\n")
      next
    }

    df_combined <- rbind(df1, df2)

    # 🛠 Reconstruct sequences
    df_combined[, Sequence := sapply(Sequence, reconstruct_sequence_from_kmers)]
# 💾 Save merged data with raw kmers and labels
    kmers_raw_path <- file.path(mark_output, paste0("raw_kmers_labels_", mark, ".tsv"))
    fwrite(df_combined, kmers_raw_path, sep = "\t")
    cat("✅ Raw kmers+labels saved at:", kmers_raw_path, "\n")

    if (!"Sequence" %in% colnames(df_combined) || all(is.na(df_combined$Sequence))) {
      cat("❌ Sequence reconstruction failed in", mark, "- skipping.\n")
    } else {
      # 💾 Save intermediate kmers merged file
      kmers_file <- file.path(mark_output, paste0("kmers_merged_file_", mark, ".tsv"))
      fwrite(df_combined, kmers_file, sep = "\t")

      # 🏷️ Check Label column
      if (!"Label" %in% colnames(df_combined)) {
        possible_labels <- c("class", "Class", "target", "Target", "type", "Type")
        label_found <- intersect(possible_labels, colnames(df_combined))
        if (length(label_found) > 0) {
          setnames(df_combined, label_found[1], "Label")
          cat("⚠️ Used alternative label column:", label_found[1], "\n")
        } else {
          cat("❌ 'Label' column missing in", mark, "- skipping.\n")
        }
      }

      # 🧹 Filter invalid sequences
      df_combined <- df_combined[!is.na(Sequence) & Sequence != "", ]
      df_combined <- df_combined[grepl("^[ACGT]+$", Sequence), ]

      if (nrow(df_combined) == 0) {
        cat("❌ No valid sequences found in", mark, "- skipping.\n")
      } else {
        # 🧬 Write FASTA
        fasta_path <- file.path(mark_output, "main.fa")
        write_fasta(df_combined, fasta_path)
        cat("✅ FASTA created at:", fasta_path, "\n")

        # 🔬 Compute DNA shape features
        shape_features <- c("MGW", "Roll", "ProT", "HelT", "EP", 
                            "Rise", "Shift", "Slide", "Tilt", "Buckle",
                            "Opening", "Shear", "Stagger", "Stretch")

        all_shapes <- tryCatch({
          getShape(fasta_path, shapeType = shape_features)
        }, error = function(e) {
          cat("❌ getShape failed for", mark, ":", e$message, "\n")
          return(NULL)
        })

        if (!is.null(all_shapes)) {
          all_seqs <- readDNAStringSet(fasta_path)
          final_df <- data.frame(SeqName = names(all_seqs), Sequence = as.character(all_seqs), stringsAsFactors = FALSE)
          final_df_diffsum <- final_df
          final_df_diffmean <- final_df

          for (feature in names(all_shapes)) {
            shape_data <- all_shapes[[feature]]
            if (is.matrix(shape_data)) {
              mad_vec <- apply(shape_data, 1, calc_median_abs_dev)
              final_df[[paste0(feature, "_median_abs_dev")]] <- mad_vec

              diff_sum <- rowSums(abs(t(apply(shape_data, 1, diff))), na.rm = TRUE)
              diff_mean <- rowMeans(abs(t(apply(shape_data, 1, diff))), na.rm = TRUE)

              final_df_diffsum[[paste0(feature, "_diffsum")]] <- diff_sum
              final_df_diffmean[[paste0(feature, "_diffmean")]] <- diff_mean
            } else {
              message("Skipping feature: ", feature, " (not a matrix)")
            }
          }

          # 📁 Save all shape-based summaries
          fwrite(final_df,       file.path(mark_output_inc, paste0("full_shape_median_abs_dev_", mark, ".csv")))
          fwrite(final_df_diffsum, file.path(mark_output_inc, paste0("shape_diffsum_", mark, ".csv")))
          fwrite(final_df_diffmean, file.path(mark_output_inc, paste0("shape_diffmean_", mark, ".csv")))

          cat("✅ Shape summary saved for", mark, "\n")
        }
      }
    }
  }
}
