#!/bin/bash
#SBATCH --job-name=shape_features
#SBATCH --output=shapefeature_Prediction_%j.log
#SBATCH --partition=extra           # <-- This is the key change
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00      # or 48:00:00 works too
#SBATCH --nodes=2              # since extra allows only 2
#SBATCH --ntasks-per-node=2    # tune based on actual CPU usage
#SBATCH --gres=gpu:2           # match GPU count



source ~/anaconda3/etc/profile.d/conda.sh

conda activate envR3


cd /home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/TFBS/

rm -rf /home/npapinen/anaconda3/envs/envR3/lib/R/library/00LOCK-data.table


Rscript /home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/ShapeFeatures/kmer_generate_single_TFBS.R



# # ================================
# # ✅ STEP 1:
# Rscript /home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/ShapeFeatures/tfbs_loop.R
# # ================================
# # ✅ STEP 2: 
# Rscript /home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/generate_fasta.R

# # # ================================
# # # ✅ STEP 3:
# Rscript /home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/ShapeFeatures/tfbs_calculation.R
# Rscript /home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/ShapeFeatures/45bp_sw10_tfbs.R

# Rscript /home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/ShapeFeatures/fullfeature.R
# Rscript /home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/ShapeFeatures/45bp_sw10_tfbs.R
# Rscript /home/npapinen/RAMANA_CEWIT_SERVER/DNABERT/pydna_epbd/shapefeatures/ShapeFeatures/difference_sum_shape_summary.R
