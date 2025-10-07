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



rm -rf /home/npapinen/anaconda3/envs/envR3/lib/R/library/00LOCK-data.table


Rscript /~/code/shape_features_cal_props/500bp.R


# # ================================
# # ✅ STEP 1:
# Rscript /~/code/shape_features_cal_props/tfbs_loop.R
# # ================================
# # ✅ STEP 2: 
# Rscript /~/code/shape_features_cal_props/generate_fasta.R

# # # ================================
# # # ✅ STEP 3:
# Rscript /~/code/shape_features_cal_props/tfbs_calculation.R
# Rscript /~/code/shape_features_cal_props/45bp_sw10_tfbs.R

# Rscript /~/code/shape_features_cal_props/fullfeature.R
# Rscript /~/code/shape_features_cal_props/45bp_sw10_tfbs.R
# Rscript /~/code/shape_features_cal_props/difference_sum_shape_summary.R
