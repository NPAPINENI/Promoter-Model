cd ..
export KMER=6
export DATA_NAME=100_percent_sequence_TATA
export CLASSES_NAME="TATA_OLD_diffsum_subsets_14"
export ARCHITECTURE=diffsum_subsets_14_TATAOLD
export MODEL_PATH=/data/projects/dna/6-new-12w-0 
export OUTPUT_PATH_BASE=/$DATA_NAME/$ARCHITECTURE

export DATA_PATH=/$DATA_NAME
export TB_PATH_BASE=/$DATA_NAME/$ARCHITECTURE
export SUMMARY_PATH_BASE=/$DATA_NAME/$ARCHITECTURE
export CUDA_VISIBLE_DEVICES=4
export WANDB_DIR=/path/

export CUDA_LAUNCH_BLOCKING=1,2,3,4,0,5,6,7

for wp in 0.1;
do 
    for wd in 0.001 ;
    do

        for lr_base in e-4 e-3 e-5 e-6;
        do
            for lr in  2$lr_base 3$lr_base 5$lr_base  1$lr_base 4$lr_base 6$lr_base;
            do
                echo "Running with wp=$wp wd=$wd lr=$lr"
                python /code/run_finetune_DNABERT-CoreProm.py \
                    --model_type dna \
                    --tokenizer_name=dna$KMER \
                    --model_name_or_path $MODEL_PATH \
                    --task_name dnaprom \
                    --additional_feature sequence \
                    --classes_name $CLASSES_NAME \
                    --architecture $ARCHITECTURE \
                    --do_train \
                    --do_eval \
                    --data_dir $DATA_PATH \
                    --max_seq_length 104 \
                    --per_gpu_eval_batch_size=200  \
                    --per_gpu_train_batch_size=200\
                    --learning_rate $lr \
                    --num_train_epochs 15 \
                    --output_dir $OUTPUT_PATH_BASE/$lr_base/$lr \
                    --tb_log_dir $TB_PATH_BASE/$lr_base/$lr \
                    --summary_dir $SUMMARY_PATH_BASE/$lr_base/$lr \
                    --evaluate_during_training \
                    --logging_steps 50 \
                    --save_steps 50 \
                    --warmup_percent $wp \
                    --hidden_dropout_prob 0.1 \
                    --overwrite_output \
                    --weight_decay $wd \
                    --wandb_tags PROMS_aditional_seq $ARCHITECTURE $DATA_NAME $lr_base\
                    --n_process 36
            done
        done
    done
done