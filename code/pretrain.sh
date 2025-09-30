export KMER=6
export LC_ALL=en_US.utf8
export LANG=en_US.utf8
# cd examples
export KMER=6
export TRAIN_FILE=/data/projects/dna/DNABERT/examples/sample_data/pre/6_3k.txt
export TEST_FILE=/data/projects/dna/DNABERT/examples/sample_data/pre/6_3k.txt


export SOURCE=/data/projects/dna/DNABERT/
export OUTPUT_PATH=/data/projects/Nimisha/home_wand_dna
export CUDA_VISIBLE_DEVICES=1

python /data/projects/dna/DNABERT/examples/run_pretrain_Orbin_masked.py \
    --output_dir $OUTPUT_PATH \
    --model_type=dna \
    --tokenizer_name=dna$KMER \
    --config_name=$SOURCE/src/transformers/dnabert-config/bert-config-$KMER/config.json \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --gradient_accumulation_steps 25 \
    --per_gpu_train_batch_size 12 \
    --per_gpu_eval_batch_size 11 \
    --save_steps 500 \
    --save_total_limit 20 \
    --max_steps 200 \
    --evaluate_during_training \
    --logging_steps 500 \
    --line_by_line \
    --learning_rate 4e-4 \
    --block_size 512 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --beta1 0.9 \
    --beta2 0.98 \
    --mlm_probability 0.025 \
    --warmup_steps 400 \
    --overwrite_output_dir \
    --n_process 24