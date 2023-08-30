MODEL=bigcode/santacoder  
CACHE_DIR=/workspaces/code-rationales/datax/df_cache_dir_v2
CHECKPOINT_DIR=/workspaces/code-rationales/data/santacoder/checkpoints
LOGGING_DIR=/workspaces/code-rationales/scripts/compatibility/santacoder/logs
nohup python3 -u /workspaces/code-rationales/sequential-rationales/huggingface/examples/pytorch/language-modeling/run_clm_santacoder.py \
    --model_name_or_path $MODEL \
    --do_train \
    --do_eval \
    --dataset_name bigcode/the-stack \
    --logging_dir $LOGGING_DIR \
    --output_dir $CHECKPOINT_DIR \
    --per_device_train_batch_size 1 \
    --evaluation_strategy steps --eval_steps 5 \
    --num_train_epochs 50 \
    --lr_scheduler_type constant \
    --learning_rate 0.00002 \
    --block_size 2048 \
    --per_device_eval_batch_size 4 \
    --save_total_limit 2 \
    --max_steps 10 \
    --word_dropout_mixture 0.5 \
    --cache_dir  $CACHE_DIR\
    --max_samples_stack 10000\
    > /workspaces/code-rationales/scripts/compatibility/santacoder/logs/output.txt &