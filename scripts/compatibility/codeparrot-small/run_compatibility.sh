MODEL=codeparrot/codeparrot-small
CACHE_DIR=/workspaces/code-rationales/datax/df_cache_dir
CHECKPOINT_DIR=/workspaces/code-rationales/data/codeparrot-small/checkpoints
LOGGING_DIR=/workspaces/code-rationales/scripts/compatibility/codeparrot-small/logs
nohup python3 -u /workspaces/code-rationales/sequential-rationales/huggingface/examples/pytorch/language-modeling/run_clm_codeparrot.py \
    --model_name_or_path $MODEL \
    --do_train \
    --do_eval \
    --dataset_name codeparrot/codeparrot-clean \
    --logging_dir $LOGGING_DIR \
    --output_dir $CHECKPOINT_DIR \
    --per_device_train_batch_size 1 \
    --evaluation_strategy steps --eval_steps 500 \
    --num_train_epochs 50 \
    --lr_scheduler_type constant \
    --learning_rate 0.00005 \
    --block_size 1024 \
    --per_device_eval_batch_size 4 \
    --save_total_limit 2 \
    --max_steps 30000 \
    --word_dropout_mixture 0.5 \
    --cache_dir  $CACHE_DIR\
    > /workspaces/code-rationales/scripts/compatibility/codeparrot-small/logs/output.txt &