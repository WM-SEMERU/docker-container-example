
MODEL=gpt2-medium
DATA_DIR=/workspaces/code-rationales/data/gpt2-medium/data
CHECKPOINT_DIR=/workspaces/code-rationales/data/gpt2-medium/checkpoints
LOGGING_DIR=/workspaces/code-rationales/scripts/compatibility/gpt2-medium/logs
nohup python3 -u /workspaces/code-rationales/sequential-rationales/huggingface/examples/pytorch/language-modeling/run_clm.py \
    --model_name_or_path $MODEL \
    --do_train \
    --do_eval \
    --train_file $DATA_DIR/train.txt \
    --validation_file $DATA_DIR/valid.txt \
    --logging_dir $LOGGING_DIR \
    --output_dir $CHECKPOINT_DIR \
    --per_device_train_batch_size 1 \
    --evaluation_strategy steps --eval_steps 500 \
    --num_train_epochs 50 \
    --lr_scheduler_type constant \
    --learning_rate 0.00001 \
    --block_size 512 \
    --per_device_eval_batch_size 4 \
    --save_total_limit 2 \
    --max_steps 45000 \
    --word_dropout_mixture 0.5 \
    > /workspaces/code-rationales/scripts/compatibility/gpt2-medium/logs/output.txt &