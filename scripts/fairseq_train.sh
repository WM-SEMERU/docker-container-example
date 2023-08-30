export TOTAL_NUM_UPDATES=100000
export WARMUP_UPDATES=10000
export LR=4.2e-05
export UPDATE_FREQ=8
export DIR=/workspaces/code-rationales/datax/bart_fairseq/checkpoint_dir_base
export MAX_TOKENS=1024
export DATA_DIR=/workspaces/code-rationales/datax/methods2test/corpus/preprocessed/fm_fc_ms_ff/bin
export SRC_LANG=input.methods
export TRG_LANG=output.tests

export CUDA_VISIBLE_DEVICES="0,1,2,3"

fairseq-train $DATA_DIR \
    --max-tokens $MAX_TOKENS \
    --max-tokens-valid $MAX_TOKENS \
    --task translation --source-lang $SRC_LANG --target-lang $TRG_LANG \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --arch bart_large \
    --criterion cross_entropy \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam \
    --reset-optimizer \
    --reset-lr-scheduler \
    --clip-norm 0.1 \
    --lr-scheduler inverse_sqrt --lr $LR --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --truncate-source \
    --skip-invalid-size-inputs-valid-test \
    --save-dir $DIR/models \
    --save-interval 1 --fp16-scale-window 512 --fp16-init-scale 2 \
    --fp16 --adam-betas '(0.9,0.98)' --adam-eps 1e-6 \
    --no-epoch-checkpoints \
    --ddp-backend=no_c10d \
    --tensorboard-logdir $DIR/tensorboard \
    --word-dropout-mixture 0.5 --word-dropout-type inverse_length