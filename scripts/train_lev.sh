#!/bin/bash

source ~/anaconda3/bin/activate gramerco

RESOURCES=/nfs/RESEARCH/bouthors/projects/gramerco/resources
DATA_BIN=$RESOURCES/AFP/AFP-bin-lev
DICT=$RESOURCES/common/dict.fr.txt
SAVE_PATH=$RESOURCES/models/levenshtein
MODEL_NAME=levT-full-GEC-1

mkdir -p $SAVE_PATH/$MODEL_NAME
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 fairseq-train $DATA_BIN \
  --ddp-backend=legacy_ddp \
  --arch correction_levenshtein_transformer_full \
  --task correction_lev_full \
  --criterion nat_loss \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --clip-norm 0.0 \
  --lr 0.00005 \
  --lr-scheduler inverse_sqrt \
  --stop-min-lr '1e-9' \
  --warmup-init-lr '1e-07' \
  --warmup-updates 4000 \
  --label-smoothing 0.1 \
  --dropout 0.3 \
  --weight-decay 0.001 \
  --max-tokens 16384\
  --save-dir $SAVE_PATH/$MODEL_NAME \
  --tensorboard-logdir $SAVE_PATH/$MODEL_NAME \
  --required-batch-size-multiple 1 \
  --log-interval 200 \
  --save-interval-updates 3000 \
  --keep-interval-updates 10 \
  --no-epoch-checkpoints \
  --max-update 4000000 \
  --max-epoch 2000 \
  --validate-interval 5 \
  --num-workers 10 \
  --skip-invalid-size-inputs-valid-test \
  --source-lang noise \
  --target-lang fr \
  --fp16 \
  --share-all-embeddings \
  --share-decoder-input-output-embed \
  2> ./logs/$MODEL_NAME.log   1> ./logs/$MODEL_NAME.out
