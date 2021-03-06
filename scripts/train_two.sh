#!/bin/bash

source ~/anaconda3/bin/activate gramerco

DATA_NAME=AFP
DATA_DIR=../resources
DATA_BIN=$DATA_DIR/$DATA_NAME/$DATA_NAME-bin-3

SAVE_PATH=$DATA_DIR/models/gramerco-two-fr
mkdir -p $SAVE_PATH
mkdir -p $SAVE_PATH/tensorboard

# word-index-infl-rd-init

CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
python train_two.py $DATA_BIN/$DATA_NAME \
      --log DEBUG \
      --model-id word-index-infl \
      --inflection-layer \
      --save $SAVE_PATH \
      --continue-from last \
      --lex $DATA_DIR/Lexique383.tsv \
      --voc $DATA_DIR/common/french.dic.50k \
      --tokenizer flaubert/flaubert_base_cased \
      --num-workers 15 \
      --tensorboard \
      -lang fr \
      --max-tokens 8192 \
      --max-sentences 128 \
      --required-batch-size-multiple 1 \
      --min-positions 7 \
      --max-positions 510 \
      --n-epochs 5 \
      -lr 0.00001 \
      -ls 0.2 \
      --valid-iter 5000  \
      --early-stopping 10 \
      --ignore-clean \
      --freeze-encoder 1 \
      --grad-cumul-iter 2 \
      --random-keep-mask 0.5 \
      --valid \
      --test \
      --word-index \
      --gpu \
      --pretrained $DATA_DIR/models/pretrained/model_weights.pt
      # --encoder-random-init \
      # --pretrained $DATA_DIR/models/pretrained/model_weights.pt \
