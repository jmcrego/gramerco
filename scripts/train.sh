#!/bin/bash

source ~/anaconda3/bin/activate gramerco

DATA_NAME=AFP
DATA_DIR=../resources
DATA_BIN=$DATA_DIR/$DATA_NAME/$DATA_NAME-bin

SAVE_PATH=$DATA_DIR/models/gramerco-fr
mkdir -p $SAVE_PATH
mkdir -p $SAVE_PATH/tensorboard

CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:200 \
python train.py $DATA_BIN/$DATA_NAME \
      --log DEBUG \
      --model-type decision2 \
      --save $SAVE_PATH \
      --continue-from none \
      --lex $DATA_DIR/Lexique383.tsv \
      --app $DATA_DIR/$DATA_NAME/$DATA_NAME-lex/lexique.app \
      --tokenizer flaubert/flaubert_base_cased \
      --num-workers 10 \
      --tensorboard \
      -lang fr \
      --max-tokens 4096 \
      --max-sentences 128 \
      --required-batch-size-multiple 8 \
      --max-positions 510 \
      --n-epochs 1 \
      -lr 0.00001 \
      --valid-iter 5000  \
      --early-stopping 5 \
      --gpu \
      --ignore-clean \
      --valid \
      --test \