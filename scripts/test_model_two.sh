#!/bin/bash

# Script for evaluating model_two architectures

source ~/anaconda3/bin/activate gramerco

DATA_DIR=../resources
DATA_NAME=AFP
DATA_BIN=$DATA_DIR/$DATA_NAME/$DATA_NAME-bin-3
DATA_SRC=$DATA_DIR/$DATA_NAME/$DATA_NAME-noise-3/$DATA_NAME.test.noise.fr
DATA_TGT=$DATA_DIR/$DATA_NAME/$DATA_NAME-noise-3/$DATA_NAME.test.tag.fr

# DATA_NAME=bin
# DATA_BIN=$DATA_DIR/debug
# DATA_SRC=$DATA_BIN/debug.fr
# DATA_TGT=$DATA_BIN/debug.tag.fr

SAVE_PATH=$DATA_DIR/models/gramerco-two-fr

CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
python test_model_two.py \
      --data-bin $DATA_BIN/$DATA_NAME \
      --file-src $DATA_SRC \
      --file-tag $DATA_TGT \
      --inflection-layer \
      --sample 1000000 \
      --log DEBUG \
      --model-iter -1 \
      --k-best 20 \
      --save $SAVE_PATH \
      --model-id word-index-debug-infl-inflection-layer \
      --lex $DATA_DIR/Lexique383.tsv \
      --voc $DATA_DIR/common/french.dic.50k \
      --tokenizer flaubert/flaubert_base_cased \
      --batch-size 20 \
      --ignore-clean \
      --return-tag-voc \
      --raw \
      --word-index \
      --out-tags $DATA_DIR/evals/test-constraint.tags \
      # --gpu \
