#!/bin/bash

source ~/anaconda3/bin/activate gramerco

DATA_DIR=../resources
DATA_NAME=AFP
DATA_BIN=$DATA_DIR/$DATA_NAME/$DATA_NAME-bin-2
DATA_SRC=$DATA_DIR/$DATA_NAME/$DATA_NAME-noise-2/$DATA_NAME.test.noise.fr
DATA_TGT=$DATA_DIR/$DATA_NAME/$DATA_NAME-noise-2/$DATA_NAME.test.tag.fr

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
      --file-tag $DATA_TGT  \
      --sample 1000000 \
      --log DEBUG \
      --model-iter -1 \
      --save $SAVE_PATH \
      --model-id freeze20k+ls0.2+cumul4+rdm0.5-normal1 \
      --lex $DATA_DIR/Lexique383.tsv \
      --voc $DATA_DIR/common/french.dic.20k \
      --tokenizer flaubert/flaubert_base_cased \
      --batch-size 20 \
      --ignore-clean \
      --return-tag-voc \
      --raw \
      --out-tags $DATA_DIR/evals/test-debug.tags \
      # --gpu \
