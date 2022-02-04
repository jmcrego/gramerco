#!/bin/bash

source ~/anaconda3/bin/activate gramerco

MOSES=/nfs/RESEARCH/bouthors/packages/mosesdecoder/scripts
DETOKENIZER=$MOSES/tokenizer/detokenizer.perl

DATA_NAME=AFP
DATA_DIR=../resources
DATA_BIN=$DATA_DIR/$DATA_NAME/$DATA_NAME-bin-2

# DATA_SRC=$DATA_DIR/dictates/test-dictates/dicts.err
# OUT_TAGS=$DATA_DIR/dictates/generated-dictates/dicts.tags

DATA_SRC=/nfs/RESEARCH/crego/projects/gramerco/data/valid-test-sets/HEADLINES_2018_8_fr.txt
OUT_TAGS=/nfs/RESEARCH/bouthors/projects/gramerco/resources/evals/io.tags

SAVE_PATH=$DATA_DIR/models/gramerco-two-fr
mkdir -p $SAVE_PATH
mkdir -p $SAVE_PATH/tensorboard

CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
python infer_two.py \
      --text "Je vais à pour voir la tour Eiffel." \
      --file $DATA_SRC \
      --log DEBUG \
      --save $SAVE_PATH \
      --model-id freeze20k+ls0.2+cumul4+rdm0.5-normal1 \
      --lex $DATA_DIR/Lexique383.tsv \
      --voc $DATA_DIR/common/french.dic.20k \
      --tokenizer flaubert/flaubert_base_cased \
      --batch-size 20 \
      --out-tags $OUT_TAGS \
      | perl $DETOKENIZER -l fr -q \
      | sed -r 's/# # #/###/g' \
      | sed -z 's/\n\n/\n/g' \
      # > $DATA_DIR/dictates/generated-dictates/dicts.infer.cor
      # --gpu \
      # --gpu-id 1
