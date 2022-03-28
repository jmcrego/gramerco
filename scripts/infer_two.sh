#!/bin/bash

source ~/anaconda3/bin/activate gramerco

MOSES=/nfs/RESEARCH/bouthors/packages/mosesdecoder/scripts
DETOKENIZER=$MOSES/tokenizer/detokenizer.perl

DATA_NAME=AFP
DATA_DIR=../resources
DATA_BIN=$DATA_DIR/$DATA_NAME/$DATA_NAME-bin-3

# DATA_SRC=$DATA_DIR/dictates/clean/dicts.err
# OUT_TAGS=$DATA_DIR/evals/dictates/dictates-gec-infl.tag
# OUT_TXT=$DATA_DIR/evals/dictates/dictates-gec-infl.cor

DATA_SRC=/nfs/RESEARCH/bouthors/projects/gramerco/resources/AFP/AFP-noise-lev/subtest.noise
OUT_TAGS=/nfs/RESEARCH/bouthors/projects/gramerco/resources/evals/synthetic/subtest-gec-infl.tag
OUT_TXT=/nfs/RESEARCH/bouthors/projects/gramerco/resources/evals/synthetic/subtest-gec-infl.cor

SAVE_PATH=$DATA_DIR/models/gramerco-two-fr
mkdir -p $SAVE_PATH
mkdir -p $SAVE_PATH/tensorboard

CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
python infer_two.py \
      --file $DATA_SRC \
      --log DEBUG \
      --save $SAVE_PATH \
      --inflection-layer \
      --model-id word-index-infl-inflection-layer \
      --lex $DATA_DIR/Lexique383.tsv \
      --voc $DATA_DIR/common/french.dic.50k \
      --tokenizer flaubert/flaubert_base_cased \
      --batch-size 20 \
      --out-tags $OUT_TAGS \
      --samepos \
      --gpu \
      | perl $DETOKENIZER -l fr -q \
      | sed -r 's/# # #/###/g' \
      | sed -z 's/\n\n/\n/g' \
      > $OUT_TXT

      # --gpu-id 1
      # --text "Le Grève àla principaux raffineries de pétrole de Koweit pourriez mené à une révolte." \
