#!/bin/bash

source ~/anaconda3/bin/activate gramerco

MOSES=/nfs/RESEARCH/bouthors/packages/mosesdecoder/scripts
DETOKENIZER=$MOSES/tokenizer/detokenizer.perl
DE_BPE=/nfs/RESEARCH/bouthors/packages/Tokenizer/build/cli/detokenize
BPE=/nfs/RESEARCH/crego/projects/gramerco/onmt-fr/bpe.32k

DATA_NAME=AFP
DATA_DIR=../resources
DATA_BIN=$DATA_DIR/$DATA_NAME/$DATA_NAME-bin-lev

SAVE_PATH=$DATA_DIR/models/levenshtein
MODEL_NAME=levT-full-GEC-1


DATA_SRC=/nfs/RESEARCH/bouthors/projects/gramerco/resources/AFP/AFP-bpe-lev/bpe.subtest.noise
DATA_INT=/nfs/RESEARCH/bouthors/projects/gramerco/resources/evals/synthetic/bpe.subtest.txt
DATA_OUT=/nfs/RESEARCH/bouthors/projects/gramerco/resources/evals/synthetic/subtest-lev

# DATA_SRC_RAW=/nfs/RESEARCH/bouthors/projects/gramerco/resources/dictates/clean/dicts.err
# DATA_SRC=/nfs/RESEARCH/bouthors/projects/gramerco/resources/dictates/bpe/bpe.dicts.err
# DATA_INT=/nfs/RESEARCH/bouthors/projects/gramerco/resources/evals/dictates/bpe.dicates-lev.txt
# DATA_OUT=/nfs/RESEARCH/bouthors/projects/gramerco/resources/evals/dictates/dictates-lev


# python data/apply_bpe.py --input-file $DATA_SRC_RAW --output-file $DATA_SRC --bpe-model $BPE

#python data/apply_bpe.py --input-file /nfs/RESEARCH/bouthors/projects/gramerco/resources/AFP/AFP-noise-lev/test.fr --output-file /nfs/RESEARCH/bouthors/projects/gramerco/resources/AFP/AFP-bpe-lev/bpe.test.fr --bpe-model $BPE

#python data/apply_bpe.py --input-file /nfs/RESEARCH/bouthors/projects/gramerco/resources/AFP/AFP-noise-lev/test.noise --output-file /nfs/RESEARCH/bouthors/projects/gramerco/resources/AFP/AFP-bpe-lev/bpe.test.noise --bpe-model $BPE

CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
fairseq-interactive $DATA_BIN \
  --task correction_lev_full \
  --path $SAVE_PATH/$MODEL_NAME/checkpoint_best.pt \
  --beam 5 \
  --buffer-size 1024 \
  --source-lang noise \
  --target-lang fr \
  --iter-decode-max-iter 9 \
  --iter-decode-eos-penalty 1 \
  --remove-bpe \
  --input $DATA_SRC \
  | sed -z 's/\n\n/\n/g' \
  | tee $DATA_INT

grep ^H  $DATA_INT | cut -f3- | $DE_BPE | perl $DETOKENIZER -l fr -q | tqdm > $DATA_OUT.cor
grep ^S  $DATA_INT | cut -f2- | $DE_BPE | perl $DETOKENIZER -l fr -q | tqdm > $DATA_OUT.err
