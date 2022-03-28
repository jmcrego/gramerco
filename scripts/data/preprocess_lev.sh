#!/bin/bash

source ~/anaconda3/bin/activate gramerco

INPUT=/nfs/RESEARCH/bouthors/projects/gramerco/resources/debug/debug-noise/debug
BPE=/nfs/RESEARCH/crego/projects/gramerco/onmt-fr/bpe.32k
OUTPUT=/nfs/RESEARCH/bouthors/projects/gramerco/resources/debug/debug-lex/debug
DATA_BIN=/nfs/RESEARCH/bouthors/projects/gramerco/resources/debug/debug-bin

DICT=/nfs/RESEARCH/bouthors/projects/gramerco/resources/common/dict.fr.txt

#for split in train dev test
#do
#	for ext in noise fr
#	do
#		python apply_bpe.py --input-file $INPUT.$split.$ext --output-file $OUTPUT.bpe.$split.$ext --bpe-model $BPE
#	done;
#done;

fairseq-preprocess \
    --source-lang fr --target-lang noise \
    --srcdict $DICT \
    --joined-dictionary \
    --trainpref $OUTPUT.bpe.train \
    --validpref $OUTPUT.bpe.train \
    --testpref $OUTPUT.bpe.test \
    --destdir $DATA_BIN \
    --workers 20
