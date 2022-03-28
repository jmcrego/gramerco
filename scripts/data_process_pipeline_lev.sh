#!/bin/bash
source ~/anaconda3/bin/activate gramerco

DATA_NAME=AFP
DATA_DIR=../resources
DATA_BPE=$DATA_DIR/$DATA_NAME/$DATA_NAME-bpe-lev
DATA_NOISE=$DATA_DIR/$DATA_NAME/$DATA_NAME-noise-lev
DATA_BIN=$DATA_DIR/$DATA_NAME/$DATA_NAME-bin-lev
DATA_RAW=/nfs/RESEARCH/crego/projects/GECor/data

DICT=$DATA_DIR/common/dict.fr.txt

MOSES=/nfs/RESEARCH/bouthors/packages/mosesdecoder/scripts
DETOKENIZER=$MOSES/tokenizer/detokenizer.perl

BPE=/nfs/RESEARCH/crego/projects/gramerco/onmt-fr/bpe.32k


#echo "combining"

#echo "train"
#find $DATA_RAW -regex "$DATA_RAW/BODY_2017_\([1-9]\|10\)_fr.txt\.split_[a-z][a-z]\.noiseA_pairs" -exec cat {} + | tqdm | perl $DETOKENIZER -l fr -q > $DATA_NOISE/train.all
#awk -F "\"*\t\"*" '{print $1}' $DATA_NOISE/train.all > $DATA_NOISE/train.fr
#awk -F "\"*\t\"*" '{print $2}' $DATA_NOISE/train.all > $DATA_NOISE/train.noise
#echo "valid"
#find $DATA_RAW -regex "$DATA_RAW/BODY_2017_11_fr.txt\.split_[a-z][a-z]\.noiseA_pairs" -exec cat {} + | tqdm | perl $DETOKENIZER -l fr -q {} > $DATA_NOISE/valid.all
#awk -F "\"*\t\"*" '{print $1}' $DATA_NOISE/valid.all > $DATA_NOISE/valid.fr
#awk -F "\"*\t\"*" '{print $2}' $DATA_NOISE/valid.all > $DATA_NOISE/valid.noise
#echo "test"
#find $DATA_RAW -regex "$DATA_RAW/BODY_2017_12_fr.txt\.split_[a-z][a-z]\.noiseA_pairs" -exec cat {} + | tqdm | perl $DETOKENIZER -l fr -q {} > $DATA_NOISE/test.all
#awk -F "\"*\t\"*" '{print $1}' $DATA_NOISE/test.all > $DATA_NOISE/test.fr
#awk -F "\"*\t\"*" '{print $2}' $DATA_NOISE/test.all > $DATA_NOISE/test.noise



#for split in train valid test
#do
#	for ext in noise fr
#	do
#		echo $split.$ext
#		python data/apply_bpe.py --input-file $DATA_NOISE/$split.$ext --output-file $DATA_BPE/bpe.$split.$ext --bpe-model $BPE
#	done;
#done;

#onmt-build-vocab --save_vocab $DICT --size 32000 $DATA_BPE/train.*

#fairseq-preprocess \
#    --source-lang fr --target-lang noise \
#    --dict-only \
#    --joined-dictionary \
#    --trainpref $DATA_BPE/bpe.train \
#    --destdir $DICT \
#    --workers 20


fairseq-preprocess \
    --source-lang fr --target-lang noise \
    --srcdict $DICT \
    --joined-dictionary \
    --trainpref $DATA_BPE/bpe.train \
    --validpref $DATA_BPE/bpe.valid \
    --testpref $DATA_BPE/bpe.test \
    --destdir $DATA_BIN \
    --workers 20


