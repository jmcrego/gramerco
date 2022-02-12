#!/bin/bash
source ~/anaconda3/bin/activate gramerco

DATA_NAME=AFP
#DATA_NAME=debug
DATA_DIR=../resources
DATA_RAW=$DATA_DIR/$DATA_NAME/$DATA_NAME-raw
DATA_LEX=$DATA_DIR/$DATA_NAME/$DATA_NAME-lex-3
DATA_NOISE=$DATA_DIR/$DATA_NAME/$DATA_NAME-noise-3
DATA_BIN=$DATA_DIR/$DATA_NAME/$DATA_NAME-bin-3

VOC=$DATA_DIR/common/french.dic.50k

mkdir -p $DATA_LEX
mkdir -p $DATA_NOISE
mkdir -p $DATA_BIN

#cat data/*.txt | python3 scripts/txt2dic.py -minf 5 > french.dic 2> log.dic &
#ls data/*.txt | parallel -j 32 "python3 scripts/noise_lex.py -adj -nom -ver Lexique383.tsv {} > {}.lex 2> {}.log" &> log.parallel &
#ls data/HEAD*.lex | parallel -j 32 "python3 scripts/add_french_noise.py -dic french.dic -rep Lexique383.tsv -app Lexique383.app {} > {}.noise" &> log.parallel &

# python noiser/txt2app.py $DATA_DIR/Lexique383.tsv -pro -adv -pre -pun > $DATA_LEX/lexique.app
# echo "app done"
#
# cat $DATA_RAW/*.txt | python noiser/txt2dic.py -minf 5 > $DATA_LEX/$DATA_NAME.dic 2> $DATA_LEX/log.dic &
# echo "dic done"
#
# ls $DATA_RAW/*.txt | parallel -j 32 'python noiser/noise_lex.py ../resources/Lexique383.tsv {} > ../resources/AFP/AFP-lex/{/.}.lex' 2> $DATA_LEX/log.parallel &
# echo "noise lex done"

# ls $DATA_LEX/*.lex | parallel -j 32 'python noiser/add_french_noise.py -dic ../resources/AFP/AFP-lex/AFP.dic -rep ../resources/Lexique383.tsv -app ../resources/AFP/AFP-lex/lexique.app {} --out ../resources/AFP/AFP-lex/{/.}.noise' &> $DATA_LEX/log.parallel2

### v2

# ls $DATA_RAW/*.txt | \
# 	parallel -j 32 "python3 noiser/noise.py --vocab ../resources/common/french.dic.20k --lexicon ../resources/Lexique383.tsv --p_clean 0 {} > ../resources/AFP/AFP-lex-2/{/.}.noise 2> {}.log" &> log.parallel
#

# echo "combining now"
#
# cat $DATA_LEX/*.noise > $DATA_LEX/$DATA_NAME.all.noise
#
# echo "Extracting subset of sentences"
#
# head -n 50000000 $DATA_LEX/$DATA_NAME.all.noise > $DATA_LEX/$DATA_NAME.noise

#python data/generate_dataset.py $DATA_LEX/$DATA_NAME.noise --word-index -to $DATA_NOISE/$DATA_NAME

#echo "splitting"

#python data/split.py $DATA_NOISE/$DATA_NAME -dev 0.002 -test 0.002 --word-index

# echo "process + binarize"
#
 python data/preprocess.py $DATA_NOISE/$DATA_NAME.train --word-index --version 2 -log info -lex $DATA_DIR/Lexique383.tsv -app $VOC --num-workers 2 -out $DATA_BIN/$DATA_NAME.train
 python data/preprocess.py $DATA_NOISE/$DATA_NAME.dev --word-index --version 2 -log info -lex $DATA_DIR/Lexique383.tsv -app $VOC --num-workers 2 -out $DATA_BIN/$DATA_NAME.dev
 python data/preprocess.py $DATA_NOISE/$DATA_NAME.test --word-index --version 2 -log info -lex $DATA_DIR/Lexique383.tsv -app $VOC --num-workers 2 -out $DATA_BIN/$DATA_NAME.test
