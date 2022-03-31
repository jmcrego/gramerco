# gramerco

The network fine-tunes a BERT-like french language model to perform a gramatical error correction task.

Thus, input is a noisy french text while output is a set of tags (one for each input word) that indicates wether the word is correct or if needs to be corrected following the output tag. The next tags are considered:

| Tag                     | Description                                                                              | Pattern      |
| ----------------------- | ---------------------------------------------------------------------------------------- | ------------ |
| ·                       | Keep current token                                                                       | (X -> X)     |
| $DELETE                 | Erase current token                                                                      | (X -> )      |
| $SWAP                   | Swap current and next tokens                                                             | (X Y -> Y X) |
| $MERGE                  | Merge current and next tokens                                                            | (X Y -> XY)  |
| $SPLIT_\<tok>           | Divide current token by space    according to prefix token                               | (XY -> X Y)  |
| $HYPHEN-MERGE           | Merge with an hyphen current and next tokens                                             | (X Y -> X-Y) |
| $HYPHEN-SPLIT           | Divide current token by the hyphen                                                       | (X-Y -> X Y) |
| $REPLACE:\<type>_\<tok> | Replace current token by tok, where pos is the gramatical category of the original token | (X -> tok)   |
| $APPEND_\<tok>          | Append tok to current token                                                              | (X -> X tok) |
| $INFLECT:\<inflection>  | Apply the g-transform/inflection to current token                                        | (X -> X')    | (X -> X') |
| $CASE:FIRST             | Flip the case of current token first character                                           | (X -> X')    |
| $CASE:UPPER             | Set case to upper for the whole token                                                    | (X -> X')    |
| $CASE:LOWER             | Set case to lower for the whole token                                                    | (X -> X')    |


There are several models (see [gec_bert.py](./model_gec/gec_bert.py)) to encompass the different tags, with corresponding tag encoders (see [tag_encoder.py](./tag_encoder.py)) :
- Complete tag enumeration in a layer
- 2 separate layers :
  - one for the tags, merging the tags requiring tokens in one single tag ($REPLACE:SPELL_\<token> $\rightarrow$ $REPLACE:SPELL)
  - one for exclusively token prediction along a set of vocabulary (words)

- 3 separate layers :
  - one for the tag categories, with tags requiring tokens and inflections merged in single representativve tags
  - one for exclusively token prediction along a set of vocabulary (words)
  - one for exclusively inflection prediction along a set of possible inflections (given by spacy)

## Preprocessing

For training, the system requires data already annotated with tags. The pipeline code contains part of useful scripts to prepare data (see [data_process_pipeline.sh](./data_process_pipeline.sh)).
An other one is used for Levenshtein Transformer ([data_process_pipeline_lev.sh](./data_process_pipeline.sh)))

## Training

Do not hesitate to adapt the arguments (data, training arguements) to your needs.

First model case :
```
bash train.sh
```
Second and third model case :
```
bash train_two.sh
```
Levenstein model case :
```
bash train_lev.sh
```

## Inference

First model case :
```
bash infer.sh
```
Second and third model case :
```
bash infer_two.sh
```
Levenstein model case :
```
bash infer_lev.sh
```