from transformers import FlaubertTokenizer, FlaubertModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tqdm import tqdm
from data.gramerco_dataset import GramercoDataset
from model_gec.gec_bert import GecBertVocModel
from tag_encoder import TagEncoder2
from tokenizer import WordTokenizer
import logging
import matplotlib.pyplot as plt
from noiser.Noise import Lexicon
from collections import defaultdict
import os
import sys
import re
import difflib
import Levenshtein
import math


separ = '￨'


MOOD_MAP = {
    "ind": "Ind",
    "imp": "Imp",
    "cnd": "Cnd",
    "sub": "Sub",
}

TENSE_MAP = {
    "pre": "Pres",
    "pas": "Past",
    "fut": "Fut",
    "imp": "Imp",
}

NBR_MAP = {
    "s": "Sing",
    "p": "Plur",
}


def read_lexicon(f, vocab):
    lem2wrd = defaultdict(set)
    wrd2lem = defaultdict(set)
    pho2wrd = defaultdict(set)
    wrd2pho = defaultdict(set)
    wrd2wrds_same_lemma = defaultdict(set)
    wrd2wrds_homophones = defaultdict(set)
    lempos2wrds = defaultdict(set)
    lempos2wrdsinf = defaultdict(set)
    lemposinf2wrd = defaultdict(set)
    if not f:
        return wrd2wrds_same_lemma, wrd2wrds_homophones, lempos2wrds

    with open(f, 'r') as fd:
        for l in fd:
            toks = l.rstrip().split('\t')
            if len(toks) < 3:
                continue
            wrd, pho, lem, pos, gender, nbre, conj = toks[0], toks[1], toks[2], toks[3], toks[4], toks[5], toks[10]
            in_vocab = wrd in vocab
            # if wrd not in vocab:
            #     continue
            if ' ' in wrd or ' ' in lem or ' ' in pho or ' ' in pos:
                continue

            if pos == "VER":
                pos = "VERB"  # use same tag as SpaCy
            elif pos.startswith("ADJ"):
                pos = "ADJ"
            elif pos.startswith("PRO"):
                pos = "PRON"
            elif pos == "NOM":
                pos = "NOUN"
            elif pos == "PRE":
                pos = "ADP"
            if in_vocab:
                lem2wrd[lem].add(wrd)
            wrd2lem[wrd].add(lem)
            if in_vocab:
                pho2wrd[pho].add(wrd)
            wrd2pho[wrd].add(pho)
            if in_vocab:
                lempos2wrds[lem + separ + pos].add(wrd)
            lempos2wrdsinf[lem + separ + pos].add((gender, nbre, conj))
            lemposinf2wrd[separ.join([lem, pos, gender, nbre, conj])].add(wrd)

    for wrd in wrd2lem:
        for lem in wrd2lem[wrd]:
            for w in lem2wrd[lem]:
                if w == wrd:
                    continue
                wrd2wrds_same_lemma[wrd].add(w)

    for wrd in wrd2pho:
        for pho in wrd2pho[wrd]:
            for w in pho2wrd[pho]:
                if w == wrd:
                    continue
                wrd2wrds_homophones[wrd].add(w)

    return {
        "common_lemma": wrd2wrds_same_lemma,
        "wrd2lem": wrd2lem,
        "homophones": wrd2wrds_homophones,
        "words_from_lemma": lempos2wrds,
        "lempos2wrdsinf": lempos2wrdsinf,
        "lemposinf2wrd": lemposinf2wrd,
    }


def inflect_tag_to_dict(tag):

    d = {
        e.split('=')[0]: e.split('=')[1] for e in tag.split(';')[1:]
        if '=' in e
    }

    d["POS"] = tag.split(';')[0]
    return d


def get_idx_from_possibilities(tagger, possibilities, voc_best=None):
    if voc_best is not None:
        words = [tagger.worder.id_to_word[i] for i in voc_best.numpy()]
    else:
        words = tagger.worder.word_to_id
    ids = [
        tagger.worder.word_to_id[word]
        for word in possibilities
        if word in words
    ]
    return torch.tensor(ids, dtype=torch.int64)


def get_homophone_idx(word, lex, tagger, voc_best=None):
    possibilities = lex["homophones"][word]
    return get_idx_from_possibilities(tagger, possibilities, voc_best=voc_best)


def get_spell_idx(word, tagger, voc_best=None):
    l = len(word)
    if voc_best is not None:
        candidates = [
            tagger.worder.id_to_word[i.item()]
            for i in voc_best
        ]
    else:
        candidates = tagger.worder.word_to_id.keys()
    possibilities = [
        candidate for candidate in candidates
        if difflib.SequenceMatcher(None, word, candidate).ratio() > 0.75
        or (
            l < 4
            and abs(len(word) - len(candidate)) <= 1
            and difflib.SequenceMatcher(None, word, candidate).ratio() > 0.45
        )
    ]
    return get_idx_from_possibilities(tagger, possibilities, voc_best=voc_best)


def get_inflection_idx(word, lex, tagger, voc_best=None):
    possibilities = lex["common_lemma"][word]
    return get_idx_from_possibilities(tagger, possibilities, voc_best=voc_best)


def get_prefix_idx(word, tagger, voc_best=None):
    if voc_best is not None:
        candidates = [
            tagger.worder.id_to_word[i.item()]
            for i in voc_best
        ]
    else:
        candidates = tagger.worder.word_to_id.keys()
    possibilities = [
        candidate for candidate in candidates
        if word.startswith(candidate)
    ]
    return get_idx_from_possibilities(tagger, possibilities)


def get_inflection(word, inflection, lex, tagger):
    inflection_spacy = inflect_tag_to_dict(inflection)
    lems = lex["wrd2lem"][word]
    # logging.info("POS = " + inflection_spacy['POS'])
    # logging.info(str(lems))
    for lem in lems:
        possibilities = lex["lempos2wrdsinf"][
            separ.join([
                lem,
                inflection_spacy['POS']
            ])
        ]
        # logging.info("possible = " + str(possibilities))
        for gender, nbre, conj in possibilities:
            gender_ = ''
            nbre_ = ''
            if gender:
                if gender == "m":
                    gender_ = "Masc"
                elif gender == "f":
                    gender_ = "Fem"
                if not "Gender" in inflection_spacy:
                    continue
            if nbre:
                nbre_ = NBR_MAP[nbre]
                if not "Number" in inflection_spacy:
                    continue
            conjs = conj.split(';')[:-1]
            for conj_ in conjs:
                c = conj_.split(':')
                if len(c) == 1:
                    inf = {"VerbForm": "Inf"}
                elif len(c) == 2:
                    tense = TENSE_MAP[c[1]]
                    inf = {"VerbForm": "Part", "Tense": tense}
                elif len(c) == 3:
                    mood = MOOD_MAP[c[0]]
                    tense = TENSE_MAP[c[1]]
                    pers = c[2][0]
                    num = NBR_MAP[c[2][1]]
                    inf = {
                        "VerbForm": "Fin",
                        "Tense": tense,
                        "Mood": mood,
                        "Number": num,
                        "Person": pers,
                    }
                else:
                    raise ValueError("conjugation value of {} invalid".format(c))

                if nbre and not "Number" in inf:
                    inf["Number"] = nbre_
                if gender:
                    inf["Gender"] = gender_

                for key in inf:
                    if (
                        (key not in inflection_spacy)
                        or (inf[key] != inflection_spacy[key])
                    ):
                        break
                else:
                    # Match found!
                    return list(lex["lemposinf2wrd"][separ.join([
                        lem,
                        inflection_spacy['POS'],
                        gender,
                        nbre,
                        conj,
                    ])])[0]
            # Not conjugation
            if inflection_spacy['POS'] != "VERB":
                # logging.info(gender)
                # logging.info(str(inflection_spacy))
                if (
                    (gender == '' or
                    gender_ == inflection_spacy["Gender"]) and
                    (nbre == '' or
                    nbre_ == inflection_spacy["Number"])
                ):
                    return list(lex["lemposinf2wrd"][separ.join([
                        lem,
                        inflection_spacy['POS'],
                        gender,
                        nbre,
                        conj,
                    ])])[0]


def get_tags_vocs_from_proposals(
    toks,
    tag_proposals,
    voc_out,
    tagger: TagEncoder2,
    lex,
    args,
):
    if len(toks) != tag_proposals.size(0):
        # logging.debug(len(toks))
        # logging.debug(tag_proposals.size(0))
        toks, tag_proposals = (
            toks[:tag_proposals.size(0)],
            tag_proposals[:len(toks)]
        )

    new_tag = torch.zeros(
        tag_proposals.size(0),
        dtype=torch.int64,
        device=voc_out.device,
    )
    new_voc = -torch.ones(
        tag_proposals.size(0),
        dtype=torch.int64,
        device=voc_out.device,
    )
    new_inflections = dict()

    if args.k_best > 0:
        _, voc_k_best = torch.topk(voc_out, args.k_best, dim=-1)
    else:
        voc_k_best = [None] * len(toks)
    for i in range(len(toks)):
        can_move_on = False
        j = 0
        while not can_move_on:
            tag = tagger._id_to_tag[tag_proposals[i, j].item()]
            # logging.info(str(i) + " HYPOTHESIS " + str(j) + " : " + tag)
            # vocs = [tagger.worder.id_to_word[i.item()] for i in voc_ids]
            if tag.startswith("$REPLACE"):
                idx = None
                if tag.startswith("$REPLACE:HOMOPHONE"):
                    idx = get_homophone_idx(
                        toks[i], lex, tagger,
                        voc_best=voc_k_best[i]
                    )
                elif tag.startswith("$REPLACE:SPELL"):
                    idx = get_spell_idx(
                        toks[i], tagger,
                        voc_best=voc_k_best[i],
                    )
                elif tag.startswith("$REPLACE:INFLECTION"):
                    idx = get_inflection_idx(
                        toks[i], lex, tagger,
                        voc_best=voc_k_best[i]
                    )
                if idx is not None and idx.any():
                    can_move_on = True
                    new_voc[i] = idx[voc_out[i][idx].argmax(-1)]
            elif tag.startswith("$SPLIT"):
                idx = get_prefix_idx(
                    toks[i], tagger,
                    voc_best=voc_k_best[i]
                )
                if idx is not None and idx.nelement():
                    can_move_on = True
                    new_voc[i] = idx[voc_out[i][idx].argmax(-1)]

            elif tag.startswith("$INFLECT"):
                inflection = tag.split(':')[-1]
                inflection = get_inflection(toks[i], inflection, lex, tagger)
                if inflection:
                    can_move_on = True
                    new_inflections[i] = inflection
            elif tag.startswith("$APPEND"):
                can_move_on = True
                new_voc[i] = voc_out[i].argmax(-1)
            else:
                can_move_on = True

            j += 1

        new_tag[i] = tag_proposals[i, j-1]

    return new_tag, new_voc, new_inflections


def apply_tags(
    toks,
    tags,
    vocs,
    infs,
    tagger: TagEncoder2,
    args,
):
    new_toks = list()
    order_edits = dict()
    for i in range(len(toks)):
        tag = tagger._id_to_tag[tags[i].item()]
        # logging.info(str(i) + " HYPOTHESIS " + str(j) + " : " + tag)
        # vocs = [tagger.worder.id_to_word[i.item()] for i in voc_ids]
        if tag.startswith("$REPLACE"):
                word = tagger.worder.id_to_word[
                    vocs[i].item()
                ]
                new_toks.append(word)
        elif tag.startswith("$SPLIT"):
            word = tagger.worder.id_to_word[
                vocs[i].item()
            ]
            assert toks[i].startswith(word)
            new_toks.append(word)
            new_toks.append(toks[i][len(word):])
        elif tag.startswith("$INFLECT"):
            new_toks.append(infs[i])
        else:
            if tag == '·':  # keep
                new_toks.append(toks[i])
            elif "$APPEND" in tag:
                new_toks.append(toks[i])
                new_toks.append(tagger.worder.id_to_word[
                    vocs[i].item()
                ])
            elif tag == "$DELETE" or tag == "$COPY":
                pass
            elif tag == "$SWAP":
                ####
                order_edits[len(new_toks)] = tag
                new_toks.append(toks[i])
                # if i != len(toks) - 1:
                #     new_toks.append(toks[i + 1])
                # new_toks.append(toks[i])
                # i = i + 1
            elif tag == "$CASE:FIRST":
                if toks[i][0].isupper():
                    new_toks.append(toks[i][0].lower() + toks[i][1:])
                elif toks[i][0].islower():
                    new_toks.append(toks[i][0].upper() + toks[i][1:])
            elif tag == "$CASE:UPPER":
                new_toks.append(toks[i].upper())
            elif tag == "$CASE:LOWER":
                new_toks.append(toks[i].lower())
            elif tag == "$HYPHEN:SPLIT":
                for t in toks[i].split('-'):
                    new_toks.append(t)
            elif tag == "$MERGE":
                ####
                order_edits[len(new_toks)] = tag
                new_toks.append(toks[i])
                # if i != len(toks) - 1:
                #     new_toks.append(toks[i] + toks[i + 1])
                # else:
                #     new_toks.append(toks[i])
                # i += 1
            elif tag == "$HYPHEN:MERGE":
                ####
                order_edits[len(new_toks)] = tag
                new_toks.append(toks[i])
                # if i != len(toks) - 1:
                #     new_toks.append(toks[i] + '-' + toks[i + 1])
                # else:
                #     new_toks.append(toks[i])
                # i += 1
            else:
                raise ValueError("Tag not recognized :" + tag)

    for i in range(len(new_toks) - 1):
        if i in order_edits:
            if order_edits[i] == "$SWAP":
                new_toks[i], new_toks[i+1] = new_toks[i+1], new_toks[i]
            elif order_edits[i] == "$MERGE":
                new_toks[i], new_toks[i+1] = '', new_toks[i] + new_toks[i+1]
            elif order_edits[i] == "$HYPHEN:MERGE":
                new_toks[i], new_toks[i+1] = '', new_toks[i] + '-' + new_toks[i+1]
    new_toks = [t for t in new_toks if t != '']


    # if args.out_tags:
    #     with open(args.out_tags, 'a') as f:
    #         f.write('\n')
    new_sentence = ' '.join(new_toks[:510])
    new_sentence = re.sub("' ", "'", new_sentence)

    return new_sentence


def apply_tags_with_constraint(
    sentence: str,
    tag_proposals,
    voc_out,
    tokenizer: WordTokenizer,
    tagger: TagEncoder2,
    lex,
    args,
    return_corrected=True,
):
    toks = tokenizer.tokenize(sentence.rstrip('\n'), max_length=510)

    res = dict()

    tags, vocs, infs = get_tags_vocs_from_proposals(
        toks,
        tag_proposals,
        voc_out,
        tagger,
        lex,
        args,
    )
    if args.return_tag_voc:
        res["vocs"] = vocs
        res["tags"] = tags

    if args.out_tags:
        tags = [tagger.id_to_tag(i.item()) for i in tagger.tag_word_to_id_vec(vocs, tags)]
        with open(args.out_tags, 'a') as f:
            f.write(' '.join([w + '|' + t for w, t in zip(toks, tags)]) + '\n')
            # logging.info(' '.join([w + '|' + t for w, t in zip(toks, tags)]) + '\n')

    if return_corrected:
        res["text"] = apply_tags(
            toks,
            tags,
            vocs,
            infs,
            tagger,
            args,
        )

    return res


def apply_tags_with_constraint_(
    sentence: str,
    tag_proposals,
    voc_out,
    tokenizer: WordTokenizer,
    tagger: TagEncoder2,
    lex,
    args,
):
    # logging.info("sentence >>> " + sentence)
    toks = tokenizer.tokenize(sentence.rstrip('\n'), max_length=510)
    # logging.info(str(toks))
    if len(toks) != tag_proposals.size(0):
        # logging.debug(len(toks))
        # logging.debug(tag_proposals.size(0))
        toks, tag_proposals = (
            toks[:tag_proposals.size(0)],
            tag_proposals[:len(toks)]
        )

    new_toks = list()
    if args.return_tag_voc:
        new_tag = torch.zeros(
            tag_proposals.size(0),
            dtype=torch.int64,
            device=voc_out.device,
        )
        new_voc = -torch.ones(
            tag_proposals.size(0),
            dtype=torch.int64,
            device=voc_out.device,
        )
    i = 0
    while i < len(toks):
        can_move_on = False
        j = 0
        while not can_move_on:
            tag = tagger._id_to_tag[tag_proposals[i, j].item()]
            # logging.info(str(i) + " HYPOTHESIS " + str(j) + " : " + tag)
            # vocs = [tagger.worder.id_to_word[i.item()] for i in voc_ids]
            if tag.startswith("$REPLACE"):
                idx = None
                if tag.startswith("$REPLACE:HOMOPHONE"):
                    idx = get_homophone_idx(toks[i], lex, tagger)
                elif tag.startswith("$REPLACE:SPELL"):
                    idx = get_spell_idx(toks[i], tagger)
                elif tag.startswith("$REPLACE:INFLECTION"):
                    idx = get_inflection_idx(toks[i], lex, tagger)
                if idx is not None and idx.any():
                    # logging.info(str(idx.cpu().numpy()))
                    can_move_on = True
                    word = tagger.worder.id_to_word[
                        idx[voc_out[i][idx].argmax(-1)].item()
                    ]
                    new_toks.append(word)
                    if args.return_tag_voc:
                        new_voc[i] = idx[voc_out[i][idx].argmax(-1)]
            elif tag.startswith("$SPLIT"):
                idx = get_prefix_idx(toks[i], tagger)
                if idx is not None and idx.nelement():
                    # logging.info(str(idx.cpu().numpy()))
                    can_move_on = True
                    word = tagger.worder.id_to_word[
                        idx[voc_out[i][idx].argmax(-1)].item()
                    ]
                    assert toks[i].startswith(word)
                    new_toks.append(word)
                    new_toks.append(toks[i][len(word):])
                    if args.return_tag_voc:
                        new_voc[i] = idx[voc_out[i][idx].argmax(-1)]
            elif tag.startswith("$INFLECT"):
                inflection = tag.split(':')[-1]
                inflection = get_inflection(toks[i], inflection, lex, tagger)
                if inflection:
                    can_move_on = True
                    new_toks.append(inflection)
            else:
                can_move_on = True
                if tag == '·':  # keep
                    new_toks.append(toks[i])
                elif "$APPEND" in tag:
                    new_toks.append(toks[i])
                    new_toks.append(tagger.worder.id_to_word[
                        voc_out[i].argmax(-1).item()
                    ])
                    if args.return_tag_voc:
                        new_voc[i] = voc_out[i].argmax(-1)
                elif tag == "$DELETE" or tag == "$COPY":
                    pass
                elif tag == "$SWAP":
                    if i != len(toks) - 1:
                        new_toks.append(toks[i + 1])
                    new_toks.append(toks[i])
                    i = i + 1
                elif tag == "$CASE:FIRST":
                    if toks[i][0].isupper():
                        new_toks.append(toks[i][0].lower() + toks[i][1:])
                    elif toks[i][0].islower():
                        new_toks.append(toks[i][0].upper() + toks[i][1:])
                elif tag == "$CASE:UPPER":
                    new_toks.append(toks[i].upper())
                elif tag == "$CASE:LOWER":
                    new_toks.append(toks[i].lower())
                elif tag == "$HYPHEN:SPLIT":
                    for t in toks[i].split('-'):
                        new_toks.append(t)
                elif tag == "$MERGE":
                    if i != len(toks) - 1:
                        new_toks.append(toks[i] + toks[i + 1])
                    else:
                        new_toks.append(toks[i])
                    i += 1
                elif tag == "$HYPHEN:MERGE":
                    if i != len(toks) - 1:
                        new_toks.append(toks[i] + '-' + toks[i + 1])
                    else:
                        new_toks.append(toks[i])
                    i += 1
                else:
                    raise ValueError("Tag not recognized :" + tag)

            j += 1
        if args.out_tags:
            with open(args.out_tags, 'a') as f:
                if i != 0:
                    f.write(' ')
                f.write('|'.join((toks[i], tag)))
                if tag == "$SPLIT":
                    f.write('_' + new_toks[-2])
                if (
                    tag.startswith("$REPLACE") or
                    tag == "$APPEND"
                ):
                    f.write('_' + new_toks[-1])
        if args.return_tag_voc:
            new_tag[i] = tag_proposals[i, j-1]

        i += 1
    if args.out_tags:
        with open(args.out_tags, 'a') as f:
            f.write('\n')
    new_sentence = ' '.join(new_toks[:510])
    new_sentence = re.sub("' ", "'", new_sentence)

    # logging.info(new_sentence)
    # logging.info("**********************")
    if args.return_tag_voc:
        return new_sentence, new_tag, new_voc

    return new_sentence

# def apply_tags(
#     sentence: str,
#     tags,
#     tokenizer: WordTokenizer,
#     tagger: TagEncoder2,
#     lexicon: Lexicon,
#     args,
# ):
#
#     toks = tokenizer.tokenize(sentence.rstrip('\n'), max_length=510)
#
#     if len(toks) != len(tags):
#         logging.debug(len(toks))
#         logging.debug(len(tags))
#         toks, tags = toks[:len(tags)], tags[:len(toks)]
#
#     if args.out_tags:
#         with open(args.out_tags, 'a') as f:
#             f.write(' '.join(['|'.join(e) for e in zip(toks, tags)]) + '\n')
#
#     assert len(toks) == len(tags)
#
#     new_toks = list()
#     i = 0
#     while i < len(toks):
#         tag = tags[i]
#         if tag == '·':  # keep
#             new_toks.append(toks[i])
#         elif "$APPEND_" in tag:
#             new_toks.append(toks[i])
#             new_toks.append(tag[8:])
#         elif tag == "$DELETE" or tag == "$COPY":
#             pass
#         elif tag == "$SWAP":
#             if i != len(toks) - 1:
#                 new_toks.append(toks[i + 1])
#             new_toks.append(toks[i])
#             i = i + 1
#         elif tag == "$CASE:FIRST":
#             if toks[i][0].isupper():
#                 new_toks.append(toks[i][0].lower() + toks[i][1:])
#             elif toks[i][0].islower():
#                 new_toks.append(toks[i][0].upper() + toks[i][1:])
#         elif tag == "$CASE:UPPER":
#             new_toks.append(toks[i].upper())
#         elif tag == "$CASE:LOWER":
#             new_toks.append(toks[i].lower())
#         elif tag == "$HYPHEN:SPLIT":
#             for t in toks[i].split('-'):
#                 new_toks.append(t)
#         elif tag == "$MERGE":
#             if i != len(toks) - 1:
#                 new_toks.append(toks[i] + toks[i + 1])
#             else:
#                 new_toks.append(toks[i])
#             i += 1
#         elif tag == "$HYPHEN:MERGE":
#             if i != len(toks) - 1:
#                 new_toks.append(toks[i] + '-' + toks[i + 1])
#             else:
#                 new_toks.append(toks[i])
#             i += 1
#         elif tag.startswith("$REPLACE"):
#             new_toks.append(tag.split('_')[-1])
#         elif tag.startswith("$INFLECT:"):
#             # new_toks.append(g_transform(toks[i], tag.split('_')[-1], lexicon))
#             new_toks.append(toks[i]+"_inflected")
#         elif tag.startswith("$SPLIT"):
#             w = tag.split('_')[-1]
#             if toks[i].startswith(w):
#                 new_toks.append(w)
#                 new_toks.append(toks[i][len(w):])
#             else:
#                 new_toks.append(toks[i])
#         else:
#             raise ValueError("Tag not recognized :" + tag)
#
#         i += 1
#
#     # logging.info(len(new_toks))
#     new_sentence = ' '.join(new_toks[:510])
#     new_sentence = re.sub("' ", "'", new_sentence)
#
#     # logging.info(new_sentence)
#     # logging.info("**********************")
#
#     return new_sentence


def infer(args):

    tokenizer = FlaubertTokenizer.from_pretrained(
        args.tokenizer
    )
    word_tokenizer = WordTokenizer(FlaubertTokenizer)
    # lexicon = Lexicon(args.lex)
    tagger = TagEncoder2(
        path_to_lex=args.lex,
        path_to_voc=args.voc,
    )
    lex = read_lexicon(args.lex, tagger.worder.word_to_id.keys())

    path_to_model = os.path.join(
        args.save_path,
        args.model_id,
        "model_best.pt",
    )
    model = GecBertVocModel(
        len(tagger),
        len(tagger.worder),
        tokenizer=tokenizer,
        tagger=tagger,
        mid=args.model_id,
    )

    device = "cuda:" + str(args.gpu_id) \
        if args.gpu and torch.cuda.is_available() else "cpu"
    if os.path.isfile(path_to_model):
        state_dict = torch.load(
            path_to_model, map_location=torch.device(device)
        )
        model.load_state_dict(state_dict["model_state_dict"])

    else:
        logging.info("Model not found at: " + path_to_model)
        return
    model.eval()

    logging.info(torch.cuda.device_count())
    logging.info("device = " + device)
    model.to(device)

    if args.text:
        txt = args.text.split('\n')
    elif args.file:
        with open(args.file, 'r') as f:
            txt = f.readlines()
    else:
        raise ValueError("No input argument. try --text or --file")

    if args.out_tags:
        with open(args.out_tags, 'w') as f:
            f.write('')
    # logging.info("-----------------------")
    # logging.info(txt)
    # logging.info([tagger.id_to_tag(tag.item()) for tag in tags])
    # logging.info(len(toks))
    # logging.info(len(tags))

    for i in range(len(txt) // args.batch_size + 1):
        if i > 0:
            print()
        batch_txt = txt[args.batch_size *
                        i: min(args.batch_size * (i + 1), len(txt))]
        for j in range(args.num_iter):
            toks = tokenizer(
                batch_txt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=510
            ).to(device)
            del toks["token_type_ids"]
            xi = toks["input_ids"][0][
                toks["attention_mask"][0].bool()
            ]
            # logging.info("noise >>> " + " ".join(map(tokenizer._convert_id_to_token, xi.cpu().numpy())))
            # logging.info(toks["input_ids"].shape)
            with torch.no_grad():
                out = model(**toks)  # tag_out, attention_mask
                for k, t in enumerate(batch_txt):
                    # tag_ids = out["tag_out"][k].argmax(-1)[out["attention_mask"][k].bool()]
                    # voc_ids = out["voc_out"][k].argmax(-1)[out["attention_mask"][k].bool()]

                    tag_out = out["tag_out"][k][out["attention_mask"][k].bool()]
                    voc_out = out["voc_out"][k][out["attention_mask"][k].bool()]

                    tag_proposals = torch.argsort(
                        tag_out, dim=-1, descending=True)
                    voc_proposals = torch.argsort(
                        voc_out, dim=-1, descending=True)

                    # tags = [tagger._id_to_tag[i.item()] for i in tag_ids]
                    # vocs = [tagger.worder.id_to_word[i.item()] for i in voc_ids]

                    # for i, tag in enumerate(tags):
                    #     if tagger.is_radical_word_tag(tag):
                    #         tags[i] = tag + '_' + vocs[i]
                    # logging.info('-' * 50)
                    # logging.info(batch_txt[k].rstrip('\n'))
                    # logging.info(' '.join(tags))

                    batch_txt[k] = apply_tags_with_constraint(
                        t,
                        tag_proposals,
                        voc_out,
                        word_tokenizer,
                        tagger,
                        lex,
                        args,
                    )["text"]
        print('\n'.join(batch_txt))


def create_logger(logfile, loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        logging.error("Invalid log level={}".format(loglevel))
        sys.exit()
    if logfile is None or logfile == 'stderr':
        logging.basicConfig(
            format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s',
            datefmt='%Y-%m-%d_%H:%M:%S',
            level=numeric_level)
    else:
        logging.basicConfig(
            filename=logfile,
            format='[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s',
            datefmt='%Y-%m-%d_%H:%M:%S',
            level=numeric_level)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--text', default=None, help="Input text")
    parser.add_argument('--file', default=None, help="Input file")
    # optional
    parser.add_argument('-v', action='store_true')
    parser.add_argument('--log', default="info", help='logging level')
    parser.add_argument(
        '--gpu',
        action='store_true',
        help="GPU usage activation."
    )
    parser.add_argument(
        '--gpu-id',
        default=0,
        type=int,
        help="GPU id, generally 0 or 1."
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='batch size for eval'
    )
    parser.add_argument(
        '--num-iter',
        type=int,
        default=1,
        help='num iteration loops to edit'
    )
    parser.add_argument(
        '--save-path',
        required=True,
        help='model save directory'
    )
    parser.add_argument(
        '--model-id',
        required=True,
        help="Model id (folder name)"
    )
    parser.add_argument(
        '--k-best',
        type=int,
        default=-1,
        help='restrict to k-best words'
    )
    parser.add_argument(
        '--lex',
        required=True,
        help='path to lexicon table.'
    )
    parser.add_argument(
        '--voc',
        required=True,
        help="Path to appendable data."
    )
    parser.add_argument(
        '--tokenizer',
        default="flaubert/flaubert_base_cased",
        help='model save directory'
    )
    parser.add_argument(
        '--return-tag-voc',
        action='store_true',
        help='Return tag and voc ids when infering.'
    )
    parser.add_argument(
        '--out-tags',
        default=None,
        help='file for the tagged output words'
    )

    args = parser.parse_args()

    create_logger("stderr", args.log)

    infer(args)
