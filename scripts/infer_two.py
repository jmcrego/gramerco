from transformers import FlaubertTokenizer, FlaubertModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tqdm import tqdm
from data.gramerco_dataset import GramercoDataset
from model_gec.gec_bert import GecBertVocModel, GecBertInflVocModel
from tag_encoder import TagEncoder2, TagEncoder3
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


separ = "￨"


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
    """Builds useful dictionaries for inference.
    f: file corresponding to a Lexicon of french words.
    vocab: list of vocabulary words used.
    """
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

    with open(f, "r") as fd:
        for l in fd:
            toks = l.rstrip().split("\t")
            if len(toks) < 3:
                continue
            wrd, pho, lem, pos, gender, nbre, conj = (
                toks[0],
                toks[1],
                toks[2],
                toks[3],
                toks[4],
                toks[5],
                toks[10],
            )
            in_vocab = wrd in vocab
            # if wrd not in vocab:
            #     continue
            if " " in wrd or " " in lem or " " in pho or " " in pos:
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
    """Transforms a string inflection tag to a dictionary 
    of grammatical specificities.
    """
    d = {e.split("=")[0]: e.split("=")[1] for e in tag.split(";")[1:] if "=" in e}

    d["POS"] = tag.split(";")[0]
    return d


def get_idx_from_possibilities(tagger, possibilities, voc_best=None):
    """Returns a tensor of indices correponding to a list of word possibilities.
    """
    if voc_best is not None:
        words = [tagger.worder.id_to_word[i] for i in voc_best.numpy()]
    else:
        words = tagger.worder.word_to_id
    ids = [tagger.worder.word_to_id[word] for word in possibilities if word in words]
    return torch.tensor(ids, dtype=torch.int64)


def get_homophone_idx(word, lex, tagger, voc_best=None):
    """Returns a tensor of indices correponding to homophones of a given word.
    """
    possibilities = lex["homophones"][word]
    return get_idx_from_possibilities(tagger, possibilities, voc_best=voc_best)


def get_spell_idx(word, tagger, voc_best=None):
    """Returns a tensor of indices correponding to closed spelling words to a given word.
    """
    l = len(word)
    if voc_best is not None:
        candidates = [tagger.worder.id_to_word[i.item()] for i in voc_best]
    else:
        candidates = tagger.worder.word_to_id.keys()
    possibilities = [
        candidate
        for candidate in candidates
        if difflib.SequenceMatcher(None, word, candidate).ratio() > 0.75
        or (
            l < 4
            and abs(len(word) - len(candidate)) <= 1
            and difflib.SequenceMatcher(None, word, candidate).ratio() > 0.45
        )
    ]
    return get_idx_from_possibilities(tagger, possibilities, voc_best=voc_best)


def get_inflection_idx(word, lex, tagger, voc_best=None):
    """Returns a tensor of indices corresponding to possible inflections of a given word.
    """
    possibilities = lex["common_lemma"][word]
    return get_idx_from_possibilities(tagger, possibilities, voc_best=voc_best)


def get_prefix_idx(word, tagger, voc_best=None):
    """Returns a tensor of indices corresponding to words being prefix of a given word.
    """
    if voc_best is not None:
        candidates = [tagger.worder.id_to_word[i.item()] for i in voc_best]
    else:
        candidates = tagger.worder.word_to_id.keys()
    possibilities = [
        candidate for candidate in candidates if word.startswith(candidate)
    ]
    return get_idx_from_possibilities(tagger, possibilities)


def get_inflection(word, inflection, lex, tagger):
    """Returns the first matching word given an original word and an inflection.
    """
    inflection_spacy = inflect_tag_to_dict(inflection)
    lems = lex["wrd2lem"][word]
    for lem in lems:
        possibilities = lex["lempos2wrdsinf"][
            separ.join([lem, inflection_spacy["POS"]])
        ]
        for gender, nbre, conj in possibilities:
            gender_ = ""
            nbre_ = ""
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
            conjs = conj.split(";")[:-1]
            for conj_ in conjs:
                c = conj_.split(":")
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
                    if (key not in inflection_spacy) or (
                        inf[key] != inflection_spacy[key]
                    ):
                        break
                else:
                    # Match found!
                    return list(
                        lex["lemposinf2wrd"][
                            separ.join(
                                [lem, inflection_spacy["POS"], gender, nbre, conj,]
                            )
                        ]
                    )[0]
            # Not conjugation
            if inflection_spacy["POS"] != "VERB":
                if (gender == "" or gender_ == inflection_spacy["Gender"]) and (
                    nbre == "" or nbre_ == inflection_spacy["Number"]
                ):
                    return list(
                        lex["lemposinf2wrd"][
                            separ.join(
                                [lem, inflection_spacy["POS"], gender, nbre, conj,]
                            )
                        ]
                    )[0]


def get_tags_vocs_from_proposals(
    toks, tag_proposals, voc_out, tagger: TagEncoder2, lex, args, infls=None,
):
    """Returns the first compatible triplet (tag, voc, inflection) in descending logit order.
    The tag_proposals is an already ordered version of the potential tags.
    The tags are tested one by one in order until one works.
    """
    if len(toks) != tag_proposals.size(0):
        toks, tag_proposals = (
            toks[: tag_proposals.size(0)],
            tag_proposals[: len(toks)],
        )

    new_tag = torch.zeros(
        tag_proposals.size(0), dtype=torch.int64, device=voc_out.device,
    )
    new_voc = -torch.ones(
        tag_proposals.size(0), dtype=torch.int64, device=voc_out.device,
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
            if tag.startswith("$REPLACE"):
                idx = None
                if tag.startswith("$REPLACE:HOMOPHONE"):
                    idx = get_homophone_idx(
                        toks[i], lex, tagger, voc_best=voc_k_best[i]
                    )
                elif tag.startswith("$REPLACE:SPELL"):
                    idx = get_spell_idx(toks[i], tagger, voc_best=voc_k_best[i],)
                elif tag.startswith("$REPLACE:INFLECTION"):
                    idx = get_inflection_idx(
                        toks[i], lex, tagger, voc_best=voc_k_best[i]
                    )
                if idx is not None and idx.any():
                    can_move_on = True
                    new_voc[i] = idx[voc_out[i][idx].argmax(-1)]
            elif tag.startswith("$SPLIT"):
                idx = get_prefix_idx(toks[i], tagger, voc_best=voc_k_best[i])
                if idx is not None and idx.nelement():
                    can_move_on = True
                    new_voc[i] = idx[voc_out[i][idx].argmax(-1)]

            elif tag.startswith("$INFLECT"):
                if infls is not None:
                    inflection = tagger.inflecter.id_to_infl[infls[i].item()]
                else:
                    inflection = tag.split(":")[-1]
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

        new_tag[i] = tag_proposals[i, j - 1]

    return new_tag, new_voc, new_inflections


def apply_tags(
    toks, tags, vocs, infs, tagger, args,
):
    """Applies tags to a sentence (list of tokens).
    """
    new_toks = list()
    order_edits = dict()
    for i in range(len(toks)):
        tag = tags[i]
        if not isinstance(tag, str):
            tag = tagger._id_to_tag[tag.item()]
        if tag.startswith("$REPLACE"):
            word = tagger.worder.id_to_word[vocs[i].item()]
            new_toks.append(word)
        elif tag.startswith("$SPLIT"):
            word = tagger.worder.id_to_word[vocs[i].item()]
            assert toks[i].startswith(word)
            new_toks.append(word)
            new_toks.append(toks[i][len(word) :])
        elif tag.startswith("$INFLECT"):
            new_toks.append(infs[i])
        else:
            if tag == "·":  # keep
                new_toks.append(toks[i])
            elif "$APPEND" in tag:
                new_toks.append(toks[i])
                new_toks.append(tagger.worder.id_to_word[vocs[i].item()])
            elif tag == "$DELETE" or tag == "$COPY":
                pass
            elif tag == "$SWAP":
                order_edits[len(new_toks)] = tag
                new_toks.append(toks[i])
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
                for t in toks[i].split("-"):
                    new_toks.append(t)
            elif tag == "$MERGE":
                order_edits[len(new_toks)] = tag
                new_toks.append(toks[i])
            elif tag == "$HYPHEN:MERGE":
                order_edits[len(new_toks)] = tag
                new_toks.append(toks[i])
            else:
                raise ValueError("Tag not recognized :" + tag)

    for i in range(len(new_toks) - 1):
        if i in order_edits:
            if order_edits[i] == "$SWAP":
                new_toks[i], new_toks[i + 1] = new_toks[i + 1], new_toks[i]
            elif order_edits[i] == "$MERGE":
                new_toks[i], new_toks[i + 1] = "", new_toks[i] + new_toks[i + 1]
            elif order_edits[i] == "$HYPHEN:MERGE":
                new_toks[i], new_toks[i + 1] = "", new_toks[i] + "-" + new_toks[i + 1]
    new_toks = [t for t in new_toks if t != ""]

    new_sentence = " ".join(new_toks[:510])
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
    infls=None,
    return_corrected=True,
):
    """Deduces tags according to constraints, then can apply them to a sentence,
    provided return_corrected==True.
    """
    toks = tokenizer.tokenize(sentence.rstrip("\n"), max_length=510)

    res = dict()

    tags, vocs, infs = get_tags_vocs_from_proposals(
        toks, tag_proposals, voc_out, tagger, lex, args, infls=infls,
    )
    res["changed"] = tags.sum().item() > 0
    if args.return_tag_voc:
        res["vocs"] = vocs
        res["tags"] = tags
        if infls is not None:
            res["infls"] = infs

    if args.out_tags:
        if infls is not None:
            tags = [
                tagger.id_to_tag(i.item())
                for i in tagger.tag_word_to_id_vec(vocs, infls, tags)
            ]
        else:
            tags = [
                tagger.id_to_tag(i.item())
                for i in tagger.tag_word_to_id_vec(vocs, tags)
            ]
        with open(args.out_tags, "a") as f:
            f.write(" ".join([w + "|" + t for w, t in zip(toks, tags)]) + "\n")
            logging.debug(" ".join([w + "|" + t for w, t in zip(toks, tags)]) + "\n")

    if return_corrected:
        res["text"] = apply_tags(toks, tags, vocs, infs, tagger, args,)

    return res


def infer(args):
    """Main inference script."""

    tokenizer = FlaubertTokenizer.from_pretrained(args.tokenizer)
    word_tokenizer = WordTokenizer(FlaubertTokenizer)
    if args.inflection_layer:
        tagger = TagEncoder3(path_to_lex=args.lex, path_to_voc=args.voc,)
    else:
        tagger = TagEncoder2(
            path_to_lex=args.lex, path_to_voc=args.voc, new_version=args.word_index,
        )
    lex = read_lexicon(args.lex, tagger.worder.word_to_id.keys())

    path_to_model = os.path.join(args.save_path, args.model_id, "model_best.pt",)
    if args.inflection_layer:
        model = GecBertInflVocModel(
            len(tagger),
            len(tagger.inflecter),
            len(tagger.worder),
            tokenizer=tokenizer,
            tagger=tagger,
            mid=args.model_id,
        )
    else:
        model = GecBertVocModel(
            len(tagger),
            len(tagger.worder),
            tokenizer=tokenizer,
            tagger=tagger,
            mid=args.model_id,
        )

    device = (
        "cuda:" + str(args.gpu_id) if args.gpu and torch.cuda.is_available() else "cpu"
    )
    if os.path.isfile(path_to_model):
        state_dict = torch.load(path_to_model, map_location=torch.device(device))
        model.load_state_dict(state_dict["model_state_dict"])

    else:
        logging.info("Model not found at: " + path_to_model)
        return
    model.eval()

    logging.info(torch.cuda.device_count())
    logging.info("device = " + device)
    model.to(device)

    if args.text:
        txt = args.text.split("\n")
    elif args.file:
        with open(args.file, "r") as f:
            txt = f.readlines()
    else:
        raise ValueError("No input argument. try --text or --file")

    if args.out_tags:
        with open(args.out_tags, "w") as f:
            f.write("")

    for i in range(len(txt) // args.batch_size + 1):
        if i > 0:
            print()
        if args.batch_size * i == min(args.batch_size * (i + 1), len(txt)):
            break
        batch_txt = txt[args.batch_size * i : min(args.batch_size * (i + 1), len(txt))]
        changed_mask = torch.ones(len(batch_txt), dtype=bool)
        for j in range(args.num_iter):
            new_changed_mask = torch.ones(len(batch_txt), dtype=bool)
            toks = tokenizer(
                batch_txt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=510,
            ).to(device)
            del toks["token_type_ids"]
            with torch.no_grad():
                out = model(**toks)  # tag_out, attention_mask
                logging.info(changed_mask.long().numpy())
                if not changed_mask.any().item():
                    break
                for k, t in enumerate(batch_txt):
                    if not changed_mask[k].item():
                        new_changed_mask[k] = changed_mask[k]
                        continue

                    tag_out = out["tag_out"][k][out["attention_mask"][k].bool()]
                    voc_out = out["voc_out"][k][out["attention_mask"][k].bool()]
                    if args.inflection_layer:
                        infl_out = out["infl_out"][k][out["attention_mask"][k].bool()]

                    tag_proposals = torch.argsort(tag_out, dim=-1, descending=True)
                    if args.inflection_layer:
                        infls_ids = infl_out.argmax(-1)

                    res = apply_tags_with_constraint(
                        t,
                        tag_proposals,
                        voc_out,
                        word_tokenizer,
                        tagger,
                        lex,
                        args,
                        infls=infls_ids,
                    )
                    batch_txt[k] = res["text"]
                    new_changed_mask[k] = res["changed"]
                changed_mask = new_changed_mask
        print("\n".join(batch_txt))


def create_logger(logfile, loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        logging.error("Invalid log level={}".format(loglevel))
        sys.exit()
    if logfile is None or logfile == "stderr":
        logging.basicConfig(
            format="[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s",
            datefmt="%Y-%m-%d_%H:%M:%S",
            level=numeric_level,
        )
    else:
        logging.basicConfig(
            filename=logfile,
            format="[%(asctime)s.%(msecs)03d] %(levelname)s %(message)s",
            datefmt="%Y-%m-%d_%H:%M:%S",
            level=numeric_level,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--text", default=None, help="Input text")
    parser.add_argument("--file", default=None, help="Input file")
    # optional
    parser.add_argument("-v", action="store_true")
    parser.add_argument("--log", default="info", help="logging level")
    parser.add_argument("--gpu", action="store_true", help="GPU usage activation.")
    parser.add_argument(
        "--gpu-id", default=0, type=int, help="GPU id, generally 0 or 1."
    )
    parser.add_argument(
        "--batch-size", type=int, default=20, help="batch size for eval"
    )
    parser.add_argument(
        "--num-iter", type=int, default=1, help="num iteration loops to edit"
    )
    parser.add_argument("--save-path", required=True, help="model save directory")
    parser.add_argument("--model-id", required=True, help="Model id (folder name)")
    parser.add_argument(
        "--k-best", type=int, default=-1, help="restrict to k-best words"
    )
    parser.add_argument("--lex", required=True, help="path to lexicon table.")
    parser.add_argument("--voc", required=True, help="Path to appendable data.")
    parser.add_argument(
        "--tokenizer",
        default="flaubert/flaubert_base_cased",
        help="model save directory",
    )
    parser.add_argument(
        "--return-tag-voc",
        action="store_true",
        help="Return tag and voc ids when infering.",
    )
    parser.add_argument(
        "--out-tags", default=None, help="file for the tagged output words"
    )
    parser.add_argument("--samepos", action="store_true", help="Use same pos tag.")
    parser.add_argument(
        "--inflection-layer",
        action="store_true",
        help="Use a separate layer for inflections.",
    )

    args = parser.parse_args()

    create_logger("stderr", args.log)

    infer(args)
