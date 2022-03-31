#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import random
import argparse
import pyonmttok
from collections import defaultdict
from keyboard import keyboard

separ = "￨"
keep = "·"
space = "~"
letters = "abcdefghijklmnopqrstuvwxyzàáâäèéêëìíîïòóôöùúûü"
symbols = "~`!@#$%^&*()_-+={}|[]\\:;\"'<>?,./"
digits = "0123456789"
letters = letters + letters.upper()
chars = letters + symbols + digits
chars = list(chars)
kbd_lc = keyboard(
    ["`1234567890-=", "qwertyuiop[]\\", "asdfghjkl;'", "zxcvbnm,./"],
    [0, 1.5, 1.85, 2.15],
    2.0,
)
kbd_uc = keyboard(
    ["~!@#$%^&*()_+", "QWERTYUIOP{}|", 'ASDFGHJKL:"', "ZXCVBNM<>?"],
    [0, 1.5, 1.85, 2.15],
    2.0,
)


def output(toks, tags, args):
    assert len(toks) == len(tags)
    if args.output_pair:
        print("{}\t{}".format(" ".join(toks), " ".join(tags)))
    else:
        print(
            "{}".format(" ".join([toks[i] + separ + tags[i] for i in range(len(toks))]))
        )


def do_spacy(nlp, l, toks):
    txt2inf = {}
    tokens = nlp(l.rstrip())
    for token in tokens:
        txt = token.text.replace(" ", space)
        if toks.count(txt) != 1:
            continue
        pos = str(token.pos_)
        if (
            pos != "NOUN" and pos != "ADJ" and pos != "VERB" and pos != "AUX"
        ):  ### only ADJ, NOUN and VERBs are considered
            continue
        lem = token.lemma_.replace(" ", space)
        inf = str(token.morph).replace("|", ";")
        if len(lem) == 0 or len(inf) == 0:
            continue
        txt2inf[txt] = lem + ";" + pos + ";" + inf
    return txt2inf


def misspell(word):
    word = list(word)
    r = random.random()  ### float between [0, 1)

    if r < 1.0 / 5:  ### add extra character in position k [0, len(word)]
        random.shuffle(chars)
        k = random.randint(0, len(word))
        word.insert(k, chars[0])

    elif r < 2.0 / 5:  ### remove character in position k [0, len(word)-1]
        k = random.randint(0, len(word) - 1)
        l = word.pop(k)

    elif r < 3.0 / 5:  ### swap characters in positions k and k+1, k in [0, len(word)-2]
        k = random.randint(0, len(word) - 2)
        word[k], word[k + 1] = (
            word[k + 1],
            word[k],
        )  # if word[k] == word[k=1] the returned word is the same!

    elif (
        r < 4.0 / 5
    ):  ### replace char in positions k [0, len(word)-1] by another close to it
        k = random.randint(0, len(word) - 1)
        c = word[k]
        if c in kbd_lc:
            near_k, _ = kbd_lc.closest(c)
        elif c in kbd_uc:
            near_k, _ = kbd_uc.closest(c)
        else:
            near_k = chars
        random.shuffle(near_k)
        word[k] = near_k[0]

    else:  ### repeat character in position k [0, len(word)-1]
        k = random.randint(0, len(word) - 1)
        word.insert(k, word[k])

    return "".join(word)


def do_replace_inflection(toks, tags, vocab, wrd2wrds_same_lemma, seen, args):
    ### replace a random word by any in wrd2wrds[word] if word in vocab, tag the previous as REPLACE:INFLECTION_word
    idxs = list(range(len(toks)))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep:
            continue
        if not toks[idx] in vocab:
            continue
        if not toks[idx] in wrd2wrds_same_lemma:
            continue
        replacement_options = list(wrd2wrds_same_lemma[toks[idx]])
        if len(replacement_options) == 0:
            continue
        k = random.randint(0, len(replacement_options) - 1)
        tags[idx] = "$REPLACE:INFLECTION_" + toks[idx]
        toks[idx] = replacement_options[k]
        output(toks, tags, args)
        seen["$REPLACE:INFLECTION_"] += 1
        return 1
    return 0


def do_replace_homophone(toks, tags, vocab, wrd2wrds_homophones, seen, args):
    ### replace a random word by any in wrd2wrds[word] if word in vocab, tag the previous as REPLACE:HOMOPHONE_word
    idxs = list(range(len(toks)))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep:
            continue
        if not toks[idx] in vocab:
            continue
        if not toks[idx] in wrd2wrds_homophones:
            continue
        replacement_options = list(wrd2wrds_homophones[toks[idx]])
        if len(replacement_options) == 0:
            continue
        k = random.randint(0, len(replacement_options) - 1)
        tags[idx] = "$REPLACE:HOMOPHONE_" + toks[idx]
        toks[idx] = replacement_options[k]
        output(toks, tags, args)
        seen["$REPLACE:HOMOPHONE_"] += 1
        return 1
    return 0


def do_replace_spell(toks, tags, vocab, seen, args):
    ### misspell a random word if word in vocab (tag it as REPLACE:SPELL_word)
    idxs = list(range(len(toks)))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep:
            continue
        if len(toks[idx]) < 2:
            continue
        if not toks[idx] in vocab:
            continue
        new_txt = misspell(toks[idx])
        if new_txt == toks[idx]:
            continue
        tags[idx] = "$REPLACE:SPELL_" + toks[idx]
        toks[idx] = new_txt
        output(toks, tags, args)
        seen["$REPLACE:SPELL_"] += 1
        return 1
    return 0


def do_append(toks, tags, wrd2pos, vocab, seen, args):
    ### remove the next word if word in vocab, tag the previous as APPEND_word
    idxs = list(range(len(toks) - 1))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep or tags[idx + 1] != keep:
            continue
        if not toks[idx + 1] in vocab:
            continue
        if not toks[idx + 1] in wrd2pos:
            continue
        pos = wrd2pos[toks[idx + 1]]  ###is a set
        if (
            "ADJ" in pos
            or "NOM" in pos
            or "VERB" in pos
            or "ADV" in pos
            or "AUX" in pos
            or "ONO" in pos
        ):  ### do not append VERB NOM ADV ADJ AUX ONO
            continue
        # print("{} {}".format(toks[idx+1], [x for x in pos]))
        tags[idx] = "$APPEND_" + toks[idx + 1]
        toks.pop(idx + 1)
        tags.pop(idx + 1)
        output(toks, tags, args)
        seen["$APPEND_"] += 1
        return 1
    return 0


def do_split(toks, tags, vocab, seen, args):
    ### join two consecutive vocab (w1 w2) if w1 and w2 in vocab, tag as SPLIT_w1
    idxs = list(range(len(toks) - 1))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep or tags[idx + 1] != keep:
            continue
        if not toks[idx] in vocab:
            continue
        if not toks[idx + 1] in vocab:
            continue
        tags[idx] = "$SPLIT_" + toks[idx]
        toks[idx] = toks[idx] + toks[idx + 1]
        toks.pop(idx + 1)
        tags.pop(idx + 1)
        output(toks, tags, args)
        seen["$SPLIT_"] += 1
        return 1
    return 0


def reinflect(txt, inflection, lempos2wrds):
    infl = inflection.split(";")
    lem = infl[0]
    pos = infl[1]
    # print("reinflect {} {}".format(lem,pos))
    for wrd in lempos2wrds[lem + separ + pos]:
        if wrd != txt:
            return wrd
    return ""


def do_inflect(toks, tags, txt2inf, lempos2wrds, seen, args):
    ### replace word i if in vocab by another with different inflection, tag it with INFLECT:inflection
    idxs = list(range(len(toks) - 1))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep:
            continue
        if toks[idx] not in txt2inf:
            continue
        curr_inflection = txt2inf[toks[idx]]
        new_txt = reinflect(toks[idx], curr_inflection, lempos2wrds)
        if new_txt == "":
            continue
        curr_inflection = ";".join(curr_inflection.split(";")[1:])  ### discard lemma
        tags[idx] = "$INFLECT:" + curr_inflection
        toks[idx] = new_txt
        output(toks, tags, args)
        seen[tags[idx]] += 1
        return 1
    return 0


def do_delete(toks, tags, vocab, seen, args):
    ### insert a random word (tag it as DELETE)
    idxs = list(range(len(toks)))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep:
            continue
        if random.random() < 0.5:
            ### random word
            lvocab = list(vocab)
            txt_new = lvocab[random.randint(0, len(lvocab) - 1)]
        else:
            ### copy toks[idx]
            txt_new = toks[idx]
        toks.insert(idx, txt_new)
        tags.insert(idx, "$DELETE")
        output(toks, tags, args)
        seen[tags[idx]] += 1
        return 1
    return 0


def do_merge(toks, tags, vocab, seen, args):
    ### split toks[idx] in two tokens and tag the first to MERGE
    idxs = list(range(len(toks)))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep:
            continue
        if len(toks[idx]) < 2:
            continue
        if not toks[idx] in vocab:
            continue
        k = random.randint(1, len(toks[idx]) - 1)
        ls = toks[idx][:k]
        rs = toks[idx][k:]
        toks[idx] = "".join(rs)
        tags[idx] = keep
        toks.insert(idx, "".join(ls))
        tags.insert(idx, "$MERGE")
        output(toks, tags, args)
        seen[tags[idx]] += 1
        return 1
    return 0


def do_swap(toks, tags, vocab, seen, args):
    ### swap tokens i and i+1 (tag the first as SWAP)
    idxs = list(range(len(toks) - 1))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep or tags[idx + 1] != keep:
            continue
        if not toks[idx] in vocab:
            continue
        if not toks[idx - 1] in vocab:
            continue
        curr_txt = toks[idx]
        curr_tag = tags[idx]
        toks[idx] = toks[idx + 1]
        tags[idx] = "$SWAP"
        toks[idx + 1] = curr_txt
        tags[idx + 1] = curr_tag
        output(toks, tags, args)
        seen[tags[idx]] += 1
        return 1
    return 0


def do_case_first(toks, tags, vocab, seen, args):
    ### change the case of first char in token
    idxs = list(range(len(toks)))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep:
            continue
        if len(toks[idx]) < 2:
            continue
        if not toks[idx] in vocab:
            continue
        if not toks[idx].isalpha():
            continue
        first = toks[idx][0]
        rest = toks[idx][1:]
        first = first.lower() if first.isupper() else first.upper()
        toks[idx] = first + rest
        tags[idx] = "$CASE:FIRST"
        output(toks, tags, args)
        seen[tags[idx]] += 1
        return 1
    return 0


def do_case_upper(toks, tags, vocab, seen, args):
    ### lower case all chars in a (uppercased) token and tag it as CASEUPPER
    idxs = list(range(len(toks)))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep:
            continue
        if not toks[idx] in vocab:
            continue
        if not toks[idx].isalpha():
            continue
        if not toks[idx].isupper():
            continue
        toks[idx] = toks[idx].lower()
        tags[idx] = "$CASE:UPPER"
        output(toks, tags, args)
        seen[tags[idx]] += 1
        return 1
    return 0


def do_case_lower(toks, tags, vocab, seen, args):
    ### upper case all chars in a (lowercased) token and tag it as CASELOWER
    idxs = list(range(len(toks)))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep:
            continue
        if not toks[idx] in vocab:
            continue
        if not toks[idx].isalpha():
            continue
        if not toks[idx].islower():
            continue
        toks[idx] = toks[idx].upper()
        tags[idx] = "$CASE:LOWER"
        output(toks, tags, args)
        seen[tags[idx]] += 1
        return 1
    return 0


def do_hyphen_split(toks, tags, vocab, seen, args):
    ### take two consecutive words and join them with an hyphen, tag it as HYPHEN_SPLIT
    idxs = list(range(len(toks) - 1))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep or tags[idx + 1] != keep:
            continue
        if not toks[idx] in vocab or not toks[idx + 1] in vocab:
            continue
        if not toks[idx].isalpha() or not toks[idx + 1].isalpha():
            continue
        toks[idx] = toks[idx] + "-" + toks[idx + 1]
        tags[idx] = "$HYPHEN:SPLIT"
        toks.pop(idx + 1)
        tags.pop(idx + 1)
        output(toks, tags, args)
        seen[tags[idx]] += 1
        return 1
    return 0


def do_hyphen_merge(toks, tags, vocab, seen, args):
    ### take a word with an hyphen and split it into two, tag the first as HYPHEN_MERGE
    idxs = list(range(len(toks)))
    random.shuffle(idxs)
    for idx in idxs:
        if tags[idx] != keep:
            continue
        if toks[idx].count("-") != 1:
            continue
        p = toks[idx].find("-", 1, len(toks[idx]) - 1)
        if p == -1:
            continue
        first = toks[idx][:p]
        second = toks[idx][p + 1 :]
        if not first in vocab or not second in vocab:
            continue
        toks[idx] = second
        toks.insert(idx, first)
        tags.insert(idx, "$HYPHEN:MERGE")
        output(toks, tags, args)
        seen[tags[idx]] += 1
        return 1
    return 0


def noise_line(
    toks,
    txt2inf,
    vocab,
    wrd2wrds_same_lemma,
    wrd2wrds_homophones,
    lempos2wrds,
    wrd2pos,
    seen,
    args,
):
    tags = [keep for t in toks]
    if random.random() < args.p_clean:
        output(toks, tags, args)  ### prints without noise
        seen["clean"] += 1

    n_attempts = 0
    n_changes = 0
    while n_attempts < args.wmax * 2 and n_changes < args.wmax:
        n_attempts += 1
        choice = random.choices(args.noises, args.weights, k=1)[0]
        if choice == "replace:inflection":
            n_changes += do_replace_inflection(
                toks, tags, vocab, wrd2wrds_same_lemma, seen, args
            )
        elif choice == "replace:homophone":
            n_changes += do_replace_homophone(
                toks, tags, vocab, wrd2wrds_homophones, seen, args
            )
        elif choice == "delete":
            n_changes += do_delete(toks, tags, vocab, seen, args)
        elif choice == "append":
            n_changes += do_append(toks, tags, wrd2pos, vocab, seen, args)
        elif choice == "inflect":
            n_changes += do_inflect(toks, tags, txt2inf, lempos2wrds, seen, args)
        elif choice == "split":
            n_changes += do_split(toks, tags, vocab, seen, args)
        elif choice == "replace:spell":
            n_changes += do_replace_spell(toks, tags, vocab, seen, args)
        elif choice == "merge":
            n_changes += do_merge(toks, tags, vocab, seen, args)
        elif choice == "swap":
            n_changes += do_swap(toks, tags, vocab, seen, args)
        elif choice == "hyphen:merge":
            n_changes += do_hyphen_merge(toks, tags, vocab, seen, args)
        elif choice == "hyphen:split":
            n_changes += do_hyphen_split(toks, tags, vocab, seen, args)
        elif choice == "case:first":
            n_changes += do_case_first(toks, tags, vocab, seen, args)
        elif choice == "case:lower":
            n_changes += do_case_lower(toks, tags, vocab, seen, args)
        elif choice == "case:upper":
            n_changes += do_case_upper(toks, tags, vocab, seen, args)


def read_vocab(f):
    vocab = set()
    if not f:
        return vocab
    with open(f, "r") as fd:
        for l in fd:
            if " " in l:
                continue
            l = l.rstrip()
            if len(l) == 0:
                continue
            vocab.add(l)
    return vocab


def read_lexicon(f, vocab):
    lem2wrd = defaultdict(set)
    wrd2lem = defaultdict(set)
    pho2wrd = defaultdict(set)
    wrd2pho = defaultdict(set)
    wrd2pos = defaultdict(set)
    wrd2wrds_same_lemma = defaultdict(set)
    wrd2wrds_homophones = defaultdict(set)
    lempos2wrds = defaultdict(set)
    if not f:
        return wrd2wrds_same_lemma, wrd2wrds_homophones, lempos2wrds, wrd2pos

    with open(f, "r") as fd:
        for l in fd:
            toks = l.rstrip().split("\t")
            if len(toks) < 3:
                continue
            wrd, pho, lem, pos = toks[0], toks[1], toks[2], toks[3]
            if wrd not in vocab:
                continue
            if " " in wrd or " " in lem or " " in pho or " " in pos:
                continue
            if pos == "VER":
                pos = "VERB"  # use same tag as SpaCy
            lem2wrd[lem].add(wrd)
            wrd2lem[wrd].add(lem)
            pho2wrd[pho].add(wrd)
            wrd2pho[wrd].add(pho)
            lempos2wrds[lem + separ + pos].add(wrd)
            wrd2pos[wrd].add(pos)

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

    return wrd2wrds_same_lemma, wrd2wrds_homophones, lempos2wrds, wrd2pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", help="Input file/s", nargs="+")
    ### optional
    parser.add_argument(
        "--vocab", help="File with words to consider (required)", required=True
    )
    parser.add_argument(
        "--lexicon", help="Lexique383.tsv lexicon file (required)", required=True
    )
    parser.add_argument(
        "--wmax",
        type=int,
        default=5,
        help="Max number of words noised per sentence (5)",
    )
    parser.add_argument(
        "--p_clean",
        type=float,
        default=0.2,
        help="Prob of printing an unnoised sentence (0.2)",
    )

    parser.add_argument(
        "--w_replace_inflection",
        type=float,
        default=5,
        help="Weight of replacing a word inflection (5)",
    )
    parser.add_argument(
        "--w_replace_homophone",
        type=float,
        default=5,
        help="Weight of replacing a word with an homophone (5)",
    )
    parser.add_argument(
        "--w_replace_spell",
        type=float,
        default=5,
        help="Weight of replacing a misspelled word (5)",
    )
    parser.add_argument(
        "--w_split", type=float, default=5, help="Weight of splitting one word (5)"
    )
    parser.add_argument(
        "--w_append", type=float, default=5, help="Weight of appending one word (5)"
    )

    parser.add_argument(
        "--w_inflect",
        type=float,
        default=5,
        help="Weight of inflecting a word (5) [Uses SpaCy]",
    )
    parser.add_argument(
        "--w_delete",
        type=float,
        default=1,
        help="Weight of deleting a random/copied word (1)",
    )
    parser.add_argument(
        "--w_swap", type=float, default=1, help="Weight of swapping two words (1)"
    )
    parser.add_argument(
        "--w_merge", type=float, default=1, help="Weight of merging two words (1)"
    )
    parser.add_argument(
        "--w_hyph_merge",
        type=float,
        default=1,
        help="Weight of merging two words with an hyphen (1)",
    )
    parser.add_argument(
        "--w_hyph_split",
        type=float,
        default=1,
        help="Weight of splitting one word with an hyphen (1)",
    )
    parser.add_argument(
        "--w_case_first",
        type=float,
        default=1,
        help="Weight of changing the first char case (1)",
    )
    parser.add_argument(
        "--w_case_upper", type=float, default=1, help="Weight of uppercasing a word (1)"
    )
    parser.add_argument(
        "--w_case_lower", type=float, default=1, help="Weight of lowercasing a word (1)"
    )

    parser.add_argument(
        "--only_vocab",
        action="store_true",
        help="Weights of noises not generating words are set to 0",
    )
    parser.add_argument(
        "--output_pair",
        action="store_true",
        help="Output tab-separated strings of words/tags",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for randomness (0)")
    args = parser.parse_args()
    if args.only_vocab:
        args.w_inflect = 0
        args.w_delete = 0
        args.w_swap = 0
        args.w_merge = 0
        args.w_hyph_merge = 0
        args.w_hyph_split = 0
        args.w_case_first = 0
        args.w_case_upper = 0
        args.w_case_lower = 0

    args.noises = [
        "replace:inflection",
        "replace:homophone",
        "replace:spell",
        "split",
        "append",
        "inflect",
        "delete",
        "swap",
        "merge",
        "hyphen:merge",
        "hyphen:split",
        "case:first",
        "case:upper",
        "case:lower",
    ]
    args.weights = [
        args.w_replace_inflection,
        args.w_replace_homophone,
        args.w_replace_spell,
        args.w_split,
        args.w_append,
        args.w_inflect,
        args.w_delete,
        args.w_swap,
        args.w_merge,
        args.w_hyph_merge,
        args.w_hyph_split,
        args.w_case_first,
        args.w_case_upper,
        args.w_case_lower,
    ]
    if args.seed:
        random.seed(args.seed)
    if args.w_inflect:
        import spacy

        nlp = spacy.load("fr_core_news_md")
    t = pyonmttok.Tokenizer("conservative", joiner_annotate=False)

    vocab = read_vocab(args.vocab)
    wrd2wrds_same_lemma, wrd2wrds_homophones, lempos2wrds, wrd2pos = read_lexicon(
        args.lexicon, vocab
    )

    tic = time.time()
    nsents = 0
    ntokens = 0
    seen = defaultdict(int)
    for f in args.files:
        sys.stderr.write("Reading {}\n".format(f))
        sys.stderr.flush()
        with open(f) as fd:
            lines = []
            for l in fd:
                toks, _ = t.tokenize(l.rstrip())
                txt2inf = do_spacy(nlp, l, toks) if args.w_inflect else []
                # for tok in toks:
                #    print('TOKEN: {}'.format(tok))
                # for txt in txt2inf:
                #    print('SPACY: {} {}'.format(txt,txt2inf[txt]))
                noise_line(
                    toks,
                    txt2inf,
                    vocab,
                    wrd2wrds_same_lemma,
                    wrd2wrds_homophones,
                    lempos2wrds,
                    wrd2pos,
                    seen,
                    args,
                )
                nsents += 1
                ntokens += len(toks)
    toc = time.time()
    sys.stderr.write(
        "Processed {} input sentences in {:.2f} seconds\n".format(nsents, toc - tic)
    )
    for k, v in sorted(seen.items(), key=lambda kv: kv[1], reverse=True):
        sys.stderr.write("{}\t{}\n".format(v, k))

