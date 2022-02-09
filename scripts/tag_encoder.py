from collections import defaultdict
import torch
from noiser.Noise import Spacy
from transformers import FlaubertTokenizer
import os
import sys
import logging
pwd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(pwd))
sys.path.append(os.path.dirname(os.path.abspath(pwd)))
try:
    from noiser.add_french_noise import read_rep, read_app
    from noiser.noise import read_vocab
except BaseException:
    from .noiser.add_french_noise import read_rep, read_app
    from .noiser.noise import read_vocab

separ = '￨'


def default_keep_tok():
    return '·'


def default_empty():
    return ''


def default_keep_id():
    return 0


# Encoder used to encode tags to integers (long)
class TagEncoder:
    """Tag Encoder compatible with GecBertModel and GecBert2DecisionsModel.
    """
    def __init__(
            self,
            path_to_lex="/home/bouthors/workspace/gramerco-repo/gramerco/resources/Lexique383.tsv",
            path_to_app="/home/bouthors/workspace/gramerco-repo/gramerco/resources/lexique.app",
    ):
        f = open(path_to_app, 'r')
        f.close()
        rep = read_rep(path_to_lex)
        app = read_app(path_to_app)

        self._id_to_tag = defaultdict(default_keep_tok)
        self._tag_to_id = defaultdict(default_keep_id)
        self._curr_cpt = 1

        self.error_type_id = {
            "DELETE": 0,
            "COPY": 1,
            "SWAP": 2,
            "MERGE": 3,
            "CASE": 4,
            "SPLIT": 5,
            "HYPHEN": 6,
            "APPEND": 7,
            "TRANSFORM": 8,
            "REPLACE": 9,
        }

        self.id_error_type = [
            "DELETE",
            "COPY",
            "SWAP",
            "MERGE",
            "CASE",
            "SPLIT",
            "HYPHEN",
            "APPEND",
            "TRANSFORM",
            "REPLACE",
        ]

        self.add_tag("$DELETE")
        self.add_tag("$COPY")
        self.add_tag("$SWAP")
        self.add_tag("$MERGE")
        self.add_tag("$CASE")
        self.add_tag("$SPLIT")
        self.add_tag("$HYPHEN")

        for pos in ["ADJ", "NOM"]:
            for genre in ["m", "f", "-"]:
                for nombre in ["s", "p", "-"]:
                    self.add_tag("$TRANSFORM_" +
                                 separ.join([pos, genre, nombre]))

        for pos in ["VER", "AUX"]:
            self.add_tag("$TRANSFORM_" + separ.join([pos, "-", "-", "inf"]))

            for genre in ["m", "f"]:
                for nombre in ["s", "p"]:
                    for tense in ["pas", "pre"]:
                        self.add_tag("$TRANSFORM_" +
                                     separ.join([pos, genre, nombre, "par", tense]))

            for tense in ["pas", "pre", "fut", "imp"]:
                for nombre in ["s", "p"]:
                    for pers in ["1", "2", "3"]:
                        self.add_tag(
                            "$TRANSFORM_" + separ.join([pos, "-", "-", "ind", tense, pers + nombre]))

            for nombre in ["s", "p"]:
                for pers in ["1", "2", "3"]:
                    self.add_tag(
                        "$TRANSFORM_" + separ.join([pos, "-", "-", "sub", "pre", pers + nombre]))

        for app_tok in app:
            self.add_tag("$APPEND_" + app_tok)

        for pos in rep["pos2mot"]:  # ART + PRO + PRE + ADV
            for tok in rep["pos2mot"][pos]:
                self.add_tag("$" + pos + "_" + tok)

    def encode_line(self, line):
        # Method used in processing for encoding + binarization w/ fairseq.
        return torch.tensor(
            list(map(self.tag_to_id, line.split(" "))), dtype=torch.int64)

    def id_to_tag(self, i):
        return self._id_to_tag[i]

    def tag_to_id(self, tag):
        return self._tag_to_id[tag]

    def add_tag(self, tag):
        self._id_to_tag[self._curr_cpt] = tag
        self._tag_to_id[tag] = self._curr_cpt
        self._curr_cpt += 1

    def get_tag_category(self, tag):
        # Method to retrieve tag category (error type) for any tag/tag id.
        if isinstance(tag, int):
            tag = self.id_to_tag(tag)
        error_type = tag[1:].split('_')[0]
        if error_type in ["ART", "PRO", "PRE", "ADV"]:
            error_type = "REPLACE"
        # DELETE, COPY, SWAP, SPLIT, HYPHEN, CASE, TRANSFORM, APPEND, REPLACE
        if error_type in self.error_type_id:
            return self.error_type_id[error_type]
        return self.error_type_id["KEEP"]

    def size(self):
        return self._curr_cpt

    def __len__(self):
        return self.size()


class TagEncoder2(TagEncoder):
    """Tag Encoder compatible with GecBertVocModel.
    For each tag ($DELETE, $INFLECT:inflection, $REPLACE:SPELL_word)
    is associated a unique id.
    This id can be decomposed in two distinct ids : error type and word.
    """
    def __init__(
            self,
            path_to_lex="/nfs/RESEARCH/bouthors/projects/gramerco/resources/Lexique383.tsv",
            path_to_voc="/nfs/RESEARCH/bouthors/projects/gramerco/resources/common/french.dic.20k",
            new_version=False,
    ):

        rep = read_rep(path_to_lex)
        voc = read_app(path_to_voc)
        self.worder = WordEncoder(path_to_voc)

        self._id_to_tag = defaultdict(default_keep_tok) # only for error type
        self._tag_to_id = defaultdict(default_keep_id) # only for error type
        self._curr_cpt = 1
        self._w_cpt = 0

        self.id_error_type = [
            "KEEP",  # .
            "DELETE",
            "SWAP",
            "MERGE",
            "HYPHEN:SPLIT",
            "HYPHEN:MERGE",
            "CASE:FIRST",
            "CASE:UPPER",
            "CASE:LOWER",
            "INFLECT",  # inflections
            "APPEND",  # word
            "REPLACE:INFLECTION",  # word
            "REPLACE:HOMOPHONE",  # word
            "REPLACE:SPELL",  # word
            "SPLIT",  # word
        ]

        if new_version:
            self.id_error_type.append(
                "$REPLACE:SAMEPOS",  # word
            )            

        self.error_type_id = {
            key: i for i, key in enumerate(self.id_error_type)
        }
        for error in self.id_error_type[1:-6]:
            self.add_tag("$" + error)

        with open(
            "/nfs/RESEARCH/bouthors/projects/gramerco/resources/common/morphs-tag.txt",
            'r'
        ) as f:
            for line in f.readlines():
                self.add_tag("$INFLECT:" + line.rstrip('\n'))

        self.add_tag("$APPEND", word=True)
        self.add_tag("$REPLACE:INFLECTION", word=True)
        self.add_tag("$REPLACE:HOMOPHONE", word=True)
        self.add_tag("$REPLACE:SPELL", word=True)
        self.add_tag("$SPLIT", word=True)

    def add_tag(self, tag, word=False):
        self._id_to_tag[self._curr_cpt] = tag
        self._tag_to_id[tag] = self._curr_cpt
        self._curr_cpt += 1
        self._w_cpt += int(word)

    def id_to_tag(self, i):
        if i < self.size() - self._w_cpt:
            return self._id_to_tag[i]
        j = i - self._curr_cpt + self._w_cpt

        # decomposition in error_type/word ids
        tag = j // len(self.worder)
        word = j % len(self.worder)

        return (
            self._id_to_tag[self._curr_cpt - tag - 1] +
            "_" +
            self.worder.id_to_word[word]
        )

    def tag_word_to_id(self, word_id, tag_id):
        # recompose error_type/word id in global tag id.
        if tag_id >= self._curr_cpt - self._w_cpt:
            return (
                self._curr_cpt - self._w_cpt +
                len(self.worder) * (self._curr_cpt - tag_id - 1) +
                word_id
            )
        return tag_id

    def tag_word_to_id_vec(self, word_id, tag_id):
        # recompose error_type/words id in global tag ids.
        # But compatible with tensors!
        ids = tag_id.clone()
        mask = (tag_id >= self._curr_cpt - self._w_cpt)
        ids[mask] = (
            self._curr_cpt - self._w_cpt +
            len(self.worder) * (self._curr_cpt - tag_id[mask] - 1) +
            word_id[mask]
        )
        return ids

    def tag_to_id(self, tag):
        # convert string tag to tag id
        if self.is_word_tag(tag):
            tags = tag.split('_')
            word = tags[-1].rstrip('\n')
            # logging.info(" >> " + str(word) + '|')
            word = self.worder.word_to_id[word]
            tag = '_'.join(tags[:-1])
            # logging.info(str(word))
            cls = self._tag_to_id[tag]
            cls = self._curr_cpt - cls - 1
            #
            # logging.info(" ".join([str(self._curr_cpt), str(self._w_cpt), str(len(self.worder)), str(cls), str(word)]))
            # logging.info(" ".join([str(type(self._curr_cpt)), str(type(self._w_cpt)), str(type(len(self.worder))), str(type(cls)), str(type(word))]))
            return self._curr_cpt - self._w_cpt + len(self.worder) * cls + word

        return self._tag_to_id[tag]

    def id_to_tag_id(self, i):
        # extract error_type id from id
        if i < self.size() - self._w_cpt:
            return i
        return self.size() - ((i - self.size() + self._w_cpt) // len(self.worder)) - 1

    def id_to_tag_id_vec(self, x):
        # extract error_type id from id
        # But compatible with tensors!
        y = x.clone()
        mask = (x >= (self.size() - self._w_cpt))
        y[mask] = (
            self.size() -
            torch.div(
                x[mask] - self.size() + self._w_cpt,
                len(self.worder),
                rounding_mode='floor'
            ) -
            1
        )
        return y

    def id_to_word_id(self, i):
        # extract word id from id
        if i < self.size() - self._w_cpt:
            # not APPEND, REPLACE, SPLIT
            return -1
        return ((i - self.size() + self._w_cpt) % len(self.worder))

    def id_to_word_id_vec(self, x):
        # extract word id from id
        # But compatible with tensors!
        y = x.new(x.shape).fill_(-1)
        mask = (x >= (self.size() - self._w_cpt))
        y[mask] = (
            torch.remainder(
                x[mask] - self.size() + self._w_cpt,
                len(self.worder)
            )
        )
        return y

    def encode_line(self, line):
        # method used to encode + binarize tags in preprocessing
        return torch.tensor(
            list(map(self.tag_to_id, line.split(" "))), dtype=torch.int64)

    def is_word_tag(self, tag):
        return '_' in tag

    def is_radical_word_tag(self, tag):
        return (tag.startswith("$APPEND")
                or tag.startswith("$REPLACE")
                or (tag == "$SPLIT")
                )

    def id_to_type_id(self, tid):
        tag = self.id_to_tag(tid)
        if self.is_word_tag(tag):
            name = self.id_to_tag(tid)[1:].split('_')[0]
            return self.error_type_id[name]
        return self.error_type_id[tag[1:]]

    def id_to_word(self, tid):
        tag = self.id_to_tag(tid)
        if self.is_word_tag(tag):
            return tag.split('_')[-1]

    def get_tag_category(self, tag):
        if isinstance(tag, int):
            tag = self.id_to_tag(tag)
        if self.is_word_tag(tag):
            error_type = tag[1:].split('_')[0]
            return self.error_type_id[error_type]
        if tag.startswith("$INFLECT"):
            return self.error_type_id["INFLECT"]
        if tag[1:] in self.error_type_id:
            return self.error_type_id[tag[1:]]
        return self.error_type_id["KEEP"]

    def get_tag_category_pure(self, tag):
        cat = self.get_tag_category(tag)
        return cat

    @property
    def infer_ids(self):
        return [
            i for i in self._id_to_tag
            if self._id_to_tag[i].startswith("$INFLECT")
        ]

    def get_num_encodable(self):
        return self.size() + self._w_cpt * (len(self.worder) - 1)


class WordEncoder:
    """Encoder combined with TagEncoder2 to uniquely encode words
    from a given word dictionary."""
    def __init__(
        self,
        path_to_voc="/nfs/RESEARCH/bouthors/projects/gramerco/resources/common/french.dic.20k",
    ):
        voc = read_vocab(path_to_voc)
        self.id_to_word = {}
        self.word_to_id = {}
        self._curr_cpt = 0
        for i, word in enumerate(voc):
            self.add_word(word)

    def add_word(self, word):
        self.id_to_word[self._curr_cpt] = word
        self.word_to_id[word] = self._curr_cpt
        self._curr_cpt += 1

    def size(self):
        return self._curr_cpt

    def __len__(self):
        return self.size()


if __name__ == "__main__":

    tagger = TagEncoder2()

    from noiser.Noise import Lexicon
    lexicon = Lexicon("../resources/Lexique383.tsv")

    txt = """$HYPHEN:SPLIT $HYPHEN:MERGE $DELETE $APPEND_le"""

    for l in txt.split('\n'):
        ids = tagger.encode_line(l)
        voc = tagger.id_to_word_id_vec(ids)
        tag = tagger.id_to_tag_id_vec(ids)

        print(ids)
        print(tag)
        print(voc)


    # for i in range(0, 90000, 100):
    #     tag = tagger.id_to_tag(i)
        # print(i, tag)
        # tagger.tag_to_id(tag)
        # tagger.id_to_tag_id(i)
        # tagger.id_to_word_id(i)
        # print(
        #     i,
        #     tag,
        #     tagger.tag_to_id(tag),
        #     '\t\t\t',
        #     tagger.id_to_tag_id(i),
        #     tagger.id_to_word_id(i),
        #     '\t\t',
        #     tagger.get_tag_category(i)
        # )
        # print(
        #     tag, '\t', tagger.id_error_type[tagger.get_tag_category(i)]
        # )

    # for i in range(300, 20000 * 5, 5000):
    #     tag = tagger.id_to_tag(i)
    #     # print(i, tag)
    #     # tagger.tag_to_id(tag)
    #     # tagger.id_to_tag_id(i)
    #     # tagger.id_to_word_id(i)
    #     print(
    #         i,
    #         tag,
    #         tagger.tag_to_id(tag),
    #         '\t\t\t',
    #         tagger.id_to_tag_id(i),
    #         tagger.id_to_word_id(i))
    # i = 3000
    # print(i, tagger.id_to_tag(i))
    # print(tagger.size())
    # print(tagger.worder.size())
    # print(tagger.error_type_id)
    #
    # print(tagger.is_word_tag("$REPLACE:SPELL_ainsi"))
    # print(tagger.is_word_tag(
    #     "$INFLECT:VERB;Mood=Ind;Number=Plur;Person=1;Tense=Pres;VerbForm=Fin"))
    # print("$REPLACE:SPELL_ainsi", tagger.tag_to_id("$REPLACE:SPELL_ainsi"))
    #
    # print(tagger.worder.word_to_id["prestement"])
    # print(tagger.worder.id_to_word[1662])
    #
    # tagger.get_tag_category("$REPLACE:SPELL_ainsi")
