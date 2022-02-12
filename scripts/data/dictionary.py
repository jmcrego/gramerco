import os
from collections import Counter
from multiprocessing import Pool

import torch
from fairseq import utils
from fairseq.data import data_utils
from fairseq.file_chunker_utils import Chunker, find_offsets
from fairseq.file_io import PathManager
from fairseq.tokenizer import tokenize_line
from itertools import accumulate


class Dictionary:
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        encoder,
        encoder_type,
        version=1
    ):
        assert encoder_type in [
            "tag_encoder", "tokenizer", "pretokenizer", "word_index"
        ]
        self.encoder = encoder
        self.encoder_type = encoder_type
        self.version = version
        if version == 1:
            if encoder_type == "tag_encoder":
                self.dic_itt = encoder._id_to_tag
                self.dic_tti = encoder._tag_to_id
            elif encoder_type == "tokenizer" or encoder_type == "pretokenizer":
                self.dic_itt = encoder.encoder
                self.dic_tti = encoder.decoder
            else:
                self.dic_itt = dict()
                self.dic_tti = dict()
        if encoder_type == "tag_encoder" or encoder_type == "word_index":
            self.unk_index = 0
        elif encoder_type == "tokenizer" or encoder_type == "pretokenizer":
            self.unk_index = self.encoder.unk_token

        if encoder_type == "word_index":
            self.unk_word = 0
        else:
            self.unk_word = (
                self.dic_itt[self.unk_index] if self.version == 1
                else self.encoder._id_to_tag[0]
            )
        self.indices = {}

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if self.encoder_type == "word_index":
            return idx
        if self.version == 2:
            return self.encoder.id_to_tag(idx)
        if idx < len(self.dic_itt):
            return self.dic_itt.get(idx, self.unk_index)

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        if self.version == 2:
            return self.encoder.get_num_encodable()
        return len(self.dic_itt)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if self.version == 2:
            return self.encoder.tag_to_id(sym)
        if sym in self.dic_tti:
            return self.dic_tti[sym]
        return self.unk_index

    def string(self, tensor):
        """Helper for converting a tensor of token indices to a string.
        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(self.string(t) for t in tensor)

        def token_string(i):
            if i == self.unk_index():
                return self.unk_index_string('<unk>')
            else:
                return self[i]

        sent = ' '.join(token_string(i) for i in tensor)

        return sent

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return "<{}>".format("unk")
        else:
            return self.unk_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    def _get_meta(self):
        return [], []

    def _load_meta(self, lines):
        return 0

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(5, len(self)).long()
        t[-1] = self.eos()
        return t

    def encode_line(
        self,
        line,
        line_tokenizer=tokenize_line,
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        reverse_order=False,
    ) -> torch.IntTensor:
        if self.encoder_type == "tokenizer":
            ids = self.encoder([line], return_tensors="pt").input_ids[0]
        elif self.encoder_type == "tag_encoder":
            ids = self.encoder.encode_line(line)
        elif self.encoder_type == "pretokenizer":
            ids = eval(line.rstrip('\n'))
            ids = [e for sub in ids for e in sub]
            ids = [0] + ids + [1]
            ids = torch.tensor(ids, dtype=torch.long)
        elif self.encoder_type == "word_index":
            ids = eval(line.rstrip('\n'))
            wids = [0]
            idx = 1
            iteration = 0
            for sub in ids:
                for e in sub:
                    wids.append(idx)
                    iteration += 1
                idx += 1
            wids.append(idx)
            ids = torch.tensor(wids, dtype=torch.long)

        if consumer is not None:
            for i, wid in enumerate(ids):
                consumer(self[i], wid.item())
        return ids
