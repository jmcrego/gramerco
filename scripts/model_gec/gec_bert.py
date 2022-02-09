from data.gramerco_dataset import GramercoDataset
from transformers import FlaubertTokenizer, FlaubertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import sys
import os
import itertools

pwd = os.path.dirname(__file__)
sys.path.append(os.path.dirname(pwd))
sys.path.append(os.path.dirname(os.path.abspath(pwd)))
try:
    from utils import word_collate
except BaseException:
    from .utils import word_collate


class GecBertModel(nn.Module):
    """Base Gec Model with 1 final layer for all possible tags.
    """
    def __init__(
        self,
        num_tag,
        encoder_name="flaubert/flaubert_base_cased",
        tokenizer=None,
        tagger=None,
        mid=None,
        freeze_encoder=False,
        dropout=0.,
    ):
        super(GecBertModel, self).__init__()

        self.tokenizer = (
            tokenizer if tokenizer else
            FlaubertTokenizer.from_pretrained(encoder_name)
        )
        self.encoder = FlaubertModel.from_pretrained(encoder_name)
        self.num_tag = num_tag
        h_size = self.encoder.attentions[0].out_lin.out_features
        self.linear_layer = nn.Linear(h_size, num_tag)
        if dropout > 0.0:
            self.dropout_layer = nn.Dropout(p=dropout)
        else:
            self.dropout_layer = None
        self.id = mid
        self.freeze_encoder = freeze_encoder

    def _tokenize_text(self, texts):
        return self.tokenizer(texts, return_tensors="pt", padding=True)

    def _is_end_of_word(self, idx):
        tok = self.tokenizer._convert_id_to_token(idx)
        return tok[-4:] in ["</w>", "<s>", "</s>"]  # and tok != '"</w>'

    def _generate_word_index(self, x):
        word_ends = x.clone().cpu().apply_(lambda t: self._is_end_of_word(t))
        tildes = (x == self.tokenizer._convert_token_to_id("~</w>"))
        # logging.debug("tildes " + str(tildes[7].float()))
        comb_tildes = tildes[:, 1:-1] | tildes[:, 2:] | tildes[:, :-2]
        # logging.debug("comb tildes " + str(comb_tildes[7].float()))
        word_ends[:, 1:-1][comb_tildes] = 0.  # consider expresions as 1 "word" to be tagged

        alone_apos = (x == self.tokenizer._convert_token_to_id("'</w>"))[:, 1:]
        word_ends[:, :-1][alone_apos] = 0.
        # logging.debug("word ends " + str(word_ends[7]))
        word_ends = torch.roll(word_ends, 1, -1)
        word_ends[:, 0] = 0
        return torch.cumsum(word_ends, -1)

    def forward_encoder(self, **inputs):
        if self.freeze_encoder:
            with torch.no_grad():
                h = self.encoder(**inputs)
        else:
            h = self.encoder(**inputs)
        # index tokens indicating how to group them into words.
        word_index = self._generate_word_index(
            inputs["input_ids"]).to(h.last_hidden_state.device)

        # agregate encoding states according to the word indexing.
        h_w = word_collate(h.last_hidden_state, word_index)
        # adapt the output attention mask to update padding.
        attention_mask_larger = word_collate(
            inputs["attention_mask"].unsqueeze(-1), word_index, agregation="max"
        ).squeeze(-1)
        return h, h_w, attention_mask_larger


    def forward(self, **inputs):
        h, h_w, attention_mask_larger = self.forward_encoder(**inputs)

        attention_mask = torch.zeros_like(
            attention_mask_larger).to(h.last_hidden_state.device)
        attention_mask[:, 1:-1] = attention_mask_larger[:, 2:]
        out = self.linear_layer(h_w)
        if self.dropout_layer:
            out = self.dropout_layer(out)
        # out = torch.softmax(out, -1)
        # out = self.ls(out)
        return {"tag_out": out, "attention_mask": attention_mask}

    def parameters(self):
        if self.freeze_encoder:
            return self.linear_layer.parameters()
        return super().parameters()


class GecBert2DecisionsModel(GecBertModel):
    """Gec Model with 2 final layers.
    One for keep/error detection.
    One for tag preditcion along all possible tags.
    """

    def __init__(
        self,
        num_tag,
        **kwargs
    ):
        super(GecBert2DecisionsModel, self).__init__(
            num_tag,
            **kwargs
        )

        h_size = self.encoder.attentions[0].out_lin.out_features
        self.linear_layer = nn.Linear(h_size, num_tag - 1)
        self.decision_layer = nn.Linear(h_size, 2)
        if "dropout" in kwargs and kwargs["dropout"] > 0.0:
            self.dropout_dec_layer = nn.Dropout(p=kwargs["dropout"])
        else:
            self.dropout_dec_layer = None

    def forward(self, **inputs):
        h, h_w, attention_mask_larger = self.forward_encoder(**inputs)

        attention_mask = torch.zeros_like(
            attention_mask_larger).to(h.last_hidden_state.device)
        attention_mask[:, 1:-1] = attention_mask_larger[:, 2:]
        out = self.linear_layer(h_w)
        out_decision = self.decision_layer(h_w)
        if self.dropout_layer:
            out = self.dropout_layer(out)
            out_decision = self.dropout_dec_layer(out_decision)
        # out = torch.softmax(out, -1)
        # out = self.ls(out)
        return {
            "tag_out": out,
            "decision_out": out_decision,
            "attention_mask": attention_mask,
        }

    def parameters(self):
        if self.freeze_encoder:
            return iter(itertools.chain(self.linear_layer.parameters(), self.decision_layer.parameters()))
        return super().parameters()


class GecBertVocModel(GecBertModel):
    """Gec Model with 2 final layers.
    One for error type tag prediction (including all inflections).
    One for word prediction on a given vocabulary
    of tens of thousands of words.
    """
    def __init__(
        self,
        num_tag,
        num_voc,
        **kwargs
    ):
        super(GecBertVocModel, self).__init__(num_tag, **kwargs)
        h_size = self.encoder.attentions[0].out_lin.out_features
        # Linear layer of vocabulary
        self.voc_layer = nn.Linear(h_size, num_voc)

    def forward(self, **inputs):
        h, h_w, attention_mask_larger = self.forward_encoder(**inputs)

        attention_mask = torch.zeros_like(
            attention_mask_larger).to(h.last_hidden_state.device)
        attention_mask[:, 1:-1] = attention_mask_larger[:, 2:]
        out_tag = self.linear_layer(h_w)
        out_voc = self.voc_layer(h_w)

        # out = torch.softmax(out, -1)
        # out = self.ls(out)
        return {"tag_out": out_tag, "voc_out": out_voc, "attention_mask": attention_mask}

    def parameters(self):
        if self.freeze_encoder:
            return iter(itertools.chain(
                self.linear_layer.parameters(),
                self.voc_layer.parameters()
            ))
        return super().parameters()


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
    create_logger("stderr", "DEBUG")
    model = GecBertModel(50).train()

    path_to_bin = (
        "/home/bouthors/workspace/gramerco-repo/gramerco/resources/bin/data.train"
    )

    bsz = 4
    dataset = GramercoDataset(path_to_bin, "fr")
    dataloader = DataLoader(dataset, shuffle=True, batch_size=bsz)

    batch = next(iter(dataloader))
    out = model(**batch["noise_data"])
    for i in range(bsz):
        x = batch["noise_data"]["input_ids"][i]
        m = batch["noise_data"]["attention_mask"][i]
        logging.debug(
            " ".join(
                [model.tokenizer._convert_id_to_token(s.item()) for s in x[m == 1]]
            )
        )
    logging.info("noise data lens = " +
                 str(batch["noise_data"]["attention_mask"].sum(-1)))
    logging.info("tag data lens = " +
                 str(batch["tag_data"]["attention_mask"].sum(-1)))
    logging.info("tag out lens = " + str(out["attention_mask"].sum(-1)))
    logging.info("over")
