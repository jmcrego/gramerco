from transformers import FlaubertTokenizer

from torch.utils.data import DataLoader, Dataset
import logging
import torch
import sys
import os
import numpy as np

from fairseq.data import FairseqDataset, data_utils, iterators
from fairseq.data import FairseqDataset
import fairseq.data.indexed_dataset as indexed_dataset

try:
    from tag_encoder import TagEncoder
except BaseException:
    pwd = os.path.dirname(__file__)
    sys.path.append(os.path.abspath(pwd))
    sys.path.append(os.path.dirname(os.path.abspath(pwd)))
    from tag_encoder import TagEncoder


class GramercoDataset(FairseqDataset):
    def __init__(
        self,
        noise_dataset,
        noise_sizes,
        clean_dataset,
        clean_sizes,
        tag_dataset,
        tag_sizes,
        tokenizer,
        tagger,
    ):
        self.noise_dataset = noise_dataset
        self.clean_dataset = clean_dataset
        self.tag_dataset = tag_dataset
        self.noise_sizes = np.array(noise_sizes)
        self.clean_sizes = np.array(clean_sizes) if clean_sizes else None
        self.tag_sizes = np.array(tag_sizes)

        self.tokenizer = tokenizer
        self.tagger = tagger

        self.pad_idx = tokenizer.convert_tokens_to_ids("<pad>")
        self.bos_idx = tokenizer.convert_tokens_to_ids("<s>")
        self.eos_idx = tokenizer.convert_tokens_to_ids("</s>")

        self.return_clean = False

    def clean(self):
        self.return_clean = True

    def soil(self):
        self.return_clean = False

    def __getitem__(self, idx):

        sample = dict()
        sample["noise_data"] = self.noise_dataset[idx]
        sample["tag_data"] = self.tag_dataset[idx]

        if self.return_clean and self.clean_dataset:
            sample["clean_data"] = self.clean_dataset[idx]

        return sample

    def __len__(self):
        return len(self.tag_dataset)

    def merge(self, key, samples, pad_idx, eos_idx=None, bos_idx=None):
        # TODO: does it work with bos, eos = None
        merged = data_utils.collate_tokens(
            [s[key] for s in samples], pad_idx, eos_idx, bos_idx, False
        )
        attention_mask = merged.ne(pad_idx).float()
        if pad_idx == -1:
            merged[~attention_mask.bool()] = 0
        return {"input_ids": merged, "attention_mask": attention_mask}

    def collater(self, samples):  # fusionne les valeurs des dict des samples
        res = dict()
        res["noise_data"] = self.merge(
            "noise_data",
            samples,
            self.pad_idx,
            eos_idx=self.eos_idx,
            bos_idx=self.bos_idx,
        )
        if self.return_clean:
            res["clean_data"] = self.merge(
                "clean_data",
                samples,
                self.pad_idx,
                eos_idx=self.eos_idx,
                bos_idx=self.bos_idx,
            )
        res["tag_data"] = self.merge("tag_data", samples, -1)
        # samples[idx] = {"noise_data": ..., "tag_data": ..., ("clean_data": ...)}
        # return {"noise_data": {"input_ids", "attention_mask"}, "tag"...,
        # ("clean"...)}
        return res

    def num_tokens(self, index):
        return max(
            self.noise_sizes[index],
            self.tag_sizes[index],
            self.clean_sizes[index] if self.return_clean and self.clean_sizes else 0,
        )

    def num_tokens_vec(self, indices):
        sizes = np.maximum(self.noise_sizes[indices], self.tag_sizes[indices])
        if self.return_clean and self.clean_sizes:
            sizes = np.maximum(sizes, self.clean_sizes[indices])
        return sizes

    def size(self, index):
        return (
            self.noise_sizes[index],
            self.tag_sizes[index],
            self.clean_sizes[index] if self.return_clean and self.clean_sizes else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        # sort by target length, then source length
        indices = np.arange(len(self), dtype=np.int64)
        indices = indices[np.argsort(
            self.tag_sizes[indices], kind="mergesort")]
        return indices[np.argsort(self.noise_sizes[indices], kind="mergesort")]

    def filter_indices_by_size(self, indices, max_sizes, min_sizes):
        if self.clean_sizes:
            mask_ignored = (
                  (self.noise_sizes[indices] > max_sizes)
                | (self.tag_sizes[indices] > max_sizes)
                | (self.clean_sizes[indices] > max_sizes)
                | (self.noise_sizes[indices] < min_sizes)
                | (self.tag_sizes[indices] < min_sizes)
                | (self.clean_sizes[indices] < min_sizes)
            )
        else:
            mask_ignored = (
                  (self.noise_sizes[indices] > max_sizes)
                | (self.tag_sizes[indices] > max_sizes)
                | (self.noise_sizes[indices] < min_sizes)
                | (self.tag_sizes[indices] < min_sizes)
            )
        ignored = indices[mask_ignored]
        indices = indices[~mask_ignored]
        return indices, ignored.tolist()


class GramercoDataset2(FairseqDataset):
    def __init__(
        self,
        noise_dataset,
        noise_sizes,
        clean_dataset,
        clean_sizes,
        tag_dataset,
        tag_sizes,
        idx_dataset,
        idx_sizes,
        tokenizer,
        tagger,
    ):
        self.noise_dataset = noise_dataset
        self.clean_dataset = clean_dataset
        self.tag_dataset = tag_dataset
        self.idx_dataset = idx_dataset
        self.noise_sizes = np.array(noise_sizes)
        self.clean_sizes = np.array(clean_sizes) if clean_sizes else None
        self.tag_sizes = np.array(tag_sizes)
        self.idx_sizes = np.array(idx_sizes)

        self.tokenizer = tokenizer
        self.tagger = tagger

        self.pad_idx = tokenizer.convert_tokens_to_ids("<pad>")
        self.bos_idx = tokenizer.convert_tokens_to_ids("<s>")
        self.eos_idx = tokenizer.convert_tokens_to_ids("</s>")

        self.return_clean = False

    def clean(self):
        self.return_clean = True

    def soil(self):
        self.return_clean = False

    def __getitem__(self, idx):

        sample = dict()
        sample["noise_data"] = self.noise_dataset[idx]
        sample["tag_data"] = self.tag_dataset[idx]

        if self.return_clean and self.clean_dataset:
            sample["clean_data"] = self.clean_dataset[idx]

        sample["word_index"] = self.idx_dataset[idx]

        return sample

    def __len__(self):
        return len(self.tag_dataset)

    def merge(self, key, samples, pad_idx, eos_idx=None, bos_idx=None):
        # TODO: does it work with bos, eos = None
        merged = data_utils.collate_tokens(
            [s[key] for s in samples], pad_idx, eos_idx, bos_idx, False
        )
        attention_mask = merged.ne(pad_idx).float()
        if pad_idx == -1:
            merged[~attention_mask.bool()] = 0
        return {"input_ids": merged, "attention_mask": attention_mask}

    def collater(self, samples):  # fusionne les valeurs des dict des samples
        res = dict()
        res["noise_data"] = self.merge(
            "noise_data",
            samples,
            self.pad_idx,
            eos_idx=self.eos_idx,
            bos_idx=self.bos_idx,
        )
        if self.return_clean:
            res["clean_data"] = self.merge(
                "clean_data",
                samples,
                self.pad_idx,
                eos_idx=self.eos_idx,
                bos_idx=self.bos_idx,
            )
        res["tag_data"] = self.merge("tag_data", samples, -1)
        res["word_index"] = self.merge("word_index", samples, -1)
        # samples[idx] = {"noise_data": ..., "tag_data": ..., ("clean_data": ...)}
        # return {"noise_data": {"input_ids", "attention_mask"}, "tag"...,
        # ("clean"...)}
        return res

    def num_tokens(self, index):
        return max(
            self.noise_sizes[index],
            self.tag_sizes[index],
            self.clean_sizes[index] if self.return_clean and self.clean_sizes else 0,
        )

    def num_tokens_vec(self, indices):
        sizes = np.maximum(self.noise_sizes[indices], self.tag_sizes[indices])
        if self.return_clean and self.clean_sizes:
            sizes = np.maximum(sizes, self.clean_sizes[indices])
        return sizes

    def size(self, index):
        return (
            self.noise_sizes[index],
            self.tag_sizes[index],
            self.clean_sizes[index] if self.return_clean and self.clean_sizes else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        # sort by target length, then source length
        indices = np.arange(len(self), dtype=np.int64)
        indices = indices[np.argsort(
            self.tag_sizes[indices], kind="mergesort")]
        return indices[np.argsort(self.noise_sizes[indices], kind="mergesort")]

    def filter_indices_by_size(self, indices, max_sizes, min_sizes):
        if self.clean_sizes:
            mask_ignored = (
                  (self.noise_sizes[indices] > max_sizes)
                | (self.tag_sizes[indices] > max_sizes)
                | (self.clean_sizes[indices] > max_sizes)
                | (self.noise_sizes[indices] < min_sizes)
                | (self.tag_sizes[indices] < min_sizes)
                | (self.clean_sizes[indices] < min_sizes)
            )
        else:
            mask_ignored = (
                  (self.noise_sizes[indices] > max_sizes)
                | (self.tag_sizes[indices] > max_sizes)
                | (self.noise_sizes[indices] < min_sizes)
                | (self.tag_sizes[indices] < min_sizes)
            )
        ignored = indices[mask_ignored]
        indices = indices[~mask_ignored]
        return indices, ignored.tolist()


def make_dataset_from_prefix(
        prefix,
        tagger,
        tokenizer,
        return_clean=False,
        ignore_clean=True,
        word_index=False,
):
    logging.debug("...noise dataset")
    noise_dataset = indexed_dataset.make_dataset(
        prefix + ".noise.fr", impl="mmap")
    logging.debug("...tag dataset")
    tag_dataset = indexed_dataset.make_dataset(prefix + ".tag.fr", impl="mmap")
    if not ignore_clean:
        logging.debug("...clean dataset")
        clean_dataset = indexed_dataset.make_dataset(
            prefix + ".fr", impl="mmap")
    if word_index:
        logging.debug("...idx dataset")
        idx_dataset = indexed_dataset.make_dataset(
            prefix + ".widx.fr", impl="mmap")

    logging.debug("Gramerco dataset build")
    if ignore_clean:
        if word_index:
            dataset = GramercoDataset2(
                noise_dataset,
                noise_dataset.sizes,
                None,
                None,
                tag_dataset,
                tag_dataset.sizes,
                idx_dataset,
                idx_dataset.sizes,
                tokenizer,
                tagger,
            )
        else:
            dataset = GramercoDataset(
                noise_dataset,
                noise_dataset.sizes,
                None,
                None,
                tag_dataset,
                tag_dataset.sizes,
                tokenizer,
                tagger,
            )
    else:
        if word_index:
            dataset = GramercoDataset2(
                noise_dataset,
                noise_dataset.sizes,
                clean_dataset,
                clean_dataset.sizes,
                tag_dataset,
                tag_dataset.sizes,
                idx_dataset,
                idx_dataset.sizes,
                tokenizer,
                tagger,
            )
        else:
            dataset = GramercoDataset(
                noise_dataset,
                noise_dataset.sizes,
                clean_dataset,
                clean_dataset.sizes,
                tag_dataset,
                tag_dataset.sizes,
                tokenizer,
                tagger,
            )
    if return_clean:
        dataset.clean()
    logging.debug("Finished building")
    return dataset


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

    tokenizer = FlaubertTokenizer.from_pretrained(
        "flaubert/flaubert_base_cased")
    tagger = TagEncoder(
        path_to_lex="/nfs/RESEARCH/bouthors/projects/gramerco/resources/Lexique383.tsv",
        path_to_app="/nfs/RESEARCH/bouthors/projects/gramerco/resources/AFP/AFP-lex/lexique.app",
    )

    dataset = make_dataset_from_prefix(
        "/nfs/RESEARCH/bouthors/projects/gramerco/resources/AFP/AFP-bin-3/AFP.dev",
        tagger,
        tokenizer,
        word_index=True,
    )

    seed = 0
    max_tokens = 1000
    max_sentences = 50
    max_positions = 512
    min_positions = 5

    with data_utils.numpy_seed(seed):
        indices = dataset.ordered_indices()

    indices, _ = dataset.filter_indices_by_size(indices, max_positions, min_positions)

    batch_sampler = dataset.batch_by_size(
        indices,
        max_tokens=max_tokens,
        max_sentences=max_sentences,
        required_batch_size_multiple=1,
    )

    dataiter = iterators.EpochBatchIterator(
        dataset=dataset,
        collate_fn=dataset.collater,
        batch_sampler=batch_sampler,
        seed=seed,
        num_shards=1,
        shard_id=0,
        num_workers=1,
        epoch=1,
        buffer_size=0,
        skip_remainder_batch=False,  #  CHANGE TO True IRL
        grouped_shuffling=True,
    )
    for i, d in enumerate(iter(dataiter.next_epoch_itr(shuffle=False))):
        # if i == 2:
        logging.info('-' * 152)
        logging.info(int(d["noise_data"]["attention_mask"][0].sum().item()))
        logging.info(int(d["word_index"]["attention_mask"][0].sum().item()))
        logging.info(int(d["tag_data"]["attention_mask"][0].sum().item()))
        logging.info(d["word_index"]["input_ids"][0].max().item() - 1)
        # break
