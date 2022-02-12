import logging
import os
import shutil
import sys
from collections import Counter
from itertools import zip_longest
from multiprocessing import Pool

from fairseq import options, tasks, utils
from fairseq.binarizer import Binarizer
from fairseq.data import indexed_dataset
from fairseq.file_chunker_utils import find_offsets

pwd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(pwd))
sys.path.append(os.path.dirname(os.path.abspath(pwd)))
from tag_encoder import TagEncoder, TagEncoder2
from transformers import FlaubertTokenizer
from dictionary import Dictionary
import argparse

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.preprocess")


def binarize_func(
    filename, vocab, output_prefix, lang, offset, end, wid, append_eos=True
):
    logging.info(output_prefix + "    " + str(wid))
    ds = indexed_dataset.make_builder(
        output_prefix + "." + lang + "." + str(wid) + ".bin",
        impl="mmap",
        vocab_size=len(vocab),
    )

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(
        filename, vocab, consumer, append_eos=append_eos, offset=offset, end=end
    )
    ds.finalize(output_prefix + "." + lang + "." + str(wid) + ".idx")
    return res


def make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers):
    logging.info("processing ::::: " + input_prefix)
    logger.info("[{}] Dictionary: {} types".format(lang, len(vocab)))
    n_seq_tok = [0, 0]
    replaced = Counter()

    def merge_result(worker_result):
        replaced.update(worker_result["replaced"])
        n_seq_tok[0] += worker_result["nseq"]
        n_seq_tok[1] += worker_result["ntok"]

    input_file = "{}{}".format(input_prefix, ("." + lang) if lang is not None else "")
    offsets = find_offsets(input_file, num_workers)
    # logging.info(offsets)
    (first_chunk, *more_chunks) = zip(offsets, offsets[1:])
    pool = None
    if num_workers > 1:
        pool = Pool(processes=num_workers - 1)
        for worker_id, (start_offset, end_offset) in enumerate(more_chunks, start=1):
            # merge_result(binarize_func(input_file, vocab, output_prefix, lang, start_offset, end_offset, worker_id))
            pool.apply_async(
                binarize_func,
                (
                    input_file,
                    vocab,
                    output_prefix,
                    lang,
                    start_offset,
                    end_offset,
                    worker_id,
                ),
                callback=merge_result,
            )
        pool.close()
    logging.info(output_prefix + "." + lang + ".bin")
    ds = indexed_dataset.make_builder(
        output_prefix + "." + lang + ".bin", impl="mmap", vocab_size=len(vocab),
    )

    merge_result(
        Binarizer.binarize(
            input_file,
            vocab,
            lambda t: ds.add_item(t),
            offset=first_chunk[0],
            end=first_chunk[1],
        )
    )
    if num_workers > 1:
        pool.join()
        for worker_id in range(1, num_workers):
            temp_file_path = output_prefix + "." + lang + "." + str(worker_id)
            ds.merge_file_(temp_file_path)
            os.remove(indexed_dataset.data_file_path(temp_file_path))
            os.remove(indexed_dataset.index_file_path(temp_file_path))

    ds.finalize(output_prefix + "." + lang + ".idx")

    logger.info(
        "[{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
            lang,
            input_file,
            n_seq_tok[0],
            n_seq_tok[1],
            100 * sum(replaced.values()) / n_seq_tok[1],
            vocab.unk_word,
        )
    )

    return ds


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
    parser.add_argument("file", help="Input files core name")
    parser.add_argument(
        "-out", required=True, help="Output files directory and basename"
    )
    ### optional
    parser.add_argument(
        "-log",
        default="info",
        help="Logging level [debug, info, warning, critical, error] (info)",
    )
    parser.add_argument("-split", default="train", help="train, test or dev")
    parser.add_argument(
        "-lex", required=True, help="Path to the Lexique table",
    )
    parser.add_argument(
        "-app", required=True, help="Path to the appendable word list",
    )
    parser.add_argument(
        "--num-workers", default=1, type=int, help="Number of workers to parallelize",
    )
    parser.add_argument(
        "--word-index", action="store_true", help="Use word index.",
    )
    parser.add_argument(
        "--version", default=1, type=int, help="Version of the Tag Encoder",
    )
    args = parser.parse_args()

    create_logger("stderr", args.log)
    if args.version == 2:
        tagger = TagEncoder2(
            path_to_lex=args.lex,
            path_to_voc=args.app,
            new_version=args.word_index
        )
    else:
        tagger = TagEncoder(path_to_lex=args.lex, path_to_app=args.app,)
    tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_cased")
    voc_tag = Dictionary(tagger, encoder_type="tag_encoder", version=args.version)
    if args.word_index:
        voc_tok_widx = Dictionary(tokenizer, encoder_type="pretokenizer")
        voc_widx = Dictionary(tokenizer, encoder_type="word_index")
    else:
        voc_tok = Dictionary(tokenizer, encoder_type="tokenizer")

    if args.word_index:
        for suffix in [".tag", ".widx1", ".widx2"]:
            suffix_out = suffix
            if suffix == ".tag":
                vocab = voc_tag
            elif suffix == ".widx1":
                vocab = voc_tok_widx
                suffix = suffix[:-1]
                suffix_out = ".noise"
            elif suffix == ".widx2":
                vocab = voc_widx
                suffix = suffix[:-1]
                suffix_out = suffix

            make_binary_dataset(
                vocab,
                args.file + suffix,
                args.out + suffix_out,
                "fr",
                num_workers=args.num_workers,
            )
    else:
        for suffix in ["", ".tag", ".noise", ".widx"]:
            vocab = voc_tag if suffix == ".tag" else voc_tok

            make_binary_dataset(
                vocab,
                args.file + suffix,
                args.out + suffix,
                "fr",
                num_workers=args.num_workers,
            )
