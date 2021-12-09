from transformers import FlaubertTokenizer, FlaubertModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tqdm import tqdm
from data.gramerco_dataset import GramercoDataset
from model_gec.gec_bert import GecBertModel
from tag_encoder import TagEncoder
from tokenizer import WordTokenizer
import logging
import matplotlib.pyplot as plt
from noiser.Noise import Lexicon
import os
import sys
import re
from infer import apply_tags


def test(args):
    tokenizer = FlaubertTokenizer.from_pretrained(
        args.tokenizer
    )
    word_tokenizer = WordTokenizer(FlaubertTokenizer)
    lexicon = Lexicon(args.lex)
    tagger = TagEncoder(
        path_to_lex=args.lex,
        path_to_app=args.app,
    )

    path_to_model = os.path.join(
        args.save_path,
        args.model_id,
        "model_best.pt",
    )
    model = model = GecBertModel(
        len(tagger),
        tagger=tagger,
        tokenizer=tokenizer,
        mid=args.model_id,
    )
    device = "cuda:" + str(args.gpu_id) \
        if args.gpu and torch.cuda.is_available() else "cpu"
    if os.path.isfile(path_to_model):
        map_loc = torch.device(device)
        state_dict = torch.load(path_to_model, map_location=map_loc)
        if isinstance(state_dict, GecBertModel):
            model = state_dict
        else:

            model.load_state_dict(
                state_dict["model_state_dict"]
            )

    else:
        logging.info("Model not found at: " + path_to_model)
        return
    model.eval()

    logging.info(torch.cuda.device_count())
    logging.info("device = " + device)
    model.to(device)

    with open(args.file_src, 'r') as f:
        txt_src = f.read(args.sample).split('\n')[:-1]

    with open(args.file_tag, 'r') as f:
        txt_tag = f.read(args.sample).split('\n')[:-1]

    txt_src = txt_src[:len(txt_tag)]
    txt_tag = txt_tag[:len(txt_src)]

    logging.info("contains {} sentences".format(len(txt_src)))

    non_zeros = [6, 11, 13, 26, 27, 40, 46]
    FP = 0
    TP = 0
    FN = 0
    TN = 0
    num_tags = 0
    num_keeps = 0

    for i in range(len(txt_src) // args.batch_size + 1):
        # if i > 0:
        #     print()
        batch_txt = txt_src[args.batch_size * i:
                            min(args.batch_size * (i + 1), len(txt_src))]
        batch_tag_ref = txt_tag[args.batch_size * i:
                                min(args.batch_size * (i + 1), len(txt_tag))]
        for j in range(args.num_iter):
            toks = tokenizer(
                batch_txt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=510
            ).to(device)
            # logging.info(toks["input_ids"].shape)
            with torch.no_grad():
                out = model(**toks)  # tag_out, attention_mask
                for k, t in enumerate(batch_txt):

                    batch_txt[k] = apply_tags(
                        t,
                        out["tag_out"][k].argmax(-1)[out["attention_mask"][k].bool()].cpu(),
                        word_tokenizer,
                        tagger,
                        lexicon
                    )

                    yy = out["tag_out"][k][out["attention_mask"][k].bool()]
                    yy = torch.softmax(yy, -1)
                    jj = yy.topk(3, dim=-1).indices.cpu()
                    ii = torch.arange(
                        jj.size(0)).unsqueeze(-1).expand(jj.shape)
                    # logging.info(jj)
                    # logging.info(yy[ii, jj])
                    # logging.info(batch_tag_ref[k].split(" "))
                    pred = jj[:, 0].bool()
                    ref = torch.tensor([tagger.tag_to_id(tag)
                                       for tag in batch_tag_ref[k].split(" ")]).bool()
                    if len(pred) != len(ref):
                        continue
                    TP += ((pred == ref) & ref).long().sum().item()
                    TN += ((pred == ref) & ~ref).long().sum().item()
                    FN += ((pred != ref) & ref).long().sum().item()
                    FP += ((pred != ref) & ~ref).long().sum().item()
                    num_tags += ref.long().sum().item()
                    num_keeps += (~ref).long().sum().item()

                    if False and i * args.batch_size + k in non_zeros:
                        logging.info("*" * 50)
                        logging.info("Noise sentence >>> " + str(t))
                        # logging.info(
                        #     " ".join(
                        #         tagger.id_to_tag(tag.item())
                        #         for tag in out["tag_out"][k].argmax(-1)[out["attention_mask"][k].bool()].cpu()
                        #     )
                        # )
                        for topk in range(1):
                            logging.info(
                                "infered tags >>> " +
                                " ".join(
                                    tagger.id_to_tag(tid.item())
                                    for tid in jj[:, topk]
                                )
                            )
                        logging.info("Corrected sentence >>> " + batch_txt[k])
                        logging.info(
                            "ref tags >>> " +
                            batch_tag_ref[k]
                        )
                        # logging.info("-" * 50)
    logging.info("TP = " + str(TP))
    logging.info("TN = " + str(TN))
    logging.info("FN = " + str(FN))
    logging.info("FP = " + str(FP))
    logging.info("non identified errors = " + str(FN / (FN + TP) * 100))
    logging.info("non identified keep = " + str(FP / (FP + TN) * 100))
    logging.info("#keep = " + str(num_keeps))
    logging.info("#tag = " + str(num_tags))
    logging.info("prop tag = " + str(num_tags / (num_tags + num_keeps) * 100))
    # print('\n'.join(batch_txt))


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

    parser.add_argument('--file-src', required=True, help="source file")
    parser.add_argument('--file-tag', required=True, help="tag file")

    # optional
    parser.add_argument('-v', action='store_true')
    parser.add_argument('--log', default="info", help='logging level')
    parser.add_argument(
        '--sample',
        type=int,
        default=0,
        help="Number of samples tested from files (faster testing if files too large)",
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help="GPU usage activation.",
    )
    parser.add_argument(
        '--gpu-id',
        default=0,
        type=int,
        help="GPU id, generally 0 or 1.",
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=20,
        help='batch size for eval',
    )
    parser.add_argument(
        '--num-iter',
        type=int,
        default=1,
        help='num iteration loops to edit',
    )
    parser.add_argument(
        '--save-path',
        required=True,
        help='model save directory'
    )
    parser.add_argument(
        '--model-id',
        required=True,
        help="Model id (folder name)",
    )
    parser.add_argument(
        '--lex',
        required=True,
        help='path to lexicon table.',
    )
    parser.add_argument(
        '--app',
        required=True,
        help="Path to appendable data.",
    )
    parser.add_argument(
        '--tokenizer',
        default="flaubert/flaubert_base_cased",
        help='model save directory',
    )

    args = parser.parse_args()

    create_logger("stderr", args.log)

    test(args)