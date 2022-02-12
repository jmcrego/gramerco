import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from tqdm import tqdm
from data.gramerco_dataset import GramercoDataset, make_dataset_from_prefix
from fairseq.data import iterators, data_utils
import fairseq.utils as fairseq_utils
from model_gec.gec_bert import GecBertVocModel
from model_gec.criterions import DecisionLoss, CompensationLoss, CrossEntropyLoss, CETwoLoss
from transformers import FlaubertTokenizer
from tag_encoder import TagEncoder, TagEncoder2

import os
import sys
import time
from utils import EarlyStopping
import shutil
import argparse
import logging


def make_iterator(dataset, args, is_eval=False, max_sentences=None):
    # Generate batch iterator (with samples sorted by size)
    # from a GramercoDataset
    with data_utils.numpy_seed(args.seed):
        indices = dataset.ordered_indices()

    # filter sentences too long
    indices, _ = dataset.filter_indices_by_size(
        indices, args.max_positions, args.min_positions)

    if max_sentences:
        ii = np.arange(len(indices))
        if not is_eval:
            with data_utils.numpy_seed(args.seed):
                np.random.shuffle(ii)

        indices = indices[
            np.sort(ii[:max_sentences])
        ]

    # implicit batch size
    batch_sampler = dataset.batch_by_size(
        indices,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        required_batch_size_multiple=args.required_batch_size_multiple,
    )

    dataiter = iterators.EpochBatchIterator(
        dataset=dataset,
        collate_fn=dataset.collater,
        batch_sampler=batch_sampler,
        seed=args.seed,
        num_shards=1,
        shard_id=0,
        num_workers=args.num_workers,
        epoch=1,  # starting epoch count
        buffer_size=1,  # default batch preloading size (not 0)
        skip_remainder_batch=is_eval,
        grouped_shuffling=not is_eval,  # shuffle batches and spread it equitably to GPUs
    )
    return dataiter


def load_data(args, tagger, tokenizer):
    # Load train, dev, test datasets
    logging.debug("LOADING \t" + args.data_path + ".train")
    train_dataset = make_dataset_from_prefix(
        args.data_path + ".train",
        tagger,
        tokenizer,
        ignore_clean=args.ignore_clean,
        word_index=args.word_index,
    )
    train_iter = make_iterator(train_dataset, args)

    if args.valid:
        logging.debug("LOADING \t" + args.data_path + ".dev")
        valid_dataset = make_dataset_from_prefix(
            args.data_path + ".dev",
            tagger,
            tokenizer,
            ignore_clean=args.ignore_clean,
            word_index=args.word_index,
        )
        valid_iter = make_iterator(
            valid_dataset,
            args,
            is_eval=True,
            max_sentences=50000)
    else:
        valid_iter = None

    if args.test:
        logging.debug("LOADING \t" + args.data_path + ".test")
        test_dataset = make_dataset_from_prefix(
            args.data_path + ".test",
            tagger,
            tokenizer,
            ignore_clean=args.ignore_clean,
            word_index=args.word_index,
        )
        test_iter = make_iterator(
            test_dataset,
            args,
            is_eval=True,
            max_sentences=50000)
    else:
        test_iter = None

    return train_iter, valid_iter, test_iter


def train(args, device):

    # init tokenizer
    tokenizer = FlaubertTokenizer.from_pretrained(args.tokenizer)
    # init tag encoder
    tagger = TagEncoder2(
        path_to_lex=args.path_to_lex,
        path_to_voc=args.path_to_voc,
        new_version=args.word_index,
    )

    # init model
    model_id = args.model_id if args.model_id else str(int(time.time()))
    model_id = model_id + "-" + args.model_type
    model = GecBertVocModel(
        len(tagger),
        len(tagger.worder),
        tokenizer=tokenizer,
        tagger=tagger,
        mid=model_id,
        freeze_encoder=(args.freeze_encoder > 0),
        dropout=args.dropout,
        word_index=args.word_index,
    ).to(device)

    try:
        os.mkdir(os.path.join(args.save, model.id))
        if args.tensorboard:
            os.mkdir(os.path.join(args.save, "tensorboard", model.id))
    except BaseException:
        ...

    if args.continue_from and args.continue_from != "none" and os.path.isfile(
        os.path.join(
            args.save,
            model.id,
            "model_{}.pt".format(args.continue_from)
        )
    ):
        # load parameters from given checkpoint
        logging.info(
            "continue from " +
            os.path.join(
                args.save,
                model.id,
                "model_{}.pt".format(args.continue_from)
            )
        )
        model_info = torch.load(
            os.path.join(
                args.save,
                model.id,
                "model_{}.pt".format(args.continue_from)
            )
        )
        logging.info("starting from iteration {}".format(model_info["num_iter"]))
        model.load_state_dict(model_info["model_state_dict"])
    else:
        model_info = None

    # load data iterators
    train_iter, valid_iter, test_iter = load_data(args, tagger, tokenizer)

    # init criterion
    criterion = CETwoLoss(label_smoothing=args.label_smoothing)

    if model_info:
        num_iter = model_info["num_iter"]
        logging.info("starting from iteration {}".format(num_iter))
    else:
        num_iter = 0
        if args.pretrained:
            logging.info("load pretrained encoder from: " + args.pretrained)
            model.encoder.load_state_dict(torch.load(
                args.pretrained
            ))
    lr = args.learning_rate / float(args.grad_cumul_iter)
    if num_iter > args.freeze_encoder:
        model.freeze_encoder = False
    # init optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if args.tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(
            args.save, "tensorboard", model.id
        ))
    if args.valid:
        stopper = EarlyStopping(patience=args.early_stopping)
    else:
        stopper = None

    torch.save(
        {
            "num_iter": num_iter,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(
            args.save,
            model.id,
            "model_best.pt"
        )
    )

    optimizer.zero_grad()

    for epoch in range(args.n_epochs):
        # epoch training loop
        logging.debug("EPOCH " + str(epoch))
        train_bs = train_iter.next_epoch_itr(shuffle=True)
        if device == "cuda":
            torch.cuda.empty_cache()
        for batch in tqdm(train_bs):
            # batch sample
            if num_iter == args.freeze_encoder:
                # unfreeze encoder paramerters
                model.freeze_encoder = False
                optimizer = optim.Adam(model.parameters(), lr=lr)

            if device == "cuda":
                batch = fairseq_utils.move_to_cuda(batch)
            model.train()
            criterion.train()

            #  TRAIN STEP
            if args.word_index:
                batch["noise_data"]["word_index"] = batch["word_index"]["input_ids"]

            out = model(
                **batch["noise_data"]
            )
            # tag_out, voc_out, attention_mask

            sizes_out = out["attention_mask"].sum(-1)
            sizes_tgt = batch["tag_data"]["attention_mask"].sum(-1)
            # select only compatible sizes in out vs target
            coincide_mask = sizes_out == sizes_tgt

            # logging.debug("coincide = " + str(coincide_mask))

            tgt = batch["tag_data"]["input_ids"]
            # logging.debug("TAGs out = " + str(out.data.argmax(-1)[:20]))
            # logging.debug("TAGs tgt = " + str(tgt.data[:20]))

            # if not coincide_mask.all():
            #     logging.debug(tokenizer.convert_ids_to_tokens(batch["noise_data"][
            #         "input_ids"][~coincide_mask][0][batch["noise_data"]["attention_mask"][~coincide_mask][0].bool()]))

            # logging.debug(str(out))
            # logging.debug(str(tgt))
            loss = criterion(out, tgt, coincide_mask, batch, tagger,
                             mask_keep_prob=args.random_keep_mask)
            loss.backward()

            if num_iter % args.grad_cumul_iter == 0:
                optimizer.step()
                optimizer.zero_grad()
            del out, sizes_out, sizes_tgt, coincide_mask, tgt, batch

            if num_iter == 0:
                torch.cuda.empty_cache()

            if args.tensorboard:
                writer.add_scalar(
                    os.path.join("Loss/train"),
                    loss.item(),
                    num_iter,
                )
                del loss

            # VALID STEP
            if (num_iter) % args.valid_iter == 0:
                # valid/save model every args.valid_iter iterations
                if os.path.isfile(
                    os.path.join(
                        args.save,
                        model.id,
                        "model_{}.pt".format(
                            num_iter - 5 * args.valid_iter
                        ),
                    )
                ):
                    os.remove(
                        os.path.join(
                            args.save,
                            model.id,
                            "model_{}.pt".format(
                                num_iter - 5 * args.valid_iter
                            ),
                        ))
                # Regular model save (checkpoint)
                torch.save(
                    {
                        "num_iter": num_iter,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(
                        args.save,
                        model.id,
                        "model_{}.pt".format(num_iter),
                    )
                )
                shutil.copy2(
                    os.path.join(
                        args.save,
                        model.id,
                        "model_{}.pt".format(num_iter),
                    ),
                    os.path.join(
                        args.save,
                        model.id,
                        "model_last.pt",
                    ),
                )
                if args.valid:
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    model.eval()
                    criterion.eval()
                    logging.info("VALIDATION at iter " + str(num_iter))
                    with torch.no_grad():
                        val_losses = list()
                        TP = 0
                        FN = 0
                        FP = 0
                        TN = 0
                        tag_word_cpts = np.zeros((tagger._w_cpt, 2))
                        accs = np.zeros(len(tagger.id_error_type))
                        lens = np.zeros(len(tagger.id_error_type))
                        for valid_batch in tqdm(valid_iter.next_epoch_itr()):
                            if device == "cuda":
                                valid_batch = fairseq_utils.move_to_cuda(
                                    valid_batch
                                )

                            if args.word_index:
                                valid_batch["noise_data"]["word_index"] = valid_batch["word_index"]["input_ids"]

                            out = model(
                                **valid_batch["noise_data"]
                            )

                            sizes_out = out["attention_mask"].sum(-1)
                            sizes_tgt = valid_batch["tag_data"]["attention_mask"].sum(-1)
                            coincide_mask = sizes_out == sizes_tgt

                            tgt_mask = valid_batch["tag_data"]["attention_mask"][
                                coincide_mask]
                            tgt_ids = valid_batch["tag_data"]["input_ids"]
                            ref_ids = tgt_ids[coincide_mask][tgt_mask.bool()]

                            ref_tag = tagger.id_to_tag_id_vec(ref_ids)
                            ref_voc = tagger.id_to_word_id_vec(ref_ids)

                            pred_tag = out["tag_out"][coincide_mask][
                                    out["attention_mask"][coincide_mask].bool()
                            ].argmax(-1)
                            pred_voc = out["voc_out"][coincide_mask][
                                    out["attention_mask"][coincide_mask].bool()
                            ].argmax(-1)
                            pred_ids = tagger.tag_word_to_id_vec(pred_voc, pred_tag)

                            # Error detection scores
                            ref_dec = ref_tag.ne(0)
                            pred_dec = pred_tag.ne(0)
                            TP += ((pred_dec == ref_dec) &
                                   ref_dec).long().sum().item()
                            TN += (
                                (pred_dec == ref_dec) & ~ref_dec
                            ).long().sum().item()
                            FN += ((pred_dec != ref_dec) &
                                   ref_dec).long().sum().item()
                            FP += ((pred_dec != ref_dec) & ~
                                   ref_dec).long().sum().item()

                            # Error type scores
                            pred_types = pred_ids.clone().cpu().apply_(
                                tagger.get_tag_category
                            ).long()
                            ref_types = ref_ids.clone().cpu().apply_(
                                tagger.get_tag_category
                            ).long()
                            for err_id in range(len(tagger.id_error_type)):
                                pred_types_i = pred_types[ref_types == err_id]
                                accs[err_id] += (
                                    pred_types_i == err_id
                                ).long().sum().item()
                                lens[err_id] += len(pred_types_i)

                            # Word prediction scores
                            for word_tag_id in range(tagger._w_cpt):
                                tid = word_tag_id + tagger._curr_cpt - tagger._w_cpt
                                tag_word_cpts[word_tag_id, 0] += (
                                    pred_tag[ref_tag == tid] == tid
                                ).long().sum().item()
                                tag_word_cpts[word_tag_id, 1] += (
                                    ref_tag == tid
                                ).long().sum().item()

                            val_loss = criterion(
                                out, tgt_ids, coincide_mask, valid_batch, tagger,
                            ).item()
                            val_losses.append(val_loss)
                        del valid_batch
                        val_loss = sum(val_losses) / len(val_losses)
                        writer.add_scalar(
                            os.path.join("Loss/valid"),
                            val_loss,
                            num_iter,
                        )

                        # Store scores in tensorboard
                        recall = TP / (TP + FN)
                        precision = TP / (TP + FP)

                        F2_score = 5 * recall * precision / \
                            (4 * precision + recall)
                        for err_id in range(len(tagger.id_error_type)):
                            writer.add_scalar(
                                "ErrorType/{}".format(tagger.id_error_type[err_id]),
                                (accs[err_id] + 1) / (lens[err_id] + 1),
                                num_iter,
                            )

                        for i in range(tagger._w_cpt):
                            if tag_word_cpts[i, 1] > 0:
                                tid = word_tag_id + tagger._curr_cpt - tagger._w_cpt
                                writer.add_scalar(
                                    "WordError/{}".format(tagger._id_to_tag[tid][1:]),
                                    tag_word_cpts[i, 0] / tag_word_cpts[i, 1],
                                    num_iter,
                                )

                        writer.add_scalar(
                            os.path.join("Error/detection_rate"),
                            recall,
                            num_iter,
                        )
                        writer.add_scalar(
                            os.path.join("Error/precision"),
                            precision,
                            num_iter,
                        )
                        writer.add_scalar(
                            os.path.join("Error/F2_score"),
                            F2_score,
                            num_iter,
                        )
                        # update for early stopping
                        stopper(val_loss)

                # Update best model save if necessary
                if stopper and stopper.counter == 0:
                    logging.debug("NEW BEST MODEL")
                    shutil.copy2(
                        os.path.join(
                            args.save, model.id, "model_{}.pt".format(num_iter)
                        ),
                        os.path.join(
                            args.save, model.id, "model_best.pt",
                        ),
                    )
            # test for early stopping
            if stopper and stopper.early_stop:
                break
            num_iter += 1
        # test for early stopping
        if stopper and stopper.early_stop:
            break

    if args.test:
        model.eval()
        if device == "cuda":
            torch.cuda.empty_cache()
        with torch.no_grad():
            test_losses = list()
            for test_batch in test_iter.next_epoch_itr():
                if device == "cuda":
                    test_batch = fairseq_utils.move_to_cuda(
                        test_batch)

                if args.word_index:
                    test_batch["noise_data"]["word_index"] = test_batch["word_index"]["input_ids"]

                out = model(
                    **test_batch["noise_data"]
                )

                sizes_out = out["attention_mask"].sum(-1)
                sizes_tgt = test_batch["tag_data"]["attention_mask"].sum(-1)
                coincide_mask = sizes_out == sizes_tgt

                # out = out["tag_out"]

                tgt = test_batch["tag_data"]["input_ids"]

                test_loss = criterion(
                    out, tgt, coincide_mask, test_batch, tagger
                ).item()
                test_losses.append(test_loss)
            test_loss = sum(test_losses) / len(test_losses)
            logging.info("MODEL test loss: " + str(test_loss))

    torch.save(
        {
            "epoch": num_iter,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(args.save, model.id, "model_final.pt")
    )
    logging.info("TRAINING OVER")


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
    logging.getLogger().setLevel(numeric_level)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("data_path", help="Input bin data path")
    parser.add_argument("--save", required=True, help="save directory")
    parser.add_argument(
        "--continue-from",
        help="Id of the model",
    )
    parser.add_argument(
        "--freeze-encoder",
        type=int,
        default=0,
        help="Freeze encoder parameters.",
    )
    parser.add_argument(
        "--model-id",
        help="Model id (default = current timestamp in seconds).",
    )
    parser.add_argument(
        "--model-type",
        choices=["normal1", "decision2", "compensation1"],
        default="normal1",
        help="Threshold number of consecutive validation scores not improved \
        to consider training over.",
    )
    # optional
    parser.add_argument("-v", action="store_true")
    parser.add_argument("--log", default="info", help="logging level")
    parser.add_argument(
        "--seed", type=int, default=0, help="Randomization seed",
    )
    parser.add_argument(
        "--path-to-lex", "--lex", required=True, help="Path to Lexique383.tsv",
    )
    parser.add_argument(
        "--path-to-voc",
        "--voc",
        required=True,
        help="Path to vocabulary file.",
    )
    parser.add_argument(
        "--tokenizer",
        default="flaubert/flaubert_base_cased",
        help="Name of Huggingface tokenizer used.",
    )
    parser.add_argument(
        "--pretrained",
        default="",
        help="Path to pretrained encoder.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers used to fetch data.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="If True: write metrics in tensorboard files.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="If True: Uses GPU resources available.",
    )
    parser.add_argument(
        "-lang", "--language", default="fr", help="language of the data"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens per batch.",
    )
    parser.add_argument(
        "--required-batch-size-multiple",
        type=int,
        default=8,
        help="batch size will be a multiplier of this value.",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=100,
        help="Maximum number of sentences per batch.",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=510,
        help="Maximum size of a tokenized sentence. Sentences too long will be forgotten.",
    )
    parser.add_argument(
        "--min-positions",
        type=int,
        default=5,
        help="Minimum size of a tokenized sentence. Sentences too short will be forgotten.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=2,
        help="Number of epochs"
    )
    # parser.add_argument(
    #     "--decision-weight",
    #     type=float,
    #     default=0.1,
    #     help="Value of the tag loss weight w.r.t. to the decision loss one."
    # )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate value."
    )
    parser.add_argument(
        "-ls",
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.,
        help="Dropout probability in the linear layers"
    )
    parser.add_argument(
        "--grad-cumul-iter",
        type=int,
        default=1,
        help="Cumulatate grad, then take optimization step every --grad-cumul-iter iterations"
    )
    parser.add_argument(
        "--random-keep-mask",
        type=float,
        default=0.,
        help="Probability of masking a keep tag in the loss computation. Used to favor error detection."
    )
    parser.add_argument(
        "--valid",
        action="store_true",
        help="if True: regularly evaluate model on validation set.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="if True: evaluate model on test set after training.",
    )
    parser.add_argument(
        "--ignore-clean",
        action="store_true",
        help="Ignore clean in triplet (noise, tag, clean)",
    )
    parser.add_argument(
        "--valid-iter",
        type=int,
        default=512,
        help="Updates interval between two validation evaluations.",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=5,
        help="Threshold number of consecutive validation scores not improved \
        to consider training over.",
    )
    parser.add_argument(
        "--word-index",
        action="store_true",
        help="Use word index.",
    )

    args = parser.parse_args()
    create_logger(None, args.log)

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    train(args, device)
