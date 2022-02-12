import numpy as np
import os
import logging
import sys
import argparse
from tqdm import tqdm

BUFFER_SIZE = 10 ** 9


def partition(
        length: int,
        dev_size: float,
        test_size: float,
        train_size: float):
    assert dev_size + test_size < 1.0

    idxs = np.arange(length)
    np.random.seed(0)
    np.random.shuffle(idxs)

    idx_1 = int(test_size * length)
    idx_2 = idx_1 + int(dev_size * length)

    chooser = np.zeros(length, dtype=np.int8)
    chooser[idxs[:idx_1]] = 2
    chooser[idxs[idx_1:idx_2]] = 1

    if train_size > 0:
        idx = int(train_size * length)

    return chooser


def write_partition(file, suffix, chooser):
    logging.info("handling " + file + suffix)
    f_train = open(file + ".train" + suffix, "w")
    f_dev = open(file + ".dev" + suffix, "w")
    f_test = open(file + ".test" + suffix, "w")

    with open(file + suffix, "r") as f:
        cpt = 0
        lines = f.readlines(BUFFER_SIZE)
        i = 0
        while lines:
            p_bar = tqdm(lines)
            p_bar.set_description(str(i))
            for line in p_bar:
                if chooser[cpt] == 0:
                    f_train.write(line)
                elif chooser[cpt] == 1:
                    f_dev.write(line)
                elif chooser[cpt] == 2:
                    f_test.write(line)
                cpt += 1
            i += 1
            lines = f.readlines(BUFFER_SIZE)

    f_train.close()
    f_dev.close()
    f_test.close()


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
        "-log",
        default="info",
        help="Logging level [debug, info, warning, critical, error] (info)",
    )

    parser.add_argument(
        "--word-index",
        action='store_true',
        help="use word index."
    )

    parser.add_argument(
        "-dev",
        default=0.01,
        type=float,
        help="dev data proportion"
    )
    parser.add_argument(
        "-test",
        default=0.01,
        type=float,
        help="test data proportion"
    )
    parser.add_argument(
        "-train",
        default=-1,
        type=float,
        help="train data proportion"
    )

    args = parser.parse_args()
    create_logger("stderr", args.log)

    logging.info("counting number of lines")
    f = open(args.file + ".tag.fr", "r")
    # length = len(f.readlines())
    length = sum(1 for line in f)
    f.close()
    chooser = partition(
        length,
        dev_size=args.dev,
        test_size=args.test,
        train_size=args.train
    )
    logging.info("partitionning over")

    write_partition(args.file, ".fr", chooser)
    write_partition(args.file, ".tag.fr", chooser)
    write_partition(args.file, ".noise.fr", chooser)
    write_partition(args.file, ".widx.fr", chooser)
