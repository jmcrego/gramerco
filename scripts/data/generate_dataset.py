#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import random
import logging
from collections import defaultdict
from tqdm import tqdm
import os
import re

separ = "￨"
keep = "·"
space = "~"
BUFFER_SIZE = 1000000000


def decode(line, get_tags=True, word_index=False):
    line = line.rstrip('\n').rstrip()
    if word_index:
        line, widx = line.split("\t")
    tuples = line.split(" ")
    tuples = [t.split(separ) for t in tuples]
    text = " ".join([t[0] for t in tuples])
    text = re.sub(" '", "'", text)
    text = re.sub("' ", "'", text)
    if get_tags:
        tags = " ".join([separ.join(t[1:]) for t in tuples])
        if word_index:
            return text, tags, widx
        return text, tags
    if word_index:
        return text, widx
    return text


def create_dataset(file, target_file, word_index=False):
    logging.info("word index = " + str(word_index))
    path_clean = os.path.abspath(target_file + ".fr")
    path_noise = os.path.abspath(target_file + ".noise.fr")
    path_tag = os.path.abspath(target_file + ".tag.fr")
    if word_index:
        path_widx = os.path.abspath(target_file + ".widx.fr")

    file_clean = open(path_clean, "w")
    file_noise = open(path_noise, "w")
    file_tag = open(path_tag, "w")
    if word_index:
        file_widx = open(path_widx, "w")

    with open(file, "r") as f:
        first = True
        tags = None
        ref = None

        N = 4
        k = -1
        data = f.readlines(BUFFER_SIZE)
        written_once = False
        tag_choice = list()
        noise_choice = list()
        widx_choice = list()
        while data:
            logging.info("---")
            for line in data:
                if k == N - 1:
                    if len(tag_choice) == 4:
                        i = random.randint(0, N - 1)
                        if written_once:
                            file_clean.write("\n")
                            file_noise.write("\n")
                            file_tag.write("\n")
                            if word_index:
                                file_widx.write("\n")
                        file_clean.write(noise_choice[0])
                        file_noise.write(noise_choice[i])
                        file_tag.write(tag_choice[i])
                        if word_index:
                            file_widx.write(widx_choice[i])
                        tag_choice = list()
                        noise_choice = list()
                        widx_choice = list()
                        written_once = True
                    else:
                        sys.exit(8)
                    k = -1

                text, tags = decode(line)
                if word_index:
                    text, tags, widx = decode(line, word_index=True)
                tag_choice.append(tags)
                noise_choice.append(text)
                if word_index:
                    widx_choice.append(widx)
                k += 1

            data = f.readlines(BUFFER_SIZE)

    file_clean.close()
    file_noise.close()
    file_tag.close()
    if word_index:
        file_widx.close()


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
    parser.add_argument("file", help="Input file/s")
    ### optional
    parser.add_argument("-v", action="store_true")
    parser.add_argument(
        "-log",
        default="info",
        help="Logging level [debug, info, warning, critical, error] (info)",
    )
    parser.add_argument(
        "-to",
        help="core file name (with path) corresponding to target dataset save files",
    )
    parser.add_argument(
        "--word-index",
        action='store_true',
        help="core file name (with path) corresponding to target dataset save files",
    )
    args = parser.parse_args()
    create_logger("stderr", args.log)
    logging.info("generate dataset")

    create_dataset(args.file, args.to, word_index=args.word_index)
