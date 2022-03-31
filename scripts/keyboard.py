#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
from collections import defaultdict
import unicodedata


class keyboard:
    def __init__(self, keys, offs, maxd=2.0):
        assert len(keys) == len(offs)
        self.key2pos = {}
        self.distance = defaultdict(dict)
        diacritic = {
            "a": "àáâä",
            "e": "èéêë",
            "i": "ìíîï",
            "o": "òóôö",
            "u": "ùúûü",
            "c": "ç",
            "A": "ÀÁÂÄ",
            "E": "ÈÉÊË",
            "I": "ÌÍÎÏ",
            "O": "ÒÓÔÖ",
            "U": "ÙÚÛÜ",
            "C": "Ç",
        }
        # norm_c = unicodedata.normalize('NFKD',c).encode('ascii','ignore').decode('ascii')

        for row in range(len(keys)):  ### row in keyboard
            k_row = keys[row]  # keys of this row
            o_row = offs[row]  # x-offset of this row
            for col, c in enumerate(list(k_row)):  ### column in keyboard
                x = col + o_row
                y = row
                self.key2pos[c] = [x, y]
                if c in diacritic:
                    for acc_c in list(diacritic[c]):
                        self.key2pos[acc_c] = [x, y]

        for c1 in self.key2pos:
            for c2 in self.key2pos:
                if c1 == c2:
                    continue
                a = abs(self.key2pos[c1][0] - self.key2pos[c2][0])
                b = abs(self.key2pos[c1][1] - self.key2pos[c2][1])
                d = math.sqrt(a ** 2 + b ** 2)
                if d < maxd:  ### consider if distance is lower than maxd
                    self.distance[c1][c2] = d

    def __contains__(
        self, c
    ):  ### implementation of the method used when invoking : entry in keyboard
        return c in self.key2pos

    def closest(self, c, k=20):
        dict_c = self.distance[c]
        list_c = sorted(dict_c.items(), key=lambda x: x[1])
        keys = [x[0] for x in list_c[:k]]
        vals = [x[1] for x in list_c[:k]]
        return keys, vals


if __name__ == "__main__":

    qwerty_lc = keyboard(
        ["`1234567890-=", "qwertyuiop[]\\", "asdfghjkl;'", "zxcvbnm,./"],
        [0, 1.5, 1.85, 2.15],
        2.0,
    )
    qwerty_uc = keyboard(
        ["~!@#$%^&*()_+", "QWERTYUIOP{}|", 'ASDFGHJKL:"', "ZXCVBNM<>?"],
        [0, 1.5, 1.85, 2.15],
        2.0,
    )
    while True:
        c = input("Type keyboard: ")
        if c in qwerty_lc:
            k, v = qwerty_lc.closest(c)
        elif c in qwerty_uc:
            k, v = qwerty_uc.closest(c)
        else:
            sys.stderr.write("input [{}] not found as key in keyboard\n".format(c))
            sys.exit()
        print(k, v)
