from transformers import FlaubertTokenizer
import re
import sys


class WordTokenizer:
    """Composite of subword tokenizer to tokenize at word level.
    """

    def __init__(self, cls_tokenizer):
        self.subword_tokenizer = cls_tokenizer.from_pretrained(
            "flaubert/flaubert_base_cased"
        )

    def tokenize(self, text, n_pass=1, max_length=None):
        for i in range(n_pass):
            toks = self._tokenize_single(text, max_length=max_length)
        return toks

    def unite_tokens(self, tokens):
        text = " ".join(tokens)
        text = re.sub(" '", "'", text)
        return text

    def _tokenize_single(self, text, max_length=None):
        toks_id = self.subword_tokenizer(text)["input_ids"][1:-1]
        toks = [self.subword_tokenizer._convert_id_to_token(
            tid) for tid in toks_id]
        if max_length:
            toks = toks[:max_length]
        final_toks = list()
        current_word = ""

        for tok in toks:
            if tok[-4:] == "</w>":
                final_toks.append(current_word + tok[:-4])
                current_word = ""
            else:
                current_word += tok

        return final_toks


class WordTokenizer2():

    def __init__(self, modelname, do_lowercase=False):
        self.flaubert_tokenizer = FlaubertTokenizer.from_pretrained(
            modelname, do_lowercase=do_lowercase)

    def get_str(self, ids, remove_eow=True):
        t = ''.join(self.flaubert_tokenizer.convert_ids_to_tokens(ids))
        if remove_eow:
            t = t.replace('</w>', '')
        return t

    def get_ids(self, l, is_split_into_words=False, add_special_tokens=False):
        # l : string
        '''
        Returns:
        ids [int] : list of ints
        '''
        return self.flaubert_tokenizer(
            l,
            is_split_into_words=is_split_into_words,
            add_special_tokens=add_special_tokens)['input_ids']

    def is_further_tokenized(self, ids):
        subwords = self.flaubert_tokenizer.convert_ids_to_tokens(ids)
        for i in range(len(subwords) - 1):
            if subwords[i].endswith('</w>'):
                return True
        return False

    def get_words_ids2words_subwords(self, ids):
        # ids [int] : list of ints
        '''
        Returns:
        words     [string] : list of strings
        ids2words [int] : list of ints
        subwords  [string] : list of strings
        '''
        subwords = self.flaubert_tokenizer.convert_ids_to_tokens(ids)
        assert(len(ids) == len(subwords))
        ids2words = []
        words = ['']  # first word prepared
        for i in range(len(subwords)):
            if subwords[i] == '<s>':
                words[-1] = subwords[i]
                ids2words.append(len(words) - 1)
                if i < len(subwords) - 1:
                    words.append('')  # word finished prepare new

            elif subwords[i] == '</s>':
                words[-1] = subwords[i]
                ids2words.append(len(words) - 1)
                if i < len(subwords) - 1:
                    words.append('')  # word finished prepare new

            elif subwords[i].endswith('</w>'):
                subwords[i] = subwords[i][:-4]
                words[-1] += subwords[i]
                ids2words.append(len(words) - 1)
                if i < len(subwords) - 1:
                    words.append('')  # word finished prepare new
            else:
                words[-1] += subwords[i]
                ids2words.append(len(words) - 1)

        assert(len(subwords) == len(ids2words))
        if len(words[-1]) == 0:
            words.pop()
        return words, ids2words, subwords


if __name__ == "__main__":

    text = "Je l'appelle Maxime anticonstitutionnellement"

    t = WordTokenizer2('flaubert/flaubert_base_cased')

    ids = t.get_ids(text)
    print('ids', ids)
    words, ids2words, subwords = t.get_words_ids2words_subwords(ids)
    print('words', words)
    print('ids2words', ids2words)
    print('subwords', subwords)
