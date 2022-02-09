from transformers import FlaubertTokenizer
import re


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
        text = " ".join(toks)
        text = re.sub(" '", "'", text)
        return text

    def _tokenize_single(self, text, max_length=None):
        toks_id = self.subword_tokenizer(text)["input_ids"][1:-1]
        toks = [self.subword_tokenizer._convert_id_to_token(tid) for tid in toks_id]
        if max_length:
            toks = toks[:max_length]
        final_toks = list()
        current_word = ""
        for tok in toks:
            if tok == "'</w>":
                current_word += "'"
            elif tok[-4:] == "</w>":
                final_toks.append(current_word + tok[:-4])
                current_word = ""
            else:
                current_word += tok

        return final_toks


if __name__ == "__main__":

    text = "Je l'appelle Maxime"

    print(text)
    t_word = WordTokenizer(FlaubertTokenizer)
    toks = t_word.tokenize(text)
    print(toks)
