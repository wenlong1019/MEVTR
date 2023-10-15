import numpy as np


class Sentence:
    def __init__(self, id, tokens, tokens_full, text):
        self.id = id
        self.tokens = tokens
        self.tokens_full = tokens_full
        self.text = text

    def print_text(self):
        return self.text

    def __repr__(self):
        return "\n".join([f"# sent_id = {self.id}"] + [f"# text = {self.text}"] + [str(t) for k, t in
                                                                                   sorted(self.tokens_full.items(),
                                                                                          key=lambda x: x[0])] + [""])

    def __iter__(self):
        for token in self.tokens:
            yield token

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index]

    def __setitem__(self, index, value):
        self.tokens[index] = value

    def make_matrix(self, target_label):
        n = len(self.tokens)
        matrix = np.zeros(n)
        try:
            for t in self:
                m = t.id
                matrix[m - 1] = target_label[t.feats]
        except KeyError:
            pass
        return matrix


class Token:
    def __init__(self, id, form, feats):
        self.id = id
        self.form = form
        self.feats = feats


class IndexEntry:

    def __init__(self, settings, sentence):
        self._id = sentence.id
        self.targets = sentence.make_matrix(settings.target_label)


def read_col_data_ner(fname):
    tokens = []
    tokens_full = {}
    sid = 0
    id = 0
    text = ""
    with open(fname, encoding='utf-8') as fhandle:
        for line in fhandle:
            if line == "\n":
                if len(tokens) != 0:
                    sid = sid + 1
                    yield Sentence(sid, tokens, tokens_full, text)
                    tokens = []
                    tokens_full = {}
                    id = 0
            else:
                try:
                    id = id + 1
                    form, _, _, feats = line.strip().split(" ")
                    tokens.append(Token(id, form, feats))
                    tokens_full[len(tokens)] = tokens[-1]
                except TypeError:
                    print(line)
