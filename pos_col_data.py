import re

import numpy as np

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")


def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()


class IndexEntry:

    def __init__(self, settings, sentence):
        self._id = sentence.id

        self.targets = sentence.make_matrix(settings.target_label)


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
                matrix[m - 1] = target_label[t.upos]
        except KeyError:
            pass
        return matrix


class Token:
    def __init__(self, id, form, lemma, upos, xpos, feats,
                 head, deprel, deps, misc, scope=None):
        self.id = int(id)
        self.form = form
        self.norm = normalize(form)
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats
        self.deprel = deprel

    def __repr__(self):
        strlist = [str(self.id), self.form, self.lemma, self.upos, self.xpos,
                   self.feats, self.deprel]

        return "\t".join(strlist)


def read_col_data(fname):
    tokens = []
    tokens_full = {}
    sid = -1
    text = ""
    with open(fname, encoding='utf-8') as fhandle:
        for line in fhandle:
            if line.startswith("# sent_id"):
                sid = line.split("=")[1].strip()
            elif line.startswith("# text"):
                text = line.split("= ")[1].strip()
            elif line.startswith("#sid"):
                sid = line.split()[1].strip()
            elif line.startswith("#"):
                continue
            elif line == "\n":
                yield Sentence(sid, tokens, tokens_full, text)
                tokens = []
                tokens_full = {}
            else:
                try:
                    t = int(line.strip().split("\t")[0])
                except:
                    continue
                tokens.append(Token(*line.strip().split("\t")))
                tokens_full[len(tokens)] = tokens[-1]
