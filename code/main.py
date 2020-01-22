from __future__ import print_function, division
import re
import sys
import os
import gensim.models
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import argparse
import json
import numpy as np
import scipy.sparse
from sklearn.decomposition import PCA
if sys.version_info[0] < 3:
    import io
    open = io.open
else:
    unicode = str
"""
    WordEmbedding Class definition
"""

class WordEmbedding:
    def __init__(self, fname, em_limit):

        # info
        print("*** Reading data from " + fname)

        # read txt by default
        binary = False

        # check file extension if .bin read binary file
        if fname[-3:] == "bin":
            binary = True

        #load model
        model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=binary, limit=em_limit)

        # model has been loaded
        assert (model is not None)

        self.words = [w for w in model.vocab if self.word_filter(w)]

        print("Number of words: ", len(self.words))

        self.vecs = np.array([model[w] for w in self.words], dtype='float32')
        self.reindex()

        norms = np.linalg.norm(self.vecs, axis=1)
        if max(norms)-min(norms) > 0.0001:
            self.normalize()

    def word_filter(self, word):
        if len(word) < 20 and word.islower() and not bool(re.search(r'\W|[0-9]', word)):
            return word

    def reindex(self):
        self.index = {w: i for i, w in enumerate(self.words)}
        self.n, _ = self.vecs.shape

        assert self.n == len(self.words) == len(self.index)

    def v(self, word):
        return self.vecs[self.index[word]]

    def normalize(self):
        self.vecs /= np.linalg.norm(self.vecs, axis=1)[:, np.newaxis]
        self.reindex()

    def save_w2v(self, filename, ext):
        with open(filename, 'wb') as fout:
            fout.write(to_utf8("%s %s\n" % self.vecs.shape))
            # store in sorted order: most frequent words at the top
            for i, word in enumerate(self.words):
                if ext == "bin":
                    fout.write(to_utf8(word) + b" " + self.vecs[i].tostring())
                elif ext == "txt":
                    fout.write(to_utf8("%s %s\n" % (word, " ".join([str(j) for j in self.vecs[i]]))))

"""
    Additional functions
"""

def doPCA(pairs, embedding, num_components = 10):
    matrix = []
    for a, b in pairs:
        center = (embedding.v(a) + embedding.v(b))/2
        matrix.extend([embedding.v(a) - center, embedding.v(b) - center])
    pca = PCA(n_components = num_components)
    pca.fit(np.array(matrix))
    return pca

def drop(u, v, s):
    return u - v * u.dot(v) * s

def to_utf8(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')

def debias(E, gender_specific_words, definitional, equalize):
    gender_direction = doPCA(definitional, E).components_[0]

    scaling = 1/gender_direction.dot(gender_direction)

    marks = np.zeros(len(E.words), dtype=bool)
    for w in gender_specific_words:
        if w in E.index:
            marks[E.index[w]] = True

    i = 0
    for w in E.words:
        if not marks[i]:
            E.vecs[i] = drop(E.vecs[i], gender_direction, scaling)
        i += 1
    E.normalize()

    lower = map(lambda x : (x[0].lower(), x[1].lower()), equalize)
    title = map(lambda x : (x[0].title(), x[1].title()), equalize)
    upper = map(lambda x : (x[0].upper(), x[1].upper()), equalize)

    for candidates in [lower, title, upper]:
        for (a, b) in candidates:
            if (a in E.index and b in E.index):
                y = drop((E.v(a) + E.v(b)) / 2, gender_direction, scaling)
                z = np.sqrt(1 - np.linalg.norm(y)**2)
                if (E.v(a) - E.v(b)).dot(gender_direction) < 0:
                    z = -z
                E.vecs[E.index[a]] = z * gender_direction + y
                E.vecs[E.index[b]] = -z * gender_direction + y
    E.normalize()

def main(args):
    with open(args.def_fn, "r") as f:
        defs = json.load(f)

    with open(args.eq_fn, "r") as f:
        equalize_pairs = json.load(f)

    with open(args.g_words_fn, "r") as f:
        gender_specific_words = json.load(f)

    E = WordEmbedding(args.i_em, args.em_limit)
    print("Saving biased vectors to file...")
    E.save_w2v(args.bias_o_em, args.o_ext)

    print("Debiasing...")
    debias(E, gender_specific_words, defs, equalize_pairs)

    print("Saving to file...")
    E.save_w2v(args.debias_o_em, args.o_ext)

    print("\n\nDone!\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--em_limit", type=int, default=50000, help="number of words to load")
    parser.add_argument("--i_em", default="../embeddings/GoogleNews-vectors-negative300.bin", help="The name of the embedding")
    parser.add_argument("--def_fn", help="JSON of definitional pairs", default="../data/definitional_pairs.json")
    parser.add_argument("--g_words_fn", help="File containing words not to neutralize (one per line)", default="../data/gender_specific_full.json")
    parser.add_argument("--eq_fn", help="JSON with equalizing pairs", default="../data/equalize_pairs.json")
    parser.add_argument("--debias_o_em", help="Output debiased embeddings file", default="../embeddings/debiased.bin")
    parser.add_argument("--bias_o_em", help="Output bieased embeddings file", default="../embeddings/biased.bin")
    parser.add_argument("--o_ext", help="Extension of output file [txt, bin]", default="bin")

    args = parser.parse_args()
    print(args)

    main(args)

