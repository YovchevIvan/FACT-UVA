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
import warnings
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

        # filter words as specified in paper and store in list
        self.words = [w for w in model.vocab if self.word_filter(w)]

        # print number of words after filtering
        print("Number of words: ", len(self.words))

        # extract word vectors and store in list
        self.vecs = np.array([model[w] for w in self.words], dtype='float32')

        # word to index look up
        self.reindex()

        # compute all norms along axis 1
        norms = np.linalg.norm(self.vecs, axis=1)

        # if diff between max norm and min norm
        # is over a threshhold normalize all vectors
        if max(norms)-min(norms) > 0.0001:
            self.normalize()

    def word_filter(self, word):
        # paper uses only words wich are less than 20 chars
        # and which do not contain non word characters or numbers
        if len(word) < 20 and word.islower() and not bool(re.search(r'\W|[0-9]', word)):
            return word

    def normalize(self):
        # normalize
        self.vecs /= np.linalg.norm(self.vecs, axis=1)[:, np.newaxis]

        # reindex
        self.reindex()

    def reindex(self):
        # create dictionary from word to its index
        self.index = {w: i for i, w in enumerate(self.words)}

        # get vec shape
        self.n, _ = self.vecs.shape

        # make sure all dims line up
        assert self.n == len(self.words) == len(self.index)

    def v(self, word):
        # return word vector based on word
        return self.vecs[self.index[word]]

    def diff(self, w_1, w_2):

    	# vector difference between two words
    	v_diff = self.v(w_1) - self.v(w_2)

    	# return normalized
    	return v_diff/np.linalg.norm(v_diff)

    def save_w2v(self, filename, ext):
        # open file to write to
        with open(filename, 'wb') as fout:

            # write out number of tokens and vector size as per
            # word2vec specs
            fout.write(to_utf8("%s %s\n" % self.vecs.shape))

            # store in sorted order: most frequent words at the top
            for i, word in enumerate(self.words):

                # write to binary (less memory not human readable)
                if ext == "bin":
                    fout.write(to_utf8(word) + b" " + self.vecs[i].tostring())
                # write to text (more memory human readable)
                elif ext == "txt":
                    fout.write(to_utf8("%s %s\n" % (word, " ".join([str(j) for j in self.vecs[i]]))))

"""
    Additional functions
"""

def doPCA(pairs, embedding, num_components = 10):
    matrix = []

    # for each pair
    for a, b in pairs:

        # get center
        center = (embedding.v(a) + embedding.v(b))/2

        # add two vecs to matrix and shift them by center
        matrix.extend([embedding.v(a) - center, embedding.v(b) - center])

    # create PCA object
    pca = PCA(n_components = num_components)

    # fit data
    pca.fit(np.array(matrix))

    return pca

def drop(u, v, s):
    # vector minus scaled dot product v*v
    return u - v * u.dot(v) * s

def to_utf8(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')

def debias(E, gender_specific_words, definitional, equalize, num_components):

    # get gender axis
    gender_direction = doPCA(definitional, E, num_components).components_[0]

    # get param
    scaling = 1/gender_direction.dot(gender_direction)

    # inint mask
    marks = np.zeros(len(E.words), dtype=bool)

    # for each gender specific word
    for w in gender_specific_words:
        # if word is in E
        if w in E.index:
            # mark word to skip
            marks[E.index[w]] = True

    i = 0
    for w in E.words:
        # for all words not market to be skipped
        if not marks[i]:
            # shift each vector
            E.vecs[i] = drop(E.vecs[i], gender_direction, scaling)
        i += 1
    # normalize
    E.normalize()

    # create maps from lower, title, upper to equalize pairs
    lower = map(lambda x : (x[0].lower(), x[1].lower()), equalize)
    title = map(lambda x : (x[0].title(), x[1].title()), equalize)
    upper = map(lambda x : (x[0].upper(), x[1].upper()), equalize)

    # for each candidate
    for candidates in [lower, title, upper]:
        # for each pair in candidate
        for (a, b) in candidates:
            # if both a and b are in the index
            if (a in E.index and b in E.index):
                # get y ais shift
                y = drop((E.v(a) + E.v(b)) / 2, gender_direction, scaling)

                # get z shift
                z = np.sqrt(1 - np.linalg.norm(y)**2)

                # differnce between a and b dot product with the gender vector
                # is negative then flip the z shift
                if (E.v(a) - E.v(b)).dot(gender_direction) < 0:
                    z = -z

                # debiasing shift to both a and b
                E.vecs[E.index[a]] = z * gender_direction + y
                E.vecs[E.index[b]] = -z * gender_direction + y
    E.normalize()

def main(args):

    # read definitional pairs
    with open(args.def_fn, "r") as f:
        defs = json.load(f)

    # read equalizing pairs
    with open(args.eq_fn, "r") as f:
        equalize_pairs = json.load(f)

    # read gender specific words
    with open(args.g_words_fn, "r") as f:
        gender_specific_words = json.load(f)

    # create word embedding
    E = WordEmbedding(args.i_em, args.em_limit)

    # dump vectors prior to debiasing
    print("Saving biased vectors to file...")
    E.save_w2v(args.bias_o_em, args.o_ext)

    # debias
    print("Debiasing...")
    debias(E, gender_specific_words, defs, equalize_pairs, args.n_comp)

    # dump debiased vectors to file
    print("Saving debiased vectors to file...")
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
    parser.add_argument("--n_comp", type=int, help="number of components for PCA", default=10)

    args = parser.parse_args()

    # get rid of annoying warning
    warnings.filterwarnings("ignore")

    main(args)

