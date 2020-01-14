from __future__ import print_function, division
import re
import sys
import gensim.models
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
	def __init__(self, fname):
		print("*** Reading data from " + fname)

		# IMPORTANT: limit argument to be removed after refactoring.
		# It is currently in there becuase it takes too long to load all the data
		model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True, limit=100000)
		self.words = sorted([w for w in model.vocab], key=lambda w: model.vocab[w].index)
		self.vecs = np.array([model[w] for w in self.words], dtype='float32')
		self.reindex()

		norms = np.linalg.norm(self.vecs, axis=1)
		if max(norms)-min(norms) > 0.0001:
			self.normalize()

	def reindex(self):
		self.index = {w: i for i, w in enumerate(self.words)}
		self.n, _ = self.vecs.shape

		assert self.n == len(self.words) == len(self.index)

	def v(self, word):
		return self.vecs[self.index[word]]

	def normalize(self):
		self.vecs /= np.linalg.norm(self.vecs, axis=1)[:, np.newaxis]
		self.reindex()

	def save_w2v(self, filename, binary=True):
		with open(filename, 'wb') as fout:
			fout.write(to_utf8("%s %s\n" % self.vecs.shape))
			# store in sorted order: most frequent words at the top
			for i, word in enumerate(self.words):
				fout.write(to_utf8(word) + b" " + self.vecs[i].tostring())

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
	# specific_set = set(gender_specific_words)

	scaling = 1/gender_direction.dot(gender_direction)

	marks = np.zeros(len(E.words), dtype=bool)
	for w in gender_specific_words:
		marks[E.index[w]] = True

	i = 0
	for w in E.words:
		if not marks[i]:
			E.vecs[i] = drop(E.vecs[i], gender_direction, scaling)
		i += 1
	E.normalize()

	# candidates = {x for e1, e2 in equalize for x in [(e1.lower(), e2.lower()),
	# 												 (e1.title(), e2.title()),
	# 												 (e1.upper(), e2.upper())]}

	lower = map(lambda x : (x[0].lower(), x[1].lower()), equalize)
	title = map(lambda x : (x[0].title(), x[1].title()), equalize)
	upper = map(lambda x : (x[0].upper(), x[1].upper()), equalize)

	for candidates in [lower, title, upper]:
		print(candidates)
		for (a, b) in candidates:
			if (a in E.index and b in E.index):
				y = drop((E.v(a) + E.v(b)) / 2, gender_direction, scaling)
				z = np.sqrt(1 - np.linalg.norm(y)**2)
				if (E.v(a) - E.v(b)).dot(gender_direction) < 0:
					z = -z
				E.vecs[E.index[a]] = z * gender_direction + y
				E.vecs[E.index[b]] = -z * gender_direction + y
	E.normalize()

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("embedding_filename", help="The name of the embedding")
	parser.add_argument("definitional_filename", help="JSON of definitional pairs")
	parser.add_argument("gendered_words_filename", help="File containing words not to neutralize (one per line)")
	parser.add_argument("equalize_filename", help="???.bin")
	parser.add_argument("debiased_filename", help="???.bin")

	args = parser.parse_args()
	print(args)

	with open(args.definitional_filename, "r") as f:
		defs = json.load(f)
	print("definitional", defs)

	with open(args.equalize_filename, "r") as f:
		equalize_pairs = json.load(f)

	with open(args.gendered_words_filename, "r") as f:
		gender_specific_words = json.load(f)
	print("gender specific", len(gender_specific_words), gender_specific_words[:10])

	E = WordEmbedding(args.embedding_filename)

	print("Debiasing...")
	debias(E, gender_specific_words, defs, equalize_pairs)

	print("Saving to file...")
	E.save_w2v(args.debiased_filename)

	print("\n\nDone!\n")

