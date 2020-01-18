from main import WordEmbedding

from numpy.linalg import norm
from numpy import unravel_index, argmax, isnan, delete, arange
import random as rand

import argparse
import json

import sys, os

from main import to_utf8

# CONSTANTS
BOLD = '\033[1m'
END = '\033[0m'
BLUE = '\033[94m'
GREEN = '\033[92m'

# Auxiliary classes

class IOController:

	def __enter__(self):
		sys.stdout = open(os.devnull, 'w')

	def __exit__(self, type, value, traceback):
		sys.stdout = sys.__stdout__

# Import needed libraries
with IOController():
	import nltk
	nltk.download('wordnet')
	from nltk.corpus import wordnet as wn

# MAIN CLASS

class AnalogyGenerator(WordEmbedding):

	def fetch_alanogy(self, x_word, y_word):

		'''
		Given a word pair x, y, randomly generate analogies x:y=z:w
		'''

		# Generate dictionary of nouns (only once)
		if not hasattr(self, 'nouns'):
			self.nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}

		# Find a third noun with which we can do the analogy
		z_word = None
		while z_word is None or z_word not in self.nouns or z_word==x_word or z_word==y_word:
			z_word = rand.choice(self.words)

		try:
			return self.complete_analogy(x_word, y_word, z_word)
		except:
			return None, None, None

	def complete_analogy(self, x_word, y_word, z_word):

		'''
		Given x, y, and z finds w such that x is to y like z is to w
		'''

		return self.find_similar(x_word, y_word, self.vecs[self.index[z_word], :][None], [z_word])

	def find_similar(self, x_word, y_word, z_vals=None, z_words=None):

		'''
		Given a word pair (x, y) finds word pair(s) (z, w) such that it holds that "x is to y as z is to w"
		'''

		if z_vals is None:
			z_vals = self.vecs
			z_words = self.words

		# Convert words to vectors
		x, y = self.v(x_word), self.v(y_word)

		# Get the direction of the analogy (what is the vector defining the analogy's transformation)
		analogy = x - y # Vector
		analogy_dir = analogy/norm(analogy) # Direction

		# Find a pair (z,w) such that z - w is most similar to the analogy's direction
		# Need to do iteratively to avoid memory errors

		best_dist = None # Best cosine distance so far
		best_pair = 0, 0 # Best pair so far

		z_idx = 0
		for z in z_vals:
			analogies = z - self.vecs # Possible analogy vectors

			norms = norm(analogies, axis=1)
			analogies_dir = analogies/norms[:, None] # Directions

			dist = analogies_dir.dot(analogy_dir) # Compute cosine distances

			dist[isnan(dist)] = -1 # Eliminate nan values (for normalized z-z)
			dist[norms>1] = -1 # Exclude pairs that are too distant

			dist_idx = arange(dist.size)

			while dist_idx.size > 1:
				w_idx = argmax(dist) # Get index of best distance for given z
				temp_val = dist[w_idx] # Get value of best distance for given z

				w_word = self.words[dist_idx[w_idx]]
				if w_word==x_word or w_word==y_word or w_word==z_words[z_idx]:
					dist = delete(dist, w_idx)
					dist_idx = delete(dist_idx, w_idx)
				else:
					break

			# Store best so far
			if best_dist is None or temp_val >= best_dist:
				best_dist = temp_val
				best_pair = (z_idx, dist_idx[w_idx])

			# Print progress
			if z_vals.shape[0]>1:
				progress = z_idx/(z_vals.shape[0]-1)*100
				print(" checked %.2f"%(progress), '%', " [%s]\t best solution: \"%s is to %s like %s is to %s\" (dist %.3f) %s\r"%("▉"*int(progress) + int(100-progress)*" ", BOLD+BLUE+x_word+END, BOLD+BLUE+y_word+END, BOLD+GREEN+z_words[best_pair[0]]+END, BOLD+GREEN+self.words[best_pair[1]]+END, best_dist, " "*10), end="")

			# Keep track of z's index without using enumarate
			z_idx += 1

		# Return word pair
		return z_words[best_pair[0]], self.words[best_pair[1]], best_dist


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--pair_seed", type = lambda x : x.split('-'), help="pair seed for analogy separated by \'-\'", default=None)
	parser.add_argument("--file_seed", type = str, default=None, help="json file containing list of paired words with which to generate a specified number of analogies")
	parser.add_argument("--n", type = int, default = None, help = "if file_seed is specified, this is taken as the number of analogies to generate")
	parser.add_argument("--complete", type = lambda x : x.split('-'), help="input x-y-z, returns w such that x:y=z:w", default=None)
	parser.add_argument("--em_limit", type=int, default=50000, help="number of words to load")
	parser.add_argument("--i_em", default="../embeddings/GoogleNews-vectors-negative300.bin", help="The name of the embedding")
	parser.add_argument("--pairs_fname", default="pairs.txt", help="The name of the output pairs file")

	args = parser.parse_args()

	# Load embeddings
	with IOController():
		E = AnalogyGenerator(args.i_em, args.em_limit)

	if args.file_seed is not None:

		assert args.n is not None, "Unspecified number of analogies to generate"

		# Load pairs
		with open(args.file_seed, 'r') as f:
			pairs = json.load(f)

		app = int(args.n/len(pairs)) # Compute number of analogies per pair

		for x, y in pairs:
			for i in range(app):

				with IOController():
					z, w, _ = E.fetch_alanogy(x, y)

				if z is not None and w is not None:
					print(dist)
					print("%s is to %s like %s is to %s"%(BOLD+BLUE+x+END, BOLD+BLUE+y+END, BOLD+GREEN+z+END, BOLD+GREEN+w+END))

	elif args.complete is not None:

		x, y, z = args.complete

		print("Completing %s is to %s like %s is to w, for any possible w ..."%(x, y, z))
		_, w, _ = E.complete_analogy(x, y, z)

		print("\nFinal result: %s is to %s like %s is to %s"%( BOLD+BLUE+x+END, BOLD+BLUE+y+END, BOLD+BLUE+z+END, BOLD+GREEN+w+END))

	elif args.pair_seed is not None:

		x, y = args.pair_seed

		if args.n is not None:

			pairs = []

			for i in range(args.n):

				with IOController():
					z, w, dist = E.fetch_alanogy(x, y)

				if z is not None and w is not None and dist is not None:
					pairs.append((z, w, dist))
					print("%s is to %s like %s is to %s"%(BOLD+BLUE+x+END, BOLD+BLUE+y+END, BOLD+GREEN+z+END, BOLD+GREEN+w+END))

			pairs.sort(key=lambda p: float(p[2]), reverse=True)
			print("\n%s\n" % BOLD + "Sorting..." + END)

			print("%s" % BOLD + "Top 10:" + END)
			for i in range(10):
				z, w, dist = pairs[i]
				print("%s is to %s like %s is to %s (%.4f)"%(BOLD+BLUE+x+END, BOLD+BLUE+y+END, BOLD+GREEN+z+END, BOLD+GREEN+w+END, dist))

			with open(args.pairs_fname, 'w') as f:
				f.write("\n".join([p[0]+":"+p[1] for p in pairs]))
		else:

			print("Completing %s is to %s like z is to w, for any possible (z,w) pair..."%(x, y))
			z, w, _ = E.find_similar(x, y)

			print("\nFinal result: %s is to %s like %s is to %s"%(BOLD+BLUE+x+END, BOLD+BLUE+y+END, BOLD+GREEN+z+END, BOLD+GREEN+w+END))
	else:
		print("Nothing to do")
