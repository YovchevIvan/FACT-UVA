import sys, os

sys.path.append('../')

from main import WordEmbedding

from numpy.linalg import norm
from numpy import unravel_index, argmax, isnan, delete, arange
import random as rand

import argparse
import json

from main import to_utf8

# CONSTANTS
BOLD = '\033[1m' # add this string to start printing in bold
END = '\033[0m' # add this string to start printing normally
BLUE = '\033[94m' # add this string to start printing in blue
GREEN = '\033[92m' # add this string to start printing in green

# Auxiliary classes

class IOController:
	# use with 'with' statement to suppress all prints inside

	def __enter__(self):
		# redirect standard output to empty file
		sys.stdout = open(os.devnull, 'w')

	def __exit__(self, type, value, traceback):
		# reset standard output
		sys.stdout = sys.__stdout__

# Import needed libraries
with IOController():
	import nltk
	nltk.download('wordnet') # download wordnet corpus to have list of nouns
	from nltk.corpus import wordnet as wn

# MAIN CLASS

class AnalogyGenerator(WordEmbedding):

	def fetch_alanogy(self, x_word, y_word, z_word=None):

		'''
		Given a word pair x, y, randomly generate analogies x:y=z:w
		'''

		# Generate dictionary of nouns (only once)
		if z_word is None and not hasattr(self, 'nouns'): # if the field self.nouns does not exist already
			# get a list of nouns (from nltk's	wordnet)
			self.nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}

		# Find a third noun (if not given) with which we can do the analogy
		# ensure the word is different from the other and a noun
		if z_word is None:
			while  z_word not in self.nouns or z_word==x_word or z_word==y_word:
				z_word = rand.choice(self.words)

		# if analogy cannot be completed, return empty 
		try:
			# now that z has been selected, find w
			return self.complete_analogy(x_word, y_word, z_word)
		except:
			return None, None, None

	def complete_analogy(self, x_word, y_word, z_word):

		'''
		Given x, y, and z finds w such that x is to y like z is to w
		'''

		# same problem as solving x:y=z:w for all (z,w) with a restricted vocabulary for z (being only the given z_word)
		return self.find_similar(x_word, y_word, self.vecs[self.index[z_word], :][None], [z_word])

	def find_similar(self, x_word, y_word, z_vals=None, z_words=None):

		'''
		Given a word pair (x, y) finds word pair(s) (z, w) such that it holds that "x is to y as z is to w"
		'''

		# if a vocabulary for z is not specified, use all embeddings
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
		best_norm = None # Norm of the vector distance for the best pair
		best_pair = None, None # Best pair so far

		z_idx = 0
		# loop through z's vocabulary
		for z in z_vals:
			analogies = z - self.vecs # Possible analogy vectors

			norms = norm(analogies, axis=1) # Magnitudes
			analogies_dir = analogies/norms[:, None] # Directions

			dist = analogies_dir.dot(analogy_dir) # Compute cosine distances

			dist[isnan(dist)] = -1 # Eliminate nan values (for normalized z-z)
			dist[norms>1] = -1 # Exclude pairs that are too distant

			dist_idx = arange(dist.size) # Store indices in vocabulary of embeddings (in case they get discarded)

			# as long as there are embeddinds left
			while dist_idx.size > 1:
				w_idx = argmax(dist) # Get index of best distance for given z
				temp_val = dist[w_idx] # Get value of best distance for given z
				temp_norm = norms[dist_idx[w_idx]] # Norm of the vector for the best pair, given z

				w_word = self.words[dist_idx[w_idx]] # Get words relative for index (index taken from dist_idx, thus original vocabulary)
				if w_word==x_word or w_word==y_word or w_word==z_words[z_idx]: # if the word is not accettable
					dist = delete(dist, w_idx) # remove its cosine distance
					dist_idx = delete(dist_idx, w_idx) # remove its index
				else:
					break

			# Store best so far (cosine distance is used as main discriminator, for equal cosine distances a lower norm is preferred)
			if best_dist is None or temp_val > best_dist or (temp_val == best_dist and temp_norm < best_norm):
				best_dist = temp_val # store best distance so far
				best_pair = (z_idx, dist_idx[w_idx]) # store indexes of respective z and w
				best_norm = temp_norm # store the magnitude of the best so far

			# Print progress
			if z_vals.shape[0]>1:
				progress = z_idx/(z_vals.shape[0]-1)*100
				print(" checked %.2f"%(progress), '%', " [%s]\t best solution: \"%s is to %s like %s is to %s\" (dist %.3f, norm %.3f) %s\r"%("▉"*int(progress) + int(100-progress)*" ", BOLD+BLUE+x_word+END, BOLD+BLUE+y_word+END, BOLD+GREEN+z_words[best_pair[0]]+END, BOLD+GREEN+self.words[best_pair[1]]+END, best_dist, best_norm, " "*10), end="")

			# Keep track of z's index without using enumarate
			z_idx += 1

		# Return word pair
		return z_words[best_pair[0]], self.words[best_pair[1]], best_dist


if __name__ == "__main__":

	# Parse input argumens (see help for definition)
	parser = argparse.ArgumentParser()
	parser.add_argument("--pair_seed", type = lambda x : x.split('-'), help="pair seed for analogy separated by \'-\'", default=None)
	parser.add_argument("--file_seed", type = str, default=None, help="json file containing list of paired words with which to generate a specified number of analogies")
	parser.add_argument("--n", type = int, default = None, help = "if file_seed is specified, this is taken as the number of analogies to generate")
	parser.add_argument("--z_file", type = str, default = None, help = "file with z words to use in completing analogy pair")
	parser.add_argument("--complete", type = lambda x : x.split('-'), help="input x-y-z, returns w such that x:y=z:w", default=None)
	parser.add_argument("--em_limit", type=int, default=50000, help="number of words to load")
	parser.add_argument("--i_em", default="../embeddings/GoogleNews-vectors-negative300.bin", help="The name of the embedding")
	parser.add_argument("--pairs_fname", default="pairs.txt", help="The name of the output pairs file")

	# Read arguments
	args = parser.parse_args()

	# Load embeddings
	with IOController():
		E = AnalogyGenerator(args.i_em, args.em_limit)

	if args.file_seed is not None: # If a file of pairs is given as seed (complte x:y=z:w for z and w)

		if args.n is not None:
			z_words = [None]*args.n
		else:
			assert args.z_file is not None, "Either specify a number of analogies of provide a file of z words"
			with open(args.z_file, 'r') as f:
				z_words = json.load(f)

		# Load pairs from file
		with open(args.file_seed, 'r') as f:
			pairs = json.load(f)

		app = int(args.n/len(pairs)) # Compute number of analogies per pair

		for x, y in pairs: # for each pair in the file
			for z in z_words: # for a specified number of times

				# generate analogy
				with IOController():
					z, w, _ = E.fetch_alanogy(x, y, z)

				# print if found analogy
				if z is not None and w is not None:
					print(dist)
					print("%s is to %s like %s is to %s"%(BOLD+BLUE+x+END, BOLD+BLUE+y+END, BOLD+GREEN+z+END, BOLD+GREEN+w+END))

	elif args.complete is not None: # if 3 words are given, solve x:y=z:w for w

		x, y, z = args.complete # get x, y, and z words (split by '-')

		# find w
		print("Completing %s is to %s like %s is to w, for any possible w ..."%(x, y, z))
		_, w, _ = E.complete_analogy(x, y, z)

		# print result
		print("\nFinal result: %s is to %s like %s is to %s"%( BOLD+BLUE+x+END, BOLD+BLUE+y+END, BOLD+BLUE+z+END, BOLD+GREEN+w+END))

	elif args.pair_seed is not None: # if a single pair is given as seed (complete x"y=z"w for z and w)

		x, y = args.pair_seed # read pair from arguments

		# if a specified number of analogies to generate is given
		if args.n is not None or args.z_file is not None:

			pairs = [] # store generated pairs

			if args.n is not None:
				z_words = [None]*args.n
			else:
				with open(args.z_file, 'r') as f:
					z_words = json.load(f)

			for z in z_words: # for n times

				# sample z and find w
				with IOController():
					z, w, dist = E.fetch_alanogy(x, y, z)

				# if succeded then print analogy
				if z is not None and w is not None and dist is not None:
					pairs.append((z, w, dist))
					print("%s is to %s like %s is to %s"%(BOLD+BLUE+x+END, BOLD+BLUE+y+END, BOLD+GREEN+z+END, BOLD+GREEN+w+END))

			# sort pairs by distance (descending)
			pairs.sort(key=lambda p: float(p[2]), reverse=True)
			print("\n%s\n" % BOLD + "Sorting..." + END)

			# show top 10 pairs (closest ones to the given seed pair)
			# print("%s" % BOLD + "Top 10:" + END)
			# for i in range(10):
			#	z, w, dist = pairs[i]
			#	print("%s is to %s like %s is to %s (%.4f)"%(BOLD+BLUE+x+END, BOLD+BLUE+y+END, BOLD+GREEN+z+END, BOLD+GREEN+w+END, dist))

			# write results in specified file
			with open(args.pairs_fname, 'w') as f:
				f.write("\n".join([p[0]+":"+p[1] for p in pairs]))
		else: # if n is not specified, loop through all possible (z,w) and fine best

			print("Completing %s is to %s like z is to w, for any possible (z,w) pair..."%(x, y))
			z, w, _ = E.find_similar(x, y)

			print("\nFinal result: %s is to %s like %s is to %s"%(BOLD+BLUE+x+END, BOLD+BLUE+y+END, BOLD+GREEN+z+END, BOLD+GREEN+w+END))
	else:
		print("Nothing to do") # either a pair, a file of pairs, or 3 words must be given
