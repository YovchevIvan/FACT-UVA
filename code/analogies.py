from main import WordEmbedding

from numpy.linalg import norm
from numpy import unravel_index, argmax, isnan

import argparse

# CONSTANTS
BOLD = '\033[1m'
END = '\033[0m'
BLUE = '\033[94m'
GREEN = '\033[92m'

# MAIN CLASS

class AnalogyGenerator(WordEmbedding):

	def fetch_similar(self, x_word, y_word):

		'''
		Given a word pair (x, y) finds word pair(s) (z, w) such that it holds that "x is to y as z is to w"
		'''

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
		for z in self.vecs:
			analogies = z - self.vecs # Possible analogy vectors

			analogies_dir = analogies/norm(analogies, axis=1)[:, None] # Directions
			analogies_dir[isnan(analogies_dir)] = -1 # Eliminate nan values (for normalized z-z)

			dist = analogies_dir.dot(analogy_dir) # Compute cosine distances

			w_idx = argmax(dist) # Get index of best distance for given z
			temp_val = dist[w_idx] # Get value of best distance for given z

			# Store best so far
			if best_dist is None or temp_val > best_dist:
				best_dist = temp_val
				best_pair = (z_idx, w_idx)

			# Print progress
			progress = z_idx/(self.vecs.shape[0]-1)*100
			print("checked %.2f"%(progress), '%', " [%s]\t best solution: \"%s is to %s like %s is to %s\" (dist %.3f) %s\r"%("▉"*int(progress) + int(100-progress)*" ", BOLD+BLUE+x_word+END, BOLD+BLUE+y_word+END, BOLD+GREEN+self.words[best_pair[0]]+END, BOLD+GREEN+self.words[best_pair[1]]+END, best_dist, " "*10), end="")

			# Keep track of z's index without using enumarate
			z_idx += 1

		# Return word pair
		return self.words[best_pair[0]], self.words[best_pair[1]]


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--pair_seed", type= lambda x : x.split('-'), help="pair seed for analogy separated by \'-\'")
	parser.add_argument("--em_limit", type=int, default=50000, help="number of words to load")
	parser.add_argument("--i_em", default="../embeddings/GoogleNews-vectors-negative300.bin", help="The name of the embedding")
	parser.add_argument("--bin", type = lambda x : x.lower()=="true" if x.lower()=="true" or x.lower()=="false" else None, default=True, help="Boolean, set to false if using txt file format")

	args = parser.parse_args()

	# Load embeddings
	E = AnalogyGenerator(args.i_em, args.bin, args.em_limit)

	x, y = args.pair_seed
	print("Completing %s is to %s like z is to w, for any possible (z,w) pair..."%(x, y))
	z, w = E.fetch_similar(x, y)
