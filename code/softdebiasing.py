import torch, torch.nn as nn
import numpy as np

import json

from main import WordEmbedding as WE, doPCA
from scipy.linalg import svd as SVD

class Det(torch.autograd.Function):
	"""
	Matrix determinant. Input should be a square matrix
	"""

	@staticmethod
	def forward(ctx, x):
		output = torch.potrf(x).diag().prod()**2
		output = torch.Tensor([output]).cuda() # remove .cuda() if you only use cpu
		ctx.save_for_backward(x, output)
		return output

	@staticmethod
	def backward(ctx, grad_output):
		x, output = ctx.saved_variables
		grad_input = None

		if ctx.needs_input_grad[0]:
			grad_input = grad_output * output * x.inverse().t()

		return grad_input

class SoftDebias():

	def __init__(self, W, N, B, ld = 0.5, lr=0.3):

		# Convert embeddings to numpy format
		W = np.array(W) # word embeddings
		N = np.array(N) # gender neutral words' embeddings

		# Get embeddings dim
		n, d = W.shape

		# Perform Singular Value Decomposition on embeddings
		# Extract U and S (and V) such that U*S*V^T = W
		U, s, _ = svd(W)

		# Create equation terms (formula 4 from appendix), as pytorch Tensors
		U = torch.Tensor(U)
		S = torch.Tensor(np.diag(s))

		I = torch.Tensor(np.eye(d)) # Identity matrix d x d

		N = torch.Tensor(N)
		B = torch.Tensor(B) # Gender subspace

		# Create tunable parameter (soft debias transform)
		self.T = nn.Parameter(torch.randn([d,d]))

		# Init values of tunable paramters using xavier uniform initialization
		nn.init.xavier_uniform_(self.T)

		# Init optimizer used to fit T
		self.optim = torch.optim.Adam([self.T], lr)

		# define determinant (differentiable) function
		det = Det.apply

		# Define function to be minimized
		self.error = lambda : det(S @ U @ (self.T.transpose() @ self.T - I) @ U.transpose() @ S) + ld*det(N @ (self.T.transpose() @ self.T) @ B)


	def fit(num_iter=10000):


		for i in range(num_iter):

			# Compute value to be minimized
			err = self.error()

			# Reset gradients
			self.optim.zero_grad()

			# Compute gradients
			err.backward()

			# Update T
			self.optim.step()

			print("step: %d, err: %.3f"%(i, err.detach().numpy()))



if __name__ == "__main__":

	# # read definitional pairs
	# with open(args.def_fn, "r") as f:
	# 	definitional = json.load(f)

	# # get gender neutral words
	# with open("neutral.json", 'r') as f:
	# 	neutral = json.load(f)

	# # create word embedding
	# E = WordEmbedding(args.i_em, args.em_limit)

	# # get gender axis
	# B = doPCA(definitional, E, args.num_components).components_

	# # get embeddings for gender neutral words
	# N = np.zeros(len(neutral), E.vecs.shape[1])
	# for i, n in enumerate(neutral):
	# 	N[i] = E.v(n)


	# # Create soft debiasing fitter object
	# soft = SoftDebias(E.vecs, N, B, args.ld, args.lr)

	# # Fit T
	# soft.fit(args.n_steps)

	# # Get T
	# T = soft.T.detach().numpy()
