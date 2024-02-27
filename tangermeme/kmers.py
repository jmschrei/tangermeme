# kmers.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import scipy
import numpy as np

from numba import njit, prange
import numba

key_type = numba.types.int64
value_type = numba.types.float64


def kmers(X, k, scores=None):
	"""Extract all k-mers found in a sequence, optionally weighted by a score.

	This function will count the number of k-mers found in each sequence and
	return a feature matrix. If `score` is provided, the counts will be weighted
	by the sum of the score across the k positions where the k-mer resides. If
	`score` is not provided, the count will just be 1 for each instance of the
	k-mer in the sequence.


	Parameters
	----------
	X: torch.Tensor, shape=(-1, len(alphabet), sequence_length)
		A one-hot encoded set of sequences.

	k: int
		The size of the k-mers to consider.
	

	Returns
	-------
	X_kmers: torch.Tensor, shape=(-1, n_kmers)
		A featurization where each row is an example from the original set of
		sequences and each column is a k-mer that could be in the sequence.
	"""

	n = X.shape[1]

	X = X.type(torch.int32)
	w = torch.arange(n).repeat(k, 1).T * n ** torch.arange(k)
	w = w[None, :, :].type(torch.int32).to(X.device)
	idxs = torch.nn.functional.conv1d(X, w).type(torch.int64)[:, 0]

	if scores is not None:
		scores = scores.unsqueeze(1).type(torch.float32)
		ws = torch.ones(1, 1, k, dtype=torch.float32)
		score_ = torch.nn.functional.conv1d(scores, ws)[:, 0]
	else:
		score_ = torch.ones(1, dtype=torch.float32).expand_as(idxs)

	X_kmers = torch.zeros((X.shape[0], n**k))
	X_kmers.scatter_add_(1, idxs, score_)
	return X_kmers


@njit(parallel=True)
def _fast_extract_gkmers(X, min_k, max_k, max_gap, max_len, max_entries):
	nx = X.shape[0]
	keys = np.zeros((nx, max_entries), dtype='int64')
	scores = np.zeros((nx, max_entries), dtype='float64')

	for xi in prange(nx):
		n = X.shape[1]
		gkmer_attrs = numba.typed.Dict.empty(key_type=key_type, 
			value_type=value_type)

		last_k_gkmers = []
		last_k_gkmers_attrs = []
		last_k_gkmers_hashes = []

		for i in range(n):
			base = int(X[xi, i, 1])
			attr = X[xi, i, 2]

			last_k_gkmers.append(np.array([i], dtype='int32'))
			last_k_gkmers_attrs.append(np.array([attr], dtype='float64'))
			last_k_gkmers_hashes.append(np.array([base+1], dtype='int64'))

		for k in range(2, max_k+1):
			for j in range(n):
				start_position = X[xi, j, 0]

				gkmers_ = []
				gkmer_attrs_ = []
				gkmer_hashes_ = []

				for i in range(j+1, n):
					position = X[xi, i, 0]
					base = int(X[xi, i, 1])
					attr = X[xi, i, 2]

					if (position - start_position) >= max_len:
						break

					for g in range(len(last_k_gkmers[j])):
						gkmer = last_k_gkmers[j][g]
						gkmer_attr = last_k_gkmers_attrs[j][g]
						gkmer_hash = last_k_gkmers_hashes[j][g]

						last_position = X[xi, gkmer, 0]
						if last_position >= position:
							break

						if (position - last_position) > max_gap:
							continue

						diff = int(position - last_position - 1)
						length = int(position - start_position)

						new_gkmer_hash = gkmer_hash + (base+1) * (5 ** length)
						new_gkmer_attr = gkmer_attr + attr
						
						gkmers_.append(i)
						gkmer_attrs_.append(new_gkmer_attr)
						gkmer_hashes_.append(new_gkmer_hash)

						if k >= min_k:
							gkmer_attrs[new_gkmer_hash] = gkmer_attrs.get(
								new_gkmer_hash, 0) + new_gkmer_attr / k

				if len(gkmers_) == 0:
					last_k_gkmers[j] = np.zeros(0, dtype='int32')
					last_k_gkmers_attrs[j] = np.zeros(0, dtype='float64')
					last_k_gkmers_hashes[j] = np.zeros(0, dtype='int64')
				else:
					last_k_gkmers[j] = np.array(gkmers_, dtype='int32')
					last_k_gkmers_attrs[j] = np.array(gkmer_attrs_, 
						dtype='float64')
					last_k_gkmers_hashes[j] = np.array(gkmer_hashes_, 
						dtype='int64')
		
		ny = len(gkmer_attrs)
		keys_ = np.empty(ny, dtype='int64')
		scores_ = np.empty(ny, dtype='float64')

		for i, key in enumerate(gkmer_attrs.keys()):
			keys_[i] = key
			scores_[i] = gkmer_attrs[key]

		idxs = np.argsort(-np.abs(scores_), kind='mergesort')[:max_entries]

		keys[xi] = keys_[idxs]
		scores[xi] = scores_[idxs]

	return keys, scores



def gapped_kmers(X, scores=None, min_k=4, max_k=8, max_gap=2, max_len=10, 
	max_gkmers=10, max_pos=None):
	"""Extract gapped k-mers from sequences and optionally scores.

	This function will extract the gapped k-mers from a set of sequences that
	are one-hot encoded and optionally scores for each position. Without scores,
	this function will return a sparse matrix where each row is an example,
	each column is a gapped k-mer, and the value is the count of that gapped
	k-mer in the data. With scores, the shape will be the same but the value 
	will be the sum of the scores across positions in the gapped k-mers.


	Parameters
	----------
	X: torch.Tensor, shape=(-1, len(alphabet), sequence_length)
		A one-hot encoded set of sequences.

	attr: torch.Tensor with shape=(-1, sequence_length) or None, optional
		A corresponding set of attribution values to use. If None, return counts
		instead of sum of attribution values. Default is None.

	min_k: int, optional
		The minimum number of non-gaps in the k-mer. Default is 4.

	max_k: int, optional
		The maximum number of non-gaps in the k-mer. Default is 8.

	max_gap: int, optional
		The maximum number of gaps in the k-mer. Default is 2.

	max_len: int, optional
		The maximum length of the k-mer, as in, the total number of gaps and
		non-gap characters. Default is 10.

	max_gkmers: int, optional
		The maximum number of gapped k-mers to return.

	top_n_gkmers: int, optional
		..


	Returns
	-------
	gkmers: scipy.sparse.csr_matrix
		A sparse matrix containing either counts or score sums for each kmer
		found.
	"""


	X_idxs = X.argmax(axis=1)
	if not scores:
		scores = torch.ones_like(X_idxs)

	if max_pos is None:
		max_pos = X.shape[-1]

	X_scores = numpy.zeros((X.shape[0], max_pos, 3))
	for i in range(X.shape[0]):
		if scores is not None:
			score_idxs = numpy.argsort(-scores[i])[:max_pos].sort().values
		else:
			score_idxs = numpy.arange(X.shape[-1])

		X_scores[i, :, 0] = score_idxs
		X_scores[i, :, 1] = X_idxs[i, score_idxs]
		X_scores[i, :, 2] = scores[i, score_idxs]

	gkmers, gkmer_scores = _fast_extract_gkmers(X_scores, min_k=min_k, max_k=max_k, 
		max_gap=max_gap, max_len=max_len, max_entries=max_gkmers)
	
	row_idxs = numpy.repeat(range(gkmers.shape[0]), gkmers.shape[1])
	csr_mat = scipy.sparse.csr_matrix((gkmer_scores.flatten(), 
		(row_idxs, gkmers.flatten())), shape=(len(gkmers), 5**max_len))

	return csr_mat