# tomtom.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com> 

import time
import math
import numpy
import numba
import torch

from numba import njit
from numba import prange
from numpy import uint8, uint64

from .tomtom import _binned_median
from .tomtom import _pairwise_max
from .tomtom import _merge_rc_results

from .tomtom import _integer_distances_and_histogram
from .tomtom import _p_values
from .tomtom import tomtom

 
@njit
def _p_value_backgrounds(f, A, B, A_csum, nq, n_bins, t_max, offset):
	"""An internal function that calculates the backgrounds for p-values.

	This method takes in the histogram of integerized scores `f` and returns 
	the background probabilities of each overlap achieving a given score. 
	These scores are calculated for the complete overlap of the query and
	target, but also for all overhangs where only part of the query and the
	target are overlapping (on either end). Additionally, background
	probabilities are calculated for all spans across the query for when the
	target is smaller than the query and has to be scanned against it.
	"""

	n = n_bins*nq + nq*offset
	nqm1 = uint64(nq-1)

	# Clear A
	for i in range(nq):
		i = uint64(i)
		for j in range(n):
			j = uint64(j)
			A[0, i, j] = 0
			A[1, i, j] = 0
	
	for i in range(nq):
		c = offset * (nq - i - 1)
		i, c = uint64(i), uint64(c)
		im1, nqmi, nqmi1 = uint64(i-1), uint64(nq-i), uint64(nq-i-1)

		if i == 0:
			for k in range(1, n_bins+1):
				k = uint64(k)
				A[0, 0, k+c] = f[0, k]
				A[1, nqm1, k+c] = f[nqm1, k]
		else:
			for k in range(n_bins*i+1):
				k = uint64(k)
				a0 = A[0, im1, k+c+offset]
				a1 = A[1, nqmi, k+c+offset]

				if a0 > 0:
					for l in range(1, n_bins+1):
						l = uint64(l)
						A[0, i, l+k+c] += a0 * f[i, l]

				if a1 > 0:
					for l in range(1, n_bins+1):
						l = uint64(l)
						A[1, nqmi1, l+k+c] += a1 * f[nqmi1, l]
		
		for k in range(n):
			k, km1 = uint64(k), uint64(k-1)

			if k > n_bins*(i+1)+c:
				A_csum[0, i, k] = 1
				A_csum[1, nqmi1, k] = 1
			else:
				A_csum[0, i, k] = A[0, i, k]
				A_csum[1, nqmi1, k] = A[1, nqmi1, k]
				if k > 0:
					A_csum[0, i, k] += A_csum[0, i, km1]
					A_csum[1, nqmi1, k] += A_csum[1, nqmi1, km1]

	###

	B[0] = -1
	for i in range(1, nq):
		_pairwise_max(B[i-1], A[0, i-1], A_csum[0, i-1], B[i], n)
		_pairwise_max(B[i], A[1, nq-i], A_csum[1, nq-i], B[i], n)

	for i in range(nq, t_max+1):
		_pairwise_max(B[i-1], A[0, nq-1], A_csum[0, nq-1], B[i], n)

	# Again, `axis` is not implemented for cumsum
	for i in range(B.shape[0]):
		for j in range(1, n):
			B[i, j] += B[i, j-1]
		
		for j in range(n):
			B[i, j] = 1 - B[i, j]
			

@njit(parallel=True)
def _tomtom(Q, T, Q_lens, T_lens, Q_norm, T_norm, rr_inv, rr_counts, n_nearest, 
	n_score_bins, n_median_bins, n_cache, reverse_complement):
	"""An internal function implementing the TOMTOM algorithm.

	This internal function is necessary to handle the numba component of the
	implementation. Here, scratchboard memory is allocated for each thread and
	the main parallel loop is called. Additionally, if reverse complements are
	being considered, values are merged across both strands.
	"""

	T_max = max(T_lens)
	
	Q_offsets = numpy.zeros(len(Q_lens)+1, dtype='int64')
	Q_offsets[1:] = numpy.cumsum(Q_lens)
	Q_max = max(Q_lens)
	
	n_in_targets = len(T_lens) // 2 if reverse_complement else len(T_lens)
	n_out_targets = n_in_targets if n_nearest == -1 else n_nearest
	n_outputs = 5 if n_nearest == -1 else 6
	nt = T.shape[-1]

	# Re-usable workspace for each thread instead of re-allocating
	# and freeing large arrays for each example.
	n = numba.get_num_threads()
	n_len = Q_max*n_score_bins + Q_max*n_cache
	
	_gamma = numpy.empty((n, nt, Q_max), dtype='float64')
	_gamma_int = numpy.empty((n, nt, Q_max), dtype='int8')
	_f = numpy.empty((n, Q_max, n_score_bins+1), dtype='float64')

	_A = numpy.empty((n, Q_max, Q_max, n_len), dtype='float64')
	_B = numpy.empty((n, T_max+1, n_len), dtype='float64')
	_A_csum = numpy.empty((n, Q_max, Q_max, n_len), dtype='float64')

	_medians = numpy.empty((n, Q_max), dtype='float64')
	_median_bins = numpy.empty((n, n_median_bins, 2), dtype='float64')

	_results = numpy.empty((n, len(T_lens), 5), dtype='float64')
	results = numpy.empty((len(Q_lens), n_out_targets, n_outputs), 
		dtype='float64') 

	for i in prange(len(Q_lens)):
		nq = Q_lens[i]
		pid = numba.get_thread_id()

		offset = _integer_distances_and_histogram(Q, T, _gamma[pid], 
			_gamma_int[pid], _f[pid], _medians[pid], _median_bins[pid], Q_norm, 
			T_norm, rr_counts, Q_offsets[i], nq, n_score_bins)

		if offset > n_cache:
			print("Offset is larger than `n_cache`. Please increase `n_cache`"
				" to at least ", offset)

		_p_value_backgrounds(_f[pid], _A[pid], _B[pid], _A_csum[pid], nq, 
			n_score_bins, T_max, offset)

		_p_values(_gamma_int[pid], _B[pid], rr_inv, T_lens, i, nq, offset, 
			_results[pid])

		if reverse_complement == 1:
			_merge_rc_results(_results[pid])
		else:
			_results[pid, :, 4] = 0

		results[i] = _results[pid, :n_in_targets]

	# Enforce symmetry
	if n_nearest == -1:
		for i in range(results.shape[-1]):
			for j in range(results.shape[0]):
				for k in range(j):
					results[j, k, i] = results[k, j, i]

	return results           
  

def symmetric_tomtom(Xs, n_score_bins=100, n_median_bins=1000, 
	n_target_bins=100, n_cache=100, reverse_complement=True, n_jobs=-1):
	"""A method for assigning p-values to motif similarity.

	This method implements the TOMTOM algorithm for assigning p-values to motif
	similarity scores. TOMTOM accounts for several issues that arise when
	motifs are scanned against each other, including correctly calculating
	scores for overlaps and accounting for motif length and information content 
	within the motifs. 

	At a high level, TOMTOM works by calculating a background distribution of 
	scores for each position in the query and then uses dynamic programming to 
	calculating a distribution of scores for each span of matches, allowing for 
	potential overhangs on either side.

	Importantly, this method implements the "complete score" version of TOMTOM
	which is more robust to edge effects. The "incomplete score" is not a good
	score and so is not implemented. 


	Parameters
	----------
	Qs: list or numpy.ndarrays or torch.Tensors with shape (len(alphabet), len)
		A list of query motifs to consider. Each query must have a shape
		according to the PyTorch format where the length is the last aspect.
		If these are PyTorch tensors they will be internally converted to a
		numpy.ndarray.

	Ts: list or numpy.ndarrays or torch.Tensors with shape (len(alphabet), len)
		A list of target motifs to compare each query against. Each target must 
		have a shape according to the PyTorch format where the length is the 
		last aspect. If these are PyTorch tensors they will be internally 
		converted to a numpy.ndarray.

	n_nearest: int or None, optional
		The number of nearest targets to keep for each query, where nearness is
		defined by the p-value. Setting this can significant reduce memory
		because, otherwise, you get a len(Qs) by len(Ts) complete matrix. If
		None, return the complete matrix. Default is None.

	n_score_bins: int, optional
		The number of bins to use when discretizing scores. A higher number is 
		not necessarily better because you need the data to support each bin
		in the distribution. This is `t` from the TOMTOM paper. Default is 100.

	n_median_bins: int, optional
		The number of bins to use when approximating the medians. More bins
		means higher precision when estimating the median but can also cause it
		to take linearly longer. Default is 1000.

	n_target_bins: int or None, optional
		Whether to use approximate hashing to speed up calculations by merging
		target columns that are similar. This can significantly speed up
		calculations and reduce memory at the cost of approximation. Each value 
		in the columns are binned and targets are merged together if all values 
		fall within the same bins, e.g., if both columns after binning are 
		[5, 11, 0, 1]. This parameter sets the number of bins to use when 
		discretizing the values in the target columns. Fewer bins means more 
		targets get merged together, which can speed up the calculations, but 
		also mean that the resulting p-values are less accurate. Conversely, 
		more bins means that fewer targets get merged together and higher 
		accuracy p-values but slower. If None, don't use approximate hashing.
		Default is 100.

	n_cache: int, optional
		A cache size to use when allocating the scratchpad. A higher number will
		linearly increase the amount of memory used but will not increase the
		amount of compute needed. Default is 250.

	reverse_complement: bool, optional
		Whether to automatically compare each query to targets and also the
		reverse complement of the target and merge the scores and p-values
		accordingly. Default is True.

	n_jobs: int, optional
		The number of threads for numba to use when parallelizing the
		processing of query sequences. If -1, use all available threads.
		Default is -1.


	Returns
	-------
	best_p_values: torch.Tensor, shape=(len(Qs), len(Ts))
		The p-value of the best alignment between each query and each target.

	best_scores: torch.Tensor, shape=(len(Qs), len(Ts))
		The scores of the best alignment between each query and each target.

	best_offsets: torch.Tensor, shape=(len(Qs), len(Ts))
		The offset of the best alignment between each query and each target.

	best_overlaps: torch.Tensor, shape=(len(Qs), len(Ts))
		The overlap of the best alignment between each query and each target.

	best_strands: torch.Tensor, shape=(len(Qs), len(Ts))
		The strand for the best alignment between each query and each target.

	best_idxs: torch.Tensor, shape=(len(Qs), len(Ts)), optional
		When returning only a number of nearest neighbors, the index in the
		original ordering of the targets corresponding to each returned
		neighbor. These will be sorted by p-value.
	"""
	
	if n_jobs != -1:
		_n_jobs = numba.get_num_threads()
		numba.set_num_threads(n_jobs)

	if isinstance(Xs[0], torch.Tensor):
		Xs = [X.numpy(force=True) for X in Xs]

	# Enforce ordering
	X_lens = numpy.array([X.shape[-1] for X in Xs], dtype='int64')
	X_idxs = numpy.argsort(X_lens, kind='stable')
	Xs = [Xs[idx] for idx in X_idxs]

	Q_lens = numpy.array([X.shape[-1] for X in Xs], dtype='int64')
	Q = numpy.concatenate(Xs, axis=-1)
	Q_norm = (Q ** 2).sum(axis=0)
	
	if reverse_complement:        
		Xs = Xs + [X[::-1, ::-1] for X in Xs]

	T_lens = numpy.array([X.shape[-1] for X in Xs], dtype='int64')
	T = numpy.concatenate(Xs, axis=-1)
	T_norm = (T ** 2).sum(axis=0)

	# Proceeds normally from here
	if Q_norm.max() == 0 or T_norm.max() == 0:
		raise ValueError("Cannot have all-zeroes as targets or query.")

	if n_target_bins is not None:
		T_min = T.min(axis=-1, keepdims=True)
		T_max = T.max(axis=-1, keepdims=True)
		T_max[T_max == T_min] = T_min[T_max == T_min] + 1

		T_ints = numpy.around((T - T_min) / (T_max - T_min) * (n_target_bins-1))
		T_ints = T_ints.T.dot(n_target_bins ** numpy.arange(len(T))[:, None])
		_, rr_idxs, rr_inv, rr_counts = numpy.unique(T_ints.flatten(), 
			return_index=True, return_inverse=True, return_counts=True)

		T = T[:, rr_idxs]
		T_norm = T_norm[rr_idxs]
		rr_inv = rr_inv.astype('uint64')
	else:
		rr_inv = numpy.arange(T.shape[-1])
		rr_counts = numpy.ones_like(rr_inv)
	
	###
	
	results = _tomtom(Q, T, Q_lens, T_lens, Q_norm, T_norm, rr_inv, rr_counts, 
		-1, n_score_bins, n_median_bins, n_cache, int(reverse_complement))

	if n_jobs != -1:
		numba.set_num_threads(_n_jobs)

	### Undo swap 

	X_idxs2 = numpy.argsort(X_idxs)
	return torch.from_numpy(results[X_idxs2][:, X_idxs2]).permute(2, 0, 1)
