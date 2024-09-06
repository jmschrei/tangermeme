# tomtom.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com> 

import math
import numpy
import numba
import torch

from numba import njit
from numba import prange
from numpy import uint64


@njit
def _binned_median(x, bins, x_min, x_max, counts):
	"""An internal function for calculating medians quickly.

	This method uses a binning-based approximation to quickly calculate medians
	in linear time with low constants. Rather than using a sorting algorithm,
	which is O(n log n) or more sophisticated approaches that are O(n) but with
	bad constants, this approach approximates the median by dividing the range
	of the array into bins, assigning points to bins in one sweep of the data,
	and then scanning over all bins until half of points have been encountered.

	To get a better approximation of the median, sufficient statistics are
	stored that enable returning the average of all points assigned to the
	bin. When there an odd number of points, or an even number and the middle
	two get assigned to the same bin, this should return the exact median.
	"""

	n, n_bins = len(x), len(bins)
	bins[:] = 0

	halfway = 0
	x_max -= x_min
	for i in range(n):
		z = int((x[i] - x_min) / x_max * (n_bins - 1))
		bins[z, 0] += counts[i]
		bins[z, 1] += x[i] * counts[i]
		halfway += counts[i]

	halfway /= 2
	count = 0
	for i in range(n_bins):
		count += bins[i, 0]
		if count >= halfway:
			return bins[i, 1] / bins[i, 0]
			
	return -99999


@njit
def _integer_distances_and_histogram(X, Y, gamma, f, Z, medians, median_bins, 
	X_norm, Y_norm, Y_counts, nq_csum, nq, n_bins):
	"""An internal function for integerized scores and the histogram.

	This function is the main workhorse for the TOMTOM algorithm. It contains
	four conceptual steps: (1) calculate the distance matrix between each column 
	in one query and each column across all targets, (2) subtract out the per
	query-column median, (3) integerize the scores into bins, and (4) calculate
	the histogram of these integers. 

	Several speed efficiencies have been built into this, including caching
	minimum and maximum values for each query column for re-use in the median
	calculation, the binned median approximation, and calculating the histogram
	simultaneously with the binned score matrix. 
	"""
	
	# Calculate the Euclidean distance between query and targets
	z_min, z_max = 9999999.9, -9999999.9
	for i in range(nq):
		z_min_, z_max_ = 9999999.9, -9999999.9
		for j in range(Y.shape[-1]):
			z = X_norm[i + nq_csum] + Y_norm[j]
			
			for k in range(Y.shape[0]):
				z -= 2 * X[k, i + nq_csum] * Y[k, j]
			  
			z = -math.sqrt(z) if z > 0 else 0
			z_max_ = max(z_max_, z)
			z_min_ = min(z_min_, z)
			Z[i, j] = z
		
		# Subtract out the median from each row
		m = _binned_median(Z[i], median_bins, z_min_, z_max_, 
			Y_counts)
		medians[i] = m

		z_min = min(z_min, z_min_ - m)
		z_max = max(z_max, z_max_ - m)
			
	# Find the minimum value and the number of bins needed to get there
	i_min = int(math.floor(z_min)) #offset
	bin_scale = int(math.floor(n_bins / (z_max - i_min))) #scale
	f[:] = 0
	
	ys = numpy.sum(Y_counts)
	# Convert the distances to bins and record the histogram of counts
	for i in range(nq):
		for j in range(Y.shape[-1]):
			x = (Z[i, j] - i_min - medians[i]) * bin_scale

			if x >= 0:
				x_int = uint64(math.floor(x + 0.5))
			else:
				x_int = uint64(math.floor(x - 0.5))
				
			gamma[i, j] = x_int
			f[i, x_int] += Y_counts[j] / ys

	return uint64(-i_min * bin_scale)


@njit
def _pairwise_max(x, y, y_csum, z, n):
	"""An internal function for the pdf of the maximum of two pdfs.

	This function takes in two probability distribution functions and
	returns the probability distribution function for the maximum of
	the two. In other words, it returns the probability distribution
	for the maximum of a randomly drawn sample from the first
	distribution and a randomly drawn sample from the second
	distribution.

	This function relies on knowing that the cumsum of y will be
	precalculated and that the cumsum of x has to be recalculated
	each call.
	"""
	
	if x[0] == -1:
		z[:] = y[:]        
	else:
		x_csum = 0
		for i in range(n):
			x_csum += x[i]
			z[i] = x[i] * y_csum[i] + y[i] * x_csum - x[i] * y[i]

 
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
	A[:] = 0
	
	for i in range(nq):
		i = uint64(i)
		for j in range(i, nq):
			j, c = uint64(j), uint64(offset * (nq - j + i - 1))
			
			if i == j:
				for l in range(1, n_bins+1):
					l = uint64(l)
					A[i, j, l+c] = f[j, l]
			else:            
				for k in range(n_bins*j+1):
					k = uint64(k)
					a = A[i, j-1, k+c+offset]
					
					if a == 0:
						continue
						
					for l in range(1, n_bins+1):
						l = uint64(l)
						A[i, j, l+k+c] += a * f[j, l]
					
			A_csum[i, j, n_bins*(j+1)+c:] = 1
			for k in range(n_bins*(j+1)+c):
				k = uint64(k)
				A_csum[i, j, k] = A[i, j, k]
				if k > 0:
					A_csum[i, j, k] += A_csum[i, j, k-1]

	###

	B[0] = -1
	for i in range(1, min(nq, t_max+1)):
		_pairwise_max(B[i-1], A[0, i-1], A_csum[0, i-1], B[i], n)
		_pairwise_max(B[i], A[nq-i, nq-1], A_csum[nq-i, nq-1], B[i], n)

	if (t_max+1) > nq:
		for i in range(nq, t_max+1):
			_pairwise_max(B[i-1], A[0, nq-1], A_csum[0, nq-1], B[i], n)
	 
	for i in range(1, min(nq, t_max+1)):
		B[i] = -1
		for j in range(nq - i + 1):
			_pairwise_max(B[i], A[j, j+i-1], A_csum[j, j+i-1], B[i], n)
	
		for j in range(i-1):
			_pairwise_max(B[i], A[0, j], A_csum[0, j], B[i], n)
			_pairwise_max(B[i], A[nq-1-j, nq-1], A_csum[nq-1-j, nq-1], 
				B[i], n)

	# Again, `axis` is not implemented for cumsum
	for i in range(B.shape[0]):
		for j in range(1, n):
			B[i, j] += B[i, j-1]
		
		for j in range(n):
			B[i, j] = 1 - B[i, j]
			

@njit
def _p_values(gamma, B_cdfs, rr_inv, T_lens, nq, offset, n_bins, results):
	"""An internal function for calculating the best match and p-values.

	This function will take in the integerized score matrix `gamma` and
	background distributions `B_cdfs` and calculate the best overlap.
	The best overlap is calculated as the best sum of scores across the
	alignment, minus a penalty for each unaligned column. After finding
	a new best overlap, the p-value is calculated by comparing the
	score to the background distribution.
	"""

	total_offset = 0
	for i, nt in enumerate(T_lens):
		results[i, 1] = 0
		
		for k in range(-nq + 1, nt):
			score = 0
			overlap = 0
			
			if k < 0:
				for j in range(-k, nq):
					if nt < nq and overlap == nt:
						break

					j, t_idx = uint64(j), uint64(total_offset + j + k)
					score += gamma[j, rr_inv[t_idx]]
					overlap += 1
			else:
				for j in range(min(nq, nt-k)):                    
					j, t_idx = uint64(j), uint64(total_offset + j + k)
					score += gamma[j, rr_inv[t_idx]]
					overlap += 1
		
			score = score + (nq - overlap) * offset
			if score >= results[i, 1]:
				if score == results[i, 1] and results[i, 2] >= overlap:
					continue

				results[i, 0] = B_cdfs[nt, uint64(score-1)]
				results[i, 1] = score
				results[i, 2] = k
				results[i, 3] = overlap

		total_offset += nt


@njit
def _merge_rc_results(results):
	"""An internal method for taking the best across two strands."""

	nt = results.shape[0]
	n = nt // 2
	
	for i in range(n):
		p = min(results[i, 0], results[i+n, 0])
		p = 1 - (1 - p) ** 2

		results[i, 0] = p
		results[i, 4] = 0
		
		if results[i, 1] <= results[i+n, 1]:                
			results[i, 1] = results[i+n, 1]
			results[i, 2] = results[i+n, 2]
			results[i, 3] = results[i+n, 3]
			results[i, 4] = 1
			

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
	
	_Z = numpy.empty((n, Q_max, nt), dtype='float64')
	_gamma = numpy.empty((n, Q_max, nt), dtype='int32')
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

		offset = _integer_distances_and_histogram(Q, T, _gamma[pid], _f[pid], 
			_Z[pid], _medians[pid], _median_bins[pid], Q_norm, T_norm, 
			rr_counts, Q_offsets[i], nq, n_score_bins)

		if offset > n_cache:
			print("Offset is larger than `n_cache`. Please increase `n_cache`"
				" to at least ", offset)

		_p_value_backgrounds(_f[pid], _A[pid], _B[pid], _A_csum[pid], nq, 
			n_score_bins, T_max, offset)

		_p_values(_gamma[pid], _B[pid], rr_inv, T_lens, nq, offset, 
			n_score_bins, _results[pid])

		if reverse_complement == 1:
			_merge_rc_results(_results[pid])
		else:
			_results[pid, :, 4] = 0

		if n_nearest == -1:
			results[i] = _results[pid, :n_in_targets]
		else:
			idxs = numpy.argsort(_results[pid, :n_in_targets, 0])[:n_nearest]
			results[i, :, :5] = _results[pid, idxs]
			results[i, :, 5] = idxs

	return results           
  

def tomtom(Qs, Ts, n_nearest=None, n_score_bins=100, n_median_bins=1000, 
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
	best_p_values: numpy.ndarray, shape=(len(Qs), len(Ts))
		The p-value of the best alignment between each query and each target.

	best_scores: numpy.ndarray, shape=(len(Qs), len(Ts))
		The scores of the best alignment between each query and each target.

	best_offsets: numpy.ndarray, shape=(len(Qs), len(Ts))
		The offset of the best alignment between each query and each target.

	best_overlaps: numpy.ndarray, shape=(len(Qs), len(Ts))
		The overlap of the best alignment between each query and each target.

	best_strands: numpy.ndarray, shape=(len(Qs), len(Ts))
		The strand for the best alignment between each query and each target.
	"""

	if n_jobs != -1:
		_n_jobs = numba.get_num_threads()
		numba.set_num_threads(n_jobs)

	if n_nearest is None:
		n_nearest = -1

	if isinstance(Ts[0], torch.Tensor):
		Ts = [T.numpy(force=True) for T in Ts]

	
	Q_lens = numpy.array([Q.shape[-1] for Q in Qs], dtype='int64')
	Q = numpy.concatenate(Qs, axis=-1)
	Q_norm = (Q ** 2).sum(axis=0)
	
	if reverse_complement:        
		Ts = Ts + [T[::-1, ::-1] for T in Ts]
	
	T_lens = numpy.array([T.shape[-1] for T in Ts], dtype='int64')
	T = numpy.concatenate(Ts, axis=-1)
	T_norm = (T ** 2).sum(axis=0)

	if n_target_bins is not None:
		T_ints = numpy.around(T / T.max(axis=-1, keepdims=True) * 
			(n_target_bins-1))
		T_ints = T_ints.T.dot(n_target_bins ** numpy.arange(len(T))[:, None])
		_, rr_idxs, rr_inv, rr_counts = numpy.unique(T_ints, return_index=True, 
			return_inverse=True, return_counts=True)

		T = T[:, rr_idxs]
		T_norm = T_norm[rr_idxs]
		rr_inv = rr_inv.astype('uint64')
	
	###
	
	results = _tomtom(Q, T, Q_lens, T_lens, Q_norm, T_norm, rr_inv, rr_counts, 
		n_nearest, n_score_bins, n_median_bins, n_cache, 
		int(reverse_complement))

	if n_jobs != -1:
		numba.set_num_threads(_n_jobs)

	return results.transpose(2, 0, 1)
