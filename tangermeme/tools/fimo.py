# fimo.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import math
import numba
import numpy
import torch
import pandas
import pyfaidx
import time

from ..io import read_meme

from tqdm import tqdm

@numba.njit('float64(float64, float64)', fastmath=True, cache=True)
def logaddexp2(x, y):
	"""Calculate the logaddexp in a numerically stable manner in base 2.

	This function is a fast implementation of the logaddexp2 function that
	operates on two numbers and is numerically stable. It should mimic the
	functionality of numpy.logaddexp2 except that it does not have the overhead
	of working on numpy arrays.


	Parameters
	----------
	x: float32
		A single number in log space.

	y: float32
		Another single number in log space.


	Returns
	-------
	z: float32
		The result of log2(pow(2, x) + pow(2, y))
	"""

	if x == float("-inf") and y == float("-inf"):
		return float("-inf")

	if x == float("inf") or y == float("inf"):
		return float("inf")

	vmax, vmin = max(x, y), min(x, y)
	return vmax + math.log2(math.pow(2, vmin - vmax) + 1)


@numba.njit(cache=True)
def _pwm_to_mapping(log_pwm, bin_size):
	"""An internal method for calculating score <-> log p-value mappings.

	This function takes in a PWM consisting of log probabilities and outputs
	a mapping between observed scores (as a convolution of the PWM across a
	one-hot encoded sequence) and log p-values. This mapping is calculated 
	quickly using dynamic programming scanning over all potential sequences.

	Importantly, the p-values are in log space meaning that values near zero
	at the start of the array are insignificant whereas those with large 
	magnitude towards the end of the array are more statistically significant.


	Parameters
	----------
	log_pwm: numpy.ndarray, shape=(len(alphabet), length)
		A position-weight matrix containing a motif encoded as the log
		probability of any character in any position.

	bin_size: float
		The size of the score bins to map to p-values. The smaller this value,
		the more bins, indicating higher precision but also longer calculation
		time.


	Returns
	-------
	smallest: int
		The number of bins between true zero and the smallest value in the
		array. In other words, the offset to subtract from binned scores to get
		p-values.

	log1mcdf: numpy.ndarray
		The log of 1 minus the cdf, or in other words, the log p-values
		associated with each score bin.
	"""

	n, l = log_pwm.shape

	log_bg = math.log2(0.25)
	int_log_pwm = numpy.round(log_pwm / bin_size).astype(numpy.int32)

	smallest, largest = 9999999, -9999999
	log_pwm_min_csum, log_pwm_max_csum = 0, 0
	for i in range(l):
		log_pwm_min = 9999999
		log_pwm_max = -9999999

		for j in range(n):
			log_pwm_min = min(log_pwm_min, int_log_pwm[j, i])
			log_pwm_max = max(log_pwm_max, int_log_pwm[j, i])

		log_pwm_min_csum += log_pwm_min
		log_pwm_max_csum += log_pwm_max

		smallest = min(smallest, log_pwm_min_csum)
		largest = max(largest, log_pwm_max_csum)

	largest += l

	logpdf = numpy.empty(largest - smallest + 1)
	old_logpdf = -numpy.inf * numpy.ones(largest - smallest + 1)
	for i in range(n):
		idx = int_log_pwm[i, 0] - smallest
		old_logpdf[idx] = logaddexp2(old_logpdf[idx], log_bg)

	for i in range(1, l):
		for j in range(largest - smallest + 1):
			logpdf[j] = -numpy.inf

		for j, x in enumerate(old_logpdf):
			if x != -numpy.inf:
				for k in range(n):
					idx = j + int_log_pwm[k, i]
					logpdf[idx] = logaddexp2(logpdf[idx], log_bg + x)

		for j in range(largest - smallest + 1):
			old_logpdf[j] = logpdf[j]

	for i in range(len(logpdf) - 2, -1, -1):
		logpdf[i] = logaddexp2(logpdf[i], logpdf[i + 1])

	return smallest, logpdf


@numba.njit(parallel=True, cache=True)
def _all_pwm_to_mapping(motifs, motif_lengths, bin_size):
	n = len(motif_lengths) - 1

	smallests = numpy.empty(n, dtype='int64')
	logpdfs = [numpy.empty(0) for i in range(n)]

	for i in numba.prange(n):
		s, e = motif_lengths[i], motif_lengths[i+1]

		smallest, logpdf = _pwm_to_mapping(motifs[:, s:e], bin_size)
		smallests[i] = smallest
		logpdfs[i] = logpdf

	return smallests, logpdfs


@numba.njit(parallel=True, fastmath=True, cache=True)
def _fast_hits(X, chrom_lengths, pwm, pwm_lengths, score_threshold, bin_size, 
	smallest, score_to_pvals, score_to_pval_lengths):
	n_motifs = len(pwm_lengths) - 1
	n_chroms = len(chrom_lengths) - 1

	hits = []
	for i in range(n_motifs):
		j = numpy.int64(1)
		k = numpy.uint64(1)
		l = numpy.float64(1.0)
		hits.append([(j, k, k, l, l) for z in range(0)])

	for k in numba.prange(n_motifs):
		n = pwm_lengths[k+1] - pwm_lengths[k]
		k = numpy.uint64(k)
		thresh = score_threshold[k]
		
		for l in range(n_chroms):        
			start = numpy.uint64(chrom_lengths[l])
			end = numpy.uint64(chrom_lengths[l+1])
			
			for i in range(end-start-n):
				i = numpy.uint64(i)
				
				score = 0.0
				for j in range(n):
					j = numpy.uint64(j)
					
					idx = X[start+i+j]
					if idx == -1:
						continue

					m_idx = numpy.uint64(j + pwm_lengths[k])
					idx = numpy.uint64(idx)
					score += pwm[idx, m_idx]

				if score > thresh:
					score_idx = int(score / bin_size) - smallest[k]                    
					score_idx += score_to_pval_lengths[k]
					hits[k].append((numpy.int64(l), i, i+n, score, 
						2.0 ** score_to_pvals[score_idx]))

	return hits


@numba.njit(cache=True)
def _fast_convert(X, mapping):
	for i in range(X.shape[0]):
		X[i] = mapping[X[i]]


def fimo(motifs, sequences, alphabet=['A', 'C', 'G', 'T'], bin_size=0.1, 
	eps=0.0001, threshold=0.0001, reverse_complement=True, return_counts=False, 
	dim=0):
	"""An implementation of the FIMO algorithm from the MEME suite.

	This function implements the "Finding Individual Motif Instances" (FIMO)
	algorithm from the MEME suite. This algorithm takes a set of PWMs and
	identifies where these PWMs have statistically significant hits against a
	set of sequences. These sequences can either come from a FASTA file, such
	as an entire genome or a set of peaks, or can be one-hot encoded sequences.

	This implementation uses numba to accelerate the inner loop, and
	parallelizes across the motif axis. No support exists for calculating
	q-values as, in my opinion, q-values do not make sense here and are both
	compute- and memory-inefficient.


	Parameters
	----------
	motifs: str or dict
		A MEME file to load containing motifs to scan, or a dictionary where
		the keys are names of motifs and the values are PWMs with shape
		(len(alphabet), pwm_length).

	sequences: str or numpy.ndarray or torch.Tensor
		A set of sequences to scan the motifs against. If this is a string,
		assumes it is a filepath to a FASTA-formatted file. If this is a numpy
		array or PyTorch tensor, will use those instead.

	alphabet: list, optional
		A list of characters to use for the alphabet, defining the order that
		characters should appear. Default is ['A', 'C', 'G', 'T'].

	bin_size: float, optional
		The size of the bins discretizing the PWM scores. The smaller the bin
		size the higher the resolution, but the less data may be available to
		support it. Default is 0.1.

	eps: float, optional
		A small pseudocount to add to the motif PWMs before taking the log.
		Default is 0.0001.

	threshold: float, optional
		The p-value threshold to use for reporting matches. Default is 0.0001.

	reverse_complement: bool, optional
		Whether to scan each motif and also the reverse complements. Default
		is True.

	return_counts: bool, optioal
		Whether to only return the count of the number of matches instead of
		dataframes containing information about each match. If True, the return
		will be a single array. Default is False

	dim: 0 or 1, optional
		Whether to return one dataframe for each motif containing all hits for
		that motif across all examples (0, default) or one dataframe for each 
		example containing all hits across all motifs to that example (1).
		Default is 0.


	Returns
	-------
	hits: list of pandas.DataFrames or numpy.ndarray
		A list of pandas.DataFrames containing motif hits, where the exact
		semantics of each dataframe are determined by `dim`. Alternatively,
		a numpy array of just the number of counts per motif if return_counts
		is set to True.
	"""

	log_threshold = math.log2(threshold)

	# Extract the motifs and potentially the reverse complements
	if isinstance(motifs, str):
		motifs_ = read_meme(motifs)
	elif isinstance(motifs, dict):
		motifs_ = motifs
	else:
		raise ValueError("`motifs` must be a dict or a filename.")

	motifs_ = list(motifs_.items())
	motifs = [(name, pwm.numpy(force=True)) for name, pwm in motifs_]
	if reverse_complement:
		for name, pwm in motifs_:
			motifs.append((name + '-rc', pwm.numpy(force=True)[::-1, ::-1]))

	# Initialize arrays to store motif properties
	n_motifs = len(motifs)

	motif_names = numpy.array([name for name, _ in motifs])
	motif_lengths = [0] + [pwm.shape[-1] for _, pwm in motifs]
	motif_lengths = numpy.cumsum(motif_lengths).astype(numpy.uint64)

	motif_pwms = numpy.concatenate([pwm for _, pwm in motifs], axis=-1)
	motif_pwms = numpy.log2(motif_pwms + eps) - math.log2(0.25)

	_smallest, _score_to_pvals = _all_pwm_to_mapping(motif_pwms, motif_lengths, 
		bin_size)
	_score_to_pvals_lengths = [0]
	_score_thresholds = numpy.empty(n_motifs, dtype=numpy.float32)

	for i in range(n_motifs):	
		_score_to_pvals_lengths.append(len(_score_to_pvals[i]))

		idx = numpy.where(_score_to_pvals[i] < log_threshold)[0]
		if len(idx) > 0:
			_score_thresholds[i] = (idx[0] + _smallest[i]) * bin_size                              
		else:
			_score_thresholds[i] = float("inf")

	_score_to_pvals = numpy.concatenate(_score_to_pvals)
	_score_to_pvals_lengths = numpy.cumsum(_score_to_pvals_lengths)

	# Extract the sequence from a FASTA or torch tensors
	if isinstance(sequences, str):
		fasta = pyfaidx.Fasta(sequences)
		sequence_names = numpy.array(list(fasta.keys()))
		X, lengths = [], [0]
		
		alphabet = ''.join(alphabet)
		alpha_idxs = numpy.frombuffer(bytearray(alphabet, 'utf8'), 
			dtype=numpy.int8)
		one_hot_mapping = numpy.zeros(256, dtype=numpy.int8) - 1
		for i, idx in enumerate(alpha_idxs):
			one_hot_mapping[idx] = i
		
		for name, chrom in fasta.items():
			chrom = chrom[:].seq.upper()
			lengths.append(lengths[-1] + len(chrom))
			
			X_idxs = numpy.frombuffer(bytearray(chrom, "utf8"), 
				dtype=numpy.int8)
			_fast_convert(X_idxs, one_hot_mapping)
			X.append(X_idxs)
			
		X = numpy.concatenate(X)
		X_lengths = numpy.array(lengths, dtype=numpy.int64)
			
	elif isinstance(sequences, (torch.Tensor, numpy.ndarray)):
		sequence_names = None
		X = ((sequences.argmax(axis=1) + 1) * sequences.sum(axis=1)) - 1
		X_lengths = numpy.arange(X.shape[0]+1) * X.shape[-1]

		if isinstance(X, torch.Tensor):
			X = X.numpy(force=True)

		X = X.astype(numpy.int8).flatten()
		X_lengths = X_lengths.astype(numpy.int64)

	# Use a fast numba function to run the core algorithm
	hits = _fast_hits(X, X_lengths, motif_pwms, motif_lengths, 
		_score_thresholds, bin_size, _smallest, _score_to_pvals, 
		_score_to_pvals_lengths)


	# Convert the results to pandas DataFrames
	names = ['sequence_name', 'start', 'end', 'score', 'p-value']
	n_ = n_motifs // 2 if reverse_complement else n_motifs

	if return_counts == True:
		counts = numpy.zeros(n_, dtype='int32')
		for i in range(n_):
			counts[i] = len(hits[i]) + len(hits[i+n_])
		return counts

	for i in range(n_):
		if reverse_complement:
			hits_ = pandas.DataFrame(hits[i] + hits[i + n_], columns=names)
			hits_['strand'] = ['+'] * len(hits[i]) + ['-'] * len(hits[i+n_])
		else:
			hits_ = pandas.DataFrame(hits[i], columns=names)
			hits_['strand'] = ['+'] * len(hits[i])

		hits_['motif_name'] = [motif_names[i] for _ in range(len(hits_))]
		hits_['motif_idx'] = numpy.ones(len(hits_), dtype='int64') * i

		if sequence_names is not None:
			hits_['sequence_name'] = sequence_names[hits_['sequence_name']]
			
		hits[i] = hits_[['motif_name', 'motif_idx', 'sequence_name', 'start', 
			'end', 'strand', 'score', 'p-value']]

	hits = hits[:n_]

	if dim == 1:
		hits = pandas.concat(hits)
		_names = numpy.unique(hits['sequence_name'])
		hits = [hits[hits['sequence_name'] == name].reset_index(drop=True) 
			for name in _names]

	return hits

