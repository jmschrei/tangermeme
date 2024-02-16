# ablate.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import numba
import numpy
import torch
import pandas

import pyfaidx
import pyBigWig

from tqdm import tqdm

from .utils import _validate_input
from .utils import one_hot_encode
from .utils import random_one_hot


def insert(X, motif, start=None, alphabet=['A', 'C', 'G', 'T']):
	"""Insert a motif into a set of sequences at a defined position.

	This function will take in a tensor of one-hot encoded sequences or a string
	that can be one-hot encoded and insert the motif into the defined 
	position. It will then return a copy of the data with the insertion, 
	leaving the  original data unperturbed.

	Importantly, an *insertion* means that the entire original sequence is still
	present, albeit in two halves with the inserted motif in the middle.
	Specifically, if we have an original sequence AAAAAACCCCAAAAAA and want to
	insert GGGG in the middle, the `insert` function will return something
	corresponding to AAAAAACCGGGGCCAAAAAA. Hence, the returned sequence will be
	longer than the original sequence.

	If the motif is a string, it will be one-hot encoded according to the
	alphabet that is provided. If a motif with batch size of 1 is provided, 
	the same motif will be inserted into all sequences. If a motif with a 
	batch size equal to that of X is provided, there will be  1-1 correspondance 
	between the motifs and the sequence, i.e., that motif at index 5 will be 
	substituted into the sequence at index 5.


	Parameters
	----------
	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to have a motif substituted into.

	motif: torch.tensor, shape=(-1, len(alphabet), motif_length)
		A one-hot encoded version of a short motif to substitute into the set of
		sequences.

	start: int or None, optional
		The starting position of where to substitute the motif. If None,
		substitute the motif into the middle of the sequence such that the 
		middle of the motif occurs at the middle of the sequence. Default is 
		None.

	alphabet : set or tuple or list, optional
		A pre-defined alphabet where the ordering of the symbols is the same
		as the index into the returned tensor, i.e., for the alphabet ['A', 'B']
		the returned tensor will have a 1 at index 0 if the character was 'A'.
		Characters outside the alphabet are ignored and none of the indexes are
		set to 1. This is not necessary or used if a one-hot encoded tensor is
		provided for the motif. Default is ['A', 'C', 'G', 'T'].


	Returns
	-------
	Y: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences that each have the motif substituted
		at the same position.
	"""

	if isinstance(motif, str):
		motif = one_hot_encode(motif, alphabet=alphabet).unsqueeze(0)

	if motif.shape[0] == 1:
		motif = motif.repeat(X.shape[0], 1, 1)

	_validate_input(X, "X", ohe=True, ohe_dim=1)
	_validate_input(motif, "motif", shape=(-1, X.shape[1], -1), ohe=True)

	if start is not None:
		if start < 0 or start > (X.shape[-1] - motif.shape[-1]):
			raise ValueError("Provided start falls off the end of the sequence")
	else:
		start = X.shape[-1] // 2
	
	return torch.cat([X[:, :, :start], motif, X[:, :, start:]], dim=-1)


def substitute(X, motif, start=None, alphabet=['A', 'C', 'G', 'T']):
	"""Substitute a motif into a set of sequences at a defined position.

	This function will take in a tensor of one-hot encoded sequences or a string
	that can be one-hot encoded and will substitute a motif at a defined 
	position. It will then return a copy of the data with the substitution, 
	leaving the  original data unperturbed.

	Importantly, a *substitution* means that part of the original sequence will
	be missing. Specifically, if we have an original sequence AAAAAACCCCAAAAAA 
	and want to substitute a GGGG in the middle, the `substitute` function will 
	return something corresponding to AAAAAAGGGGAAAAAA. Note the missing Cs. 
	Hence, the returned sequence will be the same length as the original 
	sequence.

	If the motif is a string, it will be one-hot encoded according to the
	alphabet that is provided. If a motif with batch size of 1 is provided, 
	the same motif will be substituted into all sequences. If a motif with a 
	batch size equal to that of X is provided, there will be  1-1 correspondance 
	between the motifs and the sequence, i.e., that motif at index 5 will be 
	substituted into the sequence at index 5.


	Parameters
	----------
	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to have a motif substituted into.

	motif: torch.tensor, shape=(-1, len(alphabet), motif_length)
		A one-hot encoded version of a short motif to substitute into the set of
		sequences.

	start: int or None, optional
		The starting position of where to substitute the motif. If None,
		substitute the motif into the middle of the sequence such that the 
		middle of the motif occurs at the middle of the sequence. Default is 
		None.

	alphabet : set or tuple or list, optional
		A pre-defined alphabet where the ordering of the symbols is the same
		as the index into the returned tensor, i.e., for the alphabet ['A', 'B']
		the returned tensor will have a 1 at index 0 if the character was 'A'.
		Characters outside the alphabet are ignored and none of the indexes are
		set to 1. This is not necessary or used if a one-hot encoded tensor is
		provided for the motif. Default is ['A', 'C', 'G', 'T'].


	Returns
	-------
	Y: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences that each have the motif substituted
		at the same position.
	"""

	if isinstance(motif, str):
		motif = one_hot_encode(motif, alphabet=alphabet).unsqueeze(0)

	_validate_input(X, "X", ohe=True)
	_validate_input(motif, "motif", shape=(-1, X.shape[1], -1), ohe=True)

	if motif.shape[-1] > X.shape[-1]:
		raise ValueError("Motif cannot be longer than sequence.")

	if start is not None:
		if start < 0 or start > (X.shape[-1] - motif.shape[-1]):
			raise ValueError("Provided start falls off the end of the sequence")
	else:
		start = X.shape[-1] // 2 - motif.shape[-1] // 2


	n = motif.shape[-1]

	X = torch.clone(X)
	X[:, :, start:start+n] = motif
	return X


def randomize(X, start, end, probs=[[0.25, 0.25, 0.25, 0.25]], n=1,
	random_state=None):
	"""Replace a region of the provided loci with randomly drawn sequence.

	This function will take in a batch of sequences and replace region specified
	by `start` and `end` with randomly generated sequences. It will do this `n`
	times for each sequence in X and so return a tensor with one more dimension
	than `X`. By default, the random sequences are uniformly generated, but the 
	composition of sequence can be specified with the `probs` parameter.

	Importantly, this function does not shuffle the sequence in the specified
	region but replaces it with a random substitution. If you want to shuffle or
	dinucleotide shuffle the given range, use those respective functions
	instead.


	Parameters
	----------
	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences where a portion should be randomized.

	start: int
		The starting position of where to randomize the sequence, inclusive. 

	end: int
		The ending position of where to randomize the sequence, not inclusive.

	probs: 2D matrix, optional
		A 2D matrix of probabilities, as either a list of lists, numpy array, or
		torch tensor. The shape of this matrix is either (1, len(alphabet)) or
		(len(X), len(alphabet)), and is interpreted as either having the same
		probabilities across all examples or an example-specific set of
		probabilities. Default is [[0.25, 0.25, 0.25, 0.25]].

	n: int, optional
		The number of times to shuffle that region. Default is 1.

	random_state: int, numpy.random.RandomState, or None, optional
		Whether to use a specific random seed when generating the random 
		substitution to ensure reproducibility. If None, do not use a 
		reproducible seed. Default is None.


	Returns
	-------
	X_rands: torch.tensor, shape=(-1, n, len(alphabet), length)
		A one-hot encoded set of sequences that each have a randomized
		substitution.
	"""

	if not isinstance(probs, torch.Tensor):
		probs = torch.tensor(probs)

	if not isinstance(random_state, numpy.random.RandomState):
		random_state = numpy.random.RandomState(random_state)

	_validate_input(X, "X", ohe=True)
	_validate_input(probs, "Probs", shape=(-1, -1), min_value=0, max_value=1)

	if end <= start:
		raise ValueError("End must come after start.")

	if end >= X.shape[-1] or start < 0:
		raise ValueError("Start or end are falling off the edge of X.")

	X_rands = []
	for i in range(n):
		substitute_ohe = random_one_hot((X.shape[0], probs.shape[1], end-start), 
			probs=probs, random_state=random_state)

		X_rand = substitute(X, substitute_ohe, start=start)
		X_rands.append(X_rand)

	return torch.stack(X_rands).permute(1, 0, 2, 3)


def shuffle(X, start=0, end=-1, n=1, random_state=None):
	"""Replace a region of the provided loci with a shuffled version.

	This function will take in a batch of sequences and shuffle the specified
	region between `start` and `end`. This means that the returned sequences
	will have the same number of each nucleotide in the specified region, but
	in different positions. Importantly, this only preserves the number of times
	each character in the alphabet appears, not the number of times each
	dinucleotide appears. For that, use `dinucleotide_shuffle`.

	Importantly, the i-th shuffle of each sequence uses the same shuffling.
	Put another way, every sequence is shuffled *the same way* each iteration.
	This shuffling differs across iterations.


	Parameters
	----------
	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences where a portion will be shuffled.

	start: int, optional
		The starting position of where to randomize the sequence, inclusive.
		Default is 0, shuffling the entire sequence.

	end: int, optional
		The ending position of where to randomize the sequence, not inclusive.
		Default is -1, shuffling the entire sequence.

	n: int, optional
		The number of times to shuffle that region. Default is 1.

	random_state: int, numpy.random.RandomState, or None, optional
		Whether to use a specific random seed when generating the shuffle to
		ensure reproducibility. If None, do not use a reproducible seed. Default 
		is None.


	Returns
	-------
	Y: torch.tensor, shape=(-1, n, len(alphabet), length)
		A one-hot encoded set of sequences that each have a shuffled portion.
	"""

	_validate_input(X, "X", ohe=True)
	
	if end < 0:
		end = X.shape[-1] + 1 + end

	if end <= start:
		raise ValueError("End must come after start.")

	if end > X.shape[-1] or start < 0:
		raise ValueError("Start or end are falling off the edge of X.")

	if not isinstance(random_state, numpy.random.RandomState):
		random_state = numpy.random.RandomState(random_state)

	X_shufs = []
	for i in range(n):
		idxs = numpy.arange(end-start)
		random_state.shuffle(idxs)

		X_ = torch.clone(X)
		X_[:, :, start:end] = X[:, :, start:end][:, :, idxs]
		X_shufs.append(X_)

	return torch.stack(X_shufs).permute(1, 0, 2, 3)

		
params = 'void(int64, int64, int32[:], int32[:, :], int32[:], '
params += 'int32[:, :], float32[:, :, :], int32)'
@numba.jit(params, nopython=False)
def _fast_shuffle(n_shuffles, n_chars, idxs, next_idxs, next_idxs_counts, 
	counters, shuffled_sequences, random_state):
	"""An internal function for fast dinucleotide shuffling using numba."""

	numpy.random.seed(random_state)

	for i in range(n_shuffles):
		for char in range(n_chars):
			n = next_idxs_counts[char]

			next_idxs_ = numpy.arange(n)
			next_idxs_[:-1] = numpy.random.permutation(n-1)  # Keep last index
			next_idxs[char, :n] = next_idxs[char, :n][next_idxs_]

		idx = 0
		shuffled_sequences[i, idxs[idx], 0] = 1
		for j in range(1, len(idxs)):
			char = idxs[idx]
			count = counters[i, char]
			idx = next_idxs[char, count]

			counters[i, char] += 1
			shuffled_sequences[i, idxs[idx], j] = 1


def _dinucleotide_shuffle(X, n_shuffles=1, random_state=None, verbose=False):
	"""An internal function for dinucleotide shuffling a single sequence.

	This function takes in a one-hot encoded sequence (not a string) and
	returns a set of one-hot encoded sequences that are dinucleotide
	shuffled. The approach constructs a transition matrix between
	nucleotides, keeps the first and last nucleotide constant, and then
	randomly at uniform selects transitions until all nucleotides have
	been observed. This is a Eulerian path. Because each nucleotide has
	the same number of transitions into it as out of it (except for the
	first and last nucleotides) the greedy algorithm does not need to
	check at each step to make sure there is still a path.

	This function has been adapted to work on PyTorch tensors instead of
	numpy arrays. Code has been adapted from
	https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py


	Parameters
	----------
	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to be shuffled.

	start: int, optional
		The starting position of where to randomize the sequence, inclusive.
		Default is 0, shuffling the entire sequence.

	end: int, optional
		The ending position of where to randomize the sequence, not inclusive.
		Default is -1, shuffling the entire sequence.

	n: int, optional
		The number of times to shuffle that region. Default is 1.

	random_state: int, numpy.random.RandomState, or None, optional
		Whether to use a specific random seed when generating the shuffle,
		to ensure reproducibility. If None, do not use a reproducible seed.
		Default is None.


	Returns
	-------
	shuffled_sequences: torch.tensor, shape=(n, k, -1)
		The shuffled sequences.
	"""

	if random_state is None:
		random_state = numpy.random.randint(0, 9999999)

	n_chars, seq_len = X.shape
	idxs = X.argmax(axis=0).numpy().astype(numpy.int32)

	next_idxs = numpy.zeros((n_chars, seq_len), dtype=numpy.int32)
	next_idxs_counts = numpy.zeros(n_chars, dtype=numpy.int32)

	for char in range(n_chars):
		next_idxs_ = numpy.where(idxs[:-1] == char)[0]
		n = len(next_idxs_)

		next_idxs[char][:n] = next_idxs_ + 1
		next_idxs_counts[char] = n

	shuffled_sequences = numpy.zeros((n_shuffles, *X.shape), 
		dtype=numpy.float32)
	counters = numpy.zeros((n_shuffles, n_chars), dtype=numpy.int32)

	_fast_shuffle(n_shuffles, n_chars, idxs, next_idxs, next_idxs_counts, 
		counters, shuffled_sequences, random_state)
	
	shuffled_sequences = torch.from_numpy(shuffled_sequences)

	conserved = shuffled_sequences[:, :, 1:-1].sum(dim=0)
	if conserved.max() == n_shuffles:
		if verbose:
			print("Warning: At least one position in dinucleotide shuffle " +
				"is identical across all positions.")
	if conserved.max(dim=0).values.min() == n_shuffles and n_shuffles > 1:
		raise ValueError("All dinucleotide shuffles yield identical " +
			"sequences, potentially due to a lack of diversity in sequence.")

	return shuffled_sequences


def dinucleotide_shuffle(X, start=0, end=-1, n=20, random_state=None, 
	verbose=False):
	"""Given a one-hot encoded sequence, dinucleotide shuffle it.

	This function takes in a one-hot encoded sequence (not a string) and
	returns a set of one-hot encoded sequences that are dinucleotide
	shuffled. The approach constructs a transition matrix between
	nucleotides, keeps the first and last nucleotide constant, and then
	randomly at uniform selects transitions until all nucleotides have
	been observed. This is a Eulerian path. Because each nucleotide has
	the same number of transitions into it as out of it (except for the
	first and last nucleotides) the greedy algorithm does not need to
	check at each step to make sure there is still a path.

	This function has been adapted to work on PyTorch tensors instead of
	numpy arrays. Code has been adapted from
	https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py

	Parameters
	----------
	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to be shuffled.

	start: int, optional
		The starting position of where to randomize the sequence, inclusive.
		Default is 0, shuffling the entire sequence.

	end: int, optional
		The ending position of where to randomize the sequence, not inclusive.
		Default is -1, shuffling the entire sequence.

	n: int, optional
		The number of times to shuffle that region. Default is 20.

	random_state: int or None, optional
		Whether to use a specific random seed when generating the random insert,
		to ensure reproducibility. If None, do not use a reproducible seed.
		Unlike other methods, cannot be a numpy.random.RandomState object. 
		Default is None.


	Returns
	-------
	shuffled_sequences: torch.tensor, shape=(-1, n, k, -1)
		The shuffled sequences.
	"""

	_validate_input(X, "X", shape=(-1, -1, -1), ohe=True, ohe_dim=1)

	if random_state is None:
		random_state = numpy.random.randint(0, 9999999)

	X_shufs = []
	for i in range(X.shape[0]):
		insert_ = _dinucleotide_shuffle(X[i, :, start:end], n_shuffles=n, 
			random_state=random_state+i, verbose=verbose)

		X_shuf = torch.clone(X[i:i+1]).repeat(n, 1, 1)
		X_shuf[:, :, start:end] = insert_
		X_shufs.append(X_shuf)

	return torch.stack(X_shufs)
