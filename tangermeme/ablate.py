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


def insert(X, motif, start=None, alphabet=['A', 'C', 'G', 'T']):
	"""Insert a motif into a set of sequences at a defined position.

	This function will take in a tensor of one-hot encoded sequences or a string
	that can be one-hot encoded and will insert a motif at a defined position. 
	It will then return a copy of the data with the insertion, leaving the 
	original data unperturbed.

	If the motif is a string, it will be one-hot encoded according to the
	alphabet that is provided. If a motif with batch size of 1 is provided, 
	the same motif will be inserted into all sequences. If a motif with a batch 
	size equal to that of X is provided, there will be  1-1 correspondance 
	between the motifs inserted and the sequence, i.e., that motif at index 5 
	will be inserted into the sequence at index 5.


	Parameters
	----------
	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to have a motif inserted into.

	motif: torch.tensor, shape=(-1, len(alphabet), motif_length)
		A one-hot encoded version of a short motif to insert into the set of
		sequences.

	start: int or None, optional
		The starting position of where to insert the motif. If None, insert the
		motif into the middle of the sequence such that the middle of the motif
		occurs at the middle of the sequence. Default is None.

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
		A one-hot encoded set of sequences that each have the motif inserted at
		the same position.
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


def randomize(X, start, end, probs=[0.25, 0.25, 0.25, 0.25], same=True,
	random_state=None):
	"""Replace a region of the provided loci with randomly drawn sequence.

	This function will take in a batch of sequences and replace the provided
	region with randomly generated sequence. By passing in a vector of
	probabilities, the user can alter the composition of generated characters.

	By default, this method will randomly generate a single insert and place
	it into each sequence. However, by passing `same=False` you can generate
	different inserts for each sequence.


	Parameters
	----------
	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to have a motif inserted into.

	start: int
		The starting position of where to randomize the sequence, inclusive. 

	end: int
		The ending position of where to randomize the sequence, not inclusive.

	probs: tuple, list, numpy.ndarray, optional
		An iterable of probabilities, where each value is the probability of
		that position being `on` in a one-hot encoding. The sum of the values
		must be equal to 1. Default is [0.25, 0.25, 0.25, 0.25].

	same: bool, optional
		Whether to generate a single insert and put it into each sequence (if
		True) or to generate a different insert for each sequence (if False).
		Default is True.

	random_state: int, numpy.random.RandomState, or None, optional
		Whether to use a specific random seed when generating the random insert,
		to ensure reproducibility. If None, do not use a reproducible seed.
		Default is None.


	Returns
	-------
	Y: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences that each have an insert.
	"""


	if not isinstance(numpy.random.RandomState):
		random_state = numpy.random.RandomState(random_state)

	'''
	if same:
		n = len(probs)
		
		seq = random_state.choice(range(n), probs=probs, shape=end-start)
		insertion = numpy.zeros((1, len(probs), end-start))
		insertion[:, seq, numpy.arange(end-start)] = 1
	else:
		awd
	'''

		
params = 'void(int64, int64, int64[:], int32[:, :], int32[:,], '
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


def dinucleotide_shuffle(X, n_shuffles=10, random_state=None, verbose=False):
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
	X: torch.tensor, shape=(k, -1)
		The one-hot encoded sequence. k is usually 4 for nucleotide sequences
		but can be anything in practice.

	n_shuffles: int, optional
		The number of dinucleotide shuffles to return. Default is 10.

	random_state: int or None or numpy.random.RandomState, optional
		The random seed to use to ensure determinism. If None, the
		process is not deterministic. Default is None.

	verbose: bool, optional
		Whether to print a warning if too sequence similarity is too high.


	Returns
	-------
	shuffled_sequences: torch.tensor, shape=(n_shuffles, k, -1)
		The shuffled sequences.
	"""

	_validate_input(X, "X", shape=(-1, -1), ohe=True, ohe_dim=0)

	if random_state is None:
		random_state = numpy.random.randint(0, 9999999)

	n_chars, seq_len = X.shape
	idxs = X.argmax(axis=0).numpy()

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
	if conserved.max(dim=0).values.min() == n_shuffles:
		raise ValueError("All dinucleotide shuffles yield identical " +
			"sequences, potentially due to a lack of diversity in sequence.")

	print(conserved.max(dim=0).values.min(), conserved.max())

	return shuffled_sequences
