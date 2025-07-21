# utils.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import numba
import torch

from tqdm import tqdm


def _validate_input(X, name, shape=None, dtype=None, min_value=None,
	max_value=None, ohe=False, ohe_dim=1, allow_N=False):
	"""An internal function for validating properties of the input.

	This function will take in an object and verify characteristics of it, such
	as the type, the datatype of the elements, its shape, etc. If any of these
	characteristics are not met, an error will be raised.


	Parameters
	----------
	X: torch.Tensor
		The object to be verified.

	name: str
		The name to reference the tensor by if an error is raised.

	shape: tuple or None, optional
		The shape the tensor must have. If a -1 is provided at any axis, that
		position is ignored.  If not provided, no check is performed. Default is
		None.

	dtype: torch.dtype or None, optional
		The dtype the tensor must have. If not provided, no check is performed.
		Default is None.

	min_value: float or None, optional
		The minimum value that can be in the tensor, inclusive. If None, no
		check is performed. Default is None.

	max_value: float or None, optional
		The maximum value that can be in the tensor, inclusive. If None, no
		check is performed. Default is None.

	ohe: bool, optional
		Whether the input must be a one-hot encoding, i.e., only consist of
		zeroes and ones. Default is False.

	allow_N: bool, optional
		Whether to allow the return of the character 'N' in the sequence, i.e.
		if pwm at a position is all 0's return N. Default is False.


	Returns
	X: torch.Tensor
		The same object, unmodified, for convenience.
	"""

	if not isinstance(X, torch.Tensor):
		raise ValueError("{} must be a torch.Tensor object".format(name))

	if shape is not None:
		if len(shape) != len(X.shape):
			raise ValueError("{} has shape {} but must have shape {}".format(name, 
				X.shape, shape))

		for i in range(len(shape)):
			if shape[i] != -1 and shape[i] != X.shape[i]:
				raise ValueError("{} has shape {} but must have shape {}".format(name,
					X.shape, shape))


	if dtype is not None and X.dtype != dtype:
		raise ValueError("{} has dtype {} but must have dtype {}".format(name, X.dtype, 
			dtype))

	if min_value is not None and X.min() < min_value:
		raise ValueError("{} cannot have a value below {}".format(name,
			min_value))

	if max_value is not None and X.max() > max_value:
		raise ValueError("{} cannot have a value above {}".format(name,
			max_value))

	if ohe:
		values = torch.unique(X)
		if len(values) != 2:
			raise ValueError("{} must be one-hot encoded.".format(name))

		if not all(values == torch.tensor([0, 1], device=X.device)):
			raise ValueError("{} must be one-hot encoded.".format(name))

		if allow_N:
			if not torch.all(torch.sum(X, axis=ohe_dim) <= 1):
				raise ValueError("{} must be one-hot encoded.".format(name) +
					"and contain unknown characters as all-zeroes.")
		else:
			if not torch.all(X.sum(axis=ohe_dim) == 1):
				raise ValueError("{} must be one-hot encoded ".format(name) +
					"and cannot have unknown characters.")				

	return X


def _cast_as_tensor(value, dtype=None):
	"""Cast your input as a torch tensor.

	This function will take some array-like input and cast it as a torch
	tensor with optionally a desired dtype. If X is already a torch datatype,
	ensure that the dtype matches.


	Parameters
	----------
	value: array-like
		An array-like object to be cast into a torch.tensor

	dtype: torch.dtype or None, optional
		A torch dtype to cast the values in the torch.tensor to. If None, do not
		do casting. Default is None.


	Returns
	-------
	value: torch.tensor
		A tensor that has been created from the array-like with the provided
		dtype.
	""" 

	if value is None:
		return None

	_tdtype = (torch.nn.Parameter, torch.Tensor, torch.masked.MaskedTensor)
	if isinstance(value, _tdtype):
		if dtype is None:
			return value
		elif value.dtype == dtype:
			return value
		else:
			return value.type(dtype)
			
	if isinstance(value, list):
		if all(isinstance(v, numpy.ndarray) for v in value):
			value = numpy.array(value)
		
	if isinstance(value, (float, int, list, tuple, numpy.ndarray)):
		if dtype is None:
			return torch.tensor(value)
		else:
			return torch.tensor(value, dtype=dtype)


def example_to_fasta_coords(example_df, loci_df, window=None, one_indexed=False):
	"""Converts coordinates within a given example to those in a FASTA file.
	
	Many analyses involve extracting windows from the genome and processing them
	independently. When coordinates are identified on these windows, e.g.,
	seqlets or other spans, all that is recorded is the example index and the
	relative positions from the start of the extracted window. Frequently, though,
	one wants to know these coordinates on the original genome or other entity
	encoded in a FASTA file. This function cross-references the example indexes
	and BED-formatted locus file that the examples were extracted from to get
	the coordinates on the original sequence.
	
	This cross-referencing has several steps.
	
		(1) For each span in `example_df` pull out the chrom, start, and end
		positions in `loci_df`. If `window` is provided, do not use the start and
		end in the file but, rather, extract a window centered on the middle of the
		provided coordinates.
		
		(2) Use the chrom directly, and add the start and end coordinates to the
		start of the extracted window from step (1).
	
	This function assumes by default that `example_df` is zero-indexed, in the 
	sense that the minimum start is 0. When doing the conversion, the start and
	end are simply added to the value in `loci_df`. This means that `loci_df` can
	be either zero- or one-indexed and the resulting df will match this indexing.
	If `example_df` is one-indexed, use the optional parameter.
	
	
	Parameters
	----------
	example_df: pandas.DataFrame
		A BED-formatted dataframe where the first three columns must be the example
		index, the start, and the end coordinates. It does not matter what these
		columns are named, and there can be additional columns past these.
	
	loci_df: pandas.DataFrame
		A BED-formatted dataframe where the first three columns must be the example
		index, the start, and the end coordinates. It does not matter what these
		columns are named, and there can be additional columns past these.
		
		NOTE: some processing steps may skip over elements in BED files and return
		a tensor with fewer elements than there are entries in the original BED
		file. Please check to make sure this has not happened, or everything will
		be misaligned.
	
	window: int or None, optional
		If None, use the start and end coordinates in `loci_df` directly. If an
		integer, these start and end coordinates will be replaced with a window
		centered on the middle of these original coordinates that spans this
		window length. This is useful when processing methods extracted windows
		of uniform length from a peak file, regardless of the length of that
		peak. Default is None.
		
	one_indexed: bool, optional
		Whether `example_df` is one-indexed. If it were one-indexed, the smallest
		start value would be 1 and so 1 would have to be subtracted from each
		correction. If False, assumes that the smallest start value is 0 and no
		change needs to be made. Default is False.
		
	
	Returns
	-------
	coords_df: pandas.DataFrame
		A dataframe of corrected coordinates where the first three columns are
		chrom, start, and end, and correspond to the same coordinate system as
		in `loci_df`. Put another way, if `loci_df` contains peaks on a reference
		genome, this returned file will now contain coordinates on that reference.
		The names of the first three columns will be replaced with the names of the
		first three columns in `loci_df`, and all other columns in the file will
		be left unchanged.
	"""
	
	loci_chroms = loci_df.iloc[:, 0].values
	loci_starts = loci_df.iloc[:, 1].values
	loci_ends = loci_df.iloc[:, 2].values
	
	if window is not None:
		mid = (loci_ends + loci_starts) // 2
		
		loci_starts = mid - window // 2
		loci_ends = mid + window // 2 + window % 2
	
	coords_ = example_df.iloc[:, :3].values
	idxs = coords_[:, 0]
	
	example_chroms = loci_chroms[idxs]
	example_starts = loci_starts[idxs] + coords_[:, 1] - one_indexed
	example_ends = loci_starts[idxs] + coords_[:, 2] - one_indexed
	
	names = loci_df.columns[:3]
	key_map = dict(zip(example_df.columns[:3], names))
	coords_df = example_df.copy()
	coords_df = coords_df.rename(key_map, axis='columns')
	coords_df[names[0]] = example_chroms
	coords_df[names[1]] = example_starts
	coords_df[names[2]] = example_ends
	return coords_df


def characters(pwm, alphabet=['A', 'C', 'G', 'T'], force=False, allow_N=False):
	"""Converts a PWM/one-hot encoding to a string sequence.

	This function takes in a PWM or one-hot encoding and converts it to the
	most likely sequence. When the input is a one-hot encoding, this is the
	opposite of the `one_hot_encoding` function.


	Parameters
	----------
	pwm: torch.tensor, shape=(len(alphabet), seq_len)
		A numeric representation of the sequence. This can be one-hot encoded
		or contain numeric values. These numerics can be probabilities but can
		also be frequencies.

	alphabet : set or tuple or list
		A pre-defined alphabet where the ordering of the symbols is the same
		as the index into the returned tensor. This is used to determine the
		letters in the returned sequence. Default is the DNA alphabet.

	force: bool, optional
		Whether to force a sequence to be produced even when there are ties.
		At each position that there is a tight, the character earlier in the
		sequence will be used. Default is False.
  
	allow_N: bool, optional
		Whether to allow the return of the character 'N' in the sequence, i.e.
		if pwm at a position is all 0's return N. Default is False.


	Returns
	-------
	seq: str
		A string where the length is the second dimension of PWM.
	"""
 
	#if (batch, alphabet_size, motif_size) and batch = 1, remove batch axis
	if len(pwm.shape) == 3 and pwm.shape[0] == 1:
		pwm = pwm[0]

	if len(pwm.shape) != 2:
		raise ValueError("PWM must have two dimensions where the " +
			"first dimension is the length of the alphabet and the second " +
			"dimension is the length of the sequence.")

	if pwm.shape[0] != len(alphabet):
		raise ValueError("PWM must have the same alphabet size as the " +
			"provided alphabet.")

	pwm_ismax = pwm == pwm.max(dim=0, keepdims=True).values
	if pwm_ismax.sum(axis=0).max() > 1 and force == False and allow_N == False:
		raise ValueError("At least one position in the PWM has multiple " +
			"letters with the same probability.")

	alphabet = numpy.array(alphabet)
	if isinstance(pwm, torch.Tensor):
		pwm = pwm.numpy(force=True)

	if allow_N:
		n_inds = numpy.where(pwm.sum(axis=0)==0)[0]
		dna_chars = alphabet[pwm.argmax(axis=0)]
		dna_chars[n_inds] = 'N'
	else:
		dna_chars = alphabet[pwm.argmax(axis=0)]
	
	return ''.join(dna_chars)


@numba.njit("void(int8[:, :], int8[:], int8[:])", cache=True)
def _fast_one_hot_encode(X_ohe, seq, mapping):
	"""An internal function for quickly converting bytes to one-hot indexes."""

	for i in range(len(seq)):
		idx = mapping[seq[i]]
		if idx == -1:
			continue

		if idx == -2:
			raise ValueError("Encountered character that is not in " + 
				"`alphabet` or in `ignore`.")
			
		X_ohe[i, idx] = 1


def one_hot_encode(sequence, alphabet=['A', 'C', 'G', 'T'], dtype=torch.int8, 
	ignore=['N'], desc=None, verbose=False, **kwargs):
	"""Converts a string or list of characters into a one-hot encoding.

	This function will take in either a string or a list and convert it into a
	one-hot encoding. If the input is a string, each character is assumed to be
	a different symbol, e.g. 'ACGT' is assumed to be a sequence of four 
	characters. If the input is a list, the elements can be any size.

	Although this function will be used here primarily to convert nucleotide
	sequences into one-hot encoding with an alphabet of size 4, in principle
	this function can be used for any types of sequences.

	Parameters
	----------
	sequence : str or list
		The sequence to convert to a one-hot encoding.

	alphabet : set or tuple or list
		A pre-defined alphabet where the ordering of the symbols is the same
		as the index into the returned tensor, i.e., for the alphabet ['A', 'B']
		the returned tensor will have a 1 at index 0 if the character was 'A'.
		Characters outside the alphabet are ignored and none of the indexes are
		set to 1. Default is ['A', 'C', 'G', 'T'].

	dtype : str or torch.dtype, optional
		The data type of the returned encoding. Default is int8.

	ignore: list, optional
		A list of characters to ignore in the sequence, meaning that no bits
		are set to 1 in the returned one-hot encoding. Put another way, the
		sum across characters is equal to 1 for all positions except those
		where the original sequence is in this list. Default is ['N'].


	Returns
	-------
	ohe : numpy.ndarray
		A binary matrix of shape (alphabet_size, sequence_length) where
		alphabet_size is the number of unique elements in the sequence and
		sequence_length is the length of the input sequence.
	"""

	for char in ignore:
		if char in alphabet:
			raise ValueError("Character {} in the alphabet ".format(char) + 
				"and also in the list of ignored characters.")

	if isinstance(alphabet, list):
		alphabet = ''.join(alphabet)

	ignore = ''.join(ignore)

	e = "utf8"
	seq_idxs = numpy.frombuffer(bytearray(sequence, e), dtype=numpy.int8)
	alpha_idxs = numpy.frombuffer(bytearray(alphabet, e), dtype=numpy.int8)
	ignore_idxs = numpy.frombuffer(bytearray(ignore, e), dtype=numpy.int8)

	one_hot_mapping = numpy.zeros(256, dtype=numpy.int8) - 2
	for i, idx in enumerate(alpha_idxs):
		one_hot_mapping[idx] = i

	for i, idx in enumerate(ignore_idxs):
		one_hot_mapping[idx] = -1


	n, m = len(sequence), len(alphabet)

	one_hot_encoding = numpy.zeros((n, m), dtype=numpy.int8)
	_fast_one_hot_encode(one_hot_encoding, seq_idxs, one_hot_mapping)
	return torch.from_numpy(one_hot_encoding).type(dtype).T


def reverse_complement(seq, complement_map={"A": "T", "C": "G", "G": "C", 
	"T": "A"}, allow_N=True):
	"""Return the reverse complement of a single sequence.

	This function will take in a single one-hot encoding of a sequence, or a
	single string, and return the reverse complement. If the input is a torch
	tensor, the encoding is simply flipped along both axes. If the input is
	a string, it is flipped and then each value is flipped according to the
	provided complement map. 

	Note that this function will not convert your sequence to upper-case or
	modify it in any other manner.

	
	Parameters
	----------
	seq: str or torch.Tensor w/ shape (alphabet_size, length)
		The sequence to be reverse complemented.
	
	complement_map: dict, optional
		The ordering and complement of each nucleotide. When the input is a
		string, this is used to directly convert characters. When the input is
		a torch tensor, the ordering of *keys* is assumed to be the order of 
		characters in the alphabet and the manner in which to flip depends on 
		it. Default is the nucleotide alphabet.

	allow_N: bool, optional
		Whether to allow N characters when doing the reverse complement. Only
		matters when reverse complementing strings. Default is True.
	
	Returns
	-------
	rev_comp: str or torch.Tensor w/ shape (alphabet_size, length)
		The reverse complemented string or 
	"""

	if isinstance(seq, str):
		seq_rc = []

		for char in seq:
			if char in complement_map:
				seq_rc.append(complement_map[char])
			elif char == 'N' and allow_N:
				seq_rc.append('N')
			else:
				raise ValueError("'{}' not in complement map".format(char))

		seq_rc = ''.join(reversed(seq_rc))

	elif isinstance(seq, torch.Tensor):

		chars = list(complement_map.keys())
		idxs = [chars.index(char) for char in complement_map.values()]

		seq_rc = torch.flip(seq, dims=(-1,))[idxs]

	return seq_rc


def random_one_hot(shape, probs=None, dtype='int8', random_state=None):
	"""Generate random one-hot encodings. Useful for debugging.

	This function will generate random one-hot encodings where the second to
	last dimension has a single element being a one and every other element
	being a zero. Primary used for debugging. 


	Parameters
	----------
	shape: tuple
		The shape of the 3D tensor to generate.

	probs: tuple, list, numpy.ndarray, or None optional
		A 2D array of probabilities where the first dimension is the batch size
		equal to the batch size of `X` and the second dimension is the alphabet
		size. The values should be the probability of that character occuring
		in that sequence. If a batch size of 1 is used when the batch size of
		X is greater than 1, the same probabilities are used for each sequence.
		The sum of probabilities across the alphabet axis must be equal to 1.
		If None, use a uniform distribution across the axis specified in shape.
		Default is [[0.25, 0.25, 0.25, 0.25]].

	dtype: str or numpy.dtype, optional
		The datatype to return the matrix as. Default is 'int8'.

	random_state: int or numpy.random.RandomState or None, optional
		The random state to use for generation. If None, do not use a 
		deterministic seed. Default is None.


	Returns
	-------
	ohe: torch.Tensor
		A tensor with the specified shape that is one-hot encoded.
	"""

	if not isinstance(shape, tuple) or len(shape) != 3:
		raise ValueError("Shape must be a tuple with 3 dimensions.")

	if not isinstance(random_state, numpy.random.RandomState):
		random_state = numpy.random.RandomState(random_state)

	if isinstance(probs, list):
		probs = numpy.array(probs)
		
	n = shape[1]
	ohe = numpy.zeros(shape, dtype=dtype)

	for i in range(ohe.shape[0]):
		if probs is None:
			probs_ = None
		elif probs.ndim == 1:
			probs_ = probs
		elif probs.shape[0] == 1:
			probs_ = probs[0]
		else:
			probs_ = probs[i] 

		choices = random_state.choice(n, size=shape[2], p=probs_)
		ohe[i, choices, numpy.arange(shape[2])] = 1 

	return torch.from_numpy(ohe)


def chunk(X, size=1024, overlap=0):
	"""Chunk a set of sequences into overlapping blocks.

	This function will take a set of sequences of variable length and will
	return a set of fixed-length blocks that tile the sequence with a fixed
	amount of overlap. This is useful when applying a method, such as FIMO or
	even a larger predictive model, to variable length sequences such as
	chromosomes.

	Unless the total sequence length is a multiple of the size, the final chunk
	produced from a sequence will be shorter than the size. In this case, the
	final chunk is excluded because it would not be an accurate representation
	of that sequence regardless.


	Parameters
	----------
	X: list of torch.Tensors
		A list of one-hot encoded sequences that each are of shape (4, -1).

	size: int, optional
		The size of the chunks to produce from the sequence. Default is 1024.

	overlap: int, optional
		The overlap between adjacent chunks. This is, essentially, the stride
		of the unrolling process. Default is 0.


	Returns
	-------
	y: torch.Tensor, shape=(-1, len(alphabet), size)
		A single tensor containing strides across the sequences. The chunks
		are concatenated together across examples such that the final chunk
		from the first item is followed by first chunk from the second item.
	"""

	if not isinstance(X, list):
		raise ValueError("X must be a list of tensors.")

	if not isinstance(size, int) or size <= 0:
		raise ValueError("size must be a positive integer.")

	if not isinstance(overlap, int) or overlap < 0:
		raise ValueError("overlap must be a non-negative integer.")

	return torch.cat([x.unfold(-1, size, size-overlap).permute(1, 0, 2) 
		for x in X], dim=0)


def unchunk(X, lengths=None, overlap=0):
	"""Unchunk fixed-length segments back into variable-length sequences.

	After chunking a set of variable length sequences into fixed-length chunks
	and applying some method on the chunks that results in some bp-resolution
	result, merge the chunks back into a tensor of the same length as the
	original sequence. When the final chunk was discarded due to the original
	sequence not being divisible by the chosen size, that portion will not
	be reconstructed by this method. 

	The overlap value should be the same as when the sequence was chunked, and
	should correspond to the number of positions that are shared across adjacent
	examples. When overlap is set to a value greater than 0, half of the overlap
	goes to the elements as follows:

		<------------*    |
				  overlap 
				|    *-----------> 


	Parameters
	----------
	X: list or numpy.ndarray or torch.tensor, shape=(-1, n_outputs, size)
		A set of fixed-length tensors for any number of outputs. Usually the
		result of applying `chunk` and then some form of model.

	lengths: list or numpy.ndarray or torch.tensor, shape=(-1,) or None
		The *original lengths* of the elements that were chunked. This is not
		the number of chunks produced, which will be influenced by the overlap,
		but the actual length in bp of the original sequences. If None, assume
		that all chunks in `X` come from a single variable-length sequence.
		Default is None.

	overlap: int, optional
		The number of bp overlap between adjacent chunks. Default is 0.


	Returns
	-------
	y: list of torch.Tensors or one torch.Tensor, shape=(n_outputs, -1)
		A list of variable-length tensors if `lengths` is provided, otherwise
		a single tensor.
	"""

	X = _cast_as_tensor(X)
	if X.ndim <= 2:
		raise ValueError("`X` must have at least three dimensions, with the "
			"last dimension corresponding to length.")

	lengths = _validate_input(_cast_as_tensor(lengths), "lengths", shape=(-1,), 
		min_value=0)

	size = X.shape[-1]
	lengths = (lengths - size) // (size - overlap) + 1
	lengths_csum = 0

	y = []
	for length in lengths:
		X_ = X[lengths_csum:lengths_csum+length]

		if overlap > 0:
			s = overlap // 2
			e = -(overlap - s)

			if X_.shape[0] == 1:
				X_ = X_[..., s:e].moveaxis(0, -2).reshape(*X_.shape[1:-1], -1)
			elif X_.shape[0] == 2:
				X_ = torch.cat([X_[0, ..., :e], X_[1, ..., s:]], dim=-1)
			else:
				X_ = torch.cat([X_[0, ..., :e],
								X_[1:-1, ..., s:e].moveaxis(0, -2).reshape(*X_.shape[1:-1], -1),
								X_[-1, ..., s:]], dim=-1)
		else:
			X_ = X_.moveaxis(0, -2).reshape(*X_.shape[1:-1], -1)

		y.append(X_)
		lengths_csum += length

	return y
	

def pwm_consensus(X):
	"""Take in a PWM and return the consensus.

	This function will take in a PWM that encodes the probabilities of each
	character at each position and will return the consensus, which is the
	most likely individual sequence according to that PWM. This is done by
	taking the argmax at each position. When multiple characters at the same
	position have the maximum probability, the character in the earlier
	numerical position is chosen. If a column has a sum equal to 0 (no
	characters allowed there) the returned consensus also has entirely 0s.


	Parameters
	----------
	X: torch.Tensor, numpy.ndarray, shape=(alphabet_len, pwm_len)
		A PWM containing the probabilities of each character at each position.


	Returns
	-------
	Y: torch.Tensor, shape=(alphabet_len, pwm_len)
		A PWM of the same shape as `X` except with the maximum-value character
		set to 1.
	"""

	X = _cast_as_tensor(X)
	_validate_input(X, "X", shape=(-1, -1), min_value=0, max_value=1)

	alpha_idxs = X.argmax(dim=0)

	Y = torch.zeros_like(X)
	Y[alpha_idxs, torch.arange(X.shape[-1])] = 1
	Y[:, X.sum(dim=0) == 0] = 0
	return Y


def extract_signal(loci, X, verbose=False):
	"""Extracts the signal at coordinates from a tensor of examples.

	This function takes in a dataframe with the first three columns being the
	example index, the start (inclusive) and the end (not inclusive) and
	returns the signal sum across those coordinates from the tensor. This can
	be used, for instance, to extract attributions or genomics signal from
	windows. This sum is done separately for each signal in the tensor.


	Parameters
	----------
	loci: pandas.DataFrame
		A set of loci to extract signal from. Multiple loci can be present on
		the same example in X and the starts and ends can overlap.

	X: torch.Tensor, numpy.ndarray, shape=(-1, n_signals, sequence_length)
		A 3D tensor where the first dimension corresponds to the examples, the
		second dimension corresponds to the number of signals being measured
		at each position (e.g., different ChIP-seq tracks), and the third
		position corresponds to the sequence length.

	verbose: bool, optional
		Whether to print a progress bar tracking the extraction process. Default
		is False.


	Returns
	-------
	Y: torch.Tensor, shape=(n_loci, n_signals)
		The sum of the signal across all positions for each of the loci. The
		returned value is the sum of this signal.
	"""

	_validate_input(X, "X", shape=(-1, -1, -1))

	Y = torch.zeros(loci.shape[0], X.shape[1], dtype=X.dtype, device=X.device)

	loci = loci.values[:, :3].astype(int)
	for i, (idx, start, end) in enumerate(tqdm(loci, disable=not verbose)):
		Y[i] = X[idx, :, start:end].sum(dim=-1)

	return Y
