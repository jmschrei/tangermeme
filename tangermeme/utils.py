# ablate.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import numba
import torch

from tqdm import tqdm

def _validate_input(X, name, shape=None, dtype=None, min_value=None,
	max_value=None, ohe=False, ohe_dim=1,allow_N=False):
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
	"""

	if not isinstance(X, torch.Tensor):
		raise ValueError("{} must be a torch.Tensor object".format(name))

	if shape is not None:
		if len(shape) != len(X.shape):
			raise ValueError("{} must have shape {}".format(name, shape))

		for i in range(len(shape)):
			if shape[i] != -1 and shape[i] != X.shape[i]:
				raise ValueError("{} must have shape {}".format(name, shape))


	if dtype is not None and X.dtype != dtype:
		raise ValueError("{} must have dtype {}".format(name, dtype))

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

		if ((not (X.sum(axis=1) == 1).all()) and (not allow_N)
          ) or not ((allow_N and ((X.sum(axis=ohe_dim) == 1) | (X.sum(axis=ohe_dim) == 0)).all())):
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

	dtype: torch.dtype
		A torch dtype to cast the values in the torch.tensor to


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
		n_inds = np.where(pwm.sum(axis=0)==0)[0]
		dna_chars = alphabet[pwm.argmax(axis=0)]
		dna_chars[n_inds] = 'N'
	else:
		dna_chars = alphabet[pwm.argmax(axis=0)]
    
	return ''.join(dna_chars)


@numba.njit("void(int8[:, :], int8[:], int8[:])")
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

	seq_idxs = numpy.frombuffer(bytearray(sequence, "utf8"), dtype=numpy.int8)
	alpha_idxs = numpy.frombuffer(bytearray(alphabet, "utf8"), dtype=numpy.int8)
	ignore_idxs = numpy.frombuffer(bytearray(ignore, "utf8"), dtype=numpy.int8)

	one_hot_mapping = numpy.zeros(256, dtype=numpy.int8) - 2
	for i, idx in enumerate(alpha_idxs):
		one_hot_mapping[idx] = i

	for i, idx in enumerate(ignore_idxs):
		one_hot_mapping[idx] = -1


	n, m = len(sequence), len(alphabet)

	one_hot_encoding = numpy.zeros((n, m), dtype=numpy.int8)
	_fast_one_hot_encode(one_hot_encoding, seq_idxs, one_hot_mapping)
	return torch.from_numpy(one_hot_encoding).type(dtype).T


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

	n = shape[1]
	ohe = numpy.zeros(shape, dtype=dtype)

	for i in range(ohe.shape[0]):
		if probs is None:
			probs_ = None
		elif probs.shape[0] == 1:
			probs_ = probs[0]
		else:
			probs_ = probs[i] 

		choices = random_state.choice(n, size=shape[2], p=probs_)
		ohe[i, choices, numpy.arange(shape[2])] = 1 

	return torch.from_numpy(ohe)
