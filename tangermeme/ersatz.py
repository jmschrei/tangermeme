# ersatz.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numba
import numpy
import torch

from tqdm import tqdm

from .utils import _accepts_packed
from .utils import _cat_packed
from .utils import _numba_threads
from .utils import _random_codes
from .utils import _stack_packed
from .utils import _pack_row
from .utils import _validate_input
from .utils import one_hot_encode
from .utils import PackedSequence
from .utils import random_one_hot
from .utils import TangermemeWarning


# The positions decoded at a time by the packed paths below, 16 MB of
# character indices.
_CHUNK_POSITIONS = 2 ** 24


def _codes_of(X, rows):
	"""The character index of every position of sequences `rows` of X, a
	PackedSequence or a one-hot encoding of four characters, as uint8 with 4
	for an unknown character."""

	if isinstance(X, PackedSequence):
		return X[rows].to_codes()

	values, codes = X[rows].max(dim=1)
	codes[values == 0] = 4
	return codes.to(torch.uint8).cpu()


def _map_rows(X, fn, length, alphabet):
	"""Apply `fn(codes, rows)`, which returns the character indices of the
	outputs of sequences `rows` of X given theirs, a chunk of sequences at a
	time, and pack the outputs, each `length` long, into a PackedSequence."""

	step = max(1, _CHUNK_POSITIONS // max(1, X.shape[-1]))
	parts = [PackedSequence.from_codes(fn(_codes_of(X, rows), rows), alphabet)
		for rows in (slice(i, i + step) for i in range(0, X.shape[0], step))]

	if len(parts) == 0:
		return PackedSequence(torch.zeros((0, (length + 3) // 4),
			dtype=torch.uint8), length, alphabet=alphabet)

	return _cat_packed(parts)


def _motif_codes(motif, rows, n):
	"""The character indices of the motifs of sequences `rows`, whether
	`motif` holds one motif or one per sequence, with 4 for all-zero
	columns."""

	values, codes = motif.max(dim=1)
	codes[values == 0] = 4
	codes = codes.to(torch.uint8).cpu()
	return codes.expand(n, -1)[rows] if codes.shape[0] == 1 else codes[rows]


def _packed_output(X, return_packed, alphabet):
	"""Resolve `return_packed` for X and the alphabet a packed output has,
	raising when a one-hot X does not have four characters to pack."""

	packed = isinstance(X, PackedSequence)
	if return_packed is None:
		return_packed = packed

	if return_packed and not packed and X.shape[1] != 4:
		raise ValueError("return_packed=True needs an alphabet of four "
			"characters, but X has {}.".format(X.shape[1]))

	return return_packed, X.alphabet if packed else tuple(alphabet)


def insert(
	X: torch.Tensor | PackedSequence | str,
	motif: torch.Tensor | str,
	start: int | None = None,
	alphabet: list[str] = list("ACGT"),
	return_packed: bool | None = None,
) -> torch.Tensor | PackedSequence:
	"""Insert a motif into a set of sequences at a defined position.

	This function will take in a tensor of one-hot encoded sequences or a string
	that can be one-hot encoded and insert the motif into the defined 
	position. It will then return a copy of the data with the insertion, 
	leaving the original data unperturbed.

	Importantly, an *insertion* means that the entire original sequence is still
	present, albeit in two halves with the inserted motif in the middle.
	Specifically, if we have an original sequence AAAAAACCCCAAAAAA and want to
	insert GGGG in the middle, the `insert` function will return something
	corresponding to AAAAAACCGGGGCCAAAAAA. Hence, the returned sequence will be
	longer than the original sequence.

	If the motif is a string, it will be one-hot encoded according to the
	alphabet that is provided. If a motif with batch size of 1 is provided, 
	the same motif will be inserted into all sequences. If a motif with a 
	batch size equal to that of X is provided, there will be 1-1 correspondence
	between the motifs and the sequence, i.e., the motif at index 5 will be
	substituted into the sequence at index 5.


	Parameters
	----------
	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to have a motif inserted into.
		Unknown characters are allowed and must be encoded as all-zero
		columns; they are carried through into the returned sequences.

	motif: torch.tensor, shape=(-1, len(alphabet), motif_length)
		A one-hot encoded version of a short motif to insert into the set of
		sequences.

	start: int or None, optional
		The starting position of where to insert the motif. If None,
		insert the motif into the middle of the sequence such that the
		middle of the motif occurs at the middle of the sequence. Default is
		None.

	alphabet: list or tuple, optional
		A pre-defined alphabet where the ordering of the symbols is the same
		as the index into the returned tensor, i.e., for the alphabet ['A', 'B']
		the returned tensor will have a 1 at index 0 if the character was 'A'.
		Every character of a string motif must be in the alphabet, and any
		other character, including N, raises a ValueError. This is not
		necessary or used if a one-hot encoded tensor is provided for the
		motif. Default is ['A', 'C', 'G', 'T'].

	return_packed: bool or None, optional
		Whether to return the sequences as a PackedSequence rather than a
		one-hot encoding, which needs an alphabet of four characters. `X` can
		be a PackedSequence with one batch axis, which is decoded and packed
		a chunk of sequences at a time. If None, return a PackedSequence when
		`X` is one, and with False, a PackedSequence `X` gives an int8 one-hot
		encoding. Default is None.


	Returns
	-------
	Y: torch.tensor, shape=(-1, len(alphabet), length + motif_length), or PackedSequence
		A one-hot encoded set of sequences that each have the motif inserted
		at the same position.
	"""

	if isinstance(motif, str):
		motif = one_hot_encode(motif, alphabet=alphabet).unsqueeze(0)

	_validate_input(X, "X", ohe=True, ohe_dim=1, allow_N=True)
	_validate_input(motif, "motif", shape=(-1, X.shape[1], -1), ohe=True)

	if start is not None:
		if start < 0 or start > (X.shape[-1] - motif.shape[-1]):
			raise ValueError("Provided start falls off the end of the sequence")
	else:
		start = X.shape[-1] // 2

	return_packed, packed_alphabet = _packed_output(X, return_packed, alphabet)
	if return_packed or isinstance(X, PackedSequence):
		Y = _map_rows(X, lambda codes, rows: torch.cat([codes[:, :start],
			_motif_codes(motif, rows, X.shape[0]), codes[:, start:]], dim=1),
			X.shape[-1] + motif.shape[-1], packed_alphabet)
		return Y if return_packed else Y.to_one_hot()

	if motif.shape[0] == 1:
		motif = motif.repeat(X.shape[0], 1, 1)

	return torch.cat([X[:, :, :start], motif, X[:, :, start:]], dim=-1)


def substitute(
	X: torch.Tensor | PackedSequence | str,
	motif: torch.Tensor | str,
	start: int | None = None,
	alphabet: list[str] = list("ACGT"),
	ignore: list[str] = ["N"],
	return_packed: bool | None = None,
) -> torch.Tensor | PackedSequence:
	"""Substitute a motif into a set of sequences at a defined position.

	This function will take in a tensor of one-hot encoded sequences or a string
	that can be one-hot encoded and will substitute a motif at a defined 
	position. It will then return a copy of the data with the substitution, 
	leaving the original data unperturbed.

	Importantly, a *substitution* means that part of the original sequence will
	be missing. Specifically, if we have an original sequence AAAAAACCCCAAAAAA 
	and want to substitute a GGGG in the middle, the `substitute` function will 
	return something corresponding to AAAAAAGGGGAAAAAA. Note the missing Cs. 
	Hence, the returned sequence will be the same length as the original 
	sequence.

	If the motif is a string, it will be one-hot encoded according to the
	alphabet that is provided. If a motif with batch size of 1 is provided, 
	the same motif will be substituted into all sequences. If a motif with a 
	batch size equal to that of X is provided, there will be 1-1 correspondence
	between the motifs and the sequence, i.e., the motif at index 5 will be
	substituted into the sequence at index 5.

	Finally, if all-zeros positions are present in a motif -- or if a string
	motif is passed in and some characters are present in the `ignore` list --
	the original sequence will be kept. For instance, passing in the motif
	"ACNNGT" with the default ignore list will update each sequence to have the
	dinucleotide `AC` followed by the original sequence followed by `GT`.


	Parameters
	----------
	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to have a motif substituted into.
		Unknown characters are allowed and must be encoded as all-zero
		columns, as `one_hot_encode` does for the characters in its `ignore`
		list. Those columns are left as-is unless the motif covers them.

	motif: torch.tensor, shape=(-1, len(alphabet), motif_length)
		A one-hot encoded version of a short motif to substitute into the set of
		sequences.

	start: int or None, optional
		The starting position of where to substitute the motif. If None,
		substitute the motif into the middle of the sequence such that the 
		middle of the motif occurs at the middle of the sequence. Default is 
		None.

	alphabet: list or tuple, optional
		A pre-defined alphabet where the ordering of the symbols is the same
		as the index into the returned tensor, i.e., for the alphabet ['A', 'B']
		the returned tensor will have a 1 at index 0 if the character was 'A'.
		A character of a string motif that is in neither `alphabet` nor
		`ignore` raises a ValueError. This is not necessary or used if a
		one-hot encoded tensor is provided for the motif. Default is
		['A', 'C', 'G', 'T'].

	ignore: set or tuple or list, optional
		A set of characters indicating that the original value of the sequence
		should be maintained at this position. This is only relevant when a
		string is provided for the motif and causes an all-zeros column to be
		added to the one-hot encoding of the motif. Default is ["N"].

	return_packed: bool or None, optional
		Whether to return the sequences as a PackedSequence rather than a
		one-hot encoding, which needs an alphabet of four characters. `X` can
		be a PackedSequence with one batch axis, which is decoded and packed
		a chunk of sequences at a time. If None, return a PackedSequence when
		`X` is one, and with False, a PackedSequence `X` gives an int8 one-hot
		encoding. Default is None.


	Returns
	-------
	Y: torch.tensor, shape=(-1, len(alphabet), length), or PackedSequence
		A one-hot encoded set of sequences that each have the motif substituted
		at the same position.
	"""

	if isinstance(motif, str):
		motif = one_hot_encode(motif, alphabet=alphabet, ignore=ignore)
		motif = motif.unsqueeze(0)

	_validate_input(X, "X", ohe=True, allow_N=True)
	_validate_input(motif, "motif", shape=(-1, X.shape[1], -1), ohe=True,
		allow_N=True)

	if motif.shape[-1] > X.shape[-1]:
		raise ValueError("Motif cannot be longer than sequence.")

	if start is not None:
		if start < 0 or start > (X.shape[-1] - motif.shape[-1]):
			raise ValueError("Provided start falls off the end of the sequence")
	else:
		start = X.shape[-1] // 2 - motif.shape[-1] // 2

	return_packed, packed_alphabet = _packed_output(X, return_packed, alphabet)
	if return_packed or isinstance(X, PackedSequence):
		end = start + motif.shape[-1]

		def substitute_codes(codes, rows):
			motif_codes = _motif_codes(motif, rows, X.shape[0])
			codes[:, start:end] = torch.where(motif_codes == 4,
				codes[:, start:end], motif_codes)
			return codes

		Y = _map_rows(X, substitute_codes, X.shape[-1], packed_alphabet)
		return Y if return_packed else Y.to_one_hot()

	X = torch.clone(X)
	for i in range(motif.shape[-1]):
		idxs = motif[:, :, i].sum(dim=1) == 1

		if idxs.all():
			X[:, :, start+i] = motif[:, :, i]
		elif idxs.sum() > 0:
			X[idxs, :, start+i] = motif[idxs, :, i]

	return X


def multisubstitute(
	X: torch.Tensor | PackedSequence,
	motifs: list[torch.Tensor | str],
	spacing: list[int],
	start: int | None = None,
	alphabet: list[str] = list("ACGT"),
	ignore: list[str] = ["N"],
	return_packed: bool | None = None,
) -> torch.Tensor | PackedSequence:
	"""Substitute a set of motifs into sequences with provided spacings.

	This function will take in a list of tensors of one-hot encoded sequences
	or of strings that can be one-hot encoded and will substitute the motifs
	into the sequences given the provided spacings. It will then return a copy
	of the data with the substitutions, leaving the original data unperturbed.

	This function is largely just a wrapper around the substitute function,
	calling it multiple times and figuring out the exact positioning internally.

	If the motif is a string, it will be one-hot encoded according to the
	alphabet that is provided. If a motif with batch size of 1 is provided, 
	the same motifs will be substituted into all sequences. If a motif with a 
	batch size equal to that of X is provided, there will be 1-1 correspondence
	between the motifs and the sequence, i.e., the motif at index 5 will be
	substituted into the sequence at index 5.


	Parameters
	----------
	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to have a motif substituted into.

	motifs: list of torch.tensor, shape=(-1, len(alphabet), motif_length)
		A list of strings or of one-hot encoded versions of short motifs to
		substitute into the set of sequences.

	spacing: list or int
		An integer specifying a constant spacing between all motifs or a list
		of spacings of length equal to n-1 where n is the number of motifs. If
		a list is provided, the $i$-th entry should be interpreted as the
		distance after the $i$-th motif that the $i+1$-th motif begins.

	start: int or None, optional
		The starting position of where to substitute the motifs. If None, the
		full motif arrangement is centered such that its midpoint coincides
		with the middle of the sequence. Default is None.

	alphabet : list or tuple, optional
		A pre-defined alphabet where the ordering of the symbols is the same
		as the index into the returned tensor, i.e., for the alphabet ['A', 'B']
		the returned tensor will have a 1 at index 0 if the character was 'A'.
		A character of a string motif that is in neither `alphabet` nor
		`ignore` raises a ValueError. This is not necessary or used if a
		one-hot encoded tensor is provided for the motif. Default is
		['A', 'C', 'G', 'T'].

	ignore: set or tuple or list, optional
		A set of characters indicating that the original value of the sequence
		should be maintained at this position. This is only relevant when a
		string is provided for the motif and causes an all-zeros column to be
		added to the one-hot encoding of the motif. Default is ["N"].

	return_packed: bool or None, optional
		Whether to return the sequences as a PackedSequence rather than a
		one-hot encoding, which needs an alphabet of four characters. `X` can
		be a PackedSequence with one batch axis, which is decoded and packed
		a chunk of sequences at a time. If None, return a PackedSequence when
		`X` is one, and with False, a PackedSequence `X` gives an int8 one-hot
		encoding. Default is None.


	Returns
	-------
	Y: torch.tensor, shape=(-1, len(alphabet), length), or PackedSequence
		A one-hot encoded set of sequences that each have the motifs substituted
		at the correct positions.
	"""

	motif_lengths = [len(motif) if isinstance(motif, str) else motif.shape[-1] 
		for motif in motifs]

	if not isinstance(spacing, (int, list)):
		raise ValueError("Spacings must be an integer or a list of integers.")

	if isinstance(spacing, int):
		spacing = [spacing for _ in range(len(motifs) - 1)]

	if len(spacing) != (len(motifs) - 1):
		raise ValueError("Must provide n-1 spacings for n motifs.")

	for l in spacing:
		if l < 0 or l >= X.shape[-1]:
			raise ValueError("Spacing cannot be smaller than zero or " +
				"larger than the sequence being inserted into.")

	if start is None:
		n = sum(spacing) + sum(motif_lengths)
		start = X.shape[-1] // 2 - n // 2

		if start < 0:
			raise ValueError("Sum of motif lengths and spacing cannot be " + 
				"larger than the sequence being inserted into.")

	# Packed once, the sequences stay packed through every substitution.
	packed = isinstance(X, PackedSequence) or bool(return_packed)
	for i in range(len(spacing)):
		X = substitute(X, motifs[i], start=start, alphabet=alphabet, 
			ignore=ignore, return_packed=packed or None)
		start += motif_lengths[i] + spacing[i]

	X = substitute(X, motifs[-1], start=start, alphabet=alphabet, ignore=ignore,
		return_packed=packed or None)
	return X.to_one_hot() if packed and return_packed is False else X


def delete(
	X: torch.Tensor | PackedSequence,
	start: int,
	end: int,
	return_packed: bool | None = None,
) -> torch.Tensor | PackedSequence:
	"""Delete a portion of a sequence.

	This function will take in a tensor of one-hot encoded sequences and a pair
	of numbers representing the start and the end of the portion to remove, and
	will return a tensor that is missing those positions. Essentially, those
	positions get snipped out from the tensor.

	The sequence returned from this function will be shorter than the original
	sequence. Simply, it will be as if X[:, :, start:end] were removed from the
	sequence.


	Parameters
	----------
	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to have a portion deleted from.
		Unknown characters are allowed and must be encoded as all-zero
		columns; those outside the deleted portion are kept.

	start: int
		The starting position to remove, inclusive.

	end: int
		The final position to remove, not inclusive.

	return_packed: bool or None, optional
		Whether to return the sequences as a PackedSequence rather than a
		one-hot encoding, which needs an alphabet of four characters. `X` can
		be a PackedSequence with one batch axis, which is decoded and packed
		a chunk of sequences at a time. If None, return a PackedSequence when
		`X` is one, and with False, a PackedSequence `X` gives an int8 one-hot
		encoding. Default is None.


	Returns
	-------
	Y: torch.tensor, shape=(-1, len(alphabet), length-(end-start)), or PackedSequence
		A one-hot encoded set of sequences that each have a portion deleted.
	"""

	if start < 0 or start > X.shape[-1]:
		raise ValueError("Start cannot be below zero or greater than " + 
			"the length of the sequence.")

	if end < 0 or end > X.shape[-1] or end <= start:
		raise ValueError("End must come after start, must be greater " +
			"than zero, and cannot be greater than the length of the sequence.")

	_validate_input(X, "X", ohe=True, ohe_dim=1, allow_N=True)

	return_packed, packed_alphabet = _packed_output(X, return_packed, "ACGT")
	if return_packed or isinstance(X, PackedSequence):
		Y = _map_rows(X, lambda codes, rows: torch.cat([codes[:, :start],
			codes[:, end:]], dim=1), X.shape[-1] - (end - start),
			packed_alphabet)
		return Y if return_packed else Y.to_one_hot()

	return torch.cat([X[:, :, :start], X[:, :, end:]], dim=-1)


def randomize(
	X: torch.Tensor | PackedSequence,
	start: int,
	end: int,
	probs: list | tuple | numpy.ndarray | torch.Tensor = [[0.25, 0.25, 0.25, 0.25]],
	n: int = 1,
	random_state: int | numpy.random.RandomState | None = None,
	return_packed: bool | None = None,
) -> torch.Tensor | PackedSequence:
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
		Unknown characters are allowed and must be encoded as all-zero
		columns. Those inside the randomized portion are replaced along with
		everything else, so the returned sequences only keep the ones outside
		it.

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

	return_packed: bool or None, optional
		Whether to return the sequences as a PackedSequence rather than a
		one-hot encoding, which needs an alphabet of four characters. `X` can
		be a PackedSequence with one batch axis, which is decoded and packed
		a chunk of sequences at a time. If None, return a PackedSequence when
		`X` is one, and with False, a PackedSequence `X` gives an int8 one-hot
		encoding. Default is None.


	Returns
	-------
	X_rands: torch.tensor, shape=(-1, n, len(alphabet), length), or PackedSequence
		A one-hot encoded set of sequences that each have a randomized
		substitution.
	"""

	if not isinstance(probs, torch.Tensor):
		probs = torch.tensor(probs)

	if not isinstance(random_state, numpy.random.RandomState):
		random_state = numpy.random.RandomState(random_state)

	_validate_input(X, "X", ohe=True, allow_N=True)
	_validate_input(probs, "Probs", shape=(-1, -1), min_value=0, max_value=1)

	if end <= start:
		raise ValueError("End must come after start.")

	if end > X.shape[-1] or start < 0:
		raise ValueError("Start or end are falling off the edge of X.")

	return_packed, packed_alphabet = _packed_output(X, return_packed, "ACGT")
	if return_packed or isinstance(X, PackedSequence):
		X_rands = []
		for i in range(n):
			# The same draws, in the same order, as random_one_hot below.
			codes_i = torch.from_numpy(_random_codes((X.shape[0], probs.shape[1],
				end-start), probs, random_state)).to(torch.uint8)

			def randomize_codes(codes, rows, codes_i=codes_i):
				codes[:, start:end] = codes_i[rows]
				return codes

			X_rands.append(_map_rows(X, randomize_codes, X.shape[-1],
				packed_alphabet))

		Y = _stack_packed(X_rands)
		return Y if return_packed else Y.to_one_hot()

	X_rands = []
	for i in range(n):
		substitute_ohe = random_one_hot((X.shape[0], probs.shape[1], end-start), 
			probs=probs, random_state=random_state)

		X_rand = substitute(X, substitute_ohe, start=start)
		X_rands.append(X_rand)

	return torch.stack(X_rands).permute(1, 0, 2, 3)


@_accepts_packed
def shuffle(
	X: torch.Tensor | PackedSequence,
	start: int = 0,
	end: int = -1,
	n: int = 1,
	random_state: int | numpy.random.RandomState | None = None,
	return_packed: bool | None = None,
) -> torch.Tensor | PackedSequence:
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
		Unknown characters are allowed and must be encoded as all-zero
		columns. The shuffle is a permutation of the columns in the region,
		so unknown characters inside it are moved rather than resolved and
		the number of them is preserved.

	start: int, optional
		The starting position of where to randomize the sequence, inclusive.
		Default is 0, shuffling the entire sequence.

	end: int, optional
		The ending position of where to randomize the sequence. If end is
		positive then it is non-inclusive, but if end is negative then it is
		inclusive. Default is -1, shuffling the entire sequence.

	n: int, optional
		The number of times to shuffle that region. Default is 1.

	random_state: int, numpy.random.RandomState, or None, optional
		Whether to use a specific random seed when generating the shuffle to
		ensure reproducibility. If None, do not use a reproducible seed. Default 
		is None.

	return_packed: bool or None, optional
		Whether to return the sequences as a PackedSequence rather than a
		one-hot encoding, which needs an alphabet of four characters. `X` can
		be a PackedSequence with one batch axis, which is decoded and packed
		a chunk of sequences at a time. If None, return a PackedSequence when
		`X` is one, and with False, a PackedSequence `X` gives an int8 one-hot
		encoding. Default is None.


	Returns
	-------
	Y: torch.tensor, shape=(-1, n, len(alphabet), length), or PackedSequence
		A one-hot encoded set of sequences that each have a shuffled portion.
	"""

	_validate_input(X, "X", ohe=True, allow_N=True)

	if end < 0:
		end = X.shape[-1] + 1 + end

	if end <= start:
		raise ValueError("End must come after start.")

	if end > X.shape[-1] or start < 0:
		raise ValueError("Start or end are falling off the edge of X.")

	if not isinstance(random_state, numpy.random.RandomState):
		random_state = numpy.random.RandomState(random_state)

	return_packed, packed_alphabet = _packed_output(X, return_packed, "ACGT")
	if return_packed or isinstance(X, PackedSequence):
		X_shufs = []
		for i in range(n):
			idxs = numpy.arange(end-start)
			random_state.shuffle(idxs)
			idxs = torch.from_numpy(idxs)

			def shuffle_codes(codes, rows, idxs=idxs):
				codes[:, start:end] = codes[:, start:end][:, idxs]
				return codes

			X_shufs.append(_map_rows(X, shuffle_codes, X.shape[-1],
				packed_alphabet))

		Y = _stack_packed(X_shufs)
		return Y if return_packed else Y.to_one_hot()

	X_shufs = []
	for i in range(n):
		idxs = numpy.arange(end-start)
		random_state.shuffle(idxs)

		X_ = torch.clone(X)
		X_[:, :, start:end] = X[:, :, start:end][:, :, idxs]
		X_shufs.append(X_)

	return torch.stack(X_shufs).permute(1, 0, 2, 3)


@numba.njit(nogil=True, cache=True)
def _shuffle_sequence(idxs, n_chars, n_shuffles, random_state, shuffled):
	"""Shuffle the character indices `idxs` of one sequence `n_shuffles`
	times into the rows of `shuffled`, after seeding the generator with
	`random_state`."""

	seq_len = idxs.shape[0]
	if seq_len == 0:
		return

	numpy.random.seed(random_state)

	next_idxs = numpy.zeros((n_chars, seq_len), dtype=numpy.int32)
	next_idxs_counts = numpy.zeros(n_chars, dtype=numpy.int32)
	for j in range(seq_len - 1):
		char = idxs[j]
		next_idxs[char, next_idxs_counts[char]] = j + 1
		next_idxs_counts[char] += 1

	for i in range(n_shuffles):
		for char in range(n_chars):
			n = next_idxs_counts[char]

			next_idxs_ = numpy.arange(n)
			next_idxs_[:-1] = numpy.random.permutation(n-1)  # Keep last index
			next_idxs[char, :n] = next_idxs[char, :n][next_idxs_]

		counters = numpy.zeros(n_chars, dtype=numpy.int32)
		idx = 0
		shuffled[i, 0] = idxs[idx]
		for j in range(1, seq_len):
			char = idxs[idx]
			idx = next_idxs[char, counters[char]]

			counters[char] += 1
			shuffled[i, j] = idxs[idx]


@numba.njit(nogil=True, cache=True)
def _fast_shuffle(idxs, n_chars, n_shuffles, random_states, shuffled):
	"""An internal function for fast dinucleotide shuffling using numba.

	Each row of `idxs` holds the character indices of one sequence. The row
	is shuffled `n_shuffles` times into `shuffled` after the generator is
	seeded with its own entry of `random_states`, so the shuffles of a row
	do not depend on the other rows in the call. The GIL is released.
	"""

	for k in range(idxs.shape[0]):
		_shuffle_sequence(idxs[k], n_chars, n_shuffles, random_states[k],
			shuffled[k])


@numba.njit(nogil=True, parallel=True, cache=True)
def _fast_shuffle_parallel(idxs, n_chars, n_shuffles, random_states,
	shuffled):
	"""`_fast_shuffle` with the rows split across numba's threads. Each row
	seeds the generator of the thread it runs on, so the shuffles are those
	of `_fast_shuffle` for any number of threads."""

	for k in numba.prange(idxs.shape[0]):
		_shuffle_sequence(idxs[k], n_chars, n_shuffles, random_states[k],
			shuffled[k])


@numba.njit(nogil=True, cache=True)
def _shuffle_sequence_packed(codes, start, end, n_shuffles, random_state,
	data, run_starts, run_ends, flags):
	"""Shuffle positions `start` to `end` of one sequence of character
	indices, `codes`, with 4 for an unknown character, `n_shuffles` times,
	and pack each shuffled sequence into a row of `data`.

	`run_starts` and `run_ends` hold the same number of runs of unknown
	characters, m, for each shuffle, which is the number in `codes` since
	every shuffle keeps it: those of shuffle i are written to positions i * m
	to (i + 1) * m. `flags` receives whether any position between the ends
	of the region is the same in every shuffle, whether all of them are, and
	whether a shuffle had a number of runs other than m, in which case the
	rest of its runs are not written.
	"""

	seq_len = codes.shape[0]
	region_len = end - start

	shuffled = numpy.empty((n_shuffles, region_len), dtype=numpy.uint8)
	_shuffle_sequence(codes[start:end], 5, n_shuffles, random_state, shuffled)

	any_conserved, all_conserved = False, True
	for j in range(1, region_len - 1):
		conserved = True
		for i in range(1, n_shuffles):
			if shuffled[i, j] != shuffled[0, j]:
				conserved = False
				break

		any_conserved = any_conserved or conserved
		all_conserved = all_conserved and conserved

	n_runs = run_starts.shape[0] // n_shuffles if n_shuffles > 0 else 0

	row = codes.copy()
	wrong_runs = False
	for i in range(n_shuffles):
		for j in range(region_len):
			row[start + j] = shuffled[i, j]

		r, r_end = i * n_runs, (i + 1) * n_runs
		byte = 0
		for j in range(seq_len):
			# An unknown character, 4, is stored as 4 & 3 = 0.
			byte = (byte << 2) | (row[j] & 3)
			if j % 4 == 3:
				data[i, j // 4] = byte
				byte = 0

			if row[j] == 4 and (j == 0 or row[j-1] != 4):
				if r == r_end:
					wrong_runs = True
					break

				run_starts[r] = j

			if row[j] == 4 and (j == seq_len - 1 or row[j+1] != 4):
				run_ends[r] = j + 1
				r += 1

		if seq_len % 4 != 0:
			data[i, seq_len // 4] = byte << (2 * (4 - seq_len % 4))

		wrong_runs = wrong_runs or r != r_end

	flags[0] = any_conserved
	flags[1] = all_conserved
	flags[2] = wrong_runs


@numba.njit(nogil=True, cache=True)
def _fast_shuffle_packed(codes, start, end, n_shuffles, random_states,
	run_ptr, data, run_starts, run_ends, flags):
	"""Run `_shuffle_sequence_packed` on each row of `codes`, writing the
	runs of row k from `n_shuffles * run_ptr[k]`."""

	for k in range(codes.shape[0]):
		r0, r1 = n_shuffles * run_ptr[k], n_shuffles * run_ptr[k+1]
		_shuffle_sequence_packed(codes[k], start, end, n_shuffles,
			random_states[k], data[k], run_starts[r0:r1], run_ends[r0:r1],
			flags[k])


@numba.njit(nogil=True, parallel=True, cache=True)
def _fast_shuffle_packed_parallel(codes, start, end, n_shuffles,
	random_states, run_ptr, data, run_starts, run_ends, flags):
	"""`_fast_shuffle_packed` with the rows split across numba's threads,
	with the same shuffles for any number of threads."""

	for k in numba.prange(codes.shape[0]):
		r0, r1 = n_shuffles * run_ptr[k], n_shuffles * run_ptr[k+1]
		_shuffle_sequence_packed(codes[k], start, end, n_shuffles,
			random_states[k], data[k], run_starts[r0:r1], run_ends[r0:r1],
			flags[k])


def _random_states(random_state, n):
	"""The seed of each of `n` sequences: `random_state + i` for sequence i
	when `random_state` is an integer, a new random base seed when it is
	None, and its own entry when it holds one seed per sequence."""

	if random_state is None:
		random_state = numpy.random.randint(0, 9999999)

	random_states = numpy.asarray(random_state)
	if not numpy.issubdtype(random_states.dtype, numpy.integer):
		raise ValueError("random_state must be None, an integer, or one "
			"integer per sequence.")

	if random_states.ndim == 0:
		random_states = random_states + numpy.arange(n)
	elif random_states.shape != (n,):
		raise ValueError("random_state must hold one seed per sequence, {}, "
			"but has shape {}.".format(n, random_states.shape))

	# The kernel seeds with 32-bit integers, and a larger seed wraps around,
	# as it did when the kernel took a single int32 seed.
	return random_states.astype(numpy.int64).astype(numpy.int32)


def _check_conserved(any_conserved, all_conserved, n_shuffles, verbose,
	stacklevel):
	"""Given, for each sequence, whether any position between the ends of its
	region is the same in all of its shuffles and whether all of them are,
	warn when `verbose` if any is, and raise if there is more than one
	shuffle and all of them are for some sequence, whose shuffles are then
	identical. `stacklevel` points the warning at the line that called
	dinucleotide_shuffle."""

	if verbose and any_conserved.any():
		warnings.warn("At least one position in dinucleotide shuffle is "
			"identical across all positions.", TangermemeWarning,
			stacklevel=stacklevel)

	if n_shuffles > 1 and all_conserved.any():
		raise ValueError("All dinucleotide shuffles yield identical " +
			"sequences, potentially due to a lack of diversity in sequence.")


# A call that shuffles fewer positions than this in all, the sequences times
# the shuffles times the length of the region, runs on one thread: below it,
# starting the threads took longer than they saved.
_SHUFFLE_PARALLEL_MIN = 4096


def _thread_count(n_jobs):
	"""The threads `n_jobs` asks for, with -1 meaning one per thread numba
	allows."""

	return numba.config.NUMBA_NUM_THREADS if n_jobs == -1 else int(n_jobs)


def _shuffle_packed(codes, start, end, n_shuffles, random_states,
	verbose=False, n_jobs=1, alphabet=('A', 'C', 'G', 'T')):
	"""Dinucleotide shuffle positions `start` to `end` of each sequence of
	character indices `codes`, shape (n, length) with 4 for an unknown
	character, `n_shuffles` times, seeding sequence i with random_states[i].

	The shuffles are those the one-hot path of `dinucleotide_shuffle` draws
	for the same seeds. Each is packed as it is drawn, so neither the
	shuffled character indices nor a one-hot encoding of them is held for
	more than one sequence at a time. Returns a PackedSequence of shape
	(n, n_shuffles, 4, length).
	"""

	codes = numpy.ascontiguousarray(codes.cpu().numpy(), dtype=numpy.uint8)
	n_seqs, length = codes.shape

	is_n = codes == 4
	n_runs = is_n[:, :1].sum(axis=1) + (is_n[:, 1:] & ~is_n[:, :-1]).sum(
		axis=1)
	run_ptr = numpy.zeros(n_seqs + 1, dtype=numpy.int64)
	run_ptr[1:] = numpy.cumsum(n_runs)

	data = numpy.empty((n_seqs, n_shuffles, (length + 3) // 4),
		dtype=numpy.uint8)
	run_starts = numpy.empty(n_shuffles * run_ptr[-1], dtype=numpy.int64)
	run_ends = numpy.empty(n_shuffles * run_ptr[-1], dtype=numpy.int64)
	flags = numpy.empty((n_seqs, 3), dtype=numpy.bool_)

	args = (codes, start, end, n_shuffles, random_states, run_ptr, data,
		run_starts, run_ends, flags)
	if (n_jobs == 1 or n_seqs < 2 or
			n_seqs * n_shuffles * (end - start) < _SHUFFLE_PARALLEL_MIN):
		_fast_shuffle_packed(*args)
	else:
		with _numba_threads(_thread_count(n_jobs)):
			_fast_shuffle_packed_parallel(*args)

	# A shuffle keeps the count of every dinucleotide and both ends of the
	# region, and so the number of runs of unknown characters.
	if flags[:, 2].any():
		raise RuntimeError("A dinucleotide shuffle changed the number of runs "
			"of unknown characters.")

	_check_conserved(flags[:, 0], flags[:, 1], n_shuffles, verbose,
		stacklevel=4)

	rows = numpy.repeat(numpy.arange(n_seqs * n_shuffles),
		numpy.repeat(n_runs, n_shuffles))
	n_runs = numpy.stack([rows, run_starts, run_ends], axis=1)
	return PackedSequence(torch.from_numpy(data), length,
		torch.from_numpy(n_runs), alphabet)


def _dinucleotide_shuffle(X, n_shuffles=1, random_state=None, verbose=False):
	"""An internal function for dinucleotide shuffling a single sequence.

	This is `dinucleotide_shuffle` on one sequence, with unknown characters
	allowed, returning float32 on the CPU. gReLU imports this private
	function (`grelu.data.dataset`), so its signature and output must stay
	as they are.


	Parameters
	----------
	X: torch.tensor, shape=(len(alphabet), length)
		A single one-hot encoded sequence to be shuffled. An all-zero column,
		an unknown character, is shuffled as a character of its own.

	n_shuffles: int, optional
		The number of shuffled sequences to produce. Default is 1.

	random_state: int or None, optional
		The random seed to use when generating shuffles. If None, draw a new
		seed at random. Default is None.

	verbose: bool, optional
		Whether to print a warning when at least one position is identical
		across all shuffles. Default is False.


	Returns
	-------
	shuffled_sequences: torch.tensor, shape=(n_shuffles, len(alphabet), length)
		The shuffled sequences, float32, on the CPU.
	"""

	return dinucleotide_shuffle(X.unsqueeze(0).float().cpu(), n=n_shuffles,
		random_state=random_state, verbose=verbose, allow_N=True)[0]


@_accepts_packed
def dinucleotide_shuffle(
	X: torch.Tensor | PackedSequence,
	start: int = 0,
	end: int = -1,
	n: int = 20,
	random_state: int | Sequence[int] | None = None,
	verbose: bool = False,
	allow_N: bool = False,
	return_packed: bool | None = None,
	n_jobs: int = 4,
) -> torch.Tensor | PackedSequence:
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

	The shuffles can be returned as a `PackedSequence`, which holds four
	positions in a byte. With `return_packed=True` each shuffle is packed as
	it is drawn, so the shuffles take sixteen times less memory than an int8
	one-hot encoding of them would and the one-hot encoding is never built.
	The shuffles are the same as those of the one-hot encoding.

	Parameters
	----------
	X: torch.tensor, shape=(-1, len(alphabet), length), or PackedSequence
		A one-hot encoded set of sequences to be shuffled, or a PackedSequence
		with one batch axis. Unlike the other functions in this module,
		unknown characters, encoded as all-zero columns, are rejected unless
		`allow_N=True`.

	start: int, optional
		The starting position of where to randomize the sequence, inclusive.
		Default is 0, shuffling the entire sequence.

	end: int, optional
		The ending position of where to randomize the sequence. If end is
		positive then it is non-inclusive, but if end is negative then it is
		inclusive. Default is -1, shuffling the entire sequence.

	n: int, optional
		The number of times to shuffle that region. Default is 20.

	random_state: int or sequence of int or None, optional
		Whether to use a specific random seed when generating the shuffle,
		to ensure reproducibility. If None, do not use a reproducible seed.
		Unlike other methods, cannot be a numpy.random.RandomState object.
		Note: the seed used for the i-th sequence in the batch is
		`random_state + i`, so the per-sequence stream depends on batch
		position. Two calls with the same `random_state` but different batch
		sizes will agree on the leading prefix of sequences (positions where
		both batches contain that index) but diverge otherwise; this also
		means sequence 0 of one batch is reproduced as sequence 0 of any
		other batch that uses the same `random_state`. A sequence of
		`len(X)` integers instead gives each sequence its own seed, so
		sequence i is shuffled as `dinucleotide_shuffle(X[i:i+1], ...,
		random_state=random_state[i])` would shuffle it, whatever else is in
		the batch. Default is None.

	allow_N: bool, optional
		Whether to accept unknown characters, encoded as all-zero columns, in
		`X`. Each is shuffled as a character of its own, so every shuffle
		keeps the number of unknown characters and the count of every
		dinucleotide, including those that contain one. A contiguous run of
		unknown characters therefore stays one run but can move. If False,
		an unknown character raises a ValueError. Default is False.

	return_packed: bool or None, optional
		Whether to return the shuffles as a PackedSequence rather than a
		one-hot encoding. Packing needs an alphabet of four characters. If
		None, return a PackedSequence when `X` is one. Default is None.

	n_jobs: int, optional
		The number of threads to shuffle the sequences on, or -1 for one per
		thread numba allows, capped by numba's NUMBA_NUM_THREADS. A call that
		shuffles fewer than 4,096 positions in all, the sequences times the
		shuffles times the length of the region, runs on one thread, since
		starting the threads took longer than they saved. The shuffles are
		the same for every value. Default is 4.


	Returns
	-------
	shuffled_sequences: torch.tensor, shape=(-1, n, len(alphabet), length), or PackedSequence
		The shuffled sequences. A one-hot encoding has the dtype and device
		of `X`, or int8 on the CPU when `X` is a PackedSequence. A
		PackedSequence has batch axes (-1, n), the alphabet of `X` or ACGT,
		and its tensors on the CPU.
	"""

	if isinstance(X, PackedSequence):
		if X.data.ndim != 2:
			raise ValueError("X has shape {} but must have shape {}".format(
				tuple(X.shape), (-1, 4, -1)))

		if not allow_N and len(X.n_runs) > 0:
			raise ValueError("X must be one-hot encoded and cannot have unknown"
				" characters.")

		packed, length = True, X.length
	else:
		_validate_input(X, "X", shape=(-1, -1, -1), ohe=True, ohe_dim=1,
			allow_N=allow_N)
		packed, length = False, X.shape[-1]

	if return_packed is None:
		return_packed = packed

	if return_packed and not packed and X.shape[1] != 4:
		raise ValueError("return_packed=True needs an alphabet of four "
			"characters, but X has {}.".format(X.shape[1]))

	if end < 0:
		end = length + 1 + end

	random_states = _random_states(random_state, X.shape[0])

	if return_packed:
		if packed:
			codes, alphabet = X.to_codes(), X.alphabet
		else:
			values, codes = X.max(dim=1)
			codes[values == 0] = 4
			alphabet = ('A', 'C', 'G', 'T')

		start, end, _ = slice(start, end).indices(length)
		return _shuffle_packed(codes, start, max(start, end), n, random_states,
			verbose=verbose, n_jobs=n_jobs, alphabet=alphabet)

	if packed:
		X = X.to_one_hot()

	# The character index of every position. An unknown character, an
	# all-zero column, is shuffled as a character of its own, with index
	# len(alphabet), since its maximum would read it as the first character.
	# `max` rather than `argmax`, which was 150 times slower along this axis
	# on the CPU, for 64 sequences of 2,114bp.
	values, idxs = X[:, :, start:end].max(dim=1)
	idxs[values == 0] = X.shape[1]
	idxs = idxs.cpu().numpy().astype(numpy.int32)

	shuffled = numpy.empty((X.shape[0], n, idxs.shape[-1]), dtype=numpy.int32)
	args = idxs, X.shape[1] + 1, n, random_states, shuffled
	if n_jobs == 1 or X.shape[0] < 2 or shuffled.size < _SHUFFLE_PARALLEL_MIN:
		_fast_shuffle(*args)
	else:
		with _numba_threads(_thread_count(n_jobs)):
			_fast_shuffle_parallel(*args)

	shuffled = torch.from_numpy(shuffled)

	# The first and last characters are always kept. `conserved` marks the
	# positions between them where every shuffle of a sequence agrees.
	interior = shuffled[:, :, 1:-1]
	conserved = (interior == interior[:, :1]).all(dim=1)
	_check_conserved(conserved.any(dim=1), conserved.all(dim=1), n, verbose,
		stacklevel=3)

	X_shufs = X.unsqueeze(1).repeat(1, n, 1, 1)
	alphabet = torch.arange(X.shape[1], device=X.device)
	X_shufs[:, :, :, start:end] = (shuffled.to(X.device).unsqueeze(2) ==
		alphabet[:, None]).to(X.dtype)
	return X_shufs


def _local_bins(rng, length, bin_size, min_bin_size):
	"""Draw the bins of one local shuffle of a sequence of `length`: a first
	cut between `min_bin_size` and `bin_size`, bins of `bin_size` after it,
	and a final bin merged into the one before when shorter than
	`min_bin_size`."""

	first_cut = rng.randint(min_bin_size, bin_size + 1)
	boundaries = [0, first_cut]
	boundaries.extend(range(first_cut + bin_size, length, bin_size))
	boundaries.append(length)

	bins = list(zip(boundaries[:-1], boundaries[1:]))
	if bins[-1][1] - bins[-1][0] < min_bin_size:
		# merge last two bins if the final bin is too small
		bins[-2] = (bins[-2][0], bins[-1][1])
		bins.pop()

	return bins


def _warn_unshuffled(n_unshuffled, n_bins):
	"""Warn that `n_unshuffled` of `n_bins` bins came back unshuffled."""

	if n_unshuffled > 0:
		warnings.warn("{} of {} bins came back identical to the input and are "
			"not randomized in the returned sequences. A bin whose characters "
			"form a single Eulerian path, such as a homopolymer or a short "
			"tandem repeat, has only one possible dinucleotide shuffle. "
			"Raising `bin_size` merges such a region into a more diverse "
			"neighborhood.".format(n_unshuffled, n_bins), TangermemeWarning,
			stacklevel=3)


@numba.njit(nogil=True, cache=True)
def _shuffle_bins(codes, n_shuffles, bin_ptr, bin_starts, bin_ends,
	random_states, data):
	"""For each sequence k of `codes` and each shuffle i, dinucleotide shuffle
	each of its bins once, seeding the generator for bin b with
	random_states[b], and pack the sequence into data[k, i]. The bins of
	(k, i) are bin_ptr[k * n_shuffles + i] to the next. Returns the number of
	bins that came back identical to the input."""

	n_seqs, length = codes.shape
	row = numpy.empty(length, dtype=numpy.uint8)
	unshuffled = 0
	for k in range(n_seqs):
		for i in range(n_shuffles):
			for j in range(length):
				row[j] = codes[k, j]

			p = k * n_shuffles + i
			for b in range(bin_ptr[p], bin_ptr[p + 1]):
				start, end = bin_starts[b], bin_ends[b]
				shuffled = numpy.empty((1, end - start), dtype=numpy.uint8)
				_shuffle_sequence(codes[k, start:end], 5, 1, random_states[b],
					shuffled)

				same = True
				for j in range(end - start):
					row[start + j] = shuffled[0, j]
					same = same and shuffled[0, j] == codes[k, start + j]
				unshuffled += same

			_pack_row(row, data[k], i, length)

	return unshuffled


def _local_shuffle_packed(X, n, bin_size, min_bin_size, rng, verbose,
	alphabet):
	"""The packed path of `local_dinucleotide_shuffle`: draw every bin and
	seed from `rng` in the order the one-hot path draws them, then shuffle
	and pack all of them in one call."""

	bin_ptr, starts, ends, seeds = [0], [], [], []
	for i in range(X.shape[0]):
		for j in range(n):
			for bin_start, bin_end in _local_bins(rng, X.shape[-1], bin_size,
				min_bin_size):
				starts.append(bin_start)
				ends.append(bin_end)
				seeds.append(rng.randint(0, 2**31 - 1))

			bin_ptr.append(len(starts))

	# Each bin is shuffled once, which dinucleotide_shuffle reports under
	# `verbose` as every position kept.
	if verbose:
		for bin_start, bin_end in zip(starts, ends):
			if bin_end - bin_start > 2:
				warnings.warn("At least one position in dinucleotide shuffle "
					"is identical across all positions.", TangermemeWarning,
					stacklevel=3)

	codes = _codes_of(X, slice(None)).numpy()
	data = numpy.empty((X.shape[0], n, (X.shape[-1] + 3) // 4),
		dtype=numpy.uint8)
	n_unshuffled = _shuffle_bins(codes, n, numpy.array(bin_ptr,
		dtype=numpy.int64), numpy.array(starts, dtype=numpy.int64),
		numpy.array(ends, dtype=numpy.int64), numpy.array(seeds,
		dtype=numpy.int64).astype(numpy.int32), data)

	Y = PackedSequence(torch.from_numpy(data), X.shape[-1], alphabet=alphabet)
	return Y, len(starts), n_unshuffled


@_accepts_packed
def local_dinucleotide_shuffle(
	X: torch.Tensor | PackedSequence,
	n: int = 20,
	bin_size: int = 1024,
	min_bin_size: int = 512,
	random_state: int | None = None,
	verbose: bool = False,
	return_packed: bool | None = None,
) -> torch.Tensor | PackedSequence:
	"""Dinucleotide-shuffle sequences independently within local bins.

	A dinucleotide shuffle conserves the dinucleotide composition of the
	sequence as a whole, and in doing so flattens any structure in how that
	composition varies along it. A genomic window is rarely uniform: GC
	content, repeat density and nucleotide composition drift across it, and a
	background that averages all of that away differs from the original in
	more ways than the motif content a marginalization is trying to isolate.

	This function shuffles within consecutive bins instead, so composition is
	conserved locally as well as globally. The interface otherwise follows
	`dinucleotide_shuffle`, with two differences: there is no `start` or `end`
	because the whole sequence is always shuffled, and `random_state` must be
	an integer rather than a RandomState.

	The bin boundaries are redrawn for every shuffle. The first cut is placed
	uniformly at random between `min_bin_size` and `bin_size`, and the rest
	follow at `bin_size` intervals from there. A dinucleotide shuffle holds
	the first and last character of the region it is applied to, so fixing the
	boundaries would pin those positions across every shuffle in the set;
	redrawing them spreads that over different positions instead. A trailing
	bin shorter than `min_bin_size` is merged into the one before it, so a bin
	is never shorter than `min_bin_size` and can be as long as twice
	`bin_size`.


	Parameters
	----------
	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to be shuffled. Any alphabet is
		accepted, not only DNA.

	n: int, optional
		The number of times to shuffle each sequence. Each shuffle draws its
		own bin boundaries. Default is 20.

	bin_size: int, optional
		The spacing between bin boundaries after the first cut. Must be
		strictly smaller than the sequence length, or a ValueError is raised;
		use `dinucleotide_shuffle` directly for a sequence shorter than one
		bin. A bin whose characters admit only one Eulerian path, as a
		homopolymer or a short tandem repeat does, has itself as its only
		possible shuffle and is returned unrandomized; a `TangermemeWarning`
		reports how many bins that happened to, and raising `bin_size` merges
		such a region into a more diverse neighborhood. Default is 1024.

	min_bin_size: int, optional
		The shortest a bin may be, used both as the lower bound on the random
		first cut and as the threshold below which a trailing bin is merged
		backwards. Must be no larger than `bin_size`, or a ValueError is
		raised. Default is 512.

	random_state: int or None, optional
		Whether to use a specific random seed when generating the shuffle, to
		ensure reproducibility. It seeds both the bin boundaries and the
		shuffle within each bin. Unlike `dinucleotide_shuffle`, this cannot be
		a numpy.random.RandomState object. If None, do not use a reproducible
		seed. Default is None.

	verbose: bool, optional
		Whether to print a warning when at least one position is identical
		across all shuffles of a bin. Each bin is shuffled once, so that
		condition holds for almost every bin and this emits roughly one
		warning per bin; the `TangermemeWarning` described under `bin_size` is
		the useful signal. Default is False.

	return_packed: bool or None, optional
		Whether to return the sequences as a PackedSequence rather than a
		one-hot encoding, which needs an alphabet of four characters. `X` can
		be a PackedSequence with one batch axis, which is decoded and packed
		a chunk of sequences at a time. If None, return a PackedSequence when
		`X` is one, and with False, a PackedSequence `X` gives an int8 one-hot
		encoding. Default is None.


	Returns
	-------
	shuffled_sequences: torch.tensor, shape=(-1, n, len(alphabet), length), or PackedSequence
		The shuffled sequences. Dtype and device match the input `X`.
	"""

	_validate_input(X, "X", shape=(-1, -1, -1), ohe=True, ohe_dim=1)

	if bin_size >= X.shape[-1]:
		raise ValueError(
			"Sequence length must be longer than bin_size. "
			"Use dinucleotide_shuffle directly for shorter sequences."
		)
	if min_bin_size > bin_size:
		raise ValueError("min_bin_size must be <= bin_size")

	rng = numpy.random.RandomState(random_state)

	return_packed, packed_alphabet = _packed_output(X, return_packed, "ACGT")
	if return_packed or isinstance(X, PackedSequence):
		Y, n_bins, n_unshuffled = _local_shuffle_packed(X, n, bin_size,
			min_bin_size, rng, verbose, packed_alphabet)
		_warn_unshuffled(n_unshuffled, n_bins)
		return Y if return_packed else Y.to_one_hot()

	X_shuf = X.unsqueeze(1).repeat(1, n, 1, 1)
	n_bins, n_unshuffled = 0, 0

	for i in range(X.shape[0]):
		for j in range(n):
			bins = _local_bins(rng, X.shape[-1], bin_size, min_bin_size)
			for (bin_start, bin_end) in bins:
				shuffled = dinucleotide_shuffle(
					X[i:i+1, :, bin_start:bin_end],
					n=1,
					random_state=rng.randint(0, 2**31 - 1),
					verbose=verbose,
				)
				X_shuf[i, j, :, bin_start:bin_end] = shuffled[0, 0]

				# `dinucleotide_shuffle` raises when every shuffle of a region
				# comes back identical, but only when asked for more than one.
				# It is called once per bin here, so that check cannot fire and
				# an unshuffleable bin would otherwise pass through silently.
				n_bins += 1
				n_unshuffled += torch.equal(shuffled[0, 0].to(X.device),
					X[i, :, bin_start:bin_end])

	_warn_unshuffled(n_unshuffled, n_bins)
	return X_shuf
