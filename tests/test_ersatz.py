# test_io.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numba
import numpy
import torch
import pytest
import warnings
import collections

from tangermeme.utils import characters
from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot
from tangermeme.utils import PackedSequence

from tangermeme.ersatz import insert
from tangermeme.ersatz import substitute
from tangermeme.ersatz import multisubstitute
from tangermeme.ersatz import delete
from tangermeme.ersatz import randomize
from tangermeme.ersatz import shuffle
from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.ersatz import _dinucleotide_shuffle
from tangermeme.ersatz import local_dinucleotide_shuffle
from tangermeme.ersatz import _fast_shuffle_packed

from tangermeme.utils import TangermemeWarning
from ._packed import assert_packed_equal

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


@pytest.fixture
def X():
	seq = 'CACATCATCTCATCATCTGCTGACTACTGACGTAGTCTGACTGACTGACTGACTACTGACTGACTGAC'
	return one_hot_encode(seq).unsqueeze(0)


###


def _motif(seq, form):
	"""A motif as a string or as its one-hot encoding."""

	return seq if form == "str" else one_hot_encode(seq).unsqueeze(0)


@pytest.mark.parametrize("form", ["str", "one_hot"])
def test_insert(X, form):
	X_insert = insert(X, _motif('CATCAG', form))

	assert X_insert.shape[:2] == X.shape[:2]
	assert X_insert.shape[-1] != X.shape[-1]
	assert X_insert.sum() != X.sum()

	assert_raises(AssertionError, assert_array_almost_equal,
		X_insert.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_insert, X)

	new_seq = ('CACATCATCTCATCATCTGCTGACTACTGACGTACAT' +
		'CAGGTCTGACTGACTGACTGACTACTGACTGACTGAC')
	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)

	assert X_insert.shape == new_seq_ohe.shape
	assert X_insert.sum() == new_seq_ohe.sum()
	assert_array_almost_equal(X_insert.sum(dim=-1), new_seq_ohe.sum(dim=-1))
	assert_array_almost_equal(X_insert, new_seq_ohe)


def test_insert_start(X):
	motif = 'CATCAG'
	X_insert = insert(X, motif, start=0)

	assert X_insert.shape[:2] == X.shape[:2]
	assert X_insert.shape[-1] != X.shape[-1]
	assert X_insert.sum() != X.sum()
	assert_raises(AssertionError, assert_array_almost_equal,
		X_insert.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_insert, X)

	new_seq = ('CATCAGCACATCATCTCATCATCTGCTGACTAC' +
		'TGACGTAGTCTGACTGACTGACTGACTACTGACTGACTGAC')
	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)
	assert_array_almost_equal(X_insert, new_seq_ohe)


	motif = 'CATCAGCCC'
	X_insert = insert(X, motif, start=10)

	assert X_insert.shape[:2] == X.shape[:2]
	assert X_insert.shape[-1] == 77
	assert X_insert.sum() != X.sum()
	assert_raises(AssertionError, assert_array_almost_equal,
		X_insert.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_insert, X)

	new_seq = ('CACATCATCTCATCAGCCCCATCATCTGCTGACTACTGA' +
		'CGTAGTCTGACTGACTGACTGACTACTGACTGACTGAC')
	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)
	assert_array_almost_equal(X_insert, new_seq_ohe)


def test_insert_long(X):
	motif = 'C'*1000
	X_insert = insert(X, motif)
	assert X_insert.shape == (1, 4, 1068)
	assert X_insert[0, 1].sum() > 1000


def test_insert_raises_alphabet(X):
	motif = one_hot_encode('CACCAG', alphabet=['A', 'C', 'G']).unsqueeze(0)
	with pytest.raises(ValueError, match=r"motif has shape .*\[1, 3, 6\]"):
		insert(X, motif)


def test_insert_raise_ends(X):
	assert_raises(ValueError, insert, X, 'CAGCAT', -2)
	assert_raises(ValueError, insert, X, 'CAGCAT', 1000)
	assert_raises(TypeError, insert, X, 'CAGCAT', 6.5)


def test_insert_X_with_N(X):
	# Unknown characters in X are all-zero columns. The insertion splices
	# the sequence in two and so carries them through unchanged.
	X_n = torch.clone(X)
	X_n[0, :, 10:14] = 0

	X_insert = insert(X_n, 'CATCAG')
	start = X.shape[-1] // 2

	assert X_insert.shape[-1] == X.shape[-1] + 6
	assert_array_almost_equal(X_insert[0, :, 10:14], torch.zeros(4, 4))
	assert_array_almost_equal(X_insert[:, :, :start], X_n[:, :, :start])
	assert_array_almost_equal(X_insert[0, :, start:start+6],
		one_hot_encode('CATCAG'))
	assert_array_almost_equal(X_insert[:, :, start+6:], X_n[:, :, start:])


def test_insert_str_alphabet():
	alphabet = ['A', 'C', 'G', 'U']
	X = one_hot_encode('ACGUACGU', alphabet=alphabet).unsqueeze(0)
	X_insert = insert(X, 'UUU', start=2, alphabet=alphabet)

	assert X_insert.shape == (1, 4, 11)
	assert characters(X_insert[0], alphabet=alphabet) == 'ACUUUGUACGU'


def test_insert_str_raises_N(X):
	# A string motif is encoded with N ignored, and insert rejects the
	# resulting all-zero column.
	with pytest.raises(ValueError, match="cannot have unknown characters"):
		insert(X, 'CANTG')


###


@pytest.mark.parametrize("form", ["str", "one_hot"])
def test_substitute(X, form):
	X_substitute = substitute(X, _motif('CATCAG', form))

	assert X_substitute.shape == X.shape
	assert X_substitute.sum() == X.sum()
	assert_raises(AssertionError, assert_array_almost_equal,
		X_substitute.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_substitute, X)

	new_seq = ('CACATCATCTCATCATCTGCTGACTACTGACCAT' +
		'CAGTGACTGACTGACTGACTACTGACTGACTGAC')
	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)

	assert X_substitute.shape == new_seq_ohe.shape
	assert X_substitute.sum() == new_seq_ohe.sum()
	assert_array_almost_equal(X_substitute.sum(dim=-1), new_seq_ohe.sum(dim=-1))
	assert_array_almost_equal(X_substitute, new_seq_ohe)


def test_substitute_str_multi_seqs_one_motif():
	X = random_one_hot((4, 4, 8), random_state=0)
	X_substitute = substitute(X, 'ACGT')

	assert_raises(AssertionError, assert_array_almost_equal, X, X_substitute)
	assert_array_almost_equal(X_substitute, [
		[[1, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 1, 1, 1]],

        [[0, 0, 1, 0, 0, 0, 0, 1],
         [1, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 1, 0],
         [0, 1, 0, 0, 0, 1, 0, 0]],

        [[1, 1, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 1],
         [0, 0, 0, 0, 0, 1, 1, 0]],

        [[1, 0, 1, 0, 0, 0, 0, 1],
         [0, 1, 0, 1, 0, 0, 1, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0]]])


def test_substitute_str_multi_seqs_multi_motifs():
	X = random_one_hot((4, 4, 8), random_state=0)
	motif = random_one_hot((4, 4, 4), random_state=1)
	X_substitute = substitute(X, motif)

	assert_raises(AssertionError, assert_array_almost_equal, X, X_substitute)
	assert_array_almost_equal(X_substitute, [
		[[1, 0, 0, 0, 1, 1, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 1, 0, 0, 1, 1]],

        [[0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 1, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 1, 1, 0, 1, 0, 0, 0]],

        [[1, 1, 0, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 1, 0, 0, 0, 1, 0]],

        [[1, 0, 1, 0, 0, 1, 0, 1],
         [0, 1, 0, 0, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0]]])


@pytest.mark.parametrize("form", ["str", "one_hot"])
def test_substitute_start(X, form):
	X_substitute = substitute(X, _motif('CATCAG', form), start=0)

	assert X_substitute.shape == X.shape
	assert X_substitute.sum() == X.sum()
	assert_raises(AssertionError, assert_array_almost_equal,
		X_substitute.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_substitute, X)

	new_seq = ('CATCAGATCTCATCATCTGCTGACTACT' +
		'GACGTAGTCTGACTGACTGACTGACTACTGACTGACTGAC')
	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)
	assert_array_almost_equal(X_substitute, new_seq_ohe)


	X_substitute = substitute(X, _motif('CATCAGCCC', form), start=10)

	assert X_substitute.shape == X.shape
	assert X_substitute.sum() == X.sum()
	assert_raises(AssertionError, assert_array_almost_equal,
		X_substitute.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_substitute, X)

	new_seq = ('CACATCATCTCATCAGCCCCTGACTACTGACGTA' +
		'GTCTGACTGACTGACTGACTACTGACTGACTGAC')
	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)
	assert_array_almost_equal(X_substitute, new_seq_ohe)


def test_substitute_raises_alphabet(X):
	motif = one_hot_encode('CACCAG', alphabet=['A', 'C', 'G']).unsqueeze(0)
	with pytest.raises(ValueError, match=r"motif has shape .*\[1, 3, 6\]"):
		substitute(X, motif)


def test_substitute_raises_length(X):
	assert_raises(ValueError, substitute, X, 'C'*1000)
	with pytest.raises(ValueError, match="Motif cannot be longer than"):
		substitute(X, one_hot_encode('C'*1000).unsqueeze(0))


def test_substitute_raise_ends(X):
	assert_raises(ValueError, substitute, X, 'CAGCAT', -2)
	assert_raises(ValueError, substitute, X, 'CAGCAT', 1000)
	assert_raises(IndexError, substitute, X, 'CAGCAT', 6.5)


def test_substitute_X_with_N(X):
	# Unknown characters in X are all-zero columns, as `one_hot_encode`
	# produces for the characters in its `ignore` list. They are valid
	# input and are left alone when the motif does not cover them.
	X_n = torch.clone(X)
	X_n[0, :, 10:14] = 0

	X_substitute = substitute(X_n, 'CATCAG')
	start = X.shape[-1] // 2 - 3

	assert X_substitute.shape == X_n.shape
	assert_array_almost_equal(X_substitute[0, :, 10:14], torch.zeros(4, 4))
	assert_array_almost_equal(X_substitute[:, :, :start], X_n[:, :, :start])
	assert_array_almost_equal(X_substitute[0, :, start:start+6],
		one_hot_encode('CATCAG'))


def test_substitute_X_with_N_under_motif(X):
	# An unknown character that the motif covers is overwritten by it.
	X_n = torch.clone(X)
	start = X.shape[-1] // 2 - 3
	X_n[0, :, start:start+2] = 0

	X_substitute = substitute(X_n, 'CATCAG')

	assert_array_almost_equal(X_substitute[0, :, start:start+6],
		one_hot_encode('CATCAG'))


def test_substitute_str_ignore_keeps_sequence(X):
	# The N positions of a string motif keep the original sequence.
	seq = characters(X[0])
	X_substitute = substitute(X, 'ACNNGT', start=5)

	assert characters(X_substitute[0]) == (seq[:5] + 'AC' + seq[7:9] + 'GT' +
		seq[11:])


def test_substitute_str_ignore_set(X):
	seq = characters(X[0])
	X_substitute = substitute(X, 'ACXNGT', start=5, ignore={'N', 'X'})

	assert characters(X_substitute[0]) == (seq[:5] + 'AC' + seq[7:9] + 'GT' +
		seq[11:])


###


def test_delete(X):
	X_delete = delete(X, start=0, end=5)

	assert X_delete.shape != X.shape
	assert X_delete.shape[-1] == X.shape[-1] - 5
	assert X_delete.sum() == X[:, :, 5:].sum()
	assert_raises(AssertionError, assert_array_almost_equal,
		X_delete.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_delete, X)

	X_delete = delete(X, start=10, end=17)

	assert X_delete.shape != X.shape
	assert X_delete.shape[-1] == X.shape[-1] - 7
	assert_array_almost_equal(X_delete[:, :, :10], X[:, :, :10])
	assert_array_almost_equal(X_delete[:, :, 10:], X[:, :, 17:])

	assert_raises(AssertionError, assert_array_almost_equal,
		X_delete.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_delete, X)



def test_delete_raise_ends(X):
	assert_raises(ValueError, delete, X, start=-1, end=5)
	assert_raises(ValueError, delete, X, start=100, end=5)
	assert_raises(ValueError, delete, X, start=10, end=5)

	assert_raises(ValueError, delete, X, start=0, end=-1)
	assert_raises(ValueError, delete, X, start=-5, end=-10)
	assert_raises(ValueError, delete, X, start=5, end=5)
	assert_raises(ValueError, delete, X, start=5, end=3)
	assert_raises(ValueError, delete, X, start=5, end=100)


def test_delete_X_with_N(X):
	# Unknown characters are all-zero columns and survive the deletion
	# when they fall outside the deleted portion.
	X_n = torch.clone(X)
	X_n[0, :, 10:14] = 0

	X_delete = delete(X_n, start=0, end=5)

	assert X_delete.shape[-1] == X.shape[-1] - 5
	assert_array_almost_equal(X_delete[0, :, 5:9], torch.zeros(4, 4))
	assert_array_almost_equal(X_delete, X_n[:, :, 5:])

	# Deleting the unknown characters themselves removes them.
	X_delete = delete(X_n, start=10, end=14)
	assert (X_delete.sum(dim=1) == 0).sum() == 0


###


@pytest.mark.parametrize("spacing, new_seq", [
	(0, 'CACATCATCTCATCATCTGCTGACTACTCATCAG' +
		'CATCAGCTGACTGACTGACTACTGACTGACTGAC'),
	(4, 'CACATCATCTCATCATCTGCTGACTACATCAGTA' +
		'GTCATCAGGACTGACTGACTACTGACTGACTGAC'),
	(5, 'CACATCATCTCATCATCTGCTGACTACATCAGTA' +
		'GTCCATCAGACTGACTGACTACTGACTGACTGAC'),
], ids=["spacing_0", "spacing_4", "spacing_5"])
@pytest.mark.parametrize("form", ["str", "one_hot"])
def test_multisubstitute(X, spacing, new_seq, form):
	motif = _motif('CATCAG', form)
	X_substitute = multisubstitute(X, [motif, motif], spacing=spacing)

	assert X_substitute.shape == X.shape
	assert X_substitute.sum() == X.sum()
	assert_raises(AssertionError, assert_array_almost_equal,
		X_substitute.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_substitute, X)

	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)

	assert X_substitute.shape == new_seq_ohe.shape
	assert X_substitute.sum() == new_seq_ohe.sum()
	assert_array_almost_equal(X_substitute.sum(dim=-1), new_seq_ohe.sum(dim=-1))
	assert_array_almost_equal(X_substitute, new_seq_ohe)


def test_multisubstitute_str_multi_seqs_one_motif():
	X = random_one_hot((4, 4, 8), random_state=0)
	X_substitute = multisubstitute(X, ['A', 'C', 'GT'], spacing=[0, 0])

	assert_raises(AssertionError, assert_array_almost_equal, X, X_substitute)
	assert_array_almost_equal(X_substitute, [
		[[1, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 1, 1, 1]],

        [[0, 0, 1, 0, 0, 0, 0, 1],
         [1, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 1, 0],
         [0, 1, 0, 0, 0, 1, 0, 0]],

        [[1, 1, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 1],
         [0, 0, 0, 0, 0, 1, 1, 0]],

        [[1, 0, 1, 0, 0, 0, 0, 1],
         [0, 1, 0, 1, 0, 0, 1, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0]]])



def test_multisubstitute_start(X):
	motifs = ['CATC', 'AGCCC']
	X_substitute = multisubstitute(X, motifs, spacing=[0], start=10)

	assert X_substitute.shape == X.shape
	assert X_substitute.sum() == X.sum()
	assert_raises(AssertionError, assert_array_almost_equal,
		X_substitute.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_substitute, X)

	new_seq = ('CACATCATCTCATCAGCCCCTGACTACTGACGTA' +
		'GTCTGACTGACTGACTGACTACTGACTGACTGAC')
	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)
	assert_array_almost_equal(X_substitute, new_seq_ohe)

	motifs = ['CATC', 'AGC', 'CC']
	X_substitute = multisubstitute(X, motifs, spacing=[0, 0], start=10)
	new_seq = ('CACATCATCTCATCAGCCCCTGACTACTGACGTA' +
		'GTCTGACTGACTGACTGACTACTGACTGACTGAC')
	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)
	assert_array_almost_equal(X_substitute, new_seq_ohe)


	X_substitute = multisubstitute(X, motifs, spacing=[5, 4], start=10)
	assert X_substitute.shape == X.shape
	assert X_substitute.sum() == X.sum()
	assert_raises(AssertionError, assert_array_almost_equal,
		X_substitute.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_substitute, X)

	new_seq = ('CACATCATCTCATCATCTGAGCACTACCGACGTAG' +
		'TCTGACTGACTGACTGACTACTGACTGACTGAC')
	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)
	assert_array_almost_equal(X_substitute, new_seq_ohe)


def test_multisubstitute_raises_alphabet(X):
	motif = one_hot_encode('CACCAG', alphabet=['A', 'C', 'G']).unsqueeze(0)
	with pytest.raises(ValueError, match=r"motif has shape .*\[1, 3, 6\]"):
		multisubstitute(X, [motif, motif], [0])


def test_multisubstitute_raises_length(X):
	assert_raises(ValueError, multisubstitute, X, ['C'*1000, 'C'], [0])
	assert_raises(ValueError, multisubstitute, X, ['C',
		one_hot_encode('C'*1000)], [0])


def test_multisubstitute_spacing(X):
	motifs = ['CATGG', 'CAGGA']
	assert_raises(ValueError, multisubstitute, X, motifs, [-1])
	assert_raises(ValueError, multisubstitute, X, motifs, [0, 0])
	assert_raises(ValueError, multisubstitute, X, motifs, [])
	assert_raises(ValueError, multisubstitute, X, motifs, [10000])


def test_multisubstitute_X_with_N(X):
	# multisubstitute validates X only through the substitutions it makes,
	# so it inherits their handling of unknown characters.
	X_n = torch.clone(X)
	X_n[0, :, 10:14] = 0

	X_substitute = multisubstitute(X_n, ['CATGG', 'CAGGA'], [3])

	assert X_substitute.shape == X_n.shape
	assert_array_almost_equal(X_substitute[0, :, 10:14], torch.zeros(4, 4))
	assert_array_almost_equal(X_substitute[:, :, :10], X_n[:, :, :10])

	X_multi = torch.clone(X)
	X_multi[0, :, 5] = 1
	assert_raises(ValueError, multisubstitute, X_multi, ['CATGG', 'CAGGA'], [3])


###


def test_randomize(X):
	X_rand = randomize(X, start=10, end=30, random_state=0)

	assert len(X_rand.shape) == 4
	assert X_rand.shape == (1, 1, 4, X.shape[-1])
	assert X_rand.sum() == X.sum()
	assert (X_rand != X).any()
	assert (X_rand[:, 0, :, :10] == X[:, :, :10]).all()
	assert (X_rand[:, 0, :, 30:] == X[:, :, 30:]).all()

	assert_raises(AssertionError, assert_array_almost_equal,
		X_rand.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_rand, X)


def test_randomize_probs(X):
	X_rand = randomize(X, start=10, end=30, probs=[[1.0, 0.0, 0.0, 0.0]],
		random_state=0)
	assert_array_almost_equal(X_rand[:, 0, :, 10:30].sum(dim=(0, -1)),
		[20, 0, 0, 0])

	X_rand = randomize(X.repeat(2, 1, 1), start=10, end=30,
		probs=[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], random_state=0)
	assert_array_almost_equal(X_rand[0, 0, :, 10:30].sum(dim=-1), [20, 0, 0, 0])
	assert_array_almost_equal(X_rand[1, 0, :, 10:30].sum(dim=-1), [0, 0, 20, 0])


def test_randomize_n(X):
	X_rand = randomize(X, start=10, end=30, n=12, random_state=0)
	assert X_rand.shape == (1, 12, 4, X.shape[-1])
	assert (X_rand[0, 0:1] != X_rand[0]).any()

	X_rand = randomize(X, start=10, end=30, probs=[[1.0, 0.0, 0.0, 0.0]], n=12,
		random_state=0)
	assert X_rand.shape == (1, 12, 4, X.shape[-1])
	assert (X_rand[0, 0:1] == X_rand[0]).all()

	X_rand = randomize(X.repeat(5, 1, 1), start=10, end=30, n=12,
		random_state=0)
	assert X_rand.shape == (5, 12, 4, X.shape[-1])
	assert (X_rand[:, 0:1] != X_rand).any()


def test_randomize_raises_ends(X):
	assert_raises(ValueError, randomize, X, start=-3, end=10)
	assert_raises(ValueError, randomize, X, start=500, end=10)
	assert_raises(ValueError, randomize, X, start=5, end=3)
	assert_raises(ValueError, randomize, X, start=5, end=1000)


def test_randomize_full_sequence(X):
	# `end` is exclusive (matches Python slicing conventions used elsewhere
	# in ersatz), so end == X.shape[-1] is the "randomize entire sequence"
	# case and must not raise.
	X_rand = randomize(X, start=0, end=X.shape[-1], random_state=0)
	assert X_rand.shape == (1, 1, 4, X.shape[-1])
	assert X_rand.sum() == X.sum()


def test_randomize_genome_probs(X):
	# Four decimals that sum to 1 in float64 but not after a float32 cast.
	probs = [[0.295, 0.205, 0.205, 0.295]]
	X_rand = randomize(X, start=10, end=30, probs=probs, random_state=0)
	assert X_rand.shape == (X.shape[0], 1, 4, X.shape[-1])
	assert_array_almost_equal(X_rand[:, 0, :, :10], X[:, :, :10])
	assert_array_almost_equal(X_rand[:, 0, :, 30:], X[:, :, 30:])
	assert torch.all(X_rand[:, 0, :, 10:30].sum(dim=1) == 1)

	X_rand2 = randomize(X, start=10, end=30, probs=torch.tensor(probs,
		dtype=torch.float32), random_state=0)
	assert_array_almost_equal(X_rand, X_rand2)

	X_rand3 = randomize(X, start=10, end=30, probs=numpy.array(probs),
		random_state=0)
	assert_array_almost_equal(X_rand, X_rand3)


def test_randomize_raises_probs(X):
	assert_raises(ValueError, randomize, X, start=5, end=10,
		probs=[[0.1, 0.8, 0.1]])
	assert_raises(ValueError, randomize, X, start=5, end=10,
		probs=[[0.1, 0.8, 0.1, 0.0], [0.4, 0.3, 0.2]])
	assert_raises(ValueError, randomize, X, start=5, end=10,
		probs=[0.1, 0.8, 0.1, 0.0])
	assert_raises(ValueError, randomize, X, start=5, end=10,
		probs=[[0.1, 100.8, 0.1]])


def test_randomize_X_with_N(X):
	# Unknown characters outside the randomized portion are kept; ones
	# inside it are replaced along with the rest of the window.
	X_n = torch.clone(X)
	X_n[0, :, 10:14] = 0

	X_rand = randomize(X_n, start=20, end=30, random_state=0)

	assert X_rand.shape == (1, 1, 4, X.shape[-1])
	assert_array_almost_equal(X_rand[0, 0, :, 10:14], torch.zeros(4, 4))
	assert_array_almost_equal(X_rand[0, 0, :, :20], X_n[0, :, :20])
	assert X_rand[0, 0, :, 20:30].sum() == 10

	X_rand = randomize(X_n, start=10, end=14, random_state=0)
	assert (X_rand[0, 0].sum(dim=0) == 0).sum() == 0


###


def test_shuffle(X):
	X_shuf = shuffle(X, start=10, end=30, random_state=0)

	assert X_shuf.shape == (1, 1, 4, X.shape[-1])
	assert X_shuf.sum() == X.sum()
	assert (X_shuf != X).any()
	assert (X_shuf[:, 0, :, :10] == X[:, :, :10]).all()
	assert (X_shuf[:, 0, :, 30:] == X[:, :, 30:]).all()

	assert_array_almost_equal(X_shuf[:, 0].sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_shuf[:, 0], X)


def test_shuffle_default(X):
	X_shuf = shuffle(X, random_state=0)

	assert X_shuf.shape == (1, 1, 4, X.shape[-1])
	assert X_shuf.sum() == X.sum()
	assert ((X_shuf[:, 0] != X).sum(dim=(-1, -2)) > 0).all()

	assert_array_almost_equal(X_shuf[:, 0].sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_shuf[:, 0], X)


def test_shuffle_raises_ends(X):
	assert_raises(ValueError, shuffle, X, start=-3, end=10)
	assert_raises(ValueError, shuffle, X, start=500, end=10)
	assert_raises(ValueError, shuffle, X, start=5, end=3)
	assert_raises(ValueError, shuffle, X, start=5, end=1000)


def test_shuffle_X_with_N(X):
	# The shuffle permutes the columns of the region, so unknown characters
	# inside it are moved rather than resolved and their count is kept.
	X_n = torch.clone(X)
	X_n[0, :, 10:14] = 0

	X_shuf = shuffle(X_n, start=5, end=20, random_state=0)

	assert X_shuf.shape == (1, 1, 4, X.shape[-1])
	assert (X_shuf[0, 0].sum(dim=0) == 0).sum() == 4
	assert_array_almost_equal(X_shuf[0, 0, :, :5], X_n[0, :, :5])
	assert_array_almost_equal(X_shuf[0, 0, :, 20:], X_n[0, :, 20:])
	assert_array_almost_equal(X_shuf[0, 0].sum(dim=-1), X_n[0].sum(dim=-1))

	# Unknown characters outside the shuffled region stay where they are.
	X_shuf = shuffle(X_n, start=20, end=40, random_state=0)
	assert_array_almost_equal(X_shuf[0, 0, :, 10:14], torch.zeros(4, 4))


###


def test_dinucleotide_shuffle():
	motif = one_hot_encode('CATCACGCATACG').unsqueeze(0)
	dimotif = dinucleotide_shuffle(motif, random_state=0)
	assert dimotif.shape == (1, 20, 4, 13)
	assert dimotif.dtype == torch.int8
	assert characters(dimotif[0, 0]) == 'CGCATCACATACG'
	assert characters(dimotif[0, 1]) == 'CATCGCATACACG'

	dimotif = dinucleotide_shuffle(motif, n=5, random_state=0)
	assert dimotif.shape == (1, 5, 4, 13)

	dimotif = dinucleotide_shuffle(torch.cat([motif]*8), n=7, random_state=0)
	assert dimotif.shape == (8, 7, 4, 13)


@pytest.mark.parametrize("X, alphabet", [
	(random_one_hot((8, 4, 30), random_state=0), ['A', 'C', 'G', 'T']),
	(one_hot_encode('CGATCAGCANNCACATCAGCATANNAAT', alphabet=['A', 'C', 'G',
		'T', 'N'], ignore=[]).unsqueeze(0), ['A', 'C', 'G', 'T', 'N']),
	(one_hot_encode('ATATATTAAAATTATTATATATTTATATATTTAAAAATTTTTAATA'
		).unsqueeze(0), ['A', 'C', 'G', 'T']),
], ids=["random", "large_alphabet", "missing_alphabet"])
def test_dinucleotide_shuffle_composition(X, alphabet):
	X_shuf = dinucleotide_shuffle(X, random_state=0)

	for i in range(X.shape[0]):
		dinucs = _dinuc_counts(characters(X[i], alphabet=alphabet))

		for j in range(20):
			dinucs_shuffled = _dinuc_counts(characters(X_shuf[i, j],
				alphabet=alphabet))

			for key, value in dinucs_shuffled.items():
				assert dinucs[key] == value


def test_dinucleotide_shuffle_start_end():
	X = random_one_hot((8, 4, 50), random_state=0)
	X_shuf = dinucleotide_shuffle(X, start=10, end=30, random_state=0)

	assert (X[:, :, :10].unsqueeze(1) == X_shuf[:, :, :, :10]).all()
	assert (X[:, :, 30:].unsqueeze(1) == X_shuf[:, :, :, 30:]).all()
	assert (X[:, :, 10:30].unsqueeze(1) != X_shuf[:, :, :, 10:30]).any()

	for i in range(X.shape[0]):
		seq = characters(X[i, :, 10:30])

		dinucs = collections.defaultdict(int)
		for j in range(len(seq)-1):
			dinucs[seq[j:j+2]] += 1

		for j in range(20):
			dinucs_shuffled = collections.defaultdict(int)
			dinucs_seq = characters(X_shuf[i, j, :, 10:30])

			for k in range(len(seq)-1):
				dinucs_shuffled[dinucs_seq[k:k+2]] += 1

			for key, value in dinucs_shuffled.items():
				assert dinucs[key] == value


@pytest.mark.parametrize("seq", ['AATA', 'A' * 30])
def test_dinucleotide_shuffle_raises_identical(seq):
	motif = one_hot_encode(seq).unsqueeze(0)
	with pytest.raises(ValueError, match="identical"):
		dinucleotide_shuffle(motif)


def test_dinucleotide_shuffle_raises_ohe():
	seq = 'ATATATTAAAATTATTATATATTTATATATTTAAAAATTTTTAATA'
	motif = one_hot_encode(seq).unsqueeze(0)

	assert_raises(ValueError, dinucleotide_shuffle, motif + 1)
	assert_raises(ValueError, dinucleotide_shuffle, motif.unsqueeze(0))
	assert_raises(ValueError, dinucleotide_shuffle, motif[0])
	assert_raises(ValueError, dinucleotide_shuffle, "ACGTCACGATC")


def test_dinucleotide_shuffle_raises_N():
	# Deliberately stricter than the rest of this module: unknown characters
	# are rejected unless `allow_N=True` is passed.
	seq = 'ATATATTAAAATNNNATTTAAANNNTTTTTAATA'
	motif = one_hot_encode(seq).unsqueeze(0)
	assert_raises(ValueError, dinucleotide_shuffle, motif)
	assert_raises(ValueError, dinucleotide_shuffle, motif, allow_N=False)


def test_dinucleotide_shuffle_allow_N():
	seq = 'CATNCACGCNNATACGTTAGCNCGATTACANNNNNGCATGCAN'
	X = one_hot_encode(seq).unsqueeze(0)
	X_shuf = dinucleotide_shuffle(X, n=20, random_state=0, allow_N=True)
	assert X_shuf.shape == (1, 20, 4, len(seq))
	assert X_shuf.dtype == torch.int8
	assert (X_shuf != X.unsqueeze(1)).any()

	dinucs = _dinuc_counts(seq)
	for j in range(20):
		seq_shuf = characters(X_shuf[0, j], allow_N=True)
		assert seq_shuf.count('N') == seq.count('N')
		assert _dinuc_counts(seq_shuf) == dinucs


def test_dinucleotide_shuffle_allow_N_matches_N_channel():
	# An all-zero column is shuffled as though N had a channel of its own.
	seq = 'CATNCACGCNNATACGTTAGCNCGATTACANNNNNGCATGCAN'
	X = one_hot_encode(seq).unsqueeze(0)
	X5 = one_hot_encode(seq, alphabet=['A', 'C', 'G', 'T', 'N'],
		ignore=[]).unsqueeze(0)

	X_shuf = dinucleotide_shuffle(X, n=20, random_state=0, allow_N=True)
	X5_shuf = dinucleotide_shuffle(X5, n=20, random_state=0)
	assert (X_shuf == X5_shuf[:, :, :4]).all()


def test_dinucleotide_shuffle_allow_N_run():
	seq = 'CATCACGCATACGTTAGCACGATTNNNNNNNNNNACAGCATGCACGTTAGCATCGAT'
	X = one_hot_encode(seq).unsqueeze(0)
	X_shuf = dinucleotide_shuffle(X, n=20, random_state=0, allow_N=True)

	run_starts = set()
	for j in range(20):
		seq_shuf = characters(X_shuf[0, j], allow_N=True)
		run_start = seq_shuf.index('N')
		assert seq_shuf.count('N') == 10
		assert seq_shuf[run_start:run_start+10] == 'N' * 10
		run_starts.add(run_start)

	assert len(run_starts) > 1


def test_dinucleotide_shuffle_allow_N_batch():
	X = random_one_hot((8, 4, 60), random_state=0)
	for i in range(8):
		X[i, :, 5*i:5*i+4] = 0
		X[i, :, 59-i] = 0

	X_shuf = dinucleotide_shuffle(X, n=10, random_state=0, allow_N=True)
	assert X_shuf.shape == (8, 10, 4, 60)

	for i in range(8):
		seq = characters(X[i], allow_N=True)
		dinucs = _dinuc_counts(seq)

		for j in range(10):
			seq_shuf = characters(X_shuf[i, j], allow_N=True)
			assert seq_shuf.count('N') == 5
			assert _dinuc_counts(seq_shuf) == dinucs


def test_dinucleotide_shuffle_allow_N_start_end():
	seq = 'NNNNNCATCACGCANNACGTTAGCACGATNCAGCATGCACNNNNN'
	X = one_hot_encode(seq).unsqueeze(0)
	X_shuf = dinucleotide_shuffle(X, start=10, end=35, n=20, random_state=0,
		allow_N=True)

	assert (X[:, :, :10].unsqueeze(1) == X_shuf[:, :, :, :10]).all()
	assert (X[:, :, 35:].unsqueeze(1) == X_shuf[:, :, :, 35:]).all()

	dinucs = _dinuc_counts(seq[10:35])
	for j in range(20):
		seq_shuf = characters(X_shuf[0, j, :, 10:35], allow_N=True)
		assert _dinuc_counts(seq_shuf) == dinucs


def test_dinucleotide_shuffle_allow_N_without_N():
	X = random_one_hot((4, 4, 50), random_state=0)
	X_shuf = dinucleotide_shuffle(X, n=10, random_state=0)
	X_shuf_N = dinucleotide_shuffle(X, n=10, random_state=0, allow_N=True)
	assert (X_shuf == X_shuf_N).all()


def test_dinucleotide_shuffle_allow_N_dtypes():
	seq = 'CATNCACGCNNATACGTTAGCNCGATTACANNNNNGCATGCAN'
	X = one_hot_encode(seq).unsqueeze(0)
	X_shuf = dinucleotide_shuffle(X, n=5, random_state=0, allow_N=True)

	for dtype in torch.uint8, torch.int64, torch.float16, torch.float32, \
		torch.float64:
		X_shuf_dtype = dinucleotide_shuffle(X.to(dtype), n=5, random_state=0,
			allow_N=True)
		assert X_shuf_dtype.dtype == dtype
		assert (X_shuf_dtype == X_shuf.to(dtype)).all()


def test_dinucleotide_shuffle_allow_N_device(device):
	seq = 'CATNCACGCNNATACGTTAGCNCGATTACANNNNNGCATGCAN'
	X = one_hot_encode(seq).unsqueeze(0)
	X_shuf = dinucleotide_shuffle(X, n=5, random_state=0, allow_N=True)
	X_shuf_device = dinucleotide_shuffle(X.to(device), n=5, random_state=0,
		allow_N=True)

	assert X_shuf_device.device.type == device
	assert (X_shuf_device.cpu() == X_shuf).all()


def test_dinucleotide_shuffle_allow_N_raises_all_N():
	# An all-N sequence has only itself as a shuffle, like a homopolymer.
	X = torch.zeros(1, 4, 30, dtype=torch.int8)
	assert_raises(ValueError, dinucleotide_shuffle, X, allow_N=True)


def test_dinucleotide_shuffle_random_states():
	X = random_one_hot((6, 4, 50), random_state=0)
	seeds = [11, 3, 3, 250, 0, 9999999]
	X_shuf = dinucleotide_shuffle(X, n=4, random_state=seeds)
	assert X_shuf.shape == (6, 4, 4, 50)

	for i, seed in enumerate(seeds):
		X_shuf_i = dinucleotide_shuffle(X[i:i+1], n=4, random_state=seed)
		assert (X_shuf[i] == X_shuf_i[0]).all()

	for seeds in numpy.array(seeds), torch.tensor(seeds):
		assert (dinucleotide_shuffle(X, n=4, random_state=seeds) == X_shuf).all()


def test_dinucleotide_shuffle_random_states_match_int():
	X = random_one_hot((5, 4, 40), random_state=0)
	X_shuf = dinucleotide_shuffle(X, n=3, random_state=7)
	X_shuf_seeds = dinucleotide_shuffle(X, n=3, random_state=range(7, 12))
	assert (X_shuf == X_shuf_seeds).all()


def test_dinucleotide_shuffle_random_states_allow_N():
	X = random_one_hot((3, 4, 40), random_state=0)
	X[:, :, 10:13] = 0
	X_shuf = dinucleotide_shuffle(X, n=2, random_state=[5, 1, 5], allow_N=True)

	for i, seed in enumerate([5, 1, 5]):
		X_shuf_i = dinucleotide_shuffle(X[i:i+1], n=2, random_state=seed,
			allow_N=True)
		assert (X_shuf[i] == X_shuf_i[0]).all()


def test_dinucleotide_shuffle_raises_random_states():
	X = random_one_hot((3, 4, 40), random_state=0)
	assert_raises(ValueError, dinucleotide_shuffle, X, random_state=[1, 2])
	assert_raises(ValueError, dinucleotide_shuffle, X,
		random_state=[[1, 2, 3]])
	assert_raises(ValueError, dinucleotide_shuffle, X, random_state=1.5)
	assert_raises(ValueError, dinucleotide_shuffle, X,
		random_state=[1.0, 2.0, 3.0])


def test_dinucleotide_shuffle_two_characters():
	# A region of one or two characters keeps both ends, so its only shuffle
	# is itself: one shuffle returns it, and more than one raises.
	for seq in 'A', 'AC':
		X = one_hot_encode(seq).unsqueeze(0)
		X_shuf = dinucleotide_shuffle(X, n=1, random_state=0)
		assert X_shuf.shape == (1, 1, 4, len(seq))
		assert (X_shuf[:, 0] == X).all()

		assert_raises(ValueError, dinucleotide_shuffle, X, n=3)


def _one_hot_with_N(n, length, random_state=0):
	"""Random sequences with unknown characters at the start of the first, in
	a run in the middle of the second, at the end of the third and scattered
	through the fourth."""

	X = random_one_hot((n, 4, length), random_state=random_state)
	X[0, :, :3] = 0
	X[1, :, length // 3:length // 3 + 6] = 0
	if n > 2:
		X[2, :, -2:] = 0
	if n > 3:
		X[3, :, ::9] = 0
	return X


@pytest.mark.parametrize("length", [48, 61, 62, 75])
def test_dinucleotide_shuffle_return_packed(length):
	X = random_one_hot((6, 4, length), random_state=length)
	X_shuf = dinucleotide_shuffle(X, n=4, random_state=3)
	X_packed = dinucleotide_shuffle(X, n=4, random_state=3, return_packed=True)

	assert isinstance(X_packed, PackedSequence)
	assert X_packed.shape == (6, 4, 4, length)
	assert X_packed.data.shape == (6, 4, (length + 3) // 4)
	assert X_packed.alphabet == ('A', 'C', 'G', 'T')
	assert len(X_packed.n_runs) == 0
	assert torch.equal(X_packed.to_one_hot(), X_shuf)
	assert_packed_equal(X_packed, PackedSequence.from_one_hot(X_shuf))


@pytest.mark.parametrize("dtype", [torch.int8, torch.float32, torch.float64])
def test_dinucleotide_shuffle_return_packed_allow_N(dtype):
	X = _one_hot_with_N(5, 61).to(dtype)
	X_shuf = dinucleotide_shuffle(X, n=3, random_state=8, allow_N=True)
	X_packed = dinucleotide_shuffle(X, n=3, random_state=8, allow_N=True,
		return_packed=True)

	assert torch.equal(X_packed.to_one_hot(dtype=dtype), X_shuf)
	assert_packed_equal(X_packed, PackedSequence.from_one_hot(X_shuf))


def test_dinucleotide_shuffle_packed_input():
	X = _one_hot_with_N(5, 61)
	X_shuf = dinucleotide_shuffle(X, n=3, random_state=8, allow_N=True)
	X_packed = PackedSequence.from_one_hot(X, alphabet=('A', 'C', 'G', 'U'))

	X_shuf_packed = dinucleotide_shuffle(X_packed, n=3, random_state=8,
		allow_N=True)
	assert isinstance(X_shuf_packed, PackedSequence)
	assert X_shuf_packed.alphabet == ('A', 'C', 'G', 'U')
	assert_packed_equal(X_shuf_packed, PackedSequence.from_one_hot(X_shuf,
		alphabet=('A', 'C', 'G', 'U')))

	X_shuf_one_hot = dinucleotide_shuffle(X_packed, n=3, random_state=8,
		allow_N=True, return_packed=False)
	assert X_shuf_one_hot.dtype == torch.int8
	assert torch.equal(X_shuf_one_hot, X_shuf)

	# The input is not modified.
	assert torch.equal(X_packed.to_one_hot(), X)


@pytest.mark.parametrize("start, end, n", [(0, -1, 3), (10, 40, 3),
	(-25, -1, 3), (20, -20, 3), (15, 35, 3), (3, 4, 1), (3, 5, 1),
	(55, 70, 1), (40, 10, 1)])
def test_dinucleotide_shuffle_packed_start_end(start, end, n):
	# Runs of unknown characters cross both ends of the region (10, 40).
	X = _one_hot_with_N(5, 61)
	X[4, :, 8:12] = 0
	X[4, :, 38:43] = 0

	X_shuf = dinucleotide_shuffle(X, start=start, end=end, n=n, random_state=2,
		allow_N=True)
	X_packed = dinucleotide_shuffle(X, start=start, end=end, n=n,
		random_state=2, allow_N=True, return_packed=True)
	assert_packed_equal(X_packed, PackedSequence.from_one_hot(X_shuf))


def test_dinucleotide_shuffle_packed_random_states():
	X = _one_hot_with_N(5, 50)
	seeds = [11, 3, 3, 250, 0]
	X_packed = dinucleotide_shuffle(X, n=4, random_state=seeds, allow_N=True,
		return_packed=True)

	for i, seed in enumerate(seeds):
		X_packed_i = dinucleotide_shuffle(X[i:i+1], n=4, random_state=seed,
			allow_N=True, return_packed=True)
		assert_packed_equal(X_packed[i:i+1], X_packed_i)


def test_dinucleotide_shuffle_packed_n_zero():
	X = _one_hot_with_N(3, 50)
	X_packed = dinucleotide_shuffle(X, n=0, random_state=0, allow_N=True,
		return_packed=True)

	assert X_packed.shape == (3, 0, 4, 50)
	assert X_packed.data.shape == (3, 0, 13)
	assert len(X_packed.n_runs) == 0
	assert torch.equal(X_packed.to_one_hot(), dinucleotide_shuffle(X, n=0,
		random_state=0, allow_N=True))


def test_dinucleotide_shuffle_packed_two_characters():
	for seq in 'A', 'AC', 'NC', 'NN':
		X = one_hot_encode(seq).unsqueeze(0)
		X_packed = dinucleotide_shuffle(X, n=1, random_state=0, allow_N=True,
			return_packed=True)
		assert torch.equal(X_packed.to_one_hot()[:, 0], X)

		with pytest.raises(ValueError, match="identical"):
			dinucleotide_shuffle(X, n=3, allow_N=True, return_packed=True)


def test_dinucleotide_shuffle_packed_raises():
	with pytest.raises(ValueError, match="four characters"):
		dinucleotide_shuffle(random_one_hot((2, 5, 30), random_state=0),
			return_packed=True)

	X_packed = PackedSequence.from_one_hot(_one_hot_with_N(2, 30))
	with pytest.raises(ValueError, match="unknown"):
		dinucleotide_shuffle(X_packed)

	assert_raises(ValueError, dinucleotide_shuffle, X_packed.reshape(1, 2),
		allow_N=True)
	assert_raises(ValueError, dinucleotide_shuffle, X_packed[0], allow_N=True)
	assert_raises(ValueError, dinucleotide_shuffle, X_packed, allow_N=True,
		random_state=[1, 2, 3])

	X_homopolymer = PackedSequence.from_one_hot(one_hot_encode('A' * 20
		).unsqueeze(0))
	with pytest.raises(ValueError, match="identical"):
		dinucleotide_shuffle(X_homopolymer, n=3)


@pytest.mark.parametrize("return_packed", [False, True])
def test_dinucleotide_shuffle_packed_verbose(return_packed):
	# One shuffle keeps every position, which warns under verbose=True.
	X = random_one_hot((2, 4, 30), random_state=0)
	with pytest.warns(TangermemeWarning, match="identical across") as record:
		dinucleotide_shuffle(X, n=1, verbose=True, return_packed=return_packed)

	assert record[0].filename == __file__

	with warnings.catch_warnings():
		warnings.simplefilter("error")
		dinucleotide_shuffle(X, n=1, return_packed=return_packed)


@pytest.mark.parametrize("n_jobs", [2, 4, -1])
@pytest.mark.parametrize("return_packed", [False, True])
def test_dinucleotide_shuffle_n_jobs(n_jobs, return_packed, monkeypatch):
	# The threshold is lifted so that these small calls use the threads.
	monkeypatch.setattr("tangermeme.ersatz._SHUFFLE_PARALLEL_MIN", 0)
	X = _one_hot_with_N(9, 75, random_state=2)
	X_shuf = dinucleotide_shuffle(X, n=3, random_state=4, allow_N=True,
		return_packed=return_packed)

	n_threads = numba.get_num_threads()
	X_shuf_jobs = dinucleotide_shuffle(X, n=3, random_state=4, allow_N=True,
		return_packed=return_packed, n_jobs=n_jobs)
	assert numba.get_num_threads() == n_threads

	if return_packed:
		assert_packed_equal(X_shuf_jobs, X_shuf)
	else:
		assert X_shuf_jobs.dtype == X_shuf.dtype
		assert torch.equal(X_shuf_jobs, X_shuf)


def test_dinucleotide_shuffle_n_jobs_raises(monkeypatch):
	# The check for identical shuffles still covers every sequence.
	monkeypatch.setattr("tangermeme.ersatz._SHUFFLE_PARALLEL_MIN", 0)
	X = random_one_hot((4, 4, 30), random_state=0)
	X[2] = one_hot_encode('A' * 30)
	assert_raises(ValueError, dinucleotide_shuffle, X, n=3, n_jobs=2)
	assert_raises(ValueError, dinucleotide_shuffle, X, n=3, n_jobs=2,
		return_packed=True)


@pytest.mark.parametrize("return_packed", [False, True])
def test_dinucleotide_shuffle_n_jobs_threshold(return_packed, monkeypatch):
	# A call that shuffles fewer than _SHUFFLE_PARALLEL_MIN positions in all
	# runs the serial kernel whatever n_jobs is, and a larger one the
	# parallel kernel, with the same shuffles.
	import tangermeme.ersatz

	name = "_fast_shuffle_packed_parallel" if return_packed else \
		"_fast_shuffle_parallel"
	kernel, calls = getattr(tangermeme.ersatz, name), []

	def counted(*args):
		calls.append(1)
		return kernel(*args)

	monkeypatch.setattr(tangermeme.ersatz, name, counted)
	X = _one_hot_with_N(4, 400, random_state=1)

	# 4 sequences x 2 shuffles x 399 positions = 3,192 < 4,096.
	X_small = dinucleotide_shuffle(X, end=-2, n=2, random_state=0,
		allow_N=True, return_packed=return_packed, n_jobs=4)
	assert len(calls) == 0

	# 4 sequences x 3 shuffles x 400 positions = 4,800.
	X_large = dinucleotide_shuffle(X, n=3, random_state=0, allow_N=True,
		return_packed=return_packed, n_jobs=4)
	assert len(calls) == 1

	for X_, kw in ((X_small, dict(end=-2, n=2)), (X_large, dict(n=3))):
		X_serial = dinucleotide_shuffle(X, random_state=0, allow_N=True,
			return_packed=return_packed, n_jobs=1, **kw)
		if return_packed:
			assert_packed_equal(X_, X_serial)
		else:
			assert torch.equal(X_, X_serial)

	assert len(calls) == 1


@pytest.mark.parametrize("return_packed", [False, True])
def test_dinucleotide_shuffle_n_jobs_default(return_packed, monkeypatch):
	# By default the shuffle stays on one thread however large the call, so
	# numba's threading layer is not started: under its OpenMP layer, a
	# process that has started it cannot fork worker processes afterwards.
	import tangermeme.ersatz

	def fail(*args):
		raise AssertionError("the parallel kernel ran")

	monkeypatch.setattr(tangermeme.ersatz, "_fast_shuffle_parallel", fail)
	monkeypatch.setattr(tangermeme.ersatz, "_fast_shuffle_packed_parallel",
		fail)

	X = random_one_hot((16, 4, 500), random_state=0)
	dinucleotide_shuffle(X, n=5, random_state=0, return_packed=return_packed)


def test_dinucleotide_shuffle_packed_device(device):
	X = _one_hot_with_N(4, 50)
	X_packed = dinucleotide_shuffle(X.to(device), n=3, random_state=0,
		allow_N=True, return_packed=True)

	assert X_packed.data.device.type == 'cpu'
	assert_packed_equal(X_packed, dinucleotide_shuffle(X, n=3,
		random_state=0, allow_N=True, return_packed=True))


def test_fast_shuffle_packed_flags_wrong_runs():
	# Telling the kernel that the sequence has no runs of unknown characters
	# when it has one makes it flag the shuffle rather than write past them.
	codes = numpy.array([[0, 1, 4, 4, 2, 3, 0, 1, 2, 3]], dtype=numpy.uint8)
	data = numpy.zeros((1, 2, 3), dtype=numpy.uint8)
	flags = numpy.zeros((1, 3), dtype=numpy.bool_)
	run_ptr = numpy.zeros(2, dtype=numpy.int64)
	runs = numpy.zeros(0, dtype=numpy.int64)

	_fast_shuffle_packed(codes, 0, 10, 2, numpy.array([0], dtype=numpy.int32),
		run_ptr, data, runs, runs.copy(), flags)
	assert flags[0, 2]


def test_dinucleotide_shuffle_private(device):
	# gReLU imports _dinucleotide_shuffle: one sequence in, and float32
	# shuffles on the CPU out, those of dinucleotide_shuffle with unknown
	# characters allowed.
	X = random_one_hot((1, 4, 60), random_state=0).type(torch.int8)
	X[:, :, 10:13] = 0
	expected = dinucleotide_shuffle(X, n=3, random_state=7, allow_N=True)[0]

	X_shuf = _dinucleotide_shuffle(X[0].to(device), n_shuffles=3,
		random_state=7)
	assert X_shuf.shape == (3, 4, 60)
	assert X_shuf.dtype == torch.float32
	assert X_shuf.device.type == "cpu"
	assert (X_shuf == expected.float()).all()

	X_shuf = _dinucleotide_shuffle(X[0, :, 20:40], random_state=7)
	assert X_shuf.shape == (1, 4, 20)


###


def _local_bins(first_cut, seq_len, bin_size, min_bin_size):
	"""Re-derive the bin boundaries used by `local_dinucleotide_shuffle`.

	Mirrors the boundary construction in the implementation so that tests can
	check per-bin properties without depending on the RNG draw order.
	"""

	boundaries = [0, first_cut]
	boundaries.extend(range(first_cut + bin_size, seq_len, bin_size))
	boundaries.append(seq_len)

	bins = list(zip(boundaries[:-1], boundaries[1:]))
	if bins[-1][1] - bins[-1][0] < min_bin_size:
		bins[-2] = (bins[-2][0], bins[-1][1])
		bins.pop()

	return bins


def _dinuc_counts(seq):
	"""Count the dinucleotides in a string."""

	dinucs = collections.defaultdict(int)
	for i in range(len(seq)-1):
		dinucs[seq[i:i+2]] += 1

	return dinucs


@pytest.mark.parametrize("n_characters, n", [(4, 5), (20, 3)])
def test_local_dinucleotide_shuffle(n_characters, n):
	# Any one-hot alphabet is accepted, matching dinucleotide_shuffle, which
	# this function otherwise mirrors.
	X = random_one_hot((2, n_characters, 200), random_state=0)
	X_shuf = local_dinucleotide_shuffle(X, n=n, bin_size=64, min_bin_size=32,
		random_state=0)

	assert X_shuf.shape == (2, n, n_characters, 200)
	assert X_shuf.dtype == X.dtype

	# The output is still one-hot encoded and the per-sequence nucleotide
	# composition is preserved because shuffling only permutes positions.
	assert (X_shuf.sum(dim=2) == 1).all()
	assert (X_shuf.sum(dim=-1) == X.sum(dim=-1).unsqueeze(1)).all()


def test_local_dinucleotide_shuffle_default_n():
	X = random_one_hot((1, 4, 3000), random_state=0)
	X_shuf = local_dinucleotide_shuffle(X, random_state=0)

	assert X_shuf.shape == (1, 20, 4, 3000)


def test_local_dinucleotide_shuffle_changes_sequence():
	X = random_one_hot((2, 4, 200), random_state=0)
	X_shuf = local_dinucleotide_shuffle(X, n=5, bin_size=64, min_bin_size=32,
		random_state=0)

	# Every shuffle should differ from the original somewhere, and the
	# shuffles should not all be identical to each other.
	for i in range(X.shape[0]):
		for j in range(5):
			assert (X_shuf[i, j] != X[i]).any()

		assert (X_shuf[i, 0] != X_shuf[i, 1]).any()


def test_local_dinucleotide_shuffle_composition():
	# Each bin is shuffled independently, so the dinucleotide composition is
	# only conserved within a bin, not across bin boundaries, and a
	# dinucleotide shuffle keeps the first and last character of the region
	# it is applied to, so the bin endpoints are conserved too. Rather than
	# guessing the cut points, check that *some* valid set of bins explains
	# both in each shuffle: the first cut is the only random boundary choice.
	seq_len, bin_size, min_bin_size = 200, 64, 32

	X = random_one_hot((2, 4, seq_len), random_state=0)
	X_shuf = local_dinucleotide_shuffle(X, n=3, bin_size=bin_size,
		min_bin_size=min_bin_size, random_state=0)

	cuts = range(min_bin_size, bin_size + 1)

	for i in range(X.shape[0]):
		seq = characters(X[i])

		for j in range(X_shuf.shape[1]):
			shuf_seq = characters(X_shuf[i, j])

			matches = []
			for first_cut in cuts:
				bins = _local_bins(first_cut, seq_len, bin_size, min_bin_size)
				idxs = [s for s, _ in bins] + [e-1 for _, e in bins]

				if all(_dinuc_counts(seq[s:e]) == _dinuc_counts(shuf_seq[s:e])
					for s, e in bins) and (X_shuf[i, j, :, idxs] ==
					X[i, :, idxs]).all():
					matches.append(first_cut)

			assert len(matches) > 0, ("No bin layout preserves the per-bin "
				"dinucleotide composition and endpoints of shuffle "
				"{}".format(j))


def test_local_dinucleotide_shuffle_bin_layout_valid():
	# Every bin the implementation can produce must be at least min_bin_size
	# long and no longer than 2 * bin_size (the merge of the final two bins is
	# the only way a bin exceeds bin_size).
	import tangermeme.ersatz

	seq_len, bin_size, min_bin_size = 200, 64, 32

	for seed in range(200):
		rng = numpy.random.RandomState(seed)
		bins = tangermeme.ersatz._local_bins(rng, seq_len, bin_size,
			min_bin_size)

		assert bins[0][0] == 0
		assert bins[-1][1] == seq_len

		for k in range(len(bins)-1):
			assert bins[k][1] == bins[k+1][0]  # bins tile the sequence

		for s, e in bins:
			assert e - s >= min_bin_size
			assert e - s <= 2 * bin_size


def test_local_dinucleotide_shuffle_conserves_gc_profile():
	# The reason this function exists: a whole-sequence dinucleotide shuffle
	# conserves composition globally but flattens how it varies along the
	# sequence, and a genomic window is rarely uniform. On a sequence whose two
	# halves sit at ~0.80 and ~0.18 GC, 128 bp bins hold the GC profile of every
	# 256 bp window to within 0.10, where a whole-sequence shuffle moves it by
	# 0.42.
	numpy.random.seed(0)
	gc_rich = "".join(numpy.random.choice(list("ACGT"), 1024,
		p=[.1, .4, .4, .1]))
	at_rich = "".join(numpy.random.choice(list("ACGT"), 1024,
		p=[.4, .1, .1, .4]))
	X = one_hot_encode(gc_rich + at_rich).unsqueeze(0).type(torch.float32)

	def gc_profile(x, w=256):
		gc = x[1] + x[2]
		return torch.stack([gc[i:i+w].mean()
			for i in range(0, x.shape[-1], w)])

	X_local = local_dinucleotide_shuffle(X, n=5, bin_size=128, min_bin_size=64,
		random_state=0)
	X_global = dinucleotide_shuffle(X, n=5, random_state=0)

	profile = gc_profile(X[0])
	local = max(float((gc_profile(X_local[0, j]) - profile).abs().max())
		for j in range(5))
	global_ = max(float((gc_profile(X_global[0, j]) - profile).abs().max())
		for j in range(5))

	assert local < 0.15
	assert global_ > 0.35
	assert local < global_


def test_local_dinucleotide_shuffle_random_state():
	X = random_one_hot((2, 4, 200), random_state=0)

	X_shuf0 = local_dinucleotide_shuffle(X, n=3, bin_size=64, min_bin_size=32,
		random_state=0)
	X_shuf1 = local_dinucleotide_shuffle(X, n=3, bin_size=64, min_bin_size=32,
		random_state=0)
	X_shuf2 = local_dinucleotide_shuffle(X, n=3, bin_size=64, min_bin_size=32,
		random_state=1)

	assert_array_almost_equal(X_shuf0, X_shuf1)
	assert (X_shuf0 != X_shuf2).any()


def test_local_dinucleotide_shuffle_does_not_modify_input():
	X = random_one_hot((2, 4, 200), random_state=0)
	X_orig = X.clone()

	local_dinucleotide_shuffle(X, n=3, bin_size=64, min_bin_size=32,
		random_state=0)

	assert_array_almost_equal(X, X_orig)


def test_local_dinucleotide_shuffle_single_bin_edge():
	# bin_size = seq_len - 1 is the largest legal bin size, giving a two-bin
	# layout that is immediately merged back into one bin.
	X = random_one_hot((1, 4, 100), random_state=0)
	X_shuf = local_dinucleotide_shuffle(X, n=2, bin_size=99, min_bin_size=50,
		random_state=0)

	assert X_shuf.shape == (1, 2, 4, 100)

	seq = characters(X[0])
	for j in range(2):
		assert _dinuc_counts(characters(X_shuf[0, j])) == _dinuc_counts(seq)


def test_local_dinucleotide_shuffle_warns_unshuffled():
	# A tandem repeat admits a single Eulerian path, so its dinucleotide
	# shuffle is itself. `dinucleotide_shuffle` raises on that, but only when
	# asked for more than one shuffle, and this function asks for one per bin.
	X = one_hot_encode("CAG" * 341 + "C").unsqueeze(0)

	with pytest.warns(TangermemeWarning):
		X_shuf = local_dinucleotide_shuffle(X, n=2, bin_size=256,
			min_bin_size=128, random_state=0)

	# The unshuffled sequence is still returned, not replaced or dropped.
	assert X_shuf.shape == (1, 2, 4, 1024)
	assert (X_shuf[0, 0] == X[0]).all()


def test_local_dinucleotide_shuffle_warns_partially_unshuffled():
	# One repeat inside an otherwise ordinary sequence warns about that bin
	# alone rather than the whole call.
	numpy.random.seed(0)
	flank = "".join(numpy.random.choice(list("ACGT"), 800))
	X = one_hot_encode(flank + "CAG" * 133 + "C" + flank).unsqueeze(0)

	with pytest.warns(TangermemeWarning, match="2 of 16 bins"):
		local_dinucleotide_shuffle(X, n=2, bin_size=256, min_bin_size=128,
			random_state=0)


def test_local_dinucleotide_shuffle_no_warning():
	X = random_one_hot((2, 4, 2000), random_state=0)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=TangermemeWarning)

		local_dinucleotide_shuffle(X, n=3, bin_size=256, min_bin_size=128,
			random_state=0)


def test_local_dinucleotide_shuffle_raises_shape():
	X = random_one_hot((2, 4, 200), random_state=0)

	assert_raises(ValueError, local_dinucleotide_shuffle, X[0], 20, 64, 32)
	assert_raises(ValueError, local_dinucleotide_shuffle, X.unsqueeze(0), 20,
		64, 32)

	# Validation is on the encoding, not the alphabet size, so what is rejected
	# here is a tensor that is not one-hot rather than one that is not DNA.
	assert_raises(ValueError, local_dinucleotide_shuffle, torch.randn(2, 4, 200),
		20, 64, 32)


def test_local_dinucleotide_shuffle_raises_bin_size():
	X = random_one_hot((2, 4, 200), random_state=0)

	# bin_size must be strictly smaller than the sequence length.
	assert_raises(ValueError, local_dinucleotide_shuffle, X, 20, 200)
	assert_raises(ValueError, local_dinucleotide_shuffle, X, 20, 300)


def test_local_dinucleotide_shuffle_raises_min_bin_size():
	X = random_one_hot((2, 4, 200), random_state=0)

	assert_raises(ValueError, local_dinucleotide_shuffle, X, 20, 64, 65)


###


def test_insert_batched_motif(X):
	X_batch = X.repeat(4, 1, 1)
	motifs = torch.stack([
		one_hot_encode('CATCAG'),
		one_hot_encode('GTGTGT'),
		one_hot_encode('AAAAAA'),
		one_hot_encode('TGCATG'),
	])

	X_insert = insert(X_batch, motifs, start=10)

	assert X_insert.shape[0] == 4
	assert X_insert.shape[-1] == X.shape[-1] + 6

	assert_array_almost_equal(X_insert[:, :, 10:16], motifs)


def test_substitute_end_eq_seq_length(X):
	motif = 'CATCAG'
	start = X.shape[-1] - len(motif)

	X_sub = substitute(X, motif, start=start)

	assert X_sub.shape == X.shape
	assert_array_almost_equal(X_sub[:, :, :start], X[:, :, :start])
	assert_array_almost_equal(X_sub[:, :, start:],
		one_hot_encode(motif).unsqueeze(0))


def test_substitute_batch_mismatch(X):
	X_batch = X.repeat(4, 1, 1)
	motif = torch.stack([
		one_hot_encode('CATCAG'),
		one_hot_encode('GTGTGT'),
		one_hot_encode('AAAAAA'),
	])

	assert_raises(RuntimeError, substitute, X_batch, motif, 10)


def test_multisubstitute_empty_motifs(X):
	assert_raises(ValueError, multisubstitute, X, [], spacing=[])


def test_multisubstitute_single_motif(X):
	X_sub = multisubstitute(X, ['CATCAG'], spacing=[])

	X_ref = substitute(X, 'CATCAG')
	assert_array_almost_equal(X_sub, X_ref)


def test_multisubstitute_explicit_start_overflow(X):
	assert_raises(ValueError, multisubstitute, X, ['CATCAG', 'CATCAG'],
		spacing=0, start=X.shape[-1] - 1)


def test_delete_end_eq_seq_length(X):
	X_del = delete(X, start=0, end=X.shape[-1])

	assert X_del.shape == (X.shape[0], X.shape[1], 0)


def test_randomize_n_zero(X):
	assert_raises(RuntimeError, randomize, X, 5, 10, n=0)


def test_shuffle_n_zero(X):
	assert_raises(RuntimeError, shuffle, X, 5, 10, n=0)


def test_shuffle_negative_end_one_off_check(X):
	L = X.shape[-1]

	s_neg = shuffle(X, 0, -1, random_state=0)
	s_pos = shuffle(X, 0, L, random_state=0)

	assert_array_almost_equal(s_neg, s_pos)


def test_substitute_returns_clone_not_view(X):
	X_orig = X.clone()
	X_sub = substitute(X, 'CATCAG', start=5)

	X_sub[0, 0, 5] = 99
	assert_array_almost_equal(X, X_orig)


@pytest.mark.parametrize("func", [insert, substitute])
def test_ersatz_raises_ohe(X, func):
	motif = one_hot_encode('CATCAG').unsqueeze(0)
	motif[0, 0, 0] = 1
	assert_raises(ValueError, func, X, motif)

	motif = one_hot_encode('CATCAG').unsqueeze(0)
	motif[0, 1, 0] = 2
	assert_raises(ValueError, func, X, motif)
	assert_raises(ValueError, func, X, torch.randn(1, 4, 8))


@pytest.mark.parametrize("call", [
	lambda X: insert(X, 'CATCAG'),
	lambda X: substitute(X, 'CATCAG'),
	lambda X: delete(X, start=0, end=5),
	lambda X: randomize(X, start=20, end=30),
	lambda X: shuffle(X, start=5, end=20),
], ids=["insert", "substitute", "delete", "randomize", "shuffle"])
def test_ersatz_raises_ohe_X(X, call):
	# Allowing all-zero columns must not also allow multi-hot columns or
	# values outside {0, 1}.
	X_multi = torch.clone(X)
	X_multi[0, :, 5] = 1
	assert_raises(ValueError, call, X_multi)

	assert_raises(ValueError, call, torch.clone(X) * 2)
	assert_raises(ValueError, call, torch.randn(1, 4, 68))


def test_ersatz_ops_preserve_cuda_device(cuda_device):
	# All public ersatz ops should accept a CUDA-resident X and return a
	# tensor on the same device. Previously dinucleotide_shuffle crashed
	# in the internal .numpy() conversion.
	X = random_one_hot((2, 4, 100), random_state=0).type(
		torch.float32).to(cuda_device)
	motif = one_hot_encode("ACGT").unsqueeze(0).to(cuda_device)
	X_local = local_dinucleotide_shuffle(X, n=2, bin_size=64, min_bin_size=32,
		random_state=0)

	for name, out in [
		('substitute', substitute(X, motif)),
		('multisubstitute', multisubstitute(X, [motif, motif], [5])),
		('insert', insert(X, motif, start=20)),
		('delete', delete(X, 20, 30)),
		('shuffle', shuffle(X, start=20, end=30, n=2, random_state=0)),
		('randomize', randomize(X, start=20, end=30, n=2, random_state=0)),
		('dinucleotide_shuffle', dinucleotide_shuffle(X, n=2, random_state=0)),
		('local_dinucleotide_shuffle', X_local),
	]:
		assert out.device.type == 'cuda', f"{name} produced device {out.device}"

	assert X_local.shape == (2, 2, 4, 100)


###
# Every function in ersatz takes a PackedSequence and returns one, or packs
# its output from a one-hot X under return_packed=True, with the sequences
# of the one-hot path.


def _ersatz_X():
	X = random_one_hot((7, 4, 60), random_state=3)
	X[0, :, :4] = 0
	X[2, :, 25:33] = 0
	X[4, :, -2:] = 0
	X[5, :, ::11] = 0
	return X


_motifs_per_sequence = random_one_hot((7, 4, 5), random_state=4)

_ERSATZ_CALLS = {
	"insert": lambda X, **kw: insert(X, "ACGTT", start=20, **kw),
	"insert_default_start": lambda X, **kw: insert(X, "CAT", **kw),
	"insert_per_sequence": lambda X, **kw: insert(X, _motifs_per_sequence,
		start=3, **kw),
	"substitute": lambda X, **kw: substitute(X, "ACGTT", start=20, **kw),
	"substitute_N": lambda X, **kw: substitute(X, "ANNGT", start=24, **kw),
	"substitute_per_sequence": lambda X, **kw: substitute(X,
		_motifs_per_sequence, start=55, **kw),
	"multisubstitute": lambda X, **kw: multisubstitute(X, ["ACG", "TTNA",
		"GG"], [2, 5], start=10, **kw),
	"delete": lambda X, **kw: delete(X, 22, 35, **kw),
	"randomize": lambda X, **kw: randomize(X, 10, 30, n=3, random_state=5,
		**kw),
	"randomize_probs": lambda X, **kw: randomize(X, 0, 60, probs=[[0.5, 0.25,
		0.125, 0.125]], n=2, random_state=6, **kw),
	"shuffle": lambda X, **kw: shuffle(X, n=3, random_state=7, **kw),
	"shuffle_region": lambda X, **kw: shuffle(X, start=20, end=45, n=2,
		random_state=8, **kw),
}


@pytest.mark.parametrize("name", list(_ERSATZ_CALLS))
@pytest.mark.parametrize("chunk", [None, 50])
def test_ersatz_packed(name, chunk, monkeypatch):
	if chunk is not None:
		# Several chunks of sequences per call.
		monkeypatch.setattr("tangermeme.utils._CHUNK_POSITIONS", chunk)

	call = _ERSATZ_CALLS[name]
	X = _ersatz_X()
	X_packed = PackedSequence.from_one_hot(X)
	Y = call(X)
	Y_packed = PackedSequence.from_one_hot(Y)

	assert_packed_equal(call(X, return_packed=True), Y_packed)
	assert_packed_equal(call(X_packed), Y_packed)

	Y_one_hot = call(X_packed, return_packed=False)
	assert Y_one_hot.dtype == torch.int8
	assert torch.equal(Y_one_hot, Y.to(torch.int8))
	assert torch.equal(X_packed.to_one_hot(), X)


@pytest.mark.parametrize("name", ["substitute", "insert", "delete",
	"randomize", "shuffle", "multisubstitute"])
def test_ersatz_packed_raises_alphabet(name):
	X = random_one_hot((3, 5, 60), random_state=0)
	with pytest.raises(ValueError, match="four characters"):
		if name in ("substitute", "insert"):
			_ERSATZ_CALLS[name](X, return_packed=True, alphabet=list("ACGTU"))
		elif name == "multisubstitute":
			multisubstitute(X, ["ACG", "TT"], 3, alphabet=list("ACGTU"),
				return_packed=True)
		else:
			_ERSATZ_CALLS[name](X, return_packed=True)


def test_ersatz_packed_empty():
	X_packed = PackedSequence.from_one_hot(random_one_hot((3, 4, 60),
		random_state=0))[:0]
	Y = substitute(X_packed, "ACG", start=4)
	assert Y.shape == (0, 4, 60)
	assert insert(X_packed, "ACG", start=4).shape == (0, 4, 63)


@pytest.mark.parametrize("verbose", [False, True])
def test_local_dinucleotide_shuffle_packed(verbose):
	X = random_one_hot((3, 4, 1500), random_state=9)
	X[1, :, 200:1000] = one_hot_encode('A' * 800)
	kwargs = dict(n=2, bin_size=256, min_bin_size=128, random_state=10,
		verbose=verbose)

	with warnings.catch_warnings(record=True) as w:
		warnings.simplefilter("always")
		Y = local_dinucleotide_shuffle(X, **kwargs)

	with warnings.catch_warnings(record=True) as w_packed:
		warnings.simplefilter("always")
		Y_packed = local_dinucleotide_shuffle(PackedSequence.from_one_hot(X),
			**kwargs)

	assert_packed_equal(Y_packed, PackedSequence.from_one_hot(Y))
	assert [str(w_.message) for w_ in w_packed] == [str(w_.message) for w_ in w]
	assert any("came back identical" in str(w_.message) for w_ in w)

	assert_packed_equal(local_dinucleotide_shuffle(X, return_packed=True,
		**kwargs), Y_packed)
	assert torch.equal(local_dinucleotide_shuffle(PackedSequence.from_one_hot(
		X), return_packed=False, **kwargs), Y.to(torch.int8))


def test_local_dinucleotide_shuffle_packed_raises_N():
	X = random_one_hot((2, 4, 1500), random_state=9)
	X[0, :, 10:20] = 0
	assert_raises(ValueError, local_dinucleotide_shuffle,
		PackedSequence.from_one_hot(X), bin_size=256, min_bin_size=128)


@pytest.mark.parametrize("name", ["insert", "substitute", "multisubstitute",
	"delete", "randomize", "shuffle"])
def test_ersatz_packed_raises_batch_axes(name):
	# A PackedSequence with two batch axes is rejected, as the matching 4-D
	# one-hot tensor is, rather than edited along the wrong axis.
	X = PackedSequence.from_one_hot(random_one_hot((6, 4, 40),
		random_state=1)).reshape(2, 3)
	calls = {
		"insert": lambda: insert(X, "ACG"),
		"substitute": lambda: substitute(X, "ACG"),
		"multisubstitute": lambda: multisubstitute(X, ["AC", "GT"], [3]),
		"delete": lambda: delete(X, 1, 4),
		"randomize": lambda: randomize(X, 1, 4, random_state=0),
		"shuffle": lambda: shuffle(X, 1, 10, random_state=0),
	}

	with pytest.raises(ValueError, match="X has shape"):
		calls[name]()


@pytest.mark.parametrize("width", [1, 2, 5])
def test_randomize_packed_raises_probs_width(width):
	# probs needs one column per character, as on the one-hot path, where a
	# wrong width raises from the substitution.
	X = random_one_hot((3, 4, 20), random_state=0)
	probs = torch.full((1, width), 1.0 / width)

	assert_raises(ValueError, randomize, X, 5, 15, probs=probs)
	assert_raises(ValueError, randomize, X, 5, 15, probs=probs,
		return_packed=True)
	assert_raises(ValueError, randomize, PackedSequence.from_one_hot(X), 5, 15,
		probs=probs)


@pytest.mark.parametrize("name", list(_ERSATZ_CALLS) +
	["local_dinucleotide_shuffle"])
def test_ersatz_packed_alphabet(name):
	# A PackedSequence over another alphabet gives packed output over that
	# alphabet, with the sequences of the one-hot path.
	calls = dict(_ERSATZ_CALLS, local_dinucleotide_shuffle=lambda X, **kw:
		local_dinucleotide_shuffle(X, n=2, random_state=0, **kw))
	X = _ersatz_X()
	if name == "local_dinucleotide_shuffle":
		X = random_one_hot((3, 4, 1200), random_state=1)

	alphabet = ('A', 'C', 'G', 'U')
	Y = calls[name](PackedSequence.from_one_hot(X, alphabet=alphabet))
	assert Y.alphabet == alphabet
	assert torch.equal(Y.to_one_hot(), calls[name](X).to(torch.int8))


def test_local_dinucleotide_shuffle_packed_verbose_bins():
	# Under verbose, every bin of three or more positions warns that its
	# shuffle kept every position, as on the one-hot path, including bins of
	# exactly three.
	X = random_one_hot((2, 4, 30), random_state=0)

	def count(X_):
		with warnings.catch_warnings(record=True) as record:
			warnings.simplefilter("always")
			local_dinucleotide_shuffle(X_, n=1, bin_size=3, min_bin_size=3,
				random_state=0, verbose=True)
		return sum("identical across" in str(w.message) for w in record)

	n_warnings = count(X)
	assert n_warnings == 20
	assert count(PackedSequence.from_one_hot(X)) == n_warnings
