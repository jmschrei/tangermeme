# test_io.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest
import warnings
import collections

from tangermeme.utils import characters
from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot

from tangermeme.ersatz import insert
from tangermeme.ersatz import substitute
from tangermeme.ersatz import multisubstitute
from tangermeme.ersatz import delete
from tangermeme.ersatz import randomize
from tangermeme.ersatz import shuffle
from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.ersatz import local_dinucleotide_shuffle

from tangermeme.utils import TangermemeWarning

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


@pytest.fixture
def X():
	seq = 'CACATCATCTCATCATCTGCTGACTACTGACGTAGTCTGACTGACTGACTGACTACTGACTGACTGAC'
	return one_hot_encode(seq).unsqueeze(0)


###


def test_insert_str(X):
	motif = 'CATCAG'
	X_insert = insert(X, motif)

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


def test_insert_ohe(X):
	motif = one_hot_encode('CATCAG').unsqueeze(0)
	X_insert = insert(X, motif)

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
	assert_raises(ValueError, insert, X, motif)
	assert_raises(ValueError, insert, X, motif, None, ['A', 'C', 'G'])


def test_insert_raise_ends(X):
	assert_raises(ValueError, insert, X, 'CAGCAT', -2)
	assert_raises(ValueError, insert, X, 'CAGCAT', 1000)
	assert_raises(TypeError, insert, X, 'CAGCAT', 6.5)
	assert_raises(TypeError, insert, X, 'CAGCAT', 5, 3)
	assert_raises(TypeError, insert, X, 'CAGCAT', -5, -1)
	assert_raises(TypeError, insert, X, 'CAGCAT', 5, 1000)


def test_insert_raises_ohe(X):
	motif = one_hot_encode('CATCAG').unsqueeze(0)
	motif[0, 0, 0] = 1
	assert_raises(ValueError, insert, X, motif)

	motif = one_hot_encode('CATCAG').unsqueeze(0)
	motif[0, 1, 0] = 2
	assert_raises(ValueError, insert, X, motif)
	assert_raises(ValueError, insert, X, torch.randn(1, 4, 8))


###


def test_substitute_str(X):
	motif = 'CATCAG'
	X_substitute = substitute(X, motif)

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


def test_substitute_ohe(X):
	motif = one_hot_encode('CATCAG').unsqueeze(0)
	X_substitute = substitute(X, motif)

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


def test_substitute_start_str(X):
	motif = 'CATCAG'
	X_substitute = substitute(X, motif, start=0)

	assert X_substitute.shape == X.shape
	assert X_substitute.sum() == X.sum()
	assert_raises(AssertionError, assert_array_almost_equal,
		X_substitute.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_substitute, X)

	new_seq = ('CATCAGATCTCATCATCTGCTGACTACT' +
		'GACGTAGTCTGACTGACTGACTGACTACTGACTGACTGAC')
	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)
	assert_array_almost_equal(X_substitute, new_seq_ohe)


	motif = 'CATCAGCCC'
	X_substitute = substitute(X, motif, start=10)

	assert X_substitute.shape == X.shape
	assert X_substitute.sum() == X.sum()
	assert_raises(AssertionError, assert_array_almost_equal,
		X_substitute.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_substitute, X)

	new_seq = ('CACATCATCTCATCAGCCCCTGACTACTGACGTA' +
		'GTCTGACTGACTGACTGACTACTGACTGACTGAC')
	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)
	assert_array_almost_equal(X_substitute, new_seq_ohe)


def test_substitute_start_ohe(X):
	motif = one_hot_encode('CATCAG').unsqueeze(0)
	X_substitute = substitute(X, motif, start=0)

	assert X_substitute.shape == X.shape
	assert X_substitute.sum() == X.sum()
	assert_raises(AssertionError, assert_array_almost_equal,
		X_substitute.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_substitute, X)

	new_seq = ('CATCAGATCTCATCATCTGCTGACTACT' +
		'GACGTAGTCTGACTGACTGACTGACTACTGACTGACTGAC')
	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)
	assert_array_almost_equal(X_substitute, new_seq_ohe)


	motif = one_hot_encode('CATCAGCCC').unsqueeze(0)
	X_substitute = substitute(X, motif, start=10)

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
	motif = one_hot_encode('CACCAG', alphabet=['A', 'C', 'G'])
	assert_raises(ValueError, substitute, X, motif)
	assert_raises(ValueError, substitute, X, motif, ['A', 'C', 'G'])


def test_substitute_raises_length(X):
	assert_raises(ValueError, substitute, X, 'C'*1000)
	assert_raises(ValueError, substitute, X, one_hot_encode('C'*1000))


def test_substitute_raise_ends(X):
	assert_raises(ValueError, substitute, X, 'CAGCAT', -2)
	assert_raises(ValueError, substitute, X, 'CAGCAT', 1000)
	assert_raises(IndexError, substitute, X, 'CAGCAT', 6.5)
	assert_raises(TypeError, substitute, X, 'CAGCAT', 5, 3)
	assert_raises(TypeError, substitute, X, 'CAGCAT', -5, -1)
	assert_raises(TypeError, substitute, X, 'CAGCAT', 5, 1000)


def test_substitute_raises_ohe(X):
	motif = one_hot_encode('CATCAG').unsqueeze(0)
	motif[0, 0, 0] = 1
	assert_raises(ValueError, substitute, X, motif)

	motif = one_hot_encode('CATCAG').unsqueeze(0)
	motif[0, 1, 0] = 2
	assert_raises(ValueError, substitute, X, motif)
	assert_raises(ValueError, substitute, X, torch.randn(1, 4, 8))


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


###


def test_multisubstitute_str(X):
	motifs = ['CATCAG', 'CATCAG']
	X_substitute = multisubstitute(X, motifs, spacing=0)

	assert X_substitute.shape == X.shape
	assert X_substitute.sum() == X.sum()
	assert_raises(AssertionError, assert_array_almost_equal,
		X_substitute.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_substitute, X)

	new_seq = ('CACATCATCTCATCATCTGCTGACTACTCATCAG' +
		'CATCAGCTGACTGACTGACTACTGACTGACTGAC')
	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)

	assert X_substitute.shape == new_seq_ohe.shape
	assert X_substitute.sum() == new_seq_ohe.sum()
	assert_array_almost_equal(X_substitute.sum(dim=-1), new_seq_ohe.sum(dim=-1))
	assert_array_almost_equal(X_substitute, new_seq_ohe)


def test_multisubstitute_str_even(X):
	motifs = ['CATCAG', 'CATCAG']
	X_substitute = multisubstitute(X, motifs, spacing=4)

	assert X_substitute.shape == X.shape
	assert X_substitute.sum() == X.sum()
	assert_raises(AssertionError, assert_array_almost_equal,
		X_substitute.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_substitute, X)

	new_seq = ('CACATCATCTCATCATCTGCTGACTACATCAGTA' +
		'GTCATCAGGACTGACTGACTACTGACTGACTGAC')
	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)

	assert X_substitute.shape == new_seq_ohe.shape
	assert X_substitute.sum() == new_seq_ohe.sum()
	assert_array_almost_equal(X_substitute.sum(dim=-1), new_seq_ohe.sum(dim=-1))
	assert_array_almost_equal(X_substitute, new_seq_ohe)


def test_multisubstitute_str_odd(X):
	motifs = ['CATCAG', 'CATCAG']
	X_substitute = multisubstitute(X, motifs, spacing=5)

	assert X_substitute.shape == X.shape
	assert X_substitute.sum() == X.sum()
	assert_raises(AssertionError, assert_array_almost_equal,
		X_substitute.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_substitute, X)

	new_seq = ('CACATCATCTCATCATCTGCTGACTACATCAGTA' +
		'GTCCATCAGACTGACTGACTACTGACTGACTGAC')
	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)

	assert X_substitute.shape == new_seq_ohe.shape
	assert X_substitute.sum() == new_seq_ohe.sum()
	assert_array_almost_equal(X_substitute.sum(dim=-1), new_seq_ohe.sum(dim=-1))
	assert_array_almost_equal(X_substitute, new_seq_ohe)


def test_multisubstitute_ohe(X):
	motif = one_hot_encode('CATCAG').unsqueeze(0)
	X_substitute = multisubstitute(X, [motif, motif], spacing=0)

	assert X_substitute.shape == X.shape
	assert X_substitute.sum() == X.sum()
	assert_raises(AssertionError, assert_array_almost_equal,
		X_substitute.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_substitute, X)

	new_seq = ('CACATCATCTCATCATCTGCTGACTACTCATCAG' +
		'CATCAGCTGACTGACTGACTACTGACTGACTGAC')
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
	motif = one_hot_encode('CACCAG', alphabet=['A', 'C', 'G'])
	assert_raises(ValueError, multisubstitute, X, [motif, motif], [0])
	assert_raises(ValueError, multisubstitute, X, [motif, motif], [0],
		['A', 'C', 'G'])


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


def test_randomize_raises_probs(X):
	assert_raises(ValueError, randomize, X, start=5, end=10,
		probs=[[0.1, 0.8, 0.1]])
	assert_raises(ValueError, randomize, X, start=5, end=10,
		probs=[[0.1, 0.8, 0.1, 0.0], [0.4, 0.3, 0.2]])
	assert_raises(ValueError, randomize, X, start=5, end=10,
		probs=[0.1, 0.8, 0.1, 0.0])
	assert_raises(ValueError, randomize, X, start=5, end=10,
		probs=[[0.1, 100.8, 0.1]])


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


def test_dinucleotide_shuffle_composition():
	X = random_one_hot((8, 4, 30), random_state=0)
	X_shuf = dinucleotide_shuffle(X, random_state=0)

	for i in range(X.shape[0]):
		seq = characters(X[i])

		dinucs = collections.defaultdict(int)
		for j in range(len(seq)-1):
			dinucs[seq[j:j+2]] += 1

		for j in range(20):
			dinucs_shuffled = collections.defaultdict(int)
			dinucs_seq = characters(X_shuf[i, j])

			for k in range(len(seq)-1):
				dinucs_shuffled[dinucs_seq[k:k+2]] += 1

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


def test_dinucleotide_shuffle_large_alphabet():
	alpha = ['A', 'C', 'G', 'T', 'N']
	seq = 'CGATCAGCANNCACATCAGCATANNAAT'
	motif = one_hot_encode(seq, alphabet=alpha, ignore=[]).unsqueeze(0)
	dimotif = dinucleotide_shuffle(motif, random_state=0)

	dinucs = collections.defaultdict(int)
	for i in range(len(seq)-1):
		dinucs[seq[i:i+2]] += 1

	for j in range(20):
		dinucs_shuffled = collections.defaultdict(int)
		dinucs_seq = characters(dimotif[0, j], alphabet=alpha)

		for i in range(len(seq)-1):
			dinucs_shuffled[dinucs_seq[i:i+2]] += 1

		for key, value in dinucs_shuffled.items():
			assert dinucs[key] == value


def test_dinucleotide_shuffle_missing_alphabet():
	seq = 'ATATATTAAAATTATTATATATTTATATATTTAAAAATTTTTAATA'
	motif = one_hot_encode(seq).unsqueeze(0)
	dimotif = dinucleotide_shuffle(motif, random_state=0)

	dinucs = collections.defaultdict(int)
	for i in range(len(seq)-1):
		dinucs[seq[i:i+2]] += 1

	for j in range(20):
		dinucs_shuffled = collections.defaultdict(int)
		dinucs_seq = characters(dimotif[0, j])

		for i in range(len(dinucs_seq)-1):
			dinucs_shuffled[dinucs_seq[i:i+2]] += 1

		for key, value in dinucs_shuffled.items():
			assert dinucs[key] == value


def test_dinucleotide_shuffle_raises_short():
	motif = one_hot_encode('AATA').unsqueeze(0)
	assert_raises(ValueError, dinucleotide_shuffle, motif)


def test_dinucleotide_shuffle_raises_ohe():
	seq = 'ATATATTAAAATTATTATATATTTATATATTTAAAAATTTTTAATA'
	motif = one_hot_encode(seq).unsqueeze(0)

	assert_raises(ValueError, dinucleotide_shuffle, motif + 1)
	assert_raises(ValueError, dinucleotide_shuffle, motif.unsqueeze(0))
	assert_raises(ValueError, dinucleotide_shuffle, motif[0])
	assert_raises(ValueError, dinucleotide_shuffle, "ACGTCACGATC")


def test_dinucleotide_shuffle_raises_N():
	seq = 'ATATATTAAAATNNNATTTAAANNNTTTTTAATA'
	motif = one_hot_encode(seq).unsqueeze(0)
	assert_raises(ValueError, dinucleotide_shuffle, motif)


def test_dinucleotide_shuffle_homopolymer():
	seq_ohe = one_hot_encode('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA').unsqueeze(0)
	assert_raises(ValueError, dinucleotide_shuffle, seq_ohe)


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


def test_local_dinucleotide_shuffle():
	X = random_one_hot((2, 4, 200), random_state=0)
	X_shuf = local_dinucleotide_shuffle(X, n=5, bin_size=64, min_bin_size=32,
		random_state=0)

	assert X_shuf.shape == (2, 5, 4, 200)
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
	# only conserved within a bin, not across bin boundaries. Rather than
	# guessing the cut points, check that *some* valid set of bins explains
	# each shuffle: the first cut is the only random boundary choice.
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

				if all(_dinuc_counts(seq[s:e]) == _dinuc_counts(shuf_seq[s:e])
					for s, e in bins):
					matches.append(first_cut)

			assert len(matches) > 0, ("No bin layout preserves the per-bin "
				"dinucleotide composition of shuffle {}".format(j))


def test_local_dinucleotide_shuffle_bin_endpoints():
	# A dinucleotide shuffle keeps the first and last character of the region
	# it is applied to, so the bin endpoints must be conserved. As above, at
	# least one valid first cut must be consistent with the shuffle.
	seq_len, bin_size, min_bin_size = 200, 64, 32

	X = random_one_hot((2, 4, seq_len), random_state=0)
	X_shuf = local_dinucleotide_shuffle(X, n=3, bin_size=bin_size,
		min_bin_size=min_bin_size, random_state=0)

	for i in range(X.shape[0]):
		for j in range(X_shuf.shape[1]):
			matches = []
			for first_cut in range(min_bin_size, bin_size + 1):
				bins = _local_bins(first_cut, seq_len, bin_size, min_bin_size)
				idxs = [s for s, _ in bins] + [e-1 for _, e in bins]

				if (X_shuf[i, j, :, idxs] == X[i, :, idxs]).all():
					matches.append(first_cut)

			assert len(matches) > 0


def test_local_dinucleotide_shuffle_bin_layout_valid():
	# Every bin the implementation can produce must be at least min_bin_size
	# long and no longer than 2 * bin_size (the merge of the final two bins is
	# the only way a bin exceeds bin_size).
	seq_len, bin_size, min_bin_size = 200, 64, 32

	for first_cut in range(min_bin_size, bin_size + 1):
		bins = _local_bins(first_cut, seq_len, bin_size, min_bin_size)

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


def test_local_dinucleotide_shuffle_alphabet():
	# Any one-hot alphabet is accepted, matching dinucleotide_shuffle, which
	# this function otherwise mirrors.
	X = random_one_hot((2, 20, 200), random_state=0)
	X_shuf = local_dinucleotide_shuffle(X, n=3, bin_size=64, min_bin_size=32,
		random_state=0)

	assert X_shuf.shape == (2, 3, 20, 200)
	assert X_shuf.dtype == X.dtype

	assert (X_shuf.sum(dim=2) == 1).all()
	assert (X_shuf.sum(dim=-1) == X.sum(dim=-1).unsqueeze(1)).all()


def test_local_dinucleotide_shuffle_raises_bin_size():
	X = random_one_hot((2, 4, 200), random_state=0)

	# bin_size must be strictly smaller than the sequence length.
	assert_raises(ValueError, local_dinucleotide_shuffle, X, 20, 200)
	assert_raises(ValueError, local_dinucleotide_shuffle, X, 20, 300)


def test_local_dinucleotide_shuffle_raises_min_bin_size():
	X = random_one_hot((2, 4, 200), random_state=0)

	assert_raises(ValueError, local_dinucleotide_shuffle, X, 20, 64, 65)


def test_local_dinucleotide_shuffle_preserves_cuda_device(cuda_device):
	X = random_one_hot((2, 4, 200), random_state=0).type(
		torch.float32).to(cuda_device)
	X_shuf = local_dinucleotide_shuffle(X, n=2, bin_size=64, min_bin_size=32,
		random_state=0)

	assert X_shuf.device.type == 'cuda'
	assert X_shuf.shape == (2, 2, 4, 200)


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


def test_ersatz_ops_preserve_cuda_device(cuda_device):
	# All public ersatz ops should accept a CUDA-resident X and return a
	# tensor on the same device. Previously dinucleotide_shuffle crashed
	# in the internal .numpy() conversion.
	X = random_one_hot((2, 4, 100), random_state=0).type(
		torch.float32).to(cuda_device)
	motif = one_hot_encode("ACGT").unsqueeze(0).to(cuda_device)

	for name, out in [
		('substitute', substitute(X, motif)),
		('multisubstitute', multisubstitute(X, [motif, motif], [5])),
		('insert', insert(X, motif, start=20)),
		('delete', delete(X, 20, 30)),
		('shuffle', shuffle(X, start=20, end=30, n=2, random_state=0)),
		('randomize', randomize(X, start=20, end=30, n=2, random_state=0)),
		('dinucleotide_shuffle', dinucleotide_shuffle(X, n=2, random_state=0)),
	]:
		assert out.device.type == 'cuda', f"{name} produced device {out.device}"
