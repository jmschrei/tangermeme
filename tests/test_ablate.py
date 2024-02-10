# test_io.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest

from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot

from tangermeme.ablate import insert

import pandas
import pyfaidx
import pyBigWig

from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal


@pytest.fixture
def X():
	seq = 'CACATCATCTCATCATCTGCTGACTACTGACGTAGTCTGACTGACTGACTGACTACTGACTGACTGAC'
	return one_hot_encode(seq).unsqueeze(0)


###


def test_insert_str(X):
	motif = 'CATCAG'
	X_insert = insert(X, motif)

	assert X_insert.shape == X.shape
	assert X_insert.sum() == X.sum()
	assert_raises(AssertionError, assert_array_almost_equal, 
		X_insert.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_insert, X)

	new_seq = ('CACATCATCTCATCATCTGCTGACTACTGACCAT' + 
		'CAGTGACTGACTGACTGACTACTGACTGACTGAC')
	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)

	assert X_insert.shape == new_seq_ohe.shape
	assert X_insert.sum() == new_seq_ohe.sum()
	assert_array_almost_equal(X_insert.sum(dim=-1), new_seq_ohe.sum(dim=-1))
	assert_array_almost_equal(X_insert, new_seq_ohe)


def test_insert_ohe(X):
	motif = one_hot_encode('CATCAG').unsqueeze(0)
	X_insert = insert(X, motif)

	assert X_insert.shape == X.shape
	assert X_insert.sum() == X.sum()
	assert_raises(AssertionError, assert_array_almost_equal, 
		X_insert.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_insert, X)

	new_seq = ('CACATCATCTCATCATCTGCTGACTACTGACCAT' + 
		'CAGTGACTGACTGACTGACTACTGACTGACTGAC')
	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)

	assert X_insert.shape == new_seq_ohe.shape
	assert X_insert.sum() == new_seq_ohe.sum()
	assert_array_almost_equal(X_insert.sum(dim=-1), new_seq_ohe.sum(dim=-1))
	assert_array_almost_equal(X_insert, new_seq_ohe)


def test_insert_str_multi_seqs_one_motif():
	X = random_one_hot((4, 4, 8), random_state=0)
	X_insert = insert(X, 'ACGT')
	print(X_insert)

	assert_raises(AssertionError, assert_array_almost_equal, X, X_insert)
	assert_array_almost_equal(X_insert, [
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


def test_insert_str_multi_seqs_multi_motifs():
	X = random_one_hot((4, 4, 8), random_state=0)
	motif = random_one_hot((4, 4, 4), random_state=1)
	X_insert = insert(X, motif)
	print(X_insert)

	assert_raises(AssertionError, assert_array_almost_equal, X, X_insert)
	assert_array_almost_equal(X_insert, [
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


def test_insert_start_str(X):
	motif = 'CATCAG'
	X_insert = insert(X, motif, start=0)

	assert X_insert.shape == X.shape
	assert X_insert.sum() == X.sum()
	assert_raises(AssertionError, assert_array_almost_equal, 
		X_insert.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_insert, X)

	new_seq = ('CATCAGATCTCATCATCTGCTGACTACT' + 
		'GACGTAGTCTGACTGACTGACTGACTACTGACTGACTGAC')
	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)
	assert_array_almost_equal(X_insert, new_seq_ohe)


	motif = 'CATCAGCCC'
	X_insert = insert(X, motif, start=10)

	assert X_insert.shape == X.shape
	assert X_insert.sum() == X.sum()
	assert_raises(AssertionError, assert_array_almost_equal, 
		X_insert.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_insert, X)

	new_seq = ('CACATCATCTCATCAGCCCCTGACTACTGACGTA' +
		'GTCTGACTGACTGACTGACTACTGACTGACTGAC')
	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)
	assert_array_almost_equal(X_insert, new_seq_ohe)


def test_insert_start_ohe(X):
	motif = one_hot_encode('CATCAG').unsqueeze(0)
	X_insert = insert(X, motif, start=0)

	assert X_insert.shape == X.shape
	assert X_insert.sum() == X.sum()
	assert_raises(AssertionError, assert_array_almost_equal, 
		X_insert.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_insert, X)

	new_seq = ('CATCAGATCTCATCATCTGCTGACTACT' + 
		'GACGTAGTCTGACTGACTGACTGACTACTGACTGACTGAC')
	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)
	assert_array_almost_equal(X_insert, new_seq_ohe)


	motif = one_hot_encode('CATCAGCCC').unsqueeze(0)
	X_insert = insert(X, motif, start=10)

	assert X_insert.shape == X.shape
	assert X_insert.sum() == X.sum()
	assert_raises(AssertionError, assert_array_almost_equal, 
		X_insert.sum(dim=-1), X.sum(dim=-1))
	assert_raises(AssertionError, assert_array_almost_equal, X_insert, X)

	new_seq = ('CACATCATCTCATCAGCCCCTGACTACTGACGTA' +
		'GTCTGACTGACTGACTGACTACTGACTGACTGAC')
	new_seq_ohe = one_hot_encode(new_seq).unsqueeze(0)
	assert_array_almost_equal(X_insert, new_seq_ohe)


def test_insert_raises_alphabet(X):
	motif = one_hot_encode('CACCAG', alphabet=['A', 'C', 'G'])
	assert_raises(ValueError, insert, X, motif)
	assert_raises(ValueError, insert, X, motif, ['A', 'C', 'G'])


def test_insert_raises_length(X):
	assert_raises(ValueError, insert, X, 'C'*1000)
	assert_raises(ValueError, insert, X, one_hot_encode('C'*1000))


def test_insert_raise_start(X):
	assert_raises(ValueError, insert, X, 'CAGCAT', -2)
	assert_raises(ValueError, insert, X, 'CAGCAT', 1000)
	assert_raises(TypeError, insert, X, 'CAGCAT', 6.5)


def test_insert_raises_ohe(X):
	motif = one_hot_encode('CATCAG').unsqueeze(0)
	motif[0, 0, 0] = 1
	assert_raises(ValueError, insert, X, motif)

	motif = one_hot_encode('CATCAG').unsqueeze(0)
	motif[0, 1, 0] = 2
	assert_raises(ValueError, insert, X, motif)
	assert_raises(ValueError, insert, X, torch.randn(1, 4, 8))