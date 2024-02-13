# test_io.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest
import collections

from tangermeme.utils import one_hot_encode
from tangermeme.utils import characters
from tangermeme.utils import random_one_hot

from tangermeme.ablate import insert
from tangermeme.ablate import dinucleotide_shuffle

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


###


def test_dinucleotide_shuffle():
	motif = one_hot_encode('CATCACGCATACG')
	dimotif = dinucleotide_shuffle(motif, random_state=0)

	assert dimotif.shape == (10, 4, 13)
	assert dimotif.dtype == torch.float32
	assert characters(dimotif[0]) == 'CGCATCACATACG'
	assert characters(dimotif[1]) == 'CATCGCATACACG'


def test_dinucleotide_shuffle_composition():
	seq = 'CATCACGCATACGACTACGCACTATACCATGCATGAA'
	motif = one_hot_encode(seq)
	dimotif = dinucleotide_shuffle(motif, random_state=0)

	dinucs = collections.defaultdict(int)
	for i in range(len(seq)-1):
		dinucs[seq[i:i+2]] += 1

	for j in range(10):
		dinucs_shuffled = collections.defaultdict(int)
		dinucs_seq = characters(dimotif[j])

		for i in range(len(seq)-1):
			dinucs_shuffled[dinucs_seq[i:i+2]] += 1

		for key, value in dinucs_shuffled.items():
			assert dinucs[key] == value 


def test_dinucleotide_shuffle_large_alphabet():
	alpha = ['A', 'C', 'G', 'T', 'N']
	seq = 'CGATCAGCANNCACATCAGCATANNAAT'
	motif = one_hot_encode(seq, alphabet=alpha, ignore=[])
	dimotif = dinucleotide_shuffle(motif, random_state=0)

	dinucs = collections.defaultdict(int)
	for i in range(len(seq)-1):
		dinucs[seq[i:i+2]] += 1

	for j in range(10):
		dinucs_shuffled = collections.defaultdict(int)
		dinucs_seq = characters(dimotif[j], alphabet=alpha)

		for i in range(len(seq)-1):
			dinucs_shuffled[dinucs_seq[i:i+2]] += 1

		for key, value in dinucs_shuffled.items():
			assert dinucs[key] == value 


def test_dinucleotide_shuffle_missing_alphabet():
	seq = 'ATATATTAAAATTATTATATATTTATATATTTAAAAATTTTTAATA'
	motif = one_hot_encode(seq)
	dimotif = dinucleotide_shuffle(motif, random_state=0)

	dinucs = collections.defaultdict(int)
	for i in range(len(seq)-1):
		dinucs[seq[i:i+2]] += 1

	for j in range(10):
		dinucs_shuffled = collections.defaultdict(int)
		dinucs_seq = characters(dimotif[j])

		for i in range(len(dinucs_seq)-1):
			dinucs_shuffled[dinucs_seq[i:i+2]] += 1

		for key, value in dinucs_shuffled.items():
			assert dinucs[key] == value 


def test_dinucleotide_shuffle_raises_short():
	motif = one_hot_encode('AATA')
	assert_raises(ValueError, dinucleotide_shuffle, motif)

def test_dinucleotide_shuffle_raises_ohe():
	seq = 'ATATATTAAAATTATTATATATTTATATATTTAAAAATTTTTAATA'
	motif = one_hot_encode(seq) 

	assert_raises(ValueError, dinucleotide_shuffle, motif + 1)
	assert_raises(ValueError, dinucleotide_shuffle, motif.unsqueeze(0))
	assert_raises(ValueError, dinucleotide_shuffle, motif[0])
	assert_raises(ValueError, dinucleotide_shuffle, "ACGTCACGATC")


def test_dinucleotide_shuffle_raises_N():
	seq = 'ATATATTAAAATNNNATTTAAANNNTTTTTAATA'
	motif = one_hot_encode(seq) 
	assert_raises(ValueError, dinucleotide_shuffle, motif)


def test_dinucleotide_shuffle_homopolymer():
	seq_ohe = one_hot_encode('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
	assert_raises(ValueError, dinucleotide_shuffle, seq_ohe)
