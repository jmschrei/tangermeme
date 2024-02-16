# test_io.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest
import collections

from tangermeme.utils import characters
from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot

from tangermeme.ersatz import insert
from tangermeme.ersatz import substitute
from tangermeme.ersatz import randomize
from tangermeme.ersatz import shuffle
from tangermeme.ersatz import dinucleotide_shuffle

import pandas
import pyfaidx
import pyBigWig

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
	assert_raises(TypeError, substitute, X, 'CAGCAT', 6.5)
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
