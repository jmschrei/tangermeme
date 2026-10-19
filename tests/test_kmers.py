# test_kmers.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import scipy
import torch
import pytest
import collections

from tangermeme.utils import characters
from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot
from tangermeme.utils import PackedSequence

from tangermeme.kmers import kmers
from tangermeme.kmers import gapped_kmers

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


@pytest.fixture
def X():
	return one_hot_encode('ACTACTGCAT').unsqueeze(0)


###


def test_kmers(X):
	X_kmers = kmers(X, k=5)

	assert X_kmers.shape == (1, 4**5)
	assert X_kmers.sum() == 6
	assert X_kmers[0, 308] == 1
	assert X_kmers[0, 845] == 1
	assert X_kmers[0, 723] == 1
	assert X_kmers[0, 436] == 1
	assert X_kmers[0, 109] == 1
	assert X_kmers[0, 795] == 1

	X_kmers = kmers(X, k=7)

	assert X_kmers.shape == (1, 4**7)
	assert X_kmers.sum() == 4
	assert X_kmers[0, 11572] == 1
	assert X_kmers[0, 6989] == 1
	assert X_kmers[0, 1747] == 1
	assert X_kmers[0, 12724] == 1


def test_kmers_scores(X):
	scores = torch.arange(1, 11).unsqueeze(0)
	X_kmers = kmers(X, k=7, scores=scores)

	assert X_kmers.shape == (1, 4**7)
	assert X_kmers.sum() == 154
	assert X_kmers[0, 11572] == 28
	assert X_kmers[0, 6989] == 35
	assert X_kmers[0, 1747] == 42
	assert X_kmers[0, 12724] == 49


###


def test_gkmers(X):
	X_kmers = gapped_kmers(X, min_k=5, max_k=5)

	assert isinstance(X_kmers, scipy.sparse._csr.csr_matrix)
	assert X_kmers.shape == (1, 9765625)


###


def test_kmers_k_equals_seqlen(X):
	X_kmers = kmers(X, k=X.shape[-1])

	assert X_kmers.shape == (1, 4 ** X.shape[-1])
	assert X_kmers.sum() == 1


def test_kmers_k_one(X):
	X_kmers = kmers(X, k=1)

	assert X_kmers.shape == (1, 4)
	assert_array_almost_equal(X_kmers[0], X.sum(dim=-1)[0])


def test_kmers_batched():
	X = torch.stack([
		one_hot_encode('ACTACTGCAT'),
		one_hot_encode('GGGGGGGGGG'),
	])

	X_kmers = kmers(X, k=3)

	assert X_kmers.shape == (2, 64)
	assert int(X_kmers[0].sum()) == 8
	assert int(X_kmers[1].sum()) == 8

	# The all-G sequence collapses to a single column counted 8 times.
	assert int(X_kmers[1].max()) == 8
	assert int((X_kmers[1] > 0).sum()) == 1


def test_kmers_scores_zero(X):
	scores = torch.zeros(1, X.shape[-1])
	X_kmers = kmers(X, k=5, scores=scores)

	assert X_kmers.shape == (1, 4 ** 5)
	assert X_kmers.sum() == 0


def test_gkmers_min_k_lt_max_k(X):
	X_kmers = gapped_kmers(X, min_k=4, max_k=6)

	assert isinstance(X_kmers, scipy.sparse._csr.csr_matrix)
	assert X_kmers.shape == (1, 9765625)
	assert X_kmers.sum() > 0

###
# A PackedSequence gives the counts of its one-hot encoding.


@pytest.mark.parametrize("k, length", [(1, 60), (2, 60), (3, 60), (5, 60),
	(7, 60), (7, 7)])
def test_kmers_packed_codes(k, length, monkeypatch):
	# The k-mers of a PackedSequence are read from its character indices,
	# without decoding the one-hot encoding, and equal those of the one-hot
	# encoding, with unknown characters at the start, the end and between,
	# also when k is the length of the sequences.
	X = random_one_hot((6, 4, length), random_state=1)
	X[0, :, :3] = 0
	X[2, :, 4:6] = 0
	X[4, :, -2:] = 0
	X[5] = 0
	X_kmers = kmers(X, k)
	X_packed = PackedSequence.from_one_hot(X)

	def fail(*args, **kwargs):
		raise AssertionError("decoded the one-hot encoding")

	monkeypatch.setattr(PackedSequence, "to_one_hot", fail)
	assert torch.equal(kmers(X_packed, k), X_kmers)


def test_gapped_kmers_packed():
	X = random_one_hot((5, 4, 60), random_state=0)
	X[1, :, 10:14] = 0
	X_packed = PackedSequence.from_one_hot(X)

	X_gkmers = gapped_kmers(X, min_k=4, max_k=5)
	X_gkmers_packed = gapped_kmers(X_packed, min_k=4, max_k=5)
	assert (X_gkmers_packed != X_gkmers).nnz == 0


@pytest.mark.parametrize("chunk", [1, 100, 10 ** 6])
def test_kmers_packed_chunks(chunk, monkeypatch):
	# Counting a chunk of sequences at a time gives the counts, and the
	# score-weighted counts, of the one-hot encoding.
	monkeypatch.setattr("tangermeme.utils._CHUNK_POSITIONS", chunk)
	X = random_one_hot((7, 4, 40), random_state=2)
	X[3, :, 5:9] = 0
	scores = torch.randn(7, 40, generator=torch.Generator().manual_seed(1))
	X_packed = PackedSequence.from_one_hot(X)

	assert torch.equal(kmers(X_packed, 3), kmers(X, 3))
	assert torch.equal(kmers(X_packed, 2, scores=scores), kmers(X, 2,
		scores=scores))

