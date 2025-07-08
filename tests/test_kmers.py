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