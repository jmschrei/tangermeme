# test_kmers.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import scipy
import torch
import pytest
import itertools
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


def _gapped_kmers_reference(sequence, min_k, max_k, max_len, positions=None):
	# The occurrences of each gapped k-mer: every set of min_k to max_k of the
	# positions that spans at most max_len, keyed in base 5 by the base plus
	# one at each offset from its first position, and 0 at a gap.
	if positions is None:
		positions = range(len(sequence))

	occurrences = collections.defaultdict(list)
	for k in range(min_k, max_k + 1):
		for idxs in itertools.combinations(positions, k):
			if idxs[-1] - idxs[0] < max_len:
				key = sum(('ACGT'.index(sequence[i]) + 1) * 5 ** (i - idxs[0])
					for i in idxs)
				occurrences[key].append(idxs)

	return occurrences


def _assert_top_gkmers(X_gkmers, reference, max_gkmers):
	# The row holds the max_gkmers gapped k-mers of the reference whose values
	# are largest in absolute value, or all of them.
	row = dict(zip(X_gkmers.indices.tolist(), X_gkmers.data.tolist()))
	assert len(row) == min(max_gkmers, len(reference))
	assert set(row) <= set(reference)
	assert min(abs(reference[key]) for key in row) >= max((abs(value) for
		key, value in reference.items() if key not in row), default=0)
	return row


# A max_gap of max_len - 1 allows any gap that fits in max_len.
@pytest.mark.parametrize("min_k, max_k, max_len", [(5, 5, 10), (4, 6, 10),
	(2, 3, 4), (2, 2, 3)])
def test_gkmers(X, min_k, max_k, max_len):
	reference = {key: len(idxs) for key, idxs in _gapped_kmers_reference(
		'ACTACTGCAT', min_k, max_k, max_len).items()}
	X_gkmers = gapped_kmers(X, min_k=min_k, max_k=max_k, max_gap=max_len - 1,
		max_len=max_len, max_gkmers=len(reference))

	assert isinstance(X_gkmers, scipy.sparse.csr_matrix)
	assert X_gkmers.shape == (1, 5 ** max_len)
	assert dict(zip(X_gkmers.indices.tolist(), X_gkmers.data.tolist())) == \
		reference


def test_gkmers_max_gkmers():
	# Each sequence keeps the max_gkmers gapped k-mers with the most
	# occurrences, with their counts.
	sequences = ['ACTACTGCAT', 'GGGGAAAACG']
	X = torch.stack([one_hot_encode(sequence) for sequence in sequences])
	X_gkmers = gapped_kmers(X, min_k=2, max_k=3, max_gap=3, max_len=4,
		max_gkmers=5)

	assert X_gkmers.shape == (2, 5 ** 4)
	for sequence, X_row in zip(sequences, X_gkmers):
		reference = {key: len(idxs) for key, idxs in _gapped_kmers_reference(
			sequence, 2, 3, 4).items()}
		row = _assert_top_gkmers(X_row, reference, 5)
		assert all(row[key] == reference[key] for key in row)


def test_gkmers_scores_max_pos(X):
	# Only the max_pos positions with the highest scores are used, which drops
	# positions 5 and 7, and the max_gkmers gapped k-mers whose summed scores
	# are largest in absolute value are kept, three of four negative here.
	scores = torch.tensor([[0.5, -3.0, -2.0, 1.5, 0.25, -4.0, 1.0, -3.5, 0.75,
		-2.5]])
	occurrences = _gapped_kmers_reference('ACTACTGCAT', 3, 3, 5,
		positions=[0, 1, 2, 3, 4, 6, 8, 9])
	reference = {key: sum(scores[0, i].item() for idxs in occurrence for i in
		idxs) for key, occurrence in occurrences.items()}

	for max_gkmers in [len(reference), 4]:
		X_gkmers = gapped_kmers(X, scores=scores, min_k=3, max_k=3, max_gap=4,
			max_len=5, max_gkmers=max_gkmers, max_pos=8)
		_assert_top_gkmers(X_gkmers, reference, max_gkmers)


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

	assert int(X_kmers[1].max()) == 8
	assert int((X_kmers[1] > 0).sum()) == 1


def test_kmers_scores_zero(X):
	scores = torch.zeros(1, X.shape[-1])
	X_kmers = kmers(X, k=5, scores=scores)

	assert X_kmers.shape == (1, 4 ** 5)
	assert X_kmers.sum() == 0


###
# A PackedSequence gives the counts of its one-hot encoding.


@pytest.mark.parametrize("k, length", [(1, 60), (2, 60), (3, 60), (5, 60),
	(7, 60), (7, 7)])
def test_kmers_packed_codes(k, length, monkeypatch):
	# Read from the character indices without decoding, with unknown
	# characters at the start, the end and between, and with k = length.
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
	monkeypatch.setattr("tangermeme.utils._CHUNK_POSITIONS", chunk)
	X = random_one_hot((7, 4, 40), random_state=2)
	X[3, :, 5:9] = 0
	scores = torch.randn(7, 40, generator=torch.Generator().manual_seed(1))
	X_packed = PackedSequence.from_one_hot(X)

	assert torch.equal(kmers(X_packed, 3), kmers(X, 3))
	assert torch.equal(kmers(X_packed, 2, scores=scores), kmers(X, 2,
		scores=scores))

