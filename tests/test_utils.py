# test_utils.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pandas
import pytest


from tangermeme.utils import _validate_input
from tangermeme.utils import validate_input
from tangermeme.utils import TangermemeWarning
from tangermeme.utils import characters
from tangermeme.utils import one_hot_encode
from tangermeme.utils import reverse_complement
from tangermeme.utils import random_one_hot
from tangermeme.utils import chunk
from tangermeme.utils import unchunk
from tangermeme.utils import extract_signal
from tangermeme.utils import pwm_consensus
from tangermeme.utils import example_to_fasta_coords

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal



def test_validate_input_shape():
	X = torch.randn(1)
	assert_raises(ValueError, _validate_input, X, "X", shape=(2, 1))
	assert_raises(ValueError, _validate_input, X, "X", shape=(2,))
	assert_raises(ValueError, _validate_input, X, "X", shape=(-1, -1))
	assert_raises(ValueError, _validate_input, X, "X", shape=(-1, 1))
	_validate_input(X, "X", shape=(1,))
	_validate_input(X, "X", shape=(-1,))

	X = torch.randn(5, 3)
	assert_raises(ValueError, _validate_input, X, "X", shape=(-1,))
	assert_raises(ValueError, _validate_input, X, "X", shape=(3,))
	assert_raises(ValueError, _validate_input, X, "X", shape=(5,))
	assert_raises(ValueError, _validate_input, X, "X", shape=(1, 4))
	assert_raises(ValueError, _validate_input, X, "X", shape=(-1, -1, -1))
	assert_raises(ValueError, _validate_input, X, "X", shape=(5, 3, -1))
	_validate_input(X, "X", shape=(5, 3))
	_validate_input(X, "X", shape=(-1, 3))
	_validate_input(X, "X", shape=(5, -1))
	_validate_input(X, "X", shape=(-1, -1))


def test_validate_input_dtype():
	X = torch.randn(5, 3)
	assert_raises(ValueError, _validate_input, X, "X", dtype=torch.float16)
	assert_raises(ValueError, _validate_input, X, "X", dtype=torch.float64)
	assert_raises(ValueError, _validate_input, X, "X", dtype=torch.int8)
	assert_raises(ValueError, _validate_input, X, "X", dtype=torch.int32)
	_validate_input(X, "X", dtype=torch.float32)


def test_validate_input_min_value():
	X = torch.randn(100, 100)
	assert_raises(ValueError, _validate_input, X, "X", min_value=0.0)
	_validate_input(torch.abs(X), "X", min_value=0.0)

	X[X < 0] = 0
	_validate_input(X, "X", min_value=0.0)

	X = X.type(torch.int8)
	_validate_input(X, "X", min_value=0.0)


def test_validate_input_max_value():
	X = torch.randn(100, 100)
	assert_raises(ValueError, _validate_input, X, "X", max_value=0.0)
	_validate_input(-torch.abs(X), "X", max_value=0.0)

	X[X > 0] = 0
	_validate_input(X, "X", max_value=0.0)

	X = X.type(torch.int8)
	_validate_input(X, "X", max_value=0.0)


def test_validate_input_ohe(device):
	X = random_one_hot((1, 4, 10), random_state=0).to(device)

	_validate_input(X, "X", ohe=True, ohe_dim=1)
	_validate_input(X.type(torch.float64), "X", ohe=True, ohe_dim=1)
	_validate_input(X.type(torch.int8), "X", ohe=True, ohe_dim=1)
	_validate_input(X.type(torch.int32), "X", ohe=True, ohe_dim=1)
	_validate_input(X.permute(1, 0, 2), "X", ohe=True, ohe_dim=0)
	_validate_input(X.permute(2, 0, 1), "X", ohe=True, ohe_dim=2)

	assert_raises(ValueError, _validate_input, X, "X", ohe=True, ohe_dim=0)
	assert_raises(ValueError, _validate_input, X, "X", ohe=True, ohe_dim=2)
	assert_raises(ValueError, _validate_input, X, "X", ohe=True, ohe_dim=-1)
	assert_raises(ValueError, _validate_input, X + 0.2, "X", ohe=True, 
		ohe_dim=1)
	assert_raises(ValueError, _validate_input, X - 1, "X", ohe=True, 
		ohe_dim=1)

	X[0, 2, 3] = 1
	assert_raises(ValueError, _validate_input, X, "X", ohe=True, ohe_dim=1)

	X[0, :, 3] = 0
	assert_raises(ValueError, _validate_input, X, "X", ohe=True, ohe_dim=1)

	X = random_one_hot((1, 4, 10), random_state=0)[0]
	_validate_input(X, "X", ohe=True, ohe_dim=0)
	assert_raises(ValueError, _validate_input, X, "X", ohe=True, ohe_dim=1)

	X = random_one_hot((1, 4, 10), random_state=0)[0, :, 0]
	_validate_input(X, "X", ohe=True, ohe_dim=0)
	assert_raises(IndexError, _validate_input, X, "X", ohe=True, ohe_dim=1)


def test_validate_input_allow_N(device):
	X = random_one_hot((2, 4, 10), random_state=0).to(device)
	_validate_input(X, "X", ohe=True, ohe_dim=1)

	X[0, :, 0] = 0
	assert_raises(ValueError, _validate_input, X, "X", ohe=True, ohe_dim=1)
	_validate_input(X, "X", ohe=True, ohe_dim=1, allow_N=True)

	X[0, 1, 0] = 2
	assert_raises(ValueError, _validate_input, X, "X", ohe=True, ohe_dim=1,
		allow_N=True)

	X[0, :, 0] = 1
	assert_raises(ValueError, _validate_input, X, "X", ohe=True, ohe_dim=1,
		allow_N=True)

	X[0, 0, 0] = 0
	X[0, 1, 0] = 1
	X[0, 2, 0] = -1
	X[0, 3, 0] = 1
	assert_raises(ValueError, _validate_input, X, "X", ohe=True, ohe_dim=1,
		allow_N=True)

	X = random_one_hot((2, 4, 10), random_state=0).float().to(device)
	_validate_input(X, "X", ohe=True, ohe_dim=1, allow_N=True)

	X[0, 1, 0] = 0.5
	assert_raises(ValueError, _validate_input, X, "X", ohe=True, ohe_dim=1,
		allow_N=True)


def test_validate_input_all_zero_with_allow_N(device):
	# An all-zero tensor represents a sequence where every position is
	# unknown (N). With allow_N=True this is a valid one-hot encoding
	# and must not be rejected. The previous len(unique(X)) == 2 check
	# tripped on this case because unique returned just [0].
	X = torch.zeros(1, 4, 5, dtype=torch.int8).to(device)
	_validate_input(X, "X", ohe=True, ohe_dim=1, allow_N=True)

	# Without allow_N, every position must have exactly one 1; an all-N
	# input must still raise.
	assert_raises(ValueError, _validate_input, X, "X", ohe=True, ohe_dim=1)


def test_public_validate_input_wrapper(device):
	# The public validate_input forwards to the internal helper.
	X = random_one_hot((2, 4, 10), random_state=0).to(device)
	validate_input(X, "X", ohe=True, ohe_dim=1)

	# Bad input raises as expected.
	assert_raises(ValueError, validate_input, X + 0.5, "X", ohe=True)

	# only_warn=True surfaces a TangermemeWarning instead of raising.
	with pytest.warns(TangermemeWarning):
		validate_input(X + 0.5, "X", ohe=True, only_warn=True)


##


def test_characters_ohe():
	seq = 'GCTAC'
	ohe = torch.tensor([
		[0, 0, 0, 1, 0],
		[0, 1, 0, 0, 1],
		[1, 0, 0, 0, 0],
		[0, 0, 1, 0, 0]
	])

	seq_chars = characters(ohe)

	assert isinstance(seq_chars, str)
	assert len(seq_chars) == 5
	assert seq_chars == seq


def test_character_pwm():
	seq = 'GCTAC'
	ohe = torch.tensor([
		[0.25, 0.00, 0.10, 0.95, 0.00],
		[0.20, 1.00, 1.00, 0.05, 1.00],
		[0.30, 0.00, 0.30, 0.00, 0.00],
		[0.25, 0.00, 3.00, 0.00, 0.00]
	])

	seq_chars = characters(ohe)

	assert isinstance(seq_chars, str)
	assert len(seq_chars) == 5
	assert seq_chars == seq


def test_characters_alphabet():
	seq = 'GCTAC'
	ohe = torch.tensor([
		[0.25, 0.00, 0.10, 0.95, 0.00],
		[0.20, 1.00, 1.00, 0.05, 1.00],
		[0.30, 0.00, 0.30, 0.00, 0.00],
		[0.25, 0.00, 3.00, 0.00, 0.00],
		[0.00, 0.00, 0.00, 0.00, 0.00]
	])

	seq_chars = characters(ohe, ['A', 'C', 'G', 'T', 'N'])

	assert isinstance(seq_chars, str)
	assert len(seq_chars) == 5
	assert seq_chars == seq


def test_characters_raise_alphabet():
	seq = 'GCTAC'
	ohe = torch.tensor([
		[0.25, 0.00, 0.10, 0.95, 0.00],
		[0.20, 1.00, 1.00, 0.05, 1.00],
		[0.30, 0.00, 0.30, 0.00, 0.00],
		[0.25, 0.00, 3.00, 0.00, 0.00]
	])

	assert_raises(ValueError, characters, ohe, ['A', 'C', 'G'])
	assert_raises(ValueError, characters, ohe, ['A', 'C', 'G', 'T', 'N'])


def test_characters_raise_dimensions():
	seq = 'GCTAC'
	#this will work for shape (1,4,5) but not for (N,4,5) where N > 1
	ohe = torch.tensor([[
		[0.25, 0.00, 0.10, 0.95, 0.00],
		[0.20, 1.00, 1.00, 0.05, 1.00],
		[0.30, 0.00, 0.30, 0.00, 0.00],
		[0.25, 0.00, 3.00, 0.00, 0.00]
	]])
	
	assert characters(ohe) == seq
	
	ohe = torch.concat([ohe, ohe], dim=0)
	assert_raises(ValueError, characters, ohe, ['A', 'C', 'G', 'T'])
	
	ohe = torch.tensor([0.25, 0.00, 0.10, 0.95, 0.00])
	assert_raises(ValueError, characters, ohe, ['A', 'C', 'G', 'T'])


def test_characters_raise_ties():
	seq = 'GCTAC'
	ohe = torch.tensor([
		[0.25, 0.00, 0.10, 0.95, 0.00],
		[0.20, 1.00, 1.00, 0.05, 1.00],
		[0.30, 1.00, 0.30, 0.00, 0.00],
		[0.25, 0.00, 3.00, 0.00, 0.00]
	])

	assert_raises(ValueError, characters, ohe, ['A', 'C', 'G', 'T'])
	assert characters(ohe, force=True) == seq


##


def test_one_hot_encode():
	seq = 'ACGTA'
	ohe = torch.tensor([
		[1, 0, 0, 0, 1],
		[0, 1, 0, 0, 0],
		[0, 0, 1, 0, 0],
		[0, 0, 0, 1, 0]
	])
	seq_ohe = one_hot_encode(seq)

	assert seq_ohe.dtype == torch.int8
	assert seq_ohe.shape == (4, 5)
	assert torch.all(seq_ohe == ohe)

	seq = 'CCGTC'
	ohe = torch.tensor([
		[0, 0, 0, 0, 0],
		[1, 1, 0, 0, 1],
		[0, 0, 1, 0, 0],
		[0, 0, 0, 1, 0]
	])
	seq_ohe = one_hot_encode(seq)

	assert seq_ohe.dtype == torch.int8
	assert seq_ohe.shape == (4, 5)
	assert torch.all(seq_ohe == ohe)


	seq = 'AAAAA'
	ohe = torch.tensor([
		[1, 1, 1, 1, 1],
		[0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0]
	])
	seq_ohe = one_hot_encode(seq)

	assert seq_ohe.dtype == torch.int8
	assert seq_ohe.shape == (4, 5)
	assert torch.all(seq_ohe == ohe)


def test_one_hot_encode_N():
	seq = 'ACGNNTA'
	ohe = torch.tensor([
		[1, 0, 0, 0, 0, 0, 1],
		[0, 1, 0, 0, 0, 0, 0],
		[0, 0, 1, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 1, 0]
	])
	seq_ohe = one_hot_encode(seq)

	assert seq_ohe.dtype == torch.int8
	assert seq_ohe.shape == (4, 7)
	assert torch.all(seq_ohe == ohe)

	seq = 'NNNNN'
	ohe = torch.tensor([
		[0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0]
	])
	seq_ohe = one_hot_encode(seq)

	assert seq_ohe.dtype == torch.int8
	assert seq_ohe.shape == (4, 5)
	assert torch.all(seq_ohe == ohe)


def test_one_hot_encode_dtype():
	seq = 'ACGTA'
	ohe = torch.tensor([
		[1, 0, 0, 0, 1],
		[0, 1, 0, 0, 0],
		[0, 0, 1, 0, 0],
		[0, 0, 0, 1, 0]
	])
	seq_ohe = one_hot_encode(seq, dtype=torch.float32)

	assert seq_ohe.dtype == torch.float32
	assert seq_ohe.shape == (4, 5)
	assert torch.all(seq_ohe == ohe)


def test_one_hot_encode_alphabet():
	seq = 'ACGTA'
	ohe = torch.tensor([
		[1, 0, 0, 0, 1],
		[0, 1, 0, 0, 0],
		[0, 0, 1, 0, 0],
		[0, 0, 0, 1, 0],
		[0, 0, 0, 0, 0]
	])
	seq_ohe = one_hot_encode(seq, alphabet=['A', 'C', 'G', 'T', 'Z'])
	
	assert seq_ohe.dtype == torch.int8
	assert seq_ohe.shape == (5, 5)
	assert torch.all(seq_ohe == ohe)


def test_one_hot_encode_ignore():
	seq = 'ACGTA'
	ohe = torch.tensor([
		[1, 0, 0, 0, 1],
		[0, 1, 0, 0, 0],
		[0, 0, 1, 0, 0],
		[0, 0, 0, 1, 0],
		[0, 0, 0, 0, 0]
	])
	seq_ohe = one_hot_encode(seq, alphabet=['A', 'C', 'G', 'T', 'N'], ignore=[])
	
	assert seq_ohe.dtype == torch.int8
	assert seq_ohe.shape == (5, 5)
	assert torch.all(seq_ohe == ohe)


def test_one_hot_encode_alphabet():
	seq = 'ACGTA'
	ohe = torch.tensor([
		[1, 0, 0, 0, 1],
		[0, 1, 0, 0, 0],
		[0, 0, 1, 0, 0],
		[0, 0, 0, 1, 0],
		[0, 0, 0, 0, 0]
	])
	seq_ohe = one_hot_encode(seq, alphabet=['A', 'C', 'G', 'T', 'Z'])
	
	assert seq_ohe.dtype == torch.int8
	assert seq_ohe.shape == (5, 5)
	assert torch.all(seq_ohe == ohe)


def test_one_hot_encode_raises_alphabet():
	seq = 'ACGTA'
	ohe = torch.tensor([
		[1, 0, 0, 0, 1],
		[0, 1, 0, 0, 0],
		[0, 0, 1, 0, 0],
		[0, 0, 0, 1, 0]
	])

	assert_raises(ValueError, one_hot_encode, seq, ['A', 'C', 'G'])


def test_one_hot_encode_raises_ignore():
	seq = 'ACGTA'
	ohe = torch.tensor([
		[1, 0, 0, 0, 1],
		[0, 1, 0, 0, 0],
		[0, 0, 1, 0, 0],
		[0, 0, 0, 1, 0],
		[0, 0, 0, 0, 0]
	])

	assert_raises(ValueError, one_hot_encode, seq, ['A', 'C', 'G', 'T', 'N'])


def test_one_hot_encode_lower_raises():
	seq = 'AcgTA'
	ohe = torch.tensor([
		[1, 0, 0, 0, 1],
		[0, 1, 0, 0, 0],
		[0, 0, 1, 0, 0],
		[0, 0, 0, 1, 0]
	])

	assert_raises(ValueError, one_hot_encode, seq)


###


def test_reverse_complement():
	assert 'TTGACC' == reverse_complement('GGTCAA')
	assert 'TTTTTT' == reverse_complement('AAAAAA')
	assert 'ACGT' == reverse_complement('ACGT')

	f = one_hot_encode
	assert_array_almost_equal(f('TTGACC'), reverse_complement(f('GGTCAA')))
	assert_array_almost_equal(f('TTTTTT'), reverse_complement(f('AAAAAA')))
	assert_array_almost_equal(f('ACGT'), reverse_complement(f('ACGT')))


def test_reverse_complement_mapping():
	assert 'AAAAAA' == reverse_complement('GGTCAA', 
		complement_map={'A': 'A', 'C': 'A', 'G': 'A', 'T': 'A'})

	assert 'TTGACC' == reverse_complement('GGTCaa',
		complement_map={'a': 'T', 'C': 'G', 'G': 'C', 'T': 'A'})

	f = one_hot_encode
	assert_array_almost_equal(f('TTGACC'), reverse_complement(f('GGTCAA')))
	assert_raises(AssertionError, assert_array_almost_equal, f('TTGACC'), 
		reverse_complement(f('GGTCAA'), complement_map={'A': 'T', 'G': 'C', 
			'T': 'A', 'C': 'G'}))


def test_reverse_reverse_complement():
	rc = reverse_complement
	f = one_hot_encode

	assert 'TTGACC' == rc(rc('TTGACC'))
	assert 'TTTTTT' == rc(rc('TTTTTT'))
	assert 'ACGT' == rc(rc('ACGT'))

	assert_array_almost_equal(f('TTGACC'), rc(rc(f('TTGACC'))))
	assert_array_almost_equal(f('TTTTTT'), rc(rc(f('TTTTTT'))))
	assert_array_almost_equal(f('ACGT'), rc(rc(f('ACGT'))))


def test_reverse_complement_Ns():
	assert 'TTGACCN' == reverse_complement('NGGTCAA')
	assert 'TTTNTTT' == reverse_complement('AAANAAA')
	assert 'NNN' == reverse_complement('NNN')

	f = one_hot_encode
	assert_array_almost_equal(f('TTGACCN'), reverse_complement(f('NGGTCAA')))
	assert_array_almost_equal(f('TTTNTTT'), reverse_complement(f('AAANAAA')))
	assert_array_almost_equal(f('NNN'), reverse_complement(f('NNN')))


def test_reverse_complement_disallow_Ns():
	assert_raises(ValueError, reverse_complement, "TTTGAN", allow_N=False)
	assert_raises(ValueError, reverse_complement, "N", allow_N=False)
	assert_raises(ValueError, reverse_complement, "NNNN", allow_N=False)


def test_reverse_complement_raises():
	assert_raises(ValueError, reverse_complement, "ZZ")
	assert_raises(ValueError, reverse_complement, "aCGT")


###


def test_random_one_hot():
	ohe = random_one_hot((2, 4, 9), random_state=0)

	assert ohe.shape == (2, 4, 9)
	assert ohe.dtype == torch.int8
	assert ohe.sum(axis=-2).max() == 1
	assert ohe.sum(axis=-2).min() == 1
	assert ohe.sum() == 18
	assert_array_almost_equal(ohe, [
		[[1, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 1, 1, 1, 1, 0]],

        [[0, 0, 0, 1, 0, 0, 1, 1, 1],
         [0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 1, 0, 0, 0],
         [1, 0, 0, 0, 1, 0, 0, 0, 0]]])


	ohe = random_one_hot((10, 20, 15), random_state=0)

	assert ohe.shape == (10, 20, 15)
	assert ohe.dtype == torch.int8
	assert ohe.sum(axis=-2).max() == 1
	assert ohe.sum(axis=-2).min() == 1
	assert ohe.sum() == 150


def test_random_one_hot_raises_type():
	assert_raises(ValueError, random_one_hot, [0, 3, 2])
	assert_raises(ValueError, random_one_hot, 3)


def test_random_one_hot_raises_shape():
	assert_raises(ValueError, random_one_hot, (2, 3))
	assert_raises(ValueError, random_one_hot, (3,))


###


def test_chunk():
	X0 = [torch.randn(4, 10), torch.randn(4, 14)]
	X = chunk(X0, size=2)

	assert X.dtype == X0[0].dtype
	assert X.shape == (12, 4, 2)
	assert_array_almost_equal(X, torch.stack([X0[0][:, :2], X0[0][:, 2:4], 
		X0[0][:, 4:6], X0[0][:, 6:8], X0[0][:, 8:10], X0[1][:, :2], 
		X0[1][:, 2:4], X0[1][:, 4:6], X0[1][:, 6:8], X0[1][:, 8:10],
		X0[1][:, 10:12], X0[1][:, 12:14]]))

	X0 = [torch.randn(4, 10), torch.randn(4, 14)]
	X = chunk(X0, size=5)

	assert X.dtype == X0[0].dtype
	assert X.shape == (4, 4, 5)


def test_chunk_overlap():
	X0 = [torch.randn(4, 10), torch.randn(4, 14)]

	X = chunk(X0, size=2, overlap=1)
	assert X.shape == (22, 4, 2)
	assert_array_almost_equal(X[:10], torch.stack([X0[0][:, :2], X0[0][:, 1:3], 
		X0[0][:, 2:4], X0[0][:, 3:5], X0[0][:, 4:6], X0[0][:, 5:7], 
		X0[0][:, 6:8], X0[0][:, 7:9], X0[0][:, 8:10], X0[1][:, :2]]))


	X = chunk(X0, size=5, overlap=2)
	assert X.shape == (6, 4, 5)
	assert_array_almost_equal(X, torch.stack([X0[0][:, :5], X0[0][:, 3:8], 
		X0[1][:, :5], X0[1][:, 3:8], X0[1][:, 6:11], X0[1][:, 9:]]))


def test_chunk_raises():
	assert_raises(ValueError, chunk, [torch.randn(4, 10)], -1)
	assert_raises(ValueError, chunk, [torch.randn(4, 10)], 1.2)

	assert_raises(ValueError, chunk, [torch.randn(4, 10)], 2, -1)
	assert_raises(ValueError, chunk, [torch.randn(4, 10)], 2, 0.3)

	assert_raises(ValueError, chunk, torch.randn(4, 10), 2)
	assert_raises(ValueError, chunk, torch.randn(3, 4, 10), 2)


###


def test_unchunk():
	lengths = [50, 67]
	X0 = [torch.randn(4, lengths[0]), torch.randn(4, lengths[1])]

	X = chunk(X0, size=4, overlap=3)
	X1 = unchunk(X, lengths, overlap=3)
	assert_array_almost_equal(X1[0], X0[0])
	assert_array_almost_equal(X1[1], X0[1])

	X = chunk(X0, size=4, overlap=2)
	X1 = unchunk(X, lengths, overlap=2)
	assert_array_almost_equal(X1[0], X0[0])
	assert_array_almost_equal(X1[1], X0[1][:, :66])

	X = chunk(X0, size=23, overlap=8)
	X1 = unchunk(X, lengths, overlap=8)
	assert_array_almost_equal(X1[0], X0[0][:, :38])
	assert_array_almost_equal(X1[1], X0[1][:, :53])

	X = chunk(X0, size=23, overlap=0)
	X1 = unchunk(X, lengths, overlap=0)
	assert_array_almost_equal(X1[0], X0[0][:, :46])
	assert_array_almost_equal(X1[1], X0[1][:, :46])

	X0 = X0[:1]
	X = chunk(X0, size=23, overlap=6)
	X1 = unchunk(X, lengths[:1], overlap=6)
	assert_array_almost_equal(X1[0], X0[0][:, :40])


def test_unchunk_raises():
	lengths = [50, 67]
	X0 = [torch.randn(4, lengths[0]), torch.randn(4, lengths[1])]
	X = chunk(X0[:1], size=7, overlap=2)

	assert_raises(IndexError, unchunk, X, lengths, overlap=2)
	assert_raises(ValueError, unchunk, X[0], lengths[:1], overlap=2)


###


def test_extract_signal(device):
	loci = pandas.DataFrame({
		'example_idxs': [0, 0, 1, 2],
		'start': [1, 9, 2, 4],
		'end': [8, 14, 10, 12]
	})

	X = numpy.random.RandomState(0).randn(3, 1, 15)
	X = torch.from_numpy(X).to(device)

	y = extract_signal(loci, X)
	
	assert y.shape == (4, 1)
	assert_array_almost_equal(y.cpu(), [[ 5.3088  ],
                  [ 2.891628],
                  [-0.253532],
                  [-0.917093]])

	assert y[0, 0] == X[0, :, 1:8].sum()
	assert y[1, 0] == X[0, :, 9:14].sum()
	assert y[2, 0] == X[1, :, 2:10].sum()
	assert y[3, 0] == X[2, :, 4:12].sum()


###


def test_pwm_consensus_basic():
	pwm = torch.tensor([
		[0.7, 0.1, 0.1, 0.1],
		[0.1, 0.7, 0.1, 0.1],
		[0.1, 0.1, 0.7, 0.1],
		[0.1, 0.1, 0.1, 0.7],
	])
	# pwm_consensus expects shape=(alphabet_len, pwm_len)
	Y = pwm_consensus(pwm.T)

	assert Y.shape == pwm.T.shape
	assert int(Y.sum()) == pwm.shape[0]
	assert_array_almost_equal(Y, torch.eye(4))


def test_pwm_consensus_zero_column():
	pwm = torch.tensor([
		[0.7, 0.0, 0.1, 0.1],
		[0.1, 0.0, 0.7, 0.1],
		[0.1, 0.0, 0.1, 0.7],
		[0.1, 0.0, 0.1, 0.1],
	])
	Y = pwm_consensus(pwm)

	# The all-zero column stays all-zero.
	assert int(Y[:, 1].sum()) == 0
	assert int(Y[:, 0].sum()) == 1


def test_random_one_hot_with_probs():
	X = random_one_hot((100, 4, 200), probs=[[1.0, 0.0, 0.0, 0.0]],
		random_state=0)

	assert X.shape == (100, 4, 200)
	# every position is the first character
	assert int(X[:, 0].sum()) == 100 * 200
	assert int(X[:, 1:].sum()) == 0


def test_example_to_fasta_coords_basic():
	loci_df = pandas.DataFrame({
		'chrom': ['chr1', 'chr2'],
		'start': [100, 500],
		'end': [200, 600],
	})
	example_df = pandas.DataFrame({
		'example_idx': [0, 1, 0],
		'start': [5, 10, 50],
		'end': [15, 20, 60],
		'score': [1.0, 2.0, 3.0],
	})

	out = example_to_fasta_coords(example_df, loci_df)

	assert list(out.columns) == ['chrom', 'start', 'end', 'score']
	assert list(out['chrom']) == ['chr1', 'chr2', 'chr1']
	assert list(out['start']) == [105, 510, 150]
	assert list(out['end']) == [115, 520, 160]
	assert list(out['score']) == [1.0, 2.0, 3.0]


def test_example_to_fasta_coords_window():
	loci_df = pandas.DataFrame({
		'chrom': ['chr1'],
		'start': [100],
		'end': [200],
	})
	example_df = pandas.DataFrame({
		'example_idx': [0],
		'start': [10],
		'end': [20],
	})

	out = example_to_fasta_coords(example_df, loci_df, window=50)

	# midpoint of (100, 200) is 150; window=50 centers at 150-25=125;
	# example start=10 -> 135, end=20 -> 145.
	assert int(out['start'].iloc[0]) == 135
	assert int(out['end'].iloc[0]) == 145
