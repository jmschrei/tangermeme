# test_utils.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from tangermeme.utils import characters
from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


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
	print(seq_chars)
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
	ohe = torch.tensor([[
		[0.25, 0.00, 0.10, 0.95, 0.00],
		[0.20, 1.00, 1.00, 0.05, 1.00],
		[0.30, 0.00, 0.30, 0.00, 0.00],
		[0.25, 0.00, 3.00, 0.00, 0.00]
	]])

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
	seq_ohe = one_hot_encode(seq, dtype='float32')

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


##


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
