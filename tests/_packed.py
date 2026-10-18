# _packed.py
# Helpers shared by the tests that pass PackedSequences to functions.

import torch

from tangermeme.utils import PackedSequence


def assert_same(y, y_packed):
	"""Assert that two outputs, tensors or nested lists and tuples of them,
	have the same dtypes and values."""

	if isinstance(y, torch.Tensor):
		assert y_packed.dtype == y.dtype
		assert torch.equal(y_packed, y)
	else:
		assert len(y_packed) == len(y)
		for y_, y_packed_ in zip(y, y_packed):
			assert_same(y_, y_packed_)


def assert_packed_equal(X_packed, Y_packed):
	"""Assert that two PackedSequences hold the same sequences."""

	assert isinstance(X_packed, PackedSequence)
	assert X_packed.shape == Y_packed.shape
	assert X_packed.alphabet == Y_packed.alphabet
	assert torch.equal(X_packed.data, Y_packed.data)
	assert torch.equal(X_packed.n_runs, Y_packed.n_runs)


def with_N(X):
	"""A copy of X, of at least four sequences, with unknown characters at
	the start of the first, in the middle of the fourth and at the end of the
	last."""

	X = X.clone()
	X[0, :, :5] = 0
	X[3, :, 40:47] = 0
	X[-1, :, -1] = 0
	return X


class Spy:
	"""A function of the user's, not marked as taking a PackedSequence, that
	records the dtype of the sequences it is given, positional argument
	`position`, and calls `func`."""

	def __init__(self, func, position):
		self.func, self.position, self.dtypes = func, position, []

	def __call__(self, *args, **kwargs):
		X = args[self.position]
		assert isinstance(X, torch.Tensor)
		self.dtypes.append(X.dtype)
		return self.func(*args, **kwargs)


def check_packed(fn, X, spies=()):
	"""Assert that `fn` gives the same output for X and for its
	PackedSequence, and that each of `spies` was given int8 one-hot
	sequences during the packed call."""

	y = fn(X)
	for spy in spies:
		spy.dtypes.clear()

	y_packed = fn(PackedSequence.from_one_hot(X))
	for spy in spies:
		assert len(spy.dtypes) > 0
		assert all(dtype == torch.int8 for dtype in spy.dtypes)

	assert_same(y, y_packed)


def soft_references(X, n=1, random_state=None):
	"""Uniform references, which are not a one-hot encoding."""

	return torch.full((X.shape[0], n, X.shape[1], X.shape[2]), 0.25)


class MixedReferences:
	"""References from `shuffle_fn` for the first `n_one_hot` calls and
	uniform ones after them."""

	def __init__(self, shuffle_fn, n_one_hot=1):
		self.shuffle_fn, self.n_one_hot, self.calls = shuffle_fn, n_one_hot, 0

	def __call__(self, X, n=1, random_state=None):
		self.calls += 1
		if self.calls <= self.n_one_hot:
			return self.shuffle_fn(X, n=n, random_state=0)
		return soft_references(X, n=n)
