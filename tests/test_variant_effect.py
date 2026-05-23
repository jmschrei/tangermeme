# test_variant_effect.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest
import pandas
import pyfaidx


from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal

from .toy_models import FlattenDense
from .toy_models import SumModel
from .toy_models import SmallDeepSEA

from tangermeme.utils import random_one_hot

from tangermeme.variant_effect import substitution_effect
from tangermeme.variant_effect import deletion_effect
from tangermeme.variant_effect import insertion_effect


@pytest.fixture
def X():
	torch.manual_seed(0)
	return random_one_hot((3, 4, 100), random_state=0)


@pytest.fixture
def X_del():
	torch.manual_seed(0)
	return random_one_hot((3, 4, 103), random_state=0)


@pytest.fixture
def substitutions():
	return torch.tensor([
		[0, 4, 1],
		[0, 5, 2],
		[1, 2, 1],
		[1, 3, 1],
		[1, 8, 0]])


@pytest.fixture
def deletions():
	return torch.tensor([
		[0, 4],
		[0, 5],
		[1, 2],
		[1, 3],
		[1, 8]])


###


def test_substitution_effect(X, substitutions, device):
	model = SmallDeepSEA()
	y, y_var = substitution_effect(model, X, substitutions, device=device)

	assert y.shape == (3, 1)
	assert y_var.shape == (3, 1)

	assert_array_almost_equal(y, [[-0.0851], [-0.0855], [-0.0644]], 4)
	assert_array_almost_equal(y_var, [[-0.0835], [-0.0963], [-0.0644]], 4)


def test_substitution_effect_summodel(X, substitutions, device):
	model = SumModel()
	y, y_var = substitution_effect(model, X, substitutions, device=device)

	assert y.shape == (3, 4)
	assert y_var.shape == (3, 4)

	assert all(y.sum(dim=-1) == 100)
	assert all(y_var.sum(dim=-1) == 100)

	assert_array_almost_equal(y, [
		[25, 24, 19, 32],
        [26, 18, 29, 27],
        [28, 25, 21, 26]])
	assert_array_almost_equal(y_var, [
		[25, 25, 20, 30],
        [26, 20, 29, 25],
        [28, 25, 21, 26]])


###


def test_deletion_effect(X_del, deletions, device):
	model = SmallDeepSEA()
	y, y_var = deletion_effect(model, X_del, deletions, device=device)

	assert y.shape == (3, 1)
	assert y_var.shape == (3, 1)

	assert_array_almost_equal(y, [[-0.0851], [-0.0976], [-0.0940]], 4)
	assert_array_almost_equal(y_var, [[-0.0814], [-0.0852], [-0.0940]], 4)


def test_deletion_effect_summodel(X_del, substitutions, device):
	model = SumModel()
	y, y_var = deletion_effect(model, X_del, substitutions, device=device)

	assert y.shape == (3, 4)
	assert y_var.shape == (3, 4)

	assert all(y.sum(dim=-1) == 100)
	assert all(y_var.sum(dim=-1) == 100)

	assert_array_almost_equal(y, [
		[25, 24, 19, 32],
        [25, 18, 29, 28],
        [30, 25, 19, 26]])
	assert_array_almost_equal(y_var, [
		[25, 25, 20, 30],
        [23, 18, 30, 29],
        [30, 25, 19, 26]])

	assert_array_almost_equal(y, X_del[:, :, :100].sum(dim=-1))


###


def test_insertion_effect(X, substitutions, device):
	model = SmallDeepSEA()
	y, y_var = insertion_effect(model, X, substitutions, device=device)

	assert y.shape == (3, 1)
	assert y_var.shape == (3, 1)

	assert_array_almost_equal(y, [[-0.0851], [-0.0855], [-0.0644]], 4)
	assert_array_almost_equal(y_var, [[-0.0676], [-0.1054], [-0.0644]], 4)


def test_insertion_effect_summodel(X, substitutions, device):
	model = SumModel()
	y, y_var = insertion_effect(model, X, substitutions, device=device)

	assert y.shape == (3, 4)
	assert y_var.shape == (3, 4)

	assert_array_almost_equal(y, [
		[25, 24, 19, 32],
        [26, 18, 29, 27],
        [28, 25, 21, 26]])
	assert_array_almost_equal(y_var, [
		[24., 24., 20., 32.],
        [26., 19., 28., 27.],
        [28., 25., 21., 26.]])


###


def test_deletion_effect_left(X_del, deletions, device):
	model = SumModel()

	# left=False trims from the right; left=True trims from the left.
	y_right, y_var_right = deletion_effect(model, X_del, deletions,
		left=False, device=device)
	y_left, y_var_left = deletion_effect(model, X_del, deletions,
		left=True, device=device)

	# y_before reflects which side was trimmed: right-trim uses [:100],
	# left-trim uses [-100:].
	assert_array_almost_equal(y_right, X_del[:, :, :100].sum(dim=-1))
	assert_array_almost_equal(y_left, X_del[:, :, -100:].sum(dim=-1))


def test_insertion_effect_left(X, substitutions, device):
	model = SumModel()

	y_right, y_var_right = insertion_effect(model, X, substitutions,
		left=False, device=device)
	y_left, y_var_left = insertion_effect(model, X, substitutions,
		left=True, device=device)

	# Both runs return the same y_before (predict on the unmodified X).
	assert_array_almost_equal(y_right, y_left)


def test_substitution_effect_no_variant_for_example(X, device):
	model = SumModel()

	# Only example 0 has a variant; example 1 and 2 are untouched.
	substitutions = torch.tensor([[0, 4, 1]])
	y, y_var = substitution_effect(model, X, substitutions, device=device)

	# example 0 differs, examples 1 and 2 are identical before/after
	assert_array_almost_equal(y[1], y_var[1])
	assert_array_almost_equal(y[2], y_var[2])
	assert_raises(AssertionError, assert_array_almost_equal, y[0], y_var[0])


def test_substitution_effect_does_not_mutate_X(X, substitutions, device):
	model = SumModel()
	X_before = X.clone()
	substitution_effect(model, X, substitutions, device=device)

	# substitution_effect clones X internally, so the input is unchanged.
	assert_array_almost_equal(X, X_before)
