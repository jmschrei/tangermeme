# test_variant_effect.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest
import pandas
import pyfaidx
import pyBigWig


from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


from .toy_models import FlattenDense
from .toy_models import SumModel

from tangermeme.variant_effect import marginal_substitution_effect
from tangermeme.variant_effect import marginal_deletion_effect
from tangermeme.variant_effect import marginal_insertion_effect


@pytest.fixture
def substitutions():
	return pandas.DataFrame({
		'chrom': ['chr1', 'chr1', 'chr1', 'chr1', 'chr2', 'chr2', 'chr2'], 
		'pos': [105-1, 110-1, 205-1, 206-1, 120-1, 120-1, 130-1],
		'alt': ['A', 'T', 'C', 'A', 'C', 'C', 'T']
	})



@pytest.fixture
def deletions():
	return pandas.DataFrame({
		'chrom': ['chr1', 'chr1', 'chr1', 'chr1', 'chr2', 'chr2', 'chr2'], 
		'pos': [105-1, 110-1, 205-1, 206-1, 120-1, 120-1, 130-1]
	})


@pytest.fixture
def insertions():
	return pandas.DataFrame({
		'chrom': ['chr1', 'chr1', 'chr1', 'chr1', 'chr2', 'chr2', 'chr2'], 
		'pos': [105-1, 110-1, 205-1, 206-1, 120-1, 120-1, 130-1],
		'alt': ['G', 'C', 'A', 'T', 'T', 'T', 'C']
	})


###


def test_marginal_substitution_effect(substitutions):
	torch.manual_seed(0)
	model = FlattenDense()

	y_orig, y_var = marginal_substitution_effect(model, "tests/data/test.fa", 
		substitutions, in_window=100, device='cpu')

	assert y_orig.shape == (7, 3)
	assert y_var.shape == (7, 3)

	assert_array_almost_equal(y_orig, [
		[-0.0861, -0.0909,  0.0670],
	        [ 0.2204, -0.0814, -0.1494],
	        [ 0.1383,  0.1414, -0.2976],
	        [-0.4860, -0.2796,  0.2967],
	        [ 0.2275,  0.1464, -0.1421],
	        [ 0.2275,  0.1464, -0.1421],
	        [ 0.2072,  0.1621,  0.0526]], 4)

	assert_array_almost_equal(y_var, [
		[-0.0278, -0.0767,  0.0782],
	        [ 0.2497, -0.0742, -0.0882],
	        [ 0.0596,  0.0525, -0.3298],
	        [-0.4365, -0.1979,  0.2678],
	        [ 0.1982,  0.1391, -0.2032],
	        [ 0.1982,  0.1391, -0.2032],
	        [ 0.2364,  0.1693,  0.1138]], 4)


def test_marginal_substitution_effect_counts(substitutions):
	model = SumModel()

	y_orig, y_var = marginal_substitution_effect(model, "tests/data/test.fa", 
		substitutions, in_window=100, device='cpu')

	assert y_orig.shape == (7, 4)
	assert y_var.shape == (7, 4)
	assert torch.abs(y_orig - y_var).max() == 1

	assert_array_almost_equal(y_orig, [
		[25, 28, 20, 27],
	[24, 27, 20, 29],
	[28, 28, 15, 29],
	[28, 29, 15, 28],
	[26, 23, 15, 36],
	[26, 23, 15, 36],
	[27, 23, 15, 35]])

	assert_array_almost_equal(y_var, [
		[26, 28, 19, 27],
	[24, 26, 20, 30],
	[27, 29, 15, 29],
	[29, 29, 15, 27],
	[26, 24, 15, 35],
	[26, 24, 15, 35],
	[27, 22, 15, 36]])


###


def test_marginal_deletion_effect(deletions):
	torch.manual_seed(0)
	model = FlattenDense()

	y_orig, y_var = marginal_deletion_effect(model, "tests/data/test.fa", 
		deletions, in_window=100, device='cpu')

	assert y_orig.shape == (7, 3)
	assert y_var.shape == (7, 3)

	assert_array_almost_equal(y_orig, [
		[-0.0861, -0.0909,  0.0670],
	        [ 0.2204, -0.0814, -0.1494],
	        [ 0.1383,  0.1414, -0.2976],
	        [-0.4860, -0.2796,  0.2967],
	        [ 0.2275,  0.1464, -0.1421],
	        [ 0.2275,  0.1464, -0.1421],
	        [ 0.2072,  0.1621,  0.0526]], 4)

	assert_array_almost_equal(y_var, [
		[-0.1687,  0.3454, -0.1342],
	        [ 0.1319, -0.3355, -0.3974],
	        [-0.1539,  0.1452,  0.1053],
	        [-0.3788, -0.2118, -0.0471],
	        [-0.1478, -0.2921, -0.4550],
	        [-0.1478, -0.2921, -0.4550],
	        [ 0.4193,  0.3293, -0.5298]], 4)


def test_marginal_deletion_effect_counts(deletions):
	model = SumModel()

	y_orig, y_var = marginal_deletion_effect(model, "tests/data/test.fa", 
		deletions, in_window=100, device='cpu')

	assert y_orig.shape == (7, 4)
	assert y_var.shape == (7, 4)
	assert torch.abs(y_orig - y_var).max() == 1

	assert_array_almost_equal(y_orig, [
		[25, 28, 20, 27],
		[24, 27, 20, 29],
		[28, 28, 15, 29],
		[28, 29, 15, 28],
		[26, 23, 15, 36],
		[26, 23, 15, 36],
		[27, 23, 15, 35]])

	assert_array_almost_equal(y_var, [
		[25, 28, 19, 28],
		[24, 26, 21, 29],
		[27, 29, 15, 29],
		[28, 29, 15, 28],
		[26, 24, 15, 35],
		[26, 24, 15, 35],
		[27, 22, 15, 36]])


def test_marginal_deletion_effect_odd(deletions):
	torch.manual_seed(0)
	model = FlattenDense(seq_len=101)

	y_orig, y_var = marginal_deletion_effect(model, "tests/data/test.fa", 
		deletions, in_window=101, device='cpu')

	assert y_orig.shape == (7, 3)
	assert y_var.shape == (7, 3)

	assert_array_almost_equal(y_orig, [
		[-0.0298, -0.1351, -0.1471],
	        [-0.1600,  0.3890, -0.6491],
	        [-0.0354, -0.6043, -0.1170],
	        [ 0.0782,  0.0320, -0.1014],
	        [ 0.3304, -0.1293, -0.6397],
	        [ 0.3304, -0.1293, -0.6397],
	        [ 0.6485, -0.2176, -0.5408]], 4)

	assert_array_almost_equal(y_var, [
		[ 0.0716, -0.4693, -0.1223],
	        [ 0.1453,  0.1379,  0.0679],
	        [ 0.0919, -0.4105, -0.0344],
	        [ 0.0643,  0.0537, -0.1558],
	        [ 0.2453, -0.1651, -0.5010],
	        [ 0.2453, -0.1651, -0.5010],
	        [ 0.7352, -0.2146, -0.3759]], 4)


##


def test_marginal_insertion_effect(insertions):
	torch.manual_seed(0)
	model = FlattenDense()

	y_orig, y_var = marginal_insertion_effect(model, "tests/data/test.fa", 
		insertions, in_window=100, device='cpu')

	assert y_orig.shape == (7, 3)
	assert y_var.shape == (7, 3)

	assert_array_almost_equal(y_orig, [
		[-0.0861, -0.0909,  0.0670],
	        [ 0.2204, -0.0814, -0.1494],
	        [ 0.1383,  0.1414, -0.2976],
	        [-0.4860, -0.2796,  0.2967],
	        [ 0.2275,  0.1464, -0.1421],
	        [ 0.2275,  0.1464, -0.1421],
	        [ 0.2072,  0.1621,  0.0526]], 4)

	assert_array_almost_equal(y_var, [
		[-0.0135,  0.1575, -0.1878],
	        [ 0.2277, -0.2634,  0.2301],
	        [ 0.2199, -0.0812, -0.2957],
	        [-0.2433, -0.3650, -0.0772],
	        [-0.0676,  0.0197, -0.1539],
	        [-0.0676,  0.0197, -0.1539],
	        [ 0.0320,  0.4506, -0.1432]], 4)


def test_marginal_insertion_effect_counts(insertions):
	model = SumModel()

	y_orig, y_var = marginal_insertion_effect(model, "tests/data/test.fa", 
		insertions, in_window=100, device='cpu')

	assert y_orig.shape == (7, 4)
	assert y_var.shape == (7, 4)
	assert torch.abs(y_orig - y_var).max() == 1

	assert_array_almost_equal(y_orig, [
		[25, 28, 20, 27],
		[24, 27, 20, 29],
		[28, 28, 15, 29],
		[28, 29, 15, 28],
		[26, 23, 15, 36],
		[26, 23, 15, 36],
		[27, 23, 15, 35]])

	assert_array_almost_equal(y_var, [
		[24, 28, 21, 27],
		[24, 28, 20, 28],
		[29, 28, 15, 28],
		[28, 28, 15, 29],
		[26, 23, 14, 37],
		[26, 23, 14, 37],
		[26, 24, 15, 35]])


def test_marginal_insertion_effect_odd(insertions):
	torch.manual_seed(0)
	model = FlattenDense(seq_len=101)

	y_orig, y_var = marginal_insertion_effect(model, "tests/data/test.fa", 
		insertions, in_window=101, device='cpu')

	assert y_orig.shape == (7, 3)
	assert y_var.shape == (7, 3)

	assert_array_almost_equal(y_orig, [
		[-0.0298, -0.1351, -0.1471],
	        [-0.1600,  0.3890, -0.6491],
	        [-0.0354, -0.6043, -0.1170],
	        [ 0.0782,  0.0320, -0.1014],
	        [ 0.3304, -0.1293, -0.6397],
	        [ 0.3304, -0.1293, -0.6397],
	        [ 0.6485, -0.2176, -0.5408]], 4)

	assert_array_almost_equal(y_var, [
		[-0.2063, -0.1620, -0.1345],
	        [-0.1808,  0.1657, -0.1853],
	        [ 0.0188, -0.3411, -0.3747],
	        [-0.0214, -0.6259, -0.0627],
	        [ 0.2207, -0.1184, -0.0698],
	        [ 0.2207, -0.1184, -0.0698],
	        [-0.1907, -0.0478, -0.2484]], 4)
	