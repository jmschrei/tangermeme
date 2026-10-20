# test_seqlet.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import re
import math
import itertools

import numpy
import scipy.stats
import torch
import warnings
import pandas
import pytest

from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot

from tangermeme.ersatz import substitute
from tangermeme.ersatz import shuffle

from tangermeme.seqlet import _laplacian_null
from tangermeme.seqlet import _isotonic_thresholds

from tangermeme.seqlet import tfmodisco_seqlets
from tangermeme.seqlet import recursive_seqlets
from tangermeme.seqlet import RecursiveSeqletsResult
from tangermeme.seqlet import _recursive_seqlet_tables
from tangermeme.seqlet import _recursive_seqlets_fdr
from tangermeme.seqlet import _recursive_seqlet_null
from tangermeme.seqlet import _recursive_seqlet_decode
from tangermeme.seqlet import _as_attributions

from tangermeme.utils import TangermemeWarning

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from pandas.testing import assert_frame_equal


@pytest.fixture
def X():
	return torch.load("tests/data/X_gata.torch")


@pytest.fixture
def X_attr():
	return torch.load("tests/data/X_gata_attr.torch")


@pytest.fixture
def X_contrib(X, X_attr):
	return (X * X_attr).sum(axis=1)

###


def test_laplacian_null(X_contrib):
	pos_vals, neg_vals = _laplacian_null(X_contrib, 100)

	assert isinstance(pos_vals, torch.Tensor)
	assert isinstance(neg_vals, torch.Tensor)
	assert len(pos_vals) == 46
	assert len(neg_vals) == 54

	assert_array_almost_equal(pos_vals, [
		8.4230e-07, 1.3388e-06, 1.4090e-06, 5.9829e-07, 7.1113e-07, 8.2684e-07,
        3.9205e-07, 1.3527e-06, 7.2504e-07, 2.2601e-06, 2.0950e-06, 1.6694e-07,
        9.7432e-07, 7.1160e-07, 6.0176e-07, 8.0832e-07, 8.4884e-07, 3.6767e-08,
        9.9743e-08, 2.0088e-06, 6.6184e-07, 3.0202e-07, 8.1946e-07, 7.2568e-07,
        9.5970e-07, 7.7910e-07, 2.6319e-06, 1.7704e-07, 1.6512e-07, 1.3225e-07,
        9.2874e-07, 3.7819e-08, 7.7175e-07, 1.1446e-07, 7.2544e-07, 2.0149e-06,
        3.4360e-07, 8.8779e-08, 1.0399e-06, 1.7885e-06, 1.1536e-06, 1.4078e-06,
        1.5832e-08, 1.1397e-06, 1.1607e-06, 3.9320e-07], 4)

	assert_array_almost_equal(neg_vals, [
		-5.6500e-06, -3.6982e-05, -2.2107e-05, -2.5744e-07, -3.7967e-05,
        -1.8679e-05, -1.0163e-05, -2.7614e-06, -2.1613e-05, -1.3512e-05,
        -1.3207e-07, -4.3092e-05, -8.5727e-05, -2.7825e-05, -3.8693e-06,
        -7.1185e-05, -1.5690e-05, -2.0191e-06, -1.7422e-05, -2.8874e-06,
        -3.1507e-05, -1.0230e-05, -5.3595e-05, -1.2382e-05, -3.0311e-05,
        -2.1826e-05, -1.4466e-05, -2.8317e-06, -4.4845e-05, -5.1700e-06,
        -2.1730e-05, -4.0818e-05, -1.6207e-05, -6.0059e-06, -3.5166e-06,
        -1.4431e-05, -2.7339e-06, -5.6610e-06, -1.9533e-05, -2.7035e-05,
        -4.3557e-05, -1.4896e-05, -3.9108e-05, -3.0262e-05, -1.0643e-05,
        -6.4172e-05, -8.0868e-06, -2.1420e-05, -1.2069e-05, -2.2069e-05,
        -7.4917e-06, -2.4018e-06, -1.9395e-06, -5.1572e-05], 4)


def test_laplacian_null_rand(X_contrib):
	torch.manual_seed(0)
	pos_vals, neg_vals = _laplacian_null(torch.randn_like(X_contrib), 100)

	assert isinstance(pos_vals, torch.Tensor)
	assert isinstance(neg_vals, torch.Tensor)
	assert len(pos_vals) == 48
	assert len(neg_vals) == 52

	assert_array_almost_equal(pos_vals, [
		0.59387423, 1.02254818, 1.08321971, 0.38318206, 0.48061512,
        0.58053032, 0.20510862, 1.03454702, 0.49263115, 1.81808662,
        1.67551057, 0.01073448, 0.70786767, 0.48102437, 0.38618251,
        0.56453733, 0.59952656, 0.02138944, 2.90117667, 1.60113665,
        0.43805995, 0.12736854, 0.57415862, 0.49317699, 2.14868018,
        0.38556177, 1.15258841, 0.69524781, 0.53930147, 2.13908425,
        0.01945422, 1.77433517, 0.00915822, 0.66851275, 0.53296131,
        0.49296945, 1.60633202, 0.16327354, 0.09803651, 0.76451487,
        0.31117045, 1.41090796, 0.86266415, 1.08218924, 0.85068358,
        0.17635525, 0.86881156, 0.2061009 ], 4)

	assert_array_almost_equal(neg_vals, [
		-0.36077993, -1.56747789, -0.99458238, -0.13312904, -1.60540417,
        -0.8625858 , -0.53458036, -0.24953233, -0.97556508, -0.66358447,
        -0.14826666, -1.8027814 , -3.44481374, -1.21482428, -0.10166643,
        -0.04728932, -0.74747094, -0.05776246, -0.81416415, -0.25438378,
        -1.35663542, -0.53717052, -0.98377615, -0.70029995, -0.25223898,
        -0.34229547, -0.9800762 , -1.71521117, -0.76734997, -0.0192245 ,
        -0.10075857, -0.37448858, -0.03458398, -0.27861476, -0.698965  ,
        -0.24847222, -0.89547868, -0.05675626, -1.18440235, -1.8207185 ,
        -0.71686298, -1.64935706, -1.30868821, -2.6146493 , -0.4546292 ,
        -0.96815522, -0.60798062, -0.99314707, -0.11974332, -0.2356805 ,
        -0.21787745, -2.12939304], 4)


###


def test_isotonic_thresholds(X_contrib):
	X_neg = torch.randn(100)
	X_pos = X_contrib[X_contrib > 0.03]

	vals = _isotonic_thresholds(X_pos, X_neg, increasing=True, target_fdr=0.2, 
		min_frac_neg=0.95)

	assert_array_almost_equal(vals, 0.049, 3)


@pytest.mark.parametrize("target_fdr, min_frac_neg, expected", [
	(0.01, 0.95, 0.027),
	(0.05, 0.95, 0.027),
	(0.5, 0.95, 0.035),
	(0.99, 0.95, 0.035),
	(0.2, 0.05, 0.035),
	(0.2, 0.15, 0.035),
	(0.2, 0.50, 0.027),
	(0.2, 0.90, 0.027),
])
def test_isotonic_thresholds_fdr_min_frac(X_contrib, target_fdr, min_frac_neg,
	expected):
	torch.manual_seed(0)
	X_neg = torch.randn(10000)
	X_pos = X_contrib[X_contrib > 0.02]

	val = _isotonic_thresholds(X_pos, X_neg, increasing=True,
		target_fdr=target_fdr, min_frac_neg=min_frac_neg)

	assert_array_almost_equal(val, expected, 3)


###


def test_tfmodisco_seqlets(X_contrib):
	seqlets = tfmodisco_seqlets(X_contrib)
	pos_seqlets = seqlets[seqlets['attribution'] >= 0]
	neg_seqlets = seqlets[seqlets['attribution'] < 0]

	assert isinstance(pos_seqlets, pandas.DataFrame)
	assert isinstance(neg_seqlets, pandas.DataFrame)
	assert pos_seqlets.shape == (59, 4)
	assert neg_seqlets.shape == (105, 4)

	assert_array_almost_equal(pos_seqlets.values[:15], [
		[0.0000e+00, 1.0430e+03, 1.0840e+03, 3.1515e-01],
        [0.0000e+00, 1.5110e+03, 1.5520e+03, 9.2622e-02],
        [0.0000e+00, 5.6900e+02, 6.1000e+02, 4.4792e-02],
        [0.0000e+00, 1.2220e+03, 1.2630e+03, 3.6950e-02],
        [0.0000e+00, 1.4450e+03, 1.4860e+03, 3.2022e-02],
        [0.0000e+00, 7.6400e+02, 8.0500e+02, 1.6513e-02],
        [0.0000e+00, 1.0160e+03, 1.0570e+03, 1.0608e-02],
        [0.0000e+00, 7.0200e+02, 7.4300e+02, 8.4966e-03],
        [1.0000e+00, 1.0330e+03, 1.0740e+03, 3.3020e-01],
        [1.0000e+00, 8.9600e+02, 9.3700e+02, 1.4599e-01],
        [1.0000e+00, 1.5130e+03, 1.5540e+03, 5.5456e-02],
        [1.0000e+00, 1.5940e+03, 1.6350e+03, 5.1420e-02],
        [1.0000e+00, 7.1700e+02, 7.5800e+02, 1.6273e-02],
        [1.0000e+00, 9.2400e+02, 9.6500e+02, 1.6062e-02],
        [1.0000e+00, 1.5620e+03, 1.6030e+03, 8.2083e-03]], 4)

	assert_array_almost_equal(neg_seqlets.values[:15], [
		[ 0.0000e+00,  9.8100e+02,  1.0220e+03, -2.3192e-02],
        [ 0.0000e+00,  8.8300e+02,  9.2400e+02, -2.1703e-02],
        [ 0.0000e+00,  1.0670e+03,  1.1080e+03, -1.7715e-02],
        [ 0.0000e+00,  1.4900e+03,  1.5310e+03, -1.6888e-02],
        [ 0.0000e+00,  1.4030e+03,  1.4440e+03, -1.5845e-02],
        [ 0.0000e+00,  1.3470e+03,  1.3880e+03, -1.4895e-02],
        [ 0.0000e+00,  6.7100e+02,  7.1200e+02, -1.3776e-02],
        [ 0.0000e+00,  1.1580e+03,  1.1990e+03, -1.2956e-02],
        [ 0.0000e+00,  9.2300e+02,  9.6400e+02, -1.1575e-02],
        [ 0.0000e+00,  5.9300e+02,  6.3400e+02, -1.1291e-02],
        [ 0.0000e+00,  1.1970e+03,  1.2380e+03, -1.0760e-02],
        [ 0.0000e+00,  1.1290e+03,  1.1700e+03, -1.0381e-02],
        [ 0.0000e+00,  8.1400e+02,  8.5500e+02, -1.0249e-02],
        [ 0.0000e+00,  1.5600e+03,  1.6010e+03, -9.8816e-03],
        [ 0.0000e+00,  8.4100e+02,  8.8200e+02, -9.3998e-03]], 4)

	pos_counts = numpy.unique(pos_seqlets['example_idx'].values, 
		return_counts=True)[1]
	neg_counts = numpy.unique(neg_seqlets['example_idx'].values, 
		return_counts=True)[1]

	assert_array_almost_equal(pos_counts, [ 8,  8, 12,  9, 14,  8])
	assert_array_almost_equal(neg_counts, [19, 18, 15, 19, 15, 19])
	assert all(numpy.isin(numpy.diff(pos_seqlets['example_idx']), [0, 1]))


###


def test_tfmodisco_seqlets_determinism(X_contrib):
	torch.manual_seed(0)
	seqlets0 = tfmodisco_seqlets(X_contrib)

	torch.manual_seed(0)
	seqlets1 = tfmodisco_seqlets(X_contrib)

	assert_array_almost_equal(seqlets0.values, seqlets1.values, 4)


@pytest.mark.parametrize("value", [0.0, 0.5])
def test_recursive_seqlets_constant(value):
	"""A constant input used to raise ZeroDivisionError from a zero bin width.

	Every span of a given length has the same sum, so none can be significant.
	"""

	seqlets = recursive_seqlets(torch.full((2, 100), value))

	assert seqlets.shape == (0, 5)
	assert list(seqlets.columns) == ['example_idx', 'start', 'end',
		'attribution', 'p-value']


@pytest.mark.parametrize("min_seqlet_len", [4, 6, 8])
def test_recursive_seqlets_right_edge(min_seqlet_len):
	"""The last min_seqlet_len positions used to be excluded from every seqlet.
	"""

	X = numpy.random.RandomState(0).normal(0, 0.01, size=(1, 1000))
	for start in (0, 495, 990):
		X[0, start:start+10] += 1

	seqlets = recursive_seqlets(X, min_seqlet_len=min_seqlet_len)
	seqlets = seqlets.sort_values("start").reset_index(drop=True)

	assert seqlets.shape[0] == 3
	assert seqlets.loc[0, 'start'] == 0 and seqlets.loc[0, 'end'] >= 10
	assert seqlets.loc[1, 'start'] <= 495 and seqlets.loc[1, 'end'] >= 505
	assert seqlets.loc[2, 'start'] <= 990 and seqlets.loc[2, 'end'] == 1000

	for _, (idx, start, end, attr, _) in seqlets.iterrows():
		assert_array_almost_equal(attr, X[int(idx), int(start):int(end)].sum())


def _assert_no_overlap(seqlets):
	seqlets = seqlets.sort_values(['example_idx', 'start'])
	for _, group in seqlets.groupby('example_idx'):
		assert (group['start'].values[1:] >= group['end'].values[:-1]).all()


def test_recursive_seqlets_min_seqlet_len_one(X_contrib):
	"""With min_seqlet_len=1 every length-1 span used to get a p-value of 0.

	A single position is significant only if fewer than 1% of positions score
	at least as high.
	"""

	seqlets = recursive_seqlets(X_contrib, min_seqlet_len=1)
	lengths = seqlets['end'] - seqlets['start']

	assert (lengths == 1).sum() <= 0.01 * X_contrib.numel()
	assert (seqlets['p-value'] > 0).all()
	_assert_no_overlap(seqlets)


def test_recursive_seqlets_min_seqlet_len_one_right_edge():
	"""The last position used to be left unmasked after a seqlet ending there.

	With min_seqlet_len=1 it was then called a second time as a length-1
	seqlet overlapping the first.
	"""

	X = numpy.random.RandomState(0).normal(0, 0.01, size=(1, 1000))
	X[0, 995:1000] += numpy.linspace(1, 2, 5)

	seqlets = recursive_seqlets(X, min_seqlet_len=1)

	assert ((seqlets['start'] == 995) & (seqlets['end'] == 1000)).sum() == 1
	_assert_no_overlap(seqlets)


@pytest.fixture
def X_laplace():
	rng = numpy.random.RandomState(0)
	X = rng.laplace(size=(4, 500)) * 0.01
	X[rng.uniform(size=X.shape) < 0.002] *= 40
	X[:, 100:108] += 0.05
	X[:, 300:306] += 0.03
	return X


def test_recursive_seqlets_minimum_runs():
	"""A run of values equal to the minimum used to be called with p = 0.

	Rounding in the cumulative sum could give such a span a bin index of -1.
	Here [31, 35) of example 0 is four values of -0.5.
	"""

	X = numpy.clip(numpy.random.RandomState(89).normal(size=(2, 500)), -0.5,
		None)
	seqlets = recursive_seqlets(X)

	assert seqlets.shape[0] > 0
	assert (seqlets['attribution'] > 0).all()
	assert (seqlets['p-value'] > 0).all()


def test_recursive_seqlets_minimum_mass():
	"""Blocks on a background of exact zeros used to go uncalled.

	The table of p-values held P(S > i) + P(S = 0), and with 97% of positions
	at the minimum, P(S = 0) is 0.97^4 = 0.89 for every span of length 4.
	"""

	X = torch.zeros(1, 1000)
	for start in (0, 495, 990):
		X[0, start:start+10] = 1

	seqlets = recursive_seqlets(X)
	seqlets = seqlets.sort_values("start").reset_index(drop=True)

	assert seqlets.shape[0] == 3
	assert seqlets.loc[0, 'start'] == 0 and seqlets.loc[0, 'end'] >= 10
	assert seqlets.loc[1, 'start'] <= 495 and seqlets.loc[1, 'end'] >= 505
	assert seqlets.loc[2, 'start'] <= 990 and seqlets.loc[2, 'end'] == 1000


def test_recursive_seqlets_shorter_than_min_seqlet_len():
	"""A sequence shorter than min_seqlet_len used to be indexed out of bounds.
	"""

	X = numpy.random.RandomState(0).normal(size=(2, 3))
	seqlets = recursive_seqlets(X, min_seqlet_len=4)
	assert seqlets.shape == (0, 5)


@pytest.mark.parametrize("threshold", [1.0, 0.0, -1.0])
def test_recursive_seqlets_threshold_bounds(X_laplace, threshold):
	seqlets = recursive_seqlets(X_laplace, threshold=threshold)
	if threshold <= 0:
		assert seqlets.shape == (0, 5)
	else:
		assert seqlets.shape[0] > 0


def test_tfmodisco_seqlets_empty():
	assert_raises(ValueError, tfmodisco_seqlets, torch.zeros(0, 100))


###


@pytest.fixture
def X_background():
	return numpy.random.RandomState(1).laplace(size=(4, 500)) * 0.01


def _fdr(X, X_control, thresholds=None):
	X = numpy.asarray(X, dtype=numpy.float64)
	X_control = numpy.asarray(X_control, dtype=numpy.float64)
	tables = _recursive_seqlet_tables(X, 25, 1000)
	return _recursive_seqlets_fdr(X, X_control, tables, 4, 25, 1000, thresholds)


def test_recursive_seqlets_control_default_target():
	rng = numpy.random.RandomState(0)
	X = rng.laplace(size=(40, 1000)) * 0.01
	X[:, 500:510] += 0.2
	X_control = rng.laplace(size=(40, 1000)) * 0.01

	seqlets = recursive_seqlets(X, X_control=X_control)
	expected = recursive_seqlets(X, X_control=X_control, target_fdr=0.1)

	assert seqlets.shape[0] > 0
	assert_array_almost_equal(seqlets.values, expected.values)


def test_recursive_seqlets_control_constant(X_background):
	with pytest.warns(TangermemeWarning, match="no seqlets are called on X"):
		seqlets, fdr = recursive_seqlets(numpy.zeros((2, 100)), 
			X_control=X_background, return_fdr=True)

	assert seqlets.shape == (0, 5)
	assert (fdr['n_seqlets'] == 0).all()
	assert fdr['fdr'].isna().all()


def test_recursive_seqlets_control_chance_pass():
	"""A single seqlet with no control seqlets is not an estimate of zero.

	One planted span in noise is called alone at strict thresholds, where the
	control, by chance, has none. Adding one to the control count keeps that
	from passing a target of 0.1.
	"""

	rng = numpy.random.RandomState(0)
	X = rng.laplace(size=(20, 500)) * 0.01
	X[0, 200:206] += 0.05
	X_control = rng.laplace(size=(20, 500)) * 0.01

	with pytest.warns(TangermemeWarning, match="the lowest estimate is"):
		seqlets, fdr = recursive_seqlets(X, X_control=X_control, 
			return_fdr=True)

	single = fdr[(fdr['n_seqlets'] == 1) & (fdr['n_control_seqlets'] == 0)]
	assert len(single) > 0
	assert_array_almost_equal(single['fdr'], 1.0)
	assert seqlets.shape == (0, 5)


def test_recursive_seqlets_fdr_counts(X_laplace, X_background):
	"""Each row counts the seqlets recursive_seqlets calls at that threshold,
	and scales the control count plus one by the number of positions."""

	X_control = numpy.random.RandomState(2).laplace(size=(2, 500)) * 0.01
	fdr = _fdr(X_laplace, X_control, thresholds=[1e-2, 1e-3, 1e-4])

	assert list(fdr.columns) == ['threshold', 'n_seqlets', 'n_control_seqlets',
		'fdr']
	assert_array_almost_equal(fdr['threshold'], [1e-2, 1e-3, 1e-4])

	assert (fdr['n_seqlets'] > 0).any() and (fdr['n_seqlets'] == 0).any()
	for _, (threshold, n_seqlets, n_control, value) in fdr.iterrows():
		assert n_seqlets == recursive_seqlets(X_laplace, 
			threshold=threshold).shape[0]

		if n_seqlets > 0:
			assert_array_almost_equal(value, 2 * (n_control + 1) / n_seqlets)
		else:
			assert numpy.isnan(value)


def test_recursive_seqlets_fdr_strict_thresholds(X_laplace, X_background):
	"""Rows past the first threshold that calls nothing on either input are
	filled in without calling seqlets, and match calling at each alone."""

	fdr = _fdr(X_laplace, X_background)

	empty = ((fdr['n_seqlets'] == 0) & (fdr['n_control_seqlets'] == 0)).values
	assert empty.any() and not empty.all()
	first = empty.argmax()
	assert empty[first:].all()

	for _, (threshold, n_seqlets, _, _) in fdr.iloc[::8].iterrows():
		assert n_seqlets == recursive_seqlets(X_laplace, 
			threshold=threshold).shape[0]

	for threshold in fdr['threshold'].values[first - 2:first + 3]:
		alone = _fdr(X_laplace, X_background, thresholds=[threshold])
		row = fdr[fdr['threshold'] == threshold]
		assert alone['n_seqlets'].iloc[0] == row['n_seqlets'].iloc[0]
		assert alone['n_control_seqlets'].iloc[0] == \
			row['n_control_seqlets'].iloc[0]


def test_recursive_seqlets_fdr_unsorted_thresholds(X_laplace, X_background):
	fdr = _fdr(X_laplace, X_background, thresholds=[1e-4, 1e-2, 1e-3])
	assert_array_almost_equal(fdr['threshold'], [1e-2, 1e-3, 1e-4])


def test_recursive_seqlets_fdr_no_signal():
	"""Input and control from the same process give an estimate near 1."""

	rng = numpy.random.RandomState(0)
	X = rng.laplace(size=(40, 1000))
	X_control = rng.laplace(size=(40, 1000))

	fdr = _fdr(X, X_control, thresholds=[1e-2, 3e-3])

	assert (fdr['n_seqlets'] > 50).all()
	assert ((fdr['fdr'] > 0.7) & (fdr['fdr'] < 1.4)).all()


def test_recursive_seqlets_fdr_signal():
	"""Strong planted spans are called, and few control seqlets with them."""

	rng = numpy.random.RandomState(0)
	X = rng.laplace(size=(40, 1000)) * 0.01
	X[:, 500:510] += 0.2
	X_control = rng.laplace(size=(40, 1000)) * 0.01

	seqlets, fdr = recursive_seqlets(X, X_control=X_control, return_fdr=True)

	assert seqlets.shape[0] >= 40
	for i in range(40):
		s = seqlets[seqlets['example_idx'] == i]
		assert ((s['start'] < 510) & (s['end'] > 500)).any()

	assert fdr['fdr'].min() < 0.1


def test_recursive_seqlets_fdr_control_size(X_laplace, X_background):
	"""The control count is scaled by the number of positions in the control."""

	thresholds = [1e-2, 1e-3]
	X_control = numpy.concatenate([X_background, X_background])
	fdr = _fdr(X_laplace, X_background, thresholds=thresholds)
	fdr2 = _fdr(X_laplace, X_control, thresholds=thresholds)

	assert_array_almost_equal(fdr2['n_control_seqlets'], 
		2 * fdr['n_control_seqlets'])

	called = fdr2[fdr2['n_seqlets'] > 0]
	assert len(called) > 0
	assert_array_almost_equal(called['fdr'], (called['n_control_seqlets'] + 1)
		* X_laplace.size / X_control.size / called['n_seqlets'])


###
# A numpy reference for recursive_seqlets, compared on small random inputs.
# It follows the docstring, and the code where the docstring is silent: the
# order of spans of one length and the arithmetic of the bins.


_columns = ['example_idx', 'start', 'end', 'attribution', 'p-value']


def _reference_bins(X, xmin, bin_width, n_bins):
	"""The bin of each value, with values outside the bins in the end bins.

	The difference is taken in the dtype of X and divided in float64, which is
	how recursive_seqlets bins float32 attributions.
	"""

	diff = numpy.asarray(X - xmin).astype(numpy.float64)
	bins = numpy.floor(diff / bin_width)
	return numpy.clip(bins, 0, n_bins - 1).astype(numpy.int64)


def _reference_pmfs(bins, n_bins, max_seqlet_len):
	"""P(S = s) for S the sum of k independent draws from the histogram of
	`bins`, as pmfs[k][s] for k from 1 to max_seqlet_len: the histogram
	convolved with itself k times."""

	f = numpy.bincount(bins.ravel(), minlength=n_bins) / bins.size
	pmfs = [numpy.ones(1)]
	for _ in range(max_seqlet_len):
		pmfs.append(numpy.convolve(pmfs[-1], f))

	return pmfs


def _reference_null(bins, n_bins, max_seqlet_len):
	"""P(S >= s) as rcdfs[k][s] for k from 1 to max_seqlet_len, the
	cumulative sum of P(S = s) taken from the top."""

	pmfs = _reference_pmfs(bins, n_bins, max_seqlet_len)
	return [None] + [numpy.cumsum(pmf[::-1])[::-1] for pmf in pmfs[1:]]


def _reference_spans(X, min_seqlet_len, max_seqlet_len, n_bins, X_null=None):
	"""The p-values of every span of X under the null built from X_null.

	X_null defaults to X, and X is binned in the dtype of X_null. For each
	length L, `own[L][i, a]` is the p-value of the span [a, a + L) of example
	i on its own, `key[L][i, a]` the largest p-value over its prefixes of at
	least min_seqlet_len, `tri[L][i, a]` the largest over all of its
	sub-spans of at least min_seqlet_len, and `order[L][i]` the starts sorted
	by key, lowest start first among equal keys. `entries` maps each
	(length, binned sum) that occurs to its p-value, and `pmfs` holds the
	distributions of the sums.
	"""

	X_null = X if X_null is None else X_null
	X = X.astype(X_null.dtype)
	xmin = X_null.min()
	bin_width = float(X_null.max() - xmin) / (n_bins - 1)

	null_bins = _reference_bins(X_null, xmin, bin_width, n_bins)
	pmfs = _reference_pmfs(null_bins, n_bins, max_seqlet_len)
	rcdfs = _reference_null(null_bins, n_bins, max_seqlet_len)
	bins = _reference_bins(X, xmin, bin_width, n_bins)
	csum = numpy.zeros((X.shape[0], X.shape[1] + 1), dtype=numpy.int64)
	csum[:, 1:] = numpy.cumsum(bins, axis=1)

	l = X.shape[1]
	own, key, tri, order, entries = {}, {}, {}, {}, {}
	for L in range(min_seqlet_len, min(max_seqlet_len, l) + 1):
		sums = csum[:, L:] - csum[:, :-L]
		own[L] = rcdfs[L][sums]
		for s in numpy.unique(sums):
			entries[L, int(s)] = rcdfs[L][s]

		n_starts = l - L + 1
		key[L] = numpy.max([own[m][:, :n_starts] for m in
			range(min_seqlet_len, L + 1)], axis=0)
		tri[L] = numpy.max([key[L - k][:, k:k + n_starts] for k in
			range(L - min_seqlet_len + 1)], axis=0)
		order[L] = [numpy.lexsort((numpy.arange(n_starts), key[L][i]))
			for i in range(X.shape[0])]

	return {'X': X, 'own': own, 'key': key, 'tri': tri, 'order': order,
		'entries': entries, 'pmfs': pmfs}


def _reference_decode(spans, threshold, additional_flanks=0):
	"""The seqlets called at a threshold, in the order they are called.

	Examples in order, longest spans first, and within a length the spans in
	order of their key, until a key reaches the threshold. A span is called
	if none of its positions is in a seqlet already called and every one of
	its sub-spans of at least min_seqlet_len is below the threshold. The
	p-value of a row is that of the unflanked span on its own.
	"""

	X = spans['X']
	n, l = X.shape

	rows = []
	for i in range(n):
		called = numpy.zeros(l, dtype=bool)
		for L in sorted(spans['own'], reverse=True):
			for a in spans['order'][L][i]:
				if spans['key'][L][i, a] >= threshold:
					break
				if called[a:a+L].any() or spans['tri'][L][i, a] >= threshold:
					continue

				called[a:a+L] = True
				start = max(a - additional_flanks, 0)
				end = min(a + L + additional_flanks, l)
				attr = math.fsum(X[i, start:end].astype(numpy.float64))
				rows.append((i, start, end, attr, spans['own'][L][i, a]))

	return rows


def _assert_unambiguous(spans, thresholds, rtol=1e-9):
	"""Check that the reference and the library must make the same calls.

	The two sum the null in a different order, so p-values may differ in their
	last bits. The calls agree when no p-value below the threshold is within
	rtol of the threshold or of another p-value, except where both compute
	the same number: spans of one length with the same binned sum, or two
	binned sums of one length with only sums of probability zero between them.
	"""

	limit = max(thresholds) * (1 + 1e-6)
	values = sorted((v, L, s) for (L, s), v in spans['entries'].items()
		if v < limit)

	for (v1, L1, s1), (v2, L2, s2) in zip(values[:-1], values[1:]):
		assert v2 - v1 > rtol * v2 or (v1 == v2 and L1 == L2 and
			(spans['pmfs'][L1][min(s1, s2):max(s1, s2)] == 0).all())

	values = numpy.array([v for v, _, _ in values])
	for threshold in thresholds:
		assert (numpy.abs(values - threshold) > rtol * threshold).all()


def _by_position(seqlets):
	return seqlets.sort_values(['example_idx', 'start']).reset_index(
		drop=True)


def _assert_seqlets_equal(observed, rows):
	"""Compare a frame from recursive_seqlets with rows from the reference.

	Coordinates must match exactly, and p-values and attributions within 1e-9,
	as both may differ by rounding. The frames are compared by position, and
	then as ranked: the pandas sort by p-value is not stable, so ranking the
	reference's rows in call order checks the order of ties too.
	"""

	expected = pandas.DataFrame(rows, columns=_columns)
	expected = expected.sort_values("p-value").reset_index(drop=True)

	assert list(observed.columns) == _columns
	assert observed.shape == expected.shape
	if len(expected) == 0:
		return

	for o, e in ((_by_position(observed), _by_position(expected)),
		(observed, expected)):
		assert_array_equal(o[_columns[:3]].values.astype(numpy.int64),
			e[_columns[:3]].values.astype(numpy.int64))
		assert_allclose(o['p-value'], e['p-value'], rtol=1e-9, atol=0)
		assert_allclose(o['attribution'], e['attribution'], rtol=0, atol=1e-9)


def _random_attributions(seed, n, l, dtype=numpy.float64):
	"""Laplace noise with a block of six raised positions in each example, and
	a block of four after it."""

	rng = numpy.random.RandomState(seed)
	X = rng.laplace(size=(n, l))
	X[:, l // 3:l // 3 + 6] += rng.uniform(0, 3, size=(n, 1))
	X[:, 2 * l // 3:2 * l // 3 + 4] += rng.uniform(1, 3, size=(n, 1))
	return X.astype(dtype)


def _integer_attributions():
	"""Integers from 0 to 3, with a run of 3s in three of the six examples."""

	X = numpy.random.RandomState(0).randint(0, 4, size=(6, 30))
	X[[0, 2, 3], 10:16] = 3
	return X.astype(numpy.float64)


_reference_shapes = [(1, 40), (3, 25), (5, 12), (2, 60), (4, 33), (6, 20),
	(2, 90), (7, 19)]


@pytest.mark.parametrize("min_seqlet_len, max_seqlet_len, n_bins, "
	"additional_flanks, threshold, dtype", [
	(3, 8, 16, 0, 0.043, numpy.float64),
	(1, 8, 16, 0, 0.043, numpy.float64),
	(2, 8, 16, 0, 0.043, numpy.float64),
	(4, 4, 16, 0, 0.043, numpy.float64),
	(3, 15, 16, 0, 0.043, numpy.float64),
	(3, 8, 2, 0, 0.043, numpy.float64),
	(3, 8, 7, 0, 0.043, numpy.float64),
	(3, 8, 100, 0, 0.043, numpy.float64),
	(3, 8, 16, 2, 0.043, numpy.float64),
	(3, 8, 16, 7, 0.043, numpy.float64),
	(3, 8, 16, 40, 0.043, numpy.float64),
	(3, 8, 16, 0, 0.0071, numpy.float64),
	(3, 8, 16, 0, 0.31, numpy.float64),
	(3, 8, 16, 0, 0.043, numpy.float32),
	(1, 15, 7, 3, 0.031, numpy.float32),
])
def test_recursive_seqlets_reference(min_seqlet_len, max_seqlet_len, n_bins,
	additional_flanks, threshold, dtype):
	"""recursive_seqlets matches the reference on eight random inputs.

	max_seqlet_len=15 is longer than the examples of length 12, flanks of 7
	are clipped at an end of the sequence for some seqlets and not others,
	and flanks of 40 for all of them.
	"""

	n_called = 0
	for seed, (n, l) in enumerate(_reference_shapes):
		X = _random_attributions(seed, n, l, dtype)
		spans = _reference_spans(X, min_seqlet_len, max_seqlet_len, n_bins)
		_assert_unambiguous(spans, [threshold])
		rows = _reference_decode(spans, threshold, additional_flanks)

		seqlets = recursive_seqlets(X, threshold=threshold,
			min_seqlet_len=min_seqlet_len, max_seqlet_len=max_seqlet_len,
			additional_flanks=additional_flanks, n_bins=n_bins)
		_assert_seqlets_equal(seqlets, rows)
		n_called += len(rows)

	assert n_called > 0


@pytest.mark.parametrize("n, l, min_seqlet_len, max_seqlet_len, n_bins", [
	(3, 4, 4, 4, 16),
	(3, 4, 4, 25, 16),
	(2, 10, 3, 25, 16),
	(1, 80, 3, 8, 16),
	(60, 6, 2, 5, 16),
	(4, 30, 5, 5, 2),
	(4, 30, 1, 1, 16),
])
def test_recursive_seqlets_reference_sizes(n, l, min_seqlet_len,
	max_seqlet_len, n_bins):
	"""recursive_seqlets matches the reference at the edges of the sizes:
	examples as long as min_seqlet_len, with max_seqlet_len equal or longer;
	max_seqlet_len longer than the examples; one example; many short
	examples; two bins; and min_seqlet_len equal to max_seqlet_len."""

	X = _random_attributions(0, n, l)
	spans = _reference_spans(X, min_seqlet_len, max_seqlet_len, n_bins)

	n_called = 0
	for threshold in (0.31, 0.043, 0.0071):
		_assert_unambiguous(spans, [threshold])
		rows = _reference_decode(spans, threshold)

		seqlets = recursive_seqlets(X, threshold=threshold,
			min_seqlet_len=min_seqlet_len, max_seqlet_len=max_seqlet_len,
			n_bins=n_bins)
		_assert_seqlets_equal(seqlets, rows)
		n_called += len(rows)

	assert n_called > 0


@pytest.mark.parametrize("dtype", [numpy.float64, numpy.float32])
def test_recursive_seqlets_reference_ties(dtype):
	"""recursive_seqlets matches the reference, including the order of rows,
	when many seqlets have equal p-values.

	Integers from 0 to 3 with four bins put each value in its own bin, so
	spans of one length with the same sum tie within and across examples;
	three examples share a run of 3s. A repeated example gives each seqlet
	once per copy.
	"""

	X_int = _integer_attributions()
	X_repeat = numpy.tile(_random_attributions(1, 1, 30), (4, 1))

	for X, n_bins, min_len in ((X_int, 4, 4), (X_repeat, 16, 2)):
		X = X.astype(dtype)

		for min_seqlet_len, max_seqlet_len, threshold in ((2, 6, 0.21),
			(3, 3, 0.21), (min_len, 6, 0.043)):
			spans = _reference_spans(X, min_seqlet_len, max_seqlet_len, n_bins)
			_assert_unambiguous(spans, [threshold])
			rows = _reference_decode(spans, threshold)

			seqlets = recursive_seqlets(X, threshold=threshold,
				min_seqlet_len=min_seqlet_len, max_seqlet_len=max_seqlet_len,
				n_bins=n_bins)
			assert seqlets['p-value'].duplicated().sum() >= 2
			_assert_seqlets_equal(seqlets, rows)


def test__recursive_seqlet_decode_order():
	"""_recursive_seqlet_decode returns seqlets in the order they are called,
	before any ranking: the reference's order, row for row, on the random
	inputs and on integer attributions with many equal p-values."""

	inputs = [(_random_attributions(seed, n, l), 3, 8, 16, 2, 0.043)
		for seed, (n, l) in enumerate(_reference_shapes)]
	inputs.append((_integer_attributions(), 2, 6, 4, 0, 0.21))

	for X, min_seqlet_len, max_seqlet_len, n_bins, flanks, threshold in inputs:
		spans = _reference_spans(X, min_seqlet_len, max_seqlet_len, n_bins)
		_assert_unambiguous(spans, [threshold])
		rows = _reference_decode(spans, threshold, additional_flanks=flanks)

		xmin, bin_width, rcdfs = _recursive_seqlet_tables(X, max_seqlet_len,
			n_bins)
		observed = _recursive_seqlet_decode(X, rcdfs, xmin, bin_width,
			threshold, min_seqlet_len, max_seqlet_len, flanks, n_bins)

		assert len(observed) == len(rows) > 0
		for o, e in zip(observed, rows):
			assert tuple(o[:3]) == tuple(e[:3])
			assert_allclose(o[3], e[3], rtol=0, atol=1e-9)
			assert_allclose(o[4], e[4], rtol=1e-9, atol=0)


def test_recursive_seqlets_reference_default_settings(X_laplace):
	"""recursive_seqlets matches the reference at the default n_bins,
	min_seqlet_len and max_seqlet_len."""

	for dtype in (numpy.float64, numpy.float32):
		X = X_laplace.astype(dtype)
		spans = _reference_spans(X, 4, 25, 1000)
		_assert_unambiguous(spans, [0.01])

		rows = _reference_decode(spans, 0.01)
		assert len(rows) > 5
		_assert_seqlets_equal(recursive_seqlets(X), rows)


def test_recursive_seqlets_reference_float32_bins(X_laplace):
	"""recursive_seqlets matches the reference on float32 values at bin edges.

	A tenth of the values are float32 numbers next to bin edges, each of which
	would fall in another bin if the difference from the smallest value were
	taken in float64 rather than float32, changing every p-value.
	"""

	X = X_laplace.astype(numpy.float32)
	xmin, xmax = X.min(), X.max()
	bin_width = float(xmax - xmin) / 999

	edges = (float(xmin) + numpy.arange(1, 999) * bin_width).astype(
		numpy.float32)
	near = numpy.concatenate([numpy.nextafter(edges, numpy.float32(-1)),
		edges, numpy.nextafter(edges, numpy.float32(1))])
	moves = _reference_bins(near, xmin, bin_width, 1000) != _reference_bins(
		near.astype(numpy.float64), float(xmin), bin_width, 1000)
	near = near[moves & (near > xmin) & (near < xmax)]
	assert len(near) >= X.size // 10

	rng = numpy.random.RandomState(0)
	inside = numpy.where((X.ravel() > xmin) & (X.ravel() < xmax))[0]
	replaced = rng.choice(inside, size=X.size // 10, replace=False)
	X.ravel()[replaced] = rng.choice(near, size=X.size // 10, replace=False)
	assert X.min() == xmin and X.max() == xmax

	spans = _reference_spans(X, 4, 25, 1000)
	_assert_unambiguous(spans, [0.01])
	rows = _reference_decode(spans, 0.01)
	assert len(rows) > 5
	_assert_seqlets_equal(recursive_seqlets(X), rows)


@pytest.mark.parametrize("min_seqlet_len, max_seqlet_len, threshold", [
	(4, 25, 0.01),
	(1, 25, 0.01),
	(3, 25, 0.01),
	(5, 25, 0.01),
	(7, 25, 0.01),
	(9, 25, 0.01),
	(11, 25, 0.01),
	(4, 10, 0.01),
	(4, 8, 0.01),
	(4, 6, 0.01),
	(4, 4, 0.01),
	(4, 25, 0.1),
	(4, 25, 0.001),
])
def test_recursive_seqlets_reference_attributions(X_contrib, min_seqlet_len,
	max_seqlet_len, threshold):
	"""recursive_seqlets matches the reference on real attributions, a float32
	tensor of six examples of length 2114, at the default n_bins, with
	min_seqlet_len, max_seqlet_len or threshold changed from the defaults."""

	spans = _reference_spans(X_contrib.numpy(), min_seqlet_len,
		max_seqlet_len, 1000)

	# At 0.1, two p-values of lengths 7 and 15 are 3e-10 apart, closer than
	# the guard allows, but the calls match.
	if threshold != 0.1:
		_assert_unambiguous(spans, [threshold])

	rows = _reference_decode(spans, threshold)
	assert len(rows) > 5
	_assert_seqlets_equal(recursive_seqlets(X_contrib, threshold=threshold,
		min_seqlet_len=min_seqlet_len, max_seqlet_len=max_seqlet_len), rows)


@pytest.mark.parametrize("dtype", [numpy.int64, numpy.int32])
def test_recursive_seqlets_integer_input(dtype):
	"""Integer attributions give the same output as the same values in
	float64, bit for bit: differences of integers are exact, and sums are
	taken in float64."""

	X = numpy.random.RandomState(0).randint(-20, 21, size=(4, 200))
	X[:, 50:58] += 30

	seqlets = recursive_seqlets(X.astype(dtype), threshold=0.01)
	assert len(seqlets) > 0
	assert_frame_equal(seqlets, recursive_seqlets(X.astype(numpy.float64),
		threshold=0.01), check_exact=True)


###
# Hand-worked inputs. Every value is 0 or 1 and there are two bins, so the
# bins are the values, and the null of a span of length k is Binomial(k, q),
# where q is the fraction of ones.


def test_recursive_seqlets_tie_lowest_start():
	"""Of two overlapping spans with the same p-value, the one that starts
	first is called.

	With q = 1/4, the fraction of ones, [3, 7) and [4, 8) have p-value
	q^4 = 1/256 and every other span of length 4 at least 13/256 > 0.01. The
	null is exact in floating point, so a threshold of exactly 1/256 calls
	nothing, as a p-value must be below it, and the next float above it calls
	[3, 7) again.
	"""

	X = numpy.zeros((1, 20))
	X[0, 3:8] = 1

	kwargs = {'min_seqlet_len': 4, 'max_seqlet_len': 4, 'n_bins': 2}
	seqlets = recursive_seqlets(X, threshold=0.01, **kwargs)

	assert_array_equal(seqlets[_columns[:3]].values, [[0, 3, 7]])
	assert_allclose(seqlets['attribution'], [4.0])
	assert_allclose(seqlets['p-value'], [1 / 256], rtol=1e-12)

	assert recursive_seqlets(X, threshold=1 / 256, **kwargs).shape == (0, 5)
	assert_frame_equal(recursive_seqlets(X, threshold=numpy.nextafter(1 / 256,
		1), **kwargs), seqlets, check_exact=True)


def test_recursive_seqlets_longest_first():
	"""A long span is called before the shorter spans inside it.

	With q = 0.3, the fraction of ones, a run of k ones has p-value
	q^k <= q^2 = 0.09, and every other span contains 00, 01 or 10, at
	1 - (1 - q)^2 = 0.51 or more. q^2 is one product in float64, so at a
	threshold of exactly 0.3 * 0.3 nothing is called, and at the next float
	above it [4, 10) is.
	"""

	X = numpy.zeros((1, 20))
	X[0, 4:10] = 1

	kwargs = {'min_seqlet_len': 2, 'max_seqlet_len': 6, 'n_bins': 2}
	seqlets = recursive_seqlets(X, threshold=0.1, **kwargs)

	assert_array_equal(seqlets[_columns[:3]].values, [[0, 4, 10]])
	assert_allclose(seqlets['attribution'], [6.0])
	assert_allclose(seqlets['p-value'], [0.3 ** 6], rtol=1e-12)

	assert recursive_seqlets(X, threshold=0.3 * 0.3, **kwargs).shape == (0, 5)
	assert_frame_equal(recursive_seqlets(X, threshold=numpy.nextafter(0.3 *
		0.3, 1), **kwargs), seqlets, check_exact=True)


def test_recursive_seqlets_triangle():
	"""A span is called only if every sub-span, not only every prefix, is
	below the threshold.

	With q = 0.2, the fraction of ones, [4, 9) = 11101 and [4, 8) = 1110 have
	every prefix below 0.05 (11 at 0.04, 111 at 0.008, 1110 at 0.0272, 11101 at
	0.00672), but contain [5, 8) = 110 at 0.104, so neither is called.
	[4, 7) = 111 is, and every other span contains 00, 01 or 10, at 0.36 or
	more.
	"""

	X = numpy.zeros((1, 20))
	X[0, [4, 5, 6, 8]] = 1

	seqlets = recursive_seqlets(X, threshold=0.05, min_seqlet_len=2,
		max_seqlet_len=5, n_bins=2)

	assert_array_equal(seqlets[_columns[:3]].values, [[0, 4, 7]])
	assert_allclose(seqlets['attribution'], [3.0])
	assert_allclose(seqlets['p-value'], [0.008], rtol=1e-12)


def test_recursive_seqlets_triangle_strict():
	"""Every sub-span must be strictly below the threshold.

	With q = 1/4, the fraction of ones, every p-value here is exact.
	[4, 9) = 11101 has prefixes at 1/64, 13/256 and 1/64, and contains 110 and
	101 at 10/64 = 0.15625. At a threshold of exactly 0.15625, [4, 7) = 111 is
	called instead, and at the next float above it, [4, 9) is.
	"""

	X = numpy.zeros((1, 20))
	X[0, [4, 5, 6, 8, 15]] = 1
	kwargs = {'min_seqlet_len': 3, 'max_seqlet_len': 5, 'n_bins': 2}

	seqlets = recursive_seqlets(X, threshold=0.15625, **kwargs)
	assert_array_equal(seqlets.values, [[0, 4, 7, 3.0, 1 / 64]])

	seqlets = recursive_seqlets(X, threshold=numpy.nextafter(0.15625, 1),
		**kwargs)
	assert_array_equal(seqlets.values, [[0, 4, 9, 4.0, 1 / 64]])


def test_recursive_seqlets_stricter_threshold():
	"""A stricter threshold can call more seqlets, when a long span called at
	a looser one covers two shorter ones.

	With q = 1/4, the fraction of ones, the largest p-value inside
	[3, 9) = 101101 is that of 0110, 67/256 = 0.26, so it is called at 0.3. At
	0.2, no span of length 5 or 6 is, since each contains 0110 or 1010, and
	[3, 7) = 1011, with sub-spans at most 10/64, and [8, 11) = 101 are. The fdr
	table counts the same.
	"""

	X = numpy.zeros((1, 20))
	X[0, [3, 5, 6, 8, 10]] = 1
	kwargs = {'min_seqlet_len': 3, 'max_seqlet_len': 6, 'n_bins': 2}

	loose = recursive_seqlets(X, threshold=0.3, **kwargs)
	strict = recursive_seqlets(X, threshold=0.2, **kwargs)
	assert_array_equal(loose[_columns[:3]].values, [[0, 3, 9]])
	assert_array_equal(_by_position(strict)[_columns[:3]].values,
		[[0, 3, 7], [0, 8, 11]])

	tables = _recursive_seqlet_tables(X, 6, 2)
	fdr = _recursive_seqlets_fdr(X, X, tables, 3, 6, 2, thresholds=[0.3, 0.2])
	assert_array_equal(fdr['n_seqlets'], [1, 2])
	assert_array_equal(fdr['n_control_seqlets'], [1, 2])


@pytest.mark.parametrize("n_ones", [3, 10, 18])
def test__recursive_seqlet_null_binomial(n_ones):
	"""P(S_k >= s) is scipy's binomial survival function at s - 1, for every
	k up to 25, and zero past the largest sum, k."""

	values = numpy.zeros(20)
	values[:n_ones] = 1
	rcdfs = _recursive_seqlet_null(values, 0.0, 1.0, 25, 2)

	for k in range(1, 26):
		s = numpy.arange(k + 1)
		assert_allclose(rcdfs[k, :k+1], scipy.stats.binom.sf(s - 1, k,
			n_ones / 20), rtol=1e-10)
		assert (rcdfs[k, k+1:] == 0).all()


@pytest.mark.parametrize("n_bins", [3, 5])
def test__recursive_seqlet_null_enumeration(n_bins):
	"""P(S_k >= s) is the total probability of the tuples of k bins summing to
	at least s, enumerated here for every k up to 4, and zero past the largest
	sum."""

	values = numpy.random.RandomState(0).normal(size=37)
	xmin = values.min()
	bin_width = float(values.max() - xmin) / (n_bins - 1)
	bins = _reference_bins(values, xmin, bin_width, n_bins)
	f = numpy.bincount(bins, minlength=n_bins) / values.size

	rcdfs = _recursive_seqlet_null(values, xmin, bin_width, 4, n_bins)

	for k in range(1, 5):
		tail = numpy.zeros(k * (n_bins - 1) + 1)
		for tup in itertools.product(range(n_bins), repeat=k):
			tail[:sum(tup) + 1] += numpy.prod(f[list(tup)])

		assert_allclose(rcdfs[k, :len(tail)], tail, rtol=1e-12)
		assert (rcdfs[k, len(tail):] == 0).all()


def test__recursive_seqlet_null_end_bins():
	"""Values outside the bins are counted in the end bins.

	-5 and 0 fall in bin 0, 0.5 in bin 1, and 1 and 7 in bin 2, so the
	histogram is (2, 1, 2) / 5 and S_2 is distributed as (4, 4, 9, 4, 4) / 25.
	"""

	values = numpy.array([-5, 0, 0.5, 1, 7])
	rcdfs = _recursive_seqlet_null(values, 0.0, 0.5, 2, 3)

	assert_allclose(rcdfs[1, :4], numpy.array([5, 3, 2, 0]) / 5, rtol=1e-12)
	assert_allclose(rcdfs[2, :6], numpy.array([25, 21, 17, 8, 4, 0]) / 25,
		rtol=1e-12)


@pytest.mark.parametrize("dtype", [numpy.float64, numpy.float32])
def test__recursive_seqlet_tables(X_laplace, dtype):
	"""The bins start at the smallest value of X and span its range, and the
	null at the default settings matches the reference within 1e-10. Both
	sum non-negative terms, so rounding is relative to each entry, tails
	included. A constant X has no bins.
	"""

	X = X_laplace.astype(dtype)
	xmin, bin_width, rcdfs = _recursive_seqlet_tables(X, 25, 1000)

	expected_width = float(X.max() - X.min()) / 999
	assert xmin == X.min()
	assert bin_width == expected_width

	expected = _reference_null(_reference_bins(X, X.min(), expected_width,
		1000), 1000, 25)
	for k in range(1, 26):
		assert_allclose(rcdfs[k, :len(expected[k])], expected[k], rtol=1e-10,
			atol=0)
		assert (rcdfs[k, len(expected[k]):] == 0).all()

	assert _recursive_seqlet_tables(numpy.full((2, 5), 0.3, dtype=dtype), 25,
		1000) is None


###
# Invariants of the output.


@pytest.mark.parametrize("min_seqlet_len, max_seqlet_len", [(4, 25), (1, 10),
	(6, 6)])
def test_recursive_seqlets_invariants(X_laplace, min_seqlet_len,
	max_seqlet_len):
	"""Without flanks, seqlets do not overlap and their lengths are within
	the bounds. Every p-value is above zero and below the threshold, and
	each attribution is the sum of X over the seqlet, within 1e-9 of an
	exactly rounded sum."""

	seqlets = recursive_seqlets(X_laplace, threshold=0.05,
		min_seqlet_len=min_seqlet_len, max_seqlet_len=max_seqlet_len)
	assert len(seqlets) > 10

	lengths = seqlets['end'] - seqlets['start']
	assert ((lengths >= min_seqlet_len) & (lengths <= max_seqlet_len)).all()
	_assert_no_overlap(seqlets)

	assert ((seqlets['p-value'] > 0) & (seqlets['p-value'] < 0.05)).all()

	for idx, start, end, attr in seqlets[_columns[:4]].itertuples(index=False):
		assert_allclose(attr, math.fsum(X_laplace[idx, start:end]), rtol=0,
			atol=1e-9)


def test_recursive_seqlets_example_order(X_laplace):
	"""The null is built from every example but not from their order:
	permuting the examples permutes example_idx and changes nothing else,
	bit for bit."""

	perm = numpy.array([2, 0, 3, 1])
	seqlets = recursive_seqlets(X_laplace, threshold=0.05)
	permuted = recursive_seqlets(X_laplace[perm], threshold=0.05)
	permuted['example_idx'] = perm[permuted['example_idx'].values]

	assert len(seqlets) > 0
	assert_frame_equal(_by_position(permuted), _by_position(seqlets),
		check_exact=True)


def test_recursive_seqlets_repeated_examples(X_laplace):
	"""Repeating every example leaves the proportions in the histogram, and so
	every p-value, unchanged: each seqlet is called once in each copy."""

	seqlets = recursive_seqlets(X_laplace, threshold=0.05)
	doubled = recursive_seqlets(numpy.concatenate([X_laplace, X_laplace]),
		threshold=0.05)
	assert len(seqlets) > 0
	assert len(doubled) == 2 * len(seqlets)

	first = doubled[doubled['example_idx'] < 4]
	second = doubled[doubled['example_idx'] >= 4].copy()
	second['example_idx'] -= 4

	assert_frame_equal(_by_position(first), _by_position(seqlets),
		check_exact=True)
	assert_frame_equal(_by_position(second), _by_position(seqlets),
		check_exact=True)


@pytest.mark.parametrize("scale", [0.25, 4.0])
def test_recursive_seqlets_scale(X_laplace, scale):
	"""Scaling X by a power of two is exact in floating point and leaves every
	bin unchanged, so the same seqlets are called, in the same order, with
	the same p-values and scaled attributions, bit for bit."""

	seqlets = recursive_seqlets(X_laplace, threshold=0.05)
	scaled = recursive_seqlets(X_laplace * scale, threshold=0.05)

	expected = seqlets.copy()
	expected['attribution'] *= scale
	assert len(seqlets) > 0
	assert_frame_equal(scaled, expected, check_exact=True)


@pytest.mark.parametrize("arrays", ["X", "X_control", "both"])
@pytest.mark.parametrize("layout", ["fortran", "strided"])
def test_recursive_seqlets_layout(X_laplace, X_background, layout, arrays):
	"""A Fortran-ordered or strided X gives the same frame as a C-contiguous
	one, bit for bit, and such an X_control, alone or with such an X, the
	same seqlets and fdr table. numba compiles a kernel for each layout of
	its inputs."""

	def relayout(A):
		if layout == "fortran":
			return numpy.asfortranarray(A)
		return numpy.repeat(A, 2, axis=1)[:, ::2]

	if arrays == "X":
		assert_frame_equal(recursive_seqlets(relayout(X_laplace),
			threshold=0.05), recursive_seqlets(X_laplace, threshold=0.05),
			check_exact=True)
		return

	X = relayout(X_laplace) if arrays == "both" else X_laplace
	result = recursive_seqlets(X, X_control=relayout(X_background),
		target_fdr=0.2, return_fdr=True)
	expected = recursive_seqlets(X_laplace, X_control=X_background,
		target_fdr=0.2, return_fdr=True)

	assert len(expected.seqlets) > 0
	assert_frame_equal(result.seqlets, expected.seqlets, check_exact=True)
	assert_frame_equal(result.fdr, expected.fdr, check_exact=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_recursive_seqlets_torch_exact(X_laplace, X_background, dtype):
	"""A tensor gives the same output as the array it holds, bit for bit, with
	and without a control."""

	X = torch.from_numpy(X_laplace).to(dtype)
	X_control = torch.from_numpy(X_background).to(dtype)

	assert_frame_equal(recursive_seqlets(X, threshold=0.05),
		recursive_seqlets(X.numpy(), threshold=0.05), check_exact=True)

	result = recursive_seqlets(X, X_control=X_control, target_fdr=0.2,
		return_fdr=True)
	expected = recursive_seqlets(X.numpy(), X_control=X_control.numpy(),
		target_fdr=0.2, return_fdr=True)

	assert len(result.seqlets) > 0
	assert_frame_equal(result.seqlets, expected.seqlets, check_exact=True)
	assert_frame_equal(result.fdr, expected.fdr, check_exact=True)


def test_recursive_seqlets_inputs_unchanged(X_laplace, X_background):
	"""Neither X nor X_control is modified, as an array or a tensor."""

	X = X_laplace.copy()
	X_control = X_background.astype(numpy.float32)
	X_tensor = torch.from_numpy(X_laplace.copy())

	recursive_seqlets(X, X_control=X_control, target_fdr=0.2)
	recursive_seqlets(X_tensor, threshold=0.05, additional_flanks=3)

	assert_array_equal(X, X_laplace)
	assert_array_equal(X_control, X_background.astype(numpy.float32))
	assert_array_equal(X_tensor.numpy(), X_laplace)


def test_recursive_seqlets_output_types(X_laplace, X_background):
	"""The dtypes of the output for float32 and float64 input, and the
	RecursiveSeqletsResult given with return_fdr=True."""

	int64, float64 = numpy.dtype('int64'), numpy.dtype('float64')
	for dtype in (numpy.float64, numpy.float32):
		seqlets = recursive_seqlets(X_laplace.astype(dtype), threshold=0.05)
		assert len(seqlets) > 0
		assert list(seqlets.dtypes) == [int64, int64, int64, float64, float64]

	result = recursive_seqlets(X_laplace, X_control=X_background,
		target_fdr=0.2, return_fdr=True)
	seqlets, fdr = result

	assert isinstance(result, RecursiveSeqletsResult)
	assert isinstance(result, tuple)
	assert RecursiveSeqletsResult._fields == ('seqlets', 'fdr')
	assert result[0] is seqlets is result.seqlets
	assert result[1] is fdr is result.fdr

	assert isinstance(seqlets, pandas.DataFrame)
	assert len(seqlets) > 0
	assert list(fdr.columns) == ['threshold', 'n_seqlets', 'n_control_seqlets',
		'fdr']
	assert list(fdr.dtypes) == [float64, int64, int64, float64]

	seqlets_only = recursive_seqlets(X_laplace, X_control=X_background,
		target_fdr=0.2)
	assert isinstance(seqlets_only, pandas.DataFrame)
	assert_frame_equal(seqlets_only, seqlets, check_exact=True)


@pytest.mark.parametrize("kwargs, match", [
	({'X': [[0.1, 0.2]]}, "`X` must be either a torch.Tensor or numpy.ndarray"),
	({'X': numpy.zeros((0, 10))},
		r"requires X to be non-empty; got X with shape \(0, 10\)"),
	({'X': torch.zeros(0, 100)},
		r"requires X to be non-empty; got X with shape \(0, 100\)"),
	({'X': numpy.ones((2, 10)), 'X_control': numpy.zeros((3, 0))},
		r"requires X_control to be non-empty; got X_control with shape "
		r"\(3, 0\)"),
	({'X': numpy.ones((2, 10)), 'X_control': [[0.1]]},
		"`X_control` must be either a torch.Tensor or numpy.ndarray"),
	({'X': numpy.ones((2, 10)), 'return_fdr': True},
		"return_fdr=True requires X_control"),
	({'X': numpy.ones((2, 10)), 'threshold': numpy.nan},
		"threshold must be a p-value of at most 1; got nan"),
	({'X': numpy.ones((2, 10)), 'threshold': 1.0000001},
		"threshold must be a p-value of at most 1; got 1.0000001"),
	({'X': numpy.ones((2, 10)), 'threshold': 1.5},
		"threshold must be a p-value of at most 1; got 1.5"),
	({'X': numpy.ones((2, 10)), 'threshold': numpy.inf},
		"threshold must be a p-value of at most 1; got inf"),
])
def test_recursive_seqlets_errors(kwargs, match):
	with pytest.raises(ValueError, match=match):
		recursive_seqlets(**kwargs)


def test__as_attributions():
	"""A tensor becomes the array it holds, an array keeps its values, and a
	dtype, when given, is applied. Other inputs and empty ones raise with the
	name given."""

	X = numpy.random.RandomState(0).normal(size=(2, 5))

	for X_in in (X, torch.from_numpy(X)):
		out = _as_attributions(X_in, "X")
		assert isinstance(out, numpy.ndarray)
		assert out.dtype == numpy.float64
		assert_array_equal(out, X)

		out = _as_attributions(X_in, "X", numpy.float32)
		assert out.dtype == numpy.float32
		assert_array_equal(out, X.astype(numpy.float32))

	with pytest.raises(ValueError, match="`scores` must be either"):
		_as_attributions(X.tolist(), "scores")
	with pytest.raises(ValueError, match=r"got scores with shape \(4, 0\)"):
		_as_attributions(torch.zeros(4, 0), "scores")


###
# With a control.


def _reference_fdr(X, X_control, min_seqlet_len, max_seqlet_len, n_bins):
	"""The fdr table of the reference at the 117 default thresholds, with the
	control decoded against the null and bins of X at every threshold."""

	thresholds = 10.0 ** -(1 + numpy.arange(117) / 4)
	spans = _reference_spans(X, min_seqlet_len, max_seqlet_len, n_bins)
	spans_control = _reference_spans(X_control, min_seqlet_len,
		max_seqlet_len, n_bins, X_null=X)
	_assert_unambiguous(spans, thresholds)
	_assert_unambiguous(spans_control, thresholds)

	n = numpy.array([len(_reference_decode(spans, t)) for t in thresholds])
	n_control = numpy.array([len(_reference_decode(spans_control, t))
		for t in thresholds])

	fdr = numpy.full(117, numpy.nan)
	fdr[n > 0] = (n_control[n > 0] + 1) * (X.size / X_control.size) / n[n > 0]
	return thresholds, n, n_control, fdr, spans


def _control_inputs(name):
	rng = numpy.random.RandomState(7)

	if name == "shapes":
		X = _random_attributions(0, 3, 60)
		X_control = rng.laplace(size=(5, 37))
	elif name == "range":
		X = _random_attributions(1, 3, 60)
		X_control = rng.laplace(size=(4, 50)) * 3
		X_control[0, 10:20] = X.min() - 5
		X_control[1, 30:36] = X.max() + 5
	elif name == "stronger_control":
		X = rng.laplace(size=(3, 60))
		X_control = _random_attributions(2, 3, 60)
		X_control[:, 40:46] += 4
	elif name == "float32":
		X = _random_attributions(3, 3, 60, numpy.float32)
		X_control = rng.laplace(size=(4, 60))

	return X, X_control


@pytest.mark.parametrize("min_seqlet_len, max_seqlet_len", [(3, 6), (4, 4)])
@pytest.mark.parametrize("name", ["shapes", "range", "stronger_control",
	"float32"])
def test_recursive_seqlets_control_reference(name, min_seqlet_len,
	max_seqlet_len):
	"""The fdr table matches the reference at every threshold, and the
	seqlets are those the reference calls at the largest threshold whose
	estimate is at or below target_fdr.

	The control differs from X in shape ("shapes"), has runs below and above
	the range of X ("range"), has seqlets at stricter thresholds than X
	("stronger_control"), or is float64 with float32 X ("float32").
	target_fdr falls between two distinct estimates, so that some thresholds
	reach it and some do not.
	"""

	X, X_control = _control_inputs(name)
	thresholds, n, n_control, fdr, spans = _reference_fdr(X, X_control,
		min_seqlet_len, max_seqlet_len, 16)

	estimates = numpy.unique(fdr[~numpy.isnan(fdr)])
	gaps = numpy.where(numpy.diff(estimates) > 1e-6 * estimates[1:])[0]
	assert len(gaps) > 0
	g = gaps[len(gaps) // 2]
	target_fdr = (estimates[g] + estimates[g + 1]) / 2

	result = recursive_seqlets(X, X_control=X_control, target_fdr=target_fdr,
		return_fdr=True, min_seqlet_len=min_seqlet_len,
		max_seqlet_len=max_seqlet_len, n_bins=16)

	assert_allclose(result.fdr['threshold'], thresholds, rtol=1e-12)
	assert_array_equal(result.fdr['n_seqlets'], n)
	assert_array_equal(result.fdr['n_control_seqlets'], n_control)
	assert_allclose(result.fdr['fdr'], fdr, rtol=1e-12)

	if name == "stronger_control":
		assert ((n == 0) & (n_control > 0)).any()

	passing = fdr <= target_fdr
	assert passing.any() and not passing.all()
	_assert_seqlets_equal(result.seqlets, _reference_decode(spans,
		thresholds[passing].max()))

	threshold = result.fdr['threshold'][passing].max()
	assert_frame_equal(result.seqlets, recursive_seqlets(X,
		threshold=threshold, min_seqlet_len=min_seqlet_len,
		max_seqlet_len=max_seqlet_len, n_bins=16), check_exact=True)


def test_recursive_seqlets_control_short_X():
	"""With X shorter than min_seqlet_len, nothing is called on X at any
	threshold, so every estimate is NaN and the warning says so, but the
	control is decoded against the null of X until it calls nothing."""

	X = numpy.random.RandomState(0).normal(size=(2, 3))
	X_control = _random_attributions(1, 4, 40)

	with pytest.warns(TangermemeWarning, match="no seqlets are called on X "
		"at any threshold"):
		seqlets, fdr = recursive_seqlets(X, X_control=X_control,
			return_fdr=True, max_seqlet_len=8, n_bins=16)

	_, n, n_control, _, _ = _reference_fdr(X, X_control, 4, 8, 16)

	assert seqlets.shape == (0, 5)
	assert (n == 0).all()
	assert (fdr['n_seqlets'] == 0).all()
	assert fdr['fdr'].isna().all()
	assert n_control[0] > 0
	assert_array_equal(fdr['n_control_seqlets'], n_control)


def test_recursive_seqlets_control_nested():
	"""With min_seqlet_len equal to max_seqlet_len no sub-span needs checking,
	so the spans visited at a threshold are visited first, in the same order,
	at every looser one. The seqlets called at a stricter threshold are then
	a subset of those called at a looser one, and the counts in the fdr table
	do not increase down it."""

	rng = numpy.random.RandomState(0)
	X = rng.laplace(size=(8, 200)) * 0.01
	X[:, 100:110] += numpy.linspace(0, 0.1, 8)[:, None]
	X_control = rng.laplace(size=(8, 200)) * 0.01

	fdr = recursive_seqlets(X, X_control=X_control, min_seqlet_len=5,
		max_seqlet_len=5, target_fdr=0.5, return_fdr=True).fdr

	assert fdr['n_seqlets'].iloc[0] > fdr['n_seqlets'].iloc[40] > 0
	assert (numpy.diff(fdr['n_seqlets']) <= 0).all()
	assert (numpy.diff(fdr['n_control_seqlets']) <= 0).all()

	called = None
	for threshold in (0.1, 1e-2, 1e-3, 1e-5, 1e-8):
		seqlets = recursive_seqlets(X, threshold=threshold, min_seqlet_len=5,
			max_seqlet_len=5)
		stricter = set(map(tuple, seqlets[_columns[:3]].values))
		if called is not None:
			assert stricter <= called
		called = stricter


def test_recursive_seqlets_control_largest_passing():
	"""When the thresholds that reach target_fdr are not contiguous in the
	table, seqlets are called at the largest of them."""

	rng = numpy.random.RandomState(60)
	X = rng.laplace(size=(3, 60))
	X[:, 20:26] += 2.0
	X_control = rng.laplace(size=(4, 45))

	kwargs = {'min_seqlet_len': 3, 'max_seqlet_len': 6, 'n_bins': 16}
	result = recursive_seqlets(X, X_control=X_control, target_fdr=0.3,
		return_fdr=True, **kwargs)

	passing = numpy.where(result.fdr['fdr'] <= 0.3)[0]
	assert (numpy.diff(passing) > 1).any()

	thresholds = result.fdr['threshold'].values
	assert_frame_equal(result.seqlets, recursive_seqlets(X,
		threshold=thresholds[passing[0]], **kwargs), check_exact=True)

	for idx in passing[1:]:
		seqlets = recursive_seqlets(X, threshold=thresholds[idx], **kwargs)
		assert not seqlets.equals(result.seqlets)


def test_recursive_seqlets_control_target_inclusive(X_laplace, X_background):
	"""A threshold whose estimate equals target_fdr reaches it; with the next
	float below the target, it does not."""

	fdr = recursive_seqlets(X_laplace, X_control=X_background, target_fdr=0.2,
		return_fdr=True).fdr

	# The estimate at the largest threshold with an estimate below 0.5 that
	# no larger threshold reaches.
	estimates = fdr['fdr'].values
	idx = next(i for i, v in enumerate(estimates) if v < 0.5 and
		not (estimates[:i] <= v).any())
	target = estimates[idx]
	below = numpy.nextafter(target, 0)
	assert (estimates <= below).any()

	for t, expected in ((target, fdr['threshold'][idx]),
		(below, fdr['threshold'][estimates <= below].max())):
		seqlets = recursive_seqlets(X_laplace, X_control=X_background,
			target_fdr=t)
		assert_frame_equal(seqlets, recursive_seqlets(X_laplace,
			threshold=expected), check_exact=True)

	assert fdr['threshold'][estimates <= below].max() < fdr['threshold'][idx]


def test_recursive_seqlets_control_warning_estimate():
	"""When no threshold reaches target_fdr, the warning gives the lowest
	estimate in the table, with a threshold and number of seqlets of a row
	that has it."""

	rng = numpy.random.RandomState(0)
	X = rng.laplace(size=(8, 300))
	X_control = rng.laplace(size=(8, 300))

	with pytest.warns(TangermemeWarning) as record:
		seqlets, fdr = recursive_seqlets(X, X_control=X_control,
			target_fdr=0.01, return_fdr=True)

	assert seqlets.shape == (0, 5)
	messages = [str(w.message) for w in record
		if issubclass(w.category, TangermemeWarning)]
	assert len(messages) == 1
	message = messages[0]
	match = re.search(r"the lowest estimate is (\S+), at a threshold of (\S+) "
		r"with (\d+) seqlets", message)
	assert match is not None, message

	best = fdr['fdr'].min()
	assert match.group(1) == "{:.3f}".format(best)

	rows = fdr[fdr['fdr'] == best]
	assert any(match.group(2) == "{:.1e}".format(t) and
		match.group(3) == str(k) for t, k in zip(rows['threshold'],
		rows['n_seqlets']))


@pytest.mark.parametrize("threshold", [1e-6, numpy.nan, 2.0])
def test_recursive_seqlets_control_ignores_threshold(X_laplace,
	X_background, threshold):
	"""threshold is ignored with a control, and a value that would raise
	without one does not."""

	seqlets = recursive_seqlets(X_laplace, threshold=threshold,
		X_control=X_background, target_fdr=0.2)
	assert len(seqlets) > 0
	assert_frame_equal(seqlets, recursive_seqlets(X_laplace,
		X_control=X_background, target_fdr=0.2), check_exact=True)


def test_recursive_seqlets_control_flanks(X_laplace, X_background):
	"""The seqlets returned with a control, with or without flanks, are those
	called at the chosen threshold without one, and flanks do not change the
	fdr table."""

	result = recursive_seqlets(X_laplace, X_control=X_background,
		target_fdr=0.2, return_fdr=True)
	flanked = recursive_seqlets(X_laplace, X_control=X_background,
		target_fdr=0.2, return_fdr=True, additional_flanks=3)

	assert_frame_equal(flanked.fdr, result.fdr, check_exact=True)
	assert not flanked.seqlets.equals(result.seqlets)

	threshold = result.fdr.loc[result.fdr['fdr'] <= 0.2, 'threshold'].max()
	assert not numpy.isnan(threshold)
	assert len(result.seqlets) > 0
	assert_frame_equal(result.seqlets, recursive_seqlets(X_laplace,
		threshold=threshold), check_exact=True)
	assert_frame_equal(flanked.seqlets, recursive_seqlets(X_laplace,
		threshold=threshold, additional_flanks=3), check_exact=True)


def test_recursive_seqlets_control_dtype(X_laplace, X_background):
	"""A control gives the same output, bit for bit, as the control cast to
	the dtype of X first, and a tensor control the same as its array.

	Only the last case shows that the control is binned in the dtype of X:
	over 30% of its values sit just above a bin edge of X, where their
	float32 roundings fall in the bin below.
	"""

	X32 = X_laplace.astype(numpy.float32)
	C32 = X_background.astype(numpy.float32)

	xmin = X32.min()
	bin_width = float(X32.max() - xmin) / 999
	bins = _reference_bins(X_background, float(xmin), bin_width, 1000)
	edges = float(xmin) + bins * bin_width + 1e-9 * bin_width
	moved = (bins > 0) & (_reference_bins(edges.astype(numpy.float32), xmin,
		bin_width, 1000) < _reference_bins(edges, float(xmin), bin_width, 1000))
	C_edges = numpy.where(moved, edges, X_background)
	assert moved.mean() > 0.3

	for X, X_control, cast, target_fdr in (
		(X32, X_background, C32, 0.2),
		(X_laplace, C32, C32.astype(numpy.float64), 0.2),
		(X_laplace, torch.from_numpy(X_background), X_background, 0.2),
		(X32, C_edges, C_edges.astype(numpy.float32), 0.2)):

		result = recursive_seqlets(X, X_control=X_control,
			target_fdr=target_fdr, return_fdr=True)
		expected = recursive_seqlets(X, X_control=cast,
			target_fdr=target_fdr, return_fdr=True)

		assert len(result.seqlets) > 0
		assert_frame_equal(result.seqlets, expected.seqlets, check_exact=True)
		assert_frame_equal(result.fdr, expected.fdr, check_exact=True)


def test_recursive_seqlets_control_no_threshold_typed_empty_frame():
	# A control identical to X estimates a false discovery rate above 1 at
	# every threshold, so no seqlets are returned; the empty frame keeps the
	# dtypes of a full one rather than object columns.
	torch.manual_seed(0)
	X = torch.randn(8, 200).abs()

	with warnings.catch_warnings(record=True) as caught:
		warnings.simplefilter("always")
		seqlets, fdr = recursive_seqlets(X, X_control=X, target_fdr=0.1,
			return_fdr=True)

	assert any(issubclass(c.category, TangermemeWarning) for c in caught)
	assert len(seqlets) == 0
	assert list(seqlets.columns) == ['example_idx', 'start', 'end',
		'attribution', 'p-value']
	assert seqlets['example_idx'].dtype == numpy.int64
	assert seqlets['start'].dtype == numpy.int64
	assert seqlets['end'].dtype == numpy.int64
	assert seqlets['attribution'].dtype == numpy.float64
	assert seqlets['p-value'].dtype == numpy.float64
	assert len(fdr) > 0
