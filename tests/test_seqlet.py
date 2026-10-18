# test_seqlet.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
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

from tangermeme.utils import TangermemeWarning

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_allclose


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


def test_isotonic_thresholds_fdr(X_contrib):
	torch.manual_seed(0)
	X_neg = torch.randn(10000)
	X_pos = X_contrib[X_contrib > 0.02]

	val0 = _isotonic_thresholds(X_pos, X_neg, increasing=True, target_fdr=0.01, 
		min_frac_neg=0.95)
	val1 = _isotonic_thresholds(X_pos, X_neg, increasing=True, target_fdr=0.05, 
		min_frac_neg=0.95)
	val2 = _isotonic_thresholds(X_pos, X_neg, increasing=True, target_fdr=0.5, 
		min_frac_neg=0.95)
	val3 = _isotonic_thresholds(X_pos, X_neg, increasing=True, target_fdr=0.99, 
		min_frac_neg=0.95)

	assert_array_almost_equal([val0, val1, val2, val3], 
		[0.027, 0.027, 0.035, 0.035], 3)


def test_isotonic_thresholds_min_frac(X_contrib):
	torch.manual_seed(0)
	X_neg = torch.randn(10000)
	X_pos = X_contrib[X_contrib > 0.02]

	val0 = _isotonic_thresholds(X_pos, X_neg, increasing=True, target_fdr=0.2, 
		min_frac_neg=0.05)
	val1 = _isotonic_thresholds(X_pos, X_neg, increasing=True, target_fdr=0.2, 
		min_frac_neg=0.15)
	val2 = _isotonic_thresholds(X_pos, X_neg, increasing=True, target_fdr=0.2, 
		min_frac_neg=0.50)
	val3 = _isotonic_thresholds(X_pos, X_neg, increasing=True, target_fdr=0.2, 
		min_frac_neg=0.90)

	assert_array_almost_equal([val0, val1, val2, val3], 
		[0.035, 0.035, 0.027, 0.027], 3)


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


def test_recursive_seqlets(X_contrib):
	seqlets = recursive_seqlets(X_contrib)

	assert seqlets.shape == (23, 5)

	column_names = ['example_idx', 'start', 'end', 'attribution', 'p-value']
	for name1, name2 in zip(seqlets.columns, column_names):
		assert name1 == name2

	assert_array_almost_equal(seqlets.values[:20], [
		[0.0, 1051.0, 1063.0, 0.306, 1.864e-12],
		[1.0, 1051.0, 1063.0, 0.2828, 1.269e-11],
		[4.0, 1042.0, 1055.0, 0.2294, 2.97e-09],
		[4.0, 1060.0, 1071.0, 0.1999, 1.778e-08],
		[5.0, 1069.0, 1080.0, 0.1817, 8.533e-08],
		[1.0, 913.0, 923.0, 0.1585, 9.749e-07],
		[2.0, 968.0, 978.0, 0.1317, 7.38e-06],
		[2.0, 1028.0, 1037.0, 0.0955, 7.669e-05],
		[0.0, 1520.0, 1531.0, 0.0903, 0.0003],
		[5.0, 1034.0, 1042.0, 0.0568, 0.0026],
		[1.0, 1537.0, 1544.0, 0.0529, 0.0028],
		[2.0, 1247.0, 1253.0, 0.0481, 0.0035],
		[2.0, 929.0, 935.0, 0.0478, 0.0037],
		[3.0, 1268.0, 1274.0, 0.0452, 0.0047],
		[0.0, 578.0, 585.0, 0.0441, 0.0057],
		[0.0, 1247.0, 1254.0, 0.041, 0.0066],
		[1.0, 1607.0, 1613.0, 0.0369, 0.0067],
		[4.0, 635.0, 641.0, 0.0359, 0.007],
		[4.0, 1222.0, 1228.0, 0.0343, 0.0076],
		[5.0, 885.0, 891.0, 0.0332, 0.0079]],
	4)


def test_recursive_seqlets_additional_flanks(X_contrib):
	seqlets = recursive_seqlets(X_contrib).values
	seqlets2 = recursive_seqlets(X_contrib, additional_flanks=2).values
	assert seqlets2.shape == (23, 5)

	assert_array_almost_equal(seqlets[:, 0], seqlets2[:, 0])
	assert_array_almost_equal(seqlets[:, 1], seqlets2[:, 1]+2)
	assert_array_almost_equal(seqlets[:, 2], seqlets2[:, 2]-2)


	seqlets2 = recursive_seqlets(X_contrib, additional_flanks=1).values
	assert seqlets2.shape == (23, 5)

	assert_array_almost_equal(seqlets[:, 0], seqlets2[:, 0])
	assert_array_almost_equal(seqlets[:, 1], seqlets2[:, 1]+1)
	assert_array_almost_equal(seqlets[:, 2], seqlets2[:, 2]-1)


	seqlets3 = recursive_seqlets(X_contrib, additional_flanks=4).values
	assert seqlets3.shape == (23, 5)

	assert_array_almost_equal(seqlets[:, 0], seqlets3[:, 0])
	assert_array_almost_equal(seqlets[:, 1], seqlets3[:, 1]+4)
	assert_array_almost_equal(seqlets[:, 2], seqlets3[:, 2]-4)

	seqlets4 = recursive_seqlets(X_contrib, additional_flanks=7).values
	assert seqlets4.shape == (23, 5)


def test_recursive_seqlets_min_length(X_contrib):
	seqlets = recursive_seqlets(X_contrib).values
	seqlets2 = recursive_seqlets(X_contrib, min_seqlet_len=1).values
	seqlets3 = recursive_seqlets(X_contrib, min_seqlet_len=3).values
	seqlets4 = recursive_seqlets(X_contrib, min_seqlet_len=5).values
	seqlets5 = recursive_seqlets(X_contrib, min_seqlet_len=7).values
	seqlets6 = recursive_seqlets(X_contrib, min_seqlet_len=9).values
	seqlets7 = recursive_seqlets(X_contrib, min_seqlet_len=11).values

	assert (seqlets[:, 2] - seqlets[:, 1]).min() == 4
	assert (seqlets2[:, 2] - seqlets2[:, 1]).min() == 1
	assert (seqlets3[:, 2] - seqlets3[:, 1]).min() == 3
	assert (seqlets4[:, 2] - seqlets4[:, 1]).min() == 6
	assert (seqlets5[:, 2] - seqlets5[:, 1]).min() == 7
	assert (seqlets6[:, 2] - seqlets6[:, 1]).min() == 9
	assert (seqlets7[:, 2] - seqlets7[:, 1]).min() == 11


def test_recursive_seqlets_max_length(X_contrib):
	seqlets = recursive_seqlets(X_contrib).values
	seqlets2 = recursive_seqlets(X_contrib, max_seqlet_len=10).values
	seqlets3 = recursive_seqlets(X_contrib, max_seqlet_len=8).values
	seqlets4 = recursive_seqlets(X_contrib, max_seqlet_len=6).values
	seqlets5 = recursive_seqlets(X_contrib, max_seqlet_len=4).values

	assert (seqlets[:, 2] - seqlets[:, 1]).max() == 13
	assert (seqlets2[:, 2] - seqlets2[:, 1]).max() == 10
	assert (seqlets3[:, 2] - seqlets3[:, 1]).max() == 8
	assert (seqlets4[:, 2] - seqlets4[:, 1]).max() == 6
	assert (seqlets5[:, 2] - seqlets5[:, 1]).max() == 4


def test_recursive_seqlets_p_values(X_contrib):
	seqlets = recursive_seqlets(X_contrib).values
	seqlets2 = recursive_seqlets(X_contrib, threshold=0.1).values
	seqlets3 = recursive_seqlets(X_contrib, threshold=0.001).values

	assert seqlets.shape == (23, 5)
	assert seqlets2.shape == (228, 5)
	assert seqlets3.shape == (8, 5)


###


def test_recursive_seqlets_numpy_input(X_contrib):
	seqlets_torch = recursive_seqlets(X_contrib)
	seqlets_numpy = recursive_seqlets(X_contrib.numpy())

	assert_array_almost_equal(seqlets_torch.values, seqlets_numpy.values, 4)


def test_recursive_seqlets_p_value_sorted(X_contrib):
	seqlets = recursive_seqlets(X_contrib)
	assert seqlets['p-value'].is_monotonic_increasing


def test_recursive_seqlets_schema_invariants(X_contrib):
	seqlets = recursive_seqlets(X_contrib)
	L = X_contrib.shape[-1]

	assert (seqlets['start'] < seqlets['end']).all()
	assert (seqlets['start'] >= 0).all()
	assert (seqlets['end'] <= L).all()
	assert (seqlets['example_idx'] >= 0).all()
	assert (seqlets['example_idx'] < X_contrib.shape[0]).all()


def test_tfmodisco_seqlets_determinism(X_contrib):
	torch.manual_seed(0)
	seqlets0 = tfmodisco_seqlets(X_contrib)

	torch.manual_seed(0)
	seqlets1 = tfmodisco_seqlets(X_contrib)

	assert_array_almost_equal(seqlets0.values, seqlets1.values, 4)


def test_recursive_seqlets_empty():
	assert_raises(ValueError, recursive_seqlets, torch.zeros(0, 100))


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

	A block at [990, 1000) was called as [988, 996) at min_seqlet_len=4 and
	not called at all at min_seqlet_len=8, while identical blocks at the left
	edge and in the middle were called in full.
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


def test_recursive_seqlets_additional_flanks_edges():
	"""Flanks past the right edge used to be clipped to length - 1."""

	X = numpy.random.RandomState(0).normal(0, 0.01, size=(1, 1000))
	X[0, 5:15] += 1
	X[0, 980:990] += 1

	seqlets = recursive_seqlets(X, additional_flanks=15)
	seqlets = seqlets.sort_values("start").reset_index(drop=True)

	assert seqlets.shape[0] == 2
	assert seqlets.loc[0, 'start'] == 0
	assert seqlets.loc[1, 'end'] == 1000

	for _, (idx, start, end, attr, _) in seqlets.iterrows():
		assert_array_almost_equal(attr, X[int(idx), int(start):int(end)].sum())


def _assert_no_overlap(seqlets):
	seqlets = seqlets.sort_values(['example_idx', 'start'])
	for _, group in seqlets.groupby('example_idx'):
		assert (group['start'].values[1:] >= group['end'].values[:-1]).all()


def test_recursive_seqlets_min_seqlet_len_one(X_contrib):
	"""With min_seqlet_len=1 every length-1 span used to get a p-value of 0.

	The length-1 null was never filled in, so more than 12,500 of the 12,684
	positions were called as length-1 seqlets. A single position is
	significant only if fewer than 1% of positions score at least as high.
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


def _span_p_values(X, seqlets, n_bins=1000):
	"""P(S >= s) for each seqlet, where s is the sum of its binned values.

	S is a sum of end - start independent draws from the binned values of X.
	"""

	bin_width = (X.max() - X.min()) / (n_bins - 1)
	bins = numpy.floor((X - X.min()) / bin_width).astype(int)
	f = numpy.bincount(bins.ravel(), minlength=n_bins) / bins.size

	pmfs = [numpy.ones(1)]
	for _ in range(int((seqlets['end'] - seqlets['start']).max())):
		pmfs.append(numpy.convolve(pmfs[-1], f))

	p_values = []
	for idx, start, end in seqlets.values[:, :3].astype(int):
		p_values.append(pmfs[end - start][bins[idx, start:end].sum():].sum())

	return numpy.array(p_values)


@pytest.fixture
def X_laplace():
	rng = numpy.random.RandomState(0)
	X = rng.laplace(size=(4, 500)) * 0.01
	X[rng.uniform(size=X.shape) < 0.002] *= 40
	X[:, 100:108] += 0.05
	X[:, 300:306] += 0.03
	return X


def test_recursive_seqlets_p_value_oracle(X_laplace):
	"""The p-value is P(S >= s), where s is the sum of the span's binned values.

	It used to look up the bin of the continuous sum, which is up to
	seqlet_len - 1 bins above s, in a table holding P(S > i) + P(S = 0).
	"""

	seqlets = recursive_seqlets(X_laplace, min_seqlet_len=6, max_seqlet_len=6)
	assert seqlets.shape[0] > 0

	assert_allclose(seqlets['p-value'], _span_p_values(X_laplace, seqlets),
		rtol=1e-6)


def test_recursive_seqlets_p_value_own_span(X_contrib):
	"""The reported p-value is that of the called span alone.

	It used to be the maximum over every prefix of the span from
	min_seqlet_len upward, so the ranking followed the weakest prefix: the
	seqlet with the largest attribution here was reported at 4.5e-3 rather
	than 1.9e-12.
	"""

	X = X_contrib.numpy().astype(numpy.float64)
	seqlets = recursive_seqlets(X)
	assert seqlets.shape[0] > 0

	assert_allclose(seqlets['p-value'], _span_p_values(X, seqlets), rtol=1e-6)
	assert seqlets['p-value'].is_monotonic_increasing


def test_recursive_seqlets_minimum_runs():
	"""A run of values equal to the minimum used to be called with p = 0.

	Rounding in the cumulative sum could put the span's sum just below
	seqlet_len * X.min(), giving a bin index of -1 that wrapped around to the
	end of the table. Here [31, 35) of example 0 is four values of -0.5.
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

	The out-of-bounds write sometimes segfaulted and sometimes returned an
	empty frame.
	"""

	X = numpy.random.RandomState(0).normal(size=(2, 3))
	seqlets = recursive_seqlets(X, min_seqlet_len=4)
	assert seqlets.shape == (0, 5)


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


def test_recursive_seqlets_control(X_laplace, X_background):
	"""With a control, seqlets are called at the largest threshold whose
	estimated false discovery rate is at or below target_fdr."""

	seqlets, fdr = recursive_seqlets(X_laplace, X_control=X_background,
		target_fdr=0.2, return_fdr=True)

	threshold = fdr.loc[fdr['fdr'] <= 0.2, 'threshold'].max()
	assert not numpy.isnan(threshold)
	assert seqlets.shape[0] > 0

	expected = recursive_seqlets(X_laplace, threshold=threshold)
	assert_array_almost_equal(seqlets.values, expected.values)


def test_recursive_seqlets_control_default_target():
	rng = numpy.random.RandomState(0)
	X = rng.laplace(size=(40, 1000)) * 0.01
	X[:, 500:510] += 0.2
	X_control = rng.laplace(size=(40, 1000)) * 0.01

	seqlets = recursive_seqlets(X, X_control=X_control)
	expected = recursive_seqlets(X, X_control=X_control, target_fdr=0.1)

	assert seqlets.shape[0] > 0
	assert_array_almost_equal(seqlets.values, expected.values)


def test_recursive_seqlets_control_return_type(X_laplace, X_background):
	result = recursive_seqlets(X_laplace, X_control=X_background,
		target_fdr=0.2, return_fdr=True)

	assert isinstance(result, RecursiveSeqletsResult)
	assert isinstance(result, tuple)
	assert isinstance(result.seqlets, pandas.DataFrame)
	assert result.seqlets.shape[0] > 0
	assert list(result.fdr.columns) == ['threshold', 'n_seqlets',
		'n_control_seqlets', 'fdr']

	seqlets = recursive_seqlets(X_laplace, X_control=X_background, 
		target_fdr=0.2)
	assert isinstance(seqlets, pandas.DataFrame)
	assert_array_almost_equal(seqlets.values, result.seqlets.values)


def test_recursive_seqlets_control_ignores_threshold(X_laplace, X_background):
	seqlets = recursive_seqlets(X_laplace, X_control=X_background, 
		target_fdr=0.2)
	seqlets2 = recursive_seqlets(X_laplace, threshold=1e-6, 
		X_control=X_background, target_fdr=0.2)

	assert seqlets.shape[0] > 0
	assert_array_almost_equal(seqlets.values, seqlets2.values)


def test_recursive_seqlets_control_input_types(X_laplace, X_background):
	seqlets = recursive_seqlets(X_laplace, X_control=X_background, 
		target_fdr=0.2)
	seqlets_torch = recursive_seqlets(torch.from_numpy(X_laplace),
		X_control=torch.from_numpy(X_background), target_fdr=0.2)
	seqlets_32 = recursive_seqlets(X_laplace, 
		X_control=X_background.astype(numpy.float32), target_fdr=0.2)

	assert seqlets.shape[0] > 0
	assert_array_almost_equal(seqlets.values, seqlets_torch.values)
	assert_array_almost_equal(seqlets.values, seqlets_32.values, 4)


def test_recursive_seqlets_control_unreachable():
	"""A target below every estimate warns and returns no seqlets."""

	rng = numpy.random.RandomState(0)
	X = rng.laplace(size=(40, 1000))
	X_control = rng.laplace(size=(40, 1000))

	with pytest.warns(TangermemeWarning, match="the lowest estimate is"):
		seqlets, fdr = recursive_seqlets(X, X_control=X_control, 
			target_fdr=0.01, return_fdr=True)

	assert seqlets.shape == (0, 5)
	assert list(seqlets.columns) == ['example_idx', 'start', 'end',
		'attribution', 'p-value']
	assert (fdr['fdr'].dropna() > 0.01).all()


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


def test_recursive_seqlets_return_fdr_requires_control(X_laplace):
	assert_raises(ValueError, recursive_seqlets, X_laplace, return_fdr=True)


def test_recursive_seqlets_control_empty(X_laplace):
	assert_raises(ValueError, recursive_seqlets, X_laplace, 
		X_control=numpy.zeros((0, 10)))
	assert_raises(ValueError, recursive_seqlets, X_laplace, 
		X_control=[0.1, 0.2])


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


def test_recursive_seqlets_fdr_default_thresholds(X_laplace, X_background):
	fdr = _fdr(X_laplace, X_background)

	assert fdr.shape == (117, 4)
	assert_allclose(fdr['threshold'].values[[0, -1]], [1e-1, 1e-30])
	assert (numpy.diff(fdr['threshold']) < 0).all()


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


def test_recursive_seqlets_fdr_control_range(X_laplace):
	"""Control values outside the range of X are placed in the end bins."""

	X_control = numpy.random.RandomState(2).laplace(size=(4, 500)) * 10
	fdr = _fdr(X_laplace, X_control, thresholds=[1e-2])

	assert fdr['n_control_seqlets'].iloc[0] > 0
