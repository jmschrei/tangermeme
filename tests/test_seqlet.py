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

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


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

	assert seqlets.shape == (45, 5)
	
	column_names = ['example_idx', 'start', 'end', 'attribution', 'p-value']
	for name1, name2 in zip(seqlets.columns, column_names):
		assert name1 == name2

	assert_array_almost_equal(seqlets.values[:20], [
	   [ 0.00000000e+00,  1.05200000e+03,  1.06100000e+03,  3.10099975e-01, -8.88178420e-15],
       [ 1.00000000e+00,  1.03500000e+03,  1.04300000e+03, -4.09194521e-02,  3.10862447e-15],
       [ 5.00000000e+00,  1.08000000e+03,  1.08700000e+03, -2.25865107e-02,  3.91696044e-04],
       [ 3.00000000e+00,  8.68000000e+02,  8.77000000e+02, -2.31096810e-02,  7.71604938e-04],
       [ 1.00000000e+00,  1.54400000e+03,  1.54800000e+03, -1.51427423e-02,  7.95967100e-04],
       [ 1.00000000e+00,  1.05200000e+03,  1.06100000e+03,  2.85009005e-01,  8.23045267e-04],
       [ 1.00000000e+00,  9.28000000e+02,  9.34000000e+02, -1.94051632e-02,  9.21901752e-04],
       [ 3.00000000e+00,  1.18500000e+03,  1.19200000e+03, -1.92600280e-02,  1.30565348e-03],
       [ 1.00000000e+00,  9.10000000e+02,  9.14000000e+02, -1.38847770e-02,  1.59193420e-03],
       [ 1.00000000e+00,  9.91000000e+02,  9.97000000e+02, -1.67364757e-02,  1.71210325e-03],
       [ 4.00000000e+00,  1.22800000e+03,  1.23400000e+03, -1.65439546e-02,  1.97550375e-03],
       [ 3.00000000e+00,  1.26400000e+03,  1.26800000e+03, -1.25000235e-02,  2.12257893e-03],
       [ 5.00000000e+00,  9.77000000e+02,  9.81000000e+02, -1.24873505e-02,  2.12257893e-03],
       [ 4.00000000e+00,  8.31000000e+02,  8.37000000e+02, -1.56007004e-02,  2.37060450e-03],
       [ 1.00000000e+00,  8.47000000e+02,  8.54000000e+02, -1.70986112e-02,  2.61130696e-03],
       [ 4.00000000e+00,  1.04500000e+03,  1.05300000e+03,  2.12268971e-01,  2.64335096e-03],
       [ 0.00000000e+00,  1.51700000e+03,  1.52100000e+03, -1.17767966e-02,  2.78588485e-03],
       [ 2.00000000e+00,  1.17900000e+03,  1.18300000e+03, -1.16636232e-02,  3.31652958e-03],
       [ 0.00000000e+00,  9.33000000e+02,  9.37000000e+02, -1.13330282e-02,  3.84717432e-03],
       [ 2.00000000e+00,  1.37400000e+03,  1.37800000e+03, -1.10360066e-02,  4.24515787e-03]], 4)


def test_recursive_seqlets_additional_flanks(X_contrib):
	seqlets = recursive_seqlets(X_contrib).values
	seqlets2 = recursive_seqlets(X_contrib, additional_flanks=2).values
	assert seqlets2.shape == (45, 5)

	assert_array_almost_equal(seqlets[:, 0], seqlets2[:, 0])
	assert_array_almost_equal(seqlets[:, 1], seqlets2[:, 1]+2)
	assert_array_almost_equal(seqlets[:, 2], seqlets2[:, 2]-2)

	seqlets2 = recursive_seqlets(X_contrib, additional_flanks=4).values
	assert seqlets2.shape == (45, 5)

	assert_array_almost_equal(seqlets[:, 0], seqlets2[:, 0])
	assert_array_almost_equal(seqlets[:, 1], seqlets2[:, 1]+4)
	assert_array_almost_equal(seqlets[:, 2], seqlets2[:, 2]-4)

	seqlets2 = recursive_seqlets(X_contrib, additional_flanks=7).values
	assert seqlets2.shape == (45, 5)


def test_recursive_seqlets_min_length(X_contrib):
	seqlets = recursive_seqlets(X_contrib).values
	seqlets2 = recursive_seqlets(X_contrib, min_seqlet_len=7).values
	seqlets3 = recursive_seqlets(X_contrib, min_seqlet_len=1).values

	assert (seqlets[:, 2] - seqlets[:, 1]).min() == 4
	assert (seqlets2[:, 2] - seqlets2[:, 1]).min() == 7
	assert (seqlets3[:, 2] - seqlets3[:, 1]).min() == 1


def test_recursive_seqlets_max_length(X_contrib):
	seqlets = recursive_seqlets(X_contrib).values
	seqlets2 = recursive_seqlets(X_contrib, max_seqlet_len=10).values
	seqlets3 = recursive_seqlets(X_contrib, max_seqlet_len=6).values

	assert (seqlets[:, 2] - seqlets[:, 1]).max() == 10
	assert (seqlets2[:, 2] - seqlets2[:, 1]).max() == 10
	assert (seqlets3[:, 2] - seqlets3[:, 1]).max() == 6


def test_recursive_seqlets_p_values(X_contrib):
	seqlets = recursive_seqlets(X_contrib).values
	seqlets2 = recursive_seqlets(X_contrib, threshold=0.1).values
	seqlets3 = recursive_seqlets(X_contrib, threshold=0.001).values

	assert seqlets.shape == (45, 5)
	assert seqlets2.shape == (443, 5)
	assert seqlets3.shape == (6, 5)
