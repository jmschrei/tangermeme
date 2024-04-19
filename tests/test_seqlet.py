# test_seqlet.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import pytest

from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot

from tangermeme.ersatz import substitute
from tangermeme.ersatz import shuffle

from tangermeme.seqlet import _laplacian_null
from tangermeme.seqlet import _isotonic_thresholds
from tangermeme.seqlet import tfmodisco_seqlets

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
	assert len(pos_vals) == 44
	assert len(neg_vals) == 56

	assert_array_almost_equal(pos_vals, [
		1.22090971e-05, 2.02803848e-05, 8.73393038e-06, 1.03409958e-05,
        1.48567877e-08, 1.19890020e-05, 5.79677851e-06, 1.94775753e-05,
        1.05391889e-05, 3.24013149e-05, 2.59076207e-06, 1.40893103e-05,
        1.03477460e-05, 8.78341989e-06, 1.17252129e-05, 3.00224058e-08,
        1.23023269e-05, 7.36816087e-07, 1.63371485e-06, 2.88229316e-05,
        9.63908888e-06, 4.51452994e-06, 1.05481920e-05, 1.38811578e-05,
        1.13089714e-05, 3.76958651e-05, 2.73458602e-06, 2.56476319e-06,
        2.09661720e-06, 1.34401884e-05, 7.51790364e-07, 1.12043964e-05,
        1.84327732e-06, 1.05447688e-05, 2.89086244e-05, 5.10674873e-06,
        1.47756670e-06, 1.50236519e-05, 2.56852905e-05, 1.66425310e-05,
        2.02633882e-05, 1.64449227e-05, 1.67439266e-05, 5.81314521e-06], 4)

	assert_array_almost_equal(neg_vals, [
		-3.25293399e-06, -6.37458462e-07, -4.42756448e-06, -2.62816250e-06,
        -4.54668677e-06, -2.21357570e-06, -1.18334656e-06, -2.88041964e-07,
        -2.56843125e-06, -5.09924925e-06, -1.58853433e-06, -5.16662700e-06,
        -1.03240706e-05, -3.31991814e-06, -4.22059379e-07, -8.56494049e-06,
        -1.85201256e-06, -1.98249606e-07, -1.98506852e-06, -2.06148860e-06,
        -3.03279820e-07, -3.76533139e-06, -1.19148197e-06, -6.43718280e-06,
        -1.45179240e-06, -3.62063551e-06, -2.59422131e-06, -1.70385364e-06,
        -2.96543257e-07, -5.37868551e-06, -5.79400817e-07, -2.58260018e-06,
        -4.89157851e-06, -1.91445040e-06, -6.80515854e-07, -3.79386688e-07,
        -1.69966071e-06, -2.84712286e-07, -6.38786489e-07, -2.31688862e-06,
        -3.22436621e-06, -5.22296548e-06, -1.75587631e-06, -4.68473797e-06,
        -3.61473446e-06, -1.24144364e-06, -7.71661508e-06, -9.32228745e-07,
        -2.54515768e-06, -1.41388875e-06, -2.62365432e-06, -2.29926500e-08,
        -8.60240430e-07, -2.44534877e-07, -1.88617456e-07, -6.19247835e-06], 4)


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
	torch.manual_seed(0)
	X_neg = torch.randn(100)
	X_pos = X_contrib[X_contrib > 0.03]

	vals = _isotonic_thresholds(X_pos, X_neg, increasing=True, target_fdr=0.2, 
		min_frac_neg=0.95)

	assert_array_almost_equal(vals, 0.0326, 3)


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
		[0.0326, 0.0326, 0.0406, 0.0406], 3)


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
		[0.041, 0.041, 0.033, 0.033], 3)


###

def test_tfmodisco_seqlets(X_contrib):
	pos_seqlets, neg_seqlets = tfmodisco_seqlets(X_contrib, device='cpu')

	assert isinstance(pos_seqlets, torch.Tensor)
	assert isinstance(neg_seqlets, torch.Tensor)
	assert len(pos_seqlets) == 53
	assert len(neg_seqlets) == 105

	assert_array_almost_equal(pos_seqlets[:15], [
		[0.0000e+00, 1.0440e+03, 1.0850e+03, 3.0094e-01],
        [0.0000e+00, 1.5040e+03, 1.5450e+03, 9.2409e-02],
        [0.0000e+00, 5.6900e+02, 6.1000e+02, 4.9227e-02],
        [0.0000e+00, 1.2270e+03, 1.2680e+03, 3.8508e-02],
        [0.0000e+00, 1.4460e+03, 1.4870e+03, 2.3312e-02],
        [0.0000e+00, 7.6300e+02, 8.0400e+02, 1.9889e-02],
        [0.0000e+00, 1.0180e+03, 1.0590e+03, 1.3945e-02],
        [0.0000e+00, 9.1800e+02, 9.5900e+02, 1.2988e-02],
        [0.0000e+00, 1.4760e+03, 1.5170e+03, 1.1735e-02],
        [1.0000e+00, 1.0330e+03, 1.0740e+03, 3.2346e-01],
        [1.0000e+00, 9.0400e+02, 9.4500e+02, 1.4886e-01],
        [1.0000e+00, 1.5950e+03, 1.6360e+03, 5.5283e-02],
        [1.0000e+00, 1.5130e+03, 1.5540e+03, 4.9739e-02],
        [1.0000e+00, 7.1700e+02, 7.5800e+02, 2.6031e-02],
        [1.0000e+00, 5.7700e+02, 6.1800e+02, 8.0928e-03]], 4)

	assert_array_almost_equal(neg_seqlets[:15], [
		[ 0.0000e+00,  1.1740e+03,  1.2150e+03, -2.1869e-02],
        [ 0.0000e+00,  1.0990e+03,  1.1400e+03, -1.9116e-02],
        [ 0.0000e+00,  9.6600e+02,  1.0070e+03, -1.4645e-02],
        [ 0.0000e+00,  1.0730e+03,  1.1140e+03, -1.4579e-02],
        [ 0.0000e+00,  8.3300e+02,  8.7400e+02, -1.3569e-02],
        [ 0.0000e+00,  1.4090e+03,  1.4500e+03, -1.2975e-02],
        [ 0.0000e+00,  1.1280e+03,  1.1690e+03, -1.2931e-02],
        [ 0.0000e+00,  6.3400e+02,  6.7500e+02, -1.2800e-02],
        [ 0.0000e+00,  7.8500e+02,  8.2600e+02, -1.1932e-02],
        [ 0.0000e+00,  5.4700e+02,  5.8800e+02, -1.1393e-02],
        [ 0.0000e+00,  6.8500e+02,  7.2600e+02, -1.1189e-02],
        [ 0.0000e+00,  1.5480e+03,  1.5890e+03, -1.0852e-02],
        [ 0.0000e+00,  1.3690e+03,  1.4100e+03, -1.0658e-02],
        [ 0.0000e+00,  5.9200e+02,  6.3300e+02, -9.8248e-03],
        [ 0.0000e+00,  1.1510e+03,  1.1920e+03, -9.8113e-03]], 4)

	pos_counts = torch.unique(pos_seqlets[:, 0], return_counts=True)[1]
	neg_counts = torch.unique(neg_seqlets[:, 0], return_counts=True)[1]

	assert_array_almost_equal(pos_counts, [ 9,  6, 12,  8,  9,  9])
	assert_array_almost_equal(neg_counts, [18, 21, 16, 17, 15, 18])

	assert all(pos_seqlets[:, -1] > 0)
	assert all(neg_seqlets[:, -1] < 0)
