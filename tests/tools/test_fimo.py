# test_io.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest
import pandas

from tangermeme.utils import random_one_hot
from tangermeme.ersatz import substitute
from tangermeme.predict import predict

from tangermeme.tools.fimo import _pwm_to_mapping
from tangermeme.tools.fimo import FIMO

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


@pytest.fixture
def log_pwm():
	r = numpy.random.RandomState(0)

	pwm = numpy.exp(r.randn(4, 14))
	pwm = pwm / pwm.sum(axis=0, keepdims=True)
	return numpy.log2(pwm)

@pytest.fixture
def short_log_pwm():
	r = numpy.random.RandomState(0)

	pwm = numpy.exp(r.randn(4, 2))
	pwm = pwm / pwm.sum(axis=0, keepdims=True)
	return numpy.log2(pwm)

@pytest.fixture
def long_log_pwm():
	r = numpy.random.RandomState(0)

	pwm = numpy.exp(r.randn(4, 50))
	pwm = pwm / pwm.sum(axis=0, keepdims=True)
	return numpy.log2(pwm)

@pytest.fixture
def fimo():
	return FIMO("tests/data/test.meme")


###


def test_pwm_to_mapping(log_pwm):
	smallest, mapping = _pwm_to_mapping(log_pwm, 0.1)

	assert smallest == -596
	assert mapping.shape == (586,)
	assert mapping.dtype == numpy.float64


	assert_array_almost_equal(mapping[:8], [7.4903e-08,  6.4154e-08,  
		4.2656e-08,  2.1158e-08, -1.1089e-08, -6.4833e-08, -1.1858e-07, 
		-1.8307e-07], 4)
	assert_array_almost_equal(mapping[100:108], [-0.0082, -0.0087, -0.0093, 
		-0.0099, -0.0105, -0.0112, -0.0119, -0.0126], 4)
	assert_array_almost_equal([mapping[~numpy.isinf(mapping)].min()], 
		[-28.], 4) 

	assert numpy.all(numpy.diff(mapping[~numpy.isinf(mapping)]) <= 0)
	assert numpy.isinf(mapping).sum() == 103


def test_short_pwm_to_mapping(short_log_pwm):
	smallest, mapping = _pwm_to_mapping(short_log_pwm, 0.1)

	assert smallest == -78
	assert mapping.shape == (65,)
	assert mapping.dtype == numpy.float64

	assert_array_almost_equal(mapping[:8], [ 8.3267e-17, -9.3109e-02, 
		-1.9265e-01, -1.9265e-01, -1.9265e-01, -1.9265e-01, -1.9265e-01, 
		-1.9265e-01], 4)
	assert_array_almost_equal([mapping[~numpy.isinf(mapping)].min()], 
		[-4], 4) 

	assert numpy.all(numpy.diff(mapping[~numpy.isinf(mapping)]) <= 0)
	assert numpy.isinf(mapping).sum() == 4


def test_long_pwm_to_mapping(long_log_pwm):
	smallest, mapping = _pwm_to_mapping(long_log_pwm, 0.1)

	assert smallest == -2045
	assert mapping.shape == (2035,)
	assert mapping.dtype == numpy.float64

	assert_array_almost_equal(mapping[:8], [-9.79656934e-16, -7.45058160e-09, 
		-2.23517430e-08, -3.72529047e-08, -5.96046475e-08, -9.68575534e-08, 
		-1.34110461e-07, -1.78813951e-07], 4)
	assert_array_almost_equal(mapping[100:108], [-3.3035e-14, -3.8062e-14, 
		-4.4026e-14, -5.1093e-14, -5.9458e-14, -6.9351e-14, -8.1036e-14, 
		-9.4826e-14], 4)
	assert_array_almost_equal([mapping[~numpy.isinf(mapping)].min()], 
		[-99.], 4) 

	assert numpy.all(numpy.diff(mapping[~numpy.isinf(mapping)]) <= 0)
	assert numpy.isinf(mapping).sum() == 436


def test_pwm_to_mapping_small_bins(log_pwm):
	smallest, mapping = _pwm_to_mapping(log_pwm, 0.01)

	assert smallest == -5955
	assert mapping.shape == (5850,)
	assert mapping.dtype == numpy.float64

	assert_array_almost_equal(mapping[:8], [-4.5076e-07, -4.6939e-07, 
		-4.8429e-07, -4.9546e-07, -5.1036e-07, -5.2154e-07, -5.4017e-07, 
		-5.5134e-07], 4)
	assert_array_almost_equal(mapping[100:108], [-4.5076e-07, -4.6939e-07, 
		-4.8429e-07, -4.9546e-07, -5.1036e-07, -5.2154e-07, -5.4017e-07, 
		-5.5134e-07], 4)
	assert_array_almost_equal([mapping[~numpy.isinf(mapping)].min()], 
		[-28.], 4) 

	assert numpy.all(numpy.diff(mapping[~numpy.isinf(mapping)]) <= 0)
	assert numpy.isinf(mapping).sum() == 1024


def test_pwm_to_mapping_large_bins(log_pwm):
	smallest, mapping = _pwm_to_mapping(log_pwm, 1)

	assert smallest == -60
	assert mapping.shape == (60,)
	assert mapping.dtype == numpy.float64

	assert_array_almost_equal(mapping[:8], [-3.5277e-07, -3.9577e-07, 
		-8.0423e-07, -3.0078e-06, -1.1978e-05, -4.1823e-05, -1.2694e-04, 
		-3.4126e-04], 4)
	assert_array_almost_equal([mapping[~numpy.isinf(mapping)].min()], 
		[-27], 4) 

	assert numpy.all(numpy.diff(mapping[~numpy.isinf(mapping)]) <= 0)
	assert numpy.isinf(mapping).sum() == 8


##


def test_fimo(fimo):
	assert fimo.bin_size == 0.1
	assert tuple(fimo.alphabet) == tuple(['A', 'C', 'G', 'T'])

	assert len(fimo.motif_names) == 12
	assert_array_almost_equal(fimo.motif_lengths, [10,  9, 15, 15,  8, 20, 10, 
		10, 12, 12, 20, 10])
	assert fimo.n_motifs == 12

	assert_array_almost_equal(fimo.motif_pwms[0][:, :3].detach(), [
		[ 0.2726, -2.441 , -3.7744],
		[-0.0912,  1.0043, -1.116 ],
		[ 0.3947,  0.7401, -4.0455],
		[-0.8884, -2.8234,  1.7683]], 4)
	assert_array_almost_equal(fimo.motif_pwms[1][:, :3].detach(), [
		[  0.8834, -11.2877,  -1.574 ],
        [ -1.8404,  -3.4137,  -2.0541],
        [  0.6938, -11.2877,   1.732 ],
        [ -1.9428,   1.966 ,  -3.2798]], 4)
	assert_array_almost_equal(fimo.motif_pwms[2][:, :3].detach(), [
		[ 0.911 , -1.4407,  0.829 ],
        [-2.4391, -4.6295, -0.6795],
        [ 0.5022,  1.7452, -0.1969],
        [-0.9423, -2.0565, -0.4572]], 4)
	assert_array_almost_equal(fimo._smallest, [ -536,  -535,  -520,  -470,  
		-741,  -626,  -821,  -712,  -604, -1260,  -721,  -203]) 

	assert_array_almost_equal(fimo._score_to_pval[0][:5], [ 2.495928e-08, 
		-9.287154e-07, -1.882391e-06, -3.789744e-06, -5.697101e-06], 4)
	assert_array_almost_equal(fimo._score_to_pval[1][:5], [ 5.833474e-08, 
		-3.045972e-05, -6.097871e-05, -6.097871e-05, -9.149863e-05], 4)
	assert_array_almost_equal(fimo._score_to_pval[2][:5], [5.533182e-08, 
		5.160652e-08, 4.415594e-08, 3.298005e-08, 1.435360e-08], 4)
	assert_array_almost_equal(fimo._score_to_pval[3][:5], [5.517863e-08, 
		5.424730e-08, 5.145333e-08, 4.493405e-08, 3.003285e-08], 4)


def test_fimo_forward(fimo):
	X = random_one_hot((5, 4, 30), random_state=0).type(torch.float32)
	y_hat = predict(fimo, X, device='cpu')

	assert y_hat.shape == (5, 12, 2, 11)
	assert y_hat.dtype == torch.float32

	assert_array_almost_equal(y_hat[:2, :2, :, :5], [
		[[[-13.5308, -23.0580, -20.2292, -28.6583, -19.3354],
          [-27.0516, -10.3653, -18.6258, -20.7399, -24.2299]],

         [[-22.6512, -34.5771, -44.5998, -15.7849, -25.4221],
          [-19.6063, -43.5123, -16.0468, -30.2433, -24.7719]]],


        [[[-20.7360, -38.8207,  -5.8913, -20.0417, -30.2949],
          [-23.1269, -19.9556, -11.8838, -20.5346, -15.5351]],

         [[-25.6979, -22.5750, -39.6650, -20.1621, -20.5930],
          [-22.2861, -32.6734,   0.8325, -17.1021, -20.2180]]]], 4)


def test_fimo_hits(fimo):
	X = random_one_hot((18, 4, 50), random_state=0).type(torch.float32)

	hits = fimo.hits(X, device='cpu', dim=1)
	assert len(hits) == 18
	for df in hits:
		assert isinstance(df, pandas.DataFrame)
		assert df.shape[1] == 7
		assert tuple(df.columns) == ('motif', 'start', 'end', 'strand',
			'score', 'p-value', 'seq')

	hits = fimo.hits(X, device='cpu', dim=0)
	assert len(hits) == 12
	for df in hits:
		assert isinstance(df, pandas.DataFrame)
		assert df.shape[1] == 7
		assert tuple(df.columns) == ('example_idx', 'start', 'end', 'strand',
			'score', 'p-value', 'seq')


def test_fimo_hits_ap1_axis0(fimo):
	X = random_one_hot((1, 4, 50), random_state=0).type(torch.float32)
	X = substitute(X, "TGACGTCAT")

	hits = fimo.hits(X, device='cpu')

	assert len(hits) == 12
	assert_array_almost_equal(list(map(len, hits)), [0, 0, 0, 2, 0, 0, 0, 0, 0, 
		0, 0, 0])

	for df in hits:
		assert isinstance(df, pandas.DataFrame)
		assert df.shape[1] == 7
		assert tuple(df.columns) == ('example_idx', 'start', 'end', 'strand',
			'score', 'p-value', 'seq')

	df = hits[3]
	assert tuple(df['example_idx']) == tuple([0, 0])
	assert tuple(df['start']) == tuple([18, 17])
	assert tuple(df['end']) == tuple([33, 32])
	assert tuple(df['strand']) == tuple(['+', '-'])
	assert_array_almost_equal(df['score'], [16.0935, 15.5552], 4)
	assert_array_almost_equal(df['p-value'], [2.251e-06, 3.411e-06], 4)
	assert tuple(df['seq']) == tuple(['GCGTGACGTCATCAT', 'AGCGTGACGTCATCA'])


def test_fimo_hits_ap1_axis1(fimo):
	X = random_one_hot((1, 4, 50), random_state=0).type(torch.float32)
	X = substitute(X, "TGACGTCAT")

	hits = fimo.hits(X, device='cpu', dim=1)

	assert len(hits) == 1
	assert_array_almost_equal(list(map(len, hits)), [2])

	df = hits[0]
	assert isinstance(df, pandas.DataFrame)
	assert df.shape[1] == 7
	assert tuple(df.columns) == ('motif', 'start', 'end', 'strand',
		'score', 'p-value', 'seq')

	assert tuple(df['motif']) == tuple(['FOSL2+JUND_MA1145.1', 
		'FOSL2+JUND_MA1145.1'])
	assert tuple(df['start']) == tuple([18, 17])
	assert tuple(df['end']) == tuple([33, 32])
	assert tuple(df['strand']) == tuple(['+', '-'])
	assert_array_almost_equal(df['score'], [16.0935, 15.5552], 4)
	assert_array_almost_equal(df['p-value'], [2.2510e-06, 3.4114e-06], 4)
	assert tuple(df['seq']) == tuple(['GCGTGACGTCATCAT', 'AGCGTGACGTCATCA'])


def test_fimo_hit_matrix(fimo):
	X = random_one_hot((8, 4, 50), random_state=0).type(torch.float32)
	X = substitute(X, "TGACGTCAT")

	hits = fimo.hit_matrix(X, device='cpu')

	assert hits.shape == (8, 12)
	assert hits.dtype == torch.float32
	assert_array_almost_equal(hits, [
		[ 5.5792e+00,  4.2629e-01, -1.3749e+00,  1.6094e+01, -6.5121e+00,
         -2.0851e+00, -9.2959e+00, -7.4332e+00, -2.5657e+00, -2.5400e+01,
         -6.5174e+00,  6.3290e+00],
        [-3.0198e-01,  3.8548e+00,  5.6321e+00,  1.2371e+01, -5.1043e+00,
         -3.6458e-01, -1.2893e+01,  2.1829e+00, -1.6118e+00, -1.3580e+01,
         -1.2783e+01,  4.5077e+00],
        [-2.0197e+00,  8.4375e+00, -1.2127e-01,  1.4284e+01,  3.4388e+00,
         -8.7771e+00, -3.1086e+00,  2.2791e-02,  2.9074e-01, -2.3285e+01,
         -4.4699e+00,  1.1758e+00],
        [ 9.3114e+00,  6.3890e+00, -1.1486e+00,  1.3424e+01, -1.1651e+01,
         -5.5341e+00, -1.2394e+01, -1.4051e+01, -4.8145e+00, -2.4022e+01,
          2.2819e+00,  2.1922e+00],
        [ 4.4532e+00, -5.3541e+00,  2.2351e+00,  1.6020e+01, -7.5375e-01,
         -5.9520e+00, -1.5792e+01, -5.7051e+00,  1.7323e+00, -2.7297e+00,
         -9.6288e+00,  4.7946e+00],
        [ 4.3071e+00, -5.2839e+00,  2.2718e+00,  1.3015e+01, -8.1554e+00,
         -5.4550e+00,  1.4410e+00, -2.5368e+00, -4.5676e+00, -2.8154e+01,
         -8.8911e+00,  1.1646e+00],
        [ 1.3561e+00, -2.0182e-01,  3.1246e+00,  1.2345e+01, -5.2504e+00,
          3.4990e+00, -2.0052e+01, -7.9282e+00, -7.4422e+00, -3.2059e+01,
         -1.0036e+01,  1.9807e+00],
        [ 6.0968e+00, -6.5309e+00, -5.8497e-01,  1.3620e+01, -7.4371e+00,
         -2.0684e+00, -1.5606e+01, -1.3648e+01,  4.9869e+00, -1.1849e+01,
          2.9037e+00,  4.3577e+00]], 3)
