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
    return numpy.log(pwm)

@pytest.fixture
def short_log_pwm():
    r = numpy.random.RandomState(0)

    pwm = numpy.exp(r.randn(4, 2))
    pwm = pwm / pwm.sum(axis=0, keepdims=True)
    return numpy.log(pwm)

@pytest.fixture
def long_log_pwm():
    r = numpy.random.RandomState(0)

    pwm = numpy.exp(r.randn(4, 50))
    pwm = pwm / pwm.sum(axis=0, keepdims=True)
    return numpy.log(pwm)

@pytest.fixture
def fimo():
    return FIMO("tests/data/test.meme")


###


def test_pwm_to_mapping(log_pwm):
    smallest, mapping = _pwm_to_mapping(log_pwm, 0.1)

    assert smallest == -413
    assert mapping.shape == (407,)
    assert mapping.dtype == numpy.float64

    assert_array_almost_equal(mapping[:8], [-9.79656934e-16, -7.45058160e-09, 
        -2.23517430e-08, -3.72529047e-08, -5.96046475e-08, -9.68575534e-08, 
        -1.34110461e-07, -1.78813951e-07], 4)
    assert_array_almost_equal(mapping[100:108], [-0.05485598, -0.05829404, 
        -0.06190348, -0.06568371, -0.06964217, -0.07378728, -0.07812264, 
        -0.08265364], 4)
    assert_array_almost_equal([mapping[~numpy.isinf(mapping)].min()], 
        [-19.4081], 4) 

    assert numpy.all(numpy.diff(mapping[~numpy.isinf(mapping)]) <= 0)
    assert numpy.isinf(mapping).sum() == 71


def test_short_pwm_to_mapping(short_log_pwm):
    smallest, mapping = _pwm_to_mapping(short_log_pwm, 0.1)

    assert smallest == -54
    assert mapping.shape == (45,)
    assert mapping.dtype == numpy.float64

    assert_array_almost_equal(mapping[:8], [ 2.2204e-16, -1.3353e-01, 
        -1.3353e-01, -1.3353e-01, -1.3353e-01, -1.3353e-01, -1.3353e-01, 
        -1.3353e-01], 4)
    assert_array_almost_equal([mapping[~numpy.isinf(mapping)].min()], 
        [-2.7726], 4) 

    assert numpy.all(numpy.diff(mapping[~numpy.isinf(mapping)]) <= 0)
    assert numpy.isinf(mapping).sum() == 3


def test_long_pwm_to_mapping(long_log_pwm):
    smallest, mapping = _pwm_to_mapping(long_log_pwm, 0.1)

    assert smallest == -1415
    assert mapping.shape == (1409,)
    assert mapping.dtype == numpy.float64

    assert_array_almost_equal(mapping[:8], [-9.79656934e-16, -7.45058160e-09, 
        -2.23517430e-08, -3.72529047e-08, -5.96046475e-08, -9.68575534e-08, 
        -1.34110461e-07, -1.78813951e-07], 4)
    assert_array_almost_equal(mapping[100:108], [-3.3035e-14, -3.8062e-14, 
        -4.4026e-14, -5.1093e-14, -5.9458e-14, -6.9351e-14, -8.1036e-14, 
        -9.4826e-14], 4)
    assert_array_almost_equal([mapping[~numpy.isinf(mapping)].min()], 
        [-68.6216], 4) 

    assert numpy.all(numpy.diff(mapping[~numpy.isinf(mapping)]) <= 0)
    assert numpy.isinf(mapping).sum() == 301


def test_pwm_to_mapping_small_bins(log_pwm):
    smallest, mapping = _pwm_to_mapping(log_pwm, 0.01)

    assert smallest == -4127
    assert mapping.shape == (4054,)
    assert mapping.dtype == numpy.float64

    assert_array_almost_equal(mapping[:8], [-4.5076e-07, -4.6939e-07, 
        -4.8429e-07, -4.9546e-07, -5.1036e-07, -5.2154e-07, -5.4017e-07, 
        -5.5134e-07], 4)
    assert_array_almost_equal(mapping[100:108], [-4.5076e-07, -4.6939e-07, 
        -4.8429e-07, -4.9546e-07, -5.1036e-07, -5.2154e-07, -5.4017e-07, 
        -5.5134e-07], 4)
    assert_array_almost_equal([mapping[~numpy.isinf(mapping)].min()], 
        [-19.4081], 4) 

    assert numpy.all(numpy.diff(mapping[~numpy.isinf(mapping)]) <= 0)
    assert numpy.isinf(mapping).sum() == 710


def test_pwm_to_mapping_large_bins(log_pwm):
    smallest, mapping = _pwm_to_mapping(log_pwm, 1)

    assert smallest == -42
    assert mapping.shape == (42,)
    assert mapping.dtype == numpy.float64

    assert_array_almost_equal(mapping[:8], [9.2757e-16, -2.3842e-07, 
        -3.3379e-06, -2.4200e-05, -1.2124e-04, -4.7029e-04, -1.5016e-03, 
        -4.1023e-03], 4)
    assert_array_almost_equal([mapping[~numpy.isinf(mapping)].min()], 
        [-14.5561], 4) 

    assert numpy.all(numpy.diff(mapping[~numpy.isinf(mapping)]) <= 0)
    assert numpy.isinf(mapping).sum() == 7


##


def test_fimo(fimo):
    assert fimo.bin_size == 0.1
    assert tuple(fimo.alphabet) == tuple(['A', 'C', 'G', 'T'])

    assert len(fimo.motif_names) == 12
    assert_array_almost_equal(fimo.motif_lengths, [10,  9, 15, 15,  8, 20, 10, 
        10, 12, 12, 20, 10])
    assert fimo.n_motifs == 12

    assert_array_almost_equal(fimo.motif_pwms[0][:, :3].detach(), [
        [ 0.188801, -1.693037, -2.618941],
        [-0.063408,  0.696003, -0.77401 ],
        [ 0.273411,  0.512848, -2.807429],
        [-0.616186, -1.958428,  1.225656]], 4)
    assert_array_almost_equal(fimo.motif_pwms[1][:, :3].detach(), [
        [ 0.612219, -8.517193, -1.091597],
        [-1.2764  , -2.368298, -1.424619],
        [ 0.48077 , -8.517193,  1.200495],
        [-1.347381,  1.362708, -2.275359]], 4)
    assert_array_almost_equal(fimo.motif_pwms[2][:, :3].detach(), [
        [ 0.631378, -0.999129,  0.574476],
        [-1.691733, -3.213888, -0.471284],
        [ 0.347977,  1.209617, -0.136737],
        [-0.653542, -1.426283, -0.31718 ]], 4)
    assert_array_almost_equal(fimo._smallest, [-391, -398, -366, -324, -547, 
        -440, -597, -516, -425, -947, -501, -142]) 

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
        [[[ -9.4548, -16.7310, -14.1029, -20.6167, -14.1041],
          [-18.8477,  -7.2383, -13.6544, -15.0993, -16.8987]],

         [[-16.4109, -25.3622, -33.0017, -11.6421, -19.0112],
          [-14.2926, -32.2468, -11.8239, -22.3686, -18.5594]]],


        [[[-14.4540, -28.3315,  -4.1532, -13.9553, -21.1004],
          [-16.1467, -14.5409,  -8.2615, -14.9590, -10.7953]],

         [[-18.5241, -17.0362, -29.5771, -14.6916, -14.9756],
          [-16.1551, -24.0470,   0.5747, -12.5609, -15.4019]]]], 4)


def test_fimo_hits(fimo):
    X = random_one_hot((18, 4, 50), random_state=0).type(torch.float32)

    hits = fimo.hits(X, device='cpu', dim=1)
    assert len(hits) == 18
    for df in hits:
        assert isinstance(df, pandas.DataFrame)
        assert df.shape[1] == 7
        assert tuple(df.columns) == ('motif', 'start', 'end', 'strand',
            'score', 'attr', 'seq')

    hits = fimo.hits(X, device='cpu', dim=0)
    assert len(hits) == 12
    for df in hits:
        assert isinstance(df, pandas.DataFrame)
        assert df.shape[1] == 7
        assert tuple(df.columns) == ('example_idx', 'start', 'end', 'strand',
            'score', 'attr', 'seq')


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
            'score', 'attr', 'seq')

    df = hits[3]
    assert tuple(df['example_idx']) == tuple([0, 0])
    assert tuple(df['start']) == tuple([18, 17])
    assert tuple(df['end']) == tuple([33, 32])
    assert tuple(df['strand']) == tuple(['+', '-'])
    assert_array_almost_equal(df['score'], [11.153422, 10.780101], 4)
    assert tuple(df['attr']) == tuple(['.', '.'])
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
        'score', 'attr', 'seq')

    assert tuple(df['motif']) == tuple(['FOSL2+JUND_MA1145.1', 
        'FOSL2+JUND_MA1145.1'])
    assert tuple(df['start']) == tuple([18, 17])
    assert tuple(df['end']) == tuple([33, 32])
    assert tuple(df['strand']) == tuple(['+', '-'])
    assert_array_almost_equal(df['score'], [11.153422, 10.780101], 4)
    assert tuple(df['attr']) == tuple(['.', '.'])
    assert tuple(df['seq']) == tuple(['GCGTGACGTCATCAT', 'AGCGTGACGTCATCA'])


def test_fimo_hit_matrix(fimo):
    X = random_one_hot((8, 4, 50), random_state=0).type(torch.float32)
    X = substitute(X, "TGACGTCAT")

    hits = fimo.hit_matrix(X, device='cpu')

    assert hits.shape == (8, 12)
    assert hits.dtype == torch.float32
    assert_array_almost_equal(hits, [
        [ 3.8652e+00,  2.8951e-01, -9.5847e-01,  1.1153e+01, -5.2087e+00,
         -1.4546e+00, -6.5505e+00, -5.1777e+00, -1.7942e+00, -1.9687e+01,
         -4.5260e+00,  4.3852e+00],
        [-2.1328e-01,  2.6698e+00,  3.8999e+00,  8.5722e+00, -4.2323e+00,
         -2.6135e-01, -9.6599e+00,  1.4882e+00, -1.1268e+00, -1.0801e+01,
         -8.8860e+00,  3.1225e+00],
        [-1.4063e+00,  5.8473e+00, -9.6352e-02,  9.8986e+00,  2.3795e+00,
         -6.0973e+00, -2.1882e+00,  5.1445e-03,  1.8928e-01, -1.8220e+01,
         -3.1070e+00,  8.1239e-01],
        [ 6.4528e+00,  4.4270e+00, -8.0857e-01,  9.3022e+00, -8.7719e+00,
         -3.8464e+00, -8.6978e+00, -9.7951e+00, -3.3470e+00, -1.8731e+01,
          1.5757e+00,  1.5162e+00],
        [ 3.0826e+00, -3.7339e+00,  1.5449e+00,  1.1103e+01, -1.2162e+00,
         -4.1554e+00, -1.1667e+01, -4.4523e+00,  1.1963e+00, -2.5872e+00,
         -6.6982e+00,  3.3214e+00],
        [ 2.9831e+00, -3.6683e+00,  1.5702e+00,  9.0188e+00, -6.3481e+00,
         -3.7902e+00,  3.0439e-01, -2.4600e+00, -3.1731e+00, -2.1596e+01,
         -6.1970e+00,  8.0342e-01],
        [ 9.3562e-01, -1.4396e-01,  2.1614e+00,  8.5541e+00, -4.3341e+00,
          2.4192e+00, -1.4004e+01, -6.2071e+00, -5.1894e+00, -2.4303e+01,
         -6.9739e+00,  1.3705e+00],
        [ 4.2241e+00, -4.5435e+00, -4.1188e-01,  9.4372e+00, -5.8506e+00,
         -1.4410e+00, -1.0881e+01, -1.0169e+01,  3.4529e+00, -9.6008e+00,
          2.0072e+00,  3.0185e+00]], 3)


