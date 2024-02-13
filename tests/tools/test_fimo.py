# test_io.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest
import pandas
import pyfaidx
import pyBigWig

from tangermeme.tools.fimo import _pwm_to_mapping

from nose.tools import assert_raises
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
