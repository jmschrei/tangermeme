# test_tomtom.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import pytest
import pandas

from tangermeme.io import read_meme
from tangermeme.utils import random_one_hot

from tangermeme.tools.tomtom import _binned_median
from tangermeme.tools.tomtom import _pairwise_max
from tangermeme.tools.tomtom import _merge_rc_results
from tangermeme.tools.tomtom import tomtom


from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


def generate_random_meme(n=5, min_len=4, max_len=20, random_state=0):
	state = numpy.random.RandomState(random_state)

	pwms = []
	for i in range(n):
		length = state.choice(max_len-min_len+1) + min_len
		
		pwm = state.choice(17, p=[0.2] + [0.05]*16, size=(length, 4))
		pwm[pwm.sum(axis=1) == 0] = [1, 0, 0, 0]
		pwm = pwm / pwm.sum(axis=1, keepdims=True)
		pwms.append(pwm.T)
	
	return pwms


###


def test_binned_median_odd_short():
	X = numpy.array([0, 4, 2, 1, 3], dtype='float64')
	bins = numpy.zeros((5, 2), dtype='float64')
	counts = numpy.ones(5, dtype='int64')
	assert _binned_median(X, bins, 0, 4, counts) == 2


def test_binned_median_odd_counts():
	X = numpy.array([0, 4, 2, 1, 3], dtype='float64')
	bins = numpy.zeros((5, 2), dtype='float64')
	counts = numpy.ones(5, dtype='int64')
	counts[1] = 6
	assert _binned_median(X, bins, 0, 4, counts) == 4


def test_binned_median_odd_long():
	X = numpy.array([0, 2, 8, 3, 4, 7, 9, 10, 4, 2, 1], dtype='float64')
	bins = numpy.zeros((len(X), 2), dtype='float64')
	counts = numpy.ones(len(X), dtype='int64')
	assert _binned_median(X, bins, 0, 10, counts) == 4


def test_binned_median_even():
	X = numpy.array([0, 1, 2, 3], dtype='float64')
	bins = numpy.zeros((4, 2), dtype='float64')
	counts = numpy.ones(4, dtype='int64')
	assert _binned_median(X, bins, 0, 3, counts) == 1


def test_binned_median_even_counts():
	X = numpy.array([0, 1, 2, 3], dtype='float64')
	bins = numpy.zeros((4, 2), dtype='float64')
	counts = numpy.ones(4, dtype='int64')
	counts[2] = 2
	assert _binned_median(X, bins, 0, 3, counts) == 2


def test_binned_median_long():
	X = numpy.random.RandomState(0).randn(200000)

	bins = numpy.zeros((1000, 2), dtype='float64')
	counts = numpy.ones(len(X), dtype='int64')
	assert_array_almost_equal([_binned_median(X, bins, X.min(), X.max(), 
		counts)], [numpy.median(X)], 2)


###

def test_pairwise_max():
	x = numpy.abs(numpy.random.RandomState(0).randn(100))
	x = x / x.sum()

	y = numpy.abs(numpy.random.RandomState(0).randn(100))
	y = y / y.sum()

	y_csum = numpy.cumsum(y, axis=-1)
	x_csum = numpy.cumsum(x, axis=-1)

	z = numpy.empty(100)
	_pairwise_max(x, y, y_csum, z, 100)

	assert_array_almost_equal(z[:10], [0.000475, 0.00024 , 0.000792, 0.002914, 
		0.003599, 0.002307, 0.002523, 0.000427, 0.000295, 0.001207])
	assert_array_almost_equal(z, x * y_csum + y * x_csum - x * y)


def test_pairwise_max_fallback():
	x = numpy.abs(numpy.random.RandomState(0).randn(100))
	x = x / x.sum()
	x[0] = -1

	y = numpy.abs(numpy.random.RandomState(0).randn(100))
	y = y / y.sum()

	y_csum = numpy.cumsum(y, axis=-1)

	z = numpy.empty(100)
	_pairwise_max(x, y, y_csum, z, 100)

	assert_array_almost_equal(z, y)


###


def test_merge_rc_results():
	results = numpy.random.RandomState(0).randn(1000, 4)

	idxs = results[:500, 1] > results[500:, 1]
	best_p = 1 - (1 - numpy.minimum(results[:500, 0], results[500:, 0])) ** 2
	best_scores = numpy.where(idxs, results[:500, 1], results[500:, 1])
	best_offsets = numpy.where(idxs, results[:500, 2], results[500:, 2])
	best_overlaps = numpy.where(idxs, results[:500, 3], results[500:, 3])

	_merge_rc_results(results)

	assert_array_almost_equal(results[:500, 0], best_p)
	assert_array_almost_equal(results[:500, 1], best_scores)
	assert_array_almost_equal(results[:500, 2], best_offsets)
	assert_array_almost_equal(results[:500, 3], best_overlaps)
	assert_array_almost_equal(results[:500, 4], idxs)


###


def test_tomtom():
	pwms = generate_random_meme(n=20)
	p, scores, offsets, overlaps, strands = tomtom(pwms, pwms)

	assert isinstance(p, numpy.ndarray)
	assert isinstance(scores, numpy.ndarray)
	assert isinstance(offsets, numpy.ndarray)
	assert isinstance(overlaps, numpy.ndarray)
	assert isinstance(strands, numpy.ndarray)

	assert p.dtype == numpy.float64
	assert scores.dtype == numpy.float64
	assert offsets.dtype == numpy.float64
	assert overlaps.dtype == numpy.float64
	assert overlaps.dtype == numpy.float64

	assert p.shape == (20, 20)
	assert scores.shape == (20, 20)
	assert offsets.shape == (20, 20)
	assert overlaps.shape == (20, 20)
	assert strands.shape == (20, 20)

	assert_array_almost_equal(p[0], [1.95399252e-14, 4.04372666e-01, 
		3.33899770e-01, 9.75969973e-01, 7.19287721e-01, 2.02698438e-01, 
		6.09413729e-01, 1.05202910e-01, 9.97854019e-01, 4.39470284e-01, 
		6.39529870e-01, 4.01256302e-01, 9.26477704e-01, 6.25608933e-01, 
		7.54828358e-01, 7.37538119e-01, 9.49853411e-01, 4.77573695e-01, 
		9.66135125e-01, 7.13675581e-01], 6)
	assert_array_almost_equal(scores[0], [1399., 1002.,  991.,  993.,  994., 
		1011.,  995., 1047.,  979.,  988.,  994.,  994., 986., 1015., 1003.,  
		998.,  982., 1023.,  976.,  981.])
	assert_array_almost_equal(offsets[0], [ 0., -4., -6., 14.,  2., -6.,  2.,  
		1., 11., -6., -7., -7.,  5., -4., -6.,  2.,  3.,  1., -7., -8.])
	assert_array_almost_equal(overlaps[0], [16.,  7.,  4.,  4.,  6.,  7.,  5., 
		16.,  3.,  4.,  7.,  5.,  4., 12., 10.,  8.,  5., 16., 6.,  4.])
	assert_array_almost_equal(strands[0], [])



def test_tomtom_subsets():
	pwms = generate_random_meme(n=20)
	p, scores, offsets, overlaps, strands = tomtom(pwms[:2], pwms)

	assert p.shape == (2, 20)
	assert scores.shape == (2, 20)
	assert offsets.shape == (2, 20)
	assert overlaps.shape == (2, 20)
	assert strands.shape == (2, 20)

	p2, scores2, offsets2, overlaps2, strands2 = tomtom(pwms[:5], pwms)

	assert p2.shape == (5, 20)
	assert scores2.shape == (5, 20)
	assert offsets2.shape == (5, 20)
	assert overlaps2.shape == (5, 20)
	assert strands2.shape == (5, 20)

	p3, scores3, offsets3, overlaps3, strands3 = tomtom(pwms, pwms)

	assert p3.shape == (20, 20)
	assert scores3.shape == (20, 20)
	assert offsets3.shape == (20, 20)
	assert overlaps3.shape == (20, 20)
	assert strands3.shape == (20, 20)

	assert_array_almost_equal(p, p2[:2], 6)
	assert_array_almost_equal(scores, scores2[:2], 6)
	assert_array_almost_equal(offsets, offsets2[:2], 6)
	assert_array_almost_equal(overlaps, overlaps2[:2], 6)
	assert_array_almost_equal(strands, strands2[:2], 6)

	assert_array_almost_equal(p, p3[:2], 6)
	assert_array_almost_equal(scores, scores3[:2], 6)
	assert_array_almost_equal(offsets, offsets3[:2], 6)
	assert_array_almost_equal(overlaps, overlaps3[:2], 6)
	assert_array_almost_equal(strands, strands3[:2], 6)


def test_tomtom_meme():
	pwms = list(read_meme("tests/data/test.meme").values())
	p, scores, offsets, overlaps, strands = tomtom(pwms[:1], pwms)

	assert p.shape == (1, 12)
	assert scores.shape == (1, 12)
	assert offsets.shape == (1, 12)
	assert overlaps.shape == (1, 12)
	assert strands.shape == (1, 12)

	assert_array_almost_equal(p[0], [-1.687538e-14, 0.959270, 0.990233, 
		0.501984, 0.662968, 0.993437, 0.218161, 0.999998, 0.2650769, 0.53301, 
		0.872186, 0.71878], 4)
	assert_array_almost_equal(scores[0], [879.0, 557.0, 565.0, 617.0, 582.0, 
		573.0, 626.0, 515.0, 628.0, 607.0, 599.0, 587.0], 6)
	assert_array_almost_equal(offsets[0], [0.0, 0.0, 0.0, 7.0, -2.0, 2.0, 2.0, 
		2.0, 1.0, -2.0, 7.0, 0.0], 6)
	assert_array_almost_equal(overlaps[0], [10.0, 9.0, 10.0, 8.0, 8.0, 10.0, 
		8.0, 8.0, 10.0, 8.0, 10.0, 10.0], 6)
	assert_array_almost_equal(strands[0], [])


def test_tomtom_reverse_complement():
	pwms = list(read_meme("tests/data/test.meme").values())
	p, scores, offsets, overlaps, strands = tomtom(pwms[:1], pwms, 
		reverse_complement=False)

	assert p.shape == (1, 12)
	assert scores.shape == (1, 12)
	assert offsets.shape == (1, 12)
	assert overlaps.shape == (1, 12)
	assert strands.shape == (1, 12)

	assert_array_almost_equal(p[0], [-0.0, 0.942776, 0.910807, 0.320109, 
		0.394581, 0.997576, 0.125284, 0.998832, 0.140479, 0.502672, 
		0.724997, 0.478826], 4)
	assert_array_almost_equal(scores[0], [877., 539., 563., 614., 584., 543., 
		624., 514., 628., 591., 592., 586.], 6)
	assert_array_almost_equal(offsets[0], [ 0., -1.,  0.,  6., -2.,  9.,  2.,  
		2.,  1.,  1.,  7.,  0.], 6)
	assert_array_almost_equal(overlaps[0], [10.,  9., 10.,  9.,  8., 10.,  8.,  
		8., 10., 10., 10., 10.], 6)
	assert_array_almost_equal(strands[0], [])


def test_tomtom_n_jobs():
	pwms = list(read_meme("tests/data/test.meme").values())
	p, scores, offsets, overlaps, strands = tomtom(pwms, pwms)
	p2, scores2, offsets2, overlaps2, strands2 = tomtom(pwms, pwms, n_jobs=1)

	assert_array_almost_equal(p, p2)
	assert_array_almost_equal(scores, scores2)
	assert_array_almost_equal(offsets, offsets2)
	assert_array_almost_equal(overlaps, overlaps2)
	assert_array_almost_equal(strands, strands2)


def test_tomtom_n_nearest():
	pwms = list(read_meme("tests/data/test.meme").values())
	p, scores, offsets, overlaps, strands, idxs = tomtom(pwms, pwms, 
		n_nearest=3)
	p2, scores2, offsets2, overlaps2, strands2 = tomtom(pwms, pwms)

	assert p.shape == (12, 3)
	assert scores.shape == (12, 3)
	assert offsets.shape == (12, 3)
	assert overlaps.shape == (12, 3)
	assert strands.shape == (12, 3)

	idxs = numpy.argsort(p2, axis=-1)[:, :3]
	for i, idx in enumerate(idxs):
		assert_array_almost_equal(p[i], p2[i, idx])
		assert_array_almost_equal(scores[i], scores2[i, idx])
		assert_array_almost_equal(offsets[i], offsets2[i, idx])
		assert_array_almost_equal(overlaps[i], overlaps2[i, idx])
		assert_array_almost_equal(strands[i], strands2[i, idx])


def test_tomtom_n_target_bins_small():
	pwms = list(read_meme("tests/data/test.meme").values())
	p, scores, offsets, overlaps, strands = tomtom(pwms[:1], pwms, 
		n_target_bins=10)

	assert p.shape == (1, 12)
	assert scores.shape == (1, 12)
	assert offsets.shape == (1, 12)
	assert overlaps.shape == (1, 12)
	assert strands.shape == (1, 12)

	assert_array_almost_equal(p[0], [-0.0, 0.921176, 0.994284, 0.525705, 
		0.653216, 0.985227, 0.214209, 0.999991, 0.304075, 0.455928, 0.848082, 
		0.788391], 4)
	assert_array_almost_equal(scores[0], [881., 565., 563., 617., 584., 580., 
		628., 521., 626., 614., 603., 583.], 6)
	assert_array_almost_equal(offsets[0], [ 0.,  0.,  0.,  6.,  0.,  2.,  2., 
		2.,  1., -2.,  7.,  0.], 6)
	assert_array_almost_equal(overlaps[0], [10.,  9., 10.,  9.,  8., 10.,  8.,  
		8., 10.,  8., 10., 10.], 6)
	assert_array_almost_equal(strands[0], [])
	