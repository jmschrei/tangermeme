# test_ablate.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>


import torch
torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(0)

import numpy
import pytest

from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot

from tangermeme.ablate import ablate
from tangermeme.ersatz import shuffle
from tangermeme.ersatz import substitute
from tangermeme.deep_lift_shap import deep_lift_shap

from .toy_models import SumModel
from .toy_models import FlattenDense
from .toy_models import Conv
from .toy_models import Scatter
from .toy_models import ConvDense
from .toy_models import SmallDeepSEA

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


@pytest.fixture
def X():
	X_ = random_one_hot((64, 4, 100), random_state=0).type(torch.float32)
	X_ = substitute(X_, "ACGTACGT")
	return X_


@pytest.fixture
def alpha():
	r = numpy.random.RandomState(0)
	return torch.from_numpy(r.randn(64, 1)).type(torch.float32)

@pytest.fixture
def beta():
	r = numpy.random.RandomState(1)
	return torch.from_numpy(r.randn(64, 1)).type(torch.float32)


##

def test_ablate_summodel(X):
	torch.manual_seed(0)
	model = SumModel()
	y_before, y_after = ablate(model, X, 46, 54, batch_size=5, device='cpu')

	assert y_before.shape == (64, 4)
	assert y_before.dtype == torch.float32
	assert y_before.sum() == X.sum()

	assert_array_almost_equal(y_before[:8], [
		[26., 23., 20., 31.],
        [27., 18., 29., 26.],
        [27., 25., 22., 26.],
        [21., 33., 18., 28.],
        [26., 22., 24., 28.],
        [28., 28., 22., 22.],
        [28., 16., 23., 33.],
        [20., 29., 31., 20.]])

	assert y_after.shape == (64, 20, 4)
	assert y_after.dtype == torch.float32
	assert y_after[:, 0].sum() == X.sum()
	assert_array_almost_equal(y_after[:8, 0], [
		[26., 23., 20., 31.],
        [27., 18., 29., 26.],
        [27., 25., 22., 26.],
        [21., 33., 18., 28.],
        [26., 22., 24., 28.],
        [28., 28., 22., 22.],
        [28., 16., 23., 33.],
        [20., 29., 31., 20.]])

	for i in range(8):
		assert_array_almost_equal(y_after[:, 0], y_after[:, i])
		assert_array_almost_equal(y_before, y_after[:, 0])

	y_before2, y_after2 = ablate(model, X, 0, -1, batch_size=1, 
		device='cpu')
	assert_array_almost_equal(y_before, y_before2)
	assert_array_almost_equal(y_after, y_after2)	


def test_ablate_flattendense(X):
	torch.manual_seed(0)
	model = FlattenDense()
	y_before, y_after = ablate(model, X, 46, 54, random_state=0, batch_size=8, 
		device='cpu')

	assert y_before.shape == (64, 3)
	assert y_before.dtype == torch.float32
	assert_array_almost_equal(y_before[:8], [
		[ 0.2161,  0.3207,  0.0147],
        [-0.0664, -0.0846, -0.2305],
        [-0.0801,  0.1022, -0.0152],
        [-0.0967,  0.0734, -0.0764],
        [ 0.0647, -0.1418, -0.1951],
        [-0.3011, -0.0956, -0.3683],
        [-0.2201, -0.0099, -0.2535],
        [-0.3861,  0.6259, -0.3530]], 4)

	assert y_after.shape == (64, 20, 3)
	assert y_after.dtype == torch.float32
	assert_array_almost_equal(y_after[:8, :2], [
		[[ 0.1324,  0.3553,  0.1465],
         [ 0.2455,  0.3811,  0.1003]],

        [[-0.1501, -0.0500, -0.0986],
         [-0.0370, -0.0243, -0.1448]],

        [[-0.1639,  0.1369,  0.1167],
         [-0.0508,  0.1626,  0.0705]],

        [[-0.1805,  0.1081,  0.0555],
         [-0.0674,  0.1338,  0.0092]],

        [[-0.0191, -0.1072, -0.0632],
         [ 0.0940, -0.0814, -0.1094]],

        [[-0.3848, -0.0610, -0.2364],
         [-0.2718, -0.0352, -0.2826]],

        [[-0.3039,  0.0247, -0.1216],
         [-0.1908,  0.0504, -0.1678]],

        [[-0.4698,  0.6605, -0.2211],
         [-0.3568,  0.6863, -0.2673]]], 4)

	y_before2, y_after2 = ablate(model, X, 0, -1, batch_size=1, 
		device='cpu')
	assert_array_almost_equal(y_before, y_before2, 4)
	assert_raises(AssertionError, assert_array_almost_equal, y_after, y_after2)


def test_ablate_flattendense_alpha_beta(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense()

	y_before, y_after = ablate(model, X, 46, 54, random_state=0, batch_size=8,
		args=(alpha, beta), device='cpu')
	y_before0, y_after0 = ablate(model, X, 46, 54, random_state=0, batch_size=8,
		device='cpu')

	assert y_before.shape == (64, 3)
	assert y_before.dtype == torch.float32
	assert_array_almost_equal(y_before, y_before0 * beta + alpha)
	assert_array_almost_equal(y_after, y_after0 * beta[:, None] + alpha[:, None])

	assert_array_almost_equal(y_before[:2], [
		[2.1151, 2.2849, 1.7879],
        [0.4407, 0.4519, 0.5412]], 4)

	assert y_after.shape == (64, 20, 3)
	assert y_after.dtype == torch.float32
	assert_array_almost_equal(y_after[:2, :2], [
		[[1.9791, 2.3412, 2.0021],
         [2.1628, 2.3830, 1.9270]],

        [[0.4920, 0.4307, 0.4605],
         [0.4228, 0.4150, 0.4888]]], 4)


def test_ablate_conv(X):
	torch.manual_seed(0)
	model = Conv()
	y_before, y_after = ablate(model, X, 46, 54, random_state=0, batch_size=8, 
		device='cpu')

	assert y_before.shape == (64, 12, 98)
	assert y_before.dtype == torch.float32
	assert_array_almost_equal(y_before[:2, :4, :6], [
		[[-0.2713, -0.5317, -0.3737, -0.4055, -0.3269, -0.3269],
         [-0.2988,  0.0980, -0.1013, -0.2560,  0.2595,  0.2595],
         [-0.1247,  0.1433, -0.3111, -0.3220, -0.4084, -0.4084],
         [-0.1362, -0.0067,  0.5554,  0.1810,  0.2007,  0.2007]],

        [[-0.4805, -0.1669, -0.5863, -0.0537, -0.2702, -0.1669],
         [-0.3709, -0.3077, -0.5910,  0.0165, -0.6573, -0.3077],
         [ 0.4396, -0.1558,  0.2097, -0.0929,  0.6608, -0.1558],
         [-0.0027,  0.4644,  0.1406, -0.1111, -0.3111,  0.4644]]], 4)

	assert y_after.shape == (64, 20, 12, 98)
	assert y_after.dtype == torch.float32
	assert_array_almost_equal(y_after[:2, :2, :4, :6], [
		[[[-0.2713, -0.5317, -0.3737, -0.4055, -0.3269, -0.3269],
          [-0.2988,  0.0980, -0.1013, -0.2560,  0.2595,  0.2595],
          [-0.1247,  0.1433, -0.3111, -0.3220, -0.4084, -0.4084],
          [-0.1362, -0.0067,  0.5554,  0.1810,  0.2007,  0.2007]],

         [[-0.2713, -0.5317, -0.3737, -0.4055, -0.3269, -0.3269],
          [-0.2988,  0.0980, -0.1013, -0.2560,  0.2595,  0.2595],
          [-0.1247,  0.1433, -0.3111, -0.3220, -0.4084, -0.4084],
          [-0.1362, -0.0067,  0.5554,  0.1810,  0.2007,  0.2007]]],


        [[[-0.4805, -0.1669, -0.5863, -0.0537, -0.2702, -0.1669],
          [-0.3709, -0.3077, -0.5910,  0.0165, -0.6573, -0.3077],
          [ 0.4396, -0.1558,  0.2097, -0.0929,  0.6608, -0.1558],
          [-0.0027,  0.4644,  0.1406, -0.1111, -0.3111,  0.4644]],

         [[-0.4805, -0.1669, -0.5863, -0.0537, -0.2702, -0.1669],
          [-0.3709, -0.3077, -0.5910,  0.0165, -0.6573, -0.3077],
          [ 0.4396, -0.1558,  0.2097, -0.0929,  0.6608, -0.1558],
          [-0.0027,  0.4644,  0.1406, -0.1111, -0.3111,  0.4644]]]], 4)

	for i in range(20):
		assert_array_almost_equal(y_before[:, :, :35], y_after[:, i, :, :35])
		assert_array_almost_equal(y_before[:, :, 60:], y_after[:, i, :, 60:])
		assert_raises(AssertionError, assert_array_almost_equal, 
			y_before[:, :, 35:60], y_after[:, i, :, 35:60])


def test_ablate_scatter(X):
	torch.manual_seed(0)
	model = Scatter()
	y_before, y_after = ablate(model, X, 46, 54, random_state=0, batch_size=8, 
		device='cpu')

	assert y_before.shape == (64, 100, 4)
	assert y_before.dtype == torch.float32
	assert y_before.sum() == X.sum()
	assert (y_before.sum(axis=-1) == X.sum(axis=1)).all()
	assert (y_before.sum(axis=1) == X.sum(axis=-1)).all()
	assert_array_almost_equal(y_before[:4, :5], [
		[[1., 0., 0., 0.],
         [0., 0., 0., 1.],
         [0., 1., 0., 0.],
         [1., 0., 0., 0.],
         [0., 0., 0., 1.]],

        [[0., 1., 0., 0.],
         [0., 0., 1., 0.],
         [1., 0., 0., 0.],
         [0., 0., 0., 1.],
         [1., 0., 0., 0.]],

        [[0., 1., 0., 0.],
         [0., 0., 1., 0.],
         [0., 0., 0., 1.],
         [0., 0., 1., 0.],
         [0., 0., 1., 0.]],

        [[0., 1., 0., 0.],
         [1., 0., 0., 0.],
         [0., 0., 0., 1.],
         [1., 0., 0., 0.],
         [0., 0., 1., 0.]]], 4)

	assert y_after.shape == (64, 20, 100, 4)
	assert y_after.dtype == torch.float32
	assert y_after[:, 0].sum() == X.sum()
	assert_array_almost_equal(y_after[:4, :2, :5], [
		[[[1., 0., 0., 0.],
          [0., 0., 0., 1.],
          [0., 1., 0., 0.],
          [1., 0., 0., 0.],
          [0., 0., 0., 1.]],

         [[1., 0., 0., 0.],
          [0., 0., 0., 1.],
          [0., 1., 0., 0.],
          [1., 0., 0., 0.],
          [0., 0., 0., 1.]]],


        [[[0., 1., 0., 0.],
          [0., 0., 1., 0.],
          [1., 0., 0., 0.],
          [0., 0., 0., 1.],
          [1., 0., 0., 0.]],

         [[0., 1., 0., 0.],
          [0., 0., 1., 0.],
          [1., 0., 0., 0.],
          [0., 0., 0., 1.],
          [1., 0., 0., 0.]]],


        [[[0., 1., 0., 0.],
          [0., 0., 1., 0.],
          [0., 0., 0., 1.],
          [0., 0., 1., 0.],
          [0., 0., 1., 0.]],

         [[0., 1., 0., 0.],
          [0., 0., 1., 0.],
          [0., 0., 0., 1.],
          [0., 0., 1., 0.],
          [0., 0., 1., 0.]]],


        [[[0., 1., 0., 0.],
          [1., 0., 0., 0.],
          [0., 0., 0., 1.],
          [1., 0., 0., 0.],
          [0., 0., 1., 0.]],

         [[0., 1., 0., 0.],
          [1., 0., 0., 0.],
          [0., 0., 0., 1.],
          [1., 0., 0., 0.],
          [0., 0., 1., 0.]]]], 4)


def test_ablate_convdense(X):
	torch.manual_seed(0)
	model = ConvDense()
	y_before, y_after = ablate(model, X, 46, 54, random_state=0, batch_size=2, 
		device='cpu')

	assert len(y_before) == 2
	assert y_before[0].shape == (64, 12, 98)
	assert y_before[1].shape == (64, 3)

	assert y_before[0].dtype == torch.float32
	assert y_before[1].dtype == torch.float32

	assert_array_almost_equal(y_before[0][:2, :4, 42:46], [
		[[-0.4906, -0.6971,  0.0397, -0.6363],
         [-0.3881,  0.2000,  0.2042,  0.1789],
         [ 0.8900,  0.4674,  0.5009,  0.5628],
         [-0.4831, -0.2439, -0.0296, -0.4263]],

        [[-0.5812, -0.2696, -0.7301, -0.7149],
         [ 0.2148,  0.2771,  0.1086,  0.0737],
         [ 0.2273,  0.3059,  0.5017,  0.4734],
         [-0.7614, -0.5326, -0.1356, -0.7810]]], 4)

	assert_array_almost_equal(y_before[1][:4], [
		[ 0.2161,  0.3207,  0.0147],
        [-0.0664, -0.0846, -0.2305],
        [-0.0801,  0.1022, -0.0152],
        [-0.0967,  0.0734, -0.0764]], 4)


	assert len(y_after) == 2
	assert y_after[0].shape == (64, 20, 12, 98)
	assert y_after[1].shape == (64, 20, 3)

	assert y_after[0].dtype == torch.float32
	assert y_after[1].dtype == torch.float32

	assert_array_almost_equal(y_after[0][:2, :2, :4, 42:46], [
		[[[-4.9062e-01, -6.9711e-01, -4.2652e-04, -7.0773e-01],
          [-3.8814e-01,  2.0001e-01,  8.2816e-02, -1.9818e-01],
          [ 8.9002e-01,  4.6741e-01,  4.4968e-01,  4.3157e-01],
          [-4.8307e-01, -2.4389e-01, -2.8948e-01, -3.6251e-01]],

         [[-4.9062e-01, -6.9711e-01,  1.3353e-01, -3.0229e-01],
          [-3.8814e-01,  2.0001e-01,  2.7458e-01, -6.5521e-02],
          [ 8.9002e-01,  4.6741e-01,  5.6203e-01,  5.7124e-01],
          [-4.8307e-01, -2.4389e-01, -3.2021e-01, -4.2439e-01]]],


        [[[-5.8118e-01, -2.6960e-01, -7.7023e-01, -7.8636e-01],
          [ 2.1482e-01,  2.7712e-01, -1.2824e-02, -3.0346e-01],
          [ 2.2734e-01,  3.0591e-01,  4.5041e-01,  3.4224e-01],
          [-7.6136e-01, -5.3260e-01, -3.9553e-01, -7.1724e-01]],

         [[-5.8118e-01, -2.6960e-01, -6.3627e-01, -3.8092e-01],
          [ 2.1482e-01,  2.7712e-01,  1.7894e-01, -1.7080e-01],
          [ 2.2734e-01,  3.0591e-01,  5.6276e-01,  4.8191e-01],
          [-7.6136e-01, -5.3260e-01, -4.2626e-01, -7.7912e-01]]]], 4)

	assert_array_almost_equal(y_after[1][:2, :4], [
		[[ 0.1324,  0.3553,  0.1465],
         [ 0.2455,  0.3811,  0.1003],
         [ 0.1798,  0.2700,  0.1818],
         [ 0.1989,  0.2952,  0.0110]],

        [[-0.1501, -0.0500, -0.0986],
         [-0.0370, -0.0243, -0.1448],
         [-0.1027, -0.1353, -0.0634],
         [-0.0835, -0.1101, -0.2342]]], 4)


def test_ablate_convdense_alpha(X, alpha):
	torch.manual_seed(0)
	model = ConvDense()

	y_before, y_after = ablate(model, X, 46, 54, random_state=0, 
		args=(alpha[:, :, None],), batch_size=2, device='cpu')

	assert len(y_before) == 2
	assert y_before[0].shape == (64, 12, 98)
	assert y_before[1].shape == (64, 3)

	assert y_before[0].dtype == torch.float32
	assert y_before[1].dtype == torch.float32

	assert_array_almost_equal(y_before[0][:2, :4, 42:46], [
		[[ 1.2734,  1.0669,  1.8037,  1.1278],
         [ 1.3759,  1.9641,  1.9683,  1.9430],
         [ 2.6541,  2.2315,  2.2650,  2.3268],
         [ 1.2810,  1.5202,  1.7345,  1.3378]],

        [[-0.1810,  0.1306, -0.3300, -0.3147],
         [ 0.6150,  0.6773,  0.5087,  0.4738],
         [ 0.6275,  0.7061,  0.9018,  0.8736],
         [-0.3612, -0.1324,  0.2645, -0.3808]]], 4)

	assert_array_almost_equal(y_before[1][:4], [
		[ 0.2161,  0.3207,  0.0147],
        [-0.0664, -0.0846, -0.2305],
        [-0.0801,  0.1022, -0.0152],
        [-0.0967,  0.0734, -0.0764]], 4)


	assert len(y_after) == 2
	assert y_after[0].shape == (64, 20, 12, 98)
	assert y_after[1].shape == (64, 20, 3)

	assert y_after[0].dtype == torch.float32
	assert y_after[1].dtype == torch.float32

	assert_array_almost_equal(y_after[0][:2, :2, :4, 42:46], [
		[[[ 1.2734,  1.0669,  1.7636,  1.0563],
          [ 1.3759,  1.9641,  1.8469,  1.5659],
          [ 2.6541,  2.2315,  2.2137,  2.1956],
          [ 1.2810,  1.5202,  1.4746,  1.4015]],

         [[ 1.2734,  1.0669,  1.8976,  1.4618],
          [ 1.3759,  1.9641,  2.0386,  1.6985],
          [ 2.6541,  2.2315,  2.3261,  2.3353],
          [ 1.2810,  1.5202,  1.4438,  1.3397]]],


        [[[-0.1810,  0.1306, -0.3701, -0.3862],
          [ 0.6150,  0.6773,  0.3873,  0.0967],
          [ 0.6275,  0.7061,  0.8506,  0.7424],
          [-0.3612, -0.1324,  0.0046, -0.3171]],

         [[-0.1810,  0.1306, -0.2361,  0.0192],
          [ 0.6150,  0.6773,  0.5791,  0.2294],
          [ 0.6275,  0.7061,  0.9629,  0.8821],
          [-0.3612, -0.1324, -0.0261, -0.3790]]]], 4)

	assert_array_almost_equal(y_after[1][:2, :4], [
		[[ 0.1324,  0.3553,  0.1465],
         [ 0.2455,  0.3811,  0.1003],
         [ 0.1798,  0.2700,  0.1818],
         [ 0.1989,  0.2952,  0.0110]],

        [[-0.1501, -0.0500, -0.0986],
         [-0.0370, -0.0243, -0.1448],
         [-0.1027, -0.1353, -0.0634],
         [-0.0835, -0.1101, -0.2342]]], 4)


def test_ablate_convdense_batch_size(X, alpha):
	torch.manual_seed(0)
	model = ConvDense()
	y_before0, y_after0 = ablate(model, X, 46, 54, random_state=0, 
		args=(alpha[:, :, None],), batch_size=1, device='cpu')

	y_before1, y_after1 = ablate(model, X, 46, 54, random_state=0, 
		args=(alpha[:, :, None],), batch_size=5, device='cpu')

	y_before2, y_after2 = ablate(model, X, 46, 54, random_state=0, 
		args=(alpha[:, :, None],), batch_size=68, device='cpu')

	assert len(y_before0) == len(y_before1) == len(y_before2) == 2
	assert len(y_after0) == len(y_after1) == len(y_after2) == 2
	
	assert y_before0[0].shape == (64, 12, 98)
	assert y_before0[1].shape == (64, 3)

	assert y_before1[0].shape == (64, 12, 98)
	assert y_before1[1].shape == (64, 3)

	assert y_before2[0].shape == (64, 12, 98)
	assert y_before2[1].shape == (64, 3)

	assert y_after0[0].shape == (64, 20, 12, 98)
	assert y_after0[1].shape == (64, 20, 3)

	assert y_after1[0].shape == (64, 20, 12, 98)
	assert y_after1[1].shape == (64, 20, 3)

	assert y_after2[0].shape == (64, 20, 12, 98)
	assert y_after2[1].shape == (64, 20, 3)

	assert_array_almost_equal(y_before0[0], y_before1[0])
	assert_array_almost_equal(y_before0[0], y_before2[0])

	assert_array_almost_equal(y_before0[1], y_before1[1])
	assert_array_almost_equal(y_before0[1], y_before2[1])

	assert_array_almost_equal(y_after0[0], y_after1[0])
	assert_array_almost_equal(y_after0[0], y_after2[0])

	assert_array_almost_equal(y_after0[1], y_after1[1])
	assert_array_almost_equal(y_after0[1], y_after2[1])


def test_ablate_raises_shape(X, alpha):
	torch.manual_seed(0)
	model = ConvDense()

	assert_raises(ValueError, ablate, model, X[0], 46, 54, device='cpu')
	assert_raises(ValueError, ablate, model, X[:, 0], 46, 54, device='cpu')
	assert_raises(ValueError, ablate, model, X.unsqueeze(0), 46, 54, 
		device='cpu')

	assert_raises(ValueError, ablate, model, X, 46, 54, args=(alpha[:2],), 
		device='cpu')
	assert_raises(ValueError, ablate, model, X, 46, 54, args=alpha, 
		device='cpu')
	assert_raises(ValueError, ablate, model, X, 46, 54, args=(alpha[:2, None],),
		device='cpu')


def test_ablate_raises_args(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense()

	assert_raises(ValueError, ablate, model, X, -6, 5, device='cpu')
	assert_raises(ValueError, ablate, model, X, 7, 5, device='cpu')
	assert_raises(ValueError, ablate, model, X, 7, 5000, device='cpu')
	assert_raises(ValueError, ablate, model, X, 5, 10, args=alpha, device='cpu')
	assert_raises(ValueError, ablate, model, X, 5, 10, args=(alpha[:5],), 
		device='cpu')
	assert_raises(ValueError, ablate, model, X, 5, 10, args=(alpha, beta[:5]), 
		device='cpu')


###


def test_ablate_deep_lift_shap(X):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y_before, y_after = ablate(model, X[:2], 45, 55, func=deep_lift_shap, 
		batch_size=1, n=2, n_shuffles=3, device='cpu', random_state=0)

	assert y_before.shape == (2, 4, 100)
	assert y_before.dtype == torch.float32

	assert_array_almost_equal(y_before[:, :, 48:52], [
		[[ 0.0000e+00,  0.0000e+00, -4.5916e-05, -0.0000e+00],
         [ 0.0000e+00, -0.0000e+00,  0.0000e+00,  1.7308e-03],
         [ 3.9663e-04,  0.0000e+00,  0.0000e+00,  0.0000e+00],
         [ 0.0000e+00,  3.3410e-04,  0.0000e+00,  0.0000e+00]],

        [[ 0.0000e+00,  0.0000e+00, -1.6943e-03, -0.0000e+00],
         [-0.0000e+00, -0.0000e+00, -0.0000e+00, -1.0988e-03],
         [ 7.5694e-04, -0.0000e+00,  0.0000e+00,  0.0000e+00],
         [-0.0000e+00, -1.0072e-04, -0.0000e+00,  0.0000e+00]]], 4)

	assert y_after.shape == (2, 2, 4, 100)
	assert y_after.dtype == torch.float32

	assert_array_almost_equal(y_after[:, :, :, 48:52], [
		[[[ 0.0000e+00,  1.3726e-05, -0.0000e+00, -0.0000e+00],
          [ 0.0000e+00, -0.0000e+00, -5.2354e-04,  0.0000e+00],
          [ 0.0000e+00, -0.0000e+00, -0.0000e+00, -7.4029e-04],
          [ 1.4501e-04, -0.0000e+00,  0.0000e+00,  0.0000e+00]],

         [[ 0.0000e+00, -0.0000e+00,  0.0000e+00,  0.0000e+00],
          [ 3.3442e-05, -0.0000e+00, -0.0000e+00, -3.9522e-03],
          [ 0.0000e+00,  0.0000e+00, -0.0000e+00, -0.0000e+00],
          [-0.0000e+00,  4.3980e-05,  8.9129e-04, -0.0000e+00]]],


        [[[-0.0000e+00,  1.9562e-04, -0.0000e+00,  0.0000e+00],
          [ 0.0000e+00, -0.0000e+00,  1.9479e-03,  0.0000e+00],
          [-7.3144e-04,  0.0000e+00, -0.0000e+00, -6.6028e-04],
          [-0.0000e+00, -0.0000e+00,  0.0000e+00,  0.0000e+00]],

         [[-0.0000e+00,  0.0000e+00,  0.0000e+00, -1.1882e-04],
          [-2.0654e-03, -0.0000e+00,  0.0000e+00, -0.0000e+00],
          [-0.0000e+00, -3.3294e-04,  0.0000e+00,  0.0000e+00],
          [-0.0000e+00, -0.0000e+00,  4.6053e-04,  0.0000e+00]]]], 4)

	y_before2, y_after2 = ablate(model, X[:8], 45, 55, func=deep_lift_shap, 
		batch_size=64, n=2, n_shuffles=3, device='cpu', random_state=0)
	assert_array_almost_equal(y_before, y_before2[:2], 4)
	assert_array_almost_equal(y_after, y_after2[:2], 4)


def test_ablate_deep_lift_shap_flattendense(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)
	y_before, y_after = ablate(model, X[:8], 45, 55, func=deep_lift_shap, 
		n=3, n_shuffles=5, device='cpu', random_state=0)

	assert y_before.shape == (8, 4, 100)
	assert y_before.dtype == torch.float32
	assert_array_almost_equal(y_before[:2, :, 48:52], [
		[[-0.0000, -0.0000,  0.0472,  0.0000],
         [-0.0000, -0.0000, -0.0000, -0.0628],
         [ 0.0212, -0.0000, -0.0000, -0.0000],
         [-0.0000,  0.0091, -0.0000,  0.0000]],

        [[-0.0000,  0.0000,  0.0490,  0.0000],
         [-0.0000, -0.0000, -0.0000, -0.0293],
         [ 0.0207,  0.0000, -0.0000,  0.0000],
         [-0.0000,  0.0229, -0.0000,  0.0000]]], 4)

	assert y_after.shape == (8, 3, 4, 100)
	assert y_after.dtype == torch.float32
	assert_array_almost_equal(y_after[:2, :2, :, 48:52], [
		[[[ 0.0000,  0.0081,  0.0000,  0.0000],
          [-0.0000, -0.0000, -0.0432, -0.0000],
          [ 0.0000,  0.0000, -0.0000,  0.0065],
          [-0.0068,  0.0000, -0.0000,  0.0000]],

         [[-0.0000,  0.0000,  0.0000,  0.0000],
          [-0.0049, -0.0000, -0.0000, -0.0693],
          [ 0.0000, -0.0000, -0.0000, -0.0000],
          [-0.0000,  0.0183, -0.0005,  0.0000]]],


        [[[ 0.0000,  0.0054,  0.0000,  0.0000],
          [ 0.0000, -0.0000, -0.0339, -0.0000],
          [ 0.0271, -0.0000, -0.0000, -0.0302],
          [-0.0000,  0.0000, -0.0000,  0.0000]],

         [[-0.0000, -0.0000,  0.0000,  0.0410],
          [-0.0210, -0.0000, -0.0000, -0.0000],
          [ 0.0000, -0.0118, -0.0000, -0.0000],
          [-0.0000,  0.0000, -0.0046,  0.0000]]]], 4)

	y_before2, y_after2 = ablate(model, X[:8], 45, 55, func=deep_lift_shap, 
		n=3, n_shuffles=5, batch_size=64, device='cpu', random_state=0)
	assert_array_almost_equal(y_before, y_before2)
	assert_array_almost_equal(y_after, y_after2)


def test_ablate_deep_lift_shap_vs_attribute(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)
	y_before, y_after = ablate(model, X[:8], 45, 55, func=deep_lift_shap, 
		n=3, n_shuffles=5, device='cpu', random_state=0)
	y_after = y_after.reshape(-1, *y_after.shape[2:])

	X1 = shuffle(X[:8], start=45, end=55, n=3, random_state=0).reshape(-1, 
		*X.shape[1:])

	y_before0 = deep_lift_shap(model, X[:8], n_shuffles=5, device='cpu', 
		random_state=0)
	y_after0 = deep_lift_shap(model, X1, n_shuffles=5, device='cpu', 
		random_state=0)

	assert y_before.shape == (8, 4, 100)
	assert y_before.dtype == torch.float32
	assert_array_almost_equal(y_before, y_before0, 4)

	assert y_after.shape == (24, 4, 100)
	assert y_after.dtype == torch.float32
	assert_array_almost_equal(y_after, y_after0, 4)


def test_ablate_deep_lift_shap_alpha(X, alpha):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	y_before0, y_after0 = ablate(model, X[:8], 45, 55, func=deep_lift_shap, 
		n=3, n_shuffles=5, device='cpu', random_state=0)
	y_before1, y_after1 = ablate(model, X[:8], 45, 55, func=deep_lift_shap, 
		n=3, n_shuffles=5, args=(alpha,), device='cpu', random_state=0)

	assert y_before0.shape == (8, 4, 100)
	assert y_before0.dtype == torch.float32
	assert_array_almost_equal(y_before0, y_before1, 4)

	assert y_after0.shape == (8, 3, 4, 100)
	assert y_after0.dtype == torch.float32
	assert_array_almost_equal(y_after0, y_after1, 4)


def test_ablate_deep_lift_shap_alpha_beta(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	y_before0, y_after0 = ablate(model, X[:8], 45, 55, func=deep_lift_shap, 
		n=3, n_shuffles=5, device='cpu', random_state=0)
	y_before1, y_after1 = ablate(model, X[:8], 45, 55, func=deep_lift_shap, 
		n=3, n_shuffles=5, args=(alpha, beta), device='cpu', random_state=0)

	assert y_before0.shape == (8, 4, 100)
	assert y_before0.dtype == torch.float32
	assert_raises(AssertionError, assert_array_almost_equal, y_before0, 
		y_before1, 4)

	assert y_after0.shape == (8, 3, 4, 100)
	assert y_after0.dtype == torch.float32
	assert_raises(AssertionError, assert_array_almost_equal, y_after0, 
		y_after1, 4)


def test_ablate_deep_lift_shap_n_shuffles(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	y_before0, y_after0 = ablate(model, X[:8], 45, 55, func=deep_lift_shap, 
		n_shuffles=2, device='cpu', random_state=0)
	y_before1, y_after1 = ablate(model, X[:8], 45, 55, func=deep_lift_shap, 
		n_shuffles=10, device='cpu', random_state=0)

	assert y_before0.shape == (8, 4, 100)
	assert y_before0.dtype == torch.float32
	assert_array_almost_equal(y_before0[:2, :, :3], [
		[[ 0.0000, -0.0000, -0.0000],
         [ 0.0000,  0.0000,  0.0160],
         [ 0.0000, -0.0000,  0.0000],
         [-0.0000, -0.0247,  0.0000]],

        [[-0.0000,  0.0000, -0.0076],
         [ 0.0000,  0.0000,  0.0000],
         [-0.0000,  0.0116, -0.0000],
         [-0.0000, -0.0000,  0.0000]]], 4)

	assert y_before1.shape == (8, 4, 100)
	assert y_before1.dtype == torch.float32
	assert_array_almost_equal(y_before1[:2, :, :3], [
		[[ 0.0000, -0.0000, -0.0000],
         [ 0.0000,  0.0000,  0.0115],
         [ 0.0000,  0.0000, -0.0000],
         [-0.0000, -0.0209,  0.0000]],

        [[-0.0000,  0.0000, -0.0052],
         [ 0.0000,  0.0000,  0.0000],
         [-0.0000,  0.0079, -0.0000],
         [-0.0000, -0.0000,  0.0000]]], 4)

	assert y_after0.shape == (8, 20, 4, 100)
	assert y_after0.dtype == torch.float32
	assert_array_almost_equal(y_after0[:2, :2, :, :3], [
		[[[ 0.0000, -0.0000, -0.0000],
          [ 0.0000,  0.0000,  0.0177],
          [ 0.0000,  0.0000,  0.0000],
          [-0.0000, -0.0213,  0.0000]],

         [[ 0.0000, -0.0000, -0.0000],
          [ 0.0000,  0.0000,  0.0160],
          [ 0.0000, -0.0000,  0.0000],
          [-0.0000, -0.0247,  0.0000]]],


        [[[-0.0000,  0.0000, -0.0076],
          [ 0.0000,  0.0000,  0.0000],
          [-0.0000,  0.0116, -0.0000],
          [-0.0000, -0.0000,  0.0000]],

         [[-0.0000,  0.0000, -0.0076],
          [ 0.0000,  0.0000,  0.0000],
          [-0.0000,  0.0116, -0.0000],
          [-0.0000, -0.0000,  0.0000]]]], 4)

	assert y_after1.shape == (8, 20, 4, 100)
	assert y_after1.dtype == torch.float32
	assert_array_almost_equal(y_after1[:2, :2, :, :3], [
		[[[ 0.0000, -0.0000, -0.0000],
          [ 0.0000,  0.0000,  0.0119],
          [ 0.0000,  0.0000, -0.0000],
          [-0.0000, -0.0202,  0.0000]],

         [[ 0.0000, -0.0000, -0.0000],
          [ 0.0000,  0.0000,  0.0099],
          [ 0.0000,  0.0000, -0.0000],
          [-0.0000, -0.0209,  0.0000]]],


        [[[-0.0000,  0.0000, -0.0052],
          [ 0.0000,  0.0000,  0.0000],
          [-0.0000,  0.0079, -0.0000],
          [-0.0000, -0.0000,  0.0000]],

         [[-0.0000,  0.0000, -0.0052],
          [ 0.0000,  0.0000,  0.0000],
          [-0.0000,  0.0079, -0.0000],
          [-0.0000, -0.0000,  0.0000]]]], 4)


def test_ablate_deep_lift_shap_hypothetical(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	y_before, y_after = ablate(model, X[:8], 44, 55, func=deep_lift_shap, n=3,
		n_shuffles=5, hypothetical=True, device='cpu', random_state=0)

	assert y_before.shape == (8, 4, 100)
	assert y_before.dtype == torch.float32
	assert_array_almost_equal(y_before[:2, :, :4], [
		[[ 0.0000, -0.0069, -0.0089, -0.0435],
         [ 0.0474,  0.0000,  0.0104, -0.0193],
         [ 0.0025, -0.0015, -0.0056, -0.0410],
         [-0.0237, -0.0247,  0.0063,  0.0048]],

        [[-0.0474,  0.0036, -0.0037, -0.0058],
         [ 0.0000,  0.0105,  0.0156,  0.0184],
         [-0.0449,  0.0090, -0.0004, -0.0034],
         [-0.0712, -0.0142,  0.0115,  0.0425]]], 4)

	assert y_after.shape == (8, 3, 4, 100)
	assert y_after.dtype == torch.float32
	assert_array_almost_equal(y_after[:2, :2, :, :4], [
		[[[ 0.0000, -0.0069, -0.0089, -0.0435],
          [ 0.0474,  0.0000,  0.0104, -0.0193],
          [ 0.0025, -0.0015, -0.0056, -0.0410],
          [-0.0237, -0.0247,  0.0063,  0.0048]],

         [[ 0.0000, -0.0069, -0.0082, -0.0343],
          [ 0.0474,  0.0000,  0.0111, -0.0101],
          [ 0.0025, -0.0015, -0.0049, -0.0319],
          [-0.0237, -0.0247,  0.0069,  0.0140]]],


        [[[-0.0474,  0.0036, -0.0037, -0.0063],
          [ 0.0000,  0.0105,  0.0156,  0.0179],
          [-0.0449,  0.0090, -0.0004, -0.0039],
          [-0.0712, -0.0142,  0.0115,  0.0420]],

         [[-0.0474,  0.0036, -0.0037, -0.0058],
          [ 0.0000,  0.0105,  0.0156,  0.0184],
          [-0.0449,  0.0090, -0.0004, -0.0034],
          [-0.0712, -0.0142,  0.0115,  0.0425]]]], 4)

	y_before1, y_after1 = ablate(model, X[:8], 44, 55, func=deep_lift_shap, 
		n=3, n_shuffles=5, additional_func_kwargs={'hypothetical': True}, 
		device='cpu', random_state=0)

	assert_array_almost_equal(y_before, y_before1, 4)
	assert_array_almost_equal(y_after, y_after1, 4)


def test_ablate_deep_lift_shap_raises(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	assert_raises(TypeError, ablate, model, X, 44, 55, n=3, func=deep_lift_shap, 
		device='cpu', additional_func_kwargs={'device': 'cpu'}, n_shuffles=5)
	assert_raises(TypeError, ablate, model, X, 44, 55, n=3, func=deep_lift_shap,
		device='cpu', end=10)
