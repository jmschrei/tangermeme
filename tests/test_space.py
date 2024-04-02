# test_space.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest

from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot

from tangermeme.space import space

from .toy_models import SumModel
from .toy_models import FlattenDense
from .toy_models import Conv
from .toy_models import Scatter
from .toy_models import ConvDense

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


torch.manual_seed(0)
torch.use_deterministic_algorithms(True)


@pytest.fixture
def X():
	return random_one_hot((64, 4, 100), random_state=0).type(torch.float32)


@pytest.fixture
def alpha():
	r = numpy.random.RandomState(0)
	return torch.from_numpy(r.randn(64, 1)).type(torch.float32)

@pytest.fixture
def beta():
	r = numpy.random.RandomState(1)
	return torch.from_numpy(r.randn(64, 1)).type(torch.float32)


##

def test_space_summodel(X):
	torch.manual_seed(0)
	model = SumModel()
	y_before, y_after = space(model, X, ["ACGTC", "GAGA"], 1, 
		batch_size=5, device='cpu')

	assert y_before.shape == (64, 4)
	assert y_before.dtype == torch.float32
	assert y_before.sum() == X.sum()
	assert_array_almost_equal(y_before[:8], [
		[25., 24., 19., 32.],
        [26., 18., 29., 27.],
        [28., 25., 21., 26.],
        [21., 33., 19., 27.],
        [27., 22., 23., 28.],
        [31., 27., 21., 21.],
        [26., 20., 21., 33.],
        [21., 28., 31., 20.]])

	assert y_after.shape == (64, 4)
	assert y_after.dtype == torch.float32
	assert y_after.sum() == X.sum()
	assert_array_almost_equal(y_after[:8], [
		[28., 22., 21., 29.],
        [28., 18., 29., 25.],
        [28., 24., 23., 25.],
        [21., 34., 19., 26.],
        [27., 22., 25., 26.],
        [28., 29., 22., 21.],
        [28., 17., 23., 32.],
        [22., 28., 32., 18.]])

	y_before2, y_after2 = space(model, X, ["ACGTC", "GAGA"], 1, 
		batch_size=64, device='cpu')
	assert_array_almost_equal(y_before, y_before2)
	assert_array_almost_equal(y_after, y_after2)	


def test_space_flattendense(X):
	torch.manual_seed(0)
	model = FlattenDense()
	y_before, y_after = space(model, X, ["ACGTC", "GAGA"], 1, 
		batch_size=5, device='cpu')

	assert y_before.shape == (64, 3)
	assert y_before.dtype == torch.float32
	assert_array_almost_equal(y_before[:8], [
		[ 0.1928,  0.2788, -0.1000],
        [-0.0505,  0.0174, -0.1751],
        [-0.1408,  0.0105,  0.0673],
        [-0.2375, -0.0069,  0.0835],
        [ 0.0442, -0.0692, -0.1565],
        [-0.3489, -0.0876, -0.2994],
        [-0.3894, -0.1332, -0.3717],
        [-0.3682,  0.5173, -0.4089]], 4)

	assert y_after.shape == (64, 3)
	assert y_after.dtype == torch.float32
	assert_array_almost_equal(y_after[:8], [
		[ 0.1087,  0.3662,  0.0668],
        [-0.1331,  0.0071, -0.0907],
        [-0.3239,  0.0593,  0.1154],
        [-0.3762,  0.0659,  0.0011],
        [-0.1361, -0.0605, -0.0853],
        [-0.4873, -0.1080, -0.3462],
        [-0.3656, -0.0071, -0.1459],
        [-0.4935,  0.6714, -0.3008]], 4)

	y_before2, y_after2 = space(model, X, ["ACGTC", "GAGA"], 1, 
		batch_size=64, device='cpu')
	assert_array_almost_equal(y_before, y_before2)
	assert_array_almost_equal(y_after, y_after2)


def test_space_conv(X):
	torch.manual_seed(0)
	model = Conv()
	y_before, y_after = space(model, X, ["ACGTC", "GAGA"], 1, 
		start=0, batch_size=5, device='cpu')

	assert y_before.shape == (64, 12, 98)
	assert y_before.dtype == torch.float32
	assert_array_almost_equal(y_before[:2, :4, :10], [
		[[-0.2713, -0.5317, -0.3737, -0.4055, -0.3269, -0.3269, -0.1928,
          -0.3509, -0.4816, -0.3197],
         [-0.2988,  0.0980, -0.1013, -0.2560,  0.2595,  0.2595,  0.2167,
           0.4330, -0.0124,  0.3218],
         [-0.1247,  0.1433, -0.3111, -0.3220, -0.4084, -0.4084, -0.2111,
          -0.3884, -0.3460, -0.3052],
         [-0.1362, -0.0067,  0.5554,  0.1810,  0.2007,  0.2007, -0.1166,
           0.0337,  0.1722, -0.3441]],

        [[-0.4805, -0.1669, -0.5863, -0.0537, -0.2702, -0.1669, -0.4055,
          -0.5078, -0.0848, -0.5863],
         [-0.3709, -0.3077, -0.5910,  0.0165, -0.6573, -0.3077, -0.2560,
          -0.0755,  0.1277, -0.5910],
         [ 0.4396, -0.1558,  0.2097, -0.0929,  0.6608, -0.1558, -0.3220,
           0.1234, -0.1761,  0.2097],
         [-0.0027,  0.4644,  0.1406, -0.1111, -0.3111,  0.4644,  0.1810,
           0.1603,  0.2667,  0.1406]]], 4)

	assert y_after.shape == (64, 12, 98)
	assert y_after.dtype == torch.float32
	assert_array_almost_equal(y_after[:2, :4, :10], [
		[[-0.3983, -0.2996, -0.2749, -0.3509, -0.5846, -0.1916, -0.1358,
          -0.2702, -0.0328, -0.3983],
         [-0.1937, -0.0358, -0.2188,  0.4330, -0.0807, -0.1418, -0.4189,
          -0.6573, -0.3505, -0.1937],
         [-0.2188, -0.0922, -0.1907, -0.3884, -0.4600,  0.5745, -0.0725,
           0.6608,  0.0415, -0.2188],
         [-0.3638,  0.0378,  0.0811,  0.0337,  0.1116, -0.2914,  0.0865,
          -0.3111,  0.1471, -0.3638]],

        [[-0.3983, -0.2996, -0.2749, -0.3197, -0.2685, -0.2737, -0.1358,
          -0.2702, -0.1669, -0.5863],
         [-0.1937, -0.0358, -0.2188,  0.3218, -0.1470, -0.5773, -0.4189,
          -0.6573, -0.3077, -0.5910],
         [-0.2188, -0.0922, -0.1907, -0.3052, -0.0089,  0.5948, -0.0725,
           0.6608, -0.1558,  0.2097],
         [-0.3638,  0.0378,  0.0811, -0.3441, -0.3401, -0.0938,  0.0865,
          -0.3111,  0.4644,  0.1406]]], 4)

	y_before2, y_after2 = space(model, X, ["ACGTC", "GAGA"], 1, 
		start=0, batch_size=64, device='cpu')
	assert_array_almost_equal(y_before, y_before2)
	assert_array_almost_equal(y_after, y_after2)


def test_space_scatter(X):
	torch.manual_seed(0)
	model = Scatter()
	y_before, y_after = space(model, X, ["ACGTC", "GAGA"], 1, 
		start=0, batch_size=8, device='cpu')

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

	assert y_after.shape == (64, 100, 4)
	assert y_after.dtype == torch.float32
	assert y_after.sum() == X.sum()
	assert_array_almost_equal(y_after[:4, :5], [
		[[1., 0., 0., 0.],
         [0., 1., 0., 0.],
         [0., 0., 1., 0.],
         [0., 0., 0., 1.],
         [0., 1., 0., 0.]],

        [[1., 0., 0., 0.],
         [0., 1., 0., 0.],
         [0., 0., 1., 0.],
         [0., 0., 0., 1.],
         [0., 1., 0., 0.]],

        [[1., 0., 0., 0.],
         [0., 1., 0., 0.],
         [0., 0., 1., 0.],
         [0., 0., 0., 1.],
         [0., 1., 0., 0.]],

        [[1., 0., 0., 0.],
         [0., 1., 0., 0.],
         [0., 0., 1., 0.],
         [0., 0., 0., 1.],
         [0., 1., 0., 0.]]], 4)

	y_before2, y_after2 = space(model, X, ["ACGTC", "GAGA"], 1, 
		start=0, batch_size=64, device='cpu')
	assert_array_almost_equal(y_before, y_before2)
	assert_array_almost_equal(y_after, y_after2)


def test_space_convdense(X):
	torch.manual_seed(0)
	model = ConvDense()
	y_before, y_after = space(model, X, ["ACGTC", "GAGA"], 1, 
		start=0, batch_size=2, device='cpu')

	assert len(y_before) == 2
	assert y_before[0].shape == (64, 12, 98)
	assert y_before[1].shape == (64, 3)

	assert y_before[0].dtype == torch.float32
	assert y_before[1].dtype == torch.float32

	assert_array_almost_equal(y_before[0][:2, :4, :4], [
		[[-0.7757,  0.0397, -0.8799, -1.0194],
         [ 0.0947,  0.2042, -0.2302, -0.3144],
         [ 0.3781,  0.5009,  0.8585,  0.6738],
         [-0.5986, -0.0296, -0.0995, -0.2718]],

        [[-0.6676, -0.8873, -0.8696, -0.4684],
         [-0.0768, -0.0089,  0.0244,  0.1355],
         [ 0.4828,  0.5419,  0.3170,  0.3288],
         [-0.1026, -0.4676, -0.3080, -0.2606]]], 4)

	assert_array_almost_equal(y_before[1][:4], [
		[ 0.1928,  0.2788, -0.1000],
        [-0.0505,  0.0174, -0.1751],
        [-0.1408,  0.0105,  0.0673],
        [-0.2375, -0.0069,  0.0835]], 4)

	assert len(y_after) == 2
	assert y_after[0].shape == (64, 12, 98)
	assert y_after[1].shape == (64, 3)

	assert y_after[0].dtype == torch.float32
	assert y_after[1].dtype == torch.float32

	assert_array_almost_equal(y_after[0][:2, :4, :4], [
		[[-3.8092e-01, -8.1744e-01, -7.0452e-01, -1.1013e-01],
         [-1.7080e-01, -4.1553e-01,  4.2124e-01, -1.3453e-01],
         [ 4.8191e-01,  8.3968e-01,  1.5083e-01,  8.5779e-01],
         [-7.7912e-01, -6.6461e-02, -6.1201e-01,  6.5689e-03]],

        [[-3.8092e-01, -8.1744e-01, -7.0452e-01, -4.2652e-04],
         [-1.7080e-01, -4.1553e-01,  4.2124e-01,  8.2816e-02],
         [ 4.8191e-01,  8.3968e-01,  1.5083e-01,  4.4968e-01],
         [-7.7912e-01, -6.6461e-02, -6.1201e-01, -2.8948e-01]]], 4)

	assert_array_almost_equal(y_after[1][:4], [
		[ 0.2248,  0.2780,  0.0710],
        [-0.1218,  0.1338, -0.2101],
        [-0.1292,  0.1851,  0.2033],
        [-0.2042,  0.1904,  0.2121]], 4)


def test_space_convdense_batch_size(X):
	torch.manual_seed(0)
	model = ConvDense()
	y_before, y_after = space(model, X, ["ACGTC", "GAGA"], 1, 
		batch_size=68, device='cpu')

	assert len(y_before) == 2
	assert y_before[0].shape == (64, 12, 98)
	assert y_before[1].shape == (64, 3)

	assert len(y_after) == 2
	assert y_after[0].shape == (64, 12, 98)
	assert y_after[1].shape == (64, 3)


def test_space_scatter_batch_size(X):
	torch.manual_seed(0)
	model = Scatter()
	y_before, y_after = space(model, X, ["ACGTC", "GAGA"], 1, 
		batch_size=68, device='cpu')

	assert y_before.shape == (64, 100, 4)
	assert y_after.shape == (64, 100, 4)


def test_space_raises_shape(X):
	torch.manual_seed(0)
	model = Scatter()
	assert_raises(ValueError, space, model, X[0], ["ACGTC", "ACC"], 0, 
		device='cpu')
	assert_raises(ValueError, space, model, X[:, 0], ["ACGTC", "ACC"], 0,
		device='cpu')
	assert_raises(ValueError, space, model, X.unsqueeze(0), ["ACGTC", "ACC"], 
		0, device='cpu')


def test_space_raises_args(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense()
	assert_raises(ValueError, space, model, X, ["ACGTC", "ACC"], [0, 1], 
		batch_size=2, device='cpu')
	assert_raises(ValueError, space, model, X, ["ACGTC", "ACC"], [-5], 
		batch_size=2, device='cpu')
	assert_raises(ValueError, space, model, X, ["ACGTC", "ACC"], -3, 
		batch_size=2, device='cpu')
	assert_raises(ValueError, space, model, X, ["ACGTC", "ACC"], [], 
		batch_size=2, device='cpu')
	assert_raises(ValueError, space, model, X, ["ACGTC", "ACC"], [1500], 
		batch_size=2, device='cpu')

	assert_raises(TypeError, space, model, X, ["ACGTC", "ACC"], 0, 
		batch_size=2, args=5, device='cpu')
	assert_raises(AttributeError, space, model, X, ["ACGTC", "ACC"], 0, 
		batch_size=2, args=(5,), device='cpu')
	assert_raises(ValueError, space, model, X, ["ACGTC", "ACC"], 0, 
		batch_size=2, args=alpha, device='cpu')
	assert_raises(ValueError, space, model, X, ["ACGTC", "ACC"], 0,
		batch_size=2, args=(alpha[:5],), device='cpu')
	assert_raises(ValueError, space, model, X, ["ACGTC", "ACC"], 0, 
		batch_size=2, args=(alpha, beta[:5]), device='cpu')
