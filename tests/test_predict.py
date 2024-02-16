# test_predict.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest

from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot

from tangermeme.predict import predict
from tangermeme.predict import predict_cross

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


def test_predict_summodel(X):
	torch.manual_seed(0)
	model = SumModel()
	y = predict(model, X, batch_size=8, device='cpu')

	assert y.shape == (64, 4)
	assert y.dtype == torch.float32
	assert y.sum() == X.sum()
	assert_array_almost_equal(y[:8], [
		[25., 24., 19., 32.],
        [26., 18., 29., 27.],
        [28., 25., 21., 26.],
        [21., 33., 19., 27.],
        [27., 22., 23., 28.],
        [31., 27., 21., 21.],
        [26., 20., 21., 33.],
        [21., 28., 31., 20.]])

	assert_array_almost_equal(y, model(X))
	assert_array_almost_equal(y, predict(model, X, batch_size=1, device='cpu'))
	assert_array_almost_equal(y, predict(model, X, batch_size=64, device='cpu'))


def test_predict_flattendense(X):
	torch.manual_seed(0)
	model = FlattenDense()
	y = predict(model, X, batch_size=8, device='cpu')

	assert y.shape == (64, 3)
	assert y.dtype == torch.float32

	assert_array_almost_equal(y[:8], [
		[ 0.1928,  0.2788, -0.1000],
        [-0.0505,  0.0174, -0.1751],
        [-0.1408,  0.0105,  0.0673],
        [-0.2375, -0.0069,  0.0835],
        [ 0.0442, -0.0692, -0.1565],
        [-0.3489, -0.0876, -0.2994],
        [-0.3894, -0.1332, -0.3717],
        [-0.3682,  0.5173, -0.4089]], 4)

	assert_array_almost_equal(y, model(X).detach())	
	assert_array_almost_equal(y, predict(model, X, batch_size=1, device='cpu'))
	assert_array_almost_equal(y, predict(model, X, batch_size=64, device='cpu'))


def test_predict_conv(X):
	torch.manual_seed(0)
	model = Conv()
	y = predict(model, X, batch_size=8, device='cpu')

	assert y.shape == (64, 12, 98)
	assert y.dtype == torch.float32

	assert_array_almost_equal(y[:2, :4, :10], [
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

	assert_array_almost_equal(y, model(X).detach())	
	assert_array_almost_equal(y, predict(model, X, batch_size=1, device='cpu'))
	assert_array_almost_equal(y, predict(model, X, batch_size=64, device='cpu'))


def test_predict_scatter(X):
	torch.manual_seed(0)
	model = Scatter()
	y = predict(model, X, batch_size=8, device='cpu')

	assert y.shape == (64, 100, 4)
	assert y.dtype == torch.float32
	assert y.sum() == X.sum()
	assert (y.sum(axis=-1) == X.sum(axis=1)).all()
	assert (y.sum(axis=1) == X.sum(axis=-1)).all()

	assert_array_almost_equal(y[:4, :5], [
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

	assert_array_almost_equal(y, model(X).detach())	
	assert_array_almost_equal(y, predict(model, X, batch_size=1, device='cpu'))
	assert_array_almost_equal(y, predict(model, X, batch_size=64, device='cpu'))


def test_predict_convdense(X):
	torch.manual_seed(0)
	model = ConvDense()
	y = predict(model, X, batch_size=2, device='cpu')

	assert len(y) == 2
	assert y[0].shape == (64, 12, 98)
	assert y[1].shape == (64, 3)

	assert y[0].dtype == torch.float32
	assert y[1].dtype == torch.float32

	assert_array_almost_equal(y[0][:2, :4, :4], [
		[[-0.7757,  0.0397, -0.8799, -1.0194],
         [ 0.0947,  0.2042, -0.2302, -0.3144],
         [ 0.3781,  0.5009,  0.8585,  0.6738],
         [-0.5986, -0.0296, -0.0995, -0.2718]],

        [[-0.6676, -0.8873, -0.8696, -0.4684],
         [-0.0768, -0.0089,  0.0244,  0.1355],
         [ 0.4828,  0.5419,  0.3170,  0.3288],
         [-0.1026, -0.4676, -0.3080, -0.2606]]], 4)

	assert_array_almost_equal(y[1][:4], [
		[ 0.1928,  0.2788, -0.1000],
        [-0.0505,  0.0174, -0.1751],
        [-0.1408,  0.0105,  0.0673],
        [-0.2375, -0.0069,  0.0835]], 4)


def test_predict_convdense_batch_size(X):
	torch.manual_seed(0)
	model = ConvDense()
	y = predict(model, X, batch_size=64, device='cpu')

	assert len(y) == 2
	assert y[0].shape == (64, 12, 98)
	assert y[1].shape == (64, 3)

	assert y[0].dtype == torch.float32
	assert y[1].dtype == torch.float32


def test_predict_batch_size(X):
	torch.manual_seed(0)
	model = Scatter()
	y = predict(model, X, batch_size=68, device='cpu')
	assert y.shape == (64, 100, 4)


def test_predict_raises_shape(X):
	torch.manual_seed(0)
	model = Scatter()
	assert_raises(RuntimeError, predict, model, X[0], device='cpu')
	assert_raises(RuntimeError, predict, model, X[:, 0], device='cpu')
	assert_raises(RuntimeError, predict, model, X.unsqueeze(0), device='cpu')


def test_predict_raises_args(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense()
	assert_raises(TypeError, predict, model, X, batch_size=2, args=5, 
		device='cpu')
	assert_raises(TypeError, predict, model, X, batch_size=2, args=(5,), 
		device='cpu')
	assert_raises(TypeError, predict, model, X, batch_size=2, 
		args=alpha, device='cpu')
	assert_raises(RuntimeError, predict, model, X, batch_size=2, 
		args=(alpha[:5],), device='cpu')
	assert_raises(RuntimeError, predict, model, X, batch_size=2, 
		args=(alpha, beta[:5]), device='cpu')


###


def test_predict_cross_flattendense_alpha(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense()
	y0 = predict(model, X, device='cpu').unsqueeze(1)
	y = predict_cross(model, X, args=(alpha[:5],), device='cpu')
	assert y.shape == (64, 5, 3)
	assert y.dtype == torch.float32
	assert_array_almost_equal(y, y0 + alpha[:5][None, :])
	assert_array_almost_equal(y[:2], [
		[[1.9569, 2.0429, 1.6640],
         [0.5930, 0.6790, 0.3001],
         [1.1715, 1.2575, 0.8787],
         [2.4337, 2.5197, 2.1409],
         [2.0604, 2.1464, 1.7675]],

        [[1.7136, 1.7814, 1.5889],
         [0.3497, 0.4175, 0.2250],
         [0.9283, 0.9961, 0.8036],
         [2.1904, 2.2583, 2.0658],
         [1.8171, 1.8849, 1.6924]]], 4)


	y = predict_cross(model, X, args=(alpha[5:10],), device='cpu')
	assert y.shape == (64, 5, 3)
	assert y.dtype == torch.float32
	assert_array_almost_equal(y, y0 + alpha[5:10][None, :])
	assert_array_almost_equal(y[:2], [
		[[-0.7845, -0.6985, -1.0773],
         [ 1.1429,  1.2289,  0.8501],
         [ 0.0414,  0.1274, -0.2514],
         [ 0.0896,  0.1756, -0.2032],
         [ 0.6034,  0.6894,  0.3106]],

        [[-1.0278, -0.9599, -1.1524],
         [ 0.8996,  0.9674,  0.7750],
         [-0.2018, -0.1340, -0.3265],
         [-0.1537, -0.0859, -0.2783],
         [ 0.3601,  0.4280,  0.2355]]], 4)


def test_predict_cross_flattendense_beta(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense()
	alpha = torch.zeros(X.shape[0], 1)

	y0 = predict(model, X, device='cpu').unsqueeze(1)
	y = predict_cross(model, X, args=(alpha[:5], beta[:5]), device='cpu')
	assert y.shape == (64, 5, 3)
	assert y.dtype == torch.float32
	assert_array_almost_equal(y, y0 * beta[:5][None, :])
	assert_array_almost_equal(y[:2], [
		[[ 0.3132,  0.4529, -0.1624],
         [-0.1179, -0.1706,  0.0612],
         [-0.1018, -0.1473,  0.0528],
         [-0.2069, -0.2991,  0.1073],
         [ 0.1669,  0.2413, -0.0865]],

        [[-0.0820,  0.0282, -0.2845],
         [ 0.0309, -0.0106,  0.1071],
         [ 0.0267, -0.0092,  0.0925],
         [ 0.0542, -0.0186,  0.1879],
         [-0.0437,  0.0150, -0.1516]]], 4)


	y = predict_cross(model, X, args=(alpha[5:10], beta[5:10]), device='cpu')
	assert y.shape == (64, 5, 3)
	assert y.dtype == torch.float32
	assert_array_almost_equal(y, y0 * beta[5:10][None, :])
	assert_array_almost_equal(y[:2], [
		[[-0.4437, -0.6417,  0.2302],
         [ 0.3364,  0.4865, -0.1745],
         [-0.1468, -0.2122,  0.0761],
         [ 0.0615,  0.0889, -0.0319],
         [-0.0481, -0.0695,  0.0249]],

        [[ 0.1162, -0.0400,  0.4030],
         [-0.0881,  0.0303, -0.3056],
         [ 0.0384, -0.0132,  0.1333],
         [-0.0161,  0.0055, -0.0559],
         [ 0.0126, -0.0043,  0.0437]]], 4)


def test_predict_cross_flattendense_alpha_beta(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense()
	y0 = predict(model, X, device='cpu').unsqueeze(1)
	y = predict_cross(model, X, args=(alpha[:5], beta[:5]), device='cpu')
	assert y.shape == (64, 5, 3)
	assert y.dtype == torch.float32
	assert_array_almost_equal(y, y0 * beta[None, :5] + alpha[None, :5])
	assert_array_almost_equal(y[:2], [
		[[2.0772, 2.2169, 1.6016],
         [0.2822, 0.2296, 0.4613],
         [0.8769, 0.8315, 1.0316],
         [2.0340, 1.9417, 2.3482],
         [2.0344, 2.1088, 1.7810]],

        [[1.6821, 1.7923, 1.4796],
         [0.4310, 0.3895, 0.5073],
         [1.0054, 0.9696, 1.0712],
         [2.2951, 2.2223, 2.4288],
         [1.8239, 1.8826, 1.7160]]], 4)


	y = predict_cross(model, X, args=(alpha[5:10], beta[5:10]), device='cpu')
	assert y.shape == (64, 5, 3)
	assert y.dtype == torch.float32
	assert_array_almost_equal(y, y0 * beta[None, 5:10] + alpha[None, 5:10])
	assert_array_almost_equal(y[:2], [
		[[-1.4210, -1.6190, -0.7471],
         [ 1.2865,  1.4365,  0.7756],
         [-0.2981, -0.3636, -0.0752],
         [-0.0417, -0.0143, -0.1351],
         [ 0.3625,  0.3411,  0.4355]],

        [[-0.8611, -1.0172, -0.5742],
         [ 0.8620,  0.9804,  0.6445],
         [-0.1129, -0.1646, -0.0181],
         [-0.1193, -0.0977, -0.1591],
         [ 0.4232,  0.4063,  0.4543]]], 4)


def test_predict_cross_convdense_alpha(X, alpha):
	torch.manual_seed(0)
	model = ConvDense()

	y = predict_cross(model, X, args=(alpha[:5, :, None],), 
		batch_size=2, device='cpu')

	assert len(y) == 2
	assert y[0].shape == (64, 5, 12, 98)
	assert y[1].shape == (64, 5, 3)

	assert y[0].dtype == torch.float32
	assert y[1].dtype == torch.float32

	assert_array_almost_equal(y[0][:2, :2, :3, :4], [
		[[[ 0.9883,  1.8037,  0.8841,  0.7446],
          [ 1.8588,  1.9683,  1.5339,  1.4497],
          [ 2.1421,  2.2650,  2.6226,  2.4379]],

         [[-0.3756,  0.4398, -0.4798, -0.6192],
          [ 0.4949,  0.6044,  0.1700,  0.0858],
          [ 0.7782,  0.9011,  1.2587,  1.0740]]],


        [[[ 1.0964,  0.8767,  0.8945,  1.2957],
          [ 1.6873,  1.7551,  1.7884,  1.8996],
          [ 2.2469,  2.3060,  2.0810,  2.0929]],

         [[-0.2675, -0.4872, -0.4694, -0.0682],
          [ 0.3234,  0.3912,  0.4245,  0.5357],
          [ 0.8830,  0.9421,  0.7171,  0.7290]]]], 4)

	assert_array_almost_equal(y[1][:4], [
		[[ 0.1928,  0.2788, -0.1000],
         [ 0.1928,  0.2788, -0.1000],
         [ 0.1928,  0.2788, -0.1000],
         [ 0.1928,  0.2788, -0.1000],
         [ 0.1928,  0.2788, -0.1000]],

        [[-0.0505,  0.0174, -0.1751],
         [-0.0505,  0.0174, -0.1751],
         [-0.0505,  0.0174, -0.1751],
         [-0.0505,  0.0174, -0.1751],
         [-0.0505,  0.0174, -0.1751]],

        [[-0.1408,  0.0105,  0.0673],
         [-0.1408,  0.0105,  0.0673],
         [-0.1408,  0.0105,  0.0673],
         [-0.1408,  0.0105,  0.0673],
         [-0.1408,  0.0105,  0.0673]],

        [[-0.2375, -0.0069,  0.0835],
         [-0.2375, -0.0069,  0.0835],
         [-0.2375, -0.0069,  0.0835],
         [-0.2375, -0.0069,  0.0835],
         [-0.2375, -0.0069,  0.0835]]], 4)
