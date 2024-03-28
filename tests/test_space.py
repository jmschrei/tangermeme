# test_marginalize.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest

from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot

from tangermeme.space import space
from tangermeme.space import space_cross

from tangermeme.marginalize import marginalize
from tangermeme.marginalize import marginalize_cross

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
	assert_raises(TypeError, space, model, X, ["ACGTC", "ACC"], 0, 
		batch_size=2, args=(5,), device='cpu')
	assert_raises(TypeError, space, model, X, ["ACGTC", "ACC"], 0, 
		batch_size=2, args=alpha, device='cpu')
	assert_raises(RuntimeError, space, model, X, ["ACGTC", "ACC"], 0,
		batch_size=2, args=(alpha[:5],), device='cpu')
	assert_raises(RuntimeError, space, model, X, ["ACGTC", "ACC"], 0, 
		batch_size=2, args=(alpha, beta[:5]), device='cpu')


###


def test_space_cross_flattendense_alpha(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense()
	y0_before, y0_after = space(model, X, ['ACGTGC', 'TGTT'], 3, 
		device='cpu')
	y_before, y_after = space_cross(model, X, ['ACGTGC', 'TGTT'], 3, 
		(alpha[:5],), device='cpu')

	assert y_before.shape == (64, 5, 3)
	assert y_before.dtype == torch.float32
	assert_array_almost_equal(y_before, y0_before[:, None] + alpha[:5][None, :])
	assert_array_almost_equal(y_before[:2], [
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


	assert y_after.shape == (64, 5, 3)
	assert y_after.dtype == torch.float32
	assert_array_almost_equal(y_after, y0_after[:, None] + alpha[:5][None, :])
	assert_array_almost_equal(y_after[:2], [
		[[1.9143, 1.9713, 1.6641],
         [0.5504, 0.6074, 0.3002],
         [1.1290, 1.1860, 0.8788],
         [2.3912, 2.4481, 2.1409],
         [2.0178, 2.0748, 1.7676]],

        [[1.7369, 1.6794, 1.6200],
         [0.3730, 0.3156, 0.2561],
         [0.9516, 0.8941, 0.8347],
         [2.2137, 2.1563, 2.0968],
         [1.8404, 1.7830, 1.7235]]], 4)



def test_space_cross_flattendense_beta(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense()
	alpha = torch.zeros(X.shape[0], 1)

	y0_before, y0_after = space(model, X, ['ACGTGC', 'TGTT'], 3, 
		device='cpu')
	y_before, y_after = space_cross(model, X, ['ACGTGC', 'TGTT'], 3, 
		(alpha[:5], beta[:5]), device='cpu')

	assert y_before.shape == (64, 5, 3)
	assert y_before.dtype == torch.float32
	assert_array_almost_equal(y_before, y0_before[:, None] * beta[:5][None, :])
	assert_array_almost_equal(y_before[:2], [
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

	assert y_after.shape == (64, 5, 3)
	assert y_after.dtype == torch.float32
	assert_array_almost_equal(y_after, y0_after[:, None] * beta[:5][None, :])
	assert_array_almost_equal(y_after[:2], [
		[[ 0.2441,  0.3366, -0.1624],
         [-0.0919, -0.1268,  0.0612],
         [-0.0794, -0.1095,  0.0528],
         [-0.1613, -0.2224,  0.1073],
         [ 0.1301,  0.1793, -0.0865]],

        [[-0.0441, -0.1374, -0.2340],
         [ 0.0166,  0.0518,  0.0881],
         [ 0.0143,  0.0447,  0.0761],
         [ 0.0291,  0.0908,  0.1546],
         [-0.0235, -0.0732, -0.1247]]], 4)


def test_space_cross_flattendense_alpha_beta(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense()
	y0_before, y0_after = space(model, X, ['ACGTGC', 'TGTT'], 3, 
		device='cpu')
	y_before, y_after = space_cross(model, X, ['ACGTGC', 'TGTT'], 3, 
		(alpha[:5], beta[:5]), device='cpu')

	assert y_before.shape == (64, 5, 3)
	assert y_before.dtype == torch.float32
	assert_array_almost_equal(y_before, y0_before[:, None] * beta[:5][None, :]
		+ alpha[:5][None, :])
	assert_array_almost_equal(y_before[:2], [
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

	assert y_after.shape == (64, 5, 3)
	assert y_after.dtype == torch.float32
	assert_array_almost_equal(y_after, y0_after[:, None] * beta[:5][None, :]
		+ alpha[:5][None, :])
	assert_array_almost_equal(y_after[:2], [
		[[2.0082, 2.1007, 1.6017],
         [0.3082, 0.2734, 0.4613],
         [0.8994, 0.8693, 1.0315],
         [2.0796, 2.0185, 2.3482],
         [1.9976, 2.0469, 1.7810]],

        [[1.7199, 1.6266, 1.5301],
         [0.4168, 0.4519, 0.4883],
         [0.9931, 1.0234, 1.0548],
         [2.2700, 2.3317, 2.3955],
         [1.8441, 1.7943, 1.7429]]], 4)


def test_space_cross_convdense_alpha(X, alpha):
	torch.manual_seed(0)
	model = ConvDense()
	y_before, y_after = space_cross(model, X, ["ACGTGC", "TGGTT"], 3, 
		start=0, args=(alpha[:5, :, None],), batch_size=2, device='cpu')

	assert len(y_before) == 2
	assert y_before[0].shape == (64, 5, 12, 98)
	assert y_before[1].shape == (64, 5, 3)

	assert y_before[0].dtype == torch.float32
	assert y_before[1].dtype == torch.float32

	assert_array_almost_equal(y_before[0][:2, :2, :3, :4], [
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

	assert_array_almost_equal(y_before[1][:4], [
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

	assert len(y_after) == 2
	assert y_after[0].shape == (64, 5, 12, 98)
	assert y_after[1].shape == (64, 5, 3)

	assert y_after[0].dtype == torch.float32
	assert y_after[1].dtype == torch.float32

	assert_array_almost_equal(y_after[0][:2, :2, :3, :4], [
		[[[ 1.3831,  0.9466,  0.9256,  1.4921],
          [ 1.5933,  1.3485,  1.9935,  1.9060],
          [ 2.2460,  2.6037,  1.8025,  2.1864]],

         [[ 0.0192, -0.4173, -0.4383,  0.1282],
          [ 0.2294, -0.0154,  0.6296,  0.5421],
          [ 0.8821,  1.2398,  0.4386,  0.8225]]],


        [[[ 1.3831,  0.9466,  0.9256,  1.4921],
          [ 1.5933,  1.3485,  1.9935,  1.9060],
          [ 2.2460,  2.6037,  1.8025,  2.1864]],

         [[ 0.0192, -0.4173, -0.4383,  0.1282],
          [ 0.2294, -0.0154,  0.6296,  0.5421],
          [ 0.8821,  1.2398,  0.4386,  0.8225]]]], 4)

	assert_array_almost_equal(y_after[1][:4], [
		[[ 0.3324,  0.1149, -0.0712],
         [ 0.3324,  0.1149, -0.0712],
         [ 0.3324,  0.1149, -0.0712],
         [ 0.3324,  0.1149, -0.0712],
         [ 0.3324,  0.1149, -0.0712]],

        [[-0.0377, -0.0666, -0.1223],
         [-0.0377, -0.0666, -0.1223],
         [-0.0377, -0.0666, -0.1223],
         [-0.0377, -0.0666, -0.1223],
         [-0.0377, -0.0666, -0.1223]],

        [[-0.0393, -0.0238,  0.2462],
         [-0.0393, -0.0238,  0.2462],
         [-0.0393, -0.0238,  0.2462],
         [-0.0393, -0.0238,  0.2462],
         [-0.0393, -0.0238,  0.2462]],

        [[-0.1770, -0.0393,  0.1683],
         [-0.1770, -0.0393,  0.1683],
         [-0.1770, -0.0393,  0.1683],
         [-0.1770, -0.0393,  0.1683],
         [-0.1770, -0.0393,  0.1683]]], 4)
