# test_predict.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest

from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot

from tangermeme.space import space
from tangermeme.predict import predict
from tangermeme.product import apply_product
from tangermeme.product import apply_pairwise
from tangermeme.marginalize import marginalize

from .toy_models import SumModel
from .toy_models import FlattenDense
from .toy_models import Conv
from .toy_models import Scatter
from .toy_models import ConvDense

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal

torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)


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


###


def test_apply_pairwise_predict_flattendense_alpha(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense()
	y0 = predict(model, X[:5], device='cpu').unsqueeze(1)
	y = apply_pairwise(predict, model, X[:5], args=(alpha[:5],), device='cpu')
	assert y.shape == (5, 5, 3)
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


	y = apply_pairwise(predict, model, X[:5], args=(alpha[5:10],), device='cpu')
	assert y.shape == (5, 5, 3)
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


def test_apply_pairwise_predict_flattendense_beta(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense()
	alpha = torch.zeros(X.shape[0], 1)

	y0 = predict(model, X[:5], device='cpu').unsqueeze(1)
	y = apply_pairwise(predict, model, X[:5], args=(alpha[:1].repeat(5, 1), 
		beta[:5]), device='cpu')
	
	assert y.shape == (5, 5, 3)
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


	y = apply_pairwise(predict, model, X[:5], args=(alpha[:1].repeat(5, 1), 
		beta[5:10]), device='cpu')

	assert y.shape == (5, 5, 3)
	assert y.dtype == torch.float32
	assert_array_almost_equal(y, y0 * beta[5:10][None, :])
	assert_array_almost_equal(y[:2], 
		[[[-0.4437, -0.6417,  0.2302],
          [ 0.3364,  0.4865, -0.1745],
          [-0.1468, -0.2122,  0.0761],
          [ 0.0615,  0.0889, -0.0319],
          [-0.0481, -0.0695,  0.0249]],
 
         [[ 0.1162, -0.0400,  0.4030],
          [-0.0881,  0.0303, -0.3056],
          [ 0.0384, -0.0132,  0.1333],
          [-0.0161,  0.0055, -0.0559],
          [ 0.0126, -0.0043,  0.0437]]], 4)


def test_apply_pairwise_predict_flattendense_alpha_beta(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense()
	y0 = predict(model, X[:5], device='cpu').unsqueeze(1)
	y = apply_pairwise(predict, model, X[:5], args=(alpha[:5], beta[:5]), 
		device='cpu')

	assert y.shape == (5, 5, 3)
	assert y.dtype == torch.float32

	assert_array_almost_equal(y, y0 * beta[:5] + alpha[:5])
	assert_array_almost_equal(y[:2], [[[2.0772, 2.2169, 1.6016],
         [0.2822, 0.2296, 0.4613],
         [0.8769, 0.8315, 1.0316],
         [2.0340, 1.9417, 2.3482],
         [2.0344, 2.1088, 1.7810]],

        [[1.6821, 1.7923, 1.4796],
         [0.4310, 0.3895, 0.5073],
         [1.0054, 0.9696, 1.0712],
         [2.2951, 2.2223, 2.4288],
         [1.8239, 1.8826, 1.7160]]], 4)


def test_apply_pairwise_predict_convdense_alpha(X, alpha):
	torch.manual_seed(0)
	model = ConvDense()

	y = apply_pairwise(predict, model, X[:5], args=(alpha[:5, :, None],), 
		batch_size=2, device='cpu')

	assert len(y) == 2
	assert y[0].shape == (5, 5, 12, 98)
	assert y[1].shape == (5, 5, 3)

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


###


def test_apply_product_marginalize_flattendense_alpha(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense()
	y0_before, y0_after = marginalize(model, X, 'ACGTGC', device='cpu')
	y_before, y_after = apply_product(marginalize, model, X, motif='ACGTGC', 
		args=(alpha[:5],), device='cpu')

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
		[[1.9098, 2.0546, 1.9081],
		 [0.5459, 0.6907, 0.5442],
		 [1.1245, 1.2693, 1.1228],
		 [2.3867, 2.5314, 2.3850],
		 [2.0133, 2.1581, 2.0116]],

		[[1.6194, 1.7230, 1.7036],
		 [0.2555, 0.3591, 0.3397],
		 [0.8341, 0.9377, 0.9183],
		 [2.0962, 2.1999, 2.1805],
		 [1.7229, 1.8265, 1.8071]]], 4)



def test_apply_product_marginalize_flattendense_beta(X, beta):
	torch.manual_seed(0)
	model = FlattenDense()
	alpha = torch.zeros(X.shape[0], 1)

	y0_before, y0_after = marginalize(model, X, 'ACGTGC', device='cpu')
	y_before, y_after = apply_product(marginalize, model, X, motif='ACGTGC', 
		args=(alpha[:1], beta[:5]), device='cpu')
	y_before, y_after = y_before[:, 0], y_after[:, 0]

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
		[[ 0.2368,  0.4720,  0.2340],
		 [-0.0892, -0.1777, -0.0881],
		 [-0.0770, -0.1535, -0.0761],
		 [-0.1564, -0.3118, -0.1546],
		 [ 0.1262,  0.2514,  0.1247]],

		[[-0.2350, -0.0666, -0.0981],
		 [ 0.0885,  0.0251,  0.0370],
		 [ 0.0764,  0.0217,  0.0319],
		 [ 0.1552,  0.0440,  0.0648],
		 [-0.1252, -0.0355, -0.0523]]], 4)


def test_apply_product_marginalize_flattendense_alpha_beta(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense()
	y0_before, y0_after = marginalize(model, X, 'ACGTGC', device='cpu')
	y_before, y_after = apply_product(marginalize, model, X, motif='ACGTGC', 
		args=(alpha[:5], beta[:4]), device='cpu')

	assert y_before.shape == (64, 5, 4, 3)
	assert y_before.dtype == torch.float32
	assert_array_almost_equal(y_before, y0_before[:, None, None] * 
		beta[:4][None, None, :] + alpha[:5][None, :, None])
	assert_array_almost_equal(y_before[:2, :2], [
		[[[2.0772, 2.2169, 1.6016],
		  [1.6461, 1.5935, 1.8252],
		  [1.6622, 1.6168, 1.8169],
		  [1.5572, 1.4649, 1.8714]],

		 [[0.7133, 0.8530, 0.2377],
		  [0.2822, 0.2296, 0.4613],
		  [0.2983, 0.2529, 0.4530],
		  [0.1933, 0.1010, 0.5075]]],


		[[[1.6821, 1.7923, 1.4796],
		  [1.7949, 1.7534, 1.8712],
		  [1.7907, 1.7549, 1.8565],
		  [1.8182, 1.7454, 1.9520]],

		 [[0.3182, 0.4284, 0.1157],
		  [0.4310, 0.3895, 0.5073],
		  [0.4268, 0.3910, 0.4927],
		  [0.4543, 0.3815, 0.5881]]]], 4)

	assert y_after.shape == (64, 5, 4, 3)
	assert y_after.dtype == torch.float32
	assert_array_almost_equal(y_after, y0_after[:, None, None] * 
		beta[:4][None, None, :] + alpha[:5][None, :, None])
	assert_array_almost_equal(y_after[:2, :2], [
		[[[2.0008, 2.2360, 1.9981],
		  [1.6749, 1.5863, 1.6759],
		  [1.6871, 1.6106, 1.6880],
		  [1.6076, 1.4523, 1.6095]],

		 [[0.6370, 0.8721, 0.6342],
		  [0.3110, 0.2224, 0.3120],
		  [0.3232, 0.2467, 0.3241],
		  [0.2437, 0.0884, 0.2456]]],


		[[[1.5291, 1.6974, 1.6659],
		  [1.8526, 1.7891, 1.8010],
		  [1.8405, 1.7857, 1.7960],
		  [1.9193, 1.8081, 1.8289]],

		 [[0.1652, 0.3335, 0.3020],
		  [0.4887, 0.4252, 0.4371],
		  [0.4766, 0.4218, 0.4321],
		  [0.5554, 0.4442, 0.4650]]]], 4)


def test_apply_product_marginalize_convdense_alpha(X, alpha):
	torch.manual_seed(0)
	model = ConvDense()
	y_before, y_after = apply_product(marginalize, model, X, motif="ACGTGC", 
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
		[[ 0.2472,  0.1913, -0.0212],
		 [ 0.2472,  0.1913, -0.0212],
		 [ 0.2472,  0.1913, -0.0212],
		 [ 0.2472,  0.1913, -0.0212],
		 [ 0.2472,  0.1913, -0.0212]],

		[[-0.1387,  0.0929, -0.1931],
		 [-0.1387,  0.0929, -0.1931],
		 [-0.1387,  0.0929, -0.1931],
		 [-0.1387,  0.0929, -0.1931],
		 [-0.1387,  0.0929, -0.1931]],

		[[-0.1165,  0.0957,  0.1448],
		 [-0.1165,  0.0957,  0.1448],
		 [-0.1165,  0.0957,  0.1448],
		 [-0.1165,  0.0957,  0.1448],
		 [-0.1165,  0.0957,  0.1448]],

		[[-0.2053,  0.0571,  0.1370],
		 [-0.2053,  0.0571,  0.1370],
		 [-0.2053,  0.0571,  0.1370],
		 [-0.2053,  0.0571,  0.1370],
		 [-0.2053,  0.0571,  0.1370]]], 4)


###


def test_apply_product_space_flattendense_alpha(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense()
	y0_before, y0_after = space(model, X, ['ACGTGC', 'TGTT'], [[3]], 
		device='cpu')
	y_before, y_after = apply_product(space, model, X, 
		motifs=['ACGTGC', 'TGTT'], spacing=[[3]], args=(alpha[:5],), 
		device='cpu')

	assert y_before.shape == (64, 5, 3)
	assert y_before.dtype == torch.float32
	assert_array_almost_equal(y_before, y0_before[:, None] + alpha[:5][None])
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


	assert y_after.shape == (64, 5, 1, 3)
	assert y_after.dtype == torch.float32
	assert_array_almost_equal(y_after, y0_after[:, None] + alpha[:5][None, :, 
		None])
	assert_array_almost_equal(y_after[:2, :, 0], [
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


def test_apply_product_space_flattendense_beta(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense()
	alpha = torch.zeros(X.shape[0], 1)

	y0_before, y0_after = space(model, X, ['ACGTGC', 'TGTT'], [[3]], 
		device='cpu')
	y_before, y_after = apply_product(space, model, X, 
		motifs=['ACGTGC', 'TGTT'], spacing=[[3]], args=(alpha[:1], beta[:5]), 
		device='cpu')
	y_before, y_after = y_before[:, 0], y_after[:, 0]

	assert y_before.shape == (64, 5, 3)
	assert y_before.dtype == torch.float32
	assert_array_almost_equal(y_before, y0_before[:, None] * beta[:5][None])
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

	assert y_after.shape == (64, 5, 1, 3)
	assert y_after.dtype == torch.float32
	assert_array_almost_equal(y_after, y0_after[:, None] * beta[:5][None, :, 
		None])
	assert_array_almost_equal(y_after[:2, :, 0], [
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


def test_apply_product_space_flattendense_alpha_beta(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense()
	y0_before, y0_after = space(model, X, ['ACGTGC', 'TGTT'], [[3]], 
		device='cpu')
	y_before, y_after = apply_product(space, model, X, 
		motifs=['ACGTGC', 'TGTT'], spacing=[[3]], args=(alpha[:4], beta[:5]), 
		device='cpu')

	assert y_before.shape == (64, 4, 5, 3)
	assert y_before.dtype == torch.float32
	assert_array_almost_equal(y_before, y0_before[:, None, None] * 
		beta[:5][None, None] + alpha[:4][None, :, None])
	assert_array_almost_equal(y_before[:2, :2], [
		[[[2.0772, 2.2169, 1.6016],
          [1.6461, 1.5935, 1.8252],
          [1.6622, 1.6168, 1.8169],
          [1.5572, 1.4649, 1.8714],
          [1.9309, 2.0053, 1.6775]],

         [[0.7133, 0.8530, 0.2377],
          [0.2822, 0.2296, 0.4613],
          [0.2983, 0.2529, 0.4530],
          [0.1933, 0.1010, 0.5075],
          [0.5670, 0.6414, 0.3136]]],


        [[[1.6821, 1.7923, 1.4796],
          [1.7949, 1.7534, 1.8712],
          [1.7907, 1.7549, 1.8565],
          [1.8182, 1.7454, 1.9520],
          [1.7204, 1.7791, 1.6125]],

         [[0.3182, 0.4284, 0.1157],
          [0.4310, 0.3895, 0.5073],
          [0.4268, 0.3910, 0.4927],
          [0.4543, 0.3815, 0.5881],
          [0.3565, 0.4152, 0.2486]]]], 4)

	assert y_after.shape == (64, 4, 5, 1, 3)
	assert y_after.dtype == torch.float32
	assert_array_almost_equal(y_after, y0_after[:, None, None] * 
		beta[:5][None, None, :, None] + alpha[:4][None, :, None, None])
	assert_array_almost_equal(y_after[:2, :2, :, 0], [
		[[[2.0082, 2.1007, 1.6017],
          [1.6721, 1.6373, 1.8252],
          [1.6847, 1.6546, 1.8169],
          [1.6028, 1.5417, 1.8713],
          [1.8941, 1.9434, 1.6775]],

         [[0.6443, 0.7368, 0.2378],
          [0.3082, 0.2734, 0.4613],
          [0.3208, 0.2907, 0.4530],
          [0.2389, 0.1778, 0.5074],
          [0.5302, 0.5795, 0.3136]]],


        [[[1.7199, 1.6266, 1.5301],
          [1.7807, 1.8158, 1.8522],
          [1.7784, 1.8087, 1.8401],
          [1.7932, 1.8548, 1.9186],
          [1.7405, 1.6908, 1.6394]],

         [[0.3560, 0.2627, 0.1662],
          [0.4168, 0.4519, 0.4883],
          [0.4145, 0.4448, 0.4762],
          [0.4293, 0.4909, 0.5547],
          [0.3767, 0.3269, 0.2755]]]], 4)


def test_apply_product_space_convdense_alpha(X, alpha):
  torch.manual_seed(0)
  model = ConvDense()
  y_before, y_after = apply_product(space, model, X, motifs=["ACGTGC", "TGGTT"], 
  	spacing=[[3]], start=0, args=(alpha[:5, :, None],), batch_size=2, 
  	device='cpu')

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
  assert y_after[0].shape == (64, 5, 1, 12, 98)
  assert y_after[1].shape == (64, 5, 1, 3)

  assert y_after[0].dtype == torch.float32
  assert y_after[1].dtype == torch.float32

  assert_array_almost_equal(y_after[0][:2, :2, 0, :3, :4], [
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

  assert_array_almost_equal(y_after[1][:4, :, 0], [
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
