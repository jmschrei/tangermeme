# test_saturation_mutagenesis.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest

from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot

from tangermeme.saturation_mutagenesis import _edit_distance_one
from tangermeme.saturation_mutagenesis import _attribution_score
from tangermeme.saturation_mutagenesis import saturation_mutagenesis

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
	return random_one_hot((2, 4, 10), random_state=0)


@pytest.fixture
def X0():
	return random_one_hot((2, 4, 100), random_state=0).float()


###


def test_edit_distance_one(X):
	X_one = _edit_distance_one(X[0], 0, -1)

	assert X_one.dtype == torch.int8
	assert X_one.shape == (40, 4, 10)
	assert X_one.sum() == 400

	assert_array_almost_equal(X_one[:4], [
		[[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		 [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 1, 0, 0, 1, 1, 1, 1, 0, 1]],

		[[1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
		 [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]],

		[[1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 1, 0, 0, 1, 1, 1, 1, 0, 1]],

		[[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		 [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 1, 0, 0, 1, 1, 1, 1, 0, 1]]])

	assert_array_almost_equal(X_one[-4:], [
		[[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		 [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 1, 0, 0, 1, 1, 1, 1, 0, 1]],

		[[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		 [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 1, 0, 0, 1, 1, 1, 1, 0, 1]],

		[[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 1, 0, 0, 1, 1, 1, 1, 1, 1]],

		[[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		 [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 1, 0, 0, 1, 1, 1, 1, 0, 1]]])


def test_edit_distance_one_start_end(X):
	X_one = _edit_distance_one(X[0], 2, 5)
	assert X_one.shape == (12, 4, 10)

	assert_array_almost_equal(X_one, [
		[[1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 1, 0, 0, 1, 1, 1, 1, 0, 1]],

		[[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		 [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 1, 0, 0, 1, 1, 1, 1, 0, 1]],

		[[1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
		 [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 1, 0, 0, 0, 1, 1, 1, 0, 1]],

		[[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		 [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 1, 0, 0, 1, 1, 1, 1, 0, 1]],

		[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 1, 0, 0, 1, 1, 1, 1, 0, 1]],

		[[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		 [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 1, 0, 0, 0, 1, 1, 1, 0, 1]],

		[[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
		 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
		 [0, 1, 0, 0, 1, 1, 1, 1, 0, 1]],

		[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
		 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		 [0, 1, 0, 0, 1, 1, 1, 1, 0, 1]],

		[[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		 [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
		 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
		 [0, 1, 0, 0, 0, 1, 1, 1, 0, 1]],

		[[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 1, 1, 0, 1, 1, 1, 1, 0, 1]],

		[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 1, 0, 1, 1, 1, 1, 1, 0, 1]],

		[[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		 [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 1, 0, 0, 1, 1, 1, 1, 0, 1]]])


###


def test_attribution_score():
	torch.manual_seed(0)
	y0 = torch.zeros(1, 1)
	y_hat = torch.randn(1, 4, 10, 1)

	attr = _attribution_score(y0, y_hat, None)
	attr2 = y_hat - y_hat.mean(dim=1)

	assert attr.shape == (1, 4, 10)
	assert_array_almost_equal(attr, [
		[[-1.3162, -0.6747, -0.4334, -1.1192,  0.7113,  0.7677,  0.0900,
          -1.1507,  0.5388, -1.4029],
         [ 0.1597,  0.7857, -0.0629,  0.5523,  0.9794, -0.1716, -0.9466,
          -0.7314,  0.7832,  0.6540],
         [ 0.4085, -1.0775, -0.5241,  1.1677, -0.3532, -0.6669,  0.9687,
           1.2242,  0.0425, -0.8183],
         [ 0.7479,  0.9665,  1.0204, -0.6008, -1.3375,  0.0709, -0.1121,
           0.6579, -1.3645,  1.5671]]], 4)
	assert_array_almost_equal(attr, attr2[:, :, :, 0], 4)


def test_attribution_score_average():
	torch.manual_seed(0)
	y0 = torch.zeros(1, 1) - 5
	y_hat = torch.randn(1, 4, 10, 1)

	attr = _attribution_score(y0, y_hat, None)
	attr2 = y_hat - y_hat.mean(dim=1)

	assert attr.shape == (1, 4, 10)
	assert_array_almost_equal(attr, [
		[[-1.3162, -0.6747, -0.4334, -1.1192,  0.7113,  0.7677,  0.0900,
          -1.1507,  0.5388, -1.4029],
         [ 0.1597,  0.7857, -0.0629,  0.5523,  0.9794, -0.1716, -0.9466,
          -0.7314,  0.7832,  0.6540],
         [ 0.4085, -1.0775, -0.5241,  1.1677, -0.3532, -0.6669,  0.9687,
           1.2242,  0.0425, -0.8183],
         [ 0.7479,  0.9665,  1.0204, -0.6008, -1.3375,  0.0709, -0.1121,
           0.6579, -1.3645,  1.5671]]], 4)
	assert_array_almost_equal(attr, attr2[:, :, :, 0], 4)


def test_attribution_score_target():
	torch.manual_seed(0)
	y0 = torch.zeros(1, 3)
	y_hat = torch.randn(1, 4, 10, 3)

	attr = _attribution_score(y0, y_hat, None)
	attr0 = _attribution_score(y0, y_hat, 0)
	attr1 = _attribution_score(y0, y_hat, 1)
	attr2 = _attribution_score(y0, y_hat, 2)

	assert attr.shape == (1, 4, 10)
	assert attr0.shape == (1, 4, 10)

	assert_raises(AssertionError, assert_array_almost_equal, attr0, attr1)
	assert_array_almost_equal((attr0 + attr1 + attr2) / 3, attr)


###


def test_saturation_mutagenesis(X0):
	torch.manual_seed(0)
	model = SmallDeepSEA(5)
	X_attr = saturation_mutagenesis(model, X0, device='cpu')

	assert X_attr.shape == (2, 4, 100)
	assert X_attr.dtype == torch.float32

	assert_array_almost_equal(X_attr[:, :, :3], [
		[[-2.3096e-04, -0.0000e+00,  0.0000e+00],
         [-0.0000e+00, -0.0000e+00, -5.1829e-04],
         [ 0.0000e+00,  0.0000e+00, -0.0000e+00],
         [-0.0000e+00,  5.3627e-04,  0.0000e+00]],

        [[ 0.0000e+00, -0.0000e+00,  1.1217e-03],
         [-1.9051e-06, -0.0000e+00, -0.0000e+00],
         [ 0.0000e+00,  6.1264e-04, -0.0000e+00],
         [-0.0000e+00,  0.0000e+00,  0.0000e+00]]], 4)


def test_saturation_mutagenesis_hypothetical(X0):
	torch.manual_seed(0)
	model = SmallDeepSEA(5)
	X_attr = saturation_mutagenesis(model, X0, hypothetical=True, device='cpu')
	X_attr2 = saturation_mutagenesis(model, X0, device='cpu')

	assert X_attr.shape == (2, 4, 100)
	assert X_attr.dtype == torch.float32

	assert_array_almost_equal(X_attr[:, :, :3], [
		[[-2.3096e-04, -9.0017e-04,  6.5279e-04],
         [-5.7105e-06, -4.0302e-04, -5.1829e-04],
         [ 3.5853e-04,  7.6692e-04, -1.9838e-04],
         [-1.2187e-04,  5.3627e-04,  6.3876e-05]],

        [[ 1.7478e-04, -2.4382e-04,  1.1217e-03],
         [-1.9051e-06, -7.1724e-04, -1.0386e-03],
         [ 4.5186e-05,  6.1264e-04, -5.9943e-04],
         [-2.1806e-04,  3.4841e-04,  5.1635e-04]]], 4)
	assert_array_almost_equal(X_attr * X0, X_attr2, 4)


def test_saturation_mutagenesis_start_end(X0):
	torch.manual_seed(0)
	model = SmallDeepSEA(5)
	X_attr = saturation_mutagenesis(model, X0, start=50, end=60, device='cpu')
	X_attr2 = saturation_mutagenesis(model, X0, device='cpu')

	assert X_attr.shape == (2, 4, 10)
	assert X_attr.dtype == torch.float32

	assert_array_almost_equal(X_attr[:, :, :3], [
		[[ 0.0007,  0.0000,  0.0000],
         [ 0.0000,  0.0002,  0.0007],
         [-0.0000, -0.0000, -0.0000],
         [-0.0000, -0.0000, -0.0000]],

        [[-0.0016,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000],
         [ 0.0000, -0.0007, -0.0015],
         [ 0.0000, -0.0000, -0.0000]]], 4)
	assert_array_almost_equal(X_attr, X_attr2[:, :, 50:60], 4)


def test_saturation_mutagenesis_start_end_hypothetical(X0):
	torch.manual_seed(0)
	model = SmallDeepSEA(5)
	X_attr = saturation_mutagenesis(model, X0, start=50, end=60, 
		hypothetical=True, device='cpu')
	X_attr2 = saturation_mutagenesis(model, X0, hypothetical=True, device='cpu')

	assert X_attr.shape == (2, 4, 10)
	assert X_attr.dtype == torch.float32

	assert_array_almost_equal(X_attr[:, :, :3], [
		[[ 6.7323e-04,  4.7941e-04,  7.2884e-04],
         [ 1.3862e-05,  2.4718e-04,  7.1416e-04],
         [-3.4264e-04, -2.4547e-04, -9.5634e-04],
         [-3.4445e-04, -4.8112e-04, -4.8665e-04]],

        [[-1.6091e-03,  2.2756e-04,  1.2385e-03],
         [ 1.9696e-04,  6.4452e-04,  6.1850e-04],
         [ 5.6783e-04, -7.1807e-04, -1.4681e-03],
         [ 8.4433e-04, -1.5401e-04, -3.8887e-04]]], 4)
	assert_array_almost_equal(X_attr, X_attr2[:, :, 50:60], 4)


def test_saturation_mutagenesis_ordering(X0):
	torch.manual_seed(0)
	model = SmallDeepSEA(5)
	X_attr = saturation_mutagenesis(model, X0, device='cpu')

	assert X_attr.shape == (2, 4, 100)
	assert X_attr.dtype == torch.float32

	X_attr2 = saturation_mutagenesis(model, X0[1:], device='cpu')
	assert_array_almost_equal(X_attr[1:, :, :], X_attr2, 2)


def test_saturation_mutagenesis_raw_output(X0):
	torch.manual_seed(0)
	model = SmallDeepSEA(5)
	y0, y_hat = saturation_mutagenesis(model, X0, raw_outputs=True, 
		device='cpu')

	assert y0.shape == (2, 5)
	assert y_hat.shape == (2, 4, 100, 5)

	assert_array_almost_equal(y0, [
		[ 0.1239, -0.1040, -0.2276, -0.0924, -0.0526],
        [ 0.1235, -0.1021, -0.2280, -0.1119, -0.0869]], 4)

	assert_array_almost_equal(y_hat[:, :, :3, :5], [
		[[[ 0.1239, -0.1040, -0.2276, -0.0924, -0.0526],
          [ 0.1232, -0.1041, -0.2280, -0.0957, -0.0551],
          [ 0.1270, -0.1001, -0.2249, -0.0953, -0.0535]],

         [[ 0.1240, -0.1025, -0.2279, -0.0924, -0.0528],
          [ 0.1228, -0.1043, -0.2289, -0.0934, -0.0537],
          [ 0.1239, -0.1040, -0.2276, -0.0924, -0.0526]],

         [[ 0.1258, -0.1025, -0.2281, -0.0933, -0.0516],
          [ 0.1255, -0.1027, -0.2266, -0.0957, -0.0518],
          [ 0.1270, -0.1012, -0.2243, -0.0971, -0.0554]],

         [[ 0.1250, -0.1041, -0.2280, -0.0918, -0.0532],
          [ 0.1239, -0.1040, -0.2276, -0.0924, -0.0526],
          [ 0.1246, -0.1015, -0.2275, -0.0906, -0.0547]]],


        [[[ 0.1233, -0.1008, -0.2273, -0.1121, -0.0875],
          [ 0.1200, -0.1013, -0.2281, -0.1115, -0.0888],
          [ 0.1235, -0.1021, -0.2280, -0.1119, -0.0869]],

         [[ 0.1235, -0.1021, -0.2280, -0.1119, -0.0869],
          [ 0.1182, -0.1017, -0.2305, -0.1094, -0.0886],
          [ 0.1188, -0.1067, -0.2305, -0.1109, -0.0867]],

         [[ 0.1242, -0.1030, -0.2267, -0.1128, -0.0868],
          [ 0.1235, -0.1021, -0.2280, -0.1119, -0.0869],
          [ 0.1206, -0.1055, -0.2288, -0.1117, -0.0886]],

         [[ 0.1230, -0.1015, -0.2268, -0.1136, -0.0876],
          [ 0.1212, -0.1014, -0.2300, -0.1108, -0.0856],
          [ 0.1212, -0.1036, -0.2303, -0.1100, -0.0857]]]], 4)


def test_saturation_mutagenesis_equivalence(X0):
	torch.manual_seed(0)
	model = SmallDeepSEA(5)

	X_attr = saturation_mutagenesis(model, X0, hypothetical=True, device='cpu')
	y0, y_hat = saturation_mutagenesis(model, X0, raw_outputs=True, 
		device='cpu')

	attr = _attribution_score(y0, y_hat, None)
	assert_array_almost_equal(X_attr, attr, 4)


def test_saturation_mutagenesis_sum_model(X):
	model = SumModel()
	y0, y_hat = saturation_mutagenesis(model, X, raw_outputs=True, device='cpu')

	assert y0.shape == (2, 4)
	assert y_hat.shape == (2, 4, 10, 4)

	assert_array_almost_equal(y0, [[2, 2, 0, 6], [4, 2, 3, 1]])
	assert_array_almost_equal(y_hat[:, :, :3], [
		[[[2, 2, 0, 6],
		  [3, 2, 0, 5],
		  [3, 1, 0, 6]],

		 [[1, 3, 0, 6],
		  [2, 3, 0, 5],
		  [2, 2, 0, 6]],

		 [[1, 2, 1, 6],
		  [2, 2, 1, 5],
		  [2, 1, 1, 6]],

		 [[1, 2, 0, 7],
		  [2, 2, 0, 6],
		  [2, 1, 0, 7]]],


		[[[5, 1, 3, 1],
		  [5, 2, 2, 1],
		  [4, 2, 3, 1]],

		 [[4, 2, 3, 1],
		  [4, 3, 2, 1],
		  [3, 3, 3, 1]],

		 [[4, 1, 4, 1],
		  [4, 2, 3, 1],
		  [3, 2, 4, 1]],

		 [[4, 1, 3, 2],
		  [4, 2, 2, 2],
		  [3, 2, 3, 2]]]])
