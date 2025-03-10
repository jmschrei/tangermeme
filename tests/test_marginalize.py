# test_marginalize.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest

from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot
from tangermeme.ersatz import substitute

from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.marginalize import marginalize

from .toy_models import SumModel
from .toy_models import FlattenDense
from .toy_models import Conv
from .toy_models import Scatter
from .toy_models import ConvDense
from .toy_models import SmallDeepSEA

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


##

def test_marginalize_summodel(X):
	torch.manual_seed(0)
	model = SumModel()
	y_before, y_after = marginalize(model, X, "ACGTC", batch_size=5, 
		device='cpu')

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
		[25., 24., 19., 32.],
        [26., 19., 28., 27.],
        [27., 26., 22., 25.],
        [22., 33., 18., 27.],
        [25., 23., 24., 28.],
        [30., 28., 21., 21.],
        [27., 18., 22., 33.],
        [20., 30., 30., 20.]])

	y_before2, y_after2 = marginalize(model, X, "ACGTC", batch_size=64, 
		device='cpu')
	assert_array_almost_equal(y_before, y_before2)
	assert_array_almost_equal(y_after, y_after2)	


def test_marginalize_flattendense(X):
	torch.manual_seed(0)
	model = FlattenDense()
	y_before, y_after = marginalize(model, X, "ACGTC", batch_size=8, 
		device='cpu')

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
		[ 0.1402,  0.4193, -0.0668],
        [-0.1778,  0.1163, -0.2797],
        [-0.1861,  0.1720, -0.0038],
        [-0.2052,  0.1570,  0.0039],
        [-0.0779,  0.0310, -0.2355],
        [-0.3882,  0.0477, -0.3510],
        [-0.2861,  0.0217, -0.3942],
        [-0.4546,  0.6714, -0.4246]], 4)

	y_before2, y_after2 = marginalize(model, X, "ACGTC", batch_size=64, 
		device='cpu')
	assert_array_almost_equal(y_before, y_before2)
	assert_array_almost_equal(y_after, y_after2)


def test_marginalize_conv(X):
	torch.manual_seed(0)
	model = Conv()
	y_before, y_after = marginalize(model, X, "ACGTC", start=0, batch_size=8, 
		device='cpu')

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
		[[-0.3983, -0.2996, -0.2749, -0.3509, -0.6158, -0.3269, -0.1928,
          -0.3509, -0.4816, -0.3197],
         [-0.1937, -0.0358, -0.2188,  0.4330,  0.0304,  0.2595,  0.2167,
           0.4330, -0.0124,  0.3218],
         [-0.2188, -0.0922, -0.1907, -0.3884, -0.5433, -0.4084, -0.2111,
          -0.3884, -0.3460, -0.3052],
         [-0.3638,  0.0378,  0.0811,  0.0337,  0.4894,  0.2007, -0.1166,
           0.0337,  0.1722, -0.3441]],

        [[-0.3983, -0.2996, -0.2749, -0.3197, -0.4805, -0.1669, -0.4055,
          -0.5078, -0.0848, -0.5863],
         [-0.1937, -0.0358, -0.2188,  0.3218, -0.3709, -0.3077, -0.2560,
          -0.0755,  0.1277, -0.5910],
         [-0.2188, -0.0922, -0.1907, -0.3052,  0.4396, -0.1558, -0.3220,
           0.1234, -0.1761,  0.2097],
         [-0.3638,  0.0378,  0.0811, -0.3441, -0.0027,  0.4644,  0.1810,
           0.1603,  0.2667,  0.1406]]], 4)

	y_before2, y_after2 = marginalize(model, X, "ACGTC", start=0, batch_size=64, 
		device='cpu')
	assert_array_almost_equal(y_before, y_before2)
	assert_array_almost_equal(y_after, y_after2)


def test_marginalize_scatter(X):
	torch.manual_seed(0)
	model = Scatter()
	y_before, y_after = marginalize(model, X, "ACGTC", start=0, batch_size=8, 
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

	y_before2, y_after2 = marginalize(model, X, "ACGTC", start=0, batch_size=64, 
		device='cpu')
	assert_array_almost_equal(y_before, y_before2)
	assert_array_almost_equal(y_after, y_after2)


def test_marginalize_convdense(X):
	torch.manual_seed(0)
	model = ConvDense()
	y_before, y_after = marginalize(model, X, "ACGTC", start=0, batch_size=2, 
		device='cpu')

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
		[ 0.1924,  0.2767,  0.0543],
        [-0.1214,  0.1609, -0.1117],
        [-0.1714,  0.1811,  0.2204],
        [-0.2601,  0.1425,  0.2126]], 4)


def test_marginalize_convdense_batch_size(X):
	torch.manual_seed(0)
	model = ConvDense()
	y_before, y_after = marginalize(model, X, "ACGTC", batch_size=68, 
		device='cpu')

	assert len(y_before) == 2
	assert y_before[0].shape == (64, 12, 98)
	assert y_before[1].shape == (64, 3)

	assert len(y_after) == 2
	assert y_after[0].shape == (64, 12, 98)
	assert y_after[1].shape == (64, 3)


def test_marginalize_scatter_batch_size(X):
	torch.manual_seed(0)
	model = Scatter()
	y_before, y_after = marginalize(model, X, "ACGTC", batch_size=68, 
		device='cpu')

	assert y_before.shape == (64, 100, 4)
	assert y_after.shape == (64, 100, 4)


def test_marginalize_raises_shape(X):
	torch.manual_seed(0)
	model = Scatter()
	assert_raises(ValueError, marginalize, model, X[0], "ACGTC", device='cpu')
	assert_raises(ValueError, marginalize, model, X[:, 0], "ACGTC", 
		device='cpu')
	assert_raises(ValueError, marginalize, model, X.unsqueeze(0), "ACGTC", 
		device='cpu')


def test_marginalize_raises_args(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense()
	assert_raises(TypeError, marginalize, model, X, "ACGTC", batch_size=2, 
		args=5, device='cpu')
	assert_raises(AttributeError, marginalize, model, X, "ACGTC",  batch_size=2, 
		args=(5,), device='cpu')
	assert_raises(ValueError, marginalize, model, X, "ACGTC", batch_size=2, 
		args=alpha, device='cpu')
	assert_raises(ValueError, marginalize, model, X, "ACGTC", batch_size=2, 
		args=(alpha[:5],), device='cpu')
	assert_raises(ValueError, marginalize, model, X, "ACGTC", batch_size=2, 
		args=(alpha, beta[:5]), device='cpu')


###


def test_marginalize_deep_lift_shap(X):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y_before, y_after = marginalize(model, X[:2], "ACGTC", func=deep_lift_shap, 
		device='cpu', random_state=0)

	assert y_before.shape == (2, 4, 100)
	assert y_before.dtype == torch.float32
	assert_array_almost_equal(y_before[:, :, 48:52], [
				[[ 0.0000e+00,  0.0000e+00, -5.1092e-05,  0.0000e+00],
         [-0.0000e+00, -0.0000e+00,  0.0000e+00, -6.6958e-04],
         [ 1.5839e-05, -0.0000e+00,  0.0000e+00, -0.0000e+00],
         [-0.0000e+00, -7.8783e-04,  0.0000e+00,  0.0000e+00]],

        [[-0.0000e+00,  0.0000e+00, -5.8860e-04, -0.0000e+00],
         [-2.0069e-03,  0.0000e+00, -0.0000e+00, -0.0000e+00],
         [ 0.0000e+00,  0.0000e+00, -0.0000e+00, -2.2559e-03],
         [-0.0000e+00, -6.4841e-04,  0.0000e+00,  0.0000e+00]]], 4)

	assert y_after.shape == (2, 4, 100)
	assert y_after.dtype == torch.float32
	assert_array_almost_equal(y_after[:, :, 48:52], [
				[[-3.2119e-04,  0.0000e+00,  0.0000e+00,  0.0000e+00],
         [ 0.0000e+00, -3.3051e-04, -0.0000e+00, -0.0000e+00],
         [ 0.0000e+00, -0.0000e+00,  7.4217e-04,  0.0000e+00],
         [-0.0000e+00, -0.0000e+00,  0.0000e+00, -5.0473e-04]],

        [[-5.0819e-06,  0.0000e+00,  0.0000e+00, -0.0000e+00],
         [-0.0000e+00, -9.7784e-05, -0.0000e+00,  0.0000e+00],
         [ 0.0000e+00, -0.0000e+00, -9.7142e-05, -0.0000e+00],
         [ 0.0000e+00,  0.0000e+00, -0.0000e+00,  2.8622e-05]]], 4)

	y_before2, y_after2 = marginalize(model, X[:8], "ACGTC", 
		func=deep_lift_shap, batch_size=64, device='cpu', random_state=0)
	assert_array_almost_equal(y_before, y_before2[:2], 4)
	assert_array_almost_equal(y_after, y_after2[:2], 4)


def test_marginalize_deep_lift_shap_flattendense(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)
	y_before, y_after = marginalize(model, X[:8], "ACGTC", func=deep_lift_shap, 
		device='cpu', random_state=0)

	assert y_before.shape == (8, 4, 100)
	assert y_before.dtype == torch.float32
	assert_array_almost_equal(y_before[:4, :, 48:53], [
				[[-0.0000,  0.0000,  0.0428,  0.0000,  0.0000],
         [-0.0000, -0.0000, -0.0000, -0.0558, -0.0236],
         [ 0.0214,  0.0000, -0.0000, -0.0000,  0.0000],
         [-0.0000,  0.0244, -0.0000,  0.0000, -0.0000]],

        [[-0.0000,  0.0000,  0.0393,  0.0000,  0.0000],
         [-0.0084, -0.0000, -0.0000, -0.0000, -0.0000],
         [ 0.0000, -0.0000, -0.0000, -0.0226,  0.0294],
         [-0.0000,  0.0166, -0.0000,  0.0000, -0.0000]],

        [[-0.0046,  0.0053,  0.0000,  0.0000,  0.0000],
         [-0.0000, -0.0000, -0.0000, -0.0000, -0.0158],
         [ 0.0000, -0.0000, -0.0000, -0.0000,  0.0000],
         [-0.0000,  0.0000, -0.0068,  0.0150, -0.0000]],

        [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
         [-0.0000, -0.0000, -0.0384, -0.0000, -0.0125],
         [ 0.0000,  0.0039, -0.0000,  0.0074,  0.0000],
         [-0.0065,  0.0000, -0.0000,  0.0000, -0.0000]]], 4)

	assert y_after.shape == (8, 4, 100)
	assert y_after.dtype == torch.float32
	assert_array_almost_equal(y_after[:4, :, 48:53], [
				[[-0.0023, -0.0000,  0.0000,  0.0000,  0.0000],
         [-0.0000, -0.0383, -0.0000, -0.0000, -0.0293],
         [ 0.0000, -0.0000, -0.0165, -0.0000,  0.0000],
         [-0.0000,  0.0000, -0.0000,  0.0174, -0.0000]],

        [[-0.0025,  0.0000,  0.0000,  0.0000,  0.0000],
         [-0.0000, -0.0350, -0.0000, -0.0000, -0.0282],
         [ 0.0000, -0.0000, -0.0121, -0.0000,  0.0000],
         [-0.0000,  0.0000, -0.0000,  0.0182, -0.0000]],

        [[-0.0012,  0.0000,  0.0000,  0.0000,  0.0000],
         [-0.0000, -0.0290, -0.0000, -0.0000, -0.0112],
         [ 0.0000,  0.0000, -0.0146, -0.0000,  0.0000],
         [-0.0000,  0.0000, -0.0000,  0.0198, -0.0000]],

        [[-0.0018, -0.0000,  0.0000,  0.0000,  0.0000],
         [-0.0000, -0.0403, -0.0000, -0.0000, -0.0177],
         [ 0.0000, -0.0000, -0.0077, -0.0000,  0.0000],
         [-0.0000,  0.0000,  0.0000,  0.0255, -0.0000]]], 4)

	y_before2, y_after2 = marginalize(model, X[:8], "ACGTC", func=deep_lift_shap, 
		batch_size=64, device='cpu', random_state=0)
	assert_array_almost_equal(y_before, y_before2)
	assert_array_almost_equal(y_after, y_after2)


def test_marginalize_deep_lift_shap_vs_attribute(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)
	y_before, y_after = marginalize(model, X[:8], "ACGTC", func=deep_lift_shap, 
		device='cpu', random_state=0)

	y_before0 = deep_lift_shap(model, X[:8], device='cpu', random_state=0)
	y_after0 = deep_lift_shap(model, substitute(X[:8], "ACGTC"), 
		device='cpu', random_state=0)

	assert y_before.shape == (8, 4, 100)
	assert y_before.dtype == torch.float32
	assert_array_almost_equal(y_before, y_before0, 4)

	assert y_after.shape == (8, 4, 100)
	assert y_after.dtype == torch.float32
	assert_array_almost_equal(y_after, y_after0, 4)


def test_marginalize_deep_lift_shap_alpha(X, alpha):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	y_before0, y_after0 = marginalize(model, X[:8], "ACGTC", func=deep_lift_shap, 
		device='cpu', random_state=0)
	y_before1, y_after1 = marginalize(model, X[:8], "ACGTC", func=deep_lift_shap, 
		device='cpu', random_state=0, args=(alpha,))

	assert y_before0.shape == (8, 4, 100)
	assert y_before0.dtype == torch.float32
	assert_array_almost_equal(y_before0, y_before1, 4)

	assert y_after0.shape == (8, 4, 100)
	assert y_after0.dtype == torch.float32
	assert_array_almost_equal(y_after0, y_after1, 4)


def test_marginalize_deep_lift_shap_alpha_beta(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	y_before0, y_after0 = marginalize(model, X[:8], "ACGTC", func=deep_lift_shap, 
		device='cpu', random_state=0)
	y_before1, y_after1 = marginalize(model, X[:8], "ACGTC", func=deep_lift_shap, 
		device='cpu', random_state=0, args=(alpha, beta))

	assert y_before0.shape == (8, 4, 100)
	assert y_before0.dtype == torch.float32
	assert_raises(AssertionError, assert_array_almost_equal, y_before0, 
		y_before1, 4)

	assert y_after0.shape == (8, 4, 100)
	assert y_after0.dtype == torch.float32
	assert_raises(AssertionError, assert_array_almost_equal, y_after0, 
		y_after1, 4)


def test_marginalize_deep_lift_shap_n_shuffles(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	y_before0, y_after0 = marginalize(model, X[:8], "ACGTC", func=deep_lift_shap, 
		n_shuffles=2, device='cpu', random_state=0)
	y_before1, y_after1 = marginalize(model, X[:8], "ACGTC", func=deep_lift_shap, 
		n_shuffles=10, device='cpu', random_state=0)

	assert y_before0.shape == (8, 4, 100)
	assert y_before0.dtype == torch.float32
	assert_array_almost_equal(y_before0[:2, :, :4], [
				[[ 0.0000, -0.0000, -0.0000, -0.0012],
         [ 0.0000,  0.0000,  0.0021,  0.0000],
         [ 0.0000, -0.0000, -0.0000,  0.0000],
         [-0.0000, -0.0247, -0.0000,  0.0000]],

        [[-0.0000, -0.0000, -0.0076, -0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000],
         [-0.0000,  0.0000, -0.0000, -0.0000],
         [-0.0000, -0.0000,  0.0000,  0.0350]]], 4)

	assert y_before1.shape == (8, 4, 100)
	assert y_before1.dtype == torch.float32
	assert_array_almost_equal(y_before1[:2, :, :4], [
				[[ 0.0000, -0.0000, -0.0000, -0.0268],
         [ 0.0000,  0.0000,  0.0087, -0.0000],
         [ 0.0000,  0.0000, -0.0000, -0.0000],
         [-0.0000, -0.0209,  0.0000,  0.0000]],

        [[-0.0000,  0.0000, -0.0056, -0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000],
         [-0.0000,  0.0056, -0.0000, -0.0000],
         [-0.0000, -0.0000,  0.0000,  0.0336]]], 4)

	assert y_after0.shape == (8, 4, 100)
	assert y_after0.dtype == torch.float32
	assert_array_almost_equal(y_after0[:2, :, :4], [
				[[ 0.0000, -0.0000, -0.0000, -0.0025],
         [ 0.0000,  0.0000,  0.0101,  0.0000],
         [ 0.0000, -0.0000, -0.0000,  0.0000],
         [-0.0000, -0.0247,  0.0000,  0.0000]],

        [[-0.0000,  0.0000, -0.0034, -0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000],
         [-0.0000,  0.0109,  0.0000, -0.0000],
         [-0.0000, -0.0000,  0.0000,  0.0242]]], 4)

	assert y_after1.shape == (8, 4, 100)
	assert y_after1.dtype == torch.float32
	assert_array_almost_equal(y_after1[:2, :, :4], [
				[[ 0.0000, -0.0000, -0.0000, -0.0249],
         [ 0.0000,  0.0000,  0.0103, -0.0000],
         [ 0.0000,  0.0000, -0.0000, -0.0000],
         [-0.0000, -0.0209,  0.0000,  0.0000]],

        [[-0.0000,  0.0000, -0.0051, -0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000],
         [-0.0000,  0.0108, -0.0000, -0.0000],
         [-0.0000, -0.0000,  0.0000,  0.0307]]], 4)


def test_marginalize_deep_lift_shap_hypothetical(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	y_before, y_after = marginalize(model, X[:8], "ACGTC", func=deep_lift_shap, 
		hypothetical=True, device='cpu', random_state=0)

	assert y_before.shape == (8, 4, 100)
	assert y_before.dtype == torch.float32
	assert_array_almost_equal(y_before[:2, :, :4], [
				[[ 0.0000,  0.0006, -0.0079, -0.0258],
         [ 0.0474,  0.0075,  0.0114, -0.0015],
         [ 0.0025,  0.0060, -0.0046, -0.0233],
         [-0.0237, -0.0172,  0.0073,  0.0226]],

        [[-0.0474, -0.0009, -0.0058, -0.0172],
         [ 0.0000,  0.0059,  0.0135,  0.0070],
         [-0.0449,  0.0045, -0.0025, -0.0147],
         [-0.0712, -0.0188,  0.0093,  0.0311]]], 4)

	assert y_after.shape == (8, 4, 100)
	assert y_after.dtype == torch.float32
	assert_array_almost_equal(y_after[:2, :, :4], [
				[[ 0.0000,  0.0006, -0.0081, -0.0249],
         [ 0.0474,  0.0075,  0.0113, -0.0007],
         [ 0.0025,  0.0060, -0.0047, -0.0224],
         [-0.0237, -0.0172,  0.0071,  0.0234]],

        [[-0.0474,  0.0030, -0.0066, -0.0201],
         [ 0.0000,  0.0099,  0.0127,  0.0041],
         [-0.0449,  0.0085, -0.0033, -0.0176],
         [-0.0712, -0.0148,  0.0085,  0.0282]]], 4)

	y_before1, y_after1 = marginalize(model, X[:8], "ACGTC", func=deep_lift_shap, 
		additional_func_kwargs={'hypothetical': True}, device='cpu', 
		random_state=0)

	assert_array_almost_equal(y_before, y_before1, 4)
	assert_array_almost_equal(y_after, y_after1, 4)


def test_marginalize_deep_lift_shap_raises(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	assert_raises(TypeError, marginalize, model, X, "ACGTC", func=deep_lift_shap,
		device='cpu', additional_func_kwargs={'device': 'cpu'})
	assert_raises(TypeError, marginalize, model, X, "ACGTC", 
		func=deep_lift_shap, device='cpu', end=10)
