# test_predict.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest

from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot
from tangermeme.utils import PackedSequence

from tangermeme.predict import predict
from tangermeme.predict import _to_device

from .toy_models import SumModel
from .toy_models import FlattenDense
from .toy_models import Conv
from .toy_models import Scatter
from .toy_models import ConvDense
from .toy_models import ResidualConv
from .toy_models import Transformer
from .toy_models import Conv2DExpand
from .toy_models import CustomLinear
from .toy_models import CustomSqrt
from .toy_models import DilatedConv
from .toy_models import ConvBatchNorm
from .toy_models import ConvLayerNorm
from .toy_models import MultiActivation
from .toy_models import DropoutConv
from .toy_models import MultiInputMultiOutput
from ._packed import assert_same
from ._packed import with_N

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


def test_predict_summodel(X, device):
	torch.manual_seed(0)
	model = SumModel().to(device)
	y = predict(model, X, batch_size=8, device=device)

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

	assert_array_almost_equal(y, model(X.to(device)).cpu())
	assert_array_almost_equal(y, predict(model, X, batch_size=1, device=device))
	assert_array_almost_equal(y, predict(model, X, batch_size=64, device=device))


def test_predict_flattendense(X, device):
	torch.manual_seed(0)
	model = FlattenDense().to(device)
	y = predict(model, X, batch_size=8, device=device)

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

	assert_array_almost_equal(y, model(X.to(device)).cpu().detach())
	assert_array_almost_equal(y, predict(model, X, batch_size=1, device=device))
	assert_array_almost_equal(y, predict(model, X, batch_size=64, device=device))


def test_predict_conv(X, device):
	torch.manual_seed(0)
	model = Conv().to(device)
	y = predict(model, X, batch_size=8, device=device)

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

	assert_array_almost_equal(y, model(X.to(device)).cpu().detach())
	assert_array_almost_equal(y, predict(model, X, batch_size=1, device=device))
	assert_array_almost_equal(y, predict(model, X, batch_size=64, device=device))


def test_predict_scatter(X, device):
	torch.manual_seed(0)
	model = Scatter().to(device)
	y = predict(model, X, batch_size=8, device=device)

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

	assert_array_almost_equal(y, model(X.to(device)).cpu().detach())
	assert_array_almost_equal(y, predict(model, X, batch_size=1, device=device))
	assert_array_almost_equal(y, predict(model, X, batch_size=64, device=device))
	assert_array_almost_equal(y, predict(model, X, batch_size=68, device=device))


def test_predict_convdense(X, device):
	torch.manual_seed(0)
	model = ConvDense().to(device)
	y = predict(model, X, batch_size=2, device=device)

	assert isinstance(y, list)
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

	for batch_size in (8, 64):
		y_ = predict(model, X, batch_size=batch_size, device=device)

		assert isinstance(y_, list)
		assert len(y_) == 2
		for head, head_ in zip(y, y_):
			assert head_.dtype == head.dtype
			assert_array_almost_equal(head_, head, 4)


def test_predict_raises_args(X, alpha, beta):
	torch.manual_seed(0)
	model = FlattenDense()
	assert_raises(TypeError, predict, model, X, batch_size=2, args=5)
	assert_raises(AttributeError, predict, model, X, batch_size=2, args=(5,))
	assert_raises(ValueError, predict, model, X, batch_size=2, args=alpha)
	assert_raises(ValueError, predict, model, X, batch_size=2,
		args=(alpha[:5],))
	assert_raises(ValueError, predict, model, X, batch_size=2,
		args=(alpha, beta[:5]))


def test_predict_flattendense_16bit(X, device):
	torch.manual_seed(0)
	model = FlattenDense().to(device)
	y = predict(model, X, batch_size=8, dtype=torch.float16, device=device)

	assert y.shape == (64, 3)
	assert y.dtype == torch.float16
	assert next(model.parameters()).dtype == torch.float32

	assert_array_almost_equal(y[:8], [
		[ 0.1928,  0.2788, -0.1000],
        [-0.0505,  0.0174, -0.1751],
        [-0.1408,  0.0105,  0.0673],
        [-0.2375, -0.0069,  0.0835],
        [ 0.0442, -0.0692, -0.1565],
        [-0.3489, -0.0876, -0.2994],
        [-0.3894, -0.1332, -0.3717],
        [-0.3682,  0.5173, -0.4089]], 3)

	assert_array_almost_equal(y, model(X.to(device)).cpu().detach(), 3)
	assert_array_almost_equal(y, predict(model, X, batch_size=1, device=device), 3)
	assert_array_almost_equal(y, predict(model, X, batch_size=64, device=device), 3)


###
# Tests for additional architectures (residual, attention, 2D conv, custom
# autograd ops, dilated conv, normalization, multi-activation, dropout, and
# multi-input/multi-output).


@pytest.mark.parametrize("model_factory, expected", [
	(ResidualConv, [
		[-0.4900],
		[ 0.0663],
		[-0.6028],
		[-0.1464]]),
	(lambda: Transformer().eval(), [
		[-0.5057],
		[ 0.7587],
		[-0.1567],
		[ 0.4629]]),
	(Conv2DExpand, [
		[ 0.0659],
		[ 0.0002],
		[-0.0613],
		[-0.0342]]),
	(CustomLinear, [
		[-0.0962],
		[ 0.2408],
		[ 0.1910],
		[ 0.2657]]),
	(CustomSqrt, [
		[-0.2022],
		[-0.1220],
		[-0.1642],
		[-0.1508]]),
	(DilatedConv, [
		[0.0683],
		[0.0574],
		[0.0644],
		[0.0379]]),
	(lambda: ConvBatchNorm().eval(), [
		[ 0.0440],
		[-0.0024],
		[ 0.0094],
		[ 0.0306]]),
	(ConvLayerNorm, [
		[ 0.2170],
		[ 0.3098],
		[-0.6040],
		[-0.1358]]),
	(MultiActivation, [
		[0.0712],
		[0.0853],
		[0.0732],
		[0.0559]])
], ids=[
	"ResidualConv", "Transformer", "Conv2DExpand", "CustomLinear",
	"CustomSqrt", "DilatedConv", "ConvBatchNorm", "ConvLayerNorm",
	"MultiActivation"])
def test_predict_architectures(X, model_factory, expected, device):
	torch.manual_seed(0)
	model = model_factory().to(device)
	y = predict(model, X, batch_size=8, device=device)

	assert y.shape == (64, 1)
	assert y.dtype == torch.float32
	assert_array_almost_equal(y[:4], expected, 4)
	assert_array_almost_equal(y, model(X.to(device)).cpu().detach(), 4)


def test_predict_dropout_conv(X, device):
	torch.manual_seed(0)
	model = DropoutConv().to(device)
	y = predict(model, X, batch_size=8, device=device)

	assert y.shape == (64, 1)
	assert y.dtype == torch.float32
	assert_array_almost_equal(y[:4], [
		[ 0.0921],
		[ 0.1211],
		[-0.1930],
		[-0.0334]], 4)
	# Dropout must be a no-op in eval mode, so two runs must agree exactly.
	assert_array_almost_equal(y, predict(model, X, batch_size=16, device=device))


def test_predict_multi_input_multi_output(X, alpha, beta, device):
	torch.manual_seed(0)
	model = MultiInputMultiOutput().to(device)
	y = predict(model, X, args=(alpha, beta), batch_size=8, device=device)

	assert len(y) == 2
	assert y[0].shape == (64, 12, 98)
	assert y[1].shape == (64, 3)
	assert y[0].dtype == torch.float32
	assert y[1].dtype == torch.float32

	assert_array_almost_equal(y[0][:2, :3, :4], [
		[[ 1.4927,  1.2323,  1.3904,  1.3586],
		 [ 1.4652,  1.8620,  1.6627,  1.5080],
		 [ 1.6393,  1.9074,  1.4530,  1.4420]],

		[[-0.0803,  0.2332, -0.1862,  0.3465],
		 [ 0.0293,  0.0924, -0.1909,  0.4167],
		 [ 0.8397,  0.2444,  0.6099,  0.3073]]], 4)
	assert_array_almost_equal(y[1][:4], [
		[-0.4006,  0.1456,  0.0228],
		[-0.0358, -0.2301, -0.0823],
		[-0.0012, -0.0975,  0.1534],
		[-0.1794, -0.3789, -0.0308]], 4)

	# Batch invariance: running with a different batch size yields the same output.
	y2 = predict(model, X, args=(alpha, beta), batch_size=64, device=device)
	assert_array_almost_equal(y[0], y2[0], 4)
	assert_array_almost_equal(y[1], y2[1], 4)


###
# Low-precision (fp16, bf16) tests. CUDA only: torch.autocast does not support
# these dtypes on CPU for most ops. Hardcoded expected values are captured from
# fresh runs at each dtype, so these guard against future drift in either the
# autocast path or the underlying GPU kernels.


@pytest.mark.parametrize("dtype, decimal", [(torch.float16, 3),
	(torch.bfloat16, 2)], ids=["fp16", "bf16"])
def test_predict_summodel_low_precision(X, dtype, decimal, cuda_device):
	torch.manual_seed(0)
	model = SumModel().to(cuda_device)
	y = predict(model, X, batch_size=8, dtype=dtype, device=cuda_device)

	# torch.autocast promotes reduction ops like .sum() to fp32 for numerical
	# stability, so y comes back in fp32 even when fp16 or bf16 was requested.
	assert y.shape == (64, 4)
	assert y.dtype == torch.float32
	assert_array_almost_equal(y[:4], [
		[25., 24., 19., 32.],
		[26., 18., 29., 27.],
		[28., 25., 21., 26.],
		[21., 33., 19., 27.]], decimal)


@pytest.mark.parametrize("dtype, expected, decimal", [
	(torch.float16, [
		[ 0.1927,  0.2788, -0.1000],
		[-0.0505,  0.0174, -0.1750],
		[-0.1409,  0.0106,  0.0673],
		[-0.2374, -0.0069,  0.0834]], 3),
	(torch.bfloat16, [
		[ 0.1934,  0.2793, -0.1001],
		[-0.0508,  0.0179, -0.1748],
		[-0.1416,  0.0112,  0.0669],
		[-0.2373, -0.0070,  0.0835]], 2)
], ids=["fp16", "bf16"])
def test_predict_flattendense_low_precision(X, dtype, expected, decimal,
	cuda_device):
	torch.manual_seed(0)
	model = FlattenDense().to(cuda_device)
	y = predict(model, X, batch_size=8, dtype=dtype, device=cuda_device)

	assert y.shape == (64, 3)
	assert y.dtype == dtype
	assert next(model.parameters()).dtype == torch.float32
	assert_array_almost_equal(y[:4].float(), expected, decimal)


@pytest.mark.parametrize("dtype, expected, decimal", [
	(torch.float16, [
		[[-0.2712, -0.5317, -0.3735, -0.4053],
		 [-0.2988,  0.0980, -0.1013, -0.2561]],

		[[-0.4805, -0.1667, -0.5859, -0.0536],
		 [-0.3708, -0.3076, -0.5913,  0.0166]]], 3),
	(torch.bfloat16, [
		[[-0.2715, -0.5312, -0.3750, -0.4062],
		 [-0.2988,  0.0977, -0.1016, -0.2559]],

		[[-0.4805, -0.1670, -0.5859, -0.0537],
		 [-0.3711, -0.3086, -0.5898,  0.0156]]], 2)
], ids=["fp16", "bf16"])
def test_predict_conv_low_precision(X, dtype, expected, decimal, cuda_device):
	torch.manual_seed(0)
	model = Conv().to(cuda_device)
	y = predict(model, X, batch_size=8, dtype=dtype, device=cuda_device)

	assert y.shape == (64, 12, 98)
	assert y.dtype == dtype
	assert next(model.parameters()).dtype == torch.float32
	assert_array_almost_equal(y[:2, :2, :4].float(), expected, decimal)


@pytest.mark.parametrize("dtype, decimal", [(torch.float16, 3),
	(torch.bfloat16, 2)], ids=["fp16", "bf16"])
def test_predict_scatter_low_precision(X, dtype, decimal, cuda_device):
	torch.manual_seed(0)
	model = Scatter().to(cuda_device)
	y = predict(model, X, batch_size=8, dtype=dtype, device=cuda_device)

	assert y.shape == (64, 100, 4)
	assert y.dtype == dtype
	# Scatter just permutes, so values are unchanged at any precision.
	assert_array_almost_equal(y[:2, :3, :4].float(), [
		[[1., 0., 0., 0.],
		 [0., 0., 0., 1.],
		 [0., 1., 0., 0.]],

		[[0., 1., 0., 0.],
		 [0., 0., 1., 0.],
		 [1., 0., 0., 0.]]], decimal)


@pytest.mark.parametrize("dtype, expected0, expected1, decimal", [
	(torch.float16, [
		[[-0.7759,  0.0397, -0.8799, -1.0186],
		 [ 0.0947,  0.2042, -0.2301, -0.3147]],

		[[-0.6675, -0.8872, -0.8696, -0.4683],
		 [-0.0768, -0.0090,  0.0243,  0.1355]]], [
		[ 0.1927,  0.2788, -0.1000],
		[-0.0505,  0.0174, -0.1750],
		[-0.1409,  0.0106,  0.0673],
		[-0.2374, -0.0069,  0.0834]], 3),
	(torch.bfloat16, [
		[[-0.7773,  0.0400, -0.8789, -1.0234],
		 [ 0.0942,  0.2041, -0.2314, -0.3145]],

		[[-0.6680, -0.8867, -0.8672, -0.4688],
		 [-0.0771, -0.0089,  0.0238,  0.1357]]], [
		[ 0.1934,  0.2773, -0.1001],
		[-0.0510,  0.0179, -0.1758],
		[-0.1416,  0.0112,  0.0664],
		[-0.2363, -0.0071,  0.0830]], 2)
], ids=["fp16", "bf16"])
def test_predict_convdense_low_precision(X, dtype, expected0, expected1,
	decimal, cuda_device):
	torch.manual_seed(0)
	model = ConvDense().to(cuda_device)
	y = predict(model, X, batch_size=2, dtype=dtype, device=cuda_device)

	assert len(y) == 2
	assert y[0].shape == (64, 12, 98)
	assert y[1].shape == (64, 3)
	assert y[0].dtype == dtype
	assert y[1].dtype == dtype
	assert next(model.parameters()).dtype == torch.float32
	assert_array_almost_equal(y[0][:2, :2, :4].float(), expected0, decimal)
	assert_array_almost_equal(y[1][:4].float(), expected1, decimal)


@pytest.mark.parametrize("model_class, dtype, expected, decimal", [
	(ResidualConv, torch.float16, [
		[-0.4900],
		[ 0.0663],
		[-0.6025],
		[-0.1464]], 3),
	(ResidualConv, torch.bfloat16, [
		[-0.4883],
		[ 0.0664],
		[-0.6016],
		[-0.1475]], 2),
	(Transformer, torch.float16, [
		[-0.5059],
		[ 0.7588],
		[-0.1565],
		[ 0.4629]], 3),
	(Transformer, torch.bfloat16, [
		[-0.5039],
		[ 0.7578],
		[-0.1572],
		[ 0.4629]], 2),
	(Conv2DExpand, torch.float16, [
		[ 0.0658],
		[ 0.0003],
		[-0.0614],
		[-0.0342]], 3),
	(Conv2DExpand, torch.bfloat16, [
		[ 0.0659],
		[ 0.0002],
		[-0.0623],
		[-0.0337]], 2),
	(CustomLinear, torch.float16, [
		[-0.0962],
		[ 0.2407],
		[ 0.1909],
		[ 0.2659]], 3),
	(CustomLinear, torch.bfloat16, [
		[-0.0967],
		[ 0.2412],
		[ 0.1904],
		[ 0.2656]], 2),
	(CustomSqrt, torch.float16, [
		[-0.2020],
		[-0.1216],
		[-0.1644],
		[-0.1508]], 3),
	(CustomSqrt, torch.bfloat16, [
		[-0.2002],
		[-0.1240],
		[-0.1611],
		[-0.1514]], 2),
	(DilatedConv, torch.float16, [
		[0.0682],
		[0.0574],
		[0.0643],
		[0.0378]], 3),
	(DilatedConv, torch.bfloat16, [
		[0.0679],
		[0.0574],
		[0.0645],
		[0.0376]], 2),
	(ConvBatchNorm, torch.float16, [
		[ 0.0439],
		[-0.0023],
		[ 0.0093],
		[ 0.0305]], 3),
	(ConvBatchNorm, torch.bfloat16, [
		[ 0.0439],
		[-0.0020],
		[ 0.0095],
		[ 0.0306]], 2),
	(ConvLayerNorm, torch.float16, [
		[ 0.2172],
		[ 0.3096],
		[-0.6040],
		[-0.1360]], 3),
	(ConvLayerNorm, torch.bfloat16, [
		[ 0.2168],
		[ 0.3105],
		[-0.6055],
		[-0.1367]], 2),
	(MultiActivation, torch.float16, [
		[0.0712],
		[0.0853],
		[0.0731],
		[0.0558]], 3),
	(MultiActivation, torch.bfloat16, [
		[0.0703],
		[0.0854],
		[0.0723],
		[0.0562]], 2),
	(DropoutConv, torch.float16, [
		[ 0.0921],
		[ 0.1210],
		[-0.1930],
		[-0.0334]], 3),
	(DropoutConv, torch.bfloat16, [
		[ 0.0923],
		[ 0.1211],
		[-0.1934],
		[-0.0334]], 2)
], ids=[
	"ResidualConv-fp16", "ResidualConv-bf16", "Transformer-fp16",
	"Transformer-bf16", "Conv2DExpand-fp16", "Conv2DExpand-bf16",
	"CustomLinear-fp16", "CustomLinear-bf16", "CustomSqrt-fp16",
	"CustomSqrt-bf16", "DilatedConv-fp16", "DilatedConv-bf16",
	"ConvBatchNorm-fp16", "ConvBatchNorm-bf16", "ConvLayerNorm-fp16",
	"ConvLayerNorm-bf16", "MultiActivation-fp16", "MultiActivation-bf16",
	"DropoutConv-fp16", "DropoutConv-bf16"])
def test_predict_architectures_low_precision(X, model_class, dtype, expected,
	decimal, cuda_device):
	torch.manual_seed(0)
	model = model_class().to(cuda_device)
	y = predict(model, X, batch_size=8, dtype=dtype, device=cuda_device)

	assert y.shape == (64, 1)
	assert y.dtype == dtype
	assert next(model.parameters()).dtype == torch.float32
	assert_array_almost_equal(y[:4].float(), expected, decimal)


@pytest.mark.parametrize("dtype, expected0, expected1, decimal", [
	(torch.float16, [
		[[ 1.4922,  1.2324,  1.3906,  1.3584],
		 [ 1.4648,  1.8613,  1.6621,  1.5078],
		 [ 1.6387,  1.9072,  1.4531,  1.4414]],

		[[-0.0803,  0.2334, -0.1858,  0.3467],
		 [ 0.0293,  0.0925, -0.1912,  0.4167],
		 [ 0.8398,  0.2444,  0.6099,  0.3071]]], [
		[-0.4006,  0.1458,  0.0229],
		[-0.0359, -0.2302, -0.0822],
		[-0.0012, -0.0975,  0.1534],
		[-0.1794, -0.3792, -0.0308]], 3),
	(torch.bfloat16, [
		[[ 1.4922,  1.2344,  1.3906,  1.3594],
		 [ 1.4688,  1.8594,  1.6641,  1.5078],
		 [ 1.6406,  1.9062,  1.4531,  1.4375]],

		[[-0.0801,  0.2334, -0.1855,  0.3477],
		 [ 0.0293,  0.0918, -0.1895,  0.4160],
		 [ 0.8398,  0.2451,  0.6094,  0.3086]]], [
		[-0.4023,  0.1455,  0.0236],
		[-0.0359, -0.2305, -0.0820],
		[-0.0011, -0.0972,  0.1533],
		[-0.1797, -0.3789, -0.0310]], 2)
], ids=["fp16", "bf16"])
def test_predict_multi_input_multi_output_low_precision(X, alpha, beta, dtype,
	expected0, expected1, decimal, cuda_device):
	torch.manual_seed(0)
	model = MultiInputMultiOutput().to(cuda_device)
	y = predict(model, X, args=(alpha, beta), batch_size=8,
		dtype=dtype, device=cuda_device)

	assert len(y) == 2
	assert y[0].shape == (64, 12, 98)
	assert y[1].shape == (64, 3)
	assert y[0].dtype == dtype
	assert y[1].dtype == dtype
	assert next(model.parameters()).dtype == torch.float32
	assert_array_almost_equal(y[0][:2, :3, :4].float(), expected0, decimal)
	assert_array_almost_equal(y[1][:4].float(), expected1, decimal)


###


@pytest.mark.parametrize("func, shape", [(lambda y: y, (64, 3)),
	(lambda y: y[:, 0], (64,))], ids=["identity", "projection"])
def test_predict_func(X, func, shape, device):
	torch.manual_seed(0)
	model = FlattenDense()

	y_default = predict(model, X, batch_size=8, device=device)
	y_func = predict(model, X, batch_size=8, device=device, func=func)

	assert y_func.shape == shape
	assert_array_almost_equal(y_func, func(y_default))


def test_predict_verbose(X, device):
	torch.manual_seed(0)
	model = FlattenDense()

	y0 = predict(model, X, batch_size=8, device=device, verbose=False)
	y1 = predict(model, X, batch_size=8, device=device, verbose=True)

	assert_array_almost_equal(y0, y1)


def test_predict_empty_X():
	model = FlattenDense()
	X_empty = torch.zeros((0, 4, 100))

	assert_raises(ValueError, predict, model, X_empty, batch_size=8)


def test_predict_batch_size_zero(X, device):
	model = FlattenDense()

	assert_raises(ValueError, predict, model, X, batch_size=0, device=device)


###
# A PackedSequence is decoded a batch at a time and gives the predictions of
# its one-hot encoding.


@pytest.mark.parametrize("model_class", [SumModel, FlattenDense, Conv,
	ConvDense, DilatedConv, Transformer, ConvBatchNorm])
def test_predict_packed(X, model_class, device):
	torch.manual_seed(0)
	model = model_class()
	X = with_N(X)

	y = predict(model, X, batch_size=8, device=device)
	y_packed = predict(model, PackedSequence.from_one_hot(X), batch_size=8,
		device=device)
	assert_same(y, y_packed)


@pytest.mark.parametrize("batch_size", [1, 7, 64, 100])
def test_predict_packed_batch_size(X, batch_size, device):
	torch.manual_seed(0)
	model = ConvDense()
	X = with_N(X)

	y = predict(model, X, batch_size=batch_size, device=device)
	y_packed = predict(model, PackedSequence.from_one_hot(X),
		batch_size=batch_size, device=device)
	assert_same(y, y_packed)


def test_predict_packed_args_func(X, alpha, beta, device):
	torch.manual_seed(0)
	model = MultiInputMultiOutput()
	X_packed = PackedSequence.from_one_hot(X)

	y = predict(model, X, args=(alpha, beta), batch_size=8, device=device,
		func=lambda y: [y_ * 2 for y_ in y])
	y_packed = predict(model, X_packed, args=(alpha, beta), batch_size=8,
		device=device, func=lambda y: [y_ * 2 for y_ in y])
	assert_same(y, y_packed)


def test_predict_packed_fp64(X, device):
	torch.manual_seed(0)
	model = FlattenDense().double()

	y = predict(model, X.double(), batch_size=8, device=device)
	y_packed = predict(model, PackedSequence.from_one_hot(X), batch_size=8,
		device=device)
	assert y.dtype == torch.float64
	assert_same(y, y_packed)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_predict_packed_16bit(X, dtype, cuda_device):
	torch.manual_seed(0)
	model = ConvDense()

	y = predict(model, X, batch_size=8, dtype=dtype, device=cuda_device)
	y_packed = predict(model, PackedSequence.from_one_hot(X), batch_size=8,
		dtype=dtype, device=cuda_device)
	assert_same(y, y_packed)


###
# _to_device copies a batch of integers narrower than dtype before casting
# it, and casts any other batch first. Either way it gives the tensor that
# casting and then copying gives, bit for bit, NaN included.


_FLOAT_VALUES = [0, 1, -1, 2.5, -3.75, 1 / 3, 1e-8, -1e-30, 65504, 70000,
	1e20, -1e20, float('inf'), float('-inf'), float('nan')]


def _source(dtype):
	if dtype == torch.bool:
		return torch.tensor([True, False, True])
	if dtype.is_floating_point:
		return torch.tensor(_FLOAT_VALUES, dtype=torch.float64).to(dtype)
	info = torch.iinfo(dtype)
	return torch.tensor([0, 1, info.max, info.min, info.max // 3], dtype=dtype)


def _bits(X):
	X = X.cpu()
	if X.dtype.is_floating_point:
		return X.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[
			X.element_size()])
	return X


@pytest.mark.parametrize("source", [torch.bool, torch.uint8, torch.int8,
	torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16,
	torch.float32, torch.float64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16,
	torch.float32, torch.float64, torch.int32])
def test_to_device(source, dtype, device):
	X = _source(source)
	X_ = _to_device(X, dtype, torch.device(device))
	assert X_.dtype == dtype
	assert X_.device.type == device
	assert torch.equal(_bits(X_), _bits(X.type(dtype)))


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.bool,
	torch.float16, torch.float64])
def test_predict_stored_dtype(X, alpha, beta, dtype, device):
	# Predictions from sequences and arguments stored in another dtype equal
	# those from the same values given as float32.
	torch.manual_seed(0)
	model = MultiInputMultiOutput()

	y = predict(model, X.type(dtype), args=(alpha.half(), beta.double()),
		batch_size=8, device=device)
	y0 = predict(model, X, args=(alpha.half().float(), beta), batch_size=8,
		device=device)
	assert_same(y, y0)


def test_predict_packed_raises(X):
	model = FlattenDense()
	X_packed = PackedSequence.from_one_hot(X)
	assert_raises(ValueError, predict, model, X_packed[:0])
	assert_raises(ValueError, predict, model, X_packed, args=(X[:5],))
