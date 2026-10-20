# test_saturation_mutagenesis.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import warnings

import torch
import pytest

from tangermeme.predict import predict
from tangermeme.utils import random_one_hot
from tangermeme.utils import PackedSequence
from tangermeme.utils import TangermemeWarning

from tangermeme._fastism import _FastISM
from tangermeme.saturation_mutagenesis import _attribution_score
from tangermeme.saturation_mutagenesis import saturation_mutagenesis

from .toy_models import SumModel
from .toy_models import FlattenDense
from .toy_models import Conv1
from .toy_models import Scatter
from .toy_models import ConvDense
from .toy_models import ConvPoolDense
from .toy_models import SmallDeepSEA
from .toy_models import ResidualConv
from .toy_models import DilatedConv
from .toy_models import ConvBatchNorm
from .toy_models import ConvLayerNorm
from .toy_models import ConvGLU
from .toy_models import ConvSoftmax
from .toy_models import MultiHeadAttention
from .toy_models import Conv2DExpand
from .toy_models import ProfileCount
from .toy_models import StridedPoolConv
from .toy_models import Conv2DLayouts
from .toy_models import ChannelsLastConv
from .toy_models import ReverseComplementAverage
from .toy_models import InceptionConv
from .toy_models import PadCropConv
from .toy_models import MixedOps
from .toy_models import ConvWithArgs
from .toy_models import DataDependentConv
from .toy_models import MergedBatchLength
from .toy_models import ReorderedChannels

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


@pytest.fixture
def X():
	return random_one_hot((2, 4, 10), random_state=0)


@pytest.fixture
def X0():
	return random_one_hot((2, 4, 100), random_state=0).float()


###


@pytest.mark.parametrize("offset", [0, -5])
def test_attribution_score(offset):
	torch.manual_seed(0)
	y0 = torch.zeros(1, 1) + offset
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


def test_saturation_mutagenesis(X0, device):
	torch.manual_seed(0)
	model = SmallDeepSEA(5)
	X_attr = saturation_mutagenesis(model, X0, device=device)

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


def test_saturation_mutagenesis_hypothetical(X0, device):
	torch.manual_seed(0)
	model = SmallDeepSEA(5)
	X_attr = saturation_mutagenesis(model, X0, hypothetical=True, device=device)
	X_attr2 = saturation_mutagenesis(model, X0, device=device)

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


@pytest.mark.parametrize("start, end, hypothetical, length", [
	(50, 60, False, 10),
	(50, 60, True, 10),
	(50, None, False, 50)
])
def test_saturation_mutagenesis_start_end(X0, start, end, hypothetical,
	length, device):
	# end=None leaves end at its default, -1, which maps to the length.
	torch.manual_seed(0)
	model = SmallDeepSEA(5)
	kwargs = {} if end is None else {"end": end}

	X_attr = saturation_mutagenesis(model, X0, start=start,
		hypothetical=hypothetical, device=device, **kwargs)
	X_attr2 = saturation_mutagenesis(model, X0, hypothetical=hypothetical,
		device=device)

	assert X_attr.shape == (2, 4, length)
	assert X_attr.dtype == torch.float32

	if hypothetical:
		assert_array_almost_equal(X_attr[:, :, :3], [
			[[ 6.7323e-04,  4.7941e-04,  7.2884e-04],
	         [ 1.3862e-05,  2.4718e-04,  7.1416e-04],
	         [-3.4264e-04, -2.4547e-04, -9.5634e-04],
	         [-3.4445e-04, -4.8112e-04, -4.8665e-04]],

	        [[-1.6091e-03,  2.2756e-04,  1.2385e-03],
	         [ 1.9696e-04,  6.4452e-04,  6.1850e-04],
	         [ 5.6783e-04, -7.1807e-04, -1.4681e-03],
	         [ 8.4433e-04, -1.5401e-04, -3.8887e-04]]], 4)
	else:
		assert_array_almost_equal(X_attr[:, :, :3], [
			[[ 0.0007,  0.0000,  0.0000],
	         [ 0.0000,  0.0002,  0.0007],
	         [-0.0000, -0.0000, -0.0000],
	         [-0.0000, -0.0000, -0.0000]],

	        [[-0.0016,  0.0000,  0.0000],
	         [ 0.0000,  0.0000,  0.0000],
	         [ 0.0000, -0.0007, -0.0015],
	         [ 0.0000, -0.0000, -0.0000]]], 4)

	assert_array_almost_equal(X_attr, X_attr2[:, :, start:end], 4)


def test_saturation_mutagenesis_ordering(X0, device):
	torch.manual_seed(0)
	model = SmallDeepSEA(5)
	X_attr = saturation_mutagenesis(model, X0, device=device)

	assert X_attr.shape == (2, 4, 100)
	assert X_attr.dtype == torch.float32

	X_attr2 = saturation_mutagenesis(model, X0[1:], device=device)
	assert_array_almost_equal(X_attr[1:, :, :], X_attr2, 4)


def test_saturation_mutagenesis_raw_output(X0, device):
	torch.manual_seed(0)
	model = SmallDeepSEA(5)
	y0, y_hat = saturation_mutagenesis(model, X0, raw_outputs=True, 
		device=device)

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


def test_saturation_mutagenesis_equivalence(X0, device):
	torch.manual_seed(0)
	model = SmallDeepSEA(5)

	X_attr = saturation_mutagenesis(model, X0, hypothetical=True, device=device)
	y0, y_hat = saturation_mutagenesis(model, X0, raw_outputs=True, 
		device=device)

	attr = _attribution_score(y0, y_hat, None)
	assert_array_almost_equal(X_attr, attr, 4)


def test_saturation_mutagenesis_sum_model(X, device):
	model = SumModel()
	y0, y_hat = saturation_mutagenesis(model, X, raw_outputs=True, device=device)

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


###


def test_saturation_mutagenesis_rejects_empty_X():
	# Routes through predict's empty-input check.
	model = FlattenDense()
	X = torch.zeros(0, 4, 100, dtype=torch.float32)
	with pytest.raises(ValueError, match="at least one example"):
		saturation_mutagenesis(model, X)


def test_saturation_mutagenesis_verbose(X0, device):
	model = SmallDeepSEA(1)

	X_attr0 = saturation_mutagenesis(model, X0, device=device, verbose=False)
	X_attr1 = saturation_mutagenesis(model, X0, device=device, verbose=True)

	assert_array_almost_equal(X_attr0, X_attr1)


@pytest.mark.parametrize("kwargs", [{}, {"start": 30, "end": 70}],
	ids=["full", "windowed"])
def test_saturation_mutagenesis_identity_recovers_y0(X0, kwargs, device):
	model = SmallDeepSEA(1)
	start = kwargs.get("start", 0)
	end = kwargs.get("end", X0.shape[-1])

	y0, y_hat = saturation_mutagenesis(model, X0, raw_outputs=True,
		device=device, **kwargs)

	# Identity edits are filled from y0 rather than predicted, so they must
	# equal y0 exactly, also when the window offsets the identity mask.
	for i in range(X0.shape[0]):
		for pos in range(end - start):
			orig_base = int(X0[i, :, start + pos].argmax())
			assert torch.equal(y_hat[i, orig_base, pos], y0[i])


@pytest.mark.parametrize("pos, bases", [(50, []), (30, [1, 2])],
	ids=["all_zero", "multi_hot"])
def test_saturation_mutagenesis_irregular_column(pos, bases, device):
	# An all-zero (N) or multi-hot column has no identity edit, so all four of
	# its edits must be predicted rather than filled from y0.
	torch.manual_seed(0)
	X = random_one_hot((1, 4, 100), random_state=0).float()
	X[0, :, pos] = 0
	for base in bases:
		X[0, base, pos] = 1
	model = SmallDeepSEA(1)

	y0, y_hat = saturation_mutagenesis(model, X, raw_outputs=True,
		device=device)

	for base in range(4):
		X_sub = X.clone()
		X_sub[0, :, pos] = 0
		X_sub[0, base, pos] = 1
		y_sub = predict(model, X_sub, device=device)
		assert_array_almost_equal(y_hat[0, base, pos], y_sub[0], 4)


def test_saturation_mutagenesis_matches_bruteforce(device):
	# Checks the vectorized edit construction one slot at a time.
	torch.manual_seed(0)
	X = random_one_hot((1, 4, 12), random_state=0).float()
	model = FlattenDense(seq_len=12, n_outputs=1)

	y0, y_hat = saturation_mutagenesis(model, X, raw_outputs=True,
		device=device)

	for base in range(4):
		for pos in range(12):
			X_sub = X.clone()
			X_sub[0, :, pos] = 0
			X_sub[0, base, pos] = 1
			y_sub = predict(model, X_sub, device=device)
			assert_array_almost_equal(y_hat[0, base, pos], y_sub[0], 4)


def test_saturation_mutagenesis_bruteforce_windowed_with_N(device):
	# The window offsets the edit positions, and its N column varies the number
	# of edits kept per position.
	torch.manual_seed(0)
	X = random_one_hot((1, 4, 20), random_state=0).float()
	X[0, :, 8] = 0                      # N column inside the window
	model = FlattenDense(seq_len=20, n_outputs=1)
	start, end = 5, 12

	y0, y_hat = saturation_mutagenesis(model, X, start=start, end=end,
		raw_outputs=True, device=device)
	assert y_hat.shape == (1, 4, end - start, 1)

	for base in range(4):
		for pos in range(end - start):
			X_sub = X.clone()
			X_sub[0, :, start + pos] = 0
			X_sub[0, base, start + pos] = 1
			y_sub = predict(model, X_sub, device=device)
			assert_array_almost_equal(y_hat[0, base, pos], y_sub[0], 4)


def test_saturation_mutagenesis_zero_where_X_zero(X0, device):
	model = SmallDeepSEA(1)
	X_attr = saturation_mutagenesis(model, X0, device=device,
		hypothetical=False)

	# Non-hypothetical attribution masks by X, so zeros stay zero.
	zero_mask = (X0 == 0)
	assert torch.all(X_attr[zero_mask] == 0)


def test_saturation_mutagenesis_raw_outputs_named_tuple(device):
	from tangermeme.saturation_mutagenesis import SaturationMutagenesisRawResult

	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)
	X = random_one_hot((2, 4, 100), random_state=0).type(torch.float32)

	result = saturation_mutagenesis(model, X, raw_outputs=True, device=device)

	y0, y_hat = result
	assert torch.equal(result.y0, y0)
	assert torch.equal(result.y_hat, y_hat)
	assert isinstance(result, SaturationMutagenesisRawResult)
	assert isinstance(result, tuple)

	attr = saturation_mutagenesis(model, X, raw_outputs=False, device=device)
	assert isinstance(attr, torch.Tensor)


###


class _DenseHead(torch.nn.Module):
	"""Single-tensor model sharing the dense head of a ConvDense instance.

	Used to cross-check the multi-tensor reshape branch in
	`saturation_mutagenesis` against ConvDense's second output.
	"""

	def __init__(self, src):
		super(_DenseHead, self).__init__()
		self.dense = src.dense

	def forward(self, X):
		return self.dense(X.reshape(X.shape[0], -1))


def test_saturation_mutagenesis_multi_output(device):
	torch.manual_seed(0)
	X = random_one_hot((2, 4, 100), random_state=0).float()

	model = ConvDense(n_outputs=3)
	y0, y_hat = saturation_mutagenesis(model, X, raw_outputs=True,
		device=device)

	assert isinstance(y0, list) and isinstance(y_hat, list)
	assert len(y0) == 2 and len(y_hat) == 2
	assert y0[0].shape == (2, 12, 98)
	assert y0[1].shape == (2, 3)
	assert y_hat[0].shape == (2, 4, 100, 12, 98)
	assert y_hat[1].shape == (2, 4, 100, 3)

	# Catches the reshape scrambling the alphabet and position axes.
	y0_s, y_hat_s = saturation_mutagenesis(_DenseHead(model), X,
		raw_outputs=True, device=device)
	assert_array_almost_equal(y_hat[1], y_hat_s, 4)
	assert_array_almost_equal(y0[1], y0_s, 4)

	# Identity substitutions recover y0 on both heads.
	for head_hat, head0 in zip(y_hat, y0):
		for i in range(X.shape[0]):
			for pos in range(X.shape[-1]):
				ob = int(X[i, :, pos].argmax())
				assert_array_almost_equal(head_hat[i, ob, pos], head0[i], 4)


def test_saturation_mutagenesis_multi_output_start_end(device):
	# The multi-tensor reshape used to raise when start > 0 or end < length.
	torch.manual_seed(0)
	X = random_one_hot((2, 4, 100), random_state=0).float()

	model = ConvDense(n_outputs=3)
	y0, y_hat = saturation_mutagenesis(model, X, start=50, end=60,
		raw_outputs=True, device=device)

	assert y_hat[0].shape == (2, 4, 10, 12, 98)
	assert y_hat[1].shape == (2, 4, 10, 3)

	y0_s, y_hat_s = saturation_mutagenesis(_DenseHead(model), X, start=50,
		end=60, raw_outputs=True, device=device)
	assert_array_almost_equal(y_hat[1], y_hat_s, 4)


def test_attribution_score_target_slice():
	torch.manual_seed(0)
	y0 = torch.zeros(1, 3)
	y_hat = torch.randn(1, 4, 10, 3)

	attr_slice = _attribution_score(y0, y_hat, slice(0, 2))
	attr0 = _attribution_score(y0, y_hat, 0)
	attr1 = _attribution_score(y0, y_hat, 1)

	assert attr_slice.shape == (1, 4, 10)
	assert_array_almost_equal(attr_slice, (attr0 + attr1) / 2, 4)


def test_saturation_mutagenesis_dtype_arg(X0, device):
	torch.manual_seed(0)
	model = SmallDeepSEA(1)

	X_attr = saturation_mutagenesis(model, X0, dtype=torch.float32,
		device=device)
	X_attr_default = saturation_mutagenesis(model, X0, device=device)

	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr, X_attr_default, 4)


def test_saturation_mutagenesis_args_batch_mismatch(X0):
	# args with a mismatched batch dimension must surface predict's check.
	model = FlattenDense(seq_len=100, n_outputs=1)
	with pytest.raises(ValueError, match="same first dimension"):
		saturation_mutagenesis(model, X0, args=(torch.randn(5, 1),),
			raw_outputs=True)


@pytest.mark.parametrize("start, end", [(0, 0), (0, 105), (-3, 8), (8, 3)],
	ids=["empty", "end_past_length", "negative_start", "start_after_end"])
def test_saturation_mutagenesis_raises_bad_start_end(X0, start, end):
	# X0 has length 100. A negative end is remapped from the end of the
	# sequence, but a negative start is not.
	model = SmallDeepSEA(1)
	with pytest.raises(ValueError, match="0 <= start < end <= length"):
		saturation_mutagenesis(model, X0, start=start, end=end,
			raw_outputs=True)


@pytest.mark.parametrize("dtype", [torch.int8, torch.float64, torch.int32,
	torch.bool], ids=["int8", "float64", "int32", "bool"])
def test_saturation_mutagenesis_input_dtypes(X0, dtype, device):
	# X is cast to int8 internally, so any encoding of a hard one-hot must give
	# identical results.
	torch.manual_seed(0)
	model = SmallDeepSEA(1)

	X_attr = saturation_mutagenesis(model, X0, device=device)
	X_attr_cast = saturation_mutagenesis(model, X0.type(dtype), device=device)
	assert torch.equal(X_attr, X_attr_cast)


# Attributions are not compared at low precision: their sub-1e-3 values are
# dominated by fp16 rounding noise.


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
	ids=["fp16", "bf16"])
def test_saturation_mutagenesis_low_precision(X0, dtype, cuda_device):
	torch.manual_seed(0)
	model = FlattenDense(seq_len=100, n_outputs=1).to(cuda_device)

	y0, y_hat = saturation_mutagenesis(model, X0, dtype=dtype,
		raw_outputs=True, device=cuda_device)

	assert y_hat.shape == (2, 4, 100, 1)
	assert y_hat.dtype == dtype
	assert torch.isfinite(y_hat).all()

	for i in range(X0.shape[0]):
		for pos in range(X0.shape[-1]):
			ob = int(X0[i, :, pos].argmax())
			assert_array_almost_equal(y_hat[i, ob, pos].float(),
				y0[i].float(), 2)


def test_saturation_mutagenesis_args_applied_per_example(X0, device):
	# FlattenDense adds alpha to its output, so alpha[i] must shift y0 and every
	# edit of example i, which leaves the attributions unchanged.
	torch.manual_seed(0)
	model = FlattenDense(seq_len=100, n_outputs=1)
	alpha = torch.randn(2, 1)

	y0_w, y_hat_w = saturation_mutagenesis(model, X0, args=(alpha,),
		raw_outputs=True, device=device)
	y0_n, y_hat_n = saturation_mutagenesis(model, X0, raw_outputs=True,
		device=device)

	assert_array_almost_equal(y0_w - y0_n, alpha, 4)
	for i in range(X0.shape[0]):
		delta = y_hat_w[i] - y_hat_n[i]
		assert_array_almost_equal(delta, torch.full_like(delta, float(alpha[i])),
			4)

	attr_w = saturation_mutagenesis(model, X0, args=(alpha,), device=device)
	attr_n = saturation_mutagenesis(model, X0, device=device)
	assert_array_almost_equal(attr_w, attr_n, 4)


def test_saturation_mutagenesis_batch_size_invariance(X0, device):
	torch.manual_seed(0)
	model = SmallDeepSEA(1)

	a1 = saturation_mutagenesis(model, X0, batch_size=1, device=device)
	a32 = saturation_mutagenesis(model, X0, batch_size=32, device=device)
	a_big = saturation_mutagenesis(model, X0, batch_size=10000, device=device)

	assert_array_almost_equal(a1, a32, 4)
	assert_array_almost_equal(a1, a_big, 4)


def test_saturation_mutagenesis_length_one(device):
	torch.manual_seed(0)
	X = random_one_hot((2, 4, 1), random_state=0).float()
	model = FlattenDense(seq_len=1, n_outputs=1)

	X_attr = saturation_mutagenesis(model, X, device=device)
	assert X_attr.shape == (2, 4, 1)

	y0, y_hat = saturation_mutagenesis(model, X, raw_outputs=True,
		device=device)
	assert torch.equal(y0, predict(model, X, device=device))

	for base in range(4):
		X_sub = torch.zeros_like(X)
		X_sub[:, base] = 1
		assert torch.equal(y_hat[:, base, 0], predict(model, X_sub,
			device=device))

	attr = y_hat[:, :, 0, 0] - y0
	attr = attr - attr.mean(dim=1, keepdim=True)
	assert_array_almost_equal(X_attr[:, :, 0], attr * X[:, :, 0], 6)


def test_saturation_mutagenesis_multi_output_requires_raw(X0, device):
	# Attributions are computed only for a single output tensor.
	model = ConvDense(n_outputs=3)
	with pytest.raises(ValueError, match="raw_outputs=True is required"):
		saturation_mutagenesis(model, X0, device=device)


def test_saturation_mutagenesis_func_selects_head(X0, device):
	# func is forwarded to predict and applied to both reference and perturbed
	# predictions, so selecting a head with func matches target= on that head.
	torch.manual_seed(0)
	model = FlattenDense(seq_len=100, n_outputs=3)

	attr_func = saturation_mutagenesis(model, X0, func=lambda y: y[:, 1:2],
		device=device)
	attr_target = saturation_mutagenesis(model, X0, target=1, device=device)
	assert_array_almost_equal(attr_func, attr_target, 4)


def test_saturation_mutagenesis_func_collapses_multi_output(X0, device):
	torch.manual_seed(0)
	model = ConvDense(n_outputs=3)

	attr = saturation_mutagenesis(model, X0, func=lambda ys: ys[1],
		device=device)
	assert attr.shape == (2, 4, 100)
	assert torch.equal(attr, saturation_mutagenesis(_DenseHead(model), X0,
		device=device))


def test_saturation_mutagenesis_warns_on_non_integer_input(device):
	# The int8 cast would truncate these values to zero.
	torch.manual_seed(0)
	X = random_one_hot((2, 4, 100), random_state=0).float()
	model = SmallDeepSEA(1)

	with pytest.warns(TangermemeWarning, match="non-integer values"):
		saturation_mutagenesis(model, X * 0.5, device=device)


def test_saturation_mutagenesis_skips_identity_forward_passes(X0, device):
	class _Counter(torch.nn.Module):
		def __init__(self, model):
			super(_Counter, self).__init__()
			self.model = model
			self.rows = 0

		def forward(self, X, *args):
			self.rows += X.shape[0]
			return self.model(X, *args)

	model = _Counter(SmallDeepSEA(1))
	saturation_mutagenesis(model, X0, device=device)

	n, length = X0.shape[0], X0.shape[-1]
	# n reference rows + 3 true edits per position per example.
	assert model.rows == n + 3 * length * n


###


FAST_MODELS = {
	"SumModel": SumModel,
	"FlattenDense": FlattenDense,
	"Conv1": Conv1,
	"Scatter": Scatter,
	"ConvDense": ConvDense,
	"ConvPoolDense": ConvPoolDense,
	"SmallDeepSEA": SmallDeepSEA,
	"ResidualConv": ResidualConv,
	"DilatedConv": DilatedConv,
	"ConvBatchNorm": ConvBatchNorm,
	"ConvLayerNorm": ConvLayerNorm,
	"ConvGLU": ConvGLU,
	"ConvSoftmax": ConvSoftmax,
	"MultiHeadAttention": MultiHeadAttention,
	"Conv2DExpand": Conv2DExpand,
	"ProfileCount": ProfileCount,
	"StridedPoolConv": StridedPoolConv,
	"Conv2DLayouts": Conv2DLayouts,
	"ChannelsLastConv": ChannelsLastConv,
	"ReverseComplementAverage": lambda: ReverseComplementAverage(Conv1()),
	"InceptionConv": InceptionConv,
	"PadCropConv": PadCropConv,
	"MixedOps": MixedOps,
	"MergedBatchLength": MergedBatchLength,
	"ReorderedChannels": ReorderedChannels,
}


@pytest.fixture
def force_fast(monkeypatch):
	# Attempt the fast path however small the job, and keep it however slow
	# it is, so that a toy model exercises it.
	monkeypatch.setattr(
		"tangermeme.saturation_mutagenesis._FAST_MIN_SECONDS", 0)
	monkeypatch.setattr(
		"tangermeme.saturation_mutagenesis._FAST_MAX_RELATIVE_TIME",
		float("inf"))


def _fast_and_slow(model, X, **kwargs):
	"""Run with fast=True, failing if it falls back, and with fast=False."""

	with warnings.catch_warnings():
		warnings.simplefilter("error", TangermemeWarning)
		y_fast = saturation_mutagenesis(model, X, fast=True, **kwargs)

	y_slow = saturation_mutagenesis(model, X, fast=False, **kwargs)
	return y_fast, y_slow


def _assert_close(a, b, rtol=1e-4, atol=1e-5):
	if isinstance(a, torch.Tensor):
		assert a.dtype == b.dtype
		torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
	else:
		assert type(a) is type(b) and len(a) == len(b)
		for a_, b_ in zip(a, b):
			_assert_close(a_, b_, rtol=rtol, atol=atol)


@pytest.mark.parametrize("name", FAST_MODELS)
def test_saturation_mutagenesis_fast_matches(X0, device, force_fast, name):
	torch.manual_seed(0)
	model = FAST_MODELS[name]()

	y_fast, y_slow = _fast_and_slow(model, X0, raw_outputs=True,
		batch_size=17, device=device)
	_assert_close(y_fast, y_slow)


@pytest.mark.parametrize("name, ops", [
	("SmallDeepSEA", ["aten.conv1d.padding", "aten.max_pool1d.default"]),
	("ConvBatchNorm", ["aten._native_batch_norm_legit_no_training.default"]),
	("ProfileCount", ["aten.add.Tensor", "aten.slice.Tensor"]),
	("StridedPoolConv", ["aten.max_pool1d.default",
		"aten.avg_pool1d.default"]),
	("Conv2DLayouts", ["aten.conv2d.default", "aten.max_pool2d.default",
		"aten.squeeze.dim", "aten.unsqueeze.default"]),
	("ChannelsLastConv", ["aten.layer_norm.default", "aten.linear.default",
		"aten.softmax.int", "aten.transpose.int"]),
	("ReverseComplementAverage", ["aten.flip.default"]),
	("InceptionConv", ["aten.cat.default", "aten.add.Tensor"]),
	("PadCropConv", ["aten.pad.default", "aten.slice.Tensor"]),
	("MixedOps", ["aten.embedding.default", "aten.argmax.default",
		"aten.split.Tensor", "aten.view.default", "aten.max.dim",
		"aten.expand.default", "aten.sum.dim_IntList"]),
	("MergedBatchLength", ["aten.rms_norm.default"]),
	("ReorderedChannels", ["aten.index.Tensor", "aten.zeros_like.default",
		"aten.conv1d.default"]),
])
def test_saturation_mutagenesis_fast_local_ops(X0, name, ops):
	# Each rule is exercised by at least one model, so that an equivalence
	# test cannot pass by computing everything at full length.
	torch.manual_seed(0)
	model = FAST_MODELS[name]().eval()
	X = X0.type(torch.int8)

	with torch.no_grad():
		fast = _FastISM(model, X, None, 0, 100, torch.device('cpu'),
			torch.float32, False)

	for op in ops:
		assert op in fast.local_ops


def test_saturation_mutagenesis_fast_global_ops(X0):
	# A layer norm over the length axis and a flatten into a dense layer
	# change every position, so they are computed at full length.
	torch.manual_seed(0)
	X = X0.type(torch.int8)

	with torch.no_grad():
		fast = _FastISM(ConvLayerNorm().eval(), X, None, 0, 100,
			torch.device('cpu'), torch.float32, False)
	assert fast.local_ops == ["aten.conv1d.padding"]

	with torch.no_grad():
		fast = _FastISM(FlattenDense().eval(), X, None, 0, 100,
			torch.device('cpu'), torch.float32, False)
	assert fast.local_ops == []


@pytest.mark.parametrize("start, end", [(0, 10), (90, -1), (40, 41),
	(0, 1), (99, 100), (5, 95)])
def test_saturation_mutagenesis_fast_start_end(X0, device, force_fast, start,
	end):
	torch.manual_seed(0)
	model = ProfileCount()

	y_fast, y_slow = _fast_and_slow(model, X0, start=start, end=end,
		raw_outputs=True, device=device)
	_assert_close(y_fast, y_slow)


@pytest.mark.parametrize("batch_size", [1, 2, 7, 1000])
def test_saturation_mutagenesis_fast_batch_size(X0, device, force_fast,
	batch_size):
	torch.manual_seed(0)
	model = StridedPoolConv()

	y_fast, y_slow = _fast_and_slow(model, X0, batch_size=batch_size,
		raw_outputs=True, device=device)
	_assert_close(y_fast, y_slow)


@pytest.mark.parametrize("target, hypothetical", [(None, False), (0, False),
	(slice(1, 3), True), (2, True)])
def test_saturation_mutagenesis_fast_attributions(X0, device, force_fast,
	target, hypothetical):
	torch.manual_seed(0)
	model = SmallDeepSEA(n_outputs=3)

	X_attr_fast, X_attr_slow = _fast_and_slow(model, X0, target=target,
		hypothetical=hypothetical, device=device)
	assert X_attr_fast.shape == (2, 4, 100)
	_assert_close(X_attr_fast, X_attr_slow)


def test_saturation_mutagenesis_fast_func(X0, device, force_fast):
	torch.manual_seed(0)
	model = ProfileCount()

	X_attr_fast, X_attr_slow = _fast_and_slow(model, X0,
		func=lambda y: y[0].sum(dim=-1), device=device)
	_assert_close(X_attr_fast, X_attr_slow)


def test_saturation_mutagenesis_fast_args(X0, device, force_fast):
	torch.manual_seed(0)
	X_ctl = torch.rand(2, 2, 100)

	y_fast, y_slow = _fast_and_slow(ProfileCount(n_control_tracks=2), X0,
		args=(X_ctl,), raw_outputs=True, batch_size=13, device=device)
	_assert_close(y_fast, y_slow)

	scale = torch.rand(2, 5)
	y_fast, y_slow = _fast_and_slow(ConvWithArgs(), X0, args=(X_ctl, scale),
		raw_outputs=True, device=device)
	_assert_close(y_fast, y_slow)


def test_saturation_mutagenesis_fast_irregular_columns(X0, device,
	force_fast):
	torch.manual_seed(0)
	X = X0.clone()
	X[0, :, 0] = 0
	X[0, :, 50] = 0
	X[1, :, 99] = 1

	y_fast, y_slow = _fast_and_slow(ProfileCount(), X, raw_outputs=True,
		device=device)
	_assert_close(y_fast, y_slow)


def test_saturation_mutagenesis_fast_int8(X0, device, force_fast):
	torch.manual_seed(0)
	model = InceptionConv()

	with warnings.catch_warnings():
		warnings.simplefilter("error", TangermemeWarning)
		y_int8 = saturation_mutagenesis(model, X0.type(torch.int8),
			raw_outputs=True, device=device)
		y_float = saturation_mutagenesis(model, X0, raw_outputs=True,
			device=device)

	_assert_close(y_int8, y_float, rtol=0, atol=0)


def test_saturation_mutagenesis_fast_float64(X0, force_fast):
	torch.manual_seed(0)
	model = ProfileCount().double()

	y_fast, y_slow = _fast_and_slow(model, X0, raw_outputs=True,
		dtype=torch.float64, device='cpu')
	_assert_close(y_fast, y_slow, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("name", ["ProfileCount", "ReorderedChannels"])
def test_saturation_mutagenesis_fast_autocast(X0, cuda_device, force_fast,
	dtype, name):
	# ReorderedChannels casts its input to float32, a cast that only shows up
	# in the captured graph when it is captured in the dtype autocast uses.
	torch.manual_seed(0)
	model = FAST_MODELS[name]()

	y_fast, y_slow = _fast_and_slow(model, X0, raw_outputs=True, dtype=dtype,
		device=cuda_device)
	_assert_close(y_fast, y_slow, rtol=5e-2, atol=5e-2)


def test_saturation_mutagenesis_fast_restores_model(X0, device, force_fast):
	torch.manual_seed(0)
	model = MixedOps().train()
	state = {k: v.clone() for k, v in model.state_dict().items()}

	_fast_and_slow(model, X0, raw_outputs=True, device=device)

	assert model.training
	for k, v in model.state_dict().items():
		assert v.device.type == 'cpu'
		assert torch.equal(v, state[k])


def test_saturation_mutagenesis_fast_verbose(X0, force_fast):
	torch.manual_seed(0)
	model = SmallDeepSEA()

	y_fast, y_slow = _fast_and_slow(model, X0, verbose=True, device='cpu')
	_assert_close(y_fast, y_slow)


##


def _raises_on_capture(*args, **kwargs):
	raise AssertionError("_FastISM should not be constructed")


def test_saturation_mutagenesis_fast_skips_small_jobs(X0, device,
	monkeypatch):
	# A job that fast=False finishes in well under _FAST_MIN_SECONDS is not
	# worth capturing the model for, so it runs with fast=False, silently.
	monkeypatch.setattr("tangermeme.saturation_mutagenesis._FastISM",
		_raises_on_capture)
	torch.manual_seed(0)
	model = SmallDeepSEA()

	with warnings.catch_warnings():
		warnings.simplefilter("error", TangermemeWarning)
		y_fast = saturation_mutagenesis(model, X0, raw_outputs=True,
			device=device)

	y_slow = saturation_mutagenesis(model, X0, raw_outputs=True, fast=False,
		device=device)
	_assert_close(y_fast, y_slow, rtol=0, atol=0)


def test_saturation_mutagenesis_fast_false_never_captures(X0, device,
	force_fast, monkeypatch):
	monkeypatch.setattr("tangermeme.saturation_mutagenesis._FastISM",
		_raises_on_capture)
	torch.manual_seed(0)

	with warnings.catch_warnings():
		warnings.simplefilter("error", TangermemeWarning)
		saturation_mutagenesis(SmallDeepSEA(), X0, fast=False, device=device)


def _assert_falls_back(model, X, match, device):
	with pytest.warns(TangermemeWarning, match=match):
		y_fast = saturation_mutagenesis(model, X, raw_outputs=True,
			device=device)

	y_slow = saturation_mutagenesis(model, X, raw_outputs=True, fast=False,
		device=device)
	_assert_close(y_fast, y_slow, rtol=0, atol=0)


def test_saturation_mutagenesis_fast_fallback_capture(X0, device,
	force_fast, capsys):
	torch.manual_seed(0)
	_assert_falls_back(DataDependentConv(), X0,
		"torch.export could not capture the model", device)

	# torch.export prints the partial graph of a failed capture, which the
	# warning replaces.
	captured = capsys.readouterr()
	assert "def forward" not in captured.err
	assert "def forward" not in captured.out


def test_saturation_mutagenesis_fast_fallback_hooks(X0, device, force_fast):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	model.conv1.register_forward_hook(lambda module, inputs, output:
		output * 2)

	_assert_falls_back(model, X0, "forward hooks", device)


def test_saturation_mutagenesis_fast_fallback_slower(X0, device, force_fast,
	monkeypatch):
	monkeypatch.setattr(
		"tangermeme.saturation_mutagenesis._FAST_MAX_RELATIVE_TIME", 0)
	torch.manual_seed(0)

	_assert_falls_back(SmallDeepSEA(), X0, "fast=True took", device)


def test_saturation_mutagenesis_fast_fallback_mismatch(X0, device,
	force_fast, monkeypatch):
	predict_edits = _FastISM.predict_edits
	monkeypatch.setattr(_FastISM, "predict_edits", lambda *args, **kwargs:
		predict_edits(*args, **kwargs) + 1e-2)
	torch.manual_seed(0)

	_assert_falls_back(SmallDeepSEA(), X0, "differed by up to", device)


def test_saturation_mutagenesis_fast_fallback_error(X0, device, force_fast,
	monkeypatch):
	def predict_edits(*args, **kwargs):
		raise RuntimeError("an unexpected failure")

	monkeypatch.setattr(_FastISM, "predict_edits", predict_edits)
	torch.manual_seed(0)

	_assert_falls_back(SmallDeepSEA(), X0,
		"raised RuntimeError: an unexpected failure", device)


###
# A PackedSequence gives the attributions of its one-hot encoding.


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("kwargs", [{}, {"hypothetical": True}, {"start": 20,
	"end": 60}, {"raw_outputs": True}])
def test_saturation_mutagenesis_packed(X0, fast, kwargs, device, force_fast):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	X0 = X0.clone()
	X0[1, :, 30:33] = 0

	# force_fast makes fast=True use the fast path, which warns if it cannot.
	with warnings.catch_warnings():
		warnings.simplefilter("error", TangermemeWarning)
		y = saturation_mutagenesis(model, X0, fast=fast, device=device,
			**kwargs)
		y_packed = saturation_mutagenesis(model,
			PackedSequence.from_one_hot(X0), fast=fast, device=device, **kwargs)

	if kwargs.get("raw_outputs"):
		assert torch.equal(y_packed.y0, y.y0)
		assert torch.equal(y_packed.y_hat, y.y_hat)
	else:
		assert y_packed.dtype == y.dtype
		assert torch.equal(y_packed, y)
