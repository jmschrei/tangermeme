# test_saturation_mutagenesis.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import warnings

import torch
import pytest

from tangermeme.predict import predict
from tangermeme.utils import random_one_hot
from tangermeme.utils import TangermemeWarning

from tangermeme.saturation_mutagenesis import _attribution_score
from tangermeme.saturation_mutagenesis import saturation_mutagenesis

from .toy_models import SumModel
from .toy_models import FlattenDense
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


def test_saturation_mutagenesis_start_end(X0, device):
	torch.manual_seed(0)
	model = SmallDeepSEA(5)
	X_attr = saturation_mutagenesis(model, X0, start=50, end=60, device=device)
	X_attr2 = saturation_mutagenesis(model, X0, device=device)

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


def test_saturation_mutagenesis_start_end_hypothetical(X0, device):
	torch.manual_seed(0)
	model = SmallDeepSEA(5)
	X_attr = saturation_mutagenesis(model, X0, start=50, end=60, 
		hypothetical=True, device=device)
	X_attr2 = saturation_mutagenesis(model, X0, hypothetical=True, device=device)

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


def test_saturation_mutagenesis_ordering(X0, device):
	torch.manual_seed(0)
	model = SmallDeepSEA(5)
	X_attr = saturation_mutagenesis(model, X0, device=device)

	assert X_attr.shape == (2, 4, 100)
	assert X_attr.dtype == torch.float32

	X_attr2 = saturation_mutagenesis(model, X0[1:], device=device)
	assert_array_almost_equal(X_attr[1:, :, :], X_attr2, 2)


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


def test_saturation_mutagenesis_args(X0, device):
	torch.manual_seed(0)
	model = FlattenDense(seq_len=100, n_outputs=1)

	alpha = torch.randn(2, 1)
	y0_with, y_hat_with = saturation_mutagenesis(model, X0, args=(alpha,),
		raw_outputs=True, device=device)
	y0_without, y_hat_without = saturation_mutagenesis(model, X0,
		raw_outputs=True, device=device)

	assert y0_with.shape == y0_without.shape
	assert y_hat_with.shape == y_hat_without.shape


def test_saturation_mutagenesis_identity_recovers_y0(X0, device):
	model = SmallDeepSEA(1)

	y0, y_hat = saturation_mutagenesis(model, X0, raw_outputs=True,
		device=device)

	# For each (example, position), the substitution that keeps the original
	# base is an identity edit. Its forward pass is skipped and the slot is
	# filled directly from y0, so it must equal y0 *exactly* (not just to
	# float tolerance) -- this guards the identity-skipping optimization.
	for i in range(X0.shape[0]):
		for pos in range(X0.shape[-1]):
			orig_base = int(X0[i, :, pos].argmax())
			assert torch.equal(y_hat[i, orig_base, pos], y0[i])


def test_saturation_mutagenesis_identity_recovers_y0_windowed(X0, device):
	# Identity-slot exactness must also hold inside a start/end window, which
	# exercises the windowed alignment of the identity mask.
	model = SmallDeepSEA(1)
	start, end = 30, 70

	y0, y_hat = saturation_mutagenesis(model, X0, start=start, end=end,
		raw_outputs=True, device=device)

	for i in range(X0.shape[0]):
		for pos in range(end - start):
			orig_base = int(X0[i, :, start + pos].argmax())
			assert torch.equal(y_hat[i, orig_base, pos], y0[i])


def test_saturation_mutagenesis_all_zero_column(device):
	# An all-zero column (an `N`) has no identity substitution, so all four
	# edits at that position must be genuinely predicted -- not skipped and
	# filled from y0. Validate against brute-force per-substitution predictions.
	torch.manual_seed(0)
	X = random_one_hot((1, 4, 100), random_state=0).float()
	X[0, :, 50] = 0
	model = SmallDeepSEA(1)

	y0, y_hat = saturation_mutagenesis(model, X, raw_outputs=True,
		device=device)

	for base in range(4):
		X_sub = X.clone()
		X_sub[0, :, 50] = 0
		X_sub[0, base, 50] = 1
		y_sub = predict(model, X_sub, device=device)
		assert_array_almost_equal(y_hat[0, base, 50], y_sub[0], 4)


def test_saturation_mutagenesis_multi_hot_column(device):
	# A multi-hot column likewise has no identity row; all four edits are real.
	torch.manual_seed(0)
	X = random_one_hot((1, 4, 100), random_state=0).float()
	X[0, :, 30] = 0
	X[0, 1, 30] = 1
	X[0, 2, 30] = 1
	model = SmallDeepSEA(1)

	y0, y_hat = saturation_mutagenesis(model, X, raw_outputs=True,
		device=device)

	for base in range(4):
		X_sub = X.clone()
		X_sub[0, :, 30] = 0
		X_sub[0, base, 30] = 1
		y_sub = predict(model, X_sub, device=device)
		assert_array_almost_equal(y_hat[0, base, 30], y_sub[0], 4)


def test_saturation_mutagenesis_matches_bruteforce(device):
	# Full validation of the vectorized edit construction: every (base,
	# position) slot of y_hat must equal an explicit prediction of that single
	# substitution. Identity slots reduce to the reference (== y0) and are
	# covered too. A small sequence keeps the brute-force loop cheap.
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
	# Combines the trickiest pieces of the construction: a start/end window
	# (so edit positions are offset) containing an all-zero `N` column (so the
	# kept-edit count varies within the window). Every slot must match an
	# explicit substitution at the absolute position.
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

	# When raw_outputs=False the return is still a plain Tensor.
	attr = saturation_mutagenesis(model, X, raw_outputs=False, device=device)
	assert isinstance(attr, torch.Tensor)


###


class _DenseHead(torch.nn.Module):
	"""Single-tensor model sharing the dense head of a ConvDense instance.

	Used to cross-check the multi-tensor reshape branch in
	`saturation_mutagenesis`: the second (dense) output of `ConvDense` must
	match this standalone model exactly when both are run through ISM.
	"""

	def __init__(self, src):
		super(_DenseHead, self).__init__()
		self.dense = src.dense

	def forward(self, X):
		return self.dense(X.reshape(X.shape[0], -1))


def test_saturation_mutagenesis_multi_output(device):
	# Exercises the list/tuple-output reshape branch end-to-end.
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

	# The dense head must equal the same head run as a single-tensor model.
	# This catches the alphabet/position axis scrambling the old reshape had.
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
	# Regression: the multi-tensor reshape previously hardcoded the full
	# length and raised when start>0 / end<length.
	torch.manual_seed(0)
	X = random_one_hot((2, 4, 100), random_state=0).float()

	model = ConvDense(n_outputs=3)
	y0, y_hat = saturation_mutagenesis(model, X, start=50, end=60,
		raw_outputs=True, device=device)

	assert y_hat[0].shape == (2, 4, 10, 12, 98)
	assert y_hat[1].shape == (2, 4, 10, 3)

	# Cross-check the dense head against the equivalent single-tensor model.
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
	# A slice averages over the selected targets, matching the int-target mean.
	assert_array_almost_equal(attr_slice, (attr0 + attr1) / 2, 4)


def test_saturation_mutagenesis_start_default_end(X0, device):
	# start>0 combined with the default negative end (-1 -> length).
	torch.manual_seed(0)
	model = SmallDeepSEA(5)

	X_attr = saturation_mutagenesis(model, X0, start=50, device=device)
	X_attr_full = saturation_mutagenesis(model, X0, device=device)

	assert X_attr.shape == (2, 4, 50)
	assert_array_almost_equal(X_attr, X_attr_full[:, :, 50:], 4)


def test_saturation_mutagenesis_int8_float_equivalence(X0, device):
	# The int8 cast is load-bearing; a hard one-hot must give identical
	# results whether passed as float or int8.
	torch.manual_seed(0)
	model = SmallDeepSEA(1)

	X_attr_float = saturation_mutagenesis(model, X0, device=device)
	X_attr_int8 = saturation_mutagenesis(model, X0.type(torch.int8),
		device=device)

	assert_array_almost_equal(X_attr_float, X_attr_int8, 4)


def test_saturation_mutagenesis_dtype_arg(X0, device):
	# Explicitly passing dtype should match the default (model dtype) path.
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


def test_saturation_mutagenesis_empty_span(X0):
	# A degenerate span (no positions to perturb) is rejected up front.
	model = SmallDeepSEA(1)
	with pytest.raises(ValueError, match="0 <= start < end <= length"):
		saturation_mutagenesis(model, X0, start=0, end=0, raw_outputs=True)


def test_saturation_mutagenesis_end_past_length(X0):
	# end beyond the sequence length would index past the sequence when
	# building the edits; it must be rejected up front.
	model = SmallDeepSEA(1)
	with pytest.raises(ValueError, match="0 <= start < end <= length"):
		saturation_mutagenesis(model, X0, start=0, end=X0.shape[-1] + 5,
			raw_outputs=True)


def test_saturation_mutagenesis_negative_start(X0):
	# Negative start is never remapped (unlike end) and would index before
	# the sequence; it must be rejected rather than corrupt the batch.
	model = SmallDeepSEA(1)
	with pytest.raises(ValueError, match="0 <= start < end <= length"):
		saturation_mutagenesis(model, X0, start=-3, end=8, raw_outputs=True)


def test_saturation_mutagenesis_start_after_end(X0):
	# An inverted span must be rejected.
	model = SmallDeepSEA(1)
	with pytest.raises(ValueError, match="0 <= start < end <= length"):
		saturation_mutagenesis(model, X0, start=8, end=3, raw_outputs=True)


def test_saturation_mutagenesis_input_dtypes(X0, device):
	# X is coerced to int8 internally, so any integer-valued encoding -- float64,
	# int32, or bool -- must give identical results to the float32 input.
	torch.manual_seed(0)
	model = SmallDeepSEA(1)

	X_attr = saturation_mutagenesis(model, X0, device=device)
	for X_cast in (X0.double(), X0.to(torch.int32), X0.bool()):
		X_attr_cast = saturation_mutagenesis(model, X_cast, device=device)
		assert_array_almost_equal(X_attr, X_attr_cast, 4)


# Low-precision (fp16, bf16) autocast tests. CUDA only: torch.autocast does not
# support these dtypes on CPU for most ops. The dtype is forwarded to `predict`,
# so these guard the forwarding rather than re-deriving attribution magnitudes
# (which are sub-1e-3 differences dominated by fp16 rounding noise). We assert
# the prediction dtype, finiteness, and the precision-robust identity invariant.


def test_saturation_mutagenesis_fp16(X0, cuda_device):
	torch.manual_seed(0)
	model = FlattenDense(seq_len=100, n_outputs=1).to(cuda_device)

	y0, y_hat = saturation_mutagenesis(model, X0, dtype=torch.float16,
		raw_outputs=True, device=cuda_device)

	assert y_hat.shape == (2, 4, 100, 1)
	assert y_hat.dtype == torch.float16
	assert torch.isfinite(y_hat).all()

	for i in range(X0.shape[0]):
		for pos in range(X0.shape[-1]):
			ob = int(X0[i, :, pos].argmax())
			assert_array_almost_equal(y_hat[i, ob, pos].float(),
				y0[i].float(), 2)


def test_saturation_mutagenesis_bf16(X0, cuda_device):
	torch.manual_seed(0)
	model = FlattenDense(seq_len=100, n_outputs=1).to(cuda_device)

	y0, y_hat = saturation_mutagenesis(model, X0, dtype=torch.bfloat16,
		raw_outputs=True, device=cuda_device)

	assert y_hat.shape == (2, 4, 100, 1)
	assert y_hat.dtype == torch.bfloat16
	assert torch.isfinite(y_hat).all()

	for i in range(X0.shape[0]):
		for pos in range(X0.shape[-1]):
			ob = int(X0[i, :, pos].argmax())
			assert_array_almost_equal(y_hat[i, ob, pos].float(),
				y0[i].float(), 2)


def test_saturation_mutagenesis_args_applied_per_example(X0, device):
	# FlattenDense adds a per-example alpha. Because ISM is a difference, the
	# arg must shift every perturbation of example i by exactly alpha[i] -- a
	# wrong index (e.g. alpha[0] for all) would break this -- and the resulting
	# attribution must be invariant to the additive shift.
	torch.manual_seed(0)
	model = FlattenDense(seq_len=100, n_outputs=1)
	alpha = torch.randn(2, 1)

	y0_w, y_hat_w = saturation_mutagenesis(model, X0, args=(alpha,),
		raw_outputs=True, device=device)
	y0_n, y_hat_n = saturation_mutagenesis(model, X0, raw_outputs=True,
		device=device)

	for i in range(X0.shape[0]):
		delta = y_hat_w[i] - y_hat_n[i]
		assert_array_almost_equal(delta, torch.full_like(delta, float(alpha[i])),
			4)

	attr_w = saturation_mutagenesis(model, X0, args=(alpha,), device=device)
	attr_n = saturation_mutagenesis(model, X0, device=device)
	assert_array_almost_equal(attr_w, attr_n, 4)


def test_saturation_mutagenesis_batch_size_invariance(X0, device):
	# Results must not depend on how perturbations are batched.
	torch.manual_seed(0)
	model = SmallDeepSEA(1)

	a1 = saturation_mutagenesis(model, X0, batch_size=1, device=device)
	a32 = saturation_mutagenesis(model, X0, batch_size=32, device=device)
	a_big = saturation_mutagenesis(model, X0, batch_size=10000, device=device)

	assert_array_almost_equal(a1, a32, 4)
	assert_array_almost_equal(a1, a_big, 4)


def test_saturation_mutagenesis_length_one(device):
	# A single-position sequence is a valid degenerate input.
	torch.manual_seed(0)
	X = random_one_hot((2, 4, 1), random_state=0).float()
	model = FlattenDense(seq_len=1, n_outputs=1)

	X_attr = saturation_mutagenesis(model, X, device=device)
	assert X_attr.shape == (2, 4, 1)


def test_saturation_mutagenesis_multi_output_requires_raw(X0, device):
	# Attribution aggregation assumes a single output tensor; a tuple-output
	# model must fail fast with a clear message instead of an opaque error.
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
	# func can reduce a multi-output model to a single tensor, making the
	# attribution path valid again.
	torch.manual_seed(0)
	model = ConvDense(n_outputs=3)

	attr = saturation_mutagenesis(model, X0, func=lambda ys: ys[1],
		device=device)
	assert attr.shape == (2, 4, 100)


def test_saturation_mutagenesis_warns_on_non_integer_input(device):
	# Soft / scaled one-hot inputs are truncated by the int8 cast; the user
	# must be warned rather than silently getting zeros.
	torch.manual_seed(0)
	X = random_one_hot((2, 4, 100), random_state=0).float()
	model = SmallDeepSEA(1)

	with pytest.warns(TangermemeWarning, match="non-integer values"):
		saturation_mutagenesis(model, X * 0.5, device=device)


def test_saturation_mutagenesis_no_warning_on_hard_one_hot(X0, device):
	# A valid hard one-hot must not trigger the truncation warning.
	model = SmallDeepSEA(1)
	with warnings.catch_warnings():
		warnings.simplefilter("error", TangermemeWarning)
		saturation_mutagenesis(model, X0, device=device)


def test_saturation_mutagenesis_skips_identity_forward_passes(X0, device):
	# Identity substitutions are not run through the model: only the 3 true
	# edits per position are predicted, plus one reference pass for y0.
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
