# test_dependency_map.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest

from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot
from tangermeme.utils import PackedSequence

from tangermeme.saliency import saliency
from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.saturation_mutagenesis import saturation_mutagenesis
from tangermeme.dependency_map import dependency_map

from .toy_models import FlattenDense
from .toy_models import ConvAvgDense
from .toy_models import LinearConv
from ._packed import check_packed
from ._packed import Spy

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)


@pytest.fixture
def X():
	return random_one_hot((4, 4, 12), random_state=0).type(torch.float32)


##


def test_dependency_map(X, device):
	# These values use uncorrected saliency; the corrected default is checked
	# against a brute force in test_dependency_map_matches_brute_force.
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	dmap = dependency_map(model, X, gradient_correction=False, device=device)

	assert isinstance(dmap, torch.Tensor)
	assert dmap.shape == (4, 12, 12)
	assert dmap.dtype == torch.float32
	assert dmap.device == torch.device('cpu')

	assert_array_almost_equal(dmap[0, :5, :5], [
		[0.0016, 0.0009, 0.0020, 0.0000, 0.0000],
		[0.0022, 0.0058, 0.0057, 0.0042, 0.0000],
		[0.0028, 0.0015, 0.0103, 0.0018, 0.0032],
		[0.0000, 0.0076, 0.0031, 0.0038, 0.0032],
		[0.0000, 0.0000, 0.0012, 0.0024, 0.0075]], 4)

	assert_array_almost_equal(dmap[3, :5, :5], [
		[0.0027, 0.0013, 0.0013, 0.0000, 0.0000],
		[0.0017, 0.0065, 0.0059, 0.0015, 0.0000],
		[0.0030, 0.0029, 0.0117, 0.0093, 0.0031],
		[0.0000, 0.0018, 0.0046, 0.0067, 0.0059],
		[0.0000, 0.0000, 0.0066, 0.0043, 0.0057]], 4)


def test_dependency_map_float64(device):
	# A float64 attribution function gives a float64 map, so the map can be
	# checked against a float64 reference at full precision.
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1).double()
	X = random_one_hot((2, 4, 12), random_state=0).type(torch.float64)

	dmap = dependency_map(model, X, device=device)
	assert dmap.dtype == torch.float64

	dmap32 = dependency_map(model.float(), X.float(), device=device)
	assert dmap32.dtype == torch.float32
	assert_array_almost_equal(dmap, dmap32, 5)


def test_dependency_map_holds_model_on_device(cuda_device):
	# The model is moved to the device once for the whole call, so every
	# call to func finds it there, and it is moved back afterward.
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	X = random_one_hot((2, 4, 12), random_state=0).type(torch.float32)
	devices = []

	def func(model, X, args=None, **kwargs):
		devices.append(next(model.parameters()).device.type)
		return saliency(model, X, args=args, **kwargs)

	dependency_map(model, X, func=func, device=cuda_device)

	assert devices == ["cuda"] * 4
	assert next(model.parameters()).device.type == "cpu"


def test_dependency_map_receptive_field(X, device):
	# The map is an average of absolute differences. ConvAvgDense is a
	# single width-3 convolution, so substituting a base can only move the
	# attribution of positions within one step of it, and its ReLU plus
	# global pooling lets one position's gradient depend on another
	# position's base, so off-diagonal entries must survive.
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	dmap = dependency_map(model, X, device=device)

	assert torch.all(dmap >= 0)

	j, i = torch.meshgrid(torch.arange(12), torch.arange(12), indexing='ij')
	assert dmap[:, (j - i).abs() > 2].max() < 1e-6
	assert dmap[:, (j - i).abs() <= 2].max() > 1e-3

	off_diagonal = dmap - torch.diag_embed(dmap.diagonal(dim1=-2, dim2=-1))
	assert off_diagonal.abs().max() > 1e-6


@pytest.mark.parametrize("gradient_correction", [True, False])
def test_dependency_map_matches_brute_force(gradient_correction, device):
	# For each position, the column must be the mean absolute change in the
	# summed attributions over exactly the substitutions that change a base.
	# A position whose base is re-applied contributes an exact zero, so
	# averaging over the three real substitutions rather than over all four
	# characters is what the returned values are scaled by.
	# `gradient_correction` is forwarded to `saliency` through kwargs.
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	model.eval()

	X = one_hot_encode('ACGTAC').unsqueeze(0).type(torch.float32)
	dmap = dependency_map(model, X, gradient_correction=gradient_correction,
		device=device)[0]

	attr0 = saliency(model, X, gradient_correction=gradient_correction,
		device=device).sum(dim=1)[0]

	expected = numpy.zeros((6, 6))
	for i in range(6):
		diffs = []
		for c in range(4):
			if X[0, c, i] == 1:
				continue

			X_ = X.clone()
			X_[0, :, i] = 0
			X_[0, c, i] = 1

			attr = saliency(model, X_, gradient_correction=gradient_correction,
				device=device).sum(dim=1)[0]
			diffs.append((attr - attr0).abs().numpy())

		expected[:, i] = numpy.mean(diffs, axis=0)

	assert_array_almost_equal(dmap, expected, 5)


def test_dependency_map_n_column_normalization(device):
	# An 'N' has no base to re-apply, so all four substitutions are real and
	# the average is over four rather than three.
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)

	X = one_hot_encode('ACGTAC').unsqueeze(0).type(torch.float32)
	X[0, :, 2] = 0

	dmap = dependency_map(model, X, device=device)[0]
	attr0 = saliency(model, X, device=device).sum(dim=1)[0]

	total = torch.zeros(6)
	for c in range(4):
		X_ = X.clone()
		X_[0, c, 2] = 1

		attr = saliency(model, X_, device=device).sum(dim=1)[0]
		total += (attr - attr0).abs()

	assert_array_almost_equal(dmap[:, 2], total / 4, 5)


def test_dependency_map_diagonal_for_linear_models(device):
	# A model with no nonlinearity has a constant Jacobian, so the
	# attribution at position j cannot depend on the base at any other
	# position. The map must be purely diagonal.
	torch.manual_seed(0)
	X = random_one_hot((2, 4, 15), random_state=0).type(torch.float32)

	for model in (FlattenDense(seq_len=15, n_outputs=1), LinearConv()):
		dmap = dependency_map(model, X, device=device)

		off_diagonal = dmap - torch.diag_embed(
			dmap.diagonal(dim1=-2, dim2=-1))

		assert_array_almost_equal(off_diagonal, torch.zeros(2, 15, 15), 5)
		assert dmap.diagonal(dim1=-2, dim2=-1).sum() > 0


def test_dependency_map_batch_matches_single(X, device):
	# Sequences are processed one at a time, so batching must be invisible.
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)

	dmap = dependency_map(model, X, device=device)
	dmap0 = torch.cat([dependency_map(model, X[i:i+1], device=device)
		for i in range(X.shape[0])])

	assert_array_almost_equal(dmap, dmap0, 6)


@pytest.mark.parametrize("batch_size", [1, 5, 7])
def test_dependency_map_batch_size(X, batch_size, device):
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)

	X = X[:1, :, :8]

	dmap = dependency_map(model, X, batch_size=batch_size, device=device)
	dmap0 = dependency_map(model, X, batch_size=64, device=device)

	assert_array_almost_equal(dmap, dmap0, 5)


##


def test_dependency_map_start_end(X, device):
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)

	X = X[:2]
	dmap = dependency_map(model, X, start=3, end=8, device=device)

	assert dmap.shape == (2, 12, 5)
	assert_array_almost_equal(dmap,
		dependency_map(model, X, device=device)[:, :, 3:8], 6)


def test_dependency_map_end_negative(X, device):
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)

	X = X[:1]

	assert dependency_map(model, X, end=-1, device=device).shape == (1, 12, 12)
	assert dependency_map(model, X, end=-3, device=device).shape == (1, 12, 10)
	assert_array_almost_equal(
		dependency_map(model, X, end=-3, device=device),
		dependency_map(model, X, end=10, device=device), 6)


def test_dependency_map_raises_bad_start_end(X, device):
	model = ConvAvgDense(n_outputs=1)

	for start, end in [(-1, 5), (5, 5), (8, 3), (0, 13)]:
		with pytest.raises(ValueError, match="start and end"):
			dependency_map(model, X, start=start, end=end, device=device)


##


def _deep_lift_shap_fixed(model, X, args=None, **kwargs):
	# The same three references for every sequence, so that attributions
	# before and after a substitution differ only through the substitution.
	references = random_one_hot((3, 4, 8), random_state=1).type(
		torch.float32).unsqueeze(0).expand(X.shape[0], -1, -1, -1)

	return deep_lift_shap(model, X, args=args, references=references,
		warning_threshold=1e9, **kwargs)


@pytest.mark.parametrize("func", [_deep_lift_shap_fixed,
	saturation_mutagenesis], ids=["deep_lift_shap", "saturation_mutagenesis"])
def test_dependency_map_func_matches_brute_force(func, device):
	# The same mean of absolute differences as with saliency, computed by
	# substituting each base by hand and calling func on every copy.
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	model.eval()

	X = random_one_hot((1, 4, 8), random_state=0).type(torch.float32)
	dmap = dependency_map(model, X, func=func, device=device)[0]

	attr0 = func(model, X, device=device).sum(dim=1)[0]

	expected = numpy.zeros((8, 8))
	for i in range(8):
		diffs = []
		for c in range(4):
			if X[0, c, i] == 1:
				continue

			X_ = X.clone()
			X_[0, :, i] = 0
			X_[0, c, i] = 1

			attr = func(model, X_, device=device).sum(dim=1)[0]
			diffs.append((attr - attr0).abs().numpy())

		expected[:, i] = numpy.mean(diffs, axis=0)

	assert dmap.sum() > 0
	assert_array_almost_equal(dmap, expected, 5)


def test_dependency_map_func_deep_lift_shap(device):
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	X = random_one_hot((1, 4, 8), random_state=0).type(torch.float32)

	dmap = dependency_map(model, X, func=deep_lift_shap, n_shuffles=3,
		random_state=0, device=device)

	assert dmap.shape == (1, 8, 8)
	assert torch.all(dmap >= 0)
	assert dmap.sum() > 0


def test_dependency_map_additional_func_kwargs(X, device):
	# `saliency` takes its own `func`, which collides with this function's
	# `func`, so it has to be routed through additional_func_kwargs.
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)

	dmap = dependency_map(model, X, func=saliency, device=device,
		additional_func_kwargs={'func': lambda y: y * 2})
	dmap0 = dependency_map(model, X, device=device)

	assert_array_almost_equal(dmap, dmap0 * 2, 5)


def test_dependency_map_additional_func_kwargs_not_mutated(X, device):
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	kwargs = {'target': 0}

	dependency_map(model, X[:1], device=device, additional_func_kwargs=kwargs)
	assert kwargs == {'target': 0}


def test_dependency_map_target(device):
	torch.manual_seed(0)
	model = FlattenDense(seq_len=12, n_outputs=3)
	X = random_one_hot((1, 4, 12), random_state=0).type(torch.float32)

	dmap0 = dependency_map(model, X, target=0, device=device)
	dmap1 = dependency_map(model, X, target=1, device=device)

	assert not torch.allclose(dmap0, dmap1)


def test_dependency_map_args(device):
	# FlattenDense scales its output by beta, which scales attributions and
	# therefore the whole map by the same factor.
	torch.manual_seed(0)
	model = FlattenDense(seq_len=12, n_outputs=1)
	X = random_one_hot((2, 4, 12), random_state=0).type(torch.float32)

	alpha = torch.zeros(2, 1)
	beta = torch.full((2, 1), 2.0)

	dmap = dependency_map(model, X, args=(alpha, beta), device=device)
	dmap0 = dependency_map(model, X, device=device)

	assert_array_almost_equal(dmap, dmap0 * 2, 5)


##


@pytest.mark.parametrize("packed", [False, True])
def test_dependency_map_raises_empty(X, packed, device):
	model = ConvAvgDense(n_outputs=1)
	if packed:
		X = PackedSequence.from_one_hot(X)

	with pytest.raises(ValueError, match="at least one example"):
		dependency_map(model, X[:0], device=device)


@pytest.mark.parametrize("packed", [False, True])
def test_dependency_map_raises_mismatched_args(X, packed, device):
	model = ConvAvgDense(n_outputs=1)
	if packed:
		X = PackedSequence.from_one_hot(X)

	with pytest.raises(ValueError, match="same first"):
		dependency_map(model, X, args=(torch.zeros(2, 1),), device=device)


def test_dependency_map_raises_bad_input(device):
	model = ConvAvgDense(n_outputs=1)

	assert_raises(ValueError, dependency_map, model, torch.zeros(4, 12),
		device=device)
	assert_raises(ValueError, dependency_map, model, torch.randn(4, 4, 12),
		device=device)
	assert_raises(ValueError, dependency_map, model, 'ACGTAC', device=device)


def test_dependency_map_raises_bad_func(X, device):
	# The alphabet axis is summed over, so a func that does not return a
	# single (-1, len(alphabet), length) tensor cannot be used.
	model = ConvAvgDense(n_outputs=1)

	with pytest.raises(ValueError, match="single tensor"):
		dependency_map(model, X, device=device,
			func=lambda model, X, args=None, **kwargs: (X, X))

	with pytest.raises(ValueError, match="single tensor"):
		dependency_map(model, X, device=device,
			func=lambda model, X, args=None, **kwargs: torch.zeros(X.shape[0]))


def test_dependency_map_verbose(X, device):
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)

	assert_array_almost_equal(
		dependency_map(model, X[:1], verbose=True, device=device),
		dependency_map(model, X[:1], device=device), 6)


###
# A PackedSequence, decoded at the start, gives the dependency maps of its
# one-hot encoding, and a function of the user's is given int8 one-hot
# sequences.


_SALIENCY_SPY = Spy(saliency, 1)


@pytest.mark.parametrize("func", ["saliency", "deep_lift_shap", "custom"])
def test_dependency_map_packed(X, func, device):
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	kwargs = {"saliency": dict(func=saliency), "custom": dict(
		func=_SALIENCY_SPY), "deep_lift_shap": dict(func=deep_lift_shap,
		random_state=0, n_shuffles=3)}[func]

	if func == "saliency":
		X = X.clone()
		X[1, :, 4:6] = 0

	check_packed(lambda X_: dependency_map(model, X_, device=device,
		**kwargs), X, [_SALIENCY_SPY] if func == "custom" else [])


def test_dependency_map_packed_start_end_args(device):
	torch.manual_seed(0)
	model = FlattenDense(seq_len=12, n_outputs=1)
	X = random_one_hot((2, 4, 12), random_state=0)
	alpha = torch.zeros(2, 1)
	beta = torch.full((2, 1), 2.0)

	dmap = dependency_map(model, X, args=(alpha, beta), start=3, end=8,
		device=device)
	dmap_packed = dependency_map(model, PackedSequence.from_one_hot(X),
		args=(alpha, beta), start=3, end=8, device=device)
	assert torch.equal(dmap_packed, dmap)
