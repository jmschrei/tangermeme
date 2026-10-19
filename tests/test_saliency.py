# test_saliency.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import pytest

from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot
from tangermeme.utils import PackedSequence
from tangermeme.ersatz import shuffle
from tangermeme.ersatz import substitute
from tangermeme.ersatz import multisubstitute
from tangermeme.ablate import ablate
from tangermeme.marginalize import marginalize
from tangermeme.space import space
from tangermeme.variant_effect import substitution_effect
from tangermeme.product import apply_pairwise
from tangermeme.product import apply_product

from tangermeme.saliency import saliency

from .toy_models import SumModel
from .toy_models import Conv
from .toy_models import FlattenDense
from .toy_models import ConvAvgDense
from .toy_models import ConvDense
from .toy_models import LinearConv
from ._packed import assert_same
from ._packed import with_N

from numpy.testing import assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)


@pytest.fixture
def X():
	return random_one_hot((64, 4, 100), random_state=0).type(torch.float32)


##


def test_saliency(X, device):
	# These values are the uncorrected input times gradient; the corrected
	# default is checked against its closed form further down.
	torch.manual_seed(0)
	model = FlattenDense()
	X_attr = saliency(model, X, gradient_correction=False, device=device)

	assert isinstance(X_attr, torch.Tensor)
	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert X_attr.device == torch.device('cpu')

	assert_array_almost_equal(X_attr[:2, :, :6], [
		[[-0.0004,  0.0000, -0.0000, -0.0368, -0.0000,  0.0000],
		 [ 0.0000,  0.0000, -0.0218, -0.0000, -0.0000, -0.0000],
		 [ 0.0000,  0.0000, -0.0000, -0.0000, -0.0000,  0.0000],
		 [-0.0000,  0.0090, -0.0000,  0.0000,  0.0098, -0.0371]],
		[[-0.0000,  0.0000, -0.0412, -0.0000, -0.0193,  0.0000],
		 [ 0.0470,  0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [ 0.0000,  0.0322, -0.0000, -0.0000, -0.0000,  0.0350],
		 [-0.0000,  0.0000, -0.0000,  0.0115,  0.0000, -0.0000]]], 4)


def test_saliency_hypothetical(X, device):
	torch.manual_seed(0)
	model = FlattenDense()
	X_attr = saliency(model, X, hypothetical=True, gradient_correction=False,
		device=device)

	assert X_attr.shape == X.shape

	# A single Linear layer has a constant Jacobian, so every example gets
	# the same hypothetical attributions.
	assert_array_almost_equal(X_attr[:2, :, :6], [
		[[-0.0004,  0.0268, -0.0412, -0.0368, -0.0193,  0.0134],
		 [ 0.0470,  0.0337, -0.0218, -0.0126, -0.0476, -0.0009],
		 [ 0.0021,  0.0322, -0.0378, -0.0343, -0.0290,  0.0350],
		 [-0.0241,  0.0090, -0.0260,  0.0115,  0.0098, -0.0371]],
		[[-0.0004,  0.0268, -0.0412, -0.0368, -0.0193,  0.0134],
		 [ 0.0470,  0.0337, -0.0218, -0.0126, -0.0476, -0.0009],
		 [ 0.0021,  0.0322, -0.0378, -0.0343, -0.0290,  0.0350],
		 [-0.0241,  0.0090, -0.0260,  0.0115,  0.0098, -0.0371]]], 4)


def test_saliency_hypothetical_projects_to_observed(X, device):
	# The projected attributions are exactly the hypothetical ones masked
	# down to the characters that are actually present, which is the same
	# relationship deep_lift_shap and saturation_mutagenesis have.
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)

	X_attr = saliency(model, X, device=device)
	X_attr_hypothetical = saliency(model, X, hypothetical=True, device=device)

	assert_array_almost_equal(X_attr, X_attr_hypothetical * X, 5)


def test_saliency_is_signed(X, device):
	# The gradient is not passed through an absolute value, so attributions
	# carry sign: a character can push the target output down.
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	X_attr = saliency(model, X, device=device)

	assert X_attr.min() < 0
	assert X_attr.max() > 0


def test_saliency_matches_closed_form_linear(device):
	# For a single Linear layer the gradient of the output with respect to
	# the input is exactly the weight matrix, independent of X.
	torch.manual_seed(1)
	model = FlattenDense(seq_len=6, n_outputs=1)
	X = random_one_hot((5, 4, 6), random_state=2).type(torch.float32)

	X_attr = saliency(model, X, gradient_correction=False, device=device)

	W = model.dense.weight.detach().reshape(4, 6)
	assert_array_almost_equal(X_attr, W.unsqueeze(0) * X, 5)


def test_saliency_matches_autograd(device):
	# Ground truth from a plain torch.autograd.grad call, using a model with
	# a nonlinearity so the gradient genuinely depends on X.
	torch.manual_seed(3)
	model = ConvAvgDense(n_outputs=1).to(device)
	model.eval()

	X = random_one_hot((6, 4, 15), random_state=4).type(torch.float32)
	X_attr = saliency(model, X, gradient_correction=False, device=device)

	X_ = X.clone().to(device).requires_grad_()
	y = model(X_)
	grad, = torch.autograd.grad(y[:, 0].sum(), X_)

	assert_array_almost_equal(X_attr, (grad * X_).detach().cpu(), 5)


def test_saliency_zero_column(device):
	# An all-zero ('N') column contributes nothing to grad * X, so its
	# projected attribution is exactly zero at every character.
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	X = random_one_hot((1, 4, 12), random_state=0).type(torch.float32)
	X[0, :, 5] = 0

	X_attr = saliency(model, X, device=device)
	assert_array_equal(X_attr[0, :, 5], torch.zeros(4))
	assert X_attr[0, :, 4].abs().sum() > 0


@pytest.mark.parametrize("batch_size", [1, 3, 10000])
def test_saliency_batch_size(X, batch_size, device):
	# The per-example result must not depend on which other examples happen
	# to share its batch.
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	X = X[:16]

	X_attr = saliency(model, X, batch_size=batch_size, device=device)
	X_attr0 = saliency(model, X, batch_size=16, device=device)

	assert X_attr.shape == X.shape
	assert_array_almost_equal(X_attr, X_attr0, 5)


def test_saliency_target(X, device):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=3)

	X_attr0 = saliency(model, X, target=0, device=device)
	X_attr1 = saliency(model, X, target=1, device=device)

	assert X_attr0.shape == X.shape
	assert not torch.allclose(X_attr0, X_attr1)


def test_saliency_args(X, device):
	# FlattenDense.forward(X, alpha, beta) scales the linear output by beta,
	# which scales the gradient, and so the attributions, by the same factor.
	torch.manual_seed(0)
	model = FlattenDense()
	alpha = torch.zeros(X.shape[0], 1)
	beta = torch.full((X.shape[0], 1), 2.0)

	X_attr = saliency(model, X, args=(alpha, beta), device=device)
	X_attr0 = saliency(model, X, device=device)

	assert_array_almost_equal(X_attr, X_attr0 * 2, 4)


def test_saliency_func(X, device):
	# `func` runs before the target is selected, so summing a three-output
	# head gives the same attributions as summing the three per-target maps.
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=3)

	X_attr = saliency(model, X, func=lambda y: y.sum(dim=-1, keepdims=True),
		device=device)
	X_attr0 = sum(saliency(model, X, target=i, device=device)
		for i in range(3))

	assert_array_almost_equal(X_attr, X_attr0, 4)


def test_saliency_func_reduces_multi_output_model(X, device):
	# ConvDense returns a tuple, which saliency cannot index; a `func` that
	# selects one of the two tensors makes it usable.
	torch.manual_seed(0)
	model = ConvDense()

	X_attr = saliency(model, X, func=lambda y: y[1], device=device)
	assert X_attr.shape == X.shape


def test_saliency_int8_input(device):
	# X is only upcast per batch, so an int8 encoding -- which cannot itself
	# require grad -- has to work.
	torch.manual_seed(0)
	model = FlattenDense(seq_len=6, n_outputs=1)
	X = one_hot_encode('ACGTAC').unsqueeze(0)
	assert X.dtype == torch.int8

	X_attr = saliency(model, X, device=device)

	assert X_attr.shape == (1, 4, 6)
	assert_array_almost_equal(X_attr,
		saliency(model, X.type(torch.float32), device=device), 5)


def test_saliency_dtype(X, cuda_device):
	torch.manual_seed(0)
	model = FlattenDense().to(torch.float16)

	X_attr = saliency(model, X, dtype=torch.float16, device=cuda_device)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float16


def test_saliency_autocast_bfloat16(X, device):
	# An fp32 model run under bf16 autocast returns bf16 attributions, as
	# predict and deep_lift_shap do, close to the fp32 ones.
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	X = X[:8]

	X_attr = saliency(model, X, dtype=torch.bfloat16, device=device)
	X_attr32 = saliency(model, X, device=device)

	assert X_attr.dtype == torch.bfloat16
	assert X_attr.shape == X.shape
	assert_array_almost_equal(X_attr.float(), X_attr32, 3)


def test_saliency_autocast_float16(X, cuda_device):
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	X = X[:8]

	X_attr = saliency(model, X, dtype=torch.float16, device=cuda_device)
	X_attr32 = saliency(model, X, device=cuda_device)

	assert X_attr.dtype == torch.float16
	assert_array_almost_equal(X_attr.float(), X_attr32, 3)


def test_saliency_dtype_string(X, device):
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	X = X[:8]

	# numpy has no bfloat16, so compare the tensors in torch.
	assert torch.equal(saliency(model, X, dtype="bfloat16", device=device),
		saliency(model, X, dtype=torch.bfloat16, device=device))


def test_saliency_model_without_parameters(device):
	# SumModel has no parameters to take a dtype from, so float32 is used.
	# Its output c is the count of character c, so the gradient is one at c
	# and zero elsewhere, and the corrected gradient is 0.75 and -0.25.
	model = SumModel()
	X = random_one_hot((3, 4, 10), random_state=0).type(torch.float32)

	X_attr = saliency(model, X, target=2, hypothetical=True, device=device)

	expected = torch.full((3, 4, 10), -0.25)
	expected[:, 2] = 0.75

	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr, expected, 6)


def test_saliency_preserves_model_state(X, device):
	torch.manual_seed(0)
	model = FlattenDense()
	model.train()

	saliency(model, X, device=device)

	assert model.training
	assert next(model.parameters()).device == torch.device('cpu')


def test_saliency_does_not_mutate_input(X, device):
	torch.manual_seed(0)
	model = FlattenDense()
	X0 = X.clone()

	saliency(model, X, device=device)

	assert_array_equal(X, X0)
	assert not X.requires_grad


def test_saliency_within_no_grad(X, device):
	# Perturbation functions such as marginalize may call saliency from
	# inside a no_grad block, so grad has to be re-enabled internally.
	torch.manual_seed(0)
	model = FlattenDense()

	with torch.no_grad():
		X_attr = saliency(model, X, device=device)

	assert_array_almost_equal(X_attr, saliency(model, X, device=device), 5)


def test_saliency_raises_multi_output_model(X, device):
	# Without a `func` to reduce it, a tuple-returning model cannot be
	# indexed by `target`.
	model = ConvDense()

	with pytest.raises(ValueError, match="single tensor"):
		saliency(model, X, device=device)


def test_saliency_raises_profile_output(X, device):
	# Conv returns (batch, 12, length - 2). Indexing it by `target` would
	# attribute the sum of a whole profile, so it raises instead, as
	# deep_lift_shap does.
	model = Conv()

	with pytest.raises(ValueError, match="saliency requires the model"):
		saliency(model, X, device=device)


def test_saliency_raises_empty(device):
	model = FlattenDense()
	X = random_one_hot((0, 4, 100), random_state=0).type(torch.float32)

	with pytest.raises(ValueError, match="at least one example"):
		saliency(model, X, device=device)


def test_saliency_raises_mismatched_args(X, device):
	model = FlattenDense()

	with pytest.raises(ValueError, match="same first"):
		saliency(model, X, args=(torch.zeros(2, 1),), device=device)


def test_saliency_raises_bad_input(device):
	model = FlattenDense()

	# Wrong number of dimensions, non-one-hot values, and not a tensor.
	assert_raises(ValueError, saliency, model, torch.zeros(4, 100),
		device=device)
	assert_raises(ValueError, saliency, model, torch.randn(4, 4, 100),
		device=device)
	assert_raises(ValueError, saliency, model, [[0, 1, 0, 0]], device=device)


##


def test_saliency_marginalize(X, device):
	# saliency satisfies the func= contract, so marginalize returns the
	# attributions of X and of X with the motif substituted in.
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	X = X[:4]

	y_before, y_after = marginalize(model, X, "ACGTGT", func=saliency,
		device=device)

	assert_array_almost_equal(y_before, saliency(model, X, device=device), 5)
	assert_array_almost_equal(y_after, saliency(model,
		substitute(X, "ACGTGT"), device=device), 5)


def test_saliency_ablate(X, device):
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	X = X[:4]

	y_before, y_after = ablate(model, X, 40, 60, n=3, func=saliency,
		random_state=0, device=device)

	X_perturb = shuffle(X, start=40, end=60, n=3, random_state=0)

	assert y_after.shape == (4, 3, 4, 100)
	assert_array_almost_equal(y_before, saliency(model, X, device=device), 5)
	assert_array_almost_equal(y_after.reshape(12, 4, 100), saliency(model,
		X_perturb.reshape(12, 4, 100), device=device), 5)


def test_saliency_space(X, device):
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	X = X[:4]
	motifs, spacing = ["ACG", "TTT"], [[2], [5]]

	y_before, y_afters = space(model, X, motifs, spacing, func=saliency,
		device=device)

	assert y_afters.shape == (4, 2, 4, 100)
	assert_array_almost_equal(y_before, saliency(model, X, device=device), 5)

	for k, s in enumerate(spacing):
		assert_array_almost_equal(y_afters[:, k], saliency(model,
			multisubstitute(X, motifs, s), device=device), 5)


def test_saliency_substitution_effect(X, device):
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	X = X[:4]
	substitutions = torch.tensor([[0, 10, 1], [2, 50, 3], [2, 51, 0]])

	y_before, y_after = substitution_effect(model, X, substitutions,
		func=saliency, device=device)

	X_sub = X.clone()
	for i, position, character in substitutions:
		X_sub[i, :, position] = 0
		X_sub[i, character, position] = 1

	assert_array_almost_equal(y_before, saliency(model, X, device=device), 5)
	assert_array_almost_equal(y_after, saliency(model, X_sub,
		device=device), 5)


def test_saliency_apply_pairwise(X, device):
	# FlattenDense(X, alpha, beta) scales its output by beta, so each paired
	# (alpha, beta) gives the attributions of the sequence scaled by beta.
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)
	X = X[:3]
	alpha = torch.tensor([[0.0], [1.0]])
	beta = torch.tensor([[2.0], [-1.0]])

	X_attr = apply_pairwise(saliency, model, X, args=(alpha, beta),
		device=device)
	X_attr0 = saliency(model, X, device=device)

	assert X_attr.shape == (3, 2, 4, 100)
	assert_array_almost_equal(X_attr[:, 0], X_attr0 * 2, 5)
	assert_array_almost_equal(X_attr[:, 1], -X_attr0, 5)


def test_saliency_apply_product(X, device):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)
	X = X[:3]
	alpha = torch.tensor([[0.0], [1.0]])
	beta = torch.tensor([[2.0], [-1.0], [0.5]])

	X_attr = apply_product(saliency, model, X, args=(alpha, beta),
		device=device)
	X_attr0 = saliency(model, X, device=device)

	assert X_attr.shape == (3, 2, 3, 4, 100)
	for j in range(2):
		for k, b in enumerate(beta[:, 0]):
			assert_array_almost_equal(X_attr[:, j, k], X_attr0 * b, 5)


@pytest.mark.parametrize("hypothetical", [False, True])
def test_saliency_gradient_correction_closed_form_linear(hypothetical, device):
	# For a single Linear layer the gradient is the weight matrix, so the
	# corrected gradient is the weight matrix minus its mean over the
	# alphabet at each position.
	torch.manual_seed(1)
	model = FlattenDense(seq_len=6, n_outputs=1)
	X = random_one_hot((5, 4, 6), random_state=2).type(torch.float32)

	X_attr = saliency(model, X, hypothetical=hypothetical, device=device)

	W = model.dense.weight.detach().reshape(4, 6)
	W = (W - W.mean(dim=0, keepdim=True)).unsqueeze(0)
	expected = W.expand(5, -1, -1) if hypothetical else W * X

	assert_array_almost_equal(X_attr, expected, 5)


def test_saliency_gradient_correction_matches_autograd(device):
	torch.manual_seed(3)
	model = ConvAvgDense(n_outputs=1).to(device)
	model.eval()

	X = random_one_hot((6, 4, 15), random_state=4).type(torch.float32)
	X_attr = saliency(model, X, device=device)

	X_ = X.clone().to(device).requires_grad_()
	y = model(X_)
	grad, = torch.autograd.grad(y[:, 0].sum(), X_)
	grad = grad - grad.mean(dim=1, keepdim=True)

	assert_array_almost_equal(X_attr, (grad * X_).detach().cpu(), 5)


def test_saliency_gradient_correction_ignores_off_simplex(device):
	# Adding the same weight to every character at a position changes the
	# model only off the one-hot simplex, where every position sums to one.
	# On one-hot input it adds a constant to the output, so the corrected
	# attributions must not move, while the uncorrected ones do.
	torch.manual_seed(1)
	model = FlattenDense(seq_len=6, n_outputs=1)
	X = random_one_hot((5, 4, 6), random_state=2).type(torch.float32)

	X_attr = saliency(model, X, device=device)
	X_attr_raw = saliency(model, X, gradient_correction=False, device=device)

	with torch.no_grad():
		W = model.dense.weight.reshape(4, 6)
		W += torch.randn(1, 6)

	assert_array_almost_equal(saliency(model, X, device=device), X_attr, 5)
	assert not torch.allclose(saliency(model, X, gradient_correction=False,
		device=device), X_attr_raw)


def test_saliency_gradient_correction_sums_to_zero(X, device):
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)

	X_attr = saliency(model, X, hypothetical=True, device=device)
	assert_array_almost_equal(X_attr.sum(dim=1), torch.zeros(64, 100), 6)

	X_attr = saliency(model, X, hypothetical=True, gradient_correction=False,
		device=device)
	assert X_attr.sum(dim=1).abs().max() > 1e-3

##


def test_saliency_linear_model_gradient_is_constant(device):
	# A model with no nonlinearity has a constant Jacobian, so its
	# hypothetical attributions do not depend on the sequence at all. This
	# is what makes dependency_map diagonal for linear models.
	torch.manual_seed(0)

	for model in (FlattenDense(seq_len=15, n_outputs=1), LinearConv()):
		X = random_one_hot((3, 4, 15), random_state=0).type(torch.float32)
		X_attr = saliency(model, X, hypothetical=True, device=device)

		assert_array_almost_equal(X_attr[0], X_attr[1], 5)
		assert_array_almost_equal(X_attr[0], X_attr[2], 5)


###
# A PackedSequence gives the attributions of its one-hot encoding, each batch
# decoded on the device. Given as the `func` of another function, saliency
# receives the packed copies that function makes from a packed X.


@pytest.mark.parametrize("hypothetical", [False, True])
@pytest.mark.parametrize("batch_size", [1, 7, 64])
def test_saliency_packed(X, hypothetical, batch_size, device):
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	X = with_N(X)

	X_attr = saliency(model, X, hypothetical=hypothetical,
		batch_size=batch_size, device=device)
	X_attr_packed = saliency(model, PackedSequence.from_one_hot(X),
		hypothetical=hypothetical, batch_size=batch_size, device=device)
	assert_same(X_attr, X_attr_packed)


def test_saliency_packed_args_target(X, device):
	torch.manual_seed(0)
	model = FlattenDense()
	alpha = torch.zeros(X.shape[0], 1)
	beta = torch.full((X.shape[0], 1), 2.0)

	X_attr = saliency(model, X, args=(alpha, beta), target=2, device=device)
	X_attr_packed = saliency(model, PackedSequence.from_one_hot(X),
		args=(alpha, beta), target=2, device=device)
	assert_same(X_attr, X_attr_packed)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_saliency_packed_16bit(X, dtype, cuda_device):
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)

	X_attr = saliency(model, X, dtype=dtype, device=cuda_device)
	X_attr_packed = saliency(model, PackedSequence.from_one_hot(X),
		dtype=dtype, device=cuda_device)
	assert_same(X_attr, X_attr_packed)


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.bool,
	torch.float16, torch.float64])
def test_saliency_stored_dtype(X, dtype, device):
	# Attributions of sequences and arguments stored in another dtype equal
	# those of the same values given as float32.
	torch.manual_seed(0)
	model = FlattenDense()
	alpha = torch.zeros(X.shape[0], 1)
	beta = torch.full((X.shape[0], 1), 2.0)

	X_attr = saliency(model, X.type(dtype), args=(alpha.half(),
		beta.double()), device=device)
	X_attr0 = saliency(model, X, args=(alpha, beta), device=device)
	assert_same(X_attr0, X_attr)


@pytest.mark.parametrize("name", ["marginalize", "ablate", "space",
	"substitution_effect", "apply_pairwise"])
def test_saliency_func_packed(X, name, device):
	torch.manual_seed(0)
	model = ConvAvgDense(n_outputs=1)
	X = with_N(X[:4])
	alpha = torch.tensor([[0.0], [1.0]])
	beta = torch.tensor([[2.0], [-1.0]])

	calls = {
		"marginalize": lambda X_: marginalize(model, X_, "ACGTGT",
			func=saliency, device=device),
		"ablate": lambda X_: ablate(model, X_, 40, 60, n=3, func=saliency,
			random_state=0, device=device),
		"space": lambda X_: space(model, X_, ["ACG", "TTT"], [[2], [5]],
			func=saliency, device=device),
		"substitution_effect": lambda X_: substitution_effect(model, X_,
			torch.tensor([[0, 10, 1], [2, 50, 3]]), func=saliency,
			device=device),
		"apply_pairwise": lambda X_: apply_pairwise(saliency, FlattenDense(
			n_outputs=1), X_, args=(alpha, beta), device=device),
	}

	torch.manual_seed(0)
	y = calls[name](X)
	torch.manual_seed(0)
	y_packed = calls[name](PackedSequence.from_one_hot(X))
	assert_same(y, y_packed)


def test_saliency_packed_raises(X, device):
	model = FlattenDense()
	X_packed = PackedSequence.from_one_hot(X)
	assert_raises(ValueError, saliency, model, X_packed[:0], device=device)
	assert_raises(ValueError, saliency, model, X_packed, args=(X[:5],),
		device=device)
