# test_pisa.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(0)


import pytest
import warnings

from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot

from tangermeme.ersatz import substitute
from tangermeme.ersatz import shuffle
from tangermeme.ersatz import dinucleotide_shuffle

from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.deep_lift_shap import hypothetical_attributions

from tangermeme.pisa import pisa

from .toy_models import SumModel
from .toy_models import FlattenDense
from .toy_models import Conv1
from .toy_models import Scatter
from .toy_models import ConvDense
from .toy_models import ConvPoolDense
from .toy_models import SmallDeepSEA

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


@pytest.fixture
def X():
	X_ = random_one_hot((2, 4, 100), random_state=0).type(torch.float32)
	X_ = substitute(X_, "ACGTACGT")
	return X_


@pytest.fixture
def references(X):
	return shuffle(X, n=5, random_state=0)


class LambdaWrapper(torch.nn.Module):
	"""Wrapper that runs a given forward function instead of the default.

	Several of the classes in toy_models.py return multiple outputs but the
	attributions from deep_lift_shap require that there's only one output per
	example to explain. This class helps overcome the issues with having
	multiple outputs by slicing out the output we're interested in.


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model that we want to use.

	forward: function
		A function that takes in a model and a batch of sequences and returns
		some output. Usually this is just running the forward function of the
		model and then slicing out an output.
	"""

	def __init__(self, model, forward):
		super(LambdaWrapper, self).__init__()
		self.model = model
		self._forward = forward

	def forward(self, X, *args):
		return self._forward(self.model, X, *args)


def test_pisa(X):
	torch.manual_seed(0)
	model = Conv1()

	X_attr1 = pisa(model, X, device='cpu', n_shuffles=3, 
		random_state=0, batch_size=1)
	X_attr2 = pisa(model, X, device='cpu', n_shuffles=3, 
		random_state=0, batch_size=4)

	assert X_attr1.shape == (2, 94, 4, 100)
	assert X_attr2.shape == (2, 94, 4, 100)

	assert X_attr1.dtype == torch.float32
	assert X_attr2.dtype == torch.float32

	assert_array_almost_equal(X_attr1, X_attr2)
	assert_array_almost_equal(X_attr1[:, :2, :, :5], [
		[[[ 0.0000,  0.0000, -0.0000,  0.0688, -0.0000],
          [-0.0000, -0.0000,  0.0425, -0.0000, -0.0000],
          [ 0.0000,  0.0000,  0.0000, -0.0000,  0.0000],
          [ 0.0000, -0.0642, -0.0000,  0.0000,  0.0376]],

         [[ 0.0000,  0.0000,  0.0000,  0.0221,  0.0000],
          [ 0.0000, -0.0000,  0.0043,  0.0000,  0.0000],
          [ 0.0000,  0.0000,  0.0000,  0.0000, -0.0000],
          [ 0.0000,  0.0164, -0.0000, -0.0000,  0.1269]]],


        [[[ 0.0000,  0.0000,  0.0221,  0.0000, -0.0328],
          [ 0.0000,  0.0000,  0.0000, -0.0000, -0.0000],
          [ 0.0000,  0.0149,  0.0000, -0.0000,  0.0000],
          [ 0.0000, -0.0000, -0.0000,  0.0408,  0.0000]],

         [[ 0.0000, -0.0000,  0.0776, -0.0000,  0.0103],
          [ 0.0000, -0.0000,  0.0000,  0.0000, -0.0000],
          [ 0.0000,  0.0190,  0.0000,  0.0000, -0.0000],
          [ 0.0000, -0.0000, -0.0000, -0.0592, -0.0000]]]
	], 4)


def test_pisa_deep_lift_shap(X):
	torch.manual_seed(0)
	model = Conv1()

	references = dinucleotide_shuffle(X, n=3)

	X_attr0 = pisa(model, X, device='cpu', references=references)
	X_attr1 = deep_lift_shap(model, X, device='cpu', references=references, 
		target=0)
	X_attr2 = deep_lift_shap(model, X, device='cpu', references=references,
		target=28)
	X_attr3 = deep_lift_shap(model, X, device='cpu', references=references,
		target=-1)

	assert_array_almost_equal(X_attr0[:, 0], X_attr1)
	assert_array_almost_equal(X_attr0[:, 28], X_attr2)
	assert_array_almost_equal(X_attr0[:, -1], X_attr3)


def test_pisa_convergence(X):
	torch.manual_seed(0)
	model = SmallDeepSEA()

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		pisa(model, X, device='cpu', n_shuffles=3, random_state=0,
			warning_threshold=1e-7)

		assert_raises(RuntimeWarning, deep_lift_shap, model, X, 
			device='cpu', n_shuffles=3, random_state=0, warning_threshold=1e-10)


def test_pisa_hypothetical(X):
	torch.manual_seed(0)
	model = Conv1()

	X_attr = pisa(model, X, hypothetical=True, device='cpu', 
		random_state=0)

	assert X_attr.shape == (2, 94, 4, 100)
	assert X_attr.dtype == torch.float32

	assert_array_almost_equal(X_attr[:, :3, :, :4], [
		[[[ 0.0000,  0.0482,  0.0029,  0.0731],
          [-0.0195,  0.0110,  0.0487, -0.0715],
          [ 0.0362,  0.0216,  0.0686, -0.1113],
          [ 0.0155, -0.0496, -0.0531,  0.0421]],

         [[ 0.0000, -0.0023,  0.0568, -0.0045],
          [ 0.0000, -0.0218,  0.0196,  0.0412],
          [ 0.0000,  0.0340,  0.0302,  0.0612],
          [ 0.0000,  0.0132, -0.0411, -0.0605]],

         [[ 0.0000,  0.0000, -0.0103,  0.0426],
          [ 0.0000,  0.0000, -0.0298,  0.0055],
          [ 0.0000,  0.0000,  0.0259,  0.0160],
          [ 0.0000,  0.0000,  0.0052, -0.0552]]],


        [[[ 0.0195,  0.0344,  0.0001,  0.1124],
          [ 0.0000, -0.0028,  0.0459, -0.0322],
          [ 0.0557,  0.0077,  0.0658, -0.0720],
          [ 0.0350, -0.0635, -0.0559,  0.0814]],

         [[ 0.0000, -0.0221,  0.0519, -0.0224],
          [ 0.0000, -0.0416,  0.0147,  0.0233],
          [ 0.0000,  0.0141,  0.0253,  0.0432],
          [ 0.0000, -0.0066, -0.0459, -0.0784]],

         [[ 0.0000,  0.0000, -0.0095,  0.0420],
          [ 0.0000,  0.0000, -0.0290,  0.0049],
          [ 0.0000,  0.0000,  0.0267,  0.0154],
          [ 0.0000,  0.0000,  0.0059, -0.0558]]]
	], 4)


def test_pisa_independence():
	X_ = random_one_hot((12, 4, 100), random_state=0).type(torch.float32)
	X = substitute(X_, "ACGTACGT")

	torch.manual_seed(0)
	model = Conv1()

	X_attr = pisa(model, X, device='cpu', random_state=0)
	X_attr0 = pisa(model, X[0:1], device='cpu', random_state=0)
	X_attr1 = pisa(model, X[5:6], device='cpu', random_state=0)
	X_attr2 = pisa(model, X[8:10], device='cpu', random_state=0)
	X_attr3 = pisa(model, X[0:10], device='cpu', random_state=0)

	assert_array_almost_equal(X_attr[0:1], X_attr0)
	assert_array_almost_equal(X_attr[5:6], X_attr1)
	assert_array_almost_equal(X_attr[8:10], X_attr2)
	assert_array_almost_equal(X_attr[0:10], X_attr3)

	assert_raises(AssertionError, assert_array_almost_equal, X_attr0, X_attr1)
	assert_raises(AssertionError, assert_array_almost_equal, X_attr2, 
		X_attr3[:2])


def test_pisa_random_state(X):
	torch.manual_seed(0)
	model = Conv1()

	X_attr0 = pisa(model, X, device='cpu', random_state=0)
	X_attr1 = pisa(model, X, device='cpu', random_state=1)
	X_attr2 = pisa(model, X, device='cpu', random_state=2)

	assert_raises(AssertionError, assert_array_almost_equal, X_attr0, X_attr1)
	assert_raises(AssertionError, assert_array_almost_equal, X_attr0, X_attr2)


def test_pisa_reference_tensor(X):
	torch.manual_seed(0)
	model = Conv1()

	references = shuffle(X, n=20, random_state=0)

	X_attr0 = pisa(model, X, references=references, device='cpu', random_state=0)
	X_attr1 = pisa(model, X, references=references, device='cpu', random_state=1)
	X_attr2 = pisa(model, X, references=references, device='cpu', random_state=2)

	assert_array_almost_equal(X_attr0, X_attr1)
	assert_array_almost_equal(X_attr0, X_attr2)


def test_pisa_batch_size(X):
	torch.manual_seed(0)
	model = Conv1()
	X = X[:1, :, :50]

	X_attr0 = pisa(model, X, device='cpu', random_state=0)
	X_attr1 = pisa(model, X, batch_size=1, device='cpu', random_state=0)
	X_attr2 = pisa(model, X, batch_size=1000, device='cpu', random_state=0)

	assert_array_almost_equal(X_attr0, X_attr1)
	assert_array_almost_equal(X_attr0, X_attr2)


def test_pisa_n_shuffles(X):
	torch.manual_seed(0)
	model = Conv1()

	X_attr0 = deep_lift_shap(model, X, n_shuffles=1, device='cpu', 
		random_state=0)
	X_attr1 = deep_lift_shap(model, X, n_shuffles=1, batch_size=1, device='cpu', 
		random_state=0)
	X_attr2 = deep_lift_shap(model, X, n_shuffles=30, batch_size=100000, 
		device='cpu', random_state=2)
	X_attr3 = deep_lift_shap(model, X, n_shuffles=30, batch_size=1, 
		device='cpu', random_state=2)

	assert_array_almost_equal(X_attr0, X_attr1)
	assert_array_almost_equal(X_attr2, X_attr3)
	assert_raises(AssertionError, assert_array_almost_equal, X_attr0, X_attr3)


def test_deep_lift_shap_shuffle_ordering(X):
	torch.manual_seed(0)
	model = Conv1()
	X = X[:1]

	references = dinucleotide_shuffle(X, n=1, random_state=0)

	X_attr0 = pisa(model, X, n_shuffles=1, device='cpu', random_state=0)
	X_attr1 = pisa(model, X, device='cpu', references=references)

	assert_array_almost_equal(X_attr0, X_attr1)


def test_pisa_raw_output(X):
	torch.manual_seed(0)
	model = Conv1()

	X_attr0, refs = pisa(model, X, device='cpu', raw_outputs=True, 
		random_state=0, return_references=True)
	X_attr1 = pisa(model, X, device='cpu', random_state=0)

	assert refs.shape == (2, 20, 4, 100)
	assert X_attr0.shape == (2, 20, 94, 4, 100)
	assert X_attr1.shape == (2, 94, 4, 100)

	assert_array_almost_equal(X_attr0[:, :2, :3, :, :3], [
		[[[[ 0.0054,  0.0253,  0.0155],
           [-0.0141, -0.0119,  0.0612],
           [ 0.0417, -0.0013,  0.0811],
           [ 0.0209, -0.0726, -0.0406]],

          [[ 0.0000,  0.0054,  0.0253],
           [ 0.0000, -0.0141, -0.0119],
           [ 0.0000,  0.0417, -0.0013],
           [ 0.0000,  0.0209, -0.0726]],

          [[ 0.0000,  0.0000,  0.0054],
           [ 0.0000,  0.0000, -0.0141],
           [ 0.0000,  0.0000,  0.0417],
           [ 0.0000,  0.0000,  0.0209]]],


         [[[ 0.0054,  0.0253,  0.0155],
           [-0.0141, -0.0119,  0.0612],
           [ 0.0417, -0.0013,  0.0811],
           [ 0.0209, -0.0726, -0.0406]],

          [[ 0.0000,  0.0054,  0.0253],
           [ 0.0000, -0.0141, -0.0119],
           [ 0.0000,  0.0417, -0.0013],
           [ 0.0000,  0.0209, -0.0726]],

          [[ 0.0000,  0.0000,  0.0054],
           [ 0.0000,  0.0000, -0.0141],
           [ 0.0000,  0.0000,  0.0417],
           [ 0.0000,  0.0000,  0.0209]]]],



        [[[[ 0.0054,  0.0253,  0.0155],
           [-0.0141, -0.0119,  0.0612],
           [ 0.0417, -0.0013,  0.0811],
           [ 0.0209, -0.0726, -0.0406]],

          [[ 0.0000,  0.0054,  0.0253],
           [ 0.0000, -0.0141, -0.0119],
           [ 0.0000,  0.0417, -0.0013],
           [ 0.0000,  0.0209, -0.0726]],

          [[ 0.0000,  0.0000,  0.0054],
           [ 0.0000,  0.0000, -0.0141],
           [ 0.0000,  0.0000,  0.0417],
           [ 0.0000,  0.0000,  0.0209]]],


         [[[ 0.0054,  0.0253,  0.0155],
           [-0.0141, -0.0119,  0.0612],
           [ 0.0417, -0.0013,  0.0811],
           [ 0.0209, -0.0726, -0.0406]],

          [[ 0.0000,  0.0054,  0.0253],
           [ 0.0000, -0.0141, -0.0119],
           [ 0.0000,  0.0417, -0.0013],
           [ 0.0000,  0.0209, -0.0726]],

          [[ 0.0000,  0.0000,  0.0054],
           [ 0.0000,  0.0000, -0.0141],
           [ 0.0000,  0.0000,  0.0417],
           [ 0.0000,  0.0000,  0.0209]]]]
	], 4)

	X_attr2 = hypothetical_attributions(
		(X_attr0.reshape(-1, 4, 100),), 
		(X.repeat_interleave(20*94, dim=0),), 
		(refs.reshape(-1, 4, 100).repeat_interleave(94, dim=0),)
	)[0]

	X_attr2 = X_attr2.reshape(X.shape[0], 20, 94, 4, 100)
	X_attr2 = torch.mean(X_attr2 * X.unsqueeze(1).unsqueeze(1), dim=1)

	assert_array_almost_equal(X_attr2, X_attr1, 4)


def test_pisa_return_references(X):
	torch.manual_seed(0)
	model = Conv1()

	attr, refs = pisa(model, X, n_shuffles=1, return_references=True,
		device='cpu', random_state=0)

	assert attr.shape == (2, 94, 4, 100)
	assert refs.shape == (2, 1, 4, 100)
	assert refs.dtype == torch.float32
	assert refs[:, 0].sum(dim=1).max() == 1

	assert_array_almost_equal(refs[:4, :, :, :10], [
		[[[1., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
          [0., 1., 0., 0., 1., 0., 0., 0., 0., 0.],
          [0., 0., 1., 0., 0., 0., 0., 0., 0., 1.],
          [0., 0., 0., 1., 0., 1., 1., 1., 0., 0.]]],

        [[[0., 0., 0., 1., 1., 0., 0., 0., 1., 0.],
          [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
          [0., 1., 1., 0., 0., 1., 0., 1., 0., 1.]]]])


	_, refs2 = pisa(model, X, n_shuffles=3, return_references=True,
		device='cpu', random_state=0)

	assert_array_almost_equal(refs, refs2[:, 0:1])


def test_pisa_args(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)
	alpha = torch.randn(16, 1)
	beta = torch.randn(16, 1)

	X_attr0 = pisa(model, X, device='cpu', random_state=0)[:, 0]
	X_attr1 = pisa(model, X, args=(alpha,), device='cpu', 
		random_state=0)[:, 0]
	X_attr2 = pisa(model, X, args=(alpha, beta), device='cpu', 
		random_state=0)[:, 0]

	assert X.shape == X_attr0.shape
	assert X.shape == X_attr1.shape
	assert X.shape == X_attr2.shape

	assert_array_almost_equal(X_attr0, X_attr1)
	assert_raises(AssertionError, assert_array_almost_equal, X_attr0, X_attr2)

	assert_array_almost_equal(X_attr2[:2, :, :10], [
		[[ 0.0000,  0.0000, -0.0000, -0.0209, -0.0000,  0.0000,  0.0000,
           0.0000, -0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0080,  0.0000, -0.0000,  0.0000, -0.0000,
          -0.0000, -0.0081,  0.0000],
         [ 0.0000,  0.0000, -0.0000, -0.0000, -0.0000,  0.0000, -0.0000,
           0.0000,  0.0000,  0.0000],
         [-0.0000, -0.0160,  0.0000,  0.0000,  0.0245, -0.0336,  0.0200,
           0.0050,  0.0000, -0.0085]],

        [[-0.0000, -0.0000, -0.0090, -0.0000, -0.0092,  0.0000,  0.0087,
           0.0000, -0.0000,  0.0079],
         [ 0.0000,  0.0000,  0.0000,  0.0000, -0.0000, -0.0000, -0.0000,
          -0.0000, -0.0000,  0.0000],
         [-0.0000,  0.0018, -0.0000, -0.0000, -0.0000,  0.0322, -0.0000,
           0.0000,  0.0000,  0.0000],
         [-0.0000, -0.0000,  0.0000,  0.0265,  0.0000, -0.0000,  0.0000,
          -0.0006,  0.0063, -0.0000]]
	], 4)


def test_pisa_raises(references):
	X_ = random_one_hot((16, 4, 100), random_state=0).type(torch.float32)
	X = substitute(X_, "ACGTACGT")
	
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)
	alpha = torch.randn(16, 1)
	beta = torch.randn(16, 1)

	assert_raises(ValueError, pisa, model, X[0], device='cpu')
	assert_raises(ValueError, pisa, model, X.unsqueeze(1), 
		device='cpu')
	assert_raises(RuntimeError, pisa, model, X, n_shuffles=0, 
		device='cpu')
	assert_raises(ValueError, pisa, model, X[0], device='cpu')

	assert_raises(IndexError, pisa, model, X, args=(alpha[:10],),
		device='cpu')
	assert_raises(IndexError, pisa, model, X, args=(alpha, beta[:3]),
		device='cpu')
	assert_raises(IndexError, pisa, model, X, args=(alpha[:5], 
		beta[:3]), device='cpu')
	assert_raises(IndexError, pisa, model, X, args=(alpha, beta[:3]),
		device='cpu')
	
	assert_raises(ValueError, pisa, model, X, 
		references=references[:10], device='cpu')
	assert_raises(ValueError, pisa, model, X, 
		references=references[:, :, :2], device='cpu')
	assert_raises(ValueError, pisa, model, X, 
		references=references[:, :, :, :10], device='cpu')


### Test a bunch of different models with different configurations/operations


def test_pisa_flattendense(X, references):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		X_attr = pisa(model, X, device='cpu', references=references, 
			warning_threshold=1e-5)

		assert_raises(RuntimeWarning, pisa, model, X, device='cpu', 
			random_state=0, warning_threshold=1e-10)

	X_attr2 = deep_lift_shap(model, X, device='cpu', references=references)
	
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:, 0], X_attr2)


def test_pisa_flattendense_n_outputs(X):
	model = FlattenDense(n_outputs=1)
	X_attr = pisa(model, X, device='cpu')
	assert X_attr.shape == (2, 1, 4, 100)

	model = FlattenDense(n_outputs=4)
	X_attr = pisa(model, X, device='cpu')
	assert X_attr.shape == (2, 4, 4, 100)

	model = FlattenDense(n_outputs=12)
	X_attr = pisa(model, X, device='cpu')
	assert X_attr.shape == (2, 12, 4, 100)


def test_pisa_convdense_dense_wrapper(X, references):
	torch.manual_seed(0)
	model = LambdaWrapper(ConvDense(n_outputs=1), lambda model, X: model(X)[1])

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		X_attr = pisa(model, X, device='cpu', references=references, 
			warning_threshold=1e-5)

		assert_raises(RuntimeWarning, deep_lift_shap, model, X, device='cpu', 
			random_state=0, warning_threshold=1e-8)

	X_attr2 = deep_lift_shap(model, X, device='cpu', references=references)
	assert_array_almost_equal(X_attr[:, 0], X_attr2)


def test_pisa_convdense_conv_wrapper(X, references):
	torch.manual_seed(0)
	model = LambdaWrapper(ConvDense(n_outputs=1), 
		lambda model, X: model(X)[0].sum(dim=(-1, -2)).unsqueeze(-1))

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		X_attr = pisa(model, X, device='cpu', references=references, 
			warning_threshold=1e-4)

		assert_raises(RuntimeWarning, deep_lift_shap, model, X, device='cpu', 
			random_state=0, warning_threshold=1e-8)

	X_attr2 = deep_lift_shap(model, X, device='cpu', references=references)
	assert_array_almost_equal(X_attr[:, 0], X_attr2)


### Now test some custom models with weird architectures just to check


class TorchSum(torch.nn.Module):
	def __init__(self):
		super(TorchSum, self).__init__()

	def forward(self, X):
		if len(X.shape) == 2:
			return torch.sum(X, dim=-1, keepdims=True)
		else:
			return torch.sum(X, dim=(-1, -2)).unsqueeze(-1)


def test_pisa_linear(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Flatten(),
		torch.nn.Linear(400, 5),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = pisa(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_pisa_linear_bias(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Flatten(),
		torch.nn.Linear(400, 5, bias=False),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = pisa(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_pisa_conv(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,)),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = pisa(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_pisa_conv_dilated(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,), dilation=5),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = pisa(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_pisa_conv_stride(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,), stride=5),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = pisa(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_pisa_conv_bias(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,), bias=False),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = pisa(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-4)


def test_pisa_conv_padding(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,), padding=5),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = pisa(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-4)


def test_pisa_conv_padding_same(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,), padding='same'),
		torch.nn.Flatten(),
		torch.nn.ReLU(),
		torch.nn.Linear(800, 1)
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = pisa(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_pisa_max_pool(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.MaxPool1d(4),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = pisa(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_pisa_conv_relu_pool(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,)),
		torch.nn.ReLU(),
		torch.nn.MaxPool1d(4),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = pisa(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_pisa_conv_tanh_pool(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,)),
		torch.nn.Tanh(),
		torch.nn.MaxPool1d(4),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = pisa(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_pisa_conv_elu_pool(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,)),
		torch.nn.ELU(),
		torch.nn.MaxPool1d(4),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = pisa(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_pisa_conv_relu_pool_relu(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,)),
		torch.nn.ReLU(),
		torch.nn.MaxPool1d(4),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = pisa(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_pisa_relu_conv_relu_pool_relu(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.ReLU(),
		torch.nn.Conv1d(4, 8, (5,)),
		torch.nn.ReLU(),
		torch.nn.MaxPool1d(4),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = pisa(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_pisa_relu_conv_pool_relu(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.ReLU(),
		torch.nn.Conv1d(4, 8, (5,)),
		torch.nn.MaxPool1d(4),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = pisa(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_pisa_relu_conv_pool_relu_relu(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.ReLU(),
		torch.nn.Conv1d(4, 8, (5,)),
		torch.nn.MaxPool1d(4),
		torch.nn.ReLU(),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = pisa(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_pisa_conv_relu_tanh_pool(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,)),
		torch.nn.ReLU(),
		torch.nn.Tanh(),
		torch.nn.MaxPool1d(4),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = pisa(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_pisa_conv_relu_pool_linear(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,)),
		torch.nn.ReLU(),
		torch.nn.MaxPool1d(4),
		torch.nn.Flatten(),
		torch.nn.Linear(192, 1)
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = pisa(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_pisa_conv_relu_pool_linear_linear(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,)),
		torch.nn.ReLU(),
		torch.nn.MaxPool1d(4),
		torch.nn.Flatten(),
		torch.nn.Linear(192, 10),
		torch.nn.Linear(10, 1)
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = pisa(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_pisa_conv_relu_pool_linear_relu_linear(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,)),
		torch.nn.ReLU(),
		torch.nn.MaxPool1d(4),
		torch.nn.Flatten(),
		torch.nn.Linear(192, 10),
		torch.nn.ReLU(),
		torch.nn.Linear(10, 1)
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = pisa(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)
