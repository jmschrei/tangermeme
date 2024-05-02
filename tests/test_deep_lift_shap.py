# test_deep_lift_shap.py
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

from tangermeme.deep_lift_shap import hypothetical_attributions
from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.deep_lift_shap import _captum_deep_lift_shap

from .toy_models import SumModel
from .toy_models import FlattenDense
from .toy_models import Conv
from .toy_models import Scatter
from .toy_models import ConvDense
from .toy_models import ConvPoolDense
from .toy_models import SmallDeepSEA

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


@pytest.fixture
def X():
	X_ = random_one_hot((16, 4, 100), random_state=0).type(torch.float32)
	X_ = substitute(X_, "ACGTACGT")
	return X_


@pytest.fixture
def references(X):
	return shuffle(X, n=5, random_state=0)


###


def test_hypothetical_attributions_ones(X):
	gradients = torch.ones_like(X)
	refs = shuffle(X, random_state=0)[:, 0]

	attributions = hypothetical_attributions((gradients,), (X,), (refs,))

	assert isinstance(attributions, tuple)
	assert len(attributions) == 1
	assert attributions[0].shape == X.shape
	assert_array_almost_equal(attributions[0], torch.zeros_like(X))


def test_hypothetical_attributions(X):
	torch.manual_seed(0)
	gradients = torch.randn_like(X)
	refs = shuffle(X, random_state=0)[:, 0]

	attributions = hypothetical_attributions((gradients,), (X,), (refs,))

	assert isinstance(attributions, tuple)
	assert len(attributions) == 1
	assert attributions[0].shape == X.shape
	assert_array_almost_equal(attributions[0][:3, :, :3], [
		[[ 1.4875,  0.5441, -0.0223],
         [ 0.0000,  0.0000,  0.0000],
         [ 2.5389,  0.6043,  0.6203],
         [ 4.0762,  1.0760,  1.2166]],

        [[-0.6003,  0.0000,  0.0000],
         [ 0.0000, -1.2123, -1.1973],
         [ 0.3125,  2.1779,  1.1614],
         [ 2.0122, -2.6358,  0.2413]],

        [[ 0.0000, -1.3459,  0.3138],
         [-0.7441, -0.1906,  0.6102],
         [ 1.7111,  0.0000,  0.5598],
         [ 1.5918,  0.4912,  0.0000]]], 4)


def test_hypothetical_attributions_half_precision(X):
	torch.manual_seed(0)
	gradients = torch.randn_like(X)
	refs = shuffle(X, random_state=0)[:, 0]

	attributions = hypothetical_attributions((gradients.half(),), (X.half(),), 
		(refs.half(),))

	assert attributions[0].dtype == torch.float16
	assert isinstance(attributions, tuple)
	assert len(attributions) == 1
	assert attributions[0].shape == X.shape
	assert_array_almost_equal(attributions[0][:3, :, :3], [
		[[ 1.4875,  0.5441, -0.0223],
         [ 0.0000,  0.0000,  0.0000],
         [ 2.5389,  0.6043,  0.6203],
         [ 4.0762,  1.0760,  1.2166]],

        [[-0.6003,  0.0000,  0.0000],
         [ 0.0000, -1.2123, -1.1973],
         [ 0.3125,  2.1779,  1.1614],
         [ 2.0122, -2.6358,  0.2413]],

        [[ 0.0000, -1.3459,  0.3138],
         [-0.7441, -0.1906,  0.6102],
         [ 1.7111,  0.0000,  0.5598],
         [ 1.5918,  0.4912,  0.0000]]], 2)


def test_hypothetical_attributions_independence(X):
	torch.manual_seed(0)
	gradients = torch.randn_like(X)
	refs = shuffle(X, random_state=0)[:, 0]

	attributions0 = hypothetical_attributions((gradients,), (X,), (refs,))
	attributions1 = hypothetical_attributions((gradients[:1],), (X[:1],), 
		(refs[:1],))

	assert isinstance(attributions1, tuple)
	assert len(attributions1) == 1
	assert attributions1[0].shape == X[:1].shape
	assert_array_almost_equal(attributions1[0][:, :, :3], [
		[[ 1.4875,  0.5441, -0.0223],
         [ 0.0000,  0.0000,  0.0000],
         [ 2.5389,  0.6043,  0.6203],
         [ 4.0762,  1.0760,  1.2166]]], 4)
	assert_array_almost_equal(attributions0[0][:1], attributions1[0])


def test_hypothetical_attributions_raises(X):
	gradients = torch.ones_like(X)
	refs = shuffle(X, random_state=0)[:, 0]


	assert_raises(ValueError, hypothetical_attributions, gradients, (X,), 
		(refs,))
	assert_raises(ValueError, hypothetical_attributions, (gradients,), X, 
		(refs,))
	assert_raises(ValueError, hypothetical_attributions, (gradients,), (X,), 
		refs)

	assert_raises(ValueError, hypothetical_attributions, (gradients, gradients), 
		(X,), (refs,))
	assert_raises(ValueError, hypothetical_attributions, (gradients,), (X, X), 
		(refs,))
	assert_raises(ValueError, hypothetical_attributions, (gradients,), (X,), 
		(refs, refs))

	assert_raises(ValueError, hypothetical_attributions, (gradients.numpy()), 
		(X,), (refs,))
	assert_raises(ValueError, hypothetical_attributions, (gradients,), 
		(X.numpy(),), (refs,))
	assert_raises(ValueError, hypothetical_attributions, (gradients,), (X,), 
		(refs.numpy(),))


###


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


def test_deep_lift_shap(X):
	torch.manual_seed(0)
	model = SmallDeepSEA()

	X_attr1 = deep_lift_shap(model, X[:4], device='cpu', n_shuffles=3, 
		random_state=0, batch_size=1)
	X_attr2 = deep_lift_shap(model, X[:4], device='cpu', n_shuffles=3, 
		random_state=0, batch_size=4)

	assert X_attr1.shape == X[:4].shape
	assert X_attr1.dtype == torch.float32

	assert_array_almost_equal(X_attr1, X_attr2)
	assert_array_almost_equal(X_attr1[:, :, :5], [
		[[ 0.0000e+00, -0.0000e+00,  0.0000e+00,  4.6475e-05,  0.0000e+00],
         [ 0.0000e+00,  0.0000e+00, -1.3578e-03, -0.0000e+00,  0.0000e+00],
         [ 0.0000e+00, -0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
         [-0.0000e+00, -7.6549e-04,  0.0000e+00,  0.0000e+00, -1.1941e-03]],

        [[ 0.0000e+00, -0.0000e+00,  3.5152e-04, -0.0000e+00,  4.3590e-04],
         [ 0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
         [-0.0000e+00, -3.8318e-04,  0.0000e+00,  0.0000e+00,  0.0000e+00],
         [-0.0000e+00, -0.0000e+00, -0.0000e+00, -1.8738e-03,  0.0000e+00]],

        [[ 0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
         [ 0.0000e+00, -0.0000e+00, -0.0000e+00,  0.0000e+00,  0.0000e+00],
         [-0.0000e+00,  2.6702e-03,  0.0000e+00,  4.6532e-03,  3.4494e-03],
         [-0.0000e+00, -0.0000e+00, -1.3396e-03,  0.0000e+00,  0.0000e+00]],

        [[-0.0000e+00, -4.8061e-04, -0.0000e+00,  3.0198e-03, -0.0000e+00],
         [ 0.0000e+00, -0.0000e+00,  0.0000e+00, -0.0000e+00, -0.0000e+00],
         [-0.0000e+00, -0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
         [-0.0000e+00, -0.0000e+00, -1.4029e-03,  0.0000e+00, -0.0000e+00]]], 4)


def test_deep_lift_shap_convergence(X):
	torch.manual_seed(0)
	model = SmallDeepSEA()

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		deep_lift_shap(model, X[:4], device='cpu', n_shuffles=3, random_state=0,
			warning_threshold=1e-7)

		assert_raises(RuntimeWarning, deep_lift_shap, model, X[:4], 
			device='cpu', n_shuffles=3, random_state=0, warning_threshold=1e-10)


def test_deep_lift_shap_hypothetical(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	X_attr = deep_lift_shap(model, X, hypothetical=True, device='cpu', 
		random_state=0)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000, -0.0005, -0.0100, -0.0291,  0.0008,  0.0205,  0.0073,
           0.0246, -0.0123,  0.0090],
         [ 0.0474,  0.0063,  0.0093, -0.0049, -0.0276,  0.0062, -0.0294,
          -0.0537, -0.0106,  0.0033],
         [ 0.0025,  0.0049, -0.0067, -0.0267, -0.0090,  0.0421, -0.0097,
           0.0271,  0.0102,  0.0021],
         [-0.0237, -0.0184,  0.0052,  0.0192,  0.0299, -0.0300,  0.0166,
           0.0062,  0.0119, -0.0105]],

        [[-0.0474, -0.0007, -0.0088, -0.0109, -0.0030,  0.0079,  0.0129,
           0.0104, -0.0164,  0.0098],
         [ 0.0000,  0.0062,  0.0106,  0.0133, -0.0314, -0.0064, -0.0237,
          -0.0679, -0.0147,  0.0040],
         [-0.0449,  0.0047, -0.0054, -0.0084, -0.0128,  0.0295, -0.0041,
           0.0129,  0.0062,  0.0029],
         [-0.0712, -0.0185,  0.0064,  0.0374,  0.0261, -0.0426,  0.0222,
          -0.0080,  0.0079, -0.0098]]], 4)


def test_deep_lift_shap_independence(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	X_attr = deep_lift_shap(model, X, device='cpu', random_state=0)
	X_attr0 = deep_lift_shap(model, X[0:1], device='cpu', random_state=0)
	X_attr1 = deep_lift_shap(model, X[5:6], device='cpu', random_state=0)
	X_attr2 = deep_lift_shap(model, X[8:10], device='cpu', random_state=0)
	X_attr3 = deep_lift_shap(model, X[0:10], device='cpu', random_state=0)

	assert_array_almost_equal(X_attr[0:1], X_attr0)
	assert_array_almost_equal(X_attr[5:6], X_attr1)
	assert_array_almost_equal(X_attr[8:10], X_attr2)
	assert_array_almost_equal(X_attr[0:10], X_attr3)

	assert_raises(AssertionError, assert_array_almost_equal, X_attr0, X_attr1)
	assert_raises(AssertionError, assert_array_almost_equal, X_attr2, 
		X_attr3[:2])


def test_deep_lift_shap_random_state(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	X_attr0 = deep_lift_shap(model, X, device='cpu', random_state=0)
	X_attr1 = deep_lift_shap(model, X[0:10], device='cpu', random_state=1)
	X_attr2 = deep_lift_shap(model, X[0:10], device='cpu', random_state=2)

	assert_raises(AssertionError, assert_array_almost_equal, X_attr0, X_attr1)
	assert_raises(AssertionError, assert_array_almost_equal, X_attr0, X_attr2)


def test_deep_lift_shap_reference_tensor(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	references = shuffle(X, n=20, random_state=0)

	X_attr0 = deep_lift_shap(model, X, references=references, device='cpu', 
		random_state=0)
	X_attr1 = deep_lift_shap(model, X[0:10], references=references, 
		device='cpu', random_state=1)
	X_attr2 = deep_lift_shap(model, X[0:10], references=references, 
		device='cpu', random_state=2)

	assert_array_almost_equal(X_attr0[:10], X_attr1)
	assert_array_almost_equal(X_attr0[:10], X_attr2)
	assert_array_almost_equal(X_attr0[:2, :, :10], [
		[[-0.0070,  0.0000, -0.0000, -0.0177,  0.0000,  0.0000,  0.0000,
           0.0000, -0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0085,  0.0000, -0.0000,  0.0000, -0.0000,
          -0.0000, -0.0115,  0.0000],
         [-0.0000,  0.0000, -0.0000, -0.0000,  0.0000,  0.0000, -0.0000,
           0.0000,  0.0000,  0.0000],
         [-0.0000, -0.0125,  0.0000,  0.0000,  0.0390, -0.0310,  0.0152,
           0.0005,  0.0000, -0.0105]],

        [[ 0.0000,  0.0000, -0.0074, -0.0000, -0.0044,  0.0000,  0.0081,
           0.0000, -0.0000,  0.0045],
         [ 0.0488,  0.0000,  0.0000,  0.0000, -0.0000, -0.0000, -0.0000,
          -0.0000, -0.0000, -0.0000],
         [ 0.0000,  0.0108, -0.0000, -0.0000, -0.0000,  0.0288, -0.0000,
           0.0000,  0.0000, -0.0000],
         [-0.0000, -0.0000,  0.0000,  0.0258,  0.0000, -0.0000,  0.0000,
          -0.0055,  0.0078, -0.0000]]], 4)


def test_deep_lift_shap_batch_size(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	X_attr0 = deep_lift_shap(model, X, device='cpu', random_state=0)
	X_attr1 = deep_lift_shap(model, X, batch_size=1, device='cpu', 
		random_state=0)
	X_attr2 = deep_lift_shap(model, X, batch_size=100000, device='cpu', 
		random_state=0)
	X_attr3 = deep_lift_shap(model, X, batch_size=20, device='cpu', 
		random_state=0)

	assert_array_almost_equal(X_attr0, X_attr1)
	assert_array_almost_equal(X_attr0, X_attr2)
	assert_array_almost_equal(X_attr0, X_attr3)


def test_deep_lift_shap_n_shuffles(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

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
	model = SmallDeepSEA()
	X = X[:1]

	references = dinucleotide_shuffle(X, n=1, random_state=0)

	X_attr0 = deep_lift_shap(model, X, n_shuffles=1, device='cpu', random_state=0)
	X_attr1 = deep_lift_shap(model, X, device='cpu', references=references)

	assert_array_almost_equal(X_attr0, X_attr1)


def test_deep_lift_shap_raw_output(X):
	torch.manual_seed(0)
	model = SmallDeepSEA()

	X_attr0, refs = deep_lift_shap(model, X, device='cpu', raw_outputs=True, 
		random_state=0, return_references=True)
	X_attr1 = deep_lift_shap(model, X, device='cpu', random_state=0)

	assert X_attr0.shape == (16, 20, 4, 100)
	assert X_attr1.shape == (16, 4, 100)

	assert_array_almost_equal(X_attr0[:2, :4, :, :5], [
		[[[-5.0116e-04,  3.1871e-04, -9.6379e-04,  1.8907e-03, -7.4686e-04],
          [-3.3056e-04,  2.2861e-03, -1.8157e-03, -1.0250e-03,  2.0957e-04],
          [ 2.0393e-04, -4.8745e-04,  1.5248e-04, -1.5472e-03,  3.1051e-03],
          [-8.2290e-04,  1.3244e-03, -8.3741e-04,  1.3078e-03, -2.3938e-03]],

         [[-4.1715e-04, -1.9015e-04, -2.3618e-04, -2.2531e-03, -4.3028e-05],
          [-2.2356e-04,  1.0974e-03, -2.0263e-03, -1.3769e-03,  2.0427e-03],
          [ 5.0492e-05,  1.0809e-04,  7.8811e-05,  2.5151e-03, -1.7192e-03],
          [-6.9432e-04, -3.1055e-04,  1.3494e-03, -2.7921e-03,  1.0637e-03]],

         [[-4.9491e-04,  2.1549e-03,  3.9226e-03,  6.0653e-04,  5.1495e-04],
          [ 2.1132e-04,  2.4496e-03, -3.1228e-03, -2.1822e-03, -2.0042e-03],
          [ 5.1454e-04, -1.3003e-03, -2.1690e-03,  1.3122e-03,  2.0167e-03],
          [-7.0201e-04,  2.5229e-03, -1.1406e-03,  1.5891e-03, -2.8907e-03]],

         [[-1.1855e-03,  1.0614e-03, -1.3309e-03,  7.7170e-04, -7.9025e-05],
          [-5.6384e-04,  2.9248e-03, -1.7246e-03, -6.9900e-04, -1.3818e-03],
          [ 1.7461e-04, -2.1326e-03, -6.4903e-04,  4.5011e-04,  4.1617e-03],
          [-1.4366e-03,  2.5167e-03, -8.7405e-04,  3.1704e-03, -2.4934e-03]]],


        [[[ 3.9191e-05, -1.6843e-04,  3.5755e-04, -1.4642e-03,  2.1020e-03],
          [ 6.6232e-04, -2.2308e-03, -7.2142e-04, -2.9278e-03,  7.8001e-04],
          [ 4.3716e-04, -1.2051e-03,  2.7829e-04,  1.0400e-03,  2.3251e-03],
          [ 1.4334e-05, -5.5578e-05, -1.3114e-03, -5.9387e-04,  2.5591e-03]],

         [[ 9.7985e-04, -2.9948e-03,  6.7705e-04, -3.5860e-04,  3.3796e-03],
          [ 1.9771e-03, -2.8396e-03, -1.8344e-03, -9.6604e-04,  8.0935e-04],
          [ 2.5890e-04,  4.6706e-03,  1.5568e-04,  1.8312e-03,  1.8603e-03],
          [-2.3563e-03, -3.9171e-03, -1.9084e-03,  3.6015e-04,  3.1807e-03]],

         [[ 2.5448e-03, -1.8576e-03, -7.4840e-04, -1.5695e-03,  1.8195e-04],
          [-2.9077e-04, -1.9621e-03, -2.3485e-03,  7.9775e-04, -3.1597e-03],
          [-1.3009e-03,  6.4424e-04, -1.2023e-03,  2.8251e-03,  3.9510e-04],
          [-1.2810e-03, -2.5857e-03, -1.0124e-03, -1.7189e-03, -6.6487e-05]],

         [[-3.2194e-04, -2.0264e-03,  8.4567e-04, -1.0215e-03,  1.1749e-03],
          [-2.2850e-04, -2.4731e-03, -1.1539e-03,  2.5856e-04, -2.6096e-03],
          [-6.5529e-04, -1.6158e-03, -7.1705e-04,  3.2077e-03,  5.8663e-04],
          [-1.4327e-03, -1.9454e-03,  9.8869e-05, -2.7312e-03, -5.6883e-05]]]], 
        4)

	X_attr2 = hypothetical_attributions((X_attr0.reshape(-1, 4, 100),), 
		(X.repeat_interleave(20, dim=0),), (refs.reshape(-1, 4, 100),))[0]
	X_attr2 = X_attr2.reshape(X.shape[0], 20, 4, 100)
	X_attr2 = torch.mean(X_attr2 * X.unsqueeze(1), dim=1)

	assert_array_almost_equal(X_attr2, X_attr1, 4)


def test_deep_lift_shap_return_references(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	attr, refs = deep_lift_shap(model, X, n_shuffles=1, return_references=True,
		device='cpu', random_state=0)

	assert attr.shape == X.shape
	assert refs.shape == (16, 1, 4, 100)
	assert refs.dtype == torch.float32
	assert refs[:, 0].sum(dim=1).max() == 1

	assert_array_almost_equal(refs[:4, :, :, :10], [
		[[[1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
          [0., 1., 0., 0., 1., 1., 0., 0., 0., 0.],
          [0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
          [0., 0., 0., 1., 0., 0., 0., 0., 1., 1.]]],


        [[[0., 0., 0., 0., 0., 0., 0., 1., 1., 0.],
          [1., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
          [0., 0., 1., 1., 0., 1., 0., 0., 0., 0.],
          [0., 1., 0., 0., 0., 0., 1., 0., 0., 1.]]],


        [[[0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
          [1., 0., 1., 0., 0., 0., 0., 1., 1., 1.],
          [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 1., 0., 1., 1., 0., 0., 0.]]],


        [[[0., 1., 0., 1., 0., 0., 0., 0., 1., 1.],
          [1., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 1., 1., 0., 1., 0., 0.],
          [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]]]])


	_, refs2 = deep_lift_shap(model, X, n_shuffles=3, return_references=True,
		device='cpu', random_state=0)

	assert_array_almost_equal(refs, refs2[:, 0:1])


def test_deep_lift_shap_args(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)
	alpha = torch.randn(16, 1)
	beta = torch.randn(16, 1)

	X_attr0 = deep_lift_shap(model, X, device='cpu', random_state=0)
	X_attr1 = deep_lift_shap(model, X, args=(alpha,), device='cpu', 
		random_state=0)
	X_attr2 = deep_lift_shap(model, X, args=(alpha, beta), device='cpu', 
		random_state=0)

	assert X.shape == X_attr0.shape
	assert X.shape == X_attr1.shape
	assert X.shape == X_attr2.shape

	assert_array_almost_equal(X_attr0, X_attr1)
	assert_raises(AssertionError, assert_array_almost_equal, X_attr0, X_attr2)
	assert_array_almost_equal(X_attr2[:2, :, :10], [
		[[ 0.0000, -0.0000, -0.0000, -0.0285,  0.0000,  0.0000,  0.0000,
           0.0000, -0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0091, -0.0000, -0.0000,  0.0000, -0.0000,
          -0.0000, -0.0104,  0.0000],
         [ 0.0000,  0.0000, -0.0000, -0.0000, -0.0000,  0.0000, -0.0000,
           0.0000,  0.0000,  0.0000],
         [-0.0000, -0.0180,  0.0000,  0.0000,  0.0293, -0.0294,  0.0163,
           0.0061,  0.0000, -0.0103]],

        [[ 0.0000,  0.0000,  0.0094,  0.0000,  0.0032, -0.0000, -0.0138,
          -0.0000,  0.0000, -0.0104],
         [ 0.0000, -0.0000, -0.0000, -0.0000,  0.0000,  0.0000,  0.0000,
           0.0000,  0.0000, -0.0000],
         [ 0.0000, -0.0051,  0.0000,  0.0000,  0.0000, -0.0315,  0.0000,
          -0.0000, -0.0000, -0.0000],
         [ 0.0000,  0.0000, -0.0000, -0.0399, -0.0000,  0.0000, -0.0000,
           0.0085, -0.0084,  0.0000]]], 4)


def test_deep_lift_shap_raises(X, references):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)
	alpha = torch.randn(16, 1)
	beta = torch.randn(16, 1)

	assert_raises(ValueError, deep_lift_shap, model, X[0], device='cpu')
	assert_raises(ValueError, deep_lift_shap, model, X.unsqueeze(1), 
		device='cpu')
	assert_raises(RuntimeError, deep_lift_shap, model, X, n_shuffles=0, 
		device='cpu')
	assert_raises(ValueError, deep_lift_shap, model, X[0], device='cpu')

	assert_raises(IndexError, deep_lift_shap, model, X, args=(alpha[:10],),
		device='cpu')
	assert_raises(IndexError, deep_lift_shap, model, X, args=(alpha, beta[:3]),
		device='cpu')
	assert_raises(IndexError, deep_lift_shap, model, X, args=(alpha[:5], 
		beta[:3]), device='cpu')
	assert_raises(IndexError, deep_lift_shap, model, X, args=(alpha, beta[:3]),
		device='cpu')
	
	assert_raises(IndexError, deep_lift_shap, model, X, 
		references=references[:10], device='cpu')
	assert_raises(AssertionError, deep_lift_shap, model, X, 
		references=references[:, :, :2], device='cpu')
	assert_raises(AssertionError, deep_lift_shap, model, X, 
		references=references[:, :, :, :10], device='cpu')


### Test a bunch of different models with different configurations/operations


def test_deep_lift_shap_flattendense(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)

		assert_raises(RuntimeWarning, deep_lift_shap, model, X, device='cpu', 
			random_state=0, warning_threshold=1e-10)

	assert X_attr.shape == X.shape
	assert X.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000, -0.0000, -0.0000, -0.0291,  0.0000,  0.0000,  0.0000,
           0.0000, -0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0093, -0.0000, -0.0000,  0.0000, -0.0000,
          -0.0000, -0.0106,  0.0000],
         [ 0.0000,  0.0000, -0.0000, -0.0000, -0.0000,  0.0000, -0.0000,
           0.0000,  0.0000,  0.0000],
         [-0.0000, -0.0184,  0.0000,  0.0000,  0.0299, -0.0300,  0.0166,
           0.0062,  0.0000, -0.0105]],

        [[-0.0000, -0.0000, -0.0088, -0.0000, -0.0030,  0.0000,  0.0129,
           0.0000, -0.0000,  0.0098],
         [ 0.0000,  0.0000,  0.0000,  0.0000, -0.0000, -0.0000, -0.0000,
          -0.0000, -0.0000,  0.0000],
         [-0.0000,  0.0047, -0.0000, -0.0000, -0.0000,  0.0295, -0.0000,
           0.0000,  0.0000,  0.0000],
         [-0.0000, -0.0000,  0.0000,  0.0374,  0.0000, -0.0000,  0.0000,
          -0.0080,  0.0079, -0.0000]]], 4)


def test_deep_lift_shap_convdense_dense_wrapper(X):
	torch.manual_seed(0)
	model = LambdaWrapper(ConvDense(n_outputs=1), lambda model, X: model(X)[1])

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)

		assert_raises(RuntimeWarning, deep_lift_shap, model, X, device='cpu', 
			random_state=0, warning_threshold=1e-8)

	assert X_attr.shape == X.shape
	assert X.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000, -0.0000, -0.0000, -0.0291,  0.0000,  0.0000,  0.0000,
           0.0000, -0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0093, -0.0000, -0.0000,  0.0000, -0.0000,
          -0.0000, -0.0106,  0.0000],
         [ 0.0000,  0.0000, -0.0000, -0.0000, -0.0000,  0.0000, -0.0000,
           0.0000,  0.0000,  0.0000],
         [-0.0000, -0.0184,  0.0000,  0.0000,  0.0299, -0.0300,  0.0166,
           0.0062,  0.0000, -0.0105]],

        [[-0.0000, -0.0000, -0.0088, -0.0000, -0.0030,  0.0000,  0.0129,
           0.0000, -0.0000,  0.0098],
         [ 0.0000,  0.0000,  0.0000,  0.0000, -0.0000, -0.0000, -0.0000,
          -0.0000, -0.0000,  0.0000],
         [-0.0000,  0.0047, -0.0000, -0.0000, -0.0000,  0.0295, -0.0000,
           0.0000,  0.0000,  0.0000],
         [-0.0000, -0.0000,  0.0000,  0.0374,  0.0000, -0.0000,  0.0000,
          -0.0080,  0.0079, -0.0000]]], 4)


def test_deep_lift_shap_convdense_conv_wrapper(X):
	torch.manual_seed(0)
	model = LambdaWrapper(ConvDense(n_outputs=1), 
		lambda model, X: model(X)[0].sum(dim=(-1, -2)).unsqueeze(-1))

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-4)

		assert_raises(RuntimeWarning, deep_lift_shap, model, X, device='cpu', 
			random_state=0, warning_threshold=1e-8)

	assert X_attr.shape == X.shape
	assert X.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000, -0.0000, -0.0000, -0.6011, -0.0000, -0.0000, -0.0000,
          -0.0000, -0.0000, -0.0000],
         [ 0.0000, -0.0000, -1.0128, -0.0000, -0.0000, -0.0000, -0.0000,
          -0.0000, -0.8623, -0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
           0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.9606,  0.0000,  0.0000,  0.6272,  0.8009,  0.5142,
           0.6995,  0.0000,  0.6272]],

        [[-0.0000, -0.0000, -0.9429, -0.0000, -0.8647, -0.0000, -0.6245,
          -0.0000, -0.0000, -0.8647],
         [ 0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000,
          -0.0000, -0.0000, -0.0000],
         [ 0.0000,  0.9682,  0.0000,  0.0000,  0.0000,  0.6544,  0.0000,
           0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.6095,  0.0000,  0.0000,  0.0000,
           0.4360,  0.3637,  0.0000]]], 4)


### Now test some custom models with weird architectures just to check


class TorchSum(torch.nn.Module):
	def __init__(self):
		super(TorchSum, self).__init__()

	def forward(self, X):
		if len(X.shape) == 2:
			return torch.sum(X, dim=-1, keepdims=True)
		else:
			return torch.sum(X, dim=(-1, -2)).unsqueeze(-1)


def test_deep_lift_shap_linear(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Flatten(),
		torch.nn.Linear(400, 5),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_deep_lift_shap_linear_bias(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Flatten(),
		torch.nn.Linear(400, 5, bias=False),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_deep_lift_shap_conv(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,)),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_deep_lift_shap_conv_dilated(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,), dilation=5),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_deep_lift_shap_conv_stride(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,), stride=5),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_deep_lift_shap_conv_bias(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,), bias=False),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-4)


def test_deep_lift_shap_conv_padding(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,), padding=5),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-4)


def test_deep_lift_shap_conv_padding_same(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,), padding='same'),
		torch.nn.Flatten(),
		torch.nn.ReLU(),
		torch.nn.Linear(800, 1)
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_deep_lift_shap_max_pool(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.MaxPool1d(4),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_deep_lift_shap_conv_relu_pool(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,)),
		torch.nn.ReLU(),
		torch.nn.MaxPool1d(4),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_deep_lift_shap_conv_tanh_pool(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,)),
		torch.nn.Tanh(),
		torch.nn.MaxPool1d(4),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_deep_lift_shap_conv_elu_pool(X):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,)),
		torch.nn.ELU(),
		torch.nn.MaxPool1d(4),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_deep_lift_shap_conv_relu_pool_relu(X):
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
		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_deep_lift_shap_relu_conv_relu_pool_relu(X):
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
		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_deep_lift_shap_relu_conv_pool_relu(X):
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
		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_deep_lift_shap_relu_conv_pool_relu_relu(X):
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
		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_deep_lift_shap_conv_relu_tanh_pool(X):
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
		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_deep_lift_shap_conv_relu_pool_linear(X):
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
		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_deep_lift_shap_conv_relu_pool_linear_linear(X):
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
		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


def test_deep_lift_shap_conv_relu_pool_linear_relu_linear(X):
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
		X_attr = deep_lift_shap(model, X, device='cpu', random_state=0, 
			warning_threshold=1e-5)


###


def _captum_attribute_comparison(model, X, references):
	torch.manual_seed(0)

	X_attr0 = deep_lift_shap(model, X, references=references, device='cpu', 
		random_state=0)
	X_attr1 = _captum_deep_lift_shap(model, X, references=references,
		device='cpu', random_state=0)

	assert X_attr0.shape == X_attr1.shape
	assert X_attr0.dtype == X_attr1.dtype
	assert_array_almost_equal(X_attr0, X_attr1, 5)


def test_captum_deep_lift_shap_summodel(X, references):
	model = LambdaWrapper(SumModel(), lambda model, X: model(X)[:, 0:1])
	_captum_attribute_comparison(model, X, references)


def test_captum_deep_lift_shap_flattendense(X, references):
	_captum_attribute_comparison(FlattenDense(n_outputs=1), X, references)


def test_captum_deep_lift_shap_scatter(X, references):
	model = LambdaWrapper(Scatter(), lambda model, X: model(X)[:, 0, 0:1])
	_captum_attribute_comparison(model, X, references)


def test_captum_deep_lift_shap_conv(X, references):
	model = LambdaWrapper(Conv(), lambda model, X: model(X).sum(
		dim=(-1, -2)).unsqueeze(-1))
	_captum_attribute_comparison(model, X, references)


#def test_captum_deep_lift_shap_convpooldense(X, references):
#	_captum_attribute_comparison(ConvPoolDense(), X, references)


def test_captum_deep_lift_shap_batch_size(X, references):
	_SumModel = LambdaWrapper(SumModel(), lambda model, X: model(X)[:, 0:1])
	_Scatter = LambdaWrapper(Scatter(), lambda model, X: model(X)[:, 0, 0:1])
	_Conv = LambdaWrapper(Conv(), lambda model, X: model(X).sum(dim=1)[:, 0:1])

	for model in _SumModel, FlattenDense(n_outputs=1), _Scatter, _Conv:
		torch.manual_seed(0)

		X_attr0 = deep_lift_shap(model, X, references=references, 
			batch_size=1, device='cpu', random_state=0)
		X_attr1 = _captum_deep_lift_shap(model, X, references=references, 
			batch_size=1, device='cpu', random_state=0)

		assert X_attr0.shape == X_attr1.shape
		assert X_attr0.dtype == X_attr1.dtype
		assert_array_almost_equal(X_attr0, X_attr1)


def test_captum_deep_lift_shap_n_shuffles(X, references):
	_SumModel = LambdaWrapper(SumModel(), lambda model, X: model(X)[:, 0:1])
	_Scatter = LambdaWrapper(Scatter(), lambda model, X: model(X)[:, 0, 0:1])
	_Conv = LambdaWrapper(Conv(), lambda model, X: model(X).sum(dim=1)[:, 0:1])

	for model in _SumModel, FlattenDense(n_outputs=1), _Scatter, _Conv:
		torch.manual_seed(0)

		X_attr0 = deep_lift_shap(model, X[:4], references=references, 
			n_shuffles=3, batch_size=1, device='cpu', random_state=0)
		X_attr1 = _captum_deep_lift_shap(model, X[:4], references=references,
			n_shuffles=3, batch_size=1, device='cpu', random_state=0)

		assert X_attr0.shape == X_attr1.shape
		assert X_attr0.dtype == X_attr1.dtype
		assert_array_almost_equal(X_attr0, X_attr1)


def test_captum_deep_lift_shap_args(X, references):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)
	alpha = torch.randn(16, 1)
	beta = torch.randn(16, 1)

	X_attr0 = deep_lift_shap(model, X, args=(alpha,), references=references, 
		device='cpu', random_state=0)
	X_attr1 = _captum_deep_lift_shap(model, X, args=(alpha,), 
		references=references, device='cpu', random_state=0)

	assert X_attr0.shape == X_attr1.shape
	assert X_attr0.dtype == X_attr1.dtype
	assert_array_almost_equal(X_attr0, X_attr1)


	X_attr0 = deep_lift_shap(model, X, args=(alpha, beta), 
		references=references, device='cpu', random_state=0)
	X_attr1 = _captum_deep_lift_shap(model, X, args=(alpha, beta), 
		references=references, device='cpu', random_state=0)

	assert X_attr0.shape == X_attr1.shape
	assert X_attr0.dtype == X_attr1.dtype
	assert_array_almost_equal(X_attr0, X_attr1)
