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
				[[ 0.0000, -0.0000, -0.0000, -0.0006, -0.0000],
         [ 0.0000,  0.0000, -0.0025, -0.0000,  0.0000],
         [ 0.0000, -0.0000, -0.0000,  0.0000, -0.0000],
         [-0.0000, -0.0004,  0.0000,  0.0000, -0.0014]],

        [[ 0.0000, -0.0000,  0.0006,  0.0000,  0.0020],
         [ 0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
         [-0.0000,  0.0012,  0.0000,  0.0000,  0.0000],
         [-0.0000, -0.0000, -0.0000, -0.0020,  0.0000]],

        [[ 0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
         [ 0.0000, -0.0000, -0.0000,  0.0000, -0.0000],
         [-0.0000,  0.0027,  0.0000,  0.0044,  0.0033],
         [-0.0000, -0.0000, -0.0014,  0.0000,  0.0000]],

        [[-0.0000, -0.0008, -0.0000,  0.0029, -0.0000],
         [ 0.0000, -0.0000,  0.0000, -0.0000, -0.0000],
         [-0.0000, -0.0000,  0.0000,  0.0000,  0.0000],
         [-0.0000, -0.0000, -0.0003,  0.0000, -0.0000]]], 4)


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
				[[ 0.0000, -0.0005, -0.0098, -0.0326,  0.0027,  0.0137,  0.0069,
           0.0236, -0.0124,  0.0117],
         [ 0.0474,  0.0063,  0.0095, -0.0084, -0.0257, -0.0006, -0.0298,
          -0.0546, -0.0107,  0.0059],
         [ 0.0025,  0.0049, -0.0065, -0.0302, -0.0071,  0.0353, -0.0101,
           0.0262,  0.0102,  0.0048],
         [-0.0237, -0.0184,  0.0054,  0.0157,  0.0318, -0.0368,  0.0162,
           0.0053,  0.0119, -0.0079]],

        [[-0.0474, -0.0007, -0.0085, -0.0107, -0.0015,  0.0108,  0.0151,
           0.0152, -0.0211,  0.0107],
         [ 0.0000,  0.0062,  0.0109,  0.0135, -0.0299, -0.0035, -0.0216,
          -0.0630, -0.0194,  0.0050],
         [-0.0449,  0.0047, -0.0051, -0.0082, -0.0113,  0.0324, -0.0019,
           0.0177,  0.0014,  0.0038],
         [-0.0712, -0.0185,  0.0067,  0.0377,  0.0275, -0.0397,  0.0244,
          -0.0032,  0.0031, -0.0088]]], 4)


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
	X_attr1 = deep_lift_shap(model, X[0:10], references=references[:10], 
		device='cpu', random_state=1)
	X_attr2 = deep_lift_shap(model, X[0:10], references=references[:10], 
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
		[[[-1.2537e-03,  8.9214e-04, -1.7216e-03,  1.5525e-03, -3.6719e-04],
          [-3.8154e-05,  3.1158e-03, -2.1447e-03, -2.7676e-04, -8.2222e-04],
          [ 2.5748e-04, -1.9488e-03,  2.5187e-05, -8.2059e-04,  2.1972e-03],
          [-1.3317e-03,  3.2600e-03, -1.1217e-03,  3.2719e-03, -2.3754e-03]],

         [[-4.1715e-04, -1.9015e-04, -2.3618e-04, -2.2531e-03, -4.3028e-05],
          [-2.2356e-04,  1.0974e-03, -2.0263e-03, -1.3769e-03,  2.0427e-03],
          [ 5.0492e-05,  1.0809e-04,  7.8811e-05,  2.5151e-03, -1.7192e-03],
          [-6.9432e-04, -3.1055e-04,  1.3494e-03, -2.7921e-03,  1.0637e-03]],

         [[ 5.5009e-04, -2.1254e-03,  8.3236e-04, -3.1150e-03,  3.4247e-04],
          [-8.2968e-04,  2.1356e-04, -2.9452e-03, -1.2408e-03,  1.2951e-03],
          [ 1.0232e-03, -1.2134e-03, -2.7766e-03,  3.2607e-03,  8.7921e-04],
          [-1.3670e-03,  1.4417e-04,  3.1089e-04, -2.6112e-03, -2.8219e-04]],

         [[-6.9273e-04,  8.5304e-04, -5.0310e-04,  6.1833e-04, -5.7676e-04],
          [-6.7235e-04,  2.2613e-03, -1.9021e-03, -9.0644e-04, -1.2571e-03],
          [ 1.8075e-04, -1.4490e-03, -3.9057e-04,  4.8862e-04,  3.4969e-03],
          [-1.0567e-03,  1.7016e-03, -3.3688e-04,  2.5392e-03, -2.1464e-03]]],


        [[[ 1.2693e-03, -1.0453e-03,  8.5213e-04,  1.8026e-03,  1.5205e-03],
          [ 2.0764e-03, -1.2293e-03, -4.5820e-04, -1.9499e-03, -2.0205e-04],
          [-2.1099e-04,  2.3054e-03,  2.5276e-04, -5.3881e-05,  6.6704e-04],
          [-1.3540e-03, -1.2551e-03, -7.0425e-04, -2.6836e-04,  3.1554e-03]],

         [[ 2.5604e-03, -2.7140e-03, -6.2686e-04, -1.4910e-03,  1.7278e-03],
          [-3.7230e-04, -3.5285e-03,  9.6826e-04,  1.6442e-03, -2.0139e-03],
          [-1.6076e-03,  3.1758e-03,  1.5919e-03,  3.7206e-03, -6.1896e-05],
          [-2.0704e-03, -4.3150e-03, -2.1066e-03, -2.9208e-03, -5.4428e-04]],

         [[ 9.2629e-04, -2.9686e-03,  6.0008e-04,  1.8710e-04,  1.9386e-04],
          [ 2.1494e-03, -2.2024e-03, -2.6531e-03, -4.2187e-04, -1.9708e-03],
          [-9.0424e-06,  3.1170e-04,  3.2218e-04,  2.8626e-03, -2.1286e-03],
          [-2.7281e-03, -1.5426e-03, -1.5072e-03, -2.7889e-03,  3.0662e-03]],

         [[ 1.7860e-03,  3.3834e-05,  2.8281e-03, -1.5901e-03,  1.5896e-03],
          [-2.2288e-03, -1.3378e-03,  2.1684e-03, -4.2975e-04, -4.2876e-03],
          [ 5.8548e-04,  2.8428e-03,  3.0198e-04,  1.1508e-03,  2.5912e-04],
          [ 9.2957e-05, -7.6795e-04, -1.2443e-04, -3.7598e-04, -9.3731e-04]]]], 
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
		[[[1., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
          [0., 1., 0., 0., 1., 0., 0., 0., 0., 0.],
          [0., 0., 1., 0., 0., 0., 0., 0., 0., 1.],
          [0., 0., 0., 1., 0., 1., 1., 1., 0., 0.]]],


        [[[0., 0., 0., 1., 1., 0., 0., 0., 1., 0.],
          [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
          [0., 1., 1., 0., 0., 1., 0., 1., 0., 1.]]],


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
		[[ 0.0000, -0.0000, -0.0000, -0.0320,  0.0000,  0.0000,  0.0000,
           0.0000, -0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0093, -0.0000, -0.0000, -0.0000, -0.0000,
          -0.0000, -0.0105,  0.0000],
         [ 0.0000,  0.0000, -0.0000, -0.0000, -0.0000,  0.0000, -0.0000,
           0.0000,  0.0000,  0.0000],
         [-0.0000, -0.0180,  0.0000,  0.0000,  0.0312, -0.0361,  0.0159,
           0.0052,  0.0000, -0.0077]],

        [[ 0.0000,  0.0000,  0.0090,  0.0000,  0.0016, -0.0000, -0.0161,
          -0.0000,  0.0000, -0.0114],
         [ 0.0000, -0.0000, -0.0000, -0.0000,  0.0000,  0.0000,  0.0000,
           0.0000,  0.0000, -0.0000],
         [ 0.0000, -0.0051,  0.0000,  0.0000,  0.0000, -0.0346,  0.0000,
          -0.0000, -0.0000, -0.0000],
         [ 0.0000,  0.0000, -0.0000, -0.0401, -0.0000,  0.0000, -0.0000,
           0.0034, -0.0033,  0.0000]]], 4)


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
	
	assert_raises(ValueError, deep_lift_shap, model, X, 
		references=references[:10], device='cpu')
	assert_raises(ValueError, deep_lift_shap, model, X, 
		references=references[:, :, :2], device='cpu')
	assert_raises(ValueError, deep_lift_shap, model, X, 
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
		[[ 0.0000, -0.0000, -0.0000, -0.0326,  0.0000,  0.0000,  0.0000,
           0.0000, -0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0095, -0.0000, -0.0000, -0.0000, -0.0000,
          -0.0000, -0.0107,  0.0000],
         [ 0.0000,  0.0000, -0.0000, -0.0000, -0.0000,  0.0000, -0.0000,
           0.0000,  0.0000,  0.0000],
         [-0.0000, -0.0184,  0.0000,  0.0000,  0.0318, -0.0368,  0.0162,
           0.0053,  0.0000, -0.0079]],

        [[-0.0000, -0.0000, -0.0085, -0.0000, -0.0015,  0.0000,  0.0151,
           0.0000, -0.0000,  0.0107],
         [ 0.0000,  0.0000,  0.0000,  0.0000, -0.0000, -0.0000, -0.0000,
          -0.0000, -0.0000,  0.0000],
         [-0.0000,  0.0047, -0.0000, -0.0000, -0.0000,  0.0324, -0.0000,
           0.0000,  0.0000,  0.0000],
         [-0.0000, -0.0000,  0.0000,  0.0377,  0.0000, -0.0000,  0.0000,
          -0.0032,  0.0031, -0.0000]]], 4)


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
				[[ 0.0000, -0.0000, -0.0000, -0.0326,  0.0000,  0.0000,  0.0000,
           0.0000, -0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0095, -0.0000, -0.0000, -0.0000, -0.0000,
          -0.0000, -0.0107,  0.0000],
         [ 0.0000,  0.0000, -0.0000, -0.0000, -0.0000,  0.0000, -0.0000,
           0.0000,  0.0000,  0.0000],
         [-0.0000, -0.0184,  0.0000,  0.0000,  0.0318, -0.0368,  0.0162,
           0.0053,  0.0000, -0.0079]],

        [[-0.0000, -0.0000, -0.0085, -0.0000, -0.0015,  0.0000,  0.0151,
           0.0000, -0.0000,  0.0107],
         [ 0.0000,  0.0000,  0.0000,  0.0000, -0.0000, -0.0000, -0.0000,
          -0.0000, -0.0000,  0.0000],
         [-0.0000,  0.0047, -0.0000, -0.0000, -0.0000,  0.0324, -0.0000,
           0.0000,  0.0000,  0.0000],
         [-0.0000, -0.0000,  0.0000,  0.0377,  0.0000, -0.0000,  0.0000,
          -0.0032,  0.0031, -0.0000]]], 4)


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
				[[ 0.0000, -0.0000, -0.0000, -0.6734, -0.0000, -0.0000, -0.0000,
          -0.0000, -0.0000, -0.0000],
         [ 0.0000, -0.0000, -1.0909, -0.0000, -0.0000, -0.0000, -0.0000,
          -0.0000, -0.8449, -0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
           0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.9606,  0.0000,  0.0000,  0.5665,  0.6879,  0.3987,
           0.7660,  0.0000,  0.4161]],

        [[-0.0000, -0.0000, -0.7982, -0.0000, -0.7982, -0.0000, -0.7027,
          -0.0000, -0.0000, -0.9312],
         [ 0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000,
          -0.0000, -0.0000, -0.0000],
         [ 0.0000,  0.9682,  0.0000,  0.0000,  0.0000,  0.6486,  0.0000,
           0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  0.6645,  0.0000,  0.0000,  0.0000,
           0.2482,  0.0919,  0.0000]]], 4)


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

		X_attr0 = deep_lift_shap(model, X[:4], references=references[:4], 
			n_shuffles=3, batch_size=1, device='cpu', random_state=0)
		X_attr1 = _captum_deep_lift_shap(model, X[:4], references=references[:4],
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
