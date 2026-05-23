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
from tangermeme.deep_lift_shap import _nonlinear

from .toy_models import SumModel
from .toy_models import FlattenDense
from .toy_models import Conv
from .toy_models import Scatter
from .toy_models import ConvDense
from .toy_models import ConvPoolDense
from .toy_models import SmallDeepSEA
from .toy_models import ResidualConv
from .toy_models import Conv2DExpand
from .toy_models import CustomLinear
from .toy_models import CustomSqrt
from .toy_models import CustomSqrtModule
from .toy_models import DilatedConv
from .toy_models import MultiActivation
from .toy_models import DropoutConv
from .toy_models import MultiInputMultiOutput

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


def test_deep_lift_shap(X, device):
	torch.manual_seed(0)
	model = SmallDeepSEA()

	X_attr1 = deep_lift_shap(model, X[:4], device=device, n_shuffles=3, 
		random_state=0, batch_size=1)
	X_attr2 = deep_lift_shap(model, X[:4], device=device, n_shuffles=3, 
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


def test_deep_lift_shap_convergence(X, device):
	# fp32 attribution residuals on CUDA are a few orders of magnitude larger
	# than on CPU, so the convergence threshold is loosened for the cuda pass.
	threshold = 1e-7 if device == "cpu" else 1e-4
	torch.manual_seed(0)
	model = SmallDeepSEA()

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		deep_lift_shap(model, X[:4], device=device, n_shuffles=3, random_state=0,
			warning_threshold=threshold)

		assert_raises(RuntimeWarning, deep_lift_shap, model, X[:4],
			device=device, n_shuffles=3, random_state=0, warning_threshold=1e-10)


def test_deep_lift_shap_hypothetical(X, device):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	X_attr = deep_lift_shap(model, X, hypothetical=True, device=device, 
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


def test_deep_lift_shap_independence(X, device):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	X_attr = deep_lift_shap(model, X, device=device, random_state=0)
	X_attr0 = deep_lift_shap(model, X[0:1], device=device, random_state=0)
	X_attr1 = deep_lift_shap(model, X[5:6], device=device, random_state=0)
	X_attr2 = deep_lift_shap(model, X[8:10], device=device, random_state=0)
	X_attr3 = deep_lift_shap(model, X[0:10], device=device, random_state=0)

	assert_array_almost_equal(X_attr[0:1], X_attr0)
	assert_array_almost_equal(X_attr[5:6], X_attr1)
	assert_array_almost_equal(X_attr[8:10], X_attr2)
	assert_array_almost_equal(X_attr[0:10], X_attr3)

	assert_raises(AssertionError, assert_array_almost_equal, X_attr0, X_attr1)
	assert_raises(AssertionError, assert_array_almost_equal, X_attr2, 
		X_attr3[:2])


def test_deep_lift_shap_random_state(X, device):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	X_attr0 = deep_lift_shap(model, X, device=device, random_state=0)
	X_attr1 = deep_lift_shap(model, X[0:10], device=device, random_state=1)
	X_attr2 = deep_lift_shap(model, X[0:10], device=device, random_state=2)

	assert_raises(AssertionError, assert_array_almost_equal, X_attr0, X_attr1)
	assert_raises(AssertionError, assert_array_almost_equal, X_attr0, X_attr2)


def test_deep_lift_shap_reference_tensor(X, device):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	references = shuffle(X, n=20, random_state=0)

	X_attr0 = deep_lift_shap(model, X, references=references, device=device, 
		random_state=0)
	X_attr1 = deep_lift_shap(model, X[0:10], references=references[:10], 
		device=device, random_state=1)
	X_attr2 = deep_lift_shap(model, X[0:10], references=references[:10], 
		device=device, random_state=2)

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


def test_deep_lift_shap_batch_size(X, device):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	X_attr0 = deep_lift_shap(model, X, device=device, random_state=0)
	X_attr1 = deep_lift_shap(model, X, batch_size=1, device=device, 
		random_state=0)
	X_attr2 = deep_lift_shap(model, X, batch_size=100000, device=device, 
		random_state=0)
	X_attr3 = deep_lift_shap(model, X, batch_size=20, device=device, 
		random_state=0)

	assert_array_almost_equal(X_attr0, X_attr1)
	assert_array_almost_equal(X_attr0, X_attr2)
	assert_array_almost_equal(X_attr0, X_attr3)


def test_deep_lift_shap_n_shuffles(X, device):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	X_attr0 = deep_lift_shap(model, X, n_shuffles=1, device=device, 
		random_state=0)
	X_attr1 = deep_lift_shap(model, X, n_shuffles=1, batch_size=1, device=device, 
		random_state=0)
	X_attr2 = deep_lift_shap(model, X, n_shuffles=30, batch_size=100000, 
		device=device, random_state=2)
	X_attr3 = deep_lift_shap(model, X, n_shuffles=30, batch_size=1, 
		device=device, random_state=2)

	assert_array_almost_equal(X_attr0, X_attr1)
	assert_array_almost_equal(X_attr2, X_attr3)
	assert_raises(AssertionError, assert_array_almost_equal, X_attr0, X_attr3)


def test_deep_lift_shap_input_type(X, device):
	torch.manual_seed(0)
	model = SmallDeepSEA(n_outputs=1)

	X_attr0 = deep_lift_shap(model, X, device=device, 
		n_shuffles=5, random_state=0)
	X_attr1 = deep_lift_shap(model, X.type(torch.int8), device=device, 
		n_shuffles=5, random_state=0)
	X_attr2 = deep_lift_shap(model, X.type(torch.int16), device=device,
		n_shuffles=5, random_state=0)
	X_attr3 = deep_lift_shap(model, X.type(torch.float16), device=device,
		n_shuffles=5, random_state=0)
	X_attr4 = deep_lift_shap(model, X.type(torch.bfloat16), device=device,
		n_shuffles=5, random_state=0)
	X_attr5 = deep_lift_shap(model, X.type(torch.int32), device=device,
		n_shuffles=5, random_state=0)

	assert_array_almost_equal(X_attr0, X_attr1)
	assert_array_almost_equal(X_attr0, X_attr2)
	assert_array_almost_equal(X_attr0, X_attr3)
	assert_array_almost_equal(X_attr0, X_attr4)
	assert_array_almost_equal(X_attr0, X_attr5)
	

def test_deep_lift_shap_shuffle_ordering(X, device):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	X = X[:1]

	references = dinucleotide_shuffle(X, n=1, random_state=0)

	X_attr0 = deep_lift_shap(model, X, n_shuffles=1, device=device, random_state=0)
	X_attr1 = deep_lift_shap(model, X, device=device, references=references)

	assert_array_almost_equal(X_attr0, X_attr1)


def test_deep_lift_shap_raw_output(X, device):
	torch.manual_seed(0)
	model = SmallDeepSEA()

	X_attr0, refs = deep_lift_shap(model, X, device=device, raw_outputs=True, 
		random_state=0, return_references=True)
	X_attr1 = deep_lift_shap(model, X, device=device, random_state=0)

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


def test_deep_lift_shap_return_references(X, device):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	attr, refs = deep_lift_shap(model, X, n_shuffles=1, return_references=True,
		device=device, random_state=0)

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
		device=device, random_state=0)

	assert_array_almost_equal(refs, refs2[:, 0:1])


def test_deep_lift_shap_args(X, device):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)
	alpha = torch.randn(16, 1)
	beta = torch.randn(16, 1)

	X_attr0 = deep_lift_shap(model, X, device=device, random_state=0)
	X_attr1 = deep_lift_shap(model, X, args=(alpha,), device=device, 
		random_state=0)
	X_attr2 = deep_lift_shap(model, X, args=(alpha, beta), device=device, 
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


def test_deep_lift_shap_raises(X, references, device):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)
	alpha = torch.randn(16, 1)
	beta = torch.randn(16, 1)

	assert_raises(ValueError, deep_lift_shap, model, X[0], device=device)
	assert_raises(ValueError, deep_lift_shap, model, X.unsqueeze(1), 
		device=device)
	assert_raises(RuntimeError, deep_lift_shap, model, X, n_shuffles=0, 
		device=device)
	assert_raises(ValueError, deep_lift_shap, model, X[0], device=device)

	assert_raises(IndexError, deep_lift_shap, model, X, args=(alpha[:10],),
		device=device)
	assert_raises(IndexError, deep_lift_shap, model, X, args=(alpha, beta[:3]),
		device=device)
	assert_raises(IndexError, deep_lift_shap, model, X, args=(alpha[:5], 
		beta[:3]), device=device)
	assert_raises(IndexError, deep_lift_shap, model, X, args=(alpha, beta[:3]),
		device=device)
	
	assert_raises(ValueError, deep_lift_shap, model, X, 
		references=references[:10], device=device)
	assert_raises(ValueError, deep_lift_shap, model, X, 
		references=references[:, :, :2], device=device)
	assert_raises(ValueError, deep_lift_shap, model, X, 
		references=references[:, :, :, :10], device=device)


### Test a bunch of different models with different configurations/operations


def test_deep_lift_shap_flattendense(X, device):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		X_attr = deep_lift_shap(model, X, device=device, random_state=0, 
			warning_threshold=1e-5)

		assert_raises(RuntimeWarning, deep_lift_shap, model, X, device=device, 
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


def test_deep_lift_shap_convdense_dense_wrapper(X, device):
	torch.manual_seed(0)
	model = LambdaWrapper(ConvDense(n_outputs=1), lambda model, X: model(X)[1])

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		X_attr = deep_lift_shap(model, X, device=device, random_state=0, 
			warning_threshold=1e-5)

		assert_raises(RuntimeWarning, deep_lift_shap, model, X, device=device, 
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


def test_deep_lift_shap_convdense_conv_wrapper(X, device):
	torch.manual_seed(0)
	model = LambdaWrapper(ConvDense(n_outputs=1), 
		lambda model, X: model(X)[0].sum(dim=(-1, -2)).unsqueeze(-1))

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		X_attr = deep_lift_shap(model, X, device=device, random_state=0, 
			warning_threshold=1e-4)

		assert_raises(RuntimeWarning, deep_lift_shap, model, X, device=device, 
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


def test_deep_lift_shap_linear(X, device):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Flatten(),
		torch.nn.Linear(400, 5),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device=device, random_state=0,
			warning_threshold=1e-5)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000,  0.0000,  0.0000, -0.0562,  0.0000, -0.0000,  0.0000,  0.0000,  0.0000, -0.0000],
		 [-0.0000, -0.0000,  0.0970,  0.0000,  0.0000, -0.0000, -0.0000, -0.0000, -0.0124, -0.0000],
		 [-0.0000, -0.0000, -0.0000,  0.0000, -0.0000,  0.0000,  0.0000, -0.0000,  0.0000, -0.0000],
		 [-0.0000, -0.0495, -0.0000, -0.0000, -0.0207, -0.0280, -0.0184,  0.0326,  0.0000,  0.0357]],

		[[-0.0000,  0.0000,  0.0177, -0.0000,  0.0027,  0.0000,  0.0168,  0.0000, -0.0000, -0.0092],
		 [ 0.0000, -0.0000,  0.0000,  0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0136, -0.0000,  0.0000, -0.0000,  0.0353, -0.0000, -0.0000,  0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0000,  0.0122,  0.0000, -0.0000, -0.0000,  0.0083, -0.0053,  0.0000]]], 4)


def test_deep_lift_shap_linear_bias(X, device):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Flatten(),
		torch.nn.Linear(400, 5, bias=False),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device=device, random_state=0,
			warning_threshold=1e-5)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000,  0.0000,  0.0000, -0.0505,  0.0000, -0.0000,  0.0000,  0.0000,  0.0000, -0.0000],
		 [-0.0000, -0.0000,  0.0934,  0.0000,  0.0000, -0.0000, -0.0000, -0.0000, -0.0117, -0.0000],
		 [-0.0000, -0.0000, -0.0000,  0.0000, -0.0000,  0.0000,  0.0000, -0.0000,  0.0000, -0.0000],
		 [-0.0000, -0.0477, -0.0000, -0.0000, -0.0212, -0.0250, -0.0182,  0.0317,  0.0000,  0.0349]],

		[[-0.0000,  0.0000,  0.0136, -0.0000,  0.0016,  0.0000,  0.0150,  0.0000, -0.0000, -0.0073],
		 [ 0.0000, -0.0000,  0.0000,  0.0000,  0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0130, -0.0000,  0.0000, -0.0000,  0.0285, -0.0000, -0.0000,  0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0000,  0.0104,  0.0000, -0.0000, -0.0000,  0.0070, -0.0041,  0.0000]]], 4)


def test_deep_lift_shap_conv(X, device):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,)),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device=device, random_state=0,
			warning_threshold=1e-5)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000,  0.0000,  0.0000,  0.9025, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.1260, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0379, -0.0000],
		 [-0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.6341, -0.0000, -0.0000, -0.0966, -0.1356, -0.1817, -0.1439, -0.0000, -0.3838]],

		[[ 0.0000,  0.0000,  0.4590,  0.0000,  0.6733, -0.0000,  0.5197, -0.0000, -0.0000,  0.3142],
		 [ 0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.3457, -0.0000, -0.0000, -0.0000,  0.0731, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0000, -0.2266, -0.0000, -0.0000, -0.0000, -0.3457, -0.1228, -0.0000]]], 4)


def test_deep_lift_shap_conv_dilated(X, device):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,), dilation=5),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device=device, random_state=0,
			warning_threshold=1e-5)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000,  0.0000,  0.0000,  0.5778,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
		 [-0.0000, -0.0000,  0.0112, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.1658, -0.0000],
		 [-0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.1592, -0.0000, -0.0000,  0.0564, -0.2365, -0.1947, -0.2984, -0.0000,  0.0283]],

		[[ 0.0000,  0.0000,  0.3452,  0.0000,  0.3758,  0.0000,  0.7259,  0.0000,  0.0000,  0.6784],
		 [ 0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.2153, -0.0000, -0.0000, -0.0000, -0.0166, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0000, -0.1370, -0.0000, -0.0000, -0.0000, -0.0588, -0.0551, -0.0000]]], 4)


def test_deep_lift_shap_conv_stride(X, device):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,), stride=5),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device=device, random_state=0,
			warning_threshold=1e-5)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000,  0.0000,  0.0000,  0.0998, -0.0000,  0.0000,  0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.2351, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0290, -0.0000],
		 [-0.0000, -0.0000, -0.0000,  0.0000,  0.0000, -0.0000, -0.0000, -0.0000,  0.0000, -0.0000],
		 [-0.0000, -0.2509, -0.0000,  0.0000, -0.0655, -0.1500, -0.0718, -0.0009, -0.0000, -0.0680]],

		[[ 0.0000,  0.0000,  0.0049, -0.0000, -0.0378,  0.0000,  0.1322, -0.0000, -0.0000, -0.1232],
		 [ 0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0254, -0.0000,  0.0000, -0.0000, -0.0873,  0.0000, -0.0000,  0.0000, -0.0000],
		 [-0.0000, -0.0000,  0.0000, -0.1044,  0.0000, -0.0000, -0.0000,  0.0755, -0.0991, -0.0000]]], 4)


def test_deep_lift_shap_conv_bias(X, device):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,), bias=False),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device=device, random_state=0,
			warning_threshold=1e-4)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000,  0.0000,  0.0000,  0.8116, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.1397, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0176, -0.0000],
		 [-0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.5924, -0.0000, -0.0000, -0.1294, -0.2071, -0.2220, -0.2076, -0.0000, -0.4544]],

		[[ 0.0000,  0.0000,  0.4262,  0.0000,  0.5746, -0.0000,  0.3762, -0.0000, -0.0000,  0.3317],
		 [ 0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.3874, -0.0000, -0.0000, -0.0000,  0.2103, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0000, -0.3004, -0.0000, -0.0000, -0.0000, -0.4046, -0.2499, -0.0000]]], 4)


def test_deep_lift_shap_conv_padding(X, device):
	# fp32 attribution residuals on CUDA are a few orders of magnitude larger
	# than on CPU, so the convergence threshold is loosened for the cuda pass.
	threshold = 1e-4 if device == "cpu" else 1e-2
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,), padding=5),
		torch.nn.ReLU(),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device=device, random_state=0,
			warning_threshold=threshold)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000, -0.0000,  0.0000,  0.7698, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.3090, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0379, -0.0000],
		 [-0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.4021, -0.0000, -0.0000, -0.0966, -0.1356, -0.1817, -0.1439, -0.0000, -0.3838]],

		[[-0.0000,  0.0000,  0.3432,  0.0000,  0.6733, -0.0000,  0.5197, -0.0000, -0.0000,  0.3142],
		 [ 0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.2294, -0.0000, -0.0000, -0.0000,  0.0731, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0000, -0.2165, -0.0000, -0.0000, -0.0000, -0.3457, -0.1228, -0.0000]]], 4)


def test_deep_lift_shap_conv_padding_same(X, device):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,), padding='same'),
		torch.nn.Flatten(),
		torch.nn.ReLU(),
		torch.nn.Linear(800, 1)
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device=device, random_state=0,
			warning_threshold=1e-5)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000, -0.0000, -0.0000,  0.0002, -0.0000, -0.0000,  0.0000, -0.0000, -0.0000,  0.0000],
		 [-0.0000, -0.0000,  0.0012, -0.0000,  0.0000,  0.0000,  0.0000, -0.0000, -0.0006, -0.0000],
		 [ 0.0000, -0.0000,  0.0000, -0.0000,  0.0000,  0.0000,  0.0000, -0.0000,  0.0000,  0.0000],
		 [ 0.0000,  0.0130,  0.0000,  0.0000, -0.0099, -0.0026, -0.0080,  0.0025,  0.0000, -0.0021]],

		[[-0.0000,  0.0000, -0.0049, -0.0000, -0.0027,  0.0000,  0.0167,  0.0000, -0.0000,  0.0043],
		 [ 0.0000, -0.0000, -0.0000,  0.0000,  0.0000,  0.0000, -0.0000,  0.0000, -0.0000, -0.0000],
		 [ 0.0000, -0.0031, -0.0000,  0.0000,  0.0000, -0.0019, -0.0000,  0.0000,  0.0000,  0.0000],
		 [ 0.0000,  0.0000,  0.0000, -0.0123, -0.0000, -0.0000, -0.0000, -0.0069, -0.0056, -0.0000]]], 4)


def test_deep_lift_shap_max_pool(X, device):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.MaxPool1d(4),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device=device, random_state=0,
			warning_threshold=1e-5)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000, -0.0000, -0.0000,  0.0000,  0.0000, -0.0000, -0.0000, -0.0000,  0.0000, -0.0000],
		 [-0.0000, -0.0000,  0.2000,  0.0000,  0.0000, -0.0000, -0.0000, -0.0000,  0.2000, -0.0000],
		 [ 0.0000, -0.0000, -0.0000,  0.0000,  0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000,  0.0500, -0.0000,  0.0000, -0.5500, -0.7000, -0.4500, -0.1000, -0.0000,  0.1000]],

		[[-0.0000, -0.0000,  0.3000, -0.0000,  0.3000, -0.0000, -0.2000, -0.0000, -0.0000,  0.0000],
		 [ 0.0000, -0.0000, -0.0000, -0.0000,  0.0000, -0.0000,  0.0000, -0.0000,  0.0000, -0.0000],
		 [-0.0000,  0.2000, -0.0000, -0.0000, -0.0000, -0.0500, -0.0000, -0.0000,  0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0000,  0.5500, -0.0000, -0.0000, -0.0000,  0.1500, -0.3500, -0.0000]]], 4)


def test_deep_lift_shap_conv_relu_pool(X, device):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,)),
		torch.nn.ReLU(),
		torch.nn.MaxPool1d(4),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device=device, random_state=0,
			warning_threshold=1e-5)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000,  0.0000,  0.0000,  0.1021, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0643, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.1375, -0.0000],
		 [-0.0000, -0.0000, -0.0000,  0.0000,  0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0408, -0.0000, -0.0000, -0.1820, -0.1912, -0.1507, -0.2448, -0.0000, -0.1143]],

		[[ 0.0000,  0.0000,  0.0591,  0.0000,  0.3041, -0.0000,  0.0826, -0.0000, -0.0000,  0.0267],
		 [ 0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0806, -0.0000, -0.0000, -0.0000,  0.0403, -0.0000,  0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0000,  0.0013, -0.0000, -0.0000, -0.0000, -0.1534, -0.2105, -0.0000]]], 4)


def test_deep_lift_shap_conv_tanh_pool(X, device):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,)),
		torch.nn.Tanh(),
		torch.nn.MaxPool1d(4),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device=device, random_state=0,
			warning_threshold=1e-5)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000, -0.0000, -0.0000,  0.1069, -0.0000, -0.0000,  0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0354, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.2147, -0.0000],
		 [-0.0000, -0.0000, -0.0000,  0.0000,  0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0738, -0.0000, -0.0000, -0.3306, -0.4012, -0.2828, -0.4270, -0.0000, -0.1415]],

		[[ 0.0000,  0.0000,  0.0031, -0.0000,  0.2630, -0.0000,  0.0245, -0.0000, -0.0000, -0.0002],
		 [ 0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0908, -0.0000, -0.0000, -0.0000,  0.1219, -0.0000,  0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0000, -0.0304, -0.0000, -0.0000, -0.0000, -0.1598, -0.2506, -0.0000]]], 4)


def test_deep_lift_shap_conv_elu_pool(X, device):
	torch.manual_seed(0)

	model = torch.nn.Sequential(
		torch.nn.Conv1d(4, 8, (5,)),
		torch.nn.ELU(),
		torch.nn.MaxPool1d(4),
		TorchSum()
	)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)
		X_attr = deep_lift_shap(model, X, device=device, random_state=0,
			warning_threshold=1e-5)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000, -0.0000,  0.0000,  0.1168, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0512, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.2202, -0.0000],
		 [-0.0000, -0.0000, -0.0000,  0.0000,  0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0742, -0.0000, -0.0000, -0.3211, -0.3888, -0.2736, -0.4213, -0.0000, -0.1462]],

		[[ 0.0000,  0.0000,  0.0247, -0.0000,  0.2891, -0.0000,  0.0166, -0.0000, -0.0000, -0.0036],
		 [ 0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0936, -0.0000, -0.0000, -0.0000,  0.1040, -0.0000,  0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0000, -0.0296, -0.0000, -0.0000, -0.0000, -0.1684, -0.2534, -0.0000]]], 4)


def test_deep_lift_shap_conv_relu_pool_relu(X, device):
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
		X_attr = deep_lift_shap(model, X, device=device, random_state=0,
			warning_threshold=1e-5)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000,  0.0000,  0.0000,  0.1021, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0643, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.1375, -0.0000],
		 [-0.0000, -0.0000, -0.0000,  0.0000,  0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0408, -0.0000, -0.0000, -0.1820, -0.1912, -0.1507, -0.2448, -0.0000, -0.1143]],

		[[ 0.0000,  0.0000,  0.0591,  0.0000,  0.3041, -0.0000,  0.0826, -0.0000, -0.0000,  0.0267],
		 [ 0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0806, -0.0000, -0.0000, -0.0000,  0.0403, -0.0000,  0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0000,  0.0013, -0.0000, -0.0000, -0.0000, -0.1534, -0.2105, -0.0000]]], 4)


def test_deep_lift_shap_relu_conv_relu_pool_relu(X, device):
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
		X_attr = deep_lift_shap(model, X, device=device, random_state=0,
			warning_threshold=1e-5)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000, -0.0000,  0.0000,  0.1021, -0.0000,  0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0643, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.1375, -0.0000],
		 [-0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0408, -0.0000, -0.0000, -0.1820, -0.1912, -0.1507, -0.2448, -0.0000, -0.1143]],

		[[-0.0000,  0.0000,  0.0591, -0.0000,  0.3041, -0.0000,  0.0826, -0.0000, -0.0000,  0.0267],
		 [ 0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0806, -0.0000, -0.0000, -0.0000,  0.0403,  0.0000, -0.0000,  0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0000,  0.0013, -0.0000, -0.0000, -0.0000, -0.1534, -0.2105, -0.0000]]], 4)


def test_deep_lift_shap_relu_conv_pool_relu(X, device):
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
		X_attr = deep_lift_shap(model, X, device=device, random_state=0,
			warning_threshold=1e-5)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000, -0.0000,  0.0000,  0.1021, -0.0000,  0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0643, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.1375, -0.0000],
		 [-0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0408, -0.0000, -0.0000, -0.1820, -0.1912, -0.1507, -0.2448, -0.0000, -0.1143]],

		[[-0.0000,  0.0000,  0.0591, -0.0000,  0.3041, -0.0000,  0.0826, -0.0000, -0.0000,  0.0267],
		 [ 0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0806, -0.0000, -0.0000, -0.0000,  0.0403,  0.0000, -0.0000,  0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0000,  0.0013, -0.0000, -0.0000, -0.0000, -0.1534, -0.2105, -0.0000]]], 4)


def test_deep_lift_shap_relu_conv_pool_relu_relu(X, device):
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
		X_attr = deep_lift_shap(model, X, device=device, random_state=0,
			warning_threshold=1e-5)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000, -0.0000,  0.0000,  0.1021, -0.0000,  0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0643, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.1375, -0.0000],
		 [-0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0408, -0.0000, -0.0000, -0.1820, -0.1912, -0.1507, -0.2448, -0.0000, -0.1143]],

		[[-0.0000,  0.0000,  0.0591, -0.0000,  0.3041, -0.0000,  0.0826, -0.0000, -0.0000,  0.0267],
		 [ 0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0806, -0.0000, -0.0000, -0.0000,  0.0403,  0.0000, -0.0000,  0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0000,  0.0013, -0.0000, -0.0000, -0.0000, -0.1534, -0.2105, -0.0000]]], 4)


def test_deep_lift_shap_conv_relu_tanh_pool(X, device):
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
		X_attr = deep_lift_shap(model, X, device=device, random_state=0,
			warning_threshold=1e-5)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000,  0.0000,  0.0000,  0.0922, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0539, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.1299, -0.0000],
		 [-0.0000, -0.0000, -0.0000,  0.0000,  0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0344, -0.0000, -0.0000, -0.1765, -0.1837, -0.1488, -0.2312, -0.0000, -0.1048]],

		[[ 0.0000,  0.0000,  0.0408,  0.0000,  0.2812, -0.0000,  0.1009, -0.0000, -0.0000,  0.0331],
		 [ 0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0769, -0.0000, -0.0000, -0.0000,  0.0481, -0.0000,  0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0000, -0.0000,  0.0038, -0.0000, -0.0000, -0.0000, -0.1424, -0.2023, -0.0000]]], 4)


def test_deep_lift_shap_conv_relu_pool_linear(X, device):
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
		X_attr = deep_lift_shap(model, X, device=device, random_state=0,
			warning_threshold=1e-5)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000,  0.0000,  0.0000,  0.0011, -0.0000, -0.0000,  0.0000,  0.0000,  0.0000, -0.0000],
		 [-0.0000,  0.0000, -0.0010, -0.0000, -0.0000,  0.0000, -0.0000, -0.0000,  0.0024, -0.0000],
		 [-0.0000, -0.0000, -0.0000,  0.0000,  0.0000,  0.0000, -0.0000, -0.0000,  0.0000,  0.0000],
		 [-0.0000,  0.0009, -0.0000,  0.0000, -0.0076, -0.0049, -0.0026, -0.0039, -0.0000, -0.0029]],

		[[-0.0000, -0.0000,  0.0086,  0.0000,  0.0153,  0.0000,  0.0036,  0.0000, -0.0000, -0.0041],
		 [ 0.0000, -0.0000, -0.0000,  0.0000, -0.0000,  0.0000, -0.0000, -0.0000, -0.0000,  0.0000],
		 [ 0.0000,  0.0006, -0.0000, -0.0000, -0.0000, -0.0054, -0.0000,  0.0000,  0.0000,  0.0000],
		 [ 0.0000,  0.0000,  0.0000,  0.0021, -0.0000, -0.0000, -0.0000, -0.0028, -0.0025,  0.0000]]], 4)


def test_deep_lift_shap_conv_relu_pool_linear_linear(X, device):
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
		X_attr = deep_lift_shap(model, X, device=device, random_state=0,
			warning_threshold=1e-5)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000, -0.0000, -0.0000, -0.0013,  0.0000, -0.0000, -0.0000,  0.0000,  0.0000,  0.0000],
		 [ 0.0000, -0.0000, -0.0010,  0.0000,  0.0000, -0.0000, -0.0000, -0.0000, -0.0009, -0.0000],
		 [ 0.0000,  0.0000,  0.0000, -0.0000, -0.0000,  0.0000, -0.0000, -0.0000,  0.0000, -0.0000],
		 [ 0.0000,  0.0002,  0.0000, -0.0000, -0.0002, -0.0020, -0.0013, -0.0047, -0.0000, -0.0009]],

		[[ 0.0000,  0.0000, -0.0074, -0.0000, -0.0044, -0.0000,  0.0029,  0.0000,  0.0000,  0.0001],
		 [ 0.0000, -0.0000, -0.0000, -0.0000,  0.0000,  0.0000,  0.0000, -0.0000,  0.0000,  0.0000],
		 [-0.0000, -0.0011,  0.0000,  0.0000,  0.0000, -0.0014, -0.0000, -0.0000, -0.0000,  0.0000],
		 [-0.0000, -0.0000,  0.0000, -0.0016,  0.0000, -0.0000, -0.0000, -0.0004,  0.0009, -0.0000]]], 4)


def test_deep_lift_shap_conv_relu_pool_linear_relu_linear(X, device):
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
		X_attr = deep_lift_shap(model, X, device=device, random_state=0,
			warning_threshold=1e-5)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[ 0.0000,  0.0000, -0.0000, -0.0001, -0.0000, -0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
		 [-0.0000,  0.0000, -0.0010, -0.0000, -0.0000,  0.0000, -0.0000, -0.0000,  0.0003,  0.0000],
		 [-0.0000, -0.0000, -0.0000, -0.0000,  0.0000,  0.0000, -0.0000, -0.0000,  0.0000,  0.0000],
		 [ 0.0000,  0.0004, -0.0000,  0.0000, -0.0017, -0.0009, -0.0004, -0.0014, -0.0000, -0.0012]],

		[[-0.0000, -0.0000, -0.0031,  0.0000, -0.0002,  0.0000,  0.0016,  0.0000,  0.0000, -0.0006],
		 [ 0.0000, -0.0000, -0.0000, -0.0000, -0.0000,  0.0000, -0.0000, -0.0000,  0.0000,  0.0000],
		 [ 0.0000, -0.0004, -0.0000,  0.0000, -0.0000,  0.0001, -0.0000, -0.0000, -0.0000,  0.0000],
		 [-0.0000,  0.0000,  0.0000, -0.0001, -0.0000, -0.0000, -0.0000, -0.0000,  0.0001, -0.0000]]], 4)


###


def _captum_attribute_comparison(model, X, references, device):
	torch.manual_seed(0)

	X_attr0 = deep_lift_shap(model, X, references=references, device=device,
		random_state=0)
	X_attr1 = _captum_deep_lift_shap(model, X, references=references,
		device=device, random_state=0)

	assert X_attr0.shape == X_attr1.shape
	assert X_attr0.dtype == X_attr1.dtype
	assert_array_almost_equal(X_attr0, X_attr1, 5)


def test_captum_deep_lift_shap_summodel(X, references, device):
	model = LambdaWrapper(SumModel(), lambda model, X: model(X)[:, 0:1])
	_captum_attribute_comparison(model, X, references, device)


def test_captum_deep_lift_shap_flattendense(X, references, device):
	_captum_attribute_comparison(FlattenDense(n_outputs=1), X, references, device)


def test_captum_deep_lift_shap_scatter(X, references, device):
	model = LambdaWrapper(Scatter(), lambda model, X: model(X)[:, 0, 0:1])
	_captum_attribute_comparison(model, X, references, device)


def test_captum_deep_lift_shap_conv(X, references, device):
	model = LambdaWrapper(Conv(), lambda model, X: model(X).sum(
		dim=(-1, -2)).unsqueeze(-1))
	_captum_attribute_comparison(model, X, references, device)


#def test_captum_deep_lift_shap_convpooldense(X, references):
#	_captum_attribute_comparison(ConvPoolDense(), X, references)


def test_captum_deep_lift_shap_batch_size(X, references, device):
	_SumModel = LambdaWrapper(SumModel(), lambda model, X: model(X)[:, 0:1])
	_Scatter = LambdaWrapper(Scatter(), lambda model, X: model(X)[:, 0, 0:1])
	_Conv = LambdaWrapper(Conv(), lambda model, X: model(X).sum(dim=1)[:, 0:1])

	for model in _SumModel, FlattenDense(n_outputs=1), _Scatter, _Conv:
		torch.manual_seed(0)

		X_attr0 = deep_lift_shap(model, X, references=references, 
			batch_size=1, device=device, random_state=0)
		X_attr1 = _captum_deep_lift_shap(model, X, references=references, 
			batch_size=1, device=device, random_state=0)

		assert X_attr0.shape == X_attr1.shape
		assert X_attr0.dtype == X_attr1.dtype
		assert_array_almost_equal(X_attr0, X_attr1)


def test_captum_deep_lift_shap_n_shuffles(X, references, device):
	_SumModel = LambdaWrapper(SumModel(), lambda model, X: model(X)[:, 0:1])
	_Scatter = LambdaWrapper(Scatter(), lambda model, X: model(X)[:, 0, 0:1])
	_Conv = LambdaWrapper(Conv(), lambda model, X: model(X).sum(dim=1)[:, 0:1])

	for model in _SumModel, FlattenDense(n_outputs=1), _Scatter, _Conv:
		torch.manual_seed(0)

		X_attr0 = deep_lift_shap(model, X[:4], references=references[:4], 
			n_shuffles=3, batch_size=1, device=device, random_state=0)
		X_attr1 = _captum_deep_lift_shap(model, X[:4], references=references[:4],
			n_shuffles=3, batch_size=1, device=device, random_state=0)

		assert X_attr0.shape == X_attr1.shape
		assert X_attr0.dtype == X_attr1.dtype
		assert_array_almost_equal(X_attr0, X_attr1)


def test_captum_deep_lift_shap_args(X, references, device):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)
	alpha = torch.randn(16, 1)
	beta = torch.randn(16, 1)

	X_attr0 = deep_lift_shap(model, X, args=(alpha,), references=references, 
		device=device, random_state=0)
	X_attr1 = _captum_deep_lift_shap(model, X, args=(alpha,), 
		references=references, device=device, random_state=0)

	assert X_attr0.shape == X_attr1.shape
	assert X_attr0.dtype == X_attr1.dtype
	assert_array_almost_equal(X_attr0, X_attr1)


	X_attr0 = deep_lift_shap(model, X, args=(alpha, beta), 
		references=references, device=device, random_state=0)
	X_attr1 = _captum_deep_lift_shap(model, X, args=(alpha, beta), 
		references=references, device=device, random_state=0)

	assert X_attr0.shape == X_attr1.shape
	assert X_attr0.dtype == X_attr1.dtype
	assert_array_almost_equal(X_attr0, X_attr1)


###
# Tests for additional architectures.
#
# Models whose layers have no rule in `deep_lift_shap._NON_LINEAR_OPS` are
# intentionally absent here: Transformer (MultiheadAttention/softmax + LayerNorm),
# ConvBatchNorm (BatchNorm1d), and ConvLayerNorm (LayerNorm). DLS would silently
# treat those layers as linear and the rescale rule's convergence guarantee
# would not hold, so a regression test against hardcoded values would be
# meaningless. Add a model here once its layers gain a registered rule.


def test_deep_lift_shap_residual_conv(X, references, device):
	torch.manual_seed(0)
	model = ResidualConv()
	X_attr = deep_lift_shap(model, X, references=references,
		device=device, random_state=0)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :4], [
		[[-0.0409, -0.0000, -0.0000, -0.0443],
		 [-0.0000,  0.0000, -0.0201,  0.0000],
		 [ 0.0000, -0.0000, -0.0000,  0.0000],
		 [ 0.0000, -0.0394,  0.0000,  0.0000]],

		[[-0.0000,  0.0000, -0.0313, -0.0000],
		 [-0.0345,  0.0000, -0.0000,  0.0000],
		 [ 0.0000, -0.0089,  0.0000,  0.0000],
		 [ 0.0000, -0.0000,  0.0000,  0.0414]]], 4)


def test_deep_lift_shap_conv2d_expand(X, references, device):
	torch.manual_seed(0)
	model = Conv2DExpand()
	X_attr = deep_lift_shap(model, X, references=references,
		device=device, random_state=0)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :4], [
		[[ 0.0086,  0.0000,  0.0000,  0.0043],
		 [-0.0000, -0.0000,  0.0059, -0.0000],
		 [-0.0000, -0.0000,  0.0000, -0.0000],
		 [ 0.0000,  0.0119, -0.0000,  0.0000]],

		[[ 0.0000, -0.0000,  0.0029, -0.0000],
		 [ 0.0016, -0.0000,  0.0000,  0.0000],
		 [-0.0000, -0.0079,  0.0000, -0.0000],
		 [ 0.0000,  0.0000, -0.0000,  0.0036]]], 4)


def test_deep_lift_shap_custom_linear(X, references, device):
	"""A linear torch.autograd.Function does not need registration; attribution
	flows through it via the standard gradient chain rule."""
	torch.manual_seed(0)
	model = CustomLinear()
	X_attr = deep_lift_shap(model, X, references=references,
		device=device, random_state=0)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :4], [
		[[-0.0177, -0.0000,  0.0000, -0.0008],
		 [-0.0000,  0.0000,  0.0220, -0.0000],
		 [ 0.0000, -0.0000,  0.0000, -0.0000],
		 [-0.0000, -0.0033,  0.0000,  0.0000]],

		[[-0.0000, -0.0000,  0.0011, -0.0000],
		 [-0.0083, -0.0000,  0.0000, -0.0000],
		 [ 0.0000, -0.0334,  0.0000, -0.0000],
		 [-0.0000, -0.0000,  0.0000,  0.0266]]], 4)


def test_deep_lift_shap_custom_sqrt(X, references, device):
	"""A nonlinear custom op is registered via additional_nonlinear_ops so
	DeepLIFT can apply its rescale rule to it."""
	torch.manual_seed(0)
	model = CustomSqrt()
	X_attr = deep_lift_shap(model, X, references=references,
		device=device, random_state=0,
		additional_nonlinear_ops={CustomSqrtModule: _nonlinear})

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :4], [
		[[-0.0020,  0.0000, -0.0000, -0.0030],
		 [-0.0000,  0.0000,  0.0033,  0.0000],
		 [ 0.0000, -0.0000,  0.0000, -0.0000],
		 [-0.0000,  0.0005, -0.0000,  0.0000]],

		[[-0.0000, -0.0000, -0.0001, -0.0000],
		 [-0.0029,  0.0000,  0.0000,  0.0000],
		 [ 0.0000, -0.0042,  0.0000, -0.0000],
		 [-0.0000,  0.0000, -0.0000,  0.0072]]], 4)


def test_deep_lift_shap_dilated_conv(X, references, device):
	torch.manual_seed(0)
	model = DilatedConv()
	X_attr = deep_lift_shap(model, X, references=references,
		device=device, random_state=0)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :4], [
		[[ 0.0000, -0.0000, -0.0000, -0.0008],
		 [ 0.0000,  0.0000, -0.0004,  0.0000],
		 [ 0.0000, -0.0000,  0.0000,  0.0000],
		 [-0.0000,  0.0016,  0.0000, -0.0000]],

		[[-0.0000, -0.0000, -0.0024, -0.0000],
		 [ 0.0004, -0.0000, -0.0000,  0.0000],
		 [ 0.0000, -0.0016,  0.0000,  0.0000],
		 [-0.0000,  0.0000,  0.0000, -0.0006]]], 4)


def test_deep_lift_shap_multi_activation(X, references, device):
	torch.manual_seed(0)
	model = MultiActivation()
	X_attr = deep_lift_shap(model, X, references=references,
		device=device, random_state=0)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :4], [
		[[-0.0007,  0.0000,  0.0000, -0.0001],
		 [ 0.0000, -0.0000, -0.0009, -0.0000],
		 [-0.0000, -0.0000, -0.0000,  0.0000],
		 [ 0.0000,  0.0005,  0.0000, -0.0000]],

		[[-0.0000,  0.0000,  0.0001,  0.0000],
		 [ 0.0009, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0016, -0.0000,  0.0000],
		 [-0.0000, -0.0000, -0.0000, -0.0006]]], 4)


def test_deep_lift_shap_dropout_conv(X, references, device):
	torch.manual_seed(0)
	model = DropoutConv()
	X_attr = deep_lift_shap(model, X, references=references,
		device=device, random_state=0)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :4], [
		[[ 0.0038,  0.0000, -0.0000,  0.0013],
		 [ 0.0000, -0.0000,  0.0232, -0.0000],
		 [-0.0000,  0.0000, -0.0000, -0.0000],
		 [ 0.0000, -0.0016, -0.0000,  0.0000]],

		[[ 0.0000,  0.0000, -0.0021,  0.0000],
		 [ 0.0031,  0.0000,  0.0000, -0.0000],
		 [-0.0000,  0.0062, -0.0000, -0.0000],
		 [ 0.0000, -0.0000,  0.0000,  0.0008]]], 4)


def test_deep_lift_shap_multi_input_multi_output(X, references, device):
	"""MIMO has a tuple output; LambdaWrapper picks one scalar-per-example
	target so DLS can attribute against it."""
	torch.manual_seed(0)
	model = LambdaWrapper(MultiInputMultiOutput(),
		lambda model, X: model(X)[1][:, 0:1])
	X_attr = deep_lift_shap(model, X, references=references,
		device=device, random_state=0)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :4], [
		[[ 0.0411, -0.0000,  0.0000, -0.0138],
		 [-0.0000,  0.0000,  0.0011, -0.0000],
		 [-0.0000,  0.0000, -0.0000,  0.0000],
		 [ 0.0000,  0.0129, -0.0000, -0.0000]],

		[[ 0.0000, -0.0000,  0.0156, -0.0000],
		 [-0.0300,  0.0000,  0.0000, -0.0000],
		 [ 0.0000,  0.0258, -0.0000,  0.0000],
		 [ 0.0000,  0.0000, -0.0000, -0.0028]]], 4)


###
# Activation sweep: confirm every entry in deep_lift_shap._NON_LINEAR_OPS that
# is a pointwise activation actually receives the rescale rule when wired into
# a residual block. GLU is excluded because it halves the channel count and
# would not fit a same-shape residual stream.


# Each entry is (activation_class, expected X_attr[0, :, :4]) so the sweep is
# both a smoke test for hook registration and a regression test for the
# numerical output through that activation.
_ACTIVATION_SWEEP = [
	(torch.nn.ReLU, [
		[-0.0409, -0.0000, -0.0000, -0.0443],
		[-0.0000,  0.0000, -0.0201,  0.0000],
		[ 0.0000, -0.0000, -0.0000,  0.0000],
		[ 0.0000, -0.0394,  0.0000,  0.0000]]),
	(torch.nn.ReLU6, [
		[-0.0409, -0.0000, -0.0000, -0.0443],
		[-0.0000,  0.0000, -0.0201,  0.0000],
		[ 0.0000, -0.0000, -0.0000,  0.0000],
		[ 0.0000, -0.0394,  0.0000,  0.0000]]),
	(torch.nn.RReLU, [
		[-0.0414, -0.0000, -0.0000, -0.0455],
		[-0.0000,  0.0000, -0.0180,  0.0000],
		[ 0.0000, -0.0000, -0.0000,  0.0000],
		[ 0.0000, -0.0396,  0.0000,  0.0000]]),
	(torch.nn.SELU, [
		[-0.0450, -0.0000, -0.0000, -0.0528],
		[-0.0000,  0.0000, -0.0067,  0.0000],
		[ 0.0000, -0.0000, -0.0000,  0.0000],
		[ 0.0000, -0.0409,  0.0000,  0.0000]]),
	(torch.nn.CELU, [
		[-0.0432, -0.0000, -0.0000, -0.0490],
		[-0.0000,  0.0000, -0.0124,  0.0000],
		[ 0.0000, -0.0000, -0.0000,  0.0000],
		[ 0.0000, -0.0402,  0.0000,  0.0000]]),
	(torch.nn.GELU, [
		[-0.0415, -0.0000, -0.0000, -0.0448],
		[-0.0000,  0.0000, -0.0171,  0.0000],
		[ 0.0000, -0.0000, -0.0000,  0.0000],
		[ 0.0000, -0.0391,  0.0000,  0.0000]]),
	(torch.nn.SiLU, [
		[-0.0416, -0.0000, -0.0000, -0.0447],
		[-0.0000,  0.0000, -0.0162,  0.0000],
		[ 0.0000, -0.0000, -0.0000,  0.0000],
		[ 0.0000, -0.0389,  0.0000,  0.0000]]),
	(torch.nn.Mish, [
		[-0.0419, -0.0000, -0.0000, -0.0457],
		[-0.0000,  0.0000, -0.0159,  0.0000],
		[ 0.0000, -0.0000, -0.0000,  0.0000],
		[ 0.0000, -0.0394,  0.0000,  0.0000]]),
	(torch.nn.ELU, [
		[-0.0432, -0.0000, -0.0000, -0.0490],
		[-0.0000,  0.0000, -0.0124,  0.0000],
		[ 0.0000, -0.0000, -0.0000,  0.0000],
		[ 0.0000, -0.0402,  0.0000,  0.0000]]),
	(torch.nn.LeakyReLU, [
		[-0.0409, -0.0000, -0.0000, -0.0444],
		[-0.0000,  0.0000, -0.0200,  0.0000],
		[ 0.0000, -0.0000, -0.0000,  0.0000],
		[ 0.0000, -0.0394,  0.0000,  0.0000]]),
	(torch.nn.Sigmoid, [
		[-0.0411, -0.0000, -0.0000, -0.0418],
		[-0.0000,  0.0000, -0.0165,  0.0000],
		[ 0.0000, -0.0000,  0.0000,  0.0000],
		[ 0.0000, -0.0376,  0.0000,  0.0000]]),
	(torch.nn.Tanh, [
		[-0.0436, -0.0000, -0.0000, -0.0490],
		[-0.0000,  0.0000, -0.0106,  0.0000],
		[ 0.0000, -0.0000, -0.0000,  0.0000],
		[ 0.0000, -0.0399,  0.0000,  0.0000]]),
	(torch.nn.Softplus, [
		[-0.0417, -0.0000, -0.0000, -0.0445],
		[-0.0000,  0.0000, -0.0155,  0.0000],
		[ 0.0000, -0.0000, -0.0000,  0.0000],
		[ 0.0000, -0.0388,  0.0000,  0.0000]]),
	(torch.nn.Softshrink, [
		[-0.0398, -0.0000, -0.0000, -0.0401],
		[-0.0000,  0.0000, -0.0195,  0.0000],
		[ 0.0000, -0.0000,  0.0000,  0.0000],
		[ 0.0000, -0.0376,  0.0000,  0.0000]]),
	(torch.nn.LogSigmoid, [
		[-0.0420, -0.0000, -0.0000, -0.0443],
		[-0.0000,  0.0000, -0.0138,  0.0000],
		[ 0.0000, -0.0000, -0.0000,  0.0000],
		[ 0.0000, -0.0384,  0.0000,  0.0000]]),
	(torch.nn.PReLU, [
		[-0.0415, -0.0000, -0.0000, -0.0456],
		[-0.0000,  0.0000, -0.0178,  0.0000],
		[ 0.0000, -0.0000, -0.0000,  0.0000],
		[ 0.0000, -0.0396,  0.0000,  0.0000]]),
]


@pytest.mark.parametrize("activation,expected", _ACTIVATION_SWEEP,
	ids=[a.__name__ for a, _ in _ACTIVATION_SWEEP])
def test_deep_lift_shap_residual_conv_activation(X, references, device,
		activation, expected):
	torch.manual_seed(0)
	model = ResidualConv(activation=activation)
	X_attr = deep_lift_shap(model, X, references=references,
		device=device, random_state=0)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[0, :, :4], expected, 4)


###


def test_deep_lift_shap_single_sequence(X, device):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	X_attr = deep_lift_shap(model, X[:1], n_shuffles=10, batch_size=3,
		device=device, random_state=0)

	assert X_attr.shape == (1, 4, 100)
	assert X_attr.dtype == torch.float32

	X_attr_big = deep_lift_shap(model, X[:1], n_shuffles=10, batch_size=64,
		device=device, random_state=0)
	assert_array_almost_equal(X_attr, X_attr_big, 4)


def test_deep_lift_shap_print_convergence_deltas(X, device, capsys):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	deep_lift_shap(model, X[:4], n_shuffles=2, batch_size=8, device=device,
		random_state=0, print_convergence_deltas=True)

	captured = capsys.readouterr()
	assert captured.out.strip() != ""


def test_deep_lift_shap_only_warn(X, device):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	refs = shuffle(X, n=2, random_state=0)
	X_bad = X * 2.0

	assert_raises(ValueError, deep_lift_shap, model, X_bad, references=refs,
		device=device, random_state=0)

	X_attr = deep_lift_shap(model, X_bad, references=refs, device=device,
		random_state=0, only_warn=True)

	assert X_attr.shape == X_bad.shape


def test_deep_lift_shap_preserves_model_state(X, device):
	# After deep_lift_shap returns the model should be back on its
	# original device and in its original training mode, with no
	# _NON_LINEAR_OPS attributes leaked onto its modules.
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)
	model.train()  # explicitly leave model in train mode
	orig_device = next(model.parameters()).device

	deep_lift_shap(model, X[:4], n_shuffles=2, device=device, random_state=0)

	assert model.training, "training mode was not restored"
	assert next(model.parameters()).device == orig_device, \
		"device was not restored"
	for module in model.modules():
		assert not hasattr(module, "_NON_LINEAR_OPS"), \
			"_NON_LINEAR_OPS attribute leaked onto module"


def test_deep_lift_shap_cleans_up_hooks_on_exception(X, device):
	# If the forward pass raises mid-loop, hooks and _NON_LINEAR_OPS
	# attributes must still be cleaned up by the finally clause.
	class Boom(torch.nn.Module):
		def __init__(self):
			super().__init__()
			self.dense = torch.nn.Linear(100 * 4, 1)

		def forward(self, X):
			raise RuntimeError("intentional boom")

	model = Boom()

	with pytest.raises(RuntimeError):
		deep_lift_shap(model, X[:2], n_shuffles=1, device=device,
			random_state=0)

	for module in model.modules():
		assert not hasattr(module, "_NON_LINEAR_OPS"), \
			"_NON_LINEAR_OPS leaked after exception"


def test_deep_lift_shap_dtype_param(X, device):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	X_attr0 = deep_lift_shap(model, X[:4], n_shuffles=2, dtype=torch.float32,
		device=device, random_state=0)
	X_attr1 = deep_lift_shap(model, X[:4], n_shuffles=2, dtype="float32",
		device=device, random_state=0)
	X_attr2 = deep_lift_shap(model, X[:4], n_shuffles=2, dtype=None,
		device=device, random_state=0)

	assert X_attr0.dtype == torch.float32
	assert_array_almost_equal(X_attr0, X_attr1, 4)
	assert_array_almost_equal(X_attr0, X_attr2, 4)


def test_deep_lift_shap_random_state_none_consistency(X, device):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	X_attr0 = deep_lift_shap(model, X[:4], n_shuffles=2, device=device,
		random_state=None)
	X_attr1 = deep_lift_shap(model, X[:4], n_shuffles=2, device=device,
		random_state=None)

	assert_raises(AssertionError, assert_array_almost_equal, X_attr0, X_attr1)


def test_deep_lift_shap_invalid_target(X, device):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	assert_raises(IndexError, deep_lift_shap, model, X[:2], target=999,
		n_shuffles=2, device=device, random_state=0)


def test_deep_lift_shap_empty_X(device):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)
	X_empty = torch.zeros(0, 4, 100)

	assert_raises(ValueError, deep_lift_shap, model, X_empty,
		n_shuffles=2, device=device, random_state=0)
