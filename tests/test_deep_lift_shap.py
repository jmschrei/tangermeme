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
from tangermeme.deep_lift_shap import _fp_hook
from tangermeme.deep_lift_shap import _f_hook
from tangermeme.deep_lift_shap import _b_hook
from tangermeme._deep_lift_utils import _nonlinear
from tangermeme._deep_lift_utils import _bilinear
from tangermeme._deep_lift_utils import _HookState
from tangermeme._deep_lift_utils import _disable_hooks
from tangermeme._deep_lift_utils import _hooks_disabled
from tangermeme.deep_lift_shap import integrated_gradients_op

from .toy_models import SoftmaxModel
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
from .toy_models import ConvLayerNorm
from .toy_models import ConvRMSNorm
from .toy_models import ConvSoftmax
from .toy_models import ConvBilinear
from .toy_models import ConvBilinearMatmul
from .toy_models import ConvBilinearEinsum
from .toy_models import ConvCustomGate
from .toy_models import CustomGate
from .toy_models import ConvScaledTanh
from .toy_models import ScaledTanhModule
from .toy_models import MultiHeadAttention
from .toy_models import TransformerBlock
from .toy_models import Transformer

from tangermeme.deep_lift_shap import BilinearOp
from .toy_models import AttributeNameConv

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
# intentionally absent here. DLS would silently treat those layers as linear
# and the rescale rule's convergence guarantee would not hold, so a regression
# test against hardcoded values would be meaningless. Add a model here once its
# layers gain a registered rule.


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


def test_deep_lift_shap_layernorm(X, references, device):
	# fp32 attribution residuals on CUDA are a few orders of magnitude larger
	# than on CPU, so the convergence threshold is loosened for the cuda pass.
	threshold = 1e-4 if device == "cpu" else 1e-2
	torch.manual_seed(0)
	model = ConvLayerNorm(seq_len=X.shape[-1])

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		X_attr = deep_lift_shap(model, X, references=references,
			device=device, random_state=0, warning_threshold=threshold)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :4], [
		[[ 0.0114,  0.0000, -0.0000,  0.0050],
         [ 0.0000, -0.0000,  0.0695, -0.0000],
         [-0.0000,  0.0000, -0.0000, -0.0000],
         [ 0.0000, -0.0054, -0.0000,  0.0000]],

        [[ 0.0000,  0.0000, -0.0063,  0.0000],
         [ 0.0092,  0.0000,  0.0000, -0.0000],
         [-0.0000,  0.0183, -0.0000, -0.0000],
         [ 0.0000, -0.0000,  0.0000,  0.0023]]], 4)


def test_deep_lift_shap_layernorm_local_ig(X, references, device):
	# fp32 attribution residuals on CUDA are a few orders of magnitude larger
	# than on CPU, so the convergence threshold is loosened for the cuda pass.
	threshold = 1e-4 if device == "cpu" else 1e-2
	torch.manual_seed(0)
	model = ConvLayerNorm(seq_len=X.shape[-1])
	ig_hook = integrated_gradients_op(K=8, name="layernorm")

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		X_attr = deep_lift_shap(model, X, references=references,
			device=device, random_state=0, warning_threshold=threshold,
			additional_nonlinear_ops={torch.nn.LayerNorm: ig_hook})

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :4], [
		[[ 0.0124,  0.0000, -0.0000,  0.0053],
         [ 0.0000, -0.0000,  0.0768, -0.0000],
         [-0.0000,  0.0000, -0.0000, -0.0000],
         [ 0.0000, -0.0064, -0.0000,  0.0000]],

        [[ 0.0000,  0.0000, -0.0076,  0.0000],
         [ 0.0100,  0.0000,  0.0000, -0.0000],
         [-0.0000,  0.0195, -0.0000, -0.0000],
         [ 0.0000, -0.0000,  0.0000,  0.0022]]], 4)


def test_deep_lift_shap_rmsnorm(X, references, device):
	# fp32 attribution residuals on CUDA are a few orders of magnitude larger
	# than on CPU, so the convergence threshold is loosened for the cuda pass.
	threshold = 1e-4 if device == "cpu" else 1e-2
	torch.manual_seed(0)
	model = ConvRMSNorm(seq_len=X.shape[-1])

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		X_attr = deep_lift_shap(model, X, references=references,
			device=device, random_state=0, warning_threshold=threshold)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :4], [
		[[ 0.0108,  0.0000, -0.0000,  0.0037],
         [ 0.0000, -0.0000,  0.0670, -0.0000],
         [-0.0000,  0.0000, -0.0000, -0.0000],
         [ 0.0000, -0.0042, -0.0000,  0.0000]],

        [[ 0.0000,  0.0000, -0.0062,  0.0000],
         [ 0.0092,  0.0000,  0.0000, -0.0000],
         [-0.0000,  0.0178, -0.0000, -0.0000],
         [ 0.0000, -0.0000,  0.0000,  0.0022]]], 4)


def test_deep_lift_shap_rmsnorm_local_ig(X, references, device):
	# The analytic rule and the local-IG hook are NOT the same quantity, and the
	# hardcoded values below differ from those in the test above for that reason.
	# RMSNorm couples every output to every input in the normalized window, where
	# the rescale secant and a path integral part ways; the two differ by 1.3e-2
	# here and the gap does not shrink with more quadrature points. What both must
	# satisfy is summation-to-delta, which is why the threshold is asserted.
	threshold = 1e-4 if device == "cpu" else 1e-2
	torch.manual_seed(0)
	model = ConvRMSNorm(seq_len=X.shape[-1])
	ig_hook = integrated_gradients_op(K=8, name="rmsnorm")

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		X_attr = deep_lift_shap(model, X, references=references,
			device=device, random_state=0, warning_threshold=threshold,
			additional_nonlinear_ops={torch.nn.RMSNorm: ig_hook})

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :4], [
		[[ 0.0118,  0.0000, -0.0000,  0.0039],
         [ 0.0000, -0.0000,  0.0741, -0.0000],
         [-0.0000,  0.0000, -0.0000, -0.0000],
         [ 0.0000, -0.0051, -0.0000,  0.0000]],

        [[ 0.0000,  0.0000, -0.0074,  0.0000],
         [ 0.0099,  0.0000,  0.0000, -0.0000],
         [-0.0000,  0.0190, -0.0000, -0.0000],
         [ 0.0000, -0.0000,  0.0000,  0.0021]]], 4)


def test_deep_lift_shap_norm_completeness(X, references, device):
	"""The DeepLIFT rules must satisfy summation-to-delta for both norm types.

	This checks the property the hooks exist to preserve, independent of the
	hard-coded attribution values above, and covers a non-identity gamma/beta
	(which the default-initialized toy models leave as ones/zeros).
	"""
	for cls, seed in ((ConvLayerNorm, 0), (ConvRMSNorm, 1)):
		torch.manual_seed(seed)
		model = cls(seq_len=X.shape[-1])

		# Default init leaves the affine params as identity, so randomize them to
		# exercise the gamma-scaling path in the hooks.
		norm = model.ln if hasattr(model, "ln") else model.norm
		torch.nn.init.normal_(norm.weight)
		if getattr(norm, "bias", None) is not None:
			torch.nn.init.normal_(norm.bias)

		with warnings.catch_warnings():
			warnings.simplefilter("error", category=RuntimeWarning)

			X_attr = deep_lift_shap(model, X, references=references,
				device=device, random_state=0,
				warning_threshold=1e-4 if device == "cpu" else 1e-2)

		assert X_attr.shape == X.shape


def test_deep_lift_shap_rmsnorm_explicit_eps(X, references, device):
	"""An explicitly-set eps must be honored rather than replaced by finfo.eps.

	`torch.nn.RMSNorm.eps` defaults to None (meaning the dtype epsilon), so the
	hook must key off None instead of assuming every RMSNorm uses the default;
	otherwise the rule mismatches the forward pass and convergence deltas blow up.
	"""
	torch.manual_seed(0)
	model = ConvRMSNorm(seq_len=X.shape[-1])
	model.norm = torch.nn.RMSNorm([8, X.shape[-1]], eps=1e-2)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		X_attr = deep_lift_shap(model, X, references=references,
			device=device, random_state=0,
			warning_threshold=1e-4 if device == "cpu" else 1e-2)

	assert X_attr.shape == X.shape


def test_deep_lift_shap_softmax(X, references, device):
	# fp32 attribution residuals on CUDA are a few orders of magnitude larger
	# than on CPU, so the convergence threshold is loosened for the cuda pass.
	threshold = 1e-4 if device == "cpu" else 1e-2
	torch.manual_seed(0)

	model = LambdaWrapper(SoftmaxModel(), lambda model, X: model(X)[:, 0, 0:1])

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		X_attr = deep_lift_shap(model, X, references=references,
			device=device, random_state=0,
			warning_threshold=threshold)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :4], [
		[[ 9.3886e-03, -0.0000e+00, -0.0000e+00, -1.0105e-04],
         [-0.0000e+00,  0.0000e+00,  7.2842e-05,  0.0000e+00],
         [-0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
         [-0.0000e+00,  2.8212e-05,  0.0000e+00,  0.0000e+00]],

        [[ 0.0000e+00, -0.0000e+00, -4.8106e-05, -0.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
         [ 0.0000e+00,  3.2071e-05,  0.0000e+00,  0.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  3.2071e-05]]], 4)


###
# Comprehensive coverage for the rules registered beyond the stock elementwise
# activations: LayerNorm, RMSNorm, softmax, and the three bilinear
# contractions, plus two user-defined ops that only attribute correctly once
# they are handed to `additional_nonlinear_ops`.
#
# Every model routes through exactly one new rule and shares the same
# `forward(X, alpha=0, beta=1)` signature, so a single parametrized matrix can
# put all of them through the settings the stock models are tested against
# above: convergence, batch size, shuffle count, example independence, seed,
# an explicit reference tensor, input dtype, hypothetical attributions, raw
# outputs, returned references, and extra forward args.
###


RULE_MODELS = [
	(ConvLayerNorm, None),
	(ConvRMSNorm, None),
	(ConvSoftmax, None),
	(ConvBilinear, None),
	(ConvBilinearMatmul, None),
	(ConvBilinearEinsum, None),
	(ConvScaledTanh, ScaledTanhModule),
	(ConvCustomGate, CustomGate),
]

RULE_IDS = ["layernorm", "rmsnorm", "softmax", "bilinear_elementwise",
	"bilinear_matmul", "bilinear_einsum", "custom_nonlinear", "custom_bilinear"]

RULE_PARAMS = pytest.mark.parametrize("model_cls,op", RULE_MODELS, ids=RULE_IDS)


def _rule_model(model_cls, op, seed=0):
	"""Build a rule model and the kwargs that register its op, if any.

	The two user-defined ops are absent from the default table, so they have to
	be passed in; the six built-in ones take an empty dict.
	"""

	torch.manual_seed(seed)
	model = model_cls()

	if op is None:
		return model, {}

	rule = _bilinear if op is CustomGate else _nonlinear
	return model, {"additional_nonlinear_ops": {op: rule}}


def _rule_threshold(device):
	# fp32 attribution residuals on CUDA are a few orders of magnitude larger
	# than on CPU, so the convergence threshold is loosened for the cuda pass.
	return 1e-4 if device == "cpu" else 1e-2


@RULE_PARAMS
def test_deep_lift_shap_rules_convergence(X, references, device, model_cls, op):
	"""Summation-to-delta is the property every one of these rules exists for.

	A layer with no rule is silently treated as linear, so the convergence
	delta is what separates a correct rule from an absent one.
	"""

	model, kwargs = _rule_model(model_cls, op)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		X_attr = deep_lift_shap(model, X, references=references, device=device,
			random_state=0, warning_threshold=_rule_threshold(device), **kwargs)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32


@RULE_PARAMS
def test_deep_lift_shap_rules_batch_size(X, references, device, model_cls, op):
	model, kwargs = _rule_model(model_cls, op)

	# Half the fixture: batch_size=1 runs one example-reference pair per pass,
	# so the full 16 would put the bilinear models over the runtime budget.
	X, references = X[:8], references[:8]

	X_attr0 = deep_lift_shap(model, X, references=references, device=device,
		random_state=0, **kwargs)
	X_attr1 = deep_lift_shap(model, X, references=references, batch_size=1,
		device=device, random_state=0, **kwargs)
	X_attr2 = deep_lift_shap(model, X, references=references, batch_size=100000,
		device=device, random_state=0, **kwargs)
	X_attr3 = deep_lift_shap(model, X, references=references, batch_size=20,
		device=device, random_state=0, **kwargs)

	assert_array_almost_equal(X_attr0, X_attr1, 4)
	assert_array_almost_equal(X_attr0, X_attr2, 4)
	assert_array_almost_equal(X_attr0, X_attr3, 4)


@RULE_PARAMS
def test_deep_lift_shap_rules_n_shuffles(X, device, model_cls, op):
	model, kwargs = _rule_model(model_cls, op)
	X = X[:4]

	X_attr0 = deep_lift_shap(model, X, n_shuffles=1, device=device,
		random_state=0, **kwargs)
	X_attr1 = deep_lift_shap(model, X, n_shuffles=1, batch_size=1, device=device,
		random_state=0, **kwargs)
	X_attr2 = deep_lift_shap(model, X, n_shuffles=5, batch_size=100000,
		device=device, random_state=2, **kwargs)
	X_attr3 = deep_lift_shap(model, X, n_shuffles=5, batch_size=1, device=device,
		random_state=2, **kwargs)

	# The shuffle count must not interact with the batching.
	assert_array_almost_equal(X_attr0, X_attr1, 4)
	assert_array_almost_equal(X_attr2, X_attr3, 4)
	assert_raises(AssertionError, assert_array_almost_equal, X_attr0, X_attr3)


@RULE_PARAMS
def test_deep_lift_shap_rules_independence(X, references, device, model_cls, op):
	model, kwargs = _rule_model(model_cls, op)

	X_attr = deep_lift_shap(model, X, references=references, device=device,
		random_state=0, **kwargs)
	X_attr0 = deep_lift_shap(model, X[0:1], references=references[0:1],
		device=device, random_state=0, **kwargs)
	X_attr1 = deep_lift_shap(model, X[5:6], references=references[5:6],
		device=device, random_state=0, **kwargs)
	X_attr2 = deep_lift_shap(model, X[8:10], references=references[8:10],
		device=device, random_state=0, **kwargs)

	# An example's attribution must not depend on what it is batched with. A
	# normalization rule that leaked across the batch axis would break here.
	assert_array_almost_equal(X_attr[0:1], X_attr0, 4)
	assert_array_almost_equal(X_attr[5:6], X_attr1, 4)
	assert_array_almost_equal(X_attr[8:10], X_attr2, 4)

	assert_raises(AssertionError, assert_array_almost_equal, X_attr0, X_attr1)


@RULE_PARAMS
def test_deep_lift_shap_rules_random_state(X, device, model_cls, op):
	model, kwargs = _rule_model(model_cls, op)
	X = X[:4]

	X_attr0 = deep_lift_shap(model, X, device=device, random_state=0, **kwargs)
	X_attr1 = deep_lift_shap(model, X, device=device, random_state=0, **kwargs)
	X_attr2 = deep_lift_shap(model, X, device=device, random_state=1, **kwargs)

	assert_array_almost_equal(X_attr0, X_attr1, 4)
	assert_raises(AssertionError, assert_array_almost_equal, X_attr0, X_attr2)


@RULE_PARAMS
def test_deep_lift_shap_rules_reference_tensor(X, device, model_cls, op):
	model, kwargs = _rule_model(model_cls, op)
	references = shuffle(X, n=20, random_state=0)

	X_attr0 = deep_lift_shap(model, X, references=references, device=device,
		random_state=0, **kwargs)
	X_attr1 = deep_lift_shap(model, X[0:10], references=references[:10],
		device=device, random_state=1, **kwargs)

	# An explicit reference tensor makes the result independent of the seed.
	assert_array_almost_equal(X_attr0[:10], X_attr1, 4)


@RULE_PARAMS
def test_deep_lift_shap_rules_input_type(X, references, device, model_cls, op):
	model, kwargs = _rule_model(model_cls, op)
	X, references = X[:4], references[:4]

	X_attr0 = deep_lift_shap(model, X, references=references, device=device,
		random_state=0, **kwargs)

	for dtype in (torch.int8, torch.int16, torch.int32, torch.float16,
		torch.bfloat16):
		X_attr = deep_lift_shap(model, X.type(dtype), references=references,
			device=device, random_state=0, **kwargs)

		assert X_attr.dtype == torch.float32
		assert_array_almost_equal(X_attr0, X_attr, 4)


@RULE_PARAMS
def test_deep_lift_shap_rules_hypothetical(X, references, device, model_cls, op):
	model, kwargs = _rule_model(model_cls, op)

	X_attr = deep_lift_shap(model, X, references=references, hypothetical=True,
		device=device, random_state=0, **kwargs)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32

	# Hypothetical attributions are dense; the observed ones are the
	# hypothetical values gated by the one-hot input.
	X_obs = deep_lift_shap(model, X, references=references, device=device,
		random_state=0, **kwargs)
	assert_array_almost_equal(X_attr * X.to(X_attr.device), X_obs, 4)


@RULE_PARAMS
def test_deep_lift_shap_rules_raw_outputs(X, references, device, model_cls, op):
	model, kwargs = _rule_model(model_cls, op)

	multipliers = deep_lift_shap(model, X, references=references,
		raw_outputs=True, device=device, random_state=0, **kwargs)

	# `raw_outputs` hands back one set of multipliers per reference, before
	# `hypothetical_attributions` projects them onto the alphabet.
	assert multipliers.shape == (X.shape[0], references.shape[1], *X.shape[1:])
	assert multipliers.dtype == torch.float32

	n, k = X.shape[0], references.shape[1]
	X_flat = X.unsqueeze(1).expand(-1, k, -1, -1).reshape(n * k, *X.shape[1:])
	ref_flat = references.reshape(n * k, *X.shape[1:])
	mult_flat = multipliers.cpu().reshape(n * k, *X.shape[1:])

	X_hyp = hypothetical_attributions((mult_flat,), (X_flat,), (ref_flat,))[0]
	X_hyp = X_hyp.reshape(n, k, *X.shape[1:]).mean(dim=1)

	# Doing that projection by hand and averaging over references has to
	# reproduce both cooked outputs, which pins the raw multipliers against the
	# two paths that consume them.
	X_attr_hyp = deep_lift_shap(model, X, references=references,
		hypothetical=True, device=device, random_state=0, **kwargs)
	X_attr = deep_lift_shap(model, X, references=references, device=device,
		random_state=0, **kwargs)

	assert_array_almost_equal(X_hyp, X_attr_hyp.cpu(), 4)
	assert_array_almost_equal(X_hyp * X, X_attr.cpu(), 4)


@RULE_PARAMS
def test_deep_lift_shap_rules_return_references(X, references, device,
	model_cls, op):
	model, kwargs = _rule_model(model_cls, op)

	result = deep_lift_shap(model, X, references=references,
		return_references=True, device=device, random_state=0, **kwargs)

	assert isinstance(result, tuple)
	assert len(result) == 2

	X_attr, refs = result
	assert X_attr.shape == X.shape
	assert refs.shape == references.shape
	assert_array_almost_equal(refs.cpu(), references, 4)
	assert_array_almost_equal(X_attr, result.attributions, 4)


@RULE_PARAMS
def test_deep_lift_shap_rules_args(X, references, device, model_cls, op):
	model, kwargs = _rule_model(model_cls, op)
	X, references = X[:4], references[:4]

	torch.manual_seed(0)
	alpha = torch.randn(4, 1)
	beta = torch.randn(4, 1)

	X_attr0 = deep_lift_shap(model, X, references=references, device=device,
		random_state=0, **kwargs)
	X_attr1 = deep_lift_shap(model, X, references=references, args=(alpha,),
		device=device, random_state=0, **kwargs)
	X_attr2 = deep_lift_shap(model, X, references=references, args=(alpha, beta),
		device=device, random_state=0, **kwargs)

	assert X_attr1.shape == X.shape
	assert X_attr2.shape == X.shape

	# An additive shift leaves attributions alone; a multiplicative one scales
	# them.
	assert_array_almost_equal(X_attr0, X_attr1, 4)
	assert_raises(AssertionError, assert_array_almost_equal, X_attr0, X_attr2)
	assert_array_almost_equal(X_attr0 * beta.reshape(-1, 1, 1).to(X_attr2.device),
		X_attr2, 4)



# Attributions for the first two examples, first four positions, with the
# `references` fixture pinned. Regenerating these means the rule changed.
RULE_REGRESSION = {
	"ConvLayerNorm": [
		[[ 0.0114,  0.0000, -0.0000,  0.0050],
		 [ 0.0000, -0.0000,  0.0695, -0.0000],
		 [-0.0000,  0.0000, -0.0000, -0.0000],
		 [ 0.0000, -0.0054, -0.0000,  0.0000]],

		[[ 0.0000,  0.0000, -0.0063,  0.0000],
		 [ 0.0092,  0.0000,  0.0000, -0.0000],
		 [-0.0000,  0.0183, -0.0000, -0.0000],
		 [ 0.0000, -0.0000,  0.0000,  0.0023]]],

	"ConvRMSNorm": [
		[[ 0.0108,  0.0000, -0.0000,  0.0037],
		 [ 0.0000, -0.0000,  0.0670, -0.0000],
		 [-0.0000,  0.0000, -0.0000, -0.0000],
		 [ 0.0000, -0.0042, -0.0000,  0.0000]],

		[[ 0.0000,  0.0000, -0.0062,  0.0000],
		 [ 0.0092,  0.0000,  0.0000, -0.0000],
		 [-0.0000,  0.0178, -0.0000, -0.0000],
		 [ 0.0000, -0.0000,  0.0000,  0.0022]]],

	"ConvSoftmax": [
		[[ 0.0023, -0.0000, -0.0000,  0.0149],
		 [ 0.0000,  0.0000,  0.0391, -0.0000],
		 [-0.0000,  0.0000,  0.0000, -0.0000],
		 [ 0.0000, -0.0078,  0.0000,  0.0000]],

		[[ 0.0000,  0.0000, -0.0164,  0.0000],
		 [-0.0007,  0.0000,  0.0000, -0.0000],
		 [-0.0000,  0.0145,  0.0000, -0.0000],
		 [ 0.0000, -0.0000,  0.0000,  0.0052]]],

	"ConvBilinear": [
		[[-0.0004,  0.0000,  0.0000,  0.0040],
		 [ 0.0000,  0.0000,  0.0051, -0.0000],
		 [-0.0000, -0.0000,  0.0000, -0.0000],
		 [ 0.0000, -0.0020, -0.0000,  0.0000]],

		[[-0.0000,  0.0000, -0.0054,  0.0000],
		 [ 0.0011, -0.0000,  0.0000, -0.0000],
		 [-0.0000, -0.0050,  0.0000, -0.0000],
		 [ 0.0000, -0.0000, -0.0000, -0.0014]]],

	"ConvBilinearMatmul": [
		[[ 0.0080,  0.0000,  0.0000,  0.0113],
		 [-0.0000, -0.0000, -0.1176, -0.0000],
		 [ 0.0000, -0.0000, -0.0000, -0.0000],
		 [ 0.0000,  0.0385,  0.0000,  0.0000]],

		[[ 0.0000, -0.0000,  0.0468,  0.0000],
		 [-0.0301, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.1420, -0.0000, -0.0000],
		 [ 0.0000, -0.0000, -0.0000, -0.0034]]],

	"ConvBilinearEinsum": [
		[[ 0.0080,  0.0000,  0.0000,  0.0113],
		 [-0.0000, -0.0000, -0.1176, -0.0000],
		 [ 0.0000, -0.0000, -0.0000, -0.0000],
		 [ 0.0000,  0.0385,  0.0000,  0.0000]],

		[[ 0.0000, -0.0000,  0.0468,  0.0000],
		 [-0.0301, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.1420, -0.0000, -0.0000],
		 [ 0.0000, -0.0000, -0.0000, -0.0034]]],

	"ConvScaledTanh": [
		[[ 0.0024, -0.0000, -0.0000,  0.0339],
		 [-0.0000,  0.0000,  0.0717, -0.0000],
		 [-0.0000,  0.0000,  0.0000, -0.0000],
		 [ 0.0000, -0.0243,  0.0000,  0.0000]],

		[[ 0.0000,  0.0000, -0.0276,  0.0000],
		 [-0.0021,  0.0000,  0.0000, -0.0000],
		 [-0.0000,  0.0347,  0.0000, -0.0000],
		 [ 0.0000, -0.0000,  0.0000,  0.0090]]],

	"ConvCustomGate": [
		[[-0.0004,  0.0000,  0.0000,  0.0040],
		 [ 0.0000,  0.0000,  0.0051, -0.0000],
		 [-0.0000, -0.0000,  0.0000, -0.0000],
		 [ 0.0000, -0.0020, -0.0000,  0.0000]],

		[[-0.0000,  0.0000, -0.0054,  0.0000],
		 [ 0.0011, -0.0000,  0.0000, -0.0000],
		 [-0.0000, -0.0050,  0.0000, -0.0000],
		 [ 0.0000, -0.0000, -0.0000, -0.0014]]]
}


@RULE_PARAMS
def test_deep_lift_shap_rules_regression(X, references, device, model_cls, op):
	model, kwargs = _rule_model(model_cls, op)

	X_attr = deep_lift_shap(model, X, references=references, device=device,
		random_state=0, **kwargs)

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :4],
		RULE_REGRESSION[model_cls.__name__], 4)


def test_deep_lift_shap_bilinear_matmul_matches_einsum(X, references, device):
	"""The matmul and einsum branches of BilinearOp contract the same way.

	ConvBilinearMatmul transposes its right operand and lets the op call
	`torch.matmul`; ConvBilinearEinsum hands both operands over untransposed
	and names the contraction. Same seed, same contraction, so the
	attributions must agree.
	"""

	torch.manual_seed(0)
	matmul = ConvBilinearMatmul()

	torch.manual_seed(0)
	einsum = ConvBilinearEinsum()

	X_attr0 = deep_lift_shap(matmul, X, references=references, device=device,
		random_state=0)
	X_attr1 = deep_lift_shap(einsum, X, references=references, device=device,
		random_state=0)

	assert_array_almost_equal(X_attr0, X_attr1, 4)


CUSTOM_OPS = [(ConvScaledTanh, ScaledTanhModule), (ConvCustomGate, CustomGate)]
CUSTOM_IDS = ["custom_nonlinear", "custom_bilinear"]


@pytest.mark.parametrize("model_cls,op", CUSTOM_OPS, ids=CUSTOM_IDS)
def test_deep_lift_shap_custom_op_requires_registration(X, references, device,
	model_cls, op):
	"""An unregistered op is silently treated as linear.

	That silence is the whole reason `additional_nonlinear_ops` exists, so both
	sides are pinned here: without the rule the convergence delta blows past
	the threshold, and with it summation-to-delta holds.
	"""

	threshold = _rule_threshold(device)
	model, kwargs = _rule_model(model_cls, op)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		assert_raises(RuntimeWarning, deep_lift_shap, model, X,
			references=references, device=device, random_state=0,
			warning_threshold=threshold)

		X_attr = deep_lift_shap(model, X, references=references, device=device,
			random_state=0, warning_threshold=threshold, **kwargs)

	assert X_attr.shape == X.shape


def test_deep_lift_shap_custom_bilinear_matches_builtin(X, references, device):
	"""A registered user-defined bilinear op matches the built-in BilinearOp.

	ConvCustomGate and ConvBilinear compute the same elementwise product and
	are built from the same seed, so their parameters agree and the only
	difference is whether `_bilinear` is reached through the default table or
	through `additional_nonlinear_ops`.
	"""

	torch.manual_seed(0)
	builtin = ConvBilinear()

	torch.manual_seed(0)
	custom = ConvCustomGate()

	X_attr0 = deep_lift_shap(builtin, X, references=references, device=device,
		random_state=0)
	X_attr1 = deep_lift_shap(custom, X, references=references, device=device,
		random_state=0, additional_nonlinear_ops={CustomGate: _bilinear})

	assert_array_almost_equal(X_attr0, X_attr1, 4)


def test_deep_lift_shap_custom_op_overrides_builtin(X, references, device):
	"""`additional_nonlinear_ops` takes precedence over the default table.

	Registering the generic local-IG hook for LayerNorm has to displace the
	closed-form rule, which is observable because the two disagree. They are
	not the same quantity: the rescale rule and a path integral coincide for
	an elementwise function but not for one whose outputs couple across
	positions, so the gap does not shrink with more quadrature points. Both
	satisfy summation-to-delta, which is what is checked here rather than
	agreement between them.
	"""

	torch.manual_seed(0)
	model = ConvLayerNorm()

	X_attr0 = deep_lift_shap(model, X, references=references, device=device,
		random_state=0)
	X_attr1 = deep_lift_shap(model, X, references=references, device=device,
		random_state=0, additional_nonlinear_ops={
			torch.nn.LayerNorm: integrated_gradients_op(K=8, name="layernorm")})

	assert_raises(AssertionError, assert_array_almost_equal, X_attr0, X_attr1, 4)

	# Both rules must still be complete, which is the property that matters.
	for X_attr in (X_attr0, X_attr1):
		with warnings.catch_warnings():
			warnings.simplefilter("error", category=RuntimeWarning)

			deep_lift_shap(model, X, references=references, device=device,
				random_state=0, warning_threshold=_rule_threshold(device))


@pytest.mark.skip(reason="hooks are registered by isinstance but dispatched by "
	"exact type, so subclassing any registered op raises KeyError in _b_hook. "
	"Pre-existing on main for every op, not introduced by these rules.")
def test_deep_lift_shap_subclassed_op(X, references, device):
	"""A subclass of a registered op should inherit its rule."""

	class SubclassedGate(BilinearOp):
		pass

	torch.manual_seed(0)
	model = ConvBilinear()
	model.op = SubclassedGate("...,...->...")

	X_attr = deep_lift_shap(model, X, references=references, device=device,
		random_state=0)

	assert X_attr.shape == X.shape



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


def test_deep_lift_shap_return_references_named_tuple(X, device):
	from tangermeme.results import AttributionReferencesResult

	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	result = deep_lift_shap(model, X[:2], n_shuffles=2,
		return_references=True, device=device, random_state=0)

	attributions, references = result
	assert torch.equal(result.attributions, attributions)
	assert torch.equal(result.references, references)
	assert isinstance(result, AttributionReferencesResult)
	assert isinstance(result, tuple)

	# When return_references=False the return is still a plain Tensor.
	attr = deep_lift_shap(model, X[:2], n_shuffles=2,
		return_references=False, device=device, random_state=0)
	assert isinstance(attr, torch.Tensor)


def test_deep_lift_shap_verbose(X, device):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	X_quiet = deep_lift_shap(model, X[:2], n_shuffles=2, device=device,
		random_state=0, verbose=False)
	X_loud = deep_lift_shap(model, X[:2], n_shuffles=2, device=device,
		random_state=0, verbose=True)

	assert_array_almost_equal(X_quiet, X_loud)


def test_deep_lift_shap_clears_hook_caches(X, device):
	torch.manual_seed(0)
	model = SmallDeepSEA()

	deep_lift_shap(model, X[:2], n_shuffles=2, device=device, random_state=0)

	# The forward hooks cache a detached copy of each non-linearity's input
	# and output. Those copies are as large as the activations themselves, so
	# leaving them attached pins that much memory for the life of the model.
	for module in model.modules():
		assert "input" not in module.__dict__
		assert "output" not in module.__dict__


def test_deep_lift_shap_clears_hook_caches_on_error(X, device):
	torch.manual_seed(0)
	model = SmallDeepSEA()

	# `target` is out of range for a one-output model, so the attribution loop
	# raises and unwinds through the `finally` block that clears the hooks.
	assert_raises(IndexError, deep_lift_shap, model, X[:2], target=5,
		n_shuffles=2, device=device, random_state=0)

	for module in model.modules():
		assert "input" not in module.__dict__
		assert "output" not in module.__dict__


def test_deep_lift_shap_softmax_channel_axis(X, references, device):
	"""The softmax rule follows whichever axis the module normalizes over.

	A softmax over the channel axis of an (N, C, L) tensor is an ordinary
	thing to write for genomic data, and is the case the rule refused before
	its reductions were moved off a hardcoded last axis.
	"""

	torch.manual_seed(0)
	model = ConvSoftmax(dim=1)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		X_attr = deep_lift_shap(model, X, references=references, device=device,
			random_state=0, warning_threshold=_rule_threshold(device))

	assert X_attr.shape == X.shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :4], [
		[[ 0.0032,  0.0000, -0.0000,  0.0071],
		 [ 0.0000,  0.0000,  0.0346, -0.0000],
		 [-0.0000,  0.0000,  0.0000, -0.0000],
		 [ 0.0000, -0.0172,  0.0000,  0.0000]],

		[[ 0.0000,  0.0000, -0.0122,  0.0000],
		 [ 0.0022,  0.0000,  0.0000, -0.0000],
		 [-0.0000,  0.0179,  0.0000, -0.0000],
		 [ 0.0000, -0.0000,  0.0000,  0.0134]]], 4)


def test_deep_lift_shap_softmax_axes_agree(X, references, device):
	# Normalizing over the length axis is reachable as both -1 and 2, and the
	# rule resolves the negative index rather than treating them differently.
	torch.manual_seed(0)
	neg = ConvSoftmax(dim=-1)

	torch.manual_seed(0)
	pos = ConvSoftmax(dim=2)

	X_attr0 = deep_lift_shap(neg, X, references=references, device=device,
		random_state=0)
	X_attr1 = deep_lift_shap(pos, X, references=references, device=device,
		random_state=0)

	assert_array_almost_equal(X_attr0, X_attr1, 4)


def test_deep_lift_shap_softmax_batch_axis_raises(X, references, device):
	"""Softmax over the batch axis is the one case that cannot be supported.

	DeepLIFT stacks each example with its reference along that axis, so
	normalizing over it mixes the two and the rule has no meaning.
	"""

	torch.manual_seed(0)
	model = ConvSoftmax(dim=0)

	assert_raises(ValueError, deep_lift_shap, model, X, references=references,
		device=device, random_state=0)


def test_deep_lift_shap_multihead_attention(X, references, device):
	"""Attention built from hookable modules satisfies summation-to-delta.

	This is the case the LayerNorm, softmax, and bilinear rules exist to make
	possible. Every non-linearity in the block is a module, so each one has a
	rule and the convergence guarantee holds across the whole attention
	block rather than only the parts outside it.
	"""

	torch.manual_seed(0)
	model = MultiHeadAttention()

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		X_attr = deep_lift_shap(model, X[:4], references=references[:4],
			device=device, random_state=0,
			warning_threshold=_rule_threshold(device))

	assert X_attr.shape == X[:4].shape
	assert X_attr.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :4], [
		[[ 0.0273, -0.0000, -0.0000,  0.0234],
		 [ 0.0000,  0.0000,  0.0085,  0.0000],
		 [-0.0000,  0.0000,  0.0000, -0.0000],
		 [-0.0000, -0.0267,  0.0000, -0.0000]],

		[[ 0.0000, -0.0000, -0.0519,  0.0000],
		 [ 0.0453,  0.0000, -0.0000, -0.0000],
		 [-0.0000,  0.0841,  0.0000, -0.0000],
		 [-0.0000,  0.0000,  0.0000, -0.0091]]], 4)


def test_deep_lift_shap_multihead_attention_batch_size(X, references, device):
	torch.manual_seed(0)
	model = MultiHeadAttention()

	X, references = X[:4], references[:4]

	X_attr0 = deep_lift_shap(model, X, references=references, device=device,
		random_state=0)
	X_attr1 = deep_lift_shap(model, X, references=references, batch_size=1,
		device=device, random_state=0)
	X_attr2 = deep_lift_shap(model, X, references=references, batch_size=100000,
		device=device, random_state=0)

	# Attention mixes across the length axis but never across the batch, so
	# the result must not depend on how the pairs are grouped into batches.
	assert_array_almost_equal(X_attr0, X_attr1, 4)
	assert_array_almost_equal(X_attr0, X_attr2, 4)


def test_deep_lift_shap_multihead_attention_independence(X, references, device):
	torch.manual_seed(0)
	model = MultiHeadAttention()

	X_attr = deep_lift_shap(model, X[:4], references=references[:4],
		device=device, random_state=0)
	X_attr0 = deep_lift_shap(model, X[0:1], references=references[0:1],
		device=device, random_state=0)
	X_attr2 = deep_lift_shap(model, X[2:4], references=references[2:4],
		device=device, random_state=0)

	assert_array_almost_equal(X_attr[0:1], X_attr0, 4)
	assert_array_almost_equal(X_attr[2:4], X_attr2, 4)


def test_deep_lift_shap_transformer_functional_ops(X, references, device):
	"""torch's own TransformerEncoderLayer is not fully attributable.

	Its LayerNorms are modules and do get a rule, but MultiheadAttention
	computes its softmax and both of its matmuls functionally, so there is no
	module for a rule to attach to and those steps are silently treated as
	linear. The convergence delta is the symptom, and it is asserted here so
	the limitation is recorded rather than discovered. `MultiHeadAttention`
	is the supported way to write the same block; if functional ops ever gain
	support, this is the test to update.
	"""

	torch.manual_seed(0)
	model = Transformer()

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		assert_raises(RuntimeWarning, deep_lift_shap, model, X[:4],
			references=references[:4], device=device, random_state=0,
			warning_threshold=_rule_threshold(device))


###
# Transformer blocks: the arrangement a user actually stacks, rather than the
# attention operation on its own. Pre-norm and post-norm put the LayerNorm on
# opposite sides of the residual add, a stack chains every rule through the
# block below it, and the feedforward puts a GELU/SiLU between two attention
# layers, so each is a different graph for the rules to work through.
###


TRANSFORMER_BLOCKS = [
	{},
	{"pre_norm": True},
	{"activation": "silu"},
	{"pre_norm": True, "activation": "silu"},
	{"n_blocks": 2},
	{"pre_norm": True, "n_blocks": 2},
	{"positional": True},
	{"pre_norm": True, "n_blocks": 2, "positional": True,
		"activation": "silu"},
]

TRANSFORMER_IDS = ["post_norm_gelu", "pre_norm_gelu", "post_norm_silu",
	"pre_norm_silu", "post_norm_stacked", "pre_norm_stacked",
	"learned_positional", "stacked_pre_norm_positional_silu"]

TRANSFORMER_PARAMS = pytest.mark.parametrize("kwargs,name",
	list(zip(TRANSFORMER_BLOCKS, TRANSFORMER_IDS)), ids=TRANSFORMER_IDS)

# First two examples, first four positions.
TRANSFORMER_REGRESSION = {
	"post_norm_gelu": [
		[[ 0.0282,  0.0000,  0.0000,  0.0244],
		 [ 0.0000,  0.0000, -0.0739, -0.0000],
		 [-0.0000, -0.0000, -0.0000, -0.0000],
		 [ 0.0000,  0.0276, -0.0000, -0.0000]],

		[[ 0.0000, -0.0000,  0.0647,  0.0000],
		 [ 0.0221, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0705, -0.0000, -0.0000],
		 [ 0.0000,  0.0000, -0.0000, -0.0141]]],
	"pre_norm_gelu": [
		[[ 0.0118, -0.0000,  0.0000,  0.0155],
		 [-0.0000,  0.0000, -0.0267, -0.0000],
		 [-0.0000, -0.0000, -0.0000, -0.0000],
		 [ 0.0000,  0.0029, -0.0000, -0.0000]],

		[[ 0.0000, -0.0000,  0.0323,  0.0000],
		 [ 0.0043,  0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0123, -0.0000, -0.0000],
		 [ 0.0000,  0.0000, -0.0000, -0.0050]]],
	"post_norm_silu": [
		[[ 0.0277,  0.0000,  0.0000,  0.0229],
		 [ 0.0000,  0.0000, -0.0752, -0.0000],
		 [-0.0000, -0.0000, -0.0000, -0.0000],
		 [ 0.0000,  0.0286, -0.0000, -0.0000]],

		[[ 0.0000, -0.0000,  0.0639,  0.0000],
		 [ 0.0221, -0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0727, -0.0000, -0.0000],
		 [ 0.0000,  0.0000, -0.0000, -0.0140]]],
	"pre_norm_silu": [
		[[ 0.0111,  0.0000,  0.0000,  0.0137],
		 [-0.0000,  0.0000, -0.0276, -0.0000],
		 [-0.0000, -0.0000, -0.0000, -0.0000],
		 [ 0.0000,  0.0035, -0.0000, -0.0000]],

		[[ 0.0000, -0.0000,  0.0312,  0.0000],
		 [ 0.0036,  0.0000, -0.0000, -0.0000],
		 [-0.0000, -0.0134, -0.0000, -0.0000],
		 [ 0.0000,  0.0000, -0.0000, -0.0047]]],
	"post_norm_stacked": [
		[[ 0.0025, -0.0000, -0.0000, -0.0671],
		 [ 0.0000,  0.0000,  0.0550,  0.0000],
		 [-0.0000, -0.0000,  0.0000,  0.0000],
		 [ 0.0000, -0.0062,  0.0000,  0.0000]],

		[[ 0.0000,  0.0000, -0.0695, -0.0000],
		 [ 0.0122,  0.0000,  0.0000,  0.0000],
		 [-0.0000, -0.0094,  0.0000,  0.0000],
		 [ 0.0000, -0.0000,  0.0000,  0.0619]]],
	"pre_norm_stacked": [
		[[ 0.0007,  0.0000, -0.0000, -0.0255],
		 [ 0.0000,  0.0000,  0.0244,  0.0000],
		 [-0.0000, -0.0000,  0.0000,  0.0000],
		 [ 0.0000,  0.0017,  0.0000,  0.0000]],

		[[ 0.0000, -0.0000, -0.0332, -0.0000],
		 [ 0.0032, -0.0000,  0.0000,  0.0000],
		 [-0.0000, -0.0141,  0.0000,  0.0000],
		 [ 0.0000,  0.0000,  0.0000,  0.0285]]],
	"learned_positional": [
		[[ 0.0335,  0.0000,  0.0000, -0.0149],
		 [ 0.0000, -0.0000, -0.0286, -0.0000],
		 [-0.0000,  0.0000,  0.0000,  0.0000],
		 [ 0.0000, -0.0408, -0.0000,  0.0000]],

		[[ 0.0000,  0.0000,  0.0294, -0.0000],
		 [ 0.0298,  0.0000, -0.0000, -0.0000],
		 [-0.0000,  0.0051, -0.0000,  0.0000],
		 [ 0.0000, -0.0000, -0.0000,  0.0150]]],
	"stacked_pre_norm_positional_silu": [
		[[ 0.0132,  0.0000,  0.0000,  0.0200],
		 [-0.0000, -0.0000,  0.0070, -0.0000],
		 [-0.0000,  0.0000, -0.0000, -0.0000],
		 [ 0.0000, -0.0087, -0.0000,  0.0000]],

		[[ 0.0000,  0.0000,  0.0044,  0.0000],
		 [ 0.0036, -0.0000,  0.0000, -0.0000],
		 [-0.0000,  0.0211, -0.0000, -0.0000],
		 [ 0.0000, -0.0000, -0.0000, -0.0012]]],
}


@TRANSFORMER_PARAMS
def test_deep_lift_shap_transformer_block_convergence(X, references, device,
	kwargs, name):
	"""Every arrangement must satisfy summation-to-delta end to end.

	A single unhooked op anywhere in the stack breaks the convergence
	guarantee for the whole model, so this is the check that the rules
	compose rather than only working one at a time.
	"""

	torch.manual_seed(0)
	model = TransformerBlock(**kwargs)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		X_attr = deep_lift_shap(model, X[:4], references=references[:4],
			device=device, random_state=0,
			warning_threshold=_rule_threshold(device))

	assert X_attr.shape == X[:4].shape
	assert X_attr.dtype == torch.float32


@TRANSFORMER_PARAMS
def test_deep_lift_shap_transformer_block_regression(X, references, device,
	kwargs, name):
	torch.manual_seed(0)
	model = TransformerBlock(**kwargs)

	X_attr = deep_lift_shap(model, X[:4], references=references[:4],
		device=device, random_state=0)

	assert_array_almost_equal(X_attr[:2, :, :4],
		TRANSFORMER_REGRESSION[name], 4)


@TRANSFORMER_PARAMS
def test_deep_lift_shap_transformer_block_batch_size(X, references, device,
	kwargs, name):
	torch.manual_seed(0)
	model = TransformerBlock(**kwargs)

	X, references = X[:4], references[:4]

	X_attr0 = deep_lift_shap(model, X, references=references, device=device,
		random_state=0)
	X_attr1 = deep_lift_shap(model, X, references=references, batch_size=1,
		device=device, random_state=0)
	X_attr2 = deep_lift_shap(model, X, references=references,
		batch_size=100000, device=device, random_state=0)

	# Attention and LayerNorm both mix across the length axis but never across
	# the batch, so the result must not depend on how the example-reference
	# pairs are grouped into batches.
	assert_array_almost_equal(X_attr0, X_attr1, 4)
	assert_array_almost_equal(X_attr0, X_attr2, 4)


@TRANSFORMER_PARAMS
def test_deep_lift_shap_transformer_block_independence(X, references, device,
	kwargs, name):
	torch.manual_seed(0)
	model = TransformerBlock(**kwargs)

	X_attr = deep_lift_shap(model, X[:4], references=references[:4],
		device=device, random_state=0)
	X_attr0 = deep_lift_shap(model, X[0:1], references=references[0:1],
		device=device, random_state=0)
	X_attr2 = deep_lift_shap(model, X[2:4], references=references[2:4],
		device=device, random_state=0)

	assert_array_almost_equal(X_attr[0:1], X_attr0, 4)
	assert_array_almost_equal(X_attr[2:4], X_attr2, 4)


def test_deep_lift_shap_transformer_block_positional_is_linear(X, references,
	device):
	"""A learned positional embedding is an add, so it needs no rule.

	It is a parameter added to the residual stream, which the standard
	gradient chain rule already handles. Registering nothing for it must
	still converge, and it must change the attributions -- otherwise the
	embedding is not reaching the output and the test proves nothing.
	"""

	torch.manual_seed(0)
	model = TransformerBlock(positional=True)

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		X_attr = deep_lift_shap(model, X[:4], references=references[:4],
			device=device, random_state=0,
			warning_threshold=_rule_threshold(device))

	torch.manual_seed(0)
	model_ = TransformerBlock(positional=False)
	X_attr_ = deep_lift_shap(model_, X[:4], references=references[:4],
		device=device, random_state=0)

	assert_raises(AssertionError, assert_array_almost_equal, X_attr, X_attr_, 4)


def test_transformer_block_attention_mask_forward(X, device):
	"""The masked model itself is sound, which the skipped test below is not.

	That test cannot run, so without this one nothing exercises the masked
	constructor at all and the model could stop building unnoticed.
	"""

	mask = torch.triu(torch.full((100, 100), float("-inf")), diagonal=1)

	torch.manual_seed(0)
	model = TransformerBlock(mask=mask).to(device)

	with torch.no_grad():
		y_hat = model(X[:4].to(device))

	assert y_hat.shape == (4, 1)
	assert torch.isfinite(y_hat).all()


ATTENTION_MASKS = [
	("causal", lambda v: torch.triu(torch.full((100, 100), v), diagonal=1)),
	("padding", lambda v: torch.cat([torch.zeros(100, 80),
		torch.full((100, 20), v)], dim=1)),
]
MASK_VALUES = [("neg1e9", -1e9), ("neginf", float("-inf"))]

MASK_PARAMS = pytest.mark.parametrize("mask_fn,value",
	[(fn, v) for _, fn in ATTENTION_MASKS for _, v in MASK_VALUES],
	ids=[f"{mn}_{vn}" for mn, _ in ATTENTION_MASKS for vn, _ in MASK_VALUES])


@MASK_PARAMS
def test_deep_lift_shap_attention_mask(X, references, device, mask_fn, value):
	"""A saturated attention mask must not produce NaN attributions.

	A masked logit is the same large negative number in the example and in the
	reference, so its softmax weight underflows to exactly zero in both and the
	rule's `1 / a_ref` fallback would be infinite. It is multiplied by a
	multiplier that is zero at those positions, so the contribution is zero and
	the fallback only has to be finite; leaving it infinite made the product
	`0 * inf` and every attribution in the batch NaN.

	The convergence check cannot catch this on its own, which is why finiteness
	is asserted separately: `deltas > warning_threshold` is False for NaN.
	"""

	torch.manual_seed(0)
	model = TransformerBlock(mask=mask_fn(value))

	with warnings.catch_warnings():
		warnings.simplefilter("error", category=RuntimeWarning)

		X_attr = deep_lift_shap(model, X[:4], references=references[:4],
			device=device, random_state=0,
			warning_threshold=_rule_threshold(device))

	assert X_attr.shape == X[:4].shape
	assert torch.isfinite(X_attr).all()


@MASK_PARAMS
def test_deep_lift_shap_attention_mask_is_not_vacuous(X, references, device,
	mask_fn, value):
	# The mask has to change the answer, or the test above would pass on a model
	# that ignored it. The two mask values are saturated equivalently, so they
	# must agree with each other while differing from the unmasked model.
	torch.manual_seed(0)
	masked = deep_lift_shap(TransformerBlock(mask=mask_fn(value)), X[:4],
		references=references[:4], device=device, random_state=0)

	torch.manual_seed(0)
	unmasked = deep_lift_shap(TransformerBlock(), X[:4],
		references=references[:4], device=device, random_state=0)

	assert_raises(AssertionError, assert_array_almost_equal, masked, unmasked, 4)


def test_deep_lift_shap_attention_mask_value_independence(X, references, device):
	# -1e9 and -inf both saturate the softmax to the same weights, so the
	# attributions must not depend on which one the model used.
	mask = torch.triu(torch.ones(100, 100), diagonal=1)

	torch.manual_seed(0)
	a = deep_lift_shap(TransformerBlock(mask=mask * -1e9), X[:4],
		references=references[:4], device=device, random_state=0)

	torch.manual_seed(0)
	b = deep_lift_shap(TransformerBlock(mask=torch.where(mask.bool(),
		float("-inf"), 0.0)), X[:4], references=references[:4], device=device,
		random_state=0)

	assert_array_almost_equal(a, b, 4)


def test_deep_lift_shap_fully_masked_row(X, references, device):
	"""A query attending to nothing is only attributable if the model is.

	With a large finite mask the row softmaxes to a uniform distribution and
	nothing underflows, so attribution works. With -inf, torch's own forward
	pass returns NaN for that row, so the model is ill-posed before DeepLIFT is
	involved and propagating NaN is the correct result rather than a defect.
	"""

	mask = torch.zeros(100, 100)
	mask[5, :] = -1e9

	torch.manual_seed(0)
	model = TransformerBlock(mask=mask)
	assert torch.isfinite(model(X[:4])).all()

	X_attr = deep_lift_shap(model, X[:4], references=references[:4],
		device=device, random_state=0)
	assert torch.isfinite(X_attr).all()

	torch.manual_seed(0)
	inf_model = TransformerBlock(mask=torch.where(mask.bool(),
		float("-inf"), 0.0))
	assert not torch.isfinite(inf_model(X[:4])).all()


BILINEAR_MODELS = [(ConvBilinear, None), (ConvBilinearMatmul, None),
	(ConvBilinearEinsum, None), (ConvCustomGate, CustomGate)]
BILINEAR_IDS = ["bilinear_elementwise", "bilinear_matmul", "bilinear_einsum",
	"custom_bilinear"]


@pytest.mark.parametrize("model_cls,op", BILINEAR_MODELS, ids=BILINEAR_IDS)
def test_deep_lift_shap_clears_bilinear_caches(X, device, model_cls, op):
	model, kwargs = _rule_model(model_cls, op)

	deep_lift_shap(model, X[:2], n_shuffles=2, device=device, random_state=0,
		**kwargs)

	# A bilinear op caches both of its operands so the backward rule can reach
	# them. They are as large as the activations and would otherwise stay
	# attached for the life of the model, exactly like `input`/`output`.
	for module in model.modules():
		assert "left" not in module.__dict__
		assert "right" not in module.__dict__


@pytest.mark.parametrize("model_cls,op", BILINEAR_MODELS, ids=BILINEAR_IDS)
def test_deep_lift_shap_clears_bilinear_caches_on_error(X, device, model_cls,
	op):
	model, kwargs = _rule_model(model_cls, op)

	# `target` is out of range for a one-output model, so the attribution loop
	# raises and unwinds through the `finally` block that clears the hooks.
	assert_raises(IndexError, deep_lift_shap, model, X[:2], target=5,
		n_shuffles=2, device=device, random_state=0, **kwargs)

	for module in model.modules():
		assert "left" not in module.__dict__
		assert "right" not in module.__dict__


def test_deep_lift_shap_preserves_user_attributes(X, device):
	torch.manual_seed(0)
	model = AttributeNameConv()

	deep_lift_shap(model, X[:2], n_shuffles=2, device=device, random_state=0)

	# Hooks are only registered on non-linearities, but the caches are cleared
	# across every module, so attributes that merely share a name with them
	# must survive the call.
	assert model.input == "sequence"
	assert model.output == 1
	assert model.conv.input == "kernel"
	assert model.conv.output == "logits"
	assert model.left == "5-prime"
	assert model.right == "3-prime"
	assert model.conv.left == "upstream"
	assert model.conv.right == "downstream"


###
# The hook switch in `_deep_lift_utils`, which lets a rule re-run its own
# module without the forward hooks overwriting the activations it is reading.
###


@pytest.fixture(autouse=True)
def _hook_state():
	"""Leave the global switch as it was found, whatever a test does to it."""

	previous = _HookState.enabled
	yield
	_HookState.enabled = previous


###


def test_hooks_enabled_by_default():
	assert _HookState.enabled is True
	assert _hooks_disabled() is False


def test_disable_hooks_context():
	assert _hooks_disabled() is False

	with _disable_hooks():
		assert _hooks_disabled() is True

	assert _hooks_disabled() is False


def test_disable_hooks_restores_on_exception():
	# The restore lives in a `finally`, so a hook that raises partway through
	# must not leave every later attribution silently un-hooked.
	def _raise():
		with _disable_hooks():
			raise ValueError("boom")

	assert_raises(ValueError, _raise)
	assert _hooks_disabled() is False


def test_disable_hooks_nested():
	# The context restores the previous value rather than unconditionally
	# re-enabling, so an inner exit must not re-enable hooks for an outer
	# block that is still disabled.
	with _disable_hooks():
		assert _hooks_disabled() is True

		with _disable_hooks():
			assert _hooks_disabled() is True

		assert _hooks_disabled() is True

	assert _hooks_disabled() is False


def test_disable_hooks_restores_disabled_state():
	_HookState.enabled = False

	with _disable_hooks():
		assert _hooks_disabled() is True

	assert _hooks_disabled() is True


###


def test_disable_hooks_stops_activation_caching():
	module = torch.nn.ReLU()
	X = torch.randn(2, 4)

	with _disable_hooks():
		_fp_hook(module, (X,))
		_f_hook(module, (X,), X)

	# `integrated_gradients_op` re-runs the module from inside its own backward
	# hook. If the forward hooks still fired there they would overwrite the
	# activations the rule is in the middle of reading.
	assert "input" not in module.__dict__
	assert "output" not in module.__dict__

	_fp_hook(module, (X,))
	_f_hook(module, (X,), X)

	assert "input" in module.__dict__
	assert "output" in module.__dict__


def test_disable_hooks_stops_backward_rule():
	module = torch.nn.ReLU()
	module._NON_LINEAR_OPS = {torch.nn.ReLU: lambda *args: "called"}

	with _disable_hooks():
		assert _b_hook(module, None, None) is None

	# Returning None leaves the gradient untouched, which is what a disabled
	# hook has to do; outside the block the registered rule runs again.
	assert _b_hook(module, None, None) == "called"


def test_disable_hooks_stops_bilinear_caching():
	op = BilinearOp("...,...->...")
	op._NON_LINEAR_OPS = {BilinearOp: _bilinear}

	left, right = torch.randn(2, 4), torch.randn(2, 4)
	op(left, right)

	assert torch.equal(op.left, left)
	assert torch.equal(op.right, right)

	other = torch.ones(2, 4)
	with _disable_hooks():
		op(other, other)

	# The operands cached by the original forward pass must survive a
	# re-entrant call, because `_bilinear` reads them after it returns.
	assert torch.equal(op.left, left)
	assert torch.equal(op.right, right)


def test_bilinear_does_not_cache_without_hooks():
	op = BilinearOp("...,...->...")
	left, right = torch.randn(2, 4), torch.randn(2, 4)

	out = op(left, right)

	# Outside an attribution call there is no `_NON_LINEAR_OPS`, so the op is
	# a plain elementwise product and caches nothing.
	assert torch.equal(out, left * right)
	assert "left" not in op.__dict__
	assert "right" not in op.__dict__
