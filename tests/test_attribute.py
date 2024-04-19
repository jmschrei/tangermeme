# test_attribute.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import pytest

from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot

from tangermeme.ersatz import substitute
from tangermeme.ersatz import shuffle

from tangermeme.attribute import hypothetical_attributions
from tangermeme.attribute import deep_lift_shap
from tangermeme.attribute import _captum_deep_lift_shap

from .toy_models import SumModel
from .toy_models import FlattenDense
from .toy_models import ConvDense

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


torch.manual_seed(0)
torch.use_deterministic_algorithms(True)



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


def test_deep_lift_shap(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	X_attr = deep_lift_shap(model, X, device='cpu', random_state=0)

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


def test_deep_lift_shap_summodel(X):
	torch.manual_seed(0)
	model = SumModel()

	X_attr = deep_lift_shap(model, X, device='cpu', random_state=0)

	assert X_attr.shape == X.shape
	assert X.dtype == torch.float32
	assert_array_almost_equal(X_attr[:2, :, :10], [
		[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

        [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]], 4)


def test_deep_lift_shap_convdense_dense_wrapper(X):
	class Wrapper(torch.nn.Module):
		def __init__(self, model):
			super().__init__()
			self.model = model

		def forward(self, X, alpha=0.01):
			return self.model(X, alpha)[1]

	torch.manual_seed(0)
	model = Wrapper(ConvDense(n_outputs=1))

	X_attr = deep_lift_shap(model, X, device='cpu', random_state=0)

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
	class Wrapper(torch.nn.Module):
		def __init__(self, model):
			super().__init__()
			self.model = model

		def forward(self, X, alpha=0.01):
			return self.model(X, alpha)[0].sum(dim=(-1, -2)).unsqueeze(-1)

	torch.manual_seed(0)
	model = Wrapper(ConvDense(n_outputs=1))

	X_attr = deep_lift_shap(model, X, device='cpu', random_state=0)

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


def test_deep_lift_shap_hypothetical(X):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	X_attr = deep_lift_shap(model, X, hypothetical=True, device='cpu', 
		random_state=0)

	assert X_attr.shape == X.shape
	assert X.dtype == torch.float32
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


###


def test_captum_deep_lift_shap(X, references):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	X_attr0 = deep_lift_shap(model, X, references=references, device='cpu', 
		random_state=0)
	X_attr1 = _captum_deep_lift_shap(model, X, references=references,
		device='cpu', random_state=0)

	assert X_attr0.shape == X_attr1.shape
	assert X_attr0.dtype == X_attr1.dtype
	assert_array_almost_equal(X_attr0, X_attr1)


def test_captum_deep_lift_shap_batch_size(X, references):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	X_attr0 = deep_lift_shap(model, X, references=references, batch_size=1, 
		device='cpu', random_state=0)
	X_attr1 = _captum_deep_lift_shap(model, X, references=references, 
		batch_size=1, device='cpu', random_state=0)

	assert X_attr0.shape == X_attr1.shape
	assert X_attr0.dtype == X_attr1.dtype
	assert_array_almost_equal(X_attr0, X_attr1)


def test_captum_deep_lift_shap_n_shuffles(X, references):
	torch.manual_seed(0)
	model = FlattenDense(n_outputs=1)

	X_attr0 = deep_lift_shap(model, X, references=references, n_shuffles=5, 
		batch_size=1, device='cpu', random_state=0)
	X_attr1 = _captum_deep_lift_shap(model, X, references=references,
		n_shuffles=5, batch_size=1, device='cpu', random_state=0)

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
