# test_screen.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from tangermeme.utils import random_one_hot
from tangermeme.predict import predict
from tangermeme.design import screen
from ..toy_models import SumModel
from ..toy_models import SmallDeepSEA
from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)


###


def test_screen(device):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [10]

	X_hat = screen(model, (4, 100), y, device=device, max_iter=1, random_state=0)

	assert X_hat.shape == (1, 4, 100)
	assert X_hat.dtype == torch.int8
	assert X_hat.sum(dim=1).min() == 1
	assert all(torch.unique(X_hat) == torch.tensor([0, 1], dtype=torch.int8))

	assert_array_almost_equal(X_hat[:, :, :20], [
		[[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
         [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
         [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
         [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
	])

	y_hat = predict(model, X_hat, device=device)

	assert_array_almost_equal(y_hat, [[-0.054611]])

	##

	X_hat = screen(model, (4, 100), y, device=device, max_iter=10, random_state=0)

	assert X_hat.shape == (1, 4, 100)
	assert X_hat.dtype == torch.int8
	assert all(torch.unique(X_hat) == torch.tensor([0, 1], dtype=torch.int8))

	assert_array_almost_equal(X_hat[:, :, :20], [
		[[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
         [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1]]
	])

	y_hat = predict(model, X_hat, device=device)

	assert_array_almost_equal(y_hat, [[-0.03848]])


def test_screen_summodel(device):
	torch.manual_seed(0)
	model = SumModel()
	y = [10, 0, 0, 0]

	X_hat = screen(model, (4, 10), y, device=device, max_iter=25, random_state=0)

	assert X_hat.shape == (1, 4, 10)
	assert X_hat.dtype == torch.int8
	assert all(torch.unique(X_hat) == torch.tensor([0, 1], dtype=torch.int8))

	assert_array_almost_equal(X_hat[:, :, :20], [
		[[0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]]
	])

	y_hat = predict(model, X_hat, device=device)

	assert_array_almost_equal(y_hat, [[7., 0., 1., 2.]])


def test_screen_n_best(device):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [10]
	
	X_hat = screen(model, (4, 100), y, device=device, n_best=7, 
		max_iter=10, random_state=0)

	assert X_hat.shape == (7, 4, 100)
	assert X_hat.dtype == torch.int8
	assert all(torch.unique(X_hat) == torch.tensor([0, 1], dtype=torch.int8))

	assert_array_almost_equal(X_hat[:3, :, :10], [
		[[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
         [1, 0, 0, 0, 1, 0, 0, 0, 1, 0]],

        [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 1, 0, 1, 1, 0, 0, 1, 1]],

        [[1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
         [0, 0, 1, 1, 1, 0, 0, 0, 1, 0]]])

	y_hat = predict(model, X_hat, device=device)

	assert_array_almost_equal(y_hat, [
		[-0.0385],
        [-0.0439],
        [-0.0454],
        [-0.0457],
        [-0.0466],
        [-0.0496],
        [-0.0510]
	], 4)


def test_screen_no_y(device):
	torch.manual_seed(0)
	model = SmallDeepSEA()

	X_hat = screen(model, (4, 100), device=device, random_state=0, max_iter=15)
	y_hat = predict(model, X_hat, device=device)

	assert_array_almost_equal(y_hat, [[-0.085115]])


def test_screen_custom_func(device):
	torch.manual_seed(0)
	def onlyAs(shape, random_state):
		return random_one_hot(shape, probs=[1.0, 0.0, 0.0, 0.0])
	
	model = SmallDeepSEA()
	y = [10]
	
	X_hat = screen(model, (4, 100), y, device=device, max_iter=10, 
		random_state=0, func=onlyAs)

	assert X_hat.shape == (1, 4, 100)
	assert X_hat.dtype == torch.int8
	assert all(torch.unique(X_hat) == torch.tensor([0, 1], dtype=torch.int8))

	assert_array_almost_equal(X_hat[:, :, :20], [
		[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
	])

	y_hat = predict(model, X_hat, device=device)

	assert_array_almost_equal(y_hat, [[-0.0841]], 4)


def test_screen_custom_kwargs(device):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [10]
	
	X_hat = screen(model, (4, 100), y, device=device, max_iter=1, 
		additional_func_kwargs={'probs': [1, 0, 0, 0]})

	assert X_hat.shape == (1, 4, 100)
	assert X_hat.dtype == torch.int8
	assert all(torch.unique(X_hat) == torch.tensor([0, 1], dtype=torch.int8))

	assert_array_almost_equal(X_hat[:, :, :20], [
		[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
	])

	y_hat = predict(model, X_hat, device=device)

	assert_array_almost_equal(y_hat, [[-0.0841]], 4)


def test_screen_raises(device):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [[0.5]]
	
	assert_raises(ValueError, screen, model, (1, 4, 100), y, device=device)
	assert_raises(ValueError, screen, model, (100,), y, device=device)


def test_screen_random_state_reproducibility(device):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [10]

	X_hat0 = screen(model, (4, 100), y, device=device, max_iter=1,
		random_state=0)
	X_hat1 = screen(model, (4, 100), y, device=device, max_iter=1,
		random_state=0)

	assert_array_almost_equal(X_hat0, X_hat1)


def test_screen_packed(device):
	# Drawing packed sequences gives the screen of their one-hot encoding,
	# packed.
	import functools
	from tangermeme.utils import PackedSequence

	torch.manual_seed(0)
	model = SmallDeepSEA()
	X_hat = screen(model, (4, 100), [10], device=device, max_iter=2, n_best=3,
		random_state=0)
	X_hat_packed = screen(model, (4, 100), [10], device=device, max_iter=2,
		n_best=3, random_state=0, func=functools.partial(random_one_hot,
		return_packed=True))

	assert isinstance(X_hat_packed, PackedSequence)
	assert torch.equal(X_hat_packed.to_one_hot(), X_hat)
