# test_design.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import pytest

from tangermeme.utils import random_one_hot
from tangermeme.ersatz import substitute

from tangermeme.predict import predict

from tangermeme.design import screen
from tangermeme.design import greedy_substitution
from tangermeme.design import greedy_marginalize

from tangermeme.io import read_meme
from tangermeme.utils import characters

from .toy_models import SumModel
from .toy_models import SmallDeepSEA

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)


@pytest.fixture
def X():
	return random_one_hot((1, 4, 100), random_state=0)


@pytest.fixture
def X_marg():
	return random_one_hot((5, 4, 100), random_state=0)


@pytest.fixture
def motifs():
	return [characters(m, force=True) for m in 
		read_meme("tests/data/test.meme").values()]


###


def test_screen():
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [10]

	X_hat = screen(model, (4, 100), y, device='cpu', max_iter=1, random_state=0)

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

	y_hat = predict(model, X_hat, device='cpu')

	assert_array_almost_equal(y_hat, [[-0.054611]])

	##

	X_hat = screen(model, (4, 100), y, device='cpu', max_iter=10, random_state=0)

	assert X_hat.shape == (1, 4, 100)
	assert X_hat.dtype == torch.int8
	assert all(torch.unique(X_hat) == torch.tensor([0, 1], dtype=torch.int8))

	assert_array_almost_equal(X_hat[:, :, :20], [
		[[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
         [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1]]
	])

	y_hat = predict(model, X_hat, device='cpu')

	assert_array_almost_equal(y_hat, [[-0.03848]])


def test_screen_summodel():
	torch.manual_seed(0)
	model = SumModel()
	y = [10, 0, 0, 0]

	X_hat = screen(model, (4, 10), y, device='cpu', max_iter=25, random_state=0)

	assert X_hat.shape == (1, 4, 10)
	assert X_hat.dtype == torch.int8
	assert all(torch.unique(X_hat) == torch.tensor([0, 1], dtype=torch.int8))

	assert_array_almost_equal(X_hat[:, :, :20], [
		[[0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]]
	])

	y_hat = predict(model, X_hat, device='cpu')

	assert_array_almost_equal(y_hat, [[7., 0., 1., 2.]])


def test_screen_n_best():
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [10]
	
	X_hat = screen(model, (4, 100), y, device='cpu', n_best=7, 
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

	y_hat = predict(model, X_hat, device='cpu')

	assert_array_almost_equal(y_hat, [
		[-0.0385],
        [-0.0439],
        [-0.0454],
        [-0.0457],
        [-0.0466],
        [-0.0496],
        [-0.0510]
	], 4)


def test_screen_no_y():
	torch.manual_seed(0)
	model = SmallDeepSEA()

	X_hat = screen(model, (4, 100), device='cpu', random_state=0, max_iter=15)
	y_hat = predict(model, X_hat, device='cpu')

	assert_array_almost_equal(y_hat, [[-0.085115]])


def test_screen_custom_func():
	torch.manual_seed(0)
	def onlyAs(shape, random_state):
		return random_one_hot(shape, probs=[1.0, 0.0, 0.0, 0.0])
	
	model = SmallDeepSEA()
	y = [10]
	
	X_hat = screen(model, (4, 100), y, device='cpu', max_iter=10, 
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

	y_hat = predict(model, X_hat, device='cpu')

	assert_array_almost_equal(y_hat, [[-0.0841]], 4)


def test_screen_custom_kwargs():
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [10]
	
	X_hat = screen(model, (4, 100), y, device='cpu', max_iter=1, 
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

	y_hat = predict(model, X_hat, device='cpu')

	assert_array_almost_equal(y_hat, [[-0.0841]], 4)


def test_screen_raises():
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [[0.5]]
	
	assert_raises(RuntimeError, screen, model, (5, 100), y, device='cpu')
	assert_raises(ValueError, screen, model, (1, 4, 100), y, device='cpu')
	assert_raises(ValueError, screen, model, (100,), y, device='cpu')


###


def test_greedy_substitution(X, motifs):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [[10]]

	X_hat = greedy_substitution(model, X, y, motifs, device='cpu', max_iter=1)

	assert X_hat.shape == (1, 4, 100)
	assert X_hat.dtype == torch.int8
	assert X_hat.sum(dim=1).min() == 1
	assert all(torch.unique(X_hat) == torch.tensor([0, 1], dtype=torch.int8))

	y_hat = predict(model, X_hat, device='cpu')

	assert_array_almost_equal(y_hat, [[-0.058557]])

	##

	X_hat = greedy_substitution(model, X, y, motifs, device='cpu', max_iter=4)

	assert X_hat.shape == (1, 4, 100)
	assert X_hat.dtype == torch.int8
	assert X_hat.sum(dim=1).min() == 1
	assert all(torch.unique(X_hat) == torch.tensor([0, 1], dtype=torch.int8))
	
	y_hat = predict(model, X_hat, device='cpu')

	assert_array_almost_equal(y_hat, [[-0.018134]])


def test_greedy_nucleotide_substitution(X):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [[10]]

	X_hat = greedy_substitution(model, X, y, device='cpu', max_iter=1)

	assert X_hat.shape == (1, 4, 100)
	assert X_hat.dtype == torch.int8
	assert X_hat.sum(dim=1).min() == 1
	assert all(torch.unique(X_hat) == torch.tensor([0, 1], dtype=torch.int8))
	assert abs(X_hat - X).sum() == 2

	y_hat = predict(model, X_hat, device='cpu')

	assert_array_almost_equal(y_hat, [[-0.076893]])

	##

	X_hat = greedy_substitution(model, X, y, device='cpu', max_iter=4)

	assert X_hat.shape == (1, 4, 100)
	assert X_hat.dtype == torch.int8
	assert X_hat.sum(dim=1).min() == 1
	assert all(torch.unique(X_hat) == torch.tensor([0, 1], dtype=torch.int8))
	assert abs(X_hat - X).sum() == 8
	
	y_hat = predict(model, X_hat, device='cpu')

	assert_array_almost_equal(y_hat, [[-0.057514]])


def test_greedy_nucleotide_substitution_only_As(X):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	motifs = ['A']
	y = [[10]]

	X_hat = greedy_substitution(model, X, y, motifs, device='cpu', max_iter=1, 
		reverse_complement=False)
	assert X_hat.sum(dim=(0, -1))[0] == X.sum(dim=(0, -1))[0] + 1

	X_hat = greedy_substitution(model, X, y, motifs, device='cpu', max_iter=4, 
		reverse_complement=False)
	assert X_hat.sum(dim=(0, -1))[0] == X.sum(dim=(0, -1))[0] + 4


def test_greedy_substitution_input_mask(X):
	torch.manual_seed(0)
	input_mask = torch.zeros(100, dtype=bool)
	input_mask[54:58] = True

	model = SmallDeepSEA()
	y = [[10]]

	X_hat = greedy_substitution(model, X, y, device='cpu', input_mask=input_mask,
		max_iter=4)

	assert_array_almost_equal(X[:, :, :54], X_hat[:, :, :54])
	assert_array_almost_equal(X[:, :, 58:], X_hat[:, :, 58:])
	assert abs(X[:, :, 54:58] - X_hat[:, :, 54:58]).sum() == 6


def test_greedy_substitution_output_mask(X):
	torch.manual_seed(0)
	output_mask = torch.zeros(4, dtype=bool)
	output_mask[3] = True

	model = SumModel()
	y = [[10, 10000, -26, 100]]

	X_hat = greedy_substitution(model, X, y, device='cpu', output_mask=output_mask,
		max_iter=101)

	assert all(X_hat.sum(dim=(0, 2)) == torch.tensor([0, 0, 0, 100]))


def test_greedy_substitution_reverse_complement(X):
	torch.manual_seed(0)
	output_mask = torch.zeros(4, dtype=bool)
	output_mask[3] = True

	model = SmallDeepSEA()
	y = [[10]]

	X_hat = greedy_substitution(model, X, y, ['A', 'C'], device='cpu', max_iter=101,
		reverse_complement=False)

	n_count = X.sum(dim=(0, 2))
	n_hat_count = X_hat.sum(dim=(0, 2))
	
	assert n_count[0] < n_hat_count[0]
	assert n_count[1] < n_hat_count[1]
	assert n_count[2] > n_hat_count[2]
	assert n_count[3] > n_hat_count[3]


def test_greedy_substitution_no_y(X):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [[10]]

	X_hat = greedy_substitution(model, X, device='cpu', max_iter=15)
	y_hat = predict(model, X_hat, device='cpu')

	assert_array_almost_equal(y_hat, [[-0.085115]])


###


def test_greedy_marginalize(X_marg, motifs):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [[10.0]]

	X_hat = greedy_marginalize(model, X_marg, y, motifs, device='cpu', max_iter=1)

	assert X_hat.shape == (4, 12)
	assert X_hat.dtype == torch.int8
	assert X_hat.sum(dim=0).min() == 1
	assert all(torch.unique(X_hat) == torch.tensor([0, 1], dtype=torch.int8))


