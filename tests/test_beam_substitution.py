# test_beam_substitution.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import pytest

from tangermeme.utils import random_one_hot
from tangermeme.predict import predict
from tangermeme.design import greedy_substitution
from tangermeme.design import beam_substitution
from tangermeme.io import read_meme
from tangermeme.utils import characters
from .toy_models import SmallDeepSEA
from numpy.testing import assert_array_almost_equal


torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)


@pytest.fixture
def X():
	return random_one_hot((1, 4, 100), random_state=0)


@pytest.fixture
def motifs():
	return [characters(m, force=True) for m in 
		read_meme("tests/data/test.meme").values()]


###


def test_beam_substitution(X, motifs, device):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [[10]]

	X_hat = beam_substitution(model, X, y, motifs, beam_size=4, device=device,
		max_iter=1)

	assert X_hat.shape == (1, 4, 100)
	assert X_hat.dtype == torch.int8
	assert X_hat.sum(dim=1).min() == 1
	assert all(torch.unique(X_hat) == torch.tensor([0, 1], dtype=torch.int8))

	y_hat = predict(model, X_hat, device=device)

	assert_array_almost_equal(y_hat, [[-0.058557]])

	##

	X_hat = beam_substitution(model, X, y, motifs, beam_size=4, device=device,
		max_iter=4)

	assert X_hat.shape == (1, 4, 100)
	assert X_hat.dtype == torch.int8
	assert all(torch.unique(X_hat) == torch.tensor([0, 1], dtype=torch.int8))

	y_hat = predict(model, X_hat, device=device)

	assert_array_almost_equal(y_hat, [[-0.016707]])


def test_beam_substitution_equals_greedy(X, motifs, device):
	# A beam of width 1 is exactly the greedy substitution search.
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [[10]]

	X_greedy = greedy_substitution(model, X, y, motifs, device=device,
		max_iter=4)
	X_beam = beam_substitution(model, X, y, motifs, beam_size=1, device=device,
		max_iter=4)

	assert_array_almost_equal(X_beam, X_greedy)


def test_beam_substitution_n_best(X, motifs, device):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [[10]]

	X_hat = beam_substitution(model, X, y, motifs, beam_size=4, n_best=3,
		device=device, max_iter=4)

	assert X_hat.shape == (3, 4, 100)
	assert X_hat.dtype == torch.int8

	# Returned sequences are distinct and ranked from lowest to highest loss.
	y_hat = predict(model, X_hat, device=device)
	losses = ((torch.tensor([[10.0]]) - y_hat) ** 2).sum(dim=1)
	assert (losses[:-1] <= losses[1:]).all()
	assert len(torch.unique(X_hat, dim=0)) == 3


def test_beam_substitution_max_iter_default(X, motifs, device):
	# max_iter=-1 means no iteration limit; tol stops the search instead.
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [[10]]

	X_hat = beam_substitution(model, X, y, motifs, beam_size=4, device=device)

	assert X_hat.shape == (1, 4, 100)
	assert X_hat.dtype == torch.int8


def test_beam_substitution_not_worse_than_greedy(X, motifs, device):
	# The core promise: a wider beam carries the single best edit forward at
	# every round, so its final loss can never exceed that of beam_size=1
	# (which is greedy_substitution).
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = torch.tensor([[10.0]])

	X_greedy = beam_substitution(model, X, y, motifs, beam_size=1, device=device,
		max_iter=5)
	X_beam = beam_substitution(model, X, y, motifs, beam_size=5, device=device,
		max_iter=5)

	loss_greedy = ((y - predict(model, X_greedy, device=device)) ** 2).sum()
	loss_beam = ((y - predict(model, X_beam, device=device)) ** 2).sum()

	assert loss_beam <= loss_greedy + 1e-3


def test_beam_substitution_no_y(X, device):
	# With y=None the objective is to maximize the model output, so the design
	# must not decrease it relative to the starting sequence.
	torch.manual_seed(0)
	model = SmallDeepSEA()

	y_before = predict(model, X, device=device)
	X_hat = beam_substitution(model, X, beam_size=4, device=device, max_iter=4)
	y_after = predict(model, X_hat, device=device)

	assert X_hat.shape == (1, 4, 100)
	assert y_after.sum() >= y_before.sum()


def test_beam_substitution_input_mask(X, device):
	# Only positions inside the mask may change (single-nucleotide edits).
	torch.manual_seed(0)
	input_mask = torch.zeros(100, dtype=bool)
	input_mask[54:58] = True

	model = SmallDeepSEA()
	y = [[10]]

	X_hat = beam_substitution(model, X, y, beam_size=4, device=device,
		input_mask=input_mask, max_iter=4)

	assert_array_almost_equal(X[:, :, :54], X_hat[:, :, :54])
	assert_array_almost_equal(X[:, :, 58:], X_hat[:, :, 58:])
