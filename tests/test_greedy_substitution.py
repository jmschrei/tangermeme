# test_greedy_substitution.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import pytest

from tangermeme.utils import random_one_hot
from tangermeme.predict import predict
from tangermeme.design import greedy_substitution
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
def motifs():
	return [characters(m, force=True) for m in 
		read_meme("tests/data/test.meme").values()]


###


def test_greedy_substitution(X, motifs, device):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [[10]]

	X_hat = greedy_substitution(model, X, y, motifs, device=device, max_iter=1)

	assert X_hat.shape == (1, 4, 100)
	assert X_hat.dtype == torch.int8
	assert X_hat.sum(dim=1).min() == 1
	assert all(torch.unique(X_hat) == torch.tensor([0, 1], dtype=torch.int8))

	y_hat = predict(model, X_hat, device=device)

	assert_array_almost_equal(y_hat, [[-0.058557]])

	##

	X_hat = greedy_substitution(model, X, y, motifs, device=device, max_iter=4)

	assert X_hat.shape == (1, 4, 100)
	assert X_hat.dtype == torch.int8
	assert X_hat.sum(dim=1).min() == 1
	assert all(torch.unique(X_hat) == torch.tensor([0, 1], dtype=torch.int8))
	
	y_hat = predict(model, X_hat, device=device)

	assert_array_almost_equal(y_hat, [[-0.018134]])


def test_greedy_nucleotide_substitution(X, device):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [[10]]

	X_hat = greedy_substitution(model, X, y, device=device, max_iter=1)

	assert X_hat.shape == (1, 4, 100)
	assert X_hat.dtype == torch.int8
	assert X_hat.sum(dim=1).min() == 1
	assert all(torch.unique(X_hat) == torch.tensor([0, 1], dtype=torch.int8))
	assert abs(X_hat - X).sum() == 2

	y_hat = predict(model, X_hat, device=device)

	assert_array_almost_equal(y_hat, [[-0.076893]])

	##

	X_hat = greedy_substitution(model, X, y, device=device, max_iter=4)

	assert X_hat.shape == (1, 4, 100)
	assert X_hat.dtype == torch.int8
	assert X_hat.sum(dim=1).min() == 1
	assert all(torch.unique(X_hat) == torch.tensor([0, 1], dtype=torch.int8))
	assert abs(X_hat - X).sum() == 8
	
	y_hat = predict(model, X_hat, device=device)

	assert_array_almost_equal(y_hat, [[-0.057514]])


def test_greedy_nucleotide_substitution_only_As(X, device):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	motifs = ['A']
	y = [[10]]

	X_hat = greedy_substitution(model, X, y, motifs, device=device, max_iter=1, 
		reverse_complement=False)
	assert X_hat.sum(dim=(0, -1))[0] == X.sum(dim=(0, -1))[0] + 1

	X_hat = greedy_substitution(model, X, y, motifs, device=device, max_iter=4, 
		reverse_complement=False)
	assert X_hat.sum(dim=(0, -1))[0] == X.sum(dim=(0, -1))[0] + 4


def test_greedy_substitution_input_mask(X, device):
	torch.manual_seed(0)
	input_mask = torch.zeros(100, dtype=bool)
	input_mask[54:58] = True

	model = SmallDeepSEA()
	y = [[10]]

	X_hat = greedy_substitution(model, X, y, device=device, input_mask=input_mask,
		max_iter=4)

	assert_array_almost_equal(X[:, :, :54], X_hat[:, :, :54])
	assert_array_almost_equal(X[:, :, 58:], X_hat[:, :, 58:])
	assert abs(X[:, :, 54:58] - X_hat[:, :, 54:58]).sum() == 6


def test_greedy_substitution_output_mask(X, device):
	torch.manual_seed(0)
	output_mask = torch.zeros(4, dtype=bool)
	output_mask[3] = True

	model = SumModel()
	y = [[10, 10000, -26, 100]]

	X_hat = greedy_substitution(model, X, y, device=device, output_mask=output_mask,
		max_iter=101)

	assert all(X_hat.sum(dim=(0, 2)) == torch.tensor([0, 0, 0, 100]))


def test_greedy_substitution_reverse_complement(X, device):
	torch.manual_seed(0)
	output_mask = torch.zeros(4, dtype=bool)
	output_mask[3] = True

	model = SmallDeepSEA()
	y = [[10]]

	X_hat = greedy_substitution(model, X, y, ['A', 'C'], device=device, max_iter=101,
		reverse_complement=False)

	n_count = X.sum(dim=(0, 2))
	n_hat_count = X_hat.sum(dim=(0, 2))
	
	assert n_count[0] < n_hat_count[0]
	assert n_count[1] < n_hat_count[1]
	assert n_count[2] > n_hat_count[2]
	assert n_count[3] > n_hat_count[3]


def test_greedy_substitution_no_y(X, device):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [[10]]

	X_hat = greedy_substitution(model, X, device=device, max_iter=15)
	y_hat = predict(model, X_hat, device=device)

	assert_array_almost_equal(y_hat, [[-0.01224]])


def test_greedy_substitution_max_iter_default(X, motifs, device):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [[10]]

	X_hat = greedy_substitution(model, X, y, motifs, device=device)

	assert_array_almost_equal(X_hat, X)


def test_greedy_substitution_tol_no_improvement_stops(X, motifs, device):
	torch.manual_seed(0)
	model = SmallDeepSEA()

	y_current = predict(model, X, device=device)

	X_hat = greedy_substitution(model, X, y_current, motifs, device=device,
		max_iter=10, tol=1e-3)

	assert_array_almost_equal(X_hat, X)


def test_greedy_substitution_pre_encoded_motif_tensors(X, device):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [[10]]

	motif_tensor = torch.tensor(
		[[1, 0], [0, 1], [0, 0], [0, 0]], dtype=torch.int8
	)

	assert_raises(Exception, greedy_substitution, model, X, y, [motif_tensor],
		device=device, max_iter=1)
