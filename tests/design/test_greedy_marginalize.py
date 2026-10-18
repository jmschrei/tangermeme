# test_greedy_marginalize.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import pytest

from tangermeme.utils import random_one_hot
from tangermeme.utils import PackedSequence
from tangermeme.design import greedy_marginalize
from tangermeme.io import read_meme
from tangermeme.utils import characters
from ..toy_models import SmallDeepSEA


torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)


@pytest.fixture
def X_marg():
	return random_one_hot((5, 4, 100), random_state=0)


@pytest.fixture
def motifs():
	return [characters(m, force=True) for m in 
		read_meme("tests/data/test.meme").values()]


###


def test_greedy_marginalize(X_marg, motifs, device):
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [[10.0]]

	X_hat = greedy_marginalize(model, X_marg, y, motifs, device=device, max_iter=1)

	assert X_hat.shape == (4, 12)
	assert X_hat.dtype == torch.int8
	assert X_hat.sum(dim=0).min() == 1
	assert all(torch.unique(X_hat) == torch.tensor([0, 1], dtype=torch.int8))


def test_greedy_marginalize_packed(X_marg, motifs, device):
	# Packed backgrounds give the design of their one-hot encoding, packed.
	torch.manual_seed(0)
	model = SmallDeepSEA()
	y = [[10.0]]
	X_marg = X_marg.clone()
	X_marg[0, :, 5:9] = 0
	X_marg[-1, :, -3:] = 0

	X_hat = greedy_marginalize(model, X_marg, y, motifs, device=device,
		max_iter=2)
	X_hat_packed = greedy_marginalize(model, PackedSequence.from_one_hot(X_marg),
		y, motifs, device=device, max_iter=2)
	assert isinstance(X_hat_packed, PackedSequence)
	assert torch.equal(X_hat_packed.to_one_hot(), X_hat)
