# test_greedy_marginalize.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import pytest

from tangermeme.utils import random_one_hot
from tangermeme.utils import PackedSequence
from tangermeme.utils import reverse_complement
from tangermeme.design import greedy_marginalize
from tangermeme.ersatz import substitute
from tangermeme.predict import predict
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

	# One iteration tries each motif and reverse complement at position 49
	# only, and keeps the one whose change to the predictions on the
	# backgrounds has the lowest squared error from y.
	candidates = motifs + [reverse_complement(motif) for motif in motifs]

	y_orig = predict(model, X_marg, device=device)
	losses = []
	for motif in candidates:
		y_hat = predict(model, substitute(X_marg, motif, start=49),
			device=device) - y_orig
		losses.append(((y_hat - 10.0) ** 2).mean())

	assert characters(X_hat) == candidates[torch.stack(losses).argmin()]


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


def test_greedy_marginalize_packed_alphabet(X_marg, device):
	# The design comes back labelled with `alphabet`, here an order other
	# than ACGT.
	torch.manual_seed(0)
	model = SmallDeepSEA()
	alphabet = ['C', 'A', 'T', 'G']
	X_packed = PackedSequence.from_one_hot(X_marg, alphabet=alphabet)

	X_hat = greedy_marginalize(model, X_packed, [[10.0]], ["ACGT", "TTT"],
		alphabet=alphabet, device=device, max_iter=1)
	assert X_hat.alphabet == tuple(alphabet)

	X_hat_dense = greedy_marginalize(model, X_packed.to_one_hot(), [[10.0]],
		["ACGT", "TTT"], alphabet=alphabet, device=device, max_iter=1)
	assert torch.equal(X_hat.to_one_hot(), X_hat_dense)
	assert characters(X_hat_dense, alphabet=alphabet) in ["ACGT", "TTT", "AAA"]
