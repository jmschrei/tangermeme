# test_saturation_mutagenesis_batches.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

# What the model sees from `saturation_mutagenesis`, which builds each batch of
# edited sequences on the device, and what it is left as afterwards.

import torch
import pytest

from tangermeme.predict import predict
from tangermeme.utils import random_one_hot

from tangermeme.saturation_mutagenesis import saturation_mutagenesis

from .toy_models import FlattenDense


@pytest.fixture
def X0():
	return random_one_hot((2, 4, 20), random_state=0).float()


class _Recorder(torch.nn.Module):
	"""Records the rows, device and dtype of every input to forward."""

	def __init__(self, seq_len=20, dtype=torch.float32):
		super(_Recorder, self).__init__()
		self.model = FlattenDense(seq_len=seq_len, n_outputs=2).type(dtype)
		self.calls = []

	def forward(self, X, *args):
		self.calls.append([(x.shape[0], x.device.type, x.dtype)
			for x in (X,) + args])
		return self.model(X, *args)


###


def test_saturation_mutagenesis_preserves_model_state(X0, device):
	torch.manual_seed(0)
	model = FlattenDense(seq_len=20, n_outputs=1)
	model.train()

	saturation_mutagenesis(model, X0, device=device)

	assert model.training
	assert next(model.parameters()).device.type == 'cpu'


def test_saturation_mutagenesis_preserves_model_state_raw_args(X0, device):
	torch.manual_seed(0)
	model = FlattenDense(seq_len=20, n_outputs=1)
	model.train()

	saturation_mutagenesis(model, X0, args=(torch.randn(2, 1),), start=3,
		end=9, raw_outputs=True, device=device)

	assert model.training
	assert next(model.parameters()).device.type == 'cpu'


def test_predict_preserves_model_state(X0, device):
	torch.manual_seed(0)
	model = FlattenDense(seq_len=20, n_outputs=1)
	model.train()

	predict(model, X0, device=device)

	assert model.training
	assert next(model.parameters()).device.type == 'cpu'


def test_saturation_mutagenesis_batch_composition(X0, device):
	# Each sequence's 3 * 20 = 60 edits are run in batches of at most
	# batch_size rows, in order, and batches never mix sequences.
	torch.manual_seed(0)
	model = _Recorder()

	saturation_mutagenesis(model, X0, batch_size=7, device=device)

	rows = [call[0][0] for call in model.calls]
	assert rows == [2] + ([7] * 8 + [4]) * 2
	assert all(call[0][1] == torch.device(device).type for call in model.calls)
	assert all(call[0][2] == torch.float32 for call in model.calls)


def test_saturation_mutagenesis_args_cast_to_model_dtype(X0, device):
	# args are repeated for every edit and cast to the model's dtype, as
	# `predict` casts them, including integer arguments.
	torch.manual_seed(0)
	model = _Recorder(dtype=torch.float64)
	alpha = torch.arange(2, dtype=torch.int32).reshape(2, 1)

	y0, y_hat = saturation_mutagenesis(model, X0, args=(alpha,), start=5,
		end=10, batch_size=6, raw_outputs=True, device=device)

	assert y0.dtype == torch.float64 and y_hat.dtype == torch.float64
	for call in model.calls[1:]:
		(n, x_device, x_dtype), (n_a, a_device, a_dtype) = call
		assert n == n_a and n <= 6
		assert x_device == a_device == torch.device(device).type
		assert x_dtype == a_dtype == torch.float64


def test_saturation_mutagenesis_args_follow_their_sequence(X0, device):
	# Each sequence's edits get that sequence's args: with alpha shifting the
	# output, y_hat moves by exactly alpha[i] for sequence i.
	torch.manual_seed(0)
	model = FlattenDense(seq_len=20, n_outputs=1)
	alpha = torch.tensor([[0.0], [5.0]])

	_, y_hat0 = saturation_mutagenesis(model, X0, raw_outputs=True,
		batch_size=7, device=device)
	_, y_hat1 = saturation_mutagenesis(model, X0, args=(alpha,),
		raw_outputs=True, batch_size=7, device=device)

	torch.testing.assert_close(y_hat1 - y_hat0, alpha[:, None, None].expand_as(
		y_hat0), rtol=0, atol=1e-5)
