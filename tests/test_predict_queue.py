# test_predict_queue.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

# On CUDA, `predict` queues each batch's copies without waiting for the device
# and reads a batch's outputs a few batches later. These tests check that the
# outputs are still each batch's own, in order, whatever the queue depth and
# whichever copy path an output takes. The models do exact arithmetic on 0/1
# inputs, so the expected values can be computed on the CPU.

import torch
import pytest

import tangermeme.predict as predict_module

from tangermeme.utils import random_one_hot
from tangermeme.predict import predict

from numpy.testing import assert_array_equal


@pytest.fixture
def X():
	return random_one_hot((50, 4, 30), random_state=0).type(torch.float32)


class PersistentSum(torch.nn.Module):
	"""Writes each batch's output into one buffer that it reuses, as a model
	captured in a CUDA graph does."""

	def __init__(self):
		super().__init__()
		self.scale = torch.nn.Parameter(torch.ones(1))
		self.buffer = None

	def forward(self, X):
		y = X.sum(dim=-1) * self.scale
		if self.buffer is None or self.buffer.shape != y.shape:
			self.buffer = torch.empty_like(y)

		self.buffer.copy_(y)
		return self.buffer


class Heads(torch.nn.Module):
	"""Returns a wide head and a narrow head."""

	def __init__(self):
		super().__init__()
		self.scale = torch.nn.Parameter(torch.ones(1))

	def forward(self, X):
		wide = X.reshape(X.shape[0], -1) * self.scale
		return wide, X.sum(dim=(1, 2)) * self.scale


@pytest.mark.parametrize("depth", [0, 1, 2, 100])
def test_predict_queue_persistent_buffer(X, cuda_device, depth, monkeypatch):
	monkeypatch.setattr(predict_module, "_QUEUE_DEPTH", depth)

	y = predict(PersistentSum(), X, batch_size=4, device=cuda_device)

	assert y.device.type == 'cpu'
	assert not y.is_pinned()
	assert_array_equal(y, X.sum(dim=-1))


@pytest.mark.parametrize("pinned_bytes", [0, 400, 2 ** 24])
def test_predict_queue_copy_paths(X, cuda_device, pinned_bytes, monkeypatch):
	# At 400 bytes the wide head (4 x 120 floats per batch) takes the blocking
	# copy and the narrow head the queued one.
	monkeypatch.setattr(predict_module, "_PINNED_BYTES", pinned_bytes)

	y = predict(Heads(), X, batch_size=4, device=cuda_device)

	assert isinstance(y, list) and len(y) == 2
	for y_ in y:
		assert y_.device.type == 'cpu'
		assert not y_.is_pinned()

	assert_array_equal(y[0], X.reshape(X.shape[0], -1))
	assert_array_equal(y[1], X.sum(dim=(1, 2)))


def test_predict_queue_func_cpu(X, cuda_device):
	y = predict(Heads(), X, batch_size=4, device=cuda_device,
		func=lambda y: (y[0].cpu()[:, :3], y[1]))

	assert_array_equal(y[0], X.reshape(X.shape[0], -1)[:, :3])
	assert_array_equal(y[1], X.sum(dim=(1, 2)))


def test_predict_queue_matches_cpu(X, cuda_device):
	X_ = X.type(torch.int8)
	args = (torch.arange(50, dtype=torch.float32)[:, None],)

	class Add(torch.nn.Module):
		def __init__(self):
			super().__init__()
			self.scale = torch.nn.Parameter(torch.ones(1))

		def forward(self, X, a):
			return X.sum(dim=-1) * self.scale + a

	y_cpu = predict(Add(), X_, args=args, batch_size=7, device='cpu')
	y_cuda = predict(Add(), X_, args=args, batch_size=7, device=cuda_device)

	assert y_cuda.dtype == y_cpu.dtype
	assert y_cuda.shape == y_cpu.shape
	assert_array_equal(y_cuda, y_cpu)
