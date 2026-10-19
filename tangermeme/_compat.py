# _compat.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Private helpers for device handling and model-state preservation that
the public functions share."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch


def _resolve_device(device: str | torch.device | None) -> torch.device:
	"""Resolve a user-supplied device argument to a concrete `torch.device`.

	If `device` is None, return `'cuda'` when CUDA is available and `'cpu'`
	otherwise. If `device` is a string or `torch.device`, it is returned as
	a `torch.device`.
	"""

	if device is None:
		return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	if isinstance(device, torch.device):
		return device

	return torch.device(device)


def _autocast_supported(
	device: torch.device | str,
	dtype: torch.dtype | None,
) -> bool:
	"""Return True when `torch.autocast` is meaningful for this combination.

	`torch.autocast(device_type='cpu', dtype=torch.float32)` raises on
	recent PyTorch versions; autocasting to the native dtype is also a
	no-op. This helper centralizes the check so `predict` and other
	callers can skip autocast when it would either crash or do nothing.
	"""

	if dtype is None:
		return False

	device_type = device.type if isinstance(device, torch.device) else str(device).split(':')[0]

	if device_type == 'cpu':
		return dtype in (torch.bfloat16, torch.float16)

	if device_type == 'cuda':
		return dtype != torch.float32

	return False


@contextlib.contextmanager
def _preserve_model_state(
	model: torch.nn.Module,
	device: torch.device,
) -> Iterator[None]:
	"""Move a model to `device` for the duration of a `with` block, then
	restore its original device and training mode on exit.
	"""

	try:
		orig_device = next(model.parameters()).device
	except StopIteration:
		orig_device = None

	was_training = model.training

	if orig_device is None or orig_device != device:
		model.to(device)
	model.eval()

	try:
		yield
	finally:
		if was_training:
			model.train()
		if orig_device is not None and orig_device != device:
			model.to(orig_device)
