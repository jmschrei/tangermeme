# predict.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

import collections
import contextlib
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch

from tqdm import trange

from ._compat import _autocast_supported, _preserve_model_state, _resolve_device


# On CUDA, the host does not wait for each batch to finish before preparing
# the next one. It waits for a batch's outputs only once this many later
# batches have been queued behind it.
_QUEUE_DEPTH = 2

# Outputs of up to this many bytes per batch are copied to the CPU through
# pinned memory without waiting for the device. Larger ones are copied with
# `.cpu()`, which waits, so that the pinned memory PyTorch caches stays small.
_PINNED_BYTES = 2 ** 24


def _start_cpu_copy(y, device, stream):
	"""Queue the copy of one model output to the CPU on `stream`.

	A tensor on `device` that fits in `_PINNED_BYTES` is copied into pinned
	host memory behind the work that produced it, and the host does not wait
	for it. The returned tensor must not be read until `stream` has passed
	the copy. Anything else is moved with `.cpu()`, which waits, and an
	object that is not a tensor raises as `.cpu()` does.
	"""

	if not (isinstance(y, torch.Tensor) and y.device == device
			and y.layout == torch.strided and y.nbytes <= _PINNED_BYTES):
		return y.cpu()

	y_cpu = torch.empty_like(y, device='cpu', pin_memory=True)
	y_cpu.copy_(y, non_blocking=True)

	# The device memory may be reused once `y` is freed; this keeps a block
	# allocated on another stream from being reused before the copy runs.
	y.record_stream(stream)
	return y_cpu


def _finish_cpu_copy(y, event):
	"""Wait for a batch's queued copies and move them out of pinned memory.

	The pinned buffers go back to PyTorch's cache as soon as they are copied
	into ordinary memory, so pinned memory stays bounded by a few batches no
	matter how many outputs are kept.
	"""

	event.synchronize()
	if isinstance(y, torch.Tensor):
		return y.clone() if y.is_pinned() else y
	return tuple(yi.clone() if yi.is_pinned() else yi for yi in y)


def predict(
	model: torch.nn.Module,
	X: torch.Tensor,
	args: Sequence[torch.Tensor] | None = None,
	func: Callable[..., Any] | None = None,
	batch_size: int = 32,
	dtype: str | torch.dtype | None = None,
	device: str | torch.device | None = None,
	verbose: bool = False,
) -> torch.Tensor | list[torch.Tensor]:
	"""Make batched predictions in a memory-efficient manner.

	This function will take a PyTorch model and make predictions from it using
	the forward function, with optional additional arguments to the model. The
	additional arguments must have the same batch size as the examples, and the
	i-th example will be given to the model with the i-th index of each
	additional argument. 

	Before starting predictions, the model is moved to the specified device. As 
	predictions are being made, each batch is also moved to the specified 
	device and then moved back to the CPU after predictions are made. Each batch
	is converted to the provided dtype if provided, keeping the original blob of
	examples in the original dtype. These features allow the function to work on 
	massive data sets that do not fit in GPU memory. For example, the original
	sequences can be kept as 8-bit integers for compression and each batch will
	be upcast to the desired precision. If a single batch does not fit in memory,
	try lowering the batch size.


	Parameters
	----------
	model: torch.nn.Module
		The PyTorch model to use to make predictions.

	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to make predictions for.

	args: tuple or list or None, optional
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function.
		Each per-batch slice is cast to `dtype` before being passed to the
		model — integer index tensors and boolean masks will be silently
		coerced to float, so pre-cast them or pass non-floating-point
		auxiliary inputs through a model wrapper instead. Default is None.

	func: function or None, optional 
		A function to apply to a batch of predictions after they have been made.
		If None, do nothing to them. Default is None.

	batch_size: int, optional
		The number of examples to make predictions for at a time. Default is 32.

	dtype: str or torch.dtype or None, optional
		The dtype to use with mixed precision autocasting. If None, use the dtype of
		the *model*. This allows you to use int8 to represent large data sets and
		only convert batches to the higher precision, saving memory. Default is None.

	device: str or torch.device or None, optional
		The device to move the model and batches to when making predictions. If
		None, use CUDA when available and fall back to CPU otherwise. The model's
		original device and training mode are restored after the call. Default
		is None.

	verbose: bool, optional
		Whether to display a progress bar during predictions. Default is False.


	Returns
	-------
	y: torch.Tensor or list of torch.Tensors
		The output from the model for each input example. The precise format
		is determined by the model. If the model outputs a single tensor,
		y is a single tensor concatenated across all batches. If the model
		outputs multiple tensors (list or tuple), y is always returned as a
		list (the original tuple container type is not preserved) of tensors
		which are each concatenated across all batches.
	"""

	if X.shape[0] == 0:
		raise ValueError("predict requires at least one example; got X "
			"with shape[0] == 0.")

	device = _resolve_device(device)
	dtype = _resolve_dtype(model, dtype)

	if args is not None:
		for arg in args:
			if arg.shape[0] != X.shape[0]:
				raise ValueError("Arguments must have the same first " +
					"dimension as X")

	###

	# On CUDA the copies to the device do not make the host wait, so that
	# `_predict_batches` can queue batches without a sync on each one.
	non_blocking = device.type == 'cuda'

	def batches():
		n = min(batch_size, X.shape[0])

		for start in trange(0, X.shape[0], n, disable=not verbose):
			end = start + n
			X_ = X[start:end].type(dtype).to(device, non_blocking=non_blocking)

			if args is not None:
				args_ = [a[start:end].type(dtype).to(device,
					non_blocking=non_blocking) for a in args]
			else:
				args_ = None

			yield X_, args_

	return _predict_batches(model, batches(), func=func, dtype=dtype,
		device=device)


def _resolve_dtype(
	model: torch.nn.Module,
	dtype: str | torch.dtype | None,
) -> torch.dtype:
	"""Resolve `predict`'s dtype argument: None means the dtype of the model's
	first parameter (float32 for a model without parameters), and a string
	names a torch dtype."""

	if dtype is None:
		try:
			return next(model.parameters()).dtype
		except (StopIteration, AttributeError):
			return torch.float32
	elif isinstance(dtype, str):
		return getattr(torch, dtype)
	return dtype


def _predict_batches(
	model: torch.nn.Module,
	batches: Iterable[tuple[torch.Tensor, list[torch.Tensor] | None]],
	func: Callable[..., Any] | None,
	dtype: torch.dtype,
	device: torch.device,
) -> torch.Tensor | list[torch.Tensor]:
	"""The loop inside `predict`, over batches its caller has built.

	`batches` yields `(X_, args_)` pairs that are already cast to `dtype` and
	on `device`, with `args_` a list of tensors or None. It is consumed inside
	`_preserve_model_state` and `torch.no_grad()`, so a generator may build
	each batch on the device only when it is needed, as
	`saturation_mutagenesis` does for its edited sequences. Each output has
	`func` applied and is moved to the CPU, and the outputs are concatenated
	exactly as `predict` returns them.
	"""

	use_autocast = _autocast_supported(device, dtype)

	# On CUDA, outputs are copied back to the CPU without the host waiting for
	# the device, so the host prepares and launches the next batch while the
	# device runs the current one. Only a batch's outputs are waited for,
	# `_QUEUE_DEPTH` batches later.
	queue = None
	if device.type == 'cuda':
		index = device.index
		if index is None:
			index = torch.cuda.current_device()

		out_device = torch.device('cuda', index)
		stream = torch.cuda.current_stream(out_device)
		queue = collections.deque()

	y = []
	with _preserve_model_state(model, device), torch.no_grad():
		for X_, args_ in batches:
			if X_.shape[0] == 0:
				continue

			if use_autocast:
				autocast_ctx = torch.autocast(device_type=device.type, dtype=dtype)
			else:
				autocast_ctx = contextlib.nullcontext()

			with autocast_ctx:
				if args_ is not None:
					y_ = model(X_, *args_)
				else:
					y_ = model(X_)

			# If a post-processing function is provided, apply it to the raw output
			# from the model.
			if func is not None:
				y_ = func(y_)

			# Move to the CPU
			if queue is None:
				if isinstance(y_, torch.Tensor):
					y_ = y_.cpu()
				elif isinstance(y_, (list, tuple)):
					y_ = tuple(yi.cpu() for yi in y_)
				else:
					raise ValueError("Cannot interpret output from model.")

				y.append(y_)
				continue

			if isinstance(y_, torch.Tensor):
				y_ = _start_cpu_copy(y_, out_device, stream)
			elif isinstance(y_, (list, tuple)):
				y_ = tuple(_start_cpu_copy(yi, out_device, stream) for yi in y_)
			else:
				raise ValueError("Cannot interpret output from model.")

			y.append(y_)
			queue.append((len(y) - 1, stream.record_event()))

			if len(queue) > _QUEUE_DEPTH:
				i, event = queue.popleft()
				y[i] = _finish_cpu_copy(y[i], event)

		while queue:
			i, event = queue.popleft()
			y[i] = _finish_cpu_copy(y[i], event)

	# Concatenate the outputs
	if isinstance(y[0], torch.Tensor):
		y = torch.cat(y)
	else:
		y = [torch.cat(y_) for y_ in list(zip(*y))]

	return y
