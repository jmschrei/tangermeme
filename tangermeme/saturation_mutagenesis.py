# saturation_mutagenesis.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

import contextlib
import time
import warnings
from collections.abc import Callable
from typing import Any
from typing import NamedTuple

import torch

from tqdm import trange

from ._compat import _autocast_supported
from ._compat import _preserve_model_state
from ._compat import _resolve_device
from ._fastism import _FastISM
from ._fastism import _Unsupported
from .predict import predict
from .utils import TangermemeWarning


# fast=True is not attempted for a job that fast=False would finish in less
# time than this, in seconds, because capturing and checking the model could
# take longer than it saves, and does when the check then fails.
_FAST_MIN_SECONDS = 30.0

# fast=True is used only when it takes at most this fraction of the time
# fast=False takes on the same batch of edits.
_FAST_MAX_RELATIVE_TIME = 1.0

# The largest difference between fast=True and fast=False, relative to the
# largest output, that verification accepts, by the dtype the model runs in.
# Each is set above the rounding differences of that dtype. On CUDA, float32
# convolutions run in TF32 by default, so float32 gets a looser tolerance
# there than on the CPU.
_FAST_TOLERANCE = {
	torch.float64: 1e-10,
	torch.float32: 1e-4,
	torch.float16: 1e-2,
	torch.bfloat16: 5e-2,
}
_FAST_TOLERANCE_CUDA_FLOAT32 = 3e-3


class SaturationMutagenesisRawResult(NamedTuple):
	"""Return type of `saturation_mutagenesis` when `raw_outputs=True`.

	Holds the reference predictions (`y0`, model output on the original
	sequences) alongside `y_hat`, the model output for each
	edit-distance-one perturbation. Positional unpacking
	(`y0, y_hat = saturation_mutagenesis(..., raw_outputs=True)`)
	continues to work.
	"""

	y0: torch.Tensor | list[torch.Tensor]
	y_hat: torch.Tensor | list[torch.Tensor]


def _attribution_score(y0, y_hat, target):
	"""An internal function for calculating the ISM attributions.

	This function, which is meant to be used for ISM, will take in the
	predictions before and after substitutions and a target -- which can be
	None -- and return the position-normalized differences. Specifically,
	for each example, the differences in prediction will be normalized by
	subtracting out the per-position average, and then averaged across
	all tasks if target is None.


	Parameters
	----------
	y0: torch.Tensor, shape=(-1, n_targets)
		Model predictions for each example on the original predictions.

	y_hat: torch.Tensor, shape=(-1, len(alphabet), length, n_targets)
		Model predictions for each example for each substitution.

	target: int or slice or None
		If the user wants to subset to only some targets when calculating the
		average attribution across targets. If None, use all.
	"""

	attr = y_hat[:, :, :, target] - y0[:, None, None, target]
	attr -= torch.mean(attr, dim=1, keepdims=True)

	if len(attr.shape) > 3:
		attr = torch.mean(attr, dim=tuple(range(3, len(attr.shape))))
	return attr


def _to_int8(X):
	"""Cast a one-hot encoding to int8 on the CPU, warning on soft values.

	The int8 cast truncates toward zero, so a soft encoding would silently lose
	every value with magnitude below one.
	"""

	if X.is_floating_point() and not torch.equal(X, X.round()):
		warnings.warn("X contains non-integer values which will be truncated "
			"toward zero by the int8 cast; values with magnitude < 1 become 0. "
			"Pass a hard one-hot encoding to avoid silently zeroing positions.",
			TangermemeWarning)

	return X.type(torch.int8).cpu()


def _check_span(start, end, length):
	"""Resolve a negative `end` and check that the span is non-empty.

	A negative `end` is remapped to `length + 1 + end`, so `end=-1` includes
	the last position.
	"""

	if end < 0:
		end = length + 1 + end

	if start < 0 or end > length or start >= end:
		raise ValueError("start and end must satisfy "
			"0 <= start < end <= length; got start={}, end={} for length "
			"{}.".format(start, end, length))

	return end


def _single_edits(X_i, start, end):
	"""List the true single-base edits of one sequence within [start, end).

	One substitution per position would re-apply the base already there (an
	identity edit whose prediction equals y0); it is left out here and filled
	from y0 by `_fill_identity`, saving up to 25% of the forward passes. A
	column that is all-zero (an `N`) or multi-hot is not a clean one-hot of any
	character, so it has no identity row and all of its edits are kept. The
	edits are laid out in [character, position] order to match the
	reconstruction.


	Parameters
	----------
	X_i: torch.Tensor, shape=(len(alphabet), length)
		One one-hot encoded sequence.

	start: int
		The first position to edit.

	end: int
		The position after the last one to edit.


	Returns
	-------
	edits: torch.Tensor, shape=(len(alphabet) * (end - start),)
		A boolean mask over the flattened [character, position] grid that is
		True for each edit that is made.

	edit_chars: torch.Tensor, shape=(n_edits,)
		The character each edit substitutes in.

	edit_positions: torch.Tensor, shape=(n_edits,)
		The position of each edit in the full sequence.
	"""

	ref = X_i[:, start:end]
	identity = (ref == 1) & (ref.sum(dim=0, keepdim=True) == 1)
	edits = ~identity.reshape(-1)

	edit_chars, edit_positions = torch.where(~identity)
	return edits, edit_chars, edit_positions + start


def _edited_inputs(X, args, i, edit_chars, edit_positions):
	"""Build one copy of sequence i per edit, and repeat its `args` to
	match."""

	n_edits = edit_chars.shape[0]

	X_ = X[i].repeat(n_edits, 1, 1)
	rows = torch.arange(n_edits)
	X_[rows, :, edit_positions] = 0
	X_[rows, edit_chars, edit_positions] = 1

	if args is not None:
		args_ = tuple(a[i].repeat(n_edits, *(1 for _ in a[i].shape))
			for a in args)
	else:
		args_ = None

	return X_, args_


def _fill_identity(y0, i, edits, y_edits):
	"""Scatter the predictions for one sequence's edits into the full
	[character, position] grid, filling the identity slots with y0[i]."""

	if isinstance(y_edits, torch.Tensor):
		y_hat_ = y0[i].unsqueeze(0).repeat(edits.shape[0],
			*(1 for _ in y0[i].shape))
		y_hat_[edits] = y_edits
	else:
		y_hat_ = [y0_[i].unsqueeze(0).repeat(edits.shape[0],
			*(1 for _ in y0_[i].shape)) for y0_ in y0]
		for y_hat_h, y_edit in zip(y_hat_, y_edits):
			y_hat_h[edits] = y_edit

	return y_hat_


def _stack_y_hat(y_hat, n_examples, n_chars, width):
	"""Stack per-sequence grids into (n_examples, n_chars, width, ...)
	tensors, one per model output."""

	if isinstance(y_hat[0], torch.Tensor):
		return torch.stack(y_hat).reshape(n_examples, n_chars, width,
			*y_hat[0].shape[1:])

	return [
		torch.stack(y_).reshape(n_examples, n_chars, width,
			*y_[0].shape[1:]) for y_ in zip(*y_hat)
	]


def _timed(fn, device):
	"""Call `fn` and return its result and the wall-clock seconds it took."""

	if device.type == 'cuda':
		torch.cuda.synchronize(device)
	tic = time.perf_counter()
	result = fn()
	if device.type == 'cuda':
		torch.cuda.synchronize(device)
	return result, time.perf_counter() - tic


def _verify_fast(fast, model, X, args, start, end, batch_size, dtype, device,
	func):
	"""Compare the fast path with `predict` on one batch of edits.

	The batch holds every edit at the first and last edited positions of the
	first example, where padding is read, and edits spread evenly over the
	rest. The two must agree to within a tolerance, relative to the largest
	output, set by the dtype, and the fast path must not take longer. Each is
	timed twice after a first, untimed call, alternating between them, and the
	faster time of each is compared.


	Returns
	-------
	reason: str or None
		Why the fast path should not be used, or None if it should.
	"""

	_, chars, positions = _single_edits(X[0], start, end)
	edge = torch.where((positions == start) | (positions == end - 1))[0]
	spread = torch.linspace(0, chars.shape[0] - 1, min(batch_size,
		chars.shape[0])).round().long()
	keep = torch.unique(torch.cat([edge, spread]))
	chars, positions = chars[keep], positions[keep]
	X_, args_ = _edited_inputs(X, args, 0, chars, positions)

	def run_fast():
		return fast.predict_edits(0, chars, positions, batch_size, func)

	def run_slow():
		return predict(model, X_, args=args_, func=func,
			batch_size=batch_size, dtype=dtype, device=device)

	y_fast, y_slow = run_fast(), run_slow()
	if isinstance(y_slow, torch.Tensor):
		y_fast, y_slow = [y_fast], [y_slow]

	err, scale = 0.0, 0.0
	for a, b in zip(y_fast, y_slow):
		if a.shape != b.shape:
			return ("fast=True returned an output of shape {} where fast=False "
				"returned {}".format(tuple(a.shape), tuple(b.shape)))

		a, b = a.double(), b.double()
		err = max(err, (a - b).abs().max().item())
		scale = max(scale, b.abs().max().item())

	tolerance = _FAST_TOLERANCE.get(dtype, _FAST_TOLERANCE[torch.float32])
	if dtype == torch.float32 and device.type == 'cuda':
		tolerance = _FAST_TOLERANCE_CUDA_FLOAT32
	tolerance *= scale if scale > 0 else 1
	if not err <= tolerance:
		return ("fast=True and fast=False differed by up to {:.3g} on a batch "
			"of edits, more than the tolerance of {:.3g}".format(err, tolerance))

	t_fast, t_slow = float("inf"), float("inf")
	for _ in range(2):
		t_fast = min(t_fast, _timed(run_fast, device)[1])
		t_slow = min(t_slow, _timed(run_slow, device)[1])

	if t_fast > t_slow * _FAST_MAX_RELATIVE_TIME:
		reason = ("fast=True took {:.3g}s on a batch of edits and fast=False "
			"took {:.3g}s".format(t_fast, t_slow))
		if device.type == 'cuda':
			reason += (" (on a GPU, fast=True runs many small operations per "
				"batch and gains more from a larger batch_size)")
		return reason
	return None


def _fast_path(model, X, args, start, end, batch_size, dtype, device, func):
	"""Decide whether to use fast=True, and capture and verify the model if so.

	The first batch of edits of the first example is run with fast=False and
	timed. When the whole job would then take less than `_FAST_MIN_SECONDS`
	with fast=False, capturing the model could cost more than it saves, so
	fast=False is used without a warning. Otherwise the model is captured and
	verified, and a failure of either is reported with a warning.

	The caller must keep the model on `device` and in evaluation mode while
	the returned object is used, because it holds the model's parameters.


	Returns
	-------
	fast: _FastISM or None
		The verified fast path, or None to use fast=False.

	y_first: torch.Tensor or list of torch.Tensors
		The predictions for the first `n_first` edits of the first example.

	n_first: int
		The number of edits `y_first` holds.
	"""

	device = _resolve_device(device)

	_, chars, positions = _single_edits(X[0], start, end)
	n_first = min(batch_size, chars.shape[0])
	X_, args_ = _edited_inputs(X, args, 0, chars[:n_first],
		positions[:n_first])

	y_first, t_first = _timed(lambda: predict(model, X_, args=args_,
		func=func, batch_size=batch_size, dtype=dtype, device=device), device)

	n_edits = X.shape[0] * chars.shape[0]
	if t_first * n_edits / n_first < _FAST_MIN_SECONDS:
		return None, y_first, n_first

	if dtype is None:
		try:
			dtype = next(model.parameters()).dtype
		except (StopIteration, AttributeError):
			dtype = torch.float32
	elif isinstance(dtype, str):
		dtype = getattr(torch, dtype)

	use_autocast = _autocast_supported(device, dtype)

	try:
		with torch.no_grad():
			fast = _FastISM(model, X, args, start, end, device, dtype,
				use_autocast)
		reason = _verify_fast(fast, model, X, args, start, end, batch_size,
			dtype, device, func)
	except _Unsupported as e:
		reason = str(e)
	except Exception as e:
		reason = "fast=True raised {}: {}".format(type(e).__name__,
			str(e).strip().split("\n")[0])

	if reason is None:
		return fast, y_first, n_first

	warnings.warn("saturation_mutagenesis is using fast=False because {}. "
		"Pass fast=False to skip this attempt.".format(reason),
		TangermemeWarning)
	return None, y_first, n_first



def saturation_mutagenesis(
	model: torch.nn.Module,
	X: torch.Tensor,
	args: tuple | None = None,
	start: int = 0,
	end: int = -1,
	batch_size: int = 32,
	target: int | slice | None = None,
	hypothetical: bool = False,
	raw_outputs: bool = False,
	dtype: str | torch.dtype | None = None,
	device: str | torch.device | None = None,
	verbose: bool = False,
	func: Callable[..., Any] | None = None,
	fast: bool = True,
) -> torch.Tensor | SaturationMutagenesisRawResult:
	"""Performs in-silico saturation mutagenesis on a set of sequences.
	
	This function will perform in-silico saturation mutagenesis on a set of 
	sequences and return the predictions on the original sequences and each
	of the sequences with an edit distance of one on them.
	
	By default, this function will aggregate these predictions into an
	attribution value. For each single-character substitution, the change in
	prediction (perturbed minus original) is computed and then mean-subtracted
	across the alphabet axis so the values at each position are centered on zero;
	when `target=None` these are additionally averaged across all of the model's
	outputs. The values are not Euclidean distances and are not Z-score
	normalized -- preserving the raw magnitude lets you compare the importance of
	different regions head-to-head. Unless `hypothetical=True`, the result is then
	projected onto the observed bases by multiplying by the one-hot `X`. This
	aggregation assumes that the model returns only a single tensor. This tensor
	can have multiple outputs, e.g., be of shape (batch_size, n_targets) where
	n_targets > 1, but the model cannot return multiple tensors.
	
	If you simply want the predictions before and after the substitutions
	without the method turning those into attributions because, perhaps, you
	want to define your own aggregation method, you can use `raw_outputs=True`.

	By default, the predictions for the edited sequences are computed with
	fastISM (Nair et al., Bioinformatics 2022). A single-base edit changes
	only the positions of each convolutional layer whose receptive field
	includes the edit, so most of the computation in a full forward pass on
	each edited sequence repeats the forward pass on the original. With
	`fast=True` the model is captured with `torch.export`, the activations of
	each original sequence are cached, and for each edit an operation such as
	a convolution, pooling, cropping or an elementwise function is computed
	only over the positions the edit can change. Any other operation, such as
	a dense layer over the flattened sequence, a mean over the length or
	attention, is computed in full on the cached activations with the changed
	positions written in. Models that are convolutional for most of their
	depth gain the most; a model that pools or normalizes over the whole
	length early has little to gain. The results match `fast=False` up to
	floating-point rounding rather than bit for bit.

	The first batch of edits is always computed with `fast=False` and timed.
	When the whole job would take less than thirty seconds at that rate, the
	model is not captured, because capturing and checking it can take several
	seconds, and the job continues with `fast=False`.
	Otherwise, before the fast path is used, it is checked against
	`fast=False` on one batch of edits from the first example, which includes
	both ends of the edited span. When the model cannot be captured, the two
	disagree by more than rounding, or the fast path is slower, a
	`TangermemeWarning` gives the reason and the function continues with
	`fast=False`.


	Parameters
	----------
	model: torch.nn.Module
		The PyTorch model to use to make predictions.
	
	X: torch.tensor, shape=(-1, len(alphabet), length)
		A set of one-hot encoded sequences to calculate attribution values
		for. 
	
	args: tuple or None, optional
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function.
		Default is None.
	
	start: int, optional
		The start of where to begin making perturbations to the sequence.
		Default is 0.
	
	end: int, optional
		The end of where to make perturbations to the sequence. Positive
		values follow standard Python slice semantics (non-inclusive).
		Negative values are remapped via `end = length + 1 + end` so that
		`end=-1` maps to `length` and the *last* nucleotide is included.
		Note this differs from Python's `X[start:-1]` (which would drop the
		last element). Default is -1, meaning the entire sequence.
	
	batch_size: int, optional
		The number of examples to make predictions for at a time. Default is 32.
	
	target: int or slice or None, optional
		Whether to focus on a single output/slice of outputs from the model
		when calculating attributions rather than the entire set of outputs.
		If None, use all targets when calculating attributions. Default is None.
	
	hypothetical: bool, optional
		Whether to return attributions for all possible characters at each
		position or only for the character that is actually in the sequence.
		Only matters when `raw_outputs=False`. Default is False.
	
	raw_outputs: bool, optional
		Whether to return the raw outputs from the method -- in this case,
		the predictions from the reference sequence and from each of the
		perturbations -- or the processed attribution values. Default is False.
	
	dtype: str or torch.dtype or None, optional
		The dtype to use with mixed precision autocasting. If None, use the dtype of
		the *model*. This allows you to use int8 to represent large data sets and
		only convert batches to the higher precision, saving memory. Default is None.
	
	device: str or torch.device or None, optional
		The device to move the model and batches to when making predictions. If
		None, use CUDA when available and fall back to CPU otherwise. Default
		is None.
	
	verbose: bool, optional
		Whether to display a progress bar during predictions. Default is False.

	func: function or None, optional
		A function to apply to a batch of predictions after they have been made,
		forwarded to `predict`. It is applied identically to the reference
		predictions and to every perturbation, so attributions are computed on
		the post-processed values. Use it to, e.g., select a single output head
		or apply a final non-linearity. If None, do nothing. Default is None.

	fast: bool, optional
		Whether to compute the predictions for the edited sequences with
		fastISM, recomputing only the positions of each layer that an edit can
		change, for a job that would take more than thirty seconds otherwise. The
		model is captured with `torch.export` and checked against `fast=False`
		on one batch of edits; if that fails, a `TangermemeWarning` gives the
		reason and `fast=False` is used. Results match `fast=False` up to
		floating-point rounding. On a GPU, the fast path runs many small
		operations for each batch, so it gains more from a larger `batch_size`
		than `fast=False` does. Default is True.


	Returns
	-------
	attr: torch.Tensor
		Processed attribution values. For each example and each position, the
		difference between the model's prediction on the perturbed sequence and
		on the original sequence is computed, then mean-subtracted across the
		alphabet axis so the values at each position are centered on zero.
	
	-- or, if raw_outputs=True --
	
	y0: torch.Tensor or list/tuple of torch.Tensors
		The outputs from the model for the reference sequences.
	
	y_hat: torch.Tensor or list/tuple of torch.Tensors
		The outputs from the model for each of the perturbed sequences.
	"""

	X = _to_int8(X)
	y0 = predict(model, X, args=args, func=func, dtype=dtype, device=device)

	if not raw_outputs and not isinstance(y0, torch.Tensor):
		raise ValueError("raw_outputs=True is required for models that return "
			"multiple output tensors; the attribution aggregation assumes a "
			"single output tensor.")

	end = _check_span(start, end, X.shape[-1])

	with contextlib.ExitStack() as stack:
		fast_path, y_first, n_first = None, None, 0
		if fast:
			# The fast path holds the model's parameters, so the model stays
			# on the device for the whole call rather than for each batch.
			stack.enter_context(_preserve_model_state(model,
				_resolve_device(device)))
			fast_path, y_first, n_first = _fast_path(model, X, args, start,
				end, batch_size, dtype, device, func)

		y_hat = []
		for i in trange(X.shape[0], disable=not verbose):
			edits, edit_chars, edit_positions = _single_edits(X[i], start, end)

			# The first edits of the first example were already predicted
			# while deciding whether to use the fast path.
			k = n_first if i == 0 else 0
			edit_chars, edit_positions = edit_chars[k:], edit_positions[k:]

			if edit_chars.shape[0] == 0:
				y_edits = y_first
			elif fast_path is not None:
				y_edits = fast_path.predict_edits(i, edit_chars, edit_positions,
					batch_size, func)
			else:
				X_, args_ = _edited_inputs(X, args, i, edit_chars,
					edit_positions)
				y_edits = predict(model, X_, args=args_, func=func,
					batch_size=batch_size, dtype=dtype, device=device)

			if k > 0 and edit_chars.shape[0] > 0:
				if isinstance(y_edits, torch.Tensor):
					y_edits = torch.cat([y_first, y_edits])
				else:
					y_edits = [torch.cat(y_) for y_ in zip(y_first, y_edits)]

			y_hat.append(_fill_identity(y0, i, edits, y_edits))

	y_hat = _stack_y_hat(y_hat, X.shape[0], X.shape[1], end-start)

	if not raw_outputs:
		attr = _attribution_score(y0, y_hat, target)
		return X[:, :, start:end] * attr if not hypothetical else attr
	return SaturationMutagenesisRawResult(y0=y0, y_hat=y_hat)
