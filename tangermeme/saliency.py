# saliency.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from typing import Any

import torch

from tqdm import trange

from ._compat import _autocast_supported, _preserve_model_state, _resolve_device
from .utils import _validate_input


def saliency(
	model: torch.nn.Module,
	X: torch.Tensor,
	args: Sequence[torch.Tensor] | None = None,
	target: int = 0,
	batch_size: int = 32,
	hypothetical: bool = False,
	func: Callable[..., Any] | None = None,
	dtype: str | torch.dtype | None = None,
	device: str | torch.device | None = None,
	verbose: bool = False,
) -> torch.Tensor:
	"""Calculate input-times-gradient saliency maps for a set of sequences.

	This function answers the same question that `deep_lift_shap` and
	`saturation_mutagenesis` do -- which nucleotides drive this prediction --
	but with the cheapest possible estimator: a single backward pass. The
	gradient of the model's `target`-th output with respect to the input is
	a first-order, purely local statement about how the prediction would move
	if a base were made slightly more or less present. Multiplying that
	gradient by the one-hot input projects it onto the bases that are actually
	there, which is the same projection `deep_lift_shap` and
	`saturation_mutagenesis` apply when `hypothetical=False`.

	The tradeoff against the other two methods is exactly the linearity
	assumption. A gradient is the tangent to the model at the observed
	sequence, so it says nothing about what happens when a base is changed by
	a finite amount, and it saturates: a nucleotide that a model is already
	certain about can carry a near-zero gradient despite being essential.
	`deep_lift_shap` addresses this by differencing against references, and
	`saturation_mutagenesis` by actually making every substitution. Saliency
	is what you use when you need attributions for far more sequences than
	either of those can afford, when an op in the model has no DeepLIFT rule
	registered, or as the inner attribution step of `dependency_map`, which
	calls it once per single-base substitution and so is highly sensitive to
	the cost of a single attribution.

	Attributions are calculated for a whole batch at a time. The examples in a
	batch do not interact during the forward pass, so a single `backward()`
	call on the summed output yields the correct per-example gradient for
	every example in the batch, and the result does not depend on how the
	examples happen to be grouped into batches. As with `predict`, each batch
	is moved to `device` and cast to `dtype` for the forward and backward
	pass and moved back to the CPU afterwards, so `X` itself can be kept in a
	memory-efficient dtype such as int8 even though gradients require floating
	point.

	NOTE: after `func` is applied, predictions MUST yield a
	`(batch_size, n_targets)` tensor, even if n_targets is 1. A model that
	returns a profile of shape `(batch_size, n_tasks, length)` will have
	`target` silently select an entire task rather than a single output, so
	pass a `func` that reduces the profile, or wrap the model.


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	X: torch.tensor, shape=(-1, len(alphabet), length)
		A set of one-hot encoded sequences to calculate attribution values
		for.

	args: tuple or list or None, optional
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function.
		Each per-batch slice is cast to `dtype` before being passed to the
		model, so integer index tensors and boolean masks will be silently
		coerced to float; pre-cast them or pass them through a model wrapper
		instead. Default is None.

	target: int, optional
		The output of the model to calculate attributions for. This indexes
		the second dimension of the model's predictions. Default is 0.

	batch_size: int, optional
		The number of examples to calculate attributions for at a time. The
		returned values do not depend on this. Default is 32.

	hypothetical: bool, optional
		Whether to return attributions for all possible characters at each
		position, i.e. the raw gradient, or only for the characters that are
		actually present, i.e. the gradient projected onto `X`. Note that the
		gradient is a local quantity evaluated at the observed sequence, so a
		hypothetical saliency value is the effect of infinitesimally adding a
		character that is not there rather than of substituting it in; use
		`saturation_mutagenesis` if you need the latter. Default is False.

	func: function or None, optional
		A function to apply to the output of the model before the `target`-th
		output is selected, with the same semantics as in `predict`. It runs
		inside the autograd graph, so it must be differentiable. Use it to,
		e.g., reduce a profile head to a scalar per task. If None, do nothing.
		Default is None.

	dtype: str or torch.dtype or None, optional
		The dtype to use with mixed precision autocasting. If None, use the
		dtype of the *model*. This allows you to use int8 to represent large
		data sets and only convert batches to the higher precision, saving
		memory. Default is None.

	device: str or torch.device or None, optional
		The device to move the model and batches to when calculating
		attributions. If None, use CUDA when available and fall back to CPU
		otherwise. The model's original device and training mode are restored
		after the call. Default is None.

	verbose: bool, optional
		Whether to display a progress bar as attributions are calculated.
		Default is False.


	Returns
	-------
	X_attr: torch.Tensor, shape=(-1, len(alphabet), length)
		The attribution values for each example, on the CPU. If
		`hypothetical=False` these are the gradient multiplied by `X`, so
		every position that is not the observed character is zero; if
		`hypothetical=True` these are the raw gradients. The values are
		signed -- a negative attribution means the character pushes the
		`target`-th output down.
	"""

	_validate_input(X, "X", shape=(-1, -1, -1), ohe=True, allow_N=True)

	if X.shape[0] == 0:
		raise ValueError("saliency requires at least one example; got X "
			"with shape[0] == 0.")

	device = _resolve_device(device)

	if dtype is None:
		try:
			dtype = next(model.parameters()).dtype
		except (StopIteration, AttributeError):
			dtype = torch.float32
	elif isinstance(dtype, str):
		dtype = getattr(torch, dtype)

	if args is not None:
		for arg in args:
			if arg.shape[0] != X.shape[0]:
				raise ValueError("Arguments must have the same first " +
					"dimension as X")

	###

	use_autocast = _autocast_supported(device, dtype)

	# Unlike `predict`, the output shape is known before the loop -- it is
	# exactly the shape of X -- so the batches are written into a single
	# preallocated tensor rather than concatenated afterwards. Attributions
	# are input-sized, and for `dependency_map` this avoids copying a tensor
	# as large as the whole substitution grid once per sequence.
	X_attr = None
	with _preserve_model_state(model, device):
		batch_size = min(batch_size, X.shape[0])

		for start in trange(0, X.shape[0], batch_size, disable=not verbose):
			end = start + batch_size
			X_ = X[start:end].type(dtype).to(device).requires_grad_()

			if args is not None:
				args_ = [a[start:end].type(dtype).to(device) for a in args]
			else:
				args_ = []

			if use_autocast:
				autocast_ctx = torch.autocast(device_type=device.type,
					dtype=dtype)
			else:
				autocast_ctx = contextlib.nullcontext()

			# The caller may be inside a torch.no_grad() block -- for instance
			# when saliency is passed as the `func` of a perturbation function
			# -- so grad has to be turned back on explicitly here.
			with torch.autograd.set_grad_enabled(True), autocast_ctx:
				y_ = model(X_, *args_)

				if func is not None:
					y_ = func(y_)

				if not isinstance(y_, torch.Tensor):
					raise ValueError("saliency requires a model that returns "
						"a single tensor output; got {}. Pass a `func` that "
						"reduces the output or wrap the model.".format(
							type(y_)))

				grad, = torch.autograd.grad(y_[:, target].sum(), X_)

			if not hypothetical:
				grad = grad * X_

			grad = grad.detach().cpu()

			if X_attr is None:
				X_attr = torch.empty(X.shape, dtype=grad.dtype)

			X_attr[start:end] = grad

	return X_attr
