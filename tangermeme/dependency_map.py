# dependency_map.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from tqdm import trange

from .saliency import saliency
from .utils import _validate_input


def dependency_map(
	model: torch.nn.Module,
	X: torch.Tensor,
	args: Sequence[torch.Tensor] | None = None,
	start: int = 0,
	end: int = -1,
	func: Callable[..., Any] = saliency,
	additional_func_kwargs: dict | None = None,
	verbose: bool = False,
	**kwargs: Any,
) -> torch.Tensor:
	"""Measure how substituting each position changes attributions elsewhere.

	An attribution map answers "which bases matter here"; a dependency map
	answers "which bases decide whether the other bases matter". For each
	pair of positions (j, i) it records how much the attribution at position
	j moves when the base at position i is substituted. A model that treats
	positions independently produces a purely diagonal map, because changing
	a base can only alter its own attribution. Off-diagonal structure is
	therefore direct evidence of epistasis learned by the model: cooperative
	motif pairs show up as blocks, and the flanks a motif depends on show up
	as stripes leading into it.

	This is in-silico saturation mutagenesis lifted one level up. Where
	`saturation_mutagenesis` asks how a single substitution changes the
	model's scalar prediction, this asks how that same substitution changes
	the model's entire explanation of the sequence. The cost follows: for an
	alphabet of size A and a substituted window of size W it runs A * W
	attributions per sequence, so the attribution function should be a cheap
	one. That is why `saliency` is the default -- a single backward pass --
	but `deep_lift_shap` and `saturation_mutagenesis` both satisfy the same
	contract and can be passed in when their accuracy is worth the cost.

	Substitutions that re-apply the base already present are skipped rather
	than computed, since their attributions are identical to the unperturbed
	ones and their difference is exactly zero; this saves a quarter of the
	work for DNA. Each position is then averaged over only the substitutions
	that actually changed it, so a position holding an `N` -- which has no
	base to re-apply and therefore A real substitutions rather than A - 1 --
	is on the same scale as every other position.

	Sequences are processed one at a time so that peak memory depends on the
	length of a single sequence rather than on how many were passed in.

	NOTE: the diagonal is not comparable to the rest of the map. It measures
	the direct effect of a substitution on its own attribution, which is
	non-zero even for a linear model, and is usually several times larger
	than the off-diagonal entries. Mask it with `dmap.diagonal(dim1=-2,
	dim2=-1).zero_()` before plotting when `start=0` and `end=-1`.


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	X: torch.tensor, shape=(-1, len(alphabet), length)
		A set of one-hot encoded sequences to calculate dependency maps for.
		Cast to int8 internally, in the same manner as
		`saturation_mutagenesis`, so that the substitution grid stays small;
		`func` is responsible for upcasting each batch, as `saliency`,
		`deep_lift_shap`, and `saturation_mutagenesis` all do.

	args: tuple or list or None, optional
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function.
		Default is None.

	start: int, optional
		The first position to substitute, inclusive. Attributions are still
		measured across the entire sequence; this only restricts which
		positions are perturbed, and so which columns the map has. Default
		is 0.

	end: int, optional
		The last position to substitute, exclusive. Negative values count
		back from the end of the sequence, so the default of -1 means the
		full length. Default is -1.

	func: function, optional
		The attribution function to apply before and after each substitution.
		It must satisfy `func(model, X, args=, **kwargs)` and return a tensor
		of shape `(-1, len(alphabet), length)`, which is summed over the
		alphabet axis to give one value per position. `saliency`,
		`deep_lift_shap`, and `saturation_mutagenesis` all satisfy this.
		Default is `saliency`.

	additional_func_kwargs: dict or None, optional
		Arguments to pass to `func` that would otherwise collide with the
		arguments of this function -- `start`, `end`, and `verbose` are all
		consumed here rather than forwarded, and `func` itself takes a `func`
		argument in `saliency` and `saturation_mutagenesis`. The dict is
		copied, so the caller's object is not mutated. Default is None.

	verbose: bool, optional
		Whether to display a progress bar over the sequences in `X`. This is
		not forwarded to `func`; pass `verbose` through
		`additional_func_kwargs` to see the inner progress bar instead.
		Default is False.

	kwargs: optional
		Additional arguments forwarded to `func`, e.g. `target`,
		`batch_size`, `dtype`, and `device`.


	Returns
	-------
	dmap: torch.Tensor, shape=(-1, length, end-start)
		The dependency map for each sequence, on the CPU. `dmap[:, j, i]` is
		the average absolute change in the attribution at position `j` caused
		by substituting the base at position `start + i`, averaged over the
		substitutions that actually changed that base.
	"""

	_validate_input(X, "X", shape=(-1, -1, -1), ohe=True, allow_N=True)

	if X.shape[0] == 0:
		raise ValueError("dependency_map requires at least one example; got "
			"X with shape[0] == 0.")

	if args is not None:
		for arg in args:
			if arg.shape[0] != X.shape[0]:
				raise ValueError("Arguments must have the same first " +
					"dimension as X")

	additional_func_kwargs = dict(additional_func_kwargs or {})

	X = X.type(torch.int8).cpu()
	n, n_chars, length = X.shape

	if end < 0:
		end = length + 1 + end

	if start < 0 or end > length or start >= end:
		raise ValueError("start and end must satisfy "
			"0 <= start < end <= length; got start={}, end={} for length "
			"{}.".format(start, end, length))

	def _attributions(X_, args_):
		X_attr = func(model, X_, args=args_, **kwargs, **additional_func_kwargs)

		if not isinstance(X_attr, torch.Tensor) or X_attr.ndim != 3:
			raise ValueError("func must return a single tensor of shape "
				"(-1, len(alphabet), length); got {}.".format(
					type(X_attr) if not isinstance(X_attr, torch.Tensor)
					else tuple(X_attr.shape)))

		return X_attr.type(torch.float32).sum(dim=1)

	dmap = torch.empty(n, length, end - start, dtype=torch.float32)

	for i in trange(n, disable=not verbose):
		args_ = None if args is None else tuple(a[i:i+1] for a in args)
		attr0 = _attributions(X[i:i+1], args_)[0]

		# Build only the substitutions that actually change a base. A column
		# that is a clean one-hot has one identity edit, which is dropped;
		# an all-zero ('N') or multi-hot column has none, so all of its
		# characters are kept. This mirrors `saturation_mutagenesis`.
		ref = X[i, :, start:end]
		identity = (ref == 1) & (ref.sum(dim=0, keepdim=True) == 1)

		edit_chars, edit_positions = torch.where(~identity)
		n_edits = edit_chars.shape[0]

		X_edits = X[i].repeat(n_edits, 1, 1)
		rows = torch.arange(n_edits)
		X_edits[rows, :, edit_positions + start] = 0
		X_edits[rows, edit_chars, edit_positions + start] = 1

		if args is None:
			args_ = None
		else:
			args_ = tuple(a[i].repeat(n_edits, *(1 for _ in a[i].shape))
				for a in args)

		attr = _attributions(X_edits, args_)
		attr = attr.sub_(attr0).abs_()

		# Sum the absolute differences into the substituted positions and
		# divide by how many substitutions each position actually had.
		totals = torch.zeros(end - start, length, dtype=torch.float32)
		totals.index_add_(0, edit_positions, attr)

		counts = (~identity).sum(dim=0).type(torch.float32)
		dmap[i] = (totals / counts.unsqueeze(-1)).T

	return dmap
