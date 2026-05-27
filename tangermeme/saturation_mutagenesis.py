# saturation_mutagenesis.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

from typing import NamedTuple

import numba
import torch

from tqdm import trange

from .predict import predict


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



@numba.njit("void(int8[:, :, :], int32, int32)")
def _edit_distance_one(X, start, end):
	"""An internal function for generating all sequences of edit distance 1

	This internal function, which is meant to be used for ISM, will take in a
	one-hot encoded sequence and return all sequences that have an edit distance
	of one.

	Note: the inner loop iterates over all 4 alphabet characters at each
	mutated position, *including* the character already at that position.
	One of the four "variants" per position is therefore an identity
	substitution (a no-op); it contributes a zero into the per-position
	mean used by `_attribution_score`. Callers wanting strict
	edit-distance-1 outputs should mask or skip those rows.

	Parameters
	----------
	X: torch.Tensor, shape=(length*len(alphabet), len(alphabet), length)
		A single one-hot encoded sequence.

	start: int
		The first nucleotide to begin making edits on, inclusive.

	end: int
		The end of the span. Edits are not made on this nucleotide at this
		index. Can be negative indexes.
	"""
	
	i = 0
	for j in range(4):
		for k in range(start, end):
			for l in range(4):
				X[i, l, k] = 0
				
			X[i, j, k] = 1
			i += 1



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
) -> torch.Tensor | SaturationMutagenesisRawResult:
	"""Performs in-silico saturation mutagenesis on a set of sequences.
	
	This function will perform in-silico saturation mutagenesis on a set of 
	sequences and return the predictions on the original sequences and each
	of the sequences with an edit distance of one on them.
	
	By default, this function will aggregate these predictions into an
	attribution value. This aggregation involves taking the Euclidean distance
	between the predictions before and after the substitutions and Z-score
	normalizing them across the entire example. However, this assumes that
	the model returns only a single tensor. This tensor can have multiple
	outputs, e.g., be of shape (batch_size, n_targets) where n_targets > 1,
	but the model cannot return multiple tensors.
	
	If you simply want the predictions before and after the substitutions
	without the method turning those into attributions because, perhaps, you
	want to define your own aggregation method, you can use `raw_outputs=True`.
	
	
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
		If None, use all targets when calculating distances. Default is None.  
	
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

	X = X.type(torch.int8).cpu()
	y0 = predict(model, X, args=args, dtype=dtype, device=device)
	
	if end < 0:
		end = X.shape[-1] + 1 + end
	
	y_hat = []
	for i in trange(X.shape[0], disable=not verbose):
		X_ = X[i].repeat((end-start)*X[i].shape[0], 1, 1).numpy(force=True)
		_edit_distance_one(X_, start, end)
		X_ = torch.from_numpy(X_)
	
		if args is not None:
			args_ = tuple(a[i].repeat(X_.shape[0], *(1 for _ in a[i].shape)) 
				for a in args)
		else:
			args_ = None
	
		y_hat_ = predict(model, X_, args=args_, batch_size=batch_size, 
			dtype=dtype, device=device)
	
		y_hat.append(y_hat_)
	
	if isinstance(y_hat[0], torch.Tensor):
		y_hat = torch.stack(y_hat).reshape(X.shape[0], X.shape[1], end-start, 
			*y_hat_.shape[1:])
	else:
		y_hat = [
			torch.cat(y_).reshape(X.shape[0], X.shape[2], X.shape[1], 
				*y_[0].shape[1:]).transpose(2, 1) for y_ in zip(*y_hat)
		]
	
	if raw_outputs == False:
		attr = _attribution_score(y0, y_hat, target)
		if end <= 0:
			return X * attr if hypothetical == False else attr
		return X[:, :, start:end] * attr if hypothetical == False else attr
	return SaturationMutagenesisRawResult(y0=y0, y_hat=y_hat)
