# space.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch

from .utils import _validate_input
from .utils import _cast_as_tensor
from .utils import one_hot_encode

from .ersatz import multisubstitute
from .predict import predict

from tqdm import tqdm


def space(model, X, motifs, spacing, start=None, alphabet=['A', 'C', 'G', 'T'], 
	func=predict, additional_func_kwargs={}, verbose=False, **kwargs):
	"""Runs a single spacing experiment and returns predictions.

	Given a predictive model, a set of motifs to insert and the spacings
	between them, and a set of background sequences, return the predictions 
	from the model when using the background sequences and after inserting 
	the motifs into the sequences.


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to have a motif inserted into.

	motifs: list of torch.tensor, shape=(-1, len(alphabet), motif_length)
		A list of strings or of one-hot encoded version of a short motif to 
		substitute into the set of sequences.

	spacing: torch.Tensor, shape=(-1, len(motifs)-1)
		A tensor specifying all spacings between motifs to consider. Each row
		in this tensor is a different combination of spacings between motifs
		and each column is the spacing between an adjacent pair of motifs.
		Specifically, the 1st column corresponds to the spacing between the
		first and second motif, the 2nd column corresponds to the spacing
		between the second and third motif, etc. 

	start: int or None, optional
		The starting position of where to insert the motif. If None, insert the
		motif into the middle of the sequence such that the middle of the motif
		occurs at the middle of the sequence. Default is None.

	alphabet : set or tuple or list, optional
		A pre-defined alphabet where the ordering of the symbols is the same
		as the index into the returned tensor, i.e., for the alphabet ['A', 'B']
		the returned tensor will have a 1 at index 0 if the character was 'A'.
		Characters outside the alphabet are ignored and none of the indexes are
		set to 1. This is not necessary or used if a one-hot encoded tensor is
		provided for the motif. Default is ['A', 'C', 'G', 'T'].

	func: function, optional
		A function to apply before and after making the substitutions. Default 
		is `predict`.

	additional_func_kwargs: dict, optional
		Additional named arguments to pass into the function when it is called.
		This is provided as an alternate path to route arguments into the 
		function in case they overlap, name-wise, with those in this function,
		or if you want to be absolutely sure that the arguments are making
		their way into the function. Default is {}.

	verbose: bool, optional
		Whether to display a progress bar as spacings are evaluated. Default
		is False.

	kwargs: optional
		Additional named arguments that will get passed into the function when
		it is called. Default is no arguments are passed in.


	Returns
	-------
	y_befores: torch.Tensor or list of torch.Tensors
		The predictions from the model before inserting the motif in. If the
		output from the model's forward function is a single tensor, it will
		return that. If the model outputs a list of tensors, it will return
		those.

	y_afters: torch.Tensor or list of torch.Tensors
		The predictions from the model after inserting the motif in. If the
		output from the model's forward function is a single tensor, it will
		return that. If the model outputs a list of tensors, it will return
		those.
	"""

	spacing = _validate_input(_cast_as_tensor(spacing, dtype=torch.int32), 
		"spacing", shape=(-1, len(motifs)-1))
	X = _validate_input(X, "X", shape=(-1, len(alphabet), -1))

	y_before = func(model, X, **kwargs, **additional_func_kwargs)
	y_afters = []

	for _spacing in tqdm(spacing, disable=not verbose):
		_spacing = [s.item() for s in _spacing]
		
		X_perturb = multisubstitute(X, motifs, _spacing, start=start, 
			alphabet=alphabet)

		y_after = func(model, X_perturb, **kwargs, **additional_func_kwargs)
		y_afters.append(y_after)

	if isinstance(y_afters[0], torch.Tensor):
		y_afters = torch.stack(y_afters).transpose(0, 1)
	else:
		y_afters = [torch.stack(y_).transpose(0, 1) for y_ in list(zip(
			*y_afters))]

	return y_before, y_afters
