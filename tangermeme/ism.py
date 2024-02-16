# ism.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import itertools

from .predict import predict


def _edit_distance_one(X, start, end):
	"""An internal function for generating all sequences of edit distance 1

	This internal function, which is meant to be used for ISM, will take in a
	one-hot encoded sequence and return all sequences that have an edit distance
	of one. 


	Parameters
	----------
	X: torch.Tensor, shape=(4, sequence_length)
		A single one-hot encoded sequence.

	start: int
		The first nucleotide to begin making edits on, inclusive.

	end: int
		The end of the span. Edits are not made on this nucleotide at this
		index. Can be negative indexes.


	Returns
	-------
	X_: torch.Tensor, shape=(length*len(alphabet), len(alphabet), length)
		All one-hot encoded sequences that have an edit distance of 1 from the
		original sequence. 
	"""

	end = end if end >= 0 else X.shape[-1] + 1 + end
	X_ = X.repeat((end-start)*X.shape[0], 1, 1)

	coords = itertools.product(range(start, end), range(X.shape[0]))
	for i, (j, k) in enumerate(coords):
		X_[i, :, j] = 0
		X_[i, k, j] = 1

	return X_


@torch.no_grad
def saturation_mutagenesis(model, X, args=None, start=0, end=-1, batch_size=32,
	device='cuda', verbose=False):
	"""Performs in-silico saturation mutagenesis on a set of sequences.

	This function will perform in-silico saturation mutagenesis on a set of 
	sequences and return the predictions on the original sequences and each
	of the sequences with an edit distance of one on them.


	Parameters
	----------
	model: torch.nn.Module
		The PyTorch model to use to make predictions.

	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to make predictions for.

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
		The end of where to to make perturbations to the sequence,
		non-inclusive. Default is -1.

	batch_size: int, optional
		The number of examples to make predictions for at a time. Default is 32.

	device: str or torch.device, optional
		The device to move the model and batches to when making predictions. If
		set to 'cuda' without a GPU, this function will crash and must be set
		to 'cpu'. Default is 'cuda'. 

	verbose: bool, optional
		Whether to display a progress bar during predictions. Default is False.


	Returns
	-------
	y0: torch.Tensor or list/tuple of torch.Tensors
		The outputs from the model for the reference sequences.

	y_hat: torch.Tensor or list/tuple of torch.Tensors
		The outputs from the model for each of the perturbed sequences.
	"""

	y0 = predict(model, X, args=args, device=device)
	
	y_hat = []
	for i in range(X.shape[0]):
		X_ = _edit_distance_one(X[i], start, end)
		
		y_hat_ = predict(model, X_, args=args, batch_size=batch_size, 
			device=device, verbose=verbose)
		y_hat.append(y_hat_)

	if isinstance(y_hat[0], torch.Tensor):
		y_hat = torch.stack(y_hat).reshape(X.shape[0], X.shape[2], X.shape[1], 
			*y_hat_.shape[1:]).permute(0, 2, 1, 3)

	return y0, y_hat
