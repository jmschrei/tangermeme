# design.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import numpy
import torch

from .utils import one_hot_encode
from .ersatz import substitute
from .predict import predict


def greedy_substitution(model, X, motifs, y, loss=torch.nn.MSELoss(
	reduction='none'), mask=None, tol=1e-3, max_iter=-1, args=None, start=None, 
	alphabet=['A', 'C', 'G', 'T'], batch_size=32, device='cuda', verbose=False):
	"""Greedily add motifs to achieve a desired goal. 

	This design function will greedily add motifs to achieve a desired output
	from the model. Each round, the function will iterate through all possible
	motifs, substitute each one with the given spacing, and keep the one whose
	loss function is the smallest. This process will continue until either the
	maximum number of iterations is reached (at which point, `max_iter` motifs
	will have been inserted into the sequence) or the loss falls below `tol`.

	Accordingly, the choice of loss function and desired output from the model
	is crucial for good design. Usually, the loss function can be Euclidean
	distance, but for models with more complex outputs or for subtle design
	tasks one may want to use something else, such as Jensen-Shannon divergence.


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	X: torch.tensor, shape=(1, len(alphabet), length)
		A one-hot encoded sequence to use as the base for design. This must be
		a single sequence and has the first dimension for broadcasting reasons.

	motifs: list of strings
		A list of strings where each string is a motif that can be inserted into
		the sequence. These strings will be one-hot encoded according to the
		provided alphabet.

	y: torch.Tensor or list of torch.Tensors
		A tensor or list of Tensors providing the desired output from the model.
		The type and shape must be compatible with the provided loss function
		and comparable to the output from `model`. Each tensor should have a
		shape of (1, n) where n is the number of outputs from the model. The 
		first dimension is 1 to make broadcasting work correctly.

	loss: function
		This function must take in `y` and `y_hat` where `y` is the desired
		output from the model and `y_hat` is the current prediction from the
		model given the substitutions. By default, this is the 
		torch.nn.MSELoss().

	mask: torch.Tensor
		A mask on the outputs from the model to consider. True means to include
		the outputs in the loss, False means to exclude those outputs from the
		loss. If None, use all outputs. Default is None.

	spacing: int or list or tuple
		The spacing between the substituted motifs or the range of spacings
		to try when inserting each of the next motifs.

	tol: float
		A threshold on the amount of improvement necessary according to loss,
		where the procedure will stop once the improvement is below. Default
		is 1e-3.

	max_iter: int
		The maximum number of iterations to run before terminating the
		procedure. Set to -1 for no limit. Default is -1.

	args: tuple or list or None
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function.
		Default is None.

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

	batch_size: int, optional
		The number of examples to make predictions for at a time. Default is 32.

	device: str or torch.device
		The device to move the model and batches to when making predictions. If
		set to 'cuda' without a GPU, this function will crash and must be set
		to 'cpu'. Default is 'cuda'. 

	verbose: bool, optional
		Whether to display a progress bar during predictions. Default is False.


	Returns
	-------
	X: torch.Tensor, shape=(-1, len(alphabet), length)
		The edited sequence. 
	"""

	tic = time.time()
	iteration = 0

	y_orig = predict(model, X, args=args, batch_size=batch_size, device=device, 
		verbose=verbose)
	y = y.to(device)

	mask = mask if mask is not None else torch.ones(y_orig.shape[1], dtype=bool)
	mask = mask.to(device)

	loss_prev = loss(y[:, mask], y_orig[:, mask]).mean()
	loss_orig = loss_prev

	if verbose:
		print(("Iteration 0 -- Loss: {:4.4}, Improvement: N/A, Idx: N/A, " +
			"Time (s): 0s").format(loss_prev, time.time() - tic))

	while True:
		if iteration == max_iter:
			break

		tic = time.time()
		best_improvement, best_motif_idx, best_pos = 0, -1, -1
		for idx, motif in enumerate(motifs):
			X_ = torch.cat([substitute(X, motif, start=start, 
				alphabet=alphabet) for start in range(X.shape[-1] - len(motif))
			])

			y_hat = predict(model, X_, args=args, batch_size=batch_size, 
				device=device, verbose=False)
			
			loss_curr = loss(
				y[:, mask].expand_as(y_hat[:, mask]), 
				y_hat[:, mask],
			).mean(dim=tuple(range(1, len(y_hat.shape))))

			pos = loss_curr.argmin()
			loss_curr = loss_curr[pos]

			improvement = loss_prev - loss_curr
			if improvement > best_improvement:
				best_improvement = improvement
				best_motif_idx = idx
				best_pos = pos
				best_loss = loss_curr


		if best_motif_idx != -1:
			X = substitute(X, motifs[best_motif_idx], start=best_pos, 
				alphabet=alphabet)
			loss_prev = best_loss

			if verbose:
				print(("Iteration {} -- Loss: {:4.4}, Improvement: {:4.4}, " + 
					"Motif Idx: {}, Pos Idx: {}, Time (s): {:4.4}").format(
						iteration+1, best_loss, best_improvement, 
						best_motif_idx, best_pos, time.time() - tic))

		if best_improvement <= tol:
			break

		iteration += 1

	return X