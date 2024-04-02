# ablate.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch

from .utils import one_hot_encode
from .ersatz import substitute
from .predict import predict

from .ersatz import shuffle


def ablate(model, X, start, end, n=20, shuffle_fn=shuffle, args=None, 
	batch_size=32, device='cuda', random_state=None, verbose=False):
	"""Make predictions before and after shuffling a region of sequences.

	An ablation experiment is one where a motif (or region of interest) is
	shuffled to remove any potential signal that could be in it. Predictions
	are returned before and after the region is shuffled and for a given
	number of shuffles.

	Ablation experiments can be thought of as the conceptual opposite of
	marginalization experiments. Both involve making predictions before and
	after some sequence modification, but marginalizations usually involve 
	substituting a potentially-informative motif into a set of background
	sequences, but an ablation usually involves removing drivers of signal
	from a sequence.


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to have a motif inserted into.

	start: int, optional
		The starting position of where to randomize the sequence, inclusive.
		Default is 0, shuffling the entire sequence.

	end: int, optional
		The ending position of where to randomize the sequence, not inclusive.
		Default is -1, shuffling the entire sequence.

	n: int, optional
		The number of times to shuffle that region. Default is 1.

	shuffle_fn: function
		A function that will shuffle a portion of the sequence. This can be
		`ersatz.shuffle`, `ersatz.dinucleotide_shuffle`, or any other function
		with the signature func(X, start, end, random_state) where `X` is a
		tensor with shape (-1, len(alphabet), length), `start` and `end` are
		coordinates on that sequence, and `random_state` is a seed to use to
		ensure determinism. Default is `ersatz.shuffle`. 

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

	batch_size: int, optional
		The number of examples to make predictions for at a time. Default is 32.

	device: str or torch.device
		The device to move the model and batches to when making predictions. If
		set to 'cuda' without a GPU, this function will crash and must be set
		to 'cpu'. Default is 'cuda'. 

	random_state: int, numpy.random.RandomState, or None, optional
		Whether to use a specific random seed when generating the shuffle to
		ensure reproducibility. If None, do not use a reproducible seed. Default 
		is None.

	verbose: bool, optional
		Whether to display a progress bar during predictions. Default is False.
 

	Returns
	-------
	y_before: torch.Tensor or list of torch.Tensors
		The predictions from the model before inserting the motif in. If the
		output from the model's forward function is a single tensor, it will
		return that. If the model outputs a list of tensors, it will return
		those.

	y_after: torch.Tensor or list of torch.Tensors
		The predictions from the model after inserting the motif in. If the
		output from the model's forward function is a single tensor, it will
		return that. If the model outputs a list of tensors, it will return
		those.s
	"""

	X_perturb = shuffle_fn(X, start=start, end=end, n=n, 
		random_state=random_state)

	args_n = None if args is None else tuple(a.repeat_interleave(n, dim=0) 
		for a in args)

	y_before = predict(model, X, args=args, batch_size=batch_size, 
		device=device, verbose=verbose)

	y_after = predict(model, X_perturb.reshape(-1, *X_perturb.shape[2:]), 
		args=args_n, batch_size=batch_size, device=device, 
		verbose=verbose)

	if isinstance(y_after, torch.Tensor):
		y_after = y_after.reshape(*X_perturb.shape[:2], *y_after.shape[1:])
	else:
		y_after = [y.reshape(*X_perturb.shape[:2], *y.shape[1:]) 
			for y in y_after]

	return y_before, y_after
