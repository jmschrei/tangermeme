# product.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import itertools

from tqdm import tqdm


def _apply(func, model, X, func_args, args, batch_size, device, verbose):
	"""An internal function for applying a function to a batch of data."""

	X = torch.stack(X)
	args = [torch.stack(a) for a in args]

	y = func(model, X, *func_args, args=args, batch_size=batch_size, 
		device=device, verbose=False)

	return y


def apply_product(func, model, X, *func_args, args, batch_size=32, 
	device='cuda', verbose=False):
	"""Apply a function on the cartesian product between elements in X.

	This function will apply the provided function in a batched manner on the
	cartesian product between the elemnts in X. Specifically, if X is a tuple
	where X = (A, B), then the function will be applied to all elements of A
	crossed with all elements of B. Specifically, the function will be applied
	passing A_0 B_0 as input into the model, then A_0 B_1, then A_0 B_2... until
	A_n B_m where n and m are the sizes of A and B respectively.

	This is a general purpose wrapper function that can take any of the other
	functions in tangermeme and apply them in this maner. For instance, if one
	wanted to make predictions for a set of sequences for all of some external
	condition, they would pass in the `predict` function here.


	Parameters
	----------
	func: function
		A function, likely implemented in tangermeme, to apply in a batched
		manner across the product of examples.

	model: torch.nn.Module
		The PyTorch model to use to make predictions.

	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to make predictions for.

	args: tuple or list
		A set of additional arguments to pass into the model. Each element in
		`args` should be one tensor that is input to the model. The elements do
		not need to be the same size as each other as a product will be
		constructed over all of them, as well as with `X`. If you only want
		to use one value for an argument across all function applications 

	func_args: dict or None, optional
		A dictionary of additional arguments to pass into the function call.
		Default is {}.

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
	y: torch.Tensor or list/tuple of torch.Tensors
		The output from the model for each input example. The precise format
		is determined by the model. If the model outputs a single tensor,
		y is a single tensor concatenated across all batches. If the model
		outputs multiple tensors, y is a list of tensors which are each
		concatenated across all batches.
	"""

	model = model.to(device).eval()

	X_, y, args_ = [], [], [[] for _ in args] 
	for x in tqdm(itertools.product(X, *args), disable=not verbose):
		X_.append(x[0])

		for i, arg in enumerate(x[1:]):
			args_[i].append(arg)

		if len(X_) == batch_size:
			y_ = _apply(func, model, X_, func_args=func_args, args=args_, 
				batch_size=batch_size, device=device, verbose=verbose)
			y.append(y_)

			X_, args_ = [], [[] for _ in args]
	else:
		if len(X_) > 0:
			y_ = _apply(func, model, X_, func_args=func_args, args=args_, 
				batch_size=batch_size, device=device, verbose=verbose)
			y.append(y_)

	Xal = [len(X)] + [len(a) for a in args]

	# If there is only a single output, just concatenate the tensors
	if isinstance(y[0], torch.Tensor):
		yl = y[0].shape[1:]
		y = torch.cat(y).reshape(*Xal, *yl)
	else:
		_y = []

		# If either the function or the model have multiple outputs, but the
		# other has a single output, then concatenate tensors across the
		# outputs appropriately.
		if isinstance(y[0][0], torch.Tensor):
			for y_ in list(zip(*y)):
				yl = y_[0].shape[1:]
				_y.append(torch.cat(y_).reshape(*Xal, *yl))

		# If both the function and the model have multiple outputs then you
		# have to go one layer deeper when concatenating the tensors.
		else:
			for y_task in list(zip(*y)):
				_y.append([])

				for y_ in list(zip(*y_task)):
					yl = y_[0].shape[1:]
					_y[-1].append(torch.cat(y_).reshape(*Xal, *yl))

		y = _y

	return y
