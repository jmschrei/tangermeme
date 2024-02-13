# predict.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import itertools

from tqdm import trange


def predict(model, X, args=None, batch_size=32, device='cuda', verbose=False):
	"""Make batched predictions in a memory-efficient manner.

	This function will take a PyTorch model and make predictions from it using
	the forward function, with optional additional arguments to the model. The
	additional arguments must have the same batch size as the examples, and the
	i-th example will be given to the model with the i-th index of each
	additional argument. 

	Before starting predictions, the model is moved to the specified device. As 
	predictions are being made, each batch is also moved to the specified 
	device and then moved back to the CPU after predictions are made. This is
	to allow the function to work on massive numbers of examples that would
	not necessarily each fit in memory. If the batches themselves do not fit
	in memory, try lowering the batch size.


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
		Default is None.

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
	y: torch.Tensor or list/tuple of torch.Tensors
		The output from the model for each input example. The precise format
		is determined by the model. If the model outputs a single tensor,
		y is a single tensor concatenated across all batches. If the model
		outputs multiple tensors, y is a list of tensors which are each
		concatenated across all batches.
	"""

	model = model.to(device).eval()
	y = []

	with torch.no_grad():
		# Make batched predictions
		f = X.shape[0] + batch_size
		for start in trange(0, f, batch_size, disable=not verbose):
			end = start + batch_size
			X_ = X[start:end].to(device)

			if args is not None:
				args_ = [a[start:end].to(device) for a in args]
				y_ = model(X_, *args_)
			else:
				y_ = model(X_)

			# Move to the CPU
			if isinstance(y_, torch.Tensor):
				y_ = y_.cpu()
			elif isinstance(y_, (list, tuple)):
				y_ = tuple(yi.cpu() for yi in y_)
			else:
				raise ValueError("Cannot interpret output from model.")

			y.append(y_)


	# Concatenate the outputs
	if isinstance(y[0], torch.Tensor):
		y = torch.cat(y)
	else:
		y = [torch.cat(y_) for y_ in list(zip(*y))]

	return y


def predict_cross(model, X, args, batch_size=32, device='cuda', verbose=False):
	"""Make predictions for the cross between X and args.

	This function will make predictions for the cross between the elements in X
	and the elements in args. More specifically, for each element in X,
	predictions will be made for each element in args. This function is useful
	when you need to make predictions for the same sequence across a range of
	conditions, for instance.


	Parameters
	----------
	model: torch.nn.Module
		The PyTorch model to use to make predictions.

	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to make predictions for.

	args: tuple or list
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function.
		Default is None.

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

	y = []
	n_seqs, n_args = X.shape[0], len(args[0])
	seq_idxs, arg_idxs = list(zip(*itertools.product(range(n_seqs), 
		range(n_args))))

	model = model.to(device).eval()

	with torch.no_grad():
		f = len(seq_idxs) + batch_size
		for start in trange(0, f, batch_size, disable=not verbose):
			end = start + batch_size
			sidxs, aidxs = list(seq_idxs[start:end]), list(arg_idxs[start:end])

			X_ = X[sidxs].to(device)
			args_ = [a[aidxs].to(device) for a in args]
			y_ = model(X_, *args_)

			# Move to the CPU
			if isinstance(y_, torch.Tensor):
				y_ = y_.cpu()
			elif isinstance(y_, (list, tuple)):
				y_ = tuple(yi.cpu() for yi in y_)
			else:
				raise ValueError("Cannot interpret output from model.")

			y.append(y_)

	# Concatenate the outputs
	if isinstance(y[0], torch.Tensor):
		y = torch.cat(y).reshape(n_seqs, n_args, *y[0].shape[1:])
	else:
		y = [torch.cat(y_).reshape(n_seqs, n_args, *y_[0].shape[1:]) 
			for y_ in list(zip(*y))]


	return y