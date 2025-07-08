# predict.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import itertools

from tqdm import trange


def predict(model, X, args=None, func=None, batch_size=32, dtype=None, 
	device='cuda', verbose=False):
	"""Make batched predictions in a memory-efficient manner.

	This function will take a PyTorch model and make predictions from it using
	the forward function, with optional additional arguments to the model. The
	additional arguments must have the same batch size as the examples, and the
	i-th example will be given to the model with the i-th index of each
	additional argument. 

	Before starting predictions, the model is moved to the specified device. As 
	predictions are being made, each batch is also moved to the specified 
	device and then moved back to the CPU after predictions are made. Each batch
	is converted to the provided dtype if provided, keeping the original blob of
	examples in the original dtype. These features allow the function to work on 
	massive data sets that do not fit in GPU memory. For example, the original
	sequences can be kept as 8-bit integers for compression and each batch will
	be upcast to the desired precision. If a single batch does not fit in memory,
	try lowering the batch size.


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

	func: function or None, optional 
		A function to apply to a batch of predictions after they have been made.
		If None, do nothing to them. Default is None.

	batch_size: int, optional
		The number of examples to make predictions for at a time. Default is 32.

	dtype: str or torch.dtype or None, optional
		The dtype to use with mixed precision autocasting. If None, use the dtype of
		the *model*. This allows you to use int8 to represent large data sets and
		only convert batches to the higher precision, saving memory. Defailt is None.

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
	
	if dtype is None:
		try:
			dtype = next(model.parameters()).dtype
		except:
			dtype = torch.float32
	elif isinstance(dtype, str):
		dtype = getattr(torch, dtype)

	if args is not None:
		for arg in args:
			if arg.shape[0] != X.shape[0]:
				raise ValueError("Arguments must have the same first " +
					"dimension as X")

	###

	y = []
	with torch.no_grad():
		batch_size = min(batch_size, X.shape[0])

		for start in trange(0, X.shape[0], batch_size, disable=not verbose):
			end = start + batch_size
			X_ = X[start:end].type(dtype).to(device)

			if X_.shape[0] == 0:
				continue

			with torch.autocast(device_type=device, dtype=dtype):
				if args is not None:
					args_ = [a[start:end].type(dtype).to(device) for a in args]
					y_ = model(X_, *args_)
				else:
					y_ = model(X_)

			# If a post-processing function is provided, apply it to the raw output
			# from the model.
			if func is not None:
				y_ = func(y_)

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
