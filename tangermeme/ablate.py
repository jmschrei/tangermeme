# ablate.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import inspect

from .utils import _validate_input
from .ersatz import shuffle
from .predict import predict


def ablate(model, X, start, end, n=20, shuffle_fn=shuffle, args=None, 
	random_state=None, func=predict, additional_func_kwargs=None, **kwargs):
	"""Make predictions before and after shuffling a region of sequences.

	An ablation experiment is one where a motif (or region of interest) is
	shuffled to remove any potential signal that could be in it. Outputs
	are returned before and after the region is shuffled and for a given
	number of shuffles.

	Ablation experiments can be thought of as the conceptual opposite of
	marginalization experiments. Both involve applying a function before and
	after some sequence modification, but marginalizations usually involve 
	substituting a potentially-informative motif into a set of background
	sequences, but an ablation usually involves removing drivers of signal
	from a sequence.

	By default, `ablate` will apply the `predict` function to `X` before
	and after shuffling the given sequence. However, one can pass in any 
	function, including `deep_lift_shap` or even `saturated_mutagenesis`. These 
	functions may have additional arguments and those can be passed into 
	`marginalize` as-is and will be passed along to the function. If any 
	arguments would have had the same name as those used by this function, you 
	can use the `additional_func_kwargs` input to ensure those values get to 
	the function.

	Note: if `random_state` is passed in, it will make the shuffling step
	deterministic, but it will also be added to `additional_func_kwargs` if
	there is not already a key called `random_state` in it. Essentially,
	`random_state` makes shuffling deterministic and will also make the function
	deterministic if the function accepts a random state, but if you'd like to
	set your own separate state for the function it will not be overriden.


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

	args: tuple or None, optional
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function. This
		argument is provided here because the args must be copied for each
		shuffle that occurs. Default is None.

	random_state: int or None or numpy.random.RandomState, optional
		The random seed to use to ensure determinism of both the shuffling
		step and the function if the function also takes in a random state.
		If None, the run will not be deterministic. Default is None.

	func: function, optional
		A function to apply before and after making the ablation. Default 
		is `predict`.

	additional_func_kwargs: dict, optional
		Additional named arguments to pass into the function when it is called.
		This is provided as an alternate path to route arguments into the 
		function in case they overlap, name-wise, with those in this function,
		or if you want to be absolutely sure that the arguments are making
		their way into the function. Default is {}.

	kwargs: optional
		Additional named arguments that will get passed into the function when
		it is called. Default is no arguments are passed in.
 

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
		those.
	"""

	_validate_input(X, "X", shape=(-1, -1, -1), ohe=True, allow_N=True)
	additional_func_kwargs = additional_func_kwargs or {}
	if 'random_state' in inspect.signature(func).parameters.keys():
		if 'random_state' not in additional_func_kwargs:
			additional_func_kwargs['random_state'] = random_state


	X_perturb = shuffle_fn(X, start=start, end=end, n=n, 
		random_state=random_state)

	args_n = None if args is None else tuple(a.repeat_interleave(n, dim=0) 
		for a in args)

	y_before = func(model, X, args=args, **kwargs, **additional_func_kwargs)
	y_after = func(model, X_perturb.reshape(-1, *X_perturb.shape[2:]),
		args=args_n, **kwargs, **additional_func_kwargs)

	if isinstance(y_after, torch.Tensor):
		y_after = y_after.reshape(*X_perturb.shape[:2], *y_after.shape[1:])
	else:
		y_after = [y.reshape(*X_perturb.shape[:2], *y.shape[1:]) 
			for y in y_after]

	return y_before, y_after


def ablate_annotations(model, X, annotations, **kwargs):
	"""Ablate each annotation individually and return the deltas.

	This function takes in a model, a set of sequences, and a set of annotations
	and goes through the annotations one at a time ablating the sequence. The
	model predictions before and after ablation are returned, similar to the
	saturation_mutagenesis function. Each ablation is done individually, so
	the difference in model predictions is from just one annotation at a time.


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to have a motif inserted into.

	annotations: torch.Tensor, shape=(n_annotations, 3)
		A tensor of annotations where the first column is the example_idx, the
		second column is the start position (0-indexed) and the third column is
		the end position (0-indexed, not inclusive).

	kwargs: arguments
		Additional optional arguments to pass into the `ablate` function.

	
	Returns
	-------
	y_befores: torch.Tensor or list of torch.Tensors
		The application of `func` from the model BEFORE ablating the motif. If 
		the output from the model's forward function is a single tensor, it will 
		return that. If the model outputs a list of tensors, it will return 
		those.

	y_afters: torch.Tensor or list of torch.Tensors
		The application of `func` from the model AFTER ablating the motif. If 
		the output from the model's forward function is a single tensor, it will
		return that. If the model outputs a list of tensors, it will return
		those.
	"""

	y_befores, y_afters = [], []

	for idx, start, end in annotations:
		y_before, y_after = ablate(model, X[idx:idx+1], start=start, end=end, 
			**kwargs)

		y_befores.append(y_before)
		y_afters.append(y_after)

	if isinstance(y_afters[0], torch.Tensor):
		y_befores = torch.stack(y_befores)
		y_afters = torch.stack(y_afters)
	else:
		y_befores = [torch.stack([x[i] for x in y_befores]) for i in range(len(
			y_befores))]
		y_afters = [torch.stack([x[i] for x in y_afters]) for i in range(len(
			y_afters))]

	return y_befores, y_afters
