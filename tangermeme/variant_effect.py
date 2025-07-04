# variant.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import itertools

from .io import extract_loci

from .utils import one_hot_encode
from .utils import _cast_as_tensor

from .ersatz import delete
from .ersatz import insert

from .predict import predict
from .marginalize import marginalize


def substitution_effect(model, X, substitutions, args=None, func=predict, 
	additional_func_kwargs=None, **kwargs):
	"""Apply a function before and after including one or more substitutions.

	This function will calculate the effect that substitutions have on the
	output from a model. Any number of substitutions can be added to each
	sequence and the provided `func` is applied before and after these
	substitutions are included in the sequences. By default, this `func` is the
	prediction function and so the results are the difference in predictions
	before and after substitutions are made, but if `func` is something else, 
	such as `deep_lift_shap`, the results will be the attributions before and
	after substitutions are made. At least one substitution should be provided 
	per sequence for the results to be different.

	The substitutions provided must be individual variants, i.e., that each row 
	in the tensor corresponds to a single substitution in a single example, but
	one can encode longer variants (e.g., entire motifs or just multiple
	characters) by passing in multiple rows with adjacent positions. 

	Note that substitutions are not insertions. A substitution involves changing
	one character to another character. An insertion involves adding a new
	character into a sequence and requires trimming the edges afterward. Indels
	can be represented as substitutions as long as the indel is one insertion
	and also one deletion.


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to have substitutions included in.

	substitutions: torch.tensor, shape=(-1, 3)
		A set of variants that should be substituted into each sequence. This
		tensor is formatted like a COO-sparse matrix where each row is a single
		variant, the first column is the index in `X`, the second index is the
		position in that example, and the third index is the index into the
		alphabet that should be present at that position (overriding whatever
		is currently there). 

	args: tuple or None, optional
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function. This
		argument is provided here because the args must be copied for each
		shuffle that occurs. Default is None.

	func: function, optional
		A function to apply before and after incorporating the substitution. 
		Default is `predict`.

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
	y_before: torch.Tensor
		The output from `func` before variants are included.

	y_after: torch.Tensor
		The output from `func` after the variants are included.
	"""

	substitutions = _cast_as_tensor(substitutions)

	additional_func_kwargs = additional_func_kwargs or {}

	X_var = torch.clone(X)
	X_var[substitutions[:, 0], :, substitutions[:, 1]] = 0
	X_var[substitutions[:, 0], substitutions[:, 2], substitutions[:, 1]] = 1

	y_before = func(model, X, args=args, **additional_func_kwargs, **kwargs)
	y_after = func(model, X_var, args=args, **additional_func_kwargs, **kwargs)
	return y_before, y_after


def deletion_effect(model, X, deletions, left=False, args=None, func=predict,
	additional_func_kwargs=None, **kwargs):
	"""Apply a function before and after deleting characters from a sequence.

	This function will calculate the effect that insertions have on the
	output from a model. Any number of deletions can be specified for each
	sequence and the provided `func` is applied before and after the deletions
	are taken into account. By default, this `func` is the prediction function
	and so the results are the difference in predictions before and after the
	deletions are made, but if `func` is something else, such as 
	`deep_lift_shap`, the results will be the attributions before and after the
	deletions are made. At least one deletion should be provided per sequence
	for the results to be different.

	The deletions provided must be individual characters, i.e., that each row
	in the tensor corresponds to a single deletion in a single example, but one
	can encode longer deletions (e.g., entire motifs or just multiple 
	characters) by passing in multiple rows with adjacent positions.

	Importantly, because models assume a fixed input window but deletons are by
	definition changing the length of the sequence, the provided sequences must
	be of length `model_length + max_deletions_per_sequence`. Basically, if the
	maximum number of deletions in a sequence is equal to 12 and the model
	expects a tensor of length 100 then every sequence provided must be of
	length 112, even if there are fewer than 12 deletions in a particular
	sequence.

	Simply removing all of the specified characters will lead to a set of
	sequences of differing lengths if there are a different number of deletions
	in each sequence. In order to make these sequences all the same length, we
	need to trim from each sequence a number of positions such that in total
	(these additional bases + the number of deletions provided by the user)
	characters removed per example is the same for every example. Because there
	are two ways we can trim positions -- either starting from the left end of 
	the sequence or from the right end of it -- you can use the `left` parameter
	to specify that you should trim positions on the left (`left=True`) or from
	the right (`left=False`, the default) of the sequence. 


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	X: torch.tensor, shape=(-1, len(alphabet), length + max_deletions)
		A one-hot encoded set of sequences to have deletions included in.

	deletions: torch.tensor, shape=(-1, 2)
		A set of deletions indicating characters that should be removed from
		each sequence. Each row should be a single deletion with the first
		column corresponding to the index and the second column corresponding
		to the position within that index. Multiple deletions can occur in each
		example. 

	left: bool, optional
		If False, use the first `n` positions to run through the model before
		making the deletion where `n` is the expected tensor length. If True,
		use the last `n` positions. Basically, whether we trim positions from
		the left or the right when getting sequences of the same length.
		Default is False.

	args: tuple or None, optional
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function. This
		argument is provided here because the args must be copied for each
		shuffle that occurs. Default is None.

	func: function, optional
		A function to apply before and after incorporating the substitution. 
		Default is `predict`.

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
	y_before: torch.Tensor
		The output from `func` before variants are included.

	y_after: torch.Tensor
		The output from `func` after the variants are included.
	"""

	deletions = _cast_as_tensor(deletions)

	additional_func_kwargs = additional_func_kwargs or {}

	mask = torch.zeros_like(X[:, 0]).type(torch.int32)
	mask[deletions[:, 0], deletions[:, 1]] = 1
	
	counts = mask.sum(dim=-1)
	counts = abs(counts - counts.max())

	m = mask if left == True else torch.flip(mask, dims=(-1,))
	flank = torch.cumsum(1 - m, dim=-1) <= counts[:, None]
	mask += (flank if left == True else torch.flip(flank, dims=(-1,)))
	mask = (1 - mask).type(torch.bool)
	mask = mask[:, None].repeat(1, X.shape[1], 1) 

	X_var = X[mask].reshape(X.shape[0], X.shape[1], -1)

	if left == True:
		X = X[:, :, -X_var.shape[-1]:]
	else:
		X = X[:, :, :X_var.shape[-1]]

	y_before = func(model, X, args=args, **additional_func_kwargs, **kwargs)
	y_after = func(model, X_var, args=args, **additional_func_kwargs, **kwargs)
	return y_before, y_after


def insertion_effect(model, X, insertions, left=False, args=None, func=predict,
	additional_func_kwargs=None, **kwargs):
	"""Apply a function before and after inserting characters into a sequence.

	This function will calculate the effect that insertions have on the
	output from a model. Any number of insertions can be specified for each
	sequence and the provided `func` is applied before and after the insertions
	are taken into account. By default, this `func` is the prediction function
	and so the results are the difference in predictions before and after the
	insertions are made, but if `func` is something else, such as 
	`deep_lift_shap`, the results will be the attributions before and after the
	insertions are made. At least one insertion should be provided per sequence
	for the results to be different.

	The insertions provided must be individual characters, i.e., that each row
	in the tensor corresponds to a single insertion in a single example, but one
	can encode longer insertions (e.g., entire motifs or just multiple 
	characters) by passing in multiple rows with adjacent positions.

	Simply removing all of the specified characters will lead to a set of
	sequences of differing lengths if there are a different number of insertions
	in each sequence. In order to make these sequences all the same length, we
	need to trim from each sequence a number of positions from each sequence
	equal to the number of characters that are being added in. Because there
	are two ways we can trim positions -- either starting from the left end of 
	the sequence or from the right end of it -- you can use the `left` parameter
	to specify that you should trim positions on the left (`left=True`) or from
	the right (`left=False`, the default) of the sequence. 


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to have insertions included in.

	insertions: torch.tensor, shape=(-1, 3)
		A set of insertions indicating characters that should be added to
		each sequence. Each row should be a single insertion with the first
		column corresponding to the example and the second column corresponding
		to the position within that example. Multiple insertions can occur in 
		each example. 

	left: bool, optional
		If False, use the first `n` positions to run through the model before
		making the deletion where `n` is the expected tensor length. If True,
		use the last `n` positions. Basically, whether we trim positions from
		the left or the right when getting sequences of the same length.
		Default is False.

	args: tuple or None, optional
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function. This
		argument is provided here because the args must be copied for each
		shuffle that occurs. Default is None.

	func: function, optional
		A function to apply before and after incorporating the substitution. 
		Default is `predict`.

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
	y_before: torch.Tensor
		The output from `func` before variants are included.

	y_after: torch.Tensor
		The output from `func` after the variants are included.
	"""

	insertions = _cast_as_tensor(insertions)

	additional_func_kwargs = additional_func_kwargs or {}
	X_var = []

	for i in range(X.shape[0]):
		insertions_ = insertions[insertions[:, 0] == i]
		insertions_ = insertions_[torch.argsort(insertions_[:, 1], 
			descending=True)]

		x = X[i:i+1]
		for _, j, char in insertions_:
			v = torch.zeros(1, X.shape[1], 1)
			v[:, char] = 1
			x = insert(x, v, start=j)

		if left == True:
			x = x[:, :, -X.shape[-1]:]
		else:
			x = x[:, :, :X.shape[-1]]
		
		X_var.append(x)

	X_var = torch.cat(X_var)
	y_before = func(model, X, args=args, **additional_func_kwargs, **kwargs)
	y_after = func(model, X_var, args=args, **additional_func_kwargs, **kwargs)
	return y_before, y_after
