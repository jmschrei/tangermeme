# design.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import heapq
import numba
import numpy
import torch

from tqdm import tqdm

from .utils import _cast_as_tensor
from .utils import one_hot_encode
from .utils import random_one_hot
from .utils import reverse_complement as rc

from .ersatz import substitute
from .predict import predict


@numba.njit(parallel=True, cache=True)
def _fast_tile_substitute(X, motif, idxs):
	"""This function takes a motif and inserts it at all possibilities"""
	
	n_alphabet, n_len = motif.shape
	for i in numba.prange(X.shape[0]):
		idx = idxs[i]
		for j in range(n_len):
			for k in range(n_alphabet):
				X[i, k, j+idx] = motif[k, j]


def screen(model, shape, y=None, loss=torch.nn.MSELoss(reduction='none'), tol=1e-3,
	max_iter=-1, args=None, n_best=1, alphabet=['A', 'C', 'G', 'T'], 
	batch_size=32, func=random_one_hot, additional_func_kwargs={}, dtype=None, 
	device='cuda', random_state=None, verbose=False):
	"""Screen randomly generated sequences and choose the best one.

	Potentially, the conceptually simplest method for design is to randomly
	generate a batch of examples and evaluate them using the provided model,
	keeping only the `n_best` top hits according to the loss function. This is
	called "screening", as one is "screening" a large pool of random potential
	designs for activity and keeping only those that appear good according to some
	loss function. 

	Although this function will likely be slow since each batch is independent
	from the others, i.e., you are not guaranteed to be getting closer to a
	goal with each step, you may be surprised by how good the generations are.


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	shape: tuple
		Dimensions for the randomly generated sequences, excluding the batch
		dimension. For a model expecting an input like (32, 4, 2114), where
		32 is the batch size, `shape` should be (4, 2114). 

	y: torch.Tensor or list of torch.Tensors or None
		A tensor or list of Tensors providing the desired output from the model.
		The type and shape must be compatible with the provided loss function
		and comparable to the output from `model`. Each tensor should have a
		shape of (1, n) where n is the number of outputs from the model. The 
		first dimension is 1 to make broadcasting work correctly. If None,
		simply choose the edit that yields the strongest response from the
		model. Default is None.

	loss: function, optional
		This function must take in `y` and `y_hat` where `y` is the desired
		output from the model and `y_hat` is the current prediction from the
		model given the substitutions. By default, this is the 
		torch.nn.MSELoss().

	tol: float, optional
		A threshold on how small the loss should be before terminating the screening
		procedure. Default is 1e-3.

	max_iter: int, optional
		The maximum number of iterations to run before terminating the procedure. 
		Set to -1 for no limit. Default is -1.

	args: tuple or list or None, optional
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function.
		Default is None.

	n_best: int, optional
		The number of sequences to return at the end, ranked from the lowest loss
		to the highest loss. Setting to 1 means only return the very best sequence
		observed across all generation batches. Default is 1.

	batch_size: int, optional
		The number of examples to make predictions for at a time. Default is 32.

	func: function, optional
		The function to use to generate sequences. The signature of this function
		must be that it takes in a tuple of the shape of the batch to generate, e.g. 
		(32, 4, 2114), and also a random state. Default is `random_one_hot`. 

	additional_func_kwargs: dict, optional
		Additional named arguments to pass into the function when it is called.
		This is provided as an alternate path to route arguments into the 
		function in case they overlap, name-wise, with those in this function,
		or if you want to be absolutely sure that the arguments are making
		their way into the function. Default is {}.

	dtype: str or torch.dtype or None, optional
		The dtype to use with mixed precision autocasting. If None, use the dtype of
		the *model*. This allows you to use int8 to represent large data sets and
		only convert batches to the higher precision, saving memory. Defailt is None.

	device: str or torch.device, optional
		The device to move the model and batches to when making predictions. If
		set to 'cuda' without a GPU, this function will crash and must be set
		to 'cpu'. Default is 'cuda'. 

	random_state: int or None, optional
		The random seed to use to ensure determinism of the generation function.
		If None, not deterministic. Default is None.

	verbose: bool, optional
		Whether to display a progress bar during predictions. Default is False.


	Returns
	-------
	X: torch.Tensor, shape=(n_best, len(alphabet), length)
		The screened examples with the lowest loss.
	"""

	if y is None:
		y = [9_999_999]

	y = _cast_as_tensor(y)
	
	pq = []
	iteration = 0

	while True:
		X = func((batch_size, *shape), random_state=random_state, 
			**additional_func_kwargs)

		y_hat = predict(model, X, dtype=dtype, device=device, args=args)

		current_loss = -loss(y, y_hat).numpy(force=True).sum(axis=1)
		for i in range(batch_size):
			l = current_loss[i]
			entry = [l, X[i]]

			if len(pq) < n_best:
				heapq.heappush(pq, entry)
			elif l > pq[0][0]:
				heapq.heappushpop(pq, entry)
			else:
				continue

			if verbose:
				print("Adding element with loss {:4.4}".format(l))

		iteration += 1
		if -pq[0][0] < tol or iteration == max_iter:
			break
		
		if random_state is not None:
			random_state += 1
    
	Xs = [heapq.heappop(pq)[1] for i in range(n_best)]
	X = torch.flip(torch.stack(Xs), dims=(0,))
	return X


def greedy_substitution(model, X, y=None, motifs=None, 
	loss=torch.nn.MSELoss(reduction='none'), reverse_complement=True, 
	input_mask=None, output_mask=None, tol=1e-3, max_iter=-1, args=None, 
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

	y: torch.Tensor or list of torch.Tensors or None
		A tensor or list of Tensors providing the desired output from the model.
		The type and shape must be compatible with the provided loss function
		and comparable to the output from `model`. Each tensor should have a
		shape of (1, n) where n is the number of outputs from the model. The 
		first dimension is 1 to make broadcasting work correctly. If None,
		simply choose the edit that yields the strongest response from the
		model. Default is None.

	motifs: list of strings or None
		A list of strings where each string is a motif that can be inserted into
		the sequence. These strings will be one-hot encoded according to the
		provided alphabet. If None, use the provided alphabet as the motifs to
		only change one character at a time. Default is None.

	loss: function, optional
		This function must take in `y` and `y_hat` where `y` is the desired
		output from the model and `y_hat` is the current prediction from the
		model given the substitutions. By default, this is the 
		torch.nn.MSELoss().

	reverse_complement: bool, optional
		Whether to augment the provided list of motifs with their reverse
		complements. This will double the runtime. Default is True.

	input_mask: torch.Tensor or None, optional
		A mask on input positions that can be the start of substitution. Any
		motif can be substituted in starting at each allowed position even if
		the contiguous span of the mask is shorter than the motif. True means
		that a motif can be substituted in starting at that position and False
		means that it cannot be. Default is None.

	output_mask: torch.Tensor or None, optional
		A mask on the outputs from the model to consider. True means to include
		the outputs in the loss, False means to exclude those outputs from the
		loss. If None, use all outputs. Default is None.

	tol: float, optional
		A threshold on the amount of improvement necessary according to loss,
		where the procedure will stop once the improvement is below. Default
		is 1e-3.

	max_iter: int, optional
		The maximum number of iterations to run before terminating the
		procedure. Set to -1 for no limit. Default is -1.

	args: tuple or list or None, optional
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function.
		Default is None.

	alphabet : set or tuple or list, optional
		A pre-defined alphabet where the ordering of the symbols is the same
		as the index into the returned tensor, i.e., for the alphabet ['A', 'B']
		the returned tensor will have a 1 at index 0 if the character was 'A'.
		Characters outside the alphabet are ignored and none of the indexes are
		set to 1. This is not necessary or used if a one-hot encoded tensor is
		provided for the motif. Default is ['A', 'C', 'G', 'T'].

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
	X: torch.Tensor, shape=(-1, len(alphabet), length)
		The edited sequence. 
	"""

	tic = time.time()
	iteration = 0

	X = torch.clone(X)
	
	if y is None:
		y = [[9_999_999]]
	
	y = _cast_as_tensor(y)
	if y.ndim == 1:
		y = y.unsqueeze(0)
	
	y_orig = predict(model, X, args=args, batch_size=batch_size, device=device, 
		verbose=False)

	if motifs is None:
		motifs = alphabet
		reverse_complement = False
	
	if reverse_complement:
		motifs = motifs + [rc(motif) for motif in motifs]

	if output_mask is None:
		output_mask = torch.ones(y_orig.shape[1], dtype=bool)

	if input_mask is None:
		input_mask = torch.ones(X.shape[-1], dtype=bool)

	loss_prev = loss(y[:, output_mask], y_orig[:, output_mask]).mean()
	loss_orig = loss_prev

	if verbose:
		print(("Iteration 0 -- Loss: {:4.4}, Improvement: N/A, Idx: N/A, " +
			"Time (s): 0s").format(loss_prev, time.time() - tic))

	while iteration < max_iter:
		tic = time.time()
		best_improvement, best_motif_idx, best_pos = 0, -1, -1
		for idx, motif in enumerate(tqdm(motifs, disable=not verbose)):
			motif_ohe = one_hot_encode(motif, alphabet=alphabet).numpy()
			
			input_mask_ = torch.clone(input_mask)
			if len(motif) > 1:
				input_mask_[-len(motif)+1:] = False
			input_idxs = torch.where(input_mask_ == True)[0].numpy()

			X_ = X.float().repeat(input_idxs.shape[0], 1, 1).numpy(force=True)
			_fast_tile_substitute(X_, motif_ohe, input_idxs)
			X_ = torch.from_numpy(X_).to(X.dtype)

			y_hat = predict(model, X_, args=args, batch_size=batch_size, 
				device=device, verbose=False)

			loss_curr = loss(
				y[:, output_mask].expand_as(y_hat[:, output_mask]), 
				y_hat[:, output_mask],
			).mean(dim=tuple(range(1, len(y_hat.shape))))
			
			pos = loss_curr.argmin()
			loss_curr = loss_curr[pos]
			
			improvement = loss_prev - loss_curr
			if improvement > best_improvement:
				best_improvement = improvement
				best_motif_idx = idx
				best_pos = input_idxs[pos]
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


def greedy_marginalize(model, X, y, motifs, loss=torch.nn.MSELoss(
	reduction='none'), max_spacing=12, reverse_complement=True, 
	output_mask=None, tol=1e-3, max_iter=-1, args=None, 
	alphabet=['A', 'C', 'G', 'T'], batch_size=32, device='cuda', verbose=False):
	"""Greedily builds a construct and evaluates it using marginalizations.

	This approach attempts to find a set of motifs and their orientations and
	spacings (a "construct") that yield a desired objective. Rather than 
	editing an initial sequence, this approach is just trying to greedily build
	a construct and evaluating its perform by marginalizing all other positions.
	Accordingly, rather than trying every motif at every position, it only tries
	motifs at all positions within a given spacing from the flanks of the
	construct that has been built so far.

	The algorithm proceeds like this: first, it implants each motif in the
	middle of each sequence and keeps the one that achieves the best
	improvement. Then, each motif is implanted between the left edge of the
	motif minus spacing and the right edge of the motif plus spacing (by default
	24 more nucleotides) and each position in that span (24 + orig motif length)
	are considered. This allows subsequent motifs to edit parts of previously
	implanted motifs while still significantly restricting the search space.

	This method is useful when you want to generally know what set of motifs
	and their other properties achieve your desired goal, without considering a
	specific sequence. For instance, if you train a model to predict
	accessibility and then want to design accessible regions generally, this
	approach may be more appropriate than `greedy_substiution`.


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	X: torch.tensor, shape=(1, len(alphabet), length)
		A one-hot encoded sequence to use as the base for design. This must be
		a single sequence and has the first dimension for broadcasting reasons.

	y: torch.Tensor or list of torch.Tensors
		A tensor or list of Tensors providing the desired output from the model.
		The type and shape must be compatible with the provided loss function
		and comparable to the output from `model`. Each tensor should have a
		shape of (1, n) where n is the number of outputs from the model. The 
		first dimension is 1 to make broadcasting work correctly.

	motifs: list of strings
		A list of strings where each string is a motif that can be inserted into
		the sequence. These strings will be one-hot encoded according to the
		provided alphabet.

	loss: function, optional
		This function must take in `y` and `y_hat` where `y` is the desired
		output from the model and `y_hat` is the current prediction from the
		model given the substitutions. By default, this is the 
		torch.nn.MSELoss().

	reverse_complement: bool, optional
		Whether to augment the provided list of motifs with their reverse
		complements. This will double the runtime. Default is True.

	output_mask: torch.Tensor or None, optional
		A mask on the outputs from the model to consider. True means to include
		the outputs in the loss, False means to exclude those outputs from the
		loss. If None, use all outputs. Default is None.

	tol: float, optional
		A threshold on the amount of improvement necessary according to loss,
		where the procedure will stop once the improvement is below. Default
		is 1e-3.

	max_iter: int, optional
		The maximum number of iterations to run before terminating the
		procedure. Set to -1 for no limit. Default is -1.

	args: tuple or list or None, optional
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function.
		Default is None.

	alphabet : set or tuple or list, optional
		A pre-defined alphabet where the ordering of the symbols is the same
		as the index into the returned tensor, i.e., for the alphabet ['A', 'B']
		the returned tensor will have a 1 at index 0 if the character was 'A'.
		Characters outside the alphabet are ignored and none of the indexes are
		set to 1. This is not necessary or used if a one-hot encoded tensor is
		provided for the motif. Default is ['A', 'C', 'G', 'T'].

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
	X: torch.Tensor, shape=(len(alphabet), length)
		The designed construct. 
	"""

	tic = time.time()
	iteration = 0

	y = _cast_as_tensor(y)
	if y.ndim == 1:
		y = y.unsqueeze(0)

	X = torch.clone(X)
	y_orig = predict(model, X, args=args, batch_size=batch_size, device=device, 
		verbose=False)

	if reverse_complement:
		motifs = motifs + [rc(motif) for motif in motifs]

	motifs_ohe = [one_hot_encode(motif, alphabet=alphabet, 
		allow_N=True).unsqueeze(0) for motif in motifs]

	if output_mask is None:
		output_mask = torch.ones(y_orig.shape[1], dtype=bool)

	loss_prev = loss(torch.zeros_like(y[:, output_mask]), 
		y[:, output_mask]).mean()
	loss_orig = loss_prev

	if verbose:
		print(("Iteration 0 -- Loss: {:4.4}, Improvement: N/A, Idx: N/A, " +
			"Time (s): 0s").format(loss_prev, time.time() - tic))


	left_flank = X.shape[-1] // 2
	right_flank = X.shape[-1] // 2

	chosen_motifs = []
	chosen_pos = []

	while True:
		if iteration == max_iter:
			break

		tic = time.time()
		best_improvement, best_motif_idx, best_pos = 0, -1, -1

		X_ = torch.clone(X) 
		for motif, pos in zip(chosen_motifs, chosen_pos):
			X_ = substitute(X_, motif, start=pos)

		for idx, motif_ohe in enumerate(tqdm(motifs_ohe, disable=not verbose)):
			if iteration == 0:
				lflank, rflank = left_flank - 1, right_flank
			else:
				lflank = left_flank - max_spacing - motif_ohe.shape[-1]
				rflank = right_flank + max_spacing + 1

			for pos in range(lflank, rflank):
				Xm_ = substitute(X_, motif_ohe, start=pos)

				y_hat = predict(model, Xm_, args=args, batch_size=batch_size,
					device=device, verbose=False) - y_orig

				loss_curr = loss(
					y[:, output_mask].expand_as(y_hat[:, output_mask]), 
					y_hat[:, output_mask],
				).mean()
				
				improvement = loss_prev - loss_curr
				if improvement > best_improvement:
					best_improvement = improvement
					best_motif = motifs[idx]
					best_motif_idx = idx
					best_pos = pos
					best_loss = loss_curr


		if best_motif_idx != -1:
			chosen_motifs.append(best_motif)
			chosen_pos.append(best_pos)
			loss_prev = best_loss

			left_flank = min(left_flank, best_pos)
			right_flank = max(right_flank, best_pos + len(best_motif))

			if verbose:
				print(("Iteration {} -- Loss: {:4.4}, Improvement: {:4.4}, " + 
					"Motif Idx: {}, Pos Idx: {}, Time (s): {:4.4}").format(
						iteration+1, best_loss, best_improvement, 
						best_motif_idx, best_pos, time.time() - tic))

		if best_improvement <= tol:
			break

		iteration += 1

	Xn = ['N' for _ in range(X.shape[-1])]
	for motif, pos in zip(chosen_motifs, chosen_pos):
		Xn[pos:pos+len(motif)] = list(motif)
	Xn = ''.join(Xn).strip("N")
	return one_hot_encode(Xn, alphabet=alphabet)
