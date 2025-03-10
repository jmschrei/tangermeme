# design.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import numba
import numpy
import torch

from tqdm import tqdm

from .utils import one_hot_encode
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

def greedy_substitution(model, X, motifs, y, loss=torch.nn.MSELoss(
	reduction='none'), reverse_complement=True, input_mask=None, 
	output_mask=None, tol=1e-3, max_iter=-1, args=None, 
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
	y_orig = predict(model, X, args=args, batch_size=batch_size, device=device, 
		verbose=False)

	if reverse_complement:
		motifs = motifs + [rc(motif) for motif in motifs]

	if output_mask is None:
		output_mask = torch.ones(y_orig.shape[1], dtype=bool)

	if input_mask is None:
		input_mask = torch.ones(y_orig.shape[1], dtype=bool)

	loss_prev = loss(y[:, output_mask], y_orig[:, output_mask]).mean()
	loss_orig = loss_prev

	if verbose:
		print(("Iteration 0 -- Loss: {:4.4}, Improvement: N/A, Idx: N/A, " +
			"Time (s): 0s").format(loss_prev, time.time() - tic))

	while True:
		if iteration == max_iter:
			break

		tic = time.time()
		best_improvement, best_motif_idx, best_pos = 0, -1, -1
		for idx, motif in enumerate(tqdm(motifs, disable=not verbose)):
			motif_ohe = one_hot_encode(motif, alphabet=alphabet).numpy()
			
			input_mask_ = torch.clone(input_mask)
			input_mask_[-len(motif):] = False
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


def greedy_marginalize(model, X, motifs, y, loss=torch.nn.MSELoss(
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
	X: torch.Tensor, shape=(-1, len(alphabet), length)
		The edited sequence. 
	"""

	tic = time.time()
	iteration = 0

	X = torch.clone(X)
	y_orig = predict(model, X, args=args, batch_size=batch_size, device=device, 
		verbose=False)

	if reverse_complement:
		motifs = motifs + [rc(motif) for motif in motifs]

	motifs_ohe = [one_hot_encode(motif, alphabet=alphabet, 
		allow_N=True).unsqueeze(0) for motif in motifs]

	if output_mask is None:
		output_mask = torch.ones(y_orig.shape[1], dtype=bool)

	#loss_prev = loss(y[:, output_mask], y_orig[:, output_mask]).mean()
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
