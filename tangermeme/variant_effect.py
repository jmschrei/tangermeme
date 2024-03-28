# variant.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import itertools

from .io import extract_loci
from .utils import one_hot_encode

from .ersatz import delete
from .ersatz import insert

from .predict import predict
from .marginalize import marginalize


def marginal_substitution_effect(model, sequences, variants, args=None, 
	chroms=None, in_window=2114, n_loci=None, alphabet=['A', 'C', 'G', 'T'], 
	batch_size=32, device='cuda', verbose=False):
	"""Calculates the effect of each substitution individually.

	This function will take in a reference genome and a set of substitutions
	and calculate the individual effect of each variant by changing the
	reference sequence to incorporate one substitution at a time. 


	Parameters
	----------
	sequences: dict or str
		A dictionary where the keys are names, e.g. of chromosomes, and the 
		values are one-hot encoded sequences, or a string which is the filename
		of a FASTA file.

	variants: pandas.DataFrame or str
		A VCF-formatted data frame where the columns are the name (matching a
		key in X), the (IMPORTANT!) *0-indexed position* of the variant, and 
		the new character at that position. For this function, these variants are 
		restricted to substitutions.

	device: str or torch.device, optional
		The device to move the model and batches to when making predictions. If
		set to 'cuda' without a GPU, this function will crash and must be set
		to 'cpu'. Default is 'cuda'. 

	verbose: bool, optional
		Whether to display a progress bar during predictions. Default is False.


	Returns
	-------
	y: torch.Tensor, shape=(len(variants), n_outputs)
		The effect of the variant.
	"""

	X = extract_loci(variants[variants.columns[[0, 1, 1]]], sequences, 
		chroms=chroms, in_window=in_window, n_loci=n_loci, alphabet=alphabet, 
		verbose=verbose)

	X_var = torch.stack([one_hot_encode(v, alphabet=alphabet) for v in 
		variants[variants.columns[2]]])

	return marginalize(model, X, X_var, args=args, batch_size=batch_size, 
		device=device, verbose=verbose)


def marginal_deletion_effect(model, sequences, variants, args=None, 
	chroms=None, in_window=2114, n_loci=None, alphabet=['A', 'C', 'G', 'T'], 
	batch_size=32, device='cuda', verbose=False):
	"""Calculates the effect of each substitution individually.

	This function will take in a reference genome and a set of substitutions
	and calculate the individual effect of each variant by changing the
	reference sequence to incorporate one substitution at a time. 


	Parameters
	----------
	sequences: dict or str
		A dictionary where the keys are names, e.g. of chromosomes, and the 
		values are one-hot encoded sequences, or a string which is the filename
		of a FASTA file.

	variants: pandas.DataFrame or str
		A VCF-formatted data frame where the columns are the name (matching a
		key in X) and the (IMPORTANT!) *0-indexed position* of the variant.

	device: str or torch.device, optional
		The device to move the model and batches to when making predictions. If
		set to 'cuda' without a GPU, this function will crash and must be set
		to 'cpu'. Default is 'cuda'. 

	verbose: bool, optional
		Whether to display a progress bar during predictions. Default is False.


	Returns
	-------
	y: torch.Tensor, shape=(len(variants), n_outputs)
		The effect of the variant.
	"""

	X = extract_loci(variants[variants.columns[[0, 1, 1]]], sequences, 
		chroms=chroms, in_window=in_window+1, n_loci=n_loci, alphabet=alphabet, 
		verbose=verbose)

	mid = in_window // 2 + in_window % 2
	X_var = delete(X, mid, mid+1)
	X = X[:, :, :-1] if in_window % 2 == 0 else X[:, :, 1:]

	return marginalize(model, X, X_var, args=args, batch_size=batch_size, 
		device=device, verbose=verbose)


def marginal_insertion_effect(model, sequences, variants, args=None, 
	chroms=None, in_window=2114, n_loci=None, alphabet=['A', 'C', 'G', 'T'], 
	batch_size=32, device='cuda', verbose=False):
	"""Calculates the effect of each substitution individually.

	This function will take in a reference genome and a set of substitutions
	and calculate the individual effect of each variant by changing the
	reference sequence to incorporate one substitution at a time. 


	Parameters
	----------
	sequences: dict or str
		A dictionary where the keys are names, e.g. of chromosomes, and the 
		values are one-hot encoded sequences, or a string which is the filename
		of a FASTA file.

	variants: pandas.DataFrame or str
		A VCF-formatted data frame where the columns are the name (matching a
		key in X), the (IMPORTANT!) *0-indexed position* of the variant, and 
		the character to insert.

	device: str or torch.device, optional
		The device to move the model and batches to when making predictions. If
		set to 'cuda' without a GPU, this function will crash and must be set
		to 'cpu'. Default is 'cuda'. 

	verbose: bool, optional
		Whether to display a progress bar during predictions. Default is False.


	Returns
	-------
	y: torch.Tensor, shape=(len(variants), n_outputs)
		The effect of the variant.
	"""

	X = extract_loci(variants[variants.columns[[0, 1, 1]]], sequences, 
		chroms=chroms, in_window=in_window, n_loci=n_loci, alphabet=alphabet, 
		verbose=verbose)

	X_var = torch.stack([one_hot_encode(v, alphabet=alphabet) for v in 
		variants[variants.columns[2]]])

	X_var = insert(X, X_var)
	X_var = X_var[:, :, :-1]

	return marginalize(model, X, X_var, args=args, batch_size=batch_size, 
		device=device, verbose=verbose)