# fimo.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import math
import numba
import numpy
import torch
import pandas

from ..io import read_meme
from ..predict import predict

from tqdm import tqdm


@numba.njit('float32(float32, float32)')
def logaddexp(x, y):
	"""Calculate the logaddexp in a numerically stable manner.

	This function is a fast implementation of the logaddexp function that
	operates on two numbers and is numerically stable. It should mimic the
	functionality of numpy.logaddexp except that it does not have the overhead
	of working on numpy arrays.


	Parameters
	----------
	x: float32
		A single number in log space.

	y: float32
		Another single number in log space.


	Returns
	-------
	z: float32
		The result of log(exp(x) + exp(y)) except in a numerically stable
		format.
	"""

	vmax, vmin = max(x, y), min(x, y)
	return vmax + math.log(math.exp(vmin - vmax) + 1)


@numba.njit('void(int32[:, :], float64[:], float64[:], int32, int32, float64)')
def _fast_pwm_to_cdf(int_log_pwm, old_logpdf, logpdf, alphabet_length, 
	seq_length, log_bg):
	"""A fast internal function for running the dynamic programming algorithm.

	This function is written in numba to speed up the dynamic programming used
	to convert score bins into log p-values. This is not meant to be used
	externally.
	"""

	for i in range(1, seq_length):
		logpdf[:] = -numpy.inf

		for j, x in enumerate(old_logpdf):
			if x != -numpy.inf:
				for k in range(alphabet_length):
					offset = int_log_pwm[i, k]

					v1 = logpdf[j + offset]
					v2 = log_bg + x
					logpdf[j + offset] = logaddexp(v1, v2)

		old_logpdf[:] = logpdf


def _pwm_to_mapping(log_pwm, bin_size):
	"""An internal method for calculating score <-> log p-value mappings.

	This function takes in a PWM consisting of log probabilities and outputs
	a mapping between observed scores (as a convolution of the PWM across a
	one-hot encoded sequence) and log p-values. This mapping is calculated 
	quickly using dynamic programming scanning over all potential sequences.

	Importantly, the p-values are in log space meaning that values near zero
	at the start of the array are insignificant whereas those with large 
	magnitude towards the end of the array are more statistically significant.


	Parameters
	----------
	log_pwm: numpy.ndarray, shape=(len(alphabet), length)
		A position-weight matrix containing a motif encoded as the log
		probability of any character in any position.

	bin_size: float
		The size of the score bins to map to p-values. The smaller this value,
		the more bins, indicating higher precision but also longer calculation
		time.


	Returns
	-------
	smallest: int
		The number of bins between true zero and the smallest value in the
		array. In other words, the offset to subtract from binned scores to get
		p-values.

	log1mcdf: numpy.ndarray
		The log of 1 minus the cdf, or in other words, the log p-values
		associated with each score bin.
	"""

	log_bg = math.log(0.25)
	int_log_pwm = numpy.round(log_pwm / bin_size).astype(numpy.int32).T.copy()

	smallest = int(numpy.min(numpy.cumsum(numpy.min(int_log_pwm, axis=-1), 
		axis=-1)))
	largest = int(numpy.max(numpy.cumsum(numpy.max(int_log_pwm, axis=-1), 
		axis=-1)))
	
	logpdf = -numpy.inf * numpy.ones(largest - smallest + 1)
	for i in range(log_pwm.shape[0]):
		idx = int_log_pwm[0, i] - smallest
		logpdf[idx] = numpy.logaddexp(logpdf[idx], log_bg)
	
	old_logpdf = logpdf.copy()
	logpdf[:] = 0
	
	_fast_pwm_to_cdf(int_log_pwm, old_logpdf, logpdf, log_pwm.shape[0], 
		log_pwm.shape[1], log_bg)

	log1mcdf = logpdf.copy()
	for i in range(len(logpdf) - 2, -1, -1):
		log1mcdf[i] = numpy.logaddexp(log1mcdf[i], log1mcdf[i + 1])

	return smallest, log1mcdf


class FIMO(torch.nn.Module):
	"""A motif hit caller that operates on sequences and attributions.

	This is a method for calling motif hits by scanning PWMs across one-hot
	encoded sequences as convolutions. One can get these scores directly, or
	summaries based on hits -- positions where the score goes above a certain
	p-value threshold, as calculated using dynamic programming a la FIMO.

	For efficiency, all PWMs are put in the same convolution operation 
	regardless of size and so are simultaneously scanned across all of the
	sequences, and p-values are only used to calculate thresholds on the score 
	rather than calculated for each position. 
	
	Because this method is implemented in torch, you can easily use a GPU to
	accelerate scanning and use half precision for additional speed ups (if
	your GPU supports half precision).

	There are a few ways to use this method:

		(1) Use the `.predict` method to get raw scores of each motif on each
		example on both strands.
		(2) Use the `.hits` method to get a pandas DataFrame in bed format
		showing all locations where the score is higher than the score
		threshold at the provided p-value threshold.
		(3) Use the `.hits` method with `axis=1` to get a pandas DataFrame in
		bed format showing all motif hits at each example.
		(4) Use the `.hit_matrix` method to get a matrix of size (n_examples,
		n_motifs) showing the maximum score for each motif in each example.

	
	Parameters
	----------
	motifs: str or dict
		A set of motifs to scan with. If a string, this is interpreted as a 
		filepath to a MEME file. If a dictionary, the keys are interpreted as 
		the motif names and the values are interpreted as the PWMs.
	
	alphabet : set or tuple or list, optional
		A pre-defined alphabet where the ordering of the symbols is the same
		as the index into the returned tensor, i.e., for the alphabet ['A', 'B']
		the returned tensor will have a 1 at index 0 if the character was 'A'.
		Characters outside the alphabet are ignored and none of the indexes are
		set to 1. This is not necessary or used if a one-hot encoded tensor is
		provided for the motif. Default is ['A', 'C', 'G', 'T'].

	bin_size: float, optional
		The bin size to use for the dynamic programming step when calculating 
		p-values. Default is 0.1.
	
	eps: float, optional
		A small value to add to a PWM to stabilize taking the log. Default is 
		1e-4.
	"""

	def __init__(self, motifs, alphabet=['A', 'C', 'G', 'T'], batch_size=256, 
		bin_size=0.1, eps=0.00005):
		super().__init__()
		self.bin_size = bin_size
		self.alphabet = numpy.array(alphabet)

		if isinstance(motifs, str):
			motifs = read_meme(motifs)
			
		self.motif_names = numpy.array([name for name in motifs.keys()])
		self.motif_lengths = numpy.array([len(motif) for motif in 
			motifs.values()])
		self.n_motifs = len(self.motif_names)
		
		motif_pwms = numpy.zeros((len(motifs), 4, max(self.motif_lengths)), 
			dtype=numpy.float32)
		bg = math.log(0.25)

		self._score_to_pval = []
		self._smallest = []
		for i, (name, motif) in enumerate(motifs.items()):
			motif_pwms[i, :, :len(motif)] = numpy.log(motif.T + eps) - bg
			smallest, mapping = _pwm_to_mapping(motif_pwms[i], bin_size)

			self._smallest.append(smallest)
			self._score_to_pval.append(mapping)

		self.motif_pwms = torch.nn.Parameter(torch.from_numpy(motif_pwms))
		self._smallest = numpy.array(self._smallest)

	def forward(self, X):
		"""Score a set of sequences.
		
		This method will run the PWMs against the sequences, reverse-complement 
		the sequences and run the PWMs against them again, and then return the 
		maximum per-position score after correcting for the flipping.
		
		
		Parameters
		----------
		X: torch.tensor, shape=(n, 4, length)
			A tensor containing one-hot encoded sequences.
		"""
		
		y_fwd = torch.nn.functional.conv1d(X, self.motif_pwms)
		y_bwd = torch.nn.functional.conv1d(X, torch.flip(self.motif_pwms, (1, 
			2)))
		return torch.stack([y_fwd, y_bwd]).permute(1, 2, 0, 3)
	
	@torch.no_grad()
	def hits(self, X, X_attr=None, threshold=0.0001, batch_size=256, dim=0, 
		device='cuda', verbose=False):
		"""Find motif hits that pass the given threshold.
		
		This method will first scan the PWMs over all sequences, identify where
		those scores are above the per-motif score thresholds (by converting
		the provided p-value thresholds to scores), extract those coordinates
		and provide the hits in a convenient format.


		Parameters
		----------
		X: torch.tensor, shape=(n, 4, length)
			A tensor containing one-hot encoded sequences.

		X_attr: torch.tensor, shape=(n, 4, length), optional
			A tensor containing the per-position attributions. The values in
			this tensor will be summed across all four channels in the positions
			found by the hits, so make sure that the four channels are encoding
			something summable. You may want to multiply X_attr by X before
			passing it in. If None, do not sum attributions.

		threshold: float, optional
			The p-value threshold to use when calling hits. Default is 0.0001.

		batch_size: int, optional
			The number of examples to process in parallel. Default is 256.

		dim: 0 or 1, optional
			The dimension to provide hits over. Similar to other APIs, one can
			view this as the dimension to remove from the returned results. If
			0, provide one DataFrame per motif that shows all hits for that
			motif across examples. If 1, provide one DataFrame per motif that
			shows all motif hits within each example.

		device: str or torch.device, optional
			The device to move the model and batches to when making predictions. 
			If set to 'cuda' without a GPU, this function will crash and must be 
			set to 'cpu'. Default is 'cuda'.

		verbose: bool, optional
			Whether to display a progress bar as examples are being processed.
			Default is False.


		Returns
		-------
		dfs: list of pandas.DataFrame
			A list of DataFrames containing hit calls, either with one per
			example or one per motif.
		"""

		n = self.n_motifs if dim == 0 else len(X)
		hits = [[] for i in range(n)]
		
		log_threshold = numpy.log(threshold)
		
		scores = predict(self, X, batch_size=batch_size, device=device)   
		score_thresh = torch.empty(1, scores.shape[1], 1, 1)
		for i in range(scores.shape[1]):
			idx = numpy.where(self._score_to_pval[i] < log_threshold)[0][0]
			score_thresh[0, i] = (idx + self._smallest[i]) * self.bin_size                               
		
		hit_idxs = torch.where(scores > score_thresh)        
		for idxs in tqdm(zip(*hit_idxs)):
			example_idx, motif_idx, strand_idx, pos_idx = idxs
			score = scores[(example_idx, motif_idx, strand_idx, pos_idx)].item()
			
			l = self.motif_lengths[motif_idx]
			start = pos_idx.item()
			if strand_idx == 1:
				end = start + max(self.motif_lengths)
				start = start + max(self.motif_lengths) - l
			else:
				end = start + l

			char_idxs = X[example_idx, :, start:end].argmax(axis=0).numpy(
				force=True)
			seq = ''.join(self.alphabet[char_idxs])
			strand = '+-'[strand_idx]

			if X_attr is not None:
				attr = X_attr[example_idx, :, start:end].sum(axis=1)
			else:
				attr = '.'
			
			entry_idx = example_idx.item() if dim == 0 else motif_idx.item()
			entry = entry_idx, start, end, strand, score, attr, seq
			
			idx = motif_idx if dim == 0 else example_idx
			hits[idx].append(entry)
		
		name = 'example_idx' if dim == 0 else 'motif'
		names = name, 'start', 'end', 'strand', 'score', 'attr', 'seq'        
		hits = [pandas.DataFrame(hits_, columns=names) for hits_ in hits]
		if dim == 1:
			for hits_ in hits:
				hits_['motif'] = self.motif_names[hits_['motif']]
		
		return hits
	
	@torch.no_grad()
	def hit_matrix(self, X, threshold=None, batch_size=256, device='cuda'):
		"""Return the maximum score per motif for each example.

		Parameters
		----------
		X: torch.tensor, shape=(n, 4, length)
			A tensor containing one-hot encoded sequences.

		X_attr: torch.tensor, shape=(n, 4, length), optional
			A tensor containing the per-position attributions. The values in
			this tensor will be summed across all four channels in the positions
			found by the hits, so make sure that the four channels are encoding
			something summable. You may want to multiply X_attr by X before
			passing it in. If None, do not sum attributions.

		batch_size: int, optional
			The number of examples to process in parallel. Default is 256.

		device: str or torch.device, optional
			The device to move the model and batches to when making predictions. 
			If set to 'cuda' without a GPU, this function will crash and must be 
			set to 'cpu'. Default is 'cuda'.


		Returns
		-------
		y_hat: torch.Tensor, shape=(X.shape[0], n_motifs)
			The score of the strongest motif hit in each sequence. Note that
			this is the raw score, not a p-value.
		"""

		y_hat = predict(self, X, batch_size=batch_size, device=device)
		return y_hat.max(dim=-1).values.max(dim=-1).values
