# seqlet.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar 

import math
import numpy
import torch
import pandas

from .utils import _validate_input
from .utils import characters

from sklearn.isotonic import IsotonicRegression


def _laplacian_null(X_sum, num_to_samp=10000, random_state=1234):
	"""An internal function for calculating a null distribution.

	The TF-MoDISCo seqlet calling procedure works by constructing a null
	distribution of values using Laplacian distributions for the positive and
	negative values separately. This method constructs those Laplacian
	distributions and then samples a given number of values from them.


	Parameters
	----------
	X_sum: torch.Tensor, shape=(-1, length)
		A tensor of summed attribution values over the provided window size.

	num_to_samp: int
		The number of values to sample from the fit Laplacian distributions.

	random_state: int, optional
		The random state to use for the sampling. Default is 1234.


	Returns
	-------
	pos_values: torch.Tensor, shape=(-1,)
		A vector of sampled values from the Laplacian distribution fit to the
		positive values.

	neg_values: torch.Tensor, shape=(-1,)
		A vector of sampled values from the Laplacian distribution fit to the
		negative values.
	"""

	values = X_sum.flatten()

	hist, bin_edges = torch.histogram(values, bins=1000)
	peak = torch.argmax(hist)
	l_edge, r_edge = bin_edges[peak:peak+2]
	top_values = values[(l_edge < values) & (values < r_edge)]

	hist, bin_edges = torch.histogram(top_values, bins=1000)
	peak = torch.argmax(hist)
	mu = (bin_edges[peak] + bin_edges[peak+1]).item() / 2

	pos_values = values[values >= mu]
	neg_values = values[values <=mu]

	#Take the most aggressive lambda over all percentiles
	quantiles = torch.arange(19) * 5. / 100
	lq = -torch.log(1 - quantiles)
	pos_q = torch.quantile(a=pos_values, q=quantiles)
	neg_q = torch.quantile(a=neg_values, q=1-quantiles)

	pos_lambda = torch.max(lq / (pos_q - mu))
	neg_lambda = torch.max(lq / (torch.abs(neg_q - mu)))

	prob_pos = len(pos_values) / len(values) 
	urand = torch.from_numpy(numpy.random.RandomState(random_state).uniform(
		size=(num_to_samp, 2))).to(X_sum.device)

	icdf = numpy.log(1 - urand[:, 1])

	sampled_vals = torch.where(urand[:, 0] < prob_pos, -icdf / pos_lambda + mu, 
		mu + icdf / neg_lambda)
	return sampled_vals[sampled_vals >= 0], sampled_vals[sampled_vals < 0]


def _iterative_extract_seqlets(X_sum, window_size, flank, suppress):
	"""This internal function iteratively identifies seqlets.

	This function extracts seqlets from the signal by iteratively identifying 
	the position across all examples with maximum windowed attribution sum and
	zeroing out the attributions around that position to ensure another seqlet
	cannot be identified from the same position.


	Parameters
	----------
	X_sum: torch.Tensor, shape=(-1, length)
		A tensor of summed attribution values over the provided window size.

	window_size: int
		The size of the window of attribution values to sum over when
		identifying seqlets. This is not the only component of seqlet size but
		is the most important.

	flank: int
		A number of characters on either end of the window to add to each
		seqlet. This is done primarily to remove the effect of surrounding
		positions and not have overlapping seqlets.

	suppress: int
		The number of positions to the left and right of the maximum attribution
		position to zero out the attributions of.


	Returns
	-------
	seqlets: list
		A list of tuples containing the example index, the start of the seqlet,
		the end of the seqlet, and the sum of attributions within the seqlet.
	"""

	n, d = X_sum.shape
	seqlets = []
	for i in range(n):
		while True:
			argmax = numpy.argmax(X_sum[i], axis=0)
			max_val = X_sum[i, argmax]
			if max_val == -numpy.inf:
				break

			seqlet = i, argmax - flank, argmax + window_size + flank
			seqlets.append(seqlet)

			l_idx = int(max(numpy.floor(argmax + 0.5 - suppress), 0))
			r_idx = int(min(numpy.ceil(argmax + 0.5 + suppress), d))
			X_sum[i, l_idx:r_idx] = -numpy.inf 

	return seqlets


def _isotonic_thresholds(values, null_values, increasing, target_fdr, 
	min_frac_neg=0.95):
	"""This function uses an isotonic regression to find FDR thresholds.

	Given a set of attribution values summed over the provided window and a
	set of sampled negative values from the Laplacian distribution, find the
	threshold at which real attributions are separated from the null values at
	a given FDR threshold.

	As an implementation note, this method uses the scikit-learn
	IsotonicRegression function. This method seems to accept PyTorch tensors
	as input but these tensors will be moved over to the CPU to make sure
	they are compatible. Even if other operations are done on the GPU, the
	isotonic regression component must be done on the CPU.


	Parameters
	----------
	values: torch.Tensor, shape=(-1,)
		Sampled attribution window sums from the real data.

	null_values: torch.Tensor, shape=(-1,)
		Sampled values from the Laplacian distribution that represents the null
		data.

	increasing: bool
		Whether the data should be modeled as increasing or decreasing in size.

	target_fdr: float
		The FDR threshold to use to separate positive and negative attribution
		values.

	min_frac_neg: float, optional
		The minimum number of values that need to be assigned to the null
		distribution. Default is 0.95.


	Returns
	-------
	threshold: float
		A single value representing the threshold to use on attribution window
		sums to separate seqlets from background given the FDR threshold.
	"""

	n1, n2 = len(values), len(null_values)

	X = torch.cat([values, null_values], axis=0).cpu()
	y = torch.cat([torch.ones(n1), torch.zeros(n2)], axis=0)

	w = len(values) / len(null_values)
	sample_weight = torch.cat([torch.ones(n1), torch.ones(n2)*w], axis=0)

	model = IsotonicRegression(out_of_bounds='clip', increasing=increasing)
	model.fit(X, y, sample_weight=sample_weight)
	y_hat = torch.from_numpy(model.transform(values))


	min_prec_x = model.X_min_ if increasing else model.X_max_
	min_precision = max(model.transform([min_prec_x])[0], 1e-7)
	implied_frac_neg = -1 / (1 - (1 / min_precision))

	if (implied_frac_neg > 1.0 or implied_frac_neg < min_frac_neg):
		implied_frac_neg = max(min(1.0,implied_frac_neg), min_frac_neg)

	precisions = torch.minimum(torch.maximum(1 + implied_frac_neg*(
		1 - (1 / torch.maximum(y_hat, torch.tensor(1e-7)))), torch.tensor(0.0)), 
			torch.tensor(1.0))
	precisions[-1] = 1
	return values[precisions >= (1 - target_fdr)][0].item()


def tfmodisco_seqlets(X_attr, window_size=21, flank=10, target_fdr=0.2, 
	min_passing_frac=0.03, max_passing_frac=0.2, 
	weak_threshold_for_counting_sign=0.8, device='cuda'):
	"""Extract seqlets using the same procedure as TF-MoDISco does.

	This is the initial seqlet extraction step using the same procedure that
	TF-MoDISco does. Importantly, TF-MoDISco does several post-processing steps
	on these seqlets that are interleaved in the pattern identification 
	procedure so the final set of seqlets actually used by patterns in 
	TF-MoDISco will be smaller than the set that are returned here.

	There are a few small changes here compared with the code used in the
	TF-MoDISco repository.


	Parameters
	----------
	X_attr: torch.Tensor, shape=(-1, len(alphabet), length)
		A tensor of attribution values for each position in the sequence.
		The attributions here will be summed across the length of the alphabet
		so the values must be amenable to that. This means that, most likely,
		it should be attribution values multiplied by the one-hot encodings
		so only the present characters have attributions.

	window_size: int, optional
		The size of the window of attribution values to sum over when
		identifying seqlets. This is not the only component of seqlet size but
		is the most important. Default is 21.

	flank: int, optional
		A number of characters on either end of the window to add to each
		seqlet. This is done primarily to remove the effect of surrounding
		positions and not have overlapping seqlets. Default is 10.

	target_fdr: float, optional
		A FDR value to set on attribution score sums over windows when 
		separating called seqlets from background. Default is 0.2.

	min_passing_frac: float, optional
		Require that at least this proportion of windows pass seqlet 
		identification. Default is 0.03.

	max_passing_frac: float, optional 
		Require that no more than this proportion of windows pass seqlet
		identification. Default is 0.2.

	weak_threshold_for_counting_sign: float, optional
		A minimal threshold to use when setting the final threshold value
		separating seqlets from non-seqlets.


	Returns
	-------
	pos_seqlets: torch.Tensor, shape=(-1, 4)
		A tensor containing the example index, start position, end position,
		and attribution sum for each seqlet that passes the thresholds and
		has a positive attribution sum.

	neg_seqlets: torch.Tensor, shape=(-1. 4)
		A tensor containing the example index, start position, end position,
		and attribution sum for each seqlet that passes the thresholds and
		has a negative attribution sum.
	"""

	_validate_input(X_attr, "X_attr", shape=(-1, -1, -1)) 
	suppress = int(0.5*window_size) + flank

	X_sum = X_attr.sum(axis=1).unfold(-1, window_size, 1).sum(dim=-1)
	values = X_sum.flatten()
	if len(values) > 1000000:
		values = torch.from_numpy(numpy.random.RandomState(1234).choice(
			a=values, size=1000000, replace=False)).to(device)
	
	pos_values = values[values >= 0]
	neg_values = values[values < 0]
	pos_null_values, neg_null_values = _laplacian_null(X_sum)

	pos_threshold = _isotonic_thresholds(pos_values, pos_null_values, 
		increasing=True, target_fdr=target_fdr)
	neg_threshold = _isotonic_thresholds(neg_values, neg_null_values,
		increasing=False, target_fdr=target_fdr)

	values = torch.cat([pos_values, neg_values])
	frac_passing = (sum(values >= pos_threshold) + 
		sum(values <= neg_threshold)) / len(values)

	if frac_passing < min_passing_frac:
		pos_threshold = torch.quantile(torch.abs(values), q=1-min_passing_frac) 
		neg_threshold = -pos_threshold

	if frac_passing > max_passing_frac:
		pos_threshold = torch.quantile(torch.abs(values), q=1-max_passing_frac) 
		neg_threshold = -pos_threshold

	distribution = torch.sort(torch.abs(X_sum.flatten()))[0]

	transformed_pos_threshold = numpy.sign(pos_threshold) * numpy.searchsorted(
		distribution, v=abs(pos_threshold)) / len(distribution)
	transformed_neg_threshold = numpy.sign(neg_threshold) * numpy.searchsorted(
		distribution, v=abs(neg_threshold)) / len(distribution)

	idxs = (X_sum >= pos_threshold) | (X_sum <= neg_threshold)

	X_sum[idxs] = numpy.abs(X_sum[idxs])
	X_sum[~idxs] = -numpy.inf

	X_sum[:, :flank] = -numpy.inf
	X_sum[:, -flank:] = -numpy.inf

	seqlets = _iterative_extract_seqlets(X_sum=X_sum, window_size=window_size,
		flank=flank, suppress=suppress)

	#find the weakest transformed threshold used across all tasks
	weak_thresh = min(min(transformed_pos_threshold, 
		abs(transformed_neg_threshold)) - 0.0001, 
			weak_threshold_for_counting_sign)

	threshold = distribution[int(weak_thresh * len(distribution))]

	pos_seqlets, neg_seqlets = [], []
	for example_id, start, end, in seqlets:
		attr_flank = int(0.5 * ((end-start) - window_size))
		attr_start, attr_end = start + attr_flank, end - attr_flank

		x = X_attr[example_id, :, attr_start:attr_end]
		attr = x.sum(axis=(-1, -2)).item()
		seq = characters(abs(x))

		seqlet = example_id, start.item(), end.item(), '*', attr, attr, seq

		if attr > threshold:
			pos_seqlets.append(seqlet)
		elif attr < -threshold:
			neg_seqlets.append(seqlet)

	names = 'example_idx', 'start', 'end', 'strand', 'score', 'attr', 'seq'
	pos_seqlets = pandas.DataFrame(pos_seqlets, columns=names)
	neg_seqlets = pandas.DataFrame(neg_seqlets, columns=names)
	return pos_seqlets, neg_seqlets
