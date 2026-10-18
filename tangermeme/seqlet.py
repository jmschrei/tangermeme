# seqlet.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar

from __future__ import annotations

import math
import warnings
from typing import NamedTuple

import numpy
import numba
import torch
import pandas

from .utils import _validate_input
from .utils import characters
from .utils import TangermemeWarning

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
		size=(num_to_samp, 2)))

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
		A list of 3-tuples `(example_index, start, end)`. The sum of
		attributions within the seqlet is NOT included in this output.
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


def tfmodisco_seqlets(
	X_attr: torch.Tensor,
	window_size: int = 21,
	flank: int = 10,
	target_fdr: float = 0.2,
	min_passing_frac: float = 0.03,
	max_passing_frac: float = 0.2,
	weak_threshold_for_counting_sign: float = 0.8,
) -> pandas.DataFrame:
	"""Extract seqlets using the procedure from TF-MoDISco.

	Seqlets are contiguous spans of high attribution characters. This method
	for identifying them is the one that is implemented in the TF-MoDISco
	algorithm. Importantly, TF-MoDISco does several post-processing steps
	on these seqlets that are interleaved in the pattern identification 
	procedure so the final set of seqlets actually used by patterns in 
	TF-MoDISco will be smaller than the set that are returned here.

	The seqlets returned by this procedure have been optimized to be useful
	for motif discovery, and so are generally much longer and less sensitive
	than one might initially expect. The seqlets are longer because the local
	context that patterns occur in might be useful, and because uninformative
	characters on the flanks can easily be trimmed off. The seqlets are also
	less sensitive, in the sense that sometimes spans that one might call a
	seqlet by eye are missed, to prevent noise from contaminating the found
	patterns.


	Parameters
	----------
	X_attr: torch.Tensor, shape=(-1, length)
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
		separating seqlets from non-seqlets. Default is 0.8.


	Returns
	-------
	seqlets: pandas.DataFrame, shape=(-1, 4)
		A tensor containing the example index, start position, end position,
		and attribution sum for each seqlet that passes the thresholds.
	"""

	_validate_input(X_attr, "X_attr", shape=(-1, -1))

	if X_attr.shape[0] == 0 or X_attr.shape[1] == 0:
		raise ValueError("tfmodisco_seqlets requires X_attr to be non-empty; "
			"got X_attr with shape {}.".format(tuple(X_attr.shape)))

	suppress = int(0.5*window_size) + flank

	X_sum = X_attr.unfold(-1, window_size, 1).sum(dim=-1)
	values = X_sum.flatten()
	if len(values) > 1000000:
		values = torch.from_numpy(numpy.random.RandomState(1234).choice(
			a=values, size=1000000, replace=False))
	
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

	if flank > 0:
		X_sum[:, :flank] = -numpy.inf
		X_sum[:, -flank:] = -numpy.inf

	seqlets = _iterative_extract_seqlets(X_sum=X_sum, window_size=window_size,
		flank=flank, suppress=suppress)

	#find the weakest transformed threshold used across all tasks
	weak_thresh = min(min(transformed_pos_threshold, 
		abs(transformed_neg_threshold)) - 0.0001, 
			weak_threshold_for_counting_sign)

	threshold = distribution[int(weak_thresh * len(distribution))]

	seqlets_ = []
	for example_id, start, end, in seqlets:
		attr_flank = int(0.5 * ((end-start) - window_size))
		attr_start, attr_end = start + attr_flank, end - attr_flank
		attr = X_attr[example_id, attr_start:attr_end].sum(dim=-1).item()

		seqlet = example_id, start.item(), end.item(), attr
		seqlets_.append(seqlet)

	names = 'example_idx', 'start', 'end', 'attribution'
	return pandas.DataFrame(seqlets_, columns=names)


###


@numba.njit
def _recursive_seqlet_null(values, xmin, bin_width, max_seqlet_len=25,
	n_bins=1000):
	"""An internal function building the null distributions of span sums.

	This covers the first two steps of the recursive seqlet algorithm:

	(1) Convert attribution scores into integer bins and calculate a histogram
	(2) Convert these histograms into null distributions across lengths

	The histogram is built from `values`, the flattened attributions. Returns
	`rcdfs`, where `rcdfs[k, i]` is P(S >= i) for S a sum of k binned values.
	"""

	###
	# Step 1: Calculate a histogram of binned scores
	###

	f = numpy.zeros(n_bins, dtype=numpy.float64)

	for i in range(values.shape[0]):
		x_bin = math.floor((values[i] - xmin) / bin_width)
		f[min(max(x_bin, 0), n_bins - 1)] += 1

	f = f / values.shape[0]

	###
	# Step 2: Calculate null distributions across lengths
	###

	scores = numpy.zeros((max_seqlet_len+1, n_bins*max_seqlet_len), 
		dtype=numpy.float64)
	scores[1, :n_bins] = f
	
	# rcdfs[k, i] is P(S >= i) for S a sum of k binned values. It is summed
	# down from the top because 1 - CDF loses the small tail probabilities to
	# rounding. The extra column is the zero that the sum starts from.
	rcdfs = numpy.zeros((max_seqlet_len+1, n_bins*max_seqlet_len+1),
		dtype=numpy.float64)

	for seqlet_len in range(1, max_seqlet_len+1):
		for i in range(n_bins * (seqlet_len-1)):
			for j in range(n_bins):
				scores[seqlet_len, i+j] += scores[seqlet_len-1, i] * f[j]

		for i in range(n_bins * seqlet_len - 1, -1, -1):
			rcdfs[seqlet_len, i] = rcdfs[seqlet_len, i+1] + scores[seqlet_len, i]

	return rcdfs


@numba.njit
def _recursive_seqlet_decode(X, rcdfs, xmin, bin_width, threshold=0.01,
	min_seqlet_len=4, max_seqlet_len=25, additional_flanks=0, n_bins=1000):
	"""An internal function calling seqlets given the null distributions.

	This covers the last two steps of the recursive seqlet algorithm:

	(3) Use the null distributions to calculate p-values for each possible length
	(4) Decode this matrix of p-values to find the longest seqlets

	Values of `X` outside the range the bins were built over are placed in the
	end bins.
	"""

	n, l = X.shape

	###
	# Step 3: Calculate p-values given these 1-CDFs
	###

	X_bin_csum = numpy.zeros((n, l+1), dtype=numpy.int64)
	X_csum = numpy.zeros((n, l+1))

	for i in range(n):
		for j in range(l):
			x_bin = math.floor((X[i, j] - xmin) / bin_width)
			X_bin_csum[i, j+1] = X_bin_csum[i, j] + min(max(x_bin, 0), n_bins - 1)
			X_csum[i, j+1] = X_csum[i, j] + X[i, j]


	###
	# Step 4: Decode p-values into seqlets
	###
	
	seqlets = []
	
	for i in range(n):
		p_value = numpy.ones((max_seqlet_len+1, l), dtype=numpy.float64)
		p_value[:min_seqlet_len] = 0

		for seqlet_len in range(min_seqlet_len, max_seqlet_len+1):
			for k in range(l-seqlet_len+1):
				x_ = X_bin_csum[i, k+seqlet_len] - X_bin_csum[i, k]
				p_value[seqlet_len, k] = max(rcdfs[seqlet_len, x_],
					p_value[seqlet_len-1, k])

		# Iteratively identify spans, from longest to shortest, that satisfy the
		# recursive p-value threshold.
		for j in range(max_seqlet_len - min_seqlet_len + 1):
			seqlet_len = max_seqlet_len - j

			while True:
				start = p_value[seqlet_len].argmin()
				p = p_value[seqlet_len, start]
				p_value[seqlet_len, start] = 1

				if p >= threshold:
					break

				for k in range(1, seqlet_len):
					if p_value[seqlet_len-k, start+k] >= threshold:
						break

				else:
					for end in range(start, start+seqlet_len):
						p_value[:, end] = 1

					# Report the span's own p-value, not the maximum over its
					# prefixes that p_value holds.
					x_ = X_bin_csum[i, start+seqlet_len] - X_bin_csum[i, start]
					p = rcdfs[seqlet_len, x_]

					end = min(start + seqlet_len + additional_flanks, l)
					start = max(start - additional_flanks, 0)
					attr = X_csum[i, end] - X_csum[i, start]
					seqlets.append((i, start, end, attr, p))

	return seqlets


def _as_attributions(X, name, dtype=None):
	"""Convert a tensor or array of attributions to a non-empty numpy array."""

	if isinstance(X, torch.Tensor):
		X = X.numpy()
	elif not isinstance(X, numpy.ndarray):
		raise ValueError("`{}` must be either a torch.Tensor or "
			"numpy.ndarray.".format(name))

	if X.size == 0:
		raise ValueError("recursive_seqlets requires {} to be non-empty; got "
			"{} with shape {}.".format(name, name, tuple(X.shape)))

	if dtype is not None:
		X = X.astype(dtype, copy=False)

	return X


def _recursive_seqlet_tables(X, max_seqlet_len, n_bins):
	"""Build the null distributions and the bins they are defined over.

	The bins span the values of `X`, and the histogram is built from them.
	Returns None when all of those values are equal.
	"""

	xmin, xmax = X.min(), X.max()
	if xmax == xmin:
		return None

	bin_width = float(xmax - xmin) / (n_bins - 1)
	rcdfs = _recursive_seqlet_null(X.ravel(), xmin, bin_width, max_seqlet_len,
		n_bins)
	return xmin, bin_width, rcdfs


def _recursive_seqlets_fdr(X, X_control, tables, min_seqlet_len,
	max_seqlet_len, n_bins, thresholds=None):
	"""Estimate the false discovery rate of recursive seqlet calls at each
	threshold, from the seqlets called on control attributions.

	The control is scored against the same null distributions and bins as
	`X`, and values outside the range of the bins are placed in the end
	bins. The estimate at a threshold is the number of control seqlets plus
	one, scaled to the number of positions in `X`, divided by the number of
	seqlets called on `X`, and NaN where none are called on `X`. Returns one
	row per threshold, from largest to smallest.
	"""

	if thresholds is None:
		thresholds = numpy.logspace(-1, -30, 117)
	thresholds = numpy.sort(numpy.asarray(thresholds, dtype=numpy.float64))[::-1]

	scale = X.size / X_control.size

	# With nothing called, nothing is masked, so a span passes exactly when
	# its own triangle of p-values is below the threshold. Any span passing a
	# stricter threshold would then have passed this one, so once neither
	# input has a seqlet, no stricter threshold gives one.
	exhausted = tables is None

	rows = []
	for threshold in thresholds:
		if exhausted:
			n_seqlets, n_control = 0, 0
		else:
			xmin, bin_width, rcdfs = tables
			n_seqlets = len(_recursive_seqlet_decode(X, rcdfs, xmin, bin_width,
				threshold, min_seqlet_len, max_seqlet_len, 0, n_bins))
			n_control = len(_recursive_seqlet_decode(X_control, rcdfs, xmin,
				bin_width, threshold, min_seqlet_len, max_seqlet_len, 0, n_bins))
			exhausted = n_seqlets == 0 and n_control == 0

		fdr = ((n_control + 1) * scale / n_seqlets if n_seqlets > 0
			else numpy.nan)
		rows.append((threshold, n_seqlets, n_control, fdr))

	return pandas.DataFrame(rows, columns=['threshold', 'n_seqlets',
		'n_control_seqlets', 'fdr'])


class RecursiveSeqletsResult(NamedTuple):
	"""Return type of `recursive_seqlets` with ``return_fdr=True``.
	Positional unpacking (``seqlets, fdr = recursive_seqlets(...)``) works
	alongside attribute access.

	`seqlets` is the frame of called seqlets. `fdr` has one row per threshold
	that was evaluated, from largest to smallest, with the number of seqlets
	called on `X` and on `X_control` and the estimated false discovery rate.
	"""

	seqlets: pandas.DataFrame
	fdr: pandas.DataFrame


def recursive_seqlets(
	X: torch.Tensor,
	threshold: float = 0.01,
	min_seqlet_len: int = 4,
	max_seqlet_len: int = 25,
	additional_flanks: int = 0,
	n_bins: int = 1000,
	X_control: torch.Tensor | None = None,
	target_fdr: float = 0.1,
	return_fdr: bool = False,
) -> pandas.DataFrame | RecursiveSeqletsResult:
	"""A seqlet caller implementing the recursive seqlet algorithm.

	NOTE: Currently only *positive* seqlets will be identified. To get negative
	seqlets, run this on the negated attributions (and the negated
	`X_control`, if one is used), negate the returned attribution sums, and
	drop any negative seqlet that overlaps a positive one. Running on the
	absolute value instead builds the null from both tails together, so
	negative seqlets that are weaker than the positive ones are never
	significant.

	This algorithm identifies spans of high attribution characters, called
	seqlets, using a simple approach derived from the Tomtom/FIMO algorithms.
	First, each attribution is discretized into one of `n_bins` integer bins,
	and the distribution of summed bins is created for all potential seqlet
	lengths by treating positions as independent draws from the histogram of
	all bins. Then, 1-CDFs are calculated for each distribution, giving the
	probability that a span of that length has a summed bin at least as large.
	Finally, p-values are calculated for all potential spans by looking up
	their summed bins in these 1-CDFs, yielding a (n_positions, n_lengths)
	matrix of p-values.

	These p-values rank spans; they are not error rates. Attributions are
	usually correlated along the sequence, so neighboring positions tend to
	be high together and spans reach large sums more often than independent
	draws would, which makes the p-values smaller than the chance of seeing
	such a span in sequence without real signal. The null is built from `X`,
	so it also contains the signal being tested, and a span's p-value
	depends on the other examples in `X`. As a result, a threshold of 0.01
	does not mean that 1% of the calls are false, and the same threshold can
	mean very different fractions of false calls in different models and
	datasets. To call seqlets at a target false discovery rate instead, pass
	control attributions as `X_control` (see below).

	This algorithm then identifies seqlets by defining them to have a key
	property: all internal spans of a seqlet must also have been called a
	seqlet. This means that all spans from `min_seqlet_len` to `max_seqlet_len`,
	starting at any position in the seqlet, and fully contained by the borders,
	must have a p-value below the threshold. Functionally, this means finding
	entries where the upper left triangle rooted in it is comprised entirely of
	values below the threshold. Graphically, for a candidate seqlet starting at
	X and ending at Y to be called a seqlet, all the values within the bounds
	(in addition to X) must also have a p-value below the threshold.

	::

		                        min_seqlet_len
		                         --------
		. . . . . . . | . . . . / . . . . . . . .
		. . . . . . . | . . . / . . . . . . . . .
		. . . . . . . | . . / . . . . . . . . . .
		. . . . . . . | . / . . . . . . . . . . .
		. . . . . . . | / . . . . . . . . . . . .
		. . . . . . . X . . . . . . . . Y . . . .
		. . . . . . . . . . . . . . . . . . . . .
		. . . . . . . . . . . . . . . . . . . . .


	The seqlets identified by this approach will usually be much smaller than
	those identified by the TF-MoDISco approach, including sometimes missing
	important characters on the flanks. You can set `additional_flanks` to
	a higher value if you want to include additional positions on either side.
	Importantly, the initial seqlet calls cannot overlap, but these additional
	characters are not considered when making that determination. This means
	that seqlets may appear to overlap when `additional_flanks` is set to a
	higher value.

	When `X_control` is given, the threshold is chosen from the data rather
	than passed in. Seqlets are called on `X_control`, attributions of
	sequences without real signal, with the same null distributions and bins
	as `X`, at each of 117 thresholds from 0.1 to 1e-30, four per decade.
	Every control seqlet is counted as false, so the estimated false
	discovery rate at a threshold is the number of control seqlets plus one,
	scaled to the number of positions in `X`, divided by the number of
	seqlets called on `X`. The added one keeps a strict threshold at which a
	few seqlets are called on `X` and, by chance, none on the control from
	passing as a false discovery rate of zero. Seqlets are then called on
	`X` at the largest threshold whose estimate is at or below `target_fdr`. Once a threshold calls no seqlet
	on `X` and none on `X_control`, no stricter threshold can call any, so
	the remaining thresholds are filled in without calling seqlets.

	A good control is the attributions of several dinucleotide shuffles of
	each sequence in `X`, computed with the same model, target and settings
	and concatenated along the first axis. The estimate at a threshold is
	only as precise as the number of control seqlets called there, so more
	shuffles give a steadier estimate, most of all at strict thresholds. A
	false call here is a seqlet that is no more frequent than in sequences
	with the same dinucleotide composition. Shuffling can create real motif
	matches that the model responds to, so the estimate may level off above
	zero as the threshold gets stricter. A target below that level is then
	reached by no threshold, and a target near it picks a threshold from a
	flat stretch of the curve, which moves a lot with the control drawn.
	Passing `return_fdr=True` returns the estimate at every threshold, which
	shows whether this happens.


	Parameters
	----------
	X: torch.Tensor or numpy.ndarray, shape=(-1, length)
		Attributions for each position in each example. The identity of the
		characters is not relevant for seqlet calling, so this should be the
		"projected" attributions, i.e., the attribution of the observed
		characters.

	threshold: float, optional
		The p-value threshold for calling seqlets. All positions within the
		triangle (as detailed above) must be below this threshold. This is a
		cutoff on a ranking score, not a false discovery rate; see above. It
		must be at most 1, and not NaN; at 0 or below, no seqlets are called.
		Ignored when `X_control` is given. Default is 0.01.

	min_seqlet_len: int, optional
		The minimum length that a seqlet must be, and the minimal length of
		span that must be identified as a seqlet in the recursive property.
		Default is 4.

	max_seqlet_len: int, optional
		The maximum length that a seqlet can be. Default is 25.

	additional_flanks: int, optional
		An additional value to subtract from the start, and to add to the end,
		of all called seqlets. Does not affect which seqlets are called, only
		extends their boundaries after calling. Default is 0.

	n_bins: int, optional
		The number of bins to use when estimating the PDFs and CDFs. Default is
		1000.

	X_control: torch.Tensor, numpy.ndarray, or None, optional
		Projected attributions, computed the same way as `X`, for sequences
		without real signal, such as dinucleotide-shuffled copies of the
		sequences in `X`. If given, the threshold is chosen to reach
		`target_fdr` as described above, and `threshold` is ignored. It may
		contain a different number of examples, or examples of a different
		length, than `X`; the counts are scaled by the number of positions.
		Values outside the range of `X` are placed in the end bins. If None,
		seqlets are called at `threshold`. Default is None.

	target_fdr: float, optional
		The estimated false discovery rate to call seqlets at when
		`X_control` is given. If no threshold reaches it, a
		`TangermemeWarning` gives the lowest estimate reached and no seqlets
		are returned. Ignored without `X_control`. Default is 0.1.

	return_fdr: bool, optional
		Whether to also return the estimated false discovery rate at every
		threshold. Requires `X_control`. Default is False.


	Returns
	-------
	seqlets: pandas.DataFrame, shape=(-1, 5)
		A BED-formatted dataframe containing the called seqlets, ranked from
		lowest p-value to higher p-value. The returned p-value is the p-value
		of the (location, length) span and is not influenced by the other
		values within the triangle. Use it to rank seqlets, not as the
		probability that a seqlet is false.

	fdr: pandas.DataFrame, shape=(117, 4)
		Only returned when `return_fdr=True`, as the second field of a
		`RecursiveSeqletsResult`. One row per threshold, from largest to
		smallest, with columns `threshold`, `n_seqlets` (called on `X`),
		`n_control_seqlets` (called on `X_control`) and `fdr`, the estimated
		false discovery rate described above, which is NaN where no seqlets
		are called on `X`. Seqlets are called at the largest threshold whose `fdr` is at or
		below `target_fdr`.
	"""

	X = _as_attributions(X, "X")
	if return_fdr and X_control is None:
		raise ValueError("return_fdr=True requires X_control, the control "
			"attributions that the false discovery rate is estimated from.")

	# The decoder stops at the first p-value at or above the threshold, and
	# no p-value is NaN or above 1, so such a threshold would never stop it.
	if X_control is None and not threshold <= 1:
		raise ValueError("threshold must be a p-value of at most 1; got "
			"{}.".format(threshold))

	columns = ['example_idx', 'start', 'end', 'attribution', 'p-value']

	# Every span of a constant input has the same sum, so none is significant,
	# and the bin width would be zero.
	tables = _recursive_seqlet_tables(X, max_seqlet_len, n_bins)

	fdr = None
	if X_control is not None:
		X_control = _as_attributions(X_control, "X_control", X.dtype)
		fdr = _recursive_seqlets_fdr(X, X_control, tables, min_seqlet_len,
			max_seqlet_len, n_bins)

		passing = fdr['fdr'] <= target_fdr
		if passing.any():
			threshold = fdr.loc[passing, 'threshold'].max()
		else:
			threshold = None
			estimated = fdr.dropna(subset=['fdr'])
			if len(estimated) > 0:
				best = estimated.loc[estimated['fdr'].idxmin()]
				reason = ("the lowest estimate is {:.3f}, at a threshold of "
					"{:.1e} with {} seqlets".format(best['fdr'],
					best['threshold'], int(best['n_seqlets'])))
			else:
				reason = "no seqlets are called on X at any threshold"

			warnings.warn("No threshold reaches an estimated false discovery "
				"rate at or below target_fdr={}: {}. No seqlets are returned; "
				"pass a higher target_fdr, and return_fdr=True to see the "
				"estimate at every threshold.".format(target_fdr, reason),
				TangermemeWarning, stacklevel=2)

	if tables is None or threshold is None:
		seqlets = []
	else:
		xmin, bin_width, rcdfs = tables
		seqlets = _recursive_seqlet_decode(X, rcdfs, xmin, bin_width, threshold,
			min_seqlet_len, max_seqlet_len, additional_flanks, n_bins)

	seqlets = pandas.DataFrame(seqlets, columns=columns)
	seqlets = seqlets.sort_values("p-value").reset_index(drop=True)

	if return_fdr:
		return RecursiveSeqletsResult(seqlets=seqlets, fdr=fdr)
	return seqlets
