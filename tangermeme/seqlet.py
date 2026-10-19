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

from llvmlite import ir
from numba import types
from numba.extending import intrinsic

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
		A dataframe containing the example index, start position, end position,
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


# Outputs of the null's convolution per block. At 32 or fewer, LLVM unrolls
# the block's loop fully and leaves it scalar.
_NULL_BLOCK = 64

# The null's histogram bins _NULL_CHUNK values at a time in a loop of their
# own, which vectorizes, and counts value j into histogram j mod _NULL_SPLIT
# (a power of two), so that consecutive values in one bin do not wait on each
# other's store.
_NULL_CHUNK = 2048
_NULL_SPLIT = 2


@intrinsic
def _stack_float64(typingctx, n):
	"""A pointer to `n` float64 values on the stack of the calling function,
	for a literal integer `n`. LLVM keeps such an array in registers across
	a loop when every access is at a constant offset; it cannot do that for
	a numpy array, which may alias the arrays the loop reads.
	"""

	if not isinstance(n, types.IntegerLiteral):
		return None

	def codegen(context, builder, signature, args):
		with builder.goto_entry_block():
			arr = builder.alloca(ir.ArrayType(ir.DoubleType(), n.literal_value))
		return builder.bitcast(arr, ir.DoubleType().as_pointer())

	return types.CPointer(types.float64)(n), codegen


@intrinsic
def _float64_ptr(typingctx, arr, o):
	"""A pointer to `arr[o]`, for a C-contiguous 1D float64 array `arr`.
	Unlike a slice, it computes no bounds: the caller keeps `o` in range.
	"""

	if not (isinstance(arr, types.Array) and arr.ndim == 1 and arr.layout == 'C'
		and arr.dtype == types.float64 and isinstance(o, types.Integer)):
		return None

	def codegen(context, builder, signature, args):
		ary = context.make_array(signature.args[0])(context, builder, args[0])
		return builder.gep(ary.data, [args[1]])

	return types.CPointer(types.float64)(arr, o), codegen


@numba.njit
def _recursive_seqlet_null_bins(x, xmin, bin_width, n_bins, x_bin):
	"""Write to `x_bin` the place of each value of `x` in the null's split
	histogram: its bin, clamped to the end bins as in _recursive_seqlet_bins,
	plus n_bins times (j mod _NULL_SPLIT).
	"""

	for j in range(x.shape[0]):
		b = math.floor((x[j] - xmin) / bin_width)
		x_bin[j] = min(max(b, 0), n_bins - 1) + (j & (_NULL_SPLIT - 1)) * n_bins


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

	# The counts are integers below 2**53, so the float64 sum of the split
	# histograms equals the float64 count of each bin, in any order.
	n_values = values.shape[0]
	x_bin = numpy.empty(_NULL_CHUNK, dtype=numpy.int64)
	counts = numpy.zeros(_NULL_SPLIT * n_bins, dtype=numpy.int64)

	for c0 in range(0, n_values, _NULL_CHUNK):
		m = min(_NULL_CHUNK, n_values - c0)
		_recursive_seqlet_null_bins(values[c0:c0 + m], xmin, bin_width, n_bins,
			x_bin)

		# The unsigned index (every place is >= 0) drops a wraparound check.
		for j in range(m):
			counts[numpy.uint64(x_bin[j])] += 1

	f = numpy.zeros(n_bins, dtype=numpy.float64)
	for k in range(n_bins):
		count = 0
		for s in range(_NULL_SPLIT):
			count += counts[s * n_bins + k]
		f[k] = count

	f = f / values.shape[0]

	###
	# Step 2: Calculate null distributions across lengths
	###

	scores = numpy.zeros((max_seqlet_len+1, n_bins*max_seqlet_len), 
		dtype=numpy.float64)
	scores[1, :n_bins] = f
	
	# Summed down from the top because 1 - CDF loses the small tail
	# probabilities to rounding. The extra column is the zero the sum starts
	# from.
	rcdfs = numpy.zeros((max_seqlet_len+1, n_bins*max_seqlet_len+1),
		dtype=numpy.float64)

	# Each row is the previous one convolved with f, in blocks of B outputs
	# accumulated in registers. Each output sums prev[i] * f[m - i] over
	# ascending i from +0.0, without fused multiply-adds; the zero padding
	# that lets a block read whole slices adds only +0.0 terms, which leave
	# the sums unchanged. Each slice is read from one of eight copies of its
	# padded array, copy s shifted s values left, so that the slice starts
	# on a 64-byte boundary.
	#
	# When f is finite, the sums run over the nonzero bins j of f in
	# descending order (ascending i), skipping the +0.0 terms. The previous
	# row is read from one of two sets of shifted copies and the row being
	# built is written into the other; a set is next written by a longer
	# row, so its padding stays zero. Empty input makes f NaN (0/0), which
	# must reach every output, so it keeps the loop over i.
	B = _NULL_BLOCK
	nz_j = numpy.empty(n_bins, dtype=numpy.int64)
	nz_f = numpy.empty(n_bins, dtype=numpy.float64)
	n_nz = 0
	finite = True
	for j in range(n_bins - 1, -1, -1):
		v = f[j]
		if not numpy.isfinite(v):
			finite = False
		if v != 0.0:
			nz_j[n_nz] = j
			nz_f[n_nz] = v
			n_nz += 1

	if finite:
		W = (n_bins * max(max_seqlet_len - 1, 1) + 2 * B + 7) // 8 * 8
		n_sets = 2
	else:
		W = (n_bins + 2 * B + 7) // 8 * 8
		n_sets = 1

	buf = numpy.zeros(n_sets * 8 * W + 8, dtype=numpy.float64)
	s0 = (64 - buf.ctypes.data % 64) % 64 // 8
	fp = buf[s0:s0 + 8 * W]
	for s in range(8):
		for j in range(n_bins):
			fp[s * W + B + j - s] = f[j]

	# The slice for the bin nz_j[t] in the block at m0 starts at nz_o[t] + m0
	# in a set: in copy (B + m0 - j) & 7, and m0 is a multiple of 8.
	nz_o = numpy.empty(n_nz, dtype=numpy.int64)
	for t in range(n_nz):
		j = nz_j[t]
		s = (B - j) & 7
		nz_o[t] = s * W + B - j - s

	acc = numba.carray(_stack_float64(_NULL_BLOCK), _NULL_BLOCK)

	for seqlet_len in range(1, max_seqlet_len+1):
		prev = scores[seqlet_len-1]
		cur = scores[seqlet_len]
		n_prev = n_bins * (seqlet_len-1)
		n_out = n_prev + n_bins - 1 if seqlet_len > 1 else 0

		if finite and n_out > 0:
			r0 = s0 + (seqlet_len % 2) * 8 * W
			rd = buf[r0:r0 + 8 * W]
			w0 = s0 + (1 - seqlet_len % 2) * 8 * W
			wr = buf[w0:w0 + 8 * W]
			more = seqlet_len < max_seqlet_len

			# nz_j[t_hi:t_lo] are the nonzero bins in [m0 - n_prev + 1,
			# m0 + B - 1], the bins that reach some output of the block.
			t_hi = n_nz
			t_lo = n_nz
			for m0 in range(0, n_out, B):
				for b in range(B):
					acc[b] = 0.0

				while t_hi > 0 and nz_j[t_hi - 1] <= m0 + B - 1:
					t_hi -= 1
				while t_lo > 0 and nz_j[t_lo - 1] < m0 - n_prev + 1:
					t_lo -= 1

				# The unsigned index (t >= 0) drops a wraparound check per t.
				for t in range(t_hi, t_lo):
					a = nz_f[numpy.uint64(t)]
					o = nz_o[numpy.uint64(t)] + m0
					pv = numba.carray(_float64_ptr(rd, o), _NULL_BLOCK)
					for b in range(B):
						acc[b] += pv[b] * a

				n_b = min(B, n_out - m0)
				for b in range(n_b):
					cur[m0 + b] = acc[b]

				if more:
					for s in range(8):
						o = s * W + B + m0 - s
						wv = wr[o:o + n_b]
						for b in range(n_b):
							wv[b] = acc[b]

		else:
			for m0 in range(0, n_out, B):
				for b in range(B):
					acc[b] = 0.0

				# The unsigned index (i >= 0) drops a wraparound check per i.
				for i in range(max(m0 - n_bins + 1, 0), min(m0 + B, n_prev)):
					a = prev[numpy.uint64(i)]
					st = B + m0 - i
					s = st & 7
					o = s * W + st - s
					fv = fp[o:o + B]
					for b in range(B):
						acc[b] += a * fv[b]

				for b in range(min(B, n_out - m0)):
					cur[m0 + b] = acc[b]

		for i in range(n_bins * seqlet_len - 1, -1, -1):
			rcdfs[seqlet_len, i] = rcdfs[seqlet_len, i+1] + scores[seqlet_len, i]

	return rcdfs


@numba.njit
def _sort_candidates(cand_p, cand_k, m, tmp_p, tmp_k):
	"""Sort cand_p[:m] in increasing order, moving cand_k[:m] with it.

	The sort is stable, so entries gathered in order of start come out in
	order of (p-value, start), the order in which the decoder visits them.
	Runs of 16 are sorted by insertion, then merged in passes between the
	candidates and the scratch arrays tmp_p and tmp_k, of size at least m.
	"""

	for lo in range(0, m, 16):
		hi = min(lo + 16, m)
		for a in range(lo + 1, hi):
			p, k = cand_p[a], cand_k[a]
			b = a
			while b > lo and p < cand_p[b-1]:
				cand_p[b] = cand_p[b-1]
				cand_k[b] = cand_k[b-1]
				b -= 1

			cand_p[b] = p
			cand_k[b] = k

	src_p, src_k, dst_p, dst_k = cand_p, cand_k, tmp_p, tmp_k
	in_tmp = False
	width = 16
	while width < m:
		for lo in range(0, m, 2 * width):
			mid = min(lo + width, m)
			hi = min(lo + 2 * width, m)
			a, b, c = lo, mid, lo
			while a < mid and b < hi:
				pa, pb = src_p[a], src_p[b]
				ka, kb = src_k[a], src_k[b]
				right = pb < pa
				dst_p[c] = pb if right else pa
				dst_k[c] = kb if right else ka
				step = 1 if right else 0
				b += step
				a += 1 - step
				c += 1

			while a < mid:
				dst_p[c] = src_p[a]
				dst_k[c] = src_k[a]
				a += 1
				c += 1

			while b < hi:
				dst_p[c] = src_p[b]
				dst_k[c] = src_k[b]
				b += 1
				c += 1

		src_p, src_k, dst_p, dst_k = dst_p, dst_k, src_p, src_k
		in_tmp = not in_tmp
		width *= 2

	if in_tmp:
		for c in range(m):
			cand_p[c] = src_p[c]
			cand_k[c] = src_k[c]


@numba.njit
def _sort_finite_candidates(cand_p, cand_k, m, tmp_p, tmp_k):
	"""Sort cand_p[:m] as _sort_candidates does, for values with no NaN.

	Each pair of runs is merged from both ends at once: the front takes the
	smaller head, the left run's on ties, and the back the larger tail, the
	right run's on ties. Each end follows the stable merge's rule, so between
	them they write the stable merge's output, with two independent chains of
	loads and compares per step instead of one. A NaN breaks that, since the
	two ends could then disagree on the order. The runs of 16 are sorted into
	the scratch arrays when the number of passes is odd, so that the last
	pass writes the candidates.
	"""

	n_passes = 0
	width = 16
	while width < m:
		n_passes += 1
		width *= 2

	if n_passes % 2 == 1:
		src_p, src_k, dst_p, dst_k = tmp_p, tmp_k, cand_p, cand_k
	else:
		src_p, src_k, dst_p, dst_k = cand_p, cand_k, tmp_p, tmp_k

	for lo in range(0, m, 16):
		hi = min(lo + 16, m)
		for a in range(lo, hi):
			p, k = cand_p[a], cand_k[a]
			b = a
			while b > lo and p < src_p[b-1]:
				src_p[b] = src_p[b-1]
				src_k[b] = src_k[b-1]
				b -= 1

			src_p[b] = p
			src_k[b] = k

	width = 16
	while width < m:
		for lo in range(0, m, 2 * width):
			mid = min(lo + width, m)
			hi = min(lo + 2 * width, m)
			if mid == hi:
				for c in range(lo, hi):
					dst_p[c] = src_p[c]
					dst_k[c] = src_k[c]
				continue

			# The front fills [lo, c) from a and b, the back (tc, hi) from ta
			# and tb.
			a, b, c = lo, mid, lo
			ta, tb, tc = mid - 1, hi - 1, hi - 1

			# Runs of equal length: each end writes as many outputs as a run
			# holds, so neither end can run out of a run.
			if hi - mid == mid - lo:
				for _ in range(mid - lo):
					pa, pb = src_p[a], src_p[b]
					ka, kb = src_k[a], src_k[b]
					right = pb < pa
					dst_p[c] = pb if right else pa
					dst_k[c] = kb if right else ka
					step = 1 if right else 0
					b += step
					a += 1 - step
					c += 1

					qa, qb = src_p[ta], src_p[tb]
					ja, jb = src_k[ta], src_k[tb]
					left = qb < qa
					dst_p[tc] = qa if left else qb
					dst_k[tc] = ja if left else jb
					step = 1 if left else 0
					ta -= step
					tb -= 1 - step
					tc -= 1

				continue

			# A shorter right run, at the end of a pass: the front writes
			# ceil(n/2) outputs and the back floor(n/2), at most the left
			# run's length, so neither end runs out of the left run, and
			# either may run out of the right one. The front's load from it
			# is then clamped into the pair and not taken; the back's is the
			# left run's last entry, and not taken.
			n_back = (hi - lo) // 2
			n_front = hi - lo - n_back
			for _ in range(n_back):
				pa, pb = src_p[a], src_p[min(b, hi - 1)]
				ka, kb = src_k[a], src_k[min(b, hi - 1)]
				right = (pb < pa) & (b < hi)
				dst_p[c] = pb if right else pa
				dst_k[c] = kb if right else ka
				step = 1 if right else 0
				b += step
				a += 1 - step
				c += 1

				qa, qb = src_p[ta], src_p[tb]
				ja, jb = src_k[ta], src_k[tb]
				left = (qb < qa) | (tb < mid)
				dst_p[tc] = qa if left else qb
				dst_k[tc] = ja if left else jb
				step = 1 if left else 0
				ta -= step
				tb -= 1 - step
				tc -= 1

			if n_front > n_back:
				pa, pb = src_p[a], src_p[min(b, hi - 1)]
				ka, kb = src_k[a], src_k[min(b, hi - 1)]
				right = (pb < pa) & (b < hi)
				dst_p[c] = pb if right else pa
				dst_k[c] = kb if right else ka

		src_p, src_k, dst_p, dst_k = dst_p, dst_k, src_p, src_k
		width *= 2


@numba.njit
def _recursive_seqlet_sparse(rcdfs, min_seqlet_len, max_seqlet_len, n_bins):
	"""Whether the decoders may build, for each length, only the entries whose
	triangle can still be below the threshold (_recursive_seqlet_shortest and
	_recursive_seqlet_longer) instead of the full matrices.

	A span's triangle maximum is at least that of either span one shorter
	inside it, so an entry at or above the threshold rules out every span
	holding it. That needs every decoded length to be at least 1, `rcdfs` to
	cover every length and bin sum reached, and no NaN in those entries:
	max(a, b) returns a when b is NaN, so a later max would drop it, and
	_sort_finite_candidates needs no NaN.
	"""

	if min_seqlet_len < 1 or min_seqlet_len > max_seqlet_len or n_bins < 1:
		return False
	if max_seqlet_len >= rcdfs.shape[0]:
		return False
	if max_seqlet_len * (n_bins - 1) >= rcdfs.shape[1]:
		return False

	nan = False
	for seqlet_len in range(min_seqlet_len, max_seqlet_len+1):
		for s in range(seqlet_len * (n_bins - 1) + 1):
			v = rcdfs[seqlet_len, s]
			nan |= v != v

	return not nan


@numba.njit
def _recursive_seqlet_bins(x, xmin, bin_width, n_bins, x_bin):
	"""Write the bin of each value of `x` to `x_bin`, placing values outside
	the range of the bins in the end bins. A loop of its own, over one row,
	it compiles to vector divides; inside a loop carrying a running sum the
	divides stay scalar.
	"""

	for j in range(x.shape[0]):
		b = math.floor((x[j] - xmin) / bin_width)
		x_bin[j] = min(max(b, 0), n_bins - 1)


@numba.njit
def _recursive_seqlet_shortest(X_bin_csum, rcdfs, seqlet_len, threshold,
	k_, p_, q_):
	"""Write the entries of the shortest length decoded, `seqlet_len` (at
	least 1), whose triangle is below `threshold` to `k_`, `p_` and `q_` in
	order of start: the start, the entry of the matrix of p-values and the
	largest entry of its triangle. The rows of the matrix below this length
	are zeros. Returns the number of entries written.
	"""

	l = X_bin_csum.shape[0] - 1
	m = 0
	for k in range(l-seqlet_len+1):
		x_ = X_bin_csum[k+seqlet_len] - X_bin_csum[k]
		p = max(rcdfs[seqlet_len, x_], 0.0)
		q = max(p, 0.0)
		k_[m] = k
		p_[m] = p
		q_[m] = q
		if q < threshold:
			m += 1

	return m


@numba.njit
def _recursive_seqlet_longer(X_bin_csum, rcdfs, seqlet_len, threshold,
	prev_k, prev_p, prev_q, m_prev, k_, p_, q_):
	"""Write the entries of length `seqlet_len` whose triangle is below
	`threshold` to `k_`, `p_` and `q_` in order of start, from the `m_prev`
	such entries of length seqlet_len - 1 in `prev_k`, `prev_p` and `prev_q`.

	An entry of this length below the threshold needs the entries of the
	length below at its start and at the next to be below it too, and its
	p-value and triangle maximum are computed from those two as the full
	matrices compute them. Returns the number of entries written.
	"""

	m = 0
	for a in range(m_prev-1):
		k = prev_k[a]
		if prev_k[a+1] != k + 1:
			continue

		x_ = X_bin_csum[k+seqlet_len] - X_bin_csum[k]
		p = max(rcdfs[seqlet_len, x_], prev_p[a])
		q = max(p, prev_q[a+1])
		k_[m] = k
		p_[m] = p
		q_[m] = q
		if q < threshold:
			m += 1

	return m


@numba.njit
def _grown(a, size):
	"""A new array of `size` entries whose first len(a) entries are a's."""

	b = numpy.empty(size, dtype=a.dtype)
	for i in range(a.shape[0]):
		b[i] = a[i]
	return b


@numba.njit
def _recursive_seqlet_decode_fields(X, rcdfs, xmin, bin_width, threshold=0.01,
	min_seqlet_len=4, max_seqlet_len=25, additional_flanks=0, n_bins=1000):
	"""An internal function calling seqlets given the null distributions.

	This covers the last two steps of the recursive seqlet algorithm:

	(3) Use the null distributions to calculate p-values for each possible length
	(4) Decode this matrix of p-values to find the longest seqlets

	Values of `X` outside the range the bins were built over are placed in the
	end bins. The spans are decoded as by _recursive_seqlet_decode_dense,
	which this calls when _recursive_seqlet_sparse says so. Otherwise, only
	the entries of each length whose triangle is below the threshold are
	built, from those of the length below (_recursive_seqlet_longer), in
	order of start, which are the entries the full matrices give to the
	decoder.

	Returns the seqlets in the order they are called as five arrays: the
	example index, start and end (int64), and the attribution sum and
	p-value (float64).
	"""

	if not _recursive_seqlet_sparse(rcdfs, min_seqlet_len, max_seqlet_len,
		n_bins):
		return _recursive_seqlet_decode_dense(X, rcdfs, xmin, bin_width,
			threshold, min_seqlet_len, max_seqlet_len, additional_flanks,
			n_bins)

	n, l = X.shape
	n_lens = max_seqlet_len - min_seqlet_len + 1

	X_bin = numpy.empty(l, dtype=numpy.int64)
	X_bin_csum = numpy.zeros(l+1, dtype=numpy.int64)
	X_csum = numpy.zeros(l+1)

	# The entries below the threshold of the last length built and of the one
	# before it, in order of start.
	k_a = numpy.empty(l, dtype=numpy.int64)
	p_a = numpy.empty(l, dtype=numpy.float64)
	q_a = numpy.empty(l, dtype=numpy.float64)
	k_b = numpy.empty(l, dtype=numpy.int64)
	p_b = numpy.empty(l, dtype=numpy.float64)
	q_b = numpy.empty(l, dtype=numpy.float64)

	# Row j of sorted_p and sorted_k holds the n_cands[j] entries of length
	# max_seqlet_len - j below the threshold, sorted stably by p-value from
	# an order of start, the order in which the decoder visits them.
	sorted_p = numpy.empty((n_lens, l), dtype=numpy.float64)
	sorted_k = numpy.empty((n_lens, l), dtype=numpy.int64)
	n_cands = numpy.zeros(n_lens, dtype=numpy.int64)
	tmp_p = numpy.empty(l, dtype=numpy.float64)
	tmp_k = numpy.empty(l, dtype=numpy.int64)

	# A position is in a seqlet called in example i when called holds i + 1.
	called = numpy.zeros(l, dtype=numpy.int64)

	capacity = max(n, 16)
	seqlet_idx = numpy.empty(capacity, dtype=numpy.int64)
	seqlet_start = numpy.empty(capacity, dtype=numpy.int64)
	seqlet_end = numpy.empty(capacity, dtype=numpy.int64)
	seqlet_attr = numpy.empty(capacity, dtype=numpy.float64)
	seqlet_p = numpy.empty(capacity, dtype=numpy.float64)
	n_seqlets = 0

	for i in range(n):
		_recursive_seqlet_bins(X[i], xmin, bin_width, n_bins, X_bin)

		for j in range(l):
			X_bin_csum[j+1] = X_bin_csum[j] + X_bin[j]
			X_csum[j+1] = X_csum[j] + X[i, j]

		m = 0
		for seqlet_len in range(min_seqlet_len, max_seqlet_len+1):
			if seqlet_len == min_seqlet_len:
				m = _recursive_seqlet_shortest(X_bin_csum, rcdfs, seqlet_len,
					threshold, k_a, p_a, q_a)
			else:
				m = _recursive_seqlet_longer(X_bin_csum, rcdfs, seqlet_len,
					threshold, k_a, p_a, q_a, m, k_b, p_b, q_b)
				k_a, k_b = k_b, k_a
				p_a, p_b = p_b, p_a
				q_a, q_b = q_b, q_a

			j = max_seqlet_len - seqlet_len
			for c in range(m):
				sorted_p[j, c] = p_a[c]
				sorted_k[j, c] = k_a[c]

			n_cands[j] = m
			_sort_finite_candidates(sorted_p[j], sorted_k[j], m, tmp_p, tmp_k)

		# Identify spans, from longest to shortest, that satisfy the recursive
		# p-value threshold.
		for j in range(n_lens):
			seqlet_len = max_seqlet_len - j

			for q in range(n_cands[j]):
				start = sorted_k[j, q]

				for end in range(start, start+seqlet_len):
					if called[end] == i + 1:
						break

				else:
					for end in range(start, start+seqlet_len):
						called[end] = i + 1

					# Report the span's own p-value, not the maximum over
					# its prefixes that its entry holds.
					x_ = X_bin_csum[start+seqlet_len] - X_bin_csum[start]
					p = rcdfs[seqlet_len, x_]

					end = min(start + seqlet_len + additional_flanks, l)
					start = max(start - additional_flanks, 0)
					attr = X_csum[end] - X_csum[start]

					if n_seqlets == capacity:
						capacity *= 2
						seqlet_idx = _grown(seqlet_idx, capacity)
						seqlet_start = _grown(seqlet_start, capacity)
						seqlet_end = _grown(seqlet_end, capacity)
						seqlet_attr = _grown(seqlet_attr, capacity)
						seqlet_p = _grown(seqlet_p, capacity)

					seqlet_idx[n_seqlets] = i
					seqlet_start[n_seqlets] = start
					seqlet_end[n_seqlets] = end
					seqlet_attr[n_seqlets] = attr
					seqlet_p[n_seqlets] = p
					n_seqlets += 1

	return (seqlet_idx[:n_seqlets], seqlet_start[:n_seqlets],
		seqlet_end[:n_seqlets], seqlet_attr[:n_seqlets], seqlet_p[:n_seqlets])


def _recursive_seqlet_decode(X, rcdfs, xmin, bin_width, threshold=0.01,
	min_seqlet_len=4, max_seqlet_len=25, additional_flanks=0, n_bins=1000):
	"""Call seqlets as `_recursive_seqlet_decode_fields` does, returned as a
	list of (example index, start, end, attribution, p-value) tuples in the
	order they are called."""

	fields = _recursive_seqlet_decode_fields(X, rcdfs, xmin, bin_width,
		threshold, min_seqlet_len, max_seqlet_len, additional_flanks, n_bins)
	return list(zip(*[field.tolist() for field in fields]))


@numba.njit
def _recursive_seqlet_decode_dense(X, rcdfs, xmin, bin_width, threshold=0.01,
	min_seqlet_len=4, max_seqlet_len=25, additional_flanks=0, n_bins=1000):
	"""_recursive_seqlet_decode_fields, building the full matrices of
	p-values and of triangle maxima for every example, and returning the
	same five arrays.

	Within one length, the spans are visited in the order of their entry in
	the matrix of p-values, the largest p-value over the span's prefixes,
	with ties broken by start, and the visit ends at the first entry at or
	above the threshold. A span is called when none of its positions is in a
	seqlet called already and every entry of its triangle is below the
	threshold. Each span's triangle is reduced to its largest entry while the
	matrix is built, so a span whose triangle reaches the threshold is never
	visited. Calling a span never changes the matrix, so one matrix serves
	every example and is rebuilt from ones only when `min_seqlet_len` is
	below 1, where the first row built reads the last one.
	"""

	n, l = X.shape

	X_bin_csum = numpy.zeros(l+1, dtype=numpy.int64)
	X_csum = numpy.zeros(l+1)

	capacity = max(n, 16)
	seqlet_idx = numpy.empty(capacity, dtype=numpy.int64)
	seqlet_start = numpy.empty(capacity, dtype=numpy.int64)
	seqlet_end = numpy.empty(capacity, dtype=numpy.int64)
	seqlet_attr = numpy.empty(capacity, dtype=numpy.float64)
	seqlet_p = numpy.empty(capacity, dtype=numpy.float64)
	n_seqlets = 0

	n_rows = max_seqlet_len + 1
	p_flat = numpy.ones(n_rows * l, dtype=numpy.float64)
	p_value = p_flat.reshape((n_rows, l))
	p_value[:min_seqlet_len] = 0

	# q_value[L, k] is the largest entry of the triangle of the span [k, k+L),
	# p_value[L-m, k+m] for m from 0 to L-1, where the rows of p_value below
	# min_seqlet_len are zeros. Its rows below min_seqlet_len, row 0 and the
	# last column, which row 1 reads at k = l - 1, are never written: they are
	# the zeros the recursion starts from.
	q_value = numpy.zeros((n_rows, l+1), dtype=numpy.float64)

	# A position is in a seqlet called in example i when called holds i + 1.
	# The entries of one length whose triangle is below the threshold are
	# gathered in order of start and sorted stably by p-value.
	called = numpy.zeros(l, dtype=numpy.int64)
	cand_p = numpy.empty(l, dtype=numpy.float64)
	cand_k = numpy.empty(l, dtype=numpy.int64)
	tmp_p = numpy.empty(l, dtype=numpy.float64)
	tmp_k = numpy.empty(l, dtype=numpy.int64)

	for i in range(n):
		for j in range(l):
			x_bin = math.floor((X[i, j] - xmin) / bin_width)
			X_bin_csum[j+1] = X_bin_csum[j] + min(max(x_bin, 0), n_bins - 1)
			X_csum[j+1] = X_csum[j] + X[i, j]

		if min_seqlet_len < 1 and i > 0:
			for j in range(p_flat.shape[0]):
				p_flat[j] = 1
			p_value[:min_seqlet_len] = 0

		for seqlet_len in range(min_seqlet_len, max_seqlet_len+1):
			if seqlet_len < 1:
				for k in range(l-seqlet_len+1):
					x_ = X_bin_csum[k+seqlet_len] - X_bin_csum[k]
					p_value[seqlet_len, k] = max(rcdfs[seqlet_len, x_],
						p_value[seqlet_len-1, k])
				continue

			for k in range(l-seqlet_len+1):
				x_ = X_bin_csum[k+seqlet_len] - X_bin_csum[k]
				p = max(rcdfs[seqlet_len, x_], p_value[seqlet_len-1, k])
				p_value[seqlet_len, k] = p
				q_value[seqlet_len, k] = max(p, q_value[seqlet_len-1, k+1])

		# Identify spans, from longest to shortest, that satisfy the recursive
		# p-value threshold. A span of length below 1 has no triangle beyond
		# its own entry.
		for j in range(max_seqlet_len - min_seqlet_len + 1):
			seqlet_len = max_seqlet_len - j

			m = 0
			if seqlet_len >= 1:
				for k in range(l-seqlet_len+1):
					if q_value[seqlet_len, k] < threshold:
						cand_p[m] = p_value[seqlet_len, k]
						cand_k[m] = k
						m += 1

			else:
				for k in range(min(l-seqlet_len+1, l)):
					p = p_value[seqlet_len, k]
					if p < threshold:
						cand_p[m] = p
						cand_k[m] = k
						m += 1

			_sort_candidates(cand_p, cand_k, m, tmp_p, tmp_k)

			for q in range(m):
				start = cand_k[q]

				for end in range(start, start+seqlet_len):
					if called[end] == i + 1:
						break

				else:
					for end in range(start, start+seqlet_len):
						called[end] = i + 1

					# Report the span's own p-value, not the maximum over
					# its prefixes that p_value holds.
					x_ = X_bin_csum[start+seqlet_len] - X_bin_csum[start]
					p = rcdfs[seqlet_len, x_]

					end = min(start + seqlet_len + additional_flanks, l)
					start = max(start - additional_flanks, 0)
					attr = X_csum[end] - X_csum[start]

					if n_seqlets == capacity:
						capacity *= 2
						seqlet_idx = _grown(seqlet_idx, capacity)
						seqlet_start = _grown(seqlet_start, capacity)
						seqlet_end = _grown(seqlet_end, capacity)
						seqlet_attr = _grown(seqlet_attr, capacity)
						seqlet_p = _grown(seqlet_p, capacity)

					seqlet_idx[n_seqlets] = i
					seqlet_start[n_seqlets] = start
					seqlet_end[n_seqlets] = end
					seqlet_attr[n_seqlets] = attr
					seqlet_p[n_seqlets] = p
					n_seqlets += 1

	return (seqlet_idx[:n_seqlets], seqlet_start[:n_seqlets],
		seqlet_end[:n_seqlets], seqlet_attr[:n_seqlets], seqlet_p[:n_seqlets])


@numba.njit
def _recursive_seqlet_counts(X, rcdfs, xmin, bin_width, thresholds,
	min_seqlet_len=4, max_seqlet_len=25, n_bins=1000):
	"""Count the seqlets `_recursive_seqlet_decode` calls on each row of `X`
	at each of `thresholds`, which run from largest to smallest.

	The counts are those of _recursive_seqlet_counts_dense, which this calls
	when _recursive_seqlet_sparse says so. Otherwise, only the entries of
	each length whose triangle is below the largest threshold are built, in
	order of start as in _recursive_seqlet_decode, and the sorted entries of
	each length are visited at each threshold as there. Every decoded length
	is then at least 1, so a row stops at its first threshold with no
	seqlet.
	"""

	if not _recursive_seqlet_sparse(rcdfs, min_seqlet_len, max_seqlet_len,
		n_bins):
		return _recursive_seqlet_counts_dense(X, rcdfs, xmin, bin_width,
			thresholds, min_seqlet_len, max_seqlet_len, n_bins)

	n, l = X.shape
	n_lens = max_seqlet_len - min_seqlet_len + 1
	n_thresholds = thresholds.shape[0]
	counts = numpy.zeros(n_thresholds, dtype=numpy.int64)

	largest = -numpy.inf
	for t in range(n_thresholds):
		largest = max(largest, thresholds[t])

	X_bin = numpy.empty(l, dtype=numpy.int64)
	X_bin_csum = numpy.zeros(l+1, dtype=numpy.int64)

	# The entries below the largest threshold of the last length built and of
	# the one before it, in order of start.
	k_a = numpy.empty(l, dtype=numpy.int64)
	p_a = numpy.empty(l, dtype=numpy.float64)
	q_a = numpy.empty(l, dtype=numpy.float64)
	k_b = numpy.empty(l, dtype=numpy.int64)
	p_b = numpy.empty(l, dtype=numpy.float64)
	q_b = numpy.empty(l, dtype=numpy.float64)

	# The p-values of one length's entries and their places in the order of
	# start, sorted stably by p-value.
	cand_p = numpy.empty(l, dtype=numpy.float64)
	cand_k = numpy.empty(l, dtype=numpy.int64)
	tmp_p = numpy.empty(l, dtype=numpy.float64)
	tmp_k = numpy.empty(l, dtype=numpy.int64)

	# Row j of sorted_p, sorted_k and sorted_q holds the n_cands[j] entries of
	# length max_seqlet_len - j below the largest threshold, in the decoder's
	# order, with their p-values and triangle maxima.
	sorted_p = numpy.empty((n_lens, l), dtype=numpy.float64)
	sorted_k = numpy.empty((n_lens, l), dtype=numpy.int64)
	sorted_q = numpy.empty((n_lens, l), dtype=numpy.float64)
	n_cands = numpy.zeros(n_lens, dtype=numpy.int64)

	# A position is in a seqlet called at the current row and threshold when
	# called holds that pair's stamp.
	called = numpy.zeros(l, dtype=numpy.int64)
	stamp = 0

	for i in range(n):
		_recursive_seqlet_bins(X[i], xmin, bin_width, n_bins, X_bin)

		for j in range(l):
			X_bin_csum[j+1] = X_bin_csum[j] + X_bin[j]

		m = 0
		for seqlet_len in range(min_seqlet_len, max_seqlet_len+1):
			if seqlet_len == min_seqlet_len:
				m = _recursive_seqlet_shortest(X_bin_csum, rcdfs, seqlet_len,
					largest, k_a, p_a, q_a)
			else:
				m = _recursive_seqlet_longer(X_bin_csum, rcdfs, seqlet_len,
					largest, k_a, p_a, q_a, m, k_b, p_b, q_b)
				k_a, k_b = k_b, k_a
				p_a, p_b = p_b, p_a
				q_a, q_b = q_b, q_a

			for c in range(m):
				cand_p[c] = p_a[c]
				cand_k[c] = c

			_sort_finite_candidates(cand_p, cand_k, m, tmp_p, tmp_k)

			j = max_seqlet_len - seqlet_len
			for c in range(m):
				a = cand_k[c]
				sorted_p[j, c] = cand_p[c]
				sorted_k[j, c] = k_a[a]
				sorted_q[j, c] = q_a[a]

			n_cands[j] = m

		for t in range(n_thresholds):
			threshold = thresholds[t]
			stamp += 1

			n_seqlets = 0
			for j in range(n_lens):
				seqlet_len = max_seqlet_len - j

				for q in range(n_cands[j]):
					if sorted_p[j, q] >= threshold:
						break

					if sorted_q[j, q] >= threshold:
						continue

					start = sorted_k[j, q]
					for end in range(start, start+seqlet_len):
						if called[end] == stamp:
							break

					else:
						for end in range(start, start+seqlet_len):
							called[end] = stamp

						n_seqlets += 1

			counts[t] += n_seqlets
			if n_seqlets == 0:
				break

	return counts


@numba.njit
def _recursive_seqlet_counts_dense(X, rcdfs, xmin, bin_width, thresholds,
	min_seqlet_len=4, max_seqlet_len=25, n_bins=1000):
	"""_recursive_seqlet_counts, building the full matrices of p-values and
	of triangle maxima for every row.

	Only the visit depends on the threshold, so each row builds its matrices
	and sorts each length's entries once, and every threshold visits them in
	that order. A row stops at its first threshold with no seqlet: with
	nothing called, a span passes exactly when its triangle is below the
	threshold, so no stricter threshold calls one. That needs every decoded
	length to be at least 1, so a smaller `min_seqlet_len` decodes every
	threshold. Returns the number of seqlets at each threshold, summed over
	the rows.
	"""

	n, l = X.shape
	n_rows = max_seqlet_len + 1
	n_lens = max(max_seqlet_len - min_seqlet_len + 1, 0)
	n_thresholds = thresholds.shape[0]
	counts = numpy.zeros(n_thresholds, dtype=numpy.int64)

	largest = -numpy.inf
	for t in range(n_thresholds):
		largest = max(largest, thresholds[t])

	X_bin_csum = numpy.zeros(l+1, dtype=numpy.int64)
	p_flat = numpy.ones(n_rows * l, dtype=numpy.float64)
	p_value = p_flat.reshape((n_rows, l))
	p_value[:min_seqlet_len] = 0

	# q_value[L, k] is the largest entry of the triangle of [k, k+L), built as
	# in the decoder; its rows below min_seqlet_len, row 0 and the last
	# column stay zero.
	q_value = numpy.zeros((n_rows, l+1), dtype=numpy.float64)

	cand_p = numpy.empty(l, dtype=numpy.float64)
	cand_k = numpy.empty(l, dtype=numpy.int64)
	tmp_p = numpy.empty(l, dtype=numpy.float64)
	tmp_k = numpy.empty(l, dtype=numpy.int64)

	# Row j of sorted_p, sorted_k and sorted_q holds the n_cands[j] entries of
	# length max_seqlet_len - j whose key is below the largest threshold, in
	# the decoder's order, with their p-values and keys.
	sorted_p = numpy.empty((n_lens, l), dtype=numpy.float64)
	sorted_k = numpy.empty((n_lens, l), dtype=numpy.int64)
	sorted_q = numpy.empty((n_lens, l), dtype=numpy.float64)
	n_cands = numpy.zeros(n_lens, dtype=numpy.int64)

	# A position is in a seqlet called at the current row and threshold when
	# called holds that pair's stamp.
	called = numpy.zeros(l, dtype=numpy.int64)
	stamp = 0

	stop_rows = min_seqlet_len >= 1

	for i in range(n):
		for j in range(l):
			x_bin = math.floor((X[i, j] - xmin) / bin_width)
			X_bin_csum[j+1] = X_bin_csum[j] + min(max(x_bin, 0), n_bins - 1)

		# With min_seqlet_len >= 1 every entry a row reads is written for that
		# row first, or is one of the ones and zeros set above; otherwise the
		# matrix starts over as the decoder's does.
		if not stop_rows and i > 0:
			for j in range(p_flat.shape[0]):
				p_flat[j] = 1
			p_value[:min_seqlet_len] = 0

		for seqlet_len in range(min_seqlet_len, max_seqlet_len+1):
			if seqlet_len < 1:
				for k in range(l-seqlet_len+1):
					x_ = X_bin_csum[k+seqlet_len] - X_bin_csum[k]
					p_value[seqlet_len, k] = max(rcdfs[seqlet_len, x_],
						p_value[seqlet_len-1, k])
				continue

			for k in range(l-seqlet_len+1):
				x_ = X_bin_csum[k+seqlet_len] - X_bin_csum[k]
				p = max(rcdfs[seqlet_len, x_], p_value[seqlet_len-1, k])
				p_value[seqlet_len, k] = p
				q_value[seqlet_len, k] = max(p, q_value[seqlet_len-1, k+1])

		# The key of an entry is the largest entry of its triangle, which for
		# a length below 1 is the entry itself.
		for j in range(n_lens):
			seqlet_len = max_seqlet_len - j

			m = 0
			if seqlet_len >= 1:
				for k in range(l-seqlet_len+1):
					if q_value[seqlet_len, k] < largest:
						cand_p[m] = p_value[seqlet_len, k]
						cand_k[m] = k
						m += 1

			else:
				for k in range(min(l-seqlet_len+1, l)):
					p = p_value[seqlet_len, k]
					if p < largest:
						cand_p[m] = p
						cand_k[m] = k
						m += 1

			_sort_candidates(cand_p, cand_k, m, tmp_p, tmp_k)
			for q in range(m):
				sorted_p[j, q] = cand_p[q]
				sorted_k[j, q] = cand_k[q]
				if seqlet_len >= 1:
					sorted_q[j, q] = q_value[seqlet_len, cand_k[q]]
				else:
					sorted_q[j, q] = cand_p[q]
			n_cands[j] = m

		for t in range(n_thresholds):
			threshold = thresholds[t]
			stamp += 1

			n_seqlets = 0
			for j in range(n_lens):
				seqlet_len = max_seqlet_len - j

				for q in range(n_cands[j]):
					if sorted_p[j, q] >= threshold:
						break

					if sorted_q[j, q] >= threshold:
						continue

					start = sorted_k[j, q]
					for end in range(start, start+seqlet_len):
						if called[end] == stamp:
							break

					else:
						for end in range(start, start+seqlet_len):
							called[end] = stamp

						n_seqlets += 1

			counts[t] += n_seqlets
			if n_seqlets == 0 and stop_rows:
				break

	return counts


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

	# Once neither input has a seqlet, no stricter threshold gives one, as in
	# _recursive_seqlet_counts_dense.
	exhausted = tables is None

	if not exhausted:
		xmin, bin_width, rcdfs = tables
		ordered = numpy.ascontiguousarray(thresholds)
		counts = _recursive_seqlet_counts(X, rcdfs, xmin, bin_width, ordered,
			min_seqlet_len, max_seqlet_len, n_bins)
		counts_control = _recursive_seqlet_counts(X_control, rcdfs, xmin,
			bin_width, ordered, min_seqlet_len, max_seqlet_len, n_bins)

	rows = []
	for t, threshold in enumerate(thresholds):
		if exhausted:
			n_seqlets, n_control = 0, 0
		else:
			n_seqlets, n_control = int(counts[t]), int(counts_control[t])
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
	`X` at the largest threshold whose estimate is at or below `target_fdr`.
	Once a threshold calls no seqlet on `X` and none on `X_control`, no
	stricter threshold can call any, so the remaining thresholds are filled
	in without calling seqlets.

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
		are called on `X`. Seqlets are called at the largest threshold whose
		`fdr` is at or below `target_fdr`.
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

	fields = None
	if tables is not None and threshold is not None:
		xmin, bin_width, rcdfs = tables
		fields = _recursive_seqlet_decode_fields(X, rcdfs, xmin, bin_width,
			threshold, min_seqlet_len, max_seqlet_len, additional_flanks, n_bins)

	if fields is None or fields[0].shape[0] == 0:
		dtypes = [numpy.int64, numpy.int64, numpy.int64, numpy.float64,
			numpy.float64]
		seqlets = pandas.DataFrame({name: numpy.array([], dtype=dtype)
			for name, dtype in zip(columns, dtypes)})
	else:
		seqlets = pandas.DataFrame(dict(zip(columns, fields)))

	seqlets = seqlets.sort_values("p-value").reset_index(drop=True)

	if return_fdr:
		return RecursiveSeqletsResult(seqlets=seqlets, fdr=fdr)
	return seqlets
