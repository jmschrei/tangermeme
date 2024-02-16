# ablate.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
Provides functions for the calculation of GC-content genome-wide and the
sampling of GC-matched negatives.
"""

import numpy
import torch
import pandas
import pyfaidx
import pyBigWig
import collections

from tqdm import tqdm
from scipy.stats import ks_2samp

from .io import extract_loci

from joblib import Parallel
from joblib import delayed


def _calculate_char_perc(sequence, width, chars):
	"""An internal function returning the percentage of `chars` in `sequence`.

	This method will take in a string `sequence` and return the percentage
	of each non-overlapping block of length `width` across the sequence
	containing `chars`. This is usually used to calculate GC percentage but
	can also be used to calculate the percentage of N's in a sequence.
	

	Parameters
	----------
	sequence: str
		A string made up of the alphabet 'A', 'C', 'G', 'T', 'N'.

	width: int
		The total width of the window to calculate the GC content for, e.g.,
		`1000` if you want to calculate this for 1000 bp blocks.

	chars: tuple
		The characters to look for in the sequence. The returned percentage is
		the percentage of the original sequence that is these characters.


	Returns
	-------
	perc: numpy.ndarray, shape=(len(sequence) // width)
	"""

	if len(sequence) < width:
		raise ValueError("Width is larger than the sequence.")

	if not isinstance(sequence, str):
		raise ValueError("Sequence must be provided as a string.")

	is_in = numpy.isin(list(sequence), chars)
	is_in = is_in[:len(is_in) // width * width].reshape(-1, width)
	return torch.from_numpy(is_in.mean(axis=-1)).type(torch.float32)


def _extract_and_filter_chrom(fasta, chrom, in_window, out_window, 
	max_n_perc=0.1, gc_bin_width=0.02, bigwig=None, signal_threshold=None):
	"""Calculate GC content for, and filter, one chromosome.

	This function will take in the name of a FASTA file, a chromosome, and a
	percentage of Ns that cannot be exceeded, and return a set of passing loci
	and their exact GC percentage. Optionally, it will also take in a bigwig
	and filter the loci based on having a signal threshold smaller than some
	value, where the signal is summed across the width.

	
	Parameters
	----------
	fasta: str
		The filepath to the FASTA file to extract sequences from.

	chrom: str
		The chromosome to extract from the FASTA file. Must be in the file.

	in_window: int
		The window to calculate the GC content over, corresponding to the input
		window of the downstream model that will be trained.

	out_window: int
		The window to calculate signal for and apply the signal threshold to,
		corresponding to the output window of the downstream model that will
		be trained.

	max_n_perc: float, range=(0, 1.0), optional
		The maximum percentage of N characters in each window to be considered.
		All windows with a higher percentage are discarded. Default is 0.1.

	gc_bin_width: float, range=(0, 1.0), optional
		The bin size to discretize GC content. Default is 0.02.

	bigwig: str or None, optional
		If filtering regions based on signal strength, calculate the signal
		from this bigwig. If None, do not filter based on signal strength.
		Default is None.

	signal_threshold: float or None, optional
		The maximum possible signal, summed across the entire window, that
		each window can have without being filtered. If the window has a summed
		signal higher than this value, the window is discarded. Default is None.


	Returns
	-------
	gc_percs: dict
		A dictionary where the keys are observed GC bins and the values are
		lists of loci that are in that GC bin. Each returned locus is the index
		not the true value and so corresponds to the real position integer
		divided by in_window.
	"""

	sequence = pyfaidx.Fasta(fasta)[chrom][:].seq.upper()

	gc_perc = _calculate_char_perc(sequence, in_window, ('G', 'C'))
	n_perc = _calculate_char_perc(sequence, in_window, ('N',))

	idxs = n_perc <= max_n_perc

	flank = (in_window - out_window) // 2
	if bigwig is not None:
		bw = pyBigWig.open(bigwig, "r")

		try:
			values = bw.values(chrom, 0, -1, numpy=True)
		except RuntimeError:
			return {}

		values = values[:values.shape[0] // in_window * in_window]
		values = values.reshape(-1, in_window)
		values[:, :flank] = 0
		values[:, -flank:] = 0
		values = values.sum(axis=-1)

		v_idxs = values <= signal_threshold
		idxs = idxs & v_idxs

	idxs = numpy.where(idxs)[0]

	gc_perc = ((gc_perc + gc_bin_width / 2.) // gc_bin_width).type(torch.int32)
	gc_percs = collections.defaultdict(list)
	for idx in idxs:
		gc_percs[gc_perc[idx].item()].append(idx.item())

	return gc_percs


def extract_matching_loci(loci, fasta, in_window=2114, out_window=1000, 
	max_n_perc=0.1, gc_bin_width=0.02, bigwig=None, signal_beta=0.5, 
	chroms=None, random_state=None, verbose=False):
	"""Extract matching loci given a fasta file.

	This function takes in a set of loci (a bed file or a pandas dataframe in 
	bed format) and returns a GC-matched set of negatives. This will also
	perform basic filtering to ignore regions of the genome that are too high
	in Ns. Optionally, it can take in a bigwig and a signal threshold and only
	select regions that have fewer than a threshold of counts in each region.

	Importantly, it will apply `max_n_perc` to both the loci that are passed in
	and also potential regions that can be selected. This means that if a locus
	passed in has higher than `max_n_perc` number of unspecified positions,
	it will be filtered out, and a smaller number of positions will be selected.
	This is done because the GC content of a region with many Ns in it is not
	trustworthy.


	Parameters
	----------
	fasta: str
		The filepath to the FASTA file to extract sequences from.

	chrom: str
		The chromosome to extract from the FASTA file. Must be in the file.

	in_window: int
		The window to calculate the GC content over, corresponding to the input
		window of the downstream model that will be trained.

	out_window: int
		The window to calculate signal for and apply the signal threshold to,
		corresponding to the output window of the downstream model that will
		be trained.

	max_n_perc: float, range=(0, 1.0), optional
		The maximum percentage of N characters in each window to be considered.
		All windows with a higher percentage are discarded. Default is 0.1.

	gc_bin_width: float, range=(0, 1.0), optional
		The bin size to discretize GC content. Default is 0.02.

	bigwig: str or None, optional
		If filtering regions based on signal strength, calculate the signal
		from this bigwig. If None, do not filter based on signal strength.
		Default is None.

	signal_beta: float or None, optional
		A multiplier of the robust minimum signal calculated from `loci` that
		each background region must have fewer reads then. Only relevant if a
		bigwig is passed in. Default is 0.5.

	chroms: list, tuple, or None, optional
		A set of chromosomes to use when choosing matching loci. If None, only
		use chromosomes that the loci themselves are drawn from. Default is
		None.

	random_state: numpy.random.RandomState, int or None, optional
		A random state to use for sampling loci. If a RandomState object or
		an integer, this will produce deterministic sampling. If None, sampling
		will be different each time. Default is None.

	verbose: bool, optional
		Whether to print display bars and diagnostics to ensure that the
		sampling is reasonable. When set to True, there may be a large amount
		of output. Default is False. 


	Returns
	-------
	matched_loci: pandas.DataFrame
		A bed-formatted set of matched loci sorted first by chromosome and
		then by position on the chromosome. Note that these are not sorted
		such that the i-th position in this file is a GC match for the i-th
		position in the original locus file.
	"""

	if not isinstance(random_state, numpy.random.RandomState):
		random_state = numpy.random.RandomState(random_state)

	if isinstance(loci, str):
		loci = pandas.read_csv(loci, sep='\t', usecols=[0, 1, 2], header=None, 
			index_col=False, names=['chrom', 'start', 'end'])

	if chroms is None:
		chroms = numpy.unique(loci['chrom'])

	if bigwig:
		X, y = extract_loci(loci, fasta, in_window=in_window, signals=[bigwig],
			out_window=out_window, verbose=verbose)
		robust_min = torch.quantile(y.sum(dim=(1, 2)), 0.01).item()
		threshold = robust_min * signal_beta
	else:
		X = extract_loci(loci, fasta, in_window=in_window, verbose=verbose)
		threshold = None

	X = X.type(torch.float32)
	X = X[X.sum(axis=1).mean(axis=-1) >= (1.-max_n_perc)]

	# Extract reference GC bins
	loci_gc = X.mean(axis=-1)[:, [1, 2]].sum(axis=1).numpy()
	loci_gc = ((loci_gc + gc_bin_width / 2.) // gc_bin_width).astype(int)


	loci_bin_count = numpy.zeros(int(1./gc_bin_width)+1, dtype=int)
	for gc_bin in loci_gc:
		loci_bin_count[gc_bin] += 1


	# Extract mask of already-selected loci
	mask = {chrom: [] for chrom in chroms}
	for _, (chrom, start, end) in loci.iterrows():
		if chrom not in mask:
			continue

		start = int(start // in_window)
		end = int(end // in_window) + 1

		for idx in range(start, end):
			mask[chrom].append(idx)

	for chrom, values in mask.items():
		mask[chrom] = set(values)


	# Get GC content of background regions
	f = delayed(_extract_and_filter_chrom)
	chrom_percs = Parallel(n_jobs=-1)(f(
		fasta=fasta, 
		chrom=chrom, 
		in_window=in_window,
		out_window=out_window, 
		max_n_perc=max_n_perc,
		gc_bin_width=gc_bin_width,
		bigwig=bigwig, 
		signal_threshold=threshold) 
		for chrom in tqdm(chroms, disable=not verbose))


	# Merge them into a single dictionary, keeping track of chroms
	bg_bin_count = numpy.zeros(int(1./gc_bin_width) + 1, dtype=int)
	gc_percs = {perc: [] for perc in range(len(bg_bin_count))}
	
	for chrom, percs in zip(chroms, chrom_percs):
		for key, values in percs.items():
			for value in values:
				if value not in mask[chrom]:
					gc_percs[key].append((chrom, value))
					bg_bin_count[key] += 1

	for key, value in gc_percs.items():
		random_state.shuffle(value)


	orig_bg_bin_count = bg_bin_count.copy()
	orig_loci_bin_count = loci_bin_count.copy()

	# Match the sizes
	matched_loci_bin_count = numpy.minimum(bg_bin_count, loci_bin_count)
	bg_bin_count -= matched_loci_bin_count
	loci_bin_count -= matched_loci_bin_count

	n = len(loci_bin_count)
	for i in range(n-1, -1, -1):
		if loci_bin_count[i] == 0:
			continue

		for offset in range(n):
			idx = i + offset
			if idx < n:
				count = min(bg_bin_count[idx], loci_bin_count[i])
				bg_bin_count[idx] -= count
				loci_bin_count[i] -= count
				matched_loci_bin_count[idx] += count

			if loci_bin_count[i] == 0:
				break

			idx = i - offset
			if idx > 0:
				count = min(bg_bin_count[idx], loci_bin_count[i])
				bg_bin_count[idx] -= count
				loci_bin_count[i] -= count
				matched_loci_bin_count[idx] += count

			if loci_bin_count[i] == 0:
				break

	if verbose:
		numpy.set_printoptions(suppress=True)
		print("GC Bin\tBackground Count\tPeak Count\tChosen Count")
		for i in range(n):
			print("{:2.2f}: {:8d}\t{:8d}\t{:8d}".format(
				numpy.arange(0, 1.01, gc_bin_width)[i], 
				orig_bg_bin_count[i], orig_loci_bin_count[i], 
				matched_loci_bin_count[i]))


	# Extract the loci
	matched_loci = {'chrom': [], 'start': [], 'end': []}
	for i in range(n):
		for j in range(matched_loci_bin_count[i]):
			chrom, start = gc_percs[i][j]

			matched_loci['chrom'].append(chrom)
			matched_loci['start'].append(start*in_window)
			matched_loci['end'].append((start+1)*in_window)

	matched_loci = pandas.DataFrame(matched_loci)

	if verbose:
		if bigwig is None:
			X_matched = extract_loci(matched_loci, fasta, in_window=in_window, 
				verbose=verbose)
		else:
			X_matched, y_matched = extract_loci(matched_loci, fasta, 
				signals=[bigwig], in_window=in_window, out_window=out_window, 
				verbose=verbose)

		# Extract reference GC bins
		matched_gc = X_matched.mean(axis=-1)[:, [1, 2]].sum(axis=1).numpy()
		matched_gc = ((matched_gc + gc_bin_width / 2.) // 
			gc_bin_width).astype(int)

		stats = ks_2samp(loci_gc, matched_gc)

		print("GC-bin KS test stat:{:3.3}, p-value {:3.3}".format(
			stats.statistic, stats.pvalue))

		if bigwig is not None:
			print("Peak Robust Signal Minimum: {}".format(robust_min))
			print("Matched Signal Maximum: {}".format(y_matched.sum(
				axis=(1, 2)).max()))

	matched_loci = matched_loci.sort_values(["chrom", "start"])
	return matched_loci