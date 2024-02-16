# io.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>
# Code adapted from Alex Tseng, Avanti Shrikumar, and Ziga Avsec

import numpy
import torch
import pandas

import pyfaidx
import pyBigWig

from tqdm import tqdm

from .utils import one_hot_encode


def _interleave_loci(loci, chroms=None):
	"""An internal function for processing the provided loci.

	There are two aspects that have to be considered when processing the loci.
	The first is that the user can pass in either strings containing filenames
	or pandas DataFrames. The second is that the user can pass in a single
	value or a list of values, and if a list of values the resulting dataframes
	must be interleaved.

	If a set of chromosomes is provided, each dataframe will be filtered to
	loci on those chromosomes before interleaving. If a more complicated form
	of filtering is desired, one should pre-filter the dataframes and pass those
	into this function for interleaving.


	Parameters
	----------
	loci: str, pandas.DataFrame, or list of those
		A filename to load, a pandas DataFrame in bed-format, or a list of
		either.

	chroms: list or None, optional
		A set of chromosomes to restrict the loci to. This is done before
		interleaving to ensure balance across sets of loci. If None, do not
		do filtering. Default is None.


	Returns
	-------
	interleaved_loci: pandas.DataFrame
		A single pandas DataFrame that interleaves rows from each of the
		provided examples.
	"""

	if chroms is not None:
		if not isinstance(chroms, (list, tuple)):
			raise ValueError("Provided chroms must be a list.")

	if isinstance(loci, (str, pandas.DataFrame)):
		loci = [loci]
	elif not isinstance(loci, (list, tuple)):
		raise ValueError("Provided loci must be a string or pandas " +
			"DataFrame, or a list/tuple of those.")

	names = ['chrom', 'start', 'end']
	loci_dfs = []
	for i, df in enumerate(loci):
		if isinstance(df, str):
			df = pandas.read_csv(df, sep='\t', usecols=[0, 1, 2], 
				header=None, index_col=False, names=names)
		elif isinstance(df, pandas.DataFrame):
			df = df.iloc[:, [0, 1, 2]].copy()
		else:
			raise ValueError("Provided loci must be a string or pandas " +
				"DataFrame, or a list/tuple of those.")

		if chroms is not None:
			df = df[numpy.isin(df['chrom'], chroms)]

		df['idx'] = numpy.arange(len(df)) * len(loci) + i
		loci_dfs.append(df)

	loci = pandas.concat(loci_dfs)
	loci = loci.set_index("idx").sort_index().reset_index(drop=True)
	return loci


def _load_signals(signals):
	"""An internal function for loading signals.

	The passed in signals must be a list but can either be a list of strings,
	which are interpreted as strings for bigwig files that should be opened,
	or dictionaries where the keys are chromosome names and the values are
	numpy arrays of values across the chromosome, which are kept as is.


	Parameters
	----------
	signals: list of strings or dicts or None
		A list of strings for bigwig files or dictionaries of numpy arrays.


	Returns
	-------
	_signals: list of dicts
		A list of either pointers to opened bigwig files or dictionaries of
		numpy a
	"""

	if signals is None:
		return None

	_signals = []
	for i, signal in enumerate(signals):
		if isinstance(signal, str):
			signal = pyBigWig.open(signal)
		elif not isinstance(signal, dict):
			raise ValueError("Signals must either be a list of strings " +
				"or a list of dictionaries.")
		elif not isinstance(list(signal.values())[0], numpy.ndarray):
			raise ValueError("Values in dictionaries must be numpy.ndarrays.")

		_signals.append(signal)

	return _signals


def _extract_locus_signal(signals, chrom, start, end):
	"""An internal function for extracting signal from a single locus.

	This function takes in a set of signals and a single locus and extracts
	the signal from each one of the loci.


	Parameters
	----------
	signals: list of pyBigWig objects or dictionaries
		A list of opened pyBigWig objects or dictionaries where the keys are
		chromosomes and the values are the signal at each position in the
		chromosome.

	chrom: str
		The name of the chromosome. Must be a key in the signals.

	start: int
		The starting coordinate to extract from, inclusive and base-0.

	end: int
		The ending coordinate to extract from, exclusive and base-0.


	Returns
	-------
	values: list of numpy.ndarrays, shape=(len(signals), end-start)
		The extracted signal from each of the signal files.
	"""

	if not isinstance(signals, (list, tuple)):
		raise ValueError("Provided signals must be in the form of a list.")

	values = []
	for i, signal in enumerate(signals):
		if isinstance(signal, dict):
			values_ = signal[chrom][start:end]
		else:
			try:
				values_ = signal.values(chrom, start, end, numpy=True)
			except:
				print(f"Warning: {chrom} {start} {end} not " +
					"valid bigwig indexes. Using zeros instead.")
				values_ = numpy.zeros(end-start)

		values_ = numpy.nan_to_num(values_)
		values.append(values_)

	return values


def extract_loci(loci, sequences, signals=None, in_signals=None, chroms=None, 
	in_window=2114, out_window=1000, max_jitter=0, min_counts=None,
	max_counts=None, target_idx=0, n_loci=None, alphabet=['A', 'C', 'G', 'T'], 
	verbose=False):
	"""Extract sequence and signal information for each provided locus.

	This function will take in a set of loci, sequences, and optionally signals,
	and return the sequences and signals at each of the loci. Each of these
	parameters can be a filename, which is loaded internally, or an appropriate
	Python object (see below for details). The nomenclature `in/out` refer to
	he expected inputs and outputs of the downstream machine learning model, 
	not this function.

	For each locus a sequence window of size `in_window` will be extracted from
	the sequences file and each of the `input_signals` files if provided, and
	a window of size `out_window` will be extracted from each of the `signals`
	files if provided. These windows are centered at the middle of the provided
	regions but all be of the same size, regardless of the size of the peak.

	If `max_jitter` is provided, it will expand the windows for both the input
	and output. The results are not actually jittered, but this expanded window
	allows for downstream data generators to created jittered data while
	reducing the memory footprint of the returned data.

	There are a few reasons that the returned elements may not match one-to-one
	with the provided loci:

		- (1) If any of the coordinates fall off the end of chromosomes after
		accounting for jitter, the locus will be removed.

		- (2) If any of the loci fall on chromosomes not in a provided list,
		they will be removed.

		- (3) If min_counts or max_counts are specified and the locus has a 
		number of counts not in those boundaries.
 

	Parameters
	----------
	loci: str or pandas.DataFrame or list/tuple of such
		Either the path to a bed file or a pandas DataFrame object containing
		three columns: the chromosome, the start, and the end, of each locus
		to train on. Alternatively, a list or tuple of strings/DataFrames where
		the intention is to train on the interleaved concatenation, i.e., when
		you want to train on peaks and negatives.

	sequences: str or dictionary
		Either the path to a fasta file to read from or a dictionary where the
		keys are the unique set of chromosoms and the values are one-hot
		encoded sequences as numpy arrays or memory maps.

	signals: list of strs or list of dictionaries or None, optional
		A list of filepaths to bigwig files, where each filepath will be read
		using pyBigWig, or a list of dictionaries where the keys are the same
		set of unique chromosomes and the values are numpy arrays or memory
		maps. If None, no signal tensor is returned. Default is None.

	input_signals: list of strs or list of dictionaries or None, optional
		A list of filepaths to bigwig files, where each filepath will be read
		using pyBigWig, or a list of dictionaries where the keys are the same
		set of unique chromosomes and the values are numpy arrays or memory
		maps. If None, no tensor is returned. Default is None. 

	chroms: list or None, optional
		A set of chromosomes to extact loci from. Loci in other chromosomes
		in the locus file are ignored. If None, all loci are used. Default is
		None.

	in_window: int, optional
		The input window size. Default is 2114.

	out_window: int, optional
		The output window size. Default is 1000.

	max_jitter: int, optional
		The maximum amount of jitter to add, in either direction, to the
		midpoints that are passed in. Default is 0.

	min_counts: float or None, optional
		The minimum number of counts, summed across the length of each example
		and across all tasks, needed to be kept. If None, no minimum. Default 
		is None.

	max_counts: float or None, optional
		The maximum number of counts, summed across the length of each example
		and across all tasks, needed to be kept. If None, no maximum. Default 
		is None.  

	target_idx: int, optional
		When specifying `min_counts` or `max_counts`, the single `signal`
		file to use when determining if a region has a number of counts in
		that range. Default is 0.

	n_loci: int or None, optional
		A cap on the number of loci to return. Note that this is not the
		number of loci that are considered. The difference is that some
		loci may be filtered out for various reasons, and those are not
		counted towards the total. If None, no cap. Default is None.

	verbose: bool, optional
		Whether to display a progress bar while loading. Default is False.

	Returns
	-------
	seqs: torch.tensor, shape=(n, 4, in_window+2*max_jitter)
		The extracted sequences in the same order as the loci in the locus
		file after optional filtering by chromosome.

	signals: torch.tensor, shape=(n, len(signals), out_window+2*max_jitter)
		The extracted signals where the first dimension is in the same order
		as loci in the locus file after optional filtering by chromosome and
		the second dimension is in the same order as the list of signal files.
		If no signal files are given, this is not returned.

	in_signals: torch.tensor, shape=(n, len(in_signals),out_window+2*max_jitter)
		The extracted in signals where the first dimension is in the same order
		as loci in the locus file after optional filtering by chromosome and
		the second dimension is in the same order as the list of in signal files.
		If no in signal files are given, this is not returned.
	"""

	seqs, signals_, in_signals_ = [], [], []
	in_width, out_width = in_window // 2, out_window // 2
	if signals is None and in_signals is None:
		out_width = 0

	# Load the sequences
	loci = _interleave_loci(loci, chroms)

	if isinstance(sequences, str):
		sequences = pyfaidx.Fasta(sequences)

	# Load the signal and optional in signal tracks if filenames are given
	signals = _load_signals(signals)
	in_signals = _load_signals(in_signals)

	desc = "Loading Loci"
	d = not verbose

	max_width = max(in_width, out_width)
	for chrom, start, end in tqdm(loci.values, disable=d, desc=desc):
		mid = start + (end - start) // 2
		start = mid - max(out_width, in_width) - max_jitter
		end = mid + max(out_width, in_width) + max_jitter

		if start < 0 or end >= len(sequences[chrom]): 
			continue

		if n_loci is not None and len(seqs) == n_loci:
			break 

		# Extract a window of signal using the output size
		start = mid - out_width - max_jitter
		end = mid + out_width + max_jitter + (out_window % 2)

		if signals is not None:
			signal = _extract_locus_signal(signals, chrom, start, end)

			if min_counts is not None and signal[target_idx].sum() < min_counts:
				continue

			if max_counts is not None and signal[target_idx].sum() > max_counts:
				continue

			signals_.append(signal)

		# Extract a window of signal using the input size
		start = mid - in_width - max_jitter
		end = mid + in_width + max_jitter + (in_window % 2)

		if in_signals is not None:
			in_signal = _extract_locus_signal(in_signals, chrom, start, end)
			in_signals_.append(in_signal)

		# Extract a window of sequence using the input size
		if isinstance(sequences, dict):
			seq = sequences[chrom][start:end].T
		else:
			seq = one_hot_encode(sequences[chrom][start:end].seq.upper(),
				alphabet=alphabet)

		seqs.append(seq)

	if not isinstance(sequences, dict):
		sequences.close()

	seqs = torch.from_numpy(numpy.stack(seqs))
	if signals is not None:
		signals_ = torch.from_numpy(numpy.stack(signals_))
		
		if in_signals is not None:
			in_signals_ = torch.from_numpy(numpy.stack(in_signals_))
			return seqs, signals_, in_signals_

		return seqs, signals_
	else:
		return seqs			


def read_meme(filename, n_motifs=None):
	"""Read a MEME file and return a dictionary of PWMs.

	This method takes in the filename of a MEME-formatted file to read in
	and returns a dictionary of the PWMs where the keys are the metadata
	line and the values are the PWMs.


	Parameters
	----------
	filename: str
		The filename of the MEME-formatted file to read in


	Returns
	-------
	motifs: dict
		A dictionary of the motifs in the MEME file.
	"""

	motifs = {}

	with open(filename, "r") as infile:
		motif, width, i = None, None, 0

		for line in infile:
			if motif is None:
				if line[:5] == 'MOTIF':
					motif = line.replace('MOTIF ', '').strip("\r\n")
				else:
					continue

			elif width is None:
				if line[:6] == 'letter':
					width = int(line.split()[5])
					pwm = numpy.zeros((width, 4))

			elif i < width:
				pwm[i] = list(map(float, line.strip("\r\n").split()))
				i += 1

			else:
				motifs[motif] = pwm
				motif, width, i = None, None, 0

				if n_motifs is not None and len(motifs) == n_motifs:
					break

	return motifs