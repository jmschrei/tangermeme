# io.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>
# Code adapted from Alex Tseng, Avanti Shrikumar, and Ziga Avsec

from __future__ import annotations

import os
import mmap
import warnings

import numpy
import torch
import pandas

import pyfaidx
import pybigtools

from tqdm import tqdm

from .utils import one_hot_encode  # noqa: F401, importable from here
from .utils import _one_hot_encode_rows
from .utils import _one_hot_rows_mapping
from .utils import _fast_one_hot_encode_fasta
from .utils import characters
from .utils import TangermemeWarning

from memelite.io import read_meme as memelite_read_meme


def _load_exclusion_zones(chrom_lengths, exclusion_lists):
	if exclusion_lists is not None:
		# Initialize the exclusion zones, where overlapping loci must be removed
		exclusion_zones = {}
		for chrom, size in chrom_lengths.items():
			exclusion_zones[chrom] = numpy.zeros(size // 100 + 1, dtype='bool')

		# Fill in the exclusion zones using the provided coordinates
		exclusion_list = _interleave_loci(exclusion_lists)

		for _, (chrom, start, end) in exclusion_list.iterrows():
			# A region on a chromosome absent from the sequences cannot
			# overlap a locus that can be extracted.
			if chrom not in exclusion_zones:
				continue

			# Ends are exclusive, so the last base covered is end - 1 and a
			# region with end <= start covers no base.
			if end <= start:
				continue

			exclusion_zones[chrom][start // 100:(end - 1) // 100 + 1] = True
		
		return exclusion_zones


def _interleave_loci(loci, chroms=None, summits=False):
	"""An internal function for loading and processing the provided loci.

	There are two aspects that have to be considered when processing the loci.
	The first is that the user can pass in either strings containing filenames
	or pandas DataFrames. The second is that the user can pass in a single
	value or a list of values, and if a list of values the resulting dataframes
	must be interleaved.

	If a set of chromosomes is provided, each dataframe will be filtered to
	loci on those chromosomes before interleaving. If a more complicated form
	of filtering is desired, one should pre-filter the dataframes and pass those
	into this function for interleaving.

	If one wishes to center on summits, the input data must be in BED10 format
	where the 10th column (index 9 when zero-indexing) is the relative offset
	of the summit. This will adjust the start and end of the coordinates to
	be centered on the summits.


	Parameters
	----------
	loci: str, pandas.DataFrame, or list of those
		A filename to load, a pandas DataFrame in bed-format, or a list of
		either.

	chroms: list or None, optional
		A set of chromosomes to restrict the loci to. This is done before
		interleaving to ensure balance across sets of loci. If None, do not
		do filtering. Default is None.

	summits: bool, optional
		Whether to include the summits, which are the 10th column in a BED10 file,
		when loading the dataframe.


	Returns
	-------
	interleaved_loci: pandas.DataFrame
		A single pandas DataFrame that interleaves rows from each of the
		provided examples.
	"""

	if chroms is not None:
		if not isinstance(chroms, (list, tuple)):
			raise ValueError("Provided chroms must be a list.")

		chroms = [str(chrom) for chrom in chroms]

	if isinstance(loci, (str, os.PathLike, pandas.DataFrame)):
		loci = [loci]
	elif not isinstance(loci, (list, tuple)):
		raise ValueError("Provided loci must be a string or pandas " +
			"DataFrame, or a list/tuple of those.")

	###
		
	cols = [0, 1, 2] + ([9] if summits else [])
	names = ['chrom', 'start', 'end'] + (['summit'] if summits else [])

	loci_dfs = []
	for i, df in enumerate(loci):
		# Extract the relevant columns from the dataframes
		if isinstance(df, (str, os.PathLike)):
			df = pandas.read_csv(df, sep='\t', usecols=cols,
				header=None, index_col=False, names=names)
		elif isinstance(df, pandas.DataFrame):
			df = df.iloc[:, cols].copy()
			df.columns = names
		else:
			raise ValueError("Provided loci must be a string or pandas " +
				"DataFrame, or a list/tuple of those.")

		# Chromosome names must be strings so that they match the names used by
		# pyfaidx/pybigtools. Otherwise, genomes whose chromosomes are named
		# "1", "2", etc. get read in as integers by pandas and fail to match.
		df['chrom'] = df['chrom'].astype(str)

		# If using summits, correct the coordinates to be centered on them
		if summits:
			if df.iloc[:, -1].min() < 0:
				raise ValueError("Summits cannot be negative values.")

			if ((df.iloc[:, -1] + df.iloc[:, 1]) > df.iloc[:, 2]).any():
				raise ValueError("Summit + start cannot be larger than end.")

			mid = df['start'] + df['summit']
			w = df['end'] - df['start']
			
			df['start'] = mid - w // 2
			df['end'] = mid + w // 2
			df = df.drop(columns=['summit'])
		
		# If filtering chromosomes, remove loci on unallowed chromosomes
		if chroms is not None:
			df = df[numpy.isin(df['chrom'], chroms)]

		df['idx'] = numpy.arange(len(df)) * len(loci) + i
		loci_dfs.append(df)

	loci = pandas.concat(loci_dfs)
	loci = loci.set_index("idx").sort_index().reset_index(drop=True)
	return loci


def _load_signals(signals):
	"""An internal function for loading signals.

	The passed in signals must be a list but each element can be a string,
	which is interpreted as the filename of a bigwig file to open, a bigwig
	file already opened with `pybigtools.open`, which is used as is and left
	open, or a dictionary where the keys are chromosome names and the values
	are numpy arrays of values across the chromosome. The keys of a dictionary
	are coerced to strings so that they match the chromosome names of the loci.


	Parameters
	----------
	signals: list of strings, pybigtools.BBIRead objects, or dicts, or None
		A list of filenames of bigwig files, opened bigwig files, or
		dictionaries of numpy arrays.


	Returns
	-------
	_signals: list of dicts
		A list of either pointers to opened bigwig files or dictionaries of
		numpy arrays.
	"""

	if signals is None:
		return None

	if not isinstance(signals, (list, tuple)):
		raise ValueError("Signals must be a list or tuple, even when there " +
			"is only one.")

	_signals = []
	for i, signal in enumerate(signals):
		if isinstance(signal, (str, os.PathLike)):
			signal = pybigtools.open(os.fspath(signal))
		elif isinstance(signal, pybigtools.BBIRead):
			pass
		elif not isinstance(signal, dict):
			raise ValueError("Signals must either be filenames, bigWigs " +
				"opened with pybigtools, or dictionaries.")
		elif not isinstance(list(signal.values())[0], numpy.ndarray):
			raise ValueError("Values in dictionaries must be numpy.ndarrays.")
		else:
			signal = {str(key): value for key, value in signal.items()}

		_signals.append(signal)

	return _signals


def _extract_locus_signal(signals, chrom, start, end):
	"""An internal function for extracting signal from a single locus.

	This function takes in a set of signals and a single locus and extracts
	the signal from each one of the loci.


	Parameters
	----------
	signals: list of pybigtools' BBIRead objects or dictionaries
		A list of BBIRead objects (as returned by pybigtools.open()) or dictionaries where the keys are
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

	Notes
	-----
	When `signal` is a dict, each `signal[chrom]` is assumed to be a 1-D
	array indexable by genomic position. Passing a `(n_tracks, length)`
	array under a single chromosome key will silently slice the first axis
	instead of positions and yield mis-shaped output; provide one signal
	dict per track in the outer `signals` list instead.
	"""

	if not isinstance(signals, (list, tuple)):
		raise ValueError("Provided signals must be in the form of a list.")

	values = []
	for i, signal in enumerate(signals):
		# A dict is zero-filled where it has no values, as pybigtools is:
		# a missing chromosome gives zeros and a warning, and positions past
		# the end of the array are NaN in pybigtools and so become zero.
		if isinstance(signal, dict):
			if chrom not in signal:
				warnings.warn(f"{chrom} is not in the signal dictionary. "
					"Using zeros instead.", TangermemeWarning, stacklevel=2)
				values_ = numpy.zeros(end-start, dtype=numpy.float32)
			else:
				values_ = numpy.array(signal[chrom][start:end],
					dtype=numpy.float32)

				if len(values_) < end - start:
					values_ = numpy.pad(values_, (0, end - start - len(values_)))
		else:
			try:
				values_ = numpy.array(signal.values(chrom, start, end), dtype=numpy.float32)
			except (RuntimeError, ValueError, KeyError):
				warnings.warn(
					f"{chrom} {start} {end} not valid bigwig indexes. "
					"Using zeros instead.", TangermemeWarning, stacklevel=2)
				values_ = numpy.zeros(end-start, dtype=numpy.float32)
				
		values_ = numpy.nan_to_num(values_)
		values.append(values_)

	return values


def _allocate_signal(signals, n, width):
	"""An internal function for allocating the output of bigWig signals.

	When every signal is a bigWig, extract_loci writes each locus straight
	into a row of one float32 array rather than keeping an array per locus and
	stacking them. Dictionaries, an empty list of signals and a negative
	width keep the per-locus path, whose outputs and errors differ there.


	Parameters
	----------
	signals: list of pybigtools' BBIRead objects or dictionaries, or None
		The signals, as returned by `_load_signals`.

	n: int
		The largest number of loci that can be kept.

	width: int
		The length of the window extracted from each signal.


	Returns
	-------
	values: numpy.ndarray, shape=(n, len(signals), width), or None
		An uninitialized float32 array, or None when the per-locus path is
		used.

	scratch: numpy.ndarray, shape=(width,), or None
		A float64 array that pybigtools reads each locus into, or None.
	"""

	if signals is None or len(signals) == 0 or width < 0:
		return None, None

	if not all(isinstance(signal, pybigtools.BBIRead) for signal in signals):
		return None, None

	values = numpy.empty((n, len(signals), width), dtype=numpy.float32)
	scratch = numpy.empty(width, dtype=numpy.float64)
	return values, scratch


def _write_locus_signal(signals, chrom, start, end, out, scratch):
	"""An internal function for writing the bigWig signal of a single locus.

	This gives the values `_extract_locus_signal` gives for bigWigs, but
	writes them into `out` instead of new arrays. pybigtools writes only into
	float64 arrays, so each bigWig is read into `scratch` and cast into its
	row of `out`. NaN and infinities are left for the caller to replace.


	Parameters
	----------
	signals: list of pybigtools' BBIRead objects
		A list of BBIRead objects, as returned by pybigtools.open().

	chrom: str
		The name of the chromosome.

	start: int
		The starting coordinate to extract from, inclusive and base-0.

	end: int
		The ending coordinate to extract from, exclusive and base-0.

	out: numpy.ndarray, shape=(len(signals), end-start)
		The float32 array to write the values into.

	scratch: numpy.ndarray, shape=(end-start,)
		A float64 array to read each bigWig into.
	"""

	for i, signal in enumerate(signals):
		try:
			signal.values(chrom, start, end, arr=scratch)
		except (RuntimeError, ValueError, KeyError):
			warnings.warn(
				f"{chrom} {start} {end} not valid bigwig indexes. "
				"Using zeros instead.", TangermemeWarning, stacklevel=2)
			out[i] = 0
			continue

		out[i] = scratch


def _read_signal_windows(signals, chroms, starts, width, out, kind=0,
	max_gap=4096, max_span=65536):
	"""An internal function for reading the bigWig windows of many loci.

	This gives the values `_write_locus_signal` gives when called on each
	window in turn, but makes fewer and more local reads. The windows are
	sorted by chromosome and start, and consecutive windows that overlap or
	are separated by at most `max_gap` bases are read with one call over
	their span. A span is at most `max_span` bases long, or `width` when a
	single window is longer, so the float64 array it is read into stays
	small. pybigtools gives each base the same value however long the read
	is, including `missing` for a base without data and NaN for a base past
	the end of the chromosome, so each window is cut out of the span and
	cast into its row of `out`.

	When a read over a span raises, each window in it is read alone, and a
	window whose read raises is zero-filled and returned as a failure, so the
	caller can warn in the order the loci were given.


	Parameters
	----------
	signals: list of pybigtools' BBIRead objects
		A list of BBIRead objects, as returned by pybigtools.open().

	chroms: list of str
		The chromosome of each window.

	starts: numpy.ndarray, shape=(n,), dtype=int64
		The start of each window, inclusive and base-0.

	width: int
		The length of every window.

	out: numpy.ndarray, shape=(>=n, len(signals), width)
		The float32 array whose first n rows are written, row k with window k.
		NaN and infinities are left for the caller to replace.

	kind: int, optional
		A label copied into each failure. Default is 0.

	max_gap: int, optional
		The largest number of bases between two windows read together.
		Default is 4096.

	max_span: int, optional
		The largest number of bases read in one call. Default is 65536.


	Returns
	-------
	failures: list of tuples
		(k, kind, i, chrom, start, end) for each window k whose read from
		signal i raised.
	"""

	n = len(starts)
	if n == 0:
		return []

	codes, names = pandas.factorize(numpy.array(chroms, dtype=object))
	order = numpy.lexsort((starts, codes))
	s_starts, s_codes = starts[order], codes[order]

	# A group starts at a new chromosome or after a gap of more than max_gap
	# bases. A group is then split so that the windows of each part start
	# within max_span - width bases of each other, which bounds its span.
	head = numpy.ones(n, dtype=bool)
	head[1:] = (s_codes[1:] != s_codes[:-1]) | (s_starts[1:] - s_starts[:-1] -
		width > max_gap)

	step = max_span - width
	if step > 0:
		group_start = s_starts[head][numpy.cumsum(head) - 1]
		part = (s_starts - group_start) // step
		head[1:] |= part[1:] != part[:-1]
	else:
		head[:] = True

	heads = numpy.flatnonzero(head)
	bounds = numpy.append(heads, n).tolist()
	g_starts = s_starts[heads]
	g_ends = s_starts[numpy.array(bounds[1:]) - 1] + width
	offsets = (s_starts - numpy.repeat(g_starts, numpy.diff(bounds))).tolist()
	g_chroms = [names[code] for code in s_codes[heads].tolist()]
	g_starts, g_ends = g_starts.tolist(), g_ends.tolist()
	rows = order.tolist()

	buffer = numpy.empty(max(max_span, width), dtype=numpy.float64)
	scratch = buffer[:width]

	failures = []
	for i, signal in enumerate(signals):
		out_i = out[:, i]

		for g, (chrom, start, end) in enumerate(zip(g_chroms, g_starts,
			g_ends)):
			try:
				signal.values(chrom, start, end, arr=buffer[:end - start])
			except (RuntimeError, ValueError, KeyError):
				for j in range(bounds[g], bounds[g+1]):
					k, s = rows[j], start + offsets[j]

					try:
						signal.values(chrom, s, s + width, arr=scratch)
					except (RuntimeError, ValueError, KeyError):
						failures.append((k, kind, i, chrom, s, s + width))
						out_i[k] = 0
						continue

					out_i[k] = scratch

				continue

			for j in range(bounds[g], bounds[g+1]):
				offset = offsets[j]
				out_i[rows[j]] = buffer[offset:offset + width]

	return failures


def _nan_to_num_rows(values, block_size=2**20):
	"""An internal function for applying numpy.nan_to_num in place.

	The rows are processed in blocks of about `block_size` elements, so the
	masks that numpy.nan_to_num makes stay small, and a block whose values are
	all finite is skipped because numpy.nan_to_num would leave it unchanged.


	Parameters
	----------
	values: numpy.ndarray, shape=(n, ...)
		The float32 array to modify in place.

	block_size: int, optional
		The approximate number of elements in each block. Default is 2**20.


	Returns
	-------
	values: numpy.ndarray, shape=(n, ...)
		The same array, with NaN replaced by zero and infinities by the
		largest finite float32 of the same sign.
	"""

	step = max(1, block_size // max(1, values[0].size))
	for i in range(0, len(values), step):
		block = values[i:i+step]
		if not numpy.isfinite(block).all():
			numpy.nan_to_num(block, copy=False)

	return values


def _read_fasta_windows_mmap(fasta, windows, length, alphabet, ignore):
	"""Encode fasta windows from a memory map of the file, or return None.

	Each window's bytes are gathered through the .fai index that pyfaidx
	read, skipping the line ends, and encoded straight into the output. None
	is returned, and the caller reads the windows through pyfaidx, when the
	alphabet is not ASCII, the file is compressed or cannot be mapped, the
	index describes lines that the file does not have, or a window holds a
	byte that pyfaidx would remove or decode, so that the result is always
	the one pyfaidx gives.
	"""

	table = _one_hot_rows_mapping(alphabet, ignore)
	faidx = fasta.faidx
	if table is None or length <= 0 or faidx._bgzf:
		return None

	mapping, n_characters = table
	mapping = mapping.copy()
	mapping[[ord('\n'), ord('\r')]] = -3
	mapping[128:] = -3

	chroms, starts = zip(*windows)
	codes, names = pandas.factorize(numpy.array(chroms, dtype=object))
	records = [faidx.index[name] for name in names]

	starts = numpy.array(starts, dtype=numpy.int64)
	offsets = numpy.array([r.offset for r in records], dtype=numpy.int64)[codes]
	line_bases = numpy.array([r.lenc for r in records], dtype=numpy.int64)[codes]
	line_bytes = numpy.array([r.lenb for r in records], dtype=numpy.int64)[codes]

	if (starts < 0).any() or (offsets < 0).any() or (line_bases < 1).any() or \
			(line_bytes < line_bases).any():
		return None

	try:
		fasta_map = mmap.mmap(faidx.file.fileno(), 0, access=mmap.ACCESS_READ)
	except (AttributeError, OSError, ValueError):
		return None

	X = numpy.empty((len(windows), n_characters, length), dtype=numpy.int8)
	try:
		data = numpy.frombuffer(fasta_map, dtype=numpy.uint8)
		try:
			status = _fast_one_hot_encode_fasta(X, data, starts, offsets,
				line_bases, line_bytes, mapping)
		finally:
			del data
	finally:
		fasta_map.close()

	if status == -2:
		return None

	if status >= 0:
		raise ValueError("Encountered character that is not in " +
			"`alphabet` or in `ignore`.")

	return X


def _read_fasta_windows(fasta, windows, length, alphabet, ignore):
	"""One-hot encode windows of a pyfaidx.Fasta opened from a path.

	`windows` holds a (chrom, start) pair for each window, each covering
	`length` bases inside its record. The result is identical to fetching
	each window with pyfaidx and encoding the strings with
	_one_hot_encode_rows, including its errors, which is what is done when
	the windows cannot be read from a memory map of the file.
	"""

	X = _read_fasta_windows_mmap(fasta, windows, length, alphabet, ignore)
	if X is None:
		seqs = []
		for chrom, start in windows:
			seq = fasta[chrom][start:start + length]
			if not isinstance(seq, str):
				seq = seq.seq

			seqs.append(seq)

		X = _one_hot_encode_rows(seqs, alphabet=alphabet, ignore=ignore)

	return X


def extract_loci(
	loci: str | os.PathLike | pandas.DataFrame | list,
	sequences: str | os.PathLike | pyfaidx.Fasta | dict,
	signals: list | None = None,
	in_signals: list | None = None,
	chroms: list[str] | None = None,
	in_window: int = 2114,
	out_window: int = 1000,
	max_jitter: int = 0,
	min_counts: float | None = None,
	max_counts: float | None = None,
	target_idx: int = 0,
	n_loci: int | None = None,
	summits: bool = False,
	alphabet: list[str] | tuple[str, ...] = ['A', 'C', 'G', 'T'],
	ignore: list[str] = ['N'],
	exclusion_lists: str | os.PathLike | pandas.DataFrame | list | None = None,
	return_mask: bool = False,
	verbose: bool = False,
) -> torch.Tensor | list[torch.Tensor]:
	"""Extract sequence and signal information for each provided locus.

	This function will take in a set of loci, sequences, and optionally signals,
	and return the sequences and signals at each of the loci. Each of these
	parameters can be a filename, which is loaded internally, or an appropriate
	Python object (see below for details). The nomenclature `in/out` refers to
	the expected inputs and outputs of the downstream machine learning model,
	not this function.

	For each locus a sequence window of size `in_window` will be extracted from
	the sequences file and each of the `in_signals` files if provided, and
	a window of size `out_window` will be extracted from each of the `signals`
	files if provided. These windows are centered at the middle of the provided
	regions but will all be of the same size, regardless of the size of the peak.
	The middle of a locus is `mid = start + (end - start) // 2`, and a window
	of size `w` covers `[mid - w // 2, mid + w // 2 + w % 2)`, so an odd window
	reaches one base further to the right than to the left.

	If `max_jitter` is provided, it will expand the windows for both the input
	and output. The results are not actually jittered, but this expanded window
	allows for downstream data generators to create jittered data while
	reducing the memory footprint of the returned data.

	There are a few reasons that the returned elements may not match one-to-one
	with the provided loci:

		- (1) If the input window, or the output window when `signals` is
		  given, falls off either end of its chromosome after accounting for
		  jitter, the locus will be removed.

		- (2) If any of the loci fall on chromosomes not in a provided list,
		  they will be removed.

		- (3) If min_counts or max_counts are specified and the locus has a
		  number of counts not in those boundaries, the locus will be removed.

		- (4) If the windows overlap an exclusion region, as described below,
		  the locus will be removed.

	If exclusion lists are provided, they will be used to filter out loci that
	fall in 100bp chunks that also include any of the regions in any of the
	exclusion lists. Ends are exclusive for both the regions and the windows.
	For example, if one of the exclusion lists has an element that is

		chr7    108    234

	loci will be removed if any of their bp fall within chr7 100 300, and an
	element `chr7 100 200` removes loci with any bp in chr7 100 200 but not
	those starting at 200. Regions with `end <= start` cover no bp, and regions
	on chromosomes that are not in `sequences` are ignored.

	A ValueError is raised when a locus is on a chromosome that is not in
	`sequences`, since this usually means that the chromosome names of the
	loci and the sequences do not match, and when no loci remain after
	filtering. Use `chroms` to restrict the loci to the chromosomes in
	`sequences`.


	Parameters
	----------
	loci: str, os.PathLike, pandas.DataFrame, or list/tuple of such
		Either the path to a bed file or a pandas DataFrame object containing
		three columns: the chromosome, the start, and the end, of each locus
		to train on. The three columns are taken positionally regardless of
		what they are named, and the chromosome column is coerced to a string
		so that it matches the record names used by the sequences.
		Alternatively, a list or tuple of paths/DataFrames where the
		intention is to train on the interleaved concatenation, i.e., when you
		want to train on peaks and negatives.

	sequences: str, os.PathLike, pyfaidx.Fasta, or dictionary
		Either the path to a fasta file to read from, a pyfaidx.Fasta object,
		or a dictionary where the keys are the unique set of chromosomes and
		the values are one-hot encoded sequences of shape (len(alphabet),
		chromosome length) as numpy arrays, memory maps, or torch tensors. A
		fasta file opened from a path is closed before returning; a
		pyfaidx.Fasta object is left open. The keys of a dictionary are
		coerced to strings, and the returned sequences have the dtype of its
		values.

	signals: list or None, optional
		A list whose elements are each a path to a bigwig file, which will be
		read using pybigtools, a bigwig file already opened with
		`pybigtools.open`, which is left open, or a dictionary where the keys
		are chromosomes and the values are numpy arrays or memory maps of the
		signal across each chromosome. The keys of a dictionary are coerced to
		strings. A chromosome missing from a bigwig or a dictionary gives zeros
		and a TangermemeWarning, positions past the end of the chromosome or
		array give zeros, NaN values become zero, and infinities become the
		largest finite float32 of the same sign. If None, no signal tensor is
		returned. Default is None.

	in_signals: list or None, optional
		The same as `signals`, but extracted using the input window rather
		than the output window. If None, no tensor is returned. Default is
		None.

	chroms: list or None, optional
		A set of chromosomes to extract loci from. Loci in other chromosomes
		in the locus file are ignored. Entries are coerced to strings, so
		`[1, 2]` and `['1', '2']` are equivalent. If None, all loci are
		used. Default is None.

	in_window: int, optional
		The input window size. Default is 2114.

	out_window: int, optional
		The output window size, used only for `signals`. When `signals` is
		None it has no effect, including on which loci fit on their
		chromosomes. Default is 1000.

	max_jitter: int, optional
		The maximum amount of jitter to add, in either direction, to the
		midpoints that are passed in. Default is 0.

	min_counts: float or None, optional
		The minimum number of counts, summed across the output window of
		`signals[target_idx]`, needed to be kept. A locus with exactly this
		many counts is kept. Requires `signals`. If None, no minimum. Default
		is None.

	max_counts: float or None, optional
		The maximum number of counts, summed across the output window of
		`signals[target_idx]`, needed to be kept. A locus with exactly this
		many counts is kept. Requires `signals`. If None, no maximum. Default
		is None.

	target_idx: int, optional
		When specifying `min_counts` or `max_counts`, the index into `signals`
		of the single signal to use when determining if a region has a number
		of counts in that range. Negative values index from the end, and a
		value out of range raises a ValueError. Default is 0.

	n_loci: int or None, optional
		A cap on the number of loci to return, which must be at least 1. Note
		that this is not the number of loci that are considered. The
		difference is that some loci may be filtered out for various reasons,
		and those are not counted towards the total. If None, no cap. Default
		is None.

	summits: bool, optional
		Whether to return a region centered around the summit instead of the center
		between the start and end. If True, it will add the 10th column (index 9)
		to the start to get the center of the window, and so the data must be in 
		narrowPeak format.

	alphabet : list, tuple, or str
		A pre-defined alphabet where the ordering of the symbols is the same
		as the index into the returned tensor, i.e., for the alphabet ['A', 'B']
		the returned tensor will have a 1 at index 0 if the character was 'A'.
		The fasta sequence is upper-cased before encoding. A character that is
		in neither `alphabet` nor `ignore` raises a ValueError. Only used when
		`sequences` is a fasta file. Default is ['A', 'C', 'G', 'T'].

	ignore: list, optional
		A list of characters to ignore in the sequence, meaning that no bits
		are set to 1 in the returned one-hot encoding. Put another way, the
		sum across characters is equal to 1 for all positions except those
		where the original sequence is in this list. Default is ['N'].

	exclusion_lists: str, os.PathLike, pandas.DataFrame, list, or None, optional
		Regions where overlapping loci should be filtered out, given either as a
		filename to a BED-formatted file, a pandas DataFrame in bed-format, or a
		list of either. If None, no filtering is performed based on exclusion
		zones. Default is None.

	return_mask: bool, optional
		Whether to return a tensor with one entry for each locus remaining after
		filtering by `chroms`, in the interleaved order, which is False when the
		locus was removed for any other reason or was not reached before the
		`n_loci` cap. Default is False.

	verbose: bool, optional
		Whether to display a progress bar while loading. Default is False.


	Returns
	-------
	seqs: torch.tensor, shape=(n, len(alphabet), in_window+2*max_jitter)
		The extracted sequences in the same order as the loci in the locus
		file after optional filtering by chromosome, as a contiguous tensor of
		dtype int8, or of the dtype of the values of `sequences` when it is a
		dictionary.

	signals: torch.tensor, shape=(n, len(signals), out_window+2*max_jitter)
		The extracted signals where the first dimension is in the same order
		as loci in the locus file after optional filtering by chromosome and
		the second dimension is in the same order as the list of signal files.
		If no signal files are given, this is not returned.

	in_signals: torch.tensor, shape=(n, len(in_signals), in_window+2*max_jitter)
		The extracted in signals where the first dimension is in the same order
		as loci in the locus file after optional filtering by chromosome and
		the second dimension is in the same order as the list of in signal files.
		If no in signal files are given, this is not returned.

	kept_mask: torch.tensor, shape=(n0,), dtype=bool
		A boolean vector of length equal to the number of peaks remaining after
		filtering by `chroms`, with entries being True if they were kept and
		False if they were filtered out. Applying this mask to the complete set
		of interleaved peaks will yield the returned values. Only returned if
		`return_mask=True`.

	Raises
	------
	ValueError
		If a locus is on a chromosome that is not in `sequences`, if no loci
		remain after filtering, if `min_counts` or `max_counts` is given
		without `signals`, if `target_idx` is out of range, or if `n_loci` is
		less than 1.
	"""

	if n_loci is not None and n_loci < 1:
		raise ValueError("n_loci must be at least 1 or None.")

	signals = _load_signals(signals)
	in_signals = _load_signals(in_signals)

	if min_counts is not None or max_counts is not None:
		if signals is None:
			raise ValueError("min_counts and max_counts are measured on " +
				"signals, so signals must be provided.")

		if not -len(signals) <= target_idx < len(signals):
			raise ValueError("target_idx {} is out of range for {} signals."
				.format(target_idx, len(signals)))

	seqs, signals_, in_signals_ = [], [], []
	kept_mask = []
	in_width, out_width = in_window // 2, out_window // 2
	out_extra = out_window % 2
	if signals is None:
		out_width, out_extra = 0, 0

	# Extract the length of each chromosome. Track whether we opened the
	# fasta ourselves so we know whether we are allowed to close it on
	# exit; a caller-provided pyfaidx.Fasta is theirs to manage.
	chrom_lengths = {}
	opened_fasta = False
	if isinstance(sequences, (str, os.PathLike)):
		sequences = pyfaidx.Fasta(os.fspath(sequences))
		opened_fasta = True
		for key, value in sequences.items():
			chrom_lengths[str(key)] = len(value)
	elif isinstance(sequences, pyfaidx.Fasta):
		for key, value in sequences.items():
			chrom_lengths[str(key)] = len(value)
	else:
		sequences = {str(key): value for key, value in sequences.items()}
		for key, value in sequences.items():
			chrom_lengths[key] = value.shape[-1]


	# Create the exclusion zones from the exclusion lists, if provided
	exclusion_zones = _load_exclusion_zones(chrom_lengths, exclusion_lists)

	# Load the loci
	loci = _interleave_loci(loci, chroms, summits=summits)

	missing = sorted(set(loci['chrom']) - set(chrom_lengths))
	if len(missing) > 0:
		if opened_fasta:
			sequences.close()

		raise ValueError("Loci are on chromosomes that are not in the " +
			"sequences: {}. Pass `chroms` to select the chromosomes to use."
			.format(", ".join(missing)))

	desc = "Loading Loci"
	d = not verbose

	# Each window is [mid - w//2, mid + w//2 + w%2), so an odd window reaches
	# one base further right than left. The locus is kept only if the union
	# of the windows fits on the chromosome.
	left = max(in_width, out_width) + max_jitter
	right = max(in_width + in_window % 2, out_width + out_extra) + max_jitter

	# bigWig signals are written into row len(seqs) of a preallocated array,
	# so a locus that is filtered out is overwritten by the next one.
	n_max = len(loci) if n_loci is None else min(len(loci), n_loci)
	out_values, out_scratch = _allocate_signal(signals, n_max,
		out_window + 2 * max_jitter)
	in_values, in_scratch = _allocate_signal(in_signals, n_max,
		in_window + 2 * max_jitter)
	count_filter = min_counts is not None or max_counts is not None

	# Without a count filter every locus that reaches the signals is kept, so
	# when every signal is a bigWig the loop records only the chromosome and
	# midpoint of each kept locus, and the windows are read after the loop,
	# sorted by position and grouped (_read_signal_windows).
	defer = (not count_filter
		and (out_values is not None or in_values is not None)
		and (signals is None or out_values is not None)
		and (in_signals is None or in_values is not None)
		and pandas.api.types.is_integer_dtype(loci['start'])
		and pandas.api.types.is_integer_dtype(loci['end']))
	window_chroms, window_mids = [], []

	for chrom, start, end in tqdm(loci.values, disable=d, desc=desc):
		mid = start + (end - start) // 2

		start = mid - left
		end = mid + right

		# Does it fall off the end of a chromosome?
		if start < 0 or end > chrom_lengths[str(chrom)]:
			kept_mask.append(False)
			continue

		if exclusion_zones is not None:
			s, e = start // 100, (end - 1) // 100 + 1
			if exclusion_zones[str(chrom)][s:e].any():
				kept_mask.append(False)
				continue

		# Extract a window of signal using the output size
		start = mid - out_width - max_jitter
		end = mid + out_width + max_jitter + (out_window % 2)

		if signals is not None:
			if out_values is None:
				signal = _extract_locus_signal(signals, str(chrom), start, end)
			elif not defer:
				signal = out_values[len(seqs)]
				_write_locus_signal(signals, str(chrom), start, end, signal,
					out_scratch)

				# The counts are summed after NaN and infinities are replaced.
				if count_filter:
					numpy.nan_to_num(signal[target_idx], copy=False)

			if min_counts is not None and signal[target_idx].sum() < min_counts:
				kept_mask.append(False)
				continue

			if max_counts is not None and signal[target_idx].sum() > max_counts:
				kept_mask.append(False)
				continue

			if out_values is None:
				signals_.append(signal)

		# Extract a window of signal using the input size
		start = mid - in_width - max_jitter
		end = mid + in_width + max_jitter + (in_window % 2)

		if in_signals is not None:
			if in_values is None:
				in_signal = _extract_locus_signal(in_signals, str(chrom), start,
					end)
				in_signals_.append(in_signal)
			elif not defer:
				_write_locus_signal(in_signals, str(chrom), start, end,
					in_values[len(seqs)], in_scratch)

		# Extract a window of sequence using the input size. The windows of a
		# fasta opened from a path are read together after the loop.
		if isinstance(sequences, dict):
			seq = sequences[str(chrom)][:, start:end]
		elif opened_fasta:
			seq = (str(chrom), start)
		else:
			# A Fasta opened with as_raw=True returns strings rather than
			# pyfaidx.Sequence objects.
			seq = sequences[str(chrom)][start:end]
			if not isinstance(seq, str):
				seq = seq.seq

		kept_mask.append(True)
		seqs.append(seq)

		if defer:
			window_chroms.append(str(chrom))
			window_mids.append(mid)

		if n_loci is not None and len(seqs) == n_loci:
			break 

	# The failed reads are warned about in the order the loop would have
	# made them: by locus, then signals before in_signals, then by signal.
	if defer and len(window_mids) > 0:
		mids = numpy.array(window_mids, dtype=numpy.int64)
		failures = []

		if signals is not None:
			failures += _read_signal_windows(signals, window_chroms,
				mids - out_width - max_jitter, out_window + 2 * max_jitter,
				out_values, kind=0)

		if in_signals is not None:
			failures += _read_signal_windows(in_signals, window_chroms,
				mids - in_width - max_jitter, in_window + 2 * max_jitter,
				in_values, kind=1)

		for _, _, _, chrom, start, end in sorted(failures):
			warnings.warn(f"{chrom} {start} {end} not valid bigwig indexes. "
				"Using zeros instead.", TangermemeWarning)

	if opened_fasta:
		try:
			if len(seqs) > 0:
				seqs = _read_fasta_windows(sequences, seqs, in_window +
					2 * max_jitter, alphabet, ignore)
		finally:
			sequences.close()

	if len(seqs) == 0:
		raise ValueError("No loci remain after filtering. Loci are removed " +
			"when they are not on a chromosome in `chroms`, when their windows " +
			"run off the end of a chromosome, when they overlap an exclusion " +
			"region, or when their counts fall outside min_counts/max_counts.")

	# Figure out how to format the outputs depending on the provided parameters.
	# numpy.stack keeps the memory layout of its inputs, so a stack of slices
	# of Fortran-ordered arrays is not contiguous by default. Fasta windows
	# were kept as strings, or read from a fasta opened from a path, and are
	# encoded together, straight into one C-contiguous array.
	if isinstance(sequences, dict):
		seqs = numpy.ascontiguousarray(numpy.stack(seqs))
	elif not opened_fasta:
		seqs = _one_hot_encode_rows(seqs, alphabet=alphabet, ignore=ignore)

	seqs = torch.from_numpy(seqs)
	y_return = [seqs]

	# A preallocated array is trimmed to the kept loci with a view, which is
	# contiguous; its rows past the last kept locus were never written.
	if signals is not None:
		if out_values is None:
			y_return.append(torch.from_numpy(numpy.stack(signals_)))
		else:
			out_values = _nan_to_num_rows(out_values[:len(seqs)])
			y_return.append(torch.from_numpy(out_values))

	if in_signals is not None:
		if in_values is None:
			y_return.append(torch.from_numpy(numpy.stack(in_signals_)))
		else:
			in_values = _nan_to_num_rows(in_values[:len(seqs)])
			y_return.append(torch.from_numpy(in_values))
		
	if return_mask:
		# Loci after the n_loci cap was reached were never examined and are
		# not returned, so they are False.
		kept_mask += [False] * (len(loci) - len(kept_mask))
		kept_mask = torch.tensor(kept_mask)
		y_return.append(kept_mask)

	return y_return[0] if len(y_return) == 1 else y_return


def one_hot_to_fasta(
	X: torch.Tensor | numpy.ndarray,
	filename: str,
	mode: str = 'w',
	headers: list[str] | None = None,
	alphabet: list[str] = ['A', 'C', 'G', 'T'],
) -> None:
	"""Write out one-hot encoded sequences to a FASTA file.

	This function will take a set of one-hot encoded sequences and convert them
	to characters and write them out in FASTA format. If headers are provided
	for each sequence, these are used, otherwise the numeric index is used.


	Parameters
	----------
	X: torch.Tensor, shape=(-1, len(alphabet), length)
		A set of one-hot encoded sequences to write out.

	filename: str
		The path to the FASTA file to write to.

	mode: str, optional
		The file mode to open `filename` with, e.g. 'w' to overwrite or 'a' to
		append. Default is 'w'.

	headers: list of str or None, optional
		A list of one header per sequence in `X`. If None, the numeric index of
		each sequence is used as its header. Default is None.

	alphabet: list or tuple, optional
		A pre-defined alphabet where the ordering of the symbols is the same as
		the index into the one-hot encoding. Default is ['A', 'C', 'G', 'T'].
	"""
	
	with open(filename, mode=mode) as outfile:
		for i, X_seq in enumerate(X):
			X_chars = characters(X_seq, alphabet=alphabet)
			
			if headers is None:
				outfile.write("> {}\n".format(i))
			else:
				outfile.write("> {}\n".format(headers[i]))
			
			for start in range(0, len(X_chars), 80):
				outfile.write(X_chars[start:start+80] + "\n")
				
			outfile.write("\n")


def read_meme(filename: str, n_motifs: int | None = None) -> dict[str, torch.Tensor]:
	"""Read a MEME file and return a dictionary of PWMs.

	This method takes in the filename of a MEME-formatted file to read in
	and returns a dictionary of the PWMs where the keys are the metadata
	line and the values are the PWMs.

	This function is a wrapper around the memelite one, except that it returns
	torch tensors instead of numpy arrays.


	Parameters
	----------
	filename: str
		The filename of the MEME-formatted file to read in.

	n_motifs: int or None, optional
		If provided, stop reading after this many motifs have been parsed. If
		None, read all motifs in the file. Default is None.


	Returns
	-------
	motifs: dict
		A dictionary of the motifs in the MEME file.
	"""

	motifs = memelite_read_meme(filename, n_motifs=n_motifs)
	motifs = {name: torch.from_numpy(pwm) for name, pwm in motifs.items()}
	return motifs


def read_vcf(filename: str) -> pandas.DataFrame:
	"""Read a VCF file into a pandas DataFrame

	This function takes in the name of a file that is VCF formatted and returns
	a pandas DataFrame with the comments filtered out. This will only return the
	first 9 columns (CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO, FORMAT); any
	per-sample genotype columns past column 9 are silently dropped.

	Compressed VCFs are read transparently when pandas detects the compression
	from the filename extension (e.g., `.vcf.gz` works via `pandas.read_csv`).
	BCF (binary VCF) files are NOT supported by this function.


	Parameters
	----------
	filename: str
		The path to the VCF-formatted file to read in. May be plain `.vcf` or
		gzip-compressed `.vcf.gz`.


	Returns
	-------
	vcf: pandas.DataFrame
		A pandas DataFrame containing the rows, with columns CHROM, POS, ID,
		REF, ALT, QUAL, FILTER, INFO, FORMAT.
	"""

	names = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", 
		"FORMAT"]
	dtypes = {name: str for name in names}
	dtypes['POS'] = int

	vcf = pandas.read_csv(filename, delimiter='\t', comment='#', names=names, 
		dtype=dtypes, usecols=range(9))
	return vcf
