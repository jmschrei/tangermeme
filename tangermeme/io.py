# io.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>
# Code adapted from Alex Tseng, Avanti Shrikumar, and Ziga Avsec

from __future__ import annotations

import os
import csv
import sys
import mmap
import ctypes
import warnings

from concurrent.futures import ThreadPoolExecutor

import numpy
import torch
import pandas

import figwig
import pyfaidx

from tqdm import tqdm

from .utils import one_hot_encode  # noqa: F401, importable from here
from .utils import _one_hot_encode_rows
from .utils import _one_hot_rows_mapping
from .utils import _one_hot_encode_fasta
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


# The start of a line of a BED file that holds no interval: a track or a
# browser line, or a comment.
_BED_HEADER = r'#|(track|browser)(\s|$)'


def _read_bed(filename, cols, names):
	"""An internal function for reading columns of a BED file.

	Track and browser lines, and lines that start with #, are skipped, as
	UCSC's tools skip them. A file without them is read once. Otherwise
	pandas, which takes the number of columns from the first line, reads the
	columns of the other lines wrongly, so the file is read again without
	those lines.


	Parameters
	----------
	filename: str or os.PathLike
		The BED file, which may be compressed.

	cols: list of int
		The columns to read.

	names: list of str
		The names to give the columns.


	Returns
	-------
	df: pandas.DataFrame
		The columns of the file's intervals.
	"""

	df = pandas.read_csv(filename, sep='\t', usecols=cols, header=None,
		index_col=False, names=names)
	if not df['chrom'].astype(str).str.match(_BED_HEADER).any():
		return df

	# One row per line, blank lines included, so that a row's index is its
	# line number, which is what skiprows takes.
	lines = pandas.read_csv(filename, sep='\t', usecols=[0], header=None,
		names=['chrom'], dtype=str, skip_blank_lines=False,
		quoting=csv.QUOTE_NONE)
	skip = numpy.flatnonzero(lines['chrom'].fillna('').str.match(_BED_HEADER))

	return pandas.read_csv(filename, sep='\t', usecols=cols, header=None,
		index_col=False, names=names, skiprows=skip.tolist())


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
			df = _read_bed(df, cols, names)
		elif isinstance(df, pandas.DataFrame):
			df = df.iloc[:, cols].copy()
			df.columns = names
		else:
			raise ValueError("Provided loci must be a string or pandas " +
				"DataFrame, or a list/tuple of those.")

		# Chromosome names must be strings so that they match the names used by
		# pyfaidx/figwig. Otherwise, genomes whose chromosomes are named
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


def _kept_loci(loci, chrom_lengths, exclusion_zones, left, right):
	"""An internal function for finding the loci whose windows can be read.

	A locus is kept when the union of its windows, `[mid - left, mid +
	right)` around its middle `mid = start + (end - start) // 2`, fits on its
	chromosome and does not overlap an excluded 100 bp chunk. Integer
	coordinates are checked together as int64, which gives the values that
	Python ints do while every coordinate and window is far inside its
	range. Other coordinates, such as floats, are checked one locus at a time
	with Python arithmetic, so that they raise or pass as they did before.


	Parameters
	----------
	loci: pandas.DataFrame
		The loci, with columns `chrom`, `start` and `end`, on chromosomes
		that are all keys of `chrom_lengths`.

	chrom_lengths: dict
		The length of each chromosome.

	exclusion_zones: dict or None
		A boolean array per chromosome of the 100 bp chunks that are
		excluded, as returned by `_load_exclusion_zones`, or None.

	left: int
		The number of bases the windows reach left of the middle.

	right: int
		The number of bases the windows reach right of the middle.


	Returns
	-------
	idxs: numpy.ndarray, dtype=int64, shape=(n,)
		The position in `loci` of each kept locus.

	mids: numpy.ndarray, shape=(n,)
		The middle of each kept locus, as int64 for integer coordinates.
	"""

	starts, ends = loci['start'].to_numpy(), loci['end'].to_numpy()
	bound = 2 ** 40

	if (starts.dtype.kind in 'iu' and ends.dtype.kind in 'iu' and
			0 <= left + right and max(abs(left), abs(right)) < bound and
			all(-bound < int(x.min(initial=0)) and int(x.max(initial=0)) < bound
				for x in (starts, ends))):
		starts, ends = starts.astype(numpy.int64), ends.astype(numpy.int64)
		mids = starts + (ends - starts) // 2

		codes, names = pandas.factorize(loci['chrom'])
		lengths = numpy.array([chrom_lengths[name] for name in names],
			dtype=numpy.int64)[codes]

		# Does it fall off the end of a chromosome?
		keep = (mids - left >= 0) & (mids + right <= lengths)

		# Does it overlap an excluded chunk? The chunks of every chromosome are
		# placed end to end, so that one sorted array holds every excluded
		# chunk, and the chunks [s, e) of a window hold one when fewer excluded
		# chunks come before s than before e.
		if exclusion_zones is not None:
			offsets = numpy.cumsum([0] + [len(exclusion_zones[name]) for name
				in names], dtype=numpy.int64)
			excluded = numpy.concatenate([numpy.zeros(0, dtype=numpy.int64)] +
				[numpy.flatnonzero(exclusion_zones[name]) + offsets[k]
				for k, name in enumerate(names)])

			idxs = numpy.flatnonzero(keep)
			offsets = offsets[codes[idxs]]
			s = (mids[idxs] - left) // 100 + offsets
			e = (mids[idxs] + right - 1) // 100 + 1 + offsets
			keep[idxs[numpy.searchsorted(excluded, s) <
				numpy.searchsorted(excluded, e)]] = False

		idxs = numpy.flatnonzero(keep)
		return idxs, mids[idxs]

	idxs, mids = [], []
	for i, (chrom, start, end) in enumerate(loci.values):
		mid = start + (end - start) // 2

		start = mid - left
		end = mid + right

		# Does it fall off the end of a chromosome?
		if start < 0 or end > chrom_lengths[str(chrom)]:
			continue

		if exclusion_zones is not None:
			s, e = start // 100, (end - 1) // 100 + 1
			if exclusion_zones[str(chrom)][s:e].any():
				continue

		idxs.append(i)
		mids.append(mid)

	return numpy.array(idxs, dtype=numpy.int64), numpy.array(mids)


def _load_signals(signals):
	"""An internal function for loading signals.

	The passed in signals must be a list but each element can be a string or
	os.PathLike, which is interpreted as the filename of a bigwig file to open
	with figwig, a bigwig file already opened with `figwig.BigWigReader`,
	which is used as is, or a dictionary where the keys are chromosome names
	and the values are numpy arrays of values across the chromosome. The keys
	of a dictionary are coerced to strings so that they match the chromosome
	names of the loci. A bigwig file opened with `pybigtools.open` is used as
	is, with a FutureWarning, until tangermeme 1.9.0.


	Parameters
	----------
	signals: list of strings, figwig.BigWigReader objects, or dicts, or None
		A list of filenames of bigwig files, opened bigwig files, or
		dictionaries of numpy arrays.


	Returns
	-------
	_signals: list
		A list of figwig.BigWigReader objects, bigwig files opened with
		pybigtools, or dictionaries of numpy arrays.
	"""

	if signals is None:
		return None

	if not isinstance(signals, (list, tuple)):
		raise ValueError("Signals must be a list or tuple, even when there " +
			"is only one.")

	_signals = []
	for i, signal in enumerate(signals):
		if isinstance(signal, (str, os.PathLike)):
			path = os.fspath(signal)
			if "://" in path:
				raise ValueError("Signal {} is a URL, {}, but bigwig files are "
					"read from local paths only; download it first.".format(i,
					path))

			signal = figwig.BigWigReader(path)
		elif isinstance(signal, figwig.BigWigReader):
			pass
		elif type(signal).__module__ == "pybigtools":
			warnings.warn("Signal {} is a bigwig opened with pybigtools, which "
				"is deprecated and will not be accepted from tangermeme 1.9.0; "
				"pass its path or a figwig.BigWigReader.".format(i),
				FutureWarning, stacklevel=3)

			# Reading values from a bigBed panics in pybigtools, which raises
			# an exception that is not an Exception.
			if getattr(signal, "is_bigbed", False):
				raise ValueError("Signals must be bigWig files, but signal {} is "
					"a bigBed file opened with pybigtools.".format(i))
		elif not isinstance(signal, dict):
			raise ValueError("Signals must either be filenames, bigWigs " +
				"opened with figwig.BigWigReader, or dictionaries.")
		elif len(signal) > 0 and not isinstance(next(iter(signal.values())),
				numpy.ndarray):
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
	signals: list of dictionaries or bigwig files opened with pybigtools
		A list of dictionaries where the keys are chromosomes and the values
		are the signal at each position in the chromosome, or bigwig files
		opened with `pybigtools.open`, whose `values` method is called.

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
		# A dict is zero-filled where it has no values, as a bigwig is: a
		# missing chromosome gives zeros and a warning, and positions past
		# the end of the array are zero, as past the end of a chromosome.
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


# The start of the warning figwig gives for windows on chromosomes that a
# bigWig does not have, which `_extract_signals` replaces with its own.
_FIGWIG_ABSENT_CHROMS = r"\d+ windows are on chromosomes not in "

# The number of values of the target signal that `extract_loci` holds at once
# while it measures counts for `min_counts` and `max_counts`.
_COUNT_VALUES = 1 << 24


# The number of blocks below which `_nan_to_num_rows` runs in the calling
# thread.
_NAN_TO_NUM_MIN_BLOCKS = 16


def _cpu_count():
	"""The number of CPUs this process may run on, which `n_jobs=-1` uses,
	as figwig counts them."""

	if hasattr(os, 'sched_getaffinity'):
		return len(os.sched_getaffinity(0))

	return os.cpu_count() or 1


def _nan_to_num_rows(values, block_size=2**20, n_jobs=1):
	"""An internal function for applying numpy.nan_to_num in place.

	The rows are processed in blocks of about `block_size` elements, so the
	masks that numpy.nan_to_num makes stay small, and a block whose values are
	all finite is skipped because numpy.nan_to_num would leave it unchanged.
	The blocks do not overlap and numpy releases the GIL while it checks and
	replaces them, so they are processed on up to `n_jobs` threads when there
	are at least _NAN_TO_NUM_MIN_BLOCKS, with the same result.


	Parameters
	----------
	values: numpy.ndarray, shape=(n, ...)
		The float32 array to modify in place.

	block_size: int, optional
		The approximate number of elements in each block. Default is 2**20.

	n_jobs: int, optional
		The largest number of threads to use. Default is 1.


	Returns
	-------
	values: numpy.ndarray, shape=(n, ...)
		The same array, with NaN replaced by zero and infinities by the
		largest finite float32 of the same sign.
	"""

	step = max(1, block_size // max(1, values[0].size))
	blocks = range(0, len(values), step)

	def nan_to_num_block(i):
		block = values[i:i+step]
		if not numpy.isfinite(block).all():
			numpy.nan_to_num(block, copy=False)

	if n_jobs > 1 and len(blocks) >= _NAN_TO_NUM_MIN_BLOCKS:
		with ThreadPoolExecutor(min(n_jobs, len(blocks))) as pool:
			list(pool.map(nan_to_num_block, blocks))
	else:
		for i in blocks:
			nan_to_num_block(i)

	return values


def _extract_signals(signals, chroms, starts, width, n_jobs, warn=True):
	"""An internal function for extracting signal from many loci.

	The bigWigs opened with figwig are read together in one call on `n_jobs`
	threads, and dicts and bigWigs opened with pybigtools one locus at a time
	by `_extract_locus_signal`. A locus on a chromosome that a bigWig does not
	have is zero and gives a TangermemeWarning, as it does in
	`_extract_locus_signal`. A file that figwig does not read, such as one
	with a corrupt data block, raises figwig's ValueError.


	Parameters
	----------
	signals: list of figwig.BigWigReader, pybigtools bigwigs or dicts
		A list of signals as returned by `_load_signals`.

	chroms: numpy.ndarray of str, shape=(n,)
		The chromosome of each locus.

	starts: numpy.ndarray of int, shape=(n,)
		The start of each window, inclusive and base-0.

	width: int
		The width of every window.

	n_jobs: int
		The number of threads figwig reads with, or -1 for one per CPU.

	warn: bool, optional
		Whether to give the TangermemeWarnings for chromosomes a signal does
		not have. Default is True.


	Returns
	-------
	values: numpy.ndarray, dtype=float32, shape=(n, len(signals), width)
		The extracted signal at each locus from each of the signals.
	"""

	if not warn:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", TangermemeWarning)
			return _extract_signals(signals, chroms, starts, width, n_jobs)

	readers = [j for j, signal in enumerate(signals)
		if isinstance(signal, figwig.BigWigReader)]

	if len(readers) > 0:
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", message=_FIGWIG_ABSENT_CHROMS,
				category=UserWarning)
			figwig_values = figwig.read_bigwig([signals[j] for j in readers],
				chroms, starts, width, n_jobs=n_jobs)

		_nan_to_num_rows(figwig_values, n_jobs=_cpu_count() if n_jobs == -1
			else int(n_jobs))

	if len(readers) == len(signals):
		values = figwig_values
	else:
		values = numpy.empty((len(starts), len(signals), width),
			dtype=numpy.float32)
		if len(readers) > 0:
			values[:, readers] = figwig_values

	for j in readers:
		absent = ~numpy.isin(chroms, list(signals[j].chrom_sizes))
		for i in numpy.nonzero(absent)[0]:
			warnings.warn(f"{chroms[i]} {starts[i]} {starts[i] + width} not "
				"valid bigwig indexes. Using zeros instead.", TangermemeWarning,
				stacklevel=2)

	others = [j for j in range(len(signals)) if j not in readers]
	if len(others) > 0:
		for i, (chrom, start) in enumerate(zip(chroms, starts.tolist())):
			values[i, others] = _extract_locus_signal([signals[j] for j in
				others], str(chrom), start, start + width)

	return values


# A memory map read by at least this many windows has its page table entries
# dropped on up to n_jobs threads before it is closed. Closing drops them on
# the calling thread alone, which took 34-45 ms for the 1.8 GB of hg38 pages
# that the 167,750 windows of the main benchmark map.
_UNMAP_MIN_WINDOWS = 2048

# The map is split into this many chunks per thread, since the pages that
# were read are not spread evenly over the file.
_UNMAP_CHUNKS_PER_JOB = 8

_MADVISE = None


def _madvise():
	"""libc's madvise, called through ctypes, or None when it is not used.

	ctypes releases the GIL during the call, whereas mmap.mmap.madvise holds
	it, so calls from several threads would run one at a time. It is only used
	on Linux, where MADV_DONTNEED on a shared file mapping drops the page table
	entries of the range and leaves the file and the pages it holds as they
	are.
	"""

	global _MADVISE
	if _MADVISE is None:
		_MADVISE = False
		if sys.platform.startswith('linux') and hasattr(mmap, 'MADV_DONTNEED'):
			try:
				madvise = ctypes.CDLL(None, use_errno=True).madvise
				madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
					ctypes.c_int]
				madvise.restype = ctypes.c_int
				_MADVISE = madvise
			except (OSError, AttributeError):
				pass

	return _MADVISE or None


def _close_map(fasta_map, address, n_windows, n_jobs=1):
	"""Close a read-only memory map of a file.

	Closing unmaps the map, and the kernel drops the page table entry of
	every page that was read, on the calling thread. When the map was read by
	at least _UNMAP_MIN_WINDOWS windows and n_jobs is above 1, those entries
	are first dropped with madvise(MADV_DONTNEED) on chunks of the map, on up
	to n_jobs threads, so that the close has almost nothing left to do. The
	file is not changed. The map is closed in every case, including when
	madvise fails or raises.


	Parameters
	----------
	fasta_map: mmap.mmap
		The map, opened with access=mmap.ACCESS_READ. Nothing may still hold
		a buffer of it.

	address: int or None
		The address of the first byte of the map, or None to close it
		directly.

	n_windows: int
		The number of windows read from the map.

	n_jobs: int, optional
		The largest number of threads to use. Default is 1.
	"""

	try:
		madvise = _madvise()
		if (madvise is not None and address is not None and n_jobs > 1 and
				n_windows >= _UNMAP_MIN_WINDOWS):
			size, page = len(fasta_map), mmap.PAGESIZE
			n_chunks = n_jobs * _UNMAP_CHUNKS_PER_JOB
			step = -(-size // (n_chunks * page)) * page
			chunks = [(address + s, min(step, size - s)) for s in range(0,
				size, step)]

			def drop(chunk):
				madvise(chunk[0], chunk[1], mmap.MADV_DONTNEED)

			try:
				with ThreadPoolExecutor(min(n_jobs, len(chunks))) as pool:
					list(pool.map(drop, chunks))
			except (OSError, RuntimeError):
				pass
	finally:
		fasta_map.close()


def _read_fasta_windows_mmap(fasta, windows, length, alphabet, ignore,
	names=None, n_jobs=1):
	"""Encode fasta windows from a memory map of the file, or return None.

	Each window's bytes are gathered through the .fai index that pyfaidx
	read, skipping the line ends, and encoded straight into the output, on
	at most `n_jobs` threads, which does not change the result. None
	is returned, and the caller reads the windows through pyfaidx, when the
	alphabet is not ASCII, the file is compressed or cannot be mapped, the
	index describes lines that the file does not have, or a window holds a
	byte that pyfaidx would remove or decode, so that the result is always
	the one pyfaidx gives. `windows` is as in _read_fasta_windows.
	"""

	table = _one_hot_rows_mapping(alphabet, ignore)
	faidx = fasta.faidx
	if table is None or length <= 0 or faidx._bgzf:
		return None

	mapping, n_characters = table
	mapping = mapping.copy()
	mapping[[ord('\n'), ord('\r')]] = -3
	mapping[128:] = -3

	if names is None:
		chroms, starts = zip(*windows)
		codes, names = pandas.factorize(numpy.array(chroms, dtype=object))
	else:
		codes, starts = windows

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

	X = numpy.empty((len(starts), n_characters, length), dtype=numpy.int8)
	address = None
	try:
		data = numpy.frombuffer(fasta_map, dtype=numpy.uint8)
		try:
			address = data.ctypes.data
			status = _one_hot_encode_fasta(X, data, starts, offsets,
				line_bases, line_bytes, mapping, n_jobs=n_jobs)
		finally:
			del data
	finally:
		_close_map(fasta_map, address, len(starts), n_jobs=n_jobs)

	if status == -2:
		return None

	if status >= 0:
		raise ValueError("Encountered character that is not in " +
			"`alphabet` or in `ignore`.")

	return X


def _read_fasta_windows(fasta, windows, length, alphabet, ignore,
	names=None, n_jobs=1):
	"""One-hot encode windows of a pyfaidx.Fasta opened from a path.

	`windows` holds a (chrom, start) pair for each window, each covering
	`length` bases inside its record. When `names` is given, it is instead a
	pair of int64 arrays: the index of each window's chromosome into `names`,
	and each window's start. The result is identical to fetching each window
	with pyfaidx and encoding the strings with _one_hot_encode_rows,
	including its errors, which is what is done when the windows cannot be
	read from a memory map of the file. The windows are encoded on at most
	`n_jobs` threads.
	"""

	X = _read_fasta_windows_mmap(fasta, windows, length, alphabet, ignore,
		names=names, n_jobs=n_jobs)
	if X is None:
		if names is not None:
			codes, starts = windows
			windows = zip(numpy.array(names, dtype=object)[codes].tolist(),
				starts.tolist())

		seqs = []
		for chrom, start in windows:
			seq = fasta[chrom][start:start + length]
			if not isinstance(seq, str):
				seq = seq.seq

			seqs.append(seq)

		X = _one_hot_encode_rows(seqs, alphabet=alphabet, ignore=ignore,
			n_jobs=n_jobs)

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
	n_jobs: int = 8,
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
		so that it matches the record names used by the sequences. The track
		and browser lines of a bed file, and its lines that start with #, are
		skipped. Alternatively, a list or tuple of paths/DataFrames where the
		intention is to train on the interleaved concatenation, i.e., when you
		want to train on peaks and negatives.

	sequences: str, os.PathLike, pyfaidx.Fasta, or dictionary
		Either the path to a fasta file to read from, a pyfaidx.Fasta object,
		or a dictionary where the keys are the unique set of chromosomes and
		the values are one-hot encoded sequences of shape (len(alphabet),
		chromosome length) as numpy arrays, memory maps, or torch tensors. A
		fasta file opened from a path is closed before returning; a
		pyfaidx.Fasta object is left open. The windows of a fasta file given
		as a path are read from a memory map of the file, using the .fai index
		that pyfaidx reads or builds. They are read through pyfaidx instead,
		with the same result, when they cannot be read from the map, such as
		for a compressed file, a non-ASCII alphabet, or a window holding a
		byte that pyfaidx would change, like a carriage return inside a line.
		The keys of a dictionary are coerced to strings, and the
		returned sequences have the dtype of its values.

	signals: list or None, optional
		A list whose elements are each a path to a bigwig file, a bigwig file
		already opened with `figwig.BigWigReader`, or a dictionary where the
		keys are chromosomes and the values are numpy arrays or memory maps of
		the signal across each chromosome. The keys of a dictionary are coerced
		to strings. A chromosome missing from a bigwig or a dictionary gives
		zeros and a TangermemeWarning, positions past the end of the chromosome
		or array give zeros, NaN values become zero, and infinities become the
		largest finite float32 of the same sign. The bigwig files are read by
		figwig on `n_jobs` threads, in one call for the kept loci, or in groups
		of loci under `min_counts` and `max_counts`. Only local files are read,
		so a URL raises a ValueError, and a file figwig does not read, such as
		a bigBed or a bigwig with unsorted or overlapping intervals, raises
		figwig's error. A bigwig opened with `pybigtools.open` is deprecated:
		it is read one locus at a time, with a FutureWarning, until tangermeme
		1.9.0. If None, no signal tensor is returned. Default is None.

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
		Whether to display a progress bar while loading the sequences of the
		kept loci. The windows of a fasta file given as a path are read
		together, so its bar fills in one step. Default is False.

	n_jobs: int, optional
		The number of threads that figwig reads the bigwig files given as
		paths with, and that the sequences of a fasta file are read and
		one-hot encoded with, or -1 for one per CPU. The encoding threads are
		also capped by numba's NUMBA_NUM_THREADS. The returned values do not
		depend on it. Default is 8.


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
		without `signals`, if `target_idx` is out of range, if `n_loci` is
		less than 1, or if `n_jobs` is less than 1 and not -1.

	TypeError
		If `n_jobs` is not an integer.
	"""

	if n_loci is not None and n_loci < 1:
		raise ValueError("n_loci must be at least 1 or None.")

	if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, numpy.integer)):
		raise TypeError("n_jobs must be an integer.")

	if n_jobs != -1 and n_jobs < 1:
		raise ValueError("n_jobs must be at least 1, or -1 for one thread " +
			"per CPU.")

	signals = _load_signals(signals)
	in_signals = _load_signals(in_signals)

	if min_counts is not None or max_counts is not None:
		if signals is None:
			raise ValueError("min_counts and max_counts are measured on " +
				"signals, so signals must be provided.")

		if not -len(signals) <= target_idx < len(signals):
			raise ValueError("target_idx {} is out of range for {} signals."
				.format(target_idx, len(signals)))

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

	idxs, mids = _kept_loci(loci, chrom_lengths, exclusion_zones, left, right)
	loci_chroms = loci['chrom'].values[idxs].astype(str)

	out_start = out_width + max_jitter
	out_length = out_window + 2 * max_jitter

	# The counts are measured on groups of loci, so that the windows held at
	# once stay small, first on the target alone and without warnings. Each
	# locus is compared as a float32 scalar, as its sum is. Then every signal
	# is read at each locus examined before n_loci loci are kept, which is
	# where it was read one locus at a time, so that the warnings for
	# chromosomes a signal does not have are the same, and the values at the
	# kept loci are kept.
	signals_ = None
	if min_counts is not None or max_counts is not None:
		keep, kept_values = [], []
		size = max(1, _COUNT_VALUES // out_length)
		for c in range(0, len(idxs), size):
			starts = mids[c:c+size] - out_start
			counts = _extract_signals([signals[target_idx]],
				loci_chroms[c:c+size], starts, out_length, n_jobs,
				warn=False)[:, 0].sum(axis=1)

			kept, n_examined = [], len(counts)
			for k, count in enumerate(counts):
				if min_counts is not None and count < min_counts:
					continue

				if max_counts is not None and count > max_counts:
					continue

				kept.append(k)
				if n_loci is not None and len(keep) + len(kept) == n_loci:
					n_examined = k + 1
					break

			values = _extract_signals(signals, loci_chroms[c:c+n_examined],
				starts[:n_examined], out_length, n_jobs)
			kept_values.append(values[kept])
			keep.extend(c + k for k in kept)

			if n_loci is not None and len(keep) == n_loci:
				break

		keep = numpy.array(keep, dtype=numpy.int64)
		idxs, mids, loci_chroms = idxs[keep], mids[keep], loci_chroms[keep]

		# Copied group by group, so that the values are not held twice.
		signals_ = numpy.empty((len(keep), len(signals), out_length),
			dtype=numpy.float32)
		k = 0
		while kept_values:
			values = kept_values.pop(0)
			signals_[k:k+len(values)] = values
			k += len(values)

	if n_loci is not None:
		idxs, mids = idxs[:n_loci], mids[:n_loci]
		loci_chroms = loci_chroms[:n_loci]

	if len(idxs) == 0:
		if opened_fasta:
			sequences.close()

		raise ValueError("No loci remain after filtering. Loci are removed " +
			"when they are not on a chromosome in `chroms`, when their windows " +
			"run off the end of a chromosome, when they overlap an exclusion " +
			"region, or when their counts fall outside min_counts/max_counts.")

	# Extract a window of signal using the output size
	if signals is not None and signals_ is None:
		signals_ = _extract_signals(signals, loci_chroms, mids - out_start,
			out_length, n_jobs)

	# Extract a window of signal using the input size
	if in_signals is not None:
		in_signals_ = _extract_signals(in_signals, loci_chroms,
			mids - in_width - max_jitter, in_window + 2 * max_jitter, n_jobs)

	# Extract a window of sequence using the input size. The windows of a
	# fasta opened from a path are read together, from a memory map of the
	# file where they can be, and the windows of a pyfaidx.Fasta one at a
	# time; both are one-hot encoded straight into one C-contiguous array.
	threads = _cpu_count() if n_jobs == -1 else int(n_jobs)
	try:
		if opened_fasta and mids.dtype == numpy.int64:
			codes, names = pandas.factorize(loci_chroms)
			with tqdm(total=len(mids), disable=d, desc=desc) as progress:
				seqs = _read_fasta_windows(sequences, (codes, mids - in_width -
					max_jitter), in_window + 2 * max_jitter, alphabet, ignore,
					names=[str(name) for name in names], n_jobs=threads)
				progress.update(len(mids))
		else:
			seqs = []
			for chrom, mid in tqdm(zip(loci_chroms, mids.tolist()),
					total=len(mids), disable=d, desc=desc):
				start = mid - in_width - max_jitter
				end = mid + in_width + max_jitter + (in_window % 2)

				if isinstance(sequences, dict):
					seq = sequences[str(chrom)][:, start:end]
				else:
					# A Fasta opened with as_raw=True returns strings rather
					# than pyfaidx.Sequence objects.
					seq = sequences[str(chrom)][start:end]
					if not isinstance(seq, str):
						seq = seq.seq

				seqs.append(seq)

			# numpy.stack keeps the memory layout of its inputs, so a stack of
			# slices of Fortran-ordered arrays is not contiguous by default.
			if isinstance(sequences, dict):
				seqs = numpy.ascontiguousarray(numpy.stack(seqs))
			else:
				seqs = _one_hot_encode_rows(seqs, alphabet=alphabet,
					ignore=ignore, n_jobs=threads)
	finally:
		if opened_fasta:
			sequences.close()

	# Figure out how to format the outputs depending on the provided parameters.
	seqs = torch.from_numpy(seqs)
	y_return = [seqs]

	if signals is not None:
		y_return.append(torch.from_numpy(signals_))

	if in_signals is not None:
		y_return.append(torch.from_numpy(in_signals_))
		
	if return_mask:
		# Loci after the n_loci cap was reached are not returned, so they are
		# False.
		kept_mask = numpy.zeros(len(loci), dtype=bool)
		kept_mask[idxs] = True
		y_return.append(torch.from_numpy(kept_mask))

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
