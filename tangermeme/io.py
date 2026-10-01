# io.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>
# Code adapted from Alex Tseng, Avanti Shrikumar, and Ziga Avsec

from __future__ import annotations

import os
import sys
import mmap
import ctypes
import zlib
import struct
import operator
import warnings

from concurrent.futures import ThreadPoolExecutor

import numba
import numpy
import torch
import pandas

import pyfaidx
import pybigtools

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
	signals: list of pybigtools' BBIRead objects or None
		A list of BBIRead objects, as returned by pybigtools.open(). The row
		of an entry that is None is left as it is.

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
		if signal is None:
			continue

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
	max_gap=4096, max_span=65536, names=None, rows=None):
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

	chroms: list of str, or numpy.ndarray of int when `names` is given
		The chromosome of each window, or its index into `names`.

	starts: numpy.ndarray, shape=(n,), dtype=int64
		The start of each window, inclusive and base-0.

	width: int
		The length of every window.

	out: numpy.ndarray, shape=(>=n, len(signals), width)
		The float32 array that window k is written into: its row k, or its row
		rows[k] when `rows` is given. NaN and infinities are left for the
		caller to replace.

	kind: int, optional
		A label copied into each failure. Default is 0.

	max_gap: int, optional
		The largest number of bases between two windows read together.
		Default is 4096.

	max_span: int, optional
		The largest number of bases read in one call. Default is 65536.

	names: list of str or None, optional
		The chromosome names that `chroms` indexes into, or None when `chroms`
		holds the names. Default is None.

	rows: numpy.ndarray, shape=(n,), dtype=int64, or None, optional
		The row of `out` that each window is written to, or None when window k
		is written to row k. Default is None.


	Returns
	-------
	failures: list of tuples
		(k, kind, i, chrom, start, end) for each window whose read from
		signal i raised, where k is the row of `out` it was written to.
	"""

	n = len(starts)
	if n == 0:
		return []

	if names is None:
		codes, names = pandas.factorize(numpy.array(chroms, dtype=object))
	else:
		codes = chroms

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
	rows = (order if rows is None else numpy.asarray(rows)[order]).tolist()

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


# _nan_to_num_rows checks its blocks on up to n_jobs threads when there are
# at least this many of them. Below that, starting the threads costs about as
# much as checking the blocks.
_NAN_TO_NUM_MIN_BLOCKS = 16


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


###
# A bigWig reader for extract_loci
###

# A call reads its bigWig paths with _BigWigFile when it may keep at least
# this many loci. A smaller call reads them with pybigtools, because parsing
# the data index costs about as much as a few hundred pybigtools reads.
_BIGWIG_MIN_WINDOWS = 1024

# The number of data blocks each task decompresses at once. A block holds at
# most uncompressBufSize bytes, 32 KB in ENCODE's bigWigs, so n_jobs tasks
# hold at most n_jobs * 256 * 32 KB of decompressed data.
_BIGWIG_BATCH_BLOCKS = 256

# A task inflates its blocks into a buffer of uncompressBufSize bytes per
# block, or of this many when uncompressBufSize is larger. A block that does
# not fit in what is left of the buffer is inflated by zlib.decompress.
_BIGWIG_MAX_BLOCK_BYTES = 2 ** 20

# zlib's uncompress() through ctypes, loaded on the first read.
_ZLIB_UNCOMPRESS = []


def _zlib_uncompress():
	"""zlib's uncompress() as a ctypes function, or None if it cannot be loaded.

	_inflate_bigwig_blocks calls it without the GIL, which zlib.decompress
	holds between blocks. The zlib module links the same library on Linux,
	where it is already loaded under this name. Without it, every block is
	inflated by zlib.decompress.
	"""

	if len(_ZLIB_UNCOMPRESS) == 0:
		_ZLIB_UNCOMPRESS.append(_load_zlib_uncompress())

	return _ZLIB_UNCOMPRESS[0]


def _load_zlib_uncompress():
	"""(uncompress, the dtype of a C unsigned long), or None."""

	try:
		import ctypes.util
	except ImportError:
		return None

	for name in ('libz.so.1', 'libz.dylib', 'zlib1.dll', 'z'):
		try:
			if name == 'z':
				name = ctypes.util.find_library('z')
				if name is None:
					continue

			function = ctypes.CDLL(name).uncompress
		except (OSError, AttributeError):
			continue

		function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
			ctypes.c_ulong]
		function.restype = ctypes.c_int
		return function, numpy.dtype(ctypes.c_ulong)

	return None


@numba.njit(nogil=True, cache=True)
def _inflate_bigwig_blocks(uncompress, data, starts, sizes, buffer, blocks,
	length):
	"""Inflate data blocks one after another into `buffer`, without the GIL.

	Block k is the zlib stream data[starts[k]:starts[k] + sizes[k]], and is
	inflated only when blocks[k, 5] is 1. Its words are then
	buffer[4 * blocks[k, 0]:4 * blocks[k, 1]]. A block that inflates to no
	bytes or to bytes that are not whole 32-bit words gets blocks[k, 5] = 0,
	as it would from zlib.decompress. A block that uncompress() cannot
	inflate into the rest of `buffer` gets 2, and is left to zlib.decompress.
	`length` is a one-element array of C unsigned longs. Returns the number
	of bytes of `buffer` that were used.
	"""

	position = 0
	for k in range(starts.shape[0]):
		if blocks[k, 5] != 1:
			continue

		length[0] = buffer.shape[0] - position
		source = data[starts[k]:starts[k] + sizes[k]]
		status = uncompress(buffer[position:].ctypes, length.ctypes,
			source.ctypes, sizes[k])
		n = numpy.int64(length[0])

		if status != 0:
			blocks[k, 5] = 2
		elif n == 0 or n % 4 != 0:
			blocks[k, 5] = 0
		else:
			blocks[k, 0] = position // 4
			blocks[k, 1] = (position + n) // 4
			position += n

	return position


def _pread_into(fd, buffer, offset):
	"""Read into the uint8 array `buffer` from `offset`; return the bytes read."""

	if hasattr(os, 'preadv'):
		return os.preadv(fd, [buffer], offset)

	data = os.pread(fd, len(buffer), offset)
	buffer[:len(data)] = numpy.frombuffer(data, dtype=numpy.uint8)
	return len(data)


@numba.njit(nogil=True, cache=True)
def _check_bigwig_block(words, begin, end, chrom, base_start, base_end):
	"""Whether one decompressed data block can be read by _read_bigwig_windows.

	The block is the 32-bit words `words[begin:end]`. Each section is a
	24-byte header (chromosome id, start, end, step, span, then the type and
	the item count) followed by its items. The block must hold whole
	bedGraph, varStep or fixedStep sections on the chromosome `chrom` of its
	index entry, whose items are sorted, do not overlap one another, and lie
	inside the entry's range [base_start, base_end). pybigtools sums the
	values of overlapping items, so a block that has them is left to it.
	"""

	offset = begin
	previous_end = base_start
	while offset < end:
		if offset + 6 > end or words[offset] != chrom:
			return False

		section_start = numpy.int64(words[offset + 1])
		step = numpy.int64(words[offset + 3])
		span = numpy.int64(words[offset + 4])
		kind = words[offset + 5] & 0xFF
		count = numpy.int64(words[offset + 5] >> 16)

		if kind == 1:
			size = 3
		elif kind == 2:
			size = 2
		elif kind == 3:
			size = 1
		else:
			return False

		item = offset + 6
		next_offset = item + count * size
		if next_offset > end:
			return False

		for i in range(count):
			if kind == 1:
				item_start = numpy.int64(words[item + 3 * i])
				item_end = numpy.int64(words[item + 3 * i + 1])
			elif kind == 2:
				item_start = numpy.int64(words[item + 2 * i])
				item_end = item_start + span
			else:
				item_start = section_start + i * step
				item_end = item_start + span

			if item_start < previous_end or item_end < item_start:
				return False

			previous_end = item_end

		if previous_end > base_end:
			return False

		offset = next_offset

	return True


@numba.njit(nogil=True, cache=True)
def _read_bigwig_windows(words, blocks, windows, out, signal, failed):
	"""Write the per-base values of bigWig windows into rows of `out`.

	This is the compiled half of _BigWigFile.read and runs without the GIL.
	`words` holds decompressed data blocks, one after another, as 32-bit
	words. Row b of `blocks` describes block b: its first and last word, the
	chromosome, start and end of its index entry, and whether it was
	decompressed. Row j of `windows` describes window j: the blocks [lo, hi)
	that overlap it, its start, the length of its chromosome and the row of
	`out` it is written to. The width of every window is out.shape[2].

	Each base of a window is given the value of the item that covers it, 0
	when no item does and NaN past the end of the chromosome, which is what
	pybigtools' values() gives with its default `missing` and `oob`. Items
	with a NaN value are skipped, as pybigtools skips them. A window with a
	block that fails _check_bigwig_block is not written and is marked in
	`failed`.
	"""

	values = words.view(numpy.float32)
	width = out.shape[2]
	nan = numpy.float32(numpy.nan)

	good = numpy.zeros(blocks.shape[0], dtype=numpy.bool_)
	for b in range(blocks.shape[0]):
		if blocks[b, 5] != 0:
			good[b] = _check_bigwig_block(words, blocks[b, 0], blocks[b, 1],
				blocks[b, 2], blocks[b, 3], blocks[b, 4])

	for j in range(windows.shape[0]):
		lo, hi = windows[j, 0], windows[j, 1]
		start, row = windows[j, 2], windows[j, 4]

		usable = True
		for b in range(lo, hi):
			if not good[b]:
				usable = False

		if not usable:
			failed[j] = True
			continue

		end = min(start + width, max(start, windows[j, 3]))
		for p in range(end - start):
			out[row, signal, p] = 0
		for p in range(end - start, width):
			out[row, signal, p] = nan

		for b in range(lo, hi):
			offset = blocks[b, 0]
			while offset < blocks[b, 1]:
				section_start = numpy.int64(words[offset + 1])
				step = numpy.int64(words[offset + 3])
				span = numpy.int64(words[offset + 4])
				kind = words[offset + 5] & 0xFF
				count = numpy.int64(words[offset + 5] >> 16)
				item = offset + 6

				if kind == 1:
					size = 3
				elif kind == 2:
					size = 2
				else:
					size = 1

				# The first item that ends after the window starts. Items are
				# sorted and do not overlap, so their ends are sorted too.
				if kind == 3:
					i = 0
					if step > 0 and start > section_start + span:
						i = (start - section_start - span) // step + 1
				else:
					i, k = 0, count
					while i < k:
						m = (i + k) // 2
						if kind == 1:
							item_end = numpy.int64(words[item + 3 * m + 1])
						else:
							item_end = numpy.int64(words[item + 2 * m]) + span

						if item_end <= start:
							i = m + 1
						else:
							k = m

				while i < count:
					if kind == 1:
						item_start = numpy.int64(words[item + 3 * i])
						item_end = numpy.int64(words[item + 3 * i + 1])
						value = values[item + 3 * i + 2]
					elif kind == 2:
						item_start = numpy.int64(words[item + 2 * i])
						item_end = item_start + span
						value = values[item + 2 * i + 1]
					else:
						item_start = section_start + i * step
						item_end = item_start + span
						value = values[item + i]

					if item_start >= end:
						break

					if value == value:
						for p in range(max(item_start, start) - start,
								min(item_end, end) - start):
							out[row, signal, p] = value

					i += 1

				offset = item + count * size


class _BigWigFile():
	"""A bigWig opened from a path, read with numpy, zlib and numba.

	extract_loci reads the windows of the loci it keeps from bigWig paths
	with this class, on `n_jobs` threads, and gives each base the value that
	`pybigtools.BBIRead.values(chrom, start, end)` gives it, cast to float32.
	pybigtools holds the GIL while it reads, whereas zlib.decompress and the
	compiled decoder here release it.

	The header and chromosome tree are read when the file is opened, and the
	data index (the R-tree) on the first read. A read sorts its windows by
	position, finds the data blocks that overlap each one in the index, and
	works through the blocks in batches of _BIGWIG_BATCH_BLOCKS. Each batch is
	read with one os.pread per run of adjacent blocks, decompressed, and
	decoded straight into the output rows by _read_bigwig_windows, so a block
	that several sorted, repeated or overlapping windows share is decompressed
	once, and at most n_jobs batches are held at a time. The index costs 56
	bytes per data block, about 2.3 MB for a 240 MB bigWig.

	`pybigtools_file`, the file opened with pybigtools, reads what this class
	does not:

		- a file that is not a little-endian bigWig, such as a bigBed, or
		  whose chromosome tree does not match pybigtools', or whose index
		  entries are not sorted, span two chromosomes, or overlap;
		- a window that starts before 0 or ends past 2**32 - 1, or whose
		  chromosome is not in the file;
		- a window with a data block that cannot be decompressed, holds a
		  section other than bedGraph, varStep or fixedStep, or holds items
		  that are unsorted, overlap, or lie outside the block's index entry.

	Blocks stored without compression, when uncompressBufSize is 0, are
	read as they are.
	"""

	def __init__(self, path, pybigtools_file):
		self.path = path
		self.pybigtools_file = pybigtools_file
		self._index, self._index_read = None, False

		with open(path, 'rb') as handle:
			header = handle.read(64)
			(magic, _, _, chrom_tree, _, data_index, _, _, _, _,
				self.buffer_size, _) = struct.unpack('<IHHQQQHHQQIQ', header)

			if magic != 0x888FFC26 or sys.byteorder != 'little':
				raise ValueError("not a little-endian bigWig")

			self._data_index = data_index
			self.chroms = self._read_chrom_tree(handle, chrom_tree)

		expected = {str(name): size for name, size in
			pybigtools_file.chroms().items()}
		if {name: size for name, (_, size) in self.chroms.items()} != expected:
			raise ValueError("the chromosome tree differs from pybigtools'")

	@classmethod
	def open(cls, path, pybigtools_file):
		"""A _BigWigFile for `path`, or None when it cannot read the file."""

		try:
			return cls(path, pybigtools_file)
		except (OSError, ValueError, struct.error, UnicodeDecodeError):
			return None

	@staticmethod
	def _read_chrom_tree(handle, offset):
		"""The chromosome B+ tree, as {name: (chromosome id, length)}."""

		handle.seek(offset)
		magic, _, key_size, value_size, _, _ = struct.unpack('<IIIIQQ',
			handle.read(32))
		if magic != 0x78CA8C91 or value_size != 8:
			raise ValueError("unexpected chromosome tree")

		chroms, nodes, seen = {}, [offset + 32], set()
		while nodes:
			node = nodes.pop()
			if node in seen:
				raise ValueError("the chromosome tree has a cycle")

			seen.add(node)
			handle.seek(node)
			is_leaf, _, count = struct.unpack('<BBH', handle.read(4))
			data = handle.read(count * (key_size + 8))
			for k in range(count):
				item = data[k * (key_size + 8): (k + 1) * (key_size + 8)]
				if is_leaf:
					name = item[:key_size].rstrip(b'\0').decode()
					chroms[name] = struct.unpack('<II', item[key_size:])
				else:
					nodes.append(struct.unpack('<Q', item[key_size:])[0])

		return chroms

	def _read_index(self):
		"""The data blocks in file order, from the R-tree, or None.

		Each entry of the R-tree's leaves gives a block's range, from
		(chromosome, start) to (chromosome, end), and its offset and size in
		the file. None is returned when the entries cannot be read or are not
		sorted, non-overlapping and each on one chromosome, so that the
		blocks overlapping a window can be found by binary search.
		"""

		leaf_type = numpy.dtype([('start_chrom', '<u4'), ('start', '<u4'),
			('end_chrom', '<u4'), ('end', '<u4'), ('offset', '<u8'),
			('size', '<u8')])

		try:
			with open(self.path, 'rb') as handle:
				fd = handle.fileno()
				magic = struct.unpack('<I', os.pread(fd, 4,
					self._data_index))[0]
				if magic != 0x2468ACE0:
					return None

				leaves, nodes, seen = [], [self._data_index + 48], set()
				while nodes:
					offset = nodes.pop()
					if offset in seen:
						return None

					seen.add(offset)
					is_leaf, _, count = struct.unpack('<BBH', os.pread(fd, 4,
						offset))
					size = 32 if is_leaf else 24
					data = os.pread(fd, count * size, offset + 4)
					if len(data) != count * size:
						return None

					if is_leaf:
						leaves.append(numpy.frombuffer(data, dtype=leaf_type))
					else:
						children = numpy.frombuffer(data, dtype='<u8').reshape(
							count, 3)[:, 2]
						nodes.extend(children[::-1].tolist())
		except (OSError, struct.error, OverflowError):
			return None

		leaves = numpy.concatenate(leaves) if leaves else numpy.empty(0,
			dtype=leaf_type)
		chroms = leaves['start_chrom'].astype(numpy.int64)
		starts = (chroms << 32) | leaves['start'].astype(numpy.int64)
		ends = (leaves['end_chrom'].astype(numpy.int64) << 32) | \
			leaves['end'].astype(numpy.int64)

		if (leaves['end_chrom'] != leaves['start_chrom']).any() or \
				(ends < starts).any() or (starts[1:] < ends[:-1]).any():
			return None

		return {'starts': starts, 'ends': ends, 'chroms': chroms,
			'bases': numpy.stack([leaves['start'], leaves['end']], axis=1).astype(
				numpy.int64),
			'offsets': leaves['offset'].astype(numpy.int64),
			'sizes': leaves['size'].astype(numpy.int64)}

	def read(self, rows, chroms, starts, out, signal, n_jobs=1, names=None):
		"""Write windows into out[rows[j], signal] and return the ones it did not.

		Window j covers [starts[j], starts[j] + out.shape[2]) on chroms[j]. NaN
		is written past the end of a chromosome, as pybigtools writes it. The
		positions j of the windows that must be read with pybigtools are
		returned, in increasing order.


		Parameters
		----------
		rows: numpy.ndarray, shape=(n,), dtype=int64
			The row of `out` that each window is written to. No two windows
			may share a row.

		chroms: list of str, length n, or numpy.ndarray of int when `names`
			is given
			The chromosome of each window, or its index into `names`.

		starts: numpy.ndarray, shape=(n,), dtype=int64
			The start of each window, inclusive and base-0.

		out: numpy.ndarray, shape=(m, n_signals, width), dtype=float32
			A C-contiguous array to write into.

		signal: int
			The index into the second axis of `out` that is written.

		n_jobs: int, optional
			The number of threads to decompress and decode blocks on.
			Default is 1.

		names: list of str or None, optional
			The chromosome names that `chroms` indexes into, or None when
			`chroms` holds the names. Default is None.


		Returns
		-------
		fallback: numpy.ndarray, dtype=int64
			The positions of the windows that were not written.
		"""

		n, width = len(rows), out.shape[2]
		if not self._index_read:
			self._index, self._index_read = self._read_index(), True

		index = self._index
		if index is None or n == 0:
			return numpy.arange(n, dtype=numpy.int64)

		if names is None:
			codes, names = pandas.factorize(numpy.asarray(chroms, dtype=object))
		else:
			codes = numpy.asarray(chroms)

		info = [self.chroms.get(str(name), (-1, 0)) for name in names]
		ids = numpy.array([i for i, _ in info], dtype=numpy.int64)[codes]
		sizes = numpy.array([s for _, s in info], dtype=numpy.int64)[codes]

		usable = (ids >= 0) & (starts >= 0) & (starts + width < 2**32)
		order = numpy.flatnonzero(usable)
		order = order[numpy.lexsort((starts[order], ids[order]))]

		# The blocks [lo, hi) overlap a window. The index entries are sorted
		# and do not overlap, so both their starts and their ends are sorted.
		keys = (ids[order] << 32) | starts[order]
		lo = numpy.searchsorted(index['ends'], keys, side='right')
		hi = numpy.searchsorted(index['starts'], keys + width, side='left')
		hi = numpy.maximum(lo, hi)

		# The blocks any window needs, and each window's blocks as positions
		# in that list. Windows are split into batches by their first block.
		n_blocks = len(index['starts'])
		cover = numpy.cumsum(numpy.bincount(lo, minlength=n_blocks + 1) -
			numpy.bincount(hi, minlength=n_blocks + 1))
		needed = numpy.flatnonzero(cover[:n_blocks] > 0)
		lo, hi = numpy.searchsorted(needed, lo), numpy.searchsorted(needed, hi)

		windows = numpy.stack([lo, hi, starts[order], sizes[order], rows[order]],
			axis=1)
		failed = numpy.zeros(len(order), dtype=numpy.bool_)

		batch = lo // _BIGWIG_BATCH_BLOCKS
		bounds = numpy.append(numpy.flatnonzero(numpy.diff(batch,
			prepend=-1)), len(order)).tolist()

		def read_batch(k):
			w0, w1 = bounds[k], bounds[k + 1]
			b0, b1 = int(windows[w0, 0]), int(windows[w0:w1, 1].max())
			words, blocks = self._read_blocks(fd, needed[b0:b1], index,
				uncompress)
			local = windows[w0:w1].copy()
			local[:, :2] -= b0
			_read_bigwig_windows(words, blocks, local, out, signal,
				failed[w0:w1])

		# The first batch is read on this thread when the decoder or the
		# inflater has not been compiled yet, so that each is compiled once and
		# before any thread starts. pread takes no file position, so the
		# threads share one fd.
		uncompress = _zlib_uncompress() if self.buffer_size > 0 else None
		n_batches = len(bounds) - 1
		first = int(n_batches > 0 and not (_read_bigwig_windows.signatures and
			(uncompress is None or _inflate_bigwig_blocks.signatures)))
		fd = os.open(self.path, os.O_RDONLY)
		try:
			if first:
				read_batch(0)

			if n_jobs == 1 or n_batches - first <= 1:
				for k in range(first, n_batches):
					read_batch(k)
			else:
				with ThreadPoolExecutor(min(n_jobs, n_batches - first)) as pool:
					list(pool.map(read_batch, range(first, n_batches)))
		finally:
			os.close(fd)

		fallback = numpy.concatenate([numpy.flatnonzero(~usable), order[failed]])
		return numpy.sort(fallback)

	def _read_blocks(self, fd, leaves, index, uncompress=None):
		"""Read and decompress data blocks into one array of 32-bit words.

		Returns the words and a (len(leaves), 6) array of each block's first
		and last word, its index entry's chromosome, start and end, and 1 if
		it was read or 0 if it could not be.

		The compressed blocks are read with one os.pread per run of adjacent
		blocks, into one array. `uncompress`, the result of _zlib_uncompress(),
		inflates them without the GIL. The blocks it leaves, and every block
		when it is None or the file is not compressed, are inflated with
		zlib.decompress, with the same result.
		"""

		offsets, sizes = index['offsets'][leaves], index['sizes'][leaves]
		breaks = numpy.flatnonzero(offsets[1:] != offsets[:-1] + sizes[:-1]) + 1
		breaks = [0] + breaks.tolist() + [len(leaves)] if len(leaves) > 0 else []

		# Blocks in a run are adjacent in the file, so each run is read to
		# the positions its blocks have when they are packed end to end. A
		# block that a short read does not reach in full is not read.
		ends = numpy.cumsum(sizes)
		starts = ends - sizes
		data = numpy.empty(int(sizes.sum()), dtype=numpy.uint8)
		read = numpy.ones(len(leaves), dtype=numpy.int64)
		for r0, r1 in zip(breaks[:-1], breaks[1:]):
			position, end = int(starts[r0]), int(ends[r1 - 1])
			got = _pread_into(fd, data[position:end], int(offsets[r0]))
			read[r0:r1][ends[r0:r1] > position + got] = 0

		zeros = numpy.zeros(len(leaves), dtype=numpy.int64)
		blocks = numpy.stack([zeros, zeros, index['chroms'][leaves],
			index['bases'][leaves, 0], index['bases'][leaves, 1], read], axis=1)

		used, buffer = 0, numpy.empty(0, dtype=numpy.uint8)
		if uncompress is not None and self.buffer_size > 0:
			function, ulong = uncompress
			buffer = numpy.empty(len(leaves) * min(self.buffer_size,
				_BIGWIG_MAX_BLOCK_BYTES), dtype=numpy.uint8)
			used = _inflate_bigwig_blocks(function, data, starts, sizes, buffer,
				blocks, numpy.zeros(1, dtype=ulong))
		else:
			blocks[:, 5] *= 2

		parts, position = [], used
		for k in numpy.flatnonzero(blocks[:, 5] == 2).tolist():
			raw = data[starts[k]:ends[k]]
			try:
				block = zlib.decompress(raw) if self.buffer_size > 0 else \
					raw.tobytes()
			except zlib.error:
				block = b''

			if len(block) % 4 != 0 or len(block) == 0:
				blocks[k, 5] = 0
				continue

			blocks[k, 0], blocks[k, 1], blocks[k, 5] = position // 4, (position +
				len(block)) // 4, 1
			position += len(block)
			parts.append(block)

		words = buffer[:used]
		if len(parts) > 0:
			words = numpy.concatenate([words, numpy.frombuffer(b''.join(parts),
				dtype=numpy.uint8)])

		return words.view(numpy.uint32), blocks


def _open_bigwig_files(paths, signals, values, n_max):
	"""Open with _BigWigFile each signal that was given as a bigWig path.

	Returns a list with a _BigWigFile for each such signal that it can read
	and None for every other signal, or None when no signal is read with
	_BigWigFile: when `values` was not preallocated (a dictionary is among
	the signals, or the window is negative), or when fewer than
	_BIGWIG_MIN_WINDOWS loci can be kept.
	"""

	if values is None or n_max < _BIGWIG_MIN_WINDOWS:
		return None

	readers = []
	for path, signal in zip(paths, signals):
		reader = None
		if isinstance(path, (str, os.PathLike)) and isinstance(signal,
				pybigtools.BBIRead):
			reader = _BigWigFile.open(os.fspath(path), signal)

		readers.append(reader)

	return readers if any(reader is not None for reader in readers) else None


def _signals_read_in_loop(signals, readers):
	"""The signals with None in place of each one that a _BigWigFile reads."""

	if readers is None:
		return signals

	return [signal if reader is None else None for signal, reader in
		zip(signals, readers)]


def _signals_read_after_loop(signals, readers):
	"""The signals with None in place of each one that no _BigWigFile reads.

	None is returned when no signal has a _BigWigFile.
	"""

	if readers is None:
		return None

	return [None if reader is None else signal for signal, reader in
		zip(signals, readers)]


def _read_bigwig_files(signals, readers, codes, starts, width, out, names,
	rows=None, kind=0, n_jobs=1):
	"""Read the windows of bigWig signals into `out` after the loop over loci.

	Window k of signal i is written into out[k, i], or into out[rows[k], i]
	when `rows` is given. A signal with a _BigWigFile is read by it on n_jobs
	threads, and the windows it leaves are read with pybigtools by
	_read_signal_windows, sorted and grouped, which is also how a signal
	without a _BigWigFile is read. A signal that is None is not read. A window
	whose read raises is zero-filled and returned as a failure, so that the
	caller can warn in the order of the loci.


	Parameters
	----------
	signals: list of pybigtools' BBIRead objects or None
		The signals, with None for each one that is not read.

	readers: list of _BigWigFile or None, or None
		The _BigWigFile of each signal, or None for a signal without one.
		None when no signal has one.

	codes: numpy.ndarray, shape=(n,), dtype=int64
		The chromosome of each window, as an index into `names`.

	starts: numpy.ndarray, shape=(n,), dtype=int64
		The start of each window, inclusive and base-0.

	width: int
		The length of every window, out.shape[2].

	out: numpy.ndarray, shape=(m, len(signals), width), dtype=float32
		The preallocated values. NaN and infinities are left for the caller
		to replace.

	names: list of str
		The chromosome names that `codes` indexes into.

	rows: numpy.ndarray, shape=(n,), dtype=int64, or None, optional
		The row of `out` that each window is written to, or None when window k
		is written to row k. Default is None.

	kind: int, optional
		A label copied into each failure. Default is 0.

	n_jobs: int, optional
		The number of threads each _BigWigFile reads on. Default is 1.


	Returns
	-------
	failures: list of tuples
		(row, kind, i, chrom, start, end) for each window whose read from
		signal i raised.
	"""

	if readers is None and all(signal is not None for signal in signals):
		return _read_signal_windows(signals, codes, starts, width, out,
			kind=kind, names=names, rows=rows)

	if readers is None:
		readers = [None] * len(signals)

	if rows is None:
		rows = numpy.arange(len(starts), dtype=numpy.int64)

	failures = []
	for i, (signal, reader) in enumerate(zip(signals, readers)):
		if signal is None:
			continue

		codes_, starts_, rows_ = codes, starts, rows
		if reader is not None:
			left = reader.read(rows, codes, starts, out, i, n_jobs, names=names)
			if len(left) == 0:
				continue

			codes_, starts_, rows_ = codes[left], starts[left], rows[left]

		for row, _, _, chrom, start, end in _read_signal_windows([signal],
				codes_, starts_, width, out[:, i:i+1], kind=kind, names=names,
				rows=rows_):
			failures.append((row, kind, i, chrom, start, end))

	return failures


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
		A list whose elements are each a path to a bigwig file, which is
		read by tangermeme's own reader or by pybigtools, as `n_jobs`
		describes, a bigwig file already opened with
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

	n_jobs: int, optional
		The largest number of threads to use. The sequences of a fasta file
		are one-hot encoded in blocks of loci, each block by one of numba's
		threads, on at most `n_jobs` threads, which is also capped by numba's
		NUMBA_NUM_THREADS, the number of CPUs unless it is set. Sequences
		given as a dictionary are not encoded and are unaffected. The bigWigs
		in `signals` and `in_signals` that are given as paths are read on at
		most `n_jobs` threads. When at least 1,024 loci can be kept, those
		bigWigs are read once the kept loci are known, by a reader built on
		numpy, zlib and numba that gives the same values as pybigtools and
		releases the GIL. A bigWig opened with pybigtools, the signal that a
		count filter is measured on, and the windows the reader cannot read,
		such as those on a chromosome the file lacks, are read with
		pybigtools. The results are the same as with one thread. Must be at
		least 1, and 1 encodes every locus and reads every bigWig in the
		calling thread. Default is 8.


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
		less than 1, or if `n_jobs` is not an integer of at least 1.
	"""

	if n_loci is not None and n_loci < 1:
		raise ValueError("n_loci must be at least 1 or None.")

	if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int,
			numpy.integer)) or n_jobs < 1:
		raise ValueError("n_jobs must be an integer of at least 1.")

	signal_paths, in_signal_paths = signals, in_signals
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

	# Each row's chromosome as an index into the unique names, with missing
	# values kept as their own name so that they are reported below.
	codes, names = pandas.factorize(loci['chrom'], use_na_sentinel=False)

	missing = sorted(set(names) - set(chrom_lengths))
	if len(missing) > 0:
		if opened_fasta:
			sequences.close()

		raise ValueError("Loci are on chromosomes that are not in the " +
			"sequences: {}. Pass `chroms` to select the chromosomes to use."
			.format(", ".join(missing)))

	names = [str(name) for name in names]

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

	# Every locus's window is checked at once. int64 gives the values that
	# Python ints do while the coordinates and windows are far inside its range.
	# Otherwise the coordinates are Python ints, and one that is not an integer
	# raises a TypeError, as slicing a sequence with it would.
	starts, ends = loci['start'].to_numpy(), loci['end'].to_numpy()
	if (starts.dtype.kind in 'iu' and ends.dtype.kind in 'iu' and
			max(abs(left), abs(right)) < 2 ** 40 and
			all(-2 ** 40 < int(x.min(initial=0)) and
				int(x.max(initial=0)) < 2 ** 40 for x in (starts, ends))):
		starts, ends = starts.astype(numpy.int64), ends.astype(numpy.int64)
	else:
		starts, ends = [numpy.array([operator.index(x) for x in loci[col]],
			dtype=object) for col in ('start', 'end')]

	mids = starts + (ends - starts) // 2
	lengths = numpy.array([chrom_lengths[name] for name in names],
		dtype=numpy.int64)[codes]

	# Does it fall off the end of a chromosome?
	keep = ~((mids - left < 0) | (mids + right > lengths))

	# Does it overlap an excluded 100 bp chunk? The chunks of every chromosome
	# are placed end to end, so that one sorted array holds every excluded
	# chunk, and the chunks [s, e) of a window hold one when fewer excluded
	# chunks come before s than before e.
	if exclusion_zones is not None:
		sizes = [len(exclusion_zones[name]) for name in names]
		offsets = numpy.cumsum([0] + sizes, dtype=numpy.int64)
		excluded = numpy.concatenate([numpy.zeros(0, dtype=numpy.int64)] + [
			numpy.flatnonzero(exclusion_zones[name]) + offsets[k]
			for k, name in enumerate(names)])

		idxs = numpy.flatnonzero(keep)
		offsets = offsets[codes[idxs]]
		s = (mids[idxs] - left).astype(numpy.int64) // 100 + offsets
		e = (mids[idxs] + right - 1).astype(numpy.int64) // 100 + 1 + offsets
		keep[idxs[numpy.searchsorted(excluded, s) <
			numpy.searchsorted(excluded, e)]] = False
		del sizes, offsets, excluded, s, e

	# Only the loci that remain are read, in order, until n_loci are kept.
	idxs = numpy.flatnonzero(keep)
	kept_mask = numpy.zeros(len(loci), dtype=bool)
	del starts, ends, lengths, keep

	# Without a count filter every locus that reaches the signals is kept, so
	# when every signal is a bigWig the windows of the kept loci are read
	# after the loop, sorted by position and grouped (_read_signal_windows).
	defer = (not count_filter
		and (out_values is not None or in_values is not None)
		and (signals is None or out_values is not None)
		and (in_signals is None or in_values is not None)
		and pandas.api.types.is_integer_dtype(loci['start'])
		and pandas.api.types.is_integer_dtype(loci['end']))

	# When the sequences are also a fasta opened from a path, whose windows
	# are read after the loop too, the loop would only record the loci it
	# keeps. Those are the first n_loci loci that pass the checks above, so
	# they are found without it. This holds with no signals at all as well.
	vectorized = (not count_filter and opened_fasta
		and (signals is None or out_values is not None)
		and (in_signals is None or in_values is not None)
		and mids.dtype == numpy.int64)

	# The bigWigs given as paths are read by _BigWigFile, on n_jobs threads,
	# when at least _BIGWIG_MIN_WINDOWS loci can be kept. The target of a
	# count filter is read in the loop with pybigtools, since it decides which
	# loci are kept.
	integer_loci = (pandas.api.types.is_integer_dtype(loci['start']) and
		pandas.api.types.is_integer_dtype(loci['end']))
	out_readers = _open_bigwig_files(signal_paths, signals, out_values,
		n_max if integer_loci else 0)
	in_readers = _open_bigwig_files(in_signal_paths, in_signals, in_values,
		n_max if integer_loci else 0)
	if count_filter and out_readers is not None:
		out_readers[target_idx] = None

	readers = [reader for reader in (out_readers or []) + (in_readers or [])
		if reader is not None]

	# When the loop reads the signals, those with a _BigWigFile are skipped
	# for a locus on a chromosome that every _BigWigFile's file has and are
	# read after the loop, for the kept loci only. A locus on another
	# chromosome is read in the loop, so that its warnings keep their order.
	loop_readers = len(readers) > 0 and not defer and not vectorized
	if loop_readers:
		out_loop = _signals_read_in_loop(signals, out_readers)
		in_loop = _signals_read_in_loop(in_signals, in_readers)
		readable = numpy.array([all(name in reader.chroms for reader in
			readers) for name in names], dtype=bool)
	else:
		readable = numpy.zeros(len(names), dtype=bool)

	if vectorized:
		n_remaining = len(idxs)
		idxs = idxs[:n_loci]
		kept_mask[idxs] = True

		# The progress bar counts the loci that pass the checks, as the
		# loop's does, and is filled in one step to the number kept.
		with tqdm(total=n_remaining, disable=d, desc=desc) as progress:
			progress.update(len(idxs))
	else:
		loci_iter = zip(idxs.tolist(), numpy.array(names, dtype=object)[
			codes[idxs]].tolist(), mids[idxs].tolist(), readable[
			codes[idxs]].tolist())

		for idx, chrom, mid, deferred in tqdm(loci_iter, total=len(idxs),
			disable=d, desc=desc):
			# Extract a window of signal using the output size
			start = mid - out_width - max_jitter
			end = mid + out_width + max_jitter + (out_window % 2)

			if signals is not None:
				if out_values is None:
					signal = _extract_locus_signal(signals, str(chrom), start,
						end)
				elif not defer:
					signal = out_values[len(seqs)]
					_write_locus_signal(out_loop if deferred else signals,
						str(chrom), start, end, signal, out_scratch)

					# The counts are summed after NaN and infinities are
					# replaced.
					if count_filter:
						numpy.nan_to_num(signal[target_idx], copy=False)

				if (min_counts is not None and
						signal[target_idx].sum() < min_counts):
					continue

				if (max_counts is not None and
						signal[target_idx].sum() > max_counts):
					continue

				if out_values is None:
					signals_.append(signal)

			# Extract a window of signal using the input size
			start = mid - in_width - max_jitter
			end = mid + in_width + max_jitter + (in_window % 2)

			if in_signals is not None:
				if in_values is None:
					in_signal = _extract_locus_signal(in_signals, str(chrom),
						start, end)
					in_signals_.append(in_signal)
				elif not defer:
					_write_locus_signal(in_loop if deferred else in_signals,
						str(chrom), start, end, in_values[len(seqs)], in_scratch)

			# Extract a window of sequence using the input size. The windows
			# of a fasta opened from a path are read together after the loop.
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

			kept_mask[idx] = True
			seqs.append(seq)

			if n_loci is not None and len(seqs) == n_loci:
				break

		del loci_iter

	# The chromosome, as an index into names, and the midpoint of each kept
	# locus, in order, for the reads made after the loop.
	if defer or vectorized or loop_readers:
		kept = idxs if vectorized else numpy.flatnonzero(kept_mask)
		window_codes, window_mids = codes[kept], mids[kept].astype(numpy.int64)
		del kept

	# After the loop, every signal is read when the loop read none, and
	# otherwise the signals with a _BigWigFile are read for the kept loci on
	# chromosomes that every _BigWigFile's file has.
	rows, out_after, in_after = None, signals, in_signals
	if loop_readers:
		rows = numpy.flatnonzero(readable[window_codes])
		window_codes, window_mids = window_codes[rows], window_mids[rows]
		out_after = _signals_read_after_loop(signals, out_readers)
		in_after = _signals_read_after_loop(in_signals, in_readers)

	# The failed reads are warned about in the order the loop would have
	# made them: by locus, then signals before in_signals, then by signal.
	if (defer or loop_readers) and len(window_mids) > 0:
		failures = []

		if out_after is not None:
			failures += _read_bigwig_files(out_after, out_readers, window_codes,
				window_mids - out_width - max_jitter,
				out_window + 2 * max_jitter, out_values, names, rows=rows,
				kind=0, n_jobs=n_jobs)

		if in_after is not None:
			failures += _read_bigwig_files(in_after, in_readers, window_codes,
				window_mids - in_width - max_jitter,
				in_window + 2 * max_jitter, in_values, names, rows=rows,
				kind=1, n_jobs=n_jobs)

		for _, _, _, chrom, start, end in sorted(failures):
			warnings.warn(f"{chrom} {start} {end} not valid bigwig indexes. "
				"Using zeros instead.", TangermemeWarning)

	if opened_fasta:
		try:
			if vectorized:
				if len(window_mids) > 0:
					seqs = _read_fasta_windows(sequences, (window_codes,
						window_mids - in_width - max_jitter), in_window +
						2 * max_jitter, alphabet, ignore, names=names,
						n_jobs=n_jobs)
			elif len(seqs) > 0:
				seqs = _read_fasta_windows(sequences, seqs, in_window +
					2 * max_jitter, alphabet, ignore, n_jobs=n_jobs)
		finally:
			sequences.close()

	del codes, mids, idxs, rows, readers
	if defer or vectorized or loop_readers:
		del window_codes, window_mids

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
		seqs = _one_hot_encode_rows(seqs, alphabet=alphabet, ignore=ignore,
			n_jobs=n_jobs)

	seqs = torch.from_numpy(seqs)
	y_return = [seqs]

	# A preallocated array is trimmed to the kept loci with a view, which is
	# contiguous; its rows past the last kept locus were never written.
	if signals is not None:
		if out_values is None:
			y_return.append(torch.from_numpy(numpy.stack(signals_)))
		else:
			out_values = _nan_to_num_rows(out_values[:len(seqs)],
				n_jobs=n_jobs)
			y_return.append(torch.from_numpy(out_values))

	if in_signals is not None:
		if in_values is None:
			y_return.append(torch.from_numpy(numpy.stack(in_signals_)))
		else:
			in_values = _nan_to_num_rows(in_values[:len(seqs)],
				n_jobs=n_jobs)
			y_return.append(torch.from_numpy(in_values))
		
	if return_mask:
		# Loci after the n_loci cap was reached were never read and are not
		# returned, so they are False.
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
