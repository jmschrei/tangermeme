# test_match.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pandas
import pytest

from tangermeme.utils import characters
from tangermeme.utils import one_hot_encode
from tangermeme.utils import random_one_hot

from tangermeme.io import extract_loci

from tangermeme.match import _calculate_char_perc
from tangermeme.match import _counts_from_coords
from tangermeme.match import _extract_and_filter_chrom
from tangermeme.match import extract_matching_loci

from .bigwig_writer import write_raw_bigwig

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal



###


def test_calculate_char_perc():
	seq = 'CAGCTCTAACTGATACTATCGAT'

	gc = _calculate_char_perc(seq, width=5, chars=['C', 'G'])
	at = _calculate_char_perc(seq, width=5, chars=['A', 'T'])
	n = _calculate_char_perc(seq, width=5, chars=['N'])

	assert gc.shape[0] == 4
	assert gc.dtype == numpy.float64

	assert_array_almost_equal(gc, [0.6, 0.4, 0.2, 0.4])
	assert_array_almost_equal(at, [0.4, 0.6, 0.8, 0.6])
	assert_array_almost_equal(gc+at, [1.0, 1.0, 1.0, 1.0])
	assert_array_almost_equal(n, [0.0, 0.0, 0.0, 0.0])


def test_calculate_char_perc_char():
	seq = 'ATCGATAACTACTACTACTGACGT'
	a = _calculate_char_perc(seq, width=5, chars='A')
	assert_array_almost_equal(a, [0.4, 0.4, 0.4, 0.2])


def test_calculate_char_perc_homopolymer():
	seq = 'CCCCCCCCCCCCCCCCCCCCCCC'

	gc = _calculate_char_perc(seq, width=5, chars=['C', 'G'])
	assert_array_almost_equal(gc, [1.0, 1.0, 1.0, 1.0])


def test_calculate_char_perc_N():
	seq = 'ACTATATGACACTCAGTAGCTNNNNNNNNCATCATACCATTACGACGTTCAAC'

	n = _calculate_char_perc(seq, width=10, chars=['N'])
	assert_array_almost_equal(n, [0. , 0. , 0.8, 0. , 0.])

	n = _calculate_char_perc(seq, width=10, chars='N')
	assert_array_almost_equal(n, [0. , 0. , 0.8, 0. , 0.])


def test_calculate_char_perc_short():
	seq = 'ATCGATACGT'

	gc = _calculate_char_perc(seq, width=10, chars=['C', 'G'])
	at = _calculate_char_perc(seq, width=10, chars=['A', 'T'])
	assert_array_almost_equal(gc, [0.4])
	assert_array_almost_equal(at, [0.6])


###


def test_extract_and_filter_chrom():
	regions = _extract_and_filter_chrom("tests/data/test.fa", 
		chrom='chr1', in_window=10, out_window=10)

	assert isinstance(regions, dict)
	assert len(regions) == 2
	assert_array_almost_equal(regions[25], [0, 1, 3, 5, 6, 7, 8, 10, 11, 12, 
		13, 14, 18, 22, 25, 26, 27])
	assert_array_almost_equal(regions[20], [2, 4, 9, 15, 16, 17, 19, 20, 21, 
		23, 24])


	regions = _extract_and_filter_chrom("tests/data/test.fa", 
		chrom='chr1', in_window=20, out_window=20)

	assert isinstance(regions, dict)
	assert len(regions) == 3
	assert_array_almost_equal(regions[25], [0, 3, 5, 6, 13])
	assert_array_almost_equal(regions[23], [1, 2, 4, 7, 9, 11, 12])
	assert_array_almost_equal(regions[20], [8, 10])


def test_extract_and_filter_chrom_gc_content():
	regions = _extract_and_filter_chrom("tests/data/test.fa", 
		chrom='chr1', in_window=20, out_window=20, gc_bin_width=0.05)

	for key, values in regions.items():
		chroms = ['chr1']*len(values)
		start = numpy.array(values)*20
		end = (numpy.array(values)+1)*20

		df = pandas.DataFrame({'chrom': chroms, 'start': start, 'end': end})
		X = extract_loci(df, "tests/data/test.fa", in_window=20)
		X = X.type(torch.float32)

		assert_array_almost_equal(X[:, [1, 2]].mean(axis=-1).sum(axis=1), 
			[key * 0.05]*X.shape[0])


def test_extract_and_filter_chrom_gc_content_large():
	regions = _extract_and_filter_chrom("tests/data/test.fa", 
		chrom='chr7', in_window=50, out_window=50)

	for key, values in regions.items():
		chroms = ['chr7']*len(values)
		start = numpy.array(values)*50
		end = (numpy.array(values)+1)*50

		df = pandas.DataFrame({'chrom': chroms, 'start': start, 'end': end})

		try:
			X = extract_loci(df, "tests/data/test.fa", in_window=50)
			X = X.type(torch.float32)
		except:
			continue

		assert_array_almost_equal(X[:, [1, 2]].mean(axis=-1).sum(axis=1), 
			[key * 0.02]*X.shape[0])


def test_extract_and_filter_chrom_some_N():
	regions = _extract_and_filter_chrom("tests/data/test.fa", 
		chrom='chr5', in_window=10, out_window=10)

	assert isinstance(regions, dict)
	assert len(regions) == 5
	assert_array_almost_equal(regions[30], [9, 14])
	assert_array_almost_equal(regions[25], [0, 4, 8])
	assert_array_almost_equal(regions[20], [2, 3, 15])
	assert_array_almost_equal(regions[15], [1, 7, 10, 11, 12])
	assert_array_almost_equal(regions[10], [13])


def test_extract_and_filter_chrom_many_N():
	regions = _extract_and_filter_chrom("tests/data/test.fa", 
		chrom='chr4', in_window=10, out_window=10)

	assert isinstance(regions, dict)
	assert len(regions) == 5
	assert_array_almost_equal(regions[15], [1, 7, 8])
	assert_array_almost_equal(regions[30], [2])
	assert_array_almost_equal(regions[20], [3, 20])
	assert_array_almost_equal(regions[25], [4, 5, 6, 16, 21, 22])
	assert_array_almost_equal(regions[10], [23])


def test_extract_and_filter_chrom_higher_N_filter():
	regions = _extract_and_filter_chrom("tests/data/test.fa", 
		chrom='chr4', in_window=10, out_window=10, max_n_perc=0.2)

	assert isinstance(regions, dict)
	assert len(regions) == 5
	assert_array_almost_equal(regions[15], [1, 7, 8, 15])
	assert_array_almost_equal(regions[30], [2])
	assert_array_almost_equal(regions[20], [3, 20])
	assert_array_almost_equal(regions[25], [4, 5, 6, 16, 21, 22])
	assert_array_almost_equal(regions[10], [23])


def test_extract_and_filter_chrom_no_N_filter():
	regions = _extract_and_filter_chrom("tests/data/test.fa", 
		chrom='chr4', in_window=10, out_window=10, max_n_perc=1)

	assert isinstance(regions, dict)
	assert len(regions) == 7
	assert_array_almost_equal(regions[15], [0, 1, 7, 8, 15])
	assert_array_almost_equal(regions[30], [2])
	assert_array_almost_equal(regions[20], [3, 20])
	assert_array_almost_equal(regions[25], [4, 5, 6, 16, 21, 22])
	assert_array_almost_equal(regions[10], [23])
	assert_array_almost_equal(regions[5], [17])
	assert_array_almost_equal(regions[0], [9, 10, 11, 12, 13, 14, 18, 19])


def test_extract_and_filter_chrom_N_in_windows_even():
	for window in range(10, 81, 10):
		regions = _extract_and_filter_chrom("tests/data/test.fa", 
			chrom='chr7', in_window=window, out_window=10, 
			gc_bin_width=1./window)

		for key, values in regions.items():
			chroms = ['chr7']*len(values)
			start = numpy.array(values)*window
			end = (numpy.array(values)+1)*window

			df = pandas.DataFrame({'chrom': chroms, 'start': start, 'end': end})

			try:
				X = extract_loci(df, "tests/data/test.fa", in_window=window)
				X = X.type(torch.float32)
			except:
				continue

			assert_array_almost_equal(X[:, [1, 2]].mean(axis=-1).sum(axis=1), 
				[key * 1./window]*X.shape[0])


def test_extract_and_filter_chrom_N_in_windows_odd():
	for window in range(10, 81, 5):
		regions = _extract_and_filter_chrom("tests/data/test.fa", 
			chrom='chr7', in_window=window, out_window=10, 
			gc_bin_width=1./window)

		for key, values in regions.items():
			chroms = ['chr7']*len(values)
			start = numpy.array(values)*window
			end = (numpy.array(values)+1)*window

			df = pandas.DataFrame({'chrom': chroms, 'start': start, 'end': end})

			try:
				X = extract_loci(df, "tests/data/test.fa", in_window=window)
				X = X.type(torch.float32)
			except:
				continue
				
			assert_array_almost_equal(X[:, [1, 2]].mean(axis=-1).sum(axis=1), 
				[key * 1./window]*X.shape[0])


def test_extract_and_filter_chrom_N_out_windows():
	for window in range(10, 81, 10):
		regions = _extract_and_filter_chrom("tests/data/test.fa", 
			chrom='chr7', in_window=10, out_window=window, gc_bin_width=0.1)

		for key, values in regions.items():
			chroms = ['chr7']*len(values)
			start = numpy.array(values)*10
			end = (numpy.array(values)+1)*10

			df = pandas.DataFrame({'chrom': chroms, 'start': start, 'end': end})

			try:
				X = extract_loci(df, "tests/data/test.fa", in_window=10)
				X = X.type(torch.float32)
			except:
				continue

			assert_array_almost_equal(X[:, [1, 2]].mean(axis=-1).sum(axis=1), 
				[key * 0.1]*X.shape[0])


def test_extract_and_filter_chrom_signal_threshold():
	regions = _extract_and_filter_chrom("tests/data/test.fa", 
		chrom='chr1', in_window=20, out_window=18, gc_bin_width=1.1,
		bigwig="tests/data/test.bw", signal_threshold=30)

	for key, values in regions.items():
		chroms = ['chr1']*len(values)
		start = numpy.array(values)*20
		end = (numpy.array(values)+1)*20

		df = pandas.DataFrame({'chrom': chroms, 'start': start, 'end': end})
		_, y = extract_loci(df, "tests/data/test.fa", ["tests/data/test.bw"], 
			in_window=20, out_window=18)

		assert len(y) == 14
		assert all(y.sum(axis=-1) <= 30)


	regions = _extract_and_filter_chrom("tests/data/test.fa", 
		chrom='chr1', in_window=20, out_window=18, gc_bin_width=1.1,
		bigwig="tests/data/test.bw", signal_threshold=10)

	for key, values in regions.items():
		chroms = ['chr1']*len(values)
		start = numpy.array(values)*20
		end = (numpy.array(values)+1)*20

		df = pandas.DataFrame({'chrom': chroms, 'start': start, 'end': end})
		_, y = extract_loci(df, "tests/data/test.fa", ["tests/data/test.bw"], 
			in_window=20, out_window=18)
		
		assert len(y) == 9
		assert all(y.sum(axis=-1) <= 15)


###


def test_extract_matching_loci():
	regions = extract_matching_loci("tests/data/test.bed", "tests/data/test.fa", 
		chroms=['chr1'], in_window=10, out_window=10, random_state=0)

	assert isinstance(regions, pandas.DataFrame)
	assert regions.shape == (3, 3)
	assert tuple(regions.columns) == ('chrom', 'start', 'end')

	assert numpy.unique(regions['chrom']).shape[0] == 1
	assert numpy.unique(regions['chrom']) == ('chr1',)
	assert tuple(regions['start']) == (70, 190, 240)
	assert tuple(regions['end']) == (80, 200, 250)

	X0 = extract_loci("tests/data/test.bed", "tests/data/test.fa", chroms=['chr1'], in_window=10)
	X1 = extract_loci(regions, "tests/data/test.fa", in_window=10)

	assert X0[:, [1, 2]].sum() == 13
	assert X1[:, [1, 2]].sum() == 13


def test_extract_matching_loci_int_chroms():
	# Genomes whose chromosomes are named "1", "2", etc. would otherwise get
	# read in as integers by pandas and fail to match the string names used
	# by pyfaidx.
	regions = extract_matching_loci("tests/data/test_int.bed",
		"tests/data/test_int_chroms.fa", chroms=['1'], in_window=10,
		out_window=10, random_state=0)

	regions0 = extract_matching_loci("tests/data/test.bed",
		"tests/data/test.fa", chroms=['chr1'], in_window=10, out_window=10,
		random_state=0)

	assert regions.shape == (3, 3)
	assert tuple(regions.columns) == ('chrom', 'start', 'end')
	assert tuple(numpy.unique(regions['chrom'])) == ('1',)
	assert tuple(regions['start']) == tuple(regions0['start'])
	assert tuple(regions['end']) == tuple(regions0['end'])

	# Integer chromosome names are coerced to strings as well
	regions = extract_matching_loci("tests/data/test_int.bed",
		"tests/data/test_int_chroms.fa", chroms=[1], in_window=10,
		out_window=10, random_state=0)

	assert tuple(numpy.unique(regions['chrom'])) == ('1',)
	assert tuple(regions['start']) == tuple(regions0['start'])


def test_extract_matching_loci_int_chroms_df():
	loci = pandas.read_csv("tests/data/test_int.bed", sep='\t', header=None)

	regions = extract_matching_loci(loci, "tests/data/test_int_chroms.fa",
		chroms=['1'], in_window=10, out_window=10, random_state=0)

	assert regions.shape == (3, 3)
	assert tuple(numpy.unique(regions['chrom'])) == ('1',)


def test_extract_matching_loci_int_chroms_default():
	# When chroms is None the set is derived from the loci themselves, so the
	# coercion has to happen before numpy.unique rather than only on a
	# user-provided chroms list.
	regions = extract_matching_loci("tests/data/test_int.bed",
		"tests/data/test_int_chroms.fa", in_window=10, out_window=10,
		random_state=0)

	regions0 = extract_matching_loci("tests/data/test.bed",
		"tests/data/test.fa", in_window=10, out_window=10, random_state=0)

	assert tuple(numpy.unique(regions['chrom'])) == ('1', '2')
	assert tuple(regions['start']) == tuple(regions0['start'])
	assert tuple(regions['end']) == tuple(regions0['end'])


def test_extract_matching_loci_start_edge():
	peaks = pandas.DataFrame({
		'chrom': ['chr1', 'chr1', 'chr1'],
		'start': [0, 15, 30],
		'end': [10, 25, 40]
	})

	regions = extract_matching_loci(peaks, "tests/data/test.fa", 
		chroms=['chr1'], in_window=10, out_window=10, random_state=0)
	assert regions.shape == (3, 3)


	regions = extract_matching_loci(peaks, "tests/data/test.fa", 
		chroms=['chr1'], in_window=15, out_window=10, random_state=0)
	assert regions.shape == (2, 3)


def test_extract_matching_loci_end_edge():
	peaks = pandas.DataFrame({
		'chrom': ['chr1', 'chr1', 'chr1'],
		'start': [260, 270, 280],
		'end': [270, 280, 290]
	})

	regions = extract_matching_loci(peaks, "tests/data/test.fa",
		chroms=['chr1'], in_window=10, out_window=10, random_state=0)
	assert regions.shape == (2, 3)


	regions = extract_matching_loci(peaks, "tests/data/test.fa",
		chroms=['chr1'], in_window=20, out_window=10, random_state=0)
	assert regions.shape == (1, 3)


def test_extract_matching_loci_bin_0_spillover(tmp_path):
	# bin 0 (the lowest GC bin) previously couldn't receive spillover from
	# a peak in a higher bin: the spillover loop guarded with `idx > 0`
	# instead of `>= 0`. Construct a chromosome where the only available
	# backgrounds are at GC=0 and the lone peak is at GC=1.0, so the
	# matching algorithm has nowhere else to go.
	fa_path = tmp_path / "tiny.fa"
	fa_path.write_text(">chr1\n" + "A"*100 + "G"*10 + "A"*90 + "\n")

	peaks = pandas.DataFrame({
		'chrom': ['chr1'],
		'start': [100],
		'end':   [110],
	})

	regions = extract_matching_loci(peaks, str(fa_path), in_window=10,
		out_window=10, max_n_perc=1.0, random_state=0)

	# Without the fix this returns zero rows; with the fix the peak finds
	# a GC=0 background to match against.
	assert regions.shape == (1, 3)


def test_extract_matching_loci_some_N():
	peaks = pandas.DataFrame({
		'chrom': ['chr4', 'chr4', 'chr4'],
		'start': [0, 15, 30],
		'end': [10, 15, 30]
	})

	regions = extract_matching_loci(peaks, "tests/data/test.fa", 
		chroms=['chr4'], in_window=10, out_window=10, random_state=0)

	assert isinstance(regions, pandas.DataFrame)
	assert regions.shape == (2, 3)
	assert tuple(regions.columns) == ('chrom', 'start', 'end')

	assert numpy.unique(regions['chrom']).shape[0] == 1
	assert numpy.unique(regions['chrom']) == ('chr4',)
	assert tuple(regions['start']) == (20, 80)
	assert tuple(regions['end']) == (30, 90)

	X0 = extract_loci(peaks, "tests/data/test.fa", in_window=10)
	X1 = extract_loci(regions, "tests/data/test.fa", in_window=10)

	assert X0[:, [1, 2]].sum() == 12
	assert X1[:, [1, 2]].sum() == 9


def test_extract_matching_loci_N():
	peaks = pandas.DataFrame({
		'chrom': ['chr4', 'chr4', 'chr4'],
		'start': [0, 90, 100],
		'end': [10, 100, 110]
	})

	regions = extract_matching_loci(peaks, "tests/data/test.fa", 
		chroms=['chr4'], in_window=10, out_window=10, random_state=0)

	assert isinstance(regions, pandas.DataFrame)
	assert regions.shape == (0, 3)
	assert tuple(regions.columns) == ('chrom', 'start', 'end')

	assert tuple(regions['chrom']) == tuple()
	assert tuple(regions['start']) == tuple()
	assert tuple(regions['end']) == tuple()


def test_extract_matching_loci_allow_N():
	peaks = pandas.DataFrame({
		'chrom': ['chr4', 'chr4', 'chr4'],
		'start': [0, 90, 100],
		'end': [10, 100, 110]
	})

	regions = extract_matching_loci(peaks, "tests/data/test.fa",
		chroms=['chr4'], in_window=10, out_window=10, max_n_perc=1.1,
		random_state=0)

	assert isinstance(regions, pandas.DataFrame)
	assert regions.shape == (3, 3)
	assert tuple(regions.columns) == ('chrom', 'start', 'end')

	assert tuple(regions['chrom']) == ('chr4', 'chr4', 'chr4')
	assert tuple(regions['start']) == (70, 120, 140)
	assert tuple(regions['end']) == (80, 130, 150)


###


def test_extract_matching_loci_determinism():
	regions0 = extract_matching_loci("tests/data/test.bed", "tests/data/test.fa",
		chroms=['chr1'], in_window=10, out_window=10, random_state=0)
	regions1 = extract_matching_loci("tests/data/test.bed", "tests/data/test.fa",
		chroms=['chr1'], in_window=10, out_window=10, random_state=0)

	pandas.testing.assert_frame_equal(regions0, regions1)


def test_extract_matching_loci_n_jobs_parallel():
	regions0 = extract_matching_loci("tests/data/test.bed", "tests/data/test.fa",
		chroms=['chr1'], in_window=10, out_window=10, random_state=0,
		n_jobs=1)
	regions1 = extract_matching_loci("tests/data/test.bed", "tests/data/test.fa",
		chroms=['chr1'], in_window=10, out_window=10, random_state=0,
		n_jobs=2)

	pandas.testing.assert_frame_equal(
		regions0.reset_index(drop=True), regions1.reset_index(drop=True))


def test_extract_matching_loci_chroms_none():
	peaks = pandas.DataFrame({
		'chrom': ['chr1', 'chr1', 'chr1'],
		'start': [10, 50, 100],
		'end': [20, 60, 110],
	})

	regions = extract_matching_loci(peaks, "tests/data/test.fa",
		in_window=10, out_window=10, random_state=0)

	assert isinstance(regions, pandas.DataFrame)
	assert tuple(regions.columns) == ('chrom', 'start', 'end')


##
# Signal from a bigWig. The bigWig's chr2 is shorter than the FASTA's, its
# chr3 is absent, and chr6 has a NaN and an infinite interval. Expected
# values are computed from the bases written, as float64 sums that skip the
# bases without a value.
##


@pytest.fixture(scope="module")
def match_bigwig(tmp_path_factory):
	rng = numpy.random.RandomState(0)
	sizes = {'chr1': 284, 'chr2': 150, 'chr4': 240, 'chr5': 160, 'chr6': 80}

	tracks, sections = {}, []
	for chrom, length in sizes.items():
		track = numpy.full(length, numpy.nan, dtype=numpy.float32)
		rows, pos = [], 0
		while True:
			pos += int(rng.randint(0, 6))
			width = int(rng.randint(1, 10))
			if pos + width > length:
				break
			rows.append((pos, pos + width, float(numpy.float32(rng.uniform(-2,
				6)))))
			pos += width

		if chrom == 'chr6':
			rows[0] = (rows[0][0], rows[0][1], float('nan'))
			rows[1] = (rows[1][0], rows[1][1], float('inf'))

		for start, end, value in rows:
			track[start:end] = value

		sections += [(chrom, 1, 0, 0, rows[k:k+20]) for k in range(0, len(rows),
			20)]
		tracks[chrom] = track

	path = str(tmp_path_factory.mktemp("match") / "signal.bw")
	write_raw_bigwig(path, sizes, sections)
	return path, tracks


def _expected_counts(tracks, coords):
	return numpy.array([numpy.nansum(tracks[chrom][start:end].astype(
		numpy.float64)) if chrom in tracks else numpy.nan
		for chrom, start, end in coords])


def test_counts_from_coords(match_bigwig):
	# The signal summed over each window: zero where there is none, the sum
	# of the bases that exist where a window runs past the bigWig's end of a
	# chromosome, NaN on a chromosome the bigWig lacks, and NaN values
	# skipped. Windows of different widths, and of none, in one call.
	path, tracks = match_bigwig
	rng = numpy.random.RandomState(1)

	coords = []
	for chrom in ['chr1', 'chr2', 'chr3', 'chr4', 'chr6']:
		for _ in range(40):
			start = int(rng.randint(0, 200 if chrom != 'chr6' else 70))
			coords.append((chrom, start, start + int(rng.randint(0, 50))))
	coords += [('chr2', 140, 170), ('chr2', 150, 160), ('chr2', 155, 156),
		('chr6', 0, 20)]

	counts = _counts_from_coords(path, coords)
	expected = _expected_counts(tracks, coords)

	assert counts.dtype == numpy.float64
	assert counts.shape == (len(coords),)
	assert counts.tobytes() == expected.tobytes()
	assert numpy.isnan(counts[[c == 'chr3' for c, _, _ in coords]]).all()
	assert numpy.isinf(counts[-1])


def test_counts_from_coords_past_chrom_end_bitwise(tmp_path):
	# A long region past the end of a chromosome is summed over the bases
	# before the end alone, so that the float64 sum adds the same values in
	# the same order as a slice of the chromosome does. A float64 sum of
	# float32 values is exact in any order unless they span more than about
	# 2**29, so these span 24 orders of magnitude, with both signs.
	import figwig

	rng = numpy.random.RandomState(2)
	track = (rng.choice([-1, 1], size=1000) * 10 ** rng.uniform(-12, 12,
		size=1000)).astype(numpy.float32)
	path = str(tmp_path / "wide.bw")
	figwig.write_bigwig(path, {'chr1': 1000}, 'chr1', [0], track[None])

	coords = [('chr1', start, start + width) for start in (0, 300, 700, 990)
		for width in (700, 1500, 2114)]
	counts = _counts_from_coords(path, coords)
	expected = _expected_counts({'chr1': track}, coords)
	assert counts.tobytes() == expected.tobytes()


def test_counts_from_coords_generator(match_bigwig):
	path, tracks = match_bigwig
	coords = [('chr1', 10, 30), ('chr4', 100, 101), ('chr3', 5, 9)]

	counts = _counts_from_coords(path, (c for c in coords))
	assert counts.tobytes() == _expected_counts(tracks, coords).tobytes()


@pytest.mark.parametrize("chrom", ["chr1", "chr6"])
@pytest.mark.parametrize("in_window, out_window", [(20, 18), (10, 4), (9, 2)])
@pytest.mark.parametrize("threshold", [0.0, 10.0])
def test_extract_and_filter_chrom_bigwig(match_bigwig, chrom, in_window,
	out_window, threshold):
	# The windows kept are those without the signal filter whose middle
	# out_window bases sum to at most the threshold.
	path, tracks = match_bigwig
	regions = _extract_and_filter_chrom("tests/data/test.fa", chrom, in_window,
		out_window, max_n_perc=1.0, bigwig=path, signal_threshold=threshold)
	unfiltered = _extract_and_filter_chrom("tests/data/test.fa", chrom,
		in_window, out_window, max_n_perc=1.0)

	left = (in_window - out_window) // 2
	counts = _expected_counts(tracks, [(chrom, i * in_window + left, i *
		in_window + left + out_window) for i in range(len(tracks[chrom]) //
		in_window)])

	expected = {gc: [i for i in idxs if counts[i] <= threshold]
		for gc, idxs in unfiltered.items()}
	assert regions == {gc: idxs for gc, idxs in expected.items() if idxs}


def test_extract_and_filter_chrom_bigwig_shorter_chrom(match_bigwig):
	# The bigWig's chr2 is 150 bases and the FASTA's 211: the windows past
	# 150 have no signal. This used to fail to broadcast the signal of the
	# bigWig's 150 bases against the FASTA's windows.
	path, tracks = match_bigwig
	regions = _extract_and_filter_chrom("tests/data/test.fa", "chr2", 10, 4,
		max_n_perc=1.0, bigwig=path, signal_threshold=0.0)
	unfiltered = _extract_and_filter_chrom("tests/data/test.fa", "chr2", 10, 4,
		max_n_perc=1.0)

	track = numpy.full(211, numpy.nan, dtype=numpy.float32)
	track[:150] = tracks['chr2']
	counts = _expected_counts({'chr2': track}, [('chr2', i * 10 + 3, i * 10 + 7)
		for i in range(21)])

	expected = {gc: [i for i in idxs if counts[i] <= 0.0]
		for gc, idxs in unfiltered.items()}
	assert regions == {gc: idxs for gc, idxs in expected.items() if idxs}
	assert any(i >= 15 for idxs in regions.values() for i in idxs)


def test_extract_and_filter_chrom_bigwig_missing_chrom(match_bigwig):
	# A chromosome the bigWig lacks has no windows to draw from. This used to
	# raise the KeyError of pybigtools.
	path, _ = match_bigwig
	assert _extract_and_filter_chrom("tests/data/test.fa", "chr3", 10, 4,
		bigwig=path, signal_threshold=1e9) == {}


@pytest.mark.parametrize("in_window, out_window", [(10, 4), (20, 18)])
def test_extract_matching_loci_bigwig_no_filter(tmp_path, in_window,
	out_window):
	# With 1.0 at every base, the loci's counts are out_window, so a large
	# signal_beta puts the threshold above every window's signal, and the
	# loci drawn are those drawn without a bigWig. With pybigtools 0.2.5,
	# every locus's count was NaN, so the threshold was NaN and no loci were
	# returned.
	import figwig

	sizes = {'chr1': 284, 'chr2': 211, 'chr4': 240, 'chr6': 80}
	path = str(tmp_path / "ones.bw")
	figwig.write_bigwig(path, sizes, list(sizes), [0] * len(sizes),
		numpy.ones(len(sizes)), ends=list(sizes.values()))

	kwargs = dict(in_window=in_window, out_window=out_window, max_n_perc=1.0,
		chroms=list(sizes), random_state=0)

	expected = extract_matching_loci("tests/data/test.bed", "tests/data/test.fa",
		**kwargs)
	result = extract_matching_loci("tests/data/test.bed", "tests/data/test.fa",
		bigwig=path, signal_beta=2.0, **kwargs)

	assert len(result) > 0
	pandas.testing.assert_frame_equal(result, expected)


def test_extract_matching_loci_bigwig_filter(match_bigwig):
	# With signal_beta of 0 the threshold is 0, so every locus drawn has no
	# positive signal in its middle out_window bases.
	path, tracks = match_bigwig
	result = extract_matching_loci("tests/data/test.bed", "tests/data/test.fa",
		in_window=10, out_window=4, max_n_perc=1.0, bigwig=path,
		signal_beta=0.0, chroms=['chr1', 'chr2', 'chr4', 'chr6'],
		random_state=0)

	assert len(result) > 0
	counts = _expected_counts(tracks, [(chrom, start + 3, start + 7)
		for chrom, start in zip(result['chrom'], result['start'])])
	assert (counts <= 0).all()

