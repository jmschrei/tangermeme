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
from tangermeme.match import _extract_and_filter_chrom
from tangermeme.match import extract_matching_loci

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


@pytest.fixture
def X():
	return 5


###


def test_calculate_char_perc():
	seq = 'CAGCTCTAACTGATACTATCGAT'

	gc = _calculate_char_perc(seq, width=5, chars=['C', 'G'])
	at = _calculate_char_perc(seq, width=5, chars=['A', 'T'])
	n = _calculate_char_perc(seq, width=5, chars=['N'])

	assert gc.shape[0] == 4
	assert gc.dtype == torch.float32

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


def test_calculate_char_perc_raises_long():
	seq = 'ATCGATACGT'
	assert_raises(ValueError, _calculate_char_perc, seq, width=11, 
		chars=['C', 'G'])


def test_calculate_char_perc_raises_sequence():
	assert_raises(ValueError, _calculate_char_perc, ['A', 'C', 'G', 'T'], 2, 
		['A', 'C'])
	assert_raises(ValueError, _calculate_char_perc, one_hot_encode('ACATCTG'), 
		2, ['A', 'C'])


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
	assert_array_almost_equal(regions[22], [1, 2, 4, 7, 9, 11, 12])
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



	#def extract_matching_loci(loci, fasta, in_window=2114, out_window=1000, 
	#max_n_perc=0.1, gc_bin_width=0.02, bigwig=None, signal_beta=0.5, 
	#chroms=None, random_state=None, verbose=False):


def test_extract_matching_loci():
	regions = extract_matching_loci("tests/data/test.bed", "tests/data/test.fa", 
		chroms=['chr1'], in_window=10, out_window=10, random_state=0)

	assert isinstance(regions, pandas.DataFrame)
	assert regions.shape == (5, 3)
	assert tuple(regions.columns) == ('chrom', 'start', 'end')

	assert numpy.unique(regions['chrom']).shape[0] == 1
	assert numpy.unique(regions['chrom']) == ('chr1',)
	assert tuple(regions['start']) == (70, 170, 190, 200, 240)
	assert tuple(regions['end']) == (80, 180, 200, 210, 250)

	X0 = extract_loci("tests/data/test.bed", "tests/data/test.fa", in_window=10)
	X1 = extract_loci(regions, "tests/data/test.fa", in_window=10)

	assert X0[:, [1, 2]].sum() == 20
	assert X1[:, [1, 2]].sum() == 21


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
