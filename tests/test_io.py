# test_io.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import re
import zlib
import numpy
import numba
import torch
import struct
import warnings
import pytest
import pandas
import pathlib
import pyfaidx
import pybigtools

import tangermeme.io

from tangermeme.io import _interleave_loci
from tangermeme.io import _load_signals
from tangermeme.io import _load_exclusion_zones
from tangermeme.io import _extract_locus_signal
from tangermeme.io import _read_fasta_windows
from tangermeme.io import _read_fasta_windows_mmap
from tangermeme.io import _read_signal_windows
from tangermeme.io import _write_locus_signal
from tangermeme.io import _BigWigFile
from tangermeme.io import _inflate_bigwig_blocks
from tangermeme.io import _zlib_uncompress

from tangermeme.io import read_meme
from tangermeme.io import extract_loci
from tangermeme.io import read_vcf
from tangermeme.io import one_hot_to_fasta

from tangermeme.utils import one_hot_encode
from tangermeme.utils import TangermemeWarning

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


@pytest.fixture
def short_loci1():
	return pandas.DataFrame({
		'chrom': ['chr1', 'chr1', 'chr1', 'chr2', 'chr2'],
		'start': [10, 80, 140, 25, 35],
		'end': [30, 100, 160, 55, 65]
	})


@pytest.fixture
def short_loci2():
	return pandas.DataFrame({
		'chrom': ['chr1', 'chr1', 'chr2', 'chr2', 'chr3', 'chr3', 'chr4',
			'chr5', 'chr5'],
		'start': [40, 120, 40, 120, 5, 25, 20, 50, 80],
		'end': [60, 140, 60, 140, 25, 45, 40, 70, 100]
	})


@pytest.fixture
def loci_seqs():
	return [
	    [[1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
         [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
         [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]],

        [[0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
         [1, 0, 0, 1, 1, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0, 0, 1, 0, 0, 0]],

        [[0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
         [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 1, 0, 0, 0, 1]],

        [[0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
         [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 1, 0, 0, 1, 0, 0, 1, 0, 1]],

        [[1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
         [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 1, 0, 1, 0, 1]]
    ]


@pytest.fixture
def loci2_seqs():
	return [
		[[0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
		 [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
		 [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
		 [0, 0, 0, 0, 1, 0, 0, 0, 1, 0]],

		[[0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
		 [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
		 [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
		 [1, 0, 0, 0, 1, 0, 0, 0, 1, 0]],

		[[1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
		 [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 0, 1, 0, 0, 1, 0, 1, 0, 1]],

		[[0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
		 [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
		 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
		 [0, 1, 0, 1, 0, 0, 1, 0, 0, 0]],

		[[0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
		 [0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
		 [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],

		[[0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
		 [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 0, 0, 1, 0, 0, 1, 0, 0, 1]],

		[[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
		 [1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
		 [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
		 [0, 1, 0, 0, 0, 0, 0, 0, 1, 1]],

		[[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
		 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

		[[0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
		 [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
		 [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
		 [0, 0, 1, 0, 0, 0, 0, 0, 0, 1]]
	]


@pytest.fixture
def loci_signal():
	return [
		[[1.4791311025619507, 0.49575525522232056, 0.12961287796497345,
		  0.11122498661279678, 0.7721329927444458, 1.9588080644607544,
		  0.1597556471824646, 0.1069917380809784, 1.620334506034851,
		  1.0501784086227417],
		 [0.13018153607845306, 0.6349451541900635, 0.43743935227394104,
		  0.23381824791431427, 0.5979234576225281, 1.9726061820983887,
		  0.4418468773365021, 0.778532087802887, 0.3186179995536804,
		  1.5812746286392212]],

		[[0.20437310636043549, 0.22272270917892456, 0.30441057682037354,
		  1.9459599256515503, 0.16146144270896912, 2.937054395675659,
		  1.358080267906189, 1.1391572952270508, 0.5296049118041992,
		  1.6804014444351196],
		 [1.206719994544983, 0.2273847460746765, 1.027625560760498,
		  1.3341655731201172, 0.041398968547582626, 1.630789875984192,
		  0.0022149726282805204, 0.8115938901901245, 0.09121190011501312,
		  0.322033166885376]],

		[[0.48324692249298096, 1.001604437828064, 0.6735547780990601,
		  0.363113671541214, 0.6795873045921326, 0.5142682790756226,
		  1.7785857915878296, 1.4153209924697876, 0.6760514974594116,
		  0.27945613861083984],
		 [0.32088014483451843, 0.9074110984802246, 1.125154733657837,
		  1.0642825365066528, 0.2592030167579651, 0.9836715459823608,
		  0.6187756657600403, 0.058271586894989014, 0.18631218373775482,
		  1.8474831581115723]],

		[[0.8443564176559448, 0.5371090173721313, 0.2512274384498596,
		  0.9364936351776123, 1.5986690521240234, 0.6665113568305969,
		  1.692335605621338, 0.41500651836395264, 1.6298010349273682,
		  1.1278733015060425],
		 [0.0422024242579937, 0.8806508779525757, 1.4816402196884155,
		  1.1951619386672974, 0.6161070466041565, 0.21235889196395874,
		  0.29951179027557373, 0.7038488388061523, 1.1673632860183716,
		  1.2555632591247559]],

		[[1.5983798503875732, 0.24384598433971405, 0.8742402195930481,
		  0.9665769934654236, 0.677355170249939, 1.2272214889526367,
		  0.867402970790863, 0.19692184031009674, 2.5657830238342285,
		  1.0753364562988281],
		 [0.9491047859191895, 0.4262428283691406, 0.9736483693122864,
		  0.6608085036277771, 0.6526103615760803, 2.4809725284576416,
		  0.702046811580658, 0.17438605427742004, 0.6000516414642334,
		  1.1334248781204224]]
	]


@pytest.fixture
def dict_signal():
	# One value per base of each chromosome in tests/data/test.fa: the
	# coordinate plus 10000 times the chromosome's index, so any extracted
	# window can be checked by arithmetic. Every value is exact in float32.
	lengths = {'chr1': 284, 'chr2': 211, 'chr3': 126, 'chr4': 240,
		'chr5': 160, 'chr6': 80, 'chr7': 2000}

	return {chrom: numpy.arange(length, dtype=numpy.float32) + 10000 * i
		for i, (chrom, length) in enumerate(lengths.items())}


@pytest.fixture
def dict_sequences():
	# tests/data/test.fa one-hot encoded, as C-contiguous int8 numpy arrays of
	# shape (4, length). chr5 has a Z, which is left as an all-zero column.
	fasta = pyfaidx.Fasta("tests/data/test.fa")

	return {chrom: numpy.ascontiguousarray(one_hot_encode(
		str(fasta[chrom]).upper(), ignore=['N', 'Z']).numpy())
		for chrom in fasta.keys()}



##


def test_interleave_loci_single_str(short_loci1, short_loci2):
	df = _interleave_loci("tests/data/test.bed")
	assert (df == short_loci1).all(None)
	assert_raises(ValueError, df.__eq__, short_loci2)

	df = _interleave_loci("tests/data/test2.bed")
	assert (df == short_loci2).all(None)
	assert_raises(ValueError, df.__eq__, short_loci1)


def test_interleave_loci_single_df(short_loci1, short_loci2):
	names = ['chrom', 'start', 'end']

	loci = pandas.read_csv("tests/data/test.bed", delimiter='\t',
		index_col=False, names=names, header=None)
	df = _interleave_loci(loci)
	assert (df == short_loci1).all(None)
	assert_raises(ValueError, df.__eq__, short_loci2)

	loci = pandas.read_csv("tests/data/test2.bed", delimiter='\t',
		index_col=False, names=names, header=None)
	df = _interleave_loci(loci)
	assert (df == short_loci2).all(None)
	assert_raises(ValueError, df.__eq__, short_loci1)


def test_interleave_loci_df_other_column_names(short_loci1):
	loci = pandas.read_csv("tests/data/test.bed", delimiter='\t', header=None,
		names=['a', 'b', 'c'])

	df = _interleave_loci(loci)
	assert (df == short_loci1).all(None)

	df = _interleave_loci(loci, chroms=['chr1'])
	assert list(df.columns) == ['chrom', 'start', 'end']
	assert list(df['chrom']) == ['chr1', 'chr1', 'chr1']
	assert list(df['start']) == [10, 80, 140]
	assert list(df['end']) == [30, 100, 160]


def test_interleave_loci_multi_str(short_loci1, short_loci2):
	df = _interleave_loci(["tests/data/test.bed", "tests/data/test.bed"])
	idxs = numpy.repeat(numpy.arange(5), 2)
	base_df = short_loci1.iloc[idxs].reset_index(drop=True)

	assert_raises(ValueError, df.__eq__, short_loci1)
	assert_raises(ValueError, df.__eq__, short_loci2)
	assert (df == base_df).all(None)

	#

	df = _interleave_loci(["tests/data/test.bed", "tests/data/test2.bed"])

	assert_raises(ValueError, df.__eq__, short_loci1)
	assert_raises(ValueError, df.__eq__, short_loci2)

	base_df = pandas.DataFrame({
		'chrom': ['chr1', 'chr1', 'chr1', 'chr1', 'chr1', 'chr2', 'chr2',
			'chr2', 'chr2', 'chr3', 'chr3', 'chr4', 'chr5', 'chr5'],
		'start': [10, 40, 80, 120, 140, 40, 25, 120, 35, 5, 25, 20, 50, 80],
		'end': [30, 60, 100, 140, 160, 60, 55, 140, 65, 25, 45, 40, 70, 100]
	})

	assert (df == base_df).all(None)

	#

	df = _interleave_loci(["tests/data/test2.bed", "tests/data/test.bed"])

	assert_raises(ValueError, df.__eq__, short_loci1)
	assert_raises(ValueError, df.__eq__, short_loci2)

	base_df = pandas.DataFrame({
		'chrom': ['chr1', 'chr1', 'chr1', 'chr1', 'chr2', 'chr1', 'chr2',
			'chr2', 'chr3', 'chr2', 'chr3', 'chr4', 'chr5', 'chr5'],
		'start': [40, 10, 120, 80, 40, 140, 120, 25, 5, 35, 25, 20, 50, 80],
		'end': [60, 30, 140, 100, 60, 160, 140, 55, 25, 65, 45, 40, 70, 100]
	})

	assert (df == base_df).all(None)


def test_interleave_loci_multi_chroms(short_loci1, short_loci2):
	chroms = ['chr1', 'chr2']

	df = _interleave_loci(["tests/data/test.bed", "tests/data/test.bed"],
		chroms=chroms)
	idxs = numpy.repeat(numpy.arange(5), 2)
	base_df = short_loci1.iloc[idxs].reset_index(drop=True)

	assert_raises(ValueError, df.__eq__, short_loci1)
	assert_raises(ValueError, df.__eq__, short_loci2)
	assert (df == base_df).all(None)

	#

	df = _interleave_loci(["tests/data/test.bed", "tests/data/test2.bed"],
		chroms=chroms)

	assert_raises(ValueError, df.__eq__, short_loci1)
	assert(not (df == short_loci2).all(None))

	base_df = pandas.DataFrame({
		'chrom': ['chr1', 'chr1', 'chr1', 'chr1', 'chr1', 'chr2', 'chr2',
			'chr2', 'chr2'],
		'start': [10, 40, 80, 120, 140, 40, 25, 120, 35],
		'end': [30, 60, 100, 140, 160, 60, 55, 140, 65]
	})

	assert (df == base_df).all(None)

	#

	df = _interleave_loci(["tests/data/test2.bed", "tests/data/test.bed"],
		chroms=chroms)

	assert_raises(ValueError, df.__eq__, short_loci1)
	assert(not (df == short_loci2).all(None))

	base_df = pandas.DataFrame({
		'chrom': ['chr1', 'chr1', 'chr1', 'chr1', 'chr2', 'chr1', 'chr2',
			'chr2', 'chr2'],
		'start': [40, 10, 120, 80, 40, 140, 120, 25, 35],
		'end': [60, 30, 140, 100, 60, 160, 140, 55, 65]
	})

	assert (df == base_df).all(None)


def test_interleave_loci_bed10_summits_str():
	df = _interleave_loci("tests/data/test2.bed10", summits=True)

	base_df = pandas.DataFrame({
		'chrom': ['chr1', 'chr1', 'chr2', 'chr2', 'chr3', 'chr3', 'chr4',
			'chr5', 'chr5'],
		'start': [30, 111, 35, 122, 10, 15, 11, 42, 87],
		'end': [50, 131, 55, 142, 30, 35, 31, 62, 107]
	})

	assert (df == base_df).all(None)


def test_interleave_loci_bed10_no_summits_str():
	df = _interleave_loci("tests/data/test2.bed10", summits=False)

	base_df = pandas.DataFrame({
		'chrom': ['chr1', 'chr1', 'chr2', 'chr2', 'chr3', 'chr3', 'chr4',
			'chr5', 'chr5'],
		'start': [40, 120, 40, 120, 5, 25, 20, 50, 80],
		'end': [60, 140, 60, 140, 25, 45, 40, 70, 100]
	})

	assert (df == base_df).all(None)


def test_interleave_loci_bed10_summits_df():
	loci = pandas.read_csv("tests/data/test2.bed10", delimiter='\t',
		header=None)
	df = _interleave_loci(loci, summits=True)

	base_df = pandas.DataFrame({
		'chrom': ['chr1', 'chr1', 'chr2', 'chr2', 'chr3', 'chr3', 'chr4',
			'chr5', 'chr5'],
		'start': [30, 111, 35, 122, 10, 15, 11, 42, 87],
		'end': [50, 131, 55, 142, 30, 35, 31, 62, 107]
	})

	assert (df == base_df).all(None)


def test_interleave_loci_bed10_no_summits_df():
	loci = pandas.read_csv("tests/data/test2.bed10", delimiter='\t',
		header=None)
	df = _interleave_loci(loci, summits=False)

	base_df = pandas.DataFrame({
		'chrom': ['chr1', 'chr1', 'chr2', 'chr2', 'chr3', 'chr3', 'chr4',
			'chr5', 'chr5'],
		'start': [40, 120, 40, 120, 5, 25, 20, 50, 80],
		'end': [60, 140, 60, 140, 25, 45, 40, 70, 100]
	})

	assert (df == base_df).all(None)


def test_interleave_loci_single_raises():
	assert_raises(ValueError, _interleave_loci, 5)
	assert_raises(ValueError, _interleave_loci, numpy.random.randn(5, 5))


def test_interleave_loci_multi_raises():
	assert_raises(ValueError, _interleave_loci, [5, 1])
	assert_raises(ValueError, _interleave_loci, [numpy.random.randn(5, 5)])


def test_interleave_loci_multi_raises_chroms():
	assert_raises(ValueError, _interleave_loci, ["tests/data/test2.bed",
		"tests/data/test.bed"], 'chr1')


def test_interleave_loci_raises_summits_bed():
	assert_raises(ValueError, _interleave_loci, ["tests/data/test2.bed",
		"tests/data/test.bed"], None, True)

def test_interleave_loci_raises_summits_interleave():
	assert_raises(ValueError, _interleave_loci, ["tests/data/test2.bed",
		"tests/data/test.bed"], None, True)

def test_interleave_loci_raises_summits_negative():
	base_df = pandas.DataFrame({
		'chrom': ['chr1', 'chr1', 'chr2', 'chr2'],
		'start': [40, 120, 40, 120],
		'end': [60, 140, 60, 140],
		'4': ['-', '-', '-', '-'],
		'5': ['-', '-', '-', '-'],
		'6': ['-', '-', '-', '-'],
		'7': ['-', '-', '-', '-'],
		'8': ['-', '-', '-', '-'],
		'9': ['-', '-', '-', '-'],
		'10': [-1, 0, 1, 0],
	})

	assert_raises(ValueError, _interleave_loci, base_df, None, True)


def test_interleave_loci_raises_summits_too_large():
	base_df = pandas.DataFrame({
		'chrom': ['chr1', 'chr1', 'chr2', 'chr2'],
		'start': [40, 120, 40, 120],
		'end': [60, 140, 60, 140],
		'4': ['-', '-', '-', '-'],
		'5': ['-', '-', '-', '-'],
		'6': ['-', '-', '-', '-'],
		'7': ['-', '-', '-', '-'],
		'8': ['-', '-', '-', '-'],
		'9': ['-', '-', '-', '-'],
		'10': [21, 0, 1, 0],
	})

	assert_raises(ValueError, _interleave_loci, base_df, None, True)


##


def test_load_signals_none():
	bw = _load_signals(None)
	assert bw is None


def test_load_signals_bw():
	bw = _load_signals(["tests/data/test.bw"])

	assert len(bw) == 1
	assert isinstance(bw, list)

	bw = _load_signals(["tests/data/test.bw", "tests/data/test2.bw"])

	assert len(bw) == 2
	assert isinstance(bw, list)


def test_load_signals_values():
	bw = _load_signals(["tests/data/test.bw"])

	vals = bw[0].values("chr1", 0, 20)

	assert type(vals) == numpy.ndarray
	assert vals.shape == (20,)
	assert_array_almost_equal(vals, [
		0.407911, 1.343698, 2.955252, 0.897452, 0.928617, 1.562161,
		0.662164, 1.387003, 0.963338, 1.988053, 1.373694, 1.417226,
		1.202522, 0.829855, 1.740464, 1.479131, 0.495755, 0.129613,
		0.111225, 0.772133])


def test_load_signals_multi_values():
	bw = _load_signals(["tests/data/test.bw", "tests/data/test2.bw"])

	vals = bw[0].values("chr1", 0, 10)
	assert type(vals) == numpy.ndarray
	assert vals.shape == (10,)
	assert_array_almost_equal(vals, [
		0.407911, 1.343698, 2.955252, 0.897452, 0.928617, 1.562161,
		0.662164, 1.387003, 0.963338, 1.988053])

	vals = bw[1].values("chr1", 0, 10)
	assert type(vals) == numpy.ndarray
	assert vals.shape == (10,)
	assert_array_almost_equal(vals, [
		0.210908, 1.711426, 0.292976, 0.948357, 1.946163, 0.806502,
		0.342074, 0.386286, 0.655825, 0.257574])


def test_load_signals_nan():
	bw = _load_signals(["tests/data/test3.bw"])

	vals = bw[0].values("chr1", 0, 40, missing=numpy.nan)
	assert len(bw) == 1
	assert isinstance(bw, list)
	assert numpy.isnan(vals).sum() == 20

	nan = numpy.nan
	assert_array_almost_equal(vals, [
		nan, nan, nan, nan, nan, nan, nan, nan, nan,  6.,  1., nan,  1.,
		2.,  4., 14., 28., nan, 42., 36., 41., 50.,  2.,  1., nan, 20.,
		4., nan, nan, nan, nan, nan,  1., nan,  1., nan,  1.,  5.,  3.,
		nan])


def test_load_signals_dict():
	signal = {
		'chr1': numpy.array([0.0, 0.0, 0.5, 0.0, 1.0, 0.0]),
		'chr2': numpy.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
	}

	bw = _load_signals([signal])

	assert len(bw) == 1
	assert isinstance(bw, list)
	assert isinstance(bw[0], dict)


def test_load_signals_bbiread():
	# A bigWig the caller already opened is used as is; it used to be
	# rejected even though _extract_locus_signal reads it.
	bw = pybigtools.open("tests/data/test.bw")
	signals = _load_signals([bw])

	assert len(signals) == 1
	assert signals[0] is bw


def test_load_signals_mixed():
	signal = {'chr1': numpy.zeros(6)}
	bw = _load_signals(("tests/data/test.bw", signal))

	assert isinstance(bw, list)
	assert isinstance(bw[0], pybigtools.BBIRead)
	assert isinstance(bw[1], dict)


def test_load_signals_raises_not_list():
	# A single filename or dict used to be iterated over, so each character
	# or chromosome name was opened as a bigWig, which failed with "Invalid
	# file type".
	for signals in ("tests/data/test.bw", {'chr1': numpy.zeros(6)}):
		with pytest.raises(ValueError, match="must be a list or tuple"):
			_load_signals(signals)


def test_load_signals_raises_type():
	for signal in (numpy.zeros(6), torch.zeros(6), 5):
		with pytest.raises(ValueError, match="Signals must either be"):
			_load_signals([signal])


def test_load_signals_raises_dict_values():
	with pytest.raises(ValueError, match="must be numpy.ndarrays"):
		_load_signals([{'chr1': [0.0, 1.0]}])

	with pytest.raises(ValueError, match="must be numpy.ndarrays"):
		_load_signals([{'chr1': torch.zeros(6)}])


##


def test_load_exclusion_zones_none():
	assert _load_exclusion_zones({'chr1': 250}, None) is None


def test_load_exclusion_zones():
	# One boolean per 100 bp chunk of each chromosome, True where a region
	# covers any base of the chunk. Ends are exclusive, regions with end <=
	# start cover nothing, and chromosomes without a length are skipped.
	chrom_lengths = {'chr1': 350, 'chr2': 100, 'chr3': 99}
	exclusion = pandas.DataFrame({
		0: ['chr1', 'chr1', 'chr2', 'chrM', 'chr2', 'chr3'],
		1: [0, 201, 99, 0, 50, 10],
		2: [100, 250, 100, 10, 50, 5],
	})

	zones = _load_exclusion_zones(chrom_lengths, exclusion)

	assert sorted(zones.keys()) == ['chr1', 'chr2', 'chr3']
	assert zones['chr1'].dtype == bool
	assert zones['chr1'].tolist() == [True, False, True, False]
	assert zones['chr2'].tolist() == [True, False]
	assert zones['chr3'].tolist() == [False]


def test_load_exclusion_zones_inputs(tmp_path):
	# A filename, a DataFrame, and a list mixing the two are equivalent.
	chrom_lengths = {'chr1': 350}
	a = pandas.DataFrame({0: ['chr1'], 1: [0], 2: [50]})
	b = pandas.DataFrame({0: ['chr1'], 1: [300], 2: [350]})
	filename = tmp_path / "a.bed"
	a.to_csv(filename, sep='\t', header=False, index=False)

	for exclusion in ([a, b], [str(filename), b], [filename, b]):
		zones = _load_exclusion_zones(chrom_lengths, exclusion)
		assert zones['chr1'].tolist() == [True, False, False, True]

	for exclusion in (a, str(filename), filename):
		zones = _load_exclusion_zones(chrom_lengths, exclusion)
		assert zones['chr1'].tolist() == [True, False, False, False]


##


def test_extract_locus_signal_single():
	bw = pybigtools.open("tests/data/test.bw")
	signal = _extract_locus_signal([bw], 'chr1', 3, 14)

	assert len(signal) == 1
	assert signal[0].shape == (11,)

	assert isinstance(signal, list)
	assert isinstance(signal[0], numpy.ndarray)

	assert_array_almost_equal(signal, [[0.897452, 0.928617, 1.562161, 0.662164,
		1.387003, 0.963338, 1.988053, 1.373694, 1.417226, 1.202522, 0.829855]])

	signal = _extract_locus_signal([bw], 'chr2', 16, 30)

	assert len(signal) == 1
	assert signal[0].shape == (14,)

	assert isinstance(signal, list)
	assert isinstance(signal[0], numpy.ndarray)

	assert_array_almost_equal(signal, [[0.029941, 0.262425, 0.636352, 0.368483,
		1.263775, 1.253909, 1.099941, 1.176394, 2.038911, 0.193013, 0.367099,
		0.987924, 0.60746 , 0.387713]])


def test_extract_locus_signal_multi():
	bw = pybigtools.open("tests/data/test.bw")
	bw2 = pybigtools.open("tests/data/test2.bw")
	signal = _extract_locus_signal([bw, bw2], 'chr1', 60, 67)

	assert len(signal) == 2
	assert signal[0].shape == (7,)

	assert isinstance(signal, list)
	assert isinstance(signal[0], numpy.ndarray)

	assert_array_almost_equal(signal, [[1.375648, 0.055379, 1.975644, 0.353604,
		0.106355, 0.151777, 1.656914], [0.070536, 1.10706 , 0.195981, 0.396659,
		0.646639, 0.509053, 0.814386]])

	signal = _extract_locus_signal([bw, bw2], 'chr2', 10, 15)

	assert len(signal) == 2
	assert signal[0].shape == (5,)

	assert isinstance(signal, list)
	assert isinstance(signal[0], numpy.ndarray)

	assert_array_almost_equal(signal, [[2.127791, 0.102535, 0.436831, 0.434451,
		0.895565], [0.331885, 1.177428, 0.475862, 0.340272, 0.873283]])


def test_extract_locus_signal_nan():
	bw = pybigtools.open("tests/data/test3.bw")
	signal = _extract_locus_signal([bw], 'chr1', 0, 40)[0]

	assert not numpy.isnan(signal).any()

	nan = 0.0
	assert_array_almost_equal(signal, [
		nan, nan, nan, nan, nan, nan, nan, nan, nan,  6.,  1., nan,  1.,
		2.,  4., 14., 28., nan, 42., 36., 41., 50.,  2.,  1., nan, 20.,
		4., nan, nan, nan, nan, nan,  1., nan,  1., nan,  1.,  5.,  3.,
		nan])


def test_extract_locus_signal_raises_single():
	bw = pybigtools.open("tests/data/test.bw")
	assert_raises(ValueError, _extract_locus_signal, bw, 'chr1', 3, 14)


def test_extract_locus_signal_bigwig_past_chrom_end():
	# pybigtools returns NaN past the end of a chromosome, which becomes 0.
	bw = pybigtools.open("tests/data/test.bw")
	signal = _extract_locus_signal([bw], 'chr1', 281, 286)[0]

	assert signal.shape == (5,)
	assert_array_almost_equal(signal, [1.492186, 0.317799, 0.928506, 0, 0])


def test_extract_locus_signal_bigwig_missing_chrom():
	bw = pybigtools.open("tests/data/test.bw")

	with pytest.warns(TangermemeWarning, match="chr7"):
		signal = _extract_locus_signal([bw], 'chr7', 3, 14)

	assert signal[0].dtype == numpy.float32
	assert_array_almost_equal(signal[0], numpy.zeros(11))


def test_extract_locus_signal_dict(dict_signal):
	signal = _extract_locus_signal([dict_signal], 'chr2', 16, 30)

	assert isinstance(signal, list)
	assert len(signal) == 1
	assert signal[0].dtype == numpy.float32
	assert_array_almost_equal(signal[0], numpy.arange(16, 30) + 10000)


def test_extract_locus_signal_dict_missing_chrom(dict_signal):
	# A chromosome missing from a dict gives zeros and a warning, as it does
	# for a bigWig; it used to raise KeyError.
	del dict_signal['chr2']

	with pytest.warns(TangermemeWarning, match="chr2"):
		signal = _extract_locus_signal([dict_signal], 'chr2', 16, 30)

	assert signal[0].dtype == numpy.float32
	assert_array_almost_equal(signal[0], numpy.zeros(14))


def test_extract_locus_signal_dict_past_array_end(dict_signal):
	# A dict array shorter than the window is padded with zeros, as a bigWig
	# is past the end of a chromosome; it used to come back short.
	signal = _extract_locus_signal([dict_signal], 'chr2', 205, 215)[0]

	assert signal.shape == (10,)
	assert_array_almost_equal(signal, [10205, 10206, 10207, 10208, 10209,
		10210, 0, 0, 0, 0])


##


def test_extract_loci_seq(loci_seqs):
	loci = "tests/data/test.bed"
	fasta = "tests/data/test.fa"

	X = extract_loci(loci, fasta, in_window=10)

	assert X.shape == (5, 4, 10)
	assert X.dtype == torch.int8
	assert X.sum() == 50
	assert_array_almost_equal(X, loci_seqs)


def test_extract_loci_seq_first_n(loci_seqs):
	loci = "tests/data/test.bed"
	fasta = "tests/data/test.fa"

	X = extract_loci(loci, fasta, in_window=10, n_loci=2)

	assert X.shape == (2, 4, 10)
	assert X.dtype == torch.int8
	assert X.sum() == 20
	assert_array_almost_equal(X, loci_seqs[:2])


def test_extract_loci_n_loci_mask_covers_every_locus(loci_seqs):
	# The mask has one entry per input locus. Stopping at n_loci used to leave
	# it only as long as the loci examined, so indexing the loci with it
	# raised.
	loci = "tests/data/test.bed"
	fasta = "tests/data/test.fa"

	X, mask = extract_loci(loci, fasta, in_window=10, n_loci=2,
		return_mask=True)

	assert mask.tolist() == [True, True, False, False, False]
	assert_array_almost_equal(X, loci_seqs[:2])

	# n_loci counts kept loci, so loci dropped along the way do not use up
	# the cap: the exclusion removes the first two and the next two are kept.
	exclusion = pandas.DataFrame({0: ['chr1'], 1: [10], 2: [30]})
	X, mask = extract_loci(loci, fasta, in_window=10, n_loci=2,
		exclusion_lists=exclusion, return_mask=True)

	assert mask.tolist() == [False, False, True, True, False]
	assert_array_almost_equal(X, loci_seqs[2:4])

	X_all = extract_loci(loci, fasta, in_window=10)
	assert_array_almost_equal(X, X_all[mask])

	# A cap larger than the number of loci returns them all.
	X, mask = extract_loci(loci, fasta, in_window=10, n_loci=50,
		return_mask=True)

	assert mask.tolist() == [True] * 5
	assert_array_almost_equal(X, loci_seqs)



def test_extract_loci_seq_out_window(loci_seqs):
	loci = "tests/data/test.bed"
	fasta = "tests/data/test.fa"

	X = extract_loci(loci, fasta, in_window=10, out_window=100)

	assert X.shape == (5, 4, 10)
	assert X.dtype == torch.int8
	assert_array_almost_equal(X, loci_seqs)

	X = extract_loci(loci, fasta, in_window=10, out_window=1000)

	assert X.shape == (5, 4, 10)
	assert X.dtype == torch.int8
	assert_array_almost_equal(X, loci_seqs)


def test_extract_loci_seq_jitter(loci_seqs):
	loci = "tests/data/test.bed"
	fasta = "tests/data/test.fa"

	X = extract_loci(loci, fasta, in_window=10, max_jitter=20)

	assert X.shape == (4, 4, 50)
	assert X.dtype == torch.int8
	assert X.sum() == 200

	X_true = [
		[[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
		  0, 0, 0, 0, 1, 0, 0, 0],
		 [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0,
		  0, 1, 1, 0, 0, 0, 0, 0],
		 [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
		  0, 0, 0, 0, 0, 1, 0, 1],
		 [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
		  1, 0, 0, 1, 0, 0, 1, 0]],

		[[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1,
		  0, 0, 0, 1, 0, 0, 0, 1],
		 [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,
		  1, 0, 0, 0, 1, 0, 1, 0],
		 [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
		  0, 0, 1, 0, 0, 0, 0, 0],
		 [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0,
		  0, 1, 0, 0, 0, 1, 0, 0]],

		[[1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0,
		  0, 0, 1, 0, 1, 0, 0, 0],
		 [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1,
		  0, 1, 0, 0, 0, 0, 1, 0],
		 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
		  0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0,
		  1, 0, 0, 1, 0, 1, 0, 1]],

		[[0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0,
		  0, 1, 0, 0, 1, 0, 0, 1],
		 [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1,
		  0, 0, 1, 0, 0, 1, 0, 0],
		 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		  0, 0, 0, 0, 0, 0, 0, 0],
		 [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0,
		  1, 0, 0, 1, 0, 0, 1, 0]]
		]

	assert_array_almost_equal(X[:, :, 10:-10], X_true)
	assert_array_almost_equal(X[:, :, 20:-20], loci_seqs[1:])


def test_extract_loci_keeps_locus_ending_at_chrom_length():
	# A locus whose extracted window ends exactly at chrom_length is valid
	# (end is exclusive); previously dropped by `end >= chrom_lengths[chrom]`.
	# chr2 in tests/data/test.fa is 211bp long; with in_window=10 and
	# max_jitter=0 the extracted window is [mid-5, mid+5), so mid=206 yields
	# a window of [201, 211), which fits exactly. A window of [202, 212)
	# (mid=207) genuinely runs off the end and should still be dropped.
	loci = pandas.DataFrame({
		'chrom': ['chr2', 'chr2'],
		'start': [201, 202],
		'end':   [211, 212],
	})
	fasta = "tests/data/test.fa"

	X, mask = extract_loci(loci, fasta, in_window=10, return_mask=True)

	# The boundary locus is kept; the one that overruns is dropped.
	assert mask.tolist() == [True, False]
	assert X.shape == (1, 4, 10)


def test_extract_loci_odd_in_window_at_chrom_end():
	# An odd window is [mid - w//2, mid + w//2 + 1). The chromosome-end check
	# used to leave out the extra base, so a window running one base past the
	# end was kept and came back one base short, and stacking it with a full
	# window raised "all input arrays must have the same shape". chr2 is 211
	# bp long; with in_window=11 the windows are [100, 111), [200, 211) and
	# [201, 212), and the last one overruns.
	loci = pandas.DataFrame({
		'chrom': ['chr2', 'chr2', 'chr2'],
		'start': [100, 200, 201],
		'end':   [110, 210, 211],
	})
	fasta = "tests/data/test.fa"

	X, mask = extract_loci(loci, fasta, in_window=11, return_mask=True)

	assert mask.tolist() == [True, True, False]
	assert X.shape == (2, 4, 11)

	chrom = str(pyfaidx.Fasta(fasta)['chr2'])
	assert_array_almost_equal(X[0], one_hot_encode(chrom[100:111]))
	assert_array_almost_equal(X[1], one_hot_encode(chrom[200:211]))


def test_extract_loci_odd_out_window_at_chrom_end():
	# The same check for an odd out_window: the signal windows are [100, 111),
	# [200, 211) and [201, 212), and the last one runs off chr2.
	loci = pandas.DataFrame({
		'chrom': ['chr2', 'chr2', 'chr2'],
		'start': [100, 200, 201],
		'end':   [110, 210, 211],
	})
	fasta = "tests/data/test.fa"
	bw = pybigtools.open("tests/data/test.bw")

	X, y, mask = extract_loci(loci, fasta, ["tests/data/test.bw"],
		in_window=4, out_window=11, return_mask=True)

	assert mask.tolist() == [True, True, False]
	assert X.shape == (2, 4, 4)
	assert y.shape == (2, 1, 11)
	assert_array_almost_equal(y[0, 0], bw.values('chr2', 100, 111))
	assert_array_almost_equal(y[1, 0], bw.values('chr2', 200, 211))


def test_extract_loci_seq_alphabet(loci_seqs):
	loci = "tests/data/test.bed"
	fasta = "tests/data/test.fa"

	X = extract_loci(loci, fasta, alphabet=['A', 'G', 'C', 'T'], in_window=10)

	assert X.shape == (5, 4, 10)
	assert X.dtype == torch.int8
	assert X.sum() == 50
	assert_array_almost_equal(X[:, [0, 2, 1, 3]], loci_seqs)

	X = extract_loci(loci, fasta, alphabet=['A', 'C', 'G', 'T', 'Z'],
		in_window=10)

	assert X.shape == (5, 5, 10)
	assert X.dtype == torch.int8
	assert X.sum() == 50

	expanded_loci_seqs = numpy.concatenate([loci_seqs, numpy.zeros([5, 1, 10])],
		axis=1)

	assert_array_almost_equal(X, expanded_loci_seqs)


def test_extract_loci_seq_alphabet_tuple(loci_seqs):
	# A tuple alphabet used to raise TypeError inside one_hot_encode.
	X = extract_loci("tests/data/test.bed", "tests/data/test.fa",
		alphabet=('A', 'G', 'C', 'T'), ignore=('N',), in_window=10)

	assert X.shape == (5, 4, 10)
	assert_array_almost_equal(X[:, [0, 2, 1, 3]], loci_seqs)


def test_extract_loci_int(loci_seqs):
	loci = "tests/data/test_int.bed"
	fasta = "tests/data/test_int_chroms.fa"

	X = extract_loci(loci, fasta, alphabet=['A', 'G', 'C', 'T'], in_window=10)

	assert X.shape == (5, 4, 10)
	assert X.dtype == torch.int8
	assert X.sum() == 50
	assert_array_almost_equal(X[:, [0, 2, 1, 3]], loci_seqs)

	X = extract_loci(loci, fasta, alphabet=['A', 'C', 'G', 'T', 'Z'],
		in_window=10)

	assert X.shape == (5, 5, 10)
	assert X.dtype == torch.int8
	assert X.sum() == 50

	expanded_loci_seqs = numpy.concatenate([loci_seqs, numpy.zeros([5, 1, 10])],
		axis=1)

	assert_array_almost_equal(X, expanded_loci_seqs)
	

def test_interleave_loci_int_chroms():
	# Genomes whose chromosomes are named "1", "2", etc. would otherwise get
	# read in as integers by pandas and fail to match the string names used
	# by pyfaidx/pybigtools.
	df = _interleave_loci("tests/data/test_int.bed")

	assert list(df['chrom']) == ['1', '1', '1', '2', '2']
	assert list(df['start']) == [10, 80, 140, 25, 35]
	assert list(df['end']) == [30, 100, 160, 55, 65]


def test_interleave_loci_int_chroms_df():
	loci = pandas.read_csv("tests/data/test_int.bed", delimiter='\t',
		index_col=False, header=None)
	df = _interleave_loci(loci)

	assert list(df['chrom']) == ['1', '1', '1', '2', '2']
	assert list(df['start']) == [10, 80, 140, 25, 35]


def test_interleave_loci_int_chroms_filter():
	df = _interleave_loci("tests/data/test_int.bed", chroms=['1'])
	assert list(df['chrom']) == ['1', '1', '1']

	# Integer chromosome names are coerced to strings as well
	df = _interleave_loci("tests/data/test_int.bed", chroms=[1])
	assert list(df['chrom']) == ['1', '1', '1']

	df = _interleave_loci("tests/data/test_int.bed", chroms=['2'])
	assert list(df['chrom']) == ['2', '2']


def test_extract_loci_int_chroms_filter():
	X = extract_loci("tests/data/test_int.bed", "tests/data/test_int_chroms.fa",
		chroms=['1'], in_window=10)
	X0 = extract_loci("tests/data/test.bed", "tests/data/test.fa",
		chroms=['chr1'], in_window=10)

	assert X.shape == (3, 4, 10)
	assert_array_almost_equal(X, X0)


def test_extract_loci_int_chroms_exclusion_lists():
	exclusion = pandas.DataFrame({0: [1], 1: [10], 2: [30]})
	X, mask = extract_loci("tests/data/test_int.bed",
		"tests/data/test_int_chroms.fa", in_window=10, return_mask=True,
		exclusion_lists=[exclusion])

	exclusion0 = pandas.DataFrame({0: ['chr1'], 1: [10], 2: [30]})
	X0, mask0 = extract_loci("tests/data/test.bed", "tests/data/test.fa",
		in_window=10, return_mask=True, exclusion_lists=[exclusion0])

	assert X.shape == (3, 4, 10)
	assert mask.tolist() == [False, False, True, True, True]
	assert_array_almost_equal(X, X0)
	assert mask.tolist() == mask0.tolist()


def test_extract_loci_exclusion_lists_df(tmp_path):
	exclusion = pandas.DataFrame({0: ['chr1'], 1: [10], 2: [30]})

	filename = str(tmp_path / "exclusion.bed")
	exclusion.to_csv(filename, sep='\t', header=False, index=False)

	X0, mask0 = extract_loci("tests/data/test.bed", "tests/data/test.fa",
		in_window=10, return_mask=True, exclusion_lists=[filename])

	assert X0.shape == (3, 4, 10)
	assert mask0.tolist() == [False, False, True, True, True]

	# A bare DataFrame and a list of DataFrames are equivalent to a filename
	for elist in (exclusion, [exclusion], filename):
		X, mask = extract_loci("tests/data/test.bed", "tests/data/test.fa",
			in_window=10, return_mask=True, exclusion_lists=elist)

		assert_array_almost_equal(X, X0)
		assert mask.tolist() == mask0.tolist()


def test_extract_loci_exclusion_lists_multiple(tmp_path):
	a = pandas.DataFrame({0: ['chr1'], 1: [10], 2: [30]})
	b = pandas.DataFrame({0: ['chr2'], 1: [25], 2: [55]})

	filename = str(tmp_path / "a.bed")
	a.to_csv(filename, sep='\t', header=False, index=False)

	X, mask = extract_loci("tests/data/test.bed", "tests/data/test.fa",
		in_window=10, return_mask=True, exclusion_lists=[a, b])

	assert X.shape == (1, 4, 10)
	assert mask.tolist() == [False, False, True, False, False]

	# A filename and a DataFrame can be mixed within the same list
	X0, mask0 = extract_loci("tests/data/test.bed", "tests/data/test.fa",
		in_window=10, return_mask=True, exclusion_lists=[filename, b])

	assert_array_almost_equal(X, X0)
	assert mask.tolist() == mask0.tolist()


def test_extract_loci_exclusion_lists_chrom_not_in_sequences():
	# A genome-wide exclusion list names chromosomes that a FASTA of a subset
	# of the genome lacks. Those regions cannot overlap any locus, so they are
	# skipped; they used to raise KeyError.
	exclusion = pandas.DataFrame({0: ['chrM', 'chr1', 'chrUn_KI270302v1'],
		1: [0, 10, 5], 2: [100, 30, 50]})

	X, mask = extract_loci("tests/data/test.bed", "tests/data/test.fa",
		in_window=10, return_mask=True, exclusion_lists=exclusion)
	X0, mask0 = extract_loci("tests/data/test.bed", "tests/data/test.fa",
		in_window=10, return_mask=True, exclusion_lists=exclusion.iloc[1:2])

	assert mask.tolist() == [False, False, True, True, True]
	assert mask.tolist() == mask0.tolist()
	assert_array_almost_equal(X, X0)


def test_extract_loci_exclusion_lists_end_exclusive():
	# Loci are removed when their window shares a 100 bp chunk with an
	# exclusion region. Ends are exclusive, so a region or a window ending at
	# a multiple of 100 does not reach into the next chunk; both used to.
	# chr7 is 2000 bp and in_window=10 makes each window equal to its locus.
	loci = pandas.DataFrame({
		'chrom': ['chr7'] * 6,
		'start': [85, 90, 95, 195, 200, 305],
		'end':   [95, 100, 105, 205, 210, 315],
	})
	fasta = "tests/data/test.fa"

	# [100, 200) covers only the chunk 100-199.
	exclusion = pandas.DataFrame({0: ['chr7'], 1: [100], 2: [200]})
	X, mask = extract_loci(loci, fasta, in_window=10, return_mask=True,
		exclusion_lists=exclusion)

	assert mask.tolist() == [True, True, False, False, True, True]
	assert X.shape == (4, 4, 10)

	# Overlap is decided by chunk, not by base: [250, 260) and the window
	# [200, 210) share the chunk 200-299 without sharing a base.
	exclusion = pandas.DataFrame({0: ['chr7'], 1: [250], 2: [260]})
	_, mask = extract_loci(loci, fasta, in_window=10, return_mask=True,
		exclusion_lists=exclusion)

	assert mask.tolist() == [True, True, True, False, False, True]

	# A region with end <= start covers no base, wherever it falls.
	for s in (100, 150):
		exclusion = pandas.DataFrame({0: ['chr7'], 1: [s], 2: [s]})
		_, mask = extract_loci(loci, fasta, in_window=10, return_mask=True,
			exclusion_lists=exclusion)

		assert mask.tolist() == [True] * 6


def test_extract_loci_int_chroms_filter_ints():
	X = extract_loci("tests/data/test_int.bed",
		"tests/data/test_int_chroms.fa", chroms=[1], in_window=10)
	X0 = extract_loci("tests/data/test.bed", "tests/data/test.fa",
		chroms=['chr1'], in_window=10)

	assert X.shape == (3, 4, 10)
	assert_array_almost_equal(X, X0)


def test_extract_loci_int_dict_keys(dict_sequences, dict_signal, loci_seqs):
	# Dict keys are coerced to strings like the chromosome names of the loci.
	# Integer keys used to raise KeyError: '1'.
	sequences = {int(chrom[3:]): X for chrom, X in dict_sequences.items()}
	signal = {int(chrom[3:]): y for chrom, y in dict_signal.items()}

	X, y = extract_loci("tests/data/test_int.bed", sequences, [signal],
		in_window=10, out_window=10)
	X0, y0 = extract_loci("tests/data/test.bed", dict_sequences, [dict_signal],
		in_window=10, out_window=10)

	assert_array_almost_equal(X, loci_seqs)
	assert_array_almost_equal(X, X0)
	assert_array_almost_equal(y, y0)


def test_extract_loci_seq_N(loci2_seqs):
	loci = "tests/data/test2.bed"
	fasta = "tests/data/test.fa"

	X = extract_loci(loci, fasta, in_window=10)

	assert X.shape == (9, 4, 10)
	assert X.dtype == torch.int8
	assert X.sum() == 82
	assert_array_almost_equal(X, loci2_seqs)


def test_extract_loci_seq_no_lower(loci2_seqs):
	loci = "tests/data/test2.bed"
	fasta = "tests/data/test.fa"

	X = extract_loci(loci, fasta, alphabet=['A', 'C', 'G', 'T', 'a'],
		in_window=10)

	loci2_seqs = numpy.concatenate([loci2_seqs, numpy.zeros([9, 1, 10])],
		axis=1)

	assert X.shape == (9, 5, 10)
	assert X.dtype == torch.int8
	assert X.sum() == 82
	assert_array_almost_equal(X, loci2_seqs)


def test_extract_loci_seq_interleave(loci_seqs, loci2_seqs):
	loci_seqs = numpy.concatenate([loci_seqs, loci2_seqs])[[0, 5, 1, 6, 2, 7, 3,
		8, 4, 9, 10, 11, 12, 13]]

	X = extract_loci(["tests/data/test.bed", "tests/data/test2.bed"],
		"tests/data/test.fa", in_window=10)

	assert X.shape == (14, 4, 10)
	assert X.dtype == torch.int8
	assert X.sum() == 132
	assert_array_almost_equal(X, loci_seqs)


def test_extract_loci_seq_raises_loci():
	assert_raises(ValueError, extract_loci, [0, 1, 2], "tests/data/test.fa")


def test_extract_loci_seq_raises_sequences():
	assert_raises(pyfaidx.FastaNotFoundError, extract_loci,
		"tests/data/test.bed", "ACGTG")


@pytest.mark.parametrize("n_loci", [0, -1])
def test_extract_loci_raises_n_loci(n_loci):
	# n_loci=0 used to return every locus, since the cap was only checked
	# after a locus had been kept, and a negative cap was never reached.
	with pytest.raises(ValueError, match="n_loci must be at least 1"):
		extract_loci("tests/data/test.bed", "tests/data/test.fa", in_window=10,
			n_loci=n_loci)


@pytest.mark.parametrize("kwargs", [{'min_counts': 1}, {'max_counts': 1},
	{'min_counts': 0, 'max_counts': 1}])
def test_extract_loci_raises_counts_without_signals(kwargs):
	# The counts are measured on signals, so without them the thresholds used
	# to be ignored.
	with pytest.raises(ValueError, match="signals must be provided"):
		extract_loci("tests/data/test.bed", "tests/data/test.fa", in_window=10,
			**kwargs)

	# in_signals are not what the thresholds measure.
	with pytest.raises(ValueError, match="signals must be provided"):
		extract_loci("tests/data/test.bed", "tests/data/test.fa", in_window=10,
			in_signals=["tests/data/test.bw"], **kwargs)


@pytest.mark.parametrize("target_idx", [2, 5, -3])
def test_extract_loci_raises_target_idx(target_idx):
	bw = ["tests/data/test.bw", "tests/data/test2.bw"]

	with pytest.raises(ValueError, match="target_idx"):
		extract_loci("tests/data/test.bed", "tests/data/test.fa", bw,
			in_window=10, out_window=10, min_counts=1, target_idx=target_idx)


@pytest.mark.parametrize("kwargs", [
	{'chroms': ['chr7']},
	{'in_window': 1000},
	{'signals': ["tests/data/test.bw"], 'out_window': 10, 'min_counts': 1e6},
	{'exclusion_lists': pandas.DataFrame({0: ['chr1', 'chr2'], 1: [0, 0],
		2: [300, 300]})},
])
def test_extract_loci_raises_no_loci_remain(kwargs):
	# Filtering away every locus used to fail inside numpy.stack with "need
	# at least one array to stack".
	kwargs = {'in_window': 10, **kwargs}

	with pytest.raises(ValueError, match="No loci remain"):
		extract_loci("tests/data/test.bed", "tests/data/test.fa", **kwargs)


def test_extract_loci_raises_no_loci_remain_odd_window():
	# A lone odd window running one base off the chromosome used to come back
	# one base short, since there was nothing to stack it against.
	loci = pandas.DataFrame({'chrom': ['chr2'], 'start': [201], 'end': [211]})

	with pytest.raises(ValueError, match="No loci remain"):
		extract_loci(loci, "tests/data/test.fa", in_window=11)


def test_extract_loci_raises_no_loci_given():
	loci = pandas.DataFrame({'chrom': [], 'start': [], 'end': []})

	with pytest.raises(ValueError, match="No loci remain"):
		extract_loci(loci, "tests/data/test.fa", in_window=10)


def test_extract_loci_raises_chrom_not_in_sequences():
	# A locus on a chromosome the sequences lack used to raise a bare
	# KeyError. The error names every missing chromosome.
	loci = pandas.DataFrame({
		'chrom': ['chr1', 'chrM', 'chr2', 'chrUn_KI270302v1', 'chrM'],
		'start': [10, 0, 25, 5, 30],
		'end':   [30, 20, 55, 25, 50],
	})

	with pytest.raises(ValueError, match="chrM, chrUn_KI270302v1"):
		extract_loci(loci, "tests/data/test.fa", in_window=10)

	# Filtering them out with chroms is the remedy the message gives.
	X = extract_loci(loci, "tests/data/test.fa", in_window=10,
		chroms=['chr1', 'chr2'])
	assert X.shape == (2, 4, 10)

	# A naming mismatch between the loci and the FASTA is caught the same way.
	with pytest.raises(ValueError, match="1, 2"):
		extract_loci("tests/data/test_int.bed", "tests/data/test.fa",
			in_window=10)


def test_extract_loci_raises_chrom_not_in_sequences_objects(dict_sequences):
	# The same error for sequences the caller opened, which are left open.
	loci = pandas.DataFrame({0: ['chr1', 'chrM'], 1: [10, 0], 2: [30, 20]})
	del dict_sequences['chr1']

	with pytest.raises(ValueError, match="chr1, chrM"):
		extract_loci(loci, dict_sequences, in_window=10)

	fasta = pyfaidx.Fasta("tests/data/test.fa")
	with pytest.raises(ValueError, match="chrM"):
		extract_loci(loci, fasta, in_window=10)

	assert str(fasta['chr1']) == str(pyfaidx.Fasta("tests/data/test.fa")['chr1'])


def test_extract_loci_target_idx_negative():
	# A negative target_idx indexes from the end, as for a list.
	bw = ["tests/data/test.bw", "tests/data/test2.bw"]

	X0, y0 = extract_loci("tests/data/test.bed", "tests/data/test.fa", bw,
		in_window=8, out_window=10, min_counts=7, target_idx=1)
	X, y = extract_loci("tests/data/test.bed", "tests/data/test.fa", bw,
		in_window=8, out_window=10, min_counts=7, target_idx=-1)

	assert_array_almost_equal(X, X0)
	assert_array_almost_equal(y, y0)


def test_extract_loci_does_not_close_user_provided_fasta():
	# When the caller passes a pre-opened pyfaidx.Fasta, extract_loci
	# must not close it; the caller still owns the object and needs to
	# keep using it after the call.
	fasta = pyfaidx.Fasta("tests/data/test.fa")

	extract_loci("tests/data/test.bed", fasta, in_window=10)

	# Accessing a key after the call should still work. Pre-fix this
	# raises ValueError("I/O operation on closed file") from the
	# underlying mmap inside pyfaidx.
	_ = str(fasta[list(fasta.keys())[0]])


def test_extract_loci_pathlike(tmp_path):
	# A pathlib.Path works wherever a filename does. It used to fail with
	# AttributeError for sequences and ValueError for everything else.
	exclusion = pandas.DataFrame({0: ['chr1'], 1: [10], 2: [30]})
	exclusion_file = tmp_path / "exclusion.bed"
	exclusion.to_csv(exclusion_file, sep='\t', header=False, index=False)

	bw = ["tests/data/test.bw", "tests/data/test2.bw"]
	kwargs = {'in_window': 10, 'out_window': 10, 'return_mask': True}

	y0 = extract_loci(["tests/data/test.bed", "tests/data/test2.bed"],
		"tests/data/test.fa", bw, bw, exclusion_lists=str(exclusion_file),
		**kwargs)
	y = extract_loci([pathlib.Path("tests/data/test.bed"),
		pathlib.Path("tests/data/test2.bed")], pathlib.Path("tests/data/test.fa"),
		[pathlib.Path(b) for b in bw], (pathlib.Path(bw[0]), bw[1]),
		exclusion_lists=exclusion_file, **kwargs)

	assert len(y) == len(y0) == 4
	for a, b in zip(y, y0):
		assert_array_almost_equal(a, b)

	# The exclusion list was read: it removes the three chr1 loci in 0-99.
	assert y[-1].tolist().count(False) == 3

	X = extract_loci(pathlib.Path("tests/data/test.bed"), "tests/data/test.fa",
		in_window=10)
	X0 = extract_loci("tests/data/test.bed", "tests/data/test.fa", in_window=10)
	assert_array_almost_equal(X, X0)


@pytest.mark.parametrize("kwargs", [{}, {'as_raw': True},
	{'sequence_always_upper': True}])
def test_extract_loci_pre_opened_fasta(loci2_seqs, kwargs):
	# A pre-opened Fasta gives the same sequences as its filename. With
	# as_raw=True pyfaidx returns strings rather than Sequence objects, which
	# used to raise AttributeError: 'str' object has no attribute 'seq'.
	fasta = pyfaidx.Fasta("tests/data/test.fa", **kwargs)

	X = extract_loci("tests/data/test2.bed", fasta, in_window=10)

	assert X.dtype == torch.int8
	assert_array_almost_equal(X, loci2_seqs)


@pytest.mark.parametrize("layout", ["fasta", "C", "F"])
def test_extract_loci_contiguous(dict_sequences, loci_seqs, loci_signal, layout):
	# Every returned tensor is C-contiguous. X used to keep the layout of the
	# transposed view one_hot_encode returns, strides (4 * L, 1, 4), and so did
	# X from a dict of Fortran-ordered arrays.
	if layout == "fasta":
		sequences = "tests/data/test.fa"
	elif layout == "C":
		sequences = dict_sequences
	else:
		sequences = {chrom: numpy.asfortranarray(X)
			for chrom, X in dict_sequences.items()}

	bw = ["tests/data/test.bw", "tests/data/test2.bw"]
	X, y, controls, mask = extract_loci("tests/data/test.bed", sequences, bw,
		bw, in_window=10, out_window=10, return_mask=True)

	for tensor in (X, y, controls, mask):
		assert tensor.is_contiguous()

	assert X.stride() == (40, 10, 1)
	assert_array_almost_equal(X, loci_seqs)
	assert_array_almost_equal(y, loci_signal)


def test_extract_loci_pre_opened_bigwigs(loci_signal):
	bws = [pybigtools.open("tests/data/test.bw"),
		pybigtools.open("tests/data/test2.bw")]

	X, y, controls = extract_loci("tests/data/test.bed", "tests/data/test.fa",
		bws, bws, in_window=10, out_window=10)

	assert_array_almost_equal(y, loci_signal)
	assert_array_almost_equal(controls, loci_signal)

	# The caller's bigWigs are left open.
	assert_array_almost_equal(bws[0].values('chr1', 15, 25), loci_signal[0][0])


###


def test_extract_loci(loci_signal):
	loci = "tests/data/test.bed"
	bw = ["tests/data/test.bw", "tests/data/test2.bw"]
	fasta = "tests/data/test.fa"

	X, y = extract_loci(loci, fasta, bw, in_window=10, out_window=10)

	assert X.shape == (5, 4, 10)
	assert X.dtype == torch.int8
	assert X.sum() == 50

	assert y.shape == (5, 2, 10)
	assert y.dtype == torch.float32
	assert_array_almost_equal([abs(y).sum()], [84.0259], 4)
	assert_array_almost_equal(y, loci_signal)


def test_extract_loci_widths(loci_signal):
	loci = "tests/data/test.bed"
	bw = ["tests/data/test.bw", "tests/data/test2.bw"]
	fasta = "tests/data/test.fa"

	X, y = extract_loci(loci, fasta, bw, in_window=8, out_window=14)

	assert X.shape == (5, 4, 8)
	assert X.dtype == torch.int8
	assert X.sum() == 40

	assert y.shape == (5, 2, 14)
	assert y.dtype == torch.float32
	assert_array_almost_equal([abs(y).sum()], [116.9134], 4)


def test_extract_loci_out_off(loci_signal):
	loci = "tests/data/test.bed"
	bw = ["tests/data/test.bw", "tests/data/test2.bw"]
	fasta = "tests/data/test.fa"

	X, y = extract_loci(loci, fasta, bw, in_window=8, out_window=42)

	assert X.shape == (4, 4, 8)
	assert X.dtype == torch.int8
	assert X.sum() == 32

	assert y.shape == (4, 2, 42)
	assert y.dtype == torch.float32
	assert_array_almost_equal([abs(y).sum()], [276.9355], 4)


def test_extract_loci_min_counts(loci_signal):
	loci = "tests/data/test.bed"
	bw = ["tests/data/test.bw", "tests/data/test2.bw"]
	fasta = "tests/data/test.fa"

	X, y = extract_loci(loci, fasta, bw, in_window=8, out_window=10,
		min_counts=10, target_idx=0)

	assert X.shape == (2, 4, 8)
	assert X.dtype == torch.int8
	assert X.sum() == 16

	assert y.shape == (2, 2, 10)
	assert y.dtype == torch.float32
	assert all(y[:, 0].sum(axis=-1) > 10)
	assert_array_almost_equal([abs(y).sum()], [36.2247], 4)


	X, y = extract_loci(loci, fasta, bw, in_window=8, out_window=10,
		min_counts=7, target_idx=1)

	assert X.shape == (4, 4, 8)
	assert X.dtype == torch.int8
	assert X.sum() == 32

	assert y.shape == (4, 2, 10)
	assert y.dtype == torch.float32
	assert all(y[:, 1].sum(axis=-1) > 7)
	assert_array_almost_equal([abs(y).sum()], [66.8475], 4)


def test_extract_loci_max_counts(loci_signal):
	loci = "tests/data/test.bed"
	bw = ["tests/data/test.bw", "tests/data/test2.bw"]
	fasta = "tests/data/test.fa"

	X, y = extract_loci(loci, fasta, bw, in_window=8, out_window=10,
		max_counts=10, target_idx=0)

	assert X.shape == (3, 4, 8)
	assert X.dtype == torch.int8
	assert X.sum() == 24

	assert y.shape == (3, 2, 10)
	assert y.dtype == torch.float32
	assert all(y[:, 0].sum(axis=-1) < 10)
	assert_array_almost_equal([abs(y).sum()], [47.8011], 4)


	X, y = extract_loci(loci, fasta, bw, in_window=8, out_window=10,
		max_counts=7, target_idx=1)

	assert X.shape == (1, 4, 8)
	assert X.dtype == torch.int8
	assert X.sum() == 8

	assert y.shape == (1, 2, 10)
	assert y.dtype == torch.float32
	assert all(y[:, 1].sum(axis=-1) < 10)
	assert_array_almost_equal([abs(y).sum()], [17.1784], 4)


def test_extract_loci_nans():
	loci = "tests/data/test.bed"
	bw = ["tests/data/test3.bw"]
	fasta = "tests/data/test.fa"

	X, y = extract_loci(loci, fasta, bw, in_window=8, out_window=10)

	assert X.shape == (5, 4, 8)
	assert X.dtype == torch.int8
	assert X.sum() == 40

	assert y.shape == (5, 1, 10)
	assert not numpy.isnan(y).any()
	assert_array_almost_equal([abs(y).sum()], [214.0], 4)
	assert_array_almost_equal(y, [
		[[14., 28.,  0., 42., 36., 41., 50.,  2.,  1.,  0.]],
		[[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]],
		[[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]],
		[[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]],
		[[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]]])


###


def test_extract_loci_controls(loci_signal):
	loci = "tests/data/test.bed"
	bw = ["tests/data/test.bw", "tests/data/test2.bw"]
	controls = ["tests/data/test.bw", "tests/data/test2.bw"]
	fasta = "tests/data/test.fa"

	X, y, controls = extract_loci(loci, fasta, bw, controls, in_window=16,
		out_window=10)

	assert X.shape == (5, 4, 16)
	assert X.dtype == torch.int8
	assert X.sum() == 80

	assert y.shape == (5, 2, 10)
	assert y.dtype == torch.float32
	assert_array_almost_equal([abs(y).sum()], [84.0259], 4)
	assert_array_almost_equal(y, loci_signal)

	assert controls.shape == (5, 2, 16)
	assert controls.dtype == torch.float32
	assert_array_almost_equal([abs(controls).sum()], [133.395], 4)
	assert_array_almost_equal(controls[:, :, 3:-3], loci_signal)


def test_extract_loci_in_signals_only(loci_signal):
	# Without signals no output window is extracted, so out_window must not
	# decide which loci fit. It used to: the default out_window of 1000 ran
	# every locus off these short chromosomes.
	loci = "tests/data/test.bed"
	controls = ["tests/data/test.bw", "tests/data/test2.bw"]
	fasta = "tests/data/test.fa"

	X, controls_, mask = extract_loci(loci, fasta, in_signals=controls,
		in_window=10, return_mask=True)

	assert X.shape == (5, 4, 10)
	assert controls_.shape == (5, 2, 10)
	assert controls_.dtype == torch.float32
	assert mask.tolist() == [True] * 5
	assert_array_almost_equal(controls_, loci_signal)

	X0 = extract_loci(loci, fasta, in_window=10)
	assert_array_almost_equal(X, X0)


def test_extract_loci_dict_signals_missing_chrom_and_short(dict_signal):
	# A dict signal missing a chromosome, or with an array shorter than the
	# FASTA's chromosome, used to raise KeyError or a shape mismatch in
	# numpy.stack. Both are now zero-filled, as for a bigWig.
	del dict_signal['chr1']
	dict_signal['chr2'] = dict_signal['chr2'][:205]

	loci = pandas.DataFrame({'chrom': ['chr1', 'chr2', 'chr2'],
		'start': [100, 100, 200], 'end': [110, 110, 210]})

	with pytest.warns(TangermemeWarning, match="chr1"):
		X, y = extract_loci(loci, "tests/data/test.fa", [dict_signal],
			in_window=4, out_window=11)

	assert y.shape == (3, 1, 11)
	assert_array_almost_equal(y[0, 0], numpy.zeros(11))
	assert_array_almost_equal(y[1, 0], numpy.arange(100, 111) + 10000)
	assert_array_almost_equal(y[2, 0], [10200, 10201, 10202, 10203, 10204,
		0, 0, 0, 0, 0, 0])


###


@pytest.mark.parametrize("in_window", [8, 9])
@pytest.mark.parametrize("out_window", [6, 7, 12, 13])
@pytest.mark.parametrize("max_jitter", [0, 3])
def test_extract_loci_window_coordinates(dict_sequences, dict_signal,
	in_window, out_window, max_jitter):
	# Each value of dict_signal is its own coordinate plus a per-chromosome
	# offset, so the extracted signals give the exact bases of every window:
	# [mid - w//2 - j, mid + w//2 + w%2 + j) with mid = start + (end-start)//2,
	# here for loci of odd and even length and a one-base locus.
	loci = pandas.DataFrame({'chrom': ['chr7', 'chr7', 'chr1', 'chr7'],
		'start': [100, 300, 50, 1001], 'end': [120, 331, 51, 1002]})
	offsets = {'chr1': 0, 'chr7': 60000}

	X, y, y_in = extract_loci(loci, dict_sequences, [dict_signal],
		[dict_signal], in_window=in_window, out_window=out_window,
		max_jitter=max_jitter)
	X_fasta = extract_loci(loci, "tests/data/test.fa", in_window=in_window,
		max_jitter=max_jitter)

	assert X.shape == (4, 4, in_window + 2 * max_jitter)
	assert y.shape == (4, 1, out_window + 2 * max_jitter)
	assert y_in.shape == (4, 1, in_window + 2 * max_jitter)

	for i, (chrom, start, end) in enumerate(loci.values):
		mid = start + (end - start) // 2

		s = mid - out_window // 2 - max_jitter
		e = mid + out_window // 2 + out_window % 2 + max_jitter
		assert_array_almost_equal(y[i, 0], numpy.arange(s, e) + offsets[chrom])

		s = mid - in_window // 2 - max_jitter
		e = mid + in_window // 2 + in_window % 2 + max_jitter
		assert_array_almost_equal(y_in[i, 0], numpy.arange(s, e) +
			offsets[chrom])
		assert_array_almost_equal(X[i], dict_sequences[chrom][:, s:e])
		assert_array_almost_equal(X_fasta[i], dict_sequences[chrom][:, s:e])


def test_extract_loci_window_at_chrom_start():
	# A window starting at base 0 fits and one starting at -1 does not. The
	# window checked is the union of the input and output windows, jitter
	# included.
	# The third locus, [50, 60), fits in every case.
	fasta = "tests/data/test.fa"
	loci = pandas.DataFrame({'chrom': ['chr1', 'chr1', 'chr1'],
		'start': [0, 0, 50], 'end': [10, 9, 60]})

	# in_window=10: [0, 10) and [-1, 9)
	X, mask = extract_loci(loci, fasta, in_window=10, return_mask=True)
	assert mask.tolist() == [True, False, True]
	assert_array_almost_equal(X[0], one_hot_encode(str(
		pyfaidx.Fasta(fasta)['chr1'][0:10])))

	# max_jitter=1 moves both windows one base left: [-1, 11) and [-2, 10)
	_, mask = extract_loci(loci, fasta, in_window=10, max_jitter=1,
		return_mask=True)
	assert mask.tolist() == [False, False, True]

	# An output window wider than the input window decides: with
	# out_window=12, [-1, 11) and [-2, 10)
	_, _, mask = extract_loci(loci, fasta, ["tests/data/test.bw"],
		in_window=6, out_window=12, return_mask=True)
	assert mask.tolist() == [False, False, True]

	_, _, mask = extract_loci(loci, fasta, ["tests/data/test.bw"],
		in_window=6, out_window=10, return_mask=True)
	assert mask.tolist() == [True, False, True]


@pytest.mark.parametrize("dtype", ['int32', 'uint32', 'uint64', 'Int64',
	object])
def test_extract_loci_coordinate_dtypes(dtype):
	# Coordinates of any integer dtype give what int64 coordinates do: a locus
	# off each end of its chromosome, one in an excluded chunk, and two kept.
	fasta = "tests/data/test.fa"
	loci = pandas.DataFrame({'chrom': ['chr1', 'chr1', 'chr2', 'chr4', 'chr1'],
		'start': [0, 50, 201, 100, 250], 'end': [8, 61, 211, 111, 261]})
	exclusion = pandas.DataFrame({'chrom': ['chr1'], 'start': [260],
		'end': [270]})

	X0, y0, mask0 = extract_loci(loci, fasta, ["tests/data/test.bw"],
		in_window=11, out_window=6, exclusion_lists=exclusion, return_mask=True)
	assert mask0.tolist() == [False, True, False, True, False]

	loci = loci.astype({'start': dtype, 'end': dtype})
	X, y, mask = extract_loci(loci, fasta, ["tests/data/test.bw"],
		in_window=11, out_window=6, exclusion_lists=exclusion, return_mask=True)

	assert torch.equal(X, X0)
	assert torch.equal(y, y0)
	assert torch.equal(mask, mask0)


@pytest.mark.parametrize("dtype", [numpy.int64, numpy.uint64, object])
def test_extract_loci_coordinates_near_int64_max(dtype):
	# A window that ends past 2**63 falls off the end of its chromosome; it
	# does not wrap around to a negative end that passes the check.
	fasta = "tests/data/test.fa"
	loci = pandas.DataFrame({'chrom': ['chr4', 'chr4'],
		'start': numpy.array([100, 2 ** 63 - 10], dtype=dtype),
		'end': numpy.array([111, 2 ** 63 - 1], dtype=dtype)})

	X, mask = extract_loci(loci, fasta, in_window=11, return_mask=True)
	assert mask.tolist() == [True, False]
	assert X.shape == (1, 4, 11)


def test_extract_loci_raises_float_coordinates():
	loci = pandas.DataFrame({'chrom': ['chr4'], 'start': [100.0],
		'end': [111.0]})
	assert_raises(TypeError, extract_loci, loci, "tests/data/test.fa",
		in_window=11)


def test_extract_loci_summits(dict_signal):
	# With summits=True the windows are centered on start + summit, the tenth
	# column, rather than on the middle of the locus.
	loci = pandas.DataFrame([
		['chr7', 100, 200, '.', 0, '.', 0, 0, 0, 30],
		['chr7', 100, 201, '.', 0, '.', 0, 0, 0, 0],
		['chr7', 500, 600, '.', 0, '.', 0, 0, 0, 100],
	])

	X, y = extract_loci(loci, "tests/data/test.fa", in_signals=[dict_signal],
		in_window=10, summits=True)

	assert y.shape == (3, 1, 10)
	for i, center in enumerate([130, 100, 600]):
		assert_array_almost_equal(y[i, 0], numpy.arange(center - 5,
			center + 5) + 60000)

	# Without summits the same loci are centered on their middles.
	X, y = extract_loci(loci, "tests/data/test.fa", in_signals=[dict_signal],
		in_window=10)

	for i, center in enumerate([150, 150, 550]):
		assert_array_almost_equal(y[i, 0], numpy.arange(center - 5,
			center + 5) + 60000)


def test_extract_loci_summits_file(loci2_seqs):
	# The summits of test2.bed10, applied by hand, give the same sequences.
	loci = pandas.read_csv("tests/data/test2.bed10", sep='\t', header=None)
	centers = loci[1] + loci[9]
	centered = pandas.DataFrame({0: loci[0], 1: centers - 10, 2: centers + 10})

	X = extract_loci("tests/data/test2.bed10", "tests/data/test.fa",
		in_window=10, summits=True)
	X0 = extract_loci(centered, "tests/data/test.fa", in_window=10)

	assert X.shape == (9, 4, 10)
	assert_array_almost_equal(X, X0)


def test_extract_loci_order(dict_signal):
	# Rows come back in the order of the loci, including when the loci
	# alternate between chromosomes or repeat, and a list of loci is
	# interleaved round-robin with the remainder of the longest appended.
	a = pandas.DataFrame({0: ['chr7', 'chr1', 'chr7', 'chr1', 'chr2'],
		1: [500, 100, 200, 100, 50], 2: [510, 110, 210, 110, 60]})
	b = pandas.DataFrame({0: ['chr2', 'chr7'], 1: [150, 900], 2: [160, 910]})

	_, y = extract_loci([a, b], "tests/data/test.fa", [dict_signal],
		in_window=4, out_window=2)

	centers = [60505, 10155, 105, 60905, 60205, 105, 10055]
	assert_array_almost_equal(y[:, 0, 0], [c - 1 for c in centers])
	assert_array_almost_equal(y[:, 0, 1], centers)


@pytest.mark.parametrize("min_counts, max_counts, kept", [
	(None, None, [True, True, True]),
	(198, None, [False, True, True]),
	(199, None, [False, False, True]),
	(None, 198, [True, True, False]),
	(None, 197, [True, False, False]),
	(198, 198, [False, True, False]),
	(78, 318, [True, True, True]),
])
def test_extract_loci_counts_thresholds(dict_signal, min_counts, max_counts,
	kept):
	# With out_window=4 the window around mid m on chr1 sums to
	# (m-2) + (m-1) + m + (m+1) = 4m - 2, so the mids 20, 50 and 80 sum to
	# 78, 198 and 318. Both thresholds are inclusive.
	loci = pandas.DataFrame({0: ['chr1'] * 3, 1: [15, 45, 75],
		2: [25, 55, 85]})

	X, y, y_in, mask = extract_loci(loci, "tests/data/test.fa", [dict_signal],
		[dict_signal], in_window=6, out_window=4, min_counts=min_counts,
		max_counts=max_counts, return_mask=True)

	assert mask.tolist() == kept
	assert X.shape == (sum(kept), 4, 6)
	assert y_in.shape == (sum(kept), 1, 6)
	assert_array_almost_equal(y.sum(axis=(1, 2)),
		[s for s, k in zip([78, 198, 318], kept) if k])


def test_extract_loci_counts_target_idx(dict_signal):
	# Only signals[target_idx] is thresholded; the other signals are returned
	# but do not decide.
	zeros = {chrom: numpy.zeros_like(y) for chrom, y in dict_signal.items()}
	loci = pandas.DataFrame({0: ['chr1'] * 3, 1: [15, 45, 75],
		2: [25, 55, 85]})

	_, y, mask = extract_loci(loci, "tests/data/test.fa", [dict_signal, zeros],
		in_window=6, out_window=4, min_counts=199, target_idx=0,
		return_mask=True)

	assert mask.tolist() == [False, False, True]
	assert y.shape == (1, 2, 4)
	assert_array_almost_equal(y[0, 1], numpy.zeros(4))

	_, y, mask = extract_loci(loci, "tests/data/test.fa", [dict_signal, zeros],
		in_window=6, out_window=4, max_counts=0, target_idx=1,
		return_mask=True)

	assert mask.tolist() == [True, True, True]

	with pytest.raises(ValueError, match="No loci remain"):
		extract_loci(loci, "tests/data/test.fa", [dict_signal, zeros],
			in_window=6, out_window=4, min_counts=1, target_idx=1)


def test_extract_loci_exclusion_uses_union_window():
	# The exclusion check covers the union of the windows, jitter included.
	# [100, 200) covers the chunk 100-199; the locus's in-window [205, 215)
	# does not reach it, but widened by a jitter of 10, or an out_window of
	# 30, it does.
	loci = pandas.DataFrame({0: ['chr7'], 1: [205], 2: [215]})
	exclusion = pandas.DataFrame({0: ['chr7'], 1: [100], 2: [200]})
	fasta = "tests/data/test.fa"

	X = extract_loci(loci, fasta, in_window=10, exclusion_lists=exclusion)
	assert X.shape == (1, 4, 10)

	with pytest.raises(ValueError, match="No loci remain"):
		extract_loci(loci, fasta, in_window=10, max_jitter=10,
			exclusion_lists=exclusion)

	with pytest.raises(ValueError, match="No loci remain"):
		extract_loci(loci, fasta, ["tests/data/test.bw"], in_window=10,
			out_window=30, exclusion_lists=exclusion)


def test_extract_loci_bigwig_missing_chrom_warns():
	# test3.bw only has chr1, so the two chr2 loci warn and are zero-filled.
	with pytest.warns(TangermemeWarning, match="chr2") as record:
		_, y = extract_loci("tests/data/test.bed", "tests/data/test.fa",
			["tests/data/test3.bw"], in_window=8, out_window=10)

	assert sum(issubclass(w.category, TangermemeWarning) for w in record) == 2
	assert_array_almost_equal(y[3:], numpy.zeros((2, 1, 10)))


def test_extract_loci_nan_and_inf(dict_signal):
	# NaN becomes 0 and an infinity the largest finite float32 of its sign,
	# which is what numpy.nan_to_num does, in signals and in_signals alike.
	dict_signal['chr1'][18:21] = [numpy.nan, numpy.inf, -numpy.inf]
	loci = pandas.DataFrame({0: ['chr1'], 1: [10], 2: [30]})
	big = numpy.finfo(numpy.float32).max

	_, y, y_in = extract_loci(loci, "tests/data/test.fa", [dict_signal],
		[dict_signal], in_window=10, out_window=10)

	for values in (y[0, 0], y_in[0, 0]):
		assert values.dtype == torch.float32
		assert_array_almost_equal(values, [15, 16, 17, 0, big, -big, 21, 22,
			23, 24])


@pytest.fixture
def inf_bigwig(tmp_path):
	# chr1 is 50 bp here but 284 bp in test.fa, so windows past base 50 read
	# NaN from pybigtools. A NaN interval would be read back as missing, so
	# the past-the-end positions are the only NaN.
	filename = str(tmp_path / "inf.bw")
	bw = pybigtools.open(filename, "w")
	bw.write({'chr1': 50, 'chr2': 211}, [('chr1', 0, 10, 1.0),
		('chr1', 10, 12, numpy.inf), ('chr1', 12, 14, -numpy.inf),
		('chr1', 14, 50, 2.0), ('chr2', 0, 211, 0.5)])
	return filename


def test_extract_loci_bigwig_nan_and_inf(inf_bigwig):
	# bigWigs are written into a preallocated output and nan_to_num runs once
	# at the end: NaN still becomes 0 and an infinity the largest finite
	# float32 of its sign, in signals and in_signals, as in the per-locus
	# values of _extract_locus_signal.
	loci = pandas.DataFrame({0: ['chr1', 'chr1'], 1: [7, 43], 2: [17, 53]})
	big = numpy.finfo(numpy.float32).max

	_, y, y_in = extract_loci(loci, "tests/data/test.fa", [inf_bigwig],
		[inf_bigwig], in_window=6, out_window=10)

	for tensor in (y, y_in):
		assert tensor.dtype == torch.float32
		assert tensor.is_contiguous()

	assert_array_almost_equal(y[:, 0], [[1, 1, 1, big, big, -big, -big, 2, 2,
		2], [2, 2, 2, 2, 2, 2, 2, 0, 0, 0]])
	assert_array_almost_equal(y_in[:, 0], [[1, big, big, -big, -big, 2],
		[2, 2, 2, 2, 2, 0]])

	bw = pybigtools.open(inf_bigwig)
	for i, (start, end) in enumerate([(7, 17), (43, 53)]):
		expected = _extract_locus_signal([bw], 'chr1', start, end)[0]
		assert y[i, 0].numpy().tobytes() == expected.tobytes()


@pytest.mark.parametrize("kwargs, kept", [
	({'min_counts': 14}, [True, False, True]),
	({'min_counts': 15}, [False, False, True]),
	({'max_counts': 14}, [True, True, False]),
	({'max_counts': 13}, [False, True, False]),
])
def test_extract_loci_bigwig_counts_after_nan_to_num(inf_bigwig, kwargs,
	kept):
	# The window around base 48 on chr1 runs past the end of the bigWig, so
	# its counts are 7 * 2 plus three NaN, which count as 0. A NaN left in
	# the sum would compare False against both thresholds and keep it.
	loci = pandas.DataFrame({0: ['chr1', 'chr2', 'chr1'], 1: [43, 95, 25],
		2: [53, 105, 35]})

	X, y, y_in, mask = extract_loci(loci, "tests/data/test.fa", [inf_bigwig],
		[inf_bigwig], in_window=6, out_window=10, return_mask=True, **kwargs)

	assert mask.tolist() == kept
	assert X.shape == (sum(kept), 4, 6)
	assert y_in.shape == (sum(kept), 1, 6)
	assert_array_almost_equal(y.sum(axis=(1, 2)),
		[s for s, k in zip([14, 5, 20], kept) if k])


@pytest.mark.parametrize("n_loci", [None, 1, 3, 100])
def test_extract_loci_bigwig_trimmed_rows(n_loci):
	# The preallocated outputs hold a row for every locus that could be kept,
	# and are cut to the kept loci. A locus rejected by min_counts is
	# overwritten by the next one, so each row is the values of one kept locus.
	bw = ["tests/data/test.bw", "tests/data/test2.bw"]
	X, y, y_in, mask = extract_loci("tests/data/test.bed", "tests/data/test.fa",
		bw, bw[:1], in_window=8, out_window=10, min_counts=7, target_idx=1,
		n_loci=n_loci, return_mask=True)

	n = 4 if n_loci is None else min(4, n_loci)
	assert y.shape == (n, 2, 10)
	assert y_in.shape == (n, 1, 8)
	for tensor in (X, y, y_in):
		assert tensor.is_contiguous()

	bws = [pybigtools.open(name) for name in bw]
	loci = pandas.read_csv("tests/data/test.bed", sep="\t", header=None)
	kept = loci[mask.numpy()]
	for i, (chrom, start, end) in enumerate(kept.values):
		mid = start + (end - start) // 2
		expected = numpy.stack(_extract_locus_signal(bws, chrom, mid - 5,
			mid + 5))
		assert y[i].numpy().tobytes() == expected.tobytes()

		expected = numpy.stack(_extract_locus_signal(bws[:1], chrom, mid - 4,
			mid + 4))
		assert y_in[i].numpy().tobytes() == expected.tobytes()


def test_extract_loci_does_not_modify_inputs(dict_sequences, dict_signal):
	# The loci, sequences, signals and exclusion lists are left unchanged, and
	# the returned tensors do not share memory with them.
	dict_signal['chr1'][18] = numpy.nan
	loci = pandas.DataFrame({0: ['chr1', 'chr2'], 1: [10, 25], 2: [30, 55],
		3: ['a', 'b']})
	exclusion = pandas.DataFrame({0: ['chr7', 'chrM'], 1: [0, 0], 2: [10, 10]})

	loci0 = loci.copy()
	exclusion0 = exclusion.copy()
	sequences0 = {chrom: X.copy() for chrom, X in dict_sequences.items()}
	signal0 = {chrom: y.copy() for chrom, y in dict_signal.items()}

	X, y, y_in = extract_loci(loci, dict_sequences, [dict_signal],
		[dict_signal], in_window=10, out_window=10, exclusion_lists=exclusion)

	X.fill_(1)
	y.fill_(1)
	y_in.fill_(1)

	assert loci.equals(loci0)
	assert exclusion.equals(exclusion0)
	for chrom in sequences0:
		assert numpy.array_equal(dict_sequences[chrom], sequences0[chrom])
		assert numpy.array_equal(dict_signal[chrom], signal0[chrom],
			equal_nan=True)


def test_extract_loci_return_structure():
	# One output is returned bare; more come back as a list in the order X,
	# signals, in_signals, mask. The shapes differ, so each element can be
	# identified: 1 signal of out_window 6 and 2 in_signals of in_window 10.
	loci = "tests/data/test.bed"
	fasta = "tests/data/test.fa"
	bw = ["tests/data/test.bw"]
	controls = ["tests/data/test.bw", "tests/data/test2.bw"]

	X = extract_loci(loci, fasta, in_window=10)
	assert isinstance(X, torch.Tensor)
	assert X.shape == (5, 4, 10)
	assert X.dtype == torch.int8

	cases = [
		({'signals': bw}, [(5, 1, 6)]),
		({'in_signals': controls}, [(5, 2, 10)]),
		({'return_mask': True}, [(5,)]),
		({'signals': bw, 'in_signals': controls}, [(5, 1, 6), (5, 2, 10)]),
		({'signals': bw, 'return_mask': True}, [(5, 1, 6), (5,)]),
		({'in_signals': controls, 'return_mask': True}, [(5, 2, 10), (5,)]),
		({'signals': bw, 'in_signals': controls, 'return_mask': True},
			[(5, 1, 6), (5, 2, 10), (5,)]),
	]

	for kwargs, shapes in cases:
		result = extract_loci(loci, fasta, in_window=10, out_window=6,
			**kwargs)

		assert type(result) is list
		assert len(result) == len(shapes) + 1
		assert_array_almost_equal(result[0], X)

		for tensor, shape in zip(result[1:], shapes):
			assert isinstance(tensor, torch.Tensor)
			assert tuple(tensor.shape) == shape
			assert tensor.dtype == (torch.bool if len(shape) == 1 else
				torch.float32)


def test_extract_loci_verbose(loci_signal):
	X0, y0 = extract_loci("tests/data/test.bed", "tests/data/test.fa",
		["tests/data/test.bw", "tests/data/test2.bw"], in_window=10,
		out_window=10)
	X, y = extract_loci("tests/data/test.bed", "tests/data/test.fa",
		["tests/data/test.bw", "tests/data/test2.bw"], in_window=10,
		out_window=10, verbose=True)

	assert_array_almost_equal(X, X0)
	assert_array_almost_equal(y, loci_signal)


def test_extract_loci_raises_character_not_in_alphabet():
	# chr5 has a Z at base 25. A window over it raises unless Z is ignored,
	# and a window elsewhere on chr5 is unaffected.
	loci = pandas.DataFrame({0: ['chr5', 'chr5'], 1: [20, 50], 2: [30, 60]})
	fasta = "tests/data/test.fa"

	with pytest.raises(ValueError, match="Encountered character"):
		extract_loci(loci, fasta, in_window=10)

	X = extract_loci(loci.iloc[1:], fasta, in_window=10)
	assert X.shape == (1, 4, 10)

	X = extract_loci(loci, fasta, in_window=10, ignore=['N', 'Z'])
	assert X.shape == (2, 4, 10)
	assert X[0, :, 5].tolist() == [0, 0, 0, 0]
	assert X[0].sum() == 9

	X = extract_loci(loci, fasta, in_window=10, alphabet=['A', 'C', 'G', 'T',
		'Z'])
	assert X.shape == (2, 5, 10)
	assert X[0, :, 5].tolist() == [0, 0, 0, 0, 1]


def test_extract_loci_tuples(loci_seqs, loci2_seqs):
	X = extract_loci(("tests/data/test.bed", "tests/data/test2.bed"),
		"tests/data/test.fa", in_window=10, chroms=('chr1',))
	X0 = extract_loci(["tests/data/test.bed", "tests/data/test2.bed"],
		"tests/data/test.fa", in_window=10, chroms=['chr1'])

	assert X.shape == (5, 4, 10)
	assert_array_almost_equal(X, X0)
	assert_array_almost_equal(X, [loci_seqs[0], loci2_seqs[0], loci_seqs[1],
		loci2_seqs[1], loci_seqs[2]])


def test_extract_loci_dict_sequences(dict_sequences, loci_seqs, tmp_path):
	# A dict of one-hot arrays gives the sequences of the FASTA it was made
	# from, in the dtype of its arrays: numpy arrays, memory maps and torch
	# tensors alike.
	X = extract_loci("tests/data/test.bed", dict_sequences, in_window=10)
	assert X.dtype == torch.int8
	assert_array_almost_equal(X, loci_seqs)

	floats = {chrom: X.astype(numpy.float32) for chrom, X in
		dict_sequences.items()}
	X = extract_loci("tests/data/test.bed", floats, in_window=10)
	assert X.dtype == torch.float32
	assert_array_almost_equal(X, loci_seqs)

	memmaps = {}
	for chrom, X in dict_sequences.items():
		numpy.save(tmp_path / f"{chrom}.npy", X)
		memmaps[chrom] = numpy.load(tmp_path / f"{chrom}.npy", mmap_mode='r')

	X = extract_loci("tests/data/test.bed", memmaps, in_window=10)
	assert X.dtype == torch.int8
	assert_array_almost_equal(X, loci_seqs)

	tensors = {chrom: torch.from_numpy(X) for chrom, X in
		dict_sequences.items()}
	X = extract_loci("tests/data/test.bed", tensors, in_window=10)
	assert X.dtype == torch.int8
	assert_array_almost_equal(X, loci_seqs)


def test_extract_loci_closes_fasta_it_opens(monkeypatch):
	# A FASTA opened from a filename is closed before returning, including
	# when the loci are rejected; a caller's Fasta is left open.
	closed = []
	close = pyfaidx.Fasta.close

	def recording_close(self):
		closed.append(id(self))
		close(self)

	monkeypatch.setattr(pyfaidx.Fasta, "close", recording_close)

	extract_loci("tests/data/test.bed", "tests/data/test.fa", in_window=10)
	assert len(closed) == 1

	loci = pandas.DataFrame({0: ['chrM'], 1: [0], 2: [10]})
	with pytest.raises(ValueError, match="chrM"):
		extract_loci(loci, "tests/data/test.fa", in_window=10)
	assert len(closed) == 2

	with pytest.raises(ValueError, match="No loci remain"):
		extract_loci("tests/data/test.bed", "tests/data/test.fa",
			in_window=1000)
	assert len(closed) == 3

	fasta = pyfaidx.Fasta("tests/data/test.fa")
	extract_loci("tests/data/test.bed", fasta, in_window=10)
	assert len(closed) == 3


###


# The comparison below runs extract_loci and a reference implementation on a
# random genome over a grid of settings and requires identical outputs. The
# reference is written for clarity rather than speed and shares no code with
# tangermeme.io. It phrases each rule differently from the implementation: a
# window is a start and a length, and the 100 bp chunks a region touches are
# the chunks of the bases it covers.

REFERENCE_CHROMS = {'chr1': 1500, 'chr2': 777, 'chrX': 2301, '3': 450}


def _reference_values(track, chrom, start, end):
	values = numpy.zeros(end - start, dtype=numpy.float32)
	if chrom in track:
		array = track[chrom][start:end]
		values[:len(array)] = array

	return numpy.nan_to_num(values)


def _reference_chunks(start, end):
	return {base // 100 for base in range(start, end)}


def _reference_extract_loci(loci, genome, signals, in_signals, in_window,
	out_window, max_jitter, chroms=None, min_counts=None, max_counts=None,
	target_idx=0, n_loci=None, summits=False, exclusion_lists=None):
	tables = []
	for df in loci:
		table = []
		for row in df.itertuples(index=False):
			chrom, start, end = str(row[0]), int(row[1]), int(row[2])

			if summits:
				center = start + int(row[9])
			else:
				center = start + (end - start) // 2

			if chroms is None or chrom in chroms:
				table.append((chrom, center))

		tables.append(table)

	interleaved = []
	for i in range(max(len(table) for table in tables)):
		for table in tables:
			if i < len(table):
				interleaved.append(table[i])

	excluded = set()
	if exclusion_lists is not None:
		for chrom, start, end in exclusion_lists.itertuples(index=False):
			for chunk in _reference_chunks(start, end):
				excluded.add((chrom, chunk))

	X, y, y_in, mask = [], [], [], []
	for chrom, center in interleaved:
		if n_loci is not None and len(X) == n_loci:
			mask.append(False)
			continue

		in_start = center - in_window // 2 - max_jitter
		in_end = in_start + in_window + 2 * max_jitter
		out_start = center - out_window // 2 - max_jitter
		out_end = out_start + out_window + 2 * max_jitter

		lo, hi = in_start, in_end
		if signals is not None:
			lo, hi = min(lo, out_start), max(hi, out_end)

		if lo < 0 or hi > len(genome[chrom]):
			mask.append(False)
			continue

		if any((chrom, chunk) in excluded for chunk in _reference_chunks(lo, hi)):
			mask.append(False)
			continue

		if signals is not None:
			values = [_reference_values(track, chrom, out_start, out_end)
				for track in signals]
			counts = values[target_idx].sum()

			if min_counts is not None and counts < min_counts:
				mask.append(False)
				continue

			if max_counts is not None and counts > max_counts:
				mask.append(False)
				continue

			y.append(values)

		if in_signals is not None:
			y_in.append([_reference_values(track, chrom, in_start, in_end)
				for track in in_signals])

		sequence = genome[chrom][in_start:in_end].upper()
		X.append([[int(base == character) for base in sequence]
			for character in "ACGT"])
		mask.append(True)

	return X, y, y_in, mask


@pytest.fixture(scope="module")
def reference_genome(tmp_path_factory):
	# Random sequence with a soft-masked run and an assembly gap on every
	# chromosome, three tracks of integer-valued signal so that sums are exact
	# in any order of summation, with NaN at bases that have no value and the
	# second track lacking chrX, and three tables of loci with summits.
	rng = numpy.random.RandomState(0)
	path = tmp_path_factory.mktemp("reference_genome")

	genome = {}
	for chrom, length in REFERENCE_CHROMS.items():
		sequence = rng.choice(list("ACGT"), size=length)

		start = rng.randint(0, length - 100)
		sequence[start:start+60] = numpy.char.lower(sequence[start:start+60])

		start = rng.randint(0, length - 100)
		sequence[start:start+40] = 'N'

		genome[chrom] = ''.join(sequence)

	fasta = str(path / "genome.fa")
	with open(fasta, "w") as outfile:
		for chrom, sequence in genome.items():
			outfile.write(">{}\n".format(chrom))
			for i in range(0, len(sequence), 60):
				outfile.write(sequence[i:i+60] + "\n")

	one_hot = {chrom: numpy.array([[base == character for base in
		sequence.upper()] for character in "ACGT"], dtype=numpy.int8)
		for chrom, sequence in genome.items()}

	tracks, bigwigs = [], []
	for t in range(3):
		track = {}
		for chrom, length in REFERENCE_CHROMS.items():
			if t == 1 and chrom == 'chrX':
				continue

			values = rng.choice([0, 1, 2, 3], size=length,
				p=[0.4, 0.3, 0.2, 0.1]).astype(numpy.float32)
			values[rng.uniform(0, 1, size=length) < 0.1] = numpy.nan
			track[chrom] = values

		filename = str(path / "track{}.bw".format(t))
		entries = [(chrom, i, i + 1, float(value)) for chrom in sorted(track)
			for i, value in enumerate(track[chrom]) if not numpy.isnan(value)]
		pybigtools.open(filename, "w").write({chrom: REFERENCE_CHROMS[chrom]
			for chrom in track}, entries)

		tracks.append(track)
		bigwigs.append(filename)

	loci = []
	for n in (60, 25, 40):
		rows = []
		for _ in range(n):
			chrom = str(rng.choice(list(REFERENCE_CHROMS)))
			length = rng.randint(1, 80)
			start = rng.randint(0, REFERENCE_CHROMS[chrom])
			rows.append([chrom, start, start + length, '.', 0, '.', 0.0, 0.0,
				0.0, rng.randint(0, length + 1)])

		loci.append(pandas.DataFrame(rows))

	# One-base loci whose windows, for every window size in the grid, start
	# or end exactly at the edges of chr2 and of the chunk 100-199 of chr1,
	# which the first exclusion region covers.
	centers = [('chr1', c) for c in list(range(78, 101)) + list(range(199, 223))]
	centers += [('chr2', c) for c in list(range(0, 25)) + list(range(752, 777))]
	loci.append(pandas.DataFrame([[chrom, c, c + 1, '.', 0, '.', 0.0, 0.0, 0.0,
		0] for chrom, c in centers]))

	exclusion = pandas.DataFrame([['chr1', 100, 200], ['chr1', 950, 1010],
		['chr2', 350, 350], ['chrX', 1234, 1600], ['3', 0, 50],
		['chrM', 0, 1000], ['chr2', 699, 701]])

	return {'genome': genome, 'fasta': fasta, 'one_hot': one_hot,
		'tracks': tracks, 'bigwigs': bigwigs, 'loci': loci,
		'exclusion': exclusion}


REFERENCE_FILTERS = {
	'none': lambda width, exclusion: {},
	'min_counts': lambda width, exclusion: {'min_counts': 0.9 * width},
	'max_counts': lambda width, exclusion: {'max_counts': 0.9 * width},
	'counts_target_idx': lambda width, exclusion: {'min_counts': 0.5 * width,
		'max_counts': 1.2 * width, 'target_idx': 1},
	'exclusion': lambda width, exclusion: {'exclusion_lists': exclusion},
	'chroms': lambda width, exclusion: {'chroms': ['chr1', '3']},
	'n_loci': lambda width, exclusion: {'n_loci': 50},
	'summits': lambda width, exclusion: {'summits': True},
	'combined': lambda width, exclusion: {'chroms': ['chr2', 'chrX', '3'],
		'exclusion_lists': exclusion, 'n_loci': 30, 'min_counts': 0.5 * width,
		'summits': True},
}


@pytest.mark.filterwarnings("ignore::tangermeme.utils.TangermemeWarning")
@pytest.mark.parametrize("source", ["files", "dicts"])
@pytest.mark.parametrize("mode", ["none", "signals", "in_signals", "both"])
@pytest.mark.parametrize("in_window, out_window, max_jitter", [(10, 4, 0),
	(7, 5, 3), (33, 40, 0), (1, 1, 2), (8, 13, 1)])
@pytest.mark.parametrize("filters", list(REFERENCE_FILTERS))
def test_extract_loci_matches_reference(reference_genome, source, mode,
	in_window, out_window, max_jitter, filters):
	data = reference_genome
	if source == "files":
		sequences, signal_source = data['fasta'], data['bigwigs']
	else:
		sequences, signal_source = data['one_hot'], data['tracks']

	signals = signal_source[:2] if mode in ("signals", "both") else None
	in_signals = signal_source[2:] if mode in ("in_signals", "both") else None
	kwargs = REFERENCE_FILTERS[filters](out_window + 2 * max_jitter,
		data['exclusion'])

	if signals is None and ('min_counts' in kwargs or 'max_counts' in kwargs):
		with pytest.raises(ValueError, match="signals must be provided"):
			extract_loci(data['loci'], sequences, signals, in_signals,
				in_window=in_window, out_window=out_window,
				max_jitter=max_jitter, **kwargs)
		return

	reference_signals = data['tracks'][:2] if signals is not None else None
	reference_in_signals = data['tracks'][2:] if in_signals is not None else None
	X0, y0, y_in0, mask0 = _reference_extract_loci(data['loci'], data['genome'],
		reference_signals, reference_in_signals, in_window, out_window,
		max_jitter, **kwargs)

	if len(X0) == 0:
		with pytest.raises(ValueError, match="No loci remain"):
			extract_loci(data['loci'], sequences, signals, in_signals,
				in_window=in_window, out_window=out_window,
				max_jitter=max_jitter, **kwargs)
		return

	result = extract_loci(data['loci'], sequences, signals, in_signals,
		in_window=in_window, out_window=out_window, max_jitter=max_jitter,
		return_mask=True, **kwargs)

	expected = [numpy.array(X0, dtype=numpy.int8)]
	if signals is not None:
		expected.append(numpy.array(y0, dtype=numpy.float32))
	if in_signals is not None:
		expected.append(numpy.array(y_in0, dtype=numpy.float32))

	assert len(result) == len(expected) + 1
	assert result[-1].dtype == torch.bool
	assert result[-1].tolist() == mask0

	for tensor, array in zip(result, expected):
		assert tensor.dtype == torch.from_numpy(array).dtype
		assert tensor.is_contiguous()
		assert torch.equal(tensor, torch.from_numpy(array))


def test_extract_loci_reference_grid_is_informative(reference_genome):
	# The comparison above only means something if, on the random genome,
	# every filter keeps some loci and drops others. Checked for the windows
	# (7, 5, 3), whose output window with jitter is 11 bases wide.
	data = reference_genome
	args = (data['loci'], data['genome'], data['tracks'][:2],
		data['tracks'][2:], 7, 5, 3)
	n_loci = sum(len(df) for df in data['loci'])

	X, _, _, mask = _reference_extract_loci(*args)
	baseline = sum(mask)

	# Some loci run off a chromosome end.
	assert 0 < baseline < n_loci

	for name, filters in REFERENCE_FILTERS.items():
		if name in ("none", "summits"):
			continue

		_, _, _, mask = _reference_extract_loci(*args,
			**filters(11, data['exclusion']))
		assert 0 < sum(mask) < baseline, name

	# Centering on the summits moves the windows.
	X_summits, _, _, _ = _reference_extract_loci(*args, summits=True)
	assert len(X_summits) > 0
	assert X_summits[:10] != X[:10]


###


def test_read_meme():
	keys = ["MEOX1_homeodomain_1", "HIC2_MA0738.1", "GCR_HUMAN.H11MO.0.A",
		"FOSL2+JUND_MA1145.1", "TEAD3_TEA_2", "ZN263_HUMAN.H11MO.0.A",
		"PAX7_PAX_2", "SMAD3_MA0795.1", "MEF2D_HUMAN.H11MO.0.A",
		"FOXQ1_MOUSE.H11MO.0.C", "TBX19_MA0804.1", "Hes1_MA1099.1"]

	pwm = numpy.array([
		[0.19800,	0.17700,	0.28600,	0.33900],
		[0.21900,	0.18900,	0.34000,	0.25200],
		[0.55844,	0.04496,	0.33966,	0.05694],
		[0.02600,	0.02700,	0.01000,	0.93700],
		[0.00701,	0.04605,	0.67868,	0.26827],
		[0.94300,	0.00600,	0.03100,	0.02000],
		[0.01201,	0.91191,	0.03503,	0.04104],
		[0.13487,	0.01499,	0.82218,	0.02797],
		[0.02800,	0.01200,	0.01000,	0.95000],
		[0.03297,	0.94106,	0.01499,	0.01099],
		[0.97200,	0.00600,	0.01100,	0.01100],
		[0.02300,	0.34500,	0.05500,	0.57700],
		[0.21700,	0.43400,	0.10400,	0.24500],
		[0.16200,	0.22200,	0.41600,	0.20000],
		[0.24800,	0.28800,	0.24100,	0.22300]
	]).T

	motifs = read_meme("tests/data/test.meme")

	assert len(motifs) == 12
	assert isinstance(motifs, dict)
	assert all(isinstance(key, str) for key in motifs.keys())
	assert all(isinstance(pwm, torch.Tensor) for pwm in motifs.values())

	assert all([key in motifs.keys() for key in keys])
	assert_array_almost_equal(motifs['FOSL2+JUND_MA1145.1'], pwm)


def test_read_meme_n_motifs():
	keys = ["MEOX1_homeodomain_1", "HIC2_MA0738.1", "GCR_HUMAN.H11MO.0.A",
		"FOSL2+JUND_MA1145.1", "TEAD3_TEA_2", "ZN263_HUMAN.H11MO.0.A"]

	motifs = read_meme("tests/data/test.meme", n_motifs=6)

	assert len(motifs) == 6
	assert isinstance(motifs, dict)
	assert all(isinstance(key, str) for key in motifs.keys())
	assert all(isinstance(pwm, torch.Tensor) for pwm in motifs.values())

	assert all([key in motifs.keys() for key in keys])


def test_read_meme_file_not_found():
	assert_raises(FileNotFoundError, read_meme,
		"tests/data/this_file_does_not_exist.meme")


###


def test_read_vcf_basic():
	vcf = read_vcf("tests/data/test.vcf")

	assert isinstance(vcf, pandas.DataFrame)
	assert list(vcf.columns) == ["CHROM", "POS", "ID", "REF", "ALT", "QUAL",
		"FILTER", "INFO", "FORMAT"]
	assert len(vcf) > 0


def test_read_vcf_drops_sample_columns():
	vcf = read_vcf("tests/data/test.vcf")

	assert vcf.shape[1] == 9
	assert "NA00001" not in vcf.columns
	assert "NA00002" not in vcf.columns


def test_read_vcf_chrom_is_str():
	# read_vcf forces dtype=str, so chromosome names match the record names
	# used by pyfaidx/pybigtools without any further coercion.
	vcf = read_vcf("tests/data/test.vcf")

	assert all(isinstance(chrom, str) for chrom in vcf['CHROM'])
	assert vcf['POS'].dtype == numpy.int64


###


def test_one_hot_to_fasta_basic(tmp_path):
	X = torch.stack([
		one_hot_encode('ACGT' * 25),
		one_hot_encode('GCGC' * 25),
	])

	path = tmp_path / "out.fa"
	one_hot_to_fasta(X, str(path))

	contents = path.read_text()
	assert "ACGT" in contents
	assert "GCGC" in contents
	# default headers are sequence indices
	assert "0" in contents
	assert "1" in contents


def test_one_hot_to_fasta_headers(tmp_path):
	X = torch.stack([
		one_hot_encode('A' * 100),
		one_hot_encode('C' * 100),
	])

	path = tmp_path / "out.fa"
	one_hot_to_fasta(X, str(path), headers=["seq_a", "seq_c"])

	contents = path.read_text()
	assert "seq_a" in contents
	assert "seq_c" in contents


def test_one_hot_to_fasta_headers_length_mismatch(tmp_path):
	X = torch.stack([
		one_hot_encode('A' * 10),
		one_hot_encode('C' * 10),
	])

	path = tmp_path / "out.fa"
	assert_raises(IndexError, one_hot_to_fasta, X, str(path), headers=["only_one"])


###


def test_interleave_loci_empty_dataframe():
	df = pandas.DataFrame({'chrom': [], 'start': [], 'end': []})
	result = _interleave_loci(df)

	assert len(result) == 0
	assert list(result.columns) == ['chrom', 'start', 'end']


###


# A fasta opened from a path is read through its .fai index from a memory map
# of the file. These tests check that each window equals what pyfaidx returns
# for it: across line ends of every width and kind, at both ends of every
# record, and at the end of a file that lacks a final line end. When the
# bytes are not what pyfaidx would return unchanged, or the file cannot be
# mapped, pyfaidx reads the windows instead, so the output or the error is
# the one a pyfaidx.Fasta object gives.

FASTA_GENOME_LENGTHS = {'chr1': 97, 'empty': 0, 'chr2': 70, 'chr3': 23,
	'chrLast': 420}


def _fasta_genome(seed=0):
	rng = numpy.random.RandomState(seed)
	return {chrom: ''.join(rng.choice(list('ACGTacgtN'), size=length))
		for chrom, length in FASTA_GENOME_LENGTHS.items()}


def _write_fasta(path, genome, width, newline='\n', final_newline=True):
	lines = []
	for chrom, sequence in genome.items():
		lines.append('>' + chrom)
		lines.extend(sequence[i:i+width] for i in range(0, len(sequence),
			width))

	text = newline.join(lines) + (newline if final_newline else '')
	with open(path, 'wb') as handle:
		handle.write(text.encode('utf8'))


def _every_window(genome, in_window):
	# One-base loci whose windows start at every base where they fit.
	rows = []
	for chrom, sequence in genome.items():
		for start in range(len(sequence) - in_window + 1):
			mid = start + in_window // 2
			rows.append((chrom, mid, mid + 1))

	return pandas.DataFrame(rows)


def _encode_windows(genome, loci, in_window, alphabet=['A', 'C', 'G', 'T'],
	ignore=['N']):
	X = []
	for chrom, mid, _ in loci.itertuples(index=False):
		start = mid - in_window // 2
		sequence = genome[chrom][start:start + in_window].upper()
		X.append(one_hot_encode(sequence, alphabet=alphabet, ignore=ignore))

	return torch.stack(X)


def _same_as_pyfaidx_object(loci, path, **kwargs):
	# The output, or the type and message of the error, equal those of the
	# same call given a pyfaidx.Fasta object, whose windows pyfaidx reads.
	fasta = None
	try:
		fasta = pyfaidx.Fasta(path)
		expected = extract_loci(loci, fasta, **kwargs)
	except Exception as error:
		with pytest.raises(type(error)) as info:
			extract_loci(loci, path, **kwargs)

		assert str(info.value) == str(error)
		return None
	finally:
		if fasta is not None:
			fasta.close()

	X = extract_loci(loci, path, **kwargs)
	assert X.dtype == expected.dtype
	assert X.shape == expected.shape
	assert X.is_contiguous()
	assert torch.equal(X, expected)
	return X


@pytest.mark.parametrize("width", [1, 7, 10, 60])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("final_newline", [True, False])
def test_extract_loci_fasta_every_window(tmp_path, width, newline,
	final_newline):
	genome = _fasta_genome()
	path = str(tmp_path / "genome.fa")
	_write_fasta(path, genome, width, newline, final_newline)

	for in_window in [1, 5, 8, 61]:
		loci = _every_window(genome, in_window)
		X = _same_as_pyfaidx_object(loci, path, in_window=in_window)
		assert torch.equal(X, _encode_windows(genome, loci, in_window))


def test_extract_loci_fasta_crlf_index(tmp_path):
	# With \r\n line ends the index has two bytes per line more than bases.
	genome = _fasta_genome(1)
	path = str(tmp_path / "genome.fa")
	_write_fasta(path, genome, 10, "\r\n")

	loci = _every_window(genome, 13)
	X = _same_as_pyfaidx_object(loci, path, in_window=13)
	assert torch.equal(X, _encode_windows(genome, loci, 13))

	index = pandas.read_csv(path + ".fai", sep="\t", header=None)
	nonempty = index[index[1] > 0]
	assert ((nonempty[4] - nonempty[3]) == 2).all()


def test_extract_loci_fasta_builds_missing_index(tmp_path):
	# pyfaidx builds a missing .fai when the file is opened, as before.
	genome = _fasta_genome(2)
	path = tmp_path / "genome.fa"
	_write_fasta(str(path), genome, 7)
	assert not (tmp_path / "genome.fa.fai").exists()

	loci = _every_window(genome, 9)
	X = extract_loci(loci, path, in_window=9)
	assert (tmp_path / "genome.fa.fai").exists()
	assert torch.equal(X, _encode_windows(genome, loci, 9))


def test_extract_loci_fasta_alphabet_and_ignore(tmp_path):
	# Upper-casing, ignored characters and the unknown-character error are
	# those of one_hot_encode on the upper-cased window.
	genome = {'chr1': 'ACGTNacgtnZzACGT' * 5, 'chr2': 'ACGTACGTAC' * 3}
	path = str(tmp_path / "genome.fa")
	_write_fasta(path, genome, 9)

	loci = _every_window(genome, 6)
	with pytest.raises(ValueError, match="Encountered character"):
		extract_loci(loci, path, in_window=6)

	_same_as_pyfaidx_object(loci, path, in_window=6)

	X = _same_as_pyfaidx_object(loci, path, in_window=6, ignore=['N', 'Z'])
	assert torch.equal(X, _encode_windows(genome, loci, 6, ignore=['N', 'Z']))

	alphabet = ['A', 'C', 'G', 'T', 'Z']
	X = _same_as_pyfaidx_object(loci, path, in_window=6, alphabet=alphabet)
	assert X.shape[1] == 5
	assert torch.equal(X, _encode_windows(genome, loci, 6, alphabet=alphabet))

	with pytest.raises(ValueError, match="in the alphabet and also"):
		extract_loci(loci, path, in_window=6, ignore=['A'])

	# A non-ASCII alphabet is encoded by one_hot_encode after pyfaidx reads.
	_same_as_pyfaidx_object(loci, path, in_window=6, alphabet=['A', 'C', 'G',
		'T', 'Z', 'é'])


def test_extract_loci_fasta_uses_memory_map(tmp_path):
	# A plain fasta is read from the memory map rather than through pyfaidx.
	genome = _fasta_genome(3)
	path = str(tmp_path / "genome.fa")
	_write_fasta(path, genome, 11)

	fasta = pyfaidx.Fasta(path)
	windows = [('chr1', 0), ('chrLast', 400), ('chr2', 50)]
	X = _read_fasta_windows_mmap(fasta, windows, 20, ['A', 'C', 'G', 'T'],
		['N'])
	fasta.close()

	assert X is not None
	assert X.dtype == numpy.int8
	assert X.flags['C_CONTIGUOUS']
	for x, (chrom, start) in zip(X, windows):
		assert numpy.array_equal(x, one_hot_encode(genome[chrom][start:start+20]
			.upper()).numpy())


def test_extract_loci_fasta_irregular_lines_raises(tmp_path):
	# pyfaidx rejects lines of unequal length inside a record, as before.
	path = str(tmp_path / "genome.fa")
	with open(path, "w") as handle:
		handle.write(">chr1\nACGTACGT\nACG\nACGTACGT\nACGT\n")

	loci = pandas.DataFrame([('chr1', 10, 11)])
	with pytest.raises(pyfaidx.FastaIndexingError):
		extract_loci(loci, path, in_window=4)

	_same_as_pyfaidx_object(loci, path, in_window=4)


@pytest.mark.parametrize("line", [
	# A carriage return inside a line, which pyfaidx removes.
	b"AC\rTAC",
	# A byte outside ASCII, which pyfaidx decodes as UTF-8, so that the
	# window holds one character fewer.
	"ACGéA".encode('utf8'),
	# A lone byte outside ASCII, which cannot be decoded.
	b"ACG\xffTA",
])
@pytest.mark.parametrize("in_window", [3, 5, 8])
def test_extract_loci_fasta_bytes_pyfaidx_changes(tmp_path, line, in_window):
	# The second of four lines of six bytes holds a byte that pyfaidx does not
	# return unchanged. The index is written by hand, since pyfaidx cannot
	# build one for the last two files.
	path = str(tmp_path / "genome.fa")
	with open(path, "wb") as handle:
		handle.write(b">chr1\nACGTAC\n" + line + b"\nACGTAC\nACGTAC\n")

	with open(path + ".fai", "w") as handle:
		handle.write("chr1\t24\t6\t6\t7\n")

	loci = pandas.DataFrame([('chr1', mid, mid + 1) for mid in range(
		in_window // 2, 25 - in_window + in_window // 2)])
	for n in [1, 3, len(loci)]:
		_same_as_pyfaidx_object(loci.iloc[:n], path, in_window=in_window)
		_same_as_pyfaidx_object(loci.iloc[-n:], path, in_window=in_window)

	for i in range(len(loci)):
		_same_as_pyfaidx_object(loci.iloc[i:i+1], path, in_window=in_window)


def test_extract_loci_fasta_stale_index(tmp_path):
	# An index that describes other lines than the file has, but is newer so
	# pyfaidx keeps it, gives what pyfaidx reads with it.
	genome = {'chr1': 'ACGTAACCGGTT' * 4}
	path = str(tmp_path / "genome.fa")
	_write_fasta(path, genome, 12)
	with open(path + ".fai", "w") as handle:
		handle.write("chr1\t48\t6\t10\t11\n")

	for in_window in [3, 9, 20]:
		loci = _every_window(genome, in_window)
		_same_as_pyfaidx_object(loci, path, in_window=in_window)
		for i in range(len(loci)):
			_same_as_pyfaidx_object(loci.iloc[i:i+1], path, in_window=in_window)


def _write_bgzf(path, data, block_size=100):
	# Block-gzip, as bgzip writes: gzip members carrying their compressed
	# size in a BC extra field, followed by an empty end-of-file block.
	import zlib
	import struct

	with open(path, "wb") as handle:
		for i in range(0, len(data), block_size):
			block = data[i:i+block_size]
			compressor = zlib.compressobj(6, zlib.DEFLATED, -15)
			compressed = compressor.compress(block) + compressor.flush()
			handle.write(struct.pack('<BBBBIBBHBBHH', 31, 139, 8, 4, 0, 0, 255,
				6, 66, 67, 2, len(compressed) + 25))
			handle.write(compressed)
			handle.write(struct.pack('<II', zlib.crc32(block), len(block)))

		handle.write(bytes.fromhex("1f8b08040000000000ff0600424302001b00"
			"03000000000000000000"))


def _has_biopython():
	try:
		import Bio  # noqa: F401
	except ImportError:
		return False

	return True


def _bgzf_genome(tmp_path):
	import gzip

	genome = _fasta_genome(4)
	plain = str(tmp_path / "plain.fa")
	_write_fasta(plain, genome, 10)
	with open(plain, "rb") as handle:
		data = handle.read()

	path = str(tmp_path / "genome.fa.gz")
	_write_bgzf(path, data)
	with gzip.open(path, "rb") as handle:
		assert handle.read() == data

	return genome, path


@pytest.mark.skipif(not _has_biopython(), reason="pyfaidx reads block-gzip "
	"fasta files with BioPython")
def test_extract_loci_fasta_bgzip(tmp_path):
	genome, path = _bgzf_genome(tmp_path)
	loci = _every_window(genome, 7)
	X = _same_as_pyfaidx_object(loci, path, in_window=7)
	assert torch.equal(X, _encode_windows(genome, loci, 7))


@pytest.mark.skipif(_has_biopython(), reason="BioPython is installed")
def test_extract_loci_fasta_bgzip_without_biopython(tmp_path):
	genome, path = _bgzf_genome(tmp_path)
	loci = _every_window(genome, 7)
	with pytest.raises(ImportError, match="BioPython"):
		extract_loci(loci, path, in_window=7)

	_same_as_pyfaidx_object(loci, path, in_window=7)


def _mapped_paths():
	with open("/proc/self/maps") as handle:
		return handle.read()


@pytest.mark.skipif(not pathlib.Path("/proc/self/maps").exists(),
	reason="needs /proc/self/maps")
def test_extract_loci_fasta_unmapped(tmp_path):
	# The memory map is closed on return and on the errors raised after the
	# windows are chosen.
	genome = {'chr1': 'ACGTNacgtnZzACGT' * 5}
	path = str(tmp_path / "genome_unmapped.fa")
	_write_fasta(path, genome, 9)
	loci = _every_window(genome, 6)

	extract_loci(loci.iloc[:2], path, in_window=6)
	assert "genome_unmapped.fa" not in _mapped_paths()

	with pytest.raises(ValueError, match="Encountered character"):
		extract_loci(loci, path, in_window=6)
	assert "genome_unmapped.fa" not in _mapped_paths()

	with pytest.raises(ValueError, match="in the alphabet and also"):
		extract_loci(loci, path, in_window=6, ignore=['A'])
	assert "genome_unmapped.fa" not in _mapped_paths()

	with pytest.raises(ValueError, match="No loci remain"):
		extract_loci(loci, path, in_window=1000)
	assert "genome_unmapped.fa" not in _mapped_paths()


###
# bigWig windows read together, sorted by position (_read_signal_windows)
###


class _PositionSignal():
	"""A stand-in for a bigWig whose value at a base is its position.

	It records the length of every read, so a test can see how the windows
	were grouped.
	"""

	def __init__(self, length=10**7):
		self.length = length
		self.reads = []

	def values(self, chrom, start, end, arr=None):
		self.reads.append((chrom, start, end))
		arr[:] = numpy.arange(start, end, dtype=numpy.float64)
		arr[max(0, self.length - start):] = numpy.nan
		return arr


def _per_window(signals, chroms, starts, width):
	out = numpy.full((len(starts), len(signals), width), -1, dtype=numpy.float32)
	scratch = numpy.empty(width, dtype=numpy.float64)
	for k, (chrom, start) in enumerate(zip(chroms, starts)):
		_write_locus_signal(signals, chrom, int(start), int(start) + width,
			out[k], scratch)

	return out


def _per_locus_call(*args, **kwargs):
	# max_counts=inf keeps every locus but reads each window in the loop with
	# one values() call, which is the path the grouped reads must reproduce.
	return extract_loci(*args, max_counts=float("inf"), **kwargs)


def _assert_same(a, b):
	a = a if isinstance(a, (list, tuple)) else [a]
	b = b if isinstance(b, (list, tuple)) else [b]
	assert len(a) == len(b)
	for x, y in zip(a, b):
		assert x.dtype == y.dtype
		assert x.shape == y.shape
		assert x.is_contiguous() == y.is_contiguous()
		assert torch.equal(x, y)


def test_extract_loci_bigwig_overlapping_and_repeated_windows():
	# Repeats of one locus, windows that overlap in both directions, and
	# chromosomes interleaved, so the sorted order differs from locus order.
	loci = pandas.DataFrame({
		'chrom': ['chr1', 'chr2', 'chr1', 'chr1', 'chr2', 'chr1', 'chr1',
			'chr1', 'chr3', 'chr1'],
		'start': [10, 25, 10, 12, 35, 8, 80, 10, 5, 140],
		'end': [30, 55, 30, 32, 65, 28, 100, 30, 25, 160]
	})

	X, y, y_in = extract_loci(loci, "tests/data/test.fa",
		["tests/data/test.bw", "tests/data/test2.bw"], ["tests/data/test.bw"],
		in_window=10, out_window=20)

	_assert_same([X, y, y_in], _per_locus_call(loci, "tests/data/test.fa",
		["tests/data/test.bw", "tests/data/test2.bw"], ["tests/data/test.bw"],
		in_window=10, out_window=20))

	signals = _load_signals(["tests/data/test.bw", "tests/data/test2.bw"])
	for k, (chrom, start, end) in enumerate(loci.values):
		mid = start + (end - start) // 2
		expected = _extract_locus_signal(signals, chrom, mid - 10, mid + 10)
		assert torch.equal(y[k], torch.from_numpy(numpy.stack(expected)))

	assert torch.equal(y[0], y[2]) and torch.equal(y[0], y[7])
	assert torch.equal(y[0, :, 2:], y[3, :, :-2])


def test_extract_loci_bigwig_window_at_chrom_end():
	# test3.bw has a 40 bp chr1 and test.fa a 284 bp one: the first window is
	# inside it, the second runs past its end and the third starts past it,
	# and all three are close enough to be read together.
	loci = pandas.DataFrame({
		'chrom': ['chr1', 'chr1', 'chr1', 'chr1'],
		'start': [15, 30, 50, 15],
		'end': [25, 40, 70, 25]
	})

	with warnings.catch_warnings():
		warnings.simplefilter("error")
		X, y = extract_loci(loci, "tests/data/test.fa", ["tests/data/test3.bw"],
			in_window=10, out_window=20)

	_assert_same([X, y], _per_locus_call(loci, "tests/data/test.fa",
		["tests/data/test3.bw"], in_window=10, out_window=20))

	signal = pybigtools.open("tests/data/test3.bw")
	inside = numpy.nan_to_num(signal.values("chr1", 10, 30)).astype('float32')
	assert_array_almost_equal(y[0, 0], inside)
	assert torch.equal(y[0], y[3])

	# chr1 of test3.bw ends at 40, so the window [25, 45) ends in five zeros.
	straddle = numpy.nan_to_num(signal.values("chr1", 25, 45))
	assert numpy.isnan(signal.values("chr1", 25, 45)[15:]).all()
	assert_array_almost_equal(y[1, 0], straddle)
	assert (y[1, 0, 15:] == 0).all()
	assert (y[2] == 0).all()


def test_extract_loci_bigwig_group_spans_gap(tmp_path):
	# A bigWig with data at [0, 100) and [3000, 3100) only. The windows
	# [0, 120) and [2990, 3110) are 2,870 bp apart, so one read spans the
	# empty stretch, which reads as missing, 0.0, and the window [1440, 1560)
	# lies inside it.
	path = str(tmp_path / "gap.bw")
	pybigtools.open(path, "w").write({'chr1': 10000},
		[('chr1', 0, 100, 1.5), ('chr1', 3000, 3100, 2.5)])

	sequences = {'chr1': numpy.zeros((4, 10000), dtype=numpy.int8)}
	loci = pandas.DataFrame({
		'chrom': ['chr1', 'chr1', 'chr1'],
		'start': [3040, 50, 1490],
		'end': [3060, 70, 1510]
	})

	X, y = extract_loci(loci, sequences, [path], in_window=10, out_window=120)
	_assert_same([X, y], _per_locus_call(loci, sequences, [path],
		in_window=10, out_window=120))

	expected = numpy.zeros((3, 1, 120), dtype=numpy.float32)
	expected[0, 0, 10:110] = 2.5
	expected[1, 0, :100] = 1.5
	assert torch.equal(y, torch.from_numpy(expected))

	# The same windows read with no grouping, and with one read over all.
	signals = _load_signals([path])
	starts = numpy.array([2990, 0, 1440])
	for max_gap, max_span in [(0, 120), (-1, 10**6), (4096, 65536),
		(10**6, 10**6)]:
		out = numpy.full((3, 1, 120), -1, dtype=numpy.float32)
		failures = _read_signal_windows(signals, ['chr1'] * 3, starts, 120,
			out, max_gap=max_gap, max_span=max_span)
		assert failures == []
		numpy.testing.assert_array_equal(out, _per_window(signals,
			['chr1'] * 3, starts, 120))
		numpy.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("max_gap", [-10**6, -1, 0, 1, 7, 50, 10**6])
@pytest.mark.parametrize("max_span", [1, 12, 13, 14, 40, 10**6])
def test_read_signal_windows_groups(max_gap, max_span):
	# Every row is the positions of its window, whatever the grouping, and no
	# read is longer than max(max_span, width).
	rng = numpy.random.default_rng(0)
	width = 13
	chroms = list(rng.choice(['a', 'b', 'c'], 300))
	starts = rng.integers(0, 400, 300)
	starts[:20] = starts[20:40]

	signal = _PositionSignal(length=380)
	out = numpy.full((310, 1, width), -1, dtype=numpy.float32)
	failures = _read_signal_windows([signal], chroms, starts, width, out,
		max_gap=max_gap, max_span=max_span)

	assert failures == []
	expected = (starts[:, None] + numpy.arange(width)).astype(numpy.float32)
	expected[starts[:, None] + numpy.arange(width) >= 380] = numpy.nan
	numpy.testing.assert_array_equal(out[:300, 0], expected)
	assert (out[300:] == -1).all()

	lengths = [end - start for _, start, end in signal.reads]
	assert max(lengths) <= max(max_span, width)
	assert sorted(set(chroms)) == sorted(set(c for c, _, _ in signal.reads))
	# Every window alone when a span cannot hold two starts or no gap is
	# small enough, only repeats together when it holds exactly one start,
	# and one read per chromosome when nothing splits them.
	if max_span <= width or max_gap < -width:
		assert len(signal.reads) == 300
	elif max_span == width + 1:
		assert len(signal.reads) == len(set(zip(chroms, starts.tolist())))
	elif max_gap == 10**6 and max_span == 10**6:
		assert len(signal.reads) == 3


def test_read_signal_windows_matches_per_window():
	# Random windows on test.bw, including repeats, windows past the end of
	# a chromosome and a chromosome it does not have, which fails whether it
	# is read alone or in a group, and is zero-filled.
	rng = numpy.random.default_rng(1)
	signals = _load_signals(["tests/data/test.bw", "tests/data/test2.bw"])
	width = 17
	chroms = list(rng.choice(['chr1', 'chr2', 'chr3', 'chr6', 'chr7'], 200))
	starts = rng.integers(0, 300, 200)

	for max_gap, max_span in [(0, 17), (4096, 65536), (5, 40), (10**6, 10**6)]:
		out = numpy.full((200, 2, width), -1, dtype=numpy.float32)
		failures = _read_signal_windows(signals, chroms, starts, width, out,
			kind=1, max_gap=max_gap, max_span=max_span)

		expected = _per_window(signals, chroms, starts, width)
		numpy.testing.assert_array_equal(out, expected)

		missing = [k for k, chrom in enumerate(chroms) if chrom == 'chr7']
		assert len(missing) > 0
		assert sorted(failures) == [(k, 1, i, 'chr7', int(starts[k]),
			int(starts[k]) + width) for k in missing for i in range(2)]
		assert (out[missing] == 0).all()


def test_extract_loci_bigwig_missing_chrom_warns_in_locus_order():
	# The warnings for a chromosome missing from a bigWig come in the order
	# the per-locus path gives them: by locus, signals before in_signals, and
	# by signal, even though the windows are read sorted by position.
	peaks = pandas.read_csv("tests/data/test.bed", sep="\t", header=None)
	others = pandas.DataFrame({0: ['chr2', 'chr1', 'chr2', 'chr1'],
		1: [120, 150, 30, 12], 2: [140, 170, 50, 32]})
	kwargs = dict(in_window=8, out_window=10, return_mask=True)
	signals = ["tests/data/test3.bw", "tests/data/test.bw", "tests/data/test3.bw"]
	in_signals = ["tests/data/test.bw", "tests/data/test3.bw"]

	with pytest.warns(TangermemeWarning) as record:
		result = extract_loci([peaks, others], "tests/data/test.fa", signals,
			in_signals, **kwargs)

	with pytest.warns(TangermemeWarning) as expected_record:
		expected = _per_locus_call([peaks, others], "tests/data/test.fa",
			signals, in_signals, **kwargs)

	_assert_same(result, expected)
	messages = [str(w.message) for w in record]
	assert messages == [str(w.message) for w in expected_record]
	assert len(messages) == 4 * 3
	# The first chr2 locus is the second of `others`; sorted by position,
	# its window [125, 135) would come after [35, 45).
	assert messages[0].startswith("chr2 125 135 ")


@pytest.mark.parametrize("n_loci", [1, 2, 5, 7])
def test_extract_loci_bigwig_grouped_n_loci(n_loci):
	# Only the windows of the loci kept before the cap are read.
	loci = pandas.DataFrame({
		'chrom': ['chr2', 'chr1', 'chr1', 'chr2', 'chr1', 'chr1', 'chr3',
			'chr1'],
		'start': [25, 10, 10, 35, 270, 80, 5, 140],
		'end': [55, 30, 30, 65, 280, 100, 25, 160]
	})

	kwargs = dict(in_window=10, out_window=20, n_loci=n_loci, return_mask=True)
	result = extract_loci(loci, "tests/data/test.fa", ["tests/data/test.bw"],
		["tests/data/test2.bw"], **kwargs)
	_assert_same(result, _per_locus_call(loci, "tests/data/test.fa",
		["tests/data/test.bw"], ["tests/data/test2.bw"], **kwargs))
	assert len(result[0]) == n_loci


###
# The kept loci found without the per-locus loop: a fasta opened from a path,
# bigWig signals or none, and no count filter
###


def _unlooped_loci():
	# Loci off the ends of chr1 and chr6, repeats, interleaved chromosomes,
	# chr7, which the bigWigs do not have, and chr1 past the end of test3.bw.
	return [pandas.read_csv("tests/data/test.bed", sep="\t", header=None),
		pandas.DataFrame({0: ['chr7', 'chr1', 'chr6', 'chr2', 'chr1', 'chr7',
			'chr4', 'chr1', 'chr3', 'chr1'],
			1: [500, 2, 70, 30, 270, 900, 100, 10, 60, 210],
			2: [520, 6, 80, 50, 280, 940, 111, 30, 64, 230]})]


def _with_warnings(*args, **kwargs):
	with warnings.catch_warnings(record=True) as record:
		warnings.simplefilter("always")
		result = extract_loci(*args, **kwargs)

	return result, [str(w.message) for w in record]


_UNLOOPED_SIGNALS = {
	'none': (None, None),
	'signals': (["tests/data/test.bw", "tests/data/test3.bw"], None),
	'in_signals': (None, ["tests/data/test2.bw"]),
	'both': (["tests/data/test.bw"], ["tests/data/test3.bw",
		"tests/data/test.bw"]),
}


@pytest.mark.parametrize("signals", list(_UNLOOPED_SIGNALS))
@pytest.mark.parametrize("n_loci", [None, 1, 3, 100])
@pytest.mark.parametrize("exclusion", [False, True])
@pytest.mark.parametrize("windows", [(10, 20, 0), (9, 13, 3)])
def test_extract_loci_unlooped_matches_loop(signals, n_loci, exclusion,
	windows):
	# A pyfaidx.Fasta object is read one locus at a time in the loop, and so
	# is every locus when there is a count filter, so both give what the loop
	# gives for the same loci, including the warnings and their order.
	in_window, out_window, max_jitter = windows
	signals, in_signals = _UNLOOPED_SIGNALS[signals]
	kwargs = dict(in_window=in_window, out_window=out_window,
		max_jitter=max_jitter, n_loci=n_loci, return_mask=True)
	if exclusion:
		kwargs['exclusion_lists'] = pandas.DataFrame({0: ['chr1', 'chr7'],
			1: [205, 890], 2: [210, 905]})

	loci = _unlooped_loci()
	result, messages = _with_warnings(loci, "tests/data/test.fa", signals,
		in_signals, **kwargs)

	expected, expected_messages = _with_warnings(loci,
		pyfaidx.Fasta("tests/data/test.fa"), signals, in_signals, **kwargs)
	_assert_same(result, expected)
	assert messages == expected_messages

	if signals is not None:
		expected, expected_messages = _with_warnings(loci, "tests/data/test.fa",
			signals, in_signals, max_counts=float("inf"), **kwargs)
		_assert_same(result, expected)
		assert messages == expected_messages

	# The kept loci are the first n_loci of those whose windows fit.
	fits = _with_warnings(loci, "tests/data/test.fa", signals, in_signals,
		**dict(kwargs, n_loci=None))[0][-1]
	kept = torch.nonzero(fits)[:, 0][:n_loci]
	assert torch.nonzero(result[-1])[:, 0].tolist() == kept.tolist()
	assert len(result[0]) == len(kept)
	assert 0 < len(kept) < len(fits)


@pytest.mark.parametrize("n_loci", [None, 2])
def test_extract_loci_unlooped_progress_bar(capsys, n_loci):
	# The bar counts the loci whose windows fit, and is filled in one step to
	# the number kept.
	loci = _unlooped_loci()
	n_fit = len(extract_loci(loci, "tests/data/test.fa", in_window=10))
	assert capsys.readouterr().err == ""

	X = extract_loci(loci, "tests/data/test.fa", in_window=10, n_loci=n_loci,
		verbose=True)
	err = capsys.readouterr().err
	assert "Loading Loci" in err
	assert re.findall(r"(\d+)/(\d+) \[", err)[-1] == (str(len(X)), str(n_fit))
	assert len(X) == (n_fit if n_loci is None else n_loci)


def test_extract_loci_unlooped_errors():
	# No locus fits, with and without signals; a chromosome not in the fasta.
	loci = pandas.DataFrame({0: ['chr1', 'chr6'], 1: [0, 70], 2: [4, 80]})
	for signals in [None, ["tests/data/test.bw"]]:
		with pytest.raises(ValueError, match="No loci remain"):
			extract_loci(loci, "tests/data/test.fa", signals, in_window=20)

	loci = pandas.DataFrame({0: ['chr1', 'chrZ'], 1: [100, 100],
		2: [110, 110]})
	with pytest.raises(ValueError, match="not in the sequences: chrZ"):
		extract_loci(loci, "tests/data/test.fa", ["tests/data/test.bw"],
			in_window=20)


def test_read_signal_windows_names():
	# Chromosomes given as indices into names read what the names do, and
	# the failures carry the name. One name is not used.
	rng = numpy.random.default_rng(2)
	signals = _load_signals(["tests/data/test.bw", "tests/data/test2.bw"])
	names = ['chr7', 'chr2', 'chr1', 'chr5', 'chr3']
	codes = rng.integers(0, 4, 150)
	starts = rng.integers(0, 200, 150)

	out = numpy.full((150, 2, 11), -1, dtype=numpy.float32)
	failures = _read_signal_windows(signals, codes, starts, 11, out, kind=1,
		names=names)

	expected = numpy.full((150, 2, 11), -1, dtype=numpy.float32)
	expected_failures = _read_signal_windows(signals,
		[names[code] for code in codes], starts, 11, expected, kind=1)

	numpy.testing.assert_array_equal(out, expected)
	assert sorted(failures) == sorted(expected_failures)
	assert len(failures) > 0 and all(f[3] == 'chr7' for f in failures)


@pytest.mark.parametrize("alphabet", [['A', 'C', 'G', 'T'],
	['A', 'C', 'G', 'T', '\u00e9']])
def test_read_fasta_windows_names(alphabet):
	# Chromosomes given as indices into names read what the names do, from
	# the memory map and, with an alphabet that is not ASCII, through pyfaidx.
	# One name is not used.
	fasta = pyfaidx.Fasta("tests/data/test.fa")
	rng = numpy.random.default_rng(3)
	names = ['chr4', 'chr1', 'chr3', 'chr7']
	codes = rng.integers(0, 3, 50)
	starts = rng.integers(0, 100, 50)
	windows = [(names[code], int(start)) for code, start in zip(codes, starts)]

	X = _read_fasta_windows(fasta, (codes, starts), 20, alphabet, ['N'],
		names=names)
	expected = _read_fasta_windows(fasta, windows, 20, alphabet, ['N'])
	assert X.dtype == expected.dtype and X.flags.c_contiguous
	numpy.testing.assert_array_equal(X, expected)

	X = _read_fasta_windows_mmap(fasta, (codes, starts), 20, alphabet, ['N'],
		names=names)
	if len(alphabet) == 4:
		numpy.testing.assert_array_equal(X, expected)
	else:
		assert X is None

	fasta.close()


###
# bigWig paths read with _BigWigFile
###


READER_GENOME_LENGTHS = {'chr1': 30000, 'chr2': 9000, 'chr3': 4000}


def _write_raw_bigwig(path, chroms, sections, compress=True):
	# A bigWig laid out exactly as given, for the sections pybigtools does not
	# write. Each section is one data block: (chrom, kind, step, span, items),
	# where bedGraph (kind 1) items are (start, end, value), varStep (2) are
	# (start, value), and fixedStep (3) is (start, [values]). A block given as
	# bytes is written as it is, under the index entry (chrom, start, end).
	names = list(chroms)
	ids = {name: i for i, name in enumerate(names)}
	key_size = max(len(name) for name in names)

	blocks = []
	for section in sections:
		if isinstance(section[-1], bytes):
			chrom, start, end, raw = section
			blocks.append((ids[chrom], start, end, raw, False))
			continue

		chrom, kind, step, span, items = section
		if kind == 1:
			body = b''.join(struct.pack('<IIf', s, e, v) for s, e, v in items)
			n, start = len(items), min(s for s, _, _ in items)
			end = max(e for _, e, _ in items)
		elif kind == 2:
			body = b''.join(struct.pack('<If', s, v) for s, v in items)
			n, start = len(items), min(s for s, _ in items)
			end = max(s for s, _ in items) + span
		else:
			first, values = items
			body = b''.join(struct.pack('<f', v) for v in values)
			n, start, end = len(values), first, first + step * (len(values) -
				1) + span

		header = struct.pack('<IIIIIBBH', ids[chrom], start, end, step, span,
			kind, 0, n)
		blocks.append((ids[chrom], start, end, header + body, compress))

	ctree_offset = 64 + 40
	ctree = struct.pack('<IIIIQQ', 0x78CA8C91, len(names), key_size, 8,
		len(names), 0) + struct.pack('<BBH', 1, 0, len(names))
	for i, name in enumerate(names):
		ctree += name.encode().ljust(key_size, b'\0') + struct.pack('<II', i,
			chroms[name])

	data_offset = ctree_offset + len(ctree)
	data, leaves, largest = struct.pack('<Q', len(blocks)), [], 0
	for chrom, start, end, raw, packed in blocks:
		payload = zlib.compress(raw) if packed else raw
		largest = max(largest, len(raw))
		leaves.append((chrom, start, chrom, end, data_offset + len(data),
			len(payload)))
		data += payload

	index_offset = data_offset + len(data)
	rtree = struct.pack('<IIQIIIIQII', 0x2468ACE0, len(leaves), len(leaves),
		leaves[0][0], leaves[0][1], leaves[-1][2], leaves[-1][3],
		index_offset, 1, 0) + struct.pack('<BBH', 1, 0, len(leaves))
	for leaf in leaves:
		rtree += struct.pack('<IIIIQQ', *leaf)

	header = struct.pack('<IHHQQQHHQQIQ', 0x888FFC26, 4, 0, ctree_offset,
		data_offset, index_offset, 0, 0, 0, 64, largest if compress else 0, 0)
	with open(path, 'wb') as handle:
		handle.write(header + struct.pack('<Qdddd', 0, 0, 0, 0, 0) + ctree +
			data + rtree)


def _random_bedgraph(rng, lengths, n_intervals):
	# Sorted, non-overlapping intervals with gaps and adjacent runs, and some
	# values that stress the cast: -0.0, a denormal, infinities and NaN.
	# pybigtools puts 1024 intervals in a block, so these short intervals
	# give several blocks per chromosome.
	special = [-0.0, 1e-45, float('inf'), -float('inf'), float('nan'), 0.0]
	values = []
	for chrom, n in n_intervals.items():
		position = int(rng.integers(0, 50))
		for _ in range(n):
			width = int(rng.integers(1, 5))
			if position + width > lengths[chrom]:
				break

			value = float(numpy.float32(rng.normal() * 10))
			if rng.random() < 0.05:
				value = special[int(rng.integers(len(special)))]

			values.append((chrom, position, position + width, value))
			position += width + int(rng.choice([0, 0, 1, 2, 10]))

	return values


def _random_windows(rng, lengths, n, width):
	# Windows inside each chromosome, some running past its end and some
	# starting past it.
	chroms = list(lengths)
	names = [chroms[k] for k in rng.integers(0, len(chroms), n)]
	starts = []
	for name in names:
		u = rng.random()
		if u < 0.1:
			starts.append(max(0, lengths[name] - int(rng.integers(1, width +
				1))))
		elif u < 0.13:
			starts.append(lengths[name] + int(rng.integers(0, 10)))
		else:
			starts.append(int(rng.integers(0, max(1, lengths[name] - width))))

	return names, numpy.array(starts, dtype=numpy.int64)


def _pybigtools_windows(path, chroms, starts, width):
	bw = pybigtools.open(str(path))
	out = numpy.empty((len(starts), width), dtype=numpy.float32)
	for k, (chrom, start) in enumerate(zip(chroms, starts.tolist())):
		out[k] = bw.values(chrom, start, start + width)

	return out


def _reader_windows(path, chroms, starts, width, n_jobs=1):
	# The windows through _BigWigFile.read into the second of two signals,
	# whose first is filled with a marker that must not be touched.
	reader = _BigWigFile.open(str(path), pybigtools.open(str(path)))
	assert reader is not None

	out = numpy.full((len(starts) + 3, 2, width), 7, dtype=numpy.float32)
	rows = numpy.arange(len(starts), dtype=numpy.int64)[::-1] + 3
	fallback = reader.read(numpy.ascontiguousarray(rows), chroms, starts, out,
		1, n_jobs)

	assert (out[:, 0] == 7).all()
	assert (out[:3] == 7).all()
	return out[rows, 1], fallback


def _assert_bits_equal(x, y):
	assert x.dtype == y.dtype == numpy.float32
	assert x.shape == y.shape
	assert (x.view(numpy.uint32) == y.view(numpy.uint32)).all()


@pytest.fixture
def pybigtools_bigwig(tmp_path):
	# chr3 has no data, so pybigtools leaves it out of the file.
	rng = numpy.random.default_rng(0)
	values = _random_bedgraph(rng, READER_GENOME_LENGTHS, {'chr1': 6000,
		'chr2': 1500})
	path = tmp_path / "written.bw"
	pybigtools.open(str(path), 'w').write({'chr1': 30000, 'chr2': 9000},
		values)
	return path


@pytest.fixture
def reader_fasta(tmp_path):
	# The genome is longer than the bigWigs' chromosomes, so windows near the
	# end of a chromosome run past the end of its signal.
	rng = numpy.random.RandomState(1)
	lengths = {'chr1': 30500, 'chr2': 9000, 'chr3': 4000, 'chr4': 3000}
	genome = {chrom: ''.join(rng.choice(list('ACGT'), size=length)) for chrom,
		length in lengths.items()}
	path = tmp_path / "reader.fa"
	_write_fasta(path, genome, 60)
	return str(path)


@pytest.mark.parametrize("width", [1, 13, 1000, 5000])
@pytest.mark.parametrize("n_jobs", [1, 4])
@pytest.mark.parametrize("batch_blocks", [1, 256])
def test_bigwig_file_pybigtools_written(pybigtools_bigwig, width, n_jobs,
	batch_blocks, monkeypatch):
	monkeypatch.setattr(tangermeme.io, '_BIGWIG_BATCH_BLOCKS', batch_blocks)
	bw = pybigtools.open(str(pybigtools_bigwig))
	lengths = bw.chroms()
	assert list(lengths) == ['chr1', 'chr2']

	reader = _BigWigFile.open(str(pybigtools_bigwig), bw)
	reader.read(numpy.zeros(0, dtype=numpy.int64), [], numpy.zeros(0,
		dtype=numpy.int64), numpy.zeros((0, 1, 1), dtype=numpy.float32), 0)
	assert (reader._index['chroms'] == 0).sum() >= 5

	rng = numpy.random.default_rng(width)
	chroms, starts = _random_windows(rng, lengths, 300, width)
	y, fallback = _reader_windows(pybigtools_bigwig, chroms, starts, width,
		n_jobs)
	y0 = _pybigtools_windows(pybigtools_bigwig, chroms, starts, width)

	assert len(fallback) == 0
	_assert_bits_equal(y, y0)
	assert numpy.isnan(y0).any()


def test_bigwig_file_every_window(pybigtools_bigwig):
	# A window starting at every base of chr2, including those past its end.
	starts = numpy.arange(0, 9020, dtype=numpy.int64)
	chroms = ['chr2'] * len(starts)
	for width in [1, 37]:
		y, fallback = _reader_windows(pybigtools_bigwig, chroms, starts, width,
			2)
		y0 = _pybigtools_windows(pybigtools_bigwig, chroms, starts, width)
		assert len(fallback) == 0
		_assert_bits_equal(y, y0)


@pytest.mark.parametrize("compress", [True, False])
def test_bigwig_file_section_types(tmp_path, compress):
	# bedGraph, varStep and fixedStep sections, in separate blocks and with
	# gaps between and inside them, compressed or stored as they are.
	path = tmp_path / "sections.bw"
	sections = [
		('chr1', 1, 0, 0, [(10, 20, 1.5), (20, 25, -2.5), (40, 41, 3.0)]),
		('chr1', 2, 0, 5, [(100, 4.0), (105, 5.0), (120, float('nan')),
			(130, 6.0)]),
		('chr1', 3, 10, 4, (200, [7.0, -0.0, 9.0, 1e-45])),
		('chr1', 3, 3, 3, (300, [10.0, 11.0, 12.0])),
		('chr1', 2, 0, 0, [(400, 13.0), (401, 14.0)]),
		('chr1', 1, 0, 0, [(990, 1000, float('inf'))]),
		('chr2', 3, 1, 1, (0, [float(i) for i in range(50)])),
		('chr2', 1, 0, 0, [(60, 60, 5.0), (60, 70, 6.0)]),
	]
	_write_raw_bigwig(path, {'chr1': 1000, 'chr2': 100}, sections,
		compress=compress)

	for width in [1, 9, 64, 400]:
		chroms = ['chr1'] * 1100 + ['chr2'] * 120
		starts = numpy.concatenate([numpy.arange(1100), numpy.arange(120)])
		y, fallback = _reader_windows(path, chroms, starts, width, 3)
		y0 = _pybigtools_windows(path, chroms, starts, width)
		assert len(fallback) == 0
		_assert_bits_equal(y, y0)

	assert y0[0, 10] == 1.5 and y0[0, 100] == 4.0 and y0[0, 104] == 4.0
	assert y0[0, 105] == 5.0 and y0[0, 120] == 0 and y0[0, 203] == 7.0
	assert y0[0, 204] == 0 and y0[0, 302] == 10.0 and y0[0, 303] == 11.0


@pytest.mark.parametrize("section", [
	('chr1', 1, 0, 0, [(10, 20, 1.0), (15, 25, 2.0)]),
	('chr1', 1, 0, 0, [(30, 40, 1.0), (10, 20, 2.0)]),
	('chr1', 10, 20, zlib.compress(struct.pack('<IIIIIBBH', 0, 10, 20, 0, 0,
		1, 0, 1) + struct.pack('<IIf', 15, 12, 1.0))),
	('chr1', 2, 0, 10, [(10, 1.0), (15, 2.0)]),
	('chr1', 3, 3, 5, (10, [1.0, 2.0, 3.0])),
	('chr1', 10, 20, b'not a zlib stream!!!'),
	('chr1', 10, 20, zlib.compress(b'\x00' * 22)),
	('chr1', 10, 20, zlib.compress(struct.pack('<IIIIIBBH', 0, 10, 20, 0, 0,
		4, 0, 1) + struct.pack('<IIf', 10, 20, 1.0))),
	('chr1', 10, 20, zlib.compress(struct.pack('<IIIIIBBH', 0, 10, 20, 0, 0,
		1, 0, 5) + struct.pack('<IIf', 10, 20, 1.0))),
	('chr1', 10, 20, zlib.compress(struct.pack('<IIIIIBBH', 1, 10, 20, 0, 0,
		1, 0, 1) + struct.pack('<IIf', 10, 20, 1.0))),
	('chr1', 12, 20, zlib.compress(struct.pack('<IIIIIBBH', 0, 10, 20, 0, 0,
		1, 0, 1) + struct.pack('<IIf', 10, 20, 1.0))),
])
def test_bigwig_file_unsupported_block(tmp_path, section):
	# Overlapping, unsorted or inverted items, a block that is not zlib, is
	# not whole words, has an unknown section type, overruns itself, is on
	# another chromosome, or holds an item outside its index entry. Windows
	# that touch the block are left to pybigtools; the others are read.
	path = tmp_path / "bad.bw"
	sections = [('chr1', 1, 0, 0, [(0, 5, 9.0)]), section,
		('chr1', 1, 0, 0, [(100, 110, 8.0)]), ('chr2', 1, 0, 0,
		[(0, 10, 7.0)])]
	_write_raw_bigwig(path, {'chr1': 1000, 'chr2': 100}, sections)

	chroms = ['chr1', 'chr1', 'chr1', 'chr2', 'chr1']
	starts = numpy.array([0, 12, 95, 0, 18], dtype=numpy.int64)
	y, fallback = _reader_windows(path, chroms, starts, 5, 2)

	# pybigtools raises or panics on some of these blocks, so only the other
	# windows are compared.
	assert fallback.tolist() == [1, 4]
	y0 = _pybigtools_windows(path, ['chr1', 'chr1', 'chr2'], starts[[0, 2, 3]],
		5)
	_assert_bits_equal(y[[0, 2, 3]], y0)


def test_bigwig_file_open_unsupported(tmp_path):
	# A bigBed, and a bigWig whose chromosome tree differs from pybigtools'.
	path = tmp_path / "intervals.bb"
	pybigtools.open(str(path), 'w').write({'chr1': 1000}, [('chr1', 10, 20,
		''), ('chr1', 15, 30, '')])
	assert _BigWigFile.open(str(path), pybigtools.open(str(path))) is None

	a, b = tmp_path / "a.bw", tmp_path / "b.bw"
	_write_raw_bigwig(a, {'chr1': 1000}, [('chr1', 1, 0, 0, [(0, 5, 9.0)])])
	_write_raw_bigwig(b, {'chr1': 999}, [('chr1', 1, 0, 0, [(0, 5, 9.0)])])
	assert _BigWigFile.open(str(a), pybigtools.open(str(a))) is not None
	assert _BigWigFile.open(str(a), pybigtools.open(str(b))) is None
	assert _BigWigFile.open(str(tmp_path / "missing.bw"),
		pybigtools.open(str(a))) is None


def test_bigwig_file_unsupported_index(tmp_path):
	# Index entries that overlap: every window is left to pybigtools.
	path = tmp_path / "overlapping.bw"
	_write_raw_bigwig(path, {'chr1': 1000}, [('chr1', 1, 0, 0, [(0, 50,
		1.0)]), ('chr1', 1, 0, 0, [(40, 60, 2.0)])])

	starts = numpy.array([0, 45, 100], dtype=numpy.int64)
	y, fallback = _reader_windows(path, ['chr1'] * 3, starts, 10)
	assert fallback.tolist() == [0, 1, 2]


def test_bigwig_file_windows_not_read(pybigtools_bigwig):
	# A chromosome that is not in the file, and a negative start.
	chroms = ['chr1', 'chr3', 'chr2', 'chr1']
	starts = numpy.array([5, 5, 5, -3], dtype=numpy.int64)
	y, fallback = _reader_windows(pybigtools_bigwig, chroms, starts, 10)

	assert fallback.tolist() == [1, 3]
	y0 = _pybigtools_windows(pybigtools_bigwig, chroms[:1] + chroms[2:3],
		starts[[0, 2]], 10)
	_assert_bits_equal(y[[0, 2]], y0)


def test_write_locus_signal_skips_none():
	bw = pybigtools.open("tests/data/test.bw")
	out = numpy.full((3, 10), 7, dtype=numpy.float32)
	scratch = numpy.empty(10, dtype=numpy.float64)
	_write_locus_signal([bw, None, bw], 'chr1', 0, 10, out, scratch)

	y0 = bw.values('chr1', 0, 10).astype(numpy.float32)
	assert (out[1] == 7).all()
	_assert_bits_equal(out[0], y0)
	_assert_bits_equal(out[2], y0)


def _reader_loci(rng, n):
	chroms = rng.choice(['chr1', 'chr1', 'chr2', 'chr3', 'chr4'], n)
	lengths = {'chr1': 30500, 'chr2': 9000, 'chr3': 4000, 'chr4': 3000}
	starts = numpy.array([rng.integers(0, lengths[c]) for c in chroms])
	return pandas.DataFrame({'chrom': chroms, 'start': starts,
		'end': starts + rng.integers(1, 300, n)})


def _extract_both(monkeypatch, paths, *args, n_jobs=8, **kwargs):
	# extract_loci with bigWig paths, so they are read by _BigWigFile, and
	# with the same files opened with pybigtools, so they are not.
	reads = []
	read = _BigWigFile.read

	def counting(self, rows, *a, **k):
		reads.append(len(rows))
		return read(self, rows, *a, **k)

	monkeypatch.setattr(tangermeme.io, '_BIGWIG_MIN_WINDOWS', 1)
	monkeypatch.setattr(_BigWigFile, 'read', counting)

	def opened(signals):
		return None if signals is None else [pybigtools.open(str(s)) for s in
			signals]

	with warnings.catch_warnings(record=True) as w:
		warnings.simplefilter("always")
		y = extract_loci(*args, n_jobs=n_jobs, **kwargs,
			**{key: [str(s) for s in paths[key]] for key in paths})
		n_reads = len(reads)

	with warnings.catch_warnings(record=True) as w0:
		warnings.simplefilter("always")
		y0 = extract_loci(*args, **kwargs, **{key: opened(paths[key]) for key
			in paths})

	assert len(reads) == n_reads
	return y, y0, n_reads, [str(x.message) for x in w], [str(x.message) for
		x in w0]


@pytest.mark.parametrize("n_jobs", [1, 3, 8])
@pytest.mark.parametrize("n_loci", [None, 1, 250])
def test_extract_loci_bigwig_reader(reader_fasta, pybigtools_bigwig, tmp_path,
	monkeypatch, n_jobs, n_loci):
	# Two path signals, one of which is also an in_signal, odd windows,
	# jitter, repeated loci, an exclusion list and a mask, against the same
	# call with the files opened with pybigtools.
	path = tmp_path / "sections.bw"
	_write_raw_bigwig(path, {'chr1': 30000, 'chr2': 9000, 'chr3': 4000},
		[('chr1', 3, 10, 4, (0, [float(i) for i in range(2900)])),
		('chr2', 2, 0, 3, [(i * 5, float(i)) for i in range(1700)]),
		('chr3', 1, 0, 0, [(10, 3000, 2.5)])])

	rng = numpy.random.default_rng(2)
	loci = _reader_loci(rng, 400)
	loci = pandas.concat([loci, loci.iloc[:50]])
	exclusion = pandas.DataFrame({0: ['chr1'], 1: [5000], 2: [6000]})

	y, y0, n_reads, w, w0 = _extract_both(monkeypatch, {'signals':
		[pybigtools_bigwig, path], 'in_signals': [path]}, loci, reader_fasta,
		n_jobs=n_jobs, in_window=211, out_window=101, max_jitter=7,
		n_loci=n_loci, exclusion_lists=[exclusion], return_mask=True)

	# One read per path; the first locus may be on a chromosome that is not
	# in both files, and so be read with pybigtools.
	assert n_reads == 3 or n_loci == 1
	assert w == w0
	assert len(y) == len(y0) == 4
	for x, x0 in zip(y, y0):
		assert x.dtype == x0.dtype and x.shape == x0.shape
		assert x.is_contiguous() and torch.equal(x, x0)

	assert (y[1] != 0).any() or n_loci == 1


@pytest.mark.parametrize("target_idx", [0, 1, -1])
@pytest.mark.parametrize("min_counts, max_counts", [(1, None), (None, 5000),
	(1, 5000)])
def test_extract_loci_bigwig_reader_counts(reader_fasta, pybigtools_bigwig,
	monkeypatch, target_idx, min_counts, max_counts):
	# The count filter's target is read in the loop and the other signals
	# after it, only for the loci that are kept.
	rng = numpy.random.default_rng(3)
	loci = _reader_loci(rng, 300)

	y, y0, n_reads, _, _ = _extract_both(monkeypatch, {'signals':
		[pybigtools_bigwig, pybigtools_bigwig]}, loci, reader_fasta,
		in_window=100, out_window=50, min_counts=min_counts,
		max_counts=max_counts, target_idx=target_idx, n_loci=120,
		return_mask=True)

	assert n_reads == 1
	for x, x0 in zip(y, y0):
		assert torch.equal(x, x0)

	assert 0 < y[-1].sum() < len(loci)


def test_extract_loci_bigwig_reader_missing_chrom_warns(reader_fasta,
	pybigtools_bigwig, monkeypatch):
	# chr3 and chr4 are not in the bigWig: their loci are read with
	# pybigtools in the loop, which warns in the same order.
	rng = numpy.random.default_rng(4)
	loci = _reader_loci(rng, 200)

	y, y0, n_reads, w, w0 = _extract_both(monkeypatch, {'signals':
		[pybigtools_bigwig], 'in_signals': [pybigtools_bigwig]}, loci,
		reader_fasta, in_window=100, out_window=50)

	assert n_reads == 2
	assert len(w) > 0 and w == w0
	assert all('chr3' in m or 'chr4' in m for m in w)
	for x, x0 in zip(y, y0):
		assert torch.equal(x, x0)


def test_extract_loci_bigwig_reader_missing_chrom_warning_order(reader_fasta,
	pybigtools_bigwig, tmp_path, monkeypatch):
	# The path lacks chr3 and chr4 and the object lacks chr2 and chr4, so
	# the warnings of the two are interleaved by locus, as the loop makes
	# them.
	path = tmp_path / "chr1_chr3.bw"
	_write_raw_bigwig(path, {'chr1': 30000, 'chr3': 4000}, [('chr1', 1, 0, 0,
		[(0, 100, 1.0)]), ('chr3', 1, 0, 0, [(0, 100, 2.0)])])

	rng = numpy.random.default_rng(7)
	loci = _reader_loci(rng, 300)

	monkeypatch.setattr(tangermeme.io, '_BIGWIG_MIN_WINDOWS', 1)
	results = []
	for first in [str(pybigtools_bigwig), pybigtools.open(str(
			pybigtools_bigwig))]:
		with warnings.catch_warnings(record=True) as w:
			warnings.simplefilter("always")
			y = extract_loci(loci, reader_fasta, signals=[first,
				pybigtools.open(str(path))], in_window=100, out_window=50)

		results.append((y, [str(x.message) for x in w]))

	(y, w), (y0, w0) = results
	assert w == w0
	assert {m.split()[0] for m in w} == {'chr2', 'chr3', 'chr4'}
	assert torch.equal(y[0], y0[0]) and torch.equal(y[1], y0[1])


def test_extract_loci_bigwig_reader_fallback_blocks(reader_fasta, tmp_path,
	monkeypatch):
	# Windows that touch a block pybigtools sums the overlapping items of are
	# read with pybigtools after the loop.
	path = tmp_path / "overlapping.bw"
	_write_raw_bigwig(path, {'chr1': 30000}, [('chr1', 1, 0, 0,
		[(0, 100, 1.0), (50, 200, 2.0)]), ('chr1', 1, 0, 0,
		[(1000, 2000, 3.0)])])

	loci = pandas.DataFrame({'chrom': ['chr1'] * 4, 'start': [60, 100, 1500,
		150], 'end': [61, 101, 1501, 151]})
	y, y0, _, _, _ = _extract_both(monkeypatch, {'signals': [path]}, loci,
		reader_fasta, in_window=20, out_window=20)

	assert torch.equal(y[1], y0[1])
	assert y0[1][0, 0, 0] == 3.0


def test_extract_loci_bigwig_reader_threshold(reader_fasta, pybigtools_bigwig,
	monkeypatch):
	# Calls that can keep fewer than _BIGWIG_MIN_WINDOWS loci, or that are
	# given float coordinates, do not use _BigWigFile.
	reads = []
	read = _BigWigFile.read
	monkeypatch.setattr(_BigWigFile, 'read', lambda self, rows, *a, **k:
		reads.append(len(rows)) or read(self, rows, *a, **k))

	rng = numpy.random.default_rng(5)
	loci = _reader_loci(rng, 2000)
	loci = loci[loci['chrom'].isin(['chr1', 'chr2'])]
	assert len(loci) > 1100
	kwargs = dict(signals=[str(pybigtools_bigwig)], in_window=100,
		out_window=50)

	X, y = extract_loci(loci.iloc[:10], reader_fasta, **kwargs)
	X, y = extract_loci(loci, reader_fasta, n_loci=1023, **kwargs)
	assert reads == []

	X, y = extract_loci(loci, reader_fasta, n_loci=1024, **kwargs)
	assert reads == [1024]

	# pybigtools raises for float coordinates, and so does extract_loci.
	monkeypatch.setattr(tangermeme.io, '_BIGWIG_MIN_WINDOWS', 1)
	floats = loci.iloc[:20].astype({'start': float, 'end': float})
	with pytest.raises(TypeError):
		extract_loci(floats, reader_fasta, **kwargs)

	assert reads == [1024]


@pytest.mark.parametrize("n_jobs", [0, -1, 2.5, True, "4", None])
def test_extract_loci_n_jobs_raises(n_jobs):
	with pytest.raises(ValueError, match="n_jobs"):
		extract_loci("tests/data/test.bed", "tests/data/test.fa",
			signals=["tests/data/test.bw"], n_jobs=n_jobs)


def test_extract_loci_n_jobs_same_outputs(reader_fasta, pybigtools_bigwig,
	monkeypatch):
	monkeypatch.setattr(tangermeme.io, '_BIGWIG_MIN_WINDOWS', 1)
	monkeypatch.setattr(tangermeme.io, '_BIGWIG_BATCH_BLOCKS', 1)
	rng = numpy.random.default_rng(6)
	loci = _reader_loci(rng, 500)
	loci = loci[loci['chrom'].isin(['chr1', 'chr2'])]

	y0 = extract_loci(loci, reader_fasta, signals=[str(pybigtools_bigwig)],
		in_window=100, out_window=5000, n_jobs=1)
	for n_jobs in [2, 5, 16]:
		y = extract_loci(loci, reader_fasta, signals=[str(pybigtools_bigwig)],
			in_window=100, out_window=5000, n_jobs=n_jobs)
		assert torch.equal(y[0], y0[0]) and torch.equal(y[1], y0[1])


###
# The reader for bigWig paths on the reads made after the loop
###


def _recording_bigwig_read(monkeypatch, decline=None):
	# Records the number of windows, n_jobs and the number of windows handed
	# back of each _BigWigFile.read. With `decline`, the windows at positions
	# decline(n, signal) are also handed back, after their rows are filled
	# with 7 so that a missed read shows.
	calls = []
	read = _BigWigFile.read

	def recording(self, rows, chroms, starts, out, signal, n_jobs=1,
		names=None):
		left = read(self, rows, chroms, starts, out, signal, n_jobs,
			names=names)
		calls.append((len(rows), n_jobs, len(left)))
		if decline is not None:
			extra = decline(len(rows), signal)
			out[rows[extra], signal] = 7
			left = numpy.union1d(left, extra)

		return left

	monkeypatch.setattr(_BigWigFile, 'read', recording)
	return calls


@pytest.mark.parametrize("n_jobs", [1, 3, 8])
def test_extract_loci_n_jobs_reaches_bigwig_reader(reader_fasta,
	pybigtools_bigwig, monkeypatch, n_jobs):
	# n_jobs reaches the reader on each path that uses it: loci found without
	# the loop (a fasta path), the loop that reads no signal (a pyfaidx.Fasta)
	# and the loop that reads a count filter's target.
	monkeypatch.setattr(tangermeme.io, '_BIGWIG_MIN_WINDOWS', 1)
	monkeypatch.setattr(tangermeme.io, '_BIGWIG_BATCH_BLOCKS', 1)
	calls = _recording_bigwig_read(monkeypatch)

	workers = []

	class RecordingPool(tangermeme.io.ThreadPoolExecutor):
		def __init__(self, max_workers=None, *args, **kwargs):
			workers.append(max_workers)
			super().__init__(max_workers, *args, **kwargs)

	monkeypatch.setattr(tangermeme.io, 'ThreadPoolExecutor', RecordingPool)

	rng = numpy.random.default_rng(8)
	loci = _reader_loci(rng, 300)
	path = str(pybigtools_bigwig)
	fasta = pyfaidx.Fasta(reader_fasta)

	for sequences, kwargs in [(reader_fasta, {}), (fasta, {}), (reader_fasta,
			{'min_counts': 1, 'target_idx': 1})]:
		del calls[:], workers[:]
		mask = extract_loci(loci, sequences, signals=[path, path],
			in_signals=[path], in_window=100, out_window=50, n_jobs=n_jobs,
			return_mask=True, **kwargs)[-1]

		assert len(calls) == (2 if kwargs else 3)
		assert all(c[1] == n_jobs for c in calls)

		# The reader reads the kept windows on chr1 and chr2 itself, which
		# are the only ones given to it under a count filter.
		kept = loci[mask.numpy()]
		n_readable = kept['chrom'].isin(['chr1', 'chr2']).sum()
		assert n_readable > 0
		assert all(c[0] - c[2] == n_readable for c in calls)
		assert all(c[0] == (n_readable if kwargs else len(kept)) for c in
			calls)
		if n_jobs == 1:
			assert workers == []
		else:
			assert len(workers) > 0 and max(workers) == n_jobs

	fasta.close()


@pytest.mark.parametrize("n_jobs", [1, 4])
@pytest.mark.parametrize("min_counts", [None, 1])
def test_extract_loci_bigwig_reader_declined_windows(reader_fasta,
	pybigtools_bigwig, tmp_path, monkeypatch, n_jobs, min_counts):
	# The windows the reader hands back are read with pybigtools, sorted and
	# grouped, into their own rows, next to a signal opened with pybigtools
	# and with the windows on chromosomes the file lacks.
	path = tmp_path / "chr1_chr3.bw"
	_write_raw_bigwig(path, {'chr1': 30000, 'chr3': 4000}, [('chr1', 1, 0, 0,
		[(i * 50, i * 50 + 30, float(i)) for i in range(500)]), ('chr3', 1,
		0, 0, [(0, 3000, 2.0)])])

	monkeypatch.setattr(tangermeme.io, '_BIGWIG_MIN_WINDOWS', 1)
	calls = _recording_bigwig_read(monkeypatch, decline=lambda n, signal:
		numpy.arange(signal % 2, n, 3))

	grouped = []
	read_windows = tangermeme.io._read_signal_windows

	def recording(signals, *args, rows=None, **kwargs):
		grouped.append(rows is not None)
		return read_windows(signals, *args, rows=rows, **kwargs)

	monkeypatch.setattr(tangermeme.io, '_read_signal_windows', recording)

	rng = numpy.random.default_rng(9)
	loci = _reader_loci(rng, 400)
	kwargs = dict(in_window=100, out_window=51, max_jitter=3,
		min_counts=min_counts, target_idx=1, return_mask=True)

	with warnings.catch_warnings(record=True) as w:
		warnings.simplefilter("always")
		y = extract_loci(loci, reader_fasta, signals=[str(path),
			pybigtools.open(str(pybigtools_bigwig)), str(pybigtools_bigwig)],
			in_signals=[str(pybigtools_bigwig)], n_jobs=n_jobs, **kwargs)

	assert len(calls) == 3 and any(grouped)
	del calls[:]

	with warnings.catch_warnings(record=True) as w0:
		warnings.simplefilter("always")
		y0 = extract_loci(loci, reader_fasta, signals=[pybigtools.open(str(
			path)), pybigtools.open(str(pybigtools_bigwig)), pybigtools.open(
			str(pybigtools_bigwig))], in_signals=[pybigtools.open(str(
			pybigtools_bigwig))], **kwargs)

	assert len(calls) == 0
	assert len(w0) > 0 and [str(x.message) for x in w] == [str(x.message) for
		x in w0]
	for x, x0 in zip(y, y0):
		assert x.dtype == x0.dtype and x.shape == x0.shape
		assert x.is_contiguous() and torch.equal(x, x0)

	assert not (y[1] == 7).all(dim=-1).any()


def test_read_signal_windows_rows(pybigtools_bigwig):
	# Window k is written into row rows[k], and its failure names that row.
	signals = [pybigtools.open(str(pybigtools_bigwig))]
	chroms = ['chr2', 'chr1', 'chr4', 'chr1', 'chr1']
	starts = numpy.array([100, 29990, 5, 2000, 2010])
	rows = numpy.array([5, 0, 3, 2, 6])

	out = numpy.full((8, 1, 20), 7, dtype=numpy.float32)
	failures = _read_signal_windows(signals, chroms, starts, 20, out, kind=1,
		rows=rows)

	expected = numpy.full((5, 1, 20), 7, dtype=numpy.float32)
	expected_failures = _read_signal_windows(signals, chroms, starts, 20,
		expected, kind=1)

	assert failures == [(int(rows[f[0]]),) + f[1:] for f in expected_failures]
	assert failures == [(3, 1, 0, 'chr4', 5, 25)]
	numpy.testing.assert_array_equal(out[rows].view(numpy.uint32),
		expected.view(numpy.uint32))
	assert (out[[1, 4, 7]] == 7).all()


def test_bigwig_reader_names(pybigtools_bigwig):
	# Chromosomes given as indices into names read as the names do.
	bw = pybigtools.open(str(pybigtools_bigwig))
	reader = _BigWigFile.open(str(pybigtools_bigwig), bw)
	names = ['chr4', 'chr2', 'chr1']
	codes = numpy.array([2, 1, 0, 2, 1, 2])
	starts = numpy.array([29950, 10, 3, 0, 8990, 1234])
	rows = numpy.arange(6, dtype=numpy.int64)

	out = numpy.full((6, 1, 100), 7, dtype=numpy.float32)
	left = reader.read(rows, codes, starts, out, 0, 2, names=names)

	expected = numpy.full((6, 1, 100), 7, dtype=numpy.float32)
	expected_left = reader.read(rows, [names[c] for c in codes], starts,
		expected, 0, 2)

	assert left.tolist() == expected_left.tolist() == [2]
	numpy.testing.assert_array_equal(out.view(numpy.uint32),
		expected.view(numpy.uint32))


@pytest.mark.parametrize("n_jobs", [1, 4])
def test_extract_loci_bigwig_reader_summed_overlaps(reader_fasta, tmp_path,
	monkeypatch, n_jobs):
	# The windows of blocks with overlapping items are handed back and read
	# with pybigtools, sorted and grouped. Each gets the values a read of that
	# window alone gives, which sums the overlapping items.
	path = tmp_path / "overlapping.bw"
	_write_raw_bigwig(path, {'chr1': 30000}, [('chr1', 1, 0, 0, [(0, 100, 1.0),
		(50, 200, 2.0), (150, 400, 0.5)]), ('chr1', 1, 0, 0, [(1000, 2000,
		3.0)]), ('chr1', 1, 0, 0, [(2500, 2600, 1.5), (2550, 2700, 0.25)])])

	monkeypatch.setattr(tangermeme.io, '_BIGWIG_MIN_WINDOWS', 1)
	calls = _recording_bigwig_read(monkeypatch)
	starts = numpy.arange(60, 3000, 37)
	loci = pandas.DataFrame({'chrom': 'chr1', 'start': starts, 'end': starts +
		1})

	X, y = extract_loci(loci, reader_fasta, signals=[str(path)], in_window=20,
		out_window=120, n_jobs=n_jobs)
	assert len(calls) == 1

	bw = pybigtools.open(str(path))
	scratch = numpy.empty(120, dtype=numpy.float64)
	expected = numpy.empty((len(starts), 1, 120), dtype=numpy.float32)
	for k, mid in enumerate(starts):
		_write_locus_signal([bw], 'chr1', int(mid) - 60, int(mid) + 60,
			expected[k], scratch)

	expected = numpy.nan_to_num(expected)
	assert (expected == 2.5).any() and (expected == 1.75).any()
	numpy.testing.assert_array_equal(y.numpy().view(numpy.uint32),
		expected.view(numpy.uint32))


###
# n_jobs: the sequences of a fasta file are one-hot encoded in blocks of
# rows on numba's threads, which changes no output and no error.


@pytest.mark.parametrize("n_jobs", [2, 3, 8, 1000])
def test_extract_loci_n_jobs(tmp_path, n_jobs):
	# The loci fill several blocks of rows. Every n_jobs gives the output of
	# one thread, from a path and from a pyfaidx.Fasta object, and one above
	# NUMBA_NUM_THREADS is capped rather than raising.
	genome = _fasta_genome()
	path = str(tmp_path / "genome.fa")
	_write_fasta(path, genome, 11)
	loci = _every_window(genome, 7)
	assert len(loci) > 500

	X1 = extract_loci(loci, path, in_window=7, n_jobs=1)
	X = extract_loci(loci, path, in_window=7, n_jobs=n_jobs)
	assert X.dtype == torch.int8
	assert X.is_contiguous()
	assert torch.equal(X, X1)
	assert torch.equal(X, _encode_windows(genome, loci, 7))

	fasta = pyfaidx.Fasta(path)
	X = extract_loci(loci, fasta, in_window=7, n_jobs=n_jobs)
	fasta.close()
	assert X.is_contiguous()
	assert torch.equal(X, X1)


def test_extract_loci_n_jobs_signals_and_mask():
	# With signals, in_signals and a mask, only the sequences are threaded.
	loci = pandas.concat([pandas.read_csv("tests/data/test.bed", sep="\t",
		header=None)] * 40)

	kwargs = dict(signals=["tests/data/test.bw"], in_signals=[
		"tests/data/test.bw"], in_window=10, out_window=6, max_jitter=2,
		return_mask=True)
	outputs1 = extract_loci(loci, "tests/data/test.fa", n_jobs=1, **kwargs)
	outputs = extract_loci(loci, "tests/data/test.fa", n_jobs=4, **kwargs)
	assert len(outputs) == len(outputs1) == 4
	assert outputs[0].shape[0] > 64
	for x, x1 in zip(outputs, outputs1):
		assert x.dtype == x1.dtype
		assert x.is_contiguous()
		assert torch.equal(x, x1)


@pytest.mark.parametrize("n_jobs", [1, 4])
def test_extract_loci_n_jobs_unknown_character(tmp_path, n_jobs):
	# Unknown characters under loci of several blocks raise the error one
	# thread raises, from a path and from a pyfaidx.Fasta object.
	genome = _fasta_genome()
	genome['chr2'] = genome['chr2'][:30] + 'Z' + genome['chr2'][31:]
	genome['chrLast'] = genome['chrLast'][:400] + 'Z' + genome['chrLast'][401:]
	path = str(tmp_path / "genome.fa")
	_write_fasta(path, genome, 11)
	loci = _every_window(genome, 7)

	with pytest.raises(ValueError) as info1:
		extract_loci(loci, path, in_window=7, n_jobs=1)

	with pytest.raises(ValueError) as info:
		extract_loci(loci, path, in_window=7, n_jobs=n_jobs)

	assert "Encountered character" in str(info.value)
	assert str(info.value) == str(info1.value)
	_same_as_pyfaidx_object(loci, path, in_window=7, n_jobs=n_jobs)


@pytest.mark.parametrize("n_jobs", [1, 4])
@pytest.mark.parametrize("line", [180, 3])
def test_extract_loci_n_jobs_read_through_pyfaidx(tmp_path, n_jobs, line):
	# A byte that pyfaidx removes, in a late or an early block of rows, sends
	# every window to pyfaidx, so the result is the one pyfaidx gives.
	lines = [b"ACGTAC"] * 200
	lines[line] = b"AC\rTAC"
	path = str(tmp_path / "genome.fa")
	with open(path, "wb") as handle:
		handle.write(b">chr1\n" + b"\n".join(lines) + b"\n")

	with open(path + ".fai", "w") as handle:
		handle.write("chr1\t1200\t6\t6\t7\n")

	loci = pandas.DataFrame([('chr1', mid, mid + 1) for mid in range(2, 1198)])
	_same_as_pyfaidx_object(loci, path, in_window=5, n_jobs=n_jobs)


@pytest.mark.parametrize("n_jobs", [0, -1, 1.5, "2", None])
def test_extract_loci_n_jobs_raises_without_signals(n_jobs):
	with pytest.raises(ValueError, match="n_jobs must be an integer"):
		extract_loci("tests/data/test.bed", "tests/data/test.fa", in_window=10,
			n_jobs=n_jobs)


def test_extract_loci_n_jobs_restores_numba_threads(tmp_path):
	# numba's thread count is the caller's after a call, and after a call
	# that raises inside the threaded encoding.
	genome = _fasta_genome()
	path = str(tmp_path / "genome.fa")
	_write_fasta(path, genome, 11)
	loci = _every_window(genome, 7)

	genome['chr2'] = genome['chr2'][:30] + 'Z' + genome['chr2'][31:]
	path_z = str(tmp_path / "genome_z.fa")
	_write_fasta(path_z, genome, 11)

	previous = numba.get_num_threads()
	threads = min(3, numba.config.NUMBA_NUM_THREADS)
	numba.set_num_threads(threads)
	try:
		extract_loci(loci, path, in_window=7, n_jobs=2)
		assert numba.get_num_threads() == threads

		fasta = pyfaidx.Fasta(path)
		extract_loci(loci, fasta, in_window=7, n_jobs=2)
		fasta.close()
		assert numba.get_num_threads() == threads

		with pytest.raises(ValueError, match="Encountered character"):
			extract_loci(loci, path_z, in_window=7, n_jobs=2)
		assert numba.get_num_threads() == threads
	finally:
		numba.set_num_threads(previous)


@pytest.mark.parametrize("kwargs", [
	dict(),
	dict(signals=["tests/data/test.bw"], in_signals=["tests/data/test.bw"]),
	dict(signals=["tests/data/test.bw"], min_counts=0),
	dict(fasta_object=True),
], ids=["no_signals", "signals", "count_filter", "fasta_object"])
def test_extract_loci_n_jobs_reaches_encoder(monkeypatch, kwargs):
	# n_jobs reaches the encoder whether the kept loci are found with or
	# without the per-locus loop, and from a pyfaidx.Fasta object.
	import tangermeme.io
	import tangermeme.utils

	encoder = tangermeme.utils._one_hot_encode_fasta
	calls = []

	def recording_encoder(*args, n_jobs=1):
		calls.append(n_jobs)
		return encoder(*args, n_jobs=n_jobs)

	monkeypatch.setattr(tangermeme.io, "_one_hot_encode_fasta",
		recording_encoder)
	monkeypatch.setattr(tangermeme.utils, "_one_hot_encode_fasta",
		recording_encoder)

	kwargs = dict(kwargs)
	sequences = "tests/data/test.fa"
	if kwargs.pop("fasta_object", False):
		sequences = pyfaidx.Fasta(sequences)

	loci = pandas.concat([pandas.read_csv("tests/data/test.bed", sep="\t",
		header=None)] * 40)
	X1 = extract_loci(loci, sequences, in_window=10, out_window=6, n_jobs=1,
		**kwargs)
	X = extract_loci(loci, sequences, in_window=10, out_window=6, n_jobs=3,
		**kwargs)

	if isinstance(sequences, pyfaidx.Fasta):
		sequences.close()

	assert len(calls) == 2
	assert calls == [1, 3]
	for x, x1 in zip(X, X1):
		assert torch.equal(x, x1)


###
# The inflater that _BigWigFile's tasks run without the GIL
###


class _CountingInflater():
	# _inflate_bigwig_blocks, recording each call's number of blocks and the
	# read flag it leaves on each block.
	def __init__(self, kernel):
		self.kernel, self.calls, self.flags = kernel, [], []

	@property
	def signatures(self):
		return self.kernel.signatures

	def __call__(self, uncompress, data, starts, sizes, buffer, blocks,
		length):
		used = self.kernel(uncompress, data, starts, sizes, buffer, blocks,
			length)
		self.calls.append(len(starts))
		self.flags.extend(blocks[:, 5].tolist())
		return used


def _counting_inflater(monkeypatch):
	counting = _CountingInflater(tangermeme.io._inflate_bigwig_blocks)
	monkeypatch.setattr(tangermeme.io, '_inflate_bigwig_blocks', counting)
	return counting


def test_zlib_uncompress_loads():
	import sys
	import ctypes

	uncompress = _zlib_uncompress()
	if sys.platform.startswith('linux'):
		assert uncompress is not None

	if uncompress is not None:
		assert uncompress[1].itemsize == ctypes.sizeof(ctypes.c_ulong)

	assert _zlib_uncompress() is uncompress


@pytest.mark.parametrize("capacity", [32768, 5000, 400, 0])
def test_inflate_bigwig_blocks(capacity):
	# Whole-word streams of several sizes, a stream of 22 bytes, an empty
	# stream, bytes that are not a zlib stream, and a block whose read flag
	# is 0, inflated into buffers that hold all, some or none of them.
	uncompress = _zlib_uncompress()
	if uncompress is None:
		pytest.skip("zlib's uncompress() cannot be loaded here")

	rng = numpy.random.default_rng(3)
	raws = [rng.integers(0, 4, size, dtype=numpy.uint8).tobytes() for size in
		[4, 400, 12288, 4096]] + [b'\x01' * 22, b'']
	streams = [zlib.compress(raw) for raw in raws] + [b'not a zlib stream!!!',
		zlib.compress(raws[1])]

	sizes = numpy.array([len(s) for s in streams], dtype=numpy.int64)
	starts = numpy.cumsum(sizes) - sizes
	data = numpy.frombuffer(b''.join(streams), dtype=numpy.uint8)
	buffer = numpy.full(capacity, 255, dtype=numpy.uint8)
	blocks = numpy.zeros((len(streams), 6), dtype=numpy.int64)
	blocks[:, 2:5] = [5, 6, 7]
	blocks[:, 5] = 1
	blocks[-1, 5] = 0

	used = _inflate_bigwig_blocks(uncompress[0], data, starts, sizes, buffer,
		blocks, numpy.zeros(1, dtype=uncompress[1]))

	# What zlib.decompress gives, placed one after another while they fit.
	position, expected = 0, []
	for k, stream in enumerate(streams):
		if k == len(streams) - 1:
			expected.append((0, 0, 0))
			continue

		try:
			raw = zlib.decompress(stream)
		except zlib.error:
			expected.append((0, 0, 2))
			continue

		if len(raw) >= capacity - position and len(raw) > 0:
			expected.append((0, 0, 2))
		elif len(raw) == 0 or len(raw) % 4 != 0:
			expected.append((0, 0, 0))
		else:
			expected.append((position // 4, (position + len(raw)) // 4, 1))
			assert buffer[position:position + len(raw)].tobytes() == raw
			position += len(raw)

	assert used == position
	assert [tuple(b) for b in blocks[:, [0, 1, 5]].tolist()] == expected
	assert (blocks[:, 2:5] == [5, 6, 7]).all()
	assert sum(flag == 1 for _, _, flag in expected) == {32768: 4, 5000: 3,
		400: 1, 0: 0}[capacity]


def _inflate_mode(monkeypatch, mode, buffer_size):
	# 'kernel' is the default. 'python' has no uncompress(), so every block
	# is inflated by zlib.decompress. 'retry' fits no block into the buffer,
	# and 'some' fits about two in three.
	if mode == 'python':
		monkeypatch.setattr(tangermeme.io, '_zlib_uncompress', lambda: None)
	elif mode == 'retry':
		monkeypatch.setattr(tangermeme.io, '_BIGWIG_MAX_BLOCK_BYTES', 4)
	elif mode == 'some':
		monkeypatch.setattr(tangermeme.io, '_BIGWIG_MAX_BLOCK_BYTES',
			buffer_size * 2 // 3)


@pytest.mark.parametrize("mode", ['kernel', 'python', 'retry', 'some'])
@pytest.mark.parametrize("n_jobs", [1, 4])
@pytest.mark.parametrize("batch_blocks", [2, 256])
def test_bigwig_file_inflate_paths(pybigtools_bigwig, monkeypatch, mode,
	n_jobs, batch_blocks):
	# The same values whichever way a block is inflated, with batches of two
	# blocks that split chr1's blocks across tasks, and with one batch.
	bw = pybigtools.open(str(pybigtools_bigwig))
	reader = _BigWigFile.open(str(pybigtools_bigwig), bw)
	assert reader.buffer_size > 0

	monkeypatch.setattr(tangermeme.io, '_BIGWIG_BATCH_BLOCKS', batch_blocks)
	_inflate_mode(monkeypatch, mode, reader.buffer_size)
	counting = _counting_inflater(monkeypatch)

	rng = numpy.random.default_rng(11)
	chroms, starts = _random_windows(rng, bw.chroms(), 400, 1000)
	y, fallback = _reader_windows(pybigtools_bigwig, chroms, starts, 1000,
		n_jobs)
	y0 = _pybigtools_windows(pybigtools_bigwig, chroms, starts, 1000)

	assert len(fallback) == 0
	_assert_bits_equal(y, y0)

	flags = set(counting.flags)
	if mode == 'python':
		assert counting.calls == []
	else:
		assert flags == {'kernel': {1}, 'retry': {2}, 'some': {1, 2}}[mode]

	# chr1 has at least five blocks, so batches of two split it across at
	# least three tasks.
	assert (reader._read_index()['chroms'] == 0).sum() >= 5
	if mode != 'python':
		assert len(counting.calls) >= (3 if batch_blocks == 2 else 1)


def test_bigwig_file_inflate_uncompressed(tmp_path, monkeypatch):
	# A file stored without compression is not inflated by the kernel.
	path = tmp_path / "raw.bw"
	_write_raw_bigwig(path, {'chr1': 1000}, [('chr1', 1, 0, 0, [(10, 20, 1.5),
		(30, 40, 2.5)]), ('chr1', 3, 3, 3, (300, [10.0, 11.0]))],
		compress=False)
	counting = _counting_inflater(monkeypatch)

	starts = numpy.arange(0, 400, 7, dtype=numpy.int64)
	chroms = ['chr1'] * len(starts)
	y, fallback = _reader_windows(path, chroms, starts, 50, 2)
	y0 = _pybigtools_windows(path, chroms, starts, 50)

	assert len(fallback) == 0
	_assert_bits_equal(y, y0)
	assert counting.calls == []


@pytest.mark.parametrize("mode", ['kernel', 'python', 'retry'])
def test_bigwig_file_inflate_unsupported_blocks(tmp_path, monkeypatch, mode):
	# Blocks that are not zlib, are not whole words, or hold overlapping
	# items: their windows are left to pybigtools on every path, and the
	# other windows are read.
	path = tmp_path / "bad.bw"
	sections = [('chr1', 1, 0, 0, [(0, 5, 9.0)]),
		('chr1', 10, 20, b'not a zlib stream!!!'),
		('chr1', 30, 40, zlib.compress(b'\x00' * 22)),
		('chr1', 1, 0, 0, [(100, 110, 8.0)]),
		('chr1', 1, 0, 0, [(200, 210, 7.0), (205, 215, 1.0)]),
		('chr1', 1, 0, 0, [(300, 310, 6.0)] + [(400 + i, 401 + i, float(i))
			for i in range(300)]),
		('chr2', 1, 0, 0, [(0, 10, 7.0)])]
	_write_raw_bigwig(path, {'chr1': 1000, 'chr2': 100}, sections)
	reader = _BigWigFile.open(str(path), pybigtools.open(str(path)))
	_inflate_mode(monkeypatch, mode, reader.buffer_size)
	counting = _counting_inflater(monkeypatch)

	chroms = ['chr1'] * 7 + ['chr2']
	starts = numpy.array([0, 12, 95, 32, 203, 305, 450, 0], dtype=numpy.int64)
	y, fallback = _reader_windows(path, chroms, starts, 5, 2)

	assert fallback.tolist() == [1, 3, 4]
	keep = [0, 2, 5, 6, 7]
	y0 = _pybigtools_windows(path, [chroms[k] for k in keep], starts[keep], 5)
	_assert_bits_equal(y[keep], y0)

	if mode == 'python':
		assert counting.calls == []
	else:
		assert 2 in counting.flags and 0 in counting.flags
		assert (1 in counting.flags) == (mode == 'kernel')


@pytest.mark.parametrize("mode", ['kernel', 'python'])
def test_bigwig_file_short_read(pybigtools_bigwig, monkeypatch, mode):
	# A read that stops halfway leaves unread the blocks it does not reach
	# in full, and their windows go to pybigtools, on either path.
	pread = tangermeme.io._pread_into
	monkeypatch.setattr(tangermeme.io, '_pread_into', lambda fd, buffer,
		offset: pread(fd, buffer[:len(buffer) // 2], offset))
	_inflate_mode(monkeypatch, mode, 0)

	bw = pybigtools.open(str(pybigtools_bigwig))
	rng = numpy.random.default_rng(12)
	chroms, starts = _random_windows(rng, bw.chroms(), 300, 100)
	y, fallback = _reader_windows(pybigtools_bigwig, chroms, starts, 100, 2)
	y0 = _pybigtools_windows(pybigtools_bigwig, chroms, starts, 100)

	assert 0 < len(fallback) < len(starts)
	keep = numpy.setdiff1d(numpy.arange(len(starts)), fallback)
	_assert_bits_equal(y[keep], y0[keep])

	# The same windows are handed back on both paths.
	with monkeypatch.context() as m:
		_inflate_mode(m, 'python' if mode == 'kernel' else 'kernel', 0)
		_, fallback2 = _reader_windows(pybigtools_bigwig, chroms, starts, 100,
			2)

	assert fallback.tolist() == fallback2.tolist()


def test_bigwig_file_without_preadv(pybigtools_bigwig, monkeypatch):
	# Where os.preadv is missing, each run is read with os.pread and copied.
	import os

	monkeypatch.delattr(os, 'preadv', raising=False)
	bw = pybigtools.open(str(pybigtools_bigwig))
	rng = numpy.random.default_rng(13)
	chroms, starts = _random_windows(rng, bw.chroms(), 300, 700)
	y, fallback = _reader_windows(pybigtools_bigwig, chroms, starts, 700, 3)
	y0 = _pybigtools_windows(pybigtools_bigwig, chroms, starts, 700)

	assert len(fallback) == 0
	_assert_bits_equal(y, y0)


@pytest.mark.parametrize("mode", ['python', 'retry', 'some'])
@pytest.mark.parametrize("n_jobs", [1, 4])
@pytest.mark.parametrize("count_filter", [False, True])
def test_extract_loci_bigwig_reader_inflate_paths(reader_fasta,
	pybigtools_bigwig, tmp_path, monkeypatch, mode, n_jobs, count_filter):
	# extract_loci gives the same outputs whichever way the reader inflates
	# its blocks: after the loop, and in the loop under a count filter.
	path = tmp_path / "sections.bw"
	_write_raw_bigwig(path, {'chr1': 30000, 'chr2': 9000, 'chr3': 4000},
		[('chr1', 3, 10, 4, (0, [float(i) for i in range(2900)])),
		('chr2', 2, 0, 3, [(i * 5, float(i)) for i in range(1700)]),
		('chr3', 1, 0, 0, [(10, 3000, 2.5)])])

	monkeypatch.setattr(tangermeme.io, '_BIGWIG_MIN_WINDOWS', 1)
	monkeypatch.setattr(tangermeme.io, '_BIGWIG_BATCH_BLOCKS', 2)
	rng = numpy.random.default_rng(14)
	loci = _reader_loci(rng, 400)
	kwargs = dict(signals=[str(pybigtools_bigwig), str(path)],
		in_signals=[str(pybigtools_bigwig)], in_window=301, out_window=101,
		max_jitter=5, return_mask=True, n_jobs=n_jobs)
	if count_filter:
		kwargs.update(min_counts=1.0, target_idx=1)

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		y = extract_loci(loci, reader_fasta, **kwargs)

		counting = _counting_inflater(monkeypatch)
		bw = _BigWigFile.open(str(pybigtools_bigwig), pybigtools.open(str(
			pybigtools_bigwig)))
		_inflate_mode(monkeypatch, mode, bw.buffer_size)
		y1 = extract_loci(loci, reader_fasta, **kwargs)

	assert len(y) == len(y1) == 4
	for x, x1 in zip(y, y1):
		assert x.dtype == x1.dtype and x.shape == x1.shape
		assert torch.equal(x.view(torch.uint8) if x.dtype == torch.bool else
			x.view(torch.int32) if x.dtype == torch.float32 else x,
			x1.view(torch.uint8) if x1.dtype == torch.bool else
			x1.view(torch.int32) if x1.dtype == torch.float32 else x1)

	if mode == 'python':
		assert counting.calls == []
	else:
		assert len(counting.calls) > 0
