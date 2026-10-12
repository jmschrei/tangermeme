# test_io.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import numpy
import numba
import torch
import pytest
import pandas
import pathlib
import warnings

import figwig
import pyfaidx
import pybigtools

import tangermeme.io
import tangermeme.utils

from tangermeme.io import _interleave_loci
from tangermeme.io import _load_signals
from tangermeme.io import _load_exclusion_zones
from tangermeme.io import _extract_locus_signal
from tangermeme.io import _extract_signals
from tangermeme.io import _kept_loci
from tangermeme.io import _read_fasta_windows
from tangermeme.io import _read_fasta_windows_mmap

from tangermeme.io import read_meme
from tangermeme.io import extract_loci
from tangermeme.io import read_vcf
from tangermeme.io import one_hot_to_fasta

from tangermeme.utils import one_hot_encode
from tangermeme.utils import TangermemeWarning

from .bigwig_writer import write_raw_bigwig

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

	vals = bw[0].read("chr1", [0], 20)[0]

	assert type(vals) == numpy.ndarray
	assert vals.shape == (20,)
	assert_array_almost_equal(vals, [
		0.407911, 1.343698, 2.955252, 0.897452, 0.928617, 1.562161,
		0.662164, 1.387003, 0.963338, 1.988053, 1.373694, 1.417226,
		1.202522, 0.829855, 1.740464, 1.479131, 0.495755, 0.129613,
		0.111225, 0.772133])


def test_load_signals_multi_values():
	bw = _load_signals(["tests/data/test.bw", "tests/data/test2.bw"])

	vals = bw[0].read("chr1", [0], 10)[0]
	assert type(vals) == numpy.ndarray
	assert vals.shape == (10,)
	assert_array_almost_equal(vals, [
		0.407911, 1.343698, 2.955252, 0.897452, 0.928617, 1.562161,
		0.662164, 1.387003, 0.963338, 1.988053])

	vals = bw[1].read("chr1", [0], 10)[0]
	assert type(vals) == numpy.ndarray
	assert vals.shape == (10,)
	assert_array_almost_equal(vals, [
		0.210908, 1.711426, 0.292976, 0.948357, 1.946163, 0.806502,
		0.342074, 0.386286, 0.655825, 0.257574])


def test_load_signals_nan():
	bw = _load_signals(["tests/data/test3.bw"])

	vals = bw[0].read("chr1", [0], 40, missing=numpy.nan)[0]
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
	# A bigWig the caller already opened with pybigtools is used as is, and is
	# deprecated.
	bw = pybigtools.open("tests/data/test.bw")
	with pytest.warns(FutureWarning, match="pybigtools"):
		signals = _load_signals([bw])

	assert len(signals) == 1
	assert signals[0] is bw


def test_load_signals_mixed():
	signal = {'chr1': numpy.zeros(6)}
	bw = _load_signals(("tests/data/test.bw", signal))

	assert isinstance(bw, list)
	assert isinstance(bw[0], figwig.BigWigReader)
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


@pytest.fixture
def bigbed(tmp_path):
	path = str(tmp_path / "peaks.bigBed")
	pybigtools.open(path, "w").write({'chr1': 284, 'chr2': 211},
		iter([('chr1', 10, 50, 'a\t0\t+'), ('chr2', 20, 80, 'b\t0\t-')]))
	return path


def test_load_signals_raises_bigbed(bigbed):
	# figwig refuses a bigBed path. pybigtools opens a bigBed, and reading
	# values from one used to panic, raising a pyo3 PanicException, which is
	# not an Exception.
	with pytest.raises(ValueError, match="bigBed"):
		_load_signals(["tests/data/test.bw", bigbed])

	with pytest.warns(FutureWarning), pytest.raises(ValueError,
			match="signal 1 is a bigBed"):
		_load_signals(["tests/data/test.bw", pybigtools.open(bigbed)])


@pytest.mark.parametrize("which", ["signals", "in_signals"])
def test_extract_loci_raises_bigbed(bigbed, which):
	with pytest.raises(ValueError, match="bigBed"):
		extract_loci("tests/data/test.bed", "tests/data/test.fa",
			**{which: [bigbed]}, in_window=8, out_window=10)


def test_load_signals_empty_dict():
	# A dict without chromosomes is a signal that has none of them; it used to
	# raise an IndexError.
	assert _load_signals([{}]) == [{}]


@pytest.mark.parametrize("which", ["signals", "in_signals"])
def test_extract_loci_empty_dict_signal(which):
	# Every locus is zero and warns once, as a chromosome missing from a dict
	# does.
	kwargs = {which: [{}], "in_window": 8, "out_window": 10}
	with warnings.catch_warnings(record=True) as record:
		warnings.simplefilter("always")
		_, values = extract_loci("tests/data/test.bed", "tests/data/test.fa",
			**kwargs)

	width = 10 if which == "signals" else 8
	assert values.shape == (5, 1, width)
	assert torch.equal(values, torch.zeros(5, 1, width))
	assert [w.category for w in record] == [TangermemeWarning] * 5
	assert all("is not in the signal dictionary" in str(w.message)
		for w in record)


@pytest.mark.filterwarnings("ignore::tangermeme.utils.TangermemeWarning")
def test_extract_loci_empty_dict_signal_counts():
	# An empty dict as the target of min_counts has counts of zero.
	X, y, mask = extract_loci("tests/data/test.bed", "tests/data/test.fa",
		[{}, "tests/data/test.bw"], in_window=8, out_window=10, min_counts=0.0,
		max_counts=0.0, return_mask=True)

	assert mask.tolist() == [True] * 5
	assert torch.equal(y[:, 0], torch.zeros(5, 10))

	with pytest.raises(ValueError, match="No loci remain"):
		extract_loci("tests/data/test.bed", "tests/data/test.fa", [{}],
			in_window=8, out_window=10, min_counts=0.5)


def test_load_signals_figwig():
	# Paths, pathlib.Paths and figwig readers become figwig readers, the
	# readers passed as they are, and dicts pass through.
	reader = figwig.BigWigReader("tests/data/test2.bw")
	signals = _load_signals(["tests/data/test.bw",
		pathlib.Path("tests/data/test3.bw"), reader, {'chr1': numpy.zeros(5)}])

	assert isinstance(signals[0], figwig.BigWigReader)
	assert isinstance(signals[1], figwig.BigWigReader)
	assert signals[2] is reader
	assert isinstance(signals[3], dict)


def test_load_signals_raises_url():
	# Only local files are read.
	with pytest.raises(ValueError, match="URL"):
		_load_signals(["https://example.com/signal.bw"])


def test_load_signals_raises_figwig_errors():
	# A missing file and a file that is not a bigWig raise figwig's errors.
	with pytest.raises(FileNotFoundError):
		_load_signals(["tests/data/missing.bw"])

	with pytest.raises(ValueError, match="not a bigWig"):
		_load_signals(["tests/data/test.bed"])



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


# Windows inside chromosomes, at their ends, past the end of chr1, and on
# chromosomes that test3.bw, which only has chr1, does not have.
SIGNAL_CHROMS = numpy.array(['chr1', 'chr1', 'chr2', 'chr1', 'chr3', 'chr6'])
SIGNAL_STARTS = numpy.array([0, 274, 100, 280, 3, 72])


def _per_locus_signals(paths, width):
	signals = [pybigtools.open(path) for path in paths]
	return numpy.array([_extract_locus_signal(signals, chrom, start,
		start + width) for chrom, start in zip(SIGNAL_CHROMS, SIGNAL_STARTS)])


def test_extract_signals_figwig():
	paths = ["tests/data/test.bw", "tests/data/test2.bw", "tests/data/test3.bw"]

	with warnings.catch_warnings(record=True) as record:
		warnings.simplefilter("always")
		expected = _per_locus_signals(paths, 8)

	with warnings.catch_warnings(record=True) as figwig_record:
		warnings.simplefilter("always")
		values = _extract_signals(_load_signals(paths), SIGNAL_CHROMS,
			SIGNAL_STARTS, 8, 2)

	assert values.dtype == numpy.float32
	assert values.shape == (6, 3, 8)
	assert values.flags.c_contiguous
	assert values.tobytes() == expected.tobytes()

	# One warning per locus that test3.bw lacks, with the same message.
	assert len(figwig_record) == 3
	assert all(w.category is TangermemeWarning for w in figwig_record)
	assert sorted(str(w.message) for w in figwig_record) == sorted(
		str(w.message) for w in record)


def test_extract_signals_mixed(dict_signal):
	# figwig bigWigs, pybigtools bigWigs and dicts in one list each give the
	# values they give alone.
	signals = [figwig.BigWigReader("tests/data/test.bw"), dict_signal,
		pybigtools.open("tests/data/test2.bw")]

	values = _extract_signals(signals, SIGNAL_CHROMS[:4], SIGNAL_STARTS[:4], 8,
		1)

	assert values.flags.c_contiguous
	assert values[:, 0].tobytes() == _per_locus_signals(["tests/data/test.bw"],
		8)[:4, 0].tobytes()
	assert values[:, 2].tobytes() == _per_locus_signals(["tests/data/test2.bw"],
		8)[:4, 0].tobytes()
	for i, (chrom, start) in enumerate(zip(SIGNAL_CHROMS[:4],
			SIGNAL_STARTS[:4])):
		assert values[i, 1].tobytes() == _extract_locus_signal([dict_signal],
			chrom, start, start + 8)[0].tobytes()


def test_extract_signals_figwig_error_raises(monkeypatch):
	# A file figwig does not read, such as one with a corrupt data block,
	# raises figwig's error.
	def corrupt(*args, **kwargs):
		raise ValueError("corrupt data block")

	signals = _load_signals(["tests/data/test.bw", "tests/data/test2.bw"])
	monkeypatch.setattr(figwig, "read_bigwig", corrupt)

	with pytest.raises(ValueError, match="corrupt data block"):
		_extract_signals(signals, SIGNAL_CHROMS, SIGNAL_STARTS, 8, 2)


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
	# bigWigs opened with pybigtools are deprecated, and still read.
	bws = [pybigtools.open("tests/data/test.bw"),
		pybigtools.open("tests/data/test2.bw")]

	with pytest.warns(FutureWarning, match="pybigtools"):
		X, y, controls = extract_loci("tests/data/test.bed",
			"tests/data/test.fa", bws, bws, in_window=10, out_window=10)

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


def test_extract_loci_bigwig_missing_chrom_warns_once_per_locus():
	# figwig's own warning about the missing chromosome is replaced by
	# tangermeme's, so the two chr2 loci give two warnings and no others.
	with warnings.catch_warnings(record=True) as record:
		warnings.simplefilter("always")
		extract_loci("tests/data/test.bed", "tests/data/test.fa",
			["tests/data/test3.bw"], in_window=8, out_window=10)

	assert [w.category for w in record] == [TangermemeWarning] * 2


@pytest.mark.parametrize("kwargs", [{}, {'max_jitter': 3, 'in_window': 7,
	'out_window': 5}, {'min_counts': 10.0, 'return_mask': True},
	{'max_counts': 12.0, 'target_idx': 1, 'n_loci': 2, 'return_mask': True}])
def test_extract_loci_reads_bigwig_paths_with_figwig(monkeypatch, kwargs):
	# bigWig paths are read by figwig and give the same values as bigWigs
	# opened with pybigtools, which are read one locus at a time.
	calls = []
	read_bigwig = figwig.read_bigwig

	def spy(*args, **kw):
		calls.append(kw['n_jobs'])
		return read_bigwig(*args, **kw)

	monkeypatch.setattr(figwig, "read_bigwig", spy)
	kwargs = {'in_window': 8, 'out_window': 10, **kwargs}

	paths = ["tests/data/test.bw", "tests/data/test2.bw"]
	result = extract_loci("tests/data/test.bed", "tests/data/test.fa", paths,
		paths[::-1], n_jobs=3, **kwargs)
	assert len(calls) > 0 and set(calls) == {3}

	# The same bigWigs opened with pybigtools, which is deprecated, give the
	# same values.
	bws = [pybigtools.open(path) for path in paths]
	with pytest.warns(FutureWarning, match="pybigtools"):
		expected = extract_loci("tests/data/test.bed", "tests/data/test.fa",
			bws, bws[::-1], **kwargs)

	for tensor, expected_tensor in zip(result, expected):
		assert tensor.dtype == expected_tensor.dtype
		assert tensor.is_contiguous()
		assert torch.equal(tensor, expected_tensor)


@pytest.mark.parametrize("signals, kwargs, windows", [
	# test3.bw is not the target: both chr2 loci are examined and warn,
	# although the counts on test.bw remove the first.
	(["tests/data/test.bw", "tests/data/test3.bw"], {'min_counts': 10.0},
		[(35, 45), (45, 55)]),
	# test3.bw is the target: each chr2 locus warns once, although the
	# target is read for the counts and again for the values.
	(["tests/data/test3.bw", "tests/data/test.bw"], {'min_counts': 0.0},
		[(35, 45), (45, 55)]),
	# Loci after the n_loci-th kept one are not examined and do not warn.
	(["tests/data/test3.bw"], {'min_counts': 0.0, 'n_loci': 4}, [(35, 45)]),
])
def test_extract_loci_counts_warn_once_per_examined_locus(signals, kwargs,
	windows):
	# Under min_counts, every signal warns once at each locus examined before
	# n_loci loci are kept, for chromosomes it does not have, as it did when
	# the signals were read one locus at a time. test3.bw only has chr1.
	with warnings.catch_warnings(record=True) as record:
		warnings.simplefilter("always")
		extract_loci("tests/data/test.bed", "tests/data/test.fa", signals,
			in_window=8, out_window=10, **kwargs)

	assert [w.category for w in record] == [TangermemeWarning] * len(windows)
	assert sorted(str(w.message) for w in record) == ["chr2 {} {} not valid "
		"bigwig indexes. Using zeros instead.".format(start, end)
		for start, end in windows]


def test_extract_loci_float_coordinates_raise():
	# Coordinates read as floats raise figwig's TypeError.
	loci = pandas.DataFrame({0: ['chr1', 'chr2'], 1: [10.0, 25.0],
		2: [30.0, 55.0]})

	with pytest.raises(TypeError, match="integers"):
		extract_loci(loci, "tests/data/test.fa", ["tests/data/test.bw"],
			in_window=8, out_window=10)


@pytest.mark.parametrize("sequences", ["path", "fasta", "dict"])
def test_extract_loci_float_coordinates_raise_without_signals(sequences):
	# Without signals, float coordinates raise a TypeError where the
	# sequence window is sliced.
	loci = pandas.DataFrame({0: ['chr4'], 1: [100.0], 2: [111.0]})
	if sequences == "fasta":
		sequences = pyfaidx.Fasta("tests/data/test.fa")
	elif sequences == "dict":
		sequences = {'chr4': numpy.zeros((4, 240), dtype=numpy.int8)}
	else:
		sequences = "tests/data/test.fa"

	with pytest.raises(TypeError):
		extract_loci(loci, sequences, in_window=11)


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


@pytest.mark.parametrize("exclusion", [False, True])
@pytest.mark.parametrize("left, right", [(0, 0), (5, 6), (300, 301)])
def test_kept_loci_matches_loop(exclusion, left, right):
	# Integer coordinates, checked together as int64, give what the loop over
	# loci gives for the same coordinates as Python ints, at random loci on
	# and off their chromosomes and around random excluded chunks.
	rng = numpy.random.default_rng(0)
	lengths = {'chr1': 284, 'chr2': 211, 'chr7': 2000}
	starts = rng.integers(-20, 2020, 500)
	loci = pandas.DataFrame({'chrom': rng.choice(list(lengths), 500),
		'start': starts, 'end': starts + rng.integers(0, 40, 500)})

	zones = None
	if exclusion:
		zones = {chrom: rng.random(length // 100 + 1) < 0.15
			for chrom, length in lengths.items()}

	idxs, mids = _kept_loci(loci, lengths, zones, left, right)
	idxs0, mids0 = _kept_loci(loci.astype({'start': object, 'end': object}),
		lengths, zones, left, right)

	assert idxs.dtype == mids.dtype == numpy.int64
	assert 0 < len(idxs) < len(loci)
	assert idxs.tolist() == idxs0.tolist()
	assert mids.tolist() == mids0.tolist()


def test_extract_loci_bigwig_readers():
	# figwig readers as signals and in_signals give what their paths give,
	# and can be read again.
	paths = ["tests/data/test.bw", "tests/data/test2.bw", "tests/data/test3.bw"]
	readers = [figwig.BigWigReader(path) for path in paths]
	kwargs = dict(in_window=8, out_window=10, max_jitter=2, return_mask=True)

	expected = extract_loci("tests/data/test.bed", "tests/data/test.fa", paths,
		paths[::-1], **kwargs)
	for _ in range(2):
		result = extract_loci("tests/data/test.bed", "tests/data/test.fa",
			readers, readers[::-1], **kwargs)
		for tensor, expected_tensor in zip(result, expected):
			assert torch.equal(tensor, expected_tensor)


@pytest.mark.parametrize("n_jobs", [1, 2, -1, numpy.int64(3)])
def test_extract_loci_n_jobs(n_jobs):
	paths = ["tests/data/test.bw", "tests/data/test2.bw"]
	X0, y0, y_in0 = extract_loci("tests/data/test.bed", "tests/data/test.fa",
		paths, paths, in_window=8, out_window=10, n_jobs=1)
	X, y, y_in = extract_loci("tests/data/test.bed", "tests/data/test.fa",
		paths, paths, in_window=8, out_window=10, n_jobs=n_jobs)

	assert torch.equal(X, X0)
	assert torch.equal(y, y0)
	assert torch.equal(y_in, y_in0)


@pytest.mark.parametrize("n_jobs, error", [(0, ValueError), (-2, ValueError),
	(1.5, TypeError), (True, TypeError), ("2", TypeError), (None, TypeError)])
def test_extract_loci_raises_n_jobs(n_jobs, error):
	with pytest.raises(error, match="n_jobs"):
		extract_loci("tests/data/test.bed", "tests/data/test.fa",
			["tests/data/test.bw"], in_window=8, out_window=10, n_jobs=n_jobs)


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
		chroms, starts, values = [], [], []
		for chrom in sorted(track):
			bases = numpy.flatnonzero(~numpy.isnan(track[chrom]))
			chroms += [chrom] * len(bases)
			starts.append(bases)
			values.append(track[chrom][bases])

		starts = numpy.concatenate(starts)
		figwig.write_bigwig(filename, {chrom: REFERENCE_CHROMS[chrom] for chrom
			in sorted(track)}, chroms, starts, numpy.concatenate(values),
			ends=starts + 1, missing=numpy.nan)

		tracks.append(track)
		bigwigs.append(filename)

	# The same tracks in layouts that pybigtools does not write, so that each
	# kind of section is read against the reference: varStep sections,
	# fixedStep sections over each run of bases with values, and bedGraph
	# sections without compression.
	layouts = {"varstep": [], "fixedstep": [], "uncompressed": []}
	for t, track in enumerate(tracks):
		chroms = {chrom: REFERENCE_CHROMS[chrom] for chrom in sorted(track)}
		sections = {name: [] for name in layouts}
		for chrom in chroms:
			values = track[chrom]
			bases = numpy.flatnonzero(~numpy.isnan(values))
			for k in range(0, len(bases), 200):
				block = bases[k:k+200]
				sections["varstep"].append((chrom, 2, 0, 1, [(int(b),
					float(values[b])) for b in block]))
				sections["uncompressed"].append((chrom, 1, 0, 0, [(int(b),
					int(b) + 1, float(values[b])) for b in block]))
			for run in numpy.split(bases, numpy.flatnonzero(numpy.diff(bases)
					> 1) + 1):
				sections["fixedstep"].append((chrom, 3, 1, 1, (int(run[0]),
					[float(values[b]) for b in run])))

		for name in layouts:
			filename = str(path / "track{}.{}.bw".format(t, name))
			write_raw_bigwig(filename, chroms, sections[name],
				compress=name != "uncompressed")
			layouts[name].append(filename)

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
		'tracks': tracks, 'bigwigs': bigwigs, 'layouts': layouts, 'loci': loci,
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

	_check_against_reference(data, sequences, signal_source, mode, in_window,
		out_window, max_jitter, filters)


@pytest.mark.filterwarnings("ignore::tangermeme.utils.TangermemeWarning")
@pytest.mark.parametrize("source", ["varstep", "fixedstep", "uncompressed"])
@pytest.mark.parametrize("filters", ["none", "counts_target_idx", "combined"])
def test_extract_loci_reference_layouts(reference_genome, monkeypatch, source,
	filters):
	# The reference tracks written as varStep, fixedStep and uncompressed
	# bedGraph sections, which pybigtools does not write, give the reference's
	# values, and are read by figwig without falling back to pybigtools.
	errors = []
	read_bigwig = figwig.read_bigwig

	def spy(*args, **kwargs):
		try:
			return read_bigwig(*args, **kwargs)
		except ValueError:
			errors.append(1)
			raise

	monkeypatch.setattr(figwig, "read_bigwig", spy)
	data = reference_genome
	_check_against_reference(data, data['fasta'], data['layouts'][source],
		"both", 7, 5, 3, filters)
	assert errors == []


def _check_against_reference(data, sequences, signal_source, mode, in_window,
	out_window, max_jitter, filters):
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


@pytest.mark.filterwarnings("ignore::tangermeme.utils.TangermemeWarning")
@pytest.mark.parametrize("filters, n_values", [("combined", 1),
	("combined", 33), ("counts_target_idx", 33)])
def test_extract_loci_counts_in_groups(monkeypatch, reference_genome, filters,
	n_values):
	# min_counts and max_counts are measured on groups of loci, stopping once
	# n_loci are kept. Groups of one locus and of three give what one group of
	# every locus gives.
	import tangermeme.io

	data = reference_genome
	kwargs = REFERENCE_FILTERS[filters](11, data['exclusion'])
	args = (data['loci'], data['fasta'], data['bigwigs'][:2],
		data['bigwigs'][2:])

	expected = extract_loci(*args, in_window=7, out_window=5, max_jitter=3,
		return_mask=True, **kwargs)

	monkeypatch.setattr(tangermeme.io, "_COUNT_VALUES", n_values)
	result = extract_loci(*args, in_window=7, out_window=5, max_jitter=3,
		return_mask=True, **kwargs)

	for tensor, expected_tensor in zip(result, expected):
		assert torch.equal(tensor, expected_tensor)


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






# Lines a BED file may carry before or between its rows, which are skipped.
BED_HEADERS = {
	'track': 'track name=peaks description="some peaks"\n',
	'browser': 'browser position chr1:1-100\nbrowser hide all\n',
	'column_names': '#chrom\tstart\tend\n',
	'tenx': '# id=pbmc\n# description=PBMC from a donor\n#\n',
}


@pytest.mark.parametrize("header, compressed", [("track", False),
	("browser", False), ("column_names", False), ("tenx", True)])
def test_interleave_loci_skips_bed_header_lines(tmp_path, header, compressed):
	# The rows of a file with a header are those of the file without it, with
	# integer coordinates. A track line used to raise a TypeError on string
	# arithmetic, and # lines left the coordinates as floats.
	import gzip

	rows = "chr1\t10\t30\nchr2\t25\t55\nchr2\t35\t65\n"
	opener = gzip.open if compressed else open
	suffix = ".bed.gz" if compressed else ".bed"

	with opener(tmp_path / ("plain" + suffix), "wt") as f:
		f.write(rows)
	with opener(tmp_path / ("header" + suffix), "wt") as f:
		lines = rows.splitlines(keepends=True)
		f.write(BED_HEADERS[header] + lines[0] + "# between rows\n" +
			"".join(lines[1:]))

	result = _interleave_loci(str(tmp_path / ("header" + suffix)))
	expected = _interleave_loci(str(tmp_path / ("plain" + suffix)))

	pandas.testing.assert_frame_equal(result, expected)
	assert result['start'].dtype == numpy.int64
	assert result['end'].dtype == numpy.int64


def test_extract_loci_bed_header_lines(tmp_path):
	# A BED10 file with a track line and # lines gives what the same file
	# without them gives, with summits and bigWig signals.
	rows = "chr1\t10\t30\t.\t0\t.\t0\t0\t0\t4\nchr2\t25\t55\t.\t0\t.\t0\t0\t0\t20\n"
	(tmp_path / "plain.bed").write_text(rows)
	(tmp_path / "header.bed").write_text(BED_HEADERS['track'] +
		BED_HEADERS['tenx'] + rows)

	kwargs = dict(in_window=8, out_window=10, summits=True, return_mask=True)
	result = extract_loci(str(tmp_path / "header.bed"), "tests/data/test.fa",
		["tests/data/test.bw"], **kwargs)
	expected = extract_loci(str(tmp_path / "plain.bed"), "tests/data/test.fa",
		["tests/data/test.bw"], **kwargs)

	for tensor, expected_tensor in zip(result, expected):
		assert torch.equal(tensor, expected_tensor)


##
# bigWig layouts that reach tangermeme's own handling: a chromosome shorter
# than the FASTA's, infinities and NaN, a chromosome absent from the file or
# without data, and malformed files. Each must give what a dict of the values
# written into it gives, and what the same file opened with pybigtools gives,
# which is read one locus at a time; or raise, where figwig does not read it.
##


# The chromosomes of tests/data/test.fa, without chr5, whose Z raises.
LAYOUT_CHROMS = {'chr1': 284, 'chr2': 211, 'chr3': 126, 'chr4': 240,
	'chr6': 80}
SPECIAL_VALUES = [float('nan'), float('inf'), float('-inf'), -0.0, 1e-45,
	float(numpy.finfo(numpy.float32).max), -float(numpy.finfo(
	numpy.float32).max), 1.0, -2.5]


def _intervals(rng, length, values=None):
	"""Sorted, non-overlapping (start, end, value) intervals on [0, length)."""

	rows, pos = [], int(rng.randint(0, 4))
	while True:
		width = int(rng.randint(1, 9))
		if pos + width > length:
			return rows
		value = float(numpy.float32(rng.uniform(-1, 5))) if values is None \
			else values[len(rows) % len(values)]
		rows.append((pos, pos + width, value))
		pos += width + int(rng.randint(0, 7))


LAYOUTS = ["short_chrom", "special_values", "absent_chrom",
	"chrom_without_data", "zero_length", "overlapping", "unsorted_blocks"]

# The layouts figwig raises for, at least for windows across the defect.
FIGWIG_RAISES = ("overlapping", "unsorted_blocks")


def _layout(name, rng):
	"""The chromosome lengths and bedGraph sections of one layout, and the
	values it holds by chromosome, NaN where it has none, or None for a
	layout figwig raises for."""

	chroms, sections = dict(LAYOUT_CHROMS), []
	if name == "short_chrom":
		# Shorter than the FASTA's chr1, which is 284 bases long.
		chroms["chr1"] = 200
	elif name == "absent_chrom":
		del chroms["chr3"]

	for chrom, length in chroms.items():
		if name == "chrom_without_data" and chrom == "chr4":
			continue

		rows = _intervals(rng, length, SPECIAL_VALUES if name ==
			"special_values" else None)
		if name == "overlapping":
			middle = len(rows) // 2
			rows.insert(middle, (rows[middle][0] + 1, rows[middle][1] + 4, 9.0))
		elif name == "zero_length":
			rows = [(s, s if k % 5 == 0 else e, v) for k, (s, e, v) in
				enumerate(rows)]

		blocks = [(chrom, 1, 0, 0, rows[k:k+10]) for k in range(0, len(rows),
			10)]
		if name == "unsorted_blocks":
			blocks[0], blocks[-1] = blocks[-1], blocks[0]
		sections += blocks

	if name in FIGWIG_RAISES:
		return chroms, sections, None

	values = {chrom: numpy.full(length, numpy.nan, dtype=numpy.float32)
		for chrom, length in chroms.items()}
	for chrom, _, _, _, rows in sections:
		for s, e, v in rows:
			values[chrom][s:e] = v
	return chroms, sections, values


@pytest.fixture(scope="module")
def layout_bigwigs(tmp_path_factory):
	path = tmp_path_factory.mktemp("layouts")
	rng = numpy.random.RandomState(0)
	layouts = {}
	for name in LAYOUTS:
		chroms, sections, values = _layout(name, rng)
		filename = str(path / "{}.bw".format(name))
		write_raw_bigwig(filename, chroms, sections)
		layouts[name] = (filename, values)
	return layouts


@pytest.fixture(scope="module")
def layout_loci():
	# Loci across the chromosomes, at their ends, and at the end of chr1 in
	# the short_chrom layout, whose chr1 is 200 bases long.
	rng = numpy.random.RandomState(1)
	rows = []
	for chrom, length in LAYOUT_CHROMS.items():
		for center in list(rng.randint(0, length, size=12)) + [0, 1, length - 1,
				length]:
			width = int(rng.randint(1, 20))
			start = max(int(center) - width // 2, 0)
			rows.append((chrom, start, start + width))
	rows += [('chr1', c, c + 2) for c in (185, 195, 199, 200, 205, 230)]
	return pandas.DataFrame(rows)


LAYOUT_CONFIGS = {
	# Windows past the ends of chromosomes, with jitter, in signals and in
	# in_signals.
	'windows': {'in_window': 20, 'out_window': 10, 'max_jitter': 3,
		'return_mask': True, 'in_signals': True},
	# Counts measured on the layout, in groups, stopping at n_loci.
	'counts': {'in_window': 12, 'out_window': 6, 'min_counts': 0.5,
		'target_idx': -1, 'n_loci': 12, 'return_mask': True,
		'with_test_bw': True},
}


def _layout_call(loci, signal, config):
	kwargs = dict(LAYOUT_CONFIGS[config])
	in_signals = [signal] if kwargs.pop('in_signals', False) else None
	signals = [signal]
	if kwargs.pop('with_test_bw', False):
		signals = ["tests/data/test.bw", signal]

	with warnings.catch_warnings(record=True) as record:
		warnings.simplefilter("always")
		result = extract_loci(loci, "tests/data/test.fa", signals, in_signals,
			**kwargs)

	# A bigWig opened with pybigtools warns that it is deprecated.
	messages = sorted((w.category.__name__, str(w.message)) for w in record
		if w.category is not FutureWarning)
	return result, messages


@pytest.mark.parametrize("config", list(LAYOUT_CONFIGS))
@pytest.mark.parametrize("layout", LAYOUTS)
def test_extract_loci_bigwig_layouts(layout_bigwigs, layout_loci, monkeypatch,
	layout, config):
	filename, values = layout_bigwigs[layout]

	calls, errors = [], []
	read_bigwig = figwig.read_bigwig

	def spy(*args, **kwargs):
		calls.append(1)
		try:
			return read_bigwig(*args, **kwargs)
		except ValueError:
			errors.append(1)
			raise

	monkeypatch.setattr(figwig, "read_bigwig", spy)
	if layout in FIGWIG_RAISES:
		with pytest.raises(ValueError):
			_layout_call(layout_loci, filename, config)
		return

	result, messages = _layout_call(layout_loci, filename, config)
	monkeypatch.undo()

	assert len(calls) > 0 and errors == []

	expected, expected_messages = _layout_call(layout_loci,
		pybigtools.open(filename), config)
	assert messages == expected_messages
	for tensor, expected_tensor in zip(result, expected):
		assert tensor.dtype == expected_tensor.dtype
		assert tensor.is_contiguous()
		assert tensor.numpy().tobytes() == expected_tensor.numpy().tobytes()

	if values is not None:
		track = {chrom: v for chrom, v in values.items() if chrom in
			LAYOUT_CHROMS}
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			oracle, _ = _layout_call(layout_loci, track, config)
		for tensor, oracle_tensor in zip(result, oracle):
			assert tensor.numpy().tobytes() == oracle_tensor.numpy().tobytes()


def _without_prefix(tmp_path):
	# test.bw with its chromosomes named 1, 2, ... rather than chr1, chr2.
	reader = figwig.BigWigReader("tests/data/test.bw")
	chroms, starts, values = [], [], []
	for chrom, size in reader.chrom_sizes.items():
		track = reader.read(chrom, [0], size, missing=numpy.nan)[0]
		bases = numpy.flatnonzero(~numpy.isnan(track))
		chroms += [chrom[3:]] * len(bases)
		starts.append(bases)
		values.append(track[bases])

	path = str(tmp_path / "no_prefix.bw")
	starts = numpy.concatenate(starts)
	figwig.write_bigwig(path, {chrom[3:]: size for chrom, size in
		reader.chrom_sizes.items()}, chroms, starts, numpy.concatenate(values),
		ends=starts + 1, missing=numpy.nan)
	return path


@pytest.mark.parametrize("direction", ["bigwig", "fasta"])
def test_extract_loci_chromosome_naming_mismatch(tmp_path, direction):
	# chr1 against 1, in either file: every locus is on a chromosome the
	# bigWig lacks, so it is zero and warns once, as with pybigtools.
	if direction == "bigwig":
		signal, fasta, loci = _without_prefix(tmp_path), "tests/data/test.fa", \
			"tests/data/test.bed"
	else:
		records = pyfaidx.Fasta("tests/data/test.fa")
		fasta = str(tmp_path / "no_prefix.fa")
		with open(fasta, "w") as f:
			for chrom in records.keys():
				f.write(">{}\n{}\n".format(chrom[3:], str(records[chrom])))
		loci = pandas.read_csv("tests/data/test.bed", sep="\t", header=None)
		loci[0] = loci[0].str[3:]
		signal = "tests/data/test.bw"

	results = []
	for s in (signal, pybigtools.open(signal)):
		with warnings.catch_warnings(record=True) as record:
			warnings.simplefilter("always")
			X, y = extract_loci(loci, fasta, [s], in_window=8, out_window=10)
		results.append((X, y, sorted(str(w.message) for w in record)))

	(X, y, messages), (X0, y0, messages0) = results
	messages0 = [m for m in messages0 if "deprecated" not in m]
	assert torch.equal(X, X0) and torch.equal(y, y0)
	assert torch.equal(y, torch.zeros(5, 1, 10))
	assert messages == messages0
	assert len(messages) == 5


@pytest.mark.filterwarnings("ignore::tangermeme.utils.TangermemeWarning")
@pytest.mark.parametrize("kwargs", [{}, {'min_counts': 10.0, 'target_idx': 2},
	{'max_counts': 12.0, 'target_idx': -2, 'n_loci': 2, 'return_mask': True},
	{'min_counts': 0.0, 'target_idx': 3, 'max_jitter': 2}])
def test_extract_loci_mixed_signal_kinds(dict_signal, kwargs):
	# A path, a pathlib.Path, a figwig reader and a dict in one call, in
	# signals and in_signals, give what the same signals give when every
	# bigWig is opened with pybigtools, which are read one locus at a time.
	def signals(opened):
		if opened:
			return [pybigtools.open("tests/data/test{}.bw".format(k)) for k in
				("", "2", "3")] + [dict_signal]
		return ["tests/data/test.bw", pathlib.Path("tests/data/test2.bw"),
			figwig.BigWigReader("tests/data/test3.bw"), dict_signal]

	result = extract_loci("tests/data/test.bed", "tests/data/test.fa",
		signals(False), signals(False)[::-1], in_window=8, out_window=10,
		**kwargs)
	with pytest.warns(FutureWarning):
		expected = extract_loci("tests/data/test.bed", "tests/data/test.fa",
			signals(True), signals(True)[::-1], in_window=8, out_window=10,
			**kwargs)

	for tensor, expected_tensor in zip(result, expected):
		assert tensor.numpy().tobytes() == expected_tensor.numpy().tobytes()


def test_extract_loci_in_signals_missing_chrom_warns_in_input_windows():
	# in_signals warn with the input window of each locus on a chromosome
	# they lack, once per locus, as with pybigtools. test3.bw only has chr1.
	results = []
	for signal in ("tests/data/test3.bw", pybigtools.open("tests/data/test3.bw")):
		with warnings.catch_warnings(record=True) as record:
			warnings.simplefilter("always")
			extract_loci("tests/data/test.bed", "tests/data/test.fa",
				in_signals=[signal], in_window=8, out_window=10)
		results.append(sorted(str(w.message) for w in record
			if w.category is TangermemeWarning))

	assert results[0] == results[1] == ["chr2 36 44 not valid bigwig indexes. "
		"Using zeros instead.", "chr2 46 54 not valid bigwig indexes. Using "
		"zeros instead."]


###
# A fasta opened from a path is read through its .fai index from a memory map
# of the file. These tests check that each window equals what pyfaidx returns
# for it: across line ends of every width and kind, at both ends of every
# record, and at the end of a file that lacks a final line end. When the
# bytes are not what pyfaidx would return unchanged, or the file cannot be
# mapped, pyfaidx reads the windows instead, so the output or the error is
# the one a pyfaidx.Fasta object gives.
###


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


@pytest.mark.parametrize("width", [1, 7, 60])
@pytest.mark.parametrize("newline, final_newline", [("\n", True),
	("\r\n", False)])
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
@pytest.mark.parametrize("in_window", [3, 8])
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


@pytest.mark.parametrize("mids, ignore, error, match", [
	([9, 11], ['N'], ValueError, "Encountered character"),
	([11, 9], ['N'], UnicodeDecodeError, "utf-8"),
	([9, 11], ['N', 'A'], ValueError, "in the alphabet and also"),
	([11, 9], ['N', 'A'], UnicodeDecodeError, "utf-8"),
], ids=["whole_first", "split_first", "overlap_whole_first",
	"overlap_split_first"])
def test_extract_loci_fasta_error_order(tmp_path, mids, ignore, error, match):
	# The window around base 9 holds the two bytes of an é, which pyfaidx
	# decodes to a character in neither the alphabet nor `ignore`, and the
	# window around base 11 holds only its second byte, which pyfaidx cannot
	# decode. The error is the one that reading and encoding each window in
	# turn gives, as extract_loci did one locus at a time: that of the first
	# window, from a path and from a pyfaidx.Fasta object.
	path = str(tmp_path / "genome.fa")
	with open(path, "wb") as handle:
		handle.write(b">chr1\nACGTAC\nACG" + "é".encode('utf8') + b"A\nACGTAC\n"
			b"ACGTAC\n")

	with open(path + ".fai", "w") as handle:
		handle.write("chr1\t24\t6\t6\t7\n")

	loci = pandas.DataFrame([('chr1', mid, mid + 1) for mid in mids])
	with pytest.raises(error, match=match):
		extract_loci(loci, path, in_window=3, ignore=ignore)

	fasta = pyfaidx.Fasta(path)
	with pytest.raises(error, match=match):
		extract_loci(loci, fasta, in_window=3, ignore=ignore)
	fasta.close()


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


@pytest.mark.parametrize("alphabet", [['A', 'C', 'G', 'T'],
	['A', 'C', 'G', 'T', 'é']])
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


@pytest.mark.parametrize("cause", ["index", "negative_start", "mmap_fails"])
def test_read_fasta_windows_mmap_declines(tmp_path, monkeypatch, cause):
	# The memory map is not read, and the windows go to pyfaidx, when the
	# index gives fewer bytes than bases per line, a window starts before its
	# record, or the file cannot be mapped. extract_loci then gives what a
	# pyfaidx.Fasta object gives, here the error for windows of different
	# lengths that the bad index leads pyfaidx to return.
	genome = {'chr1': 'ACGTAC' * 4}
	path = str(tmp_path / "genome.fa")
	_write_fasta(path, genome, 6)
	windows = [('chr1', 0), ('chr1', 10)]

	if cause == "index":
		with open(path + ".fai", "w") as handle:
			handle.write("chr1\t24\t6\t7\t6\n")
	elif cause == "negative_start":
		windows = [('chr1', -1), ('chr1', 10)]
	else:
		def no_map(*args, **kwargs):
			raise OSError("cannot map")

		monkeypatch.setattr(tangermeme.io.mmap, "mmap", no_map)

	fasta = pyfaidx.Fasta(path)
	assert _read_fasta_windows_mmap(fasta, windows, 4, ['A', 'C', 'G', 'T'],
		['N']) is None
	fasta.close()

	loci = pandas.DataFrame([('chr1', mid, mid + 1) for mid in (2, 7, 12)])
	_same_as_pyfaidx_object(loci, path, in_window=4)


def test_extract_loci_fasta_zero_window(tmp_path):
	# A window of no bases gives a sequence of no positions per locus, from a
	# path, which the memory map leaves to pyfaidx, as from a pyfaidx.Fasta.
	genome = _fasta_genome()
	path = str(tmp_path / "genome.fa")
	_write_fasta(path, genome, 11)
	loci = _every_window(genome, 8).iloc[::10]

	X = _same_as_pyfaidx_object(loci, path, in_window=0)
	assert X.shape == (len(loci), 4, 0)
	assert X.dtype == torch.int8


###
# n_jobs: the sequences of a fasta file are one-hot encoded in blocks of
# rows on numba's threads, which changes no output and no error.
###


@pytest.mark.parametrize("n_jobs", [3, 1000])
def test_extract_loci_n_jobs_encoding(tmp_path, n_jobs):
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
	# With signals, in_signals and a mask, the outputs are those of one
	# thread.
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


@pytest.mark.parametrize("line", [180, 3])
def test_extract_loci_n_jobs_read_through_pyfaidx(tmp_path, line):
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
	_same_as_pyfaidx_object(loci, path, in_window=5, n_jobs=4)


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


def test_cpu_count(monkeypatch):
	# The CPUs this process may run on, or every CPU where the platform does
	# not say which those are.
	if hasattr(os, 'sched_getaffinity'):
		assert tangermeme.io._cpu_count() == len(os.sched_getaffinity(0))
		monkeypatch.delattr(os, 'sched_getaffinity')

	assert tangermeme.io._cpu_count() == (os.cpu_count() or 1)


@pytest.mark.parametrize("kwargs", [
	dict(),
	dict(signals=["tests/data/test.bw"], in_signals=["tests/data/test.bw"]),
	dict(signals=["tests/data/test.bw"], min_counts=0),
	dict(fasta_object=True),
	dict(n_jobs=-1),
], ids=["no_signals", "signals", "count_filter", "fasta_object", "all_cpus"])
def test_extract_loci_n_jobs_reaches_encoder(monkeypatch, kwargs):
	# n_jobs reaches the encoder with or without signals and a count filter,
	# and from a pyfaidx.Fasta object, and -1 is the number of CPUs this
	# process may run on.
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

	n_jobs = kwargs.pop("n_jobs", 3)
	loci = pandas.concat([pandas.read_csv("tests/data/test.bed", sep="\t",
		header=None)] * 40)
	X1 = extract_loci(loci, sequences, in_window=10, out_window=6, n_jobs=1,
		**kwargs)
	X = extract_loci(loci, sequences, in_window=10, out_window=6,
		n_jobs=n_jobs, **kwargs)

	if isinstance(sequences, pyfaidx.Fasta):
		sequences.close()

	cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') \
		else os.cpu_count()
	assert calls == [1, cpus if n_jobs == -1 else n_jobs]
	for x, x1 in zip(X, X1):
		assert torch.equal(x, x1)


###
# The threaded unmap (_close_map)
###


_linux_only = pytest.mark.skipif(not pathlib.Path("/proc/self/maps").exists()
	or tangermeme.io._madvise() is None, reason="needs Linux madvise and "
	"/proc/self/maps")


class _RecordingPool(tangermeme.io.ThreadPoolExecutor):
	"""A ThreadPoolExecutor that records how many workers each pool has."""

	sizes = []

	def __init__(self, max_workers=None, *args, **kwargs):
		_RecordingPool.sizes.append(max_workers)
		super().__init__(max_workers, *args, **kwargs)


def _recording_madvise(monkeypatch, fail=None):
	# Record every madvise call, made through the real function unless
	# `fail` is an exception to raise instead.
	madvise = tangermeme.io._madvise()
	calls = []

	def recording(address, length, advice):
		calls.append((address, length, advice))
		if fail is not None:
			raise fail

		return madvise(address, length, advice)

	monkeypatch.setattr(tangermeme.io, "_madvise", lambda: recording)
	return calls


def _unmap_genome(tmp_path, name, bad=False):
	# About 31 KB, so that the map spans several pages and is split into
	# several chunks, and one window in five, which still covers every base.
	genome = {'chr1': 'ACGTNacgtnACGT' * 1500, 'chr2': 'TTGCA' * 1500}
	if bad:
		genome['chr2'] = genome['chr2'][:300] + 'Z' + genome['chr2'][301:]

	path = str(tmp_path / name)
	_write_fasta(path, genome, 11)
	return genome, path, _every_window(genome, 9).iloc[::5]


@_linux_only
@pytest.mark.parametrize("n_jobs", [2, 8])
def test_extract_loci_unmap_threads(tmp_path, monkeypatch, n_jobs):
	# A map read by at least _UNMAP_MIN_WINDOWS windows has its pages dropped
	# with madvise(MADV_DONTNEED) in page-aligned chunks that cover the whole
	# file, on up to n_jobs threads, before it is closed. The output is the
	# one n_jobs=1 gives, and the file is unmapped on return.
	import mmap
	import threading

	genome, path, loci = _unmap_genome(tmp_path, "genome_madvise.fa")
	X1 = extract_loci(loci, path, in_window=9, n_jobs=1)

	monkeypatch.setattr(tangermeme.io, "_UNMAP_MIN_WINDOWS", 1)
	monkeypatch.setattr(tangermeme.io, "ThreadPoolExecutor", _RecordingPool)
	monkeypatch.setattr(_RecordingPool, "sizes", [])
	calls = _recording_madvise(monkeypatch)
	n_threads = threading.active_count()

	X = extract_loci(loci, path, in_window=9, n_jobs=n_jobs)
	assert torch.equal(X, X1)
	assert torch.equal(X, _encode_windows(genome, loci, 9))
	assert "genome_madvise.fa" not in _mapped_paths()
	assert threading.active_count() == n_threads

	size = pathlib.Path(path).stat().st_size
	calls = sorted(calls)
	assert len(calls) > 0
	assert all(advice == mmap.MADV_DONTNEED for _, _, advice in calls)
	assert all((a - calls[0][0]) % mmap.PAGESIZE == 0 for a, _, _ in calls)
	assert all(a + n == b for (a, n, _), (b, _, _) in zip(calls, calls[1:]))
	assert calls[-1][0] + calls[-1][1] - calls[0][0] == size
	assert _RecordingPool.sizes == [min(n_jobs, len(calls))]


@_linux_only
def test_extract_loci_unmap_threads_on_error(tmp_path, monkeypatch):
	# The pages are dropped and the map closed when the encoder finds a
	# character that is in neither the alphabet nor `ignore`, and the error
	# is the one n_jobs=1 raises.
	genome, path, loci = _unmap_genome(tmp_path, "genome_madvise_bad.fa",
		bad=True)

	with pytest.raises(ValueError, match="Encountered character"):
		extract_loci(loci, path, in_window=9, n_jobs=1)

	monkeypatch.setattr(tangermeme.io, "_UNMAP_MIN_WINDOWS", 1)
	calls = _recording_madvise(monkeypatch)

	with pytest.raises(ValueError, match="Encountered character"):
		extract_loci(loci, path, in_window=9, n_jobs=4)

	assert len(calls) > 0
	assert "genome_madvise_bad.fa" not in _mapped_paths()


@_linux_only
@pytest.mark.parametrize("fail", [OSError(12, "no memory"),
	RuntimeError("can't start new thread")])
def test_extract_loci_unmap_madvise_raises(tmp_path, monkeypatch, fail):
	# madvise only speeds up the close, so when it raises the map is still
	# closed and the output is unchanged.
	genome, path, loci = _unmap_genome(tmp_path, "genome_madvise_raises.fa")
	X1 = extract_loci(loci, path, in_window=9, n_jobs=1)

	monkeypatch.setattr(tangermeme.io, "_UNMAP_MIN_WINDOWS", 1)
	calls = _recording_madvise(monkeypatch, fail=fail)

	X = extract_loci(loci, path, in_window=9, n_jobs=4)
	assert len(calls) > 0
	assert torch.equal(X, X1)
	assert "genome_madvise_raises.fa" not in _mapped_paths()


@_linux_only
@pytest.mark.parametrize("n_jobs, n_min", [(1, 1), (8, 10**9)])
def test_extract_loci_unmap_without_threads(tmp_path, monkeypatch, n_jobs,
	n_min):
	# With one thread, or fewer windows than _UNMAP_MIN_WINDOWS, the map is
	# closed directly.
	genome, path, loci = _unmap_genome(tmp_path, "genome_madvise_direct.fa")

	monkeypatch.setattr(tangermeme.io, "_UNMAP_MIN_WINDOWS", n_min)
	calls = _recording_madvise(monkeypatch)

	X = extract_loci(loci, path, in_window=9, n_jobs=n_jobs)
	assert calls == []
	assert torch.equal(X, _encode_windows(genome, loci, 9))
	assert "genome_madvise_direct.fa" not in _mapped_paths()


@_linux_only
def test_madvise_dontneed_keeps_file_contents(tmp_path):
	# MADV_DONTNEED on a read-only shared map of a file drops its pages, and
	# reading the map again reads the file's bytes, which are unchanged.
	import mmap

	path = tmp_path / "pages.bin"
	data = numpy.random.default_rng(0).integers(0, 256, size=5 * mmap.PAGESIZE
		+ 17, dtype=numpy.uint8).tobytes()
	path.write_bytes(data)

	madvise = tangermeme.io._madvise()
	with open(path, 'rb') as handle:
		fasta_map = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
		view = numpy.frombuffer(fasta_map, dtype=numpy.uint8)
		assert view.tobytes() == data

		assert madvise(view.ctypes.data, len(fasta_map),
			mmap.MADV_DONTNEED) == 0
		assert view.tobytes() == data

		del view
		fasta_map.close()

	assert path.read_bytes() == data


@pytest.mark.parametrize("cause", ["not_linux", "no_libc_madvise"])
def test_madvise_unavailable(tmp_path, monkeypatch, cause):
	# On another platform, or when libc's madvise cannot be loaded, there is
	# no madvise, and the map is closed directly with the same output.
	monkeypatch.setattr(tangermeme.io, "_MADVISE", None)
	with monkeypatch.context() as patch:
		if cause == "not_linux":
			patch.setattr(tangermeme.io.sys, "platform", "darwin")
		else:
			def no_libc(*args, **kwargs):
				raise OSError("no libc")

			patch.setattr(tangermeme.io.ctypes, "CDLL", no_libc)

		assert tangermeme.io._madvise() is None

	# The result is kept, so madvise is not looked up again.
	assert tangermeme.io._madvise() is None

	genome, path, loci = _unmap_genome(tmp_path, "genome_no_madvise.fa")
	monkeypatch.setattr(tangermeme.io, "_UNMAP_MIN_WINDOWS", 1)
	X = extract_loci(loci, path, in_window=9, n_jobs=4)
	assert torch.equal(X, _encode_windows(genome, loci, 9))
	if pathlib.Path("/proc/self/maps").exists():
		assert "genome_no_madvise.fa" not in _mapped_paths()


###
# The threaded nan_to_num (_nan_to_num_rows)
###


def _non_finite_rows(n, width):
	# Rows of finite values with NaN of both signs and several payloads,
	# both infinities and -0.0 scattered through them.
	rng = numpy.random.default_rng(0)
	values = rng.normal(size=(n, 2, width)).astype(numpy.float32)
	bits = values.view(numpy.uint32)
	flat, flat_bits = values.reshape(-1), bits.reshape(-1)
	idxs = rng.choice(flat.size, size=flat.size // 5, replace=False)
	for k, idx in enumerate(idxs):
		kind = k % 6
		if kind == 0:
			flat[idx] = numpy.nan
		elif kind == 1:
			flat_bits[idx] = 0xFFC00000
		elif kind == 2:
			flat_bits[idx] = 0x7F800001 + k
		elif kind == 3:
			flat[idx] = numpy.inf
		elif kind == 4:
			flat[idx] = -numpy.inf
		else:
			flat[idx] = -0.0

	# The first rows are all finite, so that some blocks are skipped.
	values[:3] = 1.5
	return values


@pytest.mark.parametrize("n_jobs", [1, 3])
@pytest.mark.parametrize("block_size", [1, 2**20])
def test_nan_to_num_rows_threads_match_numpy(monkeypatch, n_jobs, block_size):
	# Every block is replaced as numpy.nan_to_num replaces it, bit for bit,
	# on any number of threads: NaN of any sign or payload becomes +0.0, an
	# infinity the largest finite float32 of its sign, and -0.0 is kept.
	monkeypatch.setattr(tangermeme.io, "_NAN_TO_NUM_MIN_BLOCKS", 1)
	monkeypatch.setattr(tangermeme.io, "ThreadPoolExecutor", _RecordingPool)
	monkeypatch.setattr(_RecordingPool, "sizes", [])

	values = _non_finite_rows(23, 5)
	expected = numpy.nan_to_num(values.copy())
	out = tangermeme.io._nan_to_num_rows(values, block_size=block_size,
		n_jobs=n_jobs)

	assert out is values
	assert values.tobytes() == expected.tobytes()
	assert numpy.isfinite(values).all()

	n_blocks = len(range(0, 23, max(1, block_size // 10)))
	if n_jobs == 1:
		assert _RecordingPool.sizes == []
	else:
		assert _RecordingPool.sizes == [min(n_jobs, n_blocks)]


def test_nan_to_num_rows_few_blocks_are_serial(monkeypatch):
	# Fewer blocks than _NAN_TO_NUM_MIN_BLOCKS are checked without threads.
	monkeypatch.setattr(tangermeme.io, "ThreadPoolExecutor", _RecordingPool)
	monkeypatch.setattr(_RecordingPool, "sizes", [])

	n_min = tangermeme.io._NAN_TO_NUM_MIN_BLOCKS
	for n_blocks, sizes in [(n_min - 1, []), (n_min, [8])]:
		_RecordingPool.sizes = []
		values = _non_finite_rows(n_blocks, 5)
		expected = numpy.nan_to_num(values.copy())
		tangermeme.io._nan_to_num_rows(values, block_size=10, n_jobs=8)
		assert values.tobytes() == expected.tobytes()
		assert _RecordingPool.sizes == sizes


def test_nan_to_num_rows_empty():
	# No rows, and rows of no values, are returned as they are.
	for shape in [(0, 2, 5), (3, 2, 0)]:
		values = numpy.zeros(shape, dtype=numpy.float32)
		assert tangermeme.io._nan_to_num_rows(values, n_jobs=4) is values


@pytest.mark.parametrize("n_jobs", [2, -1])
def test_extract_loci_nan_and_inf_threads(tmp_path, monkeypatch, n_jobs):
	# extract_loci passes n_jobs, with -1 as the number of CPUs, to
	# _nan_to_num_rows for signals and in_signals, and the threaded
	# replacement, in blocks of one row each, gives the bytes that one thread
	# gives. chr1 is 50 bp in the bigWig but 284 bp in test.fa, so positions
	# past base 50 are read as NaN.
	path = str(tmp_path / "inf.bw")
	write_raw_bigwig(path, {'chr1': 50, 'chr2': 211}, [('chr1', 1, 0, 0,
		[(0, 10, 1.0), (10, 12, numpy.inf), (12, 14, -numpy.inf),
		(14, 50, 2.0)]), ('chr2', 1, 0, 0, [(0, 211, 0.5)])])

	nan_to_num_rows = tangermeme.io._nan_to_num_rows
	calls = []

	def one_row_blocks(values, n_jobs=1):
		calls.append(n_jobs)
		return nan_to_num_rows(values, block_size=1, n_jobs=n_jobs)

	monkeypatch.setattr(tangermeme.io, "_NAN_TO_NUM_MIN_BLOCKS", 1)
	monkeypatch.setattr(tangermeme.io, "_nan_to_num_rows", one_row_blocks)

	loci = pandas.DataFrame({0: ['chr1'] * 4, 1: [7, 43, 20, 45],
		2: [17, 53, 30, 55]})
	X1, y1, y_in1 = extract_loci(loci, "tests/data/test.fa", [path], [path],
		in_window=6, out_window=10, n_jobs=1)
	X, y, y_in = extract_loci(loci, "tests/data/test.fa", [path], [path],
		in_window=6, out_window=10, n_jobs=n_jobs)

	cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') \
		else os.cpu_count()
	threads = cpus if n_jobs == -1 else n_jobs
	assert calls == [1, 1, threads, threads]
	assert torch.equal(X, X1)
	assert y.numpy().tobytes() == y1.numpy().tobytes()
	assert y_in.numpy().tobytes() == y_in1.numpy().tobytes()

	big = numpy.finfo(numpy.float32).max
	assert_array_almost_equal(y[:2, 0], [[1, 1, 1, big, big, -big, -big, 2, 2,
		2], [2, 2, 2, 2, 2, 2, 2, 0, 0, 0]])
	assert numpy.isfinite(y.numpy()).all()
	assert numpy.isfinite(y_in.numpy()).all()
