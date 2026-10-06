# test_io.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest
import pandas
import pathlib
import warnings

import figwig
import pyfaidx
import pybigtools

from tangermeme.io import _interleave_loci
from tangermeme.io import _load_signals
from tangermeme.io import _load_exclusion_zones
from tangermeme.io import _extract_locus_signal
from tangermeme.io import _extract_signals

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


def test_load_signals_figwig():
	# Local paths open with figwig, and opened bigWigs and dicts pass through.
	bw = pybigtools.open("tests/data/test2.bw")
	signals = _load_signals(["tests/data/test.bw",
		pathlib.Path("tests/data/test3.bw"), bw, {'chr1': numpy.zeros(5)}],
		use_figwig=True)

	assert isinstance(signals[0], figwig.BigWigReader)
	assert isinstance(signals[1], figwig.BigWigReader)
	assert signals[2] is bw
	assert isinstance(signals[3], dict)


def test_load_signals_figwig_url(monkeypatch):
	# figwig reads local files only, so a URL goes straight to pybigtools.
	def no_figwig(path):
		raise AssertionError("figwig opened {}".format(path))

	monkeypatch.setattr(figwig, "BigWigReader", no_figwig)
	monkeypatch.setattr(pybigtools, "open", lambda path: ("pybigtools", path))

	url = "https://example.com/signal.bw"
	assert _load_signals([url], use_figwig=True) == [("pybigtools", url)]


@pytest.mark.parametrize("path", ["tests/data/test.bed",
	"tests/data/missing.bw"])
def test_load_signals_figwig_raises_as_pybigtools(path):
	# A file figwig does not read, or a missing one, raises pybigtools' error.
	with pytest.raises(Exception) as error:
		_load_signals([path])

	with pytest.raises(type(error.value)) as figwig_error:
		_load_signals([path], use_figwig=True)

	assert str(figwig_error.value) == str(error.value)


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
		values = _extract_signals(_load_signals(paths, use_figwig=True),
			SIGNAL_CHROMS, SIGNAL_STARTS, 8, 2)

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


def test_extract_signals_figwig_error_reads_with_pybigtools(monkeypatch):
	# When figwig raises for a file it does not read, the bigWigs are read
	# with pybigtools instead.
	def corrupt(*args, **kwargs):
		raise ValueError("corrupt data block")

	paths = ["tests/data/test.bw", "tests/data/test2.bw"]
	signals = _load_signals(paths, use_figwig=True)
	monkeypatch.setattr(figwig, "read_bigwig", corrupt)

	values = _extract_signals(signals, SIGNAL_CHROMS, SIGNAL_STARTS, 8, 2)
	assert values.tobytes() == _per_locus_signals(paths, 8).tobytes()


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

	bws = [pybigtools.open(path) for path in paths]
	expected = extract_loci("tests/data/test.bed", "tests/data/test.fa", bws,
		bws[::-1], **kwargs)

	for tensor, expected_tensor in zip(result, expected):
		assert tensor.dtype == expected_tensor.dtype
		assert tensor.is_contiguous()
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


@pytest.mark.filterwarnings("ignore::tangermeme.utils.TangermemeWarning")
@pytest.mark.parametrize("filters", ["min_counts", "counts_target_idx",
	"combined"])
@pytest.mark.parametrize("n_values", [1, 33, 500])
def test_extract_loci_counts_in_groups(monkeypatch, reference_genome, filters,
	n_values):
	# min_counts and max_counts are measured on groups of loci, stopping once
	# n_loci are kept. Groups of one locus, of three, and of 45 give what one
	# group of every locus gives.
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




