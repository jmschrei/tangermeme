# test_io.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pytest
import pandas
import pyfaidx
import pyBigWig

from tangermeme.io import _interleave_loci
from tangermeme.io import _load_signals
from tangermeme.io import _extract_locus_signal

from tangermeme.io import read_meme
from tangermeme.io import extract_loci

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
	
	df = pandas.read_csv("tests/data/test.bed", delimiter='\t', 
		index_col=False, names=names, header=None)
	assert (df == short_loci1).all(None)
	assert_raises(ValueError, df.__eq__, short_loci2)

	df = pandas.read_csv("tests/data/test2.bed", delimiter='\t', 
		index_col=False, names=names, header=None)
	assert (df == short_loci2).all(None)
	assert_raises(ValueError, df.__eq__, short_loci1)


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


def test_interleave_loci_single_raises():
	assert_raises(ValueError, _interleave_loci, 5)
	assert_raises(ValueError, _interleave_loci, numpy.random.randn(5, 5))


def test_ineterleave_loci_multi_raises():
	assert_raises(ValueError, _interleave_loci, [5, 1])
	assert_raises(ValueError, _interleave_loci, [numpy.random.randn(5, 5)])	


def test_ineterleave_loci_multi_raises_chroms():
	assert_raises(ValueError, _interleave_loci, ["tests/data/test2.bed", 
		"tests/data/test.bed"], 'chr1')


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


def test_load_signals_dict():
	signal = {
		'chr1': numpy.array([0.0, 0.0, 0.5, 0.0, 1.0, 0.0]),
		'chr2': numpy.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
	}

	bw = _load_signals([signal])

	assert len(bw) == 1
	assert isinstance(bw, list)
	assert isinstance(bw[0], dict)


##


def test_extract_locus_signal_single():
	bw = pyBigWig.open("tests/data/test.bw")
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


def test_extract_locus_signal_single():
	bw = pyBigWig.open("tests/data/test.bw")
	bw2 = pyBigWig.open("tests/data/test2.bw")
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


def test_extract_locus_signal_raises_single():
	bw = pyBigWig.open("tests/data/test.bw")
	assert_raises(ValueError, _extract_locus_signal, bw, 'chr1', 3, 14)


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
	])

	motifs = read_meme("tests/data/test.meme")

	assert len(motifs) == 12
	assert isinstance(motifs, dict)
	assert all(isinstance(key, str) for key in motifs.keys())
	assert all(isinstance(pwm, numpy.ndarray) for pwm in motifs.values())
	
	assert all([key in motifs.keys() for key in keys])
	assert_array_almost_equal(motifs['FOSL2+JUND_MA1145.1'], pwm) 


def test_read_meme_n_motifs():
	keys = ["MEOX1_homeodomain_1", "HIC2_MA0738.1", "GCR_HUMAN.H11MO.0.A",
		"FOSL2+JUND_MA1145.1", "TEAD3_TEA_2", "ZN263_HUMAN.H11MO.0.A"]

	motifs = read_meme("tests/data/test.meme", n_motifs=6)

	assert len(motifs) == 6
	assert isinstance(motifs, dict)
	assert all(isinstance(key, str) for key in motifs.keys())
	assert all(isinstance(pwm, numpy.ndarray) for pwm in motifs.values())
	
	assert all([key in motifs.keys() for key in keys])


