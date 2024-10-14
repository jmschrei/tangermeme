import os
import numpy
import pytest
import pandas

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


@pytest.mark.cmd
def test_cmd_tomtom():
	fname = "tests/data/test.meme"

	os.system("tangermeme tomtom -q tests/data/test2.meme " 
		"-t tests/data/test.meme > .test.tomtom")
	tomtom_results = pandas.read_csv(".test.tomtom", sep="\t")
	os.system("rm .test.tomtom")

	assert tomtom_results.shape == (4, 9)

	names = ['FOXQ1_MOUSE.H11MO.0.C', 'FOXQ1_MOUSE.H11MO.0.C', 'Hes1_MA1099.1', 
		'FOSL2+JUND_MA1145.1']
	for i, name in enumerate(tomtom_results['Target Name']):
		assert name.strip() == names[i] 

	assert_array_almost_equal(tomtom_results['p-value'], [0.000241, 0.000673, 
		0.001244, 0.004507])
	assert_array_almost_equal(tomtom_results['Score'], [594, 604, 722, 717])
	assert_array_almost_equal(tomtom_results['Offset'], [2, 2, 0, 3])
	assert_array_almost_equal(tomtom_results['Overlap'], [7, 7, 10, 10])

	strands = ['-', '-', '+', '-']
	for i, strand in enumerate(tomtom_results['Strand']):
		assert strand.strip() == strands[i]


@pytest.mark.cmd
def test_cmd_tomtom2():
	fname = "tests/data/test.meme"

	os.system("tangermeme tomtom -q tests/data/test2.meme " 
		"-t tests/data/test.meme > .test.tomtom")
	tomtom_results = pandas.read_csv(".test.tomtom", sep="\t")
	os.system("rm .test.tomtom")

	assert tomtom_results.shape == (4, 9)

	names = ['FOXQ1_MOUSE.H11MO.0.C', 'FOXQ1_MOUSE.H11MO.0.C', 'Hes1_MA1099.1', 
		'FOSL2+JUND_MA1145.1']
	for i, name in enumerate(tomtom_results['Target Name']):
		assert name.strip() == names[i] 

	assert_array_almost_equal(tomtom_results['p-value'], [0.000241, 0.000673, 
		0.001244, 0.004507])
	assert_array_almost_equal(tomtom_results['Score'], [594, 604, 722, 717])
	assert_array_almost_equal(tomtom_results['Offset'], [2, 2, 0, 3])
	assert_array_almost_equal(tomtom_results['Overlap'], [7, 7, 10, 10])

	strands = ['-', '-', '+', '-']
	for i, strand in enumerate(tomtom_results['Strand']):
		assert strand.strip() == strands[i]
