# test_plot.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>


import matplotlib
matplotlib.use("Agg")

import numpy
import torch
import pytest
import pandas

import matplotlib.pyplot as plt

from tangermeme.utils import one_hot_encode

from tangermeme.plot import plot_logo
from tangermeme.plot import plot_pwm
from tangermeme.plot import _base_path
from tangermeme.plot import get_glyph_path


@pytest.fixture(autouse=True)
def _close_figures():
	yield
	plt.close("all")


@pytest.fixture
def X_attr():
	torch.manual_seed(0)
	return torch.randn(4, 50)


###


def test_plot_logo_smoke_onehot():
	X = one_hot_encode("ACGTACGTACGTACGTACGT").type(torch.float32)
	ax = plot_logo(X)

	assert ax is not None
	assert len(ax.collections) >= 1


def test_plot_logo_smoke_random_attributions(X_attr):
	ax = plot_logo(X_attr)

	assert ax is not None
	ylo, yhi = ax.get_ylim()
	assert ylo < 0 < yhi


def test_plot_logo_numpy_input(X_attr):
	X_np = X_attr.numpy()
	ax = plot_logo(X_np)
	assert ax is not None


def test_plot_logo_start_end_slicing(X_attr):
	ax = plot_logo(X_attr, start=10, end=40)

	xlo, xhi = ax.get_xlim()
	assert int(round(xhi - xlo)) >= 29


def test_plot_logo_all_zero_attributions():
	X = torch.zeros(4, 30)
	ax = plot_logo(X)
	assert ax is not None


def test_plot_logo_color_string(X_attr):
	ax = plot_logo(X_attr, color="red")
	assert ax is not None


def test_plot_logo_color_dict(X_attr):
	color = {'A': 'red', 'C': 'blue', 'G': 'green', 'T': 'black'}
	ax = plot_logo(X_attr, color=color)
	assert ax is not None


def test_plot_logo_ax_passed():
	X = one_hot_encode("ACGT" * 10).type(torch.float32)

	fig, ax_in = plt.subplots()
	ax_out = plot_logo(X, ax=ax_in)
	assert ax_out is ax_in


def test_plot_logo_alphabet_custom():
	alphabet = ['A', 'C', 'G', 'T', 'N']
	X = torch.zeros(5, 20)
	X[0, :] = 1.0

	ax = plot_logo(X, alphabet=alphabet)
	assert ax is not None


###


def test_base_path_cache():
	verts0, codes0 = _base_path('A')
	verts1, codes1 = _base_path('A')

	# lru_cache returns the same arrays
	assert verts0 is verts1
	assert codes0 is codes1


def test_get_glyph_path_returns_path():
	from matplotlib.path import Path as MplPath
	p = get_glyph_path('C', x=0.0, y=0.0, width=1.0, height=2.0)
	assert isinstance(p, MplPath)


###


def test_plot_pwm_smoke():
	pwm = numpy.array([
		[0.7, 0.1, 0.1, 0.1],
		[0.1, 0.7, 0.1, 0.1],
		[0.1, 0.1, 0.7, 0.1],
		[0.1, 0.1, 0.1, 0.7],
	]).T
	plot_pwm(pwm, name="toy")


def test_place_new_box_does_not_mutate_input():
	from tangermeme.plot import place_new_box
	from matplotlib.transforms import Bbox

	box = Bbox.from_extents(0.0, 0.0, 10.0, 5.0)
	original = (box.x0, box.y0, box.x1, box.y1)

	# Empty box_list path: should still return a (shifted) result without
	# mutating the input.
	new_box, _ = place_new_box(box, [])
	assert (box.x0, box.y0, box.x1, box.y1) == original, \
		"place_new_box mutated the caller's box on the empty path"
	assert new_box is not box, "expected a fresh Bbox to be returned"

	# Non-empty box_list path: same guarantee under the overlap loop.
	overlap = Bbox.from_extents(0.0, -5.0, 10.0, 0.0)
	box2 = Bbox.from_extents(0.0, 0.0, 10.0, 5.0)
	original2 = (box2.x0, box2.y0, box2.x1, box2.y1)
	new_box, _ = place_new_box(box2, [overlap])
	assert (box2.x0, box2.y0, box2.x1, box2.y1) == original2


def test_place_new_bar_does_not_mutate_input():
	from tangermeme.plot import place_new_bar
	from matplotlib.transforms import Bbox

	box = Bbox.from_extents(0.0, 5.0, 10.0, 5.0)
	original = (box.x0, box.y0, box.x1, box.y1)

	new_box, _ = place_new_bar(box, [], y_step=1.0)
	assert (box.x0, box.y0, box.x1, box.y1) == original
	assert new_box is not box
