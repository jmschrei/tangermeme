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


def test_plot_categorical_scatter_respects_ax():
	from tangermeme.plot import plot_categorical_scatter

	X = torch.tensor([
		[1.0, 0.0, 0.0, 0.0, 0.5],
		[0.0, 1.0, 0.0, 0.0, 0.0],
		[0.0, 0.0, 1.0, 0.0, 0.0],
		[0.0, 0.0, 0.0, 1.0, 0.5],
	])

	# Create a user-owned figure with two axes; pass the second one in.
	fig, (ax1, ax2) = plt.subplots(2, 1)
	returned = plot_categorical_scatter(X, ax=ax2)

	# The function should draw on ax2 (the one we passed), not ax1.
	assert returned is ax2, "function did not return the passed ax"
	assert len(ax2.collections) > 0, "expected scatter on ax2"
	assert len(ax1.collections) == 0, "ax1 should be untouched"


def test_plot_categorical_scatter_defaults_to_gca():
	from tangermeme.plot import plot_categorical_scatter

	X = torch.tensor([
		[1.0, 0.0, 0.5],
		[0.0, 1.0, 0.0],
		[0.0, 0.0, 0.0],
		[0.0, 0.0, 0.5],
	])

	fig, ax = plt.subplots()
	plt.sca(ax)
	returned = plot_categorical_scatter(X)
	assert returned is ax


def test_plot_pwm_respects_ax():
	# plot_pwm should draw on the provided ax and return it, rather than
	# creating its own figure or calling plt.show().
	pwm = numpy.full((4, 6), 0.25)

	fig, (ax1, ax2) = plt.subplots(2, 1)
	returned = plot_pwm(pwm, ax=ax2)

	assert returned is ax2, "plot_pwm did not return the passed ax"
	# Something should have been drawn on ax2; the IC is zero for a
	# uniform PWM but plot_logo still sets axis labels.
	assert ax2.get_ylabel() == "Information Content (Bits)"
	# ax1 should be untouched.
	assert ax1.get_ylabel() == ""


def test_plot_pwm_defaults_to_gca():
	pwm = numpy.full((4, 6), 0.25)

	fig, ax = plt.subplots()
	plt.sca(ax)
	returned = plot_pwm(pwm)
	assert returned is ax


def test_plot_pwm_does_not_call_show(monkeypatch):
	# plot_pwm no longer calls plt.show() unconditionally.
	called = []
	monkeypatch.setattr(plt, "show", lambda *a, **k: called.append(True))

	pwm = numpy.full((4, 6), 0.25)
	fig, ax = plt.subplots()
	plot_pwm(pwm, ax=ax)

	assert called == [], "plot_pwm should not call plt.show()"


def test_plot_attributions_uses_provided_func():
	# plot_attributions previously hard-coded a call to deep_lift_shap
	# and ignored the `func=` kwarg. Verify the kwarg is now honored.
	from tangermeme.plot import plot_attributions

	calls = []

	def fake_attribute(model, X, **kwargs):
		calls.append((model, tuple(X.shape)))
		# Return something the size of X so plot_logo can render it.
		return X * 0.0

	from tests.toy_models import FlattenDense
	torch.manual_seed(0)
	model = FlattenDense()
	X = torch.zeros(1, 4, 20)
	X[0, 0] = 1.0  # all-A so plot_logo doesn't error

	axs, X_attrs = plot_attributions(model, X, func=fake_attribute)

	assert len(calls) == 1, "fake_attribute should have been called once"
	assert X_attrs.shape == X.shape
