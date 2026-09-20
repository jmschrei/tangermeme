# test_plot.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>


import matplotlib
matplotlib.use("Agg")

import warnings

import numpy
import torch
import pytest
import pandas

import matplotlib.pyplot as plt

from tangermeme.utils import one_hot_encode
from tangermeme.utils import TangermemeWarning

from tangermeme.plot import plot_logo
from tangermeme.plot import plot_pwm
from tangermeme.plot import interactive_logo
from tangermeme.plot import _base_path
from tangermeme.plot import get_glyph_path
from tangermeme.plot import _format_tooltip_value


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


def test_plot_logo_color_per_position_name_list():
	X = one_hot_encode("ACGT" * 5).type(torch.float32)
	colors = ["red"] * 10 + ["blue"] * 10

	ax = plot_logo(X, color=colors)
	facecolors = ax.collections[0].get_facecolors()

	assert len(facecolors) == 20
	numpy.testing.assert_array_almost_equal(facecolors[:10],
		numpy.tile([1.0, 0.0, 0.0, 1.0], (10, 1)))
	numpy.testing.assert_array_almost_equal(facecolors[10:],
		numpy.tile([0.0, 0.0, 1.0, 1.0], (10, 1)))


def test_plot_logo_color_per_position_rgba_array():
	X = one_hot_encode("ACGT" * 5).type(torch.float32)
	rgba = numpy.tile([0.2, 0.4, 0.6, 1.0], (20, 1))

	ax = plot_logo(X, color=rgba)
	facecolors = ax.collections[0].get_facecolors()

	numpy.testing.assert_array_almost_equal(facecolors, rgba)


def test_plot_logo_color_per_position_scalar_cmap():
	X = one_hot_encode("ACGT" * 5).type(torch.float32)
	values = numpy.linspace(0.0, 1.0, 20)

	ax = plot_logo(X, color=values, color_cmap="viridis")
	facecolors = ax.collections[0].get_facecolors()

	cmap = plt.get_cmap("viridis")
	norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
	expected = cmap(norm(values))

	numpy.testing.assert_array_almost_equal(facecolors, expected)


def test_plot_logo_color_per_position_scalar_vmin_vmax():
	X = one_hot_encode("ACGT" * 5).type(torch.float32)
	values = numpy.linspace(0.0, 1.0, 20)

	ax = plot_logo(X, color=values, color_cmap="viridis",
		color_vmin=-1.0, color_vmax=2.0)
	facecolors = ax.collections[0].get_facecolors()

	cmap = plt.get_cmap("viridis")
	norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=2.0)
	expected = cmap(norm(values))

	numpy.testing.assert_array_almost_equal(facecolors, expected)


def test_plot_logo_color_per_position_sliced():
	X = one_hot_encode("ACGT" * 5).type(torch.float32)
	colors = ["red"] * 5 + ["blue"] * 10 + ["green"] * 5

	ax = plot_logo(X, color=colors, start=5, end=15)
	facecolors = ax.collections[0].get_facecolors()

	assert len(facecolors) == 10
	numpy.testing.assert_array_almost_equal(facecolors,
		numpy.tile([0.0, 0.0, 1.0, 1.0], (10, 1)))


def test_plot_logo_color_dict_still_per_character():
	X = one_hot_encode("ACGT").type(torch.float32)
	color = {'A': 'red', 'C': 'blue', 'G': 'green', 'T': 'black'}

	ax = plot_logo(X, color=color)
	facecolors = ax.collections[0].get_facecolors()

	expected = numpy.array([
		matplotlib.colors.to_rgba('red'),
		matplotlib.colors.to_rgba('blue'),
		matplotlib.colors.to_rgba('green'),
		matplotlib.colors.to_rgba('black'),
	])
	numpy.testing.assert_array_almost_equal(facecolors, expected)


def test_plot_logo_color_length_mismatch_warns_and_falls_back():
	X = one_hot_encode("ACGT" * 5).type(torch.float32)

	with pytest.warns(TangermemeWarning):
		ax = plot_logo(X, color=["red"] * 5)

	facecolors = ax.collections[0].get_facecolors()
	assert len(facecolors) == 20

	# Falls back to the standard per-character coloring, not a single red.
	assert not numpy.allclose(facecolors,
		numpy.tile([1.0, 0.0, 0.0, 1.0], (20, 1)))


def test_plot_logo_color_array_matching_alphabet_length_is_per_position():
	# A length-4 array on a length-4 sequence is treated as per-position
	# coloring (not a single RGBA color) and must not warn.
	X = one_hot_encode("ACGT").type(torch.float32)
	colors = ["red", "green", "blue", "black"]

	with warnings.catch_warnings():
		warnings.simplefilter("error")
		ax = plot_logo(X, color=colors)

	facecolors = ax.collections[0].get_facecolors()
	expected = numpy.array([matplotlib.colors.to_rgba(c) for c in colors])
	numpy.testing.assert_array_almost_equal(facecolors, expected)


def test_plot_logo_color_per_position_rgb_without_alpha():
	X = one_hot_encode("ACGT" * 5).type(torch.float32)
	rgb = numpy.tile([0.2, 0.4, 0.6], (20, 1))

	ax = plot_logo(X, color=rgb)
	facecolors = ax.collections[0].get_facecolors()

	expected = numpy.tile([0.2, 0.4, 0.6, 1.0], (20, 1))
	numpy.testing.assert_array_almost_equal(facecolors, expected)


def test_interactive_logo_passes_per_position_color_through():
	pytest.importorskip("mpld3")
	X = one_hot_encode("ACGT" * 5).type(torch.float32)
	colors = ["red"] * 20

	ax = interactive_logo(X, color=colors)
	facecolors = ax.collections[0].get_facecolors()

	numpy.testing.assert_array_almost_equal(facecolors,
		numpy.tile([1.0, 0.0, 0.0, 1.0], (20, 1)))


def test_interactive_logo_forwards_color_cmap_and_bounds():
	pytest.importorskip("mpld3")
	X = one_hot_encode("ACGT" * 5).type(torch.float32)
	values = numpy.linspace(0.0, 1.0, 20)

	ax = interactive_logo(X, color=values, color_cmap="plasma",
		color_vmin=-1.0, color_vmax=2.0)
	facecolors = ax.collections[0].get_facecolors()

	cmap = plt.get_cmap("plasma")
	norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=2.0)
	expected = cmap(norm(values))

	numpy.testing.assert_array_almost_equal(facecolors, expected)


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


###
# interactive_logo
###


@pytest.fixture
def annotations():
	return pandas.DataFrame({
		'name': ['GATA2', 'TAL1', 'SP1'],
		'start': [5, 20, 35],
		'end': [12, 28, 42],
		'strand': ['+', '-', '+'],
		'attribution': [0.42, 0.31, 0.9],
		'p-value': [1e-5, 3.2e-3, 1e-12],
		'annotation_p-value': [2e-4, 1e-2, 5e-9],
	})


def test_format_tooltip_value_float():
	assert _format_tooltip_value(1e-12) == "1e-12"
	assert _format_tooltip_value(numpy.float64(3.2e-3)) == "0.0032"


def test_format_tooltip_value_non_float():
	assert _format_tooltip_value("+") == "+"
	assert _format_tooltip_value(7) == "7"


def test_interactive_logo_returns_ax(X_attr, annotations):
	pytest.importorskip("mpld3")
	fig, ax = plt.subplots()
	returned = interactive_logo(X_attr, ax=ax, annotations=annotations)
	assert returned is ax


def test_interactive_logo_draws_one_box_per_annotation(X_attr, annotations):
	pytest.importorskip("mpld3")
	from matplotlib.collections import PatchCollection

	fig, ax = plt.subplots()
	interactive_logo(X_attr, ax=ax, annotations=annotations)

	patch_collections = [c for c in ax.collections
		if isinstance(c, PatchCollection)]
	assert len(patch_collections) == 1
	assert len(patch_collections[0].get_paths()) == len(annotations)


def test_interactive_logo_draws_corner_labels(X_attr, annotations):
	pytest.importorskip("mpld3")
	fig, ax = plt.subplots()
	interactive_logo(X_attr, ax=ax, annotations=annotations)

	texts = [t.get_text() for t in ax.texts]
	assert "GATA2" in texts
	assert "TAL1" in texts
	assert "SP1" in texts


def test_interactive_logo_tooltip_contents(X_attr, annotations):
	mpld3 = pytest.importorskip("mpld3")
	fig, ax = plt.subplots()
	interactive_logo(X_attr, ax=ax, annotations=annotations)

	html = mpld3.fig_to_html(fig)
	assert "htmltooltip" in html
	assert "GATA2" in html
	assert "length:" in html
	# All extra columns are surfaced, including both p-values.
	assert "1e-12" in html
	assert "5e-09" in html or "5e-9" in html
	assert "strand:" in html
	# The seqlet caller's bare columns are relabeled in the tooltip.
	assert "seqlet attribution:" in html
	assert "seqlet p-value:" in html


def test_interactive_logo_none_annotations_draws_no_boxes(X_attr):
	pytest.importorskip("mpld3")
	from matplotlib.collections import PatchCollection

	fig, ax = plt.subplots()
	interactive_logo(X_attr, ax=ax, annotations=None)

	patch_collections = [c for c in ax.collections
		if isinstance(c, PatchCollection)]
	assert len(patch_collections) == 0


def test_interactive_logo_filters_out_of_view_annotations(X_attr, annotations):
	pytest.importorskip("mpld3")
	from matplotlib.collections import PatchCollection

	fig, ax = plt.subplots()
	# Window [15, 50) drops GATA2 (start 5) and keeps TAL1 and SP1.
	interactive_logo(X_attr, ax=ax, annotations=annotations, start=15, end=50)

	patch_collections = [c for c in ax.collections
		if isinstance(c, PatchCollection)]
	assert len(patch_collections[0].get_paths()) == 2

	texts = [t.get_text() for t in ax.texts]
	assert "GATA2" not in texts
	assert "TAL1" in texts


def test_interactive_logo_box_alpha(X_attr, annotations):
	pytest.importorskip("mpld3")
	from matplotlib.collections import PatchCollection

	fig, ax = plt.subplots()
	interactive_logo(X_attr, ax=ax, annotations=annotations, box_alpha=0.5)

	pc = [c for c in ax.collections if isinstance(c, PatchCollection)][0]
	facecolors = pc.get_facecolors()
	assert numpy.allclose(facecolors[:, 3], 0.5)


def test_interactive_logo_respects_ylim(X_attr, annotations):
	pytest.importorskip("mpld3")
	fig, ax = plt.subplots()
	interactive_logo(X_attr, ax=ax, annotations=annotations, ylim=(-2, 2))
	assert ax.get_ylim() == (-2, 2)


def test_interactive_logo_default_tooltip_has_background(X_attr, annotations):
	mpld3 = pytest.importorskip("mpld3")
	fig, ax = plt.subplots()
	interactive_logo(X_attr, ax=ax, annotations=annotations)

	html = mpld3.fig_to_html(fig)
	assert ".mpld3-tooltip" in html
	assert "background: rgba(255, 255, 255" in html


def test_interactive_logo_custom_tooltip_css_overrides(X_attr, annotations):
	mpld3 = pytest.importorskip("mpld3")
	fig, ax = plt.subplots()
	interactive_logo(X_attr, ax=ax, annotations=annotations,
		tooltip_css=".mpld3-tooltip { background: black; }")

	html = mpld3.fig_to_html(fig)
	assert "background: black" in html
	assert "background: rgba(255, 255, 255" not in html


def test_interactive_logo_overrides_dark_rcparam_grid(X_attr, annotations,
	monkeypatch):
	mpld3 = pytest.importorskip("mpld3")
	# Simulate a style with a heavy default grid; the function should not
	# inherit it -- the x-axis grid is cleared and the y-axis grid is light.
	monkeypatch.setitem(matplotlib.rcParams, "axes.grid", True)

	fig, ax = plt.subplots()
	interactive_logo(X_attr, ax=ax, annotations=annotations)

	xgrid = ax.xaxis.get_gridlines()
	ygrid = ax.yaxis.get_gridlines()
	assert all(not line.get_visible() for line in xgrid)
	assert all(line.get_visible() for line in ygrid)

	# mpld3 fails to apply the grid color to the gridline <line> elements, so
	# the function injects a light stroke directly. Confirm the CSS is present.
	html = mpld3.fig_to_html(fig)
	assert ".mpld3-ygrid .tick line" in html
	assert "stroke: #c4c4c4 !important" in html


def test_interactive_logo_grid_false_injects_no_grid_css(X_attr, annotations):
	mpld3 = pytest.importorskip("mpld3")
	fig, ax = plt.subplots()
	interactive_logo(X_attr, ax=ax, annotations=annotations, grid=False)

	html = mpld3.fig_to_html(fig)
	assert ".mpld3-ygrid .tick line" not in html


def test_interactive_logo_grid_false(X_attr, annotations, monkeypatch):
	pytest.importorskip("mpld3")
	monkeypatch.setitem(matplotlib.rcParams, "axes.grid", True)

	fig, ax = plt.subplots()
	interactive_logo(X_attr, ax=ax, annotations=annotations, grid=False)

	xgrid = ax.xaxis.get_gridlines()
	ygrid = ax.yaxis.get_gridlines()
	assert all(not line.get_visible() for line in xgrid)
	assert all(not line.get_visible() for line in ygrid)


def test_interactive_logo_despine_default(X_attr, annotations):
	mpld3 = pytest.importorskip("mpld3")
	fig, ax = plt.subplots()
	interactive_logo(X_attr, ax=ax, annotations=annotations)

	# Left spine hidden on the matplotlib side; mpld3 axis lines hidden via CSS.
	assert not ax.spines['left'].get_visible()
	html = mpld3.fig_to_html(fig)
	assert ".mpld3-xaxis path" in html
	assert "display: none !important" in html


def test_interactive_logo_despine_false_keeps_spines(X_attr, annotations):
	mpld3 = pytest.importorskip("mpld3")
	fig, ax = plt.subplots()
	interactive_logo(X_attr, ax=ax, annotations=annotations, despine=False)

	assert ax.spines['left'].get_visible()
	html = mpld3.fig_to_html(fig)
	assert ".mpld3-xaxis path" not in html


def test_interactive_logo_box_linewidth_default_zero(X_attr, annotations):
	pytest.importorskip("mpld3")
	from matplotlib.collections import PatchCollection

	fig, ax = plt.subplots()
	interactive_logo(X_attr, ax=ax, annotations=annotations)

	pc = [c for c in ax.collections if isinstance(c, PatchCollection)][0]
	assert numpy.allclose(pc.get_linewidth(), 0)


def test_interactive_logo_box_linewidth_positive(X_attr, annotations):
	pytest.importorskip("mpld3")
	from matplotlib.collections import PatchCollection

	fig, ax = plt.subplots()
	interactive_logo(X_attr, ax=ax, annotations=annotations, box_linewidth=2)

	pc = [c for c in ax.collections if isinstance(c, PatchCollection)][0]
	assert numpy.allclose(pc.get_linewidth(), 2)


def test_interactive_logo_coordinates_have_commas_no_scientific():
	mpld3 = pytest.importorskip("mpld3")
	# In a typical genomic window, float positions over 1000 would render in
	# scientific notation (e.g. 1.03e+03) under the default formatter.
	X_attr = torch.randn(4, 3000)
	annot = pandas.DataFrame({
		'motif_name': ['GATA2'],
		'start': [1030.0],
		'end': [1085.0],
		'p-value': [1e-12],
	})

	fig, ax = plt.subplots()
	interactive_logo(X_attr, ax=ax, annotations=annot)

	html = mpld3.fig_to_html(fig)
	assert "start: 1,030" in html
	assert "end: 1,085" in html
	assert "1.03e+03" not in html
	# length still comma-formatted, p-values still scientific.
	assert "length: 55" in html
	assert "1e-12" in html


def test_interactive_logo_index_labels_without_name():
	mpld3 = pytest.importorskip("mpld3")
	# A seqlet-caller frame (no `name` column): boxes are labeled by index and
	# the bare columns are relabeled in the tooltip.
	X_attr = torch.randn(4, 50)
	annot = pandas.DataFrame({
		'start': [5, 25],
		'end': [12, 33],
		'attribution': [0.42, 0.91],
		'p-value': [1e-5, 1e-9],
	})

	fig, ax = plt.subplots()
	interactive_logo(X_attr, ax=ax, annotations=annot)

	texts = [t.get_text() for t in ax.texts]
	assert "0" in texts
	assert "1" in texts

	html = mpld3.fig_to_html(fig)
	assert "seqlet attribution:" in html
	assert "seqlet p-value:" in html


def test_interactive_logo_label_false_draws_no_corner_text(X_attr, annotations):
	mpld3 = pytest.importorskip("mpld3")
	fig, ax = plt.subplots()
	interactive_logo(X_attr, ax=ax, annotations=annotations, label=False)

	# No corner labels drawn, but the hover tooltip is unaffected.
	assert len(ax.texts) == 0
	html = mpld3.fig_to_html(fig)
	assert "htmltooltip" in html
	assert "GATA2" in html
