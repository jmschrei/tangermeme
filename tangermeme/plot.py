# plot.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import numpy
import pandas
import logomaker

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib

from .deep_lift_shap import deep_lift_shap


#plot_logo helper functions for placing annotations

def check_box_overlap(box1, box2):
	"""Check if annotation label text boxes overlap."""
	return not(box1.x0>=box2.x1 or box2.x0>=box1.x1 or box1.y0>=box2.y1 or box2.y0>=box1.y1)


def check_box_overlap_bar(box1, box2):
	"""Check if annotation bars overlap."""
	return not(box1.x0>=box2.x1 or box2.x0>=box1.x1 or box1.y0!=box2.y0)


def place_new_box(box, box_list, n_tracks=4, show_extra=True):
    """Place annotation text so that it does not overlap with existing text."""
    box_height = box.y1 - box.y0
    box.y0 -= box_height
    box.y1 -= box_height

    if len(box_list)==0:
        return box,0

    overlap_exists = any([check_box_overlap(box,box2) for box2 in box_list])
    steps_down_taken = 0
    while overlap_exists:
        steps_down_taken += 1
        if steps_down_taken == n_tracks: #beyond the n_tracks limit: make boxes smaller
                if not show_extra:
                        box.y0 -= box_height
                        box.y1 -= box_height
                        return box, steps_down_taken
                box.y1-= box_height
                box.y0-= box_height/2
                box_height = box_height/2
                overlap_exists = any([check_box_overlap(box,box2) for box2 in box_list])
                continue

        box.y0 -= box_height
        box.y1 -= box_height
        overlap_exists = any([check_box_overlap(box,box2) for box2 in box_list])

    return box, steps_down_taken


def place_new_bar(box, box_list, y_step=None, n_tracks=4, show_extra=True):
    """
    Find a position for a new annotation bar such that it does not overlap with previously plotted bars.
    """
    if y_step is None:
        raise ValueError("y_step must be provided.")

    if len(box_list)==0:
        return box, 0

    overlap_exists = any([check_box_overlap_bar(box,box2) for box2 in box_list])
    steps_down_taken = 0
    while overlap_exists:
        steps_down_taken += 1
        box.y0 -= y_step
        box.y1 -= y_step
        overlap_exists = any([check_box_overlap_bar(box,box2) for box2 in box_list])
    return box, steps_down_taken


def plot_logo(X_attr, ax=None, color=None, annotations=None, start=None, 
	end=None, ylim=None, spacing=4, n_tracks=4, score_key='score', 
	show_extra=True, show_score=True, annot_cmap="Set1"):
	"""Make a logo plot and optionally annotate it.

	This function will take in a matrix of weights for each character in a
	sequence and produce a plot where the characters have heights proportional
	to those weights. Attribution values from a predictive model are commonly
	used to weight the characters, but the weights can come from anywhere.

	Optionally, annotations can be provided in the form of a dataframe with
	contents described below in the parameters section. These annotations will
	be displayed underneath the characters in a manner that tries to avoid
	overlap across annotations.

	This function is largely a thin-wrapper around logomaker.


	Parameters
	----------
	X_attr: torch.tensor, shape=(4, -1)
			A tensor of the attributions. Can be either the hypothetical
			attributions, where the entire matrix has values, or the projected
			attributions, where only the actual bases have their attributions
			stored, i.e., 3 values per column are zero.

	ax: matplotlib.pyplot.subplot or None, optional
			The art board to draw on. If None, choose the current artboard.

	color: str or None, optional
			The color to plot all characters as. If None, plot according to
			standard coloring. Default is None.

	annotations: pandas.DataFrame, optional
			A set of annotations with the following columns in any order except for
			`motif_name`, which can be called anything but must come first:

					- motif_name: the name of the motif
					- start: the start of the hit relative to the window provided
					- end: the end of the hit relative to the window provided
					- strand: the strand the hit is on (optional)
					- score: the score of the hit

			These will probably come from the output of the hit caller. Default is
			None.

	start: int or None, optional
			The start of the sequence to visualize. Must be non-negative and cannot
			be longer than the length of `X_attr`. If None, visualize the full
			sequence. Default is None.

	end: int or None, optional
			The end of the sequence to visuaize. Must be non-negative and cannot be
			longer than the length of `X_attr`. If `start` is provided, `end` must 
			be larger. If None, visualize the full sequence. Default is None.

	ylim: tuple or None, optional
			The lower and upper bounds of the plot. Pass the bounds in here rather
			than setting them after calling this function if you want the annotation
			spacing to adjust to it. If None, use the default bounds. Default is
			None.

	n_tracks: int, optional
			The number of rows of annotation labels to plot with bars before simply
			putting the name of the motif. Default is 4.

	score_key: str, optional
			When annotations are provided, the name of the key to use as a score.
			Must have the semantics that a higher value means a "better" annotation.
			Default is 'score'.

	show_extra: bool, optional
			Whether to show motif names past the `n_tracks` number of rows. 
			If False, do not show those motifs. Default is True.

	show_score: bool, optional
			Whether to show the score of the hit. Sometimes, the annotation can take up
			too much space already and the scoring information is not as helpful, so
			disabling this will only display the motif hit name. Default is True.

	annot_cmap: str, list, or matplotlib.colors.ListedColormap, optional
			The colormap to use for the annotations. Rows of annotation labels receive 
			distinct colors, bars are colored according to their corresponding label.
			If a string, must be a valid matplotlib qualitative colormap name 
			If a list, must be a list of colors (list of colornames or list of RGB tuples).
			Labels are colorcoded if there is more than 1 row of annotations, otherwise black.


	Returns
	-------
	ax: plt.subplot
			A subplot that contains the plot.
	"""

	try:
			import matplotlib.pyplot as plt
	except:
			raise ImportError("Must install matplotlib before using.")

	###
	# Main glyph plotting code
	###


	if start is not None and end is not None:
		X_attr = X_attr[:, start:end]

	if ax is None:
		ax = plt.gca()

	df = pandas.DataFrame(X_attr.T, columns=['A', 'C', 'G', 'T'])
	df.index.name = 'pos'

	logo = logomaker.Logo(df, ax=ax)
	logo.style_spines(visible=False)

	if color is not None:
		alpha = numpy.array(['A', 'C', 'G', 'T'])
		seq = ''.join(alpha[numpy.abs(df.values).argmax(axis=1)])
		logo.style_glyphs_in_sequence(sequence=seq, color=color)


	###
	# Handling potentially overlapping annotations
	###


	#Set annotation colormap
	if type(annot_cmap) == str:
		cmap = plt.get_cmap(annot_cmap)
	elif type(annot_cmap) == list:
		cmap = ListedColormap(annot_cmap)
	elif type(annot_cmap) == matplotlib.colors.ListedColormap:
		cmap = annot_cmap

	if annotations is not None:
		start, end = start or 0, end or X_attr.shape[-1]

		annotations_ = annotations[annotations['start'] > start]
		annotations_ = annotations_[annotations_['end'] < end]
		annotations_ = annotations_.sort_values(["start"], ascending=True)

		if len(annotations_) > 0:
			ylim = ylim or max(abs(X_attr.min()), abs(X_attr.max()))
			ax.set_ylim(-ylim, ylim)
			r = ylim*2
			y_offset_bars=ax.get_ylim()[0]/8
			y_offset_labels = 0

			#deterrmine label text size and line width according to figure size
			bbox = ax.get_position()
			fig_width, fig_height = ax.get_figure().get_size_inches()
			width_in = bbox.width * fig_width
			height_in = bbox.height * fig_height
			labelsize=width_in*1.1
			linewidth = width_in*0.25

			#plotting annotation labels
			label_box_objects = []
			visible_label_box_objects = []
			label_text_boxes = []
			text_box_colors = []
			
			for i,(_, row) in enumerate(annotations_.iterrows()):
				motif = row.values[0]
				motif_start = int(row['start'])
				motif_end = int(row['end'])
				score = row[score_key]
				motif_start -= start
				motif_end -= start

				#define label text
				text = str(motif)
				if show_score:
						text += ' ({:3.3})'.format(score)
				text=text.replace(" ", "\n")

				#plot text box in top most position...
				text_box = ax.text(motif_start, y_offset_labels, text, fontsize=labelsize)
				ax.get_figure().canvas.draw()
				bbox = text_box.get_window_extent()

				#...shift box down if it overlaps with previously drawn boxes
				bbox_new, steps_down_taken = place_new_box(bbox, label_text_boxes, n_tracks=n_tracks,show_extra=show_extra)
				bbox_new_transformed = bbox_new.transformed(ax.transData.inverted())

				#color the text according to the number of downshifts
				text_color = cmap(steps_down_taken)
				if steps_down_taken >= n_tracks: #if box is beyond the n_tracks limit, plot in smaller font size and in light grey.
						text_color = (0.7,0.7,0.7)
						text_box.set_fontsize(labelsize/2)
						if not show_extra:
								text_color = (1,1,1,0)
				else:
						visible_label_box_objects.append(text_box)

				text_box.set_position((bbox_new_transformed.x0, bbox_new_transformed.y0))
				text_box.set_color(text_color)
				text_box_colors.append(text_color)
				label_text_boxes.append(bbox_new)
				label_box_objects.append(text_box)

			#plotting annotation bars
			bars_box_objects = []
			bars_boxes = []
			bars_ymins=[]
			for i,(_, row) in enumerate(annotations_.iterrows()):
				motif = row.values[0]
				motif_start = int(row['start'])
				motif_end = int(row['end'])
				score = row[score_key]
				motif_start -= start
				motif_end -= start

				bar_color = text_box_colors[i]
				if bar_color == (0.7,0.7,0.7) or bar_color == (1,1,1,0): #for labels beyond the n_tracks limit, no bar is plotted.
						continue

				xp = [motif_start-0.1, motif_end-0.5]
				yp = [y_offset_bars, y_offset_bars]

				#plot bar in topmost place...
				bar = ax.plot(xp, yp, color='0.3', linewidth=linewidth)
				ax.get_figure().canvas.draw()

				#...shift bar down if it overlaps with previously drawn bars
				bar_box = bar[0].get_window_extent()
				bar_box_new, steps_down_taken = place_new_bar(bar_box, bars_boxes, y_step=linewidth*2, n_tracks=n_tracks,show_extra=show_extra)
				bar_box_new_transformed = bar_box_new.transformed(ax.transData.inverted())
				bar[0].set_ydata([bar_box_new_transformed.y0, bar_box_new_transformed.y1])
				bars_boxes.append(bar_box_new)
				bar[0].set_color(bar_color)
				bars_ymins.append(bar_box_new_transformed.y0)
				bars_box_objects.append(bar[0])


			#shift text boxes down under the lowest bar
			bars_ymin = min(bars_ymins)
			for label_box in label_box_objects:
					label_box.set_y(label_box.get_position()[1] + bars_ymin) 

			#if there is only one row of annotations, set colors to black
			if len(set([vis_box.get_color() for vis_box in visible_label_box_objects])) == 1:
					for label_box in visible_label_box_objects:
							label_box.set_color((0,0,0))
					for bar_box in bars_box_objects:
							bar_box.set_color((0,0,0))

	return logo


def plot_categorical_scatter(X, colors=None, **kwargs):
	"""A scatterplot of category weights across a seq, useful for attributions.
	
	Frequently, when you calculate attributions you are considering a sequence
	that is too long to be easily visualized in the standard character format,
	e.g. in `plot_logo`. One could scan the sequence a few times to find the
	regions of high attribution but that is also time-consuming. This function
	creates a scatterplot where the coloring corresponds to the category (e.g.,
	which nucleotide is there) and the value is the weight encoded in the matrix.
	Basically, this is a way to consider attributions on an entire sequence and
	get a sense for which areas you may want to follow-up with.
	
	This function is not meant solely for attributions, though. You can put any
	value in the matrix to be visualized.
	
	
	Parameters
	----------
	X: torch.Tensor, shape=(alphabet_len, length)
		A sequence to be visualized.
	
	colors: list or None, optional
		The colors to use for each category. By default, uses the standard
		nucleotide color scheme.
	
	**kwargs: any additional arguments
	"""
	
	if colors is None:
		c = [[0, 0.5, 0], [0, 0, 1], [1, 0.65, 0], [1, 0, 0]]

	pos = numpy.arange(X.shape[-1])
	for i in range(X.shape[0]):
		idxs = torch.abs(X).argmax(axis=0) == i    
		plt.scatter(pos[idxs], X.sum(axis=0)[idxs], c=[c[i]], **kwargs)


def plot_attributions(models, X, func=deep_lift_shap, attribute_kwargs=None, 
	plot_kwargs=None, layout=None):
	"""A convenience function for calculating and then plotting attributions.
	
	This function will use one or more models and calculate attributions on one or
	more sequences. These attributions are then plotted according to the format in
	layout, and returned to the user. This is just a wrapped around using an
	attribution function and `plot_logo`, and so additional arguments can be 
	passed into the attribution or the plotting functions using the respective
	kwargs arguments. `start` and `end` are likely 
	to be common arguments to pass
	into `plot_kwargs`.
	
	There are three signatures supported. (1) A list of models are provided with
	the same length as the first dimension in `X`. The i-th model gets applied to
	the i-th element of X. This is the most flexible, as the same model or
	sequence could be passed in multiple times. (2) Rather than a list of models,
	a single model object is provided, and gets applied to all the elements of X.
	(3) Rather than multiple sequences, a single sequence is provided for X as a
	2D tensor. in which case all models are applied to it.
	
	By default, `deep_lift_shap` is used to calculate attributions but any
	function can be used as long as it has the same signature.


	Parameters
	----------
	models: torch.nn.Module or list of torch.nn.Modules
		One or a set of models to apply to the sequence/s.
	
	X: torch.Tensor, shape=(-1, len(alphabet), -1) or (len(alphabet), -1)
		One or more sequences 

	func: func, optional
		The attribution function to use. Default is deep_lift_shap.
	
	attribute_kwargs: dict or None, optional
		Arguments to pass into `func`. Default is None.
	
	plot_kwargs: dict or None, optional
		Arguments to pass into `func`. Default is None.

	layout: 2-tuple or None, optional
		A layout of the subplots, the first two numbers passed into
		`plt.subplot`. If None, assume (n, 1).

	Returns
	-------
	axs: list of plt.axis
		A list of artboards.
	
	attributions: torch.Tensor
		The attributions being visualized here. This is the full output from `func`,
		even when only a portion is visualized.
	"""

	try:
		import matplotlib.pyplot as plt
	except:
		raise ImportError("Must install matplotlib before using.")
	
	
	if isinstance(models, torch.nn.Module):
		if X.ndim != 3:
			raise ValueError("X must have 3 dimensions when only providing one model.")
		
		models = [models for _ in range(X.shape[0])]
		
	elif X.ndim == 2:
		X = X.repeat(len(models), 1, 1)
	
	else:
		if len(models) != len(X):
			raise ValueError("X must have first dimension equal to number of models.")
	
	if layout is None:
		layout = (len(X), 1)
	else:
		if layout[0] * layout[1] < len(X):
			raise ValueError("Layout must have >= number of models.")
			
	attribute_kwargs = attribute_kwargs or {}
	plot_kwargs = plot_kwargs or {}
	
	X_attrs = []
	for i, model in enumerate(models):
		X_attr = deep_lift_shap(model, X[i:i+1], **attribute_kwargs)
		X_attrs.append(X_attr)
	X_attrs = torch.cat(X_attrs, dim=0).detach()
	
	axs = []
	for i in range(len(X_attrs)):
		ax = plt.subplot(*layout, i+1)
		axs.append(ax)
		
		plot_logo(X_attrs[i], ax=ax, **plot_kwargs)
	
	return axs, X_attrs

def plot_pwm(pwm, name=None, alphabet=['A', 'C', 'G', 'T'], eps=1e-7):
	"""Plots an information-content weighted PWM and its reverse complement.

	This function takes in a PWM, where the sum across all values in the
	alphabet is equal to 1, and plots the information-content weighted version
	of it, as well as the reverse complement. This should be used when you want
	to visualize a motif, perhaps from a motif database.


	Parameters
	----------
	pwm: torch.Tensor or numpy.ndarray, shape=(len(alphabet), length)
		The PWM to visualize. The rows must sum to 1.

	name: str or None, optional
		The name to put as the title for the plots. If None, do not put
		anything. Default is None.

	alphabet: list, optional
		A list of characters that comprise the alphabet. Default is
		['A', 'C', 'G', 'T'].

	eps: float, optional
		A small pseudocount to add to counts to make the log work correctly.
		Default is 1e-7.
	"""

	if isinstance(pwm, torch.Tensor):
		pwm = pwm.numpy(force=True)
	
	bg = 0.25 * numpy.log(0.25) / numpy.log(2)

	
	plt.figure(figsize=(8, 2.5))
	ax = plt.subplot(121)
	plt.title(name)
	
	ic = pwm * numpy.log(pwm + eps) / numpy.log(2) - bg
	ic = numpy.sum(ic, axis=0, keepdims=True)
	plot_logo(pwm * ic, ax=ax)
	plt.xlabel("Motif Position")
	plt.ylabel("Information Content (Bits)", fontsize=10)
	
	
	ax = plt.subplot(122)
	plt.title(name + "RC" if name is not None else "RC")
	pwm = pwm[::-1, ::-1]
	ic = pwm * numpy.log(pwm + eps) / numpy.log(2) - bg
	ic = numpy.sum(ic, axis=0, keepdims=True)
	plot_logo(pwm * ic, ax=ax)
	plt.xlabel("Motif Position")
	plt.ylabel("Information Content (Bits)", fontsize=10)
	
	plt.tight_layout()
	plt.show()
	