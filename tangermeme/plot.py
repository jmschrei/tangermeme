# plot.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import numpy
import pandas
import logomaker

from matplotlib import pyplot as plt

from .deep_lift_shap import deep_lift_shap


def plot_logo(X_attr, ax=None, color=None, annotations=None, start=None, 
	end=None, ylim=None, spacing=4, n_tracks=4, score_key='score', 
	show_extra=True, show_score=True):
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

	spacing: int or None, optional
		The number of positions between motifs to include when determining
		overlap. If there is enough overlap, kick the motif down to the next
		row of annotations. Default is 4.

	n_tracks: int, optional
		The number of tracks of annotations to plot with bars before simply
		putting the name of the motif. Default is 4.

	score_key: str, optional
		When annotations are provided, the name of the key to use as a score.
		Must have the semantics that a higher value means a "better" annotation.
		Default is 'score'.

	show_extra: bool, optional
		Whether to show motifs past the `n_tracks` number of rows that include
		the motif and the bar indicating positioning. If False, do not show
		those motifs. Default is True.
	
	show_score: bool, optional
		Whether to show the score of the hit. Sometimes, the annotation can take up
		too much space already and the scoring information is not as helpful, so
		disabling this will only display the motif hit name. Default is True.


	Returns
	-------
	ax: plt.subplot
		A subplot that contains the plot.
	"""

	try:
		import matplotlib.pyplot as plt
	except:
		raise ImportError("Must install matplotlib before using.")

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

	if annotations is not None:
		start, end = start or 0, end or X_attr.shape[-1]

		annotations_ = annotations[annotations['start'] > start]
		annotations_ = annotations_[annotations_['end'] < end]
		annotations_ = annotations_.sort_values([score_key], ascending=False)

		ylim = ylim or max(abs(X_attr.min()), abs(X_attr.max()))
		ax.set_ylim(-ylim, ylim)
		r = ylim*2

		motifs = numpy.zeros((end-start, annotations_.shape[0]))
		for _, row in annotations_.iterrows():
			motif = row.values[0]
			motif_start = int(row['start'])
			motif_end = int(row['end'])
			score = row[score_key]

			motif_start -= start
			motif_end -= start
			y_offset = 0.2
			for i in range(annotations_.shape[0]):
				if motifs[motif_start:motif_end, i].max() == 0:
					if i < n_tracks:
						text = str(motif)
						if show_score:
							text += ': ({:3.3})'.format(score)

						motifs[motif_start:motif_end, i] = 1
						y_offset += 0.2*i
						
						xp = [motif_start, motif_end]
						yp = [-ylim*y_offset, -ylim*y_offset]

						ax.plot(xp, yp, color='0.3', linewidth=2)        
						ax.text(xp[0], -ylim*(y_offset+0.1), text, 
							color='0.3', fontsize=9)
						
					elif show_extra:
						s = motif_start

						motifs[motif_start:motif_start+len(str(motif))*2, i] = 1
						y_offset += -0.1 + 0.2*(n_tracks) + 0.1*(i-n_tracks)
						
						ax.text(motif_start, -ylim*(y_offset+0.1), motif, 
							color='0.7', fontsize=9)    
						
					break

	return logo


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
	