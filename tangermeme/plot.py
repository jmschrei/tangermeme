# plot.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import numpy
import pandas
import logomaker

from matplotlib import pyplot as plt


def plot_logo(X_attr, ax, color=None, annotations=None, start=None, end=None, 
	ylim=None, spacing=4, n_tracks=4, score_key='score', show_extra=True):
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

	ax: matplotlib.pyplot.subplot
		The art board to draw on.

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
			y_offset = 0.1
			for i in range(annotations_.shape[0]):
				if motifs[motif_start:motif_end, i].max() == 0:
					if i < n_tracks:
						text = "{}: ({:3.3})".format(motif, score)
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
	