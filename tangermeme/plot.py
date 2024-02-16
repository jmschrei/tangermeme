# plot.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import pandas
import logomaker

def plot_logo(X_attr, ax, color=None, annotations=None, start=None, end=None, 
	ylim=None, spacing=4, n_tracks=4, show_extra=True):
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
		A set of annotations with the following columns in order:
			- motif: the name of the motif
			- start: the start of the hit relative to the window provided
			- end: the end of the hit relative to the window provided
			- strand: the strand the hit is on (optional)
			- score: the score of the hit
			- two more optional columns

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
		annotations_ = annotations[annotations['start'] > start]
		annotations_ = annotations_[annotations_['end'] < end]
		annotations_ = annotations_.sort_values(["score"], ascending=False)

		ylim = ylim or max(abs(X_attr.min()), abs(X_attr.max()))
		plt.ylim(-ylim, ylim)

		motifs = numpy.zeros((end-start, annotations_.shape[0]))
		for _, row in annotations_.iterrows():
			(motif, motif_start, motif_end, _, score, _, _) = row
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

						plt.plot(xp, yp, color='0.7', linewidth=2)        
						plt.text(xp[0], -ylim*(y_offset+0.1), text, 
							color='0.7', fontsize=9)
						
					elif show_extra:
						s = motif_start

						motifs[motif_start:motif_start+len(motif)*2, i] = 1
						y_offset += -0.1 + 0.2*(n_tracks) + 0.1*(i-n_tracks)
						
						plt.text(motif_start, -ylim*(y_offset+0.1), motif, 
							color='0.7', fontsize=9)    
						
					break

	return logo