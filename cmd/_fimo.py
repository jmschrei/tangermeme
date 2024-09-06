# the fimo command-line tool
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import pandas

from tangermeme.tools.fimo import fimo


import time

def _run_fimo(args):
	"""An internal function for running FIMO from the command-line."""

	hits = fimo(args.motif, args.sequence, bin_size=args.bin_size, 
		eps=args.epsilon, threshold=args.threshold, 
		reverse_complement=not args.norc)

	hits = pandas.concat(hits).sort_values('p-value')
	print(hits.to_string(index=False))
