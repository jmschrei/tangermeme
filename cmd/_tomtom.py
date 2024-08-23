# the tomtom command-line tool
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import numpy

from tangermeme.io import read_meme
from tangermeme.utils import characters
from tangermeme.utils import one_hot_encode

from tangermeme.tools.tomtom import tomtom


def _run_tomtom(args):
	"""An internal function for running TOMTOM from the command-line."""

	if args.targets is None:
		args.targets = __file__.replace("_tomtom.py", "") 
		args.targets += "JASPAR2024_CORE_non-redundant_pfms_jaspar.meme"
		f = "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_non-redundant_pfms_meme.txt"

		if not os.path.isfile(args.targets):
			print("Downloading {}...".format(f))
			os.system("wget -O {} {}".format(args.targets, f))
		

	targets = read_meme(args.targets)
	target_names = numpy.array(list(targets.keys()))
	target_pwms = list(targets.values())
	target_seqs = numpy.array([characters(x, force=True) for x in target_pwms])

	if os.path.isfile(args.query):
		queries = read_meme(args.query)
		query_names = list(queries.keys())
		query_pwms = list(queries.values())

	else:
		query_names = ['.']
		query_pwms = [one_hot_encode(args.query)]

	query_seqs = numpy.array([characters(x, force=True) for x in query_pwms])

	target_pwms = [pwm.numpy().astype('float64') for pwm in target_pwms]
	query_pwms = [pwm.numpy().astype('float64') for pwm in query_pwms]

	p, scores, offsets, overlaps, strands = tomtom(query_pwms, target_pwms, 
		n_nearest=args.n_nearest, n_score_bins=args.n_score_bins, 
		n_median_bins=args.n_median_bins, n_target_bins=args.n_target_bins, 
		n_cache=args.n_cache, reverse_complement=not args.norc, 
		n_jobs=args.n_jobs)


	q_names, q_seqs = [], []
	t_names, t_seqs, t_ps, t_scores, t_offsets = [], [], [], [], []
	t_overlaps, t_strands = [], []

	for qidx, tidx in zip(*numpy.where(p <= args.thresh)):
		q_names.append(query_names[qidx])
		q_seqs.append(query_seqs[qidx])

		t_names.append(target_names[tidx])
		t_seqs.append(target_seqs[tidx])
		t_ps.append(p[qidx, tidx])
		t_scores.append(int(scores[qidx, tidx]))
		t_offsets.append(int(offsets[qidx, tidx]))
		t_overlaps.append(int(overlaps[qidx, tidx]))
		t_strands.append('+-'[int(strands[qidx, tidx])])

	max_q_name_len = max([len(name) for name in q_names])
	max_q_seq_len = max([len(seq) for seq in q_seqs]) + 2

	max_t_name_len = max([len(name) for name in t_names])
	max_t_seq_len = max([len(seq) for seq in t_seqs]) + 2
	max_offset = max(t_offsets)

	print("Query Name\tQuery Sequence\tTarget Name\tTarget Sequence\tp-value"
		"\tScore\tOffset\tOverlap\tStrand")

	for i in numpy.argsort(t_ps):
		nq = query_pwms[0].shape[-1]
		seq, offset, overlap = t_seqs[i], t_offsets[i], t_overlaps[i]

		if overlap == nq and offset >= 0:
			s1 = seq[:offset].rjust(max_offset)
			s2 = seq[offset:offset + overlap]
			s3 = seq[offset + overlap:]
			seq = s1 + '.' + s2 + '.' + s3

		elif offset >= 0:
			s1 = seq[:offset].rjust(max_offset)
			s2 = seq[offset:] + '-' * (nq - overlap)
			seq = s1 + '.' + s2 + '.'

		else:
			s1 = ' ' * max_offset
			s2 = '-'*-offset + seq[:overlap]
			s3 = seq[overlap:]
			seq = s1 + '.' + s2 + '.' + s3


		str_format = ("{:" + str(max_q_name_len) + "}\t{:" + str(max_q_seq_len)
			+ "}\t{:" + str(max_t_name_len) + "}\t{:" + 
			str(max_t_seq_len+max_offset) + "}\t{:.8}\t{:5}\t{:6}\t{:7}\t{:6}")

		print(str_format.format(q_names[i], q_seqs[i], t_names[i], seq, t_ps[i], 
			t_scores[i], t_offsets[i], t_overlaps[i], t_strands[i]))

