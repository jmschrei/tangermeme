# bigwig_writer.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""A bigWig writer for the tests, adapted from figwig's tests/writers.py.

`write_raw_bigwig` lays a bigWig out exactly as given, including layouts that
pybigtools' writer does not produce: varStep and fixedStep sections,
uncompressed blocks, overlapping or unsorted intervals, and intervals that
cover no base.
"""

import zlib
import struct


def write_raw_bigwig(path, chroms, sections, compress=True):
	"""Write a bigWig with one data block per section and no zoom levels.

	`chroms` is a dict of chromosome lengths, in sorted order. Each section is
	(chrom, kind, step, span, items): bedGraph (kind 1) items are (start, end,
	value) tuples, varStep (kind 2) items are (start, value) tuples, and a
	fixedStep (kind 3) section's items are (start, [values]). The blocks are
	indexed in the order given.
	"""

	names = list(chroms)
	ids = {name: i for i, name in enumerate(names)}
	key_size = max(len(name) for name in names)

	blocks = []
	for chrom, kind, step, span, items in sections:
		if kind == 1:
			body = b''.join(struct.pack('<IIf', s, e, v) for s, e, v in items)
			n, start = len(items), min(s for s, _, _ in items)
			end = max(e for _, e, _ in items)
		elif kind == 2:
			body = b''.join(struct.pack('<If', s, v) for s, v in items)
			n, start = len(items), min(s for s, _ in items)
			end = max(s for s, _ in items) + span
		else:
			start, values = items
			body = b''.join(struct.pack('<f', v) for v in values)
			n, end = len(values), start + step * (len(values) - 1) + span

		header = struct.pack('<IIIIIBBH', ids[chrom], start, end, step, span,
			kind, 0, n)
		blocks.append((ids[chrom], start, end, header + body))

	# The chromosome tree as one leaf holding every chromosome.
	ctree_offset = 64 + 40
	ctree = struct.pack('<IIIIQQ', 0x78CA8C91, len(names), key_size, 8,
		len(names), 0) + struct.pack('<BBH', 1, 0, len(names))
	for name in names:
		ctree += name.encode().ljust(key_size, b'\0') + struct.pack('<II',
			ids[name], chroms[name])

	data_offset = ctree_offset + len(ctree)
	data, index, largest = struct.pack('<Q', len(blocks)), [], 0
	for chrom, start, end, raw in blocks:
		payload = zlib.compress(raw) if compress else raw
		largest = max(largest, len(raw))
		index.append((chrom, start, chrom, end, data_offset + len(data),
			len(payload)))
		data += payload

	index_offset = data_offset + len(data)
	rtree = struct.pack('<IIQIIIIQII', 0x2468ACE0, len(index), len(index),
		index[0][0], index[0][1], index[-1][2], index[-1][3], index_offset, 1,
		0) + struct.pack('<BBH', 1, 0, len(index))
	for entry in index:
		rtree += struct.pack('<IIIIQQ', *entry)

	header = struct.pack('<IHHQQQHHQQIQ', 0x888FFC26, 4, 0, ctree_offset,
		data_offset, index_offset, 0, 0, 0, 64, largest if compress else 0, 0)
	with open(path, 'wb') as handle:
		handle.write(header + struct.pack('<Qdddd', 0, 0, 0, 0, 0) + ctree +
			data + rtree)
