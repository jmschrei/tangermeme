# bigwig_writer.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""A bigWig writer for the tests, adapted from figwig's tests/writers.py.

`write_raw_bigwig` lays a bigWig out exactly as given, including layouts that
pybigtools' writer does not produce: varStep and fixedStep sections,
uncompressed blocks, more than 256 chromosomes, overlapping or unsorted
intervals, and intervals that cover no base.
"""

import zlib
import struct


def write_raw_bigwig(path, chroms, sections, compress=True,
	chrom_block_size=None):
	"""Write a bigWig with one data block per section and no zoom levels.

	`chroms` is a dict of chromosome lengths, in sorted order. Each section is
	(chrom, kind, step, span, items): bedGraph (kind 1) items are (start, end,
	value) tuples, varStep (kind 2) items are (start, value) tuples, and a
	fixedStep (kind 3) section's items are (start, [values]). The blocks are
	indexed in the order given. With `chrom_block_size` smaller than the number
	of chromosomes, the chromosome tree is a root over leaves of that many
	chromosomes, as UCSC writes it for a genome with more than 256.
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

	ctree_offset = 64 + 40
	block = chrom_block_size or len(names)
	leaves = []
	for k in range(0, len(names), block):
		leaf = struct.pack('<BBH', 1, 0, len(names[k:k + block]))
		for name in names[k:k + block]:
			leaf += name.encode().ljust(key_size, b'\0') + struct.pack('<II',
				ids[name], chroms[name])
		leaves.append((names[k], leaf))

	ctree = struct.pack('<IIIIQQ', 0x78CA8C91, block, key_size, 8, len(names), 0)
	if len(leaves) == 1:
		ctree += leaves[0][1]
	else:
		position = ctree_offset + len(ctree) + 4 + len(leaves) * (key_size + 8)
		ctree += struct.pack('<BBH', 0, 0, len(leaves))
		for first, leaf in leaves:
			ctree += first.encode().ljust(key_size, b'\0') + struct.pack('<Q',
				position)
			position += len(leaf)
		ctree += b''.join(leaf for _, leaf in leaves)

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
