# Annotating and counting motifs (tangermeme.annotate)

Row formats: `(example_idx, start, end)` for the `*_annotations`
perturbations (`references/motif-effects.md`); `(example_idx, annotation_idx)`
for counting; `(example_idx, annotation_idx, start, end)` for spacing.

```python
motif_idxs, pvals = annotate_seqlets(X, seqlets, "motifs.meme", n_nearest=1, n_jobs=4)
# (n_seqlets, n_nearest) int32 indices into the MEME file order, and float64 TOMTOM p-values
```

- Indices, not names; `-1` means no match under the threshold, and the
  default threshold is permissive, so a label is the closest motif, not a
  confident hit. `names = list(read_meme(path))` gives the same order, keys
  like `'MA2538.1 CGGBP1'`; index with `.long()` or `.numpy()`, not int32
  tensors into a list. `motifs` may also be a `{name: PWM}` dict.
- TOMTOM handles overhangs, so this is for short spans; scanning whole
  sequences is FIMO, in `memelite`, where attribution-thresholded hits from
  a redundant database double-count one span under several motifs.
- With a redundant database the label goes to whichever near-duplicate
  scored best, so counts per motif split across duplicates; use a clustered
  set (Vierstra archetypes, HOCOMOCO clustered) when counts matter. Fewer
  than 25 motifs makes the p-values depend on the target set (warning).

```python
counts = count_annotations((seqlets['example_idx'], motif_idxs[:, 0]), dtype=torch.int32)  # (n_examples, n_motifs)
co = pairwise_annotations(df2, symmetric=True, unique=True)                                 # (n_motifs, n_motifs), int64
sp = pairwise_annotations_spacing(df4, max_distance=100, dtype=torch.int32)                 # (n_motifs, n_motifs, max_distance)
```

- `count_annotations` and `pairwise_annotations_spacing` default to
  `uint8`, which wraps at 256 (300 counts read as 44) and makes
  `argsort(-counts)` put zeros first. Pass `dtype=torch.int32`. `shape=`
  pins the output size across runs; `dim=0`/`dim=1` reduce without the
  full matrix.
- `pairwise_annotations` with `symmetric=True` counts each pair in both
  orders. `symmetric=False` orders rows by the motif that came first only
  with `unique=False`; the default `unique=True` collapses repeats by
  sorting on annotation index, which discards input order.
- Spacing is `start_right - end_left`, capped at `max_distance`; the row
  order does not matter but pass one DataFrame so each motif keeps its
  coordinates. Overlapping spans within an example raise `ValueError`
  (abutting is fine); FIMO hits from a redundant database and flanked
  seqlets overlap, so merge first.
