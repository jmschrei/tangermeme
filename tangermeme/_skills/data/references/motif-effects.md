# Motif effects (marginalize, ablate, space)

Each applies `func` (default `predict`) before and after an edit and returns
both; the effect is the delta. Swap `func=deep_lift_shap` for the effect on
attributions, with its kwargs in `additional_func_kwargs`
(`references/func-pattern.md`); attribution runs before and after, each
with `n_shuffles` references, so restrict examples rather than positions
(`deep_lift_shap` has no window argument).

```python
y_before, y_after = marginalize(model, X, "CACGTG", start=None)   # PerturbationResult
y_before, y_after = ablate(model, X, start=990, end=1010, n=20, random_state=0)
y_before, y_afters = space(model, X, ["AGATAA", "CAGCTG"], spacing=[[5], [10], [20]])  # SpaceResult
```

- `marginalize` substitutes (length preserved); `start` is the motif's
  first base; the default centers it at `length // 2 - len(motif) // 2`
  (a 6-mer in 2,114 bp sits at 1054:1060). `X` may hold all-zero columns.
- `ablate` shuffles `[start, end)` `n` times per example; pass `random_state`
  or it differs per call. `shuffle_fn=partial(dinucleotide_shuffle, allow_N=True)`
  for sequences with unknown characters.
- `space`: `spacing` has shape `(n_spacings, n_motifs - 1)`, nested even
  for one gap (`[[10]]`; a flat list raises `ValueError`); sweep with
  `torch.arange(50)[:, None]`. A spacing is the number of background bases
  left between motif `i` and `i + 1`. With `start=None` the whole
  arrangement is re-centered per spacing, so the first motif moves; fix
  `start` to compare against a single-motif marginalization at one
  position. `y_afters` is `(n_examples, n_spacings, ...)`.
- Multi-task models return lists; index before subtracting.
- `marginalize_annotations(model, X, X0, annotations)` and
  `ablate_annotations(model, X, annotations)` take `(n, 3)` rows
  `(example_idx, start, end)` and return `PerturbationAnnotationsResult`
  with a leading annotation axis: `ablate_annotations` gives
  `(n_annotations, 1, ...)` before and `(n_annotations, 1, n, ...)` after,
  the `1` being the single example each row is cut from;
  `marginalize_annotations` gives `(n_annotations, len(X0), ...)` for both.

## Backgrounds for marginalize

The delta should reflect the motif, not the background's composition.
Uniform random sequence is GC-richer than a genome and gives a GC-driven
effect on some models; in one test on a MAX ChIP-seq cherimoya, non-peaks,
dinucleotide-shuffled peaks and uniform random agreed within 0.06 log
counts, so measure it on yours. In order of preference:

1. GC-matched inactive regions: `extract_matching_loci` (`references/io-loci.md`), then `extract_loci`.
2. `dinucleotide_shuffle(X_real, n=1, random_state=0)[:, 0]` or
   `local_dinucleotide_shuffle` (keeps composition per bin) of real regions.
3. `random_one_hot((n, 4, L), probs=numpy.array([[0.295, 0.205, 0.205, 0.295]]))`
   at genome frequencies; `probs` is 2-D (`references/ersatz.md`).
4. Uniform random, for prototyping.

With `func=deep_lift_shap` and `n_shuffles=1` over many backgrounds, the
backgrounds themselves stand in for reference averaging.
