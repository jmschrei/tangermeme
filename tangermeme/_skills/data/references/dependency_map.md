# Saliency and dependency maps

## saliency (tangermeme.saliency)

```python
saliency(model, X, args=None, target=0, batch_size=32, hypothetical=False,
	gradient_correction=True, func=None, dtype=None, device=None, verbose=False)
# -> (batch, len(alphabet), length) on the CPU, same sign convention as deep_lift_shap
```

One backward pass per batch, so it is the method for far more sequences than
`deep_lift_shap` or ISM can afford, or a model no rule covers. It is a
tangent at the observed sequence and saturates: a base the model is certain
about can carry a near-zero gradient. `gradient_correction=True` subtracts
the per-position mean over the alphabet (Majdandzic et al. 2023); with
`hypothetical=True` every position then sums to zero. `func=` is the
post-processing hook as in `predict`, run inside the autograd graph, so it
must be differentiable. Drops into `plot_logo`, `recursive_seqlets` (after
`.sum(dim=1)`) and the `func=` plug-point. Accepts all-zero columns, which
`deep_lift_shap` rejects.

## dependency_map (tangermeme.dependency_map)

```python
dependency_map(model, X, args=None, start=0, end=-1, func=saliency,
	additional_func_kwargs=None, device=None, verbose=False, **kwargs)
# -> (batch, length, end - start) on the CPU
```

`dmap[:, j, i]` is the mean absolute change in the attribution at position
`j` when the base at `start + i` is substituted, averaged over the three
substitutions. A model that treats positions independently gives a diagonal
map; off-diagonal blocks are motif pairs, stripes are flanks a motif depends
on. Adapted from Tomaz da Silva et al. 2025, with the substituted position on
the columns.

- `func` here is the plug-point (`saliency`, `deep_lift_shap`,
  `saturation_mutagenesis`; it must return `(-1, len(alphabet), length)`, so
  `predict` is invalid). Their own `func` post-processing hook collides;
  route it: `additional_func_kwargs={'func': lambda y: y[:, :1]}`.
- Cost is `(end - start) * (len(alphabet) - 1)` attribution calls per
  sequence. Window with `start`/`end`; the columns equal slicing a full run.
- The diagonal is a direct effect, non-zero even for a linear model, and
  several times the rest: `dmap.diagonal(dim1=-2, dim2=-1).zero_()` before
  plotting (valid for `start=0, end=-1`).
- With `func=deep_lift_shap`, hold the references fixed or the map absorbs
  reference noise (three quarters of the values on Tutorial B8). One sequence
  at a time, since references are shuffles of it:

```python
refs = dinucleotide_shuffle(X[i:i+1], n=10, random_state=0)
def dls_fixed(model, X_, args=None, **kwargs):
	return deep_lift_shap(model, X_, args=args,
		references=refs.expand(X_.shape[0], -1, -1, -1), **kwargs)
dmap = dependency_map(model, X[i:i+1], func=dls_fixed)
```
