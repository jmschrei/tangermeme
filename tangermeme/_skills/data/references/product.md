# Cartesian products over inputs (tangermeme.product)

```python
apply_pairwise(func, model, X, args=(a, b), ...)   # (len(X), len(a), ...)   a[j] with b[j]
apply_product(func, model, X, args=(a, b), ...)    # (len(X), len(a), len(b), ...)  every (a[j], b[k])
```

`func` is the first positional argument, required. `args` is required too
(`None` raises `TypeError`); the args must all have the same length and are
independent of `len(X)`. `apply_pairwise` zips the args with each other and
crosses the pairs with every example; `apply_product` crosses everything.
Metadata measured in the same cell (a cell state and its read depth) is
paired, and `apply_product` on it fabricates cells. Batches are built
iteratively; the product is never materialized.

- A single arg still adds an axis: `apply_pairwise(predict, model, X, args=(a,))[:, 0]`.
- `device=` goes to the outer call; it is forwarded to `func`.
- Any plug-point `func` works on the inside: `predict`, `deep_lift_shap`
  (`target=`, `random_state=` pass through `**kwargs`), `marginalize`
  (`motif=` passes through). A NamedTuple from the inner function
  (`PerturbationResult`, `AttributionReferencesResult`) comes back as a
  plain list, so unpack positionally. Route the inner function's own `func`
  through `additional_func_kwargs`:

```python
apply_pairwise(marginalize, model, X, args=(cell_states,),
	additional_func_kwargs={'func': deep_lift_shap}, motif="TGACGTCA")
```

- Product on the inside needs `func` bound first; passing `apply_pairwise`
  bare as `marginalize(..., func=apply_pairwise)` misaligns the positionals
  and raises `TypeError: apply_pairwise() missing 1 required positional argument: 'X'`:

```python
from functools import partial
marginalize(model, X, motif, func=partial(apply_pairwise, predict), args=(cell_states,))
```
