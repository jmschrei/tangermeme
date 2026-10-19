# The `func=` plug-point

```python
func(model, X, args=None, **kwargs) -> torch.Tensor | list[torch.Tensor]
```

`predict`, `deep_lift_shap`, `saliency` and `saturation_mutagenesis` satisfy
it. `ablate`, `marginalize`, `space`, `variant_effect.*` and the
`*_annotations` variants default `func=predict`; `dependency_map` defaults
`func=saliency` and requires `(-1, len(alphabet), length)` back, so
`predict` is invalid there. The outer function returns whatever `func`
returns, before and after the edit, so `marginalize(..., func=deep_lift_shap)`
gives attributions before and after in the same `PerturbationResult`.

`product.apply_pairwise` / `apply_product` take `func` as a required first
positional argument and so do not satisfy the contract themselves; bind
first, `func=partial(apply_pairwise, predict)`, or put the product on the
outside: `apply_pairwise(marginalize, model, X, args=(...), motif="TGACGTCA")`
(`references/product.md`).

**Other `func=` arguments are not this contract.** `design.screen(func=random_one_hot)`
is a candidate generator `func(shape, random_state=)`. `predict`,
`saliency` and `saturation_mutagenesis` take `func=` as a post-processing
hook on each batch of predictions (pick a head, apply a non-linearity),
applied identically to the reference and every perturbation.
`plot_attributions(func=deep_lift_shap)` names the attribution function.

## Routing kwargs

Bare `**kwargs` on the outer function are forwarded to `func`, so
`marginalize(model, X, motif, func=deep_lift_shap, target=3, random_state=0)`
does reach `deep_lift_shap`. The collision is a name both layers consume
(`func` inside `dependency_map`, `random_state` in `ablate`, `batch_size`
in `screen`): put anything meant for `func` in `additional_func_kwargs={...}`,
which goes only to `func`. Default `None`; the dict is copied, not mutated.

## Extra model inputs

`args=(tensor, ...)` pairs row `i` of each arg with example `i` and flows
through to `model(X, *args)`; `deep_lift_shap` reuses example `i`'s args for
its references. `marginalize` and `space` have no `args` parameter of their
own; it reaches `predict` through `**kwargs`. In `product.*`, `args` are the
axes to sweep, not per-example rows.

## Packed sequences

A `PackedSequence` as `X` reaches `predict`, `deep_lift_shap`, `pisa`,
`saliency`, `saturation_mutagenesis` and the `ersatz` shuffles given as
`shuffle_fn=` packed (also inside `functools.partial`). Any other `func` or
`shuffle_fn`, including one you write, receives its int8 one-hot encoding.
