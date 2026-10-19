# Per-position attribution (tangermeme.pisa)

PISA attributes every output position of a profile to every input position,
with the same hooks and rules as `deep_lift_shap`.

```python
pisa(model, X, args=None, batch_size=32, references=dinucleotide_shuffle,
	n_shuffles=20, return_references=False, hypothetical=False,
	warning_threshold=0.001, additional_nonlinear_ops=None,
	print_convergence_deltas=False, raw_outputs=False, device=None,
	random_state=None, verbose=False)
# -> (batch, n_outputs, len(alphabet), length)
```

- The model must return `(batch, n_outputs)`, one value per output position
  for one task, so a profile head `(batch, n_tasks, length)` is wrapped to
  `(batch, length)` first; every entry of axis 1 is explained, there is no
  `target=`. Any other trailing axis raises `ValueError`.
- `X` must be floating point: int8 raises `only Tensors of floating point
  dtype can require gradients`. The one entry point that does not upcast.
- No `only_warn`; unknown characters are rejected.
- Memory is `n_outputs` times a `deep_lift_shap` call at the same
  `batch_size`; lower `batch_size` first.
- Rules, references, seeds, convergence deltas and `additional_nonlinear_ops`
  behave as in `references/deep_lift_shap.md` and
  `references/deep_lift_shap-unhooked-ops.md`; `integrated_gradients_op`
  works here too.
