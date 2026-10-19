# DeepLIFT/SHAP attributions (tangermeme.deep_lift_shap)

```python
deep_lift_shap(
	model, X,                       # X one-hot (batch, len(alphabet), length), int8 is fine
	args=None,                      # extra model inputs, one row per example; reused for its references
	target=0,                       # indexes axis 1 of the output; the default is silent on multi-task models
	batch_size=32,                  # example-reference PAIRS per step, not examples
	references=dinucleotide_shuffle,# callable f(X, n, random_state) -> (batch, n, len(alphabet), length), or that tensor
	n_shuffles=20,
	return_references=False,        # True -> AttributionReferencesResult(attributions, references)
	hypothetical=False,             # True -> scores for every base, for CWMs; not seqlet-caller input
	warning_threshold=0.001,        # ABSOLUTE bound on each pair's convergence delta
	additional_nonlinear_ops=None,  # {ModuleType: rule}, see references/deep_lift_shap-unhooked-ops.md
	print_convergence_deltas=False,
	raw_outputs=False,              # True -> multipliers before projection, not per-reference attributions
	only_warn=False,
	dtype=None, device=None,        # device None -> CUDA if available; model device and mode restored after
	random_state=None,
)                                   # -> (batch, len(alphabet), length) on the CPU, model dtype
```

The model must return `(batch, n_outputs)`; trailing axes of length 1 are
dropped, any other trailing axis (a profile) raises `ValueError`. Wrap first:
`references/model-wrapping.md`.

## Defaults that silently give the wrong answer

- `target=0` on a multi-task model attributes output 0 with no warning.
- No `random_state` means different references, and different attributions,
  on every call. Reference `k` of every example is seeded `random_state + k`,
  so the same references come from
  `torch.stack([dinucleotide_shuffle(X, n=1, random_state=[r + k] * len(X))[:, 0] for k in range(n_shuffles)], dim=1)`;
  `dinucleotide_shuffle(X, n=n_shuffles, random_state=r)` gives different ones.
  Comparing attributions across models or tasks needs the same references:
  precompute once and pass `references=` the tensor.
- The convergence warning compares each delta with the absolute
  `warning_threshold`. A model whose outputs are of order 1e-3 can be 30%
  wrong with no warning. Check the relative error yourself:
  `python scripts/audit_deeplift.py --model module:expr` (skill root
  relative; see the next section), or `attr.sum((1, 2))` against
  `model(X)[:, target] - model(refs).mean over references`.
- The seed fixes the references, not the kernels: on a GPU two calls with
  the same seed differ at 1e-8 to 1e-5. For bit-identity set
  `torch.backends.cudnn.deterministic = True` and `benchmark = False`
  first, or compare with a tolerance.

## Convergence deltas

Attributions of each example-reference pair sum to the difference in output.
The delta is the violation. Relative to the output difference: 1e-5 or
below is floating-point noise (higher on CUDA than on the CPU, an order of
magnitude higher under `dtype=torch.bfloat16` or `float16`, 1e-16 at fp64);
anything larger means an operation the hooks do not cover and the logo may
still look motif-like. The warning is a `RuntimeWarning`, one per batch.
Agreement with ISM does not detect a missing rule: on a model with an
unhooked softmax the wrong and the corrected attributions correlated equally
with ISM. Finding and fixing the operation: `references/deep_lift_shap-unhooked-ops.md`.
A model whose package ships its own rules (cherimoya:
`cherimoya.deep_lift_shap.attribution_ops()`; bpnet-lite: `_ProfileLogitScaling`,
`_Log`, `_Exp` with `_nonlinear`) needs them passed as `additional_nonlinear_ops`,
or every pair warns.

## References

- A callable gets `(X, n, random_state)`. `local_dinucleotide_shuffle`
  qualifies and keeps composition per bin instead of per window; its
  `bin_size` defaults to 1024 and must be shorter than the sequence, so bind
  it for inputs under about 1 kb:
  `references=partial(local_dinucleotide_shuffle, bin_size=128, min_bin_size=64)`.
  Shuffling a sub-span (`partial(dinucleotide_shuffle, start=, end=)`) gives
  exactly 0 outside it.
- Unknown characters (all-zero columns) are rejected. Route:
  `only_warn=True, references=partial(dinucleotide_shuffle, allow_N=True)`;
  those positions get 0. `only_warn=True` alone still raises inside the
  default shuffle.
- A tensor `references=` may be a `PackedSequence` of batch shape
  `(batch, n_shuffles)`; references come back packed when `X` or
  `references` is packed, otherwise in the model's dtype (float32 for int8 `X`).

## Memory and speed

Every hooked module caches activations for `2 x batch_size` rows, so memory
follows `batch_size x length x width x depth`, not parameter count: a 197 KB
cherimoya of width 32 on 2,114 bp took 3.7 GB at `batch_size=64`. Lower
`batch_size` or `n_shuffles` on OOM; attributions are unchanged. The first
call in a process pays 15 to 35 s of CUDA warm-up; time the second. Models
need not live on the GPU: keep them on the CPU and pass `device=`, each call
moves one over and back.

## Output

Projected (default): the observed base's score, zero elsewhere; sum over
axis 1 for the per-position track that `recursive_seqlets` takes
(`references/seqlets.md`). `hypothetical=True`: a score for every base at
every position, the input for CWMs. `raw_outputs=True`: the multipliers
before projection, `(batch, n_shuffles, ...)`, which `hypothetical_attributions`
turns into attributions. `plot_logo(X_attr[0], ax=ax)` draws one example
(`references/plot.md`). As `func=` inside `marginalize`, `ablate` or
`variant_effect.*`, route its kwargs through `additional_func_kwargs`
(`references/func-pattern.md`).

tangermeme below 1.5.0 attributed stacked or overlapping max-pools, reused
modules, GLU and half precision wrongly; recompute on 1.5.0 or later.
