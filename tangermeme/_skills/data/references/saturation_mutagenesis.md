# Saturation mutagenesis (tangermeme.saturation_mutagenesis)

```python
saturation_mutagenesis(model, X, args=None, start=0, end=-1, batch_size=32,
	target=None,          # int | slice | None; None AVERAGES over every output
	hypothetical=False, raw_outputs=False, dtype=None, device=None, verbose=False,
	func=None,            # post-processing hook on each batch of predictions, not the plug-point
	fast=True)
# -> (batch, len(alphabet), end - start), projected onto the observed bases
```

Forward passes only, through `predict`, so no rule can be missing. Use it
when DeepLIFT deltas cannot be fixed, or the model is massively multi-task
(every output comes from the same passes). Agreement between ISM and
DeepLIFT does not validate the DeepLIFT rules: a wrong and a corrected
attribution correlated equally with ISM on a test model.

- `target=None` averages over all outputs; on a 919-task model that is one
  noisy map, not 919. Pass an int or slice. The output shape is `(batch, 4, length)` either way.
- Non-integer `X` (a soft or scaled one-hot) is truncated toward zero by the
  int8 cast, usually to all zeros, with only a `TangermemeWarning`.
- Cost is `(end - start) * (len(alphabet) - 1)` sequences per example;
  window with `start`/`end`, which equals slicing a full run up to rounding.
  Triage motif hits by summing `(X_ism * X)` over each hit's window.
- The score at the observed base is the mean-centered prediction change,
  not Z-scored, so regions compare head to head. `hypothetical=True` gives
  all bases; `raw_outputs=True` gives `SaturationMutagenesisRawResult(y0, y_hat)`
  with `y_hat` `(batch, 4, length, n_outputs)` and the identity slot holding `y0`.

## `fast=True` (fastISM, the default)

Captures the model with `torch.export`, caches each original sequence's
activations, and computes every edit only over the positions each
convolutional layer's receptive field reaches; dense, norm-over-length and
attention layers run at full length on the cached activations. Results
match `fast=False` up to rounding.

- A job that would finish in under thirty seconds at `fast=False` skips
  capture silently, so toy models never use it.
- Otherwise one batch is computed both ways; capture failure (data-dependent
  control flow, forward hooks), disagreement beyond rounding, or a slower
  fast path fall back with a `TangermemeWarning` saying why.
- Gains: about 2x on ChromBPNet, 4x on Beluga; none on a model that
  normalizes or pools over the whole length early (LegNet, cherimoya).
- On a GPU use `batch_size` of 256 or so; at 32 the fast path is often not
  faster and falls back.
