# Saliency and dependency maps in tangermeme

Two related functions, in two modules:

- `tangermeme.saliency.saliency` — input-times-gradient attribution, the
  cheapest of the three attribution methods (one backward pass per batch).
- `tangermeme.dependency_map.dependency_map` — substitutes every base at every
  position and records how far each substitution moves the attributions
  *everywhere else*.

## saliency

```python
saliency(
    model, X,
    args=None,
    target=0,                       # indexes dim 1 of the model output
    batch_size=32,
    hypothetical=False,
    func=None,                      # POST-PROCESSING hook, as in predict
    dtype=None, device=None,        # device None -> CUDA if available else CPU
    verbose=False,
)                                   # -> (-1, len(alphabet), length), on CPU
```

Returns the **same shape and sign convention** as `deep_lift_shap` and
`saturation_mutagenesis`: signed values of shape `(-1, len(alphabet), length)`,
zero everywhere except the observed base unless `hypothetical=True`. So it drops
straight into `plot_logo`, `recursive_seqlets` (after `.sum(dim=1)`), and the
`func=` plug-point.

**When to use it instead of the other two:** you have far more sequences than
`deep_lift_shap` or ISM can afford, or an op in your model has no DeepLIFT rule
registered. **The trade-off is the linearity assumption** — a gradient is the
tangent at the observed sequence, so it saturates. A base the model is already
certain about can carry a near-zero gradient despite being essential. Prefer
`deep_lift_shap` when you can afford it; see
[deep_lift_shap.md](deep_lift_shap.md).

`func=` here is the `predict`-style post-processing hook (pick a head, reduce a
profile), **not** the `func(model, X)` plug-point — see
[func-pattern.md](func-pattern.md). It runs inside the autograd graph, so it
must be differentiable.

## dependency_map

```python
dependency_map(
    model, X,
    args=None,
    start=0, end=-1,                # which positions get substituted
    func=saliency,                  # THE PLUG-POINT func, not post-processing
    additional_func_kwargs=None,
    verbose=False,
    **kwargs,                       # forwarded to func: target, batch_size, device, ...
)                                   # -> (-1, length, end-start), on CPU
```

`dmap[:, j, i]` is the mean absolute change in the attribution at position `j`
caused by substituting the base at position `start + i`.

**What it is for.** An attribution map says which bases matter; a dependency map
says which bases decide whether the other bases matter. A model that treats
positions independently produces a **purely diagonal** map, so off-diagonal
structure is direct evidence of epistasis the model learned — cooperative motif
pairs show up as blocks, and the flanks a motif depends on show up as stripes.

### Three footguns

1. **The diagonal is not comparable to the rest of the map.** It measures the
   direct effect of a substitution on its own attribution, is non-zero even for
   a linear model, and is typically several times the off-diagonal values, so it
   dominates any color scale. Mask it before plotting:
   `dmap.diagonal(dim1=-2, dim2=-1).zero_()` (valid when `start=0, end=-1`).

2. **Cost is `(end - start) * (len(alphabet) - 1)` attributions per sequence.**
   This is ISM's cost with an attribution in place of each forward pass. Use
   `start`/`end` to restrict to a window rather than scanning a full-length
   locus; the columns are identical to running the full sequence and slicing.

3. **`func` is the plug-point, not post-processing.** `saliency` and
   `saturation_mutagenesis` each take their own `func`, which collides. Route
   the inner one through `additional_func_kwargs`:

   ```python
   dependency_map(model, X, func=saliency,
       additional_func_kwargs={'func': lambda y: y[:, :1]})
   ```

`func` must return `(-1, len(alphabet), length)`; the alphabet axis is summed to
one value per position. `saliency`, `deep_lift_shap`, and
`saturation_mutagenesis` all qualify. Note `deep_lift_shap` rejects `N`
(all-zero) columns, which `saliency` allows.

## Worked example

```python
import torch
from tangermeme.dependency_map import dependency_map
from tangermeme.utils import random_one_hot

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 12, 3)
        self.dense = torch.nn.Linear(12, 1)
    def forward(self, X):
        return self.dense(torch.relu(self.conv(X)).mean(dim=-1))

torch.manual_seed(0)
X = random_one_hot((2, 4, 50), random_state=0).type(torch.float32)
dmap = dependency_map(Model(), X, start=10, end=30, device='cpu')
print(dmap.shape)
# torch.Size([2, 50, 20])

dmap = dependency_map(Model(), X, device='cpu')
dmap.diagonal(dim1=-2, dim2=-1).zero_()      # the diagonal dominates
print(dmap.shape)
# torch.Size([2, 50, 50])
```
