# Wrapping models

- `X` is `(batch, len(alphabet), length)`; `y = model(X)` must work.
- `predict` accepts a model returning a list of tensors and returns a list;
  `deep_lift_shap`, `pisa`, `saliency`, `saturation_mutagenesis` and
  `design.*` need a single tensor. `deep_lift_shap` and `pisa` read it as
  `(batch, n_outputs)`: `target=` indexes axis 1, `pisa` explains every
  entry of it, trailing axes of length 1 are dropped, any other trailing
  axis (a profile's length) raises `ValueError` naming the shape.
- Several outputs in one tensor `(batch, n_tasks)`: select with `target=`.
  Separate tensors in a list: wrap to return one.
- Put preprocessing and head selection in the wrapper, not in a custom
  `func`. `predict`, `deep_lift_shap`, `pisa`, `saturation_mutagenesis`,
  `design.*` and `product.*` move the model to `device`, switch to eval,
  run without gradients where appropriate, and restore device and mode
  afterward. A sweep over N models keeps them on the CPU in a dict and
  passes `device=` to each call; only one occupies the GPU at a time.

```python
class CountWrapper(torch.nn.Module):
	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, X, *args):
		profile, counts = self.model(X, *args)   # BPNet-style heads
		return counts                            # (batch, 1); or profile.sum(-1)
```

Recipes that need a wrapper and the one trap in each:

| wrapper | trap |
|---|---|
| reverse-complement averaging, `torch.flip(X, dims=(-1, -2))` | stranded outputs must be flipped along their output axis too; as written it is right only for strand-agnostic outputs |
| bake in a zero control track so `args=` is not needed | cherimoya ships `ControlWrapper`, `LogCountWrapper`, `ProfileWrapper`; bpnet-lite ships `CountWrapper`, `ProfileWrapper` |
| concatenate inputs into one tensor and split inside `forward` | for functions with no `args=` |
| `X.permute(0, 2, 1)` for length-first models (`enformer_pytorch`); `y['human']`; `y.sum(dim=-2)` | |
| pad or trim to compare input lengths on one locus set | padding with N is out of distribution; re-`extract_loci` at each model's `in_window` instead |
| run several models and concatenate outputs | feeds `marginalize` or ISM as one model |

Wrappers stack: `CountWrapper(ControlWrapper(model))`. A wrapper does not
hide a model's own fused layers from DeepLIFT: cherimoya needs
`additional_nonlinear_ops=cherimoya.deep_lift_shap.attribution_ops()`,
bpnet-lite's profile head needs `_ProfileLogitScaling` registered with
`_nonlinear` (`references/deep_lift_shap.md`).
