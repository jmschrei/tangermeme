# Finding and fixing operations DeepLIFT does not hook

A rule is registered per module type. Covered: the elementwise activations
(`ReLU`, `ReLU6`, `LeakyReLU`, `RReLU`, `PReLU`, `ELU`, `CELU`, `SELU`,
`GELU`, `SiLU`, `Mish`, `Sigmoid`, `LogSigmoid`, `Tanh`, `Softplus`,
`Softshrink`), `MaxPool1d`, `MaxPool2d`, and the coupling ops `Softmax`
(any non-batch axis), `LayerNorm`, `RMSNorm`, `GLU` and
`tangermeme.deep_lift_shap.BilinearOp`. `Conv*`, `Linear`, `BatchNorm*`,
`Embedding`, `AvgPool*`, adds, concatenations, reshapes and `masked_fill`
with a constant are linear and need nothing. Anything else non-linear is
treated as linear, silently.

Two things escape the rules and need different checks. A **custom module**
with no rule is visible in `model.modules()`. A **functional call**,
`torch.softmax`, `.softmax()`, `F.gelu`, `a * b` between two activations,
`matmul`, `einsum`, is not a module, so nothing hooks it.

## Audit

`scripts/audit_deeplift.py` at the skill root walks one forward pass,
reports every op whose owning module has no rule, then runs `deep_lift_shap`
and measures summation-to-delta relative to the output scale:

```
python <skill root>/scripts/audit_deeplift.py --model my_models:Net() --length 2114 --target 3
```

`--model` is `module:expression`, evaluated after importing the module from
the working directory. Importable too: `audit(model, X)` returns
`[(module type, op), ...]`; `check_convergence(model, X, target=)` returns
the relative error and the number of library warnings. `mul`, `div`,
`matmul`, `bmm` and `einsum` are reported on purpose, since a product with a
constant (`scores / d ** 0.5`) is linear and a product of two activations is
not; read the line to tell them apart. The audit sees one input's path; the
relative convergence error is the measurement that settles it. On the
fixtures it was run on: a conv model with a functional softmax gate reported
`('GatedConv', 'softmax')` and `('GatedConv', 'mul')` with relative error
0.30 and zero library warnings; `TransformerEncoderLayer` reports its fused
`multi_head_attention_forward`; `enformer_pytorch`'s `AttentionPool` reports
`softmax` and `mul`.

## Route 1: rewrite the op as a module (preferred)

The ops are parameterless, so the rewrite changes structure only and the
checkpoint loads unchanged. Functional `softmax` becomes `torch.nn.Softmax`;
a functional product, `matmul` or `einsum` becomes `BilinearOp(equation)`,
which is `torch.matmul` for `equation=None`, the elementwise product for
`"...,...->..."`, and `torch.einsum(equation, left, right)` otherwise, with
transposes, reshapes and scaling kept outside it. Subclass the original when
its parameters live in the parent:

```python
class GatedHookable(Gated):
	def __init__(self, seq_len=100):
		super().__init__(seq_len)
		self.softmax = torch.nn.Softmax(dim=-1)
		self.gate = BilinearOp("...,...->...")

	def forward(self, X):
		h = self.gate(self.softmax(self.a(X)), self.b(X))
		return self.d(h.reshape(h.shape[0], -1))

hookable = GatedHookable()
hookable.load_state_dict(model.state_dict())
with torch.no_grad():
	assert torch.allclose(model(X), hookable(X), atol=1e-6)
```

`load_state_dict` accepts a model that computes something else, so the
`allclose` is not optional. Then re-run the audit. `torch.nn.MultiheadAttention`
and `TransformerEncoderLayer` have no functional form to swap; write the
block out of `BilinearOp` and `torch.nn.Softmax` (`MultiHeadAttention` and
`TransformerBlock` in the package's `tests/toy_models.py` are the worked
versions). Attention pooling as in `enformer_pytorch` needs both its softmax
and its window product replaced; swapping only the softmax leaves the delta
where it was.

## Route 1b: a custom elementwise module reuses the rescale rule

```python
from tangermeme.deep_lift_shap import _nonlinear
X_attr = deep_lift_shap(model, X, additional_nonlinear_ops={MyActivation: _nonlinear})
```

`_nonlinear` divides `delta_out / delta_in`, so input and output shapes must
match; a module that also reduces raises a shape error. Split it: the
elementwise part in its own module, registered, and the reduction in the
parent. The same dict overrides a built-in rule on a key collision. A
subclass of a registered type (`class MyReLU(torch.nn.ReLU)`) raises
`KeyError` in the backward hook, because hooks are installed by `isinstance`
and dispatched by exact type; register the subclass here.

## Route 2: `integrated_gradients_op` for a module with no closed form

```python
from tangermeme.deep_lift_shap import integrated_gradients_op
X_attr = deep_lift_shap(model, X, additional_nonlinear_ops={AttentionPool: integrated_gradients_op(K=8)})
```

Integrates the module's Jacobian along the straight path from reference to
observed activation with `K` Gauss-Legendre nodes. Single-input modules only
(`BilinearOp` raises). Register the smallest module that contains the
unhooked ops: a block that also holds a ReLU puts a kink in the path and
`K=8` still warned where the pooling layer alone converged at `K=4`. Pick
the smallest `K` that satisfies summation-to-delta; cost grows with `K`.
It is not a check on a closed-form rule: for an elementwise op the two agree
to floating-point noise, for a coupling op (`RMSNorm`, attention pooling)
they differ by orders of magnitude more and raising `K` does not close the
gap. Both satisfy summation-to-delta; neither is the more accurate one.

## Limitations

- A row masked entirely with `-inf` makes the forward pass `NaN` before
  attribution starts; use a large finite mask and check
  `torch.isfinite(model(X)).all()`.
- A module called several times in one forward pass (a shared `self.relu`,
  a weight-shared layer) is handled; no rewrite needed.
