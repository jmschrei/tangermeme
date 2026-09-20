# DeepLIFT/SHAP attributions in tangermeme

`tangermeme.deep_lift_shap.deep_lift_shap` runs a model "backwards" to score the
contribution of each input base to an output. It corrects several issues found in
other DeepLIFT/SHAP implementations and supports custom non-linearities.

## Signature (defaults that matter)

```python
deep_lift_shap(
    model, X,
    args=None,
    target=0,                       # WHICH output — see below
    batch_size=32,                  # example-REFERENCE pairs, not examples — see below
    references=dinucleotide_shuffle,# callable OR a precomputed tensor
    n_shuffles=20,                  # references per example when references is a fn
    return_references=False,
    hypothetical=False,
    warning_threshold=0.001,        # additivity (convergence-delta) check
    additional_nonlinear_ops=None,
    print_convergence_deltas=False,
    raw_outputs=False,
    only_warn=False,
    dtype=None, device=None,        # device None -> CUDA if available else CPU
    random_state=None,
    verbose=False,
)
```

## Footgun #1 — `target=` is mandatory for multi-task models

`target` defaults to `0`. If your model emits multiple outputs in one tensor and
you forget `target=`, you will silently get attributions for output 0, not an
error. Always set it explicitly for multi-task models:

```python
X_attr = deep_lift_shap(model, X, target=267, random_state=0)
```

If the model returns *multiple tensors* (a list), `target` cannot select among
them — wrap the model so its forward returns a single tensor first (see
`references/model-wrapping.md`).

## Footgun #2 — reproducibility needs `random_state=`

The default `references=dinucleotide_shuffle` draws `n_shuffles` random
backgrounds per example, so attributions vary run to run. Pass `random_state=` to
make them deterministic (required when capturing regression values). For comparing
attributions **across tasks or models**, holding `random_state` *and* `n_shuffles`
fixed is mandatory — otherwise each call attributes against different backgrounds
and the differences you see are noise, not signal (see
`references/comparing-models.md`). A custom `references=` callable has the
signature `f(X, n, random_state) -> (n_examples, n, len(alphabet), length)`; if you
only shuffle a sub-span, attributions are exactly `0` outside it — don't interpret
those positions.

`ersatz.local_dinucleotide_shuffle` satisfies that contract and can be passed
straight in. It shuffles within consecutive bins rather than across the whole
sequence, so the background keeps the GC and repeat structure varying along the
window instead of averaging it flat — closer to the original in every respect
except the motif content you are trying to isolate.

## Footgun #3 — `only_warn=True` still raises inside shuffling

`only_warn=True` downgrades validation errors on `X`, but the internal
`dinucleotide_shuffle` re-validates and will still raise on a malformed `X`.
Workaround: precompute references from a valid one-hot input and pass them in. Note
the standalone `dinucleotide_shuffle` parameter is `n=` (the `n_shuffles` name is
`deep_lift_shap`'s own), and a precomputed `references=` tensor must have shape
`(batch, n_shuffles, len(alphabet), length)`:

```python
refs = dinucleotide_shuffle(X_valid, n=20, random_state=0)  # (batch, 20, 4, length)
X_attr = deep_lift_shap(model, X, references=refs, only_warn=True)
```

## `batch_size` counts example-reference pairs

`batch_size` is the number of example×reference pairs run at once, not the number
of examples. With the default `n_shuffles=20`, a single example already expands to
20 forward/backward passes, so `batch_size=32` does **not** mean 32 examples in
flight. If you hit OOM, lower `batch_size` (or `n_shuffles`); the attributions are
unchanged, only the per-step memory differs.

## Convergence deltas — the correctness signal (read this)

DeepLIFT/SHAP has an additive property: per-example attributions should sum to the
prediction difference between the sequence and its references. The convergence
delta measures the violation, and its **magnitude tells you the cause**:

- **delta > ~0.01**: a non-linearity the hooks don't handle is unregistered — a
  real **correctness bug**. Fix it (below), don't ship the attributions.
- **delta ~1e-3 to 1e-5 (often 1e-7 on CPU)**: benign floating-point error.
  Architecture-dependent; safe to ignore.

tangermeme auto-warns above `warning_threshold`; use `print_convergence_deltas=True`
to see per example-reference-pair values.

**High deltas do NOT produce garbage logos — this is the dangerous part.** A model
with an unregistered op can still highlight motif-shaped patterns that look real but
*vanish* once the op is registered. Motif-shaped ≠ correct; only low deltas are.
Cross-check a suspicious logo against the actual prediction.

### Registering a custom non-linearity

A rule is registered per module *type*. These are the types `deep_lift_shap` and
`pisa` both cover:

- **Elementwise activations** (rescale rule) — `ReLU`, `ReLU6`, `LeakyReLU`,
  `RReLU`, `PReLU`, `ELU`, `CELU`, `SELU`, `GELU`, `SiLU`, `Mish`, `GLU`,
  `Sigmoid`, `LogSigmoid`, `Tanh`, `Softplus`, `Softshrink`.
- **Pooling** — `MaxPool1d`, `MaxPool2d`.
- **Coupling ops with closed forms** — `Softmax`, `LayerNorm`, `RMSNorm`, and
  `tangermeme.deep_lift_shap.BilinearOp`.

Anything not in that list and not linear is a hole. `Conv*`, `Linear`,
`BatchNorm*`, `Embedding`, `AvgPool*`, adds, concatenations and
reshapes need no custom rule.

```python
from tangermeme.deep_lift_shap import _nonlinear
X_attr = deep_lift_shap(model, X, additional_nonlinear_ops={MyActivation: _nonlinear})
```

The catch: `_nonlinear` divides `delta_out / delta_in`, so it **must be registered
on a layer with equal input and output shape**. If your op also reduces (e.g. a
profile head that does `logits * softmax(logits)` then `.sum()`), split it: put the
elementwise, shape-preserving part in its own `nn.Module`, register *that*, and do
the reduction in the parent wrapper. Registering the reducing layer raises a
size-mismatch error.

### Precision: CPU vs CUDA, and the fp64 escape hatch

The same model gives **higher deltas on CUDA than CPU** (parallel reductions reorder
float sums) — thresholds tuned on CPU may need raising on GPU. To disambiguate
"unregistered op" from "precision noise," re-run a few examples on CPU. For
genuinely precision-driven deltas, `deep_lift_shap(model.double(), X.double(),
references=refs.double(), ...)` drops them to ~1e-16 (slower; fp64).

When the delta is real rather than precision noise, the next two sections find
the op responsible and fix it. When a closed-form rule is impractical to derive,
reach for `integrated_gradients_op` instead. When nothing can be hooked at all,
switch to ISM — see `references/saturation_mutagenesis.md`.

## Auditing a model for unhooked operations

Two things can go wrong, and they need different checks. A **custom module** with no
rule is visible in `model.modules()`. A **functional** call — `torch.matmul`,
`F.softmax` — is not a module at all, so there is nothing for a rule to attach to
and nothing in the module list to notice. Both are silently treated as linear.

Drop this in and run it before trusting any attribution from an unfamiliar model.
It walks the forward pass, attributes every torch call to the innermost module
executing at the time, and reports the ones whose owner has no rule:

```python
import torch
from torch.nn.modules.module import register_module_forward_hook
from torch.nn.modules.module import register_module_forward_pre_hook

from tangermeme.deep_lift_shap import BilinearOp

RULED = {
    torch.nn.ReLU, torch.nn.ReLU6, torch.nn.RReLU, torch.nn.SELU,
    torch.nn.CELU, torch.nn.GELU, torch.nn.SiLU, torch.nn.Mish, torch.nn.GLU,
    torch.nn.ELU, torch.nn.LeakyReLU, torch.nn.Sigmoid, torch.nn.Tanh,
    torch.nn.Softplus, torch.nn.Softshrink, torch.nn.LogSigmoid,
    torch.nn.PReLU, torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.Softmax,
    torch.nn.LayerNorm, torch.nn.RMSNorm, BilinearOp,
}

LINEAR = {
    "linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "add", "sub",
    "cat", "stack", "reshape", "view", "permute", "transpose", "flatten",
    "unsqueeze", "squeeze", "chunk", "split", "sum", "mean", "pad", "to",
    "type", "clone", "detach", "contiguous", "dim", "size", "dropout",
    "avg_pool1d", "avg_pool2d", "embedding", "batch_norm", "expand",
    "repeat", "narrow", "index_select", "roll", "__get__", "__getitem__",
    "_set_grad_enabled", "empty", "zeros", "ones", "arange", "t",
}


def audit(model, X, extra_rules=()):
    """List every op in `model`'s forward that no DeepLIFT rule covers."""

    ruled, stack, found = RULED | set(extra_rules), [], set()

    def _push(module, inputs):
        stack.append(module)

    def _pop(module, inputs, outputs):
        stack.pop()

    class _Trace(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            name = getattr(func, "__name__", "?")
            if name not in LINEAR and stack and type(stack[-1]) not in ruled:
                found.add((type(stack[-1]).__name__, name))
            return func(*args, **(kwargs or {}))

    pre = register_module_forward_pre_hook(_push)
    post = register_module_forward_hook(_pop)
    try:
        with _Trace(), torch.no_grad():
            model(X)
    finally:
        pre.remove()
        post.remove()

    return sorted(found)
```

Each finding is `(owning module type, operation)`. Measured on four models:

| model | `audit` reports | meaning |
|---|---|---|
| conv + `MultiheadAttention` | `('MultiheadAttention', 'multi_head_attention_forward')` | `MultiheadAttention` module internally invokes a fused kernel |
| `TransformerEncoderLayer` | the above, plus `('TransformerEncoderLayer', 'relu')` | functional activation, rewritable |
| the `Gated` model below, `torch.softmax(a(X), -1) * b(X)` | `('Gated', 'mul')`, `('Gated', 'softmax')` | functional ops, rewritable |
| attention built from `BilinearOp` + `nn.Softmax` | only the constant `'div'` and `'mul'` | false positives — see below |

**`mul`, `div`, `matmul`, `bmm` and `einsum` are reported deliberately.** A product
is linear when one operand is a constant (`scores / head_dim ** 0.5`, a `* beta`
scaling) and bilinear when both vary. The audit cannot tell those apart, so it
reports both and you dismiss the constant ones by reading the line. A product of
two activations that it reports is real and needs `BilinearOp`.

### The convergence delta is the empirical test

The audit is a static read of one forward pass; it can miss an op reached only on
another input, and it flags things that turn out to be fine. The delta is the
measurement that settles it, because an unhooked non-linearity breaks
summation-to-delta by construction:

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    X_attr = deep_lift_shap(model, X, references=refs, random_state=0)

unhooked = [c for c in caught if issubclass(c.category, RuntimeWarning)]
```

An empty list means every non-linearity on the path your input took is hooked.
A non-empty one means something is not, and the audit above tells you what.
Use `warnings.simplefilter("error", RuntimeWarning)` to turn it into a hard
failure in a script. A fully hooked model sits near floating-point noise; an
unhooked non-linearity raises the delta by orders of magnitude, and the more of
the model's output it accounts for the larger the jump. The absolute values are
model-specific, so compare a model against itself before and after a fix rather
than against a number quoted here.

## Fixing what the audit finds

### Route 1 — rewrite the op as a hookable module (preferred)

These ops are parameterless, so the rewrite changes the *structure* of the
forward and nothing else, and the original weights load straight in. Swap a
functional `softmax` for `torch.nn.Softmax`, a functional product or `matmul` or
`einsum` for `BilinearOp`, a functional `F.gelu` for `torch.nn.GELU`.

`BilinearOp(equation)` is `torch.matmul` when `equation is None`, the elementwise
product `left * right` for the string `"...,...->..."`, and `torch.einsum(equation,
left, right)` otherwise. It uses its operands exactly as passed, so any transpose,
reshape or scaling stays outside it.

```python
import torch
from tangermeme.deep_lift_shap import BilinearOp


class Gated(torch.nn.Module):
    """The original: a functional softmax and a functional product."""

    def __init__(self, seq_len=100):
        super().__init__()
        self.a = torch.nn.Conv1d(4, 8, 3, padding='same')
        self.b = torch.nn.Conv1d(4, 8, 3, padding='same')
        self.d = torch.nn.Linear(8 * seq_len, 1)

    def forward(self, X):
        h = torch.softmax(self.a(X), dim=-1) * self.b(X)
        return self.d(h.reshape(h.shape[0], -1))


class GatedHookable(Gated):
    """The rewrite: same parameters, every non-linearity behind a module."""

    def __init__(self, seq_len=100):
        super().__init__(seq_len)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.gate = BilinearOp("...,...->...")

    def forward(self, X):
        h = self.gate(self.softmax(self.a(X)), self.b(X))
        return self.d(h.reshape(h.shape[0], -1))


model = Gated()                                 # the trained original
hookable = GatedHookable()
hookable.load_state_dict(model.state_dict())    # parameter set is unchanged
```

Subclassing the original is the cheapest form of the rewrite when the parameters
live in the parent, as above. When they do not, write the new class standalone and
`load_state_dict(model.state_dict())` still works as long as every parameter keeps
its name — `Softmax` and `BilinearOp` add none.

**Always verify the transfer before attributing**, since `load_state_dict` is happy
with a model that computes something else entirely:

```python
with torch.no_grad():
    assert torch.allclose(model(X), hookable(X), atol=1e-6)
```

Then re-run the delta check. On the pair above the original emits one convergence
warning and the rewrite emits none.

`torch.nn.MultiheadAttention` has no functional form to swap — see the limitations
below. Write the block out of `BilinearOp` and `torch.nn.Softmax` instead; the
worked example is `MultiHeadAttention` in `tests/toy_models.py`, and
`TransformerBlock` in the same file is the stacked pre-norm/post-norm version.

### Route 1b — a custom elementwise module reuses the built-in rescale rule

A module of your own that is elementwise and shape-preserving does not need a new
rule, only registration of the existing one:

```python
from tangermeme._deep_lift_utils import _nonlinear

X_attr = deep_lift_shap(model, X, references=refs, random_state=0,
    additional_nonlinear_ops={MyActivation: _nonlinear})
```

`_nonlinear` divides `delta_out / delta_in`, so it **must go on a layer with equal
input and output shape**. If your op also reduces (a profile head that does
`logits * softmax(logits)` then `.sum()`, say), split it: put the elementwise,
shape-preserving part in its own `nn.Module`, register *that*, and do the
reduction in the parent wrapper. Registering the reducing layer raises a
size-mismatch error. The same dictionary overrides a built-in rule when the key
collides, which is how you replace one you disagree with.

### Route 2 — `integrated_gradients_op` for a module with no closed form

If deriving a closed-form rule (and/or its corresponding Jacobian-vector-product) is too
involved, you may register a numeric rule for it instead. `integrated_gradients_op(K=8)`
returns a hook that integrates the module's Jacobian along the straight-line path from
the reference activation to the observed one:

```python
from tangermeme.deep_lift_shap import integrated_gradients_op

X_attr = deep_lift_shap(model, X, references=refs, random_state=0,
    additional_nonlinear_ops={MyOpaqueModule: integrated_gradients_op(K=8)})
```

Caveats:

- **Single-input modules only.** The rule re-runs `module(z)` with one tensor.
  Registering it for a two-operand module such as `BilinearOp` raises
  `TypeError: forward() missing 1 required positional argument`.
- **It is slower.** The rule evaluates the module `K` times, so the cost grows
  with `K`, and the overhead on the whole call depends on how much of the model's
  cost sits outside the wrapped module and on the device. The factor varies enough
  between models and between CPU and CUDA that quoting one here would mislead;
  time your own model with and without it.
- **Choice of K** Minimally, select a `K` that satisfies summation-to-delta. Increasing
  `K`, if runtime/compute allow, can improve attribution quality in some cases. There is
  currently no well-defined guidance for selecting `K`. Choice of `K` is partly dependent
  on the nature of the function whose partial derivatives we are integrating, and partly on
  the data regime where it is being evaluated. 

Prefer route 1 when it applies. Route 2 is the fallback for what route 1 cannot
reach.

### `integrated_gradients_op` is a fallback, not a validator

It is tempting to register it over a layer that already has a closed-form rule to
"check" that rule. That does not work, because the two are not the same quantity.
A path integral and the rescale secant coincide only for an elementwise function:

- elementwise (an unregistered activation, rescale vs. `integrated_gradients_op`)
  — they agree to floating-point noise.
- coupling across positions (`RMSNorm`, closed form vs. `integrated_gradients_op`)
  — they differ by orders of magnitude more, and **raising `K` does not close the
  gap**. That is how you tell the two apart: quadrature error shrinks with
  `K`, a difference in method does not.

Both satisfy summation-to-delta. Neither is a more accurate version of the other,
so a disagreement tells you nothing about which is right. Summation-to-delta is
the only property to check.

## Known limitations

- **`torch.nn.MultiheadAttention` is not attributable as-is.** It dispatches into a
  fused `multi_head_attention_forward` that computes its softmax and both matmuls
  below the module layer, so there is no module for a rule to attach to and no
  functional call to swap. Rewrite the block.
- **`torch.nn.TransformerEncoderLayer` is not attributable as-is** for the same
  reason — it contains a `MultiheadAttention` — and additionally calls its
  activation functionally.
- **Subclassing a registered op raises `KeyError`.** Hooks are registered by
  `isinstance` but dispatched by exact `type`, so `class MyReLU(torch.nn.ReLU)`
  gets hooks installed and then fails in the backward hook with
  `KeyError: <class 'MyReLU'>`. Pass the subclass through
  `additional_nonlinear_ops={MyReLU: _nonlinear}` instead of relying on
  inheritance. This predates the transformer rules and applies to every op.
- **A row masked entirely with `-inf` gives `NaN`.** Causal and padding masks are
  handled, but a query attending to nothing makes torch's own softmax return `NaN`
  in the forward pass, so the model is ill-posed before attribution is involved.
  Use a large finite mask instead, which softmaxes that row to uniform. Check
  `torch.isfinite(model(X)).all()` first when your model masks attention.

### Reused modules corrupt attributions — check for this

A module assigned once and called several times, the common `self.relu` reused
across layers, is a single instance in the module tree. The forward hooks cache
`module.input` and `module.output` on that one instance, so the second call
overwrites what the first stored and the backward rule reads the wrong
activations. Weight sharing is the same problem.

It is not silent, but it surfaces as a raised convergence delta rather than as
anything naming the cause, so it is easy to misread as an unhooked op. When the
call sites see different shapes it raises instead, usually a shape mismatch from
inside a hook.

```python
from collections import Counter

def find_reused_modules(model, X):
	"""Modules called more than once in a forward pass, with their counts."""
	counts, handles = Counter(), []
	for name, module in model.named_modules():
		handles.append(module.register_forward_hook(
			lambda mod, inp, out, n=name: counts.update([n])))
	try:
		model(X)
	finally:
		for h in handles:
			h.remove()
	return {n: c for n, c in counts.items() if c > 1 and n != ""}

find_reused_modules(model, X)   # {} is clean; {'relu': 2} is the problem
```

Report any hit whose type has a rule in the table above. The fix is to give
each call site its own instance — `self.relu1`, `self.relu2` — which changes
nothing about the model, since these ops are parameterless. For a genuinely
weight-shared parameterized layer, instantiate one module per call site and tie
the parameters instead of the modules.

## hypothetical vs projected attributions

- Default (`hypothetical=False`): attributions are projected onto the observed
  bases — what you plot as a logo of the actual sequence.
- `hypothetical=True`: per-base hypothetical contributions across all four bases.
  Use these to build motif patterns / contribution-weight matrices (CWMs)
  downstream — **not** as the seqlet-caller input.

The seqlet callers take the projected output collapsed over the channel axis
(`X_attr.sum(dim=1)`) — see `references/seqlets.md`.

## Return type

Returns a tensor shaped like `X` (e.g. `(batch, 4, length)`). With
`return_references=True` it returns an `AttributionReferencesResult(attributions,
references)` NamedTuple — unpack positionally or by attribute.

## Plotting

```python
from tangermeme.plot import plot_logo
plot_logo(X_attr[0], ax=ax)   # tangermeme has its own logo plotting (no logomaker)
```

## Composing with perturbations

`deep_lift_shap` satisfies the `func=` contract, so it drops into `marginalize`,
`ablate`, `variant_effect.*`, etc. to get attributions before/after an edit —
route attribution kwargs via `additional_func_kwargs` (see
`references/func-pattern.md`).

## Related references

`references/saturation_mutagenesis.md` (the forward-pass
alternative), `references/seqlets.md` (consuming projected attributions),
`references/model-wrapping.md` (single-tensor requirement),
`references/comparing-models.md` (shared references across models).
