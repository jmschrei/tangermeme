# Sequence design with tangermeme.design

`tangermeme.design` builds sequences that produce a desired model output. It is
discrete/combinatorial (screening and greedy motif insertion). For gradient-based
minimal edits of a template, use the sibling `ledidi` library instead — see the
end of this file.

## Naming (matches the project conventions)

- `X_bar` — designed sequences (what these functions return).
- `y_bar` — the desired/target output (passed as `y=`).

## screen — random generate, keep the best

```python
from tangermeme.design import screen

X_bar = screen(model, shape=(4, 2114), y=y_bar, batch_size=1000, n_best=1, random_state=0)
```

- `shape` is the **per-sequence** shape `(len(alphabet), length)` — it **excludes
  the batch dimension**. `batch_size` is how many candidates are drawn and screened
  per iteration (it is *not* forwarded to `predict`'s own batch_size).
- Generates candidates (default `func=random_one_hot`), scores each against `y` with
  `loss`, keeps the top `n_best`. Loops until `loss < tol` (set a positive
  `max_iter` as a safety cap). Simplest baseline; seed/sanity-check before greedy.

## greedy_substitution — build toward a goal with a motif library

```python
from tangermeme.design import greedy_substitution

X_bar = greedy_substitution(model, X, y=y_bar, motifs=motif_list,
                            reverse_complement=True, max_iter=10)
```

- Each round, tries substituting every motif at candidate positions and keeps the
  single edit that most reduces `loss`; repeats until `loss < tol` or `max_iter`
  motifs have been placed.
- **`max_iter` defaults to `-1`, which is a silent no-op** — the loop is
  `iteration < max_iter`, so a non-positive value runs zero iterations and returns
  `X` unchanged. The greedy functions require a **positive `max_iter`**.
- **`X` must have batch size 1** — it designs one sequence at a time.
- `reverse_complement=True` also considers each motif's reverse complement.
- `input_mask` restricts which positions may be edited (e.g. the middle 200 bp →
  ~10× speedup); `output_mask` restricts which outputs the loss is computed over.
- **Two modes from one function:** pass real motifs → greedy motif implantation;
  pass `['A', 'C', 'G', 'T']` as `motifs` → greedy single-nucleotide (ISM-style)
  design that escapes a motif library (use `reverse_complement=False` and expect
  many more iterations).
- **Elimination is automatic** — set a low target and the same function removes
  harmful motifs without being told which are bad; add/remove can interleave.

## beam_substitution — greedy_substitution with a beam

```python
from tangermeme.design import beam_substitution

X_bar = beam_substitution(model, X, y=y_bar, motifs=motif_list,
                          beam_size=4, n_best=1, max_iter=10)
```

- A strict generalization of `greedy_substitution`: instead of committing to the
  single best edit each round, it keeps the `beam_size` lowest-loss **complete**
  sequences and expands all of them. **`beam_size=1` reproduces
  `greedy_substitution` exactly.**
- Every beam member is always a full, fixed-length sequence (each step
  *substitutes* into an existing sequence, never grows one), so there is no
  partial-sequence scoring problem — the search is over trajectories through
  edit-space. Larger beams recover good multi-edit combinations that greedy
  prunes after a locally-suboptimal first edit.
- Each round expands all members, pools every candidate, and keeps the **global**
  top `beam_size` by **absolute loss** (not per-step improvement); current members
  are carried forward so the beam never regresses, and identical sequences are
  de-duplicated so it does not collapse onto one sequence.
- `n_best` returns that many sequences (≤ `beam_size`), ranked low→high loss,
  shape `(n_best, len(alphabet), length)`.
- **`max_iter` differs from the greedy functions:** here `-1` means *no limit*
  (like `screen`), with `tol` as the stop — not the greedy no-op. Cost scales
  ~linearly with `beam_size`. `input_mask`/`output_mask`/`reverse_complement`/
  `args`/`loss` behave exactly as in `greedy_substitution`.

## greedy_marginalize — build a construct against backgrounds

```python
from tangermeme.design import greedy_marginalize
construct = greedy_marginalize(model, X_backgrounds, y=y_delta, motifs=motif_list,
                               max_iter=5)   # positive max_iter required (see above)
```

Semantics differ from `greedy_substitution` in three ways that are easy to miss:

- `y` is the **desired change** `f(X + construct) − f(X)`, *not* a target prediction.
- `X` is a **set of background sequences**, not one sequence to edit.
- it returns a **variable-width one-hot construct** (e.g. `(4, 51)`), not a full
  sequence — decode with `characters(construct, allow_N=True)`; it contains `N`
  where overlapping motifs cancel. `max_spacing` (default 12) bounds the search.

## The loss / target contract (footgun)

- `loss` defaults to `torch.nn.MSELoss(reduction='none')`. **`reduction='none'`
  is required** — design needs a per-candidate / per-example loss vector to rank
  edits, not a single scalar. PyTorch losses average over the batch by default, which
  hides which example is best; a custom `loss(y, y_hat)` must take `y` of shape
  `(n, ...)` and return a per-example `(n, ...)` (don't reduce it yourself either).
- **Balancing-losses idiom:** instead of `output_mask`-ing off-target tasks, set
  `y = predict(model, X)` and overwrite only the tasks you want to change — this
  holds the others near baseline rather than ignoring them.
- `y` (the target `y_bar`) and `loss` must be shaped consistently with the model
  output. For multi-output models, either pass a list `y` matching the outputs or
  wrap the model to a single objective head (see [model-wrapping.md](model-wrapping.md)),
  and use `output_mask` to focus the loss.
- The choice of loss + target *is* the design objective — get these right before
  worrying about iterations. Euclidean/MSE is the usual default; supply a custom
  callable for subtler objectives (e.g. maximize one head while holding another).

## When to use ledidi instead

`tangermeme.design` is discrete and greedy. When you want **gradient-based,
minimal edits** to a specific template sequence to hit precise output
characteristics, reach for the `ledidi` library — it optimizes a continuous
relaxation and tends to find smaller, more targeted edits than greedy
substitution.

## Related references

[model-wrapping.md](model-wrapping.md) (collapsing multi-output models to a design
objective), [func-pattern.md](func-pattern.md) (the `func=` candidate generator in
`screen`).
