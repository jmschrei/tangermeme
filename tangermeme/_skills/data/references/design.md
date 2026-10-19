# Sequence design (tangermeme.design)

Discrete search: random screening and greedy or beam motif substitution.
Gradient-based minimal edits of a template are `ledidi` (its own skill).
`X_bar` is the designed sequence, `y_bar` the target passed as `y=`.

```python
X_bar = screen(model, shape=(4, 2114), y=y_bar, batch_size=1000, n_best=1, max_iter=4, random_state=0)
X_bar = greedy_substitution(model, X[i:i+1], y=y_bar, motifs=motif_strings, max_iter=10, input_mask=mask)
X_bar = beam_substitution(model, X[i:i+1], y=y_bar, motifs=motif_strings, beam_size=4, n_best=1, max_iter=10)
construct = greedy_marginalize(model, X_backgrounds, y=y_delta, motifs=motif_strings, max_iter=5)
```

- The model must return one tensor `(batch, n_outputs)`; wrap a
  `(profile, counts)` model first (`references/model-wrapping.md`). `y` is
  `(1, n_outputs)`, or a list for a multi-output model. `loss` returns one
  value per element (`MSELoss(reduction='none')` is the default); a reducing
  loss breaks the per-candidate ranking.
- `greedy_substitution` and `beam_substitution` take one sequence
  (`X.shape[0] == 1`, else `ValueError`); loop over examples.
- `reverse_complement=True` (default) also tries each motif's reverse
  complement; `verbose` prints motif indices into `motifs + reverse
  complements`, so an index `>= len(motifs)` is a reverse complement.
  `input_mask` marks allowed start positions; a motif may run past the
  masked span. Pass `['A', 'C', 'G', 'T']` as `motifs` for single-base
  design.
- `screen` draws `batch_size` candidates per iteration (not `predict`'s
  batch size), keeps the top `n_best`, and warns about a broadcast when
  `y=None`; `shape` excludes the batch axis.
- `greedy_marginalize`: `y` is the desired change `f(X + construct) - f(X)`,
  `X` is a set of backgrounds, and the return is a variable-width construct
  (decode with `characters(construct, allow_N=True)`; `N` where motifs
  cancel), bounded by `max_spacing` (default 12).

## Stop rules

| | `max_iter=-1` (default) | `tol` stops on |
|---|---|---|
| `greedy_substitution` | **runs zero rounds, returns `X` unchanged** | `improvement <= tol` |
| `beam_substitution`, `greedy_marginalize` | no limit | `improvement <= tol` |
| `screen` | no limit (loops forever with an unreachable `tol`) | the `n_best`-th candidate's loss `<= tol` |

Always pass a positive `max_iter`. `tol` is an improvement floor, not a loss
target: a `tol` equal to the loss you want ends the run after one round.
There is no "reach a value" rule. For a threshold goal use a one-sided loss,
`lambda y, y_hat: (y - y_hat).clamp(min=0) ** 2`, which stops one round after
the threshold is crossed because no edit improves a loss of 0; near the
threshold the squared gap makes improvements small, so with `tol=1e-3` it
can stop about 0.03 short. Once the loss saturates at 0, `beam_substitution`
ranks its tied rows in insertion order and fills the remaining slots with
pointless edits: re-rank `n_best` rows with `predict` and take the highest
prediction or fewest changed bases.

Before choosing a target, predict the start and a few hundred random
sequences; on a K562 ChromBPNet, random hg38-frequency sequence already
predicted 6 to 9.8 log counts. The first round pays numba compilation and
GPU warm-up (14 s against 2.7 s after, for 9,680 candidates on an H200).
Balancing idiom: `y = predict(model, X)` with only the target tasks
overwritten holds the others near baseline instead of masking them out.
