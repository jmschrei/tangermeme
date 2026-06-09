# Plotting logos and annotations (tangermeme.plot)

`plot_logo` draws attribution/PWM logos (it replaces logomaker). The function call
itself is simple; the **annotation subsystem** is where the non-obvious behavior
lives, and it is the connective tissue between seqlets/FIMO and the figure.

```python
from tangermeme.plot import plot_logo

ax = plot_logo(X_attr[i], ax=ax, start=900, end=1200)
```

- Pass `X_attr` of shape `(len(alphabet), length)` (one example). Positive
  characters stack up, negative stack down, ordered by magnitude.
- For a **probability PWM** (columns sum to 1), use `plot_pwm` instead — it draws
  the information-content-weighted logo. `plot_logo` is for attributions/weights.
- It draws **only the logo panel** — add titles/labels/limits with normal matplotlib
  afterward. It will make an `ax` if you don't pass one.
- `start`/`end` subset the plotted window **and** are required for annotation
  coordinates to line up, because annotations use absolute coordinates.

## Annotations contract

`annotations=` is a pandas DataFrame with columns `motif_name`, `start`, `end`,
`strand` (currently unused), `score`. Coordinates are 0-indexed; `score` is both
shown and used to order overlapping annotations. Extra columns are ignored.

```python
plot_logo(X_attr[i], ax=ax, annotations=hits, start=900, end=1200)
```

### Seqlet footgun: set score_key and filter per example

Seqlet DataFrames use the column `attribution`, not `score`, so pass
`score_key='attribution'`. And a seqlet DataFrame holds **all examples in one
frame** — filter to the example you're plotting first, or you'll draw another
example's coordinates:

```python
s = seqlets[seqlets['example_idx'] == i]
plot_logo(X_attr[i], ax=ax, annotations=s, score_key='attribution',
          start=900, end=1200)
```

## Three interoperable annotation sources

The same `annotations=` arg accepts: a hand-built DataFrame, FIMO hits (from the
external `memelite` package — `fimo(motifs, X, dim=1)[i]`), and seqlets from
`recursive_seqlets` / `tfmodisco_seqlets`.

## Track packing

Overlapping annotations are greedily packed into rows; past `n_tracks` (default 4)
they collapse into a compact gray name-only strip. `show_extra=False` hides the
overflow; raise `n_tracks` to show more. `color=` sets a single uniform character
color (handy for cross-model overlays); `ylim=` fixes the y-axis for comparison.

## Related references

[seqlets.md](seqlets.md) and [annotate.md](annotate.md) (producing annotations),
[deep_lift_shap.md](deep_lift_shap.md) (producing `X_attr`).
