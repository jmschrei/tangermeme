# Logos and annotations (tangermeme.plot)

```python
plot_logo(X_attr[i], ax=ax, start=900, end=1200, annotations=None, score_key='score',
	n_tracks=4, show_extra=True, show_score=True, ylim=None, color=None)
```

- One example, `(len(alphabet), length)`; positives stack up, negatives
  down. Probability PWMs go to `plot_pwm` (information content).
- Without `ax=` it draws on `plt.gca()`; pass `ax=` in a loop.
- `ylim` is read only when `annotations` is given, and then as a scalar
  half-range, whatever the type hint says; for a shared axis call
  `ax.set_ylim(lo, hi)` after `plot_logo`.
- `color=`: one color, a `{char: color}` dict, or a per-position array of
  colors or numbers mapped through `color_cmap` (`color_vmin`/`color_vmax`).
- `plot_attributions(models, X, func=deep_lift_shap, attribute_kwargs=, plot_kwargs=, layout=)`
  attributes and plots a grid; `interactive_logo` adds hover tooltips
  (needs the `interactive` extra).

## Annotations

A DataFrame with `start`, `end` (0-indexed, in the full sequence's
coordinates, the same frame as the `start`/`end` arguments), a score column
named by `score_key`, and the label read from **column 0**, whatever it is
called. FIMO frames lead with `motif_name`; seqlet frames lead with
`example_idx` and hold every example, so filter to one example, pass
`score_key='attribution'`, and insert a name column first:

```python
s = seqlets[seqlets['example_idx'] == i].copy()
s.insert(0, 'motif_name', [names[j] for j in motif_idxs[s.index, 0].tolist()])
plot_logo(X_attr[i], ax=ax, annotations=s, score_key='attribution', start=900, end=1200)
```

A row is drawn only when `row.start > start` and `row.end < end`: a hit
flush with either edge, or at position 0 with the default `start=0`, is
dropped silently. Rows pack into tracks in `start` order, not score order;
past `n_tracks` they collapse into a gray strip (`show_extra=False` hides it).
