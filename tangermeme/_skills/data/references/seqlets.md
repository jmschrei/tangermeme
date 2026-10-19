# Seqlet calling (tangermeme.seqlet)

Both callers take per-position projected attribution sums, `(n, length)`:
`X_attr.sum(dim=1)` of the default `deep_lift_shap` output. A `(n, 4, length)`
or hypothetical tensor fails inside numba with a `TypingError`.

## recursive_seqlets

```python
recursive_seqlets(X, threshold=0.01, min_seqlet_len=4, max_seqlet_len=25,
	additional_flanks=0, n_bins=1000, X_control=None, target_fdr=0.1, return_fdr=False)
# DataFrame: example_idx, start, end, attribution, p-value; sorted by p-value; positive seqlets only
```

- The `p-value` column is a ranking score: positions are treated as
  independent draws from a histogram that contains the signal being
  tested, so `threshold=0.01` is not a 1% false-call rate and means
  different rates in different models. A sequence's p-values depend on the
  other sequences in the call; pass many together, never one.
- Negative seqlets: run on `-attr_sums` (and `X_control=-attr_control`),
  negate the returned `attribution`, drop overlaps with positive calls.
  Not `attr_sums.abs()`: the pooled null hides weaker negatives.
- `min_seqlet_len=1` makes every position pass on its own, so one weak base
  splits a motif. `additional_flanks` pads `start`/`end` and re-sums
  `attribution` over the padded span (the p-value stays on the core);
  flanked spans can overlap, which `pairwise_annotations_spacing` rejects.

### Choosing the threshold with a control

`X_control` picks the largest threshold whose estimated false discovery
rate is at or below `target_fdr` and ignores `threshold`. Control: a few
dinucleotide shuffles of each sequence attributed with the same settings.

```python
shuffles = dinucleotide_shuffle(X, n=3, random_state=0)                 # (n, 3, 4, length)
attr_control = torch.cat([deep_lift_shap(model, shuffles[:, k], target=t, random_state=0).sum(dim=1)
	for k in range(3)])
seqlets, fdr = recursive_seqlets(attr_sums, X_control=attr_control, target_fdr=0.1, return_fdr=True)
```

The estimate is (control calls + 1, scaled to the size of `X`) over calls
on `X`, per threshold; `fdr` has one row per threshold from 0.1 to 1e-30.
When no threshold reaches the target the call warns and returns an empty
frame (typed columns; object columns on 1.5.0 and earlier). A model that
responds to composition (an accessibility count head) calls as many seqlets
on shuffles as on peaks: on a K562 ChromBPNet the estimate bottomed out at
0.16. Look at `fdr` first; fall back to `threshold`. More shuffles steady
the estimate; each costs an attribution pass.

## tfmodisco_seqlets

```python
tfmodisco_seqlets(attr_sums, window_size=21, flank=10)   # example_idx, start, end, attribution; both signs
```

TF-MoDISco's caller: longer, less sensitive spans for motif discovery; it
lowers its threshold to reach a minimum count, so filter by attribution
afterward; boundaries are left-offset (each position holds the sum of the
window to its right).

Label and count the spans: `references/annotate.md`. Draw them:
`references/plot.md` (filter to one example, `score_key='attribution'`, a
name column first).
