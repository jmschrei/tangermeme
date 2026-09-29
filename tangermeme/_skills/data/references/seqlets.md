# Seqlet calling (recursive_seqlets / tfmodisco_seqlets)

Seqlets are contiguous spans of high attribution — the "where are the motifs"
step. `tangermeme.seqlet` has two callers with genuinely different behavior.

## The #1 footgun: input is a 2D projected track

Both callers take **per-position projected attribution sums**, shape `(n, length)`
— i.e. the default (`hypothetical=False`) `deep_lift_shap` output collapsed over the
channel axis:

```python
attr_sums = X_attr.sum(dim=1)        # (n, 4, length) -> (n, length)
```

Do **not** pass the `(n, 4, length)` attributions or hypothetical attributions.
(See `references/deep_lift_shap.md`: hypothetical is for building CWMs, not
for seqlet calling.)

## recursive_seqlets — sharp, p-value based

```python
from tangermeme.seqlet import recursive_seqlets

seqlets = recursive_seqlets(attr_sums, threshold=0.01, additional_flanks=0)
# DataFrame columns: example_idx, start, end, attribution, p-value (sorted by p-value)
```

- Returns **positive seqlets only**. For negative ones, run on `-attr_sums` (and
  `X_null=-attr_null` if you use a control), negate the returned `attribution`,
  and drop any that overlap a positive seqlet. Do not run on `attr_sums.abs()`:
  the null then pools both tails, so negatives weaker than the positives are never
  significant.
- `threshold` is a p-value **per span**, computed as if positions were independent
  draws from a histogram. Attributions are correlated along the sequence and the
  default histogram contains the signal being tested, so the fraction of calls that
  are false is not the threshold: on ChromBPNet K562 ATAC-seq count attributions,
  an estimated 61-94% of the calls at `0.01` were no more frequent than on control
  sequences. Choose the threshold with a control (below) rather than by habit.
- `min_seqlet_len` (default 4) is also the smallest span the recursive property must
  hold over. At 1, every single position inside a seqlet must pass the threshold on
  its own, so one weak base splits a motif. `max_seqlet_len` (default 25) raises cost.
- `additional_flanks` pads the reported `start`/`end` **and** the reported
  `attribution` is re-summed over the padded span, so it grows with the flanks. The
  `p-value` is unchanged — it is computed on the un-padded core. Flanks may overlap
  neighbors even though the cores cannot, so flanked spans are not disjoint, their
  attributions can double-count the same positions, and
  `pairwise_annotations_spacing` rejects them (see `references/annotate.md`).
- **Over-calls on a single example** (it can't fit a null from one sequence) — pass
  many examples, or pass a control as `X_null`.

### Choosing the threshold with a control

Attribute two independent dinucleotide shuffles of the sequences with the same
model, target and settings as `X`. One builds the null (`X_null`), the other counts
false calls (`X_control`); counting on the same control that built the null
underestimates the false calls.

```python
from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.seqlet import recursive_seqlets
from tangermeme.seqlet import recursive_seqlets_fdr

shuffles = dinucleotide_shuffle(X, n=2, random_state=0)
attr_null = deep_lift_shap(model, shuffles[:, 0], random_state=0).sum(dim=1)
attr_control = deep_lift_shap(model, shuffles[:, 1], random_state=0).sum(dim=1)

fdr = recursive_seqlets_fdr(attr_sums, attr_control, X_null=attr_null)
threshold = fdr.loc[fdr['fdr'] <= 0.2, 'threshold'].max()   # NaN if none passes
seqlets = recursive_seqlets(attr_sums, threshold=threshold, X_null=attr_null)
```

- `fdr` has one row per threshold (0.1 to 1e-10 by default): seqlets called on
  `X`, on `X_control`, and their ratio scaled by size. Look at the whole curve: it
  often flattens well above zero, because shuffles keep chance motif matches that
  the model responds to, and then no threshold reaches a low target.
- Pass the **same `X_null`** to `recursive_seqlets` as to `recursive_seqlets_fdr`,
  or the threshold means something else. `X_null=None` in both is also valid (the
  null is then pooled over `X`); only the counting control is required.
- With `X_null`, a span whose summed bins exceed anything the control can produce
  gets p = 0.

## tfmodisco_seqlets — longer, discovery-oriented

```python
from tangermeme.seqlet import tfmodisco_seqlets

seqlets = tfmodisco_seqlets(attr_sums, window_size=21, flank=10)
# DataFrame columns: example_idx, start, end, attribution (positive + negative sets)
```

- Replicates TF-MoDISco's caller: seqlets are **longer and less sensitive** by
  design (local context helps motif discovery; flanks get trimmed downstream).
- **Adaptively lowers its threshold to hit a minimum count → it over-calls**; expect
  to filter the results by attribution afterward.
- Boundaries are **left-offset/asymmetric** because each position is replaced by the
  sum of the window to its right (the chosen position is the left edge).

## Which to use

`recursive_seqlets` for sharp, minimal spans and per-seqlet p-values;
`tfmodisco_seqlets` when feeding a motif-discovery pipeline that wants generous
spans. Both flow straight into annotation and plotting.

## Related references

- **Label / count seqlets** → `references/annotate.md` (`annotate_seqlets`,
  `count_annotations`, `pairwise_annotations`).
- **Plot seqlets** → `references/plot.md`. Three things to get right: filter to
  one example, pass `score_key='attribution'`, and move a name column to the
  front — `plot_logo` reads the label from column 0, which in a seqlet frame is
  `example_idx`.
- **Where the attributions come from** → `references/deep_lift_shap.md`
  (projected vs hypothetical).
