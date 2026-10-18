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
- The `p-value` column is a **ranking score, not an error rate**. It is computed as
  if positions were independent draws from a histogram, but attributions are
  correlated along the sequence and the default histogram contains the signal being
  tested, so p-values come out smaller than the real chance of a span in
  signal-free sequence. `threshold=0.01` does not mean 1% of calls are false, and the
  same threshold means different false-call rates in different models and datasets.
  Rank seqlets by it; choose the threshold with a control (below), not by habit.
- `min_seqlet_len` (default 4) is also the smallest span the recursive property must
  hold over. At 1, every single position inside a seqlet must pass the threshold on
  its own, so one weak base splits a motif. `max_seqlet_len` (default 25) raises cost.
- `additional_flanks` pads the reported `start`/`end` **and** the reported
  `attribution` is re-summed over the padded span, so it grows with the flanks. The
  `p-value` is unchanged — it is computed on the un-padded core. Flanks may overlap
  neighbors even though the cores cannot, so flanked spans are not disjoint, their
  attributions can double-count the same positions, and
  `pairwise_annotations_spacing` rejects them (see `references/annotate.md`).
- **Over-calls on a single example** (it can't fit a null from one sequence), and by
  default a sequence's p-values depend on the other sequences passed with it. Pass
  many examples, or a fixed reference set (e.g. attributions of many sequences, or of
  many dinucleotide shuffles) as `X_null` to score every sequence or batch against
  the same null.

### Choosing the threshold with a control

Attribute several dinucleotide shuffles of each sequence with the same model,
target and settings as `X`, concatenate them, and count the seqlets called on them
at each threshold. Leave `X_null` unset: the null is pooled over `X`, and the
shuffles are only counted.

```python
import torch

from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.seqlet import recursive_seqlets
from tangermeme.seqlet import recursive_seqlets_fdr

shuffles = dinucleotide_shuffle(X, n=3, random_state=0)       # (n, 3, 4, length)
attr_control = torch.cat([deep_lift_shap(model, shuffles[:, k], random_state=0).sum(dim=1)
	for k in range(3)])

fdr = recursive_seqlets_fdr(attr_sums, attr_control)
threshold = fdr.loc[fdr['fdr'] <= 0.2, 'threshold'].max()   # NaN if none passes
seqlets = recursive_seqlets(attr_sums, threshold=threshold)
```

- `fdr` has one row per threshold (0.1 to 1e-30 by default): seqlets called on `X`,
  on the control, and their ratio scaled by size. It is a property of the whole set
  of calls at that threshold, not of any single seqlet.
- **More shuffles, steadier estimate.** The estimate is only as precise as the number
  of control seqlets at a threshold; with one shuffle, a strict threshold may rest on
  a few dozen. Each shuffle costs one more attribution pass.
- **Look at the whole curve.** "False" here means no more frequent than in sequences
  with the same dinucleotide composition. Shuffles keep chance motif matches that the
  model responds to, so the curve can level off above zero: a target below that
  level is reached by no threshold, and a target near it picks a threshold from a
  flat stretch that moves a lot between controls.
- With `X_null`, pass the **same `X_null`** to `recursive_seqlets` as to
  `recursive_seqlets_fdr`, or the threshold means something else, and count on a
  control other than `X_null`. A span whose summed bins exceed anything `X_null` can
  produce gets p = 0.

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

`recursive_seqlets` for sharp, minimal spans ranked by p-value;
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
