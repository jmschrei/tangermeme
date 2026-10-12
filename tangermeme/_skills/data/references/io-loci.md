# Loading data with tangermeme.io

`tangermeme.io` turns genomic files into the tensors the rest of the library
consumes. The central footgun is that `extract_loci` returns a **different number
of elements depending on which arguments you set** — unpacking blindly is the #1
mistake.

## extract_loci — the variable return

```python
from tangermeme.io import extract_loci

extract_loci(
    loci,                 # BED/narrowPeak path (track, browser and # lines are
                          # skipped), DataFrame, or list of them
    sequences,            # FASTA path, pyfaidx.Fasta, or {chrom: tensor} dict
    signals=None,         # list of bigWig paths/figwig readers -> OUTPUT signal tensor
    in_signals=None,      # list of bigWig -> INPUT signal tensor (e.g. controls)
    chroms=None,
    in_window=2114,       # input window length (sequence + in_signals)
    out_window=1000,      # output window length (signals)
    max_jitter=0,         # EXPANDS windows for downstream jittering; not applied here
    min_counts=None, max_counts=None, target_idx=0,
    n_loci=None, summits=False,
    alphabet=['A','C','G','T'], ignore=['N'],
    exclusion_lists=None,  # BED path, DataFrame, or list of them
    return_mask=False,
    verbose=False,
    n_jobs=8,             # threads for bigWig paths and FASTA windows; -1 = every CPU
)
```

### Return order (this is the footgun)

The result is a list assembled in this fixed order, and a **bare object** (not a
list) is returned when only one element is present:

1. `X` — one-hot sequences `(n, len(alphabet), in_window)`, dtype **int8** —
   **always present**
2. `y` — output signals `(n, n_signals, out_window)` (single signal still gets a
   signal axis; order = file order) — only if `signals` is given
3. `X_in` — input signals — only if `in_signals` is given
4. `mask` — kept-locus boolean tensor — only if `return_mask=True`

**Leave `X` as int8.** `predict`, `deep_lift_shap` and `saturation_mutagenesis`
upcast each batch to the model's dtype, so the int8 blob is the whole memory win.
Only `.float()` it for `pisa`, which doesn't upcast, or when you call `model(X)`
yourself.

So unpack to match exactly what you requested:

```python
X = extract_loci(loci, fasta)                                  # 1 -> bare tensor
X, y = extract_loci(loci, fasta, signals=bws)                  # 2
X, y, mask = extract_loci(loci, fasta, signals=bws, return_mask=True)  # 3
X, y, X_in, mask = extract_loci(loci, fasta, signals=bws,
                                in_signals=ctrls, return_mask=True)     # 4
```

Get the count wrong and you will silently bind a tensor to the wrong variable.

### Why returned rows may not match input loci

Loci are dropped when they fall off chromosome ends (after jitter), sit on
chromosomes not in `chroms`, fail `min_counts`/`max_counts` (measured on
`signals[target_idx]`, both inclusive, and requiring `signals`), or share a
100bp chunk with an `exclusion_lists` region. **Use `return_mask=True` whenever
you need to align results back to the input rows** — the mask has one entry per
locus left after `chroms` filtering, and loci past an `n_loci` cap are `False`.

Two situations raise `ValueError` rather than dropping loci: a locus on a
chromosome that is not in `sequences` (usually a `chr1` vs `1` naming
mismatch; otherwise pass `chroms=` to keep only the chromosomes the FASTA has),
and no loci remaining after filtering. Exclusion regions on chromosomes absent
from the FASTA are ignored, so a genome-wide blacklist works with a partial
FASTA.

### Pass bigWigs as paths or figwig readers

bigWigs given as local paths, or opened with `figwig.BigWigReader`, are read
by figwig, all kept loci in one call on `n_jobs` threads; a dict is read one
locus at a time. Only local files are read: a URL raises a `ValueError`, so
download remote bigWigs first. A file figwig does not read (a bigBed, a bigWig
with overlapping intervals or unsorted blocks) raises figwig's `ValueError`.
A bigWig opened with `pybigtools.open` is deprecated, with a `FutureWarning`,
and will not be accepted from tangermeme 1.9.0. Until then it is read one locus
at a time: on 167,750 loci with hg38 and one bigWig, the call takes 3.7 s with a
pybigtools object and 0.35 s with the path, at `n_jobs=8`.

### Threads and readers

`n_jobs` (default 8) is the most threads `extract_loci` uses: figwig reads the
bigWig paths on them, and the windows of a FASTA path are read and one-hot
encoded on them. The results are identical for every value. Pass `n_jobs=1`
inside a DataLoader worker or any other process that already runs in
parallel, so the threads do not multiply.

Pass the FASTA as a path. Its windows are then read from a memory map of the
file through its `.fai`, where those of a `pyfaidx.Fasta` object are read one
at a time: on 167,750 loci of hg38 with one bigWig, at `n_jobs=8`, the call
takes 0.35 s with the path and 1.2 s with a `pyfaidx.Fasta`. A compressed
FASTA, or one with bytes pyfaidx would change, is read through pyfaidx with
the same result. The first call in
a new environment compiles numba kernels, about 2.5 s, once.

### Multiple loci files are interleaved, not concatenated

Passing a list of BED/narrowPeak files **interleaves** them round-robin until the
shortest is exhausted, then appends the remainder — it does not concatenate them in
order. `n_loci` then truncates that interleaved list. Useful for mixing a
locus-of-interest with genomic background; surprising if you expected file-order
concatenation.

### Chromosome names are always strings

Chromosome names from BED files and DataFrames are coerced to strings, so genomes
that name their chromosomes `1`, `2`, ... (Ensembl-style) match the FASTA/bigWig
names instead of being read in as integers by pandas. `chroms=[1, 2]` and
`chroms=['1', '2']` are equivalent.

### Window semantics

`in_window` (sequence + `in_signals`) and `out_window` (`signals`) are centered on
each locus, independent of the locus's own width: a window of size `w` is
`[mid - w//2, mid + w//2 + w%2)` with `mid = start + (end - start)//2`, so an odd
window has its extra base on the right. `out_window` has no effect when
`signals` is None. `summits=True` centers on the narrowPeak summit instead of
the region midpoint. `max_jitter` *expands* both windows so a downstream data
generator can jitter cheaply — it does not jitter the returned data itself.

## read_meme — motif PWMs

```python
from tangermeme.io import read_meme
motifs = read_meme("motifs.meme")        # dict: name -> PWM tensor (4, length)
```

Wraps the memelite reader. Note: FIMO/Tomtom scanning moved out of tangermeme to
`memesuite-lite` — use that package for motif scanning.

## read_vcf — variants

```python
from tangermeme.io import read_vcf
vcf = read_vcf("variants.vcf")           # pandas DataFrame, comments stripped
```

Feed variants into `tangermeme.variant_effect.*` to score substitution / deletion
/ insertion effects.

## Related references

`references/notebook-walkthrough.md` (loading is step 3 of the
end-to-end flow), `references/model-wrapping.md` (the
`(batch, channels, length)` layout the loaded `X` follows),
`references/motif-effects.md` (consuming the loaded sequences).
