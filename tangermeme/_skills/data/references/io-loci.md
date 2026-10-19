# Loading data (tangermeme.io, tangermeme.match, tangermeme.utils)

## extract_loci

```python
extract_loci(
	loci,                 # BED/narrowPeak path (track, browser, # lines skipped), DataFrame, or a list of them
	sequences,            # FASTA path (fastest), pyfaidx.Fasta, or {chrom: tensor}
	signals=None,         # list of bigWig paths or figwig.BigWigReader -> y; a bare path raises
	in_signals=None,      # list of bigWigs -> X_in (controls)
	chroms=None, in_window=2114, out_window=1000, max_jitter=0,
	min_counts=None, max_counts=None, target_idx=0,   # inclusive, on signals[target_idx]; need signals
	n_loci=None, summits=False, alphabet=['A','C','G','T'], ignore=['N'],
	exclusion_lists=None, # BED path, DataFrame, or list
	return_mask=False, verbose=False, n_jobs=4, return_packed=False,
)
```

**The return count follows the arguments.** `X` alone comes back as a bare
int8 tensor `(n, 4, in_window)`; otherwise a list in the order `X`, `y`
`(n, n_signals, out_window)` if `signals`, `X_in` if `in_signals`, `mask` if
`return_mask`. Unpack to match what you asked for. Leave `X` int8: every
function except `pisa` upcasts per batch.

- Rows are dropped, so `n` is not `len(loci)`: windows past a chromosome
  end, chromosomes not in `chroms`, counts outside the thresholds, a 100 bp
  chunk shared with an exclusion region, loci past `n_loci`. `return_mask=True`
  gives one bool per locus (after `chroms` filtering) to align results to the
  input. A locus on a chromosome the FASTA lacks raises `ValueError`
  (usually `chr1` vs `1`; pass `chroms=`); so does an empty result.
- A list of loci files is interleaved round-robin, not concatenated.
- Window of size `w`: `mid = start + (end - start) // 2`,
  `[mid - w//2, mid + w//2 + w%2)`. To center on one 0-based base `b`, pass
  `start=b, end=b+1`; it lands at index `w//2`. `summits=True` centers on the
  narrowPeak summit. `max_jitter` widens both windows; it does not jitter.
- Chromosome names are coerced to strings, so `1`-style names work.
- Pass bigWigs and the FASTA as paths: a `pyfaidx.Fasta` is read one window
  at a time (3x slower), a `pybigtools` object 10x slower and deprecated for
  1.9.0; URLs raise. `n_jobs=1` inside a process that already runs in
  parallel. The first call compiles numba kernels (about 2.5 s).
- `return_packed=True` packs each chunk as it is read, so the full one-hot
  is never held; four-letter alphabets only.

## Coordinates back and forth

- `utils.example_to_fasta_coords(example_df, loci_df, window=)` maps
  `(example_idx, start, end)` from seqlets or variants back to genome
  coordinates through the same centering rule.
- `utils.extract_signal(loci, X)` sums a tensor over `(example_idx, start, end)` rows.
- `utils.chunk(X, size=1024, overlap=0)` / `unchunk` split long sequences
  into overlapping blocks and back.
- `io.one_hot_to_fasta(X, filename)` writes sequences, all-zero columns as `N`.

## Matched backgrounds (tangermeme.match)

```python
extract_matching_loci(loci, fasta, in_window=2114, out_window=1000, max_n_perc=0.1,
	gc_bin_width=0.02, bigwig=None, signal_beta=0.5, chroms=None, random_state=None, n_jobs=1)
# -> DataFrame (chrom, start, end), one GC-matched non-overlapping region per input locus
```

`bigwig` plus `signal_beta` keeps only regions whose summed signal is below
that quantile of the loci's. 50 loci on chr22 took 0.4 s. Feed the frame to
`extract_loci` for marginalization backgrounds (`references/motif-effects.md`).

## read_meme, read_vcf

`read_meme(path)` returns `{name: PWM (4, length)}`. `read_vcf(path)`
returns the nine VCF columns as strings with `POS` as int, comments
stripped, nothing split or classified; the variant recipe is in
`references/variant-effect.md`. FIMO and TOMTOM scanning live in `memelite`,
not here.
