# Sequence editing (tangermeme.ersatz, tangermeme.utils)

All take `X` `(n, len(alphabet), length)`, int8 or float, with all-zero
columns allowed except in `dinucleotide_shuffle` without `allow_N=True`
and in `local_dinucleotide_shuffle`. Random functions return `(n, k, A, L)`
even at `k=1`.

| call | returns | note |
|---|---|---|
| `insert(X, motif, start=None)` | `(n, A, L + len)` | lengthens; default centers |
| `substitute(X, motif, start=None)` | `(n, A, L)` | overwrites; default start `L // 2 - len // 2` |
| `multisubstitute(X, motifs, spacing, start=None)` | `(n, A, L)` | `spacing[i]` is the bases left between motif `i` and `i + 1` |
| `delete(X, start, end)` | `(n, A, L - (end - start))` | half-open |
| `randomize(X, start, end, probs=[[.25]*4], n=1, random_state=None)` | `(n, k, A, L)` | `probs` 2-D; a Python list is cast to float32 and numpy then rejects `[[0.295, 0.205, 0.205, 0.295]]` as not summing to 1; pass `numpy.array([[...]])` or a float64 tensor |
| `shuffle(X, start=0, end=-1, n=1, random_state=None)` | `(n, k, A, L)` | `random_state` int or `RandomState` |
| `dinucleotide_shuffle(X, start=0, end=-1, n=20, random_state=None, allow_N=False, n_jobs=1, return_packed=False)` | `(n, k, A, L)` int8 | `random_state` int or one int per sequence, never a `RandomState`; keeps the first and last base; raises on a region with one Eulerian path (homopolymer) when `n > 1` |
| `local_dinucleotide_shuffle(X, n=20, bin_size=1024, min_bin_size=512, random_state=None)` | `(n, k, A, L)` | raises when `length <= bin_size` |

Motifs are strings or one-hot tensors. `random_one_hot((n, A, L), probs=, random_state=)`
takes the same `probs` form (a list works there; `randomize` is the one that
casts). `return_packed=True` or a `PackedSequence` in gives a `PackedSequence` out.

- `reverse_complement(X)` is per sequence: a string, one `(A, L)` tensor, or
  a `PackedSequence` with batch axes. A `(n, A, L)` tensor is not rejected
  and returns the wrong shape; use `X.flip(1).flip(2)` for ACGT or loop.
- `one_hot_encode("ACGTN")` is int8 with an all-zero column for `N`;
  uppercase only. `characters(X)` on such a column raises `At least one
  position in the PWM has multiple letters with the same probability`;
  `characters(X, allow_N=True)` round-trips, `force=True` decodes `N` as `A`.
- `PackedSequence.from_one_hot(X)` (not the constructor) packs; `.data` is
  the uint8 `(n, ceil(L/4))` bytes, `.shape` reports the one-hot shape;
  `.to_one_hot(dtype=, device=)` and `.to_codes()` decode; indexing along
  batch axes stays packed; `save`/`load(mmap=True)`.
- `read_meme` PWMs are float64; cast before combining with float32 tensors.
