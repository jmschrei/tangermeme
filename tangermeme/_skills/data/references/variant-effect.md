# Variant effects (tangermeme.variant_effect)

All three apply `func` (default `predict`) before and after the edit and
return `PerturbationResult(y_before, y_after)`; the distance is yours to
compute. A multi-output model gives lists. Positions are indices into the
window `X`, not genome coordinates; alphabet indices follow `alphabet`
(A 0, C 1, G 2, T 3).

| function | rows | `X` length | note |
|---|---|---|---|
| `substitution_effect(model, X, subs)` | `(example_idx, position, new_char_idx)` | model length | rows on one example are applied together, one score; duplicate `(example, position)` raises `ValueError` |
| `deletion_effect(model, X, dels, left=False)` | `(example_idx, position)` | model length + max deletions per example | every sequence is trimmed to the same post-deletion length; `y_before` is on the trimmed slice, `X[:, :, :model_length]` for `left=False` |
| `insertion_effect(model, X, ins, left=False)` | `(example_idx, position, char_idx)` | model length | result trimmed back; a string `S` is the rows `(k, p, S[-1])`, ..., `(k, p, S[0])`: rows at one position go in reversed, adjacent positions interleave |

Independent scores need one example copy per variant: `X[0:1].repeat(n, 1, 1)`
with row `i` on copy `i`. Slicing `X[k:k+1]` means re-basing `example_idx` to 0.
`deletion_effect` only raises when deletions consume the whole sequence; a
model-length `X` fails in the model, or silently scores a shorter window on a
length-flexible model. The edited sequences are not returned; `func=lambda model, X, args=None, **kw: X`
returns them as `y_after`.

## From a VCF

```python
vcf = read_vcf("variants.vcf.gz")                       # POS is 1-based
snps = vcf[(vcf.REF.str.len() == 1) & (vcf.ALT.str.len() == 1)]
loci = pandas.DataFrame({"chrom": snps.CHROM, "start": snps.POS - 1, "end": snps.POS})
X, mask = extract_loci(loci, "genome.fa", in_window=2114, return_mask=True)
snps = snps[mask.numpy()]
center = 2114 // 2                                       # the variant base lands here
idx = {c: i for i, c in enumerate("ACGT")}
subs = torch.tensor([[i, center, idx[a]] for i, a in enumerate(snps.ALT)])
y_before, y_after = substitution_effect(model, X, subs)
```

Check that `X[i, idx[ref], center] == 1` for each row before trusting the
scores. Indels: the anchor base is `REF[0] == ALT[0]`. A deletion of `D`
bases after the anchor: window `start=POS-1-center, end=start+2114+D`,
rows `(i, center + 1 + j)` for `j in range(D)`, `left=False`, so `y_before`
is the plain centered window. An insertion of `S` after the anchor: rows
`(i, center + 1, idx[c])` for `c in reversed(S)`, all at one position.
Report back with `example_to_fasta_coords` (`references/io-loci.md`).
Scores vary at the 1e-6 level with batch composition on the CPU; that is
float noise, not dependence.
