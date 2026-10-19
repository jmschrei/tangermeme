---
name: tangermeme
description: Use for any task involving the tangermeme library — post-training analysis of genomic sequence-to-function (S2F) deep learning models. Triggers on predictions, DeepLIFT/SHAP attributions, saliency, PISA, dependency maps, marginalization/ablation/spacing of motifs, saturation mutagenesis (ISM), variant effect scoring, sequence design, seqlet calling, motif annotation, loading loci/FASTA/bigWig/MEME/VCF, GC-matched backgrounds, packed sequences, or wrapping a PyTorch genomic model for these analyses. This is a router skill — read the relevant file under references/ for details and footguns before writing tangermeme code.
---

# tangermeme

Post-training analysis of any PyTorch sequence model: batched prediction,
attribution, perturbation experiments, sequence design, on raw outputs with
no distance baked in. This file routes; the reference files hold the
signatures, defaults and silent failures. Read the file for the task before
writing code: several defaults select the wrong output, return a variable
number of objects, or run zero iterations without a word.

## Task to reference file

| task | read |
|---|---|
| attribution via **DeepLIFT/SHAP**: which bases drive this prediction, logos, hypothetical scores for CWMs, references, memory | `references/deep_lift_shap.md` |
| a **convergence delta warning**, a transformer, attention pooling, a functional softmax or product, a custom module, "are these attributions correct" | `references/deep_lift_shap-unhooked-ops.md` and `scripts/audit_deeplift.py` |
| **ISM / saturation mutagenesis**, `fast=True` | `references/saturation_mutagenesis.md` |
| **saliency** (input x gradient, one backward pass) and **dependency maps** (epistasis, motif-flank interactions) | `references/dependency_map.md` |
| **PISA**: attribution of every output position | `references/pisa.md` |
| effect of a motif or region: **marginalize, ablate, spacing**; choosing backgrounds | `references/motif-effects.md` |
| **variant effects** from a VCF: substitution, deletion, insertion | `references/variant-effect.md` |
| **seqlets** from attributions, choosing a threshold with a control | `references/seqlets.md` |
| **motif labels and counts**: TOMTOM, co-occurrence, spacing | `references/annotate.md` |
| **logos** with seqlet or motif annotations | `references/plot.md` |
| a function over a **product of inputs** (sequence x cell state x depth) | `references/product.md` |
| composing predict / attribution through a perturbation (`func=`, `additional_func_kwargs`, `args=`) | `references/func-pattern.md` |
| a model with **several inputs or outputs**, a profile head, a control track; N models in one sweep | `references/model-wrapping.md` |
| **loading** loci, FASTA, bigWig, MEME, VCF; GC-matched background loci; window to genome coordinates | `references/io-loci.md` |
| **editing sequences**: insert, substitute, delete, shuffle, random draws, reverse complement, one-hot, `PackedSequence` | `references/ersatz.md` |
| **designing** sequences to a target output: screen, greedy, beam | `references/design.md` |

The one script, `scripts/audit_deeplift.py` (path relative to this skill's
root), lists the operations in a model's forward pass that DeepLIFT has no
rule for and measures summation-to-delta relative to the output scale; run
it before trusting attributions of an unfamiliar architecture.

## Conventions

- Layout `(batch, channels, length)`, channels the one-hot alphabet, default
  `ACGT`. `X`/`y` observed, `X_bar`/`y_bar` designed or target, `y_hat`
  predictions, `X_attr` attributions.
- `predict(model, X, args=None, func=None, batch_size=32, dtype=None, device=None)`
  upcasts each batch from int8 and returns the model's dtype; a
  multi-output model returns a list. `device=None` means CUDA if available
  on every function that takes it; the model's device and train/eval mode
  are restored afterward. The first GPU call in a process pays 15 to 35 s
  of warm-up.
- Perturbation functions return NamedTuples (`PerturbationResult(y_before, y_after)`,
  `SpaceResult`, `AttributionReferencesResult`); unpack positionally or by
  attribute. `extract_loci` returns one to four objects depending on its arguments.
- `PackedSequence` (`tangermeme.utils`): four-letter sequences at 2 bits a
  base, 16x smaller than int8 one-hot, unknown characters kept as runs.
  Every function that takes one-hot `X` takes one and returns what the
  one-hot gives; sequences come back packed (`ersatz`, references, designs),
  predictions and attributions as tensors. `kmers`, `gapped_kmers`,
  `gc_content`, `chunk` and `unchunk` live in `tangermeme.kmers` and
  `tangermeme.utils`; FIMO and TOMTOM scanning live in `memelite`.
