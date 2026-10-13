# tangermeme

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/tangermeme?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=ORANGE&left_text=downloads)](https://pepy.tech/projects/tangermeme) [![Unit Tests](https://github.com/jmschrei/tangermeme/actions/workflows/python-package.yml/badge.svg)](https://github.com/jmschrei/tangermeme/actions/workflows/python-package.yml) [![Documentation Status](https://readthedocs.org/projects/tangermeme/badge/?version=latest)](https://tangermeme.readthedocs.io/en/latest/?badge=latest)

[[paper](https://doi.org/10.1038/s41592-026-03254-z)][[preprint](https://www.biorxiv.org/content/10.1101/2025.08.08.669296v2)][[docs](https://tangermeme.readthedocs.io/en/latest/index.html)][[tutorials](https://github.com/jmschrei/tangermeme/tree/main/docs/tutorials)][[vignettes](https://github.com/jmschrei/tangermeme/tree/main/docs/vignettes)][[release notes](https://tangermeme.readthedocs.io/en/latest/whats_new.html)]

<a href="https://claude.ai/artifact/Mq1LZZJEUn8t3W35vFu7fv">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmschrei/tangermeme/main/docs/_static/codemap-dark.png">
<img src="https://raw.githubusercontent.com/jmschrei/tangermeme/main/docs/_static/codemap-light.png" alt="The tangermeme workflow map: modules arranged in six stages from loading data to design, with the links of the marginalize module drawn" width="100%">
</picture>
</a>

Every public function in tangermeme, arranged by where it sits in an analysis. Click the map to open the [interactive version](https://claude.ai/artifact/Mq1LZZJEUn8t3W35vFu7fv), where selecting a module draws its calls, data flow and `func=` links and lists its functions with links to the API reference and tutorials.

> [!NOTE]
> tangermeme is described in [Nature Methods](https://doi.org/10.1038/s41592-026-03254-z). If you use tangermeme in your work, please consider citing the paper (BibTeX [below](#citation)). Citations allow me to continue developing software like this for the community.

Training sequence-based (sometimes called sequence-to-function or S2F) machine learning models has become widespread when studying genomics. But what have these models learned, and what do we even do with them after training? tangermeme provides robust and easy-to-use tools for the what-to-do-after-training question. It implements everything except the model: loading sequences and signal, atomic sequence operations such as inserting a motif or shuffling one out, efficient application of a model to the edited sequences, attribution methods for dissecting what the model has learned, seqlet calling and annotation for turning those attributions into statements about motifs, and methods for designing new sequences. If the MEME suite is the toolkit for analyzing biological sequences on their own, tangermeme is the toolkit for analyzing them alongside a predictive model.

tangermeme aims to be assumption-free. Models can be any PyTorch model, including multi-input and multi-output ones, as long as `y = model(X)` works. Functions do not assume a distance and instead return the raw predictions, and when a loss function is needed it can be supplied by the user. Sequence operations and model operations are kept separate, so any sequence operation can be paired with any model operation: a marginalization can return DeepLIFT/SHAP attributions or saturation mutagenesis scores instead of predictions by passing `func=deep_lift_shap` or `func=saturation_mutagenesis`. Although the documentation describes best practices, our hope is that being assumption-free makes adapting tangermeme to your setting as frictionless as possible. Every function is unit-tested and written with compute- and memory-efficiency in mind, and, although the library was built with DNA in mind, every function works with any alphabet.

Please see the documentation and tutorials linked at the top of this README for more extensive documentation. If you only read one vignette, read THIS ONE: [Inspecting what Cis-Regulatory Features a Model has Learned](https://tangermeme.readthedocs.io/en/latest/vignettes/Inspecting_What_Cis-Regulatory_Features_a_Model_Has_Learned.html).

<br>
<img src="https://github.com/user-attachments/assets/20b186e7-73af-46c7-a7b6-5484c714036e" width=60%>


## Installation

```bash
pip install tangermeme
```

Or, with [uv](https://docs.astral.sh/uv/):

```bash
uv add tangermeme
```

tangermeme requires Python 3.10 or later. If PyTorch is already installed, installation should take less than five minutes. If PyTorch needs to be installed, this command should do so but may take up to ten minutes, most of it spent resolving dependencies. `uv` is significantly faster than `pip` at this resolution step.

Optional extras:

- `tangermeme[interactive]` — install `mpld3` for `tangermeme.plot.interactive_logo`, an interactive sequence logo with hover tooltips over each annotation.
- `tangermeme[docs]` — install the Sphinx toolchain (sphinx, sphinx-rtd-theme, nbsphinx, jupyter, nbconvert) for building the documentation site locally.

### Development install

Contributors should clone the repository and use `uv` to create a reproducible environment from `uv.lock`:

```bash
git clone https://github.com/jmschrei/tangermeme.git
cd tangermeme
uv sync --extra dev
uv run pytest
```

This installs the package in editable mode along with the testing toolchain (`pytest`, `captum`, `ruff`, `build`, `twine`). `uv run pytest -m gpu` runs the CUDA tests, which are skipped automatically on a machine without a GPU.

## Claude Code Skill

tangermeme ships an [Agent Skill](https://docs.claude.com/en/docs/claude-code/skills) for [Claude Code](https://claude.com/claude-code) that teaches the agent how to use the library correctly — the API contracts, the footguns (e.g., `deep_lift_shap` needs `target=` for multi-task models, `extract_loci` has a variable-length return), and the multi-step workflows. It is bundled with the package but, because Claude Code does not scan installed Python packages, you install it once into your personal skills directory:

```bash
tangermeme-install-skills
```

This copies the skill to `~/.claude/skills/tangermeme/`, where it is available to Claude Code in **every** project. Re-run with `--force` after upgrading tangermeme to refresh it. Once installed, just ask Claude Code to do tangermeme tasks ("compute DeepLIFT/SHAP attributions on these peaks", "marginalize this motif", "set up a notebook to inspect what my model learned") and the skill is consulted automatically. The skill is a router (`SKILL.md`) that points to detailed reference files loaded on demand, so it adds essentially no context cost until it is used.

If you would rather not copy files into your home directory, point Claude Code at the bundled copy in place instead:

```bash
export CLAUDE_SKILLS_PATH="$(tangermeme-install-skills --print-path)"
```

## Speed

tangermeme is written so that the bookkeeping around a model costs as little as possible. The tables below compare it with other packages on one machine: one thread of an AMD EPYC 9575F for one-hot encoding, and one NVIDIA H200 for saturation mutagenesis and DeepLIFT/SHAP, with tangermeme 1.5.1, Python 3.11 and PyTorch 2.14.1. Each time is the median of repeated calls after a warm-up call. Before timing, each one-hot encoding was checked against a reference encoding, each set of saturation mutagenesis predictions against tangermeme's with `fast=False`, and both DeepLIFT/SHAP implementations for attributions that sum to the difference in predictions.

#### One-hot encoding

Time to one-hot encode chromosome 1 of hg38 (248,956,422 bp, upper-cased) on one thread.

| Package | Function | Output dtype | Time (s) | Slower than tangermeme |
|---|---|---|---:|---:|
| **tangermeme** 1.5.1 | `one_hot_encode(X)` | int8 | **0.13** | — |
| SeqPro 0.23.0 | `NucleotideAlphabet.ohe` | uint8 | 0.60 | 4.7x |
| ChromBPNet | `dna_to_one_hot` | int8 | 7.29 | 57x |
| **tangermeme** 1.5.1 | `one_hot_encode(X, dtype=torch.float32)` | float32 | **0.79** | — |
| CREsted 1.10.0 | `one_hot_encode_sequence` | float32 | 1.78 | 2.2x |
| gReLU 1.1.0 | `strings_to_one_hot` | float32 | 12.9 | 16x |
| Selene SDK 0.6.0 | `sequence_to_encoding` | float32 | 14.9 | 19x |

#### Saturation mutagenesis

Time to predict the effect of every single-base substitution across 2,114 bp with the count head of two models: a BPNet model of GATA2 ChIP-seq in K562 (64 filters) and the ChromBPNet model of K562 ATAC-seq (ENCSR868FGK, 512 filters). tangermeme is timed with its default, `fast=True`, which computes the predictions with fastISM, and with `fast=False`, which runs every edited sequence through the model. The jobs are large enough for `fast=True` to take the fast path, which it skips for a job it estimates would take under thirty seconds. Captum has no saturation mutagenesis routine. Its `Occlusion`, with a window of one position and a baseline of one character at every position, returns the effect of substituting that character at each position, so four calls give every substitution; this also predicts the identity substitution at each position, which tangermeme skips, so Captum makes four forward passes per position where tangermeme with `fast=False` makes three. The loop is the batched implementation a user would write by hand: build every mutant in Python, skipping the identity substitutions, and run them through `predict`. Each method was timed at its fastest batch size of 256 and 1,024. The predictions of every method agree with those of `fast=False` to within 2.0e-3 for BPNet, whose predicted log counts range from -0.4 to 2.3, and to within 3.3e-4 for ChromBPNet, whose predictions range from 6.3 to 8.7.

| Model | Sequences | tangermeme (s) | tangermeme, `fast=False` (s) | Captum `Occlusion` (s) | Loop + `predict` (s) |
|---|---:|---:|---:|---:|---:|
| BPNet (GATA2) | 320 | **19.6** | 77.9 (4.0x) | 104.5 (5.3x) | 105.3 (5.4x) |
| ChromBPNet | 32 | **14.2** | 54.3 (3.8x) | 74.5 (5.3x) | 57.6 (4.1x) |

#### DeepLIFT/SHAP

DeepLIFT/SHAP in tangermeme runs at the same speed as Captum's `DeepLiftShap`. With 20 dinucleotide shuffles per sequence, the same references given to both, and 20 example-reference pairs per forward and backward pass, attributing 100 sequences with ChromBPNet took 3.5 s with tangermeme and 3.4 s with Captum, and attributing 1,000 sequences with the BPNet model took 8.6 s and 8.3 s.

Where the two differ is in what they return. DeepLIFT/SHAP needs a rule for each non-linear operation, and an operation without one is treated as linear, which breaks the property that the attributions sum to the difference in prediction between a sequence and its reference. The profile head of ChromBPNet ends in a softmax-weighted sum of the profile logits. tangermeme is told about it with `additional_nonlinear_ops`. Captum has no argument for this, since its rules live in a module-level dictionary, and run as is it treats the operation as linear without a warning. Below are the convergence deltas, the gap between the summed attributions and the difference in predictions, over 400 example-reference pairs with cuDNN's TF32 convolutions turned off.

| ChromBPNet head | tangermeme median (max) | Captum median (max) |
|---|---:|---:|
| Counts | 3.9e-7 (1.9e-6) | 3.9e-7 (1.9e-6) |
| Profile | 5.8e-7 (1.5e-5) | 0.18 (0.78) |

## Usage

tangermeme aims to be as low-level and simple as possible. Models can be any PyTorch model or any wrapper of a PyTorch model as long as the forward function is still exposed, i.e., `y = model(X)` still works. If you have a model that takes multiple inputs or returns multiple outputs and you want to simplify it for the purpose of ISM or sequence design, you can wrap it however you would like and still use these functions. All data are PyTorch tensors, one-hot encoded sequences have shape `(batch, len(alphabet), length)`, and broadcasting is supported wherever possible. Being this flexible sometimes results in bugs, so please report any anomalies when doing fancy broadcasting or model wrapping.

The examples below form one walkthrough: run this block first to define a small convolutional model, which takes one-hot sequences of shape `(batch, 4, length)` and returns two outputs per sequence, and a batch of 100 random sequences of 2,000 bp. Substitute your own model and sequences.

```python
import torch

from tangermeme.utils import random_one_hot


class Model(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.conv = torch.nn.Conv1d(4, 16, kernel_size=21, padding=10)
		self.relu = torch.nn.ReLU()
		self.linear = torch.nn.Linear(16, 2)

	def forward(self, X):
		h = self.relu(self.conv(X))
		return self.linear(h.mean(dim=-1))


torch.manual_seed(0)
model = Model()
X = random_one_hot((100, 4, 2000), random_state=0).float()
```

#### Ersatz

tangermeme implements atomic sequence operations to help you ask "what if?" questions of your data. These operations can be found in `tangermeme.ersatz`. For example, if you want to insert a subsequence or motif into the middle of a sequence you can use the `insert` function.

```python
from tangermeme.ersatz import insert
from tangermeme.utils import one_hot_encode   # Convert a sequence into a one-hot encoding
from tangermeme.utils import characters   # Convert a one-hot encoding back into a string

seq = one_hot_encode("AAAAAA").unsqueeze(0)
merge = insert(seq, "GCGC")[0]

print(characters(merge))
# AAAGCGCAAA
```

Sometimes, when people say "insert" what they really mean is "substitute", where a block of characters are changed without changing the length of the sequence. Most functions in tangermeme that involve adding a motif to a sequence use substitutions instead of insertions.

```python
from tangermeme.ersatz import substitute

seq = one_hot_encode("AAAAAA").unsqueeze(0)
merge = substitute(seq, "GCGC")[0]

print(characters(merge))
# AGCGCA
```

If you want to dinucleotide shuffle a sequence, you can use the `dinucleotide_shuffle` command.

```python
from tangermeme.ersatz import dinucleotide_shuffle

seq = one_hot_encode('CATCGACAGACTACGCTAC').unsqueeze(0)
shuf = dinucleotide_shuffle(seq, random_state=0)

print(characters(shuf[0, 0]))
# CAGATACACGACGCTCTAC
print(characters(shuf[0, 1]))
# CGATAGACACGCTCACTAC
```

Both shuffling and dinucleotide shuffling can be applied to an entire sequence, but they can also be applied to *portions* of the sequence by supplying `start` and `end` parameters if you want to, for instance, eliminate a motif by shuffling the nucleotides.

```python
seq = one_hot_encode('CATCGACAGACGCATACTCAGACTTACGCTAC').unsqueeze(0)
shuf = dinucleotide_shuffle(seq, start=5, end=25, random_state=0)

print(characters(shuf[0, 0]))
# CATCG AGCGACTCAGATACACACTT ACGCTAC    Spacing added to emphasize that the flanks are identical
print(characters(shuf[0, 1]))
# CATCG ACGAGCATCACACTAGACTT ACGCTAC
```

A dinucleotide shuffle preserves the composition of the sequence as a whole but averages away how that composition varies along it. `local_dinucleotide_shuffle` shuffles within bins instead, so a background built from a window whose GC content changes along its length keeps that structure. The bin boundaries are redrawn for every shuffle.

```python
from tangermeme.ersatz import local_dinucleotide_shuffle

shuf = local_dinucleotide_shuffle(X[:2], n=5, bin_size=200, min_bin_size=100, random_state=0)

print(shuf.shape)
# torch.Size([2, 5, 4, 2000])
```

#### Loading Data

`tangermeme.io` reads the formats you need to get sequences and signal into a model. `extract_loci` takes a BED file of loci (here, peaks), a FASTA file of the genome, and optionally bigWig files of signal, and returns one-hot encoded sequences centered on each locus along with the signal over a window of each. It reads the FASTA through a memory map and the bigWigs with [figwig](https://github.com/jmschrei/figwig), both on `n_jobs` threads.

```python
from tangermeme.io import extract_loci

X_peaks, y_peaks = extract_loci("peaks.bed.gz", "hg38.fa", signals=["signal.bw"], in_window=2114, out_window=1000)

print(X_peaks.shape, y_peaks.shape)
# torch.Size([4872, 4, 2114]) torch.Size([4872, 1, 1000])
```

`read_meme` reads a MEME-format motif file, `read_vcf` reads a VCF of variants, and `one_hot_to_fasta` writes one-hot encoded sequences back out as a FASTA file.

#### Prediction

Once you have your sequence of interest, you usually want to apply a predictive model to it. `tangermeme.predict` implements `predict`, which will handle constructing batches from a big blob of data, moving these batches to the GPU (or whatever device you specify) one at a time and moving the results back to the CPU, and making sure the inference is done without gradients and in evaluation mode. The model's device and training mode are restored afterwards. Also, there can be a cool progress bar.

```python
from tangermeme.predict import predict

y = predict(model, X, batch_size=64)

print(y.shape)
# torch.Size([100, 2])
```

#### DeepLIFT/SHAP Attributions

A powerful form of analysis is to run your predictive model backwards to highlight the input characters driving predictions. If the model predictions are accurate one can interpret these highlights -- or attributions -- as the actual driver of experimental signal. One of these attribution methods is called DeepLIFT/SHAP (merging ideas from DeepLIFT and DeepSHAP). tangermeme has a built-in implementation that is simpler, more robust, and corrects a few issues with other implementations of DeepLIFT/SHAP.

```python
from tangermeme.deep_lift_shap import deep_lift_shap

X_attr = deep_lift_shap(model, X, target=0, random_state=0)

print(X_attr.shape)
# torch.Size([100, 4, 2000])
```

Note that for multi-task models a target must be set to calculate attributions for one output at a time.

![image](https://github.com/jmschrei/tangermeme/assets/3916816/db628856-182a-427a-be95-2dc33def11b0)

DeepLIFT/SHAP needs a rule for every non-linear operation in the model, and implementations such as Captum's silently treat an operation they do not have a rule for as linear (see [Speed](#speed)). tangermeme checks the convergence deltas of every example-reference pair and warns when attributions do not sum to the difference in predictions, which is how such a missing rule shows up. It has rules for sixteen element-wise activations along with GLU, max pooling, softmax, LayerNorm and RMSNorm, `BilinearOp` for writing the matrix products of attention as hookable modules, `integrated_gradients_op` for any layer without a closed-form rule, and `additional_nonlinear_ops` for registering your own. See [Tutorial A3](https://tangermeme.readthedocs.io/en/latest/tutorials/Tutorial_A3_Deep_LIFT_SHAP.html) and the [vignette on attribution trickiness](https://tangermeme.readthedocs.io/en/latest/vignettes/Attribution_Trickiness_and_DeepLiftShap_Implementations.html) for details.

#### PISA

DeepLIFT/SHAP explains one output at a time. Pairwise Influence by Sequence Attribution (PISA) runs it for every output, giving the influence of each input position on each output, which for a profile model shows which parts of the predicted profile each motif is responsible for. Here, the attributions for both outputs of the model are returned.

```python
from tangermeme.pisa import pisa

X_pisa = pisa(model, X[:5], random_state=0)

print(X_pisa.shape)
# torch.Size([5, 2, 4, 2000])
```

#### Saliency

DeepLIFT/SHAP and saturation mutagenesis both cost far more than a single forward pass. When you have more sequences than either can afford, or your model uses an op that has no DeepLIFT rule registered, `saliency` gives you input times gradient attributions for one backward pass per batch. It returns the same shape and sign convention as the other two, so it plots and composes identically. By default the mean gradient across the alphabet at each position is subtracted first, following Majdandzic et al. (Genome Biology 2023); pass `gradient_correction=False` for the uncorrected values.

```python
from tangermeme.saliency import saliency

X_attr = saliency(model, X, target=267)
```

The trade-off is the linearity assumption: gradients saturate, so a nucleotide the model is already confident about can carry a near-zero value despite being essential. Prefer DeepLIFT/SHAP when you can afford it, and use saliency to triage which regions deserve the more careful method.


#### Dependency Maps

Attributions say which bases matter; a dependency map, adapted from [Tomaz da Silva et al. (Nature Genetics 2025)](https://doi.org/10.1038/s41588-025-02347-3), says which bases decide whether the other bases matter. It substitutes every base at every position and records how far the attributions *elsewhere* in the sequence move, so a model with no interactions gives a purely diagonal map and any off-diagonal structure is evidence of epistasis the model learned -- cooperating motifs appear as blocks, and the flanks a motif depends on appear as stripes.

```python
from tangermeme.dependency_map import dependency_map

dmap = dependency_map(model, X, start=975, end=1030, target=267)
```

`func` chooses the attribution method and defaults to `saliency`; `deep_lift_shap` and `saturation_mutagenesis` can be passed in when their accuracy is worth the cost. Because this runs one attribution per substitution, use `start` and `end` to restrict the substituted region. Note that the diagonal is non-zero even for a linear model and dominates any color scale, so mask it before plotting.


#### Marginalization

Given a predictive model and a set of known motifs, a common question is to ask what motifs affect the model's predictions. Rather than trying to scan these motifs against the genome and averaging predictions at all sites -- which is challenging and computationally costly -- you can simply substitute in the motif of interest into a background set of sequences and see what the difference in predictions is. Because `tangermeme` aims to be assumption-free, these functions take in a batch of examples that you specify, and return the predictions before and after adding the motif in for each example. If the model returns more than one tensor, `y_before` and `y_after` will each be a list of tensors. If the model is multi-input, additional inputs can be specified as a tuple passed into `args`.

```python
from tangermeme.marginalize import marginalize

y_before, y_after = marginalize(model, X, "CTCAGTGATG")

print(y_before.shape, y_after.shape)
# torch.Size([100, 2]) torch.Size([100, 2])
```

By default, these functions use the nucleotide alphabet, but you can pass in any alphabet if you're using these models in other settings.

Below, we can see how a BPNet model trained to predict GATA2 binding responds to marginalizing over a GATA motif.

<img src="https://github.com/jmschrei/tangermeme/assets/3916816/66f776e1-b49b-4b31-9e1f-88bce0096400" width="600">

Importantly, all methods that modify the sequence can take in an optional `func` parameter to change the function that gets applied before and after performing the sequence modification (in this case, substituting in a motif). By default, this function is just the `predict` function but it can just as easily be the `deep_lift_shap` method to give you attributions before and after.

```python
attr_before, attr_after = marginalize(model, X, "CTCAGTGATG", func=deep_lift_shap)
```

![image](https://github.com/jmschrei/tangermeme/assets/3916816/9d453d4c-aba8-449b-9911-c6e17ef4ab77)

`ablate`, `marginalize`, `space`, `tangermeme.product.apply_pairwise`, `tangermeme.product.apply_product`, and the three variant effect functions all take `func`, and `predict`, `deep_lift_shap`, and `saturation_mutagenesis` can each be passed in directly. Use `additional_func_kwargs={...}` to pass arguments through to `func`.

#### Ablation

The conceptual opposite of marginalization is ablation; rather than adding information to a sequence in the form of a motif, you remove information from it by shuffling out a portion of the sequence. This function will handle shuffling (or dinucleotide shuffling) the given number of times and returns the predictions for each shuffle.

```python
from tangermeme.ablate import ablate

y_before, y_after = ablate(model, X, 995, 1005, random_state=0)

print(y_before.shape, y_after.shape)
# torch.Size([100, 2]) torch.Size([100, 20, 2])
```

As you might expect, if we shuffle a sequence that has the GATA motif in the middle, we see the opposite as the marginalization experiment.

<img src="https://github.com/jmschrei/tangermeme/assets/3916816/871a88bd-3e2c-4538-b928-1ca8921cfe56" width="600">

#### Spacing

Motifs do not occur in isolation in the genome and so, frequently, we want to measure how these motifs interact with each other and how this changes across different spacings. This function takes in a set of motifs and a tensor of spacings, with one row per combination of spacings to try and one column per adjacent pair of motifs, and returns the predictions before and after inserting the full set of motifs at each combination.

```python
from tangermeme.space import space

spacing = torch.arange(0, 50, 10)[:, None]   # five spacings between the two motifs
y_before, y_afters = space(model, X, ["CTCAGTGATG", "CTCAGTGATG"], spacing=spacing)

print(y_before.shape, y_afters.shape)
# torch.Size([100, 2]) torch.Size([100, 5, 2])
```

By running this function across several spacings one can, for instance, measure the cooperative effect that nearby AP-1 motifs have on predictions and observe that this cooperativity fades as the motifs get further apart.

<img src="https://github.com/jmschrei/tangermeme/assets/3916816/60bb5465-8c1c-4f32-b0a2-c2a9f9bb9889" width="600">

#### Saturation Mutagenesis

Given an observed sequence a simple question is "what positions are driving model predictions?" One simple way to answer this question is through saturation mutagenesis, i.e., compare the predictions on the original sequence with that of sequences that comprehensively each contain one mutation with respect to that original sequence. This is another form of attribution method that is conceptually similar to deep mutational scanning but using a predictive model instead of running an experiment.

```python
from tangermeme.saturation_mutagenesis import saturation_mutagenesis

X_ism = saturation_mutagenesis(model, X[:5], target=0)

print(X_ism.shape)
# torch.Size([5, 4, 2000])
```

For a job that would take more than thirty seconds, `saturation_mutagenesis` computes these predictions with fastISM, which recomputes only the part of each layer that an edit can change, after checking it against the direct computation on one batch of edits. Pass `fast=False` to compute every edited sequence in full.

By default, the change in prediction for each substitution is mean-subtracted across the alphabet at each position and multiplied by the one-hot sequence, giving the influence of each *observed* nucleotide on the prediction, with the same shape as `X`. Pass `raw_outputs=True` to get the predictions for the original sequences and for every substitution, and combine them however you would like. See the tutorial for more details.

In a region with an AP-1 motif, we can run ISM on Beluga and look at AP-1 factor tasks to identify that the AP-1 motif is what is driving the predictions.

<img src="https://github.com/jmschrei/tangermeme/assets/3916816/34fe309b-bf67-465f-911f-9d83c011ad61" width="800">

#### Variant Effect

A common use case of these predictive models is evaluating the effect that mutations in sequence have on predictions. In practice, there are several situations where calculating variant effect is helpful but potentially the most clinically relevant one is having a list of variants that have been implicated by some sort of association study and wanting to screen them for some likelihood of being truly causal. In this case, given a model that predicts a readout of interest, e.g. protein binding and chromatin accessibility, one can see how the predictions change before and after variants are incorporated into the sequence. Presumably, the mutations that are actual drivers of phenotype will cause changes in predicted readout, and those that are passengers will not have as strong an effect.

The simplest form of variant is the substitution, where one or more characters in a sequence are changed to another character. Each row of `substitutions` is `(example index, position, new character index)`, and more than one substitution can be encoded in each sequence.

```python
from tangermeme.variant_effect import substitution_effect

substitutions = torch.tensor([
	[0, 1058, 0],
	[1, 1000, 3],
])

y, y_var = substitution_effect(model, X, substitutions)

print(y.shape, y_var.shape)
# torch.Size([100, 2]) torch.Size([100, 2])
```

When using a BPNet model that predicts GATA2 binding, we can see that a single substitution encoded at this position in the example seems to knock out predicted signal in the middle of the window entirely. Quite a strong effect.

![image](https://github.com/jmschrei/tangermeme/assets/3916816/9f241e09-3510-4c59-8694-4c6b5b77a16b)

tangermeme can also handle deletions and insertions. These situations are a bit more complicated because the sequences before and after incorporating variants are different lengths and so trimming needs to be done. See the tutorials for more explanation for how to control this trimming.

```python
from tangermeme.variant_effect import deletion_effect
from tangermeme.variant_effect import insertion_effect

deletions = torch.tensor([[0, 1000]])   # (example index, position)
insertions = torch.tensor([[0, 1000, 2]])   # (example index, position, character index)

y, y_var = deletion_effect(model, X, deletions)
y, y_var = insertion_effect(model, X, insertions)
```

#### Design

Given a trained predictive model, one can try to design a sequence that causes the model to give desired predictions. `greedy_substitution` tries every given motif at every position each iteration, and takes the motif + position combination that yields predictions closest to the desired output from the model. `output_mask` restricts the loss to the outputs you care about, ignoring the effect on all others.

```python
from tangermeme.design import greedy_substitution

X0 = random_one_hot((1, 4, 2000), random_state=1).float()
y = torch.tensor([[5.0, 0.0]])
motifs = ["GATAAG", "TGACTCA", "CACGTG"]

X_hat = greedy_substitution(model, X0, y, motifs, output_mask=torch.tensor([True, False]), max_iter=3)

print(X_hat.shape)
# torch.Size([1, 4, 2000])
```

When the model is the Beluga model and the goal is to design a sequence that yields strong AP-1 binding but ignore the effect on all other tasks, this function inserts three AP-1 binding sites close together. The predictions from the model are much higher for the AP-1 tasks on the designed sequence than the original sequence.

<img src="https://github.com/jmschrei/tangermeme/assets/3916816/519d0648-af02-4e75-bef2-958b5a6629a4" width="600">

`beam_substitution` generalizes the greedy search by keeping the `beam_size` best sequences each round rather than one, which recovers combinations of edits that a greedy search prunes after a locally suboptimal first edit; `beam_size=1` reproduces `greedy_substitution`. `greedy_marginalize` greedily builds a construct of motifs, orientations and spacings rather than editing one sequence, scoring each candidate by marginalizing it over the sequences in `X`, and `screen` draws random sequences and keeps the best. For gradient-based design of edits, see [ledidi](https://github.com/jmschrei/ledidi).

#### Seqlet Calling

In contrast to motif scanning, which usually relies solely on nucleotide sequence to determine whether a motif matches, seqlet calling is the identification of spans of nucleotides that have high attribution score. Seqlet calling methods usually do not rely on sequence at all because they are the first step in identifying repeating patterns based on having high attributions.

```python
from tangermeme.seqlet import recursive_seqlets

seqlets = recursive_seqlets(X_attr.sum(dim=1))   # You pass in a 2D tensor, not a 3D one

print(seqlets.head())
#    example_idx  start   end  attribution   p-value
# 0           25   1062  1067     0.000261  0.000026
# 1           53    828   832     0.000210  0.000142
# 2           12    594   601     0.000286  0.000174
# 3           28   1239  1244     0.000245  0.000200
# 4           24    966   973     0.000337  0.000207
```

![image](https://github.com/user-attachments/assets/152f63ab-ef58-42df-902a-1c1be2641868)

The TF-MoDISco seqlet calling algorithm is also implemented as `tfmodisco_seqlets`. See the seqlet tutorial or API documentation for more details.

Because seqlets are called entirely based on attributions, it is sometimes unclear whether the sequence content is similar to any known motif. You can use `annotate_seqlets` to match seqlets to a motif database, such as one downloaded from [JASPAR](https://jaspar.elixir.no/downloads/), using Tomtom from [memesuite-lite](https://github.com/jmschrei/memesuite-lite). By default this will give you the nearest motif match for each seqlet but can give you any number of matches you want.

```python
from tangermeme.annotate import annotate_seqlets
from tangermeme.io import read_meme

motifs = read_meme("motifs.meme")
motif_idxs, motif_pvalues = annotate_seqlets(X, seqlets, motifs)

print(motif_idxs.shape, motif_pvalues.shape)
# torch.Size([1526, 1]) torch.Size([1526, 1])
```

![image](https://github.com/user-attachments/assets/5c4ed1ab-1c1a-4d33-96e8-b25613970a22)

#### Annotations

In `tangermeme`, an annotation is any genomic span. This means it can be motif hits, seqlets, hit calls, etc., as long as it has a defined start and end coordinate. Given a set of annotations you can do many things. First, you can count the number of times each annotation appears in each sequence.

```python
from tangermeme.annotate import count_annotations

y = count_annotations((seqlets['example_idx'], motif_idxs[:, 0]))

print(y.shape)
# torch.Size([100, 1007])
```

`pairwise_annotations` counts how often each pair of annotations occurs in the same sequence, and `pairwise_annotations_spacing` counts how often each pair occurs at each number of nucleotides apart.

For each annotation, you can get results before and after ablating the annotation from the sequence. Here is an example of running DeepLIFT/SHAP on an example after shuffling each motif instance independently, with only one line of code, for the first ten seqlets.

```python
from tangermeme.ablate import ablate_annotations

annotations = torch.tensor(seqlets[['example_idx', 'start', 'end']].values[:10])
y_before, y_after = ablate_annotations(model, X, annotations, func=deep_lift_shap)

print(y_before.shape, y_after.shape)
# torch.Size([10, 1, 4, 2000]) torch.Size([10, 1, 20, 4, 2000])
```

![image](https://github.com/user-attachments/assets/997ca242-8c27-46b6-8204-b898e5f20166)

#### Plotting

`tangermeme.plot` draws sequence logos with matplotlib alone. `plot_logo` plots attributions over a window, with annotations such as seqlets drawn beneath them, and `plot_pwm` plots a motif. `interactive_logo`, from the `interactive` extra, draws the same logo with a tooltip over each annotation.

## Related Packages

- [memesuite-lite](https://github.com/jmschrei/memesuite-lite): implementations of FIMO and Tomtom that do not require PyTorch. The command-line FIMO and Tomtom tools that used to ship with tangermeme now live there, with expanded functionality.
- [bpnet-lite](https://github.com/jmschrei/bpnet-lite): a minimal implementation of BPNet and ChromBPNet, whose models produced several of the figures and timings above.
- [ledidi](https://github.com/jmschrei/ledidi): gradient-based design of minimal edits to a sequence so that a model predicts a desired output.
- [figwig](https://github.com/jmschrei/figwig): a fast, multithreaded reader and writer of bigWig files, which `extract_loci` uses.

## Citation

If you use tangermeme in your work, please cite:

> Schreiber, J. Tangermeme: a toolkit for understanding *cis*-regulatory logic using deep learning models. *Nature Methods* (2026). https://doi.org/10.1038/s41592-026-03254-z

```bibtex
@article{schreiber2026tangermeme,
	author = {Schreiber, Jacob},
	title = {Tangermeme: a toolkit for understanding cis-regulatory logic using deep learning models},
	journal = {Nature Methods},
	year = {2026},
	doi = {10.1038/s41592-026-03254-z},
	url = {https://doi.org/10.1038/s41592-026-03254-z}
}
```
