# tangermeme

[![Unit Tests](https://github.com/jmschrei/tangermeme/actions/workflows/python-package.yml/badge.svg)](https://github.com/jmschrei/tangermeme/actions/workflows/python-package.yml)

[[docs](https://tangermeme.readthedocs.io/en/latest/index.html)][[tutorials](https://github.com/jmschrei/tangermeme/tree/main/docs/tutorials)]

> [!NOTE] 
> tangermeme is under active development. The API has largely been decided on, but may change slightly across versions until the first major release.

The [MEME Suite](https://meme-suite.org/meme/) is a collection of biological sequence analysis tools that rely almost solely on a collection of sequences or the motifs derived from them; tangermeme is an extension of this concept to biological sequence analysis when you have a collection of sequences *and a predictive model.* Hence, it implements many atomic sequence operations such as adding a motif to a sequence or shuffling it out, efficient tools for applying predictive models to these sequences, methods for dissecting what these predictive models have learned, and tools for designing new sequences using these models. tangermeme aims to be assumption free: models can be multi-input or multi-output, functions do not assume a distance and instead return the raw predictions, and when loss functions are necessary they can be supplied by the user. Although we will provide best practices for how to use these functions, our hope is that being assumption-free makes adaptation of tangermeme into your settings as frictionless as possible. All functions are unit-tested and implemented with both compute- and memory-efficient in mind. Finally, although the library was built with operations on DNA sequences in mind, all functions are extensible to any alphabet.

In addition to a library of functions to help you apply predictive models to sequences, future iterations of tangermeme will include PyTorch-based/GPU accelerated command-line tools that range from reimplementations of some of the tools in the MEME suite to new tools for sequence analysis that include attribution scores.

## Installation

`pip install tangermeme`


## Usage

tangermeme aims to be as low-level and simple as possible. This means that models can be any PyTorch model or any wrapper of a PyTorch model as long as the forward function is still exposed, i.e., `y = model(X)` still works. This also means that if you have a model that potentially takes in multiple inputs or outputs and you want to simplify it for the purpose of ISM or sequence design that you can take your model and wrap it however you would like and still use these functions. It also means that all data are PyTorch tensors and that broadcasting is supported wherever possible. Being this flexible sometimes results in bugs, however, so please report any anomalies when doing fancy broadcasting or model wrapping.

#### Ersatz

tangermeme implements atomic sequence operations to help you ask "what if?" questions of your data. These operations can be found in `tangermeme.ersatz`. For example, if you want to insert a subsequence or motif into the middle of a sequence you can use the `insert` function.

```python
from tangermeme.erastz import insert
from tangermeme.utils import one_hot_encode   # Convert a sequence into a one-hot encoding
from tangermeme.utils import characters   # Convert a one-hot encoding back into a string

seq = one_hot_encode("AAAAAA").unsqueeze(0)
merge = insert(seq, "GCGC")[0]

print(characters(merge))
# AAAGCGCGCAAA
```

If you want to dinucleotide shuffle a sequence, you can use the `dinucleotide_shuffle` command.

```python
from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.utils import one_hot_encode
from tangermeme.utils import characters

seq = one_hot_encode('CATCGACAGACTACGCTAC').unsqueeze(0)
shuf = dinucleotide_shuffle(seq, random_state=0)

print(characters(shuf[0, 0]))
# CAGACACGATACGCTCTAC
print(characters(shuf[0, 1]))
# CGACATACGAGCTCACTAC
```

Both shuffling and dinucleotide shuffling can be applied to entire sequence, but they can also be applied to *portions* of the sequence by supplying `start` and `end` parameters if you want to, for instance, eliminate a motif by shuffling the nucleotides.

```python
from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.utils import one_hot_encode
from tangermeme.utils import characters

seq = one_hot_encode('CATCGACAGACGCATACTCAGACTTACGCTAC').unsqueeze(0)
shuf = dinucleotide_shuffle(seq, start=5, end=25, random_state=0)

print(characters(shuf[0, 0]))
# CATCG AGCGACTCAGATACACACTT ACGCTAC    Spacing added to emphasize that the flanks are identical
print(characters(shuf[0, 1]))
# CATCG ACGAGCATCACACTAGACTT ACGCTAC
```

#### Prediction

Once you have your sequence of interest, you usually want to apply a predictive model to it. `tangermeme.predict` implements `predict`, which will handle constructing batches from a big blob of data, moving these batches to the GPU (or whatever device you specify) one at a time and moving the results back to the CPU, and making sure the inference is done without gradients and in evaluation mode. Also, there can be a cool progress bar.

```python
from tangermeme.predict import predict

y = predict(model, X, batch_size=2)
```

#### Marginalization

Given a predictive model and a set of known motifs, a common question is to ask what motifs affect the model's predictions. Rather than trying to scan these motifs against the genome and averaging predictions at all sites -- which is challenging and computationally costly -- you can simply substitute in the motif of interest into a background set of sequences and see what the difference in predictions is. Because `tangermeme` aims to be assumption-free, these functions take in a batch of examples that you specify, and return the predictions before and after adding the motif in for each example. If the model is multi-task, `y_before` and `y_after` will be a tuple of outputs. If the model is multi-input, additional inputs can be specified as a tuple passed into `args`. 

```python
y_before, y_after = marginalize(model, X, "CTCAGTGATG")
```

By default, these functions use the nucleotide alphabet, but you can pass in any alphabet if you're using these models in other settings.

Below, we can see how a BPNet model trained to predict GATA2 binding responds to marginalizing over a GATA motif.

<img src="https://github.com/jmschrei/tangermeme/assets/3916816/66f776e1-b49b-4b31-9e1f-88bce0096400" width="600">


#### Ablation

The conceptual opposite of marginalization is ablation; rather than adding information to a sequence in the form of a motif, you remove information from it by shuffling out a portion of the sequence. This function will handle shuffling (or dinucleotide shuffling) the given number of times and returns the predictions for each shuffle. 

```python
y_before, y_after = ablate(model, X, 995, 1005)
```

As you might expect, if we shuffle a sequence that has the GATA motif in the middle, we see the opposite as the marginalization experiment.

<img src="https://github.com/jmschrei/tangermeme/assets/3916816/871a88bd-3e2c-4538-b928-1ca8921cfe56" width="600">

#### Spacing

Motifs do not occur in isolation in the genome and so, frequently, we want to measure how these motifs interact with each other and how this changes across different spacings. This function takes in a set of motifs and either a fixed distance or one distance for each adjacent pair of motifs, returning the predictions before and after insertion of the full set of motifs.

```python
y_before, y_after = space(model, X, ["CTCAGTGATG", "CTCAGTGATG"], spacing=10)
```

By running this function across several spacings one can, for instance, measure the cooperative effect that nearby AP-1 motifs have on predictions and observe that this cooperativity fades as the motifs get further apart.

<img src="https://github.com/jmschrei/tangermeme/assets/3916816/60bb5465-8c1c-4f32-b0a2-c2a9f9bb9889" width="600">

#### Saturation Mutagenesis

Given an observed sequence a simple question is "what positions are driving model predictions?" One simple way to answer this question is through saturation mutagenesis, i.e., compare the predictions on the original sequence with that of sequences that comprehensively each contain one mutation with respect to that original sequence. This is conceptually similar to deep mutational scanning but using a predictive model instead of running an experiment. In a region with an AP-1 motif, we can run ISM on Beluga and look at AP-1 factor tasks to identify that the AP-1 motif is what is driving the predictions.

```python
from tangermeme.ism import saturation_mutagenesis

y_ref, y_ism = saturation_mutagenesis(model, X)
```

This yields a tensor of a similar shape as the original sequence `X` where each value contains the predicted value (or values) for making the associated substitution. There are numerous ways to combine the predictions of each variant with the predictions on the original sequence and tangermeme allows you to use whichever approach you would like. However, a common one is the Euclidean distance followed by mean-subtracting to get the influence that each <i>observed</i> nucleotide has on the prediction. See the tutorial for more details.

<img src="https://github.com/jmschrei/tangermeme/assets/3916816/34fe309b-bf67-465f-911f-9d83c011ad61" width="800">

#### Variant Effect

A common use case of these predictive models is evaluating the effect that individual mutations have on predictions. Basically, if you have a list of potentially-pathogenic variants, an easy way to screen them for likelihood of being causal is seeing the effect they have on predictions for potentially thousands of experimental readouts. These functions can be divided into those that evaluate the marginal effect of each individual variant or the joint effect of all variants. The second is a little more challenging than the first, and so only marginal effects are currently implemented.

In these functions, one passes in the model, a dictionary or filename for an entire genome/set of sequences, the variants in pandas format (see the tutorial for more details), and the window to extract, and gets back the predictions before and after the variants are incorporated.

```python
from tangermeme.variant_effect import marginal_substitution_effect

y_orig, y_var = marginal_substitution_effect(model, X, variants, in_window=10)
````

#### Design

Given a trained predictive model, one can try to design a sequence that has desired attributions. Specifically, one can try to design a sequence that causes the model to give desired predictions.

Currently, the only design algorithm implemented in tangermeme is a greedy substitution algorithm that will try every given motif at every position each iteration, and take the motif + position combo that yields predictions closest to the desired output from the model.

```python
from tangermeme.design import greedy_substitution

X_hat = greedy_substitution(model, X, motifs, y, mask=idxs, max_iter=3, verbose=True)
```

When the model is the Beluga model and the goal is to design a sequence that yields strong AP-1 binding but ignore the effect on all other tasks, this function inserts three AP-1 binding sites close together. The predictions from the model are much higher for the AP-1 tasks on the designed sequence than the original sequence.

<img src="https://github.com/jmschrei/tangermeme/assets/3916816/519d0648-af02-4e75-bef2-958b5a6629a4" width="600">


