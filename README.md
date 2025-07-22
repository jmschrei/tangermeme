# tangermeme

[![Downloads](https://static.pepy.tech/badge/tangermeme)](https://pepy.tech/project/tangermeme) [![Unit Tests](https://github.com/jmschrei/tangermeme/actions/workflows/python-package.yml/badge.svg)](https://github.com/jmschrei/tangermeme/actions/workflows/python-package.yml) [![Documentation Status](https://readthedocs.org/projects/tangermeme/badge/?version=latest)](https://tangermeme.readthedocs.io/en/latest/?badge=latest)

[[docs](https://tangermeme.readthedocs.io/en/latest/index.html)][[tutorials](https://github.com/jmschrei/tangermeme/tree/main/docs/tutorials)][[vignettes](https://github.com/jmschrei/tangermeme/tree/main/docs/vignettes)]

> [!NOTE] 
> tangermeme is under active development. The API has largely been decided on, but may change slightly across versions until the first major release.

Training sequence-based machine learning models has become widespread when studying genomics. But, what have these models learned, and what do we even do with them after training? `tangermeme` aims to provide robust and easy-to-use tools for the what-to-do-after-training question. `tangermeme` implements many atomic sequence operations such as adding a motif to a sequence or shuffling it out, efficient tools for applying predictive models to these sequences, methods for dissecting what these predictive models have learned, and tools for designing new sequences using these models. `tangermeme` aims to be assumption free: models can be multi-input or multi-output, functions do not assume a distance and instead return the raw predictions, and when loss functions are necessary they can be supplied by the user. Although we will provide best practices for how to use these functions, our hope is that being assumption-free makes adaptation of tangermeme into your settings as frictionless as possible. All functions are unit-tested and implemented with both compute- and memory-efficient in mind. Finally, although the library was built with operations on DNA sequences in mind, all functions are extensible to any alphabet.

Please see the documentation and tutorials linked at the top of this README for more extensive documentation. If you only read one vignette, read THIS ONE: [Inspecting what Cis-Regulatory Features a Model has Learned](https://tangermeme.readthedocs.io/en/latest/vignettes/Inspecting_What_Cis-Regulatory_Features_a_Model_Has_Learned.html).

<br>
<img src="https://github.com/user-attachments/assets/20b186e7-73af-46c7-a7b6-5484c714036e" width=60%>


## Installation

`pip install tangermeme`

## Roadmap

This first release focused on the core prediction-based functionality (e.g., marginalizations, ISM, etc..) that subsequent releases will build on. Although my focus will largely follow my research projects and the feedback I receive from the community, here is a roadmap for what I currently plan to focus on in the next few releases.

- v0.1.0: ✔️ Prediction-based functionality
- v0.2.0: ✔️ Attribution-based functionality (e.g., attribution marginalization, support for DeepLIFT, seqlet calling..)
- v0.3.0: ✔️ PyTorch ports for MEME and TOMTOM and command-line tools for the prediction- and attribution- based functionality 
- v0.4.0: ✔️ Focus on interleaving tools and iterative approaches
- v0.5.0: More sophisticated methods for motif discovery

## Command-line Tools

> [!WARNING]
> These FIMO and Tomtom command-line tools have been moved to [memesuite-lite](https://github.com/jmschrei/memesuite-lite), where their functionality has also been expanded and the PyTorch requirement has been removed. Please use those!

## Usage

tangermeme aims to be as low-level and simple as possible. This means that models can be any PyTorch model or any wrapper of a PyTorch model as long as the forward function is still exposed, i.e., `y = model(X)` still works. This also means that if you have a model that potentially takes in multiple inputs or outputs and you want to simplify it for the purpose of ISM or sequence design that you can take your model and wrap it however you would like and still use these functions. It also means that all data are PyTorch tensors and that broadcasting is supported wherever possible. Being this flexible sometimes results in bugs, however, so please report any anomalies when doing fancy broadcasting or model wrapping.

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
from tangermeme.utils import one_hot_encode   # Convert a sequence into a one-hot encoding
from tangermeme.utils import characters   # Convert a one-hot encoding back into a string

seq = one_hot_encode("AAAAAA").unsqueeze(0)
merge = substitute(seq, "GCGC")[0]

print(characters(merge))
# AGCGCA
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

#### DeepLIFT/SHAP Attributions

A powerful form of analysis is to run your predictive model backwards to highlight the input characters driving predictions. If the model predictions are accurate one can interpret these highlights -- or attributions -- as the actual driver of experimental signal. One of these attribution methods is called DeepLIFT/SHAP (merging ideas from DeepLIFT and DeepSHAP). tangermeme has a built-in implementation that is simpler, more robust, and corrects a few issues with other implementations of DeepLIFT/SHAP.

```python
from tangermeme.deep_lift_shap import deep_lift_shap

X_attr = deep_lift_shap(model, X, target=267, random_state=0)
```

Note that for multi-task models a target must be set to calculate attributions for one output at a time.

![image](https://github.com/jmschrei/tangermeme/assets/3916816/db628856-182a-427a-be95-2dc33def11b0)


#### Marginalization

Given a predictive model and a set of known motifs, a common question is to ask what motifs affect the model's predictions. Rather than trying to scan these motifs against the genome and averaging predictions at all sites -- which is challenging and computationally costly -- you can simply substitute in the motif of interest into a background set of sequences and see what the difference in predictions is. Because `tangermeme` aims to be assumption-free, these functions take in a batch of examples that you specify, and return the predictions before and after adding the motif in for each example. If the model is multi-task, `y_before` and `y_after` will be a tuple of outputs. If the model is multi-input, additional inputs can be specified as a tuple passed into `args`. 

```python
y_before, y_after = marginalize(model, X, "CTCAGTGATG")
```

By default, these functions use the nucleotide alphabet, but you can pass in any alphabet if you're using these models in other settings.

Below, we can see how a BPNet model trained to predict GATA2 binding responds to marginalizing over a GATA motif.

<img src="https://github.com/jmschrei/tangermeme/assets/3916816/66f776e1-b49b-4b31-9e1f-88bce0096400" width="600">

Importantly, all methods that modify the sequence can take in an optional `func` parameter to change the function that gets applied before and after performing the sequence modification (in this case, substituting in a motif). By default, this function is just the `predict` function but it can just as easily be the `deep_lift_shap` method to give you attributions before and after.

```python
attr_before, attr_after = marginalize(model, X, "CTCAGTGATG", func=deep_lift_shap)
```

![image](https://github.com/jmschrei/tangermeme/assets/3916816/9d453d4c-aba8-449b-9911-c6e17ef4ab77)


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

Given an observed sequence a simple question is "what positions are driving model predictions?" One simple way to answer this question is through saturation mutagenesis, i.e., compare the predictions on the original sequence with that of sequences that comprehensively each contain one mutation with respect to that original sequence. This is another form of attribution method that is conceptually similar to deep mutational scanning but using a predictive model instead of running an experiment. In a region with an AP-1 motif, we can run ISM on Beluga and look at AP-1 factor tasks to identify that the AP-1 motif is what is driving the predictions.

```python
from tangermeme.ism import saturation_mutagenesis

X_attr = saturation_mutagenesis(model, X)
```

This yields a tensor of a similar shape as the original sequence `X` where each value contains the predicted value (or values) for making the associated substitution. There are numerous ways to combine the predictions of each variant with the predictions on the original sequence and tangermeme allows you to use whichever approach you would like. However, a common one is the Euclidean distance followed by mean-subtracting to get the influence that each <i>observed</i> nucleotide has on the prediction. See the tutorial for more details.

<img src="https://github.com/jmschrei/tangermeme/assets/3916816/34fe309b-bf67-465f-911f-9d83c011ad61" width="800">

#### Variant Effect

A common use case of these predictive models is evaluating the effect that mutations in sequence have on predictions. In practice, there are several situations where calculating variant effect is helpful but potentially the most clinically relevant one is having a list of variants that have been implicated by some sort of association study and wanting to screen them for some likelihood of being truly causal. In this case, given a model that predicts a readout of interest, e.g. protein binding and chromatin accessibility, one can see how the predictions change before and after variants are incorporated into the sequence. Presumably, the mutations that are actual drivers of phenotype will cause changes in predicted readout, and those that are passengers will not have as strong an effect.

The simplest form of variant is the substitution, where one or more characters in a sequence are changed to another character. We can easily get predictions before and after substitutions are included in the sequence. Importantly, more than one substitution can be encoded in each sequence.

```python
from tangermeme.variant_effect import substitution_effect

substitutions = torch.tensor([
    [0, 1058, 0]
])

y, y_var = substitution_effect(model, X, substitutions)
````

When using a BPNet model that predicts GATA2 binding, we can see that a single substitution encoded at this position in the example seems to knock out predicted signal in the middle of the window entirely. Quite a strong effect.

![image](https://github.com/jmschrei/tangermeme/assets/3916816/9f241e09-3510-4c59-8694-4c6b5b77a16b)

tangermeme can also handle deletions and insertions. These situations are a bit more complicated because the sequences before and after incorporating variants are different lengths and so trimming needs to be done. See the tutorials for more explanation for how to control this trimming.

```python
from tangermeme.variant_effect import deletion_effect
from tangermeme.variant_effect import insertion_effect

y, y_var = deletion_effect(model, X, deletions)
y, y_var = insertion_effect(model, X, insertions)
```

#### Design

Given a trained predictive model, one can try to design a sequence that has desired attributions. Specifically, one can try to design a sequence that causes the model to give desired predictions.

Currently, the only design algorithm implemented in tangermeme is a greedy substitution algorithm that will try every given motif at every position each iteration, and take the motif + position combo that yields predictions closest to the desired output from the model.

```python
from tangermeme.design import greedy_substitution

X_hat = greedy_substitution(model, X, motifs, y, mask=idxs, max_iter=3, verbose=True)
```

When the model is the Beluga model and the goal is to design a sequence that yields strong AP-1 binding but ignore the effect on all other tasks, this function inserts three AP-1 binding sites close together. The predictions from the model are much higher for the AP-1 tasks on the designed sequence than the original sequence.

<img src="https://github.com/jmschrei/tangermeme/assets/3916816/519d0648-af02-4e75-bef2-958b5a6629a4" width="600">

#### Seqlet Calling

In contrast to motif scanning, which usually relies solely on nucleotide sequence to determine whether a motif matches, seqlet calling is the identification of spans of nucleotides that have high attribution score. Seqlet calling methods usually do not rely on sequence at all because they are the first step in identifying repeating patterns based on having high attributions.

```python
from tangermeme.seqlet import recursive_seqlets

seqlets = recursive_seqlets(X_attr.sum(dim=1)) # You pass in a 2D tensor, not a 3D one
```

![image](https://github.com/user-attachments/assets/152f63ab-ef58-42df-902a-1c1be2641868)

The TF-MoDISco seqlet calling algorithm is also implemented. See the seqlet tutorial or API documentation for more details.

Because seqlets are called entirely based on attributions, it is sometimes unclear whether the sequence content is similar to any known motif. Now, you can use `annotate_seqlets` to match seqlets to a motif database using TOMTOM! By default this will give you the nearest motif match for each seqlet but can give you any number of matches you want.

```python
motifs = read_meme("motifs.meme.txt")
motif_idxs, motif_pvalues = annotate_seqlets(X, seqlets, motifs)
```

![image](https://github.com/user-attachments/assets/5c4ed1ab-1c1a-4d33-96e8-b25613970a22)


#### Annotations

In `tangermeme`, an annotation is any genomic span. This means it can be motif hits, seqlets, hit calls, etc., as long as it has a defined start and end coordinate. Given a set of annotations you can do many things. First, you can count the number of times each annotation appears in each sequence. 

```python
from tangermeme.annotate import count_annotations

y = count_annotations((seqlets['example_idx'], motif_idxs[:, 0]))
```

For each annotation, you can get results before and after ablating the annotation from the sequence. Here is an example of running DeepLIFT/SHAP on an example after shuffling each motif instance independently, with only one line of code.

```python
from tangermeme.ablate import ablate_annotations

y_before, y_after = ablate_annotations(model, X, annotations, func=deep_lift_shap)
```

![image](https://github.com/user-attachments/assets/997ca242-8c27-46b6-8204-b898e5f20166)




