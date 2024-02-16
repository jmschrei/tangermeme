# tangermeme

[![Unit Tests](https://github.com/jmschrei/tangermeme/actions/workflows/python-package.yml/badge.svg)](https://github.com/jmschrei/tangermeme/actions/workflows/python-package.yml)

The [MEME Suite](https://meme-suite.org/meme/) is a collection of biological sequence analysis tools that rely almost solely on sequence or motifs derived from them; tangermeme is an extension of this concept to biological sequence analysis when you have sequence *and a predictive model.* Hence, it implements many atomic sequence operations such as adding a motif to a sequence or removing one through shuffling, but also efficient tools for applying predictive models to these sequences and analyzing their results. All functions are unit tested and implemented with both compute- and memory-efficient in mind. Although the library was built with operations on DNA sequences in mind, all functions are extensible to any alphabet.

In addition to a library of functions to help you apply predictive models to sequences, tangermeme includes PyTorch-based/GPU accelerated command-line tools that range from reimplementations of some of the tools in the MEME suite to new tools for sequence analysis that include attribution scores.

## Installation

`pip install tangermeme`


## Usage

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
