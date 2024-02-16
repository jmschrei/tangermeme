# tangermeme

![](https://github.com/jmschrei/pomegranate/actions/workflows/python-package.yml/badge.svg)

The [MEME Suite](https://meme-suite.org/meme/) is a collection of biological sequence analysis tools that rely almost solely on sequence or motifs derived from them; tangermeme is an extension of this concept to biological sequence analysis when you have sequence *and a predictive model.* Hence, it implements many atomic sequence operations such as adding a motif to a sequence or removing one through shuffling, but also efficient tools for applying predictive models to these sequences and analyzing their results. All functions are unit tested and implemented with both compute- and memory-efficient in mind. Although the library was built with operations on DNA sequences in mind, all functions are extensible to any alphabet.

In addition to a library of functions to help you apply predictive models to sequences, tangermeme includes PyTorch-based/GPU accelerated command-line tools that range from reimplementations of some of the tools in the MEME suite to new tools for sequence analysis that include attribution scores.

## Installation

`pip install tangermeme`


## Usage

tangermeme implements atomic sequence operations to help you ask "what if?" questions of your data.

```
from tangermeme.erastz import insert
from tangermeme.utils import one_hot_encode  # Convert a sequence into a one-hot encoding
from tangermeme.utils import characters 
```
