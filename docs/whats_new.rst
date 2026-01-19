.. currentmodule:: tangermeme


===============
Release History
===============


Version 1.0.2
=============

Highlights
----------

	- Fixed `extract_matching_loci` not respecting the `chrom` parameter


Version 1.0.1
=============

Highlights
----------

	- The slowest unit tests have been refocused, bringing total unit test time from ~75s to ~33s.


deep_lift_shap
--------------

	- Add conversion to model dtype to improve usability, and appropriate unit test.


design
------

	- Changed `greedy_substitution` without `y` to not make a pseudo target and instead truly just try to maximize predictions. In principle, the pseudo target works identically, but can lead to overflow of values in some settings and is generally less precise.



match
-----

	- Set default n_jobs from -1 to 1 to avoid Child Process Errors on small tasks.


Version 1.0.0
=============

Highlights
----------

	- Our first major release, corresponding to the paper publication.
	- Changes the `recursive_seqlet` calling algorithm slightly to be more principled
	- Adds in new design methods and features


seqlets
-------

	- The `recursive_seqlet` algorithm has been slightly altered to make the calculated p-values more faithful. Rather than calculating a null as the empirically observed attribution sum across different lengths, where the "p-value" is just the probability that the observed attribution is higher, null distributions for different lengths are inferred from the previous lengths


design
------

	- `screen` is added in as a new design method that randomly generates sequences and chooses the one with the best predictions. Each batch is fast because nothing special is done, but also each batch is independent from the others and so there is no guarantee that each iteration yields better results
	- Design methods now allow you to not pass in a `y` target value and instead will try to just maximize the predictions.



Version 0.5.1
=============

Highlights
----------

	- Add `summits` to `extract_loci` to center on summits when a BED10 file is provided
	- Improve casting of indexes for variant effect predictions
	- Slight improvement to the usability of deep_lift_shap



Version 0.5.0
=============

Highlights
----------

	- The Tomtom and FIMO tools have been moved to memesuite-lite so they can be used without a PyTorch dependency
	- All internals tools that used Tomtom and FIMO now call the memesuite-lite versions

annotate
--------

	- The call to tomtom now goes to memesuite-lite


io
----

	- `read_meme` now calls the memesuite-lite function and wraps the numpy arrays into torch tensors.
	- `return_filtered` has been added as an optional parameter to `extract_loci` where, if set to true, returns a list of indexes for the loci that are kept or discarded. Note that the indexes are into the INTERLEAVED LOCI, not the original set of indices.


plot
----

	- Improved the placement of annotation labels in `plot_logo`. Thanks Nikolaus Mandlburger!
	- Fixed a bug where annotations were extended an additional basepair to the right


seqlet
------

	- The `recursive_seqlet` algorithm has been slightly modified to more closely match the provided description. This change involves using the calculated p-values instead of the maximum p-value for each position across all seqlets of smaller size. As a consequence, motifs should not be be shifted to the right anymore.


utils
-----

	- Added a `example_to_fasta_coordinates` which will convert the relative coordinates in examples to exact coordinates on the genome when provided with a BED file of examples and a FASTA file. This is useful if you have seqlet coordinates for each example and need to convert them to positions on the genome.



Version 0.4.4
=============

io
----

	- Added `one_hot_to_fasta` which takes a 3D one-hot encoded tensor and an optional list of headers and outputs a FASTA file with those sequences. 


plot
----

	- Added `plot_attributions` which wraps the calculation and the visualization of attributions between multiple models and multiple sequences.
	- Added `show_score` to `plot_logo` where you can optionally hide the score from the visualization


predict
-------

	- Added `dtype` to `predict`, which will autocast the model and the data to the desired dtype to increase speed. Currently only supports the dtypes supported by torch.autocast. This allows datasets to be represented as torch.uint8 and only converted to higher precision in each batch, yielding significant memory savings.


tools/fimo
----------

	- Fixed a bug to allow dict[str: numpy.ndarray] to be used for the motifs. Thanks @SeppeDeWinter!



Version 0.4.3
=============

ersatz
------

	- Substitute now accepts Ns or all-zeroes positions as inputs and, at those positions, will not alter the original sequence. If only one motif is given, this will be the same across all background sequences. If one motif is given per background sequence, this is done on a per-background example.
	- The above change means that higher-level functions like `marginalize` can now be run with motifs that contain missing characters, without any changes needed.
	- The default `start` and `end` of `dinucleotide_shuffle` have been set to `None` because using `0` and `-1` meant that the last provided position never got shuffled.


design
------

	- Changed `mask` parameter to `output_mask` 
	- Added `input_mask` which restricts what positions can be the start of motifs, so design can be restricted to subsets of the sequence or certain important elements can be ignored.
	- Significantly sped up the creation of sequences with tiled motifs implanted using a numba function, which can speed up design 3-10x.
	- Added in `greedy_marginalize` which design constructs using marginalizations




Version 0.4.1
=============

Highlights
----------


plots
-----

	- Fixed a bug where `plot_logo` raises an error when `start` and `end` are not provided but `annotations` are. 
	- Fixed a bug where `plot_logo` plots annotations using calls to `plt` instead of directly on the provided artboard.


tools
-----

	- Sped up `tomtom` by using more compact dtypes and avoiding cache misses
	- Added `symmetric_tomtom` which takes in a set of items and orders them such that the smaller item is always the query and the larger one is always the target. This reduces the number of background distributions that need to be made from a quadratic number to a linear one, significantly speeding up the algorithm.


utils
-----

	- Added `reverse_complement` function that can convert one-hot encodings and strings. Thanks @Al-Murphy!


Version 0.4.0
==============

Highlights
----------

	- At a high level, this release focuses on quick ways to understand what a model has learned. This means extending seqlet calling functionality as well as introducing handling of annotations, which are any sort of notation of span along the genome -- seqlet calls, motif matches, and hit calls.


annotate
--------

	- Added in a new file for handling annotations.
	- Includes a `count_annotations` function for converting a sparse list of annotations into a dense matrix of counts.
	- Also includes a `pairwise_annotations` function for looking at pairs of motifs that are learned.
	- Also includes a `pairwise_annotations_space` function for looking at spacing between pairs of functions.
	- Also includes an `annotate_seqlet` function for annotating seqlets using TOMTOM and a reference database.


seqlet
------

	- Added in `recursive_seqlets`, which calls seqlets using a recursive definition that all spans within a seqlet must also be independently called as seqlets.


plot
----

	- Added in `plot_pwm` that takes in a PWM whose rows sum to 1 and plots the information content weighted characters as well as the reverse complement.


utils
-----

	- Added in a `pwm_consensus` function that takes in a single PWM and returns a one-hot encoded version of the consensus sequence.
	- Added in an `extract_signal` function for extracting sums over variable-length spans from tensors. 



Version 0.3.0
==============

Highlights
----------

	- Added in a new TOMTOM implementation and a revamped FIMO implementation
	- TOMTOM and FIMO both have command-line tools in `tangermeme`


FIMO
----

	- The PyTorch implementation has been exchanged for a numba based one.
	- The new signature is a single function called `fimo`
	- A command-line tool can be used with the signature `tangermeme fimo ...`


TOMTOM
------

	- A numba-based implementation has been added in the function `tomtom`
	- A command-line tool can be used with the signature `tangermeme tomtom ...`


utils
-----
	
	- `chunk` and `unchunk` have been added in to chunk long sequences into blocks that can be operated on by methods with fixed-window inputs, such as machine learning models, and for converting the predictions from these approaches back into a contiguous format.


match
-----

	- Implemented updates to substantially reduce memory use and runtime of extract_matching_loci. This was mainly achieved by
	1) Avoid using io.extract_loci, which one hot encodes all loci into a single large tensor. Instead, the locus sequences are extracted one by one, keeping only one in memory at a time. The N and GC percentages are calculated directly from the sequence, and only those values are stored.
	2) Calculate genome wide N and GC percentages by taking slices of the chromosomal DNA sequences and using the count method of python strings. This is significantly faster than the previous approach using numpy isin, and avoids keeping several copies of the sequence in memory at the same time.

	- Various other changes:
	1) Counts from regions that cannot be extracted from a provided bigwig file (such as for a missing chromosome) are now set to nan rather than 0. This will effect the threshold value used for filtering background regions.
	2) Small change to the binning strategy for gc values, which could mean that matching loci generated in a previous version will not be reproduced exactly in all cases, even when using the same random seed.
	3) Enable the handling of 'N' in sequences or [0,0,0,0], i.e. an ambiguous genomic positions. Updated the `characters()` and the `_validate_input()` in `utils` module to enable this.


Version 0.2.3
==============

match
-----

	- Expanded the `ignore` parameter to ignore all non-ACGT characters.


Version 0.2.2
==============

plot
----

	- Fixed issue in `plot_logo` raised by @sandyfloren where passing in annotations without passing in `start` or `end` would raise an error. Now, `start` defaults to 0 and `end` defaults to the length of the sequence.


tools
-----

	- FIMO is now base 2 instead of base e, to better match the MEME-suite tool. p-values should remain the same but scores will change.
	- FIMO `hits` will now return p-values, and will longer return an uninformative `attr` column


product
-------

	- `apply_pairwise` has been added along with documentation and unit tests 


match
-----

	- Fixes an issue with trying to calculate the mean over an array of integers by changing the array to be dtype float. via @adamyhe



Version 0.2.1
==============

deep_lift_shap
--------------

	- Removed the autocasting to 32-bit floats, enabling attributions to be calculated at other resolutions
	- Removes ~100 LOC and the DeepLiftShap object, integrating that code directly into the `deep_lift_shap` function
	- Only assigns hooks once at the beginning of the function and clears them upon an error or completion of function, instead of assigning and clearing hooks every batch


Version 0.2.0
==============

Highlights
----------

	- Alters the API of several functions to make them more general, with the option of taking in a function to apply instead of defaulting to predict, while still back compatible
	- Adds in `deep_lift_shap` and `seqlet` to operate on attributions


deep_lift_shap
-------------
	
	- Added in a stand-alone implementation of deep_lift_shap 
	- This implementation resolves several issues with Captum, e.g., with pooling layers
	- Allows batching of example-reference pairs across examples (so batch_size can be > than n_shuffles)
	- Allows batch_size to be much smaller than n_shuffles with the results aggregated once all references have been processed to allow large models to be run
	- Allows additional non_linear operations to be registered by passing in a dictionary
	- Allows the raw multipliers to be returned with `raw_output=True` or the aggregated attribution scores


ism
----

	- Changes the default output from the raw output (which you can get with `raw_output=True`) to defaultly aggregated attribution values to make the API compatible


marginalize
------------

	- Change the signature to take in an optional function that gets applied before/after the substitution, default is predict
	- Change the signature to take in **kwargs that get passed into the optional function
	- Change the signature to take in `additional_func_kwargs` that is an alternative and safer way to pass arguments into the function


ablate
------

	- Change the signature to take in an optional function that gets applied before/after the ablation, default is predict
	- Change the signature to take in **kwargs that get passed into the optional function
	- Change the signature to take in `additional_func_kwargs` that is an alternative and safer way to pass arguments into the function


space
------

	- Change the signature to take in an optional function that gets applied before/after the substitutions, default is predict
	- Change the signature to take in **kwargs that get passed into the optional function
	- Change the signature to take in `additional_func_kwargs` that is an alternative and safer way to pass arguments into the function


variant_effect
--------------

	- Change the name of `marginal_substitution_effect` to `substitution_effect`
	- Change the API of `substitution_effect` to take in a tensor of original sequences and a tensor of substitutions
	- Change the API of `substitution_effect` to take in an optional function and **kwargs and `additional_func_kwargs` to pass into `func`
	- Change the name of `marginal_deletion_effect` to `deletion_effect`
	- Change the API of `deletion_effect` to take in a tensor of original sequences and a tensor of deletions
	- Change the API of `deletion_effect` to take in an optional function and **kwargs and `additional_func_kwargs` to pass into `func`
	- Change the name of `marginal_insertion_effect` to `insertion_effect`
	- Change the API of `insertion_effect` to take in a tensor of original sequences and a tensor of insertions
	- Change the API of `insertion_effect` to take in an optional function and **kwargs and `additional_func_kwargs` to pass into `func`


seqlet
------

	- Add a new file for the identification of seqlets
	- Add `tfmodisco_seqlets` which is a simplified and documented version of the seqlet calling in `tfmodisco` that returns dataframes


Version 0.1.0
==============

Highlights
----------

	- This is the first major release of tangermeme and contains the first version of the core functionality.


ersatz
------

	- This module implements common sequence manipulation methods such as substitutions, insertions, deletions, and shufflings of sequences.


predict
-------

	- This module implements efficient batched prediction that can handle models that accept multiple inputs or multiple outputs.


marginalize
-----------

	- This module implements marginalization experiments, where predictions are made for a set of sequences, a motif is substituted into the middle, and then new predictions are made for the new sequences.


space
-----
	
	- This module implements spacing experiments where predictions are made for a set of sequences, a set of motifs are inserted with a given spacing, and then new predictions are made for the new sequences.


io
---

	- This module implements I/O functions for common data types as well as for extracting examples for machine learning models.


ism
---

	- This module implements in silico saturated mutagenesis (ISM).


variant_effect
--------------

	- This module implements functions for evaluating the marginal effect of variants on model predictions.


Version 0.0.1
==============

Highlights
----------

	- Initial release
