.. currentmodule:: tangermeme


===============
Release History
===============

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