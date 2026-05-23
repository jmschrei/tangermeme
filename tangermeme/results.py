# results.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Shared return types for the perturbation-experiment functions
(`ablate`, `marginalize`, and `variant_effect.{substitution,deletion,
insertion}_effect`), plus their per-annotation counterparts. Centralized
here so agents see one shape, not three near-duplicates."""

from __future__ import annotations

from typing import NamedTuple

import torch


class PerturbationResult(NamedTuple):
	"""Result of applying a function before and after a perturbation.

	Returned by `ablate`, `marginalize`, and the three `variant_effect`
	functions. Positional unpacking (``y_before, y_after = ...``)
	continues to work alongside attribute access (``result.y_before``).
	"""

	y_before: torch.Tensor | list[torch.Tensor]
	y_after: torch.Tensor | list[torch.Tensor]


class PerturbationAnnotationsResult(NamedTuple):
	"""Result of applying a perturbation per annotation, stacked along
	axis 0.

	Returned by `ablate_annotations` and `marginalize_annotations`.
	Pluralized field names reflect that each tensor contains one entry
	per annotation.
	"""

	y_befores: torch.Tensor | list[torch.Tensor]
	y_afters: torch.Tensor | list[torch.Tensor]
