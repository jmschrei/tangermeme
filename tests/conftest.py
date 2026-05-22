"""Shared pytest configuration for the tangermeme test suite.

The `device` fixture parametrizes any test (or upstream fixture) that depends
on it over ``cpu`` and ``cuda``. The ``cuda`` parameter is tagged with the
``gpu`` marker and is automatically skipped on machines without CUDA support,
so the default ``pytest`` invocation always runs at least the CPU pass.

Selecting passes from the command line:
    pytest                      # cpu pass; cuda pass too if CUDA is available
    pytest -m gpu               # only the cuda pass
    pytest -m "not gpu"         # only the cpu pass
"""

import pytest
import torch


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA is not available on this machine",
                ),
            ],
        ),
    ]
)
def device(request):
    return request.param
