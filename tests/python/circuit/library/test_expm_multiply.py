# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for matrix-exponential helpers."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse.linalg
from qiskit_fermions.circuit.library._expm_multiply import _expm_multiply_fci
from qiskit_fermions.operators import FermionOperator


def test_expm_multiply_fci_supplies_exact_trace():
    """The helper reads the exact fixed-sector trace off the native kernel."""
    # scalar 1.5 over the dim-3 sector plus n_0 (occupied in one of the three strings): 1.5*3 + 2.0
    operator = FermionOperator.from_dict(
        {
            (): 1.5,
            ((True, 0), (False, 0)): 2.0,
            ((True, 0), (False, 1)): 7.0,  # off-diagonal: contributes nothing to the trace
        }
    )
    vec = np.array([1.0, 0.0, 0.0], dtype=complex)

    with patch(
        "scipy.sparse.linalg.expm_multiply", wraps=scipy.sparse.linalg.expm_multiply
    ) as call:
        _expm_multiply_fci(operator, vec, 3, 2)

    assert call.call_args.kwargs["traceA"] == pytest.approx(8.5)


def test_expm_multiply_fci_scales_the_trace():
    """``scale`` multiplies the operator, so it must multiply the trace too."""
    operator = FermionOperator.from_dict({(): 1.5, ((True, 0), (False, 0)): 2.0})
    vec = np.array([1.0, 0.0, 0.0], dtype=complex)
    scale = -1j * 0.4

    with patch(
        "scipy.sparse.linalg.expm_multiply", wraps=scipy.sparse.linalg.expm_multiply
    ) as call:
        _expm_multiply_fci(operator, vec, 3, 2, scale=scale)

    assert call.call_args.kwargs["traceA"] == pytest.approx(scale * 8.5)


@pytest.mark.parametrize("scale", [1.0, -1j * 0.7, 0.3 + 0.2j])
def test_expm_multiply_fci_matches_dense_expm(scale):
    """The result matches a dense matrix exponential of the same operator."""
    rng = np.random.default_rng(11)
    norb, nelec = 4, (2, 1)
    terms: dict = {(): 3.0}
    for i in range(2 * norb):
        terms[((True, i), (False, i))] = 1.0 + 0.25 * i
    for i, j in [(0, 1), (1, 0), (4, 5), (5, 4)]:
        terms[((True, i), (False, j))] = 0.4

    operator = FermionOperator.from_dict(terms)
    kernel = operator._fci_linear_operator_(norb, nelec)
    dim = kernel.shape[0]

    dense = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        basis = np.zeros(dim, dtype=complex)
        basis[col] = 1.0
        dense[:, col] = kernel.matvec(basis)

    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    vec /= np.linalg.norm(vec)

    expected = scipy.linalg.expm(scale * dense) @ vec
    actual = _expm_multiply_fci(operator, vec, norb, nelec, scale=scale)

    np.testing.assert_allclose(actual, expected, atol=1e-11)
