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

import numpy as np
import pytest
import qiskit_fermions.circuit.library._expm_multiply as expm_multiply_module
import scipy.linalg
import scipy.sparse.linalg
from qiskit_fermions.circuit.library._expm_multiply import _expm_multiply_fci
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.protocols.linear_operator import scipy_linear_operator_from_kernel


def _large_trace_operator(norb: int) -> FermionOperator:
    """A number operator scaled so the sector trace is large, where ``traceA`` matters most."""
    return FermionOperator.from_dict({((True, i), (False, i)): 50.0 for i in range(2 * norb)})


def test_expm_multiply_fci_uses_the_trace_to_precondition(monkeypatch):
    """Supplying the exact trace makes SciPy converge in far fewer matrix-vector products.

    ``traceA`` cannot change the result -- SciPy factors out ``exp(traceA / n)`` and multiplies it
    back -- so the only observable effect of getting it right is how much work SciPy does. A wrong
    (or dropped) trace shows up here as a collapse in that saving.
    """
    norb, nelec = 4, (2, 2)
    operator = _large_trace_operator(norb)
    kernel = operator._fci_linear_operator_(norb, nelec)
    dim = kernel.shape[0]
    rng = np.random.default_rng(0)
    vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    vec /= np.linalg.norm(vec)

    # Count the matrix-vector products SciPy performs, by wrapping the operator it is handed. The
    # native kernel's own ``matvec`` is read-only (a compiled type), so the count is taken one layer
    # out, at the SciPy ``LinearOperator`` the helper builds.
    matvecs = 0
    wrap_kernel = scipy_linear_operator_from_kernel

    def counting_wrapper(kernel_):
        nonlocal matvecs
        linop = wrap_kernel(kernel_)

        def counting_matvec(v):
            nonlocal matvecs
            matvecs += 1
            return linop.matvec(v)

        return scipy.sparse.linalg.LinearOperator(
            shape=linop.shape, matvec=counting_matvec, rmatvec=linop.rmatvec, dtype=linop.dtype
        )

    monkeypatch.setattr(expm_multiply_module, "scipy_linear_operator_from_kernel", counting_wrapper)
    _expm_multiply_fci(operator, vec, norb, nelec, scale=-1j)
    with_trace = matvecs

    # Baseline: the same exponential without the trace hint, as the code did before.
    matvecs = 0
    scipy.sparse.linalg.expm_multiply(-1j * counting_wrapper(kernel), vec, traceA=0.0)
    without_trace = matvecs

    # Measured separation is ~200x; assert an order of magnitude to stay robust across SciPy versions.
    assert with_trace * 10 < without_trace


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


@pytest.mark.parametrize("scale", [-1j * 0.7, 0.3 + 0.2j])
def test_expm_multiply_fci_scales_the_trace(monkeypatch, scale):
    """``scale`` must multiply the trace too, since ``trace(c * A) == c * trace(A)``.

    Unlike the preconditioning above, this has no signature in the result: an unscaled trace still
    yields the right vector, only worse-conditioned. So it is observed where it is applied, by
    capturing the shift the helper hands to SciPy.
    """
    norb, nelec = 4, (2, 2)
    operator = _large_trace_operator(norb)
    kernel = operator._fci_linear_operator_(norb, nelec)
    dim = kernel.shape[0]
    vec = np.zeros(dim, dtype=complex)
    vec[0] = 1.0

    traces = []
    expm_multiply = scipy.sparse.linalg.expm_multiply

    def recording_expm_multiply(operator_, vector, **kwargs):
        traces.append(kwargs["traceA"])
        return expm_multiply(operator_, vector, **kwargs)

    monkeypatch.setattr(scipy.sparse.linalg, "expm_multiply", recording_expm_multiply)
    _expm_multiply_fci(operator, vec, norb, nelec, scale=scale)

    assert traces == [pytest.approx(scale * kernel.trace)]
