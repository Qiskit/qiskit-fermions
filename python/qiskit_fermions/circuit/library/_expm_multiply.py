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

"""Shared matrix-exponential helpers for fermionic circuit simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import scipy.sparse.linalg

from qiskit_fermions.protocols.linear_operator import scipy_linear_operator_from_kernel

if TYPE_CHECKING:
    from qiskit_fermions.protocols.linear_operator import _SupportsFciLinearOperator


def _expm_multiply_fci(
    operator: _SupportsFciLinearOperator,
    vec: np.ndarray,
    norb: int,
    nelec: int | tuple[int, int],
    scale: complex = 1.0,
) -> np.ndarray:
    r"""Applies ``exp(scale * operator)`` to ``vec`` on the ``(norb, nelec)`` FCI sector.

    This goes through the internal ``_fci_linear_operator_`` carrier rather than the public
    :meth:`.SupportsLinearOperator._linear_operator_` protocol method, because the native kernel it
    returns exposes the operator's exact fixed-sector ``trace`` alongside its matrix-vector action.
    SciPy uses that trace to precondition the exponential (it factors out ``exp(traceA / n)``), which
    is not a correctness input but is a large win in both speed and accuracy for an operator with a
    sizeable trace. A SciPy ``LinearOperator`` has nowhere to carry the value (and drops attributes
    when scaled), so the trace is read off the kernel here instead.

    ``scale`` multiplies the operator, so it must multiply the trace too: ``trace(c * A)`` is
    ``c * trace(A)``. It is applied to the trace *before* SciPy sees it, since ``scale * linop``
    produces a composed operator that no longer carries the kernel's metadata.

    Args:
        operator: the operator to exponentiate, exposing the internal FCI kernel carrier.
        vec: the state vector to apply the exponential to.
        norb: the number of spatial orbitals.
        nelec: the electron count -- an integer for a spinless sector, or an ``(n_alpha, n_beta)``
            pair for a spinful one.
        scale: a scalar multiplying the operator inside the exponential.

    Returns:
        The vector ``exp(scale * operator) @ vec``.
    """
    kernel = operator._fci_linear_operator_(norb, nelec)
    linop = scipy_linear_operator_from_kernel(kernel)
    return cast(
        np.ndarray,
        scipy.sparse.linalg.expm_multiply(scale * linop, vec, traceA=scale * kernel.trace),
    )
