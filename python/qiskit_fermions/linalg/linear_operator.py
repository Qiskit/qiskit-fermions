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

"""Conversion of an operator into a SciPy linear operator."""

from __future__ import annotations

import scipy.sparse.linalg

from qiskit_fermions.protocols import SupportsLinearOperator


def linear_operator(
    operator: SupportsLinearOperator, norb: int, nelec: int | tuple[int, int]
) -> scipy.sparse.linalg.LinearOperator:
    """Returns a SciPy ``LinearOperator`` for an operator on the ``(norb, nelec)`` FCI sector.

    This is a thin, type-agnostic wrapper around the :class:`.SupportsLinearOperator` protocol
    method, mirroring the free-function style of :func:`ffsim.linear_operator`.

    Args:
        operator: the operator to convert, implementing :class:`.SupportsLinearOperator`.
        norb: the number of spatial orbitals.
        nelec: the electron count -- an integer for a spinless sector, or an ``(n_alpha, n_beta)``
            pair for a spinful one.

    Returns:
        A :class:`scipy.sparse.linalg.LinearOperator` applying ``operator`` on the requested sector.
    """
    return operator._linear_operator_(norb, nelec)
