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

"""A protocol to indicate linear operator conversion support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
import scipy.sparse.linalg

if TYPE_CHECKING:
    from qiskit_fermions._lib.linalg.fci import FciLinearOperator


class SupportsLinearOperator(Protocol):
    """The linear-operator conversion contract this package implements.

    This is the same protocol that :class:`ffsim.SupportsLinearOperator` describes, so
    :func:`ffsim.linear_operator` dispatches to the method below on any object of this package. It is
    stated here because the contract holds whether or not ffsim is installed: the native FCI kernel
    backing it depends only on ``scipy``, so without ffsim you call the method directly, as the
    doctest below does.

    .. doctest::

        >>> import numpy as np
        >>> from qiskit_fermions.operators import FermionOperator
        >>> num_op = FermionOperator.from_dict({((True, 0), (False, 0)): 1.0})
        >>> linop = num_op._linear_operator_(norb=2, nelec=1)
        >>> linop.shape
        (2, 2)
        >>> linop.matvec(np.array([1.0, 0.0], dtype=complex))
        array([1.+0.j, 0.+0.j])
    """

    def _linear_operator_(
        self, norb: int, nelec: int | tuple[int, int]
    ) -> scipy.sparse.linalg.LinearOperator:
        """Returns a :class:`scipy.sparse.linalg.LinearOperator` for this operator on the ``(norb, nelec)`` FCI sector."""


class _SupportsFciLinearOperator(Protocol):
    """A protocol for operators carrying a native FCI matrix-vector kernel.

    This is the internal contract through which the :class:`.OperatorTrait` implementations of this
    package support the :class:`ffsim.SupportsLinearOperator` protocol. This is achieved by
    attaching ``_linear_operator_`` methods to the operator classes via a Python wrapper around the
    Rust-backed :func:`_fci_linear_operator_` carrier, rendering the internal wrapper function
    independent of any concrete operator type.
    """

    def _fci_linear_operator_(self, norb: int, nelec: int | tuple[int, int]) -> FciLinearOperator:
        """Returns a native FCI matrix-vector view of this operator on the ``(norb, nelec)`` sector."""


def scipy_linear_operator_from_fci(  # noqa: D417
    self: _SupportsFciLinearOperator, norb: int, nelec: int | tuple[int, int]
) -> scipy.sparse.linalg.LinearOperator:
    """Returns a SciPy ``LinearOperator`` for this operator on the ``(norb, nelec)`` FCI sector.

    This implements the :class:`SupportsLinearOperator` protocol, so an operator carrying a native
    FCI kernel can be passed directly to :func:`scipy.sparse.linalg.expm_multiply` or to
    :func:`ffsim.linear_operator`. It depends only on the internal
    :class:`~qiskit_fermions.protocols._SupportsFciLinearOperator` contract -- the
    ``_fci_linear_operator_`` carrier -- rather than on any concrete operator type, and wraps that
    native matrix-vector kernel in a genuine :class:`scipy.sparse.linalg.LinearOperator`;
    :func:`~scipy.sparse.linalg.expm_multiply` requires the adjoint action, so both ``matvec`` and
    ``rmatvec`` are supplied.

    The native kernel requires a contiguous one-dimensional ``complex128`` vector, whereas SciPy's
    machinery may feed a :class:`~scipy.sparse.linalg.LinearOperator` real probe vectors (from its
    one-norm estimator) or non-contiguous ``(dim, 1)`` column slices. The ``matvec``/``rmatvec``
    wrappers coerce the input with ``numpy.ascontiguousarray(v, complex128).reshape(-1)``; the numpy
    handles are bound once here (per operator) rather than re-resolved on every matvec inside the
    :func:`~scipy.sparse.linalg.expm_multiply` loop.

    This is attached to the operator classes as ``_linear_operator_`` at import time (see
    :mod:`qiskit_fermions.operators`): the native operator classes are compiled types whose
    instances cannot themselves subclass SciPy's :class:`~scipy.sparse.linalg.LinearOperator`, so
    the protocol method is provided in Python.

    Args:
        norb: the number of spatial orbitals.
        nelec: the electron count -- an integer for a spinless sector, or an ``(n_alpha, n_beta)``
            pair for a spinful one.

    Returns:
        A :class:`scipy.sparse.linalg.LinearOperator` applying this operator on the requested sector.
    """
    return scipy_linear_operator_from_kernel(self._fci_linear_operator_(norb, nelec))


def scipy_linear_operator_from_kernel(
    kernel: FciLinearOperator,
) -> scipy.sparse.linalg.LinearOperator:
    """Wraps a native FCI kernel in a SciPy :class:`~scipy.sparse.linalg.LinearOperator`.

    The native kernel requires a contiguous one-dimensional ``complex128`` vector, whereas SciPy's
    machinery may feed a :class:`~scipy.sparse.linalg.LinearOperator` real probe vectors (from its
    one-norm estimator) or non-contiguous ``(dim, 1)`` column slices, so the ``matvec``/``rmatvec``
    wrappers coerce their input.

    This deliberately returns only the SciPy operator and not the kernel's ``trace``: a SciPy
    ``LinearOperator`` has nowhere to carry that metadata, and composing one (scaling, adding) drops
    any attribute attached to it. Callers that need the exact trace should hold on to the ``kernel``
    they passed in and read its ``trace`` directly.

    Args:
        kernel: the native FCI matrix-vector kernel to wrap.

    Returns:
        A :class:`scipy.sparse.linalg.LinearOperator` applying ``kernel`` on its sector.
    """
    # Bind the coercion handles once per operator (not once per matvec): SciPy hands a
    # LinearOperator real probe vectors or non-contiguous (dim, 1) columns, which the native kernel
    # cannot slice directly.
    ascontiguousarray = np.ascontiguousarray
    complex128 = np.complex128

    def matvec(vec):
        return kernel.matvec(ascontiguousarray(vec, complex128).reshape(-1))

    def rmatvec(vec):
        return kernel.rmatvec(ascontiguousarray(vec, complex128).reshape(-1))

    return scipy.sparse.linalg.LinearOperator(
        shape=kernel.shape,
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=kernel.dtype,
    )
