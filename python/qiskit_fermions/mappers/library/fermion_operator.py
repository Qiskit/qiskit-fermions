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

"""Conversion of an operator into a :class:`.FermionOperator`."""

from __future__ import annotations

from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.protocols import SupportsFermionOperator


def fermion_operator(operator: SupportsFermionOperator) -> FermionOperator:
    """Converts an operator into a :class:`.FermionOperator`.

    This is a thin, type-agnostic wrapper around the :class:`.SupportsFermionOperator` protocol
    method, mirroring the free-function style of :func:`ffsim.fermion_operator`.

    Args:
        operator: the operator to convert, implementing :class:`.SupportsFermionOperator`.

    Returns:
        The operator converted into a :class:`.FermionOperator`.

    .. doctest::

        >>> from qiskit_fermions.mappers.library import fermion_operator
        >>> from qiskit_fermions.operators import MajoranaOperator
        >>> maj_op = MajoranaOperator.from_dict({(0, 1): 1})
        >>> fer_op = fermion_operator(maj_op)
        >>> print(format(fer_op.normal_ordered().simplify()))
          0.000000e0 +1.000000e0j * ()
          0.000000e0 -2.000000e0j * (+0 -0)
    """
    return operator._fermion_operator_()
