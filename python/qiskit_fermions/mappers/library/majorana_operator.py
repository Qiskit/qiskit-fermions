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

"""Conversion of an operator into a :class:`.MajoranaOperator`."""

from __future__ import annotations

from qiskit_fermions.operators import MajoranaOperator
from qiskit_fermions.protocols import SupportsMajoranaOperator


def majorana_operator(operator: SupportsMajoranaOperator) -> MajoranaOperator:
    """Converts an operator into a :class:`.MajoranaOperator`.

    This is a thin, type-agnostic wrapper around the :class:`.SupportsMajoranaOperator` protocol
    method, mirroring the free-function style of :func:`.fermion_operator`.

    Args:
        operator: the operator to convert, implementing :class:`.SupportsMajoranaOperator`.

    Returns:
        The operator converted into a :class:`.MajoranaOperator`.

    .. doctest::

        >>> from qiskit_fermions.mappers.library import majorana_operator
        >>> from qiskit_fermions.operators import FermionOperator
        >>> fer_op = FermionOperator.from_dict({((True, 0), (False, 0)): 1})
        >>> maj_op = majorana_operator(fer_op)
        >>> print(format(maj_op.normal_ordered().simplify()))
         5.000000e-1 +0.000000e0j * ()
          0.000000e0-5.000000e-1j * (γ'0 γ0)
    """
    return operator._majorana_operator_()
