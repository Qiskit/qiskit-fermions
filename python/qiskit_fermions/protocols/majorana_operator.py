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

"""A protocol to indicate MajoranaOperator conversion support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from qiskit_fermions.operators import MajoranaOperator


class SupportsMajoranaOperator(Protocol):
    """An equivalent of the :class:`.SupportsFermionOperator` protocol, targeting Majorana operators.

    See :func:`.majorana_operator` for the type-agnostic helper function dispatching to the method
    below.

    .. doctest::

        >>> from qiskit_fermions.operators import FermionOperator
        >>> fer_op = FermionOperator.from_dict({((True, 0), (False, 0)): 1})
        >>> maj_op = fer_op._majorana_operator_()
        >>> print(format(maj_op.normal_ordered().simplify()))
         5.000000e-1 +0.000000e0j * ()
          0.000000e0-5.000000e-1j * (γ'0 γ0)
    """

    def _majorana_operator_(self) -> MajoranaOperator:
        """Converts this operator into a :class:`.MajoranaOperator` instance."""
