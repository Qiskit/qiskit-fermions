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

"""A protocol to indicate FermionOperator conversion support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from qiskit_fermions.operators import FermionOperator


class SupportsFermionOperator(Protocol):
    """An equivalent of the :class:`ffsim.SupportsFermionOperator` protocol.

    See :func:`.fermion_operator` for the type-agnostic helper function dispatching to the method
    below.

    .. doctest::

        >>> from qiskit_fermions.operators import MajoranaOperator
        >>> maj_op = MajoranaOperator.from_dict({(0, 1): 1})
        >>> fer_op = maj_op._fermion_operator_()
        >>> print(format(fer_op.normal_ordered().simplify()))
          0.000000e0 +1.000000e0j * ()
          0.000000e0 -2.000000e0j * (+0 -0)
    """

    def _fermion_operator_(self) -> FermionOperator:
        """Converts this operator into a :class:`.FermionOperator` instance."""
