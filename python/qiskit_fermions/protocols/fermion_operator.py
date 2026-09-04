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
    """This package's operator-conversion contract, converting an operator to a :class:`.FermionOperator`.

    See :func:`.fermion_operator` for the type-agnostic helper function dispatching to the method
    below.

    .. caution::
       Despite the shared method name, this is *not* the same contract as
       :class:`ffsim.SupportsFermionOperator`: this one returns a
       :class:`~qiskit_fermions.operators.FermionOperator`, whereas ffsim's returns an
       :class:`ffsim.FermionOperator`. The two types are unrelated. Since
       :func:`ffsim.fermion_operator` dispatches purely on the method name, calling it on an operator
       of this package returns *this* package's type rather than ffsim's; use
       :func:`.fermion_operator` when that is what you mean.

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
