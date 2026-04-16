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

"""Hamiltonian evolution gate."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import FermionGate

if TYPE_CHECKING:
    from qiskit_fermions._lib.operators.fermion_operator import FermionOperator
    from qiskit_fermions._lib.operators.majorana_operator import MajoranaOperator


class Evolution(FermionGate):
    """Implements the time-evolution of an operator."""

    def __init__(
        self, num_fermions: int, operator: FermionOperator | MajoranaOperator, time: float = 1.0
    ) -> None:
        """Initializes an Evolution gate.

        Args:
            num_fermions: the number of fermionic modes on which this gate acts.
            operator: the operator under which to time-evolve the acted-upon fermionic modes.
            time: the evolution time.
        """
        self.operator = operator
        """The operator under which to time-evolve the acted-upon fermionic modes."""

        super().__init__("Evolution", num_fermions, [time])
