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

from ...operators import FermionOperator, MajoranaOperator
from .. import FermionGate


class Evolution(FermionGate):
    """TODO."""

    def __init__(
        self, num_fermions: int, operator: FermionOperator | MajoranaOperator, time: float = 1.0
    ) -> None:
        """TODO."""
        self.operator = operator
        super().__init__("Evolution", num_fermions, [time])
