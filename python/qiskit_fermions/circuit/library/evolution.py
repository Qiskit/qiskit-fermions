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

from qiskit_fermions.operators.protocol import OperatorTrait

from .. import FermionicGate


class Evolution(FermionicGate):
    """Implements the time-evolution of an operator."""

    def __init__(self, num_modes: int, operator: OperatorTrait, time: float = 1.0) -> None:
        """Initializes an Evolution gate.

        Args:
            num_modes: the number of fermionic modes on which this gate acts.
            operator: the operator under which to time-evolve the acted-upon fermionic modes.
            time: the evolution time.
        """
        self.operator = operator
        """The operator under which to time-evolve the acted-upon fermionic modes."""

        super().__init__(
            "Evolution", num_modes, [time], label=f"evolve({' '.join(str(operator).split())})"
        )

    def _define(self) -> None:
        from qiskit_fermions.circuit import FermionicCircuit

        definition = FermionicCircuit(self.num_modes)

        # when the operator being evolved has groups use those for the decomposition, otherwise
        # decompose into all individual terms
        iterator = (
            self.operator.iter_terms
            if self.operator.groups is None
            else self.operator.split_out_groups
        )

        for item in iterator():
            if isinstance(item, tuple):
                # iterating over terms rather than operator groups
                item = self.operator.__class__.from_terms([item])

            # reduce each operator to act only on the non-idle part of the register
            active = item.get_support()
            num_active = len(active)
            active_idx = iter(range(num_active))
            idle_idx = iter(range(num_active, self.num_modes))
            permutation = [
                next(active_idx) if idx in active else next(idle_idx)
                for idx in range(self.num_modes)
            ]
            relabeled = item.relabel_modes(permutation)

            definition.append(
                Evolution(num_active, relabeled, time=self.params[0]),
                [definition.fermions[idx] for idx in sorted(active)],
            )

        self._definition = definition._inner
