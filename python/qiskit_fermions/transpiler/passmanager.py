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

"""The passmanager interface for fermionic circuits."""

from qiskit.passmanager import BasePassManager

from qiskit_fermions.circuit import FermionicCircuit, FermionicDAGCircuit

from .converters import FermionicCircuitToDAG, FermionicDAGToCircuit


class FermionicPassManager(BasePassManager[FermionicDAGCircuit]):
    """A transpiler pass manager converting one :class:`.FermionicCircuit` to another.

    The internal representation (IR) of this pass manager is :class:`.FermionicDAGCircuit`, but the
    input and output type are :class:`.FermionicCircuit`.
    """

    def _passmanager_frontend(
        self, input_program: FermionicCircuit, **kwargs
    ) -> FermionicDAGCircuit:
        return FermionicCircuitToDAG().run(input_program)

    def _passmanager_backend(
        self, passmanager_ir: FermionicDAGCircuit, in_program: FermionicCircuit, **kwargs
    ) -> FermionicCircuit:
        return FermionicDAGToCircuit().run(passmanager_ir)
