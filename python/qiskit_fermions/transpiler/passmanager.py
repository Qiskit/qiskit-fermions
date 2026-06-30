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

"""PassManager interfaces for FermionicCircuits."""

from typing import TypeAlias

from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.passmanager import BasePassManager, GenericPass

from qiskit_fermions.circuit import FermionicCircuit, FermionicDAGCircuit

FermionicDAGCircuitPass: TypeAlias = GenericPass[FermionicDAGCircuit, FermionicDAGCircuit]


class FermionicCircuitToDAG(GenericPass[FermionicCircuit, FermionicDAGCircuit]):
    """TODO."""

    def run(self, passmanager_ir: FermionicCircuit) -> FermionicDAGCircuit:
        """TODO."""
        return circuit_to_dag(passmanager_ir._inner, copy_operations=True)


class FermionicDAGToCircuit(GenericPass[FermionicDAGCircuit, FermionicCircuit]):
    """TODO."""

    def run(self, passmanager_ir: FermionicDAGCircuit) -> FermionicCircuit:
        """TODO."""
        out = FermionicCircuit(passmanager_ir.num_qubits())
        out._inner = dag_to_circuit(passmanager_ir, copy_operations=False)
        return out


class FermionicPassManager(BasePassManager[FermionicDAGCircuit]):
    """A transpiler pass manager converting one :class:`.FermionicCircuit` to another.

    .. note::
       Qiskit is currently working on native support of transpiler pipelines involving more than a
       single intermediate representation. Once that gets more formalized, the implementation here
       will be aligned with the resulting interfaces. See also `this tracking issue
       <https://github.com/Qiskit/qiskit/issues/16115>`_.
    """

    def _passmanager_frontend(
        self, input_program: FermionicCircuit, **kwargs
    ) -> FermionicDAGCircuit:
        return FermionicCircuitToDAG().run(input_program)

    def _passmanager_backend(
        self, passmanager_ir: FermionicDAGCircuit, in_program: FermionicCircuit, **kwargs
    ) -> FermionicCircuit:
        return FermionicDAGToCircuit().run(passmanager_ir)
