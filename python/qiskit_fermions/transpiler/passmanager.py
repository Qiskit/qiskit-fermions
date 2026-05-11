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

from collections.abc import Callable, Iterable

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager import BasePassManager
from qiskit.transpiler import StagedPassManager

from qiskit_fermions.circuit import FermionicCircuit


class FermionicPassManager(BasePassManager):
    """A transpiler pass manager converting one :class:`.FermionicCircuit` to another.

    .. note::
       Qiskit is currently working on native support of transpiler pipelines involving more than a
       single intermediate representation. Once that gets more formalized, the implementation here
       will be aligned with the resulting interfaces. See also `this tracking issue
       <https://github.com/Qiskit/qiskit/issues/16115>`_.
    """

    def _passmanager_frontend(self, input_program: FermionicCircuit, **kwargs) -> DAGCircuit:
        return circuit_to_dag(input_program._inner, copy_operations=True)

    def _passmanager_backend(
        self, passmanager_ir: DAGCircuit, in_program: FermionicCircuit, **kwargs
    ) -> FermionicCircuit:
        out = FermionicCircuit(passmanager_ir.num_qubits())
        out._inner = dag_to_circuit(passmanager_ir, copy_operations=False)
        return out


class FermionicToQubitConverter(BasePassManager):
    """A transpiler pass manager converting a :class:`.FermionicCircuit` to a :class:`~qiskit.circuit.QuantumCircuit`.

    .. note::
       Qiskit is currently working on native support of transpiler pipelines involving more than a
       single intermediate representation. Once that gets more formalized, the implementation here
       will be aligned with the resulting interfaces. See also `this tracking issue
       <https://github.com/Qiskit/qiskit/issues/16115>`_.
    """

    def _passmanager_frontend(self, input_program: FermionicCircuit, **kwargs) -> DAGCircuit:
        return circuit_to_dag(input_program._inner, copy_operations=True)

    def _passmanager_backend(
        self, passmanager_ir: DAGCircuit, in_program: FermionicCircuit, **kwargs
    ) -> QuantumCircuit:
        return dag_to_circuit(passmanager_ir, copy_operations=False)


class FermionicStagedPassManager(StagedPassManager):
    """The staged fermion-to-qubit transpilation pipeline.

    This

    .. note::
       Qiskit is currently working on native support of transpiler pipelines involving more than a
       single intermediate representation. Once that gets more formalized, the implementation here
       will be aligned with the resulting interfaces. See also `this tracking issue
       <https://github.com/Qiskit/qiskit/issues/16115>`_.
    """

    def _passmanager_frontend(self, input_program: FermionicCircuit, **kwargs) -> DAGCircuit:
        return circuit_to_dag(input_program._inner, copy_operations=True)

    def __init__(self, stages: Iterable[str] | None = None, **kwargs) -> None:  # noqa: D107
        stages = stages or [
            "optimization",
            "layout",
            "synthesis",
            "quantum",
        ]
        super().__init__(stages, **kwargs)

    def run(  # noqa: D102
        self,
        in_programs: FermionicCircuit | list[FermionicCircuit],
        callback: Callable | None = None,
        num_processes: int | None = None,
        *,
        property_set: dict[str, object] | None = None,
        **kwargs,
    ) -> QuantumCircuit:
        self._update_passmanager()
        return super().run(
            in_programs,
            callback,
            num_processes=num_processes,
            property_set=property_set,
            **kwargs,
        )
