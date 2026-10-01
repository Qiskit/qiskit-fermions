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

"""FermionicCircuit tests."""

from __future__ import annotations

import pickle

import pytest
from qiskit.circuit import Barrier, CircuitInstruction
from qiskit.circuit.library import XGate
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.operators import FermionOperator, ann, cre
from qiskit_fermions.transpiler import FermionicCircuitToDAG


def test_invalid_gate():
    circ = FermionicCircuit(1)
    with pytest.raises(ValueError):
        circ.append(XGate(), circ.modes)


def _evolution_circuit() -> FermionicCircuit:
    """Builds a small circuit holding an ``Evolution`` gate over a ``FermionOperator``."""
    num_modes = 4
    hamil = FermionOperator.from_dict(
        {(cre(0), ann(2)): 2.0, (cre(2), ann(0)): 2.0, (cre(1), ann(3)): -2.0}
    )
    circ = FermionicCircuit(num_modes)
    circ.append(Evolution(num_modes, hamil, time=1.5), circ.modes)
    circ.barrier()
    return circ


def _barrier_nodes(circ: FermionicCircuit) -> list[CircuitInstruction]:
    """Returns every barrier instruction held by ``circ``."""
    return [instr for instr in circ._inner.data if isinstance(instr.operation, Barrier)]


def test_barrier_all_modes():
    """A barrier without arguments spans every mode of the circuit."""
    circ = FermionicCircuit(4)
    circ.barrier()

    assert circ.count_ops() == {"barrier": 1}
    (barrier,) = _barrier_nodes(circ)
    assert list(barrier.qubits) == circ.modes


def test_barrier_subset_of_modes():
    """A barrier can be restricted to a subset of the modes."""
    circ = FermionicCircuit(4)
    circ.barrier(circ.register[1], circ.register[2])

    (barrier,) = _barrier_nodes(circ)
    assert list(barrier.qubits) == [circ.register[1], circ.register[2]]
    assert barrier.operation.num_qubits == 2


def test_barrier_label():
    """A barrier carries the label it was given."""
    circ = FermionicCircuit(2)
    circ.barrier(label="sync")

    (barrier,) = _barrier_nodes(circ)
    assert barrier.operation.label == "sync"


def test_barrier_survives_decompose():
    """Decomposing a circuit leaves its barriers in place."""
    circ = _evolution_circuit()
    decomposed = circ.decompose()

    assert circ.count_ops()["barrier"] == 1
    assert decomposed.count_ops()["barrier"] == 1


def test_pickle():
    """Regression test for https://github.com/Qiskit/qiskit-fermions/issues/225.

    A ``FermionicCircuit`` holding an ``Evolution`` gate failed to pickle because the native
    operator it wraps carried no pickle protocol, and separately because its ``__module__``
    was not resolvable via ``importlib`` (see :mod:`qiskit_fermions`'s ``sys.modules`` aliasing).
    """
    circ = _evolution_circuit()
    reconstructed = pickle.loads(pickle.dumps(circ))

    assert reconstructed.modes == circ.modes
    assert reconstructed.count_ops() == circ.count_ops() == {"Evolution": 1, "barrier": 1}


def test_pickle_dag():
    """Same as :func:`test_pickle`, but for the ``FermionicDAGCircuit`` conversion."""
    dag = FermionicCircuitToDAG().run(_evolution_circuit())
    reconstructed = pickle.loads(pickle.dumps(dag))

    original_nodes = dag.named_nodes("Evolution")
    reconstructed_nodes = reconstructed.named_nodes("Evolution")
    assert len(original_nodes) == len(reconstructed_nodes) == 1
    # the barrier the circuit also carries has to survive the round trip alongside the gate
    assert len(dag.op_nodes()) == len(reconstructed.op_nodes()) == 2

    original_op = original_nodes[0].op.operator
    reconstructed_op = reconstructed_nodes[0].op.operator
    assert original_op.equiv(reconstructed_op)
