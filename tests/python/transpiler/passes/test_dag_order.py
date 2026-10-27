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

"""Tests for the order in which passes that rebuild a DAG visit its nodes.

A pass that substitutes a node by a sub-circuit leaves the order in which the DAG *stores* its nodes
out of sync with the dependencies between them. A pass that rebuilds its output node by node has to
iterate in circuit order, and not in storage order, or the factors of successive substitutions end
up interleaved: a well-formed circuit, but for a different unitary, because the gates of two
substitutions share qubits and do not commute.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from qiskit.passmanager import MultiStagePassManager
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.circuit.library.synthesis import FermionicLieTrotter
from qiskit_fermions.mappers.library import jordan_wigner
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.transpiler import (
    FermionicCircuitToDAG,
    FermionicPassManager,
    QuantumDAGToCircuit,
)
from qiskit_fermions.transpiler.passes import (
    F2FSynthesis,
    F2QSynthesis,
    MapperFnEvolutionSynthesis,
    QDriftTrotterization,
    RelabelModes,
    TrivialF2QLayout,
)

NUM_MODES = 3


def _hop_groups(*hops: tuple[int, int, float]) -> FermionOperator:
    """A Hermitian operator holding one conjugate-paired group per hop between two modes."""
    terms = {}
    for i, j, coeff in hops:
        terms[((True, i), (False, j))] = coeff
        terms[((True, j), (False, i))] = coeff
    operator = FermionOperator.from_dict(terms)
    # the operator does not preserve the order in which its terms were added, so a term's group (the
    # pair of modes it hops between) is derived from the terms as stored
    pairs = [frozenset(idx for _, idx in term) for term, _ in operator.iter_terms()]
    ids = {pair: group for group, pair in enumerate(sorted(set(pairs), key=sorted))}
    operator.groups = [ids[pair] for pair in pairs]
    return operator


def _evolve(operator: FermionOperator, time: float) -> np.ndarray:
    qubit_op = SparsePauliOp.from_sparse_observable(jordan_wigner(operator, NUM_MODES))
    return scipy.linalg.expm(-1j * time * qubit_op.to_matrix())


def _lie_trotter(operator: FermionOperator, time: float) -> np.ndarray:
    """The first-order product over the groups of ``operator``, in group order."""
    unitary = np.eye(2**NUM_MODES, dtype=complex)
    for group in sorted(set(operator.groups)):
        unitary = _evolve(operator.split_out_groups([group])[0], time) @ unitary
    return unitary


def _substituted_dag(gates: list[tuple[FermionOperator, float]]):
    """A DAG after a pass replaced each of its evolutions by a sub-circuit."""
    circuit = FermionicCircuit(NUM_MODES)
    for operator, time in gates:
        circuit.append(Evolution(NUM_MODES, operator, time=time), circuit.modes)
    dag = FermionicCircuitToDAG().run(circuit)
    F2FSynthesis({"Evolution": FermionicLieTrotter()}).run(dag)
    return dag


def _signature(dag, nodes) -> list[tuple[str, tuple[int, ...], float]]:
    """Name, modes and evolution time of each node, which identify a gate of the examples below."""
    return [
        (
            node.name,
            tuple(dag.find_bit(mode).index for mode in node.qargs),
            round(float(node.op.params[0]), 9) if node.op.params else 0.0,
        )
        for node in nodes
    ]


def _overlapping_dag():
    return _substituted_dag(
        [
            (_hop_groups((0, 1, 0.7), (1, 2, 0.3)), 0.6),
            (_hop_groups((1, 2, 0.4), (0, 1, 0.2)), 0.5),
            (_hop_groups((0, 1, 0.9), (1, 2, 0.5)), 0.8),
        ]
    )


def test_the_fixture_has_storage_out_of_sync_with_the_circuit():
    """If this fails, the tests below no longer exercise the bug they are named after."""
    dag = _overlapping_dag()
    assert _signature(dag, dag.op_nodes()) != _signature(dag, dag.topological_op_nodes())


def test_f2f_synthesis_visits_the_gates_in_circuit_order():
    """The pass substitutes nodes itself, so it must not iterate in storage order.

    A first pass splits each gate into plain (not atomic) factors, which leaves the storage of the
    DAG out of sync with its wires. A second pass, with a plugin that records the order in which it
    is handed the gates, then has to see them in circuit order.
    """

    class Split:
        def synthesize(self, gate):
            circuit = FermionicCircuit(gate.num_modes)
            for factor in gate.operator.split_out_groups():
                circuit.append(
                    Evolution(gate.num_modes, factor, time=gate.params[0]), circuit.modes
                )
            return circuit

    class Recorder:
        def __init__(self):
            self.seen = []

        def synthesize(self, gate):
            self.seen.append((round(float(gate.params[0]), 9), sorted(gate.operator.get_support())))
            return FermionicCircuit(gate.num_modes)

    circ = FermionicCircuit(NUM_MODES)
    for operator, time in (
        (_hop_groups((0, 1, 0.7), (1, 2, 0.3)), 0.6),
        (_hop_groups((1, 2, 0.4), (0, 1, 0.2)), 0.5),
        (_hop_groups((0, 1, 0.9), (1, 2, 0.5)), 0.8),
    ):
        circ.append(Evolution(NUM_MODES, operator, time=time), circ.modes)
    dag = FermionicCircuitToDAG().run(circ)
    F2FSynthesis({"Evolution": Split()}).run(dag)
    assert _signature(dag, dag.op_nodes()) != _signature(dag, dag.topological_op_nodes())

    recorder = Recorder()
    F2FSynthesis({"Evolution": recorder}).run(dag)

    assert [time for time, _ in recorder.seen] == [0.6, 0.6, 0.5, 0.5, 0.8, 0.8]


def test_synthesis_lowers_successive_substitutions_in_circuit_order():
    """The factors of one substituted gate must not be interleaved with those of the next."""
    first = _hop_groups((0, 1, 0.7), (1, 2, 0.3))
    second = _hop_groups((1, 2, 0.4), (0, 1, 0.2))
    third = _hop_groups((0, 1, 0.9), (1, 2, 0.5))
    times = (0.6, 0.5, 0.8)

    circ = FermionicCircuit(NUM_MODES)
    for operator, time in zip((first, second, third), times, strict=True):
        circ.append(Evolution(NUM_MODES, operator, time=time), circ.modes)

    synth = F2QSynthesis()
    synth.methods["Evolution"] = MapperFnEvolutionSynthesis(jordan_wigner)
    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        optimization=FermionicPassManager(F2FSynthesis({"Evolution": FermionicLieTrotter()})),
        layout=TrivialF2QLayout(),
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )

    lowered = Operator(pm.run(circ)).data

    expected = (
        _lie_trotter(third, times[2])
        @ _lie_trotter(second, times[1])
        @ _lie_trotter(first, times[0])
    )
    np.testing.assert_allclose(lowered, expected, atol=1e-10)


def test_relabel_modes_keeps_the_circuit_order_of_a_substituted_dag():
    """A permutation that changes nothing must leave the circuit exactly as it was."""
    dag = _overlapping_dag()
    before = _signature(dag, dag.topological_op_nodes())

    relabeled = RelabelModes(list(range(NUM_MODES))).run(dag)

    assert _signature(relabeled, relabeled.topological_op_nodes()) == before


def test_qdrift_keeps_the_circuit_order_of_a_substituted_dag():
    """With one group per gate every sample is deterministic, so only the order can differ.

    The sampled gates sit on the whole register, so a gate is identified by its evolution time
    instead: it is the time of the input gate times a constant, so the order of the times is the
    order of the circuit.
    """
    dag = _substituted_dag(
        [
            (_hop_groups((0, 1, 1.0)), 0.1),
            (_hop_groups((1, 2, 1.0)), 0.2),
            (_hop_groups((0, 1, 1.0)), 0.3),
        ]
    )
    assert _signature(dag, dag.op_nodes()) != _signature(dag, dag.topological_op_nodes())

    sampled = QDriftTrotterization(1, rng=0).run(dag)

    times = [float(node.op.params[0]) for node in sampled.topological_op_nodes()]
    assert times == sorted(times)
    assert len(set(times)) == len(times)
