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

"""Fermionic Trotterization pass tests."""

from __future__ import annotations

import numpy as np
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution, InitializeModes, OrbitalRotation
from qiskit_fermions.circuit.library.synthesis import FermionicLieTrotter, FermionicSuzukiTrotter
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.transpiler.passes import FermionicTrotterization, RelabelModes
from qiskit_fermions.transpiler.passmanager import FermionicPassManager

NUM_MODES = 3


def _hopping_chain() -> FermionOperator:
    """Returns a Hermitian hopping chain whose two groups do not commute with each other."""
    return FermionOperator.from_terms_with_groups(
        [
            (((True, 0), (False, 1)), -1.0, 0),
            (((True, 1), (False, 0)), -1.0, 0),
            (((True, 1), (False, 2)), -0.8, 1),
            (((True, 2), (False, 1)), -0.8, 1),
        ]
    )


def _circuit(*extra_gates) -> FermionicCircuit:
    circuit = FermionicCircuit(NUM_MODES)
    for gate in extra_gates:
        circuit.append(gate, circuit.modes)
    circuit.append(Evolution(NUM_MODES, _hopping_chain(), time=0.5), circuit.modes)
    return circuit


def _evolutions(circuit: FermionicCircuit) -> list[Evolution]:
    return [
        instruction.operation
        for instruction in circuit._inner.data
        if isinstance(instruction.operation, Evolution)
    ]


def test_pass_selects_the_synthesis_method():
    synthesis = FermionicSuzukiTrotter(order=2)

    out = FermionicPassManager(FermionicTrotterization(synthesis)).run(_circuit())

    (evolution,) = _evolutions(out)
    assert evolution.synthesis is synthesis


def test_pass_changes_the_resulting_decomposition():
    """Selecting a second-order formula must actually change what the gate decomposes into."""
    circuit = _circuit()
    default = dict(circuit.decompose().count_ops())

    out = FermionicPassManager(FermionicTrotterization(FermionicSuzukiTrotter(order=2))).run(
        circuit
    )

    # two groups: first-order emits one factor each, the order-2 palindrome emits 2*2-1 = 3
    assert default == {"Evolution": 2}
    assert dict(out.decompose().count_ops()) == {"Evolution": 3}


def test_pass_does_not_mutate_the_input_gate():
    """A gate instance may be shared between circuits, so the pass must not modify it in place."""
    gate = Evolution(NUM_MODES, _hopping_chain(), time=0.5)
    circuit = FermionicCircuit(NUM_MODES)
    circuit.append(gate, circuit.modes)

    FermionicPassManager(FermionicTrotterization(FermionicSuzukiTrotter(order=2))).run(circuit)

    assert isinstance(gate.synthesis, FermionicLieTrotter)


def test_pass_leaves_other_gates_untouched():
    occupation = np.array([True, False, False])
    rotation = np.eye(NUM_MODES, dtype=complex)
    circuit = _circuit(InitializeModes(occupation), OrbitalRotation(rotation))

    out = FermionicPassManager(FermionicTrotterization(FermionicSuzukiTrotter(order=2))).run(
        circuit
    )

    assert dict(out.count_ops()) == {
        "InitializeModes": 1,
        "OrbitalRotation": 1,
        "Evolution": 1,
    }


def test_filter_selects_a_subset(subtests):
    synthesis = FermionicSuzukiTrotter(order=2)

    with subtests.test("rejecting everything"):
        out = FermionicPassManager(
            FermionicTrotterization(synthesis, filter=lambda _node: False)
        ).run(_circuit())
        (evolution,) = _evolutions(out)
        assert isinstance(evolution.synthesis, FermionicLieTrotter)

    with subtests.test("accepting everything"):
        out = FermionicPassManager(
            FermionicTrotterization(synthesis, filter=lambda _node: True)
        ).run(_circuit())
        (evolution,) = _evolutions(out)
        assert evolution.synthesis is synthesis


def test_filter_receives_the_dag_node():
    """The predicate is handed the node, so it can inspect the gate it is deciding about."""
    seen = []

    def record(node):
        seen.append(node.op)
        return True

    FermionicPassManager(
        FermionicTrotterization(FermionicSuzukiTrotter(order=2), filter=record)
    ).run(_circuit())

    assert len(seen) == 1
    assert isinstance(seen[0], Evolution)


def test_pass_preserves_the_operator_and_time():
    operator = _hopping_chain()
    out = FermionicPassManager(FermionicTrotterization(FermionicSuzukiTrotter(order=2))).run(
        _circuit()
    )

    (evolution,) = _evolutions(out)
    assert evolution.operator.equiv(operator, 1e-12)
    assert evolution.params[0] == 0.5
    assert evolution.num_modes == NUM_MODES


def test_operators_stay_intact_for_downstream_passes():
    """The evolution stays one node, so a later pass still sees a whole operator to work with.

    This is the reason the pass selects a synthesis method instead of expanding the gate: expanding
    would hand :class:`.RelabelModes` one fragment per factor rather than the full operator.
    """
    circuit = _circuit()
    pass_manager = FermionicPassManager(
        [FermionicTrotterization(FermionicSuzukiTrotter(order=2)), RelabelModes([2, 1, 0])]
    )

    out = pass_manager.run(circuit)

    (evolution,) = _evolutions(out)
    assert len(list(evolution.operator.iter_terms())) == 4
