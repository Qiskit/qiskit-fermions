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
import pytest
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


def test_pass_expands_the_evolution():
    """By default the pass applies the synthesis, so no separate expansion step is needed."""
    circuit = _circuit()

    out = FermionicPassManager(FermionicTrotterization(FermionicSuzukiTrotter(order=2))).run(
        circuit
    )

    # two groups: first-order emits one factor each, the order-2 palindrome emits 2*2-1 = 3
    assert dict(circuit.count_ops()) == {"Evolution": 1}
    assert dict(out.count_ops()) == {"Evolution": 3}


def test_pass_expansion_is_idempotent():
    """The emitted factors are atomic, so running the pass again must not split them further."""
    once = FermionicPassManager(FermionicTrotterization(FermionicSuzukiTrotter(order=2))).run(
        _circuit()
    )
    thrice = FermionicPassManager(
        [FermionicTrotterization(FermionicSuzukiTrotter(order=2))] * 3
    ).run(_circuit())

    assert dict(once.count_ops()) == dict(thrice.count_ops()) == {"Evolution": 3}
    assert all(evolution.atomic for evolution in _evolutions(thrice))


def test_pass_leaves_atomic_gates_alone():
    """An atomic gate is a terminal factor: it carries no definition and must not be expanded.

    This is what keeps the pass a fixed point over :class:`.QDriftTrotterization`'s output, whose
    sampled gates are all atomic.
    """
    synthesis = FermionicSuzukiTrotter(order=2)
    circuit = FermionicCircuit(NUM_MODES)
    circuit.append(Evolution(NUM_MODES, _hopping_chain(), time=0.5, atomic=True), circuit.modes)

    out = FermionicPassManager(FermionicTrotterization(synthesis)).run(circuit)

    (evolution,) = _evolutions(out)
    assert evolution.atomic
    # the method is still selected, it is simply never consulted for an atomic gate
    assert evolution.synthesis is synthesis


def test_apply_false_only_selects_the_synthesis_method():
    """With ``apply=False`` the gate stays whole and merely carries the method."""
    synthesis = FermionicSuzukiTrotter(order=2)

    out = FermionicPassManager(FermionicTrotterization(synthesis, apply=False)).run(_circuit())

    (evolution,) = _evolutions(out)
    assert evolution.synthesis is synthesis
    assert dict(out.count_ops()) == {"Evolution": 1}
    # the selection still governs a later expansion
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
        "Evolution": 3,
    }


def test_filter_selects_a_subset(subtests):
    synthesis = FermionicSuzukiTrotter(order=2)

    with subtests.test("rejecting everything"):
        out = FermionicPassManager(
            FermionicTrotterization(synthesis, filter=lambda _node: False)
        ).run(_circuit())
        # a rejected node is left untouched, so it is neither expanded nor re-tagged
        (evolution,) = _evolutions(out)
        assert isinstance(evolution.synthesis, FermionicLieTrotter)

    with subtests.test("accepting everything"):
        out = FermionicPassManager(
            FermionicTrotterization(synthesis, filter=lambda _node: True)
        ).run(_circuit())
        assert dict(out.count_ops()) == {"Evolution": 3}


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
    """With ``apply=False`` the gate is re-tagged, so its operator and time must survive intact."""
    operator = _hopping_chain()
    out = FermionicPassManager(
        FermionicTrotterization(FermionicSuzukiTrotter(order=2), apply=False)
    ).run(_circuit())

    (evolution,) = _evolutions(out)
    assert evolution.operator.equiv(operator, 1e-12)
    assert evolution.params[0] == 0.5
    assert evolution.num_modes == NUM_MODES


def test_expansion_preserves_the_total_evolution_time():
    """Every group must end up evolved for the gate's full time, however the formula splits it."""
    out = FermionicPassManager(FermionicTrotterization(FermionicSuzukiTrotter(order=2))).run(
        _circuit()
    )

    # The order-2 palindrome is A(t/2) B(t) A(t/2) for two groups: the outer group is applied twice
    # at half the time and the inner one once at the full time, so each group accumulates exactly
    # `time`. Each factor is narrowed onto its support, so its operator carries gate-local mode
    # indices and the modes it acts on are what tell the two groups apart.
    times: dict[tuple[int, ...], float] = {}
    for instruction in out._inner.data:
        modes = tuple(out._inner.find_bit(mode).index for mode in instruction.qubits)
        times[modes] = times.get(modes, 0.0) + instruction.operation.params[0]

    assert times.keys() == {(0, 1), (1, 2)}
    assert all(total == pytest.approx(0.5) for total in times.values())


def test_expansion_composes_with_relabel_modes():
    """Expanding no longer constrains the pass order: RelabelModes works on the factors too.

    The factors a synthesis method emits are narrowed onto their support, so their operators carry
    gate-local mode indices. :class:`.RelabelModes` rewires by permuting ``qargs``, which is
    independent of that, and its automatic model translates through ``qargs`` before gathering.
    """
    out = FermionicPassManager(
        [FermionicTrotterization(FermionicSuzukiTrotter(order=2)), RelabelModes([2, 1, 0])]
    ).run(_circuit())

    assert dict(out.count_ops()) == {"Evolution": 3}
    assert out.metadata["permutation"] == [2, 1, 0]
