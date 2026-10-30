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

"""Tests for the MergeEvolutions optimization pass."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.circuit.library.synthesis import FermionicSuzukiTrotter
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.operators.fermion_action import ann, cre
from qiskit_fermions.transpiler import FermionicCircuitToDAG
from qiskit_fermions.transpiler.passes import F2FSynthesis, MergeEvolutions


def _number(mode: int, coeff: float = 1.0) -> FermionOperator:
    return FermionOperator.from_dict({(cre(mode), ann(mode)): coeff})


def _hopping() -> FermionOperator:
    return FermionOperator.from_dict({(cre(0), ann(1)): -1.0, (cre(1), ann(0)): -1.0})


def _merge(circ: FermionicCircuit, **kwargs):
    """Runs the merge pass on ``circ`` and returns the resulting DAG."""
    return MergeEvolutions(**kwargs).run(FermionicCircuitToDAG().run(circ))


def _times(dag) -> list[float]:
    return [node.op.params[0] for node in dag.topological_op_nodes()]


# --- positive cases -----------------------------------------------------------------------------


def test_merge_two_evolutions_sums_the_time():
    """Two neighboring evolutions under the same operator fuse into one with the summed time."""
    circ = FermionicCircuit(2)
    circ.append(Evolution(2, _hopping(), time=0.25), circ.modes)
    circ.append(Evolution(2, _hopping(), time=0.5), circ.modes)

    dag = _merge(circ)

    assert _times(dag) == pytest.approx([0.75])


def test_merge_run_of_three():
    circ = FermionicCircuit(2)
    for time in (0.1, 0.2, 0.3):
        circ.append(Evolution(2, _hopping(), time=time), circ.modes)

    assert _times(_merge(circ)) == pytest.approx([0.6])


def test_merge_keeps_groups_of_the_first_operator():
    """The merged gate carries the operator, and with it the groups, of the first gate."""
    operator = FermionOperator.from_terms_with_groups(
        [(((True, 0), (False, 1)), -1.0, 0), (((True, 1), (False, 0)), -1.0, 0)]
    )
    circ = FermionicCircuit(2)
    circ.append(Evolution(2, operator, time=0.1), circ.modes)
    circ.append(Evolution(2, operator, time=0.2), circ.modes)

    (node,) = _merge(circ).topological_op_nodes()

    assert node.op.operator.has_groups()


def test_merge_within_tolerance():
    circ = FermionicCircuit(2)
    circ.append(Evolution(2, _number(0, 1.0), time=0.1), circ.modes)
    circ.append(Evolution(2, _number(0, 1.0 + 1e-9), time=0.2), circ.modes)

    assert len(_times(_merge(circ, atol=1e-6))) == 1
    assert len(_times(_merge(circ, atol=1e-12))) == 2


def test_merge_across_gate_boundary_of_repeated_circuit():
    """The seam between two repetitions of a palindromic circuit is fused."""
    palindrome = FermionicCircuit(2)
    palindrome.append(Evolution(2, _number(0), time=0.25), palindrome.modes)
    palindrome.append(Evolution(2, _hopping(), time=1.0), palindrome.modes)
    palindrome.append(Evolution(2, _number(0), time=0.25), palindrome.modes)

    circ = FermionicCircuit(2)
    for _ in range(2):
        circ.compose(palindrome, inplace=True)

    assert _times(_merge(circ)) == pytest.approx([0.25, 1.0, 0.5, 1.0, 0.25])


def test_merge_removes_the_seams_of_an_unmerged_suzuki_schedule():
    """The pass reproduces what ``merge_adjacent`` does inside the synthesis."""
    operator = FermionOperator.from_terms_with_groups(
        [
            (((True, 0), (False, 1)), -1.0, 0),
            (((True, 1), (False, 0)), -1.0, 0),
            (((True, 1), (False, 2)), -1.0, 1),
            (((True, 2), (False, 1)), -1.0, 1),
        ]
    )
    circ = FermionicCircuit(3)
    circ.append(Evolution(3, operator, time=0.7), circ.modes)

    def expanded(merge_adjacent: bool):
        synthesis = FermionicSuzukiTrotter(order=2, reps=3, merge_adjacent=merge_adjacent)
        dag = FermionicCircuitToDAG().run(circ)
        return F2FSynthesis({"Evolution": synthesis}).run(dag)

    unmerged = expanded(False)
    reference = expanded(True)
    merged = MergeEvolutions().run(unmerged)

    assert len(merged.op_nodes()) == len(reference.op_nodes())
    assert sum(_times(merged)) == pytest.approx(sum(_times(reference)))


# --- negative cases -----------------------------------------------------------------------------


def test_no_merge_single_evolution():
    circ = FermionicCircuit(2)
    circ.append(Evolution(2, _hopping(), time=0.3), circ.modes)

    assert _times(_merge(circ)) == pytest.approx([0.3])


def test_no_merge_of_different_operators():
    circ = FermionicCircuit(2)
    circ.append(Evolution(2, _number(0), time=0.1), circ.modes)
    circ.append(Evolution(2, _number(1), time=0.2), circ.modes)

    assert _times(_merge(circ)) == pytest.approx([0.1, 0.2])


def test_no_merge_when_not_adjacent():
    """An operation between two evolutions on shared wires blocks the merge."""
    circ = FermionicCircuit(2)
    circ.append(Evolution(2, _number(0), time=0.1), circ.modes)
    circ.append(Evolution(2, _number(1), time=0.2), circ.modes)
    circ.append(Evolution(2, _number(0), time=0.3), circ.modes)

    assert _times(_merge(circ)) == pytest.approx([0.1, 0.2, 0.3])


def test_no_merge_across_barrier():
    circ = FermionicCircuit(2)
    circ.append(Evolution(2, _hopping(), time=0.1), circ.modes)
    circ.barrier()
    circ.append(Evolution(2, _hopping(), time=0.2), circ.modes)

    names = [node.op.name for node in _merge(circ).topological_op_nodes()]

    assert names == ["Evolution", "barrier", "Evolution"]


def test_no_merge_across_wire_order():
    """The operator's labels are relative to the wires, so reordered wires are different gates."""
    circ = FermionicCircuit(2)
    circ.append(Evolution(2, _number(0), time=0.1), circ.modes)
    circ.append(Evolution(2, _number(0), time=0.2), circ.modes[::-1])

    assert _times(_merge(circ)) == pytest.approx([0.1, 0.2])


def test_no_merge_of_atomic_with_non_atomic():
    """Merging must never turn an atomic factor into a splittable one, or the reverse."""
    circ = FermionicCircuit(2)
    circ.append(Evolution(2, _hopping(), time=0.1, atomic=True), circ.modes)
    circ.append(Evolution(2, _hopping(), time=0.2), circ.modes)

    assert _times(_merge(circ)) == pytest.approx([0.1, 0.2])


def test_merge_atomic_gates_stays_atomic():
    circ = FermionicCircuit(2)
    circ.append(Evolution(2, _hopping(), time=0.1, atomic=True), circ.modes)
    circ.append(Evolution(2, _hopping(), time=0.2, atomic=True), circ.modes)

    (node,) = _merge(circ).topological_op_nodes()

    assert node.op.atomic
    assert node.op.params[0] == pytest.approx(0.3)


def test_no_merge_of_different_synthesis():
    """Gates that decompose differently must not be collapsed into one that picks a single way."""
    circ = FermionicCircuit(2)
    circ.append(
        Evolution(2, _hopping(), time=0.1, synthesis=FermionicSuzukiTrotter(order=2)), circ.modes
    )
    circ.append(
        Evolution(2, _hopping(), time=0.2, synthesis=FermionicSuzukiTrotter(order=4)), circ.modes
    )

    assert _times(_merge(circ)) == pytest.approx([0.1, 0.2])


class _WeightedSynthesis:
    """A synthesis method whose settings include an array, which ``==`` cannot compare."""

    def __init__(self, weights: list[float]) -> None:
        self.weights = np.asarray(weights, dtype=float)

    def synthesize(self, gate):  # pragma: no cover - never called, only compared
        raise NotImplementedError


def test_no_merge_of_different_array_settings():
    """Array-valued settings are compared by value: different arrays must not merge, or raise."""
    circ = FermionicCircuit(2)
    for time, weights in ((0.1, [1.0, 2.0]), (0.2, [1.0, 3.0])):
        synthesis = _WeightedSynthesis(weights)
        circ.append(Evolution(2, _hopping(), time=time, synthesis=synthesis), circ.modes)

    assert _times(_merge(circ)) == pytest.approx([0.1, 0.2])


def test_merge_of_equal_array_settings():
    """Equal arrays held in distinct objects describe the same method, so the gates merge."""
    circ = FermionicCircuit(2)
    for time in (0.1, 0.2):
        synthesis = _WeightedSynthesis([1.0, 2.0])
        circ.append(Evolution(2, _hopping(), time=time, synthesis=synthesis), circ.modes)

    assert _times(_merge(circ)) == pytest.approx([0.3])


def test_merge_stretches_split_by_a_different_operator():
    """A run is cut into stretches: ``A A B B`` becomes ``A B``, not one gate."""
    circ = FermionicCircuit(2)
    circ.append(Evolution(2, _number(0), time=0.1), circ.modes)
    circ.append(Evolution(2, _number(0), time=0.2), circ.modes)
    circ.append(Evolution(2, _number(1), time=0.3), circ.modes)
    circ.append(Evolution(2, _number(1), time=0.4), circ.modes)

    assert _times(_merge(circ)) == pytest.approx([0.3, 0.7])
