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

"""Tests for the MergeOrbitalRotations optimization pass."""

from __future__ import annotations

import numpy as np
from qiskit.passmanager import MultiStagePassManager
from qiskit.quantum_info import Statevector
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import (
    Evolution,
    InitializeModes,
    OrbitalRotation,
)
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.operators.fermion_action import ann, cre
from qiskit_fermions.transpiler import FermionicCircuitToDAG, QuantumDAGToCircuit
from qiskit_fermions.transpiler.passes import (
    F2QSynthesis,
    GivensDecompositionOrbitalRotationSynthesis,
    MergeOrbitalRotations,
    TrivialF2QLayout,
    TrivialOccupationInitializeModesSynthesis,
)

from ...utils import random_unitary


def _nodes(dag) -> list[tuple[str, list[int]]]:
    """Returns the ``(op name, sorted global mode indices)`` of a DAG in topological order."""
    return [
        (node.op.name, sorted(dag.find_bit(qubit).index for qubit in node.qargs))
        for node in dag.topological_op_nodes()
    ]


def _merge(circ: FermionicCircuit):
    """Runs the merge pass on ``circ`` and returns the resulting DAG."""
    dag = FermionicCircuitToDAG().run(circ)
    return MergeOrbitalRotations().run(dag)


# --- positive cases: consecutive same-mode rotations merge --------------------------------------


def test_merge_two_rotations():
    """Two consecutive rotations on the same modes fuse into one gate."""
    circ = FermionicCircuit(3)
    circ.append(OrbitalRotation(random_unitary(3, seed=1)), circ.modes)
    circ.append(OrbitalRotation(random_unitary(3, seed=2)), circ.modes)

    assert _nodes(_merge(circ)) == [("OrbitalRotation", [0, 1, 2])]


def test_merge_three_rotations():
    """A run of three consecutive rotations collapses to a single gate."""
    circ = FermionicCircuit(2)
    for seed in (1, 2, 3):
        circ.append(OrbitalRotation(random_unitary(2, seed=seed)), circ.modes)

    assert _nodes(_merge(circ)) == [("OrbitalRotation", [0, 1])]


def test_merged_unitary_is_product_in_circuit_order():
    """The merged rotation_unitary is the run's unitaries multiplied in circuit order."""
    u1 = random_unitary(3, seed=1)
    u2 = random_unitary(3, seed=2)
    circ = FermionicCircuit(3)
    circ.append(OrbitalRotation(u1), circ.modes)
    circ.append(OrbitalRotation(u2), circ.modes)

    (node,) = _merge(circ).topological_op_nodes()
    # later rotation (u2) multiplies from the left
    np.testing.assert_allclose(node.op.rotation_unitary, u2 @ u1, atol=1e-12)


def test_merge_preserves_state():
    """Merging a rotation run yields the same state as the hand-composed single rotation.

    The merged circuit (one rotation carrying ``u2 @ u1``) must be equivalent under simulation to
    the unmerged circuit (``u1`` then ``u2``). Both are compared as fully synthesized qubit circuits
    to exercise the actual lowering path.
    """
    u1 = random_unitary(3, seed=1)
    u2 = random_unitary(3, seed=2)
    circ = FermionicCircuit(3)
    circ.append(InitializeModes([1, 1, 0]), circ.modes)
    circ.append(OrbitalRotation(u1), circ.modes)
    circ.append(OrbitalRotation(u2), circ.modes)

    def synth() -> F2QSynthesis:
        s = F2QSynthesis()
        s.methods["InitializeModes"] = TrivialOccupationInitializeModesSynthesis()
        s.methods["OrbitalRotation"] = GivensDecompositionOrbitalRotationSynthesis()
        return s

    reference = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=synth(),
        output=QuantumDAGToCircuit(),
    ).run(circ)
    merged = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        optimization=MergeOrbitalRotations(),
        layout=TrivialF2QLayout(),
        synthesis=synth(),
        output=QuantumDAGToCircuit(),
    ).run(circ)

    sv_ref = Statevector(reference).data
    sv_merged = Statevector(merged).data
    k = int(np.argmax(np.abs(sv_merged)))
    phase = sv_ref[k] / sv_merged[k]
    np.testing.assert_allclose(sv_ref, phase * sv_merged, atol=1e-10)


# --- negative cases: unmergeable shapes are left untouched --------------------------------------


def test_no_merge_single_rotation():
    """A lone rotation is left untouched."""
    circ = FermionicCircuit(2)
    circ.append(OrbitalRotation(random_unitary(2, seed=1)), circ.modes)

    assert _nodes(_merge(circ)) == [("OrbitalRotation", [0, 1])]


def test_no_merge_when_not_adjacent():
    """An operation between two rotations on shared wires blocks the merge."""
    circ = FermionicCircuit(2)
    circ.append(OrbitalRotation(random_unitary(2, seed=1)), circ.modes)
    number_op = FermionOperator.from_dict({(cre(0), ann(0)): 1.0})
    circ.append(Evolution(2, number_op, time=0.5), circ.modes)
    circ.append(OrbitalRotation(random_unitary(2, seed=2)), circ.modes)

    assert _nodes(_merge(circ)) == [
        ("OrbitalRotation", [0, 1]),
        ("Evolution", [0, 1]),
        ("OrbitalRotation", [0, 1]),
    ]


def test_no_merge_across_different_mode_sets():
    """Rotations on different mode sets do not merge into each other."""
    circ = FermionicCircuit(3)
    circ.append(OrbitalRotation(random_unitary(2, seed=1)), circ.modes[:2])
    circ.append(OrbitalRotation(random_unitary(2, seed=2)), circ.modes[1:])

    assert _nodes(_merge(circ)) == [
        ("OrbitalRotation", [0, 1]),
        ("OrbitalRotation", [1, 2]),
    ]


def test_merge_per_half_independently():
    """Consecutive rotations on disjoint spin halves each merge within their own half."""
    circ = FermionicCircuit(4)
    circ.append(OrbitalRotation(random_unitary(2, seed=1)), circ.modes[:2])
    circ.append(OrbitalRotation(random_unitary(2, seed=2)), circ.modes[:2])
    circ.append(OrbitalRotation(random_unitary(2, seed=3)), circ.modes[2:])
    circ.append(OrbitalRotation(random_unitary(2, seed=4)), circ.modes[2:])

    assert sorted(_nodes(_merge(circ))) == [
        ("OrbitalRotation", [0, 1]),
        ("OrbitalRotation", [2, 3]),
    ]


# --- interaction with a preceding InitializeModes -----------------------------------------------


def test_merge_leaves_preceding_init_untouched():
    """A run of rotations after an InitializeModes collapses without disturbing the init.

    The pass merges only the rotations; the preceding :class:`.InitializeModes` is left in place, so
    the result is the ``Init -> OrbitalRotation`` shape a downstream Slater-preparation optimization
    (once available) can recognize.
    """
    circ = FermionicCircuit(3)
    circ.append(InitializeModes([1, 1, 0]), circ.modes)
    circ.append(OrbitalRotation(random_unitary(3, seed=1)), circ.modes)
    circ.append(OrbitalRotation(random_unitary(3, seed=2)), circ.modes)

    assert _nodes(_merge(circ)) == [
        ("InitializeModes", [0, 1, 2]),
        ("OrbitalRotation", [0, 1, 2]),
    ]
