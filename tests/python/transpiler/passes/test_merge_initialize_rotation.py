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

"""Tests for the MergeSlaterDeterminantPreparation optimization pass."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager import MultiStagePassManager
from qiskit.quantum_info import Statevector
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import (
    UCJ,
    Evolution,
    InitializeModes,
    OrbitalRotation,
    PrepareSlaterDeterminant,
)
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.operators.fermion_action import ann, cre
from qiskit_fermions.transpiler import FermionicCircuitToDAG, QuantumDAGToCircuit
from qiskit_fermions.transpiler.passes import (
    F2QSynthesis,
    GivensDecompositionOrbitalRotationSynthesis,
    GivensDecompositionSlaterDeterminantSynthesis,
    MergeSlaterDeterminantPreparation,
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


def _merge(circ: FermionicCircuit) -> list[tuple[str, list[int]]]:
    """Runs the merge pass on ``circ`` and returns its post-merge node list."""
    dag = FermionicCircuitToDAG().run(circ)
    out = MergeSlaterDeterminantPreparation().run(dag)
    return _nodes(out)


def _block_diag(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """Assembles a block-spin orbital rotation from independent alpha/beta rotations."""
    norb = mat_a.shape[0]
    full = np.zeros((2 * norb, 2 * norb), dtype=complex)
    full[:norb, :norb] = mat_a
    full[norb:, norb:] = mat_b
    return full


# --- positive cases: the three supported patterns fuse -----------------------------------------


def test_merge_full_sector():
    """An InitializeModes followed by an OrbitalRotation on the same modes fuses into one gate."""
    circ = FermionicCircuit(4)
    circ.append(InitializeModes([1, 1, 0, 0]), circ.modes)
    circ.append(OrbitalRotation(random_unitary(4, seed=1)), circ.modes)

    assert _merge(circ) == [("PrepareSlaterDeterminant", [0, 1, 2, 3])]


def test_merge_per_sector():
    """Two separate per-sector init+rotation pairs each fuse independently."""
    circ = FermionicCircuit(4)
    circ.append(InitializeModes([1, 0]), circ.modes[:2])
    circ.append(OrbitalRotation(random_unitary(2, seed=1)), circ.modes[:2])
    circ.append(InitializeModes([1, 0]), circ.modes[2:])
    circ.append(OrbitalRotation(random_unitary(2, seed=2)), circ.modes[2:])

    assert _merge(circ) == [
        ("PrepareSlaterDeterminant", [0, 1]),
        ("PrepareSlaterDeterminant", [2, 3]),
    ]


def test_merge_global_init_per_spin_rotations():
    """A full-register init split by two per-spin rotations emits two gates (one per sector)."""
    circ = FermionicCircuit(4)
    circ.append(InitializeModes([1, 0, 1, 0]), circ.modes)
    circ.append(OrbitalRotation(random_unitary(2, seed=1)), circ.modes[:2])
    circ.append(OrbitalRotation(random_unitary(2, seed=2)), circ.modes[2:])

    assert _merge(circ) == [
        ("PrepareSlaterDeterminant", [0, 1]),
        ("PrepareSlaterDeterminant", [2, 3]),
    ]


def test_merge_global_init_per_spin_rotations_beta_first():
    """Pattern 3 fires regardless of the order the two per-spin rotations appear in."""
    circ = FermionicCircuit(4)
    circ.append(InitializeModes([1, 0, 1, 0]), circ.modes)
    circ.append(OrbitalRotation(random_unitary(2, seed=2)), circ.modes[2:])
    circ.append(OrbitalRotation(random_unitary(2, seed=1)), circ.modes[:2])

    assert _merge(circ) == [
        ("PrepareSlaterDeterminant", [0, 1]),
        ("PrepareSlaterDeterminant", [2, 3]),
    ]


def test_merge_splits_occupation_per_sector():
    """Pattern 3 splits the global occupation at the sector boundary onto the two gates."""
    circ = FermionicCircuit(4)
    circ.append(InitializeModes([True, False, False, True]), circ.modes)
    circ.append(OrbitalRotation(random_unitary(2, seed=1)), circ.modes[:2])
    circ.append(OrbitalRotation(random_unitary(2, seed=2)), circ.modes[2:])

    dag = FermionicCircuitToDAG().run(circ)
    out = MergeSlaterDeterminantPreparation().run(dag)
    gates = [node.op for node in out.topological_op_nodes()]

    assert all(isinstance(gate, PrepareSlaterDeterminant) for gate in gates)
    alpha, beta = gates
    np.testing.assert_array_equal(alpha.occupation, [True, False])
    np.testing.assert_array_equal(beta.occupation, [False, True])


# --- negative cases: unmatched shapes are left untouched ----------------------------------------


def test_no_merge_when_not_adjacent():
    """An operation between the init and rotation blocks the fusion."""
    circ = FermionicCircuit(2)
    circ.append(InitializeModes([1, 0]), circ.modes)
    number_op = FermionOperator.from_dict({(cre(0), ann(0)): 1.0})
    circ.append(Evolution(2, number_op, time=0.5), circ.modes)
    circ.append(OrbitalRotation(random_unitary(2, seed=1)), circ.modes)

    assert _merge(circ) == [
        ("InitializeModes", [0, 1]),
        ("Evolution", [0, 1]),
        ("OrbitalRotation", [0, 1]),
    ]


def test_no_merge_when_rotation_is_not_a_spin_half():
    """A single rotation on a partial sub-range that is not a contiguous spin half does not fuse."""
    circ = FermionicCircuit(4)  # norb = 2: the only valid halves are [0, 1] and [2, 3]
    circ.append(InitializeModes([1, 1, 0, 0]), circ.modes)
    circ.append(OrbitalRotation(random_unitary(3, seed=1)), circ.modes[:3])

    assert _merge(circ) == [
        ("InitializeModes", [0, 1, 2, 3]),
        ("OrbitalRotation", [0, 1, 2]),
    ]


def test_no_merge_when_odd_number_of_modes():
    """An odd-mode register cannot be split into two spin halves, so Pattern 3 does not fire.

    The single rotation on modes ``[0, 1]`` does not cover the full init range ``[0, 1, 2]``, so
    Pattern 1/2 falls through to the Pattern 3 spin-half check, which bails out on the odd mode
    count and leaves the circuit unchanged.
    """
    circ = FermionicCircuit(3)
    circ.append(InitializeModes([1, 1, 0]), circ.modes)
    circ.append(OrbitalRotation(random_unitary(2, seed=1)), circ.modes[:2])

    assert _merge(circ) == [
        ("InitializeModes", [0, 1, 2]),
        ("OrbitalRotation", [0, 1]),
    ]


def test_run_rejects_multiple_registers():
    """``run`` only supports a single fermionic register."""
    dag = DAGCircuit()
    dag.add_qreg(QuantumRegister(2, "a"))
    dag.add_qreg(QuantumRegister(2, "b"))

    with pytest.raises(NotImplementedError, match=r"more than .*a single register"):
        MergeSlaterDeterminantPreparation().run(dag)


def test_no_merge_rotation_without_init():
    """An OrbitalRotation with no preceding InitializeModes is left untouched."""
    circ = FermionicCircuit(2)
    circ.append(OrbitalRotation(random_unitary(2, seed=1)), circ.modes)

    assert _merge(circ) == [("OrbitalRotation", [0, 1])]


def test_merge_only_one_per_spin_rotation():
    """A full-register init with only one per-spin rotation fuses, identity-padding the other half.

    The unrotated (beta) half is prepared with an identity rotation, which synthesizes to just its
    reference X gates -- so the fusion still emits two :class:`.PrepareSlaterDeterminant` gates at no
    extra gate cost while unlocking the reduced Slater synthesis on the rotated (alpha) half.
    """
    circ = FermionicCircuit(4)
    circ.append(InitializeModes([1, 0, 1, 0]), circ.modes)
    circ.append(OrbitalRotation(random_unitary(2, seed=1)), circ.modes[:2])

    assert _merge(circ) == [
        ("PrepareSlaterDeterminant", [0, 1]),
        ("PrepareSlaterDeterminant", [2, 3]),
    ]


def test_merge_only_one_per_spin_rotation_beta():
    """The single-rotation fusion also works when it is the beta half that is rotated."""
    circ = FermionicCircuit(4)
    circ.append(InitializeModes([1, 0, 1, 0]), circ.modes)
    circ.append(OrbitalRotation(random_unitary(2, seed=1)), circ.modes[2:])

    assert _merge(circ) == [
        ("PrepareSlaterDeterminant", [0, 1]),
        ("PrepareSlaterDeterminant", [2, 3]),
    ]


def test_merge_single_rotation_identity_pad_preserves_state():
    """Identity-padding the unrotated half preserves the state up to a global phase, no phase gates.

    Compares the merged circuit (one reduced Slater prep + one identity-padded prep) against the
    unmerged InitializeModes + single OrbitalRotation reference lowered by the same square path.
    """
    circ = FermionicCircuit(4)
    circ.append(InitializeModes([1, 0, 1, 0]), circ.modes)
    circ.append(OrbitalRotation(random_unitary(2, seed=3)), circ.modes[:2])

    reference = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=_reference_synth(),
        output=QuantumDAGToCircuit(),
    ).run(circ)
    merged = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        optimization=MergeSlaterDeterminantPreparation(),
        layout=TrivialF2QLayout(),
        synthesis=_merged_synth(),
        output=QuantumDAGToCircuit(),
    ).run(circ)

    _assert_equal_up_to_global_phase(reference, merged)
    # the identity-padded half adds no gates; the reduced decomposition carries no phase gates
    assert "p" not in merged.count_ops()


# --- equivalence + payoff -----------------------------------------------------------------------


def _reference_synth() -> F2QSynthesis:
    """The trusted square path for a separate InitializeModes + OrbitalRotation."""
    synth = F2QSynthesis()
    synth.methods["InitializeModes"] = TrivialOccupationInitializeModesSynthesis()
    synth.methods["OrbitalRotation"] = GivensDecompositionOrbitalRotationSynthesis()
    return synth


def _merged_synth() -> F2QSynthesis:
    """The reduced Slater path plus the square fallbacks for anything not fused."""
    synth = _reference_synth()
    synth.methods["PrepareSlaterDeterminant"] = GivensDecompositionSlaterDeterminantSynthesis()
    return synth


def _assert_equal_up_to_global_phase(qc_a, qc_b):
    sv_a = Statevector(qc_a).data
    sv_b = Statevector(qc_b).data
    k = int(np.argmax(np.abs(sv_b)))
    phase = sv_a[k] / sv_b[k]
    np.testing.assert_allclose(sv_a, phase * sv_b, atol=1e-10)


def test_merge_preserves_state_and_reduces_gate_count():
    """Transpiling the merged circuit yields the same state, with fewer gates and no phases."""
    circ = FermionicCircuit(4)
    circ.append(InitializeModes([1, 1, 0, 0]), circ.modes)
    circ.append(OrbitalRotation(random_unitary(4, seed=9)), circ.modes)

    reference = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=_reference_synth(),
        output=QuantumDAGToCircuit(),
    ).run(circ)
    merged = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        optimization=MergeSlaterDeterminantPreparation(),
        layout=TrivialF2QLayout(),
        synthesis=_merged_synth(),
        output=QuantumDAGToCircuit(),
    ).run(circ)

    _assert_equal_up_to_global_phase(reference, merged)

    ref_ops = reference.count_ops()
    merged_ops = merged.count_ops()
    assert merged_ops.get("xx_plus_yy", 0) < ref_ops.get("xx_plus_yy", 0)
    # the reduced decomposition carries no diagonal phase gates
    assert "p" not in merged_ops


def test_merge_ucj_workflow():
    """The InitializeModes.from_hartree_fock + decomposed UCJ workflow fuses (pattern 3).

    A user places an :class:`.InitializeModes` at the front and appends a :class:`.UCJ`. After
    ``decompose()`` the ansatz expands to per-spin rotations and evolutions; the leading init and
    the ansatz's first two per-spin rotations directly follow one another and must fuse into two
    :class:`.PrepareSlaterDeterminant` gates. The later rotations follow an :class:`.Evolution`, not
    the init, and are correctly left untouched.
    """
    norb, nelec, n_reps = 2, (1, 1), 1
    diag_coulomb_mats = np.zeros((n_reps, 2, norb, norb))
    orbital_rotations = np.stack([random_unitary(norb, seed=5) for _ in range(n_reps)])
    ucj = UCJ(norb, "balanced", diag_coulomb_mats, orbital_rotations)

    circ = FermionicCircuit(2 * norb)
    circ.append(InitializeModes.from_hartree_fock(norb, nelec), circ.modes)
    circ.append(ucj, circ.modes)

    merged = _merge(circ.decompose())

    # the two leading per-spin rotations fused; the post-Evolution rotations remain
    assert merged[:2] == [
        ("PrepareSlaterDeterminant", [0, 1]),
        ("PrepareSlaterDeterminant", [2, 3]),
    ]
    assert ("Evolution", [0, 1, 2, 3]) in merged
    assert merged.count(("OrbitalRotation", [0, 1])) == 1
    assert merged.count(("OrbitalRotation", [2, 3])) == 1
