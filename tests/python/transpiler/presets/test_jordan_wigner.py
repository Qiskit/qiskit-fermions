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

"""Preset Jordan-Wigner transpiler test."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate, XXPlusYYGate
from qiskit.quantum_info import Statevector
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import (
    Evolution,
    InitializeModes,
    OrbitalRotation,
    PrepareSlaterDeterminant,
)
from qiskit_fermions.mappers.library import jordan_wigner
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager

from ...utils import random_unitary


def test_preset_jordan_wigner():
    hamil = FermionOperator.from_dict(
        {
            ((True, 0), (False, 2)): 2.0,
            ((True, 2), (False, 0)): 2.0,
            ((True, 1), (False, 3)): -2.0,
            ((True, 3), (False, 1)): -2.0,
        }
    )
    time = 1.5
    num_modes = 4
    evo = Evolution(num_modes, hamil, time=time)

    init = InitializeModes([True, False, True, False])

    rot_mat = np.zeros((num_modes, num_modes), dtype=complex)
    rot_mat[0, 1] = 1
    rot_mat[1, 0] = 1
    rot_mat[2, 3] = 1
    rot_mat[3, 2] = 1
    orb_rot = OrbitalRotation(rot_mat)

    circ = FermionicCircuit(num_modes)
    circ.append(init, circ.modes)
    circ.append(orb_rot, circ.modes)
    circ.append(evo, circ.modes)

    pm = generate_preset_jw_pass_manager()

    qu_circ = pm.run(circ)

    expected = QuantumCircuit(num_modes)
    expected.x((0, 2))
    expected.append(XXPlusYYGate(np.pi, -np.pi / 2), (0, 1))
    expected.append(XXPlusYYGate(np.pi, -np.pi / 2), (2, 3))
    expected.p(np.pi, 0)
    expected.p(np.pi, 2)
    expected.append(
        PauliEvolutionGate(jordan_wigner(hamil, num_qubits=num_modes).simplify(), time=time),
        expected.qubits,
    )

    assert Statevector(qu_circ) == Statevector(expected)


def test_preset_jordan_wigner_identity_rotation():
    """The preset lowers rotation-free circuits (only X gates, idle qubits) to the correct state.

    Smoke test for the ``F2QSynthesis`` output-DAG construction. The pass used to build its output by
    copying the fermionic input DAG and deleting the mode wires (``copy_empty_like`` +
    ``remove_qubits``), which left the DAG's internal wire graph inconsistent. A circuit synthesizing
    to only single-qubit gates on a subset of qubits -- an identity ``OrbitalRotation`` lowers to
    just ``X`` gates with idle qubits -- surfaced this as a ``not a DAG`` panic in the downstream
    ``Depth`` analysis pass.

    Note this is a *smoke* test, not a deterministic reproducer: the panic originates in Qiskit's
    rustworkx-backed DAG and its reproducibility depends on process memory layout, so it cannot be
    triggered on demand. The test therefore only asserts that the path used to fail now lowers
    cleanly to the expected state; see the pull request for the standalone ``depth()`` reducer that
    reproduces the underlying corruption deterministically.
    """
    num_modes = 4
    occupation = [True, True, False, False]
    identity = np.eye(num_modes, dtype=complex)

    orbital = FermionicCircuit(num_modes)
    orbital.append(InitializeModes(occupation), orbital.modes)
    orbital.append(OrbitalRotation(identity), orbital.modes)

    reference = QuantumCircuit(num_modes)
    reference.x((0, 1))
    expected = Statevector(reference)

    assert Statevector(generate_preset_jw_pass_manager().run(orbital)) == expected


def test_preset_jordan_wigner_prepare_slater_determinant():
    """The preset lowers a PrepareSlaterDeterminant to the reduced Slater synthesis.

    The prepared state must agree (up to a global phase, which the reduced decomposition drops) with
    the equivalent separate InitializeModes + OrbitalRotation lowered by the same preset.
    """
    num_modes = 6
    occupation = [True, True, True, False, False, False]
    rotation = random_unitary(num_modes, seed=42)

    slater = FermionicCircuit(num_modes)
    slater.append(PrepareSlaterDeterminant(occupation, rotation), slater.modes)

    unmerged = FermionicCircuit(num_modes)
    unmerged.append(InitializeModes(occupation), unmerged.modes)
    unmerged.append(OrbitalRotation(rotation), unmerged.modes)

    pm = generate_preset_jw_pass_manager()
    sv_slater = Statevector(pm.run(slater)).data
    sv_unmerged = Statevector(pm.run(unmerged)).data

    # align the global phase on the largest-magnitude amplitude, then compare
    k = int(np.argmax(np.abs(sv_unmerged)))
    phase = sv_slater[k] / sv_unmerged[k]
    np.testing.assert_allclose(sv_slater, phase * sv_unmerged, atol=1e-10)


def test_preset_jordan_wigner_merges_rotation_run():
    """The preset's optimization stage merges a run of consecutive OrbitalRotation gates.

    A run of two OrbitalRotations on the same modes is collapsed by MergeOrbitalRotations into a
    single rotation before synthesis, so the preset yields the same state (up to a global phase) as
    the equivalent pre-composed single rotation while lowering only one decomposition.
    """
    num_modes = 6
    occupation = [True, True, True, False, False, False]
    u1 = random_unitary(num_modes, seed=11)
    u2 = random_unitary(num_modes, seed=12)

    circ = FermionicCircuit(num_modes)
    circ.append(InitializeModes(occupation), circ.modes)
    circ.append(OrbitalRotation(u1), circ.modes)
    circ.append(OrbitalRotation(u2), circ.modes)

    pm = generate_preset_jw_pass_manager()
    merged = pm.run(circ)

    # equivalent unmerged reference: the two rotations pre-composed into one full OrbitalRotation
    reference = FermionicCircuit(num_modes)
    reference.append(InitializeModes(occupation), reference.modes)
    reference.append(OrbitalRotation(u2 @ u1), reference.modes)
    lowered_reference = pm.run(reference)

    sv_merged = Statevector(merged).data
    sv_reference = Statevector(lowered_reference).data
    k = int(np.argmax(np.abs(sv_reference)))
    phase = sv_merged[k] / sv_reference[k]
    np.testing.assert_allclose(sv_merged, phase * sv_reference, atol=1e-10)

    # the run collapses to a single rotation, so it lowers to the same gate count as the
    # pre-composed reference -- rather than twice the decomposition two separate rotations require
    assert merged.count_ops().get("xx_plus_yy", 0) == lowered_reference.count_ops().get(
        "xx_plus_yy", 0
    )


def test_preset_jordan_wigner_merges_initialize_rotation():
    """The preset's optimization stage fuses InitializeModes + OrbitalRotation.

    An InitializeModes immediately followed by an OrbitalRotation is merged into a
    PrepareSlaterDeterminant and lowered with the reduced Slater synthesis, so the preset yields
    the same state (up to a global phase) as the unmerged pair but with fewer gates and no phases.
    """
    num_modes = 6
    occupation = [True, True, True, False, False, False]
    rotation = random_unitary(num_modes, seed=17)

    circ = FermionicCircuit(num_modes)
    circ.append(InitializeModes(occupation), circ.modes)
    circ.append(OrbitalRotation(rotation), circ.modes)

    pm = generate_preset_jw_pass_manager()
    merged = pm.run(circ)

    # the reduced decomposition drops the diagonal phase gates the full orbital rotation would emit
    assert "p" not in merged.count_ops()

    # equivalent reference lowered without the merge (a bare OrbitalRotation keeps its phases)
    reference = FermionicCircuit(num_modes)
    reference.append(OrbitalRotation(rotation), reference.modes)
    ref_ops = pm.run(reference).count_ops()
    assert merged.count_ops().get("xx_plus_yy", 0) < ref_ops.get("xx_plus_yy", 0)
