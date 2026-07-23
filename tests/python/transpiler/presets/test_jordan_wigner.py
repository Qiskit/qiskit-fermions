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
from qiskit_fermions.circuit.library import Evolution, InitializeModes, OrbitalRotation
from qiskit_fermions.mappers.library import jordan_wigner
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager


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
