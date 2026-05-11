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
    circ.append(init, circ.fermions)
    circ.append(orb_rot, circ.fermions)
    circ.append(evo, circ.fermions)

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
