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

"""Orbital rotation synthesis tests."""

from __future__ import annotations

from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import OrbitalRotation
from qiskit_fermions.transpiler import FermionPassManager, FermionStagedPassManager
from qiskit_fermions.transpiler.passes import (
    F2QSynthesis,
    OrbitalRotationSynthesis,
    TrivialF2QLayout,
)
from qiskit_fermions.transpiler.passmanager import FermionToQubitConverter

from ...utils import random_unitary


def test_orbital_rotation_global_gate_synthesis():
    num_modes = 6
    rotation_unitary = random_unitary(num_modes, seed=42)
    circ = FermionicCircuit(num_modes)
    rot = OrbitalRotation(rotation_unitary)
    circ.append(rot, circ.fermions)

    synth = F2QSynthesis()
    synth.plugins[OrbitalRotation] = OrbitalRotationSynthesis()

    pm = FermionStagedPassManager()
    pm.layout = FermionPassManager(TrivialF2QLayout())
    pm.synthesis = FermionToQubitConverter(synth)

    qu_circ = pm.run(circ)

    ops = qu_circ.count_ops()
    assert ops == {"xx_plus_yy": 15, "p": 6}


def test_initialize_modes_local_gate_synthesis():
    num_modes = 6
    rotation_unitary_a = random_unitary(num_modes // 2, seed=42)
    rotation_unitary_b = random_unitary(num_modes // 2, seed=43)
    circ = FermionicCircuit(num_modes)
    rot_a = OrbitalRotation(rotation_unitary_a)
    circ.append(rot_a, circ.fermions[:3])
    rot_b = OrbitalRotation(rotation_unitary_b)
    circ.append(rot_b, circ.fermions[3:])

    synth = F2QSynthesis()
    synth.plugins[OrbitalRotation] = OrbitalRotationSynthesis()

    pm = FermionStagedPassManager()
    pm.layout = FermionPassManager(TrivialF2QLayout())
    pm.synthesis = FermionToQubitConverter(synth)

    qu_circ = pm.run(circ)

    ops = qu_circ.count_ops()
    assert ops == {"xx_plus_yy": 6, "p": 6}
