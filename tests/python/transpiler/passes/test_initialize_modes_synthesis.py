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

"""Mode initialization synthesis tests."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_fermions.circuit import FermionCircuit
from qiskit_fermions.circuit.library import InitializeModes
from qiskit_fermions.transpiler import FermionPassManager, FermionStagedPassManager
from qiskit_fermions.transpiler.passes import (
    F2QSynthesis,
    InitializeModesSynthesis,
    TrivialF2QLayout,
)
from qiskit_fermions.transpiler.passmanager import FermionToQubitConverter


def test_initialize_modes_global_gate_synthesis():
    occupation = [True, False, True, False]
    num_fermions = len(occupation)
    circ = FermionCircuit(num_fermions)
    init = InitializeModes(occupation)
    circ.append(init, circ.fermions)

    synth = F2QSynthesis()
    synth.plugins[InitializeModes] = InitializeModesSynthesis()

    pm = FermionStagedPassManager()
    pm.layout = FermionPassManager(TrivialF2QLayout())
    pm.synthesis = FermionToQubitConverter(synth)

    qu_circ = pm.run(circ)

    expected = QuantumCircuit(num_fermions)
    expected.x(0)
    expected.x(2)

    assert Statevector(qu_circ) == Statevector(expected)


def test_initialize_modes_local_gate_synthesis():
    occupation = [True, False]
    num_fermions = 4
    circ = FermionCircuit(num_fermions)
    init = InitializeModes(occupation)
    circ.append(init, circ.fermions[1:3])

    synth = F2QSynthesis()
    synth.plugins[InitializeModes] = InitializeModesSynthesis()

    pm = FermionStagedPassManager()
    pm.layout = FermionPassManager(TrivialF2QLayout())
    pm.synthesis = FermionToQubitConverter(synth)

    qu_circ = pm.run(circ)

    expected = QuantumCircuit(num_fermions)
    expected.x(1)

    assert Statevector(qu_circ) == Statevector(expected)
