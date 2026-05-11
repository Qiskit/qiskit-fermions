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

"""Fermion-to-qubit synthesis tests."""

from __future__ import annotations

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.transpiler.passes import F2QSynthesis, TrivialF2QLayout
from qiskit_fermions.transpiler.passmanager import (
    FermionicPassManager,
    FermionicStagedPassManager,
    FermionicToQubitConverter,
)


def test_missing_plugin():
    """Test the handling of a missing fermion-to-qubit plugin."""
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
    circ = FermionicCircuit(num_modes)
    evo = Evolution(num_modes, hamil, time=time)
    circ.append(evo, circ.fermions)

    pm = FermionicStagedPassManager()
    pm.layout = FermionicPassManager(TrivialF2QLayout())
    pm.synthesis = FermionicToQubitConverter(F2QSynthesis())

    with pytest.raises(TypeError, match="No plugin registered"):
        _ = pm.run(circ)


def test_unsupported_gate():
    """Test the handling of an unsupported circuit instruction."""
    circ = QuantumCircuit(2)
    circ.x(0)

    pm = PassManager([TrivialF2QLayout(), F2QSynthesis()])

    with pytest.raises(ValueError, match="unsupported circuit instruction"):
        _ = pm.run(circ)
