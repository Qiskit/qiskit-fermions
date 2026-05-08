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

"""Fermion-operator evolution tests."""

from __future__ import annotations

from qiskit_fermions.circuit import FermionCircuit
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.operators import FermionOperator


def test_evolution_gate():
    hamil = FermionOperator.from_dict(
        {
            ((True, 0), (False, 1)): 2.0,
            ((True, 1), (False, 0)): 2.0,
            ((True, 2), (False, 3)): -2.0,
            ((True, 3), (False, 2)): -2.0,
        }
    )
    time = 1.5
    num_modes = 4
    circ = FermionCircuit(num_modes)
    evo = Evolution(num_modes, hamil, time=time)
    circ.append(evo, circ.fermions)

    ops = circ.count_ops()
    assert ops == {"Evolution": 1}


def test_evolution_gate_decompose():
    hamil = FermionOperator.from_dict(
        {
            ((True, 0), (False, 1)): 2.0,
            ((True, 1), (False, 0)): 2.0,
            ((True, 2), (False, 3)): -2.0,
            ((True, 3), (False, 2)): -2.0,
        }
    )
    time = 1.5
    num_modes = 4
    circ = FermionCircuit(num_modes)
    evo = Evolution(num_modes, hamil, time=time)
    circ.append(evo, circ.fermions)

    decomposed = circ.decompose()

    ops = decomposed.count_ops()
    assert ops == {"Evolution": 4}


def test_evolution_gate_decompose_with_groups():
    hamil = FermionOperator.zero()
    hamil += FermionOperator.from_dict(
        {
            ((True, 0), (False, 1)): 2.0,
            ((True, 1), (False, 0)): 2.0,
        }
    )
    hamil += FermionOperator.from_dict(
        {
            ((True, 2), (False, 3)): -2.0,
            ((True, 3), (False, 2)): -2.0,
        }
    )
    hamil.groups = [0, 0, 1, 1]
    time = 1.5
    num_modes = 4
    circ = FermionCircuit(num_modes)
    evo = Evolution(num_modes, hamil, time=time)
    circ.append(evo, circ.fermions)

    decomposed = circ.decompose()

    ops = decomposed.count_ops()
    assert ops == {"Evolution": 2}
