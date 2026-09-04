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

from qiskit_fermions.circuit import FermionicCircuit
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
    circ = FermionicCircuit(num_modes)
    evo = Evolution(num_modes, hamil, time=time)
    circ.append(evo, circ.modes)

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
    circ = FermionicCircuit(num_modes)
    evo = Evolution(num_modes, hamil, time=time)
    circ.append(evo, circ.modes)

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
    circ = FermionicCircuit(num_modes)
    evo = Evolution(num_modes, hamil, time=time)
    circ.append(evo, circ.modes)

    decomposed = circ.decompose()

    ops = decomposed.count_ops()
    assert ops == {"Evolution": 2}


def test_atomic_evolution_is_not_decomposed_further():
    """An atomic factor is left in place rather than expanded -- or, worse, dropped.

    Guarding both failure modes: a definition that re-emits the gate recurses forever, while an
    *empty* definition makes ``decompose()`` discard the evolution entirely.
    """
    hamil = FermionOperator.from_dict({((True, 0), (False, 1)): 1.0})
    circ = FermionicCircuit(2)
    circ.append(Evolution(2, hamil, time=0.5, atomic=True), circ.modes)

    assert circ.decompose(reps=5).count_ops() == {"Evolution": 1}


def test_repeated_decomposition_reaches_a_fixed_point():
    """Regression test for a recursion that had no base case.

    A single-term factor used to decompose into an identical single-term ``Evolution``, so the
    expansion never terminated -- which is what made ``inverse()`` raise ``RecursionError``.
    """
    hamil = FermionOperator.from_dict(
        {
            ((True, 0), (False, 1)): 2.0,
            ((True, 1), (False, 0)): 2.0,
            ((True, 2), (False, 3)): -2.0,
            ((True, 3), (False, 2)): -2.0,
        }
    )
    circ = FermionicCircuit(4)
    circ.append(Evolution(4, hamil, time=1.5), circ.modes)

    once = circ.decompose().count_ops()
    assert once == {"Evolution": 4}
    for reps in (2, 3, 6):
        assert circ.decompose(reps=reps).count_ops() == once


def test_inverse_negates_the_time():
    hamil = FermionOperator.from_dict({((True, 0), (False, 1)): 1.0, ((True, 1), (False, 0)): 1.0})
    evo = Evolution(2, hamil, time=0.5)

    inverse = evo.inverse()

    assert isinstance(inverse, Evolution)
    assert inverse.params[0] == -0.5
    assert inverse.operator.equiv(hamil, 1e-12)
    assert inverse.inverse().params[0] == 0.5


def test_inverse_preserves_the_synthesis_and_atomicity():
    hamil = FermionOperator.from_dict({((True, 0), (False, 1)): 1.0})
    evo = Evolution(2, hamil, time=0.5, atomic=True)

    inverse = evo.inverse()

    assert inverse.atomic
    assert inverse.synthesis is evo.synthesis
