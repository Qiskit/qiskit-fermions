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

import numpy as np
import pytest
from qiskit.passmanager import MultiStagePassManager
from qiskit.quantum_info import Operator
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import OrbitalRotation
from qiskit_fermions.mappers.library import jordan_wigner
from qiskit_fermions.transpiler import (
    FermionicCircuitToDAG,
    FermionicPassManager,
    QuantumDAGToCircuit,
)
from qiskit_fermions.transpiler.passes import (
    F2FSynthesis,
    F2QSynthesis,
    HopEvolutionSynthesis,
    MapperFnEvolutionSynthesis,
    TrivialF2QLayout,
)

from ...utils import random_unitary


def _pass_manager(**kwargs) -> MultiStagePassManager:
    """The Jordan-Wigner pipeline for an orbital rotation, assembled from its stages."""
    synth = F2QSynthesis()
    synth.methods["Evolution"] = MapperFnEvolutionSynthesis(jordan_wigner)
    return MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        optimization=FermionicPassManager(F2FSynthesis({"OrbitalRotation": "Givens"})),
        layout=TrivialF2QLayout(),
        synthesis=synth,
        qubit=generate_preset_pass_manager(
            hls_config=HLSConfig(PauliEvolution=[HopEvolutionSynthesis(), "default"]), **kwargs
        ),
        output=QuantumDAGToCircuit(),
    )


def test_orbital_rotation_global_gate_synthesis():
    num_modes = 6
    rotation_unitary = random_unitary(num_modes, seed=42)
    circ = FermionicCircuit(num_modes)
    circ.append(OrbitalRotation(rotation_unitary), circ.modes)

    qu_circ = _pass_manager(optimization_level=0, basis_gates=["xx_plus_yy", "p", "rz", "u3"]).run(
        circ
    )

    ops = qu_circ.count_ops()
    # the Givens mesh of n modes has n (n - 1) / 2 rotations between adjacent modes, each of which
    # is exactly one XXPlusYYGate
    assert ops["xx_plus_yy"] == 15


def test_orbital_rotation_local_gate_synthesis():
    num_modes = 6
    circ = FermionicCircuit(num_modes)
    circ.append(OrbitalRotation(random_unitary(num_modes // 2, seed=42)), circ.modes[:3])
    circ.append(OrbitalRotation(random_unitary(num_modes // 2, seed=43)), circ.modes[3:])

    qu_circ = _pass_manager(optimization_level=0, basis_gates=["xx_plus_yy", "p", "rz", "u3"]).run(
        circ
    )

    assert qu_circ.count_ops()["xx_plus_yy"] == 6


@pytest.mark.parametrize("num_modes", [2, 3, 4, 5])
def test_orbital_rotation_unitary_matches_the_single_particle_block(num_modes):
    """The lowered circuit implements the rotation on the one-particle sector.

    Under Jordan-Wigner the amplitude ``<j|R|i>`` between the single-occupancy states of modes ``i``
    and ``j`` is ``U[j, i]`` (``a†_i -> sum_j U[j, i] a†_j``), up to the global phase that the
    rotation carries on the vacuum, which is none.
    """
    unitary = random_unitary(num_modes, seed=7 + num_modes)
    circ = FermionicCircuit(num_modes)
    circ.append(OrbitalRotation(unitary), circ.modes)

    matrix = Operator(_pass_manager(optimization_level=0).run(circ)).data

    for source in range(num_modes):
        for target in range(num_modes):
            np.testing.assert_allclose(
                matrix[1 << target, 1 << source], unitary[target, source], atol=1e-10
            )
    np.testing.assert_allclose(matrix[0, 0], 1.0, atol=1e-10)


def test_successive_rotations_are_lowered_in_circuit_order():
    """Two decomposed rotations must reach the qubit circuit as one after the other.

    Substituting a node by a sub-circuit leaves the DAG's storage order out of sync with its wire
    dependencies, so a synthesis that walked the nodes in storage order interleaved the factors of
    the two rotations. The result was a valid circuit that implemented the wrong unitary.
    """
    num_modes = 3
    first = random_unitary(num_modes, seed=1)
    second = random_unitary(num_modes, seed=2)
    circ = FermionicCircuit(num_modes)
    circ.append(OrbitalRotation(first), circ.modes)
    circ.append(OrbitalRotation(second), circ.modes)

    def lowered(unitary):
        single = FermionicCircuit(num_modes)
        single.append(OrbitalRotation(unitary), single.modes)
        return Operator(_pass_manager(optimization_level=0).run(single)).data

    both = Operator(_pass_manager(optimization_level=0).run(circ)).data

    # a rotation is applied after the other, so the unitaries multiply, later on the left
    np.testing.assert_allclose(both, lowered(second) @ lowered(first), atol=1e-10)


def test_orbital_rotation_reaching_the_synthesis_is_rejected():
    circ = FermionicCircuit(2)
    circ.append(OrbitalRotation(random_unitary(2, seed=1)), circ.modes)

    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=F2QSynthesis(),
        output=QuantumDAGToCircuit(),
    )

    with pytest.raises(TypeError, match="No plugin registered"):
        pm.run(circ)
