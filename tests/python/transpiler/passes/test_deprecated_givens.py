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

"""Tests for the deprecated Givens fermion-to-qubit plugins."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.passmanager import MultiStagePassManager
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import OrbitalRotation, PrepareSlaterDeterminant
from qiskit_fermions.mappers.library import jordan_wigner
from qiskit_fermions.transpiler import FermionicCircuitToDAG, QuantumDAGToCircuit
from qiskit_fermions.transpiler.passes import (
    F2QSynthesis,
    F2QSynthesisPluginManager,
    GivensDecompositionOrbitalRotationSynthesis,
    GivensDecompositionSlaterDeterminantSynthesis,
    MapperFnEvolutionSynthesis,
    TrivialF2QLayout,
    TrivialOccupationInitializeModesSynthesis,
)

from ...utils import random_unitary, reference_pass_manager


def _legacy_pass_manager(gate: str, plugin) -> MultiStagePassManager:
    """The pipeline that used to be assembled: the Givens plugin maps the gate straight to qubits."""
    synthesis = F2QSynthesis()
    synthesis.methods[gate] = plugin
    synthesis.methods["Evolution"] = MapperFnEvolutionSynthesis(jordan_wigner)
    synthesis.methods["InitializeModes"] = TrivialOccupationInitializeModesSynthesis()
    return MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=synthesis,
        qubit=generate_preset_pass_manager(optimization_level=0),
        output=QuantumDAGToCircuit(),
    )


def test_the_deprecation_is_announced():
    with pytest.warns(DeprecationWarning, match="F2FSynthesis"):
        GivensDecompositionOrbitalRotationSynthesis()
    with pytest.warns(DeprecationWarning, match="F2FSynthesis"):
        GivensDecompositionSlaterDeterminantSynthesis()


def test_the_old_entry_points_are_still_registered():
    manager = F2QSynthesisPluginManager()

    assert manager.method("OrbitalRotation", "GivensDecomposition") is (
        GivensDecompositionOrbitalRotationSynthesis
    )
    assert manager.method("PrepareSlaterDeterminant", "GivensDecompositionSlater") is (
        GivensDecompositionSlaterDeterminantSynthesis
    )


def test_the_old_config_still_builds_the_pass():
    with pytest.warns(DeprecationWarning):
        synth = F2QSynthesis(
            {
                "OrbitalRotation": "GivensDecomposition",
                "PrepareSlaterDeterminant": "GivensDecompositionSlater",
            }
        )

    assert isinstance(synth.methods["OrbitalRotation"], GivensDecompositionOrbitalRotationSynthesis)


def test_minimize_2q_gate_count_is_ignored_with_a_warning():
    with pytest.warns(DeprecationWarning), pytest.warns(UserWarning, match="no longer supported"):
        plugin = GivensDecompositionSlaterDeterminantSynthesis(minimize_2q_gate_count=True)

    assert plugin.minimize_2q_gate_count is False


@pytest.mark.parametrize("num_modes", [2, 3, 5])
def test_an_orbital_rotation_matches_the_modern_pipeline(num_modes):
    circ = FermionicCircuit(num_modes)
    circ.append(OrbitalRotation(random_unitary(num_modes, seed=num_modes)), circ.modes)
    with pytest.warns(DeprecationWarning):
        plugin = GivensDecompositionOrbitalRotationSynthesis()

    legacy = _legacy_pass_manager("OrbitalRotation", plugin).run(circ)
    modern = reference_pass_manager(optimization_level=0).run(circ)

    assert dict(legacy.count_ops()) == dict(modern.count_ops())
    assert Operator(legacy).equiv(Operator(modern))


@pytest.mark.parametrize("occupation", [[1, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 1, 1, 0]])
def test_a_slater_determinant_matches_the_modern_pipeline(occupation):
    num_modes = len(occupation)
    circ = FermionicCircuit(num_modes)
    circ.append(PrepareSlaterDeterminant(occupation, random_unitary(num_modes, seed=3)), circ.modes)
    with pytest.warns(DeprecationWarning):
        plugin = GivensDecompositionSlaterDeterminantSynthesis()

    legacy = _legacy_pass_manager("PrepareSlaterDeterminant", plugin).run(circ)
    modern = reference_pass_manager(optimization_level=0).run(circ)

    assert dict(legacy.count_ops()) == dict(modern.count_ops())
    overlap = abs(np.vdot(Statevector(legacy).data, Statevector(modern).data))
    np.testing.assert_allclose(overlap, 1.0, atol=1e-9)
