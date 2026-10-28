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

"""End-to-end tests of the Slater determinant preparation through the Jordan-Wigner pipeline.

The fermionic synthesis is tested on its own in
``tests/python/circuit/library/test_slater_determinant_synthesis.py``; these tests cover what the
pipeline does with it: the reduced gate count under Jordan-Wigner, and the state that comes out.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import (
    InitializeModes,
    OrbitalRotation,
    PrepareSlaterDeterminant,
)

from ...utils import (
    SLATER_STRUCTURE_CASES,
    draw_multi_block_target,
    givens_2mode,
    multi_block_rotation,
    random_unitary,
    reference_pass_manager,
)


def _synthesize(num_modes, occupation, rotation):
    """Lowers a ``PrepareSlaterDeterminant`` through the Jordan-Wigner pipeline."""
    circ = FermionicCircuit(num_modes)
    circ.append(PrepareSlaterDeterminant(occupation, rotation), circ.modes)
    return reference_pass_manager(optimization_level=0).run(circ)


def _reference(num_modes, occupation, rotation):
    """The faithful ``InitializeModes`` + ``OrbitalRotation``, which never block-splits."""
    circ = FermionicCircuit(num_modes)
    circ.append(InitializeModes(occupation), circ.modes)
    circ.append(OrbitalRotation(rotation), circ.modes)
    return reference_pass_manager(optimization_level=0).run(circ)


def _overlap(qc_a, qc_b) -> float:
    return float(abs(Statevector(qc_a).inner(Statevector(qc_b))))


def _assert_matches_reference(num_modes, occupation, rotation):
    """Asserts the state equals the reference one up to a global phase; returns the circuit."""
    qc = _synthesize(num_modes, occupation, rotation)
    reference = _reference(num_modes, occupation, rotation)
    np.testing.assert_allclose(_overlap(qc, reference), 1.0, atol=1e-9)
    return qc


def _xx_plus_yy_count(qc) -> int:
    return qc.count_ops().get("xx_plus_yy", 0)


def test_matches_reference():
    """The reduced Slater synthesis prepares the same state as the full InitializeModes+rotation."""
    num_modes = 6
    occupation = [True, True, True, False, False, False]
    _assert_matches_reference(num_modes, occupation, random_unitary(num_modes, seed=42))


def test_scattered_occupation():
    """A non-leading, non-contiguous occupation still prepares the correct state.

    The reduced decomposition seeds a *leading*-``m`` reference regardless of where the occupied
    modes sit; ``occupation``'s positions enter only via the column selection of the rotation.
    """
    num_modes = 6
    occupation = [False, True, False, True, True, False]
    _assert_matches_reference(num_modes, occupation, random_unitary(num_modes, seed=7))


def test_reduced_gate_count_and_no_phases():
    """The Slater synthesis uses fewer rotations than the full orbital rotation and no phases."""
    num_modes, nocc = 6, 3
    occupation = [i < nocc for i in range(num_modes)]
    rotation = random_unitary(num_modes, seed=1)

    ops = _synthesize(num_modes, occupation, rotation).count_ops()

    # at most m(n-m) XXPlusYYGates, m X gates, and crucially no phase gates
    assert 0 < ops.get("xx_plus_yy", 0) <= nocc * (num_modes - nocc)
    assert ops.get("x", 0) == nocc
    assert not {"p", "rz"} & set(ops)

    ref_ops = _reference(num_modes, occupation, rotation).count_ops()
    assert ops.get("xx_plus_yy", 0) < ref_ops.get("xx_plus_yy", 0)


def test_every_rotation_is_a_hop_between_adjacent_qubits():
    """Every rotation is nearest-neighbor, hence a single ``XXPlusYYGate``."""
    num_modes = 6
    occupation = [i < num_modes - 1 for i in range(num_modes)]
    rotation = givens_2mode(0.6, 0, num_modes - 1, num_modes)

    qc = _assert_matches_reference(num_modes, occupation, rotation)

    spans = []
    for instruction in qc.data:
        if instruction.operation.name == "xx_plus_yy":
            i, j = (qc.find_bit(qubit).index for qubit in instruction.qubits)
            spans.append(abs(i - j))
    assert spans
    assert set(spans) == {1}
    # nothing else was needed to get there
    assert set(qc.count_ops()) <= {"x", "xx_plus_yy"}


def test_full_occupation():
    """A fully occupied reference needs no rotations, only the X gates."""
    num_modes = 4
    qc = _synthesize(num_modes, [True] * num_modes, random_unitary(num_modes, seed=2))

    assert qc.count_ops() == {"x": num_modes}


@pytest.mark.parametrize(
    ("label", "num_modes", "occupied_columns", "build_rotation"),
    SLATER_STRUCTURE_CASES,
    ids=[case[0] for case in SLATER_STRUCTURE_CASES],
)
def test_structures(label, num_modes, occupied_columns, build_rotation):
    """The state is right across occupied-space structures."""
    rotation = build_rotation()
    occupation = [i in occupied_columns for i in range(num_modes)]
    _assert_matches_reference(num_modes, occupation, rotation)


def test_randomized_sweep():
    """A seeded sweep of multi-block targets."""
    rng = np.random.default_rng(20260724)
    trials = 0
    for _ in range(120):
        drawn = draw_multi_block_target(rng)
        if drawn is None:
            continue
        num_modes, occupation, rotation, _ = drawn
        trials += 1
        _assert_matches_reference(num_modes, occupation, rotation)
    assert trials > 60


def test_a_pure_permutation_emits_no_two_qubit_gate():
    qc = _assert_matches_reference(4, [False, True, False, True], np.eye(4, dtype=complex))

    assert qc.count_ops() == {"x": 2}


def test_block_splitting_synthesizes_each_cluster_on_its_own_window():
    """Two disjoint clusters cost one rotation each, not a transport across the gap between."""
    num_modes = 8
    rotation = multi_block_rotation(num_modes, [[0, 1], [5, 6]], seed=31)
    occupation = [i in (0, 3, 5) for i in range(num_modes)]

    qc = _assert_matches_reference(num_modes, occupation, rotation)

    # one hop per cluster, each between the adjacent qubits of its own window
    assert _xx_plus_yy_count(qc) == 2
