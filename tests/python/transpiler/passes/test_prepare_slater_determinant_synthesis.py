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

"""Slater determinant preparation synthesis tests."""

from __future__ import annotations

import numpy as np
from qiskit.passmanager import MultiStagePassManager
from qiskit.quantum_info import Statevector
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import (
    InitializeModes,
    OrbitalRotation,
    PrepareSlaterDeterminant,
)
from qiskit_fermions.transpiler import FermionicCircuitToDAG, QuantumDAGToCircuit
from qiskit_fermions.transpiler.passes import (
    F2QSynthesis,
    GivensDecompositionOrbitalRotationSynthesis,
    GivensDecompositionSlaterDeterminantSynthesis,
    TrivialF2QLayout,
    TrivialOccupationInitializeModesSynthesis,
)

from ...utils import random_unitary


def _synthesize(circ: FermionicCircuit, synth: F2QSynthesis):
    pm = MultiStagePassManager(
        input=FermionicCircuitToDAG(),
        layout=TrivialF2QLayout(),
        synthesis=synth,
        output=QuantumDAGToCircuit(),
    )
    return pm.run(circ)


def _slater_synth() -> F2QSynthesis:
    synth = F2QSynthesis()
    synth.methods["PrepareSlaterDeterminant"] = GivensDecompositionSlaterDeterminantSynthesis()
    return synth


def _reference_synth() -> F2QSynthesis:
    """Synthesizes the equivalent InitializeModes + OrbitalRotation via the trusted square path."""
    synth = F2QSynthesis()
    synth.methods["InitializeModes"] = TrivialOccupationInitializeModesSynthesis()
    synth.methods["OrbitalRotation"] = GivensDecompositionOrbitalRotationSynthesis()
    return synth


def _assert_equal_up_to_global_phase(qc_a, qc_b):
    sv_a = Statevector(qc_a).data
    sv_b = Statevector(qc_b).data
    # align the global phase on the largest-magnitude amplitude, then compare
    k = int(np.argmax(np.abs(sv_b)))
    phase = sv_a[k] / sv_b[k]
    np.testing.assert_allclose(sv_a, phase * sv_b, atol=1e-10)


def test_prepare_slater_determinant_synthesis_matches_reference():
    """The reduced Slater synthesis prepares the same state as the full InitializeModes+rotation."""
    num_modes = 6
    occupation = [True, True, True, False, False, False]
    rotation = random_unitary(num_modes, seed=42)

    slater = FermionicCircuit(num_modes)
    slater.append(PrepareSlaterDeterminant(occupation, rotation), slater.modes)
    qc_slater = _synthesize(slater, _slater_synth())

    reference = FermionicCircuit(num_modes)
    reference.append(InitializeModes(occupation), reference.modes)
    reference.append(OrbitalRotation(rotation), reference.modes)
    qc_reference = _synthesize(reference, _reference_synth())

    _assert_equal_up_to_global_phase(qc_slater, qc_reference)


def test_prepare_slater_determinant_synthesis_scattered_occupation():
    """A non-leading, non-contiguous occupation still prepares the correct state.

    The reduced decomposition seeds a *leading*-``m`` reference (X gates on the first ``m`` modes)
    regardless of where the occupied modes actually sit; ``occupation``'s positions enter only via
    the column selection ``rotation_unitary[:, occupied]``. This checks that this leading-reference
    convention is nonetheless correct for an occupation whose ``True`` entries are neither leading
    nor contiguous.
    """
    num_modes = 6
    occupation = [False, True, False, True, True, False]
    rotation = random_unitary(num_modes, seed=7)

    slater = FermionicCircuit(num_modes)
    slater.append(PrepareSlaterDeterminant(occupation, rotation), slater.modes)
    qc_slater = _synthesize(slater, _slater_synth())

    reference = FermionicCircuit(num_modes)
    reference.append(InitializeModes(occupation), reference.modes)
    reference.append(OrbitalRotation(rotation), reference.modes)
    qc_reference = _synthesize(reference, _reference_synth())

    _assert_equal_up_to_global_phase(qc_slater, qc_reference)


def test_prepare_slater_determinant_synthesis_reduced_gate_count():
    """The Slater synthesis uses fewer gates than the full orbital rotation and emits no phases."""
    num_modes = 6
    nocc = 3
    occupation = [i < nocc for i in range(num_modes)]
    rotation = random_unitary(num_modes, seed=1)

    slater = FermionicCircuit(num_modes)
    slater.append(PrepareSlaterDeterminant(occupation, rotation), slater.modes)
    ops = _synthesize(slater, _slater_synth()).count_ops()

    # at most m(n-m) XXPlusYYGates, m X gates, and crucially no phase gates
    assert ops.get("xx_plus_yy", 0) <= nocc * (num_modes - nocc)
    assert ops.get("x", 0) == nocc
    assert "p" not in ops

    # the full orbital-rotation synthesis of the same rotation uses the n(n-1)/2 brick-wall + phases
    reference = FermionicCircuit(num_modes)
    reference.append(InitializeModes(occupation), reference.modes)
    reference.append(OrbitalRotation(rotation), reference.modes)
    ref_ops = _synthesize(reference, _reference_synth()).count_ops()
    assert ops.get("xx_plus_yy", 0) < ref_ops.get("xx_plus_yy", 0)


def _givens_2mode(theta: float, p: int, q: int, num_modes: int) -> np.ndarray:
    """A real orbital rotation mixing only modes ``p`` and ``q`` (identity elsewhere).

    Used to build occupied spaces that factor into disjoint unit-orbitals plus a genuinely-mixing
    block, exercising the peel-off fast path.
    """
    rot = np.eye(num_modes, dtype=complex)
    rot[p, p] = np.cos(theta)
    rot[p, q] = -np.sin(theta)
    rot[q, p] = np.sin(theta)
    rot[q, q] = np.cos(theta)
    return rot


def test_prepare_slater_determinant_synthesis_peels_disjoint_unit_orbital():
    """A disjoint unit-orbital is placed directly, not transported out with bare SWAPs.

    The rotation mixes only modes 0 and 1; occupying column 3 (a bare basis vector on mode 3) and
    column 1 (a superposition of modes 0 and 1) factors the occupied space into the disjoint
    unit-orbital ``e_3`` plus one genuine two-mode rotation. The reduced sweep therefore needs a
    single ``XXPlusYYGate`` -- the two transport SWAPs the leading-reference sweep would emit for
    ``e_3`` are peeled away -- while the prepared state is unchanged.
    """
    num_modes = 4
    occupation = [False, True, False, True]
    rotation = _givens_2mode(0.6, 0, 1, num_modes)

    slater = FermionicCircuit(num_modes)
    slater.append(PrepareSlaterDeterminant(occupation, rotation), slater.modes)
    qc_slater = _synthesize(slater, _slater_synth())
    ops = qc_slater.count_ops()

    # only the genuine 0-1 mixing survives; no bare-SWAP transport of the e_3 orbital
    assert ops.get("xx_plus_yy", 0) == 1
    assert ops.get("x", 0) == 2

    reference = FermionicCircuit(num_modes)
    reference.append(InitializeModes(occupation), reference.modes)
    reference.append(OrbitalRotation(rotation), reference.modes)
    qc_reference = _synthesize(reference, _reference_synth())
    _assert_equal_up_to_global_phase(qc_slater, qc_reference)


def test_prepare_slater_determinant_synthesis_permutation_no_two_qubit_gates():
    """A pure permutation rotation (here the identity) emits only X gates, no ``XXPlusYYGate``.

    With ``rotation`` the identity and a non-leading, non-contiguous occupation the entire occupied
    space is a signed permutation, so every occupied orbital peels: the state is prepared with X
    gates placed directly on the occupied modes and zero two-qubit gates.
    """
    num_modes = 4
    occupation = [False, True, False, True]
    rotation = np.eye(num_modes, dtype=complex)

    slater = FermionicCircuit(num_modes)
    slater.append(PrepareSlaterDeterminant(occupation, rotation), slater.modes)
    qc_slater = _synthesize(slater, _slater_synth())
    ops = qc_slater.count_ops()

    assert ops.get("xx_plus_yy", 0) == 0
    assert ops.get("x", 0) == 2
    assert "p" not in ops

    reference = FermionicCircuit(num_modes)
    reference.append(InitializeModes(occupation), reference.modes)
    reference.append(OrbitalRotation(rotation), reference.modes)
    qc_reference = _synthesize(reference, _reference_synth())
    _assert_equal_up_to_global_phase(qc_slater, qc_reference)


def test_prepare_slater_determinant_synthesis_multiple_disjoint_units():
    """Several disjoint unit-orbitals peel independently around a single mixing block.

    The rotation mixes only the adjacent modes 0 and 1; occupying modes 0, 4, 5 gives two disjoint
    unit-orbitals (``e_4``, ``e_5``) plus a single occupied orbital spanning the adjacent pair {0, 1}.
    (Occupying *both* mixed columns 0 and 1 would fill the whole {0, 1} subspace and reduce to a plain
    permutation -- selecting only column 0 keeps a genuine two-mode rotation.) The two disjoint units
    peel to direct X gates, leaving a single ``XXPlusYYGate`` on the adjacent mixing pair.
    """
    num_modes = 6
    occupation = [True, False, False, False, True, True]
    rotation = _givens_2mode(0.5, 0, 1, num_modes)

    slater = FermionicCircuit(num_modes)
    slater.append(PrepareSlaterDeterminant(occupation, rotation), slater.modes)
    qc_slater = _synthesize(slater, _slater_synth())
    ops = qc_slater.count_ops()

    assert ops.get("xx_plus_yy", 0) == 1
    assert ops.get("x", 0) == 3

    reference = FermionicCircuit(num_modes)
    reference.append(InitializeModes(occupation), reference.modes)
    reference.append(OrbitalRotation(rotation), reference.modes)
    qc_reference = _synthesize(reference, _reference_synth())
    _assert_equal_up_to_global_phase(qc_slater, qc_reference)


def test_prepare_slater_determinant_synthesis_generic_rotation_not_peeled():
    """A generic entangled rotation has nothing to peel; the sweep is unchanged.

    This guards the fast path against regressing the common case: a fully-mixing occupied space must
    still be synthesized by the ordinary reduced Givens sweep (its ``m(n-m)`` bound intact) and
    prepare the correct state.
    """
    num_modes = 6
    nocc = 3
    occupation = [i < nocc for i in range(num_modes)]
    rotation = random_unitary(num_modes, seed=42)

    slater = FermionicCircuit(num_modes)
    slater.append(PrepareSlaterDeterminant(occupation, rotation), slater.modes)
    qc_slater = _synthesize(slater, _slater_synth())
    ops = qc_slater.count_ops()

    # nothing peels: the full leading-reference sweep runs (up to m(n-m) two-qubit gates, m X gates)
    assert ops.get("x", 0) == nocc
    assert 0 < ops.get("xx_plus_yy", 0) <= nocc * (num_modes - nocc)

    reference = FermionicCircuit(num_modes)
    reference.append(InitializeModes(occupation), reference.modes)
    reference.append(OrbitalRotation(rotation), reference.modes)
    qc_reference = _synthesize(reference, _reference_synth())
    _assert_equal_up_to_global_phase(qc_slater, qc_reference)


def test_prepare_slater_determinant_synthesis_full_occupation():
    """A fully occupied reference (m == n) needs no Givens rotations, only the X gates."""
    num_modes = 4
    occupation = [True] * num_modes
    rotation = random_unitary(num_modes, seed=2)

    slater = FermionicCircuit(num_modes)
    slater.append(PrepareSlaterDeterminant(occupation, rotation), slater.modes)
    ops = _synthesize(slater, _slater_synth()).count_ops()

    # m == n: the occupied space is the whole space, so any rotation within it is discarded
    assert ops.get("x", 0) == num_modes
    assert ops.get("xx_plus_yy", 0) == 0
