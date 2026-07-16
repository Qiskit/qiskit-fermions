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

"""Structural tests for the UCJ ansatz gate."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import UCJ, OrbitalRotation

from ...utils import random_unitary


def _balanced_tensors(norb, n_reps, *, seed):
    """Returns random ``(diag_coulomb_mats, orbital_rotations)`` for the spin-balanced variant."""
    rng = np.random.default_rng(seed)
    mats = rng.standard_normal((n_reps, 2, norb, norb))
    mats = mats + mats.transpose(0, 1, 3, 2)  # symmetric aa/ab
    rotations = np.stack([random_unitary(norb, seed=seed + k) for k in range(n_reps)])
    return mats, rotations


def test_ucj_infers_balanced_variant():
    """A ``(L, 2, norb, norb)`` diag-Coulomb tensor selects the balanced variant and 2*norb modes."""
    norb = 3
    mats, rotations = _balanced_tensors(norb, 2, seed=0)
    gate = UCJ(norb, (1, 1), mats, rotations)
    assert gate.num_modes == 2 * norb
    assert gate._variant == "balanced"


def test_ucj_infers_unbalanced_variant():
    """A ``(L, 3, norb, norb)`` diag-Coulomb tensor with paired rotations selects unbalanced."""
    norb = 3
    n_reps = 2
    rng = np.random.default_rng(1)
    mats = rng.standard_normal((n_reps, 3, norb, norb))
    rotations = np.stack(
        [
            np.stack([random_unitary(norb, seed=10 + k), random_unitary(norb, seed=20 + k)])
            for k in range(n_reps)
        ]
    )
    gate = UCJ(norb, (2, 1), mats, rotations)
    assert gate.num_modes == 2 * norb
    assert gate._variant == "unbalanced"


def test_ucj_infers_spinless_variant():
    """A ``(L, norb, norb)`` diag-Coulomb tensor selects the spinless variant."""
    norb = 3
    n_reps = 2
    rng = np.random.default_rng(2)
    mats = rng.standard_normal((n_reps, norb, norb))
    rotations = np.stack([random_unitary(norb, seed=30 + k) for k in range(n_reps)])
    # spinful nelec -> 2*norb modes; integer nelec -> norb modes
    assert UCJ(norb, (1, 1), mats, rotations).num_modes == 2 * norb
    spinless = UCJ(norb, 2, mats, rotations)
    assert spinless.num_modes == norb
    assert spinless._variant == "spinless"


def test_ucj_rejects_inconsistent_shapes():
    """Tensor shapes inconsistent with norb or with each other raise ValueError."""
    norb = 3
    mats, rotations = _balanced_tensors(norb, 2, seed=3)
    with pytest.raises(ValueError, match="Inconsistent"):
        UCJ(norb + 1, (1, 1), mats, rotations)  # norb mismatch


def test_ucj_rejects_integer_nelec_for_balanced_tensors():
    """An integer nelec with balanced-shaped tensors is rejected (spinless variant only)."""
    norb = 3
    mats, rotations = _balanced_tensors(norb, 2, seed=4)
    with pytest.raises(ValueError, match="spinless variant"):
        UCJ(norb, 2, mats, rotations)


def test_ucj_default_reference_is_hartree_fock():
    """The default reference occupation is the Hartree-Fock determinant."""
    norb = 3
    mats, rotations = _balanced_tensors(norb, 1, seed=5)
    gate = UCJ(norb, (2, 1), mats, rotations)
    # block-spin order: alpha modes 0..3 (first 2 occupied), beta modes 3..6 (first 1 occupied)
    expected = [True, True, False, True, False, False]
    np.testing.assert_array_equal(gate.reference_occupation, expected)


def test_ucj_rejects_wrong_reference_length():
    """A reference occupation whose length does not match the mode count raises ValueError."""
    norb = 3
    mats, rotations = _balanced_tensors(norb, 1, seed=6)
    with pytest.raises(ValueError, match="reference_occupation has length"):
        UCJ(norb, (1, 1), mats, rotations, reference_occupation=[True, False])


def test_ucj_rejects_spin_sector_exceeding_norb():
    """A spin sector with more electrons than orbitals raises instead of spilling into the block."""
    norb = 2
    mats, rotations = _balanced_tensors(norb, 1, seed=11)
    with pytest.raises(ValueError, match="exceeding the norb"):
        UCJ(norb, (3, 0), mats, rotations)


def test_ucj_rejects_spinless_nelec_exceeding_norb():
    """A spinless electron count above ``norb`` raises rather than indexing out of range."""
    norb = 2
    n_reps = 1
    rng = np.random.default_rng(12)
    mats = rng.standard_normal((n_reps, norb, norb))
    mats = mats + mats.transpose(0, 2, 1)
    rotations = np.stack([random_unitary(norb, seed=12)])
    with pytest.raises(ValueError, match="exceeds the norb"):
        UCJ(norb, 3, mats, rotations)


def test_ucj_define_gate_sequence():
    """The gate definition is InitializeModes + per-rep (rotation, evolution, rotation) + final."""
    norb = 3
    n_reps = 2
    mats, rotations = _balanced_tensors(norb, n_reps, seed=7)
    final = random_unitary(norb, seed=99)
    gate = UCJ(norb, (1, 1), mats, rotations, final_orbital_rotation=final)

    circ = FermionicCircuit(gate.num_modes)
    circ.append(gate, circ.modes)
    ops = circ.decompose().count_ops()

    assert ops["InitializeModes"] == 1
    assert ops["Evolution"] == n_reps
    # per rep: U^dagger (2 local) + U (2 local) = 4; plus a final rotation (2 local)
    assert ops["OrbitalRotation"] == 4 * n_reps + 2


def test_ucj_spinless_true_uses_norb_modes():
    """A true spinless system (integer nelec) acts on norb modes with only same-spin terms."""
    norb = 3
    n_reps = 1
    rng = np.random.default_rng(8)
    mats = rng.standard_normal((n_reps, norb, norb))
    mats = mats + mats.transpose(0, 2, 1)
    rotations = np.stack([random_unitary(norb, seed=40)])
    gate = UCJ(norb, 2, mats, rotations)
    circ = FermionicCircuit(gate.num_modes)
    circ.append(gate, circ.modes)
    ops = circ.decompose().count_ops()
    assert ops["InitializeModes"] == 1
    assert ops["Evolution"] == n_reps
    # spinless places a single rotation on all norb modes: U^dagger + U = 2 per rep
    assert ops["OrbitalRotation"] == 2 * n_reps


def test_ucj_accepts_complex_diag_coulomb_with_zero_imaginary_part():
    """Complex-typed diag-Coulomb mats with a negligible imaginary part are coerced to real."""
    norb = 3
    mats, rotations = _balanced_tensors(norb, 1, seed=11)
    gate = UCJ(norb, (1, 1), mats.astype(complex), rotations)
    assert gate.diag_coulomb_mats.dtype == np.float64
    np.testing.assert_allclose(gate.diag_coulomb_mats, mats)


def test_ucj_rejects_complex_diag_coulomb_with_nonzero_imaginary_part():
    """A genuinely complex diag-Coulomb matrix is rejected rather than silently truncated."""
    norb = 3
    mats, rotations = _balanced_tensors(norb, 1, seed=12)
    complex_mats = mats.astype(complex)
    complex_mats[0, 0, 0, 1] += 0.5j
    with pytest.raises(ValueError, match="imaginary part"):
        UCJ(norb, (1, 1), complex_mats, rotations)


def test_ucj_from_t_amplitudes_rejects_tuple_n_reps_for_balanced():
    """A tuple n_reps is only valid for the unbalanced variant; balanced/spinless reject it."""
    nocc, nvrt = 1, 2
    t2 = np.zeros((nocc, nocc, nvrt, nvrt))
    with pytest.raises(ValueError, match="only valid for the 'unbalanced' variant"):
        UCJ.from_t_amplitudes((1, 1), t2, variant="balanced", n_reps=(2, 3))


def test_ucj_from_t_amplitudes_empty_interaction_pairs_zeros_layer():
    """An empty-list interaction_pairs zeros the whole diagonal-Coulomb block (unlike None)."""
    nocc, nvrt = 1, 2
    rng = np.random.default_rng(13)
    t2 = rng.standard_normal((nocc, nocc, nvrt, nvrt))
    t2 = t2 + t2.transpose(1, 0, 3, 2)  # a nontrivial symmetric t2 so the factorization is nonzero
    gate = UCJ.from_t_amplitudes(2, t2, variant="spinless", interaction_pairs=[])
    # [] allows no interactions -> every diagonal-Coulomb matrix is zeroed
    np.testing.assert_array_equal(gate.diag_coulomb_mats, 0.0)
    # None (no restriction) leaves the genuine factorization terms in place
    gate_none = UCJ.from_t_amplitudes(2, t2, variant="spinless", interaction_pairs=None)
    assert np.any(gate_none.diag_coulomb_mats != 0.0)


def test_orbital_rotation_from_t1_amplitudes_is_unitary():
    """OrbitalRotation.from_t1_amplitudes builds a unitary of the right size."""
    t1 = np.array([[0.1, 0.2], [0.3, -0.1]])  # 2 occ, 2 virt
    gate = OrbitalRotation.from_t1_amplitudes(t1)
    assert isinstance(gate, OrbitalRotation)
    assert gate.num_modes == 4
    u = gate.rotation_unitary
    np.testing.assert_allclose(u.conj().T @ u, np.eye(4), atol=1e-12)
