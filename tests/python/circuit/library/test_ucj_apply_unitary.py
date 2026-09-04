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

"""Tests for applying a UCJ gate to an ffsim state vector (SupportsApplyUnitary).

Each spin variant is validated against ffsim's own ``UCJOpSpin{Balanced,Unbalanced,less}``: the same
operator is applied by ffsim directly and through our :class:`.UCJ` gate, and the resulting state
vectors must agree. Since the gate now wraps the ffsim operator unchanged, these tests pin the
circuit this package builds from it (the ``U exp(i J) U^dagger`` layer order and the diagonal
Coulomb convention), not the tensors themselves.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import UCJ

ffsim = pytest.importorskip("ffsim")


def _apply_ours(ucj_op, norb, nelec, reference):
    """Wraps an ffsim UCJ operator in our UCJ gate and applies it to ``reference``.

    Applies the :class:`.UCJ` gate itself (via ffsim's ``SupportsApplyUnitary`` protocol), so these
    correctness checks exercise the gate's public :meth:`.UCJ._apply_unitary_placed_` path, which
    delegates to its ``_build_definition()``, rather than the definition circuit directly.
    """
    return ffsim.apply_unitary(reference, UCJ(ucj_op), norb=norb, nelec=nelec)


def test_ucj_spin_balanced_matches_ffsim():
    """The balanced UCJ gate reproduces ffsim's UCJOpSpinBalanced state vector."""
    norb = 4
    nelec = (2, 2)
    for with_final in (False, True):
        ucj_op = ffsim.random.random_ucj_op_spin_balanced(
            norb, n_reps=2, with_final_orbital_rotation=with_final, seed=1234 + with_final
        )
        reference = ffsim.hartree_fock_state(norb, nelec)
        expected = ffsim.apply_unitary(reference, ucj_op, norb=norb, nelec=nelec)
        result = _apply_ours(ucj_op, norb, nelec, reference)
        np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucj_spin_unbalanced_matches_ffsim():
    """The unbalanced UCJ gate reproduces ffsim's UCJOpSpinUnbalanced state vector."""
    norb = 4
    nelec = (3, 1)
    for with_final in (False, True):
        ucj_op = ffsim.random.random_ucj_op_spin_unbalanced(
            norb, n_reps=2, with_final_orbital_rotation=with_final, seed=2024 + with_final
        )
        reference = ffsim.hartree_fock_state(norb, nelec)
        expected = ffsim.apply_unitary(reference, ucj_op, norb=norb, nelec=nelec)
        result = _apply_ours(ucj_op, norb, nelec, reference)
        np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucj_spinless_shortcut_via_zeroed_balanced_ab_block_matches_ffsim():
    """A spinless ffsim operator run on a spinful sector is reproduced via a zeroed-ab balanced gate.

    A ``UCJOpSpinless`` always builds a true 1-register spinless gate here (see the class docstring),
    so the two-register/shared-matrix/no-cross-spin reading that ffsim's ``UCJOpSpinless`` also
    supports for a tuple ``nelec`` is not directly constructible. This confirms the same physics
    remains fully expressible via a ``UCJOpSpinBalanced`` with its alpha-beta block zeroed
    (aa == bb == the spinless matrix, no ab term), the equivalence this design decision relies on.
    """
    norb = 4
    nelec = (2, 2)
    ucj_op = ffsim.random.random_ucj_op_spinless(norb, n_reps=2, seed=55)
    reference = ffsim.hartree_fock_state(norb, nelec)
    expected = ffsim.apply_unitary(reference, ucj_op, norb=norb, nelec=nelec)

    zero_ab = np.zeros_like(ucj_op.diag_coulomb_mats)
    balanced_op = ffsim.UCJOpSpinBalanced(
        diag_coulomb_mats=np.stack([ucj_op.diag_coulomb_mats, zero_ab], axis=1),
        orbital_rotations=ucj_op.orbital_rotations,
        final_orbital_rotation=ucj_op.final_orbital_rotation,
    )
    result = ffsim.apply_unitary(reference, UCJ(balanced_op), norb=norb, nelec=nelec)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucj_spinless_on_spinless_sector_matches_ffsim():
    """The spinless UCJ gate applied to a true spinless sector (integer nelec) matches ffsim."""
    norb = 4
    nelec = 3
    ucj_op = ffsim.random.random_ucj_op_spinless(
        norb, n_reps=2, with_final_orbital_rotation=True, seed=77
    )
    reference = ffsim.slater_determinant(norb, list(range(nelec)))
    expected = ffsim.apply_unitary(reference, ucj_op, norb=norb, nelec=nelec)
    result = _apply_ours(ucj_op, norb, nelec, reference)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucj_gate_apply_unitary_matches_ffsim():
    """The bare gate's _apply_unitary_ applied to the HF reference matches ffsim's UCJ operator.

    UCJ is a pure unitary carrying no reference: the caller supplies the reference state, exactly as
    ffsim's own operator does.
    """
    norb = 4
    nelec = (2, 2)
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(
        norb, n_reps=2, with_final_orbital_rotation=True, seed=7
    )
    reference = ffsim.hartree_fock_state(norb, nelec)
    expected = ffsim.apply_unitary(reference, ucj_op, norb=norb, nelec=nelec)

    result = UCJ(ucj_op)._apply_unitary_(reference.copy(), norb, nelec, copy=True)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucj_gate_through_circuit_matches_ffsim():
    """A UCJ gate appended to a FermionicCircuit and driven by ffsim.apply_unitary matches ffsim."""
    norb = 4
    nelec = (2, 2)
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(norb, n_reps=2, seed=8)
    reference = ffsim.hartree_fock_state(norb, nelec)
    expected = ffsim.apply_unitary(reference, ucj_op, norb=norb, nelec=nelec)

    gate = UCJ(ucj_op)
    circ = FermionicCircuit(2 * norb)
    circ.append(gate, circ.modes)
    result = ffsim.apply_unitary(reference, circ, norb=norb, nelec=nelec)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucj_gate_subset_placement_matches_global_embedding():
    """A UCJ placed on a subset of a larger register is applied on its absolute (global) modes.

    A spinless UCJ acting on ``norb_local`` orbitals is appended to a ``norb_global``-orbital
    register on the non-identity mode subset ``placement``. The oracle is the *same* ansatz whose
    tensors are embedded into ``norb_global``-sized tensors on exactly those orbitals (identity
    rotations and zero diagonal Coulomb on the untouched orbitals), built and applied natively by
    ffsim. This guards the ``_apply_unitary_placed_`` routing: with the placement ignored, the gate
    would act on orbitals ``0..norb_local`` instead of ``placement`` and disagree.
    """
    norb_local = 2
    norb_global = 4
    nelec = 2  # spinless integer nelec on the global register
    placement = [1, 3]  # global orbitals the local UCJ modes map onto (non-identity)

    ucj_op = ffsim.random.random_ucj_op_spinless(
        norb_local, n_reps=2, with_final_orbital_rotation=True, seed=909
    )

    # embed the local tensors onto the global orbitals: identity rotations everywhere except the
    # placed rows/columns, zero diagonal Coulomb except the placed block
    embed = np.ix_(placement, placement)
    global_rotations = np.stack(
        [np.eye(norb_global, dtype=complex) for _ in ucj_op.orbital_rotations]
    )
    for k, rot in enumerate(ucj_op.orbital_rotations):
        global_rotations[k][embed] = rot
    global_diag_coulomb = np.zeros((len(ucj_op.diag_coulomb_mats), norb_global, norb_global))
    for k, mat in enumerate(ucj_op.diag_coulomb_mats):
        global_diag_coulomb[k][embed] = mat
    global_final = np.eye(norb_global, dtype=complex)
    global_final[embed] = ucj_op.final_orbital_rotation

    global_ucj = ffsim.UCJOpSpinless(
        diag_coulomb_mats=global_diag_coulomb,
        orbital_rotations=global_rotations,
        final_orbital_rotation=global_final,
    )
    # a reference determinant occupying exactly the placed orbitals; both the oracle and our path
    # transform this same state (UCJ is a pure unitary and carries no reference of its own)
    reference = ffsim.slater_determinant(norb_global, placement)
    expected = ffsim.apply_unitary(reference, global_ucj, norb=norb_global, nelec=nelec)

    # our path: the local UCJ gate, placed on the global orbitals via ``placement`` in the circuit,
    # applied to the same reference.
    gate = UCJ(ucj_op)
    circ = FermionicCircuit(norb_global)
    circ.append(gate, [circ.modes[p] for p in placement])
    result = ffsim.apply_unitary(reference, circ, norb=norb_global, nelec=nelec)

    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucj_single_rep_matches_hand_composed_ffsim_primitives():
    """A single-rep UCJ matches a reference composed directly from ffsim's gate primitives.

    This cross-check does not go through ``UCJOpSpinBalanced``: it applies ffsim's orbital-rotation
    and diagonal-Coulomb-evolution kernels by hand in the ``U exp(i J) U^dagger`` order, giving an
    independent oracle for the diagonal Coulomb operator convention the :class:`.UCJ` gate builds.
    """
    norb = 3
    nelec = (2, 1)

    ucj_op = ffsim.random.random_ucj_op_spin_balanced(norb, n_reps=1, seed=555)
    orbital_rotation = ucj_op.orbital_rotations[0]
    mat_aa, mat_ab = ucj_op.diag_coulomb_mats[0]

    reference = ffsim.hartree_fock_state(norb, nelec)

    # hand-composed reference: U^dagger, then exp(-i * (-1) * J), then U (i.e. U exp(i J) U^dagger)
    vec = ffsim.apply_orbital_rotation(reference, orbital_rotation.conj().T, norb=norb, nelec=nelec)
    vec = ffsim.apply_diag_coulomb_evolution(
        vec, (mat_aa, mat_ab, mat_aa), time=-1.0, norb=norb, nelec=nelec
    )
    expected = ffsim.apply_orbital_rotation(vec, orbital_rotation, norb=norb, nelec=nelec)

    result = _apply_ours(ucj_op, norb, nelec, reference)
    np.testing.assert_allclose(result, expected, atol=1e-10)
