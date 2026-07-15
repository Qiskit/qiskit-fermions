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

"""Integration test: an LUCJ ansatz built from the fermionic gate library, checked against ffsim.

The local unitary cluster Jastrow (LUCJ) ansatz combines all three simulation-capable gates in
:mod:`qiskit_fermions.circuit.library` -- :class:`.InitializeModes` (the Hartree-Fock reference),
:class:`.OrbitalRotation`, and :class:`.Evolution` (of a diagonal Coulomb operator) -- so it
exercises the whole ``SupportsApplyUnitary`` path end to end. ffsim's own
:class:`ffsim.UCJOpSpinBalanced` provides the ground truth: we reconstruct the *same* operator with
qiskit-fermions gates and require the resulting state vectors to agree.

A spin-balanced UCJ operator has the form

    |Psi> = [prod_k U_k exp(i J_k) U_k^dagger] (U_final) |HF>

where each ``U_k`` is an orbital rotation applied to both spin sectors and each ``J_k`` is the
diagonal Coulomb operator ``J = 1/2 sum_{ij, sigma tau} J^{sigma tau}_{ij} n_{i sigma} n_{j tau}``.
See ffsim ``python/ffsim/variational/ucj_spin_balanced.py`` for the reference decomposition.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution, InitializeModes, OrbitalRotation
from qiskit_fermions.operators import FermionOperator, ann, cre

ffsim = pytest.importorskip("ffsim")


def _block_diag(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """Assembles a block-spin orbital rotation from independent alpha/beta rotations."""
    norb = mat_a.shape[0]
    full = np.zeros((2 * norb, 2 * norb), dtype=complex)
    full[:norb, :norb] = mat_a
    full[norb:, norb:] = mat_b
    return full


def _diag_coulomb_operator(mat_aa: np.ndarray, mat_ab: np.ndarray, norb: int) -> FermionOperator:
    """Builds the diagonal Coulomb operator ``J`` as a :class:`.FermionOperator`.

    ``J = 1/2 sum_{ij, sigma tau} J^{sigma tau}_{ij} n_{i sigma} n_{j tau}`` in the block-spin mode
    convention (mode ``p`` is alpha orbital ``p``, mode ``norb + p`` is beta orbital ``p``). For a
    spin-balanced operator the beta-beta block equals ``mat_aa`` and the beta-alpha block equals
    ``mat_ab`` transposed; both are supplied as symmetric matrices. Each number operator ``n_m`` is
    the two-mode product ``a^dagger_m a_m``, so ``n_{i sigma} n_{j tau}`` is a single four-operator
    term. All such terms commute, so the resulting :class:`.Evolution` is exact.
    """
    # per-spin block matrices in the (alpha, beta) sector order
    blocks = {
        (0, 0): mat_aa,  # alpha-alpha
        (0, 1): mat_ab,  # alpha-beta
        (1, 0): mat_ab.T,  # beta-alpha
        (1, 1): mat_aa,  # beta-beta (spin-balanced)
    }

    terms: dict[tuple, complex] = {}
    for (sigma, tau), mat in blocks.items():
        offset_i = sigma * norb
        offset_j = tau * norb
        for i in range(norb):
            for j in range(norb):
                coeff = 0.5 * mat[i, j]
                if coeff == 0.0:
                    continue
                mode_i = offset_i + i
                mode_j = offset_j + j
                # n_{i sigma} n_{j tau} = a^dagger_{mode_i} a_{mode_i} a^dagger_{mode_j} a_{mode_j}
                term = (cre(mode_i), ann(mode_i), cre(mode_j), ann(mode_j))
                terms[term] = terms.get(term, 0.0) + coeff

    return FermionOperator.from_dict(terms)


def _hartree_fock_occupation(norb: int, nelec: tuple[int, int]) -> list[bool]:
    """Returns the block-spin occupation of the Hartree-Fock determinant for ``(norb, nelec)``."""
    n_alpha, n_beta = nelec
    occ = [False] * (2 * norb)
    for i in range(n_alpha):
        occ[i] = True
    for i in range(n_beta):
        occ[norb + i] = True
    return occ


def _lucj_circuit(
    ucj_op: ffsim.UCJOpSpinBalanced,
    norb: int,
    nelec: tuple[int, int],
    *,
    seed_reference: bool = True,
    group_diag_coulomb: bool = False,
) -> FermionicCircuit:
    """Translates an ffsim ``UCJOpSpinBalanced`` into an equivalent qiskit-fermions circuit.

    Builds ``[prod_k U_k exp(i J_k) U_k^dagger] (U_final) |HF>`` using :class:`.InitializeModes`,
    :class:`.OrbitalRotation`, and :class:`.Evolution` on ``2 * norb`` block-spin modes. When
    ``seed_reference`` is ``True`` the circuit opens with an :class:`.InitializeModes` that produces
    the Hartree-Fock reference; otherwise the caller is expected to supply the reference state.
    ``group_diag_coulomb`` assigns a single group index to each diagonal Coulomb operator so
    :class:`.Evolution` decomposes it group-by-group (exact, since the terms commute).
    """
    circ = FermionicCircuit(2 * norb)

    if seed_reference:
        circ.append(InitializeModes(_hartree_fock_occupation(norb, nelec)), circ.modes)

    for orbital_rotation, (mat_aa, mat_ab) in zip(
        ucj_op.orbital_rotations, ucj_op.diag_coulomb_mats, strict=True
    ):
        full = _block_diag(orbital_rotation, orbital_rotation)
        diag_coulomb = _diag_coulomb_operator(mat_aa, mat_ab, norb)
        if group_diag_coulomb and len(diag_coulomb):
            diag_coulomb.groups = [0] * len(diag_coulomb)

        # U_k^dagger, then exp(i J_k) == exp(-i * (-1) * J_k), then U_k. Gates are applied in append
        # order, so the block realizes U_k exp(i J_k) U_k^dagger acting on the incoming state.
        circ.append(OrbitalRotation(full.conj().T), circ.modes)
        circ.append(Evolution(2 * norb, diag_coulomb, time=-1.0), circ.modes)
        circ.append(OrbitalRotation(full), circ.modes)

    if ucj_op.final_orbital_rotation is not None:
        final = _block_diag(ucj_op.final_orbital_rotation, ucj_op.final_orbital_rotation)
        circ.append(OrbitalRotation(final), circ.modes)

    return circ


def test_lucj_matches_ffsim_ucj_spin_balanced():
    """The reconstructed LUCJ circuit reproduces ffsim's UCJOpSpinBalanced state vector."""
    norb = 4
    nelec = (2, 2)

    ucj_op = ffsim.random.random_ucj_op_spin_balanced(norb, n_reps=2, seed=1234)

    reference = ffsim.hartree_fock_state(norb, nelec)
    expected = ffsim.apply_unitary(reference, ucj_op, norb=norb, nelec=nelec)

    circ = _lucj_circuit(ucj_op, norb, nelec)
    # InitializeModes seeds the Hartree-Fock reference, so no incoming state is needed
    result = ffsim.apply_unitary(reference, circ, norb=norb, nelec=nelec)

    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_lucj_matches_ffsim_with_final_orbital_rotation():
    """The reconstruction also matches when the UCJ operator carries a final orbital rotation."""
    norb = 4
    nelec = (2, 2)

    ucj_op = ffsim.random.random_ucj_op_spin_balanced(
        norb, n_reps=2, with_final_orbital_rotation=True, seed=2024
    )

    reference = ffsim.hartree_fock_state(norb, nelec)
    expected = ffsim.apply_unitary(reference, ucj_op, norb=norb, nelec=nelec)

    circ = _lucj_circuit(ucj_op, norb, nelec)
    result = ffsim.apply_unitary(reference, circ, norb=norb, nelec=nelec)

    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_lucj_via_circuit_apply_unitary_seeds_from_none():
    """The DAG-walk path (with InitializeModes seeding from a None vector) agrees with ffsim."""
    norb = 4
    nelec = (2, 2)

    ucj_op = ffsim.random.random_ucj_op_spin_balanced(norb, n_reps=2, seed=99)

    reference = ffsim.hartree_fock_state(norb, nelec)
    expected = ffsim.apply_unitary(reference, ucj_op, norb=norb, nelec=nelec)

    circ = _lucj_circuit(ucj_op, norb, nelec)
    # InitializeModes accepts a None seed and produces the reference itself
    result = circ._apply_unitary_(None, norb, nelec, copy=True)

    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_lucj_grouped_diag_coulomb_matches_ffsim():
    """Grouping the diagonal Coulomb terms (group-by-group Evolution) is exact and matches ffsim."""
    norb = 4
    nelec = (2, 2)

    ucj_op = ffsim.random.random_ucj_op_spin_balanced(norb, n_reps=2, seed=7)

    reference = ffsim.hartree_fock_state(norb, nelec)
    expected = ffsim.apply_unitary(reference, ucj_op, norb=norb, nelec=nelec)

    circ = _lucj_circuit(ucj_op, norb, nelec, group_diag_coulomb=True)
    result = ffsim.apply_unitary(reference, circ, norb=norb, nelec=nelec)

    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_lucj_single_rep_matches_hand_composed_ffsim_primitives():
    """A single-rep LUCJ matches a reference composed directly from ffsim's gate primitives.

    This cross-check does not go through ``UCJOpSpinBalanced``: it applies the ffsim orbital-rotation
    and diagonal-Coulomb-evolution kernels by hand in the ``U exp(i J) U^dagger`` order, giving an
    independent oracle for the operator convention used by :func:`_diag_coulomb_operator`.
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

    circ = _lucj_circuit(ucj_op, norb, nelec)
    result = ffsim.apply_unitary(reference, circ, norb=norb, nelec=nelec)

    np.testing.assert_allclose(result, expected, atol=1e-10)
