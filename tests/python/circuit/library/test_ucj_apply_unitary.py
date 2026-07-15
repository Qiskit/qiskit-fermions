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

Each spin variant is validated against ffsim's own ``UCJOpSpin{Balanced,Unbalanced,less}``: we take
the tensors from a random ffsim UCJ operator, rebuild the ansatz with our :class:`.UCJ` gate, and
require the resulting state vectors to agree. A separate group of tests validates
:meth:`.UCJ.from_t_amplitudes` against ffsim's ``from_t_amplitudes`` (exact factorization).
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import UCJ

ffsim = pytest.importorskip("ffsim")


def _apply_ours(ucj_op, norb, nelec, reference):
    """Rebuilds an ffsim UCJ operator with our UCJ gate and applies it to ``reference``."""
    gate = UCJ(
        norb,
        nelec,
        ucj_op.diag_coulomb_mats,
        ucj_op.orbital_rotations,
        final_orbital_rotation=ucj_op.final_orbital_rotation,
    )
    return ffsim.apply_unitary(reference, gate._build_definition(), norb=norb, nelec=nelec)


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


def test_ucj_spinless_on_spinful_sector_matches_ffsim():
    """The spinless UCJ gate applied to a spinful sector matches ffsim."""
    norb = 4
    nelec = (2, 2)
    ucj_op = ffsim.random.random_ucj_op_spinless(norb, n_reps=2, seed=55)
    reference = ffsim.hartree_fock_state(norb, nelec)
    expected = ffsim.apply_unitary(reference, ucj_op, norb=norb, nelec=nelec)
    result = _apply_ours(ucj_op, norb, nelec, reference)
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


def test_ucj_gate_apply_unitary_seeds_from_none():
    """The bare gate's _apply_unitary_ seeds the reference from a None vector and matches ffsim."""
    norb = 4
    nelec = (2, 2)
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(
        norb, n_reps=2, with_final_orbital_rotation=True, seed=7
    )
    reference = ffsim.hartree_fock_state(norb, nelec)
    expected = ffsim.apply_unitary(reference, ucj_op, norb=norb, nelec=nelec)

    gate = UCJ(
        norb,
        nelec,
        ucj_op.diag_coulomb_mats,
        ucj_op.orbital_rotations,
        final_orbital_rotation=ucj_op.final_orbital_rotation,
    )
    result = gate._apply_unitary_(None, norb, nelec, copy=True)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucj_gate_through_circuit_matches_ffsim():
    """A UCJ gate appended to a FermionicCircuit and driven by ffsim.apply_unitary matches ffsim."""
    norb = 4
    nelec = (2, 2)
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(norb, n_reps=2, seed=8)
    reference = ffsim.hartree_fock_state(norb, nelec)
    expected = ffsim.apply_unitary(reference, ucj_op, norb=norb, nelec=nelec)

    gate = UCJ(norb, nelec, ucj_op.diag_coulomb_mats, ucj_op.orbital_rotations)
    circ = FermionicCircuit(2 * norb)
    circ.append(gate, circ.modes)
    result = ffsim.apply_unitary(reference, circ, norb=norb, nelec=nelec)
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


def test_ucj_from_t_amplitudes_balanced_matches_ffsim():
    """UCJ.from_t_amplitudes (balanced) matches ffsim's from_t_amplitudes, including LUCJ locality."""
    pyscf = pytest.importorskip("pyscf")
    import pyscf.cc as _pyscf_cc

    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["H", (0, 0, 0)], ["H", (0, 0, 0.74)]], basis="6-31g", symmetry="Dooh", verbose=0
    )
    scf = pyscf.scf.RHF(mol).run()
    mol_data = ffsim.MolecularData.from_scf(scf)
    norb, nelec = mol_data.norb, mol_data.nelec
    ccsd = _pyscf_cc.CCSD(scf).run()
    t1, t2 = ccsd.t1, ccsd.t2

    reference = ffsim.hartree_fock_state(norb, nelec)
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]

    for kwargs_ffsim, kwargs_ours in [
        ({}, {}),
        (
            {"n_reps": 2, "interaction_pairs": (pairs_aa, pairs_ab)},
            {"n_reps": 2, "interaction_pairs": (pairs_aa, pairs_ab)},
        ),
    ]:
        ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, t1=t1, **kwargs_ffsim)
        expected = ffsim.apply_unitary(reference, ucj_op, norb=norb, nelec=nelec)

        gate = UCJ.from_t_amplitudes(nelec, t2, t1=t1, variant="balanced", **kwargs_ours)
        result = ffsim.apply_unitary(reference, gate._build_definition(), norb=norb, nelec=nelec)
        np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucj_from_t_amplitudes_spinless_matches_ffsim():
    """UCJ.from_t_amplitudes (spinless) matches ffsim's UCJOpSpinless.from_t_amplitudes."""
    pyscf = pytest.importorskip("pyscf")
    import pyscf.cc as _pyscf_cc

    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["H", (0, 0, 0)], ["H", (0, 0, 0.74)]], basis="6-31g", symmetry="Dooh", verbose=0
    )
    scf = pyscf.scf.RHF(mol).run()
    mol_data = ffsim.MolecularData.from_scf(scf)
    norb, nelec = mol_data.norb, mol_data.nelec
    ccsd = _pyscf_cc.CCSD(scf).run()

    reference = ffsim.hartree_fock_state(norb, nelec)
    ucj_op = ffsim.UCJOpSpinless.from_t_amplitudes(ccsd.t2, t1=ccsd.t1)
    expected = ffsim.apply_unitary(reference, ucj_op, norb=norb, nelec=nelec)

    gate = UCJ.from_t_amplitudes(nelec, ccsd.t2, t1=ccsd.t1, variant="spinless")
    result = ffsim.apply_unitary(reference, gate._build_definition(), norb=norb, nelec=nelec)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucj_from_t_amplitudes_unbalanced_matches_ffsim():
    """UCJ.from_t_amplitudes (unbalanced) matches ffsim's UCJOpSpinUnbalanced.from_t_amplitudes."""
    pyscf = pytest.importorskip("pyscf")
    import pyscf.cc as _pyscf_cc

    mol = pyscf.gto.Mole()
    mol.build(atom=[["O", (0, 0, 0)], ["H", (0, 0, 0.97)]], basis="sto-3g", spin=1, verbose=0)
    scf = pyscf.scf.UHF(mol).run()
    ccsd = _pyscf_cc.UCCSD(scf).run()
    norb = scf.mo_coeff[0].shape[1]
    nelec = ((mol.nelectron + mol.spin) // 2, (mol.nelectron - mol.spin) // 2)

    reference = ffsim.hartree_fock_state(norb, nelec)
    ucj_op = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(ccsd.t2, t1=ccsd.t1)
    expected = ffsim.apply_unitary(reference, ucj_op, norb=norb, nelec=nelec)

    gate = UCJ.from_t_amplitudes(nelec, ccsd.t2, t1=ccsd.t1, variant="unbalanced")
    result = ffsim.apply_unitary(reference, gate._build_definition(), norb=norb, nelec=nelec)
    np.testing.assert_allclose(result, expected, atol=1e-10)
