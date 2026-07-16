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


def test_ucj_gate_accepts_numpy_int_nelec():
    """A UCJ built with a numpy-integer (e.g. ``np.int64``) spinless nelec matches the ``int`` build.

    A numpy scalar is not a Python ``int``, so a naive ``isinstance(nelec, int)`` check would infer
    the spinful variant, build the wrong (aa/ab/bb) diagonal-Coulomb layer, and crash inside ffsim's
    own ``isinstance(int)`` classification. The applied state must match the plain-``int`` build.
    """
    norb = 4
    ucj_op = ffsim.random.random_ucj_op_spinless(
        norb, n_reps=2, with_final_orbital_rotation=True, seed=5
    )

    def build(nelec):
        return UCJ(
            norb,
            nelec,
            ucj_op.diag_coulomb_mats,
            ucj_op.orbital_rotations,
            final_orbital_rotation=ucj_op.final_orbital_rotation,
        )

    gate_int = build(2)
    gate_np = build(np.int64(2))
    assert gate_int._spinless and gate_np._spinless
    assert gate_int._variant == gate_np._variant == "spinless"

    # the two builds are mathematically identical; they may differ only at floating-point rounding
    expected = gate_int._apply_unitary_(None, norb, 2, copy=True)
    result = gate_np._apply_unitary_(None, norb, np.int64(2), copy=True)
    np.testing.assert_allclose(result, expected, atol=1e-12)


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
    # the UCJ gate seeds its own reference determinant on the placed orbitals (both local modes
    # occupied -> global orbitals ``placement``), so the oracle must transform that same determinant
    reference = ffsim.slater_determinant(norb_global, placement)
    expected = ffsim.apply_unitary(reference, global_ucj, norb=norb_global, nelec=nelec)

    # our path: the local UCJ gate, placed on the global orbitals via ``placement`` in the circuit.
    # the gate seeds its own reference determinant on the placed modes, so pass ``None`` through.
    gate = UCJ(
        norb_local,
        nelec,
        ucj_op.diag_coulomb_mats,
        ucj_op.orbital_rotations,
        final_orbital_rotation=ucj_op.final_orbital_rotation,
        reference_occupation=[True, True],  # both local orbitals occupied -> global orbitals 1, 3
    )
    circ = FermionicCircuit(norb_global)
    circ.append(gate, [circ.modes[p] for p in placement])
    result = ffsim.apply_unitary(None, circ, norb=norb_global, nelec=nelec)

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
