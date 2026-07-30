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


def _apply_ours(ucj_op, norb, nelec, reference, variant):
    """Rebuilds an ffsim UCJ operator with our UCJ gate and applies it to ``reference``.

    Applies the :class:`.UCJ` gate itself (via ffsim's ``SupportsApplyUnitary`` protocol), so these
    correctness checks exercise the gate's public :meth:`.UCJ._apply_unitary_placed_` path -- which
    delegates to its ``_build_definition()`` -- rather than the definition circuit directly.
    """
    gate = UCJ(
        norb,
        variant,
        ucj_op.diag_coulomb_mats,
        ucj_op.orbital_rotations,
        final_orbital_rotation=ucj_op.final_orbital_rotation,
    )
    return ffsim.apply_unitary(reference, gate, norb=norb, nelec=nelec)


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
        result = _apply_ours(ucj_op, norb, nelec, reference, "balanced")
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
        result = _apply_ours(ucj_op, norb, nelec, reference, "unbalanced")
        np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucj_spinless_shortcut_via_zeroed_balanced_ab_block_matches_ffsim():
    """A spinless ffsim operator run on a spinful sector is reproduced via a zeroed-ab balanced gate.

    ``Variant.SPINLESS`` now always means a true 1-register spinless gate (see the class docstring),
    so the two-register/shared-matrix/no-cross-spin shortcut ffsim's ``UCJOpSpinless`` supports for a
    tuple ``nelec`` is no longer directly constructible. This confirms the same physics remains fully
    expressible via ``Variant.BALANCED`` with its alpha-beta block zeroed (aa == bb == the spinless
    matrix, no ab term) -- the equivalence this design decision relies on.
    """
    norb = 4
    nelec = (2, 2)
    ucj_op = ffsim.random.random_ucj_op_spinless(norb, n_reps=2, seed=55)
    reference = ffsim.hartree_fock_state(norb, nelec)
    expected = ffsim.apply_unitary(reference, ucj_op, norb=norb, nelec=nelec)

    zero_ab = np.zeros_like(ucj_op.diag_coulomb_mats)
    balanced_diag_coulomb_mats = np.stack([ucj_op.diag_coulomb_mats, zero_ab], axis=1)
    gate = UCJ(
        norb,
        "balanced",
        balanced_diag_coulomb_mats,
        ucj_op.orbital_rotations,
        final_orbital_rotation=ucj_op.final_orbital_rotation,
    )
    result = ffsim.apply_unitary(reference, gate, norb=norb, nelec=nelec)
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
    result = _apply_ours(ucj_op, norb, nelec, reference, "spinless")
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

    gate = UCJ(
        norb,
        "balanced",
        ucj_op.diag_coulomb_mats,
        ucj_op.orbital_rotations,
        final_orbital_rotation=ucj_op.final_orbital_rotation,
    )
    result = gate._apply_unitary_(reference.copy(), norb, nelec, copy=True)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucj_gate_through_circuit_matches_ffsim():
    """A UCJ gate appended to a FermionicCircuit and driven by ffsim.apply_unitary matches ffsim."""
    norb = 4
    nelec = (2, 2)
    ucj_op = ffsim.random.random_ucj_op_spin_balanced(norb, n_reps=2, seed=8)
    reference = ffsim.hartree_fock_state(norb, nelec)
    expected = ffsim.apply_unitary(reference, ucj_op, norb=norb, nelec=nelec)

    gate = UCJ(norb, "balanced", ucj_op.diag_coulomb_mats, ucj_op.orbital_rotations)
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
    gate = UCJ(
        norb_local,
        "spinless",
        ucj_op.diag_coulomb_mats,
        ucj_op.orbital_rotations,
        final_orbital_rotation=ucj_op.final_orbital_rotation,
    )
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

    result = _apply_ours(ucj_op, norb, nelec, reference, "balanced")
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
        result = ffsim.apply_unitary(reference, gate, norb=norb, nelec=nelec)
        np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucj_from_t_amplitudes_spinless_matches_ffsim():
    """UCJ.from_t_amplitudes (spinless) matches ffsim's UCJOpSpinless.from_t_amplitudes.

    ``variant="spinless"`` now always means a true 1-register spinless gate (see the class
    docstring), so this exercises a genuinely spinless (integer nelec) sector rather than ffsim's
    dropped two-register/shared-matrix shortcut for a spinful tuple nelec -- that equivalence is
    covered separately by
    ``test_ucj_spinless_shortcut_via_zeroed_balanced_ab_block_matches_ffsim``.
    """
    norb = 4
    nelec = 3
    rng = np.random.default_rng(99)
    nocc, nvrt = nelec, norb - nelec
    t2 = rng.standard_normal((nocc, nocc, nvrt, nvrt))
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)
    t1 = rng.standard_normal((nocc, nvrt))

    reference = ffsim.slater_determinant(norb, list(range(nelec)))
    ucj_op = ffsim.UCJOpSpinless.from_t_amplitudes(t2, t1=t1)
    expected = ffsim.apply_unitary(reference, ucj_op, norb=norb, nelec=nelec)

    gate = UCJ.from_t_amplitudes(nelec, t2, t1=t1, variant="spinless")
    result = ffsim.apply_unitary(reference, gate, norb=norb, nelec=nelec)
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
    result = ffsim.apply_unitary(reference, gate, norb=norb, nelec=nelec)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_ucj_from_t_amplitudes_unbalanced_with_same_spin_terms_matches_ffsim():
    """UCJ.from_t_amplitudes (unbalanced) matches ffsim when the aa/bb blocks factorize non-trivially.

    The OH/sto-3g system used above happens to have same-spin (aa/bb) blocks that factorize to zero
    terms, so it never exercises ``_assemble_same_spin_unbalanced``. Here we use synthetic UCCSD-shaped
    ``(t2aa, t2ab, t2bb)`` amplitudes with genuine same-spin correlation, which the double
    factorization turns into complex same-spin orbital rotations. The rotation buffer must be complex:
    truncating the imaginary part makes the rotations non-unitary and collapses the state norm.
    """
    rng = np.random.default_rng(1234)
    nocc_a, nocc_b, nvrt_a, nvrt_b = 2, 1, 2, 3
    norb = nocc_a + nvrt_a  # == nocc_b + nvrt_b == 4
    nelec = (nocc_a, nocc_b)

    def _rand(shape):
        return rng.standard_normal(shape) + 0j

    t2aa = _rand((nocc_a, nocc_a, nvrt_a, nvrt_a))
    t2ab = _rand((nocc_a, nocc_b, nvrt_a, nvrt_b))
    t2bb = _rand((nocc_b, nocc_b, nvrt_b, nvrt_b))
    # UCCSD same-spin amplitudes are antisymmetric under occupied and virtual exchange
    t2aa = t2aa - t2aa.transpose(1, 0, 2, 3)
    t2aa = t2aa - t2aa.transpose(0, 1, 3, 2)
    t2bb = t2bb - t2bb.transpose(1, 0, 2, 3)
    t2bb = t2bb - t2bb.transpose(0, 1, 3, 2)
    t2 = (t2aa, t2ab, t2bb)

    reference = ffsim.hartree_fock_state(norb, nelec)
    ucj_op = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(t2)
    expected = ffsim.apply_unitary(reference, ucj_op, norb=norb, nelec=nelec)

    gate = UCJ.from_t_amplitudes(nelec, t2, variant="unbalanced")
    result = ffsim.apply_unitary(reference, gate, norb=norb, nelec=nelec)

    np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-10)
    np.testing.assert_allclose(result, expected, atol=1e-10)
