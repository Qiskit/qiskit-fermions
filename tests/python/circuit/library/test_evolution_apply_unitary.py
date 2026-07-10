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

"""Tests for applying an Evolution gate to an ffsim state vector (SupportsApplyUnitary)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import Evolution
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.operators.library import FCIDump

ffsim = pytest.importorskip("ffsim")
scipy_sparse_linalg = pytest.importorskip("scipy.sparse.linalg")


def test_evolution_apply_unitary_matches_ffsim_oracle():
    """The evolution matches an oracle built from ffsim's own FermionOperator."""
    # hopping a†_0 a_1 + a†_1 a_0 on (alpha) orbitals 0 and 1
    hamil = FermionOperator.from_dict(
        {
            ((True, 0), (False, 1)): 1.0,
            ((True, 1), (False, 0)): 1.0,
        }
    )
    norb = 2
    nelec = (1, 1)
    time = 0.37

    vec0 = ffsim.slater_determinant(norb, ([0], [0]))

    # oracle: identical operator built natively with ffsim helpers, evolved via linear_operator
    ffsim_op = ffsim.FermionOperator(
        {
            (ffsim.cre_a(0), ffsim.des_a(1)): 1.0,
            (ffsim.cre_a(1), ffsim.des_a(0)): 1.0,
        }
    )
    linop = ffsim.linear_operator(ffsim_op, norb=norb, nelec=nelec)
    expected = scipy_sparse_linalg.expm_multiply(-1j * time * linop, vec0, traceA=0.0)

    vec0_before = vec0.copy()
    result = Evolution(2, hamil, time=time)._apply_unitary_(vec0, norb, nelec, copy=True)

    np.testing.assert_allclose(result, expected, atol=1e-10)
    # copy=True must leave the input untouched
    np.testing.assert_array_equal(vec0, vec0_before)


def test_evolution_apply_unitary_through_circuit_with_placement():
    """A subset-placed Evolution gate is relabeled to global modes and evolved correctly."""
    norb = 2
    nelec = (1, 1)
    time = 0.37

    # local hopping on the gate's own modes 0 and 1
    local_hamil = FermionOperator.from_dict(
        {
            ((True, 0), (False, 1)): 1.0,
            ((True, 1), (False, 0)): 1.0,
        }
    )
    # place it on global modes [0, 1] (both alpha orbitals)
    circ = FermionicCircuit(2 * norb)
    circ.append(Evolution(2, local_hamil, time=time), [circ.modes[0], circ.modes[1]])

    vec0 = ffsim.slater_determinant(norb, ([0], [0]))

    # oracle on the equivalent global operator
    ffsim_op = ffsim.FermionOperator(
        {
            (ffsim.cre_a(0), ffsim.des_a(1)): 1.0,
            (ffsim.cre_a(1), ffsim.des_a(0)): 1.0,
        }
    )
    linop = ffsim.linear_operator(ffsim_op, norb=norb, nelec=nelec)
    expected = scipy_sparse_linalg.expm_multiply(-1j * time * linop, vec0, traceA=0.0)

    result = circ._apply_unitary_(vec0, norb, nelec, copy=True)

    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_evolution_apply_unitary_via_ffsim_apply_unitary():
    """The public ffsim.apply_unitary entry point works on a FermionicCircuit."""
    norb = 2
    nelec = (1, 1)
    time = 0.37

    hamil = FermionOperator.from_dict(
        {
            ((True, 0), (False, 1)): 1.0,
            ((True, 1), (False, 0)): 1.0,
        }
    )
    circ = FermionicCircuit(2 * norb)
    circ.append(Evolution(2 * norb, hamil, time=time), circ.modes)

    vec0 = ffsim.slater_determinant(norb, ([0], [0]))

    ffsim_op = ffsim.FermionOperator(
        {
            (ffsim.cre_a(0), ffsim.des_a(1)): 1.0,
            (ffsim.cre_a(1), ffsim.des_a(0)): 1.0,
        }
    )
    linop = ffsim.linear_operator(ffsim_op, norb=norb, nelec=nelec)
    expected = scipy_sparse_linalg.expm_multiply(-1j * time * linop, vec0, traceA=0.0)

    result = ffsim.apply_unitary(vec0, circ, norb=norb, nelec=nelec)

    np.testing.assert_allclose(result, expected, atol=1e-10)


def _spinless_evolution_oracle(terms, norb, nelec, time, vec0):
    """Independent exact-diagonalization reference for a spinless evolution ``exp(-i t H) |vec0>``.

    Builds ``H`` as a dense matrix in the ``C(norb, nelec)`` determinant basis, addressing each
    determinant with pyscf's ``cistring`` (the same ordering ffsim/pyscf use for the state vector),
    then exponentiates it exactly with ``scipy.linalg.expm``. This deliberately avoids ffsim's own
    ``FermionOperator`` machinery so it is a true cross-check of the spinless path rather than a
    tautology.
    """
    import itertools

    import scipy.linalg
    from pyscf.fci import cistring

    def addr(occ):
        string = 0
        for orb in occ:
            string |= 1 << orb
        return cistring.str2addr(norb, nelec, string)

    def apply_ladder(occ, action, mode):
        occ = list(occ)
        if action:  # creation
            if mode in occ:
                return None, 0
            sign = (-1) ** sum(1 for o in occ if o < mode)
            return tuple(sorted([*occ, mode])), sign
        # annihilation
        if mode not in occ:
            return None, 0
        sign = (-1) ** sum(1 for o in occ if o < mode)
        occ.remove(mode)
        return tuple(occ), sign

    dets = list(itertools.combinations(range(norb), nelec))
    dim = len(dets)
    hamil_mat = np.zeros((dim, dim), dtype=complex)
    for term, coeff in terms.items():
        for det in dets:
            occ, sign, ok = det, 1, True
            for action, mode in reversed(term):  # ladder ops act right-to-left
                occ, s = apply_ladder(occ, action, mode)
                if occ is None:
                    ok = False
                    break
                sign *= s
            if ok:
                hamil_mat[addr(occ), addr(det)] += coeff * sign

    return scipy.linalg.expm(-1j * time * hamil_mat) @ vec0


def test_evolution_apply_unitary_spinless_matches_exact_diagonalization():
    """The spinless (integer nelec) path matches an independent exact-diagonalization oracle."""
    pytest.importorskip("pyscf")

    norb = 5
    nelec = 2  # spinless: integer nelec -> C(5, 2) = 10 dimensional FCI space
    time = 0.37

    # a particle-conserving spinless operator (hopping + number + density-density)
    terms = {
        ((True, 0), (False, 1)): 0.7 + 0.2j,
        ((True, 1), (False, 0)): 0.7 - 0.2j,  # hermitian conjugate of the hop above
        ((True, 2), (False, 2)): 0.5,
        ((True, 0), (False, 0), (True, 3), (False, 3)): 1.3,
    }
    hamil = FermionOperator.from_dict(terms)

    rng = np.random.default_rng(0)
    dim = 10
    vec0 = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)

    expected = _spinless_evolution_oracle(terms, norb, nelec, time, vec0)

    vec0_before = vec0.copy()
    result = Evolution(norb, hamil, time=time)._apply_unitary_(vec0, norb, nelec, copy=True)

    np.testing.assert_allclose(result, expected, atol=1e-10)
    # copy=True must leave the input untouched
    np.testing.assert_array_equal(vec0, vec0_before)


def test_ffsim_operator_conversion_rejects_spin_nonconserving():
    """ffsim's guard rejects operators that do not conserve particle number and spin-z.

    The evolution path now applies the operator through its own native FCI kernel
    (``FermionOperator._linear_operator_``), which -- unlike ``ffsim.linear_operator`` -- does not
    guard sector conservation: a spin-non-conserving term maps amplitude out of the fixed
    ``(norb, nelec)`` sector and is silently dropped rather than raising. This guard therefore only
    remains where conversion still goes through ffsim's *own* ``FermionOperator`` data structure, so
    that is what this test exercises.
    """
    from qiskit_fermions.circuit.library._ffsim import to_ffsim_operator

    norb = 2
    nelec = (1, 1)
    # cre_a(0) des_b(0): moves an electron from the beta to the alpha sector -> not Sz-conserving
    hamil = FermionOperator.from_dict({((True, 0), (False, 2)): 1.0})

    ffsim_op = to_ffsim_operator(hamil, norb, nelec)
    with pytest.raises(ValueError):
        ffsim.linear_operator(ffsim_op, norb=norb, nelec=nelec)


def test_evolution_apply_unitary_matches_ffsim_molecular_hamiltonian():
    """Evolving under a full FCIDump Hamiltonian matches ffsim's MolecularHamiltonian path."""
    fcidump_file = str(Path(__file__).parent / "../../../h2.fcidump")

    fcidump = FCIDump.from_file(fcidump_file)
    norb = fcidump.norb
    nelec = (1, 1)
    time = 1.0

    # our path: FermionOperator.from_fcidump -> Evolution -> ffsim.apply_unitary
    hamil = FermionOperator.from_fcidump(fcidump)
    circ = FermionicCircuit(2 * norb)
    circ.append(Evolution(2 * norb, hamil, time=time), circ.modes)

    initial = ffsim.hartree_fock_state(norb, nelec)
    result = ffsim.apply_unitary(initial, circ, norb=norb, nelec=nelec)

    # reference: load the same FCIDump into ffsim's native MolecularHamiltonian and evolve directly
    mol_hamiltonian = ffsim.MolecularData.from_fcidump(fcidump_file).hamiltonian
    linop = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
    expected = scipy_sparse_linalg.expm_multiply(-1j * time * linop, initial, traceA=0.0)

    np.testing.assert_allclose(result, expected, atol=1e-10)
