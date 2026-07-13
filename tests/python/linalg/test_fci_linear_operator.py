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

"""Tests for the native FCI ``LinearOperator`` returned by ``FermionOperator._linear_operator_``."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse.linalg as ssl
from qiskit_fermions.operators import FermionOperator

ffsim = pytest.importorskip("ffsim")


def test_fci_linear_operator_shape_and_dtype_spinful():
    """The wrapper reports the FCI sector dimension and a complex128 dtype."""
    op = FermionOperator.from_dict({((True, 0), (False, 1)): 1.0})
    linop = op._linear_operator_(3, (2, 1))  # C(3, 2) * C(3, 1) = 3 * 3 = 9
    assert linop.shape == (9, 9)
    assert linop.dtype == np.dtype("complex128")


def test_fci_linear_operator_shape_spinless():
    """A spinless (integer nelec) sector has dimension ``C(norb, nelec)``."""
    op = FermionOperator.from_dict({((True, 0), (False, 1)): 1.0})
    linop = op._linear_operator_(5, 2)  # C(5, 2) = 10
    assert linop.shape == (10, 10)


def test_fci_linear_operator_matvec_matches_ffsim():
    """The native spinful matvec matches ffsim's linear operator applied to the same vector."""
    hamil = FermionOperator.from_dict(
        {
            ((True, 0), (False, 1)): 1.0,
            ((True, 1), (False, 0)): 1.0,
        }
    )
    norb, nelec = 2, (1, 1)
    ffsim_op = ffsim.FermionOperator(
        {
            (ffsim.cre_a(0), ffsim.des_a(1)): 1.0,
            (ffsim.cre_a(1), ffsim.des_a(0)): 1.0,
        }
    )
    ff_linop = ffsim.linear_operator(ffsim_op, norb=norb, nelec=nelec)
    ours = hamil._linear_operator_(norb, nelec)

    rng = np.random.default_rng(1)
    vec = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    np.testing.assert_allclose(ours.matvec(vec), ff_linop.matvec(vec), atol=1e-12)


def test_fci_linear_operator_rmatvec_is_adjoint():
    """``rmatvec`` applies the operator's adjoint (``A.H @ v``)."""
    hamil = FermionOperator.from_dict(
        {
            ((True, 0), (False, 1)): 0.7 + 0.2j,
            ((True, 1), (False, 0)): 0.3 - 0.5j,
        }
    )
    norb, nelec = 2, (1, 1)
    linop = hamil._linear_operator_(norb, nelec)
    adjoint_linop = hamil.adjoint()._linear_operator_(norb, nelec)

    rng = np.random.default_rng(2)
    vec = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    # A.H @ v (rmatvec) must equal (A^dagger) @ v (the adjoint operator's matvec)
    np.testing.assert_allclose(linop.rmatvec(vec), adjoint_linop.matvec(vec), atol=1e-12)


def test_fci_linear_operator_expm_multiply_matches_ffsim_oracle():
    """Evolving via the native spinful operator matches an ffsim-built oracle."""
    hamil = FermionOperator.from_dict(
        {
            ((True, 0), (False, 1)): 1.0,
            ((True, 1), (False, 0)): 1.0,
        }
    )
    norb, nelec, time = 2, (1, 1), 0.37
    vec0 = ffsim.slater_determinant(norb, ([0], [0]))

    ffsim_op = ffsim.FermionOperator(
        {
            (ffsim.cre_a(0), ffsim.des_a(1)): 1.0,
            (ffsim.cre_a(1), ffsim.des_a(0)): 1.0,
        }
    )
    ff_linop = ffsim.linear_operator(ffsim_op, norb=norb, nelec=nelec)
    expected = ssl.expm_multiply(-1j * time * ff_linop, vec0, traceA=0.0)

    ours = hamil._linear_operator_(norb, nelec)
    result = ssl.expm_multiply(-1j * time * ours, vec0, traceA=0.0)

    np.testing.assert_allclose(result, expected, atol=1e-10)


def _spinless_evolution_oracle(terms, norb, nelec, time, vec0):
    """Independent exact-diagonalization reference for a spinless ``exp(-i t H) |vec0>``.

    Mirrors the oracle in ``tests/python/circuit/library/test_evolution_apply_unitary.py``: builds
    ``H`` densely in the ``C(norb, nelec)`` determinant basis addressed by pyscf's ``cistring`` (the
    ordering the native kernel targets) and exponentiates it exactly.
    """
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
        if mode not in occ:  # annihilation
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


def test_fci_linear_operator_expm_multiply_spinless_matches_exact_diagonalization():
    """The native spinless (integer nelec) path matches an independent exact-diagonalization oracle."""
    pytest.importorskip("pyscf")

    norb = 5
    nelec = 2  # spinless: C(5, 2) = 10
    time = 0.37

    terms = {
        ((True, 0), (False, 1)): 0.7 + 0.2j,
        ((True, 1), (False, 0)): 0.7 - 0.2j,
        ((True, 2), (False, 2)): 0.5,
        ((True, 0), (False, 0), (True, 3), (False, 3)): 1.3,
    }
    hamil = FermionOperator.from_dict(terms)

    rng = np.random.default_rng(0)
    vec0 = rng.standard_normal(10) + 1j * rng.standard_normal(10)

    expected = _spinless_evolution_oracle(terms, norb, nelec, time, vec0)

    ours = hamil._linear_operator_(norb, nelec)
    result = ssl.expm_multiply(-1j * time * ours, vec0, traceA=0.0)

    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_fci_linear_operator_matvec_rejects_wrong_length():
    """A state vector whose length does not match the sector dimension raises ValueError."""
    op = FermionOperator.from_dict({((True, 0), (False, 1)): 1.0})
    linop = op._linear_operator_(2, (1, 1))  # dim 4
    with pytest.raises(ValueError):
        linop.matvec(np.ones(3, dtype=complex))


def test_fci_linear_operator_rejects_too_many_orbitals():
    """A norb beyond the bitmask limit raises a catchable ValueError, not a Rust panic."""
    op = FermionOperator.from_dict({((True, 0), (False, 1)): 1.0})
    with pytest.raises(ValueError, match="exceeds the maximum"):
        op._linear_operator_(65, 2)
