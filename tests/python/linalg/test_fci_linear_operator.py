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

import numpy as np
import pytest
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


def test_fci_linear_operator_matvec_cross_spin_matches_ffsim():
    """A cross-spin (mixed alpha-and-beta) term matches ffsim for both matvec and rmatvec.

    The other oracle tests only exercise same-spin (alpha-only) hops. This covers the mixed path,
    where a term acts non-trivially on *both* spin blocks -- a spin exchange plus a cross-spin
    density-density -- and checks it against ffsim's linear operator and its adjoint.
    """
    norb, nelec = 3, (2, 1)
    # Block-spin modes: m < norb is alpha orbital m; m >= norb is beta orbital m - norb.
    hamil = FermionOperator.from_dict(
        {
            # spin exchange a^dag_{0a} a^dag_{2b} a_{0b} a_{2a}
            ((True, 0), (True, norb + 2), (False, norb + 0), (False, 2)): 0.6 - 0.3j,
            # cross-spin density-density n^a_1 n^b_1
            ((True, 1), (False, 1), (True, norb + 1), (False, norb + 1)): 1.1,
        }
    )
    ffsim_op = ffsim.FermionOperator(
        {
            (ffsim.cre_a(0), ffsim.cre_b(2), ffsim.des_b(0), ffsim.des_a(2)): 0.6 - 0.3j,
            (ffsim.cre_a(1), ffsim.des_a(1), ffsim.cre_b(1), ffsim.des_b(1)): 1.1,
        }
    )
    ff_linop = ffsim.linear_operator(ffsim_op, norb=norb, nelec=nelec)
    ours = hamil._linear_operator_(norb, nelec)

    dim = ff_linop.shape[0]
    rng = np.random.default_rng(3)
    vec = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    np.testing.assert_allclose(ours.matvec(vec), ff_linop.matvec(vec), atol=1e-12)
    np.testing.assert_allclose(ours.rmatvec(vec), ff_linop.rmatvec(vec), atol=1e-12)


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
