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

"""Test the Givens decomposition."""

import numpy as np
import pytest
from qiskit_fermions.linalg import givens_decomposition, givens_decomposition_slater

from ..utils import random_unitary


@pytest.mark.parametrize("dim", range(1, 11))
def test_givens_decomposition(dim: int):
    unitary = random_unitary(dim)

    givens_rotations, phase_shifts = givens_decomposition(unitary)
    reconstructed = np.diag(phase_shifts)
    for c, s, i, j in givens_rotations[::-1]:
        givens_mat = np.eye(dim, dtype=complex)
        givens_mat[np.ix_((i, j), (i, j))] = [
            [c, s],
            [-s.conjugate(), c],
        ]
        reconstructed @= givens_mat.conj()

    np.testing.assert_allclose(unitary, reconstructed)


def _reconstruct_slater(rotations, m: int, n: int) -> np.ndarray:
    """Reconstruct the occupied orbitals by applying the rotations to ``[I_m | 0]``."""
    reconstructed = np.eye(m, n, dtype=complex)
    for c, s, i, j in rotations:
        col_i = reconstructed[:, i].copy()
        col_j = reconstructed[:, j].copy()
        reconstructed[:, i] = c * col_i + s.conjugate() * col_j
        reconstructed[:, j] = c * col_j - s * col_i
    return reconstructed


@pytest.mark.parametrize("norb, nocc", [(6, 3), (7, 2), (5, 4), (4, 4), (5, 1), (8, 4)])
def test_givens_decomposition_slater(norb: int, nocc: int):
    # occupied-orbital coefficients: nocc rows of a random unitary, in a basis of norb orbitals
    coeffs = random_unitary(norb, seed=norb * 100 + nocc).T[list(range(nocc))]
    max_full = nocc * (norb - nocc)

    rotations = givens_decomposition_slater(coeffs)

    # diamond-pattern bound and adjacency
    assert len(rotations) <= max_full
    assert all(abs(i - j) == 1 for _, _, i, j in rotations)

    # the reconstructed occupied space matches the target Slater determinant (fidelity 1)
    reconstructed = _reconstruct_slater(rotations, nocc, norb)
    overlap = abs(np.linalg.det(reconstructed @ coeffs.conj().T)) ** 2
    assert overlap == pytest.approx(1.0)
