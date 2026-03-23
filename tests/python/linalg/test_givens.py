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
from qiskit_fermions.linalg import givens_decomposition


def random_unitary(dim: int, *, seed=None, dtype=complex) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((dim, dim)).astype(dtype, copy=False)
    z += 1j * rng.standard_normal((dim, dim)).astype(dtype, copy=False)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    return q * (d / np.abs(d))


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
