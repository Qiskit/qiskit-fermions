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

"""Testing utilities."""

import itertools

import numpy as np


def random_unitary(dim: int, *, seed=None, dtype=complex) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((dim, dim)).astype(dtype, copy=False)
    z += 1j * rng.standard_normal((dim, dim)).astype(dtype, copy=False)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    return q * (d / np.abs(d))


def random_two_body_tensor(dim: int, *, rank: int | None = None, seed=None) -> np.ndarray:
    """Generates a random real two-body tensor with the standard two-body symmetries.

    Mirrors ``ffsim.random.random_two_body_tensor`` with ``dtype=float``: it builds the tensor as a
    sum of outer products of symmetric "Cholesky" matrices, which makes the reshaped
    ``(dim**2, dim**2)`` matrix real symmetric positive semidefinite and hence exactly
    double-factorizable by both the Cholesky and eigendecomposition paths.
    """
    rng = np.random.default_rng(seed)
    if rank is None:
        rank = dim * (dim + 1) // 2
    cholesky_vecs = rng.standard_normal((rank, dim, dim))
    cholesky_vecs += cholesky_vecs.transpose((0, 2, 1))
    return np.einsum("ipr,iqs->prqs", cholesky_vecs, cholesky_vecs)


def random_t2_amplitudes(norb: int, nocc: int, *, seed=None) -> np.ndarray:
    """Generates random spin-restricted ``t2`` amplitudes.

    Mirrors ``ffsim.random.random_t2_amplitudes`` with ``dtype=float``: the amplitudes satisfy the
    restricted-CCSD symmetry ``t2[i, j, a, b] == t2[j, i, b, a]``, which makes the corresponding
    reshaped matrix real symmetric and exactly representable by the explicit factorization.
    """
    rng = np.random.default_rng(seed)
    nvrt = norb - nocc
    t2 = np.zeros((nocc, nocc, nvrt, nvrt))
    pairs = itertools.product(range(nocc), range(nocc, norb))
    for (i, a), (j, b) in itertools.combinations_with_replacement(pairs, 2):
        val = rng.standard_normal()
        t2[i, j, a - nocc, b - nocc] = val
        t2[j, i, b - nocc, a - nocc] = val
    return t2
