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

# The tests in this module were adapted from the ffsim library
# (https://github.com/qiskit-community/ffsim, Apache-2.0, (C) IBM), specifically
# ``tests/python/linalg/double_factorized_decomposition_test.py``. Only the tests covering the
# decompositions ported here are included, and they are adapted to this package's API (which returns
# a list of per-term ``(Z, U)`` tuples rather than stacked arrays) and to ``numpy.einsum`` (in place
# of ``opt_einsum``). The tests exercising the ``optimize=True`` "compressed" code paths and the
# pyscf-based molecular fixtures from the original are intentionally not ported.

"""Tests for double factorization utilities."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from qiskit_fermions.linalg import (
    double_factorized,
    double_factorized_t2,
    double_factorized_t2_alpha_beta,
    modified_cholesky,
    reconstruct_t2,
    reconstruct_t2_alpha_beta,
)

from ..utils import random_t2_amplitudes, random_two_body_tensor, random_unitary

RNG = np.random.default_rng(139632037091916421993148931543991464292)


def _stack_terms(terms):
    """Stacks a list of ``(Z, U)`` terms into ``(diag_coulomb_mats, orbital_rotations)`` arrays."""
    if not terms:
        return np.empty((0,)), np.empty((0,))
    diag_coulomb_mats = np.array([z for z, _ in terms])
    orbital_rotations = np.array([u for _, u in terms])
    return diag_coulomb_mats, orbital_rotations


def _stack_alpha_beta_terms(terms):
    """Stacks alpha-beta terms into ``(n, 3, norb, norb)`` and ``(n, 2, norb, norb)`` arrays."""
    if not terms:
        return np.empty((0,)), np.empty((0,))
    diag_coulomb_mats = np.array([np.stack(z) for z, _ in terms])
    orbital_rotations = np.array([np.stack(u) for _, u in terms])
    return diag_coulomb_mats, orbital_rotations


def _reconstruct_two_body(terms) -> np.ndarray:
    """Reconstructs a two-body tensor from its double-factorized ``(Z, U)`` terms."""
    diag_coulomb_mats, orbital_rotations = _stack_terms(terms)
    return np.einsum(
        "kpi,kqi,kij,krj,ksj->pqrs",
        orbital_rotations,
        orbital_rotations,
        diag_coulomb_mats,
        orbital_rotations,
        orbital_rotations,
    )


@pytest.mark.parametrize("dim", range(6))
def test_modified_cholesky(dim: int):
    """Test modified Cholesky decomposition on a random tensor."""
    # construct a random positive definite matrix
    unitary = random_unitary(dim, seed=RNG)
    eigs = RNG.uniform(size=dim)
    mat = unitary @ np.diag(eigs) @ unitary.T.conj()
    cholesky_vecs = modified_cholesky(mat, 1e-8, None)
    reconstructed = np.einsum("ji,ki->jk", cholesky_vecs, cholesky_vecs.conj())
    np.testing.assert_allclose(reconstructed, mat, atol=1e-8)


@pytest.mark.parametrize("dim, cholesky", itertools.product(range(6), [False, True]))
def test_double_factorized_random(dim: int, cholesky: bool):
    """Test double-factorized decomposition on a random tensor."""
    two_body_tensor = random_two_body_tensor(dim, seed=RNG)
    terms = double_factorized(two_body_tensor, 1e-8, None, cholesky)
    if dim == 0:
        assert terms == []
        return
    reconstructed = _reconstruct_two_body(terms)
    np.testing.assert_allclose(reconstructed, two_body_tensor, atol=1e-8)


@pytest.mark.parametrize("cholesky", [True, False])
def test_double_factorized_tol_max_vecs(cholesky: bool):
    """Test double-factorized decomposition error threshold and max vecs."""
    dim = 5
    two_body_tensor = random_two_body_tensor(dim, seed=RNG)
    full = double_factorized(two_body_tensor, 1e-8, None, cholesky)

    # test max_vecs caps the number of returned terms
    max_vecs = 3
    terms = double_factorized(two_body_tensor, 1e-8, max_vecs, cholesky)
    assert len(terms) == max_vecs
    assert len(terms) <= len(full)

    # test that a looser tolerance discards terms
    loose = double_factorized(two_body_tensor, 1e-1, None, cholesky)
    assert len(loose) <= len(full)
    reconstructed = _reconstruct_two_body(loose) if loose else np.zeros_like(two_body_tensor)
    np.testing.assert_allclose(reconstructed, two_body_tensor, atol=1e-1)

    # test error threshold and max vecs together
    terms = double_factorized(two_body_tensor, 1e-1, max_vecs, cholesky)
    assert len(terms) <= max_vecs


@pytest.mark.parametrize("norb, nocc", [(4, 2), (5, 2), (5, 3)])
def test_double_factorized_t2_amplitudes_random(norb: int, nocc: int):
    """Test double factorization of random t2 amplitudes."""
    t2 = random_t2_amplitudes(norb, nocc, seed=RNG).astype(complex)
    terms = double_factorized_t2(t2, 1e-8, None)
    reconstructed = reconstruct_t2(terms, nocc=nocc)
    np.testing.assert_allclose(reconstructed, t2, atol=1e-8)

    diag_coulomb_mats, orbital_rotations = _stack_terms(terms)
    n_reps = len(terms)
    even_index = list(range(0, n_reps, 2))
    odd_index = list(range(1, n_reps, 2))
    np.testing.assert_allclose(
        diag_coulomb_mats[even_index], -diag_coulomb_mats[odd_index], atol=1e-8
    )
    np.testing.assert_allclose(
        orbital_rotations[even_index], orbital_rotations[odd_index].conj(), atol=1e-8
    )


def test_double_factorized_t2_tol_max_terms():
    """Test double-factorized t2 decomposition error threshold and max terms."""
    norb, nocc = 5, 2
    t2 = random_t2_amplitudes(norb, nocc, seed=RNG).astype(complex)
    full = double_factorized_t2(t2, 1e-8, None)

    # test max_terms caps the number of returned terms
    max_terms = 3
    terms = double_factorized_t2(t2, 1e-8, max_terms)
    assert len(terms) == max_terms
    assert len(terms) <= len(full)

    # test error threshold and max terms together
    terms = double_factorized_t2(t2, 1e-1, max_terms)
    assert len(terms) <= max_terms


def test_double_factorized_t2_alpha_beta_random():
    """Test double factorization of opposite-spin t2 amplitudes with random tensor."""
    shape = (3, 6, 7, 4)
    t2ab = RNG.standard_normal(shape)
    terms = double_factorized_t2_alpha_beta(t2ab.astype(complex), 1e-8, None)
    nocc_a, nocc_b, nvrt_a, _ = t2ab.shape
    norb = nocc_a + nvrt_a
    reconstructed = reconstruct_t2_alpha_beta(terms, norb, nocc_a, nocc_b)
    np.testing.assert_allclose(reconstructed, t2ab, atol=1e-8)

    diag_coulomb_mats, orbital_rotations = _stack_alpha_beta_terms(terms)
    n_reps = len(terms)
    index_0 = list(range(0, n_reps, 4))
    index_1 = list(range(1, n_reps, 4))
    index_2 = list(range(2, n_reps, 4))
    index_3 = list(range(3, n_reps, 4))

    np.testing.assert_allclose(diag_coulomb_mats[index_0], -diag_coulomb_mats[index_1], atol=1e-8)
    np.testing.assert_allclose(diag_coulomb_mats[index_0], -diag_coulomb_mats[index_2], atol=1e-8)
    np.testing.assert_allclose(diag_coulomb_mats[index_0], diag_coulomb_mats[index_3], atol=1e-8)

    np.testing.assert_allclose(
        orbital_rotations[index_0, 0], orbital_rotations[index_1, 0], atol=1e-8
    )
    np.testing.assert_allclose(
        orbital_rotations[index_0, 0], orbital_rotations[index_2, 0].conj(), atol=1e-8
    )
    np.testing.assert_allclose(
        orbital_rotations[index_0, 0], orbital_rotations[index_3, 0].conj(), atol=1e-8
    )


def test_double_factorized_t2_alpha_beta_tol_max_terms():
    """Test double-factorized alpha-beta t2 decomposition error threshold and max terms."""
    shape = (3, 6, 7, 4)
    t2ab = RNG.standard_normal(shape).astype(complex)
    full = double_factorized_t2_alpha_beta(t2ab, 1e-8, None)

    # test max_terms caps the number of returned terms
    max_terms = 5
    terms = double_factorized_t2_alpha_beta(t2ab, 1e-8, max_terms)
    assert len(terms) == max_terms
    assert len(terms) <= len(full)

    # test error threshold and max terms together
    terms = double_factorized_t2_alpha_beta(t2ab, 1e-1, max_terms)
    assert len(terms) <= max_terms
