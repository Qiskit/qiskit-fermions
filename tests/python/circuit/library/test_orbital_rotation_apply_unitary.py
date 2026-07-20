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

"""Tests for applying an OrbitalRotation gate to an ffsim state vector (SupportsApplyUnitary)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import scipy.linalg
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import OrbitalRotation

from ...utils import random_unitary

ffsim = pytest.importorskip("ffsim")


def _orbital_rotation_oracle(
    full: np.ndarray, norb: int, nelec: int | tuple[int, int], vec0: np.ndarray
) -> np.ndarray:
    """Independent exact-diagonalization reference for ``exp(G) |vec0>``.

    ``G = sum_ij log(full)_ij a†_i a_j`` is the generator of the orbital rotation ``full``. This
    builds ``G`` as a dense matrix in the FCI determinant basis, addressing each determinant with
    pyscf's ``cistring`` (the ordering ffsim/pyscf use), then exponentiates it exactly with
    ``scipy.linalg.expm``. It deliberately avoids both ffsim's Givens kernel and this repo's native
    FCI matvec, so it is a true cross-check rather than a tautology.

    For a spinful ``nelec`` the ``2 * norb`` modes follow the block-spin convention: modes
    ``0..norb`` are alpha orbitals, modes ``norb..2*norb`` are beta orbitals, and the flat state
    index is ``addr_a * dim_b + addr_b`` (alpha slow, beta fast).
    """
    from pyscf.fci import cistring

    log_mat = scipy.linalg.logm(full)

    if isinstance(nelec, int):
        num_modes = norb
        spin_sizes = [(norb, nelec)]
    else:
        num_modes = 2 * norb
        n_alpha, n_beta = nelec
        spin_sizes = [(norb, n_alpha), (norb, n_beta)]

    # enumerate determinants as (alpha occupation, beta occupation) products
    det_lists = [list(itertools.combinations(range(n), k)) for n, k in spin_sizes]
    dims = [len(d) for d in det_lists]
    dim = int(np.prod(dims))

    def sector_addr(occ, n, k):
        string = 0
        for orb in occ:
            string |= 1 << orb
        return cistring.str2addr(n, k, string)

    def to_spinorb_occ(sector_occs):
        # combine per-sector occupations into a single occupation set over ``num_modes`` modes
        occ = set(sector_occs[0])
        if len(sector_occs) == 2:
            occ |= {orb + norb for orb in sector_occs[1]}
        return occ

    def from_spinorb_occ(occ):
        # split a spin-orbital occupation set back into per-sector occupations; return None if the
        # per-sector electron counts do not match the target sector
        if len(spin_sizes) == 1:
            alpha = sorted(occ)
            if len(alpha) != spin_sizes[0][1]:
                return None
            return [alpha]
        alpha = sorted(o for o in occ if o < norb)
        beta = sorted(o - norb for o in occ if o >= norb)
        if len(alpha) != spin_sizes[0][1] or len(beta) != spin_sizes[1][1]:
            return None
        return [alpha, beta]

    def full_addr(sector_occs):
        if len(sector_occs) == 1:
            return sector_addr(sector_occs[0], *spin_sizes[0])
        addr_a = sector_addr(sector_occs[0], *spin_sizes[0])
        addr_b = sector_addr(sector_occs[1], *spin_sizes[1])
        return addr_a * dims[1] + addr_b

    def apply_ladder(occ, action, mode):
        occ = sorted(occ)
        if action:  # creation
            if mode in occ:
                return None, 0
            sign = (-1) ** sum(1 for o in occ if o < mode)
            return set([*occ, mode]), sign
        if mode not in occ:  # annihilation
            return None, 0
        sign = (-1) ** sum(1 for o in occ if o < mode)
        occ.remove(mode)
        return set(occ), sign

    all_dets = list(itertools.product(*det_lists))
    hamil = np.zeros((dim, dim), dtype=complex)
    for i in range(num_modes):
        for j in range(num_modes):
            coeff = log_mat[i, j]
            if coeff == 0.0:
                continue
            for sector_occs in all_dets:
                occ = to_spinorb_occ(sector_occs)
                # a†_i a_j : annihilate j (right), then create i (left)
                occ2, s1 = apply_ladder(occ, False, j)
                if occ2 is None:
                    continue
                occ3, s2 = apply_ladder(occ2, True, i)
                if occ3 is None:
                    continue
                out = from_spinorb_occ(occ3)
                if out is None:
                    continue
                hamil[full_addr(out), full_addr(sector_occs)] += coeff * s1 * s2

    return scipy.linalg.expm(hamil) @ vec0


def _block_diag(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """Assembles a block-spin orbital rotation from independent alpha/beta rotations."""
    norb = mat_a.shape[0]
    full = np.zeros((2 * norb, 2 * norb), dtype=complex)
    full[:norb, :norb] = mat_a
    full[norb:, norb:] = mat_b
    return full


def test_orbital_rotation_apply_unitary_spinful_matches_oracle():
    """A spin-balanced (single-matrix) rotation matches the independent exact-diagonalization oracle."""
    norb = 3
    nelec = (2, 1)
    rot = random_unitary(norb, seed=1)
    full = _block_diag(rot, rot)

    vec0 = ffsim.slater_determinant(norb, ([0, 1], [0]))
    vec0_before = vec0.copy()

    result = OrbitalRotation(full)._apply_unitary_(vec0, norb, nelec, copy=True)
    expected = _orbital_rotation_oracle(full, norb, nelec, vec0)

    np.testing.assert_allclose(result, expected, atol=1e-10)
    # copy=True must leave the input untouched
    np.testing.assert_array_equal(vec0, vec0_before)


def test_orbital_rotation_apply_unitary_spinful_independent_blocks_match_oracle():
    """Independent alpha/beta rotations match the exact-diagonalization oracle."""
    norb = 3
    nelec = (2, 1)
    rot_a = random_unitary(norb, seed=1)
    rot_b = random_unitary(norb, seed=2)
    full = _block_diag(rot_a, rot_b)

    vec0 = ffsim.slater_determinant(norb, ([0, 1], [0]))

    result = OrbitalRotation(full)._apply_unitary_(vec0, norb, nelec, copy=True)
    expected = _orbital_rotation_oracle(full, norb, nelec, vec0)

    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_orbital_rotation_apply_unitary_spinless_matches_oracle():
    """The spinless (integer nelec) path matches the exact-diagonalization oracle."""
    norb = 5
    nelec = 2  # spinless: C(5, 2) = 10 dimensional FCI space
    rot = random_unitary(norb, seed=3)

    rng = np.random.default_rng(0)
    vec0 = rng.standard_normal(10) + 1j * rng.standard_normal(10)
    vec0_before = vec0.copy()

    result = OrbitalRotation(rot)._apply_unitary_(vec0, norb, nelec, copy=True)
    expected = _orbital_rotation_oracle(rot, norb, nelec, vec0)

    np.testing.assert_allclose(result, expected, atol=1e-10)
    np.testing.assert_array_equal(vec0, vec0_before)


def test_orbital_rotation_apply_unitary_matches_ffsim():
    """The spinful and spinless paths agree with ffsim.apply_orbital_rotation directly."""
    norb = 3
    nelec = (2, 1)
    rot_a = random_unitary(norb, seed=1)
    rot_b = random_unitary(norb, seed=2)
    full = _block_diag(rot_a, rot_b)
    vec0 = ffsim.slater_determinant(norb, ([0, 1], [0]))

    result = OrbitalRotation(full)._apply_unitary_(vec0, norb, nelec, copy=True)
    expected = ffsim.apply_orbital_rotation(vec0.copy(), (rot_a, rot_b), norb=norb, nelec=nelec)
    np.testing.assert_allclose(result, expected, atol=1e-10)

    # spinless
    norb_s = 5
    nelec_s = 2
    rot = random_unitary(norb_s, seed=3)
    rng = np.random.default_rng(0)
    vec0_s = rng.standard_normal(10) + 1j * rng.standard_normal(10)
    result_s = OrbitalRotation(rot)._apply_unitary_(vec0_s, norb_s, nelec_s, copy=True)
    expected_s = ffsim.apply_orbital_rotation(vec0_s.copy(), rot, norb=norb_s, nelec=nelec_s)
    np.testing.assert_allclose(result_s, expected_s, atol=1e-10)


def test_orbital_rotation_apply_unitary_through_circuit_with_placement():
    """A subset-placed OrbitalRotation is embedded onto its global modes and applied correctly."""
    norb = 3
    nelec = (2, 1)
    rot = random_unitary(norb, seed=5)
    vec0 = ffsim.slater_determinant(norb, ([0, 1], [0]))

    # place the norb x norb rotation on the alpha modes [0, norb) of the 2*norb register
    circ_alpha = FermionicCircuit(2 * norb)
    circ_alpha.append(OrbitalRotation(rot), [circ_alpha.modes[i] for i in range(norb)])
    result_alpha = circ_alpha._apply_unitary_(vec0, norb, nelec, copy=True)
    expected_alpha = ffsim.apply_orbital_rotation(vec0.copy(), (rot, None), norb=norb, nelec=nelec)
    np.testing.assert_allclose(result_alpha, expected_alpha, atol=1e-10)

    # and on the beta modes [norb, 2*norb)
    circ_beta = FermionicCircuit(2 * norb)
    circ_beta.append(OrbitalRotation(rot), [circ_beta.modes[norb + i] for i in range(norb)])
    result_beta = circ_beta._apply_unitary_(vec0, norb, nelec, copy=True)
    expected_beta = ffsim.apply_orbital_rotation(vec0.copy(), (None, rot), norb=norb, nelec=nelec)
    np.testing.assert_allclose(result_beta, expected_beta, atol=1e-10)


def test_orbital_rotation_apply_unitary_via_ffsim_apply_unitary():
    """The public ffsim.apply_unitary entry point works on a FermionicCircuit."""
    norb = 3
    nelec = (2, 1)
    rot = random_unitary(norb, seed=5)
    vec0 = ffsim.slater_determinant(norb, ([0, 1], [0]))

    circ = FermionicCircuit(2 * norb)
    circ.append(OrbitalRotation(rot), [circ.modes[i] for i in range(norb)])

    result = ffsim.apply_unitary(vec0, circ, norb=norb, nelec=nelec)
    expected = ffsim.apply_orbital_rotation(vec0.copy(), (rot, None), norb=norb, nelec=nelec)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_orbital_rotation_apply_unitary_general_path_matches_fast_path(monkeypatch):
    """With ffsim disabled the native generator path matches the ffsim fast path."""
    import qiskit_fermions.circuit.library.orbital_rotation as orbital_rotation_module

    norb = 3
    nelec = (2, 1)
    full = _block_diag(random_unitary(norb, seed=1), random_unitary(norb, seed=2))
    vec0 = ffsim.slater_determinant(norb, ([0, 1], [0]))

    fast = OrbitalRotation(full)._apply_unitary_(vec0, norb, nelec, copy=True)

    monkeypatch.setattr(orbital_rotation_module, "HAS_FFSIM", False)
    general = OrbitalRotation(full)._apply_unitary_(vec0, norb, nelec, copy=True)

    np.testing.assert_allclose(general, fast, atol=1e-10)


def test_orbital_rotation_apply_unitary_rejects_spin_mixing(monkeypatch):
    """A spinful rotation mixing the alpha/beta sectors is rejected, with or without ffsim."""
    import qiskit_fermions.circuit.library.orbital_rotation as orbital_rotation_module

    norb = 3
    nelec = (2, 1)
    full = random_unitary(2 * norb, seed=7)  # dense: nonzero alpha/beta off-diagonal blocks
    vec0 = ffsim.slater_determinant(norb, ([0, 1], [0]))

    with pytest.raises(ValueError, match="mixes the alpha and beta spin sectors"):
        OrbitalRotation(full)._apply_unitary_(vec0, norb, nelec, copy=True)

    monkeypatch.setattr(orbital_rotation_module, "HAS_FFSIM", False)
    with pytest.raises(ValueError, match="mixes the alpha and beta spin sectors"):
        OrbitalRotation(full)._apply_unitary_(vec0, norb, nelec, copy=True)


def test_orbital_rotation_apply_unitary_accepts_block_diagonal_with_float_noise():
    """A block-diagonal rotation carrying only float round-off in its off-blocks is accepted.

    The spin-mixing check tolerates round-off so a genuinely block-diagonal rotation (e.g. built via
    ``expm`` of a block-diagonal generator) is not wrongly rejected. The result must still match the
    exact-diagonalization oracle for the intended block-diagonal rotation.
    """
    norb = 3
    nelec = (2, 1)
    full = _block_diag(random_unitary(norb, seed=1), random_unitary(norb, seed=2))
    # inject sub-tolerance noise into the nominally-zero alpha/beta off-blocks
    full[:norb, norb:] += 1e-14
    full[norb:, :norb] += 1e-14
    vec0 = ffsim.slater_determinant(norb, ([0, 1], [0]))

    result = OrbitalRotation(full)._apply_unitary_(vec0, norb, nelec, copy=True)
    # the oracle uses the full (noisy) matrix directly; the accepted result must agree with it
    expected = _orbital_rotation_oracle(full, norb, nelec, vec0)
    np.testing.assert_allclose(result, expected, atol=1e-10)
