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

"""Tests for seeding an InitializeModes gate into an ffsim state vector (SupportsApplyUnitary)."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import InitializeModes, OrbitalRotation

from ...utils import random_unitary

ffsim = pytest.importorskip("ffsim")


def test_initialize_modes_apply_unitary_spinful_matches_ffsim():
    """A spinful occupation seeds the same determinant as ffsim.slater_determinant."""
    norb = 3
    nelec = (2, 1)
    # block-spin occupation: alpha orbitals 0,1 (modes 0,1); beta orbital 0 (mode norb + 0)
    occupation = [True, True, False, True, False, False]

    result = InitializeModes(occupation)._apply_unitary_(None, norb, nelec, copy=True)
    expected = ffsim.slater_determinant(norb, ([0, 1], [0]))

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_initialize_modes_apply_unitary_spinless_matches_ffsim():
    """A spinless occupation seeds the same determinant as ffsim.slater_determinant."""
    norb = 5
    nelec = 2
    occupation = [True, False, True, False, False]  # orbitals 0 and 2 occupied

    result = InitializeModes(occupation)._apply_unitary_(None, norb, nelec, copy=True)
    expected = ffsim.slater_determinant(norb, [0, 2])

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_initialize_modes_apply_unitary_accepts_numpy_int_nelec():
    """A numpy-integer (e.g. ``np.int64``) spinless nelec is treated as spinless, matching ``int``.

    A numpy scalar is not a Python ``int``, so a naive ``isinstance(nelec, int)`` check would route
    it to the spinful branch and fail to unpack it as a ``(n_alpha, n_beta)`` pair. The seed must
    match the plain-``int`` spinless path.
    """
    norb = 5
    occupation = [True, False, True, False, False]  # orbitals 0 and 2 occupied

    expected = InitializeModes(occupation)._apply_unitary_(None, norb, 2, copy=True)
    result = InitializeModes(occupation)._apply_unitary_(None, norb, np.int64(2), copy=True)

    np.testing.assert_array_equal(result, expected)


def test_initialize_modes_apply_unitary_vec_none_seeds_from_nothing():
    """The vec=None path produces the determinant for both spinful and spinless systems."""
    # spinful
    norb = 3
    nelec = (2, 1)
    occupation = [True, True, False, True, False, False]
    result = InitializeModes(occupation)._apply_unitary_(None, norb, nelec, copy=False)
    np.testing.assert_allclose(result, ffsim.slater_determinant(norb, ([0, 1], [0])), atol=1e-12)

    # spinless
    norb_s = 4
    nelec_s = 2
    occ_s = [True, False, False, True]
    result_s = InitializeModes(occ_s)._apply_unitary_(None, norb_s, nelec_s, copy=False)
    np.testing.assert_allclose(result_s, ffsim.slater_determinant(norb_s, [0, 3]), atol=1e-12)


def test_initialize_modes_apply_unitary_agreeing_vec_logs_and_seeds(caplog):
    """A non-None vec that agrees with the occupation's sector is accepted, logged, and overwritten."""
    norb = 3
    nelec = (2, 1)
    occupation = [True, True, False, True, False, False]
    expected = ffsim.slater_determinant(norb, ([0, 1], [0]))

    # a real, correctly sized (but different) vector in the same sector
    rng = np.random.default_rng(0)
    incoming = rng.standard_normal(len(expected)) + 1j * rng.standard_normal(len(expected))
    incoming_before = incoming.copy()

    with caplog.at_level(logging.INFO, logger="qiskit_fermions.circuit.library.initialize_modes"):
        result = InitializeModes(occupation)._apply_unitary_(incoming, norb, nelec, copy=True)

    np.testing.assert_allclose(result, expected, atol=1e-12)
    # the incoming amplitudes are replaced by the determinant, and the input is left untouched
    np.testing.assert_array_equal(incoming, incoming_before)
    assert any("agrees with the occupation" in rec.message for rec in caplog.records)


def test_initialize_modes_apply_unitary_through_circuit_with_placement():
    """A subset-placed InitializeModes seeds onto its global modes (straddling alpha and beta)."""
    norb = 3
    nelec = (1, 1)

    # a single gate placed on a subset of the 2*norb register: its two local modes map onto global
    # alpha mode 2 and global beta mode (norb + 1), so the seeded determinant is ([2], [1])
    circ = FermionicCircuit(2 * norb)
    circ.append(InitializeModes([True, True]), [circ.modes[2], circ.modes[norb + 1]])

    result = circ._apply_unitary_(None, norb, nelec, copy=True)
    expected = ffsim.slater_determinant(norb, ([2], [1]))

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_initialize_modes_apply_unitary_end_to_end_full_circuit():
    """Seeding via InitializeModes then rotating matches the external-seed flow."""
    norb = 3
    nelec = (2, 1)
    occupation = [True, True, False, True, False, False]
    rot = random_unitary(norb, seed=5)

    # our flow: InitializeModes seeds the state, then OrbitalRotation acts on it
    circ = FermionicCircuit(2 * norb)
    circ.append(InitializeModes(occupation), circ.modes)
    circ.append(OrbitalRotation(rot), [circ.modes[i] for i in range(norb)])
    result = circ._apply_unitary_(None, norb, nelec, copy=True)

    # reference: the current external-seed flow (build the determinant, then apply the same rotation)
    vec0 = ffsim.slater_determinant(norb, ([0, 1], [0]))
    expected = ffsim.apply_orbital_rotation(vec0.copy(), (rot, None), norb=norb, nelec=nelec)

    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_initialize_modes_apply_unitary_via_ffsim_apply_unitary_with_real_vec():
    """ffsim.apply_unitary (which always passes a real array) drives a circuit starting with the seed."""
    norb = 3
    nelec = (2, 1)
    occupation = [True, True, False, True, False, False]
    rot = random_unitary(norb, seed=5)

    circ = FermionicCircuit(2 * norb)
    circ.append(InitializeModes(occupation), circ.modes)
    circ.append(OrbitalRotation(rot), [circ.modes[i] for i in range(norb)])

    # a correctly sized (arbitrary) vector in the target sector; InitializeModes overwrites it
    vec0 = ffsim.slater_determinant(norb, ([0, 1], [0]))
    result = ffsim.apply_unitary(vec0, circ, norb=norb, nelec=nelec)

    expected = ffsim.apply_orbital_rotation(vec0.copy(), (rot, None), norb=norb, nelec=nelec)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_initialize_modes_apply_unitary_native_fallback_matches_ffsim(monkeypatch):
    """With ffsim disabled the native seed kernel matches ffsim.slater_determinant."""
    import qiskit_fermions.circuit.library.initialize_modes as initialize_modes_module

    monkeypatch.setattr(initialize_modes_module, "HAS_FFSIM", False)

    # spinful, non-trivial ranks
    norb = 6
    nelec = (3, 2)
    # alpha orbitals 1,3,5 (modes 1,3,5); beta orbitals 0,4 (modes norb+0, norb+4)
    alpha_occ = [False, True, False, True, False, True]
    beta_occ = [True, False, False, False, True, False]
    occupation = alpha_occ + beta_occ
    result = InitializeModes(occupation)._apply_unitary_(None, norb, nelec, copy=False)
    expected = ffsim.slater_determinant(norb, ([1, 3, 5], [0, 4]))
    np.testing.assert_allclose(result, expected, atol=1e-12)

    # spinless
    norb_s = 6
    nelec_s = 3
    occ_s = [False, True, False, True, False, True]
    result_s = InitializeModes(occ_s)._apply_unitary_(None, norb_s, nelec_s, copy=False)
    expected_s = ffsim.slater_determinant(norb_s, [1, 3, 5])
    np.testing.assert_allclose(result_s, expected_s, atol=1e-12)


def test_initialize_modes_apply_unitary_rejects_sector_mismatch(monkeypatch):
    """An occupation whose electron counts disagree with nelec is rejected, with or without ffsim."""
    import qiskit_fermions.circuit.library.initialize_modes as initialize_modes_module

    # spinful: occupation sets 3 alpha but nelec asks for 2
    norb = 3
    nelec = (2, 1)
    occupation = [True, True, True, True, False, False]  # 3 alpha, 1 beta
    with pytest.raises(ValueError, match="do not match the requested nelec"):
        InitializeModes(occupation)._apply_unitary_(None, norb, nelec, copy=False)

    monkeypatch.setattr(initialize_modes_module, "HAS_FFSIM", False)
    with pytest.raises(ValueError, match="do not match the requested nelec"):
        InitializeModes(occupation)._apply_unitary_(None, norb, nelec, copy=False)

    # spinless: total count disagrees
    with pytest.raises(ValueError, match="do not match the requested nelec"):
        InitializeModes([True, True, True])._apply_unitary_(None, 3, 2, copy=False)


def test_initialize_modes_apply_unitary_rejects_wrong_length_vec():
    """A non-None vec whose length disagrees with the occupation's sector is rejected."""
    norb = 3
    nelec = (2, 1)
    occupation = [True, True, False, True, False, False]
    wrong = np.zeros(5, dtype=complex)  # sector dim is C(3,2)*C(3,1) = 9
    with pytest.raises(ValueError, match="does not match the dimension"):
        InitializeModes(occupation)._apply_unitary_(wrong, norb, nelec, copy=True)


def test_fermionic_circuit_apply_unitary_accepts_none_vec():
    """Regression: FermionicCircuit._apply_unitary_ does not crash on a None vec with copy=True."""
    norb = 2
    nelec = (1, 1)
    circ = FermionicCircuit(2 * norb)
    circ.append(InitializeModes([True, False, True, False]), circ.modes)

    result = circ._apply_unitary_(None, norb, nelec, copy=True)
    expected = ffsim.slater_determinant(norb, ([0], [0]))
    np.testing.assert_allclose(result, expected, atol=1e-12)
