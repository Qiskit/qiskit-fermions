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

"""Tests for the OrbitalRotation gate."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import OrbitalRotation
from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager

from ...utils import random_unitary


def test_orbital_rotation_coerces_real_dtype():
    """A real-dtype rotation matrix is coerced to complex on construction.

    The synthesis path (``givens_decomposition``) requires a complex matrix; without coercion a
    real-valued unitary reaches the Rust binding and raises an opaque cast error. Constructing from a
    real array must store a complex ``rotation_unitary`` and synthesize without error.
    """
    num_modes = 4
    # a real-valued orthogonal matrix (real special case of a unitary)
    real_rotation = np.linalg.qr(random_unitary(num_modes, seed=3).real)[0]
    assert real_rotation.dtype.kind == "f"

    gate = OrbitalRotation(real_rotation)
    assert gate.rotation_unitary.dtype == np.complex128

    circ = FermionicCircuit(num_modes)
    circ.append(gate, circ.modes)
    # lowering must not raise (previously: "'ndarray' object cannot be cast as 'ndarray'")
    generate_preset_jw_pass_manager().run(circ)


def test_orbital_rotation_from_t1_amplitudes_is_unitary():
    """``OrbitalRotation.from_t1_amplitudes`` builds a unitary of the right size.

    (Moved here from the UCJ tests, where it had ended up despite testing neither UCJ nor anything
    UCJ-specific.)
    """
    t1 = np.array([[0.1, 0.2], [0.3, -0.1]])  # 2 occ, 2 virt
    gate = OrbitalRotation.from_t1_amplitudes(t1)
    assert isinstance(gate, OrbitalRotation)
    assert gate.num_modes == 4
    u = gate.rotation_unitary
    np.testing.assert_allclose(u.conj().T @ u, np.eye(4), atol=1e-12)


def test_spin_mixing_rotation_is_rejected():
    """A rotation with nonzero alpha/beta off-blocks cannot act on a fixed ``(n_alpha, n_beta)``.

    Such a rotation does not conserve the per-spin electron counts, so ``_resolve_orbital_rotation``
    refuses it rather than silently producing a state outside the requested sector. This error path
    had no test at all.
    """
    ffsim = pytest.importorskip("ffsim")

    norb = 2
    # A Givens rotation between an alpha and a beta orbital: block-off-diagonal by construction.
    generator = np.zeros((2 * norb, 2 * norb), dtype=complex)
    generator[0, norb] = 0.3
    generator[norb, 0] = -0.3
    mixing = scipy.linalg.expm(generator)
    assert not np.allclose(mixing[:norb, norb:], 0.0), "fixture must actually mix the sectors"

    gate = OrbitalRotation(mixing)
    vec = ffsim.hartree_fock_state(norb, (1, 1))
    with pytest.raises(ValueError, match="mixes the alpha and beta spin sectors"):
        gate._apply_unitary_placed_(vec, norb, (1, 1), True, list(range(2 * norb)))


def test_block_diagonal_rotation_with_roundoff_is_accepted():
    """The off-block check tolerates floating-point round-off.

    A rotation assembled via ``expm`` of a block-diagonal generator carries tiny nonzero off-blocks;
    the guard above must not reject those. This is the other side of the tolerance that
    ``test_spin_mixing_rotation_is_rejected`` exercises, and pins why the check uses ``allclose``
    rather than an exact comparison.
    """
    ffsim = pytest.importorskip("ffsim")

    norb = 2
    generator = np.zeros((2 * norb, 2 * norb), dtype=complex)
    # block-diagonal: an alpha-alpha rotation and a beta-beta one
    generator[0, 1], generator[1, 0] = 0.4, -0.4
    generator[norb, norb + 1], generator[norb + 1, norb] = -0.2, 0.2
    block_diagonal = scipy.linalg.expm(generator)

    gate = OrbitalRotation(block_diagonal)
    vec = ffsim.hartree_fock_state(norb, (1, 1))
    # must not raise
    out = gate._apply_unitary_placed_(vec, norb, (1, 1), True, list(range(2 * norb)))
    assert np.isclose(np.linalg.norm(out), 1.0)
