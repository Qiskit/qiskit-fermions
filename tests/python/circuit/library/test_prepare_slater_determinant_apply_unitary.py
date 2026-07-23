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

"""Tests for applying a PrepareSlaterDeterminant gate to an ffsim state vector.

The gate is validate-then-rotate: given a reference state vector confined to its occupation
subspace, the leading :class:`.InitializeModes` validates (and returns it unchanged) and the
:class:`.OrbitalRotation` rotates it. These tests pin the correctness contract that matters for the
later merge optimization pass: applying the merged gate to a reference vector must yield the *exact
same* final state as applying an :class:`.InitializeModes` and an :class:`.OrbitalRotation`
separately.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit_fermions.circuit import FermionicCircuit
from qiskit_fermions.circuit.library import (
    InitializeModes,
    OrbitalRotation,
    PrepareSlaterDeterminant,
)

from ...utils import random_unitary

ffsim = pytest.importorskip("ffsim")


def _block_diag(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """Assembles a block-spin orbital rotation from independent alpha/beta rotations."""
    norb = mat_a.shape[0]
    full = np.zeros((2 * norb, 2 * norb), dtype=complex)
    full[:norb, :norb] = mat_a
    full[norb:, norb:] = mat_b
    return full


def test_prepare_slater_determinant_apply_unitary_spinless_matches_ffsim():
    """The spinless gate validates the reference and applies the rotation (== ffsim rotation)."""
    norb = 5
    nelec = 2
    occupation = [i < nelec for i in range(norb)]
    rotation = random_unitary(norb, seed=3)

    reference = ffsim.slater_determinant(norb, list(range(nelec)))
    reference_before = reference.copy()

    gate = PrepareSlaterDeterminant(occupation, rotation)
    result = gate._apply_unitary_(reference, norb, nelec, copy=True)

    # InitializeModes only validates (no-op on a valid reference), so the effect is the rotation
    expected = ffsim.apply_orbital_rotation(reference.copy(), rotation, norb=norb, nelec=nelec)
    np.testing.assert_allclose(result, expected, atol=1e-10)
    # copy=True must leave the input untouched
    np.testing.assert_array_equal(reference, reference_before)


def test_prepare_slater_determinant_apply_unitary_spinful_matches_ffsim():
    """The spinful (block-spin) gate matches ffsim's per-spin orbital rotation."""
    norb = 3
    nelec = (2, 1)
    occupation = [i < nelec[0] for i in range(norb)] + [i < nelec[1] for i in range(norb)]
    rot_a = random_unitary(norb, seed=1)
    rot_b = random_unitary(norb, seed=2)
    full = _block_diag(rot_a, rot_b)

    reference = ffsim.slater_determinant(norb, ([0, 1], [0]))

    gate = PrepareSlaterDeterminant(occupation, full)
    result = gate._apply_unitary_(reference, norb, nelec, copy=True)

    expected = ffsim.apply_orbital_rotation(
        reference.copy(), (rot_a, rot_b), norb=norb, nelec=nelec
    )
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_prepare_slater_determinant_matches_separate_init_and_rotation():
    """The merge-invariance contract: the merged gate equals Init then Rotation applied separately.

    This is the property the later optimization pass relies on -- fusing an ``InitializeModes`` and an
    ``OrbitalRotation`` into a single ``PrepareSlaterDeterminant`` must leave the simulated state
    exactly unchanged.
    """
    norb = 4
    nelec = 2
    occupation = [i < nelec for i in range(norb)]
    rotation = random_unitary(norb, seed=7)

    reference = ffsim.slater_determinant(norb, list(range(nelec)))

    # separate: InitializeModes (validate) then OrbitalRotation (rotate)
    unmerged = FermionicCircuit(norb)
    unmerged.append(InitializeModes(occupation), unmerged.modes)
    unmerged.append(OrbitalRotation(rotation), unmerged.modes)
    expected = unmerged._apply_unitary_(reference, norb, nelec, copy=True)

    # merged
    merged = PrepareSlaterDeterminant(occupation, rotation)
    result = merged._apply_unitary_(reference, norb, nelec, copy=True)

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_prepare_slater_determinant_apply_unitary_rejects_wrong_reference():
    """A reference vector outside the occupation subspace is rejected by the leading validator."""
    norb = 4
    nelec = 2
    # occupation names orbital 0 occupied, but the reference occupies orbitals 2 and 3
    occupation = [True, True, False, False]
    rotation = random_unitary(norb, seed=9)
    wrong_reference = ffsim.slater_determinant(norb, [2, 3])

    gate = PrepareSlaterDeterminant(occupation, rotation)
    with pytest.raises(ValueError, match="amplitude outside the subspace"):
        gate._apply_unitary_(wrong_reference, norb, nelec, copy=True)


def test_prepare_slater_determinant_apply_unitary_with_placement():
    """A subset-placed gate is embedded onto its global modes and applied correctly.

    Placing the gate on the alpha modes of a block-spin register validates+rotates only that sector.
    """
    norb = 3
    nelec = (2, 1)
    occupation = [i < nelec[0] for i in range(norb)]  # local alpha occupation
    rotation = random_unitary(norb, seed=5)

    reference = ffsim.slater_determinant(norb, ([0, 1], [0]))

    circ = FermionicCircuit(2 * norb)
    circ.append(
        PrepareSlaterDeterminant(occupation, rotation),
        [circ.modes[i] for i in range(norb)],
    )
    result = circ._apply_unitary_(reference, norb, nelec, copy=True)

    expected = ffsim.apply_orbital_rotation(
        reference.copy(), (rotation, None), norb=norb, nelec=nelec
    )
    np.testing.assert_allclose(result, expected, atol=1e-10)
