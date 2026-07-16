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

"""Tests for the native ``slater_determinant_statevector`` FCI seed kernel."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit_fermions._lib.linalg.fci import slater_determinant_statevector


def test_slater_determinant_statevector_spinless_is_one_hot():
    """A spinless occupation seeds a one-hot vector of length ``C(norb, nelec)``."""
    # occupy orbitals 0 and 1 of 4 (the lowest string) -> C(4, 2) = 6 dimensional, address 0
    vec = slater_determinant_statevector(4, 0b0011, None)
    assert vec.shape == (6,)
    assert vec.dtype == np.dtype("complex128")
    np.testing.assert_array_equal(np.flatnonzero(vec), [0])
    assert vec[0] == 1.0


def test_slater_determinant_statevector_spinful_flat_index():
    """A spinful occupation seeds the block-spin flat index ``addr_a * dim_b + addr_b``."""
    # norb=2, alpha occ {1}, beta occ {0}: dim_a = dim_b = C(2, 1) = 2, addr_a = 1, addr_b = 0
    # -> flat index 1 * 2 + 0 = 2 in a length-4 vector
    vec = slater_determinant_statevector(2, 0b10, 0b01)
    assert vec.shape == (4,)
    np.testing.assert_array_equal(np.flatnonzero(vec), [2])
    assert vec[2] == 1.0


def test_slater_determinant_statevector_rejects_out_of_range_mode():
    """An occupation bit set outside ``0..norb`` raises a catchable ValueError, not a Rust panic."""
    with pytest.raises(ValueError, match="outside the range"):
        slater_determinant_statevector(2, 0b100, None)  # bit 2 set, but only orbitals 0, 1 exist


def test_slater_determinant_statevector_rejects_too_many_orbitals():
    """A norb beyond the bitmask limit raises a catchable ValueError, not a Rust panic."""
    with pytest.raises(ValueError, match="exceeds the maximum"):
        slater_determinant_statevector(65, 0b1, None)
