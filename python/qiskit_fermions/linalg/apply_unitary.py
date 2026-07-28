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

"""Application of a unitary transformation to a state vector."""

from __future__ import annotations

import numpy as np

from qiskit_fermions.protocols import SupportsApplyUnitary


def apply_unitary(
    vec: np.ndarray,
    operator: SupportsApplyUnitary,
    norb: int,
    nelec: int | tuple[int, int],
    copy: bool = True,
) -> np.ndarray:
    """Applies a unitary transformation to a state vector.

    This is a thin, type-agnostic wrapper around the :class:`.SupportsApplyUnitary` protocol
    method, mirroring the free-function style of :func:`ffsim.apply_unitary`.

    Args:
        vec: the state vector to apply the unitary transformation to.
        operator: the object with a unitary effect, implementing :class:`.SupportsApplyUnitary`.
        norb: the number of spatial orbitals.
        nelec: the electron count -- an integer for a spinless sector, or an ``(n_alpha, n_beta)``
            pair for a spinful one.
        copy: whether to copy the vector before operating on it.

    Returns:
        The transformed vector.
    """
    return operator._apply_unitary_(vec, norb, nelec, copy)
