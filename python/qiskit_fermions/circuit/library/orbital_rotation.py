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

"""Orbital rotation gate."""

from __future__ import annotations

import numpy as np

from .. import FermionicGate


class OrbitalRotation(FermionicGate):
    """Implements an orbital rotation."""

    def __init__(self, rotation_unitary: np.ndarray) -> None:
        """Initializing an instance of this gate can be done with the arguments listed below.

        Args:
            rotation_unitary: the unitary matrix representing the orbital rotation coefficients.
        """
        self.rotation_unitary = rotation_unitary
        """The unitary matrix representing the orbital rotation coefficients."""

        super().__init__("OrbitalRotation", self.rotation_unitary.shape[0], [])
