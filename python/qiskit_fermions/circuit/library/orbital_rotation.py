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
    r"""Implements an orbital rotation.

    Given an :math:`n \times n` unitary matrix :math:`U` (``rotation_unitary``), this gate
    implements the single-particle basis change that maps the creation operators as

    .. math::

        a^\dagger_i \mapsto \sum_j U_{ji} a^\dagger_j,

    which is equivalent to applying the many-body unitary

    .. math::

        \exp\left(\sum_{ij} \log(U)_{ij} \, a^\dagger_i a_j\right).

    The number of fermionic modes the gate acts on is the dimension :math:`n` of
    ``rotation_unitary``.
    """

    def __init__(self, rotation_unitary: np.ndarray) -> None:
        r"""Initializing an instance of this gate can be done with the arguments listed below.

        Args:
            rotation_unitary: the :math:`n \times n` unitary matrix :math:`U` defining the orbital
                rotation via :math:`a^\dagger_i \mapsto \sum_j U_{ji} a^\dagger_j`. It must be
                square and unitary; this is the caller's responsibility and is not verified.
        """
        self.rotation_unitary = rotation_unitary
        """The unitary matrix representing the orbital rotation coefficients."""

        super().__init__("OrbitalRotation", self.rotation_unitary.shape[0], [])
