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

"""Mode initialization gate."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .. import FermionGate


class InitializeModes(FermionGate):
    """Implements the fermionic mode initialization.

    .. caution::
       This is an early development prototype. Beware of changes to its interface without warning
       during the pre-release development of this package.
    """

    def __init__(self, occupation: Sequence[bool]) -> None:
        """Initializes the modes of a :class:`.FermionRegister` with the provided occupation.

        Args:
            occupation: a sequence of booleans indicating the occupation for each mode in the
                :class:`.FermionRegister` being initialized by this gate.
        """
        self.occupation = np.asarray(occupation, dtype=bool)
        """The sequence of booleans indicating the occupation for each mode in the
        :class:`~.FermionRegister` being initialized by this gate."""

        super().__init__("InitializeModes", len(self.occupation), [])
