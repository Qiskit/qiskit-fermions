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

"""FermionGate."""

from __future__ import annotations

from qiskit.circuit import Gate


class FermionGate(Gate):
    """TODO."""

    def __init__(
        self,
        name: str,
        num_fermions: int,
        /,
        params: list | None = None,
        *,
        label: str | None = None,
    ) -> None:
        """TODO."""
        if params is None:
            params = []
        super().__init__(name, num_fermions, params, label)
