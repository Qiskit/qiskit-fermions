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

from typing import cast

from qiskit.circuit import Gate


class FermionGate(Gate):
    """The base class for all fermionic gates.

    To ensure consistency only subclasses of this gate class can be added to instances of
    :class:`.FermionCircuit`. As such, this class (mostly) serves as a type (for the time being).

    .. caution::
       Since this is a subclass of :class:`~qiskit.circuit.Gate` the documentation of its methods
       may refer to `qubits`. Those references should be interpreted as referring to `fermions` in
       the context of instances of this subclass.

       It may also happen that some of the inherited methods may not always make sense because of
       this `re-interpretation`. You have been warned.
    """

    def __init__(
        self,
        name: str,
        num_modes: int,
        /,
        params: list | None = None,
        *,
        label: str | None = None,
    ) -> None:
        """Initializes a FermionGate instance."""
        if params is None:
            params = []
        super().__init__(name, num_modes, params, label)

    @property
    def num_modes(self) -> int:
        """The number of fermionic modes that this gate acts upon."""
        return cast(int, self._num_qubits)
