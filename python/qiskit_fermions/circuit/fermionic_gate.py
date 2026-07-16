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

"""FermionicGate."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from qiskit.circuit import Gate

if TYPE_CHECKING:
    import numpy as np


class FermionicGate(Gate):
    """The base class for all fermionic gates.

    To ensure consistency only subclasses of this gate class can be added to instances of
    :class:`.FermionicCircuit`. As such, this class (mostly) serves as a type (for the time being).

    .. caution::
       Since this is a subclass of :class:`~qiskit.circuit.Gate` the documentation of its methods
       may refer to `qubits`. Those references should be interpreted as referring to `fermions` in
       the context of instances of this subclass.

       It may also happen that some of the inherited methods may not always make sense because of
       this `re-interpretation`. You have been warned.
    """

    def __init__(  # noqa: D107
        self,
        name: str,
        num_modes: int,
        /,
        params: list | None = None,
        *,
        label: str | None = None,
    ) -> None:
        if params is None:
            params = []
        super().__init__(name, num_modes, params, label)

    @property
    def num_modes(self) -> int:
        """The number of fermionic modes that this gate acts upon."""
        return cast(int, self._num_qubits)

    def _apply_unitary_(
        self, vec: np.ndarray | None, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        """Applies this gate to an ffsim state vector, implementing ffsim's protocol.

        This is the identity-placement entry point of ffsim's ``SupportsApplyUnitary`` protocol: it
        assumes the gate acts on the modes ``0..num_modes`` of the state vector and delegates to the
        placement-aware :meth:`_apply_unitary_placed_`, which every concrete fermionic gate implements.
        See that method for the semantics of ``vec`` (including whether a ``None`` vector is accepted),
        ``norb``, ``nelec``, and ``copy``.

        Raises:
            NotImplementedError: if this gate does not implement ``_apply_unitary_placed_`` (a bare
                :class:`.FermionicGate` used only as a type marker), and therefore cannot be applied to
                a state vector.
        """
        placed = getattr(self, "_apply_unitary_placed_", None)
        if placed is None:
            raise NotImplementedError(
                f"'{type(self).__name__}' does not implement '_apply_unitary_placed_' and so cannot "
                "be applied to a state vector via ffsim's SupportsApplyUnitary protocol."
            )
        return cast("np.ndarray", placed(vec, norb, nelec, copy, list(range(self.num_modes))))
