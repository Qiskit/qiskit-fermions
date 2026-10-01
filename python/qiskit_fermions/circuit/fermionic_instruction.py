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

"""FermionicInstruction."""

from __future__ import annotations

import numbers
from typing import cast

from qiskit.circuit import Instruction


class FermionicInstruction(Instruction):
    """The base class for all instructions acting on fermionic modes.

    This is the type that the fermion-to-qubit synthesis stage dispatches on, so it is what makes an
    instruction "fermionic" as far as this package is concerned.

    Most fermionic instructions are unitary and should subclass :class:`.FermionicGate` instead, which
    adds the :class:`~qiskit.circuit.Gate` interface on top of this one. Subclass this class directly
    only for an instruction that cannot be a gate, the motivating case being one that acts on
    classical bits: :class:`~qiskit.circuit.Gate` fixes its classical-bit count at zero, so a
    measurement (:class:`.FermionicMeasure`) cannot be a gate.

    .. caution::
       Since this is a subclass of :class:`~qiskit.circuit.Instruction` the documentation of its
       methods may refer to `qubits`. Those references should be interpreted as referring to
       `fermions` in the context of instances of this subclass.

       It may also happen that some of the inherited methods may not always make sense because of
       this `re-interpretation`. You have been warned.
    """

    @property
    def num_modes(self) -> int:
        """The number of fermionic modes that this instruction acts upon."""
        return cast(int, self._num_qubits)

    @staticmethod
    def _normalize_nelec(nelec: int | tuple[int, int]) -> int | tuple[int, int]:
        """Normalizes an integral ``nelec`` to a plain :class:`int`.

        ffsim (and the native FCI kernels) classify the spinless vs. spinful sector with
        ``isinstance(nelec, int)``, which a numpy integer (e.g. ``np.int64``) fails -- it would be
        misrouted to the spinful path and crash deeper in. Coercing integral values to ``int`` at the
        entry points keeps the classification correct; a ``(n_alpha, n_beta)`` tuple is passed through
        unchanged. Applied at both apply-unitary entry points (:meth:`.FermionicGate._apply_unitary_`
        and :meth:`.FermionicCircuit._apply_unitary_placed_`) since the DAG walk bypasses the former.
        """
        if isinstance(nelec, numbers.Integral):
            return int(nelec)
        return nelec
