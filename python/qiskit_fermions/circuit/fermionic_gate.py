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

import copy
from typing import TYPE_CHECKING, Self, cast

from qiskit.circuit import Gate

from .fermionic_instruction import FermionicInstruction

if TYPE_CHECKING:
    import numpy as np

    from .library.synthesis import F2FSynthesisPlugin


class FermionicGate(FermionicInstruction, Gate):
    """The base class for all fermionic gates.

    This is the `unitary` fermionic instruction: it adds the :class:`~qiskit.circuit.Gate` interface
    on top of :class:`.FermionicInstruction`, from which it inherits :attr:`~.num_modes`. Almost every
    fermionic operation belongs here; subclass :class:`.FermionicInstruction` directly only for one
    that cannot be a gate, such as a measurement. As such, this class (mostly) serves as a type (for
    the time being).

    .. rubric:: Breaking a gate down in fermionic space

    Every fermionic gate has a :attr:`synthesis` and an :attr:`atomic` flag, which the
    fermion-to-fermion synthesis framework relies on (see
    :mod:`~qiskit_fermions.circuit.library.synthesis`). The :attr:`synthesis` is the method that
    rewrites the gate into a circuit of smaller fermionic gates, and it is ``None`` for a gate that
    does not break down. A gate that is :attr:`atomic` is never broken down, however it is asked to.
    Both are chosen when the gate is built and are read-only afterwards.

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
        synthesis: F2FSynthesisPlugin | None = None,
        atomic: bool = False,
    ) -> None:
        if params is None:
            params = []

        # both read-only (see their properties): the definition is cached the first time it is built,
        # so a later re-assignment of either would be silently ignored
        self._synthesis = synthesis
        self._atomic = atomic

        super().__init__(name, num_modes, params, label)

    @property
    def synthesis(self) -> F2FSynthesisPlugin | None:
        """The fermion-to-fermion synthesis method with which this gate is broken down, if any.

        This is the customizable part of breaking a gate down: any object implementing the
        :class:`.F2FSynthesisPlugin` protocol can be given, and the :class:`.F2FSynthesis` transpiler
        pass selects one per gate type for a whole circuit. ``None`` marks a gate that does not break
        down in fermionic space.

        What the circuit that the method produces is equivalent to is specified by the gate, and
        documented there. Where the gate's decomposition is that circuit,
        :meth:`~.FermionicCircuit.decompose` yields it; a gate may deviate from this and document so.

        This is read-only: the circuit that :meth:`~.FermionicCircuit.decompose` yields is built once
        and then cached, so a synthesis method assigned after the fact could not take effect. Pass it
        to the constructor instead, or use :meth:`with_synthesis`.
        """
        return self._synthesis

    @property
    def atomic(self) -> bool:
        r"""Whether this gate is a factor that must not be broken down any further.

        A synthesis marks the factors it emits as atomic, because they are the result of the split it
        chose to perform: breaking them down again would discard that choice. A gate that is atomic
        for the stronger reason that there is nothing left to split is marked the same way.

        An atomic gate has no definition at all, so :meth:`~.FermionicCircuit.decompose` leaves it in
        place rather than expanding it, and the :class:`.F2FSynthesis` pass leaves it alone. It still
        maps and synthesizes normally at the fermion-to-qubit level, which is where a factor is turned
        into actual operations.

        This is read-only, for the same reason as :attr:`synthesis`.
        """
        return self._atomic

    def with_synthesis(self, synthesis: F2FSynthesisPlugin | None) -> Self:
        """Returns a copy of this gate that is broken down with another synthesis method.

        The gate itself is left as it is, because a gate instance can be shared between circuits.

        Args:
            synthesis: the fermion-to-fermion synthesis method of the copy.

        Returns:
            A shallow copy of this gate whose :attr:`synthesis` is ``synthesis``.
        """
        gate = copy.copy(self)
        gate._synthesis = synthesis
        # the cached definition belongs to the method it was built with
        gate._definition = None
        return gate

    def _define(self) -> None:
        # A gate that is atomic, or that has no synthesis, has no fermionic definition: it is left
        # untouched by `decompose()` and lowered directly by the fermion-to-qubit stage. Note that
        # this must leave `_definition` as `None` rather than assigning an empty circuit -- an empty
        # definition would make `decompose()` *drop* the gate instead of keeping it.
        if self._atomic or self._synthesis is None:
            return

        self._definition = self._synthesis.synthesize(self)._inner

    def _apply_unitary_(
        self, vec: np.ndarray | None, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        """Applies this gate to an ffsim state vector, implementing ffsim's protocol.

        This is the identity-placement entry point of ffsim's :class:`ffsim.SupportsApplyUnitary`
        protocol: it assumes the gate acts on the modes ``0..num_modes`` of the state vector and
        delegates to the placement-aware :meth:`_apply_unitary_placed_` (see
        :class:`.SupportsApplyUnitaryPlaced`), which every concrete fermionic gate implements. See
        that method for the semantics of ``vec`` (including whether a ``None`` vector is accepted),
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
        nelec = self._normalize_nelec(nelec)
        return cast("np.ndarray", placed(vec, norb, nelec, copy, list(range(self.num_modes))))
