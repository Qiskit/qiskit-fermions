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

"""Hamiltonian evolution gate."""

from __future__ import annotations

import numpy as np

from qiskit_fermions.operators.protocol import OperatorTrait
from qiskit_fermions.utils.optionals import HAS_FFSIM

from .. import FermionicGate


class Evolution(FermionicGate):
    r"""Implements the time evolution of an operator.

    Given a fermionic ``operator`` :math:`H` and an evolution ``time`` :math:`t`, this gate
    implements the unitary

    .. math::

        U = e^{-i t H}.

    For :math:`U` to be unitary, :math:`H` must be Hermitian. This is the caller's responsibility
    and is not verified.

    .. note::
       How this evolution is decomposed into a circuit is not fixed by this gate alone. The default
       :meth:`_define` implementation splits the evolution group-by-group when the ``operator`` has
       :attr:`~qiskit_fermions.operators.FermionOperator.groups` assigned, and term-by-term
       otherwise, yielding a first-order product formula (which is exact only when the individual
       factors mutually commute). The transpilation process may further alter this decomposition
       (for example, :class:`.QDriftTrotterization` replaces it with a randomized product formula).
    """

    def __init__(self, num_modes: int, operator: OperatorTrait, time: float = 1.0) -> None:
        r"""Initializing an instance of this gate can be done with the arguments listed below.

        Args:
            num_modes: the number of fermionic modes on which this gate acts.
            operator: the Hermitian operator :math:`H` under which to time evolve the acted-upon
                fermionic modes.
            time: the evolution time :math:`t` entering the exponent of :math:`e^{-i t H}`. A
                negative value evolves backwards in time.
        """
        self.operator = operator
        """The operator under which to time evolve the acted-upon fermionic modes."""

        super().__init__(
            "Evolution", num_modes, [time], label=f"evolve({' '.join(str(operator).split())})"
        )

    def _define(self) -> None:
        from qiskit_fermions.circuit import FermionicCircuit

        definition = FermionicCircuit(self.num_modes)

        # when the operator being evolved has groups use those for the decomposition, otherwise
        # decompose into all individual terms
        iterator = (
            self.operator.iter_terms
            if self.operator.groups is None
            else self.operator.split_out_groups
        )

        for item in iterator():
            if isinstance(item, tuple):
                # iterating over terms rather than operator groups
                item = self.operator.__class__.from_terms([item])

            # reduce each operator to act only on the non-idle part of the register
            active = item.get_support()
            num_active = len(active)
            active_idx = iter(range(num_active))
            idle_idx = iter(range(num_active, self.num_modes))
            permutation = [
                next(active_idx) if idx in active else next(idle_idx)
                for idx in range(self.num_modes)
            ]
            relabeled = item.relabel_modes(permutation)

            definition.append(
                Evolution(num_active, relabeled, time=self.params[0]),
                [definition.modes[idx] for idx in sorted(active)],
            )

        self._definition = definition._inner

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        """Applies this gate's ``exp(-i * time * operator)`` to an ffsim state vector.

        This implements ffsim's ``SupportsApplyUnitary`` protocol. See
        :meth:`_apply_unitary_placed_` for the details; this method assumes the gate's operator is
        already expressed in the space defined by ``(norb, nelec)`` (i.e. an identity mode placement).
        """
        return self._apply_unitary_placed_(vec, norb, nelec, copy, list(range(self.num_modes)))

    @HAS_FFSIM.require_in_call
    def _apply_unitary_placed_(
        self,
        vec: np.ndarray,
        norb: int,
        nelec: int | tuple[int, int],
        copy: bool,
        freg_indices: list[int],
    ) -> np.ndarray:
        """Applies ``exp(-i * time * operator)`` after relabeling the operator to global modes.

        The operator is relabeled onto its global modes, converted to an ``ffsim.FermionOperator``,
        turned into a ``scipy`` ``LinearOperator`` via ``ffsim.linear_operator``, and applied to the
        vector via ``scipy.sparse.linalg.expm_multiply``. This mirrors ffsim's own ``_apply_unitary_``
        implementations (e.g. for its UCCSD operators).

        Args:
            vec: the state vector to act on.
            norb: the number of spatial orbitals of the *global* state vector.
            nelec: either a single integer for a spinless system, or a pair of integers storing the
                numbers of spin alpha and spin beta fermions. An integer selects the spinless mode
                interpretation (the operator's ``norb`` modes are alpha orbitals); a pair selects the
                spinful ``(orb, spin)`` interpretation of the operator's ``2 * norb`` modes.
            copy: whether to copy the vector before operating on it.
            freg_indices: the absolute (global) mode indices that this gate's local modes map onto.
                The operator is relabeled from its local modes to these global modes before being
                applied, mirroring the transpiler's synthesis pass.

        Returns:
            The transformed vector.

        Raises:
            MissingOptionalLibraryError: if ``ffsim`` is not installed.
            ValueError: if the operator does not conserve particle number and the z-component of spin
                (raised by ``ffsim.linear_operator``).
        """
        import ffsim
        import scipy.sparse.linalg

        from ._ffsim import to_ffsim_operator

        if copy:
            vec = vec.copy()

        operator = self.operator.relabel_modes(freg_indices)
        ffsim_op = to_ffsim_operator(operator, norb, nelec)
        linop = ffsim.linear_operator(ffsim_op, norb=norb, nelec=nelec)
        return scipy.sparse.linalg.expm_multiply(-1j * self.params[0] * linop, vec, traceA=0.0)
