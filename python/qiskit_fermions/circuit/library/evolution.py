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

    def _apply_unitary_placed_(
        self,
        vec: np.ndarray,
        norb: int,
        nelec: int | tuple[int, int],
        copy: bool,
        freg_indices: list[int],
    ) -> np.ndarray:
        """Applies ``exp(-i * time * operator)`` after relabeling the operator to global modes.

        The operator is relabeled onto its global modes and turned into a ``scipy`` ``LinearOperator``
        via its ``_linear_operator_`` protocol method (backed by a native FCI matrix-vector kernel),
        then applied to the vector via ``scipy.sparse.linalg.expm_multiply``. This mirrors ffsim's own
        ``_apply_unitary_`` implementations (e.g. for its UCCSD operators).

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
            NotImplementedError: if the gate's ``operator`` is not a
                :class:`~qiskit_fermions.operators.FermionOperator`. The simulation path is currently
                backed by the native FCI kernel, which only ``FermionOperator`` exposes (via
                ``_linear_operator_``); evolving other operator types is not yet supported.
            ValueError: if the operator does not conserve the ``(norb, nelec)`` sector. Evolving
                under ``exp(-i * time * operator)`` only yields a unitary when ``operator`` maps the
                sector to itself; a term that leaves the sector would be silently projected to zero by
                the native kernel, producing a non-unitary, physically meaningless result. Such an
                operator is therefore rejected rather than applied. For a pair ``nelec`` the alpha and
                beta electron counts must *each* be conserved (particle number and the z-component of
                spin), matching the fixed sector the kernel represents.
        """
        import scipy.sparse.linalg

        from qiskit_fermions.operators import FermionOperator

        # The simulation path relies on the operator exposing ``_linear_operator_`` (the native FCI
        # kernel) and ``conserves_sector``, which today only ``FermionOperator`` provides. Until
        # ``_linear_operator_`` is folded into ``OperatorTrait`` so any operator can be evolved, reject
        # a non-``FermionOperator`` operator with a clear error rather than failing obscurely deeper in.
        if not isinstance(self.operator, FermionOperator):
            raise NotImplementedError(
                "Evolution can only be applied to a state vector when its operator is a "
                f"'FermionOperator'; got '{type(self.operator).__name__}'. Simulating the evolution "
                "of other operator types is not yet supported."
            )

        if copy:
            vec = vec.copy()

        # `self.operator` is narrowed to `FermionOperator` by the guard above, and `relabel_modes`
        # preserves the type, so the relabeled operator exposes `conserves_sector`/`_linear_operator_`.
        operator = self.operator.relabel_modes(freg_indices)
        # A term that leaves the fixed (norb, nelec) sector is projected to zero by the native kernel,
        # which would turn `exp(-i t H)` into a non-unitary map. Reject rather than silently produce a
        # wrong state. Spinless: one block of `norb`; spinful: alpha [0, norb) and beta [norb, 2*norb)
        # blocks, each of which must be conserved.
        block_sizes = [norb] if isinstance(nelec, int) else [norb, norb]
        if not operator.conserves_sector(block_sizes):
            raise ValueError(
                "Evolution requires an operator that conserves the (norb, nelec) sector: every term "
                "must preserve the particle number"
                + (" of each spin species" if not isinstance(nelec, int) else "")
                + f" (norb={norb}, nelec={nelec})."
            )
        linop = operator._linear_operator_(norb, nelec)
        # ``traceA`` is only a balancing hint for scipy (it factors out ``exp(traceA / n)`` to
        # improve conditioning), not a correctness input: an inexact value costs at most some
        # numerical conditioning. Passing 0.0 avoids scipy estimating the trace itself and mirrors
        # ffsim's own ``_apply_unitary_`` implementations.
        return scipy.sparse.linalg.expm_multiply(-1j * self.params[0] * linop, vec, traceA=0.0)
