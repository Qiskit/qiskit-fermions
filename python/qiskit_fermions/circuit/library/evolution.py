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

from typing import TYPE_CHECKING

from qiskit_fermions.operators import OperatorTrait

from .. import FermionicGate
from ._expm_multiply import _expm_multiply_fci

if TYPE_CHECKING:
    import numpy as np

    from .synthesis import FermionicEvolutionSynthesis


class Evolution(FermionicGate):
    r"""Implements the time evolution of an operator.

    Given a fermionic ``operator`` :math:`H` and an evolution ``time`` :math:`t`, this gate
    implements the unitary

    .. math::

        U = e^{-i t H}.

    For :math:`U` to be unitary, :math:`H` must be Hermitian. This is the caller's responsibility
    and is not verified.

    .. note::
       Breaking this gate down in fermionic space (into a circuit of smaller :class:`.Evolution`
       gates) is **optional**, and it is what the :attr:`synthesis` attribute governs. The gate can
       equally be handed straight to the fermion-to-qubit stage, which maps and synthesizes it whole,
       however many terms its ``operator`` holds. See
       :mod:`~qiskit_fermions.circuit.library.synthesis`.

       When it `is` decomposed (by :meth:`~.FermionicCircuit.decompose`, for instance)
       :attr:`synthesis` decides how. It defaults to :class:`.FermionicLieTrotter`, which splits the
       evolution group-by-group when the ``operator`` has
       :attr:`~qiskit_fermions.operators.OperatorTrait.groups` assigned (see
       :ref:`grouping_explanation`), and term-by-term otherwise (exact only when the individual factors
       mutually commute). The transpilation process may further alter the decomposition;
       :class:`.QDriftTrotterization`, for example, replaces the evolution with a randomized sample of
       its terms.

    .. note::
       The state-vector simulation path does `not` go through :attr:`synthesis`: it exponentiates the
       whole ``operator`` exactly. Simulating this gate therefore incurs no Trotter error, whereas a
       decomposed circuit generally does.
    """

    def __init__(
        self,
        num_modes: int,
        operator: OperatorTrait,
        time: float = 1.0,
        *,
        synthesis: FermionicEvolutionSynthesis | None = None,
        atomic: bool = False,
    ) -> None:
        r"""Initializing an instance of this gate can be done with the arguments listed below.

        Args:
            num_modes: the number of fermionic modes on which this gate acts.
            operator: the Hermitian operator :math:`H` under which to time evolve the acted-upon
                fermionic modes.
            time: the evolution time :math:`t` entering the exponent of :math:`e^{-i t H}`. A
                negative value evolves backwards in time.
            synthesis: the fermion-to-fermion synthesis method with which to decompose this gate, when
                it gets decomposed at all. If ``None`` (the default), a
                :class:`.FermionicLieTrotter` instance is used.
            atomic: whether this gate is a fully decomposed factor which must not be broken down any
                further in fermionic space. See :attr:`atomic`.
        """
        from .synthesis import FermionicLieTrotter

        self.operator = operator
        """The operator under which to time evolve the acted-upon fermionic modes."""

        # both read-only (see their properties): the definition is cached the first time it is built,
        # so a later re-assignment of either would be silently ignored
        self._synthesis: FermionicEvolutionSynthesis = synthesis or FermionicLieTrotter()
        self._atomic = atomic

        super().__init__(
            "Evolution", num_modes, [time], label=f"evolve({' '.join(str(operator).split())})"
        )

    @property
    def synthesis(self) -> FermionicEvolutionSynthesis:
        """The fermion-to-fermion synthesis method with which this gate is decomposed.

        This is read-only: the gate's definition is built once and then cached, so a synthesis method
        assigned after the fact could not take effect. Pass it to the constructor instead.
        """
        return self._synthesis

    @property
    def atomic(self) -> bool:
        r"""Whether this gate is a factor that must not be decomposed any further.

        A :class:`.FermionicEvolutionSynthesis` marks the factors it emits as atomic, because they
        are the result of the split it chose to perform: breaking them down again would discard that
        choice. The evolution of a single operator term is atomic for the stronger reason that there
        is nothing left to split.

        An atomic gate has no definition at all, so :meth:`~.FermionicCircuit.decompose` leaves it in
        place rather than expanding it. It still maps and synthesizes normally at the
        fermion-to-qubit level, which is where a factor is turned into actual operations.

        This is read-only, for the same reason as :attr:`synthesis`.
        """
        return self._atomic

    def _define(self) -> None:
        if self.atomic:
            # A fully decomposed factor has no fermionic definition: it is left untouched by
            # `decompose()` and lowered directly by the fermion-to-qubit stage. Note that this must
            # leave `_definition` as `None` rather than assigning an empty circuit -- an empty
            # definition would make `decompose()` *drop* the evolution instead of keeping it.
            return

        self._definition = self.synthesis.synthesize(self)._inner

    def inverse(self, annotated: bool = False) -> Evolution:
        r"""Returns the inverse of this gate, :math:`e^{+i t H}`.

        Args:
            annotated: ignored. The inverse of this gate is another :class:`.Evolution`, so it never
                needs to be an :class:`~qiskit.circuit.AnnotatedOperation`.

        Returns:
            An :class:`.Evolution` gate evolving the same operator for the negated time.
        """
        # `Instruction.inverse` would recurse through the definition and return a plain `Gate`,
        # losing the operator and this gate's synthesis method; negating the time is both cheaper and
        # exact.
        return Evolution(
            self.num_modes,
            self.operator,
            time=-self.params[0],
            synthesis=self.synthesis,
            atomic=self.atomic,
        )

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
        backed by a native FCI matrix-vector kernel, then applied to the vector via
        ``scipy.sparse.linalg.expm_multiply``. This mirrors ffsim's own ``_apply_unitary_``
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
            NotImplementedError: if the gate's ``operator`` is not a
                :class:`~qiskit_fermions.operators.FermionOperator`. The simulation path is currently
                backed by the native FCI kernel, which only ``FermionOperator`` exposes (via
                :class:`.SupportsLinearOperator`); evolving other operator types is not yet supported.
            ValueError: if the operator does not conserve the ``(norb, nelec)`` sector. Evolving
                under ``exp(-i * time * operator)`` only yields a unitary when ``operator`` maps the
                sector to itself; a term that leaves the sector would be silently projected to zero by
                the native kernel, producing a non-unitary, physically meaningless result. Such an
                operator is therefore rejected rather than applied. For a pair ``nelec`` the alpha and
                beta electron counts must *each* be conserved (particle number and the z-component of
                spin), matching the fixed sector the kernel represents.
        """
        from qiskit_fermions.operators import FermionOperator

        # The simulation path relies on the operator exposing ``SupportsLinearOperator`` (the native FCI
        # kernel) and ``conserves_sector``, which today only ``FermionOperator`` provides. Until
        # ``SupportsLinearOperator`` is folded into ``OperatorTrait`` so any operator can be evolved,
        # reject a non-``FermionOperator`` operator with a clear error rather than failing obscurely
        # deeper in.
        if not isinstance(self.operator, FermionOperator):
            raise NotImplementedError(
                "Evolution can only be applied to a state vector when its operator is a "
                f"'FermionOperator'; got '{type(self.operator).__name__}'. Simulating the evolution "
                "of other operator types is not yet supported."
            )

        if copy:
            vec = vec.copy()

        # `self.operator` is narrowed to `FermionOperator` by the guard above, and `relabel_modes`
        # preserves the type, so the relabeled operator exposes `conserves_sector` and satisfies
        # `SupportsLinearOperator`.
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
        return _expm_multiply_fci(operator, vec, norb, nelec, scale=-1j * self.params[0])
