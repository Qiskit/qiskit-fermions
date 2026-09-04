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

"""Unitary coupled-cluster (UCC) ansatz gate."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from qiskit_fermions._lib.operators.fermion_operator import FermionOperator
from qiskit_fermions.operators.fermion_action import ann, cre
from qiskit_fermions.utils.optionals import HAS_FFSIM

from .. import FermionicGate
from .evolution import Evolution
from .orbital_rotation import OrbitalRotation

if TYPE_CHECKING:
    from qiskit_fermions.circuit import FermionicCircuit


class UCC(FermionicGate):
    r"""Implements the unitary coupled-cluster singles and doubles (UCCSD) ansatz.

    A unitary coupled-cluster operator has the form

    .. math::

        e^{T - T^\dagger}, \qquad
        T = \sum_{ia} t^a_i\, a^\dagger_a a_i
          + \sum_{ijab} t^{ab}_{ij}\, a^\dagger_a a^\dagger_b a_j a_i,

    with :math:`i, j` occupied and :math:`a, b` virtual orbitals, and the amplitudes
    :math:`t_1 = t^a_i` and :math:`t_2 = t^{ab}_{ij}` supplied by the operator this gate wraps.

    The operator itself is built by `ffsim <https://qiskit-community.github.io/ffsim/>`__, and this
    gate turns it into a :class:`.FermionicCircuit`. That division of labor is deliberate: ffsim owns
    the ansatz math (the amplitude conventions and the parameter-vector packing that a variational
    optimizer drives), while this gate expresses the result as fermionic modes so that the transpiler
    can lower it through *any* fermion-to-qubit encoding. ffsim ships no Qiskit gate for UCCSD at all,
    so this is the only route from one of its UCCSD operators to a circuit; see the
    :ref:`ffsim relationship guide <ffsim_relationship_explanation>`.

    Accepts any of ffsim's four UCCSD operators, whose type fixes the spin variant and the number of
    modes this gate acts on. All four act on ``2 * norb`` block-spin modes (mode ``p`` is alpha
    orbital ``p``, mode ``norb + p`` is beta orbital ``p``), with the occupied orbitals ordered before
    the virtual ones:

    - :external:class:`~ffsim.UCCSDOpRestrictedReal` and
      :external:class:`~ffsim.UCCSDOpRestricted` share one spatial parametrization between both spin
      sectors: ``t1`` has shape ``(nocc, nvrt)`` and ``t2`` has shape ``(nocc, nocc, nvrt, nvrt)``.
    - :external:class:`~ffsim.UCCSDOpUnrestrictedReal` and
      :external:class:`~ffsim.UCCSDOpUnrestricted` parameterize the spin sectors independently:
      ``t1`` is a pair ``(t1a, t1b)`` and ``t2`` a triple ``(t2aa, t2ab, t2bb)``.

    .. note::
       The cluster operator only ever sees the part of a same-spin :math:`t_2` block that is
       symmetric under the *simultaneous* exchange :math:`t_2[i,j,a,b] = t_2[j,i,b,a]`, because the
       underlying excitation :math:`a^\dagger_a a^\dagger_b a_j a_i` is invariant under relabeling
       the pairs :math:`(i,a) \leftrightarrow (j,b)`. Coupled-cluster amplitudes (from PySCF, or
       ffsim's own :external:func:`~ffsim.uccsd_generator_restricted`) always carry that symmetry, so
       this gate and ffsim agree exactly on them. A hand-built ``t2`` without it describes the same
       ansatz here as its symmetrized counterpart, whereas ffsim reads the raw tensor.

    .. note::
       Unlike :class:`.UCJ`, this ansatz carries no final orbital rotation of its own: its
       :math:`t_1` amplitudes already provide the single excitations, so a trailing rotation would be
       redundant freedom. ffsim's operators do expose a ``final_orbital_rotation``; when one is set,
       this gate appends it as a closing :class:`.OrbitalRotation`.

    .. note::
       Because the individual excitation terms of :math:`T - T^\dagger` do **not** commute, the
       circuit :meth:`~qiskit.circuit.Gate.definition` this gate produces is a *first-order product
       formula* (Trotter) approximation of the exponential, not an exact decomposition (the usual
       situation for UCC ansatz circuits). The state-vector simulation path
       (:meth:`_apply_unitary_placed_`), by contrast, applies the exponential *exactly* via
       ``scipy``'s ``expm_multiply``. Consequently the simulated gate and its synthesized circuit
       agree only up to the Trotter error; use a higher-order product formula during transpilation to
       tighten it.

    .. note::
       ffsim does not support Windows (through its unconditional PySCF dependency), so this gate
       requires the ``ffsim`` extra (``pip install "qiskit-fermions[ffsim]"``) and is unavailable
       there. Use `WSL <https://learn.microsoft.com/windows/wsl/>`__ on Windows.

    .. caution::
       This is an early development prototype. Beware of changes to its interface without warning
       during the pre-release development of this package.

    .. invisible-code-block: python

        >>> from qiskit_fermions.utils.optionals import HAS_FFSIM

    .. skip: start if(not HAS_FFSIM)

    .. doctest::

        >>> import ffsim
        >>> import numpy as np
        >>> from qiskit_fermions.circuit.library import UCC
        >>> uccsd_op = ffsim.UCCSDOpRestrictedReal(
        ...     t1=np.zeros((1, 1)), t2=np.zeros((1, 1, 1, 1))
        ... )
        >>> gate = UCC(uccsd_op)
        >>> gate.norb, gate.num_modes
        (2, 4)

    .. skip: end
    """

    def __init__(self, uccsd_op: Any) -> None:
        """Initializing an instance of this gate can be done with the argument listed below.

        Args:
            uccsd_op: the ffsim UCCSD operator to build the circuit from, one of
                :external:class:`~ffsim.UCCSDOpRestrictedReal`,
                :external:class:`~ffsim.UCCSDOpRestricted`,
                :external:class:`~ffsim.UCCSDOpUnrestrictedReal` or
                :external:class:`~ffsim.UCCSDOpUnrestricted`. Its type determines the spin variant
                (see the class docstring).

        Raises:
            MissingOptionalLibraryError: if ``ffsim`` is not installed.
            TypeError: if ``uccsd_op`` is not one of ffsim's four UCCSD operator types.
        """
        HAS_FFSIM.require_now("UCC")
        import ffsim

        restricted = (ffsim.UCCSDOpRestricted, ffsim.UCCSDOpRestrictedReal)
        unrestricted = (ffsim.UCCSDOpUnrestricted, ffsim.UCCSDOpUnrestrictedReal)
        if not isinstance(uccsd_op, restricted + unrestricted):
            raise TypeError(
                "UCC requires one of ffsim's UCCSD operators (UCCSDOpRestricted, "
                "UCCSDOpRestrictedReal, UCCSDOpUnrestricted or UCCSDOpUnrestrictedReal), but got "
                f"{type(uccsd_op).__name__}."
            )

        self.uccsd_op = uccsd_op
        """The ffsim UCCSD operator this gate builds its circuit from."""

        self._unrestricted = isinstance(uccsd_op, unrestricted)

        super().__init__("UCC", 2 * uccsd_op.norb, [])

    @property
    def norb(self) -> int:
        """The number of spatial orbitals."""
        return int(self.uccsd_op.norb)

    def _nocc(self) -> int | tuple[int, int]:
        """Returns the number of occupied orbitals, per spin sector when unrestricted."""
        if self._unrestricted:
            t1a, t1b = cast("tuple[np.ndarray, np.ndarray]", self.uccsd_op.t1)
            return int(t1a.shape[0]), int(t1b.shape[0])
        return int(cast(np.ndarray, self.uccsd_op.t1).shape[0])

    def cluster_operator(self) -> FermionOperator:
        r"""Returns the anti-Hermitian cluster generator :math:`T - T^\dagger`.

        The generator is expressed in the block-spin mode convention (mode ``p`` is alpha orbital
        ``p``, mode ``norb + p`` is beta orbital ``p``). Occupied orbitals are ordered before virtual
        ones.

        Being anti-Hermitian, this generator relates to the ansatz unitary by
        :math:`e^{T - T^\dagger} = e^{-i H}` with the Hermitian :math:`H = i (T - T^\dagger)`. That
        :math:`H` is what :class:`.Evolution` consumes, since it requires a Hermitian operator to
        produce a unitary, and it is how :meth:`_build_definition` expresses the ansatz.

        The returned operator carries :attr:`~qiskit_fermions.operators.FermionOperator.groups` that
        pair every excitation with its Hermitian conjugate. That grouping is load-bearing:
        :class:`.Evolution` decomposes group-by-group, so each group becomes one factor
        :math:`e^{-i H_k}` of the product formula, and multiplying by :math:`i` leaves every group
        individually Hermitian -- hence every factor a genuine unitary. Splitting term-by-term
        instead would not be (see the comment below).

        Returns:
            The cluster generator :math:`T - T^\dagger` as a
            :class:`~qiskit_fermions.operators.FermionOperator`.
        """
        excitations = self._excitation_operator()
        adjoint = excitations.adjoint()

        # Pair each excitation with its conjugate in a shared group. Without this, splitting the
        # generator term-by-term (as Evolution does by default) yields individually *non-Hermitian*
        # factors -- each exp(-i H_k) would then be non-unitary and the Trotterized circuit would not
        # even preserve the norm. Grouping T_k with its conjugate makes every group anti-Hermitian, so
        # every Trotter factor is a genuine unitary.
        #
        # Taking the adjoint conjugates each action and reverses its order, which leaves the *multiset*
        # of modes acted on unchanged -- so that sorted multiset is exactly the key uniting a term with
        # its conjugate. Note it must be the multiset and not the mode `set`: repeated modes are
        # possible (`a+_2 a+_2 a_0 a_0` alongside `a+_2 a_0`), and a set would collapse two distinct
        # supports into one key.
        #
        # The group indices are then assigned in sorted key order rather than by first encounter.
        # `Evolution` emits one product-formula factor per group in index order and the excitations do
        # not commute, so that order changes the Trotter error; deriving it from set/dict iteration
        # would tie it to element hashing, and hence to PYTHONHASHSEED, making the synthesized circuit
        # differ between processes.
        def support(actions: Sequence[tuple[bool, int]]) -> tuple[int, ...]:
            return tuple(sorted(mode for _, mode in actions))

        keys = sorted(
            {
                support(actions)
                for operator in (excitations, adjoint)
                for actions, _ in operator.iter_terms()
            }
        )
        groups = {key: index for index, key in enumerate(keys)}

        terms_with_groups = []
        for operator, sign in ((excitations, 1.0), (adjoint, -1.0)):
            for actions, coeff in operator.iter_terms():
                terms_with_groups.append((actions, sign * coeff, groups[support(actions)]))

        return FermionOperator.from_terms_with_groups(terms_with_groups)

    def _excitation_operator(self) -> FermionOperator:
        r"""Builds the (non-Hermitian) cluster operator :math:`T = T_1 + T_2`.

        The terms are accumulated into a dict keyed by the action tuple, because distinct
        ``(i, j, a, b)`` index combinations can map onto the *same* action tuple (for instance the
        restricted same-spin doubles, where ``(i, j, a, b)`` and ``(j, i, b, a)`` produce the same
        actions); their coefficients must add rather than overwrite each other.
        """
        norb = self.norb
        terms: dict[tuple, complex] = {}

        def mode(sigma: int, orb: int) -> int:
            """Maps a ``(spin sector, orbital)`` pair onto its block-spin mode index."""
            return orb + sigma * norb

        def add(actions: tuple, coeff: complex) -> None:
            if coeff == 0.0:
                return
            terms[actions] = terms.get(actions, 0.0) + coeff

        if self._unrestricted:
            self._add_unrestricted_terms(add, mode)
        else:
            self._add_restricted_terms(add, mode)

        return FermionOperator.from_dict(terms)  # type: ignore[arg-type]

    def _add_restricted_terms(self, add, mode) -> None:
        r"""Adds the restricted (spin-summed) :math:`T_1` and :math:`T_2` terms.

        Follows ffsim's ``singles_excitations_restricted``/``doubles_excitations_restricted``
        convention: the singles are applied identically in both spin sectors, and the doubles carry a
        factor ``1/2`` on the two same-spin blocks and ``1`` on the alpha-beta block.
        """
        t1 = cast(np.ndarray, self.uccsd_op.t1)
        t2 = cast(np.ndarray, self.uccsd_op.t2)
        nocc, nvrt = t1.shape

        for i, a in itertools.product(range(nocc), range(nvrt)):
            coeff = complex(t1[i, a])
            for sigma in (0, 1):
                add((cre(mode(sigma, nocc + a)), ann(mode(sigma, i))), coeff)

        for i, j, a, b in itertools.product(range(nocc), range(nocc), range(nvrt), range(nvrt)):
            coeff = complex(t2[i, j, a, b])
            for sigma in (0, 1):
                add(
                    (
                        cre(mode(sigma, nocc + a)),
                        cre(mode(sigma, nocc + b)),
                        ann(mode(sigma, j)),
                        ann(mode(sigma, i)),
                    ),
                    0.5 * coeff,
                )
            add(
                (
                    cre(mode(0, nocc + a)),
                    cre(mode(1, nocc + b)),
                    ann(mode(1, j)),
                    ann(mode(0, i)),
                ),
                coeff,
            )

    def _add_unrestricted_terms(self, add, mode) -> None:
        r"""Adds the unrestricted (per-spin) :math:`T_1` and :math:`T_2` terms.

        Follows ffsim's ``singles_excitations_unrestricted``/``doubles_excitations_unrestricted``
        convention: a factor ``1/4`` on each same-spin doubles block and ``1`` on the alpha-beta
        block. Note that each spin sector uses its *own* occupied count as the virtual-orbital
        offset, so the two sectors may split the same ``norb`` orbitals differently.
        """
        t1a, t1b = cast("tuple[np.ndarray, np.ndarray]", self.uccsd_op.t1)
        t2aa, t2ab, t2bb = cast("tuple[np.ndarray, np.ndarray, np.ndarray]", self.uccsd_op.t2)
        nocc = (t1a.shape[0], t1b.shape[0])
        nvrt = (t1a.shape[1], t1b.shape[1])

        for sigma, t1s in enumerate((t1a, t1b)):
            for i, a in itertools.product(range(nocc[sigma]), range(nvrt[sigma])):
                add((cre(mode(sigma, nocc[sigma] + a)), ann(mode(sigma, i))), complex(t1s[i, a]))

        for sigma, t2s in enumerate((t2aa, t2bb)):
            no, nv = nocc[sigma], nvrt[sigma]
            for i, j, a, b in itertools.product(range(no), range(no), range(nv), range(nv)):
                add(
                    (
                        cre(mode(sigma, no + a)),
                        cre(mode(sigma, no + b)),
                        ann(mode(sigma, j)),
                        ann(mode(sigma, i)),
                    ),
                    0.25 * complex(t2s[i, j, a, b]),
                )

        for i, j, a, b in itertools.product(
            range(nocc[0]), range(nocc[1]), range(nvrt[0]), range(nvrt[1])
        ):
            add(
                (
                    cre(mode(0, nocc[0] + a)),
                    cre(mode(1, nocc[1] + b)),
                    ann(mode(1, j)),
                    ann(mode(0, i)),
                ),
                complex(t2ab[i, j, a, b]),
            )

    def _apply_unitary_placed_(
        self,
        vec: np.ndarray,
        norb: int,
        nelec: int | tuple[int, int],
        copy: bool,
        freg_indices: list[int],
    ) -> np.ndarray:
        """Applies the ansatz after placing its modes onto the vector's global modes.

        This builds the gate's definition (the cluster-operator evolution) and applies it to ``vec``,
        with the definition circuit placed onto the global modes ``freg_indices``. Because the
        definition's single :class:`.Evolution` carries the *whole* cluster generator, its own
        ``_apply_unitary_placed_`` exponentiates it exactly (via ``scipy``'s ``expm_multiply``) -- so
        this path incurs no Trotter error, unlike the synthesized circuit. See :meth:`_define` for
        the exact gate sequence.

        Args:
            vec: the state vector to act on.
            norb: the number of spatial orbitals of the *global* state vector.
            nelec: either a single integer for a spinless system, or a pair of integers storing the
                numbers of spin alpha and spin beta fermions.
            copy: whether to copy the vector before operating on it.
            freg_indices: the absolute (global) mode indices that this gate's local modes map onto.

        Returns:
            The transformed vector.
        """
        return self._build_definition()._apply_unitary_placed_(vec, norb, nelec, copy, freg_indices)

    def _build_definition(self) -> FermionicCircuit:
        r"""Builds the ansatz as a :class:`.FermionicCircuit` (shared by ``_define``).

        The cluster exponential :math:`e^{T - T^\dagger}` is expressed as the evolution
        :math:`e^{-i t H}` with :math:`H = i (T - T^\dagger)` and :math:`t = 1`. Since
        :math:`T - T^\dagger` is anti-Hermitian, that :math:`H` is Hermitian -- which is what
        :class:`.Evolution` requires of its operator.

        The generator carries conjugate-paired groups (see :meth:`.cluster_operator`), which
        :class:`.Evolution` decomposes group-by-group. That keeps every factor of the product formula
        Hermitian and hence unitary; a term-by-term split would not be.
        """
        from qiskit_fermions.circuit import FermionicCircuit

        definition = FermionicCircuit(self.num_modes)

        # e^{T - T^dag} == e^{-i * 1.0 * H} for the Hermitian H = i (T - T^dag)
        definition.append(
            Evolution(self.num_modes, self.cluster_operator() * 1j, time=1.0),  # type: ignore[arg-type]
            definition.modes,
        )

        # ffsim's UCCSD operators can carry a trailing orbital rotation; it acts per spin sector
        if self.uccsd_op.final_orbital_rotation is not None:
            norb = self.norb
            rotation = OrbitalRotation(self.uccsd_op.final_orbital_rotation)
            definition.append(rotation, definition.modes[:norb])
            definition.append(
                OrbitalRotation(self.uccsd_op.final_orbital_rotation), definition.modes[norb:]
            )

        return definition

    def _define(self) -> None:
        self._definition = self._build_definition()._inner
