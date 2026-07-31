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

"""Unitary coupled cluster (UCC) ansatz gate."""

from __future__ import annotations

import itertools
import sys
from enum import Enum
from typing import TYPE_CHECKING, cast

import numpy as np

from qiskit_fermions._lib.operators.fermion_operator import FermionOperator
from qiskit_fermions.operators.fermion_action import ann, cre

from .. import FermionicGate
from .evolution import Evolution

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qiskit_fermions.circuit import FermionicCircuit


class UCC(FermionicGate):
    r"""Implements the unitary coupled cluster (UCC) ansatz.

    A unitary coupled cluster operator has the form

    .. math::

        e^{T - T^\dagger}

    where :math:`T = T_1 + T_2` is the cluster operator built from the single and double fermionic
    excitations, parameterized by the :math:`t_1` and :math:`t_2` amplitudes. Since
    :math:`T - T^\dagger` is anti-Hermitian, its exponential is unitary.

    This gate supports three spin variants (see :class:`.UCC.Variant`), selected explicitly by the
    ``variant`` argument and validated against the shapes of the supplied amplitudes (mirroring
    ffsim's :external:class:`~ffsim.UCCSDOpRestrictedReal` and
    :external:class:`~ffsim.UCCSDOpUnrestrictedReal`):

    - **restricted** -- a single spin-summed amplitude pair. The cluster operator is

      .. math::

          \begin{align}
          T_1 &= \sum_{ia} t_{ia}\left(
          a^\dagger_{a\alpha} a_{i\alpha} + a^\dagger_{a\beta} a_{i\beta}\right), \\
          T_2 &= \sum_{ijab} t_{ijab}\left[
          \frac12\left(
          a^\dagger_{a\alpha} a^\dagger_{b\alpha} a_{j\alpha} a_{i\alpha}
          + a^\dagger_{a\beta} a^\dagger_{b\beta} a_{j\beta} a_{i\beta}\right)
          + a^\dagger_{a\alpha} a^\dagger_{b\beta} a_{j\beta} a_{i\alpha}\right],
          \end{align}

      with ``t1`` of shape ``(nocc, nvrt)`` and ``t2`` of shape ``(nocc, nocc, nvrt, nvrt)``. Acts
      on ``2 * norb`` block-spin modes.
    - **unrestricted** -- independent per-spin amplitudes. The cluster operator is

      .. math::

          \begin{align}
          T_1 &= \sum_{ia} t^{(\alpha)}_{ia} a^\dagger_{a\alpha} a_{i\alpha}
          + \sum_{IA} t^{(\beta)}_{IA} a^\dagger_{A\beta} a_{I\beta}, \\
          T_2 &= \frac14 \sum_{ijab} t^{(\alpha\alpha)}_{ijab}
          a^\dagger_{a\alpha} a^\dagger_{b\alpha} a_{j\alpha} a_{i\alpha}
          + \frac14 \sum_{IJAB} t^{(\beta\beta)}_{IJAB}
          a^\dagger_{A\beta} a^\dagger_{B\beta} a_{J\beta} a_{I\beta}
          + \sum_{iJaB} t^{(\alpha\beta)}_{iJaB}
          a^\dagger_{a\alpha} a^\dagger_{B\beta} a_{J\beta} a_{i\alpha},
          \end{align}

      with ``t1`` a pair ``(t1a, t1b)`` and ``t2`` a triple ``(t2aa, t2ab, t2bb)``. Acts on
      ``2 * norb`` block-spin modes. Note that the occupied/virtual split is resolved **per spin
      sector**, so the two sectors may have different numbers of occupied orbitals.
    - **spinless** -- a single register of ``norb`` spinless modes,

      .. math::

          T_1 = \sum_{ia} t_{ia}\, a^\dagger_a a_i, \qquad
          T_2 = \frac14 \sum_{ijab} t_{ijab}\,
          a^\dagger_a a^\dagger_b a_j a_i,

      with the same amplitude shapes as the ``"restricted"`` variant. Acts on ``norb`` modes.

    In every variant the occupied orbitals are ordered before the virtual ones, so orbital
    :math:`i < n_\text{occ}` is occupied and orbital :math:`n_\text{occ} + a` is virtual.

    .. note::
       Unlike :class:`.UCJ`, this ansatz carries no final orbital rotation: its :math:`t_1`
       amplitudes already provide the single excitations, so a trailing rotation would be redundant
       freedom. Append an :class:`.OrbitalRotation` explicitly if you want one.

    .. note::
       By default only the symmetry the cluster operator actually enforces is imposed on the
       same-spin :math:`t_2` blocks. The stricter antisymmetry of the standard coupled-cluster
       convention is available opt-in via the ``antisymmetric`` flag (see :attr:`.antisymmetric`),
       which both validates supplied amplitudes and shrinks the parameter vector accordingly. It is
       not supported for the ``"restricted"`` variant, whose single :math:`t_2` also carries the
       cross-spin amplitudes.

    .. note::
       Because the individual excitation terms of :math:`T - T^\dagger` do **not** commute, the
       circuit :meth:`~qiskit.circuit.Gate.definition` this gate produces is a *first-order product
       formula* (Trotter) approximation of the exponential, not an exact decomposition -- the usual
       situation for UCC ansatz circuits. The state-vector simulation path
       (:meth:`_apply_unitary_placed_`), by contrast, applies the exponential *exactly* via
       ``scipy``'s ``expm_multiply``. Consequently the simulated gate and its synthesized circuit
       agree only up to the Trotter error; use a higher-order product formula during transpilation to
       tighten it.

    .. caution::
       This is an early development prototype. Beware of changes to its interface without warning
       during the pre-release development of this package.
    """

    class Variant(Enum):
        """The spin variant of a :class:`.UCC` ansatz.

        Mirrors ffsim's UCCSD operator flavors
        (:external:class:`~ffsim.UCCSDOpRestrictedReal` and
        :external:class:`~ffsim.UCCSDOpUnrestrictedReal`), plus a spinless variant. Passed explicitly
        to :meth:`.UCC.__init__` (or its string value) and validated against the supplied amplitude
        shapes.
        """

        SPINLESS = "spinless"
        RESTRICTED = "restricted"
        UNRESTRICTED = "unrestricted"

    def __init__(
        self,
        variant: UCC.Variant | str,
        t1: np.ndarray | tuple[np.ndarray, np.ndarray],
        t2: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray],
        *,
        antisymmetric: bool = False,
        atol: float = 1e-8,
    ) -> None:
        r"""Initializing an instance of this gate can be done with the arguments listed below.

        Args:
            variant: the spin variant, a :class:`.UCC.Variant` (or its string value
                ``"restricted"``, ``"unrestricted"``, or ``"spinless"``). Determines the number of
                modes this gate acts on (see the class docstring) and the expected amplitude shapes.
            t1: the :math:`t_1` (singles) amplitudes. For the ``"restricted"`` and ``"spinless"``
                variants, a single array of shape ``(nocc, nvrt)``. For the ``"unrestricted"``
                variant, a pair ``(t1a, t1b)``.
            t2: the :math:`t_2` (doubles) amplitudes. For the ``"restricted"`` and ``"spinless"``
                variants, a single array of shape ``(nocc, nocc, nvrt, nvrt)``. For the
                ``"unrestricted"`` variant, a triple ``(t2aa, t2ab, t2bb)``.
            antisymmetric: whether the same-spin :math:`t_2` blocks obey the *separate* occupied and
                virtual antisymmetry (the standard coupled-cluster convention, see
                :attr:`.antisymmetric`). When ``True`` the supplied blocks are validated against it
                and the parameter vector is restricted to the corresponding subspace. Not supported
                for the ``"restricted"`` variant.
            atol: the absolute tolerance for the ``antisymmetric`` validation.

        Raises:
            ValueError: if ``variant`` is not recognized, if the amplitude shapes are inconsistent
                with each other or with ``variant``, if ``antisymmetric`` is requested for the
                ``"restricted"`` variant, or if ``antisymmetric`` is requested but a same-spin
                :math:`t_2` block violates that antisymmetry.
        """
        variant = self._normalize_variant(variant)
        self._variant = variant
        self._validate_antisymmetric_supported(variant, antisymmetric)

        self.antisymmetric = antisymmetric
        r"""Whether the same-spin :math:`t_2` blocks obey the separate occupied/virtual antisymmetry.

        The cluster operator only ever sees the part of a same-spin :math:`t_2` block that is
        symmetric under the *simultaneous* exchange
        :math:`t_2[i,j,a,b] = t_2[j,i,b,a]`, because the underlying excitation
        :math:`a^\dagger_a a^\dagger_b a_j a_i` is invariant under relabeling the pairs
        :math:`(i,a) \leftrightarrow (j,b)`. That weaker symmetry is therefore always imposed.

        The standard coupled-cluster convention additionally makes the block antisymmetric in each
        index pair *separately*, :math:`t_2[i,j,a,b] = -t_2[j,i,a,b] = -t_2[i,j,b,a]`, which is a
        strict subspace of the above. Setting this flag opts into that convention: the supplied
        amplitudes are validated against it, and :meth:`.num_parameters` /
        :meth:`.from_parameters` / :meth:`.to_parameters` switch to the smaller parameter basis that
        spans exactly this subspace.
        """
        self._atol = atol

        if variant is UCC.Variant.UNRESTRICTED:
            # the variant selects which argument shapes are valid; mypy cannot narrow the unions
            self.t1: np.ndarray | tuple[np.ndarray, ...] = tuple(
                np.asarray(t, dtype=complex) for t in cast("tuple", t1)
            )
            """The :math:`t_1` (singles) amplitudes."""
            self.t2: np.ndarray | tuple[np.ndarray, ...] = tuple(
                np.asarray(t, dtype=complex) for t in cast("tuple", t2)
            )
            """The :math:`t_2` (doubles) amplitudes."""
        else:
            self.t1 = np.asarray(cast(np.ndarray, t1), dtype=complex)
            self.t2 = np.asarray(cast(np.ndarray, t2), dtype=complex)

        norb = self._validate_shapes()
        self.norb = norb
        """The number of spatial orbitals (or spinless modes, for the spinless variant)."""

        if antisymmetric:
            # validated only after the shapes are known to be consistent, so the transpose-based
            # checks below cannot fail on a malformed tensor instead of a genuinely asymmetric one
            self._validate_antisymmetric_amplitudes()

        num_modes = norb if variant is UCC.Variant.SPINLESS else 2 * norb

        super().__init__("UCC", num_modes, [])

    @classmethod
    def from_t_amplitudes(
        cls,
        t2: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray],
        *,
        t1: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
        variant: UCC.Variant | str = "restricted",
        antisymmetric: bool = False,
        atol: float = 1e-8,
    ) -> Self:
        r"""Constructs a UCC ansatz from coupled-cluster :math:`t_2` (and optional :math:`t_1`) amplitudes.

        This is a convenience constructor mirroring :meth:`.UCJ.from_t_amplitudes`. Unlike the (L)UCJ
        ansatz -- which *factorizes* the amplitudes into diagonal Coulomb layers -- the UCC ansatz
        uses the amplitudes directly as its parameters, so this simply defaults an omitted ``t1`` to
        zeros of the shape implied by ``t2``, giving a doubles-only (UCCD) ansatz.

        Args:
            t2: the :math:`t_2` amplitudes. For the ``"restricted"`` and ``"spinless"`` variants, a
                single array of shape ``(nocc, nocc, nvrt, nvrt)``. For the ``"unrestricted"``
                variant, a triple ``(t2aa, t2ab, t2bb)``.
            t1: the optional :math:`t_1` amplitudes. For ``"unrestricted"``, a pair ``(t1a, t1b)``;
                otherwise a single array of shape ``(nocc, nvrt)``. Defaults to zeros.
            variant: the spin variant to build, a :class:`.UCC.Variant` (or its string value
                ``"restricted"``, ``"unrestricted"``, or ``"spinless"``).
            antisymmetric: whether to assert the standard coupled-cluster antisymmetry of the
                same-spin :math:`t_2` blocks (see :attr:`.antisymmetric`). Amplitudes from a genuine
                coupled-cluster calculation satisfy it, so this is a cheap way to confirm they
                survived whatever preprocessing produced them. Not supported for the ``"restricted"``
                variant.
            atol: the absolute tolerance for the ``antisymmetric`` validation.

        Returns:
            The constructed :class:`.UCC` gate.

        Raises:
            ValueError: if ``variant`` is not recognized, if the amplitude shapes are inconsistent
                with each other or with ``variant``, if ``antisymmetric`` is requested for the
                ``"restricted"`` variant, or if ``antisymmetric`` is requested but a same-spin
                :math:`t_2` block violates that antisymmetry.
        """
        variant = cls._normalize_variant(variant)

        if t1 is None:
            t1 = cls._zero_t1_like(t2, variant)

        return cls(variant, t1, t2, antisymmetric=antisymmetric, atol=atol)

    @classmethod
    def num_parameters(
        cls,
        norb: int,
        nocc: int | tuple[int, int],
        variant: UCC.Variant | str,
        *,
        antisymmetric: bool = False,
    ) -> int:
        r"""Returns the number of parameters of a UCC ansatz with the given settings.

        Args:
            norb: the number of spatial orbitals (or spinless modes, for the spinless variant).
            nocc: the number of occupied orbitals. For the ``"unrestricted"`` variant a pair
                ``(nocc_a, nocc_b)`` giving the per-spin occupations; otherwise a single integer.
            variant: the spin variant, a :class:`.UCC.Variant` (or its string value
                ``"restricted"``, ``"unrestricted"``, or ``"spinless"``).
            antisymmetric: whether the same-spin :math:`t_2` blocks are restricted to the standard
                coupled-cluster antisymmetric subspace (see :attr:`.antisymmetric`), which needs
                strictly fewer parameters. Not supported for the ``"restricted"`` variant.

        Returns:
            The number of parameters.

        Raises:
            ValueError: if ``variant`` is not recognized, if ``nocc`` is a pair for a variant other
                than :attr:`.UCC.Variant.UNRESTRICTED` (or an integer for that variant), or if
                ``antisymmetric`` is requested for the ``"restricted"`` variant.
        """
        variant = cls._normalize_variant(variant)
        nocc = cls._normalize_nocc(nocc, variant)
        cls._validate_antisymmetric_supported(variant, antisymmetric)

        if variant is UCC.Variant.UNRESTRICTED:
            nocc_a, nocc_b = cast("tuple[int, int]", nocc)
            nvrt_a, nvrt_b = norb - nocc_a, norb - nocc_b
            return (
                # t1a and t1b
                nocc_a * nvrt_a
                + nocc_b * nvrt_b
                # the same-spin t2aa and t2bb blocks (the cross-spin t2ab is unconstrained)
                + cls._same_spin_block_num_parameters(nocc_a, nvrt_a, antisymmetric)
                + nocc_a * nocc_b * nvrt_a * nvrt_b
                + cls._same_spin_block_num_parameters(nocc_b, nvrt_b, antisymmetric)
            )

        # restricted and spinless both carry a single t1/t2 pair
        nocc_int = cast(int, nocc)
        nvrt = norb - nocc_int
        return nocc_int * nvrt + cls._same_spin_block_num_parameters(nocc_int, nvrt, antisymmetric)

    @classmethod
    def from_parameters(
        cls,
        params: np.ndarray,
        norb: int,
        nocc: int | tuple[int, int],
        variant: UCC.Variant | str,
        *,
        antisymmetric: bool = False,
    ) -> Self:
        r"""Constructs a UCC ansatz from a real-valued parameter vector.

        With ``antisymmetric=False`` (the default) the parameter ordering matches ffsim's
        :external:class:`~ffsim.UCCSDOpRestrictedReal` /
        :external:class:`~ffsim.UCCSDOpUnrestrictedReal` convention, so a vector produced by ffsim's
        own ``to_parameters`` round-trips through this method.

        With ``antisymmetric=True`` the same-spin :math:`t_2` blocks are instead built from the
        smaller basis spanning the standard coupled-cluster antisymmetric subspace (see
        :attr:`.antisymmetric`), so the expected vector length differs and ffsim's vectors no longer
        apply.

        Args:
            params: the real-valued parameter vector.
            norb: the number of spatial orbitals (or spinless modes, for the spinless variant).
            nocc: the number of occupied orbitals. For the ``"unrestricted"`` variant a pair
                ``(nocc_a, nocc_b)``; otherwise a single integer.
            variant: the spin variant, a :class:`.UCC.Variant` (or its string value
                ``"restricted"``, ``"unrestricted"``, or ``"spinless"``).
            antisymmetric: whether to build the same-spin :math:`t_2` blocks in the antisymmetric
                subspace (see :attr:`.antisymmetric`). The resulting amplitudes then satisfy that
                antisymmetry by construction. Not supported for the ``"restricted"`` variant.

        Returns:
            The constructed :class:`.UCC` gate.

        Raises:
            ValueError: if ``variant`` is not recognized, if ``antisymmetric`` is requested for the
                ``"restricted"`` variant, or if ``len(params)`` does not match
                :meth:`.num_parameters` for the given settings.
        """
        variant = cls._normalize_variant(variant)
        nocc = cls._normalize_nocc(nocc, variant)
        cls._validate_antisymmetric_supported(variant, antisymmetric)
        expected = cls.num_parameters(norb, nocc, variant, antisymmetric=antisymmetric)
        if len(params) != expected:
            raise ValueError(
                "The number of parameters passed did not match the number expected based on the "
                f"given settings. Expected {expected} but got {len(params)}."
            )

        index = 0
        t1: np.ndarray | tuple[np.ndarray, np.ndarray]
        t2: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]

        if variant is UCC.Variant.UNRESTRICTED:
            nocc_a, nocc_b = cast("tuple[int, int]", nocc)
            nvrt_a, nvrt_b = norb - nocc_a, norb - nocc_b
            # ffsim's ordering is t1a, t1b, t2aa, t2ab, t2bb -- note that the cross-spin block sits
            # *between* the two same-spin blocks, not after them
            t1a, index = cls._t1_from_parameters(params, index, nocc_a, nvrt_a)
            t1b, index = cls._t1_from_parameters(params, index, nocc_b, nvrt_b)
            t2aa, index = cls._same_spin_t2_from_parameters(
                params, index, nocc_a, nvrt_a, antisymmetric
            )
            shape_ab = (nocc_a, nocc_b, nvrt_a, nvrt_b)
            n_ab = int(np.prod(shape_ab))
            t2ab = np.asarray(params[index : index + n_ab], dtype=float).reshape(shape_ab)
            index += n_ab
            t2bb, index = cls._same_spin_t2_from_parameters(
                params, index, nocc_b, nvrt_b, antisymmetric
            )
            return cls(variant, (t1a, t1b), (t2aa, t2ab, t2bb), antisymmetric=antisymmetric)

        nocc_int = cast(int, nocc)
        nvrt = norb - nocc_int
        t1, index = cls._t1_from_parameters(params, index, nocc_int, nvrt)
        t2, index = cls._same_spin_t2_from_parameters(params, index, nocc_int, nvrt, antisymmetric)

        return cls(variant, t1, t2, antisymmetric=antisymmetric)

    def to_parameters(self) -> np.ndarray:
        r"""Converts this UCC ansatz to a real-valued parameter vector.

        The inverse of :meth:`.from_parameters`, using the same ordering and the same basis this
        gate's :attr:`.antisymmetric` flag selects -- so ``from_parameters(gate.to_parameters(), ...)``
        round-trips as long as the flag is passed consistently.

        .. note::
           Only the independent amplitude entries implied by the variant's symmetries (and by
           :attr:`.antisymmetric`) are written out; see :meth:`.num_parameters`. Amplitudes violating
           those symmetries -- or carrying a non-negligible imaginary part -- are therefore not
           recoverable from the parameter vector.

        .. note::
           The round-trip is two-sided and holds at *any* parameter scale: the amplitudes are this
           ansatz's parameters directly, so both directions are a plain re-indexing.

        Returns:
            The real-valued parameter vector.
        """
        cls = type(self)
        variant = self._variant
        antisymmetric = self.antisymmetric
        params = np.empty(
            cls.num_parameters(self.norb, self._nocc(), variant, antisymmetric=antisymmetric)
        )

        index = 0
        if variant is UCC.Variant.UNRESTRICTED:
            t1a, t1b = cast("tuple[np.ndarray, np.ndarray]", self.t1)
            t2aa, t2ab, t2bb = cast("tuple[np.ndarray, np.ndarray, np.ndarray]", self.t2)
            index = cls._t1_to_parameters(params, index, t1a)
            index = cls._t1_to_parameters(params, index, t1b)
            index = cls._same_spin_t2_to_parameters(params, index, t2aa, antisymmetric)
            params[index : index + t2ab.size] = t2ab.real.ravel()
            index += t2ab.size
            cls._same_spin_t2_to_parameters(params, index, t2bb, antisymmetric)
            return params

        index = cls._t1_to_parameters(params, index, cast(np.ndarray, self.t1))
        cls._same_spin_t2_to_parameters(params, index, cast(np.ndarray, self.t2), antisymmetric)
        return params

    def cluster_operator(self) -> FermionOperator:
        r"""Returns the anti-Hermitian cluster generator :math:`T - T^\dagger`.

        The generator is expressed in the block-spin mode convention (mode ``p`` is alpha orbital
        ``p``, mode ``norb + p`` is beta orbital ``p``) for the spinful variants, and directly on the
        ``norb`` modes for the spinless variant. Occupied orbitals are ordered before virtual ones.

        The returned operator carries :attr:`~qiskit_fermions.operators.FermionOperator.groups` that
        pair every excitation with its Hermitian conjugate; see :meth:`.hermitian_generator` for why
        that grouping matters.

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

    def hermitian_generator(self) -> FermionOperator:
        r"""Returns the Hermitian generator :math:`H = i (T - T^\dagger)` this ansatz evolves under.

        The ansatz unitary is :math:`e^{T - T^\dagger} = e^{-i H}` for this :math:`H`, which is the
        form :class:`.Evolution` consumes (it requires a Hermitian operator to produce a unitary). The
        grouping of :meth:`.cluster_operator` is preserved, so each group of :math:`H` is individually
        Hermitian and every factor of the resulting product formula is unitary.

        Returns:
            The Hermitian generator :math:`i (T - T^\dagger)` as a
            :class:`~qiskit_fermions.operators.FermionOperator`.
        """
        return self.cluster_operator() * 1j

    @staticmethod
    def _normalize_variant(variant: UCC.Variant | str) -> UCC.Variant:
        """Normalizes a variant argument, raising the same ``ValueError`` as ``__init__``."""
        try:
            return UCC.Variant(variant)
        except ValueError:
            raise ValueError(
                f"Unknown UCC variant {variant!r}; expected 'restricted', 'unrestricted', or "
                "'spinless'."
            ) from None

    @staticmethod
    def _normalize_nocc(nocc: int | tuple[int, int], variant: UCC.Variant) -> int | tuple[int, int]:
        """Validates ``nocc`` against ``variant``, rejecting the wrong arity.

        The unrestricted variant resolves its occupied/virtual split per spin sector, so it requires
        a ``(nocc_a, nocc_b)`` pair; the other variants take a single integer. A mismatch would
        otherwise be silently misinterpreted (a pair truncated to its first element, or an integer
        broadcast to both sectors), so reject it explicitly.
        """
        is_pair = isinstance(nocc, tuple)
        if variant is UCC.Variant.UNRESTRICTED and not is_pair:
            raise ValueError(
                f"The 'unrestricted' variant requires a (nocc_a, nocc_b) pair; got nocc={nocc!r}."
            )
        if variant is not UCC.Variant.UNRESTRICTED and is_pair:
            raise ValueError(
                f"A tuple nocc={nocc!r} is only valid for the 'unrestricted' variant; pass a single "
                f"integer for the '{variant.value}' variant."
            )
        return nocc

    @staticmethod
    def _validate_antisymmetric_supported(variant: UCC.Variant, antisymmetric: bool) -> None:
        r"""Rejects ``antisymmetric=True`` for the ``restricted`` variant.

        The restricted ``t2`` does double duty: the *same* tensor supplies both the same-spin
        amplitudes (where the separate occupied/virtual antisymmetry is the standard convention) and
        the alpha-beta amplitudes (where it is not -- there is no antisymmetry to impose between an
        alpha and a beta excitation, since exchanging them is not an exchange of identical operators).
        Imposing the antisymmetry on the restricted tensor would therefore silently constrain the
        cross-spin channel too, yielding an over-constrained ansatz. Refuse explicitly rather than
        offer a flag that means something different here than everywhere else.
        """
        if antisymmetric and variant is UCC.Variant.RESTRICTED:
            raise ValueError(
                "antisymmetric=True is not supported for the 'restricted' variant: its single t2 "
                "tensor supplies both the same-spin amplitudes (where the antisymmetry is the "
                "standard convention) and the alpha-beta amplitudes (where it does not apply), so "
                "imposing it would also constrain the cross-spin channel. Use the 'unrestricted' "
                "variant to constrain only the same-spin blocks, or pass antisymmetric=False."
            )

    def _validate_antisymmetric_amplitudes(self) -> None:
        r"""Validates that the same-spin ``t2`` blocks obey the separate occupied/virtual antisymmetry.

        Only the same-spin blocks are checked; the unrestricted cross-spin ``t2ab`` carries no such
        symmetry. The ``restricted`` variant never reaches here (see
        :meth:`._validate_antisymmetric_supported`).
        """
        if self._variant is UCC.Variant.UNRESTRICTED:
            t2aa, _, t2bb = cast("tuple[np.ndarray, np.ndarray, np.ndarray]", self.t2)
            blocks = [("t2aa", t2aa), ("t2bb", t2bb)]
        else:  # SPINLESS: the single register is exactly one same-spin sector
            blocks = [("t2", cast(np.ndarray, self.t2))]

        for name, t2 in blocks:
            self._check_antisymmetric_block(t2, name, self._atol)

    @staticmethod
    def _check_antisymmetric_block(t2: np.ndarray, name: str, atol: float) -> None:
        """Raises if ``t2`` is not antisymmetric under occupied and virtual exchange separately.

        ``t2 + t2.transpose(...)`` is exactly the symmetric (i.e. antisymmetry-violating) part of the
        tensor under each exchange, so the check needs no index loops.
        """
        for label, violation in (
            ("occupied (i <-> j)", t2 + t2.transpose(1, 0, 2, 3)),
            ("virtual (a <-> b)", t2 + t2.transpose(0, 1, 3, 2)),
        ):
            worst = float(np.abs(violation).max(initial=0.0))
            if worst > atol:
                raise ValueError(
                    f"antisymmetric=True was requested but the {name} amplitudes are not "
                    f"antisymmetric under {label} exchange: largest violation {worst:.3e} exceeds "
                    f"atol={atol:.3e}. Pass antisymmetric=False to allow the full "
                    "exchange-symmetric family."
                )

    @classmethod
    def _same_spin_block_num_parameters(cls, nocc: int, nvrt: int, antisymmetric: bool) -> int:
        """Returns the parameter count of one same-spin ``t2`` block, per symmetry convention."""
        if antisymmetric:
            # the independent entries are the strictly-upper-triangular occupied pairs combined with
            # the strictly-upper-triangular virtual pairs; everything else follows by antisymmetry
            # (and the i == j / a == b entries must vanish)
            return len(cls._antisymmetric_double_indices(nocc, nvrt))
        # the upper triangle of the combined occupied-virtual pair index
        n_pairs = nocc * nvrt
        return n_pairs * (n_pairs + 1) // 2

    @staticmethod
    def _antisymmetric_double_indices(nocc: int, nvrt: int) -> list[tuple[int, int, int, int]]:
        """Returns the independent ``(i, j, a, b)`` entries of an antisymmetric same-spin ``t2``."""
        return [
            (i, j, a, b)
            for i, j in itertools.combinations(range(nocc), 2)
            for a, b in itertools.combinations(range(nvrt), 2)
        ]

    @staticmethod
    def _zero_t1_like(
        t2: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray], variant: UCC.Variant
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Builds zero ``t1`` amplitudes of the shape implied by ``t2`` (a doubles-only ansatz)."""
        if variant is UCC.Variant.UNRESTRICTED:
            t2ab = np.asarray(cast("tuple", t2)[1])
            if t2ab.ndim != 4:
                raise ValueError(
                    "The unrestricted t2ab amplitudes must be 4-dimensional "
                    f"(nocc_a, nocc_b, nvrt_a, nvrt_b); got shape {t2ab.shape}."
                )
            nocc_a, nocc_b, nvrt_a, nvrt_b = t2ab.shape
            return np.zeros((nocc_a, nvrt_a)), np.zeros((nocc_b, nvrt_b))
        t2_arr = np.asarray(cast(np.ndarray, t2))
        if t2_arr.ndim != 4:
            raise ValueError(
                "The t2 amplitudes must be 4-dimensional (nocc, nocc, nvrt, nvrt); got shape "
                f"{t2_arr.shape}."
            )
        nocc, _, nvrt, _ = t2_arr.shape
        return np.zeros((nocc, nvrt))

    @staticmethod
    def _t1_from_parameters(
        params: np.ndarray, index: int, nocc: int, nvrt: int
    ) -> tuple[np.ndarray, int]:
        """Reads an ``(nocc, nvrt)`` ``t1`` block off ``params``, returning it and the new index."""
        n = nocc * nvrt
        t1 = np.asarray(params[index : index + n], dtype=float).reshape((nocc, nvrt))
        return t1, index + n

    @staticmethod
    def _t1_to_parameters(params: np.ndarray, index: int, t1: np.ndarray) -> int:
        """Writes ``t1`` into ``params`` at ``index``, returning the new index."""
        n = int(t1.size)
        params[index : index + n] = t1.real.ravel()
        return index + n

    @staticmethod
    def _occ_vrt_pairs(nocc: int, nvrt: int) -> list[tuple[int, int]]:
        """Returns the ``(i, a)`` occupied-virtual index pairs, in ffsim's parameter order."""
        return list(itertools.product(range(nocc), range(nvrt)))

    @classmethod
    def _same_spin_t2_from_parameters(
        cls, params: np.ndarray, index: int, nocc: int, nvrt: int, antisymmetric: bool
    ) -> tuple[np.ndarray, int]:
        """Reads one same-spin ``t2`` block off ``params``, in the basis ``antisymmetric`` selects."""
        if antisymmetric:
            return cls._antisymmetric_t2_from_parameters(params, index, nocc, nvrt)
        return cls._exchange_symmetric_t2_from_parameters(params, index, nocc, nvrt)

    @classmethod
    def _same_spin_t2_to_parameters(
        cls, params: np.ndarray, index: int, t2: np.ndarray, antisymmetric: bool
    ) -> int:
        """Writes one same-spin ``t2`` block into ``params``, inverting :meth:`._same_spin_t2_from_parameters`."""
        if antisymmetric:
            return cls._antisymmetric_t2_to_parameters(params, index, t2)
        return cls._exchange_symmetric_t2_to_parameters(params, index, t2)

    @classmethod
    def _exchange_symmetric_t2_from_parameters(
        cls, params: np.ndarray, index: int, nocc: int, nvrt: int
    ) -> tuple[np.ndarray, int]:
        """Reads a ``t2`` block off ``params``, filling in its ``(i, a) <-> (j, b)`` symmetry.

        Every ``t2`` block this ansatz parameterizes -- the restricted and spinless single block, and
        each same-spin block of the unrestricted variant -- is symmetric under the *simultaneous*
        exchange of the occupied and virtual indices, i.e. ``t2[i, j, a, b] == t2[j, i, b, a]``. It is
        therefore determined by the upper triangle of the occupied-virtual pair index. This mirrors
        ffsim's ``UCCSDOpRestrictedReal``/``UCCSDOpUnrestrictedReal`` ``from_parameters`` ordering
        exactly: the parameters run over ``combinations_with_replacement`` of those pairs, and each
        parameter sets both ``t2[i, j, a, b]`` and its ``(j, i, b, a)`` partner.
        """
        t2 = np.zeros((nocc, nocc, nvrt, nvrt))
        for (i, a), (j, b) in itertools.combinations_with_replacement(
            cls._occ_vrt_pairs(nocc, nvrt), 2
        ):
            t2[i, j, a, b] = params[index]
            t2[j, i, b, a] = params[index]
            index += 1
        return t2, index

    @classmethod
    def _exchange_symmetric_t2_to_parameters(
        cls, params: np.ndarray, index: int, t2: np.ndarray
    ) -> int:
        """Inverts :meth:`._exchange_symmetric_t2_from_parameters`, reading the same entries back."""
        nocc, _, nvrt, _ = t2.shape
        for (i, a), (j, b) in itertools.combinations_with_replacement(
            cls._occ_vrt_pairs(nocc, nvrt), 2
        ):
            params[index] = t2[i, j, a, b].real
            index += 1
        return index

    @classmethod
    def _antisymmetric_t2_from_parameters(
        cls, params: np.ndarray, index: int, nocc: int, nvrt: int
    ) -> tuple[np.ndarray, int]:
        """Expands the independent ``(i < j, a < b)`` entries into a fully antisymmetric ``t2``.

        Each parameter fixes four entries of the block via
        ``t2[i, j, a, b] == -t2[j, i, a, b] == -t2[i, j, b, a] == t2[j, i, b, a]``, so the result
        satisfies the separate occupied and virtual antisymmetry by construction (see
        :attr:`.antisymmetric`). Entries with ``i == j`` or ``a == b`` are left zero, as that
        antisymmetry requires.
        """
        t2 = np.zeros((nocc, nocc, nvrt, nvrt))
        for i, j, a, b in cls._antisymmetric_double_indices(nocc, nvrt):
            value = params[index]
            t2[i, j, a, b] = value
            t2[j, i, b, a] = value
            t2[j, i, a, b] = -value
            t2[i, j, b, a] = -value
            index += 1
        return t2, index

    @classmethod
    def _antisymmetric_t2_to_parameters(cls, params: np.ndarray, index: int, t2: np.ndarray) -> int:
        """Inverts :meth:`._antisymmetric_t2_from_parameters`, reading the independent entries back."""
        nocc, _, nvrt, _ = t2.shape
        for i, j, a, b in cls._antisymmetric_double_indices(nocc, nvrt):
            params[index] = t2[i, j, a, b].real
            index += 1
        return index

    def _nocc(self) -> int | tuple[int, int]:
        """Returns the number of occupied orbitals, per spin sector for the unrestricted variant."""
        if self._variant is UCC.Variant.UNRESTRICTED:
            t1a, t1b = cast("tuple[np.ndarray, np.ndarray]", self.t1)
            return int(t1a.shape[0]), int(t1b.shape[0])
        return int(cast(np.ndarray, self.t1).shape[0])

    def _validate_shapes(self) -> int:
        """Validates the amplitude shapes, returning the implied ``norb``.

        Raises:
            ValueError: if the amplitude shapes are inconsistent with each other, or if the
                unrestricted variant's two spin sectors imply different orbital counts.
        """
        variant = self._variant

        if variant is not UCC.Variant.UNRESTRICTED:
            t1_arr = cast(np.ndarray, self.t1)
            t2_arr = cast(np.ndarray, self.t2)
            if t1_arr.ndim != 2:
                raise ValueError(
                    "The t1 amplitudes must be 2-dimensional (nocc, nvrt); got shape "
                    f"{t1_arr.shape}."
                )
            nocc, nvrt = t1_arr.shape
            if t2_arr.shape != (nocc, nocc, nvrt, nvrt):
                raise ValueError(
                    f"Inconsistent {variant.value} UCC amplitude shapes: t2 should have shape "
                    f"{(nocc, nocc, nvrt, nvrt)} (implied by t1's shape {t1_arr.shape}) but got "
                    f"{t2_arr.shape}."
                )
            return int(nocc + nvrt)

        t1_tuple = cast("tuple[np.ndarray, ...]", self.t1)
        t2_tuple = cast("tuple[np.ndarray, ...]", self.t2)
        if len(t1_tuple) != 2:
            raise ValueError(
                "The 'unrestricted' variant expects a (t1a, t1b) pair of t1 amplitudes; got "
                f"{len(t1_tuple)} array(s)."
            )
        if len(t2_tuple) != 3:
            raise ValueError(
                "The 'unrestricted' variant expects a (t2aa, t2ab, t2bb) triple of t2 amplitudes; "
                f"got {len(t2_tuple)} array(s)."
            )
        t1a, t1b = t1_tuple
        t2aa, t2ab, t2bb = t2_tuple
        if t1a.ndim != 2 or t1b.ndim != 2:
            raise ValueError(
                "The unrestricted t1 amplitudes must each be 2-dimensional (nocc, nvrt); got "
                f"shapes {t1a.shape} and {t1b.shape}."
            )
        nocc_a, nvrt_a = t1a.shape
        nocc_b, nvrt_b = t1b.shape
        # both spin sectors span the same set of spatial orbitals, so their occupied/virtual splits
        # must add up to the same norb -- otherwise the block-spin register is ill-defined
        if nocc_a + nvrt_a != nocc_b + nvrt_b:
            raise ValueError(
                "The unrestricted alpha and beta t1 amplitudes imply different numbers of spatial "
                f"orbitals: {nocc_a + nvrt_a} (alpha) vs {nocc_b + nvrt_b} (beta)."
            )
        expected = {
            "t2aa": ((nocc_a, nocc_a, nvrt_a, nvrt_a), t2aa.shape),
            "t2ab": ((nocc_a, nocc_b, nvrt_a, nvrt_b), t2ab.shape),
            "t2bb": ((nocc_b, nocc_b, nvrt_b, nvrt_b), t2bb.shape),
        }
        for name, (want, got) in expected.items():
            if got != want:
                raise ValueError(
                    f"Inconsistent unrestricted UCC amplitude shapes: {name} should have shape "
                    f"{want} (implied by the t1 amplitudes) but got {got}."
                )
        return int(nocc_a + nvrt_a)

    def _excitation_operator(self) -> FermionOperator:
        r"""Builds the (non-Hermitian) cluster operator :math:`T = T_1 + T_2`.

        The terms are accumulated into a dict keyed by the action tuple, because distinct
        ``(i, j, a, b)`` index combinations can map onto the *same* action tuple (for instance the
        restricted same-spin doubles, where ``(i, j, a, b)`` and ``(j, i, b, a)`` produce the same
        actions); their coefficients must add rather than overwrite each other.
        """
        variant = self._variant
        norb = self.norb
        # spinless acts on a single register, so its (single) sector carries no spin offset
        spinless = variant is UCC.Variant.SPINLESS

        terms: dict[tuple, complex] = {}

        def mode(sigma: int, orb: int) -> int:
            """Maps a ``(spin sector, orbital)`` pair onto its block-spin mode index."""
            return orb if spinless else orb + sigma * norb

        def add(actions: tuple, coeff: complex) -> None:
            if coeff == 0.0:
                return
            terms[actions] = terms.get(actions, 0.0) + coeff

        match variant:
            case UCC.Variant.UNRESTRICTED:
                self._add_unrestricted_terms(add, mode)
            case UCC.Variant.RESTRICTED:
                self._add_restricted_terms(add, mode)
            case _:  # SPINLESS
                self._add_spinless_terms(add, mode)

        return FermionOperator.from_dict(terms)  # type: ignore[arg-type]

    def _add_restricted_terms(self, add, mode) -> None:
        r"""Adds the restricted (spin-summed) :math:`T_1` and :math:`T_2` terms.

        Follows ffsim's ``singles_excitations_restricted``/``doubles_excitations_restricted``
        convention: the singles are applied identically in both spin sectors, and the doubles carry a
        factor ``1/2`` on the two same-spin blocks and ``1`` on the alpha-beta block.
        """
        t1 = cast(np.ndarray, self.t1)
        t2 = cast(np.ndarray, self.t2)
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
        t1a, t1b = cast("tuple[np.ndarray, np.ndarray]", self.t1)
        t2aa, t2ab, t2bb = cast("tuple[np.ndarray, np.ndarray, np.ndarray]", self.t2)
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

    def _add_spinless_terms(self, add, mode) -> None:
        r"""Adds the spinless :math:`T_1` and :math:`T_2` terms on the single ``norb``-mode register.

        The doubles carry the same ``1/4`` factor as an unrestricted same-spin block, since a
        spinless register is exactly one such sector.
        """
        t1 = cast(np.ndarray, self.t1)
        t2 = cast(np.ndarray, self.t2)
        nocc, nvrt = t1.shape

        for i, a in itertools.product(range(nocc), range(nvrt)):
            add((cre(mode(0, nocc + a)), ann(mode(0, i))), complex(t1[i, a]))

        for i, j, a, b in itertools.product(range(nocc), range(nocc), range(nvrt), range(nvrt)):
            add(
                (
                    cre(mode(0, nocc + a)),
                    cre(mode(0, nocc + b)),
                    ann(mode(0, j)),
                    ann(mode(0, i)),
                ),
                0.25 * complex(t2[i, j, a, b]),
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
            Evolution(self.num_modes, self.hermitian_generator(), time=1.0),  # type: ignore[arg-type]
            definition.modes,
        )

        return definition

    def _define(self) -> None:
        self._definition = self._build_definition()._inner
