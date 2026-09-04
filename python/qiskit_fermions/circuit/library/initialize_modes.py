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

"""Mode initialization gate."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from math import comb

import numpy as np

from qiskit_fermions.utils.optionals import HAS_FFSIM

from .. import FermionicGate

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

# The absolute tolerance for the subspace-confinement check. There is no project-wide tolerance
# constant; this is the de-facto convention (numpy's ``allclose`` default, matching the Rust
# operator-method default) applied consistently across the apply-unitary stack.
_ATOL = 1e-8


@HAS_FFSIM.require_in_call("InitializeModes._apply_unitary_")
def _occupation_axis_mask(norb: int, nocc: int, occupied: int, empty: int) -> np.ndarray:
    """Returns the determinants of one FCI axis that match a partial occupation.

    Marks every determinant of the ``(norb, nocc)`` single-spin axis whose orbital occupation agrees
    with the constraint: every orbital whose bit is set in ``occupied`` is filled, and every orbital
    whose bit is set in ``empty`` is not. Orbitals named in neither are left free.

    The determinants are enumerated in the FCI address order that
    :func:`ffsim.addresses_to_strings` defines, which is the order the state vector this mask indexes
    is laid out in.

    Args:
        norb: the number of spatial orbitals of the axis.
        nocc: the number of occupied orbitals of the axis.
        occupied: a bitmask of the orbitals constrained to be occupied.
        empty: a bitmask of the orbitals constrained to be empty.

    Returns:
        A boolean mask over the axis, ``True`` where the determinant matches the constraint.
    """
    import ffsim

    strings = ffsim.addresses_to_strings(range(ffsim.dim(norb, nocc)), norb=norb, nelec=nocc)
    return np.fromiter(
        ((s & occupied) == occupied and not (s & empty) for s in strings),
        dtype=bool,
        count=len(strings),
    )


class InitializeModes(FermionicGate):
    """Prepares (or, under simulation, certifies) a fermionic mode occupation.

    This gate declares an intended occupation of the modes it is placed on. Its behavior depends on
    how the circuit is consumed:

    - **Transpiled** (synthesized to qubit gates): it *produces* the state -- the synthesis plugin
      emits the gates that set the named modes to their occupation, as one would expect of an
      initialization gate.
    - **Simulated** (:meth:`_apply_unitary_`): it acts as a *validator* rather than a producer. Given
      a state vector it checks that the vector's amplitude is confined to the subspace its
      :attr:`occupation` defines and returns the vector unchanged, certifying -- without mutating --
      that the incoming state is the intended reference so the transforms that follow act on it.

    In both modes the gate constrains only the orbitals the occupation names (and, per spin sector,
    only along that sector's axis), so several :class:`InitializeModes` gates can be placed **in
    parallel** to seed disjoint fragments of a state independently -- e.g. one gate per spin sector,
    or one per orbital group. A spinful gate may cover a single sector (any fragment of it) or fully
    specify both sectors, but a partial straddle of both is rejected (see
    :meth:`_apply_unitary_placed_`).

    Use :meth:`from_hartree_fock` to construct the occupation of a Hartree-Fock reference.

    .. caution::
       This is an early development prototype. Beware of changes to its interface without warning
       during the pre-release development of this package.
    """

    def __init__(self, occupation: Sequence[bool]) -> None:
        """Initializing an instance of this gate can be done with the arguments listed below.

        Args:
            occupation: a sequence of booleans indicating the occupation for each mode in the
                :class:`.FermionicRegister` being initialized by this gate.
        """
        self.occupation = np.asarray(occupation, dtype=bool)
        """The sequence of booleans indicating the occupation for each mode in the
        :class:`~.FermionicRegister` being initialized by this gate."""

        super().__init__("InitializeModes", len(self.occupation), [])

    @classmethod
    def from_hartree_fock(cls, norb: int, nelec: int | tuple[int, int]) -> Self:
        """Builds the gate for the Hartree-Fock reference occupation of ``(norb, nelec)``.

        The Hartree-Fock determinant fills the lowest-indexed orbitals of each spin sector. Whether
        the reference is spinless or spinful is inferred from ``nelec`` (an ``int`` selects the
        spinless interpretation of the ``norb`` modes; a ``(n_alpha, n_beta)`` pair selects the
        spinful block-spin interpretation of the ``2 * norb`` modes). Place the returned gate on the
        matching register to certify a Hartree-Fock reference state.

        Args:
            norb: the number of spatial orbitals.
            nelec: either a single integer for a spinless system, or a pair of integers storing the
                numbers of spin alpha and spin beta fermions.

        Returns:
            An :class:`InitializeModes` gate whose occupation is the Hartree-Fock determinant.

        Raises:
            ValueError: if the electron count exceeds the ``norb`` orbitals available in a sector.
        """
        nelec = cls._normalize_nelec(nelec)
        if isinstance(nelec, int):
            if nelec > norb:
                raise ValueError(f"nelec={nelec!r} exceeds the norb={norb} spinless modes.")
            occ = [i < nelec for i in range(norb)]
            return cls(occ)

        n_alpha, n_beta = nelec
        if n_alpha > norb or n_beta > norb:
            raise ValueError(
                f"nelec={nelec!r} has a spin sector exceeding the norb={norb} available orbitals."
            )
        occ = [False] * (2 * norb)
        for i in range(n_alpha):
            occ[i] = True
        for i in range(n_beta):
            occ[norb + i] = True
        return cls(occ)

    def _apply_unitary_placed_(
        self,
        vec: np.ndarray,
        norb: int,
        nelec: int | tuple[int, int],
        copy: bool,
        freg_indices: list[int],
    ) -> np.ndarray:
        r"""Asserts that ``vec`` is confined to this gate's occupation subspace, returning it unchanged.

        Unlike a transform gate, this gate does not modify the state: it *checks* that ``vec``'s
        amplitude lives entirely in the subspace its :attr:`occupation` defines, and if so returns
        ``vec`` untouched. The subspace is the set of determinants whose occupation agrees with this
        gate on the orbitals it names, with every unnamed orbital left free -- so a partial occupation
        (a fragment of a sector) accepts a whole family of determinants and the check composes with
        other parallel :class:`InitializeModes` gates.

        The gate's local :attr:`occupation` (one flag per local mode) is placed onto the global
        register via ``freg_indices``: local mode ``i`` constrains global mode ``freg_indices[i]``.
        The global modes are then interpreted under the ``(norb, nelec)`` convention:

        - **Spinless** (``nelec`` is an ``int``): the ``norb`` modes are orbitals directly; the check
          is over the ``C(norb, nelec)``-dimensional space.
        - **Spinful** (``nelec`` is a pair): under the block-spin convention modes ``0..norb`` are
          alpha orbitals and modes ``norb..2*norb`` are beta orbitals. A gate touching a single
          sector constrains that sector's axis of the ``(dim_a, dim_b)`` state (a set of full rows
          for an alpha gate, or full columns for a beta gate) and leaves the other axis free, so it
          composes with parallel gates on the other sector. A gate may also cover *both* sectors, but
          only when it pins a complete determinant (every orbital of both sectors named, none left
          free); a *partial* straddle -- constraining some orbitals of both sectors while leaving
          others free -- is rejected, since it is not a product of per-axis subspaces and cannot
          compose (use one gate per sector instead).

        The check is on *confinement*, not equality: an incoming amplitude may carry any phase and
        any magnitude within the subspace (a global phase or normalization is physically irrelevant),
        so this validates the reference without pinning it to a specific determinant vector.

        Args:
            vec: the state vector to validate. Its length must match the ``(norb, nelec)`` sector
                dimension.
            norb: the number of spatial orbitals of the *global* state vector.
            nelec: either a single integer for a spinless system, or a pair of integers storing the
                numbers of spin alpha and spin beta fermions. An integer selects the spinless mode
                interpretation (the ``norb`` modes are orbitals); a pair selects the spinful
                ``(orb, spin)`` block-spin interpretation of the ``2 * norb`` modes.
            copy: accepted for protocol conformance but has no effect -- this gate does not mutate the
                state, so ``vec`` is returned as-is regardless.
            freg_indices: the absolute (global) mode indices that this gate's local modes map onto.

        Returns:
            The input ``vec``, unchanged, once its confinement to the occupation subspace is verified.

        Raises:
            ValueError: if an occupied mode falls outside the range implied by ``norb``; if a spinful
                gate partially straddles both spin sectors (without pinning a full determinant); if
                ``vec``'s length does not match the ``(norb, nelec)`` sector dimension; or if ``vec``
                has amplitude outside the subspace the occupation defines.
        """
        num_modes = norb if isinstance(nelec, int) else 2 * norb

        # place the local occupation onto its global modes, keeping the occupied/empty split
        placed = sorted(
            (int(g), bool(occ)) for g, occ in zip(freg_indices, self.occupation, strict=True)
        )
        global_modes = [g for g, _ in placed]

        if global_modes and (global_modes[0] < 0 or global_modes[-1] >= num_modes):
            raise ValueError(
                f"InitializeModes places a mode outside the range [0, {num_modes}) "
                f"implied by norb={norb} and nelec={nelec!r}."
            )

        if isinstance(nelec, int):
            occupied_bits = sum(1 << g for g, occ in placed if occ)
            empty_bits = sum(1 << g for g, occ in placed if not occ)
            mask = _occupation_axis_mask(norb, nelec, occupied_bits, empty_bits)
            # the sole axis: amplitude on any determinant outside the mask must vanish
            if len(vec) != len(mask):
                raise ValueError(
                    f"InitializeModes received a state vector of length {len(vec)}, which does not "
                    f"match the dimension {len(mask)} of the (norb={norb}, nelec={nelec!r}) sector."
                )
            if not np.allclose(vec[~mask], 0.0, atol=_ATOL):
                raise ValueError(
                    "InitializeModes: the state vector has amplitude outside the subspace defined by "
                    f"its occupation for the (norb={norb}, nelec={nelec!r}) sector."
                )
            return vec

        # Spinful: split the placed modes into the two spin sectors. A gate constrains one axis of the
        # (dim_a, dim_b) state per sector it touches -- an alpha gate fixes rows, a beta gate columns.
        n_alpha, n_beta = nelec
        alpha = [(g, occ) for g, occ in placed if g < norb]
        beta = [(g - norb, occ) for g, occ in placed if g >= norb]

        # A gate touching *both* sectors is a joint constraint. It is a well-defined per-axis product
        # only when it pins a complete determinant -- every orbital of both sectors named, none left
        # free (i.e. it covers all 2*norb modes). A partial straddle (some orbitals free in a sector)
        # is not a product of per-axis subspaces and cannot compose with parallel gates, so reject it.
        if alpha and beta and not (len(alpha) == norb and len(beta) == norb):
            raise ValueError(
                "InitializeModes straddles both spin sectors without pinning a full determinant "
                f"(some orbitals are left free) under norb={norb}. A spinful InitializeModes must "
                "either lie entirely within one spin sector or fully specify both; place a separate "
                "gate per sector to seed disjoint fragments in parallel."
            )

        dim_a = comb(norb, n_alpha)
        dim_b = comb(norb, n_beta)
        if len(vec) != dim_a * dim_b:
            raise ValueError(
                f"InitializeModes received a state vector of length {len(vec)}, which does not match "
                f"the dimension {dim_a * dim_b} of the (norb={norb}, nelec={nelec!r}) sector."
            )
        vec_2d = np.asarray(vec).reshape(dim_a, dim_b)

        # Constrain each touched axis. A sector the gate does not touch is left free (its mask is the
        # whole axis), so a single-sector gate constrains only its own axis and the other stays open.
        if alpha:
            mask_a = _occupation_axis_mask(
                norb,
                n_alpha,
                sum(1 << g for g, occ in alpha if occ),
                sum(1 << g for g, occ in alpha if not occ),
            )
            if not np.allclose(vec_2d[~mask_a, :], 0.0, atol=_ATOL):
                raise ValueError(
                    "InitializeModes: the state vector has amplitude outside the alpha-sector "
                    f"subspace defined by its occupation for the (norb={norb}, nelec={nelec!r}) sector."
                )
        if beta:
            mask_b = _occupation_axis_mask(
                norb,
                n_beta,
                sum(1 << g for g, occ in beta if occ),
                sum(1 << g for g, occ in beta if not occ),
            )
            if not np.allclose(vec_2d[:, ~mask_b], 0.0, atol=_ATOL):
                raise ValueError(
                    "InitializeModes: the state vector has amplitude outside the beta-sector "
                    f"subspace defined by its occupation for the (norb={norb}, nelec={nelec!r}) sector."
                )
        return vec
