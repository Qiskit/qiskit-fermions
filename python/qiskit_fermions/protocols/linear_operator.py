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

"""Support for handing an operator to ffsim's simulation entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit_fermions.utils.optionals import HAS_FFSIM

if TYPE_CHECKING:
    import ffsim
    import scipy.sparse.linalg

    from qiskit_fermions.operators import FermionOperator


def _ffsim_fermion_operator(
    operator: FermionOperator, norb: int, spinless: bool
) -> ffsim.FermionOperator:
    """Converts a :class:`.FermionOperator` of this package into an :class:`ffsim.FermionOperator`.

    This deliberately does **not** go through the :class:`.SupportsFermionOperator` protocol. Despite
    the shared method name, ``_fermion_operator_`` returns *this* package's ``FermionOperator``,
    whereas ffsim's simulation entry points need ffsim's own unrelated type; see the caution in
    :mod:`qiskit_fermions.protocols`.

    The two types differ in how they index a spin orbital. This package uses a flat, convention-free
    mode index and encodes spin in that index under the block-spin convention (mode ``m < norb`` is
    alpha orbital ``m``; mode ``m >= norb`` is beta orbital ``m - norb``), while ffsim carries a
    separate ``spin`` boolean alongside a *spatial* orbital index. A spinless sector maps every mode
    onto an alpha orbital directly, since ffsim represents that sector as an empty beta space.

    The conversion reads this package's flat term buffers rather than iterating term objects, which
    keeps a Hamiltonian's worth of terms out of intermediate Python lists. The order of the ladder
    operators *within* a term is preserved: reordering them flips the sign of a cross-spin term.

    Args:
        operator: the operator to convert.
        norb: the number of spatial orbitals, which fixes the alpha/beta split of the mode index.
        spinless: whether to read the modes as a single spinless (alpha-only) sector.

    Returns:
        The equivalent :class:`ffsim.FermionOperator`.
    """
    import ffsim

    coeffs = operator.get_coeffs()
    actions = operator.get_actions()
    modes = operator.get_modes()
    boundaries = operator.get_boundaries()

    # Bind the action constructors once, rather than re-resolving them per ladder operator.
    cre, des = ffsim.cre, ffsim.des

    coeff_map: dict[tuple[tuple[bool, bool, int], ...], complex] = {}
    for index, coeff in enumerate(coeffs):
        start, stop = boundaries[index], boundaries[index + 1]
        if spinless:
            term = tuple((cre if actions[i] else des)(False, modes[i]) for i in range(start, stop))
        else:
            term = tuple(
                (cre if actions[i] else des)(modes[i] >= norb, modes[i] % norb)
                for i in range(start, stop)
            )
        # Distinct terms of this package's operator can share a key, since it does not canonicalize
        # on construction. Accumulate rather than overwrite, or a repeated term would be dropped.
        coeff_map[term] = coeff_map.get(term, 0.0) + coeff

    return ffsim.FermionOperator(coeff_map)


@HAS_FFSIM.require_in_call("FermionOperator._linear_operator_")
def _linear_operator(  # noqa: D417
    self: FermionOperator, norb: int, nelec: int | tuple[int, int]
) -> scipy.sparse.linalg.LinearOperator:
    """Returns a SciPy ``LinearOperator`` for this operator on the ``(norb, nelec)`` FCI sector.

    This implements :class:`ffsim.SupportsLinearOperator` by converting to an
    :class:`ffsim.FermionOperator` and handing that to :func:`ffsim.linear_operator`: simulation is
    ffsim's concern, and this package owns the mapper-agnostic circuit rather than a second
    simulation backend.

    ffsim rejects an operator that does not conserve both particle number and the z-component of
    spin, since such an operator has no action on a fixed sector, and silently ignores a term acting
    on an orbital outside ``[0, norb)``.

    Args:
        norb: the number of spatial orbitals.
        nelec: the electron count -- an integer for a spinless sector, or an ``(n_alpha, n_beta)``
            pair for a spinful one.

    Returns:
        A :class:`scipy.sparse.linalg.LinearOperator` applying this operator on the requested sector.
    """
    import ffsim

    operator = _ffsim_fermion_operator(self, norb, isinstance(nelec, int))
    return ffsim.linear_operator(operator, norb=norb, nelec=nelec)


@HAS_FFSIM.require_in_call("FermionOperator._trace_")
def _trace(  # noqa: D417
    self: FermionOperator, norb: int, nelec: int | tuple[int, int]
) -> complex:
    """Returns the trace of this operator on the ``(norb, nelec)`` FCI sector.

    This implements :class:`ffsim.SupportsTrace` via :func:`ffsim.trace`, mirroring
    ``_linear_operator_``.

    Args:
        norb: the number of spatial orbitals.
        nelec: the electron count -- an integer for a spinless sector, or an ``(n_alpha, n_beta)``
            pair for a spinful one.

    Returns:
        The trace of this operator on the requested sector.
    """
    import ffsim

    spinless = isinstance(nelec, int)
    operator = _ffsim_fermion_operator(self, norb, spinless)
    # ffsim's own trace protocol requires a pair, even though its `linear_operator` accepts a bare
    # integer and reads it as an empty beta sector. Spell that pair out to keep both spellings of a
    # spinless sector working here.
    sector = (nelec, 0) if spinless else nelec
    return complex(ffsim.trace(operator, norb=norb, nelec=sector))
