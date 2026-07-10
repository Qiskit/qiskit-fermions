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

"""Glue code for using `ffsim <https://github.com/qiskit-community/ffsim>`__ as a simulation engine.

The functions in this module bridge :class:`~qiskit_fermions.operators.FermionOperator` (a *spinless*
operator whose modes are indexed ``0 .. 2 * norb - 1``) to ffsim's ``FermionOperator`` (a *spinful*
operator with a separate ``(orbital, spin)`` index). The mapping follows the spin convention used by
the ``FermionOperator.from_1body/2body_*_spin*`` constructors: alpha (spin-up) orbital ``i`` is mode
``i``, and beta (spin-down) orbital ``i`` is mode ``i + norb``. Hence mode ``m`` maps to ffsim
``(orb=m % norb, spin=m // norb)`` with spin ``0`` for alpha and ``1`` for beta.

``ffsim`` (and its transitive ``scipy``/``pyscf`` dependencies) is an optional dependency, so all
imports here are performed lazily inside the functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qiskit_fermions.utils.optionals import HAS_FFSIM

if TYPE_CHECKING:
    import ffsim

    from qiskit_fermions._lib.operators.fermion_operator import FermionOperator


def to_ffsim_operator(operator: FermionOperator, norb: int) -> ffsim.FermionOperator:
    """Converts a :class:`~qiskit_fermions.operators.FermionOperator` to an ``ffsim.FermionOperator``.

    Args:
        operator: the spinless fermionic operator to convert.
        norb: the number of spatial orbitals. The operator's modes are interpreted as spin-orbitals
            under the convention that mode ``m`` corresponds to ffsim ``(orb=m % norb, spin=m // norb)``
            (spin ``0`` is alpha, spin ``1`` is beta).

    Returns:
        The equivalent ``ffsim.FermionOperator``.

    Raises:
        MissingOptionalLibraryError: if ``ffsim`` is not installed.
        ValueError: if the operator acts on a mode outside the ``[0, 2 * norb)`` range implied by
            ``norb``.
    """
    HAS_FFSIM.require_now("converting a FermionOperator to an ffsim.FermionOperator")
    import ffsim

    # (action, spin) -> ffsim ladder-operator constructor
    actions = {
        (True, 0): ffsim.cre_a,
        (True, 1): ffsim.cre_b,
        (False, 0): ffsim.des_a,
        (False, 1): ffsim.des_b,
    }

    data: dict[tuple, complex] = {}
    for term, coeff in operator.iter_terms():
        ffsim_term = []
        for action, mode in term:
            if not 0 <= mode < 2 * norb:
                raise ValueError(
                    f"Mode {mode} is outside the range [0, {2 * norb}) implied by norb={norb}. "
                    "Ensure the operator's modes match the number of spatial orbitals."
                )
            spin = mode // norb
            orb = mode % norb
            ffsim_term.append(actions[(action, spin)](orb))
        data[tuple(ffsim_term)] = coeff

    return ffsim.FermionOperator(data)


def apply_fermion_operator_evolution(
    operator: FermionOperator,
    time: float,
    vec: np.ndarray,
    norb: int,
    nelec: int | tuple[int, int],
    copy: bool,
) -> np.ndarray:
    """Applies ``exp(-i * time * operator)`` to an ffsim state vector.

    This mirrors ffsim's own :meth:`_apply_unitary_` implementations (e.g. for its UCCSD operators):
    the operator is converted to an ``ffsim.FermionOperator``, turned into a ``scipy`` ``LinearOperator``
    via ``ffsim.linear_operator``, and applied to the vector via ``scipy.sparse.linalg.expm_multiply``.

    Args:
        operator: the (spinless) fermionic operator to time evolve under.
        time: the evolution time.
        vec: the state vector to act on.
        norb: the number of spatial orbitals.
        nelec: either a single integer for a spinless system, or a pair of integers storing the
            numbers of spin alpha and spin beta fermions.
        copy: whether to copy the vector before operating on it.

    Returns:
        The transformed vector, or ``NotImplemented`` for a spinless (integer ``nelec``) system.

    Raises:
        MissingOptionalLibraryError: if ``ffsim`` is not installed.
        ValueError: if the operator does not conserve particle number and the z-component of spin
            (raised by ``ffsim.linear_operator``).
    """
    if isinstance(nelec, int):
        # ffsim's FermionOperator LinearOperator requires a spinful (n_alpha, n_beta) sector.
        return NotImplemented

    HAS_FFSIM.require_now("applying a FermionOperator evolution to a state vector")
    import ffsim
    import scipy.sparse.linalg

    if copy:
        vec = vec.copy()

    linop = ffsim.linear_operator(to_ffsim_operator(operator, norb), norb=norb, nelec=nelec)
    return scipy.sparse.linalg.expm_multiply(-1j * time * linop, vec, traceA=0.0)
