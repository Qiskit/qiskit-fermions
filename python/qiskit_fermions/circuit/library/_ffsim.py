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

The :func:`to_ffsim_operator` function bridges :class:`~qiskit_fermions.operators.FermionOperator`
(whose modes carry no intrinsic spin label) to ffsim's ``FermionOperator`` (which indexes ladder
operators by a separate ``(orbital, spin)`` pair). Two mode interpretations are supported, selected by
the type of ``nelec``:

* **Spinful** (``nelec`` is an ``(n_alpha, n_beta)`` pair): the operator's ``2 * norb`` modes are
  spin-orbitals following the convention of the ``FermionOperator.from_1body/2body_*_spin*``
  constructors -- alpha (spin-up) orbital ``i`` is mode ``i`` and beta (spin-down) orbital ``i`` is
  mode ``i + norb``. Hence mode ``m`` maps to ffsim ``(orb=m % norb, spin=m // norb)`` with spin ``0``
  for alpha and ``1`` for beta.
* **Spinless** (``nelec`` is a single integer): the operator's ``norb`` modes map directly onto ffsim's
  alpha orbitals, i.e. mode ``m`` becomes ``(orb=m, spin=alpha)``. ffsim represents a spinless system
  as the ``(nelec, 0)`` spinful sector, so keeping every mode on alpha confines the operator to that
  sector (the FCI space has dimension ``C(norb, nelec)``).

``ffsim`` (and its transitive ``scipy``/``pyscf`` dependencies) is an optional dependency, so all
imports here are performed lazily inside the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit_fermions.utils.optionals import HAS_FFSIM

if TYPE_CHECKING:
    import ffsim

    from qiskit_fermions._lib.operators.fermion_operator import FermionOperator


@HAS_FFSIM.require_in_call
def to_ffsim_operator(
    operator: FermionOperator, norb: int, nelec: int | tuple[int, int]
) -> ffsim.FermionOperator:
    """Converts a :class:`~qiskit_fermions.operators.FermionOperator` to an ``ffsim.FermionOperator``.

    The mode interpretation is selected by the type of ``nelec``, mirroring ffsim's own
    ``norb, nelec`` convention:

    * **Spinful** (``nelec`` is an ``(n_alpha, n_beta)`` pair): the operator's ``2 * norb`` modes are
      spin-orbitals under the convention that mode ``m`` corresponds to ffsim ``(orb=m % norb,
      spin=m // norb)`` (spin ``0`` is alpha, spin ``1`` is beta). This matches the
      ``from_1body/2body_*_spin*`` constructors.
    * **Spinless** (``nelec`` is a single integer): the operator's ``norb`` modes are mapped directly
      onto ffsim's alpha (spin-up) orbitals, i.e. mode ``m`` becomes ``(orb=m, spin=alpha)``. ffsim
      represents a spinless system as the ``(nelec, 0)`` spinful sector, so placing every mode on alpha
      keeps the operator entirely within that sector.

    Args:
        operator: the fermionic operator to convert.
        norb: the number of spatial orbitals.
        nelec: either a single integer for a spinless system, or a pair of integers storing the numbers
            of spin alpha and spin beta fermions. Only its type is inspected here, to choose the mode
            interpretation (see above).

    Returns:
        The equivalent ``ffsim.FermionOperator``.

    Raises:
        MissingOptionalLibraryError: if ``ffsim`` is not installed.
        ValueError: if the operator acts on a mode outside the range implied by ``norb`` (``[0, norb)``
            for a spinless system, else ``[0, 2 * norb)``).
    """
    import ffsim

    # (action, spin) -> ffsim ladder-operator constructor
    actions = {
        (True, 0): ffsim.cre_a,
        (True, 1): ffsim.cre_b,
        (False, 0): ffsim.des_a,
        (False, 1): ffsim.des_b,
    }

    spinless = isinstance(nelec, int)
    num_modes = norb if spinless else 2 * norb

    data: dict[tuple, complex] = {}
    for term, coeff in operator.iter_terms():
        ffsim_term = []
        for action, mode in term:
            if not 0 <= mode < num_modes:
                raise ValueError(
                    f"Mode {mode} is outside the range [0, {num_modes}) implied by norb={norb}. "
                    "Ensure the operator's modes match the number of spatial orbitals."
                )
            # spinless: every mode is an alpha orbital; spinful: split into (orb, spin) blocks.
            spin = 0 if spinless else mode // norb
            orb = mode if spinless else mode % norb
            ffsim_term.append(actions[(action, spin)](orb))
        data[tuple(ffsim_term)] = coeff

    return ffsim.FermionOperator(data)
