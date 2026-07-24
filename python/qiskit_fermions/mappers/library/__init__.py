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

# ruff: noqa: D205,D212,D415
"""
==============
Mapper Library
==============

.. currentmodule:: qiskit_fermions.mappers.library

This module provides efficient implementations of commonly used operator representation mapper
routines.

.. autosummary::
   :toctree: ../stubs/

   jordan_wigner
   fermion_jordan_wigner
   fermion_to_majorana
   majorana_to_fermion
   edge_vertex_to_fermion
   edge_vertex_to_majorana
   transfer_vertex_to_fermion
   transfer_vertex_to_majorana
   transfer_vertex_to_edge_vertex
"""

from __future__ import annotations

from qiskit.quantum_info import SparseObservable

from qiskit_fermions._lib.mappers.mappers_library.edge_vertex import (
    edge_vertex_to_fermion,
    edge_vertex_to_majorana,
)
from qiskit_fermions._lib.mappers.mappers_library.jordan_wigner import (
    fermion_jordan_wigner,
)
from qiskit_fermions._lib.mappers.mappers_library.majorana_fermion import (
    fermion_to_majorana,
    majorana_to_fermion,
)
from qiskit_fermions._lib.mappers.mappers_library.transfer_vertex import (
    transfer_vertex_to_edge_vertex,
    transfer_vertex_to_fermion,
    transfer_vertex_to_majorana,
)
from qiskit_fermions.operators import FermionOperator
from qiskit_fermions.operators.protocol import OperatorTrait


def jordan_wigner(operator: OperatorTrait, num_qubits: int) -> SparseObservable:
    """Map an operator to a :class:`~qiskit.quantum_info.SparseObservable` under the Jordan-Wigner
    transformation.

    This is the type-agnostic entry point to the Jordan-Wigner transformation. It dispatches on the
    concrete type of ``operator`` to the appropriate direct implementation. Today only
    :class:`.FermionOperator` is supported, delegating to :func:`.fermion_jordan_wigner`; any other
    operator type raises a :class:`TypeError` until a direct implementation exists for it.

    Args:
        operator: the operator to map.
        num_qubits: the number of qubits for the resulting qubit operator. This must be strictly
            greater than the largest mode index in ``operator`` (any additional qubits are padded
            with the identity).

    Returns:
        The mapped qubit operator.

    Raises:
        TypeError: if ``operator`` is of a type for which no Jordan-Wigner implementation exists.
        ValueError: if ``num_qubits`` is too small to hold the operator's support.

    .. doctest::

        >>> from qiskit_fermions.mappers.library import jordan_wigner
        >>> from qiskit_fermions.operators import FermionOperator
        >>> fop = FermionOperator.from_dict({((True, 0), (False, 0)): 0.1})
        >>> jordan_wigner(fop, 2).simplify()
        <SparseObservable with 2 terms on 2 qubits: (0.05+0j)() + (-0.05+0j)(Z_0)>

    An operator type without a direct implementation raises a :class:`TypeError`:

    .. doctest::

        >>> from qiskit_fermions.operators import MajoranaOperator
        >>> mop = MajoranaOperator.from_dict({(0, 1): 1.0})
        >>> jordan_wigner(mop, 2)
        Traceback (most recent call last):
        ...
        TypeError: jordan_wigner does not support operator type 'MajoranaOperator'; only FermionOperator is supported today.
    """
    match operator:
        case FermionOperator():
            return fermion_jordan_wigner(operator, num_qubits)
        case _:
            raise TypeError(
                f"jordan_wigner does not support operator type "
                f"{type(operator).__name__!r}; only FermionOperator is supported today."
            )


__all__ = [
    "edge_vertex_to_fermion",
    "edge_vertex_to_majorana",
    "fermion_jordan_wigner",
    "fermion_to_majorana",
    "jordan_wigner",
    "majorana_to_fermion",
    "transfer_vertex_to_edge_vertex",
    "transfer_vertex_to_fermion",
    "transfer_vertex_to_majorana",
]
