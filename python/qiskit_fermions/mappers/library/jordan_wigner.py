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

"""A wrapper of jordan-wigner mappings."""

from __future__ import annotations

from qiskit.quantum_info import SparseObservable

from qiskit_fermions._lib.mappers.mappers_library.jordan_wigner import (
    edge_vertex_jordan_wigner,
    fermion_jordan_wigner,
    majorana_jordan_wigner,
    transfer_vertex_jordan_wigner,
)
from qiskit_fermions.operators import (
    EdgeVertexOperator,
    FermionOperator,
    MajoranaOperator,
    OperatorTrait,
    TransferVertexOperator,
)


def jordan_wigner(operator: OperatorTrait, num_qubits: int) -> SparseObservable:
    """Map an operator to a ``SparseObservable`` under the Jordan-Wigner transformation.

    This is the type-agnostic entry point to the Jordan-Wigner transformation. It dispatches on the
    concrete type of ``operator`` to the appropriate direct implementation:
    :func:`.fermion_jordan_wigner`, :func:`.majorana_jordan_wigner`,
    :func:`.edge_vertex_jordan_wigner` or :func:`.transfer_vertex_jordan_wigner`. Anything that is
    not one of the four operator types raises a :class:`TypeError`.

    Args:
        operator: the operator to map.
        num_qubits: the number of qubits for the resulting qubit operator. This must be strictly
            greater than the largest mode index in ``operator`` (any additional qubits are padded
            with the identity). Note that for a :class:`.MajoranaOperator` the bound is counted in
            *fermionic* modes, i.e. in the largest Majorana index divided by two.

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

    Every operator type has a direct implementation, so the dispatch is total over them. Anything
    else -- an object that is not a fermionic operator at all -- raises a :class:`TypeError`:

    .. doctest::

        >>> from qiskit.quantum_info import SparseObservable
        >>> jordan_wigner(SparseObservable.identity(2), 2)
        Traceback (most recent call last):
        ...
        TypeError: jordan_wigner does not support operator type 'SparseObservable'.
    """
    match operator:
        case FermionOperator():
            return fermion_jordan_wigner(operator, num_qubits)
        case MajoranaOperator():
            return majorana_jordan_wigner(operator, num_qubits)
        case EdgeVertexOperator():
            return edge_vertex_jordan_wigner(operator, num_qubits)
        case TransferVertexOperator():
            return transfer_vertex_jordan_wigner(operator, num_qubits)
        case _:
            # Kept as a guard even though the four arms above cover every operator type: it is what
            # turns a non-operator argument into a clear error instead of an obscure failure deeper
            # in, and it is what a future operator type will land on until it gains an implementation.
            raise TypeError(
                f"jordan_wigner does not support operator type {type(operator).__name__!r}."
            )
