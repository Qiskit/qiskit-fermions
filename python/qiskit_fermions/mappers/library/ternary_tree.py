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

"""A wrapper of ternary-tree mappings."""

from __future__ import annotations

from collections.abc import Callable

from qiskit.quantum_info import SparseObservable

from qiskit_fermions._lib.mappers.mappers_library.ternary_tree import (
    edge_vertex_ternary_tree,
    fermion_ternary_tree,
    majorana_ternary_tree,
    transfer_vertex_ternary_tree,
)
from qiskit_fermions._lib.mappers.ternary_tree_structures import TernaryTreeEncoding
from qiskit_fermions.operators import (
    EdgeVertexOperator,
    FermionOperator,
    MajoranaOperator,
    OperatorTrait,
    TransferVertexOperator,
)

from ._dispatch import dispatch_mapper

_IMPLEMENTATIONS: dict[type, Callable[..., SparseObservable]] = {
    FermionOperator: fermion_ternary_tree,
    MajoranaOperator: majorana_ternary_tree,
    EdgeVertexOperator: edge_vertex_ternary_tree,
    TransferVertexOperator: transfer_vertex_ternary_tree,
}


def ternary_tree(
    operator: OperatorTrait, encoding: TernaryTreeEncoding, num_qubits: int
) -> SparseObservable:
    """Map an operator to a ``SparseObservable`` under a ternary-tree encoding.

    This is the type-agnostic entry point to the ternary-tree mappers, mirroring
    :func:`.jordan_wigner`. It dispatches on the concrete type of ``operator`` to the matching direct
    implementation, listed under :ref:`ternary_tree_mappers` in the
    :mod:`~qiskit_fermions.mappers.library` documentation. An operator type without one raises a
    :class:`TypeError`.

    Args:
        operator: the operator to map.
        encoding: the compiled encoding to map through. Compile it once with
            :class:`~qiskit_fermions.mappers.TernaryTreeEncoding` and reuse it across every operator
            mapped through the same tree.
        num_qubits: the number of qubits for the resulting qubit operator. Must be at least the
            encoding's mode count; a larger value pads with the identity.

    Returns:
        The mapped qubit operator, subject to the same simplification and grouping caveats as
        :func:`.fermion_ternary_tree`.

    Raises:
        TypeError: if ``operator`` is of a type for which no ternary-tree implementation exists.
        ValueError: if ``operator`` acts on a mode the encoding does not cover, or ``num_qubits`` is
            smaller than the encoding's mode count.
        OverflowError: if ``num_qubits`` is negative or does not fit in 32 bits.

    .. doctest::

        >>> from qiskit_fermions.mappers import TernaryTree, TernaryTreeEncoding
        >>> from qiskit_fermions.mappers.library import ternary_tree
        >>> from qiskit_fermions.operators import FermionOperator
        >>> fop = FermionOperator.from_dict({((True, 0), (False, 0)): 0.1})
        >>> enc = TernaryTreeEncoding(TernaryTree.breadth_first(2, 3))
        >>> ternary_tree(fop, enc, 2).simplify()
        <SparseObservable with 2 terms on 2 qubits: (0.05+0j)() + (-0.05+0j)(Z_1 Z_0)>

    A chain along ``Z`` reproduces :func:`.jordan_wigner`, which is what makes the tree a
    generalisation of it rather than an alternative to it:

    .. doctest::

        >>> from qiskit_fermions.mappers.library import jordan_wigner
        >>> chain = TernaryTreeEncoding(TernaryTree.chain(2, "Z"))
        >>> bool((ternary_tree(fop, chain, 2) - jordan_wigner(fop, 2)).simplify().num_terms == 0)
        True

    Every operator type has a direct implementation, so the dispatch is total over them. Anything
    else (an object that is not a fermionic operator at all) raises a :class:`TypeError`:

    .. doctest::

        >>> from qiskit.quantum_info import SparseObservable
        >>> ternary_tree(SparseObservable.identity(2), enc, 2)
        Traceback (most recent call last):
        ...
        TypeError: ternary_tree does not support operator type 'SparseObservable'.
    """
    return dispatch_mapper("ternary_tree", _IMPLEMENTATIONS, operator, encoding, num_qubits)
