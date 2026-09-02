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

"""TransferVertexOperator mapper."""

from collections.abc import Callable
from operator import and_

from qiskit_fermions.mappers._typing import T
from qiskit_fermions.operators import TransferVertexOperator
from qiskit_fermions.operators.transfer_action import TransferAction


def map_transfer_vertex_generators(
    operator: TransferVertexOperator,
    map_action: Callable[[TransferAction], T],
    identity: Callable[[], T],
    compose: Callable[[T, T], T] | None = None,
) -> T:
    """Map a :class:`.TransferVertexOperator` to another operator type.

    This is a generic function to aid in implementing new mappers for
    :class:`.TransferVertexOperator` instances. At its core, it simply iterates over the
    terms of the operator, mapping each encountered :class:`.TransferAction` with the
    user-provided ``map_action`` function. In combination with the user-provided ``identity``
    generator, this allows mapping to arbitrary output types.

    .. note::
       The output type ``T`` must support multiplication by a scalar via ``__mul__``.
       If ``compose=None`` it must also support composition of two instances via ``__and__``.

    .. note::
       The mapping written out below is a deliberately minimal illustration of this function, not a
       replacement for :func:`.transfer_vertex_jordan_wigner`: it covers only nearest-neighbor
       interactions, and it is neither parallelized nor memory-bounded. Reach for the library
       function rather than copying this one.

    .. doctest::

        >>> from qiskit_fermions.mappers import map_transfer_vertex_generators
        >>> from qiskit_fermions.operators import TransferAction, TransferVertexOperator
        >>> from qiskit.quantum_info import SparsePauliOp
        >>>
        >>> def jordan_wigner_nearest_neighbor(mode: TransferAction) -> SparsePauliOp:
        ...     left, right = mode
        ...     if left == right:
        ...         return SparsePauliOp.from_sparse_list(
        ...             [("Z", [left], 1.0)], num_qubits=num_qubits
        ...         )
        ...     if abs(left - right) != 1:
        ...         raise NotImplementedError(
        ...             "This mapping only handles nearest neighbor interactions"
        ...         )
        ...     # `T_lr` and `T_rl` differ, so the orientation must be compared rather than
        ...     # differenced. The coefficient is -1/2 either way; the Pauli letters swap instead.
        ...     lo, hi = min(left, right), max(left, right)
        ...     pauli = "XX" if left < right else "YY"
        ...     return SparsePauliOp.from_sparse_list(
        ...         [(pauli, [lo, hi], -0.5)], num_qubits=num_qubits
        ...     )
        >>>
        >>> num_qubits = 4
        >>> def identity() -> SparsePauliOp:
        ...     return SparsePauliOp.from_sparse_list([("", [], 1)], num_qubits)
        >>>
        >>> op = TransferVertexOperator.from_dict({
        ...     ((0, 0),): 2.0,
        ...     ((0, 1),): 0.5,
        ...     ((1, 1), (1, 2)): 1.0,
        ... })
        >>> qop = map_transfer_vertex_generators(op, jordan_wigner_nearest_neighbor, identity)
        >>> print([(label, complex(coeff)) for label, coeff in sorted(qop.label_iter())])
        [('IIII', 0j), ('IIIZ', (2+0j)), ('IIXX', (-0.25+0j)), ('IXYI', -0.5j)]

    Args:
        operator: the operator to be mapped.
        map_action: the function to map a single :class:`.TransferAction` to the desired
            output type.
        identity: the function to generate the multiplicative identity instance of the output type.
        compose: an optional function to implement the compositiion logic of two output type
            instances. If this is not provided, it will default to using :py:func:`operator.and_`.

    Returns:
        The mapped operator.
    """
    if compose is None:
        compose = and_

    mapped_operator: T = 0 * identity()
    for terms, coeff in operator.iter_terms():
        mapped_terms = identity()

        for term in terms:
            mapped_terms = compose(map_action(term), mapped_terms)

        mapped_operator += coeff * mapped_terms

    return mapped_operator
