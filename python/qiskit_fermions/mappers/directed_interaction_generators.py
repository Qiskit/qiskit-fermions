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

"""DirectedInteractionOperator mapper."""

from collections.abc import Callable
from operator import and_
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from qiskit_fermions._lib.operators.directed_interaction_operator import (
        DirectedInteractionOperator,
    )
    from qiskit_fermions.operators.directed_interaction import DirectedInteraction

T = TypeVar("T")


def map_directed_interaction_generators(
    operator: "DirectedInteractionOperator",
    map_interaction: Callable[["DirectedInteraction"], T],
    identity: Callable[[], T],
    compose: Callable[[T, T], T] | None = None,
) -> T:
    """Map a :class:`.DirectedInteractionOperator` to another operator type.

    This is a generic function to aid in implementing new mappers for
    :class:`.DirectedInteractionOperator` instances. At its core, it simply iterates over the
    terms of the operator, mapping each encountered :class:`.DirectedInteraction` with the
    user-provided ``map_interaction`` function. In combination with the user-provided ``identity``
    generator, this allows mapping to arbitrary output types.

    .. note::
       The output type ``T`` must support multiplication by a scalar via ``__mul__``.
       If ``compose=None`` it must also support composition of two instances via ``__and__``.

    .. doctest::

        >>> from qiskit_fermions.mappers import map_directed_interaction_generators
        >>> from qiskit_fermions.operators import DirectedInteraction, DirectedInteractionOperator
        >>> from qiskit.quantum_info import SparsePauliOp
        >>>
        >>> def jordan_wigner_nearest_neighbor(mode: DirectedInteraction) -> SparsePauliOp:
        ...     match abs(mode[0] - mode[1]):
        ...         case 0:
        ...             pauli = "Z"
        ...             qubits = [mode[0]]
        ...             coeff = 1.0
        ...         case 1:
        ...             pauli = "YY"
        ...             qubits = [mode[0], mode[1]]
        ...             coeff = -0.5
        ...         case _:
        ...             raise NotImplementedError(
        ...                 "This mapping only handles nearest neighbor interactions"
        ...             )
        ...
        ...     return SparsePauliOp.from_sparse_list([(pauli, qubits, coeff)], num_qubits=num_qubits)
        >>>
        >>> num_qubits = 4
        >>> def identity() -> SparsePauliOp:
        ...     return SparsePauliOp.from_sparse_list([("", [], 1)], num_qubits)
        >>>
        >>> op = DirectedInteractionOperator.from_dict({
        ...     ((0, 0),): 2.0,
        ...     ((0, 1),): 0.5,
        ...     ((1, 1), (1, 2)): 1.0,
        ... })
        >>> qop = map_directed_interaction_generators(op, jordan_wigner_nearest_neighbor, identity)
        >>> print([(label, complex(coeff)) for label, coeff in sorted(qop.label_iter())])
        [('IIII', 0j), ('IIIZ', (2+0j)), ('IIYY', (-0.25+0j)), ('IYXI', 0.5j)]

    Args:
        operator: the operator to be mapped.
        map_interaction: the function to map a single :class:`.DirectedInteraction` to the desired
            output type.
        identity: the function to generate the multiplicative identity instance of the output type.
        compose: an optional function to implement the compositiion logic of two output type
            instances. If this is not provided, it will default to using :py:func:`operator.and_`.

    Returns:
        The mapped operator.
    """
    if compose is None:
        compose = and_

    mapped_operator: T = 0 * identity()  # type: ignore[assignment,operator]
    for terms, coeff in operator.iter_terms():
        mapped_terms = identity()

        for term in terms:
            mapped_terms = compose(map_interaction(term), mapped_terms)

        mapped_operator += coeff * mapped_terms

    return mapped_operator
