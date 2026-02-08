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

"""FermionOperator mapper."""

from collections.abc import Callable
from operator import and_
from typing import TypeVar

from ..operators import FermionAction, FermionOperator

T = TypeVar("T")


def map_fermion_action_generators(
    operator: FermionOperator,
    map_action: Callable[[FermionAction], T],
    identity: Callable[[], T],
    compose: Callable[[T, T], T] | None = None,
) -> T:
    """Map a :class:`.FermionOperator` to another operator type.

    This is a generic function to aid in implementing new mappers for :class:`.FermionOperator`
    instances. At its core, it simply iterates over the terms of the operator, mapping each
    encountered :class:`.FermionAction` with the user-provided ``map_action`` function. In
    combination with the user-provided ``identity`` generator, this allows mapping to arbitrary
    output types.

    .. note::
       The output type ``T`` must support multiplication by a scalar via ``__mul__``.
       If ``compose=None`` it must also support composition of two instances via ``__and__``.

    .. doctest::
        >>> from qiskit_fermions.mappers import map_fermion_action_generators
        >>> from qiskit_fermions.operators import FermionAction, FermionOperator, ann, cre
        >>> from qiskit.quantum_info import SparsePauliOp
        >>>
        >>> def jordan_wigner(action: FermionAction) -> SparsePauliOp:
        ...     act, idx = action
        ...     qubits = list(range(idx + 1))
        ...     return SparsePauliOp.from_sparse_list(
        ...         [
        ...             ("Z" * idx + "X", qubits, 0.5),
        ...             ("Z" * idx + "Y", qubits, -0.5j if act else 0.5j),
        ...         ],
        ...         num_qubits=num_qubits,
        ...     )
        >>>
        >>> num_qubits = 2
        >>> def identity() -> SparsePauliOp:
        ...     return SparsePauliOp.from_sparse_list([("", [], 1)], num_qubits)
        >>>
        >>> op = FermionOperator.from_dict({(cre(0), ann(1)): 2.0})
        >>> qop = map_fermion_action_generators(op, jordan_wigner, identity)
        >>> print([(label, complex(coeff)) for label, coeff in sorted(qop.label_iter())])
        [('II', 0j), ('XX', (0.5+0j)), ('XY', -0.5j), ('YX', 0.5j), ('YY', (0.5+0j))]

    Args:
        operator: the operator to be mapped.
        map_action: the function to map a single :class:`.FermionAction` to the desired output type.
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
            mapped_terms = compose(map_action(term), mapped_terms)

        mapped_operator += coeff * mapped_terms

    return mapped_operator
