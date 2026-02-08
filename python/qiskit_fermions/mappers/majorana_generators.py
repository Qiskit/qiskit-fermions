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

"""MajoranaOperator mapper."""

from collections.abc import Callable
from operator import and_
from typing import TypeVar

from ..operators import MajoranaAction, MajoranaOperator

T = TypeVar("T")


def map_majorana_action_generators(
    operator: MajoranaOperator,
    map_action: Callable[[MajoranaAction], T],
    identity: Callable[[], T],
    compose: Callable[[T, T], T] | None = None,
) -> T:
    """Map a :class:`.MajoranaOperator` to another operator type.

    This is a generic function to aid in implementing new mappers for :class:`.MajoranaOperator`
    instances. At its core, it simply iterates over the terms of the operator, mapping each
    encountered :class:`.MajoranaAction` with the user-provided ``map_action`` function. In
    combination with the user-provided ``identity`` generator, this allows mapping to arbitrary
    output types.

    .. note::
       The output type ``T`` must support multiplication by a scalar via ``__mul__``.
       If ``compose=None`` it must also support composition of two instances via ``__and__``.

    .. doctest::
        >>> from qiskit_fermions.mappers import map_majorana_action_generators
        >>> from qiskit_fermions.operators import MajoranaAction, MajoranaOperator, gamma
        >>> from qiskit.quantum_info import SparsePauliOp
        >>>
        >>> def jordan_wigner(mode: MajoranaAction) -> SparsePauliOp:
        ...     idx = mode // 2
        ...     qubits = list(range(idx + 1))
        ...     pauli = "Y" if mode % 2 else "X"
        ...     return SparsePauliOp.from_sparse_list(
        ...         [("Z" * idx + pauli, qubits, 1.0)],
        ...         num_qubits=num_qubits,
        ...     )
        >>>
        >>> num_qubits = 2
        >>> def identity() -> SparsePauliOp:
        ...     return SparsePauliOp.from_sparse_list([("", [], 1)], num_qubits)
        >>>
        >>> op = MajoranaOperator.from_dict({
        ...     (0, 2): 0.5,
        ...     (1, 3): 0.5,
        ...     (0, 3): 0.5j,
        ...     (1, 2): -0.5j,
        ... })
        >>> qop = map_majorana_action_generators(op, jordan_wigner, identity)
        >>> print([(label, complex(coeff)) for label, coeff in sorted(qop.label_iter())])
        [('II', 0j), ('XX', (0.5-0j)), ('XY', -0.5j), ('YX', 0.5j), ('YY', (0.5+0j))]

    Args:
        operator: the operator to be mapped.
        map_action: the function to map a single :class:`.MajoranaAction` to the desired output type.
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


# TODO: map_majorana_even_generators
#
# The idea of this function is to provide a quick means to implement a mapping for even Majorana
# operators (i.e. `op.is_even()` returns `True`). In those cases, every term is known to have an
# even number of actions inside of it, allowing us to process them in pairs. Every pair of actions
# will thus correspond to a `tuple[int, int] = (i, j)` for which 4 cases are possible:
#   1. both `i` and `j` are odd
#   2. `i` is odd, `j` is even
#   3. `i` is even, `j` is odd
#   4. both `i` and `j` are even
#
# All these cases can be broken down into a mapping using only 2 user-provided functions:
#   A. `map_pair`: which implements a mapping for `(gamma(i, True), gamma(j, True)` (i.e. case 1)
#   B. `map_single`: which implements a mapping for `(gamma(i, True), gamma(i, False)`
#
# The cases above then simply become compositions of the results of A and B.
#
# However, I dislike the arbitrary choice of picking case 1 over 4 for `map_pair` and would much
# rather have the user make this choice.
# Therefore, I would even propose to outsource these case distinctions entirely to the user-provided
# function.
# All that this function would then do is to iterate over pairs rather than individual actions.
# This also avoids hard-coding an assignment assumption about the nature of gamma/gamma' mapping to
# even/odd indices, respectively.
