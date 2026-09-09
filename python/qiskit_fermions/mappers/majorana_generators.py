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

from qiskit_fermions.mappers._typing import T
from qiskit_fermions.operators import MajoranaOperator
from qiskit_fermions.operators.majorana_action import MajoranaAction


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
       :class:`~qiskit.quantum_info.SparseObservable` does not, which is why the example below
       names :meth:`~qiskit.quantum_info.SparseObservable.compose` explicitly; a type with an
       ``__and__`` (such as :class:`~qiskit.quantum_info.SparsePauliOp`) can rely on the default.

    .. note::
       The mapping written out below is a minimal illustration of this function rather than a
       replacement for :func:`.majorana_jordan_wigner`, which is parallelized and bounds the memory
       it uses while assembling the result.

    .. doctest::

        >>> from qiskit_fermions.mappers import map_majorana_action_generators
        >>> from qiskit_fermions.operators import MajoranaAction, MajoranaOperator, gamma
        >>> from qiskit.quantum_info import SparseObservable
        >>>
        >>> def jordan_wigner(mode: MajoranaAction) -> SparseObservable:
        ...     idx = mode // 2
        ...     qubits = list(range(idx + 1))
        ...     pauli = "Y" if mode % 2 else "X"
        ...     return SparseObservable.from_sparse_list(
        ...         [("Z" * idx + pauli, qubits, 1.0)],
        ...         num_qubits=num_qubits,
        ...     )
        >>>
        >>> num_qubits = 2
        >>> def identity() -> SparseObservable:
        ...     return SparseObservable.identity(num_qubits)
        >>>
        >>> op = MajoranaOperator.from_dict({
        ...     (0, 2): 0.5,
        ...     (1, 3): 0.5,
        ...     (0, 3): 0.5j,
        ...     (1, 2): -0.5j,
        ... })
        >>> qop = map_majorana_action_generators(
        ...     op, jordan_wigner, identity, compose=SparseObservable.compose
        ... )
        >>> print(sorted(qop.simplify().to_sparse_list()))
        [('XX', [0, 1], (0.5-0j)), ('XY', [0, 1], 0.5j), ('YX', [0, 1], -0.5j), ('YY', [0, 1], (0.5+0j))]

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

    mapped_operator: T = 0 * identity()
    for terms, coeff in operator.iter_terms():
        mapped_terms = identity()

        for term in terms:
            mapped_terms = compose(map_action(term), mapped_terms)

        mapped_operator += coeff * mapped_terms

    return mapped_operator
