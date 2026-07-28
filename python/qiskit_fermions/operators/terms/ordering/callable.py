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

"""Ordering the terms of an operator by a user-provided key."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qiskit_fermions.operators import OperatorTrait


def order_terms(
    op: OperatorTrait,
    key: Callable[[tuple], Any],
    *,
    reverse: bool = False,
) -> OperatorTrait:
    """Returns a copy of an operator with its terms reordered by a user-provided key.

    Unlike :func:`.canonical_order`, which imposes a single fixed, coefficient-independent order,
    this function lets you sort the terms by any criterion you like: the ``key`` function is applied
    to each term exactly as :func:`sorted` would, and the terms are rebuilt into a new operator of
    the same type in the resulting order. This is useful whenever the *order* of the terms matters
    for a downstream computation (e.g. randomized product-formula sampling), since it gives you a
    deterministic, reproducible order independent of how the operator happened to be assembled.

    The terms themselves are left untouched -- this only reorders them, it does not simplify or
    normal-order the operator. This works for any of the built-in operator types implementing the
    :class:`~qiskit_fermions.operators.OperatorTrait` protocol; the returned operator is of the same
    type as the input.

    .. note::
       The ``key`` receives each term as the same tuple that the operator's
       :meth:`~qiskit_fermions.operators.OperatorTrait.iter_terms` yields -- or, when the operator
       tracks groups (see :attr:`~qiskit_fermions.operators.OperatorTrait.groups`), the longer tuple
       from :meth:`~qiskit_fermions.operators.OperatorTrait.iter_terms_with_groups`, whose trailing
       element is the term's group index. The exact layout of the term itself depends on the
       operator type. Group indices are preserved -- each term carries its group index along as it
       moves -- and if the input tracks no groups, neither does the result.

    .. doctest::

        >>> from qiskit_fermions.operators import FermionOperator
        >>> from qiskit_fermions.operators.terms.ordering import order_terms
        >>> op = FermionOperator.from_dict(
        ...     {
        ...         ((True, 1), (False, 0)): 1.0,  # a†_1 a_0, |coeff| = 1
        ...         ((True, 0), (False, 1)): 3.0,  # a†_0 a_1, |coeff| = 3
        ...     }
        ... )
        >>> ordered = order_terms(op, key=lambda term: abs(term[1]), reverse=True)
        >>> list(ordered.iter_terms())
        [([(True, 0), (False, 1)], (3+0j)), ([(True, 1), (False, 0)], (1+0j))]

    Args:
        op: the operator whose terms to reorder.
        key: a function of a single term, used exactly as the ``key`` argument of :func:`sorted`. It
            receives each term as the tuple described in the note above.
        reverse: if ``True``, the terms are sorted in descending order of ``key``.

    Returns:
        A new operator of the same type with its terms in the requested order.
    """
    if op.groups is None:
        terms = sorted(op.iter_terms(), key=key, reverse=reverse)
        return op.__class__.from_terms(terms)

    terms = sorted(op.iter_terms_with_groups(), key=key, reverse=reverse)
    return op.__class__.from_terms_with_groups(terms)
