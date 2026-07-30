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

"""A protocol for operator types."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator, Sequence
from typing import Protocol

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class OperatorTrait(Protocol):
    """A protocol indicating all methods implemented by operator classes."""

    @classmethod
    def zero(cls) -> Self:
        """Constructs the additive identity operator."""

    @classmethod
    def one(cls) -> Self:
        """Constructs the multiplicative identity operator."""

    def __iadd__(self, other: Self):
        """Adds another operator to this one."""

    def __add__(self, other: Self) -> Self:
        """Adds two operators."""

    def __isub__(self, other: Self):
        """Subtracts another operator from this one."""

    def __sub__(self, other: Self) -> Self:
        """Subtracts two operators."""

    def __imul__(self, other: complex):
        """Multiplies this operator by a scalar."""

    def __mul__(self, other: complex) -> Self:
        """Multiplies an operator by a scalar."""

    def __idiv__(self, other: complex):
        """Divides this operator by a scalar."""

    def __div__(self, other: complex) -> Self:
        """Divides an operator by a scalar."""

    def __neg__(self) -> Self:
        """Negates this operator."""

    def __iand__(self, other: Self):
        """Composes (left-multiplies) another operator onto this one."""

    def __and__(self, other: Self) -> Self:
        """Composes (left-multiplies) two operators."""

    def __imatmul__(self, other: Self):
        """Takes the dot-product (right-multiplication) of another operator onto this one."""

    def __matmul__(self, other: Self) -> Self:
        """Takes the dot-product (right-multiplication) two operators."""

    def __pow__(self, exponent: int, modulo: int | None) -> Self:
        """Exponentiates this operator by the integer exponent."""

    def __len__(self) -> int:
        """Returns the length of this operator."""

    def iter_terms(self) -> Iterator:
        """Iterates over the terms of this operator."""

    @classmethod
    def from_terms(cls, terms: Iterable) -> Self:
        """Constructs a new operator from an iterator (see also :meth:`.iter_terms`)."""

    def iter_terms_with_groups(self) -> Iterator:
        """Iterates over the terms of this operator with their group indices."""

    @classmethod
    def from_terms_with_groups(cls, terms: Iterable) -> Self:
        """Constructs a new operator from an iterator (see also :meth:`.iter_terms_with_groups`)."""

    def equiv(self, other: Self, atol: float) -> bool:
        """Checks this operator with another for equivalence up to the specified absolute tolerance."""

    def ichop(self, atol: float):
        """Trims coefficients below the absolute tolerance from this operator."""

    def simplify(self, atol: float) -> Self:
        """Simplifies the terms of this operator, discarding those below the absolute tolerance."""

    def adjoint(self) -> Self:
        """Returns the adjoint of this operator."""

    def get_support(self) -> frozenset[int]:
        """Returns the set of mode indices which this operator acts upon."""

    def relabel_modes(self, permutation: list[int]) -> Self:
        """Relabels the modes of the operator."""

    def normal_ordered(self, *args, **kwargs) -> Self:
        """Returns the normal-ordered form of this operator.

        .. note::
           A specific implementation of this method may take additional arguments.
        """

    def get_coeffs(self) -> list[complex]:
        """Returns the term coefficients."""

    @property
    def groups(self) -> list[int] | None:
        """Returns the groups indices."""

    @groups.setter
    def groups(self, groups: list[int] | None) -> None:
        """Sets the groups indices."""

    def has_groups(self) -> bool:
        """Returns whether this operator tracks group indices.

        This is equivalent to (but cheaper than) checking ``op.groups is not None``, because it does
        not copy the group indices out of the operator in order to inspect them.
        """

    def num_groups(self) -> int | None:
        """Returns the number of groups."""

    def split_out_groups(self, group_indices: Sequence[int] | None = None) -> list[Self]:
        """Splits this operator into an optional list of new operators based on its :attr:`.groups`.

        .. note::
           If ``group_indices`` is omitted, every group is built, in index order. Otherwise, only
           the requested indices are built, in the given order.
        """
