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

"""Private typing helpers shared by the representation mapper generators."""

import sys
from typing import Protocol, TypeVar, runtime_checkable

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


@runtime_checkable
class MappableOperator(Protocol):
    """The structural output-type contract of the ``map_*_generators`` functions.

    A mapper generator builds up its result by scaling mapped terms by a (complex) coefficient and
    summing them, so the output type must support scalar multiplication and addition of two
    instances. This mirrors the requirement stated in each generator's docstring and lets the output
    type variable be bound rather than unbounded (an unbounded ``TypeVar`` supports no operators).
    """

    def __rmul__(self, other: complex) -> Self: ...

    def __add__(self, other: Self) -> Self: ...


T = TypeVar("T", bound=MappableOperator)
"""The output type variable of the ``map_*_generators`` functions."""
