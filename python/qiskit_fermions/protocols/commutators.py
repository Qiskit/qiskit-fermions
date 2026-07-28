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

"""A protocol to indicate commutator computation support."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T", bound="SupportsCommutators")


@runtime_checkable
class SupportsCommutators(Protocol):
    """A runtime-checkable Protocol indicating support for efficient commutator generation.

    Implementation of this protocol requires the three methods below.
    """

    @staticmethod
    def _commutator_(op_a: T, op_b: T) -> T:
        """Computes the commutator of two operator instances.

        See :func:`.commutator` for more details.
        """
        ...

    @staticmethod
    def _anti_commutator_(op_a: T, op_b: T) -> T:
        """Computes the anti-commutator of two operator instances.

        See :func:`.anti_commutator` for more details.
        """
        ...

    @staticmethod
    def _double_commutator_(op_a: T, op_b: T, op_c: T, sign: bool) -> T:
        """Computes the double-commutator of three operator instances.

        See :func:`.double_commutator` for more details.
        """
        ...
