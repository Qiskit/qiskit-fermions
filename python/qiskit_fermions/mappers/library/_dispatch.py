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

"""Type dispatch shared by the type-agnostic mapper wrappers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from qiskit.quantum_info import SparseObservable


def dispatch_mapper(
    name: str,
    implementations: Mapping[type, Callable[..., SparseObservable]],
    operator: Any,
    *args: Any,
) -> SparseObservable:
    """Call the implementation registered for the type of ``operator``.

    Args:
        name: the wrapper's name, used in the error message.
        implementations: the direct implementation for each supported operator type.
        operator: the operator to map.
        *args: the remaining arguments, passed through after ``operator``.

    Returns:
        Whatever the matching implementation returns.

    Raises:
        TypeError: if no implementation matches. This is what turns a non-operator argument into a
            clear error instead of an obscure failure deeper in, and it is what a future operator
            type will land on until it gains an implementation.
    """
    for operator_type, implementation in implementations.items():
        if isinstance(operator, operator_type):
            return implementation(operator, *args)
    raise TypeError(f"{name} does not support operator type {type(operator).__name__!r}.")
