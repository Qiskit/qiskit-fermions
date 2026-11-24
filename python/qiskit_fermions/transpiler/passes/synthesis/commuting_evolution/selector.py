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

"""Selectors: the policy that picks one of all the variants the candidates built."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from qiskit import QuantumCircuit
from qiskit.transpiler.exceptions import TranspilerError

from .cost import device_cost
from .problem import Context, Problem, Variant


class Selector(Protocol):
    """Chooses one of the variants, or returns ``None`` to decline the gate.

    A selector is called as ``selector(variants, problem, context)`` once per gate, with every
    :class:`.Variant` the candidates built, in tie-break order (the order of the plugin's
    candidates, then each candidate's own order), the gate's :class:`.Problem` and its
    :class:`.Context`. It returns one of the variants, or ``None`` to decline the gate. Any callable
    with this signature is a selector, a plain function included. :class:`.MinCost` is the built-in
    one.
    """

    def __call__(
        self, variants: Sequence[Variant], problem: Problem, context: Context
    ) -> Variant | None:
        """Chooses one of ``variants``.

        Args:
            variants: every variant the candidates built for the gate, in tie-break order (the order
                of the plugin's candidates, then each candidate's own order).
            problem: the gate's terms.
            context: what is known about the device.

        Returns:
            One of ``variants``, or ``None`` to decline the gate.
        """
        ...


class MinCost:
    """A :class:`.Selector` that chooses the variant of the lowest cost.

    Ties go to the earlier variant.

    Each variant is scored once. A variant whose cost raises
    :class:`~qiskit.transpiler.exceptions.TranspilerError` (a routed cost that cannot route it) is
    dropped, and when no variant is left the gate is declined. A single variant is returned without
    being scored.

    Args:
        cost: maps a circuit and its :class:`.Context` to a value; smaller is better. The values of
            all variants must be comparable with ``<`` among themselves, a number or a tuple of
            numbers, say. The default, :func:`.device_cost`, routes onto the device when one is
            known.
    """

    def __init__(self, cost: Callable[[QuantumCircuit, Context], Any] = device_cost) -> None:
        """Configures the selector; see the class documentation for the argument."""
        self._cost = cost

    def __call__(
        self, variants: Sequence[Variant], problem: Problem, context: Context
    ) -> Variant | None:
        """Selects the cheapest of ``variants``."""
        if len(variants) <= 1:
            return variants[0] if variants else None
        best: tuple[Any, int] | None = None
        for i, variant in enumerate(variants):
            try:
                score = (self._cost(variant.circuit, context), i)
            except TranspilerError:
                continue
            if best is None or score < best:
                best = score
        return None if best is None else variants[best[1]]
