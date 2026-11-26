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

"""The built-in cost functions of :class:`.MinCost`.

A cost function takes a circuit and its :class:`.Context` and returns any comparable value; smaller
is better. The built-in ones return the pair ``(two-qubit depth, two-qubit gate count)``, compared
lexicographically, of the circuit translated into the basis of the device's target, or into the
``basis`` argument (by default :data:`SCORING_BASIS`) when no target is known.

The routed cost transpiles at ``optimization_level=1`` with ``seed_transpiler=SEED``, so that it is
deterministic. Level 1 does not include the two-qubit peephole optimization of the higher levels, so
it can rank circuits differently from what the final pipeline would measure. It can also change with
the Qiskit version, because the transpiler's routing does.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING, Any

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap, PassManager, generate_preset_pass_manager
from qiskit.transpiler.passes.routing.algorithms import ApproximateTokenSwapper

if TYPE_CHECKING:
    from .problem import Context

SCORING_BASIS = ("cz", "rz", "sx", "x")
"""The default basis of the scoring when no target is known."""

SEED = 42
"""The seed of every scoring transpile."""


def _two_qubit(instruction) -> bool:
    return len(instruction.qubits) == 2


def _score(circuit: QuantumCircuit) -> tuple[int, int]:
    return circuit.depth(_two_qubit), circuit.num_nonlocal_gates()


@cache
def _translation(basis: tuple[str, ...]) -> PassManager:
    """High-level synthesis and basis translation into ``basis``, and nothing else.

    There is no layout, routing or optimization.
    """
    return generate_preset_pass_manager(optimization_level=0, basis_gates=list(basis))


def abstract_cost(
    circuit: QuantumCircuit, context: Context | None = None, *, basis: Sequence[str] = SCORING_BASIS
) -> tuple[int, int]:
    """The two-qubit depth and gate count of the circuit in ``basis``, without routing.

    The circuit is translated by Qiskit's high-level synthesis and basis translation, as in a
    transpile without layout, routing or optimization, so every gate is counted the way the device
    would execute it. The count covers every two-qubit gate of the result.

    Args:
        circuit: the circuit to score.
        context: ignored; accepted so that this is a cost function for :class:`.MinCost`.
        basis: the basis gates to translate into. To score in another basis, bind it with
            :func:`functools.partial` before passing the function to :class:`.MinCost`.

    Returns:
        ``(two-qubit depth, two-qubit gate count)``.
    """
    return _score(_translation(tuple(basis)).run(circuit))


def device_cost(
    circuit: QuantumCircuit, context: Context, *, basis: Sequence[str] = SCORING_BASIS
) -> tuple[int, int]:
    """The two-qubit depth and count after a quick transpile onto the device, when one is known.

    The circuit is transpiled into the basis of the context's target, or into ``basis`` without one.
    With a coupling map and the physical qubits, qubit ``i`` of the circuit is placed on device
    qubit ``physical_qubits[i]``, so that the router sees exactly the connectivity the real circuit
    will see. With a coupling map only, the router chooses the placement. Without a coupling map,
    nothing is routed.

    A router may leave the qubits permuted at the end. The circuit that follows expects every qubit
    back in its place, so a permutation is undone with SWAP gates on device edges before the circuit
    is counted. Otherwise a circuit that the router leaves permuted would look cheaper than one that
    needs no routing at all.

    Args:
        circuit: the circuit to score.
        context: where it will run.
        basis: the basis gates to translate into when the context has no target. To score in another
            basis, bind it with :func:`functools.partial` before passing the function to
            :class:`.MinCost`.

    Returns:
        ``(two-qubit depth, two-qubit gate count)``.

    Raises:
        TranspilerError: if the circuit cannot be routed onto the coupling map. :class:`.MinCost`
            drops such a circuit.
    """
    target = context.target
    coupling = context.coupling_map
    if coupling is None and target is not None:
        coupling = target.build_coupling_map()
    if coupling is None:
        names = tuple(basis) if target is None else tuple(sorted(target.operation_names))
        return _score(_translation(names).run(circuit))
    # the target carries its own connectivity and basis
    device: dict[str, Any] = (
        {"target": target}
        if target is not None
        else {"coupling_map": coupling, "basis_gates": list(basis)}
    )
    if context.physical_qubits is not None:
        size = coupling.size()
        placed = QuantumCircuit(size)
        placed.compose(circuit, qubits=list(context.physical_qubits), inplace=True)
        circuit, initial_layout = placed, list(range(size))
    else:
        initial_layout = None
    out = transpile(
        circuit,
        **device,
        initial_layout=initial_layout,
        optimization_level=1,
        seed_transpiler=SEED,
    )
    return _score(_undo_permutation(out, coupling, device))


def _undo_permutation(
    out: QuantumCircuit, coupling: CouplingMap, device: dict[str, Any]
) -> QuantumCircuit:
    """``out`` with its final permutation undone by SWAP gates on device edges, re-optimized.

    The SWAP gates come from Qiskit's token swapper on the coupling graph; the second transpile,
    onto the same ``device`` (the transpile arguments that name the target, or the coupling map and
    the basis), keeps every qubit in place (trivial layout, no routing) and merges them with the
    gates around them where it can.
    """
    if out.layout is None:
        return out
    initial = out.layout.initial_index_layout(filter_ancillas=True)
    final = out.layout.final_index_layout(filter_ancillas=True)
    if final == initial:
        return out
    restored = QuantumCircuit(out.num_qubits)
    restored.compose(out, inplace=True)  # copies the gates and the global phase, but not the layout
    graph = coupling.graph.to_undirected(multigraph=False)
    # every circuit qubit goes back from where it ended to where it started; the ancillas may go
    # anywhere
    swaps = ApproximateTokenSwapper(graph, seed=SEED).map(dict(zip(final, initial, strict=True)))
    for a, b in swaps:
        restored.swap(a, b)
    return transpile(
        restored,
        **device,
        initial_layout=list(range(out.num_qubits)),
        layout_method="trivial",
        routing_method="none",
        optimization_level=1,
        seed_transpiler=SEED,
    )
