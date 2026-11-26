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

"""The high-level synthesis plugin for the evolution under mutually commuting Pauli terms."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from qiskit.transpiler.passes.synthesis.plugin import HighLevelSynthesisPlugin

from ._layers import sorted_pair
from .candidates import DEFAULT_CANDIDATES, Candidate
from .problem import Context, Declined, Problem, Report, Variant
from .selector import MinCost, Selector


class CommutingDiagonalizationSynthesis(HighLevelSynthesisPlugin):
    """Exact synthesis of a :class:`~qiskit.circuit.library.PauliEvolutionGate` of commuting terms.

    The gate is ``exp(-i t H)`` with ``H`` a sum of mutually commuting Pauli terms with real
    coefficients. For each gate the plugin asks every candidate for its circuits (its *variants*),
    and lets the selector choose one of them.

    * A **candidate** (see :class:`.Candidate`) builds exact circuits from the terms
      (:class:`.Problem`) and what is known about the device (:class:`.Context`). The built-in ones
      write the evolution as ``U† exp(-i t M) U``, with a Clifford ``U`` and a middle part ``M``
      that is diagonal, or holds generic two-qubit blocks, or diagonalize every term on its own. Any
      object with a ``name`` and a ``generate`` method can be a candidate.
    * The **selector** (see :class:`.Selector`) sees all variants at once and returns one, or
      ``None`` to decline. The default, :class:`.MinCost` with :func:`.device_cost`, takes the
      variant of the lowest two-qubit depth, then count, after a quick routed transpile onto the
      device when one is known. Ties go to the earlier variant: the order of ``candidates``, then
      each candidate's own order.

    The candidates that use the device's connectivity need the physical qubits of the gate, so they
    offer nothing unless the plugin runs after the layout stage, or ``edges`` are given.

    **Example.** The terms below commute, so the plugin synthesizes the gate; its output is exact.

    .. doctest::

        >>> import numpy as np
        >>> from qiskit import QuantumCircuit
        >>> from qiskit.circuit.library import PauliEvolutionGate
        >>> from qiskit.quantum_info import Operator, SparsePauliOp
        >>> from qiskit.transpiler import generate_preset_pass_manager
        >>> from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
        >>>
        >>> op = SparsePauliOp.from_sparse_list(
        ...     [("XX", [0, 1], 0.7), ("ZZ", [0, 1], 0.3), ("ZZ", [2, 3], 0.2)], num_qubits=4
        ... )
        >>> circuit = QuantumCircuit(4)
        >>> _ = circuit.append(PauliEvolutionGate(op, time=0.5), [0, 1, 2, 3])
        >>> config = HLSConfig(PauliEvolution=["commuting_diagonalization", "default"])
        >>> pm = generate_preset_pass_manager(
        ...     basis_gates=["cx", "rz", "sx", "x"], hls_config=config
        ... )
        >>> exact = Operator(PauliEvolutionGate(op, time=0.5))
        >>> bool(np.allclose(Operator(pm.run(circuit)).data, exact.data))
        True

    The ``"default"`` entry is the fallback for the gates the plugin declines, such as a group whose
    terms do not commute.

    **Declines** (returns ``None``, so the next method of the ``HLSConfig`` runs) a gate it does not
    handle: anything but a :class:`~qiskit.circuit.library.PauliEvolutionGate` over a single
    operator with a plain real time (a list of operators or a parameterized time included). It also
    declines a gate it handles when the operator is not a Hermitian sum of mutually commuting
    Paulis, or when the selector returns ``None`` (the default one does when no candidate built a
    circuit, or none of them can be routed onto the coupling map). The gate's own ``synthesis``
    attribute is ignored.

    Args:
        candidates: the candidates to ask, in the order in which ties are broken. Their names must
            be unique.
        selector: chooses one of the variants. ``None`` (the default) is :class:`.MinCost` with
            :func:`.device_cost`.
        edges: allowed CZ pairs for the candidates that use them, in the qubit indices of the
            circuit that is being transpiled: its virtual qubits before the layout stage, its
            physical ones after it. Each gate uses the pairs between two of its own qubits. They
            take precedence over the device's.
        callback: called with a :class:`.Report` for every gate the plugin handles, including the
            ones it then declines. A gate it does not handle is passed on without a report.

    Raises:
        ValueError: if two candidates have the same name.
        ValueError: if ``edges`` holds something other than pairs of two distinct non-negative qubit
            indices.
    """

    def __init__(
        self,
        candidates: Iterable[Candidate] = DEFAULT_CANDIDATES,
        selector: Selector | None = None,
        *,
        edges: Iterable[tuple[int, int]] | None = None,
        callback: Callable[[Report], None] | None = None,
    ):
        """Configures the plugin; see the class documentation for the arguments."""
        super().__init__()
        candidates = tuple(candidates)  # read only once, so that an iterator works too
        names = [c.name for c in candidates]
        if len(names) > len(set(names)):
            raise ValueError(f"the candidates' names must be unique, got duplicates in {names}")
        pairs = None if edges is None else [tuple(e) for e in edges]  # read only once, likewise
        if pairs is not None:
            bad = [e for e in pairs if len(e) != 2 or e[0] == e[1] or min(e) < 0]
            if bad:
                raise ValueError(
                    f"edges must be pairs of two distinct non-negative qubit indices, got {bad}"
                )
        self.candidates = candidates
        self.selector: Selector = MinCost() if selector is None else selector
        self.edges = None if pairs is None else tuple(sorted_pair((a, b)) for a, b in pairs)
        self.callback = callback

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Synthesizes one ``PauliEvolutionGate``, or returns ``None`` to decline it.

        When the plugin declines, the high-level synthesis tries the next method.

        ``qubits`` are the gate's physical qubits, which the high-level synthesis passes only after
        the layout stage. The option ``input_qubits``, which it passes always, gives the gate's
        qubits in the circuit, so that ``edges`` can be mapped onto the gate before the layout stage
        too. Called directly without either, the gate's qubits are ``0, 1, ...``.
        """
        try:
            problem = Problem.from_gate(high_level_object)
        except Declined as reason:
            self._report(None, None, [], None, str(reason))
            return None
        if problem is None:
            return None  # not a gate this plugin handles: not reported
        context = Context(
            problem.num_qubits,
            coupling_map=coupling_map,
            target=target,
            physical_qubits=None if qubits is None else tuple(qubits),
            user_edges=self._edges_of(problem.num_qubits, options.get("input_qubits", qubits)),
        )
        variants: list[Variant] = []
        for candidate in self.candidates:
            built = candidate.generate(problem, context)
            variants += [Variant(candidate.name, name, circuit) for name, circuit in built.items()]
        chosen = self.selector(variants, problem, context)
        self._report(
            problem,
            context,
            variants,
            chosen,
            None if chosen is not None else "the selector chose no circuit",
        )
        return None if chosen is None else chosen.circuit

    def _edges_of(
        self, num_qubits: int, circuit_qubits: Sequence[int] | None
    ) -> tuple[tuple[int, int], ...] | None:
        """The user's pairs between two of the gate's qubits, in the gate's own indices.

        ``None`` if the user gave no pairs.
        """
        if self.edges is None:
            return None
        where = {
            q: i
            for i, q in enumerate(range(num_qubits) if circuit_qubits is None else circuit_qubits)
        }
        return tuple(
            sorted_pair((where[a], where[b])) for a, b in self.edges if a in where and b in where
        )

    def _report(
        self,
        problem: Problem | None,
        context: Context | None,
        variants: list[Variant],
        chosen: Variant | None,
        declined: str | None,
    ) -> None:
        if self.callback is not None:
            self.callback(Report(problem, context, tuple(variants), chosen, declined))
