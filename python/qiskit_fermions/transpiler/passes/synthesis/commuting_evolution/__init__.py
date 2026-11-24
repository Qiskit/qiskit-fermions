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

# ruff: noqa: D212,D415
r"""
.. _qiskit_fermions-transpiler-passes-synthesis-commuting_evolution:

=======================================
Synthesis of commuting Pauli evolutions
=======================================

The evolution :math:`e^{-i t H}` of a Hamiltonian whose Pauli terms all commute is exactly
:math:`U^\dagger e^{-i t M} U`, with a Clifford :math:`U` and a middle part :math:`M` that is cheap
to synthesize: a diagonal, when :math:`U` turns every term into a Z-string, or a layer of generic
two-qubit blocks next to a diagonal, when :math:`U` only brings every term onto one qubit or one
pair of qubits. Many Clifford circuits do that, and which one is best depends on the structure of
the terms, on the device, and on what you count as best. The
:class:`.CommutingDiagonalizationSynthesis` plugin therefore separates two jobs:

* **Candidates** build exact circuits. Each one is a way of finding a circuit, and may offer several
  *variants* of it. They look only at the Pauli terms (the :class:`.Problem`) and at what is known
  about the device (the :class:`.Context`), so they need no knowledge of the model or of the fermion
  encoding that produced the terms.
* A **selector** chooses one of all the candidates' variants. The default, :class:`.MinCost`, takes
  the cheapest under a cost function.

Both are plain Python objects, so either can be replaced. The plugin is a Qiskit
:class:`~qiskit.transpiler.passes.synthesis.plugin.HighLevelSynthesisPlugin`. Select it by name,
with the default method as a fallback for the gates it declines:

.. code-block:: python

    from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig

    hls_config = HLSConfig(PauliEvolution=["commuting_diagonalization", "default"])

or pass an instance, to choose its candidates and its selector. To take one particular circuit,
narrow the candidates down to it: a single variant is returned without being scored. Below, the
plugin takes exactly the ladder variant of the :class:`.Local` candidate:

.. doctest::

    >>> from qiskit.circuit.library import PauliEvolutionGate
    >>> from qiskit.quantum_info import SparsePauliOp
    >>> from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution import (
    ...     DEFAULT_CANDIDATES,
    ...     CommutingDiagonalizationSynthesis,
    ...     Local,
    ...     MinCost,
    ... )
    >>> operator = SparsePauliOp(["XXII", "ZZII", "IIYY"], [0.3, 0.5, 0.7])
    >>> gate = PauliEvolutionGate(operator, time=0.4)
    >>> plugin = CommutingDiagonalizationSynthesis(candidates=[Local(variants=["ladder"])])
    >>> plugin.run(gate) is not None
    True

A cost function maps a circuit and its context to anything comparable with ``<``. Here the default
candidates are ranked by their gate count, and a ``callback`` receives a :class:`.Report` of every
variant that was built and of the one that was chosen, which is the cheapest of them:

.. doctest::

    >>> def gate_count(circuit, context):
    ...     return circuit.size()
    >>> reports = []
    >>> plugin = CommutingDiagonalizationSynthesis(
    ...     selector=MinCost(gate_count), callback=reports.append
    ... )
    >>> _ = plugin.run(gate)
    >>> report = reports[0]
    >>> cheapest = min(gate_count(v.circuit, report.context) for v in report.variants)
    >>> gate_count(report.chosen.circuit, report.context) == cheapest
    True

A candidate of your own needs only a ``name`` and a ``generate`` method that receives the
:class:`.Problem` and the :class:`.Context` and returns its circuits by variant name; the plugin
wraps each of them in a :class:`.Variant` for the selector. This one offers Qiskit's own synthesis,
so that the built-in candidates have to beat it:

.. doctest::

    >>> from qiskit import QuantumCircuit
    >>> from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution import Context, Problem
    >>> class Stock:
    ...     name = "stock"
    ...     def generate(self, problem: Problem, context: Context) -> dict[str, QuantumCircuit]:
    ...         circuit = QuantumCircuit(problem.num_qubits)
    ...         circuit.append(PauliEvolutionGate(problem.operator, problem.time), circuit.qubits)
    ...         return {"": circuit.decompose()}
    >>> plugin = CommutingDiagonalizationSynthesis(candidates=[*DEFAULT_CANDIDATES, Stock()])
    >>> plugin.run(gate) is not None
    True

The plugin does not group terms. It handles a :class:`~qiskit.circuit.library.PauliEvolutionGate`
only if all of its terms commute with each other, and declines it otherwise, so splitting a
Hamiltonian into commuting groups, one gate per group, is the caller's job. A gate whose operator is
a list of such groups, the form in which Qiskit's product formulas receive a grouping, is declined
as well: the next method then synthesizes it with the gate's own product formula. To synthesize the
groups with this plugin, put each of them into a gate of its own.

.. currentmodule:: qiskit_fermions.transpiler.passes.synthesis.commuting_evolution

.. rubric:: The plugin

.. autosummary::
   :toctree: ../stubs/

   CommutingDiagonalizationSynthesis
   Report

.. rubric:: Problem, context and variants

.. autosummary::
   :toctree: ../stubs/

   Problem
   Context
   Variant

.. rubric:: Candidates

.. autosummary::
   :toctree: ../stubs/

   Candidate
   Plain
   Majority
   TwinFold
   Local
   Search

.. autodata:: DEFAULT_CANDIDATES

.. rubric:: Selectors and costs

.. autosummary::
   :toctree: ../stubs/

   Selector
   MinCost
   abstract_cost
   device_cost
"""

from .candidates import (
    DEFAULT_CANDIDATES,
    Candidate,
    Local,
    Majority,
    Plain,
    Search,
    TwinFold,
)
from .cost import abstract_cost, device_cost
from .plugin import CommutingDiagonalizationSynthesis
from .problem import Context, Problem, Report, Variant
from .selector import MinCost, Selector

__all__ = [
    "DEFAULT_CANDIDATES",
    "Candidate",
    "CommutingDiagonalizationSynthesis",
    "Context",
    "Local",
    "Majority",
    "MinCost",
    "Plain",
    "Problem",
    "Report",
    "Search",
    "Selector",
    "TwinFold",
    "Variant",
    "abstract_cost",
    "device_cost",
]
