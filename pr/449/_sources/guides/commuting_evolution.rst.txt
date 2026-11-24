.. _commuting_evolution_explanation:

Synthesize the evolution under commuting Pauli terms
====================================================

.. meta::
   :description: How to synthesize the PauliEvolutionGate of a group of mutually commuting Pauli
                 terms with the CommutingDiagonalizationSynthesis plugin, on an abstract register
                 or on the qubit pairs of a device, and how to choose its candidates and its cost.

.. important::

   The concepts in this guide are currently available only in the Python API.

When all the Pauli terms of a Hamiltonian :math:`H` commute with each other, its evolution is
exactly

.. math::

   e^{-itH} = U^\dagger e^{-itM} U,

with a Clifford circuit :math:`U` and a middle part :math:`M` that is cheap to synthesize: a
diagonal, or a layer of generic two-qubit blocks next to a diagonal. A term-by-term product
formula ignores this and spends a change of basis on every term. The
:class:`~qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.CommutingDiagonalizationSynthesis`
plugin instead builds several such circuits for every
:class:`~qiskit.circuit.library.PauliEvolutionGate` of commuting terms, and keeps the cheapest. It
looks only at the Pauli terms, so it needs no knowledge of the model or of the fermion encoding
that produced them.

Groups of commuting terms arise wherever a Hamiltonian is split before it is synthesized: by a
grouping of its terms (see :ref:`grouping_explanation`), and by a product formula in fermionic
space that evolves one group at a time (see :ref:`fermionic_synthesis_explanation`). The
:ref:`1D <1d_fermi_hubbard>` and :ref:`2D <2d_fermi_hubbard>` Fermi-Hubbard guides use the plugin
for such groups in a complete pipeline. This guide shows the plugin on its own.

Select the plugin by name
-------------------------

The plugin is a Qiskit
:class:`~qiskit.transpiler.passes.synthesis.plugin.HighLevelSynthesisPlugin` registered as
``"commuting_diagonalization"``. The groups in this section hold the backward hops of a chain
under the encoding of the 1D Fermi-Hubbard guide, a ``Z X Z`` on every window of three qubits. Any
two of them commute:

.. plot::
   :context:
   :nofigs:
   :include-source:

   >>> from qiskit import QuantumCircuit
   >>> from qiskit.circuit.library import PauliEvolutionGate
   >>> from qiskit.quantum_info import SparsePauliOp
   >>>
   >>> def zxz_windows(num_qubits):
   ...     """A circuit with the evolution under a Z X Z on every window of three qubits."""
   ...     terms = [("ZXZ", [k - 1, k, k + 1], 0.5 + 0.1 * k) for k in range(1, num_qubits - 1)]
   ...     operator = SparsePauliOp.from_sparse_list(terms, num_qubits=num_qubits)
   ...     circuit = QuantumCircuit(num_qubits)
   ...     _ = circuit.append(PauliEvolutionGate(operator, time=0.3), range(num_qubits))
   ...     return circuit

Running only the high-level synthesis shows how a gate is built, before any translation into a
basis. Without any configuration, it uses Qiskit's default synthesis for the gate:

.. plot::
   :context: close-figs
   :include-source:
   :alt: Qiskit's default circuit for four Z X Z windows on six qubits: a staircase of CX gates
         across the qubits, with the rotation of each window between its CX gates, and the
         mirrored staircase.

   >>> from qiskit.transpiler.passes import HighLevelSynthesis
   >>>
   >>> HighLevelSynthesis()(zxz_windows(6)).draw("mpl", fold=-1)
   <Figure size ... with 1 Axes>

The default synthesis treats the windows as a sequence of terms. Neighboring windows overlap, so
their two-qubit gates form one staircase along the chain, and a longer chain needs a longer
staircase.

To use the plugin instead, name it in an
:class:`~qiskit.transpiler.passes.synthesis.high_level_synthesis.HLSConfig`, with Qiskit's
default method as a fallback for the gates it declines:

.. plot::
   :context: close-figs
   :include-source:
   :alt: The plugin's circuit for four Z X Z windows on six qubits: two layers of CZ gates on
         neighboring pairs, an H, Rz, H sequence on the four inner qubits, and the two CZ layers
         again in reverse order.

   >>> from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
   >>>
   >>> plugin_config = HLSConfig(PauliEvolution=["commuting_diagonalization", "default"])
   >>> synthesized = HighLevelSynthesis(hls_config=plugin_config)(zxz_windows(6))
   >>> synthesized.draw("mpl")
   <Figure size ... with 1 Axes>

The two CZ layers on the left are the Clifford :math:`U`. They turn every window into an ``X`` on
its middle qubit, which the H gates turn into a ``Z``, so the middle part is one Rz rotation per
term, all of them in parallel. The mirrored layers on the right undo :math:`U`.

That one Clifford serves every window, however long the chain, so the plugin's circuit keeps the
same two-qubit depth while the default synthesis grows with the chain. Transpiling both into a CZ
basis for chains of growing length shows the difference:

.. plot::
   :context: close-figs
   :include-source:
   :alt: Two-qubit depth against the number of qubits, from 8 to 64. The default synthesis grows
         linearly with the chain, while the plugin's circuit stays at the same small depth.

   >>> import matplotlib.pyplot as plt
   >>> from qiskit.transpiler import generate_preset_pass_manager
   >>>
   >>> def two_qubit_depth(circuit):
   ...     return circuit.depth(lambda instruction: len(instruction.qubits) == 2)
   >>>
   >>> def transpile(circuit, hls_config=None, **options):
   ...     pass_manager = generate_preset_pass_manager(
   ...         optimization_level=1,
   ...         basis_gates=["cz", "rz", "sx", "x"],
   ...         hls_config=hls_config,
   ...         seed_transpiler=1,
   ...         **options,
   ...     )
   ...     return pass_manager.run(circuit)
   >>>
   >>> sizes = [8, 16, 24, 32, 48, 64]
   >>> default_depths = [two_qubit_depth(transpile(zxz_windows(n))) for n in sizes]
   >>> plugin_depths = [two_qubit_depth(transpile(zxz_windows(n), plugin_config)) for n in sizes]
   >>>
   >>> figure, axes = plt.subplots(figsize=(6, 3.5), layout="constrained")
   >>> _ = axes.plot(sizes, default_depths, "o-", label="default synthesis")
   >>> _ = axes.plot(sizes, plugin_depths, "s-", label="commuting_diagonalization")
   >>> _ = axes.set(xlabel="number of qubits", ylabel="two-qubit depth", ylim=(0, None))
   >>> _ = axes.legend()
   >>> figure
   <Figure size ... with 1 Axes>

.. invisible-code-block: python

   >>> len(set(plugin_depths)) == 1
   True
   >>> all(plugin < default for plugin, default in zip(plugin_depths, default_depths))
   True
   >>> default_depths == sorted(default_depths) and default_depths[0] < default_depths[-1]
   True

The plugin handles a gate only if all of its terms commute. It declines any other gate, so the
next method of the :class:`~qiskit.transpiler.passes.synthesis.high_level_synthesis.HLSConfig`
synthesizes it, and it never groups terms itself: a gate per commuting group is the input it
expects.

See which circuit the plugin chose
----------------------------------

Every way the plugin builds a circuit is a *candidate*, and a candidate can offer several
*variants* of its circuit. A *selector* then chooses one variant among all of them; the default
one ranks them by two-qubit depth, then two-qubit gate count.

The Z X Z windows are an easy case, for which the candidates that share one Clifford all find
equally good circuits. The ``Z X X Z`` windows of four qubits, the backward hops of a chain whose
spin orbitals are interleaved, are harder, and set the candidates apart:

.. plot::
   :context: close-figs
   :nofigs:
   :include-source:

   >>> def zxxz_windows(num_qubits):
   ...     """A circuit with the evolution under a Z X X Z on every window of four qubits."""
   ...     terms = [
   ...         ("ZXXZ", [k - 1, k, k + 1, k + 2], 0.5 + 0.1 * k) for k in range(1, num_qubits - 2)
   ...     ]
   ...     operator = SparsePauliOp.from_sparse_list(terms, num_qubits=num_qubits)
   ...     circuit = QuantumCircuit(num_qubits)
   ...     _ = circuit.append(PauliEvolutionGate(operator, time=0.3), range(num_qubits))
   ...     return circuit

To see the variants and the choice, pass an instance of the plugin with a ``callback``, which
receives a :class:`.Report` for every gate:

.. plot::
   :context: close-figs
   :nofigs:
   :include-source:

   >>> from qiskit_fermions.transpiler.passes import CommutingDiagonalizationSynthesis
   >>>
   >>> reports = []
   >>> plugin = CommutingDiagonalizationSynthesis(callback=reports.append)
   >>> _ = transpile(zxxz_windows(12), HLSConfig(PauliEvolution=[plugin]))
   >>> report = reports[0]
   >>> [variant.label for variant in report.variants]
   ['plain[elimination-order]', 'majority[elimination-order]', 'local[tree]', 'local[ladder]',
    'search[star-layered]', 'search[star-elimination-order]', 'search[tree-colored-layered]',
    'search[tree-colored-elimination-order]']
   >>> report.chosen.label
   'search[tree-colored-layered]'

Each label names the candidate and, in brackets, its variant. The ``local`` candidate diagonalizes
every term on its own, the others share one Clifford across the group. Every variant is exact: it
implements the evolution, including its global phase, so the choice between them is only one of
cost. :func:`.device_cost` is the cost the default selector ranks by, and scoring every variant
with it shows why the plugin chose as it did:

.. plot::
   :context: close-figs
   :include-source:
   :alt: Horizontal bars of the two-qubit depth and the two-qubit gate count of every variant, with
         the chosen variant highlighted. The depths spread widely across the candidates. The
         chosen variant is the shallowest, but another variant needs fewer two-qubit gates.

   >>> from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution import device_cost
   >>>
   >>> labels = [variant.label for variant in report.variants]
   >>> costs = [device_cost(variant.circuit, report.context) for variant in report.variants]
   >>> colors = ["tab:orange" if v is report.chosen else "tab:gray" for v in report.variants]
   >>>
   >>> figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True, layout="constrained")
   >>> for panel, index, title in zip(axes, (0, 1), ("two-qubit depth", "two-qubit gates")):
   ...     _ = panel.barh(labels, [cost[index] for cost in costs], color=colors)
   ...     _ = panel.set(title=title)
   >>> _ = axes[0].invert_yaxis()
   >>> figure
   <Figure size ... with 2 Axes>

.. invisible-code-block: python

   >>> costs[labels.index(report.chosen.label)] == min(costs)
   True
   >>> len({cost[0] for cost in costs}) > 2
   True
   >>> by_label = dict(zip(labels, costs))
   >>> report.chosen.candidate == "search"
   True
   >>> by_label["local[tree]"][0] < by_label["plain[elimination-order]"][0]
   True
   >>> min(cost[1] for cost in costs) < by_label[report.chosen.label][1]
   True

How well a candidate does depends on the structure of the terms, which is why the plugin asks
them all. Here, the ``search`` candidate finds the shallowest circuit, and the ``local`` one
beats the plain elimination. The ranking is by depth first, so the chosen variant need not have
the fewest two-qubit gates. Ties go to the variant that comes first, in the order of the
candidates and then in each candidate's own order.

The chosen circuit shows how the ``search`` candidate gets there:

.. plot::
   :context: close-figs
   :include-source:
   :alt: The search circuit for nine Z X X Z windows on twelve qubits: a Clifford of CX and CZ
         gates, several of them between distant qubits, then one Rz rotation on a single qubit per
         window, and a second Clifford that undoes the first.

   >>> report.chosen.circuit.draw("mpl", fold=-1)
   <Figure size ... with 1 Axes>

.. invisible-code-block: python

   >>> rotations = [i for i in report.chosen.circuit.data if i.operation.name in ("rz", "rzz")]
   >>> {len(i.qubits) for i in rotations}
   {1}

Its Clifford turns every window into a Z on a single qubit, so the middle part is one Rz rotation
per term. The price is in the Clifford: several of its two-qubit gates act on qubits that are far
apart.

Use the device's connectivity
-----------------------------

The candidates listed so far choose their two-qubit gates from the terms alone, so on a device
their circuits might need routing. For the Z X Z windows that costs nothing, because their shared
Clifford happens to act on neighbors only. The Z X X Z windows are different.

The ``cz-local`` candidate puts every two-qubit gate on a pair of qubits that can interact, so it
has to know those pairs. A preset pass manager runs the high-level synthesis in its ``init``
stage, before the layout is chosen, so the plugin cannot read the pairs off the device there. You
can pass them yourself as ``edges``, in the qubit indices of the circuit you transpile. The
example below places the circuit's qubits in order on a line of qubits, so its pairs are the
neighboring qubits:

.. plot::
   :context: close-figs
   :nofigs:
   :include-source:

   >>> from qiskit.transpiler import CouplingMap
   >>>
   >>> def line_pairs(num_qubits):
   ...     return [(q, q + 1) for q in range(num_qubits - 1)]
   >>>
   >>> def on_a_line(circuit, hls_config=None):
   ...     """Transpiles onto a line of qubits, with the circuit's qubits placed in order."""
   ...     num_qubits = circuit.num_qubits
   ...     line = CouplingMap.from_line(num_qubits)
   ...     layout = list(range(num_qubits))
   ...     return transpile(circuit, hls_config, coupling_map=line, initial_layout=layout)
   >>>
   >>> reports = []
   >>> aware = CommutingDiagonalizationSynthesis(edges=line_pairs(12), callback=reports.append)
   >>> _ = on_a_line(zxxz_windows(12), HLSConfig(PauliEvolution=[aware]))
   >>> reports[0].chosen.label
   'cz-local[diagonal]'

With the pairs, the plugin chooses the ``cz-local`` circuit. Routing the three syntheses onto lines
of growing length shows why:

.. plot::
   :context: close-figs
   :include-source:
   :alt: Two-qubit depth and two-qubit gate count after routing onto a line, against the number of
         qubits from 8 to 64, for the default synthesis, the plugin without edges, and the plugin
         with edges. The default synthesis grows linearly in both. The plugin without edges stays
         at a constant depth but needs as many gates as the default. The plugin with edges is the
         cheapest in both, at a constant depth of a fraction of the others.

   >>> def with_edges(num_qubits):
   ...     plugin = CommutingDiagonalizationSynthesis(edges=line_pairs(num_qubits))
   ...     return HLSConfig(PauliEvolution=[plugin])
   >>>
   >>> configs = {
   ...     "default synthesis": lambda num_qubits: None,
   ...     "plugin without edges": lambda num_qubits: plugin_config,
   ...     "plugin with edges": with_edges,
   ... }
   >>> routed = {
   ...     name: [on_a_line(zxxz_windows(n), config(n)) for n in sizes]
   ...     for name, config in configs.items()
   ... }
   >>>
   >>> metrics = {
   ...     "two-qubit depth": two_qubit_depth,
   ...     "two-qubit gates": QuantumCircuit.num_nonlocal_gates,
   ... }
   >>> figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
   >>> for panel, (title, metric) in zip(axes, metrics.items()):
   ...     for (name, circuits), style in zip(routed.items(), ("o-", "s--", "^-")):
   ...         _ = panel.plot(sizes, [metric(c) for c in circuits], style, label=name)
   ...     _ = panel.set(xlabel="number of qubits", title=title, ylim=(0, None))
   >>> _ = axes[0].legend()
   >>> figure
   <Figure size ... with 2 Axes>

.. invisible-code-block: python

   >>> values = {
   ...     (name, title): [metric(c) for c in circuits]
   ...     for name, circuits in routed.items()
   ...     for title, metric in metrics.items()
   ... }
   >>> len(set(values["plugin with edges", "two-qubit depth"])) == 1
   True
   >>> all(
   ...     best < other
   ...     for title in metrics
   ...     for name in ("default synthesis", "plugin without edges")
   ...     for best, other in zip(values["plugin with edges", title], values[name, title])
   ... )
   True
   >>> all(
   ...     blind < default
   ...     for blind, default in zip(
   ...         values["plugin without edges", "two-qubit depth"],
   ...         values["default synthesis", "two-qubit depth"],
   ...     )
   ... )
   True
   >>> all(
   ...     blind >= default
   ...     for blind, default in zip(
   ...         values["plugin without edges", "two-qubit gates"],
   ...         values["default synthesis", "two-qubit gates"],
   ...     )
   ... )
   True

Without the pairs, the plugin's circuit is still much shallower than the default synthesis, but
routing it costs at least as many two-qubit gates: its shared Clifford has gates between distant
qubits, which the router has to bring together with SWAP gates. The ``cz-local`` circuit needs no
routing at all. Compared with the ``search`` circuit for the same twelve qubits above, it does not
reduce the windows as far, only to Z-strings on two neighboring qubits. In return, its CZ layers
act on neighbors only, and so do the RZZ rotations of its diagonal middle part:

.. plot::
   :context: close-figs
   :include-source:
   :alt: The cz-local circuit for nine Z X X Z windows on twelve qubits: two layers of CZ gates on
         neighboring pairs, S-dagger and H gates on the inner qubits, two layers of RZZ rotations
         on neighboring pairs, and the mirrored gates again.

   >>> reports[0].chosen.circuit.draw("mpl", fold=-1)
   <Figure size ... with 1 Axes>

.. invisible-code-block: python

   >>> chosen = reports[0].chosen.circuit
   >>> pairs_used = {
   ...     tuple(sorted(chosen.find_bit(qubit).index for qubit in instruction.qubits))
   ...     for instruction in chosen.data
   ...     if len(instruction.qubits) == 2
   ... }
   >>> pairs_used <= set(line_pairs(12))
   True

When the pass manager knows the device's coupling map, as here, the default selector scores every
variant after a quick transpile onto it, so a variant that needs routing pays for it in the
ranking. When the high-level synthesis runs after the layout stage instead, the plugin reads the
pairs off the device itself.

Choose the candidates and the cost
----------------------------------

The candidates and the selector are both arguments of the plugin, so you can trade circuit
quality against compile time, or rank the circuits by what matters on your device.

Leave out a candidate
~~~~~~~~~~~~~~~~~~~~~

The ``search`` candidate is by far the most expensive to run, because it tries many changes of
basis for every gate. For a circuit with many large groups, you might prefer to leave it out:

.. plot::
   :context: close-figs
   :nofigs:
   :include-source:

   >>> from qiskit_fermions.transpiler.passes.synthesis.commuting_evolution import (
   ...     DEFAULT_CANDIDATES,
   ...     MinCost,
   ... )
   >>>
   >>> gate = zxxz_windows(12).data[0].operation
   >>> without_search = [c for c in DEFAULT_CANDIDATES if c.name != "search"]
   >>> reports = []
   >>> fast = CommutingDiagonalizationSynthesis(candidates=without_search, callback=reports.append)
   >>> _ = fast.run(gate)
   >>> reports[0].chosen.label
   'local[tree]'

.. invisible-code-block: python

   >>> fast_cost = device_cost(reports[0].chosen.circuit, reports[0].context)
   >>> fast_cost[0] > by_label[report.chosen.label][0]
   True

The plugin is much faster on this gate now, but it settles for the ``local[tree]`` variant of the
bar chart above, which is deeper than what the ``search`` candidate found. Narrowing the candidates
down to a single variant takes that circuit directly, without scoring it.

Rank by a different cost
~~~~~~~~~~~~~~~~~~~~~~~~

The default selector ranks by two-qubit depth first. On a device where the errors of two-qubit
gates dominate and idle qubits cost little, the number of two-qubit gates is the better measure.
:class:`.MinCost` takes any cost function that maps a circuit and its :class:`.Context` to
something that compares with ``<``, so swapping the two entries of the built-in
:func:`.device_cost` ranks by count first:

.. plot::
   :context: close-figs
   :nofigs:
   :include-source:

   >>> def count_first(circuit, context):
   ...     depth, count = device_cost(circuit, context)
   ...     return count, depth
   >>>
   >>> reports = []
   >>> by_count = CommutingDiagonalizationSynthesis(
   ...     selector=MinCost(count_first), callback=reports.append
   ... )
   >>> _ = by_count.run(gate)
   >>> reports[0].chosen.label
   'search[star-layered]'

.. invisible-code-block: python

   >>> count_cost = device_cost(reports[0].chosen.circuit, reports[0].context)
   >>> default_cost = by_label[report.chosen.label]
   >>> count_cost[1] < default_cost[1] and count_cost[0] > default_cost[0]
   True

This is the variant with the fewest two-qubit gates in the bar chart, at a slightly larger depth
than the default choice.

The built-in cost functions, the built-in candidates, and the interface of a candidate of your own
are described in :mod:`qiskit_fermions.transpiler.passes.synthesis.commuting_evolution`.

Next steps
----------

- :ref:`1d_fermi_hubbard` uses the plugin for the flow sets of a chain, where it finds the
  constant-depth circuit of a hand-derived construction by itself.
- :ref:`2d_fermi_hubbard` uses it on a square lattice, with the qubit pairs of a local encoding.
- :ref:`fermionic_synthesis_explanation` splits an evolution into groups in fermionic space.
- :ref:`grouping_explanation` describes how the terms of an operator are grouped.
