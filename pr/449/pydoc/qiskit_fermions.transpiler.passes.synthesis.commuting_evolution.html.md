<a id="module-qiskit_fermions.transpiler.passes.synthesis.commuting_evolution"></a>

<a id="qiskit-fermions-transpiler-passes-synthesis-commuting-evolution"></a>

# Synthesis of commuting Pauli evolutions

The evolution $e^{-i t H}$ of a Hamiltonian whose Pauli terms all commute is exactly
$U^\dagger e^{-i t M} U$, with a Clifford $U$ and a middle part $M$ that is cheap
to synthesize: a diagonal, when $U$ turns every term into a Z-string, or a layer of generic
two-qubit blocks next to a diagonal, when $U$ only brings every term onto one qubit or one
pair of qubits. Many Clifford circuits do that, and which one is best depends on the structure of
the terms, on the device, and on what you count as best. The
[`CommutingDiagonalizationSynthesis`](../stubs/qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.CommutingDiagonalizationSynthesis.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.CommutingDiagonalizationSynthesis) plugin therefore separates two jobs:

* **Candidates** build exact circuits. Each one is a way of finding a circuit, and may offer several
  *variants* of it. They look only at the Pauli terms (the [`Problem`](../stubs/qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem)) and at what is known
  about the device (the [`Context`](../stubs/qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context)), so they need no knowledge of the model or of the fermion
  encoding that produced the terms.
* A **selector** chooses one of all the candidates’ variants. The default, [`MinCost`](../stubs/qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.MinCost.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.MinCost), takes
  the cheapest under a cost function.

Both are plain Python objects, so either can be replaced. The plugin is a Qiskit
[`HighLevelSynthesisPlugin`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.transpiler.passes.synthesis.plugin.HighLevelSynthesisPlugin). Select it by name,
with the default method as a fallback for the gates it declines:

```python
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig

hls_config = HLSConfig(PauliEvolution=["commuting_diagonalization", "default"])
```

or pass an instance, to choose its candidates and its selector. To take one particular circuit,
narrow the candidates down to it: a single variant is returned without being scored. Below, the
plugin takes exactly the ladder variant of the [`Local`](../stubs/qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Local.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Local) candidate:

```pycon
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
```

A cost function maps a circuit and its context to anything comparable with `<`. Here the default
candidates are ranked by their gate count, and a `callback` receives a [`Report`](../stubs/qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Report.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Report) of every
variant that was built and of the one that was chosen, which is the cheapest of them:

```pycon
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
```

A candidate of your own needs only a `name` and a `generate` method that receives the
[`Problem`](../stubs/qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem) and the [`Context`](../stubs/qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context) and returns its circuits by variant name; the plugin
wraps each of them in a [`Variant`](../stubs/qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Variant.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Variant) for the selector. This one offers Qiskit’s own synthesis,
so that the built-in candidates have to beat it:

```pycon
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
```

The candidates that use the device’s connectivity need the physical qubits of the gate, so they
offer circuits only when the high-level synthesis happens after the layout stage, with an initial
layout that places the problem sensibly on the device, or when the plugin is given `edges`. When
run in the `init` stage, which is the default, the plugin sees no qubits and uses only the other
candidates.

The plugin does not group terms. It handles a [`PauliEvolutionGate`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.PauliEvolutionGate)
only if all of its terms commute with each other, and declines it otherwise, so splitting a
Hamiltonian into commuting groups, one gate per group, is the caller’s job. A gate whose operator is
a list of such groups, the form in which Qiskit’s product formulas receive a grouping, is declined
as well: the next method then synthesizes it with the gate’s own product formula. To synthesize the
groups with this plugin, put each of them into a gate of its own.

### The plugin

### Problem, context and variants

### Candidates

### DEFAULT_CANDIDATES *= (<qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.candidates.clifford.Plain object>, <qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.candidates.clifford.Majority object>, <qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.candidates.clifford.TwinFold object>, <qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.candidates.local.Local object>, <qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.candidates.cz_local.CzLocal object>, <qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.candidates.search.Search object>)*

Built-in immutable sequence.

If no argument is given, the constructor returns an empty tuple.
If iterable is specified the tuple is initialized from iterable’s items.

If the argument is a tuple, the return value is the same object.

### Selectors and costs
