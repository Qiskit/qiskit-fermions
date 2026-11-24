# CommutingDiagonalizationSynthesis

### *class* CommutingDiagonalizationSynthesis(candidates=(<qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.candidates.clifford.Plain object>, <qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.candidates.clifford.Majority object>, <qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.candidates.clifford.TwinFold object>, <qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.candidates.local.Local object>, <qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.candidates.cz_local.CzLocal object>, <qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.candidates.search.Search object>), selector=None, \*, edges=None, callback=None)

Exact synthesis of a [`PauliEvolutionGate`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.PauliEvolutionGate) of commuting terms.

Bases: [`HighLevelSynthesisPlugin`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.transpiler.passes.synthesis.plugin.HighLevelSynthesisPlugin)

The gate is `exp(-i t H)` with `H` a sum of mutually commuting Pauli terms with real
coefficients. For each gate the plugin asks every candidate for its circuits (its *variants*),
and lets the selector choose one of them.

* A **candidate** (see [`Candidate`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Candidate.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Candidate)) builds exact circuits from the terms
  ([`Problem`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Problem)) and what is known about the device ([`Context`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Context)). The built-in ones
  write the evolution as `U† exp(-i t M) U`, with a Clifford `U` and a middle part `M`
  that is diagonal, or holds generic two-qubit blocks, or diagonalize every term on its own. Any
  object with a `name` and a `generate` method can be a candidate.
* The **selector** (see [`Selector`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Selector.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Selector)) sees all variants at once and returns one, or
  `None` to decline. The default, [`MinCost`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.MinCost.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.MinCost) with [`device_cost()`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.device_cost.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.device_cost), takes the
  variant of the lowest two-qubit depth, then count, after a quick routed transpile onto the
  device when one is known. Ties go to the earlier variant: the order of `candidates`, then
  each candidate’s own order.

The candidates that use the device’s connectivity need the physical qubits of the gate, so they
offer nothing unless the plugin runs after the layout stage, or `edges` are given.

**Example.** The terms below commute, so the plugin synthesizes the gate; its output is exact.

```pycon
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
```

The `"default"` entry is the fallback for the gates the plugin declines, such as a group whose
terms do not commute.

**Declines** (returns `None`, so the next method of the `HLSConfig` runs) a gate it does not
handle: anything but a [`PauliEvolutionGate`](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.library.PauliEvolutionGate) over a single
operator with a plain real time (a list of operators or a parameterized time included). It also
declines a gate it handles when the operator is not a Hermitian sum of mutually commuting
Paulis, or when the selector returns `None` (the default one does when no candidate built a
circuit, or none of them can be routed onto the coupling map). The gate’s own `synthesis`
attribute is ignored.

* **Parameters:**
  * **candidates** (*Iterable* *[*[*Candidate*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Candidate.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Candidate) *]*) – the candidates to ask, in the order in which ties are broken. Their names must
    be unique.
  * **selector** ([*Selector*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Selector.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Selector) *|* *None*) – chooses one of the variants. `None` (the default) is [`MinCost`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.MinCost.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.MinCost) with
    [`device_cost()`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.device_cost.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.device_cost).
  * **edges** (*Iterable* *[*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple) *[*[*int*](https://docs.python.org/3/builtins/functions.html#int) *,* [*int*](https://docs.python.org/3/builtins/functions.html#int) *]* *]*  *|* *None*) – allowed CZ pairs for the candidates that use them, in the qubit indices of the
    circuit that is being transpiled: its virtual qubits before the layout stage, its
    physical ones after it. Each gate uses the pairs between two of its own qubits. They
    take precedence over the device’s.
  * **callback** (*Callable* *[* *[*[*Report*](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Report.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Report) *]* *,* *None* *]*  *|* *None*) – called with a [`Report`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Report.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Report) for every gate the plugin handles, including the
    ones it then declines. A gate it does not handle is passed on without a report.
* **Raises:**
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if two candidates have the same name.
  * [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) – if `edges` holds something other than pairs of two distinct non-negative qubit
        indices.

Configures the plugin; see the class documentation for the arguments.

### Methods

#### run(high_level_object, coupling_map=None, target=None, qubits=None, \*\*options)

Synthesizes one `PauliEvolutionGate`, or returns `None` to decline it.

When the plugin declines, the high-level synthesis tries the next method.

`qubits` are the gate’s physical qubits, which the high-level synthesis passes only after
the layout stage. The option `input_qubits`, which it passes always, gives the gate’s
qubits in the circuit, so that `edges` can be mapped onto the gate before the layout stage
too. Called directly without either, the gate’s qubits are `0, 1, ...`.
