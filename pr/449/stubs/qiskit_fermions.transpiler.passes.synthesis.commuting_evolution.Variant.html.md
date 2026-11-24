# Variant

### *class* Variant(candidate, variant, circuit)

One circuit a candidate offers for a problem.

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

### Attributes

* **Parameters:**
  * **candidate** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str))
  * **variant** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str))
  * **circuit** ([*QuantumCircuit*](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit))

#### label

`"candidate[variant]"`, or the candidate’s name alone when [`variant`](#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Variant.variant) is empty.

#### candidate *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

The [`name`](qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Candidate.html.md#qiskit_fermions.transpiler.passes.synthesis.commuting_evolution.Candidate.name) of the candidate that built it.

#### variant *: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*

Which of the candidate’s circuits it is, or `""` for a candidate with a single one.

#### circuit *: [QuantumCircuit](https://quantum.cloud.ibm.com/docs/api/qiskit/qiskit.circuit.QuantumCircuit)*

The circuit, on the gate’s qubits. It implements the evolution exactly.
